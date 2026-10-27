import numpy as np
import cv2
import mujoco
from src.utils.keyboard import KeyListener
from src.utils.mujoco_helper import load_model_with_board
from src.utils.camera import get_intrinsics_from_model

class BaseCalibrator:
    def __init__(self, cfg):
        self.cfg = cfg
        sim = cfg["simulation"]; robot = cfg["robot"]; cam_cfg = cfg["camera"]
        
        self.model = load_model_with_board(cfg)
        self.data = mujoco.MjData(self.model)
        self.render_w = int(cam_cfg["render_width"])
        self.render_h = int(cam_cfg["render_height"])
        self.renderer = mujoco.Renderer(self.model, height=self.render_h, width=self.render_w)
        
        self.cam_name = cam_cfg["name"]
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)

        self.joint_names = list(robot["joint_names"])
        self.joint_jids = []; self.joint_adr = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.joint_jids.append(jid)
            self.joint_adr.append(int(self.model.jnt_qposadr[jid]))

        self.site_name = robot["ee_site"]
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)

        kf_name = robot.get("keyframe") or ""
        if kf_name:
            kf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, kf_name)
            if kf_id >= 0: mujoco.mj_resetDataKeyframe(self.model, self.data, kf_id)

        self.target_qpos = [float(self.data.qpos[adr]) for adr in self.joint_adr]
        self.key_map = list(robot.get("key_map", []))
        while len(self.key_map) < len(self.joint_names): self.key_map.append({"inc": None, "dec": None})
        self.joint_step = float(robot.get("joint_step", 0.02))
        self.smoothing = float(robot.get("smoothing", 0.10))
        self.control_mode = robot.get("control_mode", "joint")
        self.cart_cfg = robot.get("cartesian", {})

        self.key_listener = KeyListener()
        # self._setup_board_offsets()
        
        if cfg["board"]["scene"].get("enabled", True):
            self.board_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cfg["board"]["scene"]["body_name"])
        else:
            self.board_body_id = -1

        mujoco.mj_forward(self.model, self.data)

    def _setup_board_offsets(self):
        b = self.cfg["board"]
        try:
            _c = np.asarray(self.charuco_board.getChessboardCorners())
        except AttributeError:
            _c = np.asarray(self.charuco_board.chessboardCorners)
        _min = _c[:, :2].min(axis=0); _max = _c[:, :2].max(axis=0)
        self.offset_x = float((_min[0] + _max[0]) / 2.0)
        self.offset_y = float((_min[1] + _max[1]) / 2.0)

    def _camera_on_front_side(self):
        if self.board_body_id < 0: return True
        p_board = self.data.xpos[self.board_body_id]
        R_board = self.data.xmat[self.board_body_id].reshape(3, 3)
        n_world = R_board @ np.array([0.0, 0.0, 1.0])
        cam_pos = self.data.cam_xpos[self.cam_id]
        return float((cam_pos - p_board) @ n_world) > 0.0

    def control_robot_joint(self):
        kl = self.key_listener
        n = min(len(self.joint_names), len(self.key_map))
        for i in range(n):
            km = self.key_map[i]
            if km.get("inc") and kl.take_edge(km["inc"]): self.target_qpos[i] += self.joint_step
            if km.get("dec") and kl.take_edge(km["dec"]): self.target_qpos[i] -= self.joint_step
        for i in range(len(self.joint_names)):
            adr = self.joint_adr[i]; cur = self.data.qpos[adr]
            self.data.qpos[adr] = cur + (self.target_qpos[i] - cur) * self.smoothing

    def control_robot_cartesian(self, dt):
        kl = self.key_listener
        v_lin = float(self.cart_cfg.get("linear_speed", 0.05))
        v_ang = float(self.cart_cfg.get("angular_speed", 0.5))
        lam   = float(self.cart_cfg.get("damping", 0.05))
        max_qv= float(self.cart_cfg.get("max_joint_vel", 1.0))

        twist = np.zeros(6)
        for i, km in enumerate(self.key_map[:6]):
            speed = v_lin if i < 3 else v_ang
            if km.get("inc") and kl.is_pressed(km["inc"]): twist[i] += speed
            if km.get("dec") and kl.is_pressed(km["dec"]): twist[i] -= speed
        if not np.any(twist): return False

        jacp = np.zeros((3, self.model.nv)); jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        dof_cols = [int(self.model.jnt_dofadr[jid]) for jid in self.joint_jids]
        J_6 = np.vstack([jacp[:, dof_cols], jacr[:, dof_cols]])

        JJt = J_6 @ J_6.T + lam * lam * np.eye(6)
        try:
            dq = J_6.T @ np.linalg.solve(JJt, twist) * dt
        except np.linalg.LinAlgError:
            return False

        dq = np.clip(dq, -max_qv * dt, max_qv * dt)
        for i, adr in enumerate(self.joint_adr):
            new_val = self.data.qpos[adr] + dq[i]
            if self.model.jnt_limited[self.joint_jids[i]]:
                lo, hi = self.model.jnt_range[self.joint_jids[i]]
                new_val = np.clip(new_val, lo, hi)
            self.data.qpos[adr] = new_val
            self.target_qpos[i] = new_val
        return True

    def is_settled(self, tol=1e-3):
        if self.control_mode == "cartesian":
            for km in self.key_map[:6]:
                if km.get("inc") and self.key_listener.is_pressed(km["inc"]): return False
                if km.get("dec") and self.key_listener.is_pressed(km["dec"]): return False
            return True
        return all(abs(self.data.qpos[self.joint_adr[i]] - self.target_qpos[i]) < tol for i in range(len(self.joint_names)))

    def detect_board(self, image_bgr, min_corners=6):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = self.charuco_detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < min_corners: return None, None, 0
        all_corners = self.charuco_board.getChessboardCorners()
        ids_flat = ch_ids.flatten().astype(int)
        obj_pts = np.asarray(all_corners)[ids_flat].reshape(-1, 3).astype(np.float64)
        img_pts = ch_corners.reshape(-1, 2).astype(np.float64)
        obj_pts[:, 0] -= self.offset_x; obj_pts[:, 1] -= self.offset_y
        return img_pts, obj_pts, len(ch_ids)

    def step_simulation(self, dt):
        mujoco.mj_forward(self.model, self.data)
        if self.control_mode == "cartesian":
            self.control_robot_cartesian(dt)
        else:
            self.control_robot_joint()
        mujoco.mj_forward(self.model, self.data)