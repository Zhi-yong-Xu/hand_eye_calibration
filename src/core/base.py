# src/core/base.py
import numpy as np
import cv2
import mujoco
import mujoco.viewer
from src.utils.keyboard import KeyListener
from src.utils.mujoco_helper import load_model_with_board

class BaseCalibrator:
    def __init__(self, cfg):
        self.cfg = cfg
        robot = cfg["robot"]
        cam_cfg = cfg["camera"]

        self.model = load_model_with_board(cfg)
        self.data = mujoco.MjData(self.model)
        self.render_w = int(cam_cfg["render_width"])
        self.render_h = int(cam_cfg["render_height"])
        self.renderer = mujoco.Renderer(self.model, height=self.render_h, width=self.render_w)

        self.cam_name = cam_cfg["name"]
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        if self.cam_id < 0:
            raise RuntimeError(f"XML 中找不到相机 '{self.cam_name}'")

        self.joint_names = list(robot["joint_names"])
        self.joint_jids = []
        self.joint_adr = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"找不到关节 '{name}'")
            self.joint_jids.append(jid)
            self.joint_adr.append(int(self.model.jnt_qposadr[jid]))

        self.site_name = robot["ee_site"]
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)
        if self.site_id < 0:
            raise RuntimeError(f"找不到 site '{self.site_name}'")

        # 严格索引：keyframe 必须显式配置（可为空字符串表示不加载 keyframe）
        kf_name = robot["keyframe"]
        if kf_name:
            kf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, kf_name)
            if kf_id < 0:
                raise RuntimeError(f"找不到 keyframe '{kf_name}'")
            mujoco.mj_resetDataKeyframe(self.model, self.data, kf_id)

        self.target_qpos = [float(self.data.qpos[adr]) for adr in self.joint_adr]

        self.joint_step = float(robot["joint_step"])
        self.smoothing = float(robot["smoothing"])

        # ★ 顺序修正：先读 control_mode（因为 key_map 校验依赖它）
        self.control_mode = robot["control_mode"]
        if self.control_mode not in ("joint", "cartesian"):
            raise ValueError(f"robot.control_mode 必须是 'joint' 或 'cartesian'，得到: {self.control_mode}")

        # ★ 按模式分支校验 key_map
        self.key_map = list(robot["key_map"])
        if self.control_mode == "joint":
            # joint 模式：每个关节必须有一对键
            if len(self.key_map) < len(self.joint_names):
                raise ValueError(
                    f"joint 模式下 robot.key_map 长度 {len(self.key_map)} 少于关节数 {len(self.joint_names)}，"
                    f"必须为每个关节显式配置 inc/dec"
                )
        else:  # cartesian
            # cartesian 模式：需要 6 项（x/y/z/rx/ry/rz），与关节数无关
            if len(self.key_map) < 6:
                raise ValueError(
                    f"cartesian 模式下 robot.key_map 长度 {len(self.key_map)} 少于 6，"
                    f"必须为 6D 空间速度（x/y/z/rx/ry/rz）各配置一对键"
                )

        # 严格索引：cartesian 模式下必须显式配置全部 4 个字段
        if self.control_mode == "cartesian":
            self.cart_cfg = robot["cartesian"]
            for key in ("linear_speed", "angular_speed", "damping", "max_joint_vel"):
                if key not in self.cart_cfg:
                    raise KeyError(f"robot.cartesian 缺少必需字段: '{key}'")
        else:
            self.cart_cfg = {}

        self.key_listener = KeyListener()

        # 严格索引：scene.enabled 必须显式配置
        if cfg["board"]["scene"]["enabled"]:
            self.board_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, cfg["board"]["scene"]["body_name"]
            )
            if self.board_body_id < 0:
                raise RuntimeError(f"标定板 body '{cfg['board']['scene']['body_name']}' 不存在")
        else:
            self.board_body_id = -1

        mujoco.mj_forward(self.model, self.data)

    def _setup_board_offsets(self):
        b = self.cfg["board"]
        if b["origin"] == "center":
            try:
                _c = np.asarray(self.charuco_board.getChessboardCorners())
            except AttributeError:
                _c = np.asarray(self.charuco_board.chessboardCorners)
            _min = _c[:, :2].min(axis=0)
            _max = _c[:, :2].max(axis=0)
            self.offset_x = float((_min[0] + _max[0]) / 2.0)
            self.offset_y = float((_min[1] + _max[1]) / 2.0)
        else:
            raise NotImplementedError(f"board.origin='{b['origin']}' 未实现，目前仅支持 'center'")

    def _camera_on_front_side(self):
        if self.board_body_id < 0:
            return True
        p_board = self.data.xpos[self.board_body_id]
        R_board = self.data.xmat[self.board_body_id].reshape(3, 3)
        n_world = R_board @ np.array([0.0, 0.0, 1.0])
        cam_pos = self.data.cam_xpos[self.cam_id]
        return float((cam_pos - p_board) @ n_world) > 0.0

    def control_robot_joint(self):
        kl = self.key_listener
        # ★ joint 模式下已严格校验 key_map 长度足够，无需 min()
        for i in range(len(self.joint_names)):
            km = self.key_map[i]
            if km.get("inc") and kl.take_edge(km["inc"]):
                self.target_qpos[i] += self.joint_step
            if km.get("dec") and kl.take_edge(km["dec"]):
                self.target_qpos[i] -= self.joint_step
        for i in range(len(self.joint_names)):
            adr = self.joint_adr[i]
            cur = self.data.qpos[adr]
            self.data.qpos[adr] = cur + (self.target_qpos[i] - cur) * self.smoothing

    def control_robot_cartesian(self, dt):
        kl = self.key_listener
        # 严格索引（在 __init__ 已校验 cart_cfg 完整性）
        v_lin  = float(self.cart_cfg["linear_speed"])
        v_ang  = float(self.cart_cfg["angular_speed"])
        lam    = float(self.cart_cfg["damping"])
        max_qv = float(self.cart_cfg["max_joint_vel"])

        twist = np.zeros(6)
        for i, km in enumerate(self.key_map[:6]):
            speed = v_lin if i < 3 else v_ang
            if km.get("inc") and kl.is_pressed(km["inc"]):
                twist[i] += speed
            if km.get("dec") and kl.is_pressed(km["dec"]):
                twist[i] -= speed
        if not np.any(twist):
            return False

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
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
                if km.get("inc") and self.key_listener.is_pressed(km["inc"]):
                    return False
                if km.get("dec") and self.key_listener.is_pressed(km["dec"]):
                    return False
            return True
        return all(
            abs(self.data.qpos[self.joint_adr[i]] - self.target_qpos[i]) < tol
            for i in range(len(self.joint_names))
        )

    def detect_board(self, image_bgr, min_corners):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = self.charuco_detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < min_corners:
            return None, None, 0
        all_corners = self.charuco_board.getChessboardCorners()
        ids_flat = ch_ids.flatten().astype(int)
        obj_pts = np.asarray(all_corners)[ids_flat].reshape(-1, 3).astype(np.float64)
        img_pts = ch_corners.reshape(-1, 2).astype(np.float64)
        obj_pts[:, 0] -= self.offset_x
        obj_pts[:, 1] -= self.offset_y
        return img_pts, obj_pts, len(ch_ids)

    def step_simulation(self, dt):
        mujoco.mj_forward(self.model, self.data)
        if self.control_mode == "cartesian":
            self.control_robot_cartesian(dt)
        else:
            self.control_robot_joint()
        mujoco.mj_forward(self.model, self.data)