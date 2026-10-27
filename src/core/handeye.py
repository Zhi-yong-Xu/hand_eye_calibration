# src/core/handeye.py
import time
import json
import math
import cv2
import numpy as np
import mujoco
import mujoco.viewer
from src.core.base import BaseCalibrator
from src.utils.camera import get_intrinsics_from_model
from src.utils.kinematics import rotation_angle_deg

HAND_EYE_METHODS = {"TSAI": cv2.CALIB_HAND_EYE_TSAI, "PARK": cv2.CALIB_HAND_EYE_PARK}
FUNC_KEYS = {"c", "p", "n", "space", "enter", "esc"}

class HandEyeCalibrator(BaseCalibrator):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        cam_cfg = cfg["camera"]
        self.intr_mode = cam_cfg.get("intrinsics_mode", "analytic")
        self._intr_calibrated = True
        self._pending_K = None; self._pending_dist = None

        if self.intr_mode == "manual":
            m = cam_cfg["manual_intrinsics"]
            self.K = np.array([[m["fx"], 0, m["cx"]], [0, m["fy"], m["cy"]], [0, 0, 1]], dtype=np.float64)
            print(f"[内参] manual: fx={m['fx']:.4f} fy={m['fy']:.4f}")
        elif self.intr_mode == "analytic":
            self.K = get_intrinsics_from_model(self.model, self.cam_id, self.render_w, self.render_h)
        elif self.intr_mode == "calibrate":
            self.K = get_intrinsics_from_model(self.model, self.cam_id, self.render_w, self.render_h)
            self._intr_calibrated = False
        
        self.dist = np.zeros((5, 1), dtype=np.float64)

        b = cfg["board"]
        dict_id = getattr(cv2.aruco, b["aruco_dict"])
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (int(b["squares_x"]), int(b["squares_y"])), float(b["square_len"]), float(b["marker_len"]), aruco_dict)
        self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board, cv2.aruco.CharucoParameters(), cv2.aruco.DetectorParameters())
        
        self._setup_board_offsets()
        self.min_corners_handeye = int(cfg["board"].get("min_corners_handeye", 6))

        self.R_gripper2base = []; self.t_gripper2base = []
        self.R_target2cam = []; self.t_target2cam = []
        self.reproj_errors = []; self.intr_obj_pts = []; self.intr_img_pts = []
        self.sampled_qpos = []
        
        self._validate_keys()

    def _validate_keys(self):
        used = set()
        for km in self.key_map:
            if km.get("inc"): used.add(km["inc"])
            if km.get("dec"): used.add(km["dec"])
        conflict = used & FUNC_KEYS
        if conflict: raise RuntimeError(f"按键冲突: {sorted(conflict)}")

    def get_site_pose_in_base(self):
        pos_w = self.data.site_xpos[self.site_id].copy()
        mat_w = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        base_body = self.cfg["simulation"].get("base_body") or ""
        if base_body:
            base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body)
            bp = self.data.xpos[base_id].copy()
            bR = self.data.xmat[base_id].reshape(3, 3).copy()
            return bR.T @ mat_w, bR.T @ (pos_w - bp)
        return mat_w, pos_w

    def get_true_cam2site_at(self, qpos_snapshot):
        saved = self.data.qpos.copy()
        self.data.qpos[:] = qpos_snapshot
        mujoco.mj_forward(self.model, self.data)
        
        T_w_cam = np.eye(4); T_w_cam[:3, :3] = self.data.cam_xmat[self.cam_id].reshape(3, 3); T_w_cam[:3, 3] = self.data.cam_xpos[self.cam_id]
        T_w_site = np.eye(4); T_w_site[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3); T_w_site[:3, 3] = self.data.site_xpos[self.site_id]
        result = np.linalg.inv(T_w_site) @ T_w_cam
        
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return result

    def collect_current(self, image_bgr):
        if not self._camera_on_front_side():
            print("[丢弃] 相机在板子背面")          # ★ 补回诊断
            return False, 999.0
            
        img_pts, obj_pts, _ = self.detect_board(image_bgr, self.min_corners_handeye)
        if img_pts is None:
            print(f"[丢弃] 检测角点不足 {self.min_corners_handeye} 个")  # ★ 补回诊断
            return False, 0.0

        ok, rvec, tvec = cv2.solvePnP(obj_pts.astype(np.float32), img_pts.astype(np.float32), self.K.astype(np.float32), self.dist.astype(np.float32), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tvec[2, 0] <= 0: 
            print("[丢弃] PnP 失败或深度负值")      # ★ 补回诊断
            return False, 999.0

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.dist)
        rms = float(np.sqrt(np.mean(np.sum((img_pts - proj.reshape(-1, 2))**2, axis=1))))
        if rms > float(self.cfg["calibration"]["max_reproj_err_px"]): 
            print(f"[丢弃] 重投影误差 {rms:.3f}px 超限")  # ★ 补回诊断
            return False, rms

        R_t2c, _ = cv2.Rodrigues(rvec)
        R_s2b, t_s2b = self.get_site_pose_in_base()

        self.R_gripper2base.append(R_s2b); self.t_gripper2base.append(np.asarray(t_s2b).reshape(3, 1))
        self.R_target2cam.append(R_t2c); self.t_target2cam.append(np.asarray(tvec).reshape(3, 1))
        self.reproj_errors.append(rms)
        self.intr_obj_pts.append(obj_pts.reshape(-1, 1, 3).astype(np.float32))
        self.intr_img_pts.append(img_pts.reshape(-1, 1, 2).astype(np.float32))
        self.sampled_qpos.append(self.data.qpos.copy())
        return True, rms

    def run_intrinsic_calibration(self):
        if len(self.intr_obj_pts) < 5: 
            print(f"[内参] 数据不足 {len(self.intr_obj_pts)}/5")
            return
        print(f"\n[内参标定] {len(self.intr_obj_pts)} 组")
        K_init = get_intrinsics_from_model(self.model, self.cam_id, self.render_w, self.render_h)
        ret, K_calib, dist_calib, _, _ = cv2.calibrateCamera(self.intr_obj_pts, self.intr_img_pts, (self.render_w, self.render_h), K_init, np.zeros((5, 1), dtype=np.float32), flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        if ret:
            self._pending_K = K_calib.astype(np.float64); self._pending_dist = dist_calib.astype(np.float64)
            print("[K_calib]\n", K_calib)
            print("\n>>> 按 P 应用 / 按 N 取消")

    def run_calibration(self):
        if self.intr_mode == "calibrate" and not self._intr_calibrated: 
            print("[拒绝] calibrate 模式请先按 C 内参标定 + P 应用")
            return
        
        n = len(self.R_gripper2base)
        min_samples = int(self.cfg["calibration"]["min_samples"])
        if n < min_samples: 
            print(f"[失败] 样本不足: {n}/{min_samples}，请继续采集")
            return
        
        print(f"\n=== 手眼标定 === 样本 {n}, 平均 RMS {np.mean(self.reproj_errors):.4f} px")

        results = {}
        for name, method in HAND_EYE_METHODS.items():
            try:
                R, t = cv2.calibrateHandEye(self.R_gripper2base, self.t_gripper2base, self.R_target2cam, self.t_target2cam, method=method)
                T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(t).flatten()
                results[name] = T
            except Exception as e:
                print(f"  [{name}] 失败: {e}")

        if not results: 
            print("[失败] 所有算法均失败，请检查数据质量")
            return

        cons_mm = None
        if len(results) > 1:
            tvecs = np.array([T[:3, 3] for T in results.values()])
            cons_mm = float(np.linalg.norm(np.std(tvecs, axis=0))) * 1000
            print(f"[一致性] 平移 std = {cons_mm:.2f} mm")

        chosen = self.cfg["calibration"].get("method", "TSAI")
        if chosen not in results: chosen = list(results.keys())[0]
        T_opencv = results[chosen]
        
        R_cv2mj = np.diag([1.0, -1.0, -1.0])
        T_mujoco = T_opencv.copy(); T_mujoco[:3, :3] = T_opencv[:3, :3] @ R_cv2mj
        
        # ★ 补回矩阵打印
        print(f"\n采用算法: {chosen}")
        print("\n[结果 T_cam2site OpenCV]")
        print(T_opencv)
        print("\n[结果 T_cam2site MuJoCo]")
        print(T_mujoco)
        
        T_true = self.get_true_cam2site_at(self.sampled_qpos[-1] if self.sampled_qpos else self.data.qpos.copy())
        
        print("\n[真值 T_cam2site MuJoCo]")
        print(T_true)
        
        ang_err = rotation_angle_deg(T_true[:3, :3], T_mujoco[:3, :3])
        t_err = float(np.linalg.norm(T_true[:3, 3] - T_mujoco[:3, 3]))
        
        print(f"\n[验证] 旋转误差 {ang_err:.4f}° | 平移误差 {t_err*1000:.3f} mm")

        calib = self.cfg["calibration"]
        np.save(calib["result_npy"], T_mujoco)
        with open(calib["result_json"], "w", encoding="utf-8") as f:
            json.dump({
                "method": chosen, "samples": n, "control_mode": self.control_mode,
                "mean_reproj_rms_px": float(np.mean(self.reproj_errors)),
                "consistency_trans_std_mm": cons_mm,
                "T_cam2site_mujoco": T_mujoco.tolist(),
                "T_cam2site_opencv": T_opencv.tolist(),
                "T_true_mujoco": T_true.tolist(),
                "verify_angle_deg": ang_err, "verify_trans_m": t_err,
            }, f, indent=2)
        print(f"[保存] {calib['result_npy']} / {calib['result_json']}")

    def run(self):
        target_dt = 1.0 / float(self.cfg["simulation"].get("target_fps", 60))
        min_samples = int(self.cfg["calibration"]["min_samples"])
        cv2.namedWindow("Calibration View", cv2.WINDOW_NORMAL)
        quit_flag = False

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                t0 = time.perf_counter()
                self.step_simulation(target_dt)

                self.renderer.update_scene(self.data, camera=self.cam_name)
                image_bgr = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
                img_pts, _, ncorner = self.detect_board(image_bgr, self.min_corners_handeye)
                
                vis = image_bgr.copy()
                if img_pts is not None:
                    for p in img_pts.astype(int): cv2.circle(vis, tuple(p), 3, (0, 255, 0), -1)

                back = not self._camera_on_front_side() and self.board_body_id >= 0
                status = "Back!" if back else ("Ready" if self.is_settled() else "Moving")
                color = (0, 0, 255) if back else (0, 255, 0)
                extra = " | Pending K (P)" if self._pending_K is not None else ""
                cv2.putText(vis, f"Samples: {len(self.R_gripper2base)}/{min_samples} | Corners: {ncorner} | {status}{extra}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow("Calibration View", vis)
                cv2.waitKey(1)

                if self.key_listener.take_edge("esc"): quit_flag = True
                
                if self.key_listener.take_edge("space") and self.is_settled():
                    ok, _ = self.collect_current(image_bgr)
                    if ok: 
                        print(f"[采集] {len(self.R_gripper2base)} 组")
                    else:
                        print("[采集失败] 见上方原因")
                        
                if self.key_listener.take_edge("c"): 
                    self.run_intrinsic_calibration()
                    
                if self.key_listener.take_edge("p") and self._pending_K is not None:
                    self.K = self._pending_K; self.dist = self._pending_dist
                    self._intr_calibrated = True; self._pending_K = None
                    print(">>> 已应用新内参")
                    
                if self.key_listener.take_edge("n"): 
                    self._pending_K = None
                    print(">>> 已取消")
                    
                if self.key_listener.take_edge("enter"): 
                    print(f"[ENTER] 触发标定，当前样本数 {len(self.R_gripper2base)}")
                    self.run_calibration()

                viewer.sync()
                elapsed = time.perf_counter() - t0
                if elapsed < target_dt: time.sleep(target_dt - elapsed)
                if quit_flag: break
                
        cv2.destroyAllWindows()
        self.key_listener.stop()