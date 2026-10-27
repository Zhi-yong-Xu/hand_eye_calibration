# src/core/intrinsic.py
import time
import math
import cv2
import numpy as np
import mujoco
import mujoco.viewer
from src.core.base import BaseCalibrator
from src.utils.camera import get_intrinsics_from_model

class IntrinsicCalibrator(BaseCalibrator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.K_true = get_intrinsics_from_model(self.model, self.cam_id, self.render_w, self.render_h)

        b = cfg["board"]
        dict_id = getattr(cv2.aruco, b["aruco_dict"])
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (int(b["squares_x"]), int(b["squares_y"])),
            float(b["square_len"]), float(b["marker_len"]), aruco_dict)

        dparams = cv2.aruco.DetectorParameters()
        dparams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        dparams.cornerRefinementWinSize = 5
        dparams.cornerRefinementMaxIterations = 100
        dparams.cornerRefinementMinAccuracy = 0.001
        self.charuco_detector = cv2.aruco.CharucoDetector(
            self.charuco_board, cv2.aruco.CharucoParameters(), dparams)

        self._setup_board_offsets()

        # 严格索引
        self.min_corners = int(cfg["board"]["min_corners_intrinsic"])

        self.obj_pts = []
        self.img_pts = []
        self.sample_meta = []

    def _pnp_diagnose(self, obj_pts, img_pts):
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts.astype(np.float32), img_pts.astype(np.float32),
            self.K_true.astype(np.float32), np.zeros((5, 1), np.float32),
            flags=cv2.SOLVEPNP_SQPNP)
        return (rvec, tvec) if ok else None

    def _sample_metadata(self, obj_pts, img_pts, rvec, tvec):
        R_t2c, _ = cv2.Rodrigues(rvec)
        p_cam = R_t2c @ np.mean(obj_pts, axis=0) + tvec.flatten()
        depth = float(p_cam[2])
        n_cam = R_t2c @ np.array([0.0, 0.0, 1.0])
        tilt_deg = math.degrees(math.acos(float(np.clip(n_cam[2], -1.0, 1.0))))
        return dict(depth=depth, tilt=tilt_deg, img_center=np.mean(img_pts, axis=0))

    def collect(self, image_bgr):
        if not self._camera_on_front_side():
            print("[丢弃] 相机在板子背面")
            return False
        img_pts, obj_pts, ncorner = self.detect_board(image_bgr, self.min_corners)
        if img_pts is None:
            return False
        pnp = self._pnp_diagnose(obj_pts, img_pts)
        if pnp is None:
            return False
        meta = self._sample_metadata(obj_pts, img_pts, *pnp)
        print(f"[采集] 第 {len(self.obj_pts)+1} 帧 | 角点 {ncorner} | 深度 {meta['depth']*1000:.0f}mm | 倾斜 {meta['tilt']:.1f}°")
        self.obj_pts.append(obj_pts.reshape(-1, 1, 3).astype(np.float32))
        self.img_pts.append(img_pts.reshape(-1, 1, 2).astype(np.float32))
        self.sample_meta.append(meta)
        return True

    def _coverage_report(self):
        depths = [m['depth'] for m in self.sample_meta]
        tilts = [m['tilt'] for m in self.sample_meta]
        depth_ratio = max(depths) / min(depths) if min(depths) > 1e-6 else 0
        print("\n" + "=" * 60 + "\n 采样覆盖度评估\n" + "=" * 60)
        print(f" 样本数: {len(self.sample_meta)}")
        print(f" 深度范围: {min(depths):.3f} ~ {max(depths):.3f} m (比值 {depth_ratio:.2f})")
        print(f" 倾斜范围: {min(tilts):.1f}° ~ {max(tilts):.1f}°")
        if depth_ratio < 1.5:
            print(" ⚠ 深度比值 < 1.5，建议靠近和远离各采几帧")
        if max(tilts) < 20:
            print(" ⚠ 最大倾斜 < 20°，建议绕两轴各倾斜 20°~40°")
        print("=" * 60 + "\n")

    def _robust_calibrate(self):
        obj_list, img_list = list(self.obj_pts), list(self.img_pts)
        K = dist = None
        for it in range(3):
            K_init = self.K_true.copy() if it == 0 else K.copy()
            dist_init = np.zeros((5, 1), dtype=np.float32) if it == 0 else dist.copy()
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_list, img_list, (self.render_w, self.render_h),
                K_init, dist_init, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
            if not ret:
                return None
            rms_arr = []
            for i in range(len(obj_list)):
                proj, _ = cv2.projectPoints(obj_list[i], rvecs[i], tvecs[i], K, dist)
                d = img_list[i].reshape(-1, 2) - proj.reshape(-1, 2)
                rms_arr.append(float(np.sqrt(np.mean(np.sum(d * d, axis=1)))))
            rms_arr = np.array(rms_arr)
            med = float(np.median(rms_arr))
            mad = float(np.median(np.abs(rms_arr - med))) + 1e-6
            keep = rms_arr <= max(med + 3.0 * 1.4826 * mad, 0.5)
            print(f"  鲁棒迭代 {it+1}: RMS 中位数 {med:.4f}px, 保留 {int(np.sum(keep))}/{len(obj_list)} 帧")
            if np.all(keep) or len(obj_list) <= 8:
                break
            obj_list = [obj_list[i] for i in range(len(obj_list)) if keep[i]]
            img_list = [img_list[i] for i in range(len(img_list)) if keep[i]]
        return K, dist

    def run_calibration(self):
        if len(self.obj_pts) < 8:
            print(f"[失败] 数据不足 ({len(self.obj_pts)}/8)")
            return
        self._coverage_report()
        result = self._robust_calibrate()
        if not result:
            return
        K, dist = result
        diff = K - self.K_true
        err_fx = abs(diff[0, 0]) / self.K_true[0, 0] * 100
        err_fy = abs(diff[1, 1]) / self.K_true[1, 1] * 100
        print(f"\n[K_calib]\n{K}\n\n[K_true]\n{self.K_true}")
        print(f"\n fx 误差 {err_fx:.3f}% | fy 误差 {err_fy:.3f}%")
        print("\n[复制到 config]")
        print("camera:\n  intrinsics_mode: \"manual\"\n  manual_intrinsics:")
        for k, v in [("fx", K[0, 0]), ("fy", K[1, 1]), ("cx", K[0, 2]), ("cy", K[1, 2])]:
            print(f"    {k}: {v:.4f}")

    def run(self):
        target_dt = 1.0 / float(self.cfg["simulation"]["target_fps"])

        # ★ 启动时打印操作说明
        print("=" * 58)
        print(f" 内参标定 | 控制模式: {self.control_mode}")
        if self.control_mode == "cartesian":
            print(" 笛卡尔键位: q/a=±x  w/s=±y  e/d=±z  r/f=±rx  t/g=±ry  y/h=±rz")
        else:
            print(f" 关节键位: 前 {len(self.joint_names)} 个键对应 J1~J{len(self.joint_names)}")
        print(" SPACE 采集 | ENTER 计算 | ESC 退出")
        print(" 采集要点：深度变化 ≥1.5x、3x3 网格覆盖 ≥6 格、倾斜 ≥20°")
        print("=" * 58)

        cv2.namedWindow("Intrinsic Calibration", cv2.WINDOW_NORMAL)
        quit_flag = False

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                t0 = time.perf_counter()
                self.step_simulation(target_dt)

                self.renderer.update_scene(self.data, camera=self.cam_name)
                img_bgr = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
                img_pts, _, ncorner = self.detect_board(img_bgr, self.min_corners)

                vis = img_bgr.copy()
                for m in self.sample_meta:
                    cx = int(np.clip(m['img_center'][0], 0, self.render_w - 1))
                    cy = int(np.clip(m['img_center'][1], 0, self.render_h - 1))
                    cv2.drawMarker(vis, (cx, cy), (0, 165, 255), cv2.MARKER_CROSS, 12, 2)
                if img_pts is not None:
                    for p in img_pts.astype(int):
                        cv2.circle(vis, tuple(p), 3, (0, 255, 0), -1)

                back = not self._camera_on_front_side() and self.board_body_id >= 0
                status = "Back!" if back else ("Ready" if self.is_settled() else "Moving")
                color = (0, 0, 255) if back else (0, 255, 0)
                cv2.putText(vis, f"Captured: {len(self.obj_pts)} | Corners: {ncorner} | {status}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow("Intrinsic Calibration", vis)
                cv2.waitKey(1)

                if self.key_listener.take_edge("esc"):
                    quit_flag = True
                if self.key_listener.take_edge("space") and self.is_settled():
                    if self.collect(img_bgr):
                        print(f"[累计] {len(self.obj_pts)} 组")
                if self.key_listener.take_edge("enter"):
                    self.run_calibration()

                viewer.sync()
                elapsed = time.perf_counter() - t0
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
                if quit_flag:
                    break
        cv2.destroyAllWindows()
        self.key_listener.stop()