import numpy as np
import math

def get_intrinsics_from_model(model, cam_id, w, h):
    sensor = np.asarray(model.cam_sensorsize[cam_id], dtype=np.float64)
    intr = np.asarray(model.cam_intrinsic[cam_id], dtype=np.float64)
    fovy = float(model.cam_fovy[cam_id])
    if (sensor[0] > 1e-6 and sensor[1] > 1e-6
            and intr[0] > 1e-6 and intr[1] > 1e-6):
        fx = w * (intr[0] / sensor[0]); fy = h * (intr[1] / sensor[1])
        cx = w / 2.0 + w * (intr[2] / sensor[0]); cy = h / 2.0 + h * (intr[3] / sensor[1])
        print(f"[内参] sensorsize: fx={fx:.4f} fy={fy:.4f} cx={cx:.2f} cy={cy:.2f}")
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    if fovy > 1e-6:
        fy = h / (2.0 * math.tan(math.radians(fovy) / 2.0)); fx = fy
        cx, cy = w / 2.0, h / 2.0
        print(f"[内参] fovy: fx=fy={fx:.4f} cx={cx:.2f} cy={cy:.2f}")
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    raise RuntimeError("相机既无有效 sensorsize 也无 fovy，无法解析内参！")

def board_physical_size(b):
    px = b["px_per_square"]; m = b.get("margin_px", 0); sq = b["square_len"]
    return ((b["squares_x"]*px + 2*m)/px*sq, (b["squares_y"]*px + 2*m)/px*sq)