# calibrate_intrinsics.py
import argparse
import yaml
from src.core.intrinsic import IntrinsicCalibrator

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="相机内参标定")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    IntrinsicCalibrator(cfg).run()