import argparse
import yaml
from src.core.handeye import HandEyeCalibrator

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手眼标定")
    parser.add_argument("--config", type=str, default="config_ee_panda.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    HandEyeCalibrator(cfg).run()