import argparse
import json
import cv2
import numpy as np
import yaml

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_charuco_board(b):
    dict_id = getattr(cv2.aruco, b["aruco_dict"])
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.CharucoBoard(
        (b["squares_x"], b["squares_y"]),
        b["square_len"], b["marker_len"], aruco_dict)

def generate_charuco_image(b, board):
    w = b["squares_x"] * b["px_per_square"] + 2 * b["margin_px"]
    h = b["squares_y"] * b["px_per_square"] + 2 * b["margin_px"]
    return board.generateImage((w, h), marginSize=b["margin_px"], borderBits=b["border_bits"])

def generate_chessboard_image(b):
    px = b["px_per_square"]
    w, h = b["squares_x"] * px, b["squares_y"] * px
    img = np.ones((h, w), dtype=np.uint8) * 255
    for i in range(b["squares_y"]):
        for j in range(b["squares_x"]):
            if (i + j) % 2 == 0:
                img[i * px:(i + 1) * px, j * px:(j + 1) * px] = 0
    return img

def physical_size(b):
    px = b["px_per_square"]
    m = b["margin_px"]           # 严格索引
    sq = b["square_len"]
    W = (b["squares_x"] * px + 2 * m) / px * sq
    H = (b["squares_y"] * px + 2 * m) / px * sq
    return W, H

def save_meta(b):
    W, H = physical_size(b)
    meta = {
        "board_type": b["type"],
        "squares_x": b["squares_x"],
        "squares_y": b["squares_y"],
        "square_len_m": b["square_len"],
        "marker_len_m": b["marker_len"] if b["type"] == "charuco" else None,
        "aruco_dict": b["aruco_dict"],
        "origin": b["origin"],
        "inner_corner_grid": [b["squares_x"] - 1, b["squares_y"] - 1],
        "px_per_square": b["px_per_square"],
        "margin_px": b["margin_px"],
        "board_physical_size_m": [W, H],
    }
    with open(b["meta_path"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="生成标定板图片与元数据")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    b = cfg["board"]

    if b["type"] == "charuco":
        img = generate_charuco_image(b, build_charuco_board(b))
    elif b["type"] == "chessboard":
        img = generate_chessboard_image(b)
    else:
        raise ValueError(f"不支持的 board.type: {b['type']}")

    cv2.imwrite(b["image_path"], img)
    save_meta(b)
    W, H = physical_size(b)

    print("=" * 55)
    print(f"  标定板已生成: {b['image_path']}")
    print("=" * 55)
    print(f"  类型        : {b['type']}")
    print(f"  格子数      : {b['squares_x']} x {b['squares_y']}")
    print(f"  格子物理尺寸: {b['square_len']} m")
    if b["type"] == "charuco":
        print(f"  Marker 尺寸 : {b['marker_len']} m")
        print(f"  Aruco 字典  : {b['aruco_dict']}")
    print(f"  内角点数    : ({b['squares_x'] - 1}, {b['squares_y'] - 1})")
    print(f"  物理尺寸    : {W * 100:.2f} cm x {H * 100:.2f} cm")
    print(f"  图片尺寸    : {img.shape[1]} x {img.shape[0]} px")
    print(f"  原点位置    : {b['origin']}")
    print(f"  元数据      : {b['meta_path']}")

if __name__ == "__main__":
    main()