# extract_info.py
import argparse
import json
import os
import sys
import mujoco
import numpy as np

def load_board_meta(meta_path):
    """严格读取标定板元数据，不存在直接抛异常"""
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"\n[错误] 找不到标定板元数据文件: {meta_path}\n"
            f"请先运行: python generate_board.py --config <你的配置文件>\n"
            f"以生成 charuco_board.png 和 charuco_board_meta.json，然后再运行本脚本。"
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    # 校验关键字段是否齐全，缺一不可
    required_keys = ["squares_x", "squares_y", "square_len_m", "marker_len_m", 
                     "aruco_dict", "px_per_square", "margin_px"]
    for k in required_keys:
        if k not in meta or meta[k] is None:
            raise KeyError(f"[错误] 标定板元数据缺失关键字段: '{k}'，请检查 {meta_path}")
    return meta

def extract_info(xml_path, meta_path):
    print(f"正在解析模型: {xml_path} ...\n")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as e:
        raise RuntimeError(f"[错误] 无法加载 MuJoCo 模型: {e}")
    
    # 加载并严格校验标定板元数据（如果失败，直接在这里抛异常退出）
    print(f"正在读取标定板参数: {meta_path} ...\n")
    board_meta = load_board_meta(meta_path)

    # 1. 提取相机
    cameras = []
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        fovy = model.cam_fovy[i]
        cameras.append({"name": name, "fovy": fovy})
    if not cameras:
        raise RuntimeError("[错误] 模型中没有找到任何 <camera>，无法进行标定！")

    # 2. 提取关节
    joints = []
    for i in range(model.njnt):
        jnt_type = model.jnt_type[i]
        if jnt_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joints.append(name)
    if not joints:
        raise RuntimeError("[错误] 模型中没有找到任何关节！")

    # 3. 提取 Site
    sites = []
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        sites.append(name)
    if not sites:
        raise RuntimeError("[错误] 模型中没有找到任何 <site>，手眼标定必须指定末端 site！")

    # 4. 提取 Body (过滤 world)
    bodies = []
    for i in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        bodies.append(name)
    if not bodies:
        raise RuntimeError("[错误] 模型中没有找到任何 <body>！")

    # 5. 提取 Keyframe
    keyframes = []
    for i in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
        keyframes.append(name)
    if not keyframes:
        print("[警告] 模型中没有定义任何 <keyframe>，配置中将留空字符串 \"\"。")
        kf_name = ""
    else:
        kf_name = keyframes[0]

    # 6. 检查离屏渲染缓冲
    off_w = model.vis.global_.offwidth
    off_h = model.vis.global_.offheight
    fb_ok = off_w >= 1280 and off_h >= 720

    # 智能推断工作空间大小 (仅用于给 pos 提供建议)
    ws_max = np.max(model.body_pos, axis=0)
    ws_min = np.min(model.body_pos, axis=0)
    rec_pos_x = round(max(0.5, ws_max[0] * 0.8), 2)
    rec_pos_y = 0.0
    rec_pos_z = round(max(0.3, ws_max[2] * 0.5), 2)

    # ================= 打印报告 =================
    print("=" * 60)
    print(" 🔍 模型与资产信息提取报告")
    print("=" * 60)

    if fb_ok:
        print(f"[离屏渲染] 当前: {off_w}x{off_h} ✅ 满足 1280x720")
    else:
        print(f"[离屏渲染] 当前: {off_w}x{off_h} ❌ 不满足 1280x720")
        raise RuntimeError(
            f"\n[错误] XML 的离屏渲染分辨率不足！\n"
            f"请在 XML 的 <mujoco> 根节点下添加:\n"
            f"<visual><global offwidth=\"1280\" offheight=\"720\"/></visual>"
        )

    print(f"[相机列表] 共 {len(cameras)} 个")
    for c in cameras: print(f"  - {c['name']} (fovy={c['fovy']:.1f}°)")

    print(f"[关节列表] 共 {len(joints)} 个")
    for j in joints: print(f"  - {j}")

    print(f"[Site 列表] 共 {len(sites)} 个")
    for s in sites: print(f"  - {s}")

    print(f"[Body 列表] 共 {len(bodies)} 个 (基座通常为第一个)")
    for b in bodies[:3]: print(f"  - {b}")
    if len(bodies) > 3: print(f"  - ... (省略 {len(bodies)-3} 个)")

    print(f"[Keyframe 列表] 共 {len(keyframes)} 个")
    if keyframes:
        for k in keyframes: print(f"  - {k}")
    else:
        print("  (无)")

    # ================= 生成完整的 YAML 模板 =================
    print("\n" + "=" * 60)
    print(" 📋 请将以下完整内容保存为 config.yaml，并稍作修改")
    print("=" * 60)

    cam_name = cameras[0]['name']
    base_body = bodies[0]
    ee_site = sites[-1]

    print("```yaml")
    print("# =====================================================")
    print("# 由 extract_info.py 自动生成的配置模板")
    print("# =====================================================")
    print()
    print("simulation:")
    print(f'  model_path: "{xml_path}"')
    print(f'  camera_name: "{cam_name}"')
    print(f'  base_body: "{base_body}"')
    print(f'  render_width: 1280')
    print(f'  render_height: 720')
    print(f'  target_fps: 60')
    print()
    print("robot:")
    print(f'  type: "my_robot"')
    print(f'  joint_names: {joints}')
    print(f'  ee_site: "{ee_site}"')
    print(f'  keyframe: "{kf_name}"')
    print(f'  control_mode: "cartesian"')
    print(f'  cartesian:')
    print(f'    linear_speed: 0.05')
    print(f'    angular_speed: 0.5')
    print(f'    damping: 0.05')
    print(f'    max_joint_vel: 1.0')
    print(f'  joint_step: 0.02')
    print(f'  smoothing: 0.10')
    print(f'  key_map:')
    print(f'    - {{ inc: "q", dec: "a" }}')
    print(f'    - {{ inc: "w", dec: "s" }}')
    print(f'    - {{ inc: "e", dec: "d" }}')
    print(f'    - {{ inc: "r", dec: "f" }}')
    print(f'    - {{ inc: "t", dec: "g" }}')
    print(f'    - {{ inc: "y", dec: "h" }}')
    print()
    print("board:")
    print(f'  type: "{board_meta.get("board_type", "charuco")}"')
    print(f'  squares_x: {board_meta["squares_x"]}')
    print(f'  squares_y: {board_meta["squares_y"]}')
    print(f'  square_len: {board_meta["square_len_m"]}')
    print(f'  marker_len: {board_meta["marker_len_m"]}')
    print(f'  aruco_dict: "{board_meta["aruco_dict"]}"')
    print(f'  origin: "{board_meta.get("origin", "center")}"')
    print(f'  px_per_square: {board_meta["px_per_square"]}')
    print(f'  margin_px: {board_meta["margin_px"]}')
    print(f'  border_bits: 1')
    print(f'  image_path: "charuco_board.png"')
    print(f'  meta_path: "{meta_path}"')
    print(f'  min_corners_handeye: 6')
    print(f'  scene:')
    print(f'    enabled: true')
    print(f'    body_name: "calib_auto_board"')
    print(f'    texture_name: "calib_auto_board_tex"')
    print(f'    material_name: "calib_auto_board_mat"')
    print(f'    pos: [{rec_pos_x}, {rec_pos_y}, {rec_pos_z}]  # ⚠️ 请根据实际工作空间微调')
    print(f'    euler: [0, -1.5708, 0]')
    print(f'    thickness: 0.004')
    print(f'    add_collision: false')
    print()
    print("camera:")
    print(f'  name: "{cam_name}"')
    print(f'  render_width: 1280')
    print(f'  render_height: 720')
    if cameras[0]['fovy'] > 0:
        print(f'  intrinsics_mode: "analytic"    # 检测到 fovy={cameras[0]["fovy"]:.1f}，直接使用')
    else:
        print(f'  intrinsics_mode: "manual"      # 无 fovy，需手动标定')
    print(f'  manual_intrinsics:')
    print(f'    fx: 869.0')
    print(f'    fy: 869.0')
    print(f'    cx: 640.0')
    print(f'    cy: 360.0')
    print()
    print("calibration:")
    print(f'  min_samples: 10')
    print(f'  max_reproj_err_px: 1.0')
    print(f'  method: "TSAI"')
    print(f'  verify_ok_angle_deg: 1.0')
    print(f'  verify_ok_trans_m: 0.01')
    print(f'  result_npy: "hand_eye_result.npy"')
    print(f'  result_json: "hand_eye_result.json"')
    print("```")
    print("=" * 60)
    print(" 💡 关于 board.scene.pos 的推荐说明：")
    print(f"    代码尝试根据你的模型自动推算了板子位置: [{rec_pos_x}, {rec_pos_y}, {rec_pos_z}]")
    print(f"    机器人大部分 Body 的坐标范围是: X[{ws_min[0]:.2f}, {ws_max[0]:.2f}], Y[{ws_min[1]:.2f}, {ws_max[1]:.2f}], Z[{ws_min[2]:.2f}, {ws_max[2]:.2f}]")
    print(f"    如果这个位置不合适，请调整 pos，确保标定板出现在相机的视野中心。")
    print(f"    euler: [0, -1.5708, 0] 表示板子竖直朝 -X 方向；如果是平放用 [0, 0, 0]。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo 模型与标定板资产信息提取工具")
    parser.add_argument("--model", type=str, required=True, help="MuJoCo XML 模型路径")
    parser.add_argument("--meta", type=str, default="charuco_board_meta.json", help="标定板元数据路径")
    args = parser.parse_args()
    
    try:
        extract_info(args.model, args.meta)
    except Exception as e:
        print(f"\n{e}\n", file=sys.stderr)
        sys.exit(1)