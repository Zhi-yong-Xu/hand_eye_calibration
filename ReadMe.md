## 项目介绍
这是一个基于mujoco的手眼标定仿真项目，封装了相机内参标定和手眼标定的主要流程，用户可通过填写.yaml文件进行手眼标定

## 📁 项目目录结构

```
project_root/
├── calibrate_intrinsics.py       # 内参标定入口
├── hand_eye_calibration.py       # 手眼标定入口
└── src/
    ├── __init__.py
    ├── utils/
    │   ├── __init__.py
    │   ├── camera.py             # 相机内参、物理尺寸计算
    │   ├── kinematics.py         # 旋转矩阵、四元数、欧拉角转换
    │   ├── keyboard.py           # 键盘监听类
    │   └── mujoco_helper.py      # MuJoCo 模型加载与标定板注入
    └── core/
        ├── __init__.py
        ├── base.py               # 标定器基类（机器人控制、板子检测等）
        ├── intrinsic.py          # 内参标定核心逻辑类
        └── handeye.py            # 手眼标定核心逻辑类
```

## 模型 XML 前置要求 

请确保您的机器人 XML 文件（如 kuka.xml 或 panda.xml）的 <mujoco> 根节点内包含以下 <visual> 配置：
```
<visual>
  <global offwidth="1280" offheight="720"/>
</visual>
```
如果您的 XML 中已有 <visual> 节点, 跳过。


## 快速开始
### 标定板生成
```
python generate_board.py
```
### 内参标定
```
python calibrate_intrinsics.py --config config/kuka.yaml
```

### 手眼标定
```
python hand_eye_calibration.py --config config/kuka.yaml
```

### config 文件配置教程
```
# =====================================================
# 手眼标定配置  (Eye-in-Hand, 标定目标 = ee_site)
# =====================================================

# ---------- 1. 仿真环境 ----------
simulation:
  model_path: "path/to/your/model.xml"   # MuJoCo 模型 XML 的路径
  camera_name: "wrist_cam"               # XML 中相机的名字 <camera name="xxx">
  base_body: "base_link"                 # 机器人基座的名字 <body name="xxx">
  render_width: 1280                     # 渲染分辨率宽（必须与 XML 的 offwidth 一致）
  render_height: 720                     # 渲染分辨率高（必须与 XML 的 offheight 一致）
  target_fps: 60                         # 主循环帧率

# ---------- 2. 机器人控制 ----------
robot:
  type: "your_robot"                     # 仅用于打印日志显示
  joint_names: ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"] 
  ee_site: "tool0"                       # 末端执行器 site 的名字 <site name="xxx">
  keyframe: "home"                       # 初始关键帧名字（没有可写 ""）

  # 控制模式：
  # joint     : 按键直接控制关节角（适合手眼标定，能产生大角度旋转）
  # cartesian : 按键控制末端笛卡尔速度（适合内参标定，直观调深度和视角）
  control_mode: "cartesian"              

  # 笛卡尔模式参数（仅 control_mode="cartesian" 时生效）
  cartesian:
    linear_speed: 0.05                   # 末端平移速度 m/s
    angular_speed: 0.5                   # 末端旋转速度 rad/s
    damping: 0.05                        # DLS 阻尼系数（0.01~0.1）
    max_joint_vel: 1.0                   # 单关节最大速度 rad/s

  # 关节模式参数（仅 control_mode="joint" 时生效）
  joint_step: 0.02                       # 每按一次键关节角变化量 rad
  smoothing: 0.10                        # 每帧向目标插值系数

  # 按键映射（前3对对应位置，后3对对应姿态）
  # ⚠️ 警告：这里的按键不能与功能键（space/enter/esc/c/p/n）重复
  key_map:
    - { inc: "q", dec: "a" }             # +X / -X (或 J1+ / J1-)
    - { inc: "w", dec: "s" }             # +Y / -Y (或 J2+ / J2-)
    - { inc: "e", dec: "d" }             # +Z / -Z (或 J3+ / J3-)
    - { inc: "r", dec: "f" }             # +RX / -RX (或 J4+ / J4-)
    - { inc: "t", dec: "g" }             # +RY / -RY (或 J5+ / J5-)
    - { inc: "y", dec: "h" }             # +RZ / -RZ (或 J6+ / J6-)

# ---------- 3. 标定板 ----------
board:
  type: "charuco"
  squares_x: 4                           # X方向格子数
  squares_y: 6                           # Y方向格子数
  square_len: 0.04                       # 格子物理边长 m
  marker_len: 0.03                       # ArUco marker 物理边长 m
  aruco_dict: "DICT_6X6_250"             # ArUco 字典类型
  origin: "center"                       # 板子原点（目前仅支持 center）
  px_per_square: 200                     # 生成图片时每格像素数
  margin_px: 0                           # 图片留白像素
  border_bits: 1
  image_path: "charuco_board.png"        # 生成的板子图片路径
  meta_path: "charuco_board_meta.json"   # 元数据存档路径

  min_corners_handeye: 6                 # 手眼标定最少需要的角点数

  # 标定板在 MuJoCo 场景中的自动注入设置
  scene:
    enabled: true
    body_name: "calib_auto_board"
    texture_name: "calib_auto_board_tex"
    material_name: "calib_auto_board_mat"
    
    # ⚠️ 最重要：板子在世界坐标系中的位置和朝向
    # pos: 板子几何中心坐标 [x, y, z]
    # euler: MuJoCo内旋XYZ欧拉角，弧度制。 
    #        [0, 0, 0] 平放
    #        [0, -1.5708, 0] 竖放，朝向机器人的方向（-X）
    pos: [0.5, 0.0, 0.3]
    euler: [0, -1.5708, 0]
    
    thickness: 0.004                     # 板子厚度
    add_collision: false                 # 是否开启碰撞

# ---------- 4. 相机与内参 ----------
camera:
  name: "wrist_cam"
  render_width: 1280
  render_height: 720

  # analytic  : 从 XML 中解析（推荐仿真使用，fovy=45 对应 fx≈869）
  # manual    : 使用下方 manual_intrinsics 填写的值（自己标定出的结果）
  # calibrate : 启动用解析值，按 C 标定后需按 P 应用
  intrinsics_mode: "analytic"

  manual_intrinsics:
    fx: 869.0
    fy: 869.0
    cx: 640.0
    cy: 360.0

# ---------- 5. 标定算法参数 ----------
calibration:
  min_samples: 10                        # 最少采集多少组数据进行标定
  max_reproj_err_px: 1.0                 # 重投影误差阈值（超过该值的数据丢弃）
  method: "TSAI"                         # TSAI 或 PARK
  verify_ok_angle_deg: 1.0               # 验证旋转误差的 "优秀" 阈值
  verify_ok_trans_m: 0.01                # 验证平移误差的 "优秀" 阈值
  result_npy: "hand_eye_result.npy"      # 输出矩阵文件
  result_json: "hand_eye_result.json"    # 输出报告文件
```
