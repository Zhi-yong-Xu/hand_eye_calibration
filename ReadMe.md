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