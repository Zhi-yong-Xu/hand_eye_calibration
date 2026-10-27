import os
import mujoco
import numpy as np
from src.utils.kinematics import euler_xyz_to_quat
from src.utils.camera import board_physical_size

def ensure_framebuffer(spec, need_w, need_h):
    vis = spec.visual
    for attr in ("glob", "global_", "globalvisual"):
        if hasattr(vis, attr):
            g = getattr(vis, attr)
            if hasattr(g, "offwidth"):
                g.offwidth = max(int(getattr(g, "offwidth", 0)), need_w)
                g.offheight = max(int(getattr(g, "offheight", 0)), need_h)
                print(f"[framebuffer] offwidth={g.offwidth} offheight={g.offheight}")
                return

def load_model_with_board(cfg):
    xml_path = cfg["simulation"]["model_path"]
    spec = mujoco.MjSpec.from_file(xml_path)
    ensure_framebuffer(spec, int(cfg["camera"]["render_width"]), int(cfg["camera"]["render_height"]))

    scene = cfg["board"]["scene"]
    if not scene.get("enabled", True):
        return spec.compile()

    protected_bodies = {"calibration_board", "calib_auto_board"}
    protected_tex    = {"checkerboard_tex", "calib_auto_board_tex"}
    protected_mat    = {"checkerboard_mat", "calib_auto_board_mat"}
    if scene.get("body_name"):     protected_bodies.add(scene["body_name"])
    if scene.get("texture_name"):  protected_tex.add(scene["texture_name"])
    if scene.get("material_name"): protected_mat.add(scene["material_name"])

    def _remove_old(body):
        for i in reversed(range(len(body.bodies))):
            c = body.bodies[i]
            if c.name in protected_bodies: body.bodies.pop(i)
            else: _remove_old(c)
    _remove_old(spec.worldbody)
    for t in list(spec.textures):
        if t.name in protected_tex: spec.textures.remove(t)
    for m in list(spec.materials):
        if m.name in protected_mat: spec.materials.remove(m)

    pos = np.array(scene["pos"], dtype=float)
    euler = np.array(scene["euler"], dtype=float)
    quat = euler_xyz_to_quat(euler)

    tex = spec.add_texture()
    tex.name = scene["texture_name"]; tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.file = os.path.abspath(cfg["board"]["image_path"])

    mat = spec.add_material(); mat.name = scene["material_name"]
    for i in range(len(mat.textures)): mat.textures[i] = ""
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = scene["texture_name"]

    body = spec.worldbody.add_body()
    body.name = scene["body_name"]
    body.pos = [float(v) for v in pos]; body.quat = [float(v) for v in quat]

    geom = body.add_geom()
    geom.name = scene["body_name"] + "_geom"
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    W, H = board_physical_size(cfg["board"])
    geom.size = [W/2.0, H/2.0, float(scene.get("thickness", 0.004))/2.0]
    geom.material = scene["material_name"]
    c = 1 if scene.get("add_collision", False) else 0
    geom.contype = c; geom.conaffinity = c

    model = spec.compile()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene["body_name"])
    print(f"[模型] 注入 '{scene['body_name']}' (id={bid})")
    return model