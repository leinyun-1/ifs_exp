import os
import math
import pickle as pkl
import numpy as np
import taichi as ti
import taichi_three as t3
print(t3.__file__)
from tqdm import tqdm
from taichi_three.transform import rotationX, rotationY


def load_proj_params(pkl_path):
    with open(pkl_path, "rb") as f:
        params = pkl.load(f)
    return params


def get_subject_vids(params, sub):
    prefix = f"{sub}_"
    vids = []
    for k in params.keys():
        if k.startswith(prefix):
            try:
                vids.append(int(k.split("_")[-1]))
            except Exception:
                continue
    return sorted(list(set(vids)))


def add_default_lights(scene, num=6):
    light_dir = np.array([0, 0, 1])
    for l in range(num):
        rotate = np.matmul(
            rotationX(math.radians(np.random.uniform(-30, 30))),
            rotationY(math.radians(360 // num * l)),
        )
        direction = [*np.matmul(rotate, light_dir)]
        scene.add_light(t3.Light(direction, color=[1.0, 1.0, 1.0]))


def set_camera_from_param(camera, intrinsic, extrinsic):
    camera.set_intrinsic(fx=intrinsic[0, 0], fy=intrinsic[1, 1], cx=intrinsic[0, 2], cy=intrinsic[1, 2])
    trans = extrinsic[:, :3]
    T = extrinsic[:, 3]
    pos = -trans.T @ T
    camera.set_extrinsic(trans.T, pos)
    camera._init()


def save_depth(zbuf_field, out_path, zbuf_scale=1.0, save_vis=True, vis_path=None):
    """
    保存真值 depth（= 1 / zbuf）为 npz，同时可选保存 8-bit 可视化图。
    """
    z = zbuf_field.to_numpy().astype(np.float32)
    # 只对有效 z 求倒数，0 保持为 0
    depth = np.zeros_like(z, dtype=np.float32)
    valid = z > 0
    depth[valid] = 1.0 / z[valid]
    # Taichi field is (W, H). Convert to (H, W) for standard image conventions.
    depth = depth.T[::-1,:]
    np.savez_compressed(out_path, depth=depth)

    if save_vis:
        vis = np.clip(z * zbuf_scale, 0.0, 1.0).astype(np.float32)
        ti.imwrite(vis, vis_path)


def _get_mesh_path(mesh_root, sub):
    return os.path.join(mesh_root, sub, f"{sub}.obj")


def _scan_max_n(mesh_root, subjects):
    max_n = 1
    first_obj = None
    for sub in subjects:
        path = _get_mesh_path(mesh_root, sub)
        if not os.path.exists(path):
            continue
        obj = t3.readobj(path, scale=1)
        for key in ["vi", "f", "vt", "vn"]:
            if obj.get(key) is not None:
                max_n = max(max_n, obj[key].shape[0])
        if first_obj is None:
            first_obj = obj
    return max_n, first_obj


def render_depth(
    mesh_root="/home/leinyun/dataset/mesh",
    render_root="/home/leinyun/dataset/Thuman2.1_render_1129",
    res=(2048, 2048),
    zbuf_scale=1.0,
):
    ti.init(ti.cpu)

    proj_path = os.path.join(render_root, "ProjParams.pkl")
    params = load_proj_params(proj_path)

    depth_dir = os.path.join(render_root, "depth")
    depth_vis_dir = os.path.join(render_root, "depth_vis")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(depth_vis_dir, exist_ok=True)

    subjects = [d for d in os.listdir(mesh_root) if os.path.isdir(os.path.join(mesh_root, d))]
    subjects = sorted(subjects)[512:]

    #max_n, first_obj = _scan_max_n(mesh_root, subjects)
    max_n = 1000000
    first_obj = t3.readobj(_get_mesh_path(mesh_root, subjects[0]), scale=1)
    if first_obj is None:
        print(f"[WARN] 未找到可用 mesh: {mesh_root}")
        return

    scene = t3.Scene()
    camera = t3.Camera(res=res)
    scene.add_camera(camera)
    add_default_lights(scene)

    model = t3.StaticModel(max_n, obj=first_obj)
    scene.add_model(model)
    scene.init()

    for sub in tqdm(subjects, desc="subjects"):
        mesh_path = _get_mesh_path(mesh_root, sub)
        if not os.path.exists(mesh_path):
            continue

        obj = t3.readobj(mesh_path, scale=1)
        model.from_obj(obj)
        model.type[None] = 0

        vids = get_subject_vids(params, sub)
        for vid in vids:
            key = f"{sub}_{vid}"
            if key not in params:
                continue
            K = params[key]["K"]
            R = params[key]["R"]
            t = params[key]["t"]
            if K.ndim == 3:
                K = K[0]
            if R.ndim == 3:
                R = R[0]
            if t.ndim == 2:
                t = t[0]

            extrinsic = np.zeros((3, 4), dtype=np.float32)
            extrinsic[:, :3] = R
            extrinsic[:, 3] = t

            set_camera_from_param(camera, K, extrinsic)
            scene.render()

            depth_path = os.path.join(depth_dir, f"{sub}_{vid}.npz")
            depth_vis_path = os.path.join(depth_vis_dir, f"{sub}_{vid}.png")
            save_depth(camera.zbuf, depth_path, zbuf_scale=zbuf_scale, save_vis=True, vis_path=depth_vis_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render mesh depth (zbuf) using ProjParams")
    parser.add_argument("--mesh_root", type=str, default="/home/leinyun/dataset/mesh")
    parser.add_argument("--render_root", type=str, default="/home/leinyun/dataset/Thuman2.1_render_1129")
    parser.add_argument("--res", type=int, default=2048)
    parser.add_argument("--zbuf_scale", type=float, default=1.0)
    args = parser.parse_args()

    render_depth(
        mesh_root=args.mesh_root,
        render_root=args.render_root,
        res=(args.res, args.res),
        zbuf_scale=args.zbuf_scale,
    )
