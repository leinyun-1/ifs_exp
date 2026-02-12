# 现在我需要你在render_multi_1.py同级目录下创建一个render_smpl.py文件。每个sub都有一个对应的smpl mesh，
# 存储在/home/leinyun/dataset/smplx/{sub}/mesh_smplx.obj，然后通过Thuman2.1_render_1129/ProjParams.pkl读取
# 相机参数，利用taichi渲染得到smpl深度图和global法线图，存储在 Thuman2.1_render_1129/smpl_depth/{sub}_{vid}.png  
# 和 Thuman2.1_render_1129/smpl_global_normal/{sub}_{vid}.png


import os
import math
import pickle as pkl
import numpy as np
import taichi as ti
import taichi_three as t3
import trimesh
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


def _rotation_matrix(rx_deg=0.0, ry_deg=0.0, rz_deg=0.0):
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def build_obj_and_normals(obj_path, rotate_xyz=(0.0, 0.0, 0.0)):
    obj = t3.readobj(obj_path, scale=1)
    mesh = trimesh.load(obj_path, force="mesh")

    rx, ry, rz = rotate_xyz
    if rx != 0.0 or ry != 0.0 or rz != 0.0:
        rot = _rotation_matrix(rx, ry, rz)
        if obj["vi"] is not None:
            obj["vi"] = (obj["vi"] @ rot.T).astype(np.float32)
        mesh.vertices = (mesh.vertices @ rot.T).astype(np.float32)

    normals = mesh.vertex_normals.astype(np.float32)
    if obj["vi"].shape[0] != normals.shape[0]:
        # fallback: pad or truncate to match
        n = obj["vi"].shape[0]
        normals = normals[:n]
        if normals.shape[0] < n:
            pad = np.zeros((n - normals.shape[0], 3), dtype=np.float32)
            normals = np.concatenate([normals, pad], axis=0)
    return obj, normals


def add_default_lights(scene, num=6):
    light_dir = np.array([0, 0, 1])
    for l in range(num):
        rotate = np.matmul(
            rotationX(math.radians(np.random.uniform(-30, 30))),
            rotationY(math.radians(360 // num * l)),
        )
        direction = [*np.matmul(rotate, light_dir)]
        scene.add_light(t3.Light(direction, color=[1.0, 1.0, 1.0]))


def set_camera_from_param(camera, intrinsic, extrinsic, res):
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
    depth = depth.T[::-1,:] # Taichi field is (W, H). Convert to (H, W) for standard image conventions.
    np.savez_compressed(out_path, depth=depth)

    if save_vis:
        vis = np.clip(z * zbuf_scale, 0.0, 1.0).astype(np.float32)
        ti.imwrite(vis, vis_path)


def save_normal(normal_field, out_path, normalized=True):
    # 使用 Taichi 写文件，避免轴翻转问题
    if normalized:
        ti.imwrite(normal_field, out_path)
    else:
        normal = (normal_field.to_numpy() + 1.0) / 2.0
        normal = np.clip(normal, 0.0, 1.0).astype(np.float32)
        ti.imwrite(normal, out_path)


def render_smpl(
    smpl_root="/home/leinyun/dataset/smplx",
    render_root="/home/leinyun/dataset/Thuman2.1_render_1129",
    res=(2048, 2048),
    rotate_xyz=(0.0, 0.0, 0.0),
    zbuf_scale=1.0,
):
    ti.init(ti.cpu)

    proj_path = os.path.join(render_root, "ProjParams.pkl")
    params = load_proj_params(proj_path)

    depth_dir = os.path.join(render_root, "smpl_depth_npz")
    depth_vis_dir = os.path.join(render_root, "smpl_depth")
    normal_dir = os.path.join(render_root, "smpl_global_normal")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(depth_vis_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    subjects = [d for d in os.listdir(smpl_root) if os.path.isdir(os.path.join(smpl_root, d))]
    subjects = sorted(subjects)[1:]

    scene = t3.Scene()
    camera = t3.Camera(res=res)
    scene.add_camera(camera)
    add_default_lights(scene)

    # 初始化一次模型，后续仅用 from_obj 替换网格，避免 materialization 错误
    if not subjects:
        return
    first_sub = subjects[0]
    first_path = os.path.join(smpl_root, first_sub, "mesh_smplx.obj")
    if not os.path.exists(first_path):
        return
    first_obj, first_normals = build_obj_and_normals(first_path, rotate_xyz=rotate_xyz)
    model = t3.Model(obj=first_obj, col_n=first_obj["vi"].shape[0])
    model.vc.from_numpy(first_normals)
    model.type[None] = 1  # use vertex color
    scene.add_model(model)
    scene.init()

    for sub in tqdm(subjects, desc="subjects"):
        smpl_path = os.path.join(smpl_root, sub, "mesh_smplx.obj")
        if not os.path.exists(smpl_path):
            continue
        obj, normals = build_obj_and_normals(smpl_path, rotate_xyz=rotate_xyz)
        model.from_obj(obj)
        model.vc.from_numpy(normals)
        model.type[None] = 1  # ensure vertex color mode

        vids = get_subject_vids(params, sub)
        for vid in vids:
            key = f"{sub}_{vid}"
            if key not in params:
                continue
            K = params[key]["K"][0]
            R = params[key]["R"][0]
            t = params[key]["t"][0]
            extrinsic = np.zeros((3, 4), dtype=np.float32)
            extrinsic[:, :3] = R
            extrinsic[:, 3] = t

            set_camera_from_param(camera, K, extrinsic, res)
            scene.render()

            depth = camera.zbuf
            #normal = camera.img

            depth_path = os.path.join(depth_dir, f"{sub}_{vid}.npz")
            depth_vis_path = os.path.join(depth_vis_dir, f"{sub}_{vid}.png")
            normal_path = os.path.join(normal_dir, f"{sub}_{vid}.png")
            save_depth(depth, depth_path, zbuf_scale=zbuf_scale, save_vis=False, vis_path=depth_vis_path)
            #save_normal(normal, normal_path, normalized=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render SMPL depth & global normal")
    parser.add_argument("--smpl_root", type=str, default="/home/leinyun/dataset/smplx")
    parser.add_argument("--render_root", type=str, default="/home/leinyun/dataset/Thuman2.1_render_1129")
    parser.add_argument("--res", type=int, default=2048)
    parser.add_argument("--rotate_x", type=float, default=0.0)
    parser.add_argument("--rotate_y", type=float, default=0.0)
    parser.add_argument("--rotate_z", type=float, default=0.0)
    parser.add_argument("--zbuf_scale", type=float, default=1.0)
    args = parser.parse_args()

    render_smpl(
        args.smpl_root,
        args.render_root,
        res=(args.res, args.res),
        rotate_xyz=(args.rotate_x, args.rotate_y, args.rotate_z),
        zbuf_scale=args.zbuf_scale,
    )
