import taichi as ti
import taichi_three as t3
import numpy as np
from taichi_three.transform import *
from tqdm import tqdm
import os
import time
import cv2
import json
import pickle as pkl
import multiprocessing as mp
from functools import partial
import math
import traceback
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

def find_border(img):
    img_1 = np.sum(img, axis=2)
    img_x = np.sum(img_1, axis=0)
    img_y = np.sum(img_1, axis=1)
    x_min = img_x.shape[0]
    x_max = 0
    y_min = img_y.shape[0]
    y_max = 0
    for x in range(img_x.shape[0]):
        if img_x[x] > 0:
            x_min = x
            break
    for x in range(img_x.shape[0]-1, 0, -1):
        if img_x[x] > 0:
            x_max = x
            break
    for y in range(img_y.shape[0]):
        if img_y[y] > 0:
            y_min = y
            break
    for y in range(img_y.shape[0]-1, 0, -1):
        if img_y[y] > 0:
            y_max = y
            break
    return x_min, x_max, y_min, y_max

class StaticRenderer:
    def __init__(self):
        ti.init(ti.cpu)
        self.scene = t3.Scene()
        self.N = 10
    
    def change_all(self):
        save_obj = []
        save_tex = []
        for model in self.scene.models:
            save_obj.append(model.init_obj)
            save_tex.append(model.init_tex)
        ti.init(ti.cpu)
        print('init')
        self.scene = t3.Scene()
        for i in range(len(save_obj)):
            model = t3.StaticModel(self.N, obj=save_obj[i], tex=save_tex[i])
            self.scene.add_model(model)

    def check_update(self, obj):
        temp_n = self.N
        self.N = max(obj['vi'].shape[0], self.N)
        self.N = max(obj['f'].shape[0], self.N)
        if not (obj['vt'] is None):
            self.N = max(obj['vt'].shape[0], self.N)

        if self.N > temp_n:
            self.N *= 2
            self.change_all()
            self.camera_light()
    
    def add_model(self, obj, tex=None):
        self.check_update(obj)
        model = t3.StaticModel(self.N, obj=obj, tex=tex)
        self.scene.add_model(model)
    
    def modify_model(self, index, obj, tex=None):
        self.check_update(obj)
        self.scene.models[index].init_obj = obj
        self.scene.models[index].init_tex = tex
        self.scene.models[index]._init()
    
    def camera_light(self):
        camera = t3.Camera(res=res)
        camera1 = t3.Camera(res=res)
        self.scene.add_camera(camera)
        self.scene.add_camera(camera1)
        light_dir = np.array([0, 0, 1])
        for l in range(6):
            rotate = np.matmul(rotationX(math.radians(np.random.uniform(-30, 30))),
                            rotationY(math.radians(360 // 6 * l)))
            dir = [*np.matmul(rotate, light_dir)]
            light = t3.Light(dir, color=[1.0, 1.0, 1.0])
            self.scene.add_light(light)

def set_lights(scene, off_count=2):
    """在已有 6 盏灯中随机关闭 off_count 盏，其余方向/强度保持初始化时的值。"""
    total_lights = len(scene.lights)
    off_count = min(off_count, total_lights)
    off_idxs = set(random.sample(range(total_lights), off_count))

    for idx, light in enumerate(scene.lights):
        if idx in off_idxs:
            light.color[None] = ti.Vector([0.0, 0.0, 0.0])
        else:
            # 保持原有方向与强度
            light.color[None] = ti.Vector([1, 1, 1])

def render_mv(renderer, data_path, param, num_views, data_id, save_path, res=(1024, 1024), enable_gpu=False, dis_scale=1):
    img_path = os.path.join(data_path, data_id,  'material_0.jpeg')
    obj_path = os.path.join(data_path, data_id, data_id + '.obj')
    
    ### 此部分代码会无效重复
    img_save_path = os.path.join(save_path, 'image')
    mask_save_path = os.path.join(save_path, 'mask')
    normal_save_path = os.path.join(save_path, 'normal')
    #parameter_save_path = os.path.join(save_path, 'ProjParams.pkl')
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    # if not os.path.exists(parameter_save_path):
    #     os.makedirs(parameter_save_path)
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)
    if not os.path.exists(normal_save_path):
        os.makedirs(normal_save_path)
    ###
        
    texture = ti.imread(img_path)
    obj = t3.readobj(obj_path, scale=1)
    
    if len(renderer.scene.models) >= 1:
        renderer.modify_model(0, obj, texture)
    else:
        renderer.add_model(obj, texture)
    
    vi = obj['vi']
    median = np.median(vi, axis=0)  # + (np.random.randn(3) - 0.5) * 0.2
    vmin = vi.min(0)
    vmax = vi.max(0)
    median[1] = (vmax[1] * 4 + vmin[1] * 3) / 7

    r_color = np.zeros((obj['vi'].shape[0], 3))
    r_color[:, 0] = 1
    renderer.scene.models[0].modify_color(r_color)

    # 三种光照组合
    light_setups = [
        [[0.0, 0.0, 1.0]],
        [[0.2, 0.2, 1.0]],
        [[-0.2, 0.1, 1.0]],
    ]
    max_lights = max(len(ls) for ls in light_setups)

    angle_mul = 360 / num_views
    for vid in tqdm(range(num_views), desc='vid'):
        renderer.scene.models[0].type[None] = 0
        dis = vmax[1] - vmin[1]
        dis *= dis_scale
        ori_vec = np.array([0, 0, dis])
        # 俯仰抖动
        pitch = random.uniform(-30.0, 10.0)
        rotate = np.matmul(rotationY(math.radians(vid * angle_mul)), rotationX(math.radians(pitch)))
        fwd = np.matmul(rotate, ori_vec)
        target = median
        pos = target + fwd

        if vid == 0:
            set_lights(renderer.scene)
            # 第一个视角计算相机内参
            fx = res[0] * 0.5
            fy = res[1] * 0.5
            cx = fx
            cy = fy
            renderer.scene.cameras[0].set(pos=pos, target=target)
            renderer.scene.cameras[0].set_intrinsic(fx, fy, cx, cy)
            renderer.scene.cameras[0]._init()
            renderer.scene.single_render(0)

            img = renderer.scene.cameras[0].img.to_numpy()
            x_min, x_max, y_min, y_max = find_border(img)

            x_min -= 20
            x_max += 20
            x_len = x_max - x_min
            y_min = (y_max + y_min - x_len) // 2
            scale = res[0] / x_len
            fx = res[0] / 2 * scale
            fy = res[1] / 2 * scale
            cx = scale * (cx - y_min)
            cy = scale * (cy - x_min)

        # 复用第一个视角的内参
        renderer.scene.cameras[1].set_intrinsic(fx, fy, cx, cy)
        renderer.scene.cameras[1].set(pos=pos, target=target)
        renderer.scene.cameras[1]._init()

        # 渲染并保存图像和其他数据（每个视角三种光照）
        for lid, dirs in enumerate(light_setups):
            set_lights(renderer.scene)
            renderer.scene.render()
            camera1 = renderer.scene.cameras[1]
            ti.imwrite(camera1.img, os.path.join(img_save_path, f"{data_id}_{vid}_{lid}.png"))
            ti.imwrite(camera1.normal_map, os.path.join(normal_save_path, f"{data_id}_{vid}_{lid}_normal.png"))

            # 保存相机参数（按视角记录一次即可）
            if lid == 0:
                name = data_id + '_' + str(vid)
                extrinsic = camera1.export_extrinsic()
                intrinsic = camera1.export_intrinsic()
                param[name] = {}
                param[name]['K'] = intrinsic[None]
                param[name]['R'] = extrinsic[:3, :3][None]
                param[name]['t'] = extrinsic[:, 3][None]

            # 保存掩码
            renderer.scene.models[0].type[None] = 1
            renderer.scene.render()
            camera1 = renderer.scene.cameras[1]
            mask = camera1.img.to_numpy()
            ti.imwrite(mask, os.path.join(mask_save_path, f"{data_id}_{vid}_{lid}_mask.png"))
            renderer.scene.models[0].type[None] = 0

def worker_render(data_id, data_root, save_path, res, num_views, dis_scale):
    """单进程渲染任务，输出单独 pkl 文件"""
    renderer = StaticRenderer()  # 每个进程自己的 Renderer
    param_local = {}
    pkl_path = os.path.join(save_path,'param',f"{data_id}_params.pkl")
    if os.path.exists(pkl_path):
        print(f"[Warning] {data_id} 已存在参数文件，跳过渲染")
        return

    try:
        render_mv(renderer, data_root, param_local, num_views, data_id, save_path, res, False, dis_scale)
    except Exception as e:
        print(f"[Error] {data_id} 渲染失败: {e}")
        traceback.print_exc()
        return

    # 保存单个 pkl 文件
    os.makedirs(os.path.join(save_path, 'param'), exist_ok=True)
    pkl_path = os.path.join(save_path,'param',f"{data_id}_params.pkl")  
    with open(pkl_path, 'wb') as f:
        pkl.dump(param_local, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/leinyun/winshare_1/dataset/Thuman2.1_norm")
    parser.add_argument("--save_path", type=str, default='/home/leinyun/dataset/Thuman2.1_render_1129')
    parser.add_argument("--res", type=int, default=2048)
    parser.add_argument("--views", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    data_root = args.data_root
    save_path = args.save_path
    res = (args.res, args.res)
    num_views = args.views
    os.makedirs(save_path, exist_ok=True)

    # 获取所有任务
    data_ids = os.listdir(data_root)

    # 多进程执行
    with mp.Pool(processes=args.num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(
                partial(worker_render,
                        data_root=data_root,
                        save_path=save_path,
                        res=res,
                        num_views=num_views,
                        dis_scale=2),
                data_ids
            ),
            total=len(data_ids),
            desc="Rendering"
        ))

    # 合并所有 pkl
    param_final = {}
    param_root = os.path.join(save_path, 'param')
    for fname in os.listdir(param_root):
        if fname.endswith("_params.pkl"):
            with open(os.path.join(param_root, fname), 'rb') as f:
                param_final.update(pkl.load(f))

    # 保存总文件
    with open(os.path.join(save_path, 'ProjParams.pkl'), 'wb') as f:
        pkl.dump(param_final, f)

    print(f"✅ 渲染完成，合并 {len(param_final)} 条相机参数")
