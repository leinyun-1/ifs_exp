from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 数据增强测试：复用 BaseIFS 的增强逻辑
from Dataset.BaseIFS import BaseIFS


class _AugmentOnly(BaseIFS):
    """仅为测试数据增强而构造的轻量类，不依赖数据目录。"""

    def __init__(self):
        # 不调用父类 __init__，只设置增强所需字段
        self.is_train = True
        self.roi = 512
        self.aug_cfg = {
            "enable": True,
            "color_jitter": {"p": 1.0, "brightness": 0.2, "contrast": 0.2, "saturation": 0.2},
            "gaussian_blur": {"p": 1.0, "radius": 1.0},
            "gaussian_noise": {"p": 1.0, "std": 0.01},
            "random_mask_drop": {"p": 1.0, "max_ratio": 0.2},
            "geom_jitter": {"enable": False, "p": 0.7, "pad_ratio": 0.1, "scale_min": 0.9, "scale_max": 1.1, "trans_ratio": 0.1},
        }

    def run(self, image: Image.Image, mask: Image.Image):
        # 根据输入动态更新 roi，保持几何抖动参数合理
        self.roi = min(image.size)
        image, mask, _ = self._augment_render(image, mask, intrinsic=np.eye(3))
        return image, mask


def test_HG():
    import torch 
    from Model.HourGlass import HGFilter
    from types import SimpleNamespace
    config = {
        "num_stack": 4,
        "num_hourglass": 2,
        "hourglass_dim": 256,
        "hg_down": 'ave_pool',
        "norm": 'group'
    }
    config = SimpleNamespace(**config)
    net = HGFilter(config).cuda()
    x = torch.randn(1,3,512,512).cuda()
    y = net(x)
    print(y.shape)

def test_mlp():
    import torch 
    from Model.SurfaceClassifier import SurfaceClassifier
    chnnels = [323, 1024, 512, 256, 128, 1]
    net = SurfaceClassifier(chnnels).cuda()
    x = torch.randn(1,323,8000).cuda()
    y = net(x)
    print(y.shape)

def make_train_txt():
    file_path = "/home/leinyun/dataset/Thuman2.1_render_1129/test1.txt"
    with open(file_path,'w') as f:
        for i in range(2152,2158):
            f.write(f"{i:04d}\n")

def test_pifu():
    import torch 
    from Model.Pifu import IFSNet
    from Dataset.PifuIFS import get_dataloader,ifs_pack

    net = IFSNet(tm=None,rov=-1).cuda()
    dataset = get_dataloader(path="/root/leinyu/data/thuman2/ft_local/dataset ../grid_samples_64_12")
    dataloader = torch.utils.data.DataLoader(dataset)
    for b_data in dataloader:
        b_data = ifs_pack('cuda',b_data)
        print(b_data['images'].shape)
        y,loss,_ = net.forward(epoch=0,bidx=0,data=b_data)
        print(y.shape)
        print(loss)
        break 

def make_dataset():
    import os 
    import shutil
    import pickle 
    from PIL import Image 
    dataset_root = '/root/leinyu/data/thuman2/ft_local/dataset/Res2048/0418'
    image_root = os.path.join(dataset_root,'image')
    mask_root = os.path.join(dataset_root,'mask')
    param_path = os.path.join(dataset_root,'ProjParams.pkl')

    eval_image_root = '/mnt/aigc_cq/private/leinyu/code/skyreels_v2/result/i2v_1.3b_lora_0822/0825/images'
    eval_images = sorted(os.listdir(eval_image_root))
    eval_images = eval_images[::10][:-1] # 从0-80共81图取出均匀环绕8图
    for i,image in enumerate(eval_images):
        src_path = os.path.join(eval_image_root,image)
        dest_path = os.path.join(image_root,'woman_'+ str(i*6)+'.png')
        dest_mask_path = os.path.join(mask_root,'woman_'+str(i*6)+'_mask.png')

        Image.open(src_path).resize((2048,2048), resample=Image.NEAREST).save(dest_path)
        Image.open(src_path).resize((2048,2048), resample=Image.NEAREST).save(dest_mask_path)
        #shutil.copy(src_path,dest_path)

    # cam_params = pickle.load(open(param_path, "rb"), encoding="iso-8859-1")

    # views = [0, 6, 12, 18, 24, 30, 36, 42]
    # for vid in views:
    #     param = {}
    #     param['K'] = cam_params['0000_'+str(vid)]['K']
    #     param['R'] = cam_params['0000_'+str(vid)]['R']
    #     param['t'] = cam_params['0000_'+str(vid)]['t']
    #     cam_params['woman_' + str(vid)] = param.copy()
    
    # # 回存更新后的相机参数到原始路径
    # with open(param_path, 'wb') as f:
    #     pickle.dump(cam_params, f)
    
    

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from concurrent.futures import ProcessPoolExecutor, as_completed

def schedule_centers(bmin, size, rom, rov, dia, stride):
    sos = size / rom
    candates = np.mgrid[0:rom-rov+1:stride, 0:rom-rov+1:stride, 0:rom-rov+1:stride].reshape(3, -1)
    deltas = np.mgrid[0:dia, 0:dia, 0:dia].reshape(3, -1)
    indices, centers = [], []
    for i in range(candates.shape[1]):
        for j in range(deltas.shape[1]):
            idx = candates[:, i]
            delta = deltas[:, j]
            center = bmin + sos * idx + sos * rov / 2 + sos * delta / dia
            indices.append(idx * dia + delta)
            centers.append(center)
    return np.array(indices), np.array(centers)

def draw_boxes(ax, centers, rov, sos, dia, color='blue', alpha=0.05):
    # 在细网格坐标下画出一个 rov^3 的方块投影（仅示意，不画全部）
    for c in centers:
        # 这里 c 是真实坐标，转换到细网格坐标：
        # fine_coord = (c - bmin) / sos * dia   （原始 code 里 labels shape = rom*dia）
        pass

def visualize(bmin=np.array([0,0,0]), size=np.array([1,1,1]), rom=128, rov=32, stride=32, dia=2):
    sos = size / rom
    idx, ctr = schedule_centers(bmin, size, rom, rov, dia, stride)
    # 转到细网格整数坐标，便于理解“放大”后的位置
    fine_centers = (ctr - bmin) / sos * dia
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(fine_centers[:,0], fine_centers[:,1], fine_centers[:,2], c='r', s=10, label='centers (fine grid)')
    ax.set_xlabel('x (fine vox)')
    ax.set_ylabel('y (fine vox)')
    ax.set_zlabel('z (fine vox)')
    ax.set_title(f'rom={rom}, rov={rov}, stride={stride}, dia={dia}\nlabels grid size={(rom*dia)}^3')
    ax.legend()
    plt.show()

def run_augment(image_path: Path, mask_path: Path, out_dir: Path, samples: int = 4):
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")

    augmenter = _AugmentOnly()
    # 启用几何抖动以覆盖所有增强功能
    augmenter.aug_cfg["geom_jitter"]["enable"] = True

    for idx in range(samples):
        aug_img, aug_mask = augmenter.run(image, mask)
        aug_img.save(out_dir / f"aug_image_{idx}.png")
        aug_mask.save(out_dir / f"aug_mask_{idx}.png")
    print(f"保存增强结果到 {out_dir}, 共 {samples} 组")


from Dataset.ScenaroIFS import ScenaroIFS
from Dataset.PathedIFS import PathedIFS
def test_dataset_py():
    # 按你的真实路径填写，格式: "<root_path> <grid_sample_path>"
    data_path = "/home/leinyun/dataset ../grid_samples_48_16"
    ds = PathedIFS(path=data_path, split="train",  roi=1536, nov=8, yaw_list=[0,2,4,6,8,10,12,14], dataset_type="A+")

    # 取一个样本验证 shape
    sample = ds[10]
    print("images shape:", sample["images"][0].shape)          # [V,3,H,W]
    print("projection shape:", sample["projection"].shape)  # [V,3,4]

    outdir = '../tmp/ifs_dataset_debug'
    os.makedirs(outdir,exist_ok=True)
    for idx,img in enumerate(sample["images"]):
        img = img.permute(1,2,0).numpy()
        Image.fromarray((img*255).astype(np.uint8)).save(outdir + '/' + f"dataset_debug_{idx}.png")


def test_data_set_file(render_dir='/home/leinyun/dataset/Thuman2.1_render_1129'):
    """
    校验多光照 A 型数据：
    1) param/{sub}_params.pkl 大小是否一致；
    2) image/ 下 {sub}_{vid}_{lid}.png 覆盖是否完整（lid 期望 {0,1,2}），无多余。
    """
    import os
    import glob
    import pickle

    param_dir = os.path.join(render_dir, "param")
    image_dir = os.path.join(render_dir, "image")

    param_files = sorted(glob.glob(os.path.join(param_dir, "*_params.pkl")))
    if not param_files:
        print(f"[WARN] 未找到 param 文件: {param_dir}")
        return

    # 1) pkl 文件大小一致性
    sizes = {os.path.basename(p): os.path.getsize(p) for p in param_files}
    uniq = set(sizes.values())
    if len(uniq) == 1:
        print(f"[OK] param 文件大小一致，共 {len(param_files)} 个，大小 {uniq.pop()} 字节")
    else:
        print("[WARN] param 文件大小不一致：")
        for name, sz in sizes.items():
            print(f"  {name}: {sz}")

    # 2) image 覆盖情况
    expected_lids = {0, 1, 2}
    missing, extra = [], []

    for pkl_path in param_files:
        sub = os.path.basename(pkl_path).replace("_params.pkl", "")
        with open(pkl_path, "rb") as f:
            params = pickle.load(f)
        vids = set(int(k.split("_")[-1]) for k in params.keys())
        for vid in vids:
            files = glob.glob(os.path.join(image_dir, f"{sub}_{vid}_*.png"))
            lids_found = set()
            for fp in files:
                try:
                    lids_found.add(int(os.path.splitext(fp)[0].split("_")[-1]))
                except Exception:
                    continue
            miss = expected_lids - lids_found
            if miss:
                missing.append((sub, vid, sorted(list(miss))))
            if len(lids_found) > len(expected_lids):
                extras = sorted(list(lids_found - expected_lids))
                if extras:
                    extra.append((sub, vid, extras))

    if not missing and not extra:
        print("[OK] image 覆盖完整，每个 (sub, vid) 有 lid 0/1/2 且无多余。")
    else:
        if missing:
            print("[MISS] 缺失：")
            for sub, vid, lids in missing:
                print(f"  {sub} vid={vid} 缺 lid={lids}")
        if extra:
            print("[EXTRA] 多余：")
            for sub, vid, lids in extra:
                print(f"  {sub} vid={vid} 多余 lid={lids}")


import os
import glob
import trimesh
def _check_mesh_bbox_task(path, threshold):
    try:
        mesh = trimesh.load(path, force='mesh')
        bmin = mesh.bounds[0]
        bmax = mesh.bounds[1]
        bad = (bmin < -threshold).any() or (bmax > threshold).any()
        return path, bmin, bmax, bad, None
    except Exception as e:
        return path, None, None, True, str(e)


def test_mesh_bbox(mesh_dir, threshold=1.5, workers=None):
    """
    检查 mesh_dir/{sub}/{sub}.obj 的 bounding box 范围，找出异常 mesh。
    判定标准：任一轴的 min/max 超出 [-threshold, threshold] 视为异常。
    支持多进程 workers 加速，默认为 CPU 核数。
    """

    objs = glob.glob(os.path.join(mesh_dir, "*", "*.obj"))
    if not objs:
        print(f"[WARN] 未找到 obj 文件于 {mesh_dir}")
        return

    if workers is None:
        workers = os.cpu_count() or 4

    bad = []
    from tqdm import tqdm
    if workers == 1:
        for path in tqdm(objs, desc="checking bbox"):
            res = _check_mesh_bbox_task(path, threshold)
            if res[3]:  # bad
                bad.append((res[0], res[1], res[2], res[4]))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_check_mesh_bbox_task, path, threshold): path for path in objs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="checking bbox"):
                path, bmin, bmax, is_bad, err = fut.result()
                if is_bad:
                    bad.append((path, bmin, bmax, err))

    if not bad:
        print(f"[OK] 所有 mesh 的 bbox 均在 [{-threshold}, {threshold}] 范围内")
    else:
        print("[WARN] 检测到异常 bbox：")
        for path, bmin, bmax, err in bad:
            print(f"  {path}")
            print(f"    bmin: {bmin}, bmax: {bmax}, err: {err}")



if __name__ == "__main__":
    # 手动设置模式与参数，避免命令行冗余
    MODE = "augment"  # 取值 "augment" 或 "visualize"

    # 数据增强参数
    IMAGE_PATH = Path("/home/leinyun/winshare_1/dataset/Thuman2.1_render/img/0000/0.png")
    MASK_PATH = Path("/home/leinyun/winshare_1/dataset/Thuman2.1_render/mask/0000/0.png")
    OUTPUT_DIR = Path("./tmp/ifs_augmented")
    SAMPLES = 4

    make_train_txt()
    #visualize(rom=512, rov=64, stride=32, dia=2)
    #run_augment(IMAGE_PATH, MASK_PATH, OUTPUT_DIR, SAMPLES)
    #test_dataset_py()
    #test_data_set_file()
    #test_mesh_bbox('/home/leinyun/dataset/mesh',workers=4)
    
