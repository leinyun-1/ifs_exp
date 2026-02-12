import os
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import trimesh
import random

"""
支持多视角数据集组织：
 A（默认）
    root/
        render/
            image/{sub}_{vid}.png
            mask/{sub}_{vid}_mask.png
            ProjParams.pkl  # 集中存储 K/R/t
 A+（多光照）
    root/
        render/
            image/{sub}_{vid}_{lid}.png  # lid ∈ {0,1,2}，随机选一张
            mask/{sub}_{vid}_{lid}_mask.png
            ProjParams.pkl  # 键仍为 {sub}_{vid}
 B
    root/
        img/{sub}/{vid}.png
        mask/{sub}/{vid}.png
        parameter/{sub}/{vid}_intrinsic.npy  # 3x3
        parameter/{sub}/{vid}_extrinsic.npy  # 3x4
"""


class BaseIFS(Dataset):
    def __init__(self, path, split, roi, nov, yaw_list, dataset_type="A"):
        super(BaseIFS, self).__init__()

        self.is_train = split == "train"

        self.roi = roi
        self.nov = nov
        self.yaw_list = yaw_list
        self.dataset_type = dataset_type

        [path, grid_sample_path] = path.split(' ')

        # Path setup
        if self.dataset_type in ("A", "A+"):
            self.dir_render = path
            self.dir_images = os.path.join(self.dir_render, "image")
            self.dir_params = os.path.join(self.dir_render, "ProjParams.pkl")
            self.dir_masks = os.path.join(self.dir_render, "mask")
            self.dir_points = os.path.join(self.dir_render, "pts")
            self.dir_depths = os.path.join(self.dir_render, "depth")
            self.dir_normal = os.path.join(self.dir_render, "normal")
            self.dir_texture = os.path.join(self.dir_render, "texture")
            self.dir_colors = os.path.join(self.dir_render, "color")

            self.dir_meshes = os.path.join(path, "../","mesh")
            self.dir_smplxs = os.path.join(path, "../","smplx")

        else:
            # dataset B
            self.dir_render = os.path.join(path, "Thuman2.1_render_1129")
            self.dir_images = os.path.join(self.dir_render, "img")
            self.dir_masks = os.path.join(self.dir_render, "mask")
            self.dir_params_root = os.path.join(self.dir_render, "parameter")
            self.dir_points = os.path.join(self.dir_render, "pts")
            self.dir_depths = os.path.join(self.dir_render, "depth")
            self.dir_normal = os.path.join(self.dir_render, "normal")
            self.dir_texture = os.path.join(self.dir_render, "texture")
            self.dir_colors = os.path.join(self.dir_render, "color")

            self.dir_meshes = os.path.join('/home/leinyun/winshare_1/dataset/', "Thuman2.1_norm")
            self.dir_smplxs = os.path.join('/home/leinyun/winshare_1/dataset/', "Thuman2.1_smplx_norm")

        self.dir_samples = grid_sample_path

        if self.dataset_type in ("A", "A+"):
            if self.is_train:
                dir_subjects = os.path.join(self.dir_render, "train.txt")
            else:
                dir_subjects = os.path.join(self.dir_render, f"{split}.txt") 
                #dir_subjects = os.path.join(self.dir_render, "test1.txt") # 测试特殊样例
            self.subjects = sorted(list(set(np.loadtxt(dir_subjects, dtype=str))))
            self.cam_params = pickle.load(open(self.dir_params, "rb"), encoding="iso-8859-1")
        else:
            # 从子目录名推断 subject
            self.subjects = sorted([d for d in os.listdir(self.dir_images) if os.path.isdir(os.path.join(self.dir_images, d))])
            self.cam_params = None

        # 数据增强开关，可通过 set_augmentations 进行动态调整
        self.aug_cfg = {
            "enable": True,
            "color_jitter": {"p": 0.3, "brightness": 0.2, "contrast": 0.2, "saturation": 0.2},
            "gaussian_blur": {"p": 0.2, "radius": 1.0},
            "gaussian_noise": {"p": 0.3, "std": 0.01},
            "random_mask_drop": {"p": 0.2, "max_ratio": 0.2},
            "geom_jitter": {
                "enable": True,
                "pad_ratio": 0.1,
                "pad_p": 0.3,
                "scale_min": 0.9,
                "scale_max": 1.1,
                "scale_p": 0.3,
                "trans_ratio": 0.1,
                "trans_p": 1.0,
            },
        }

    def _load_images(self, subject, vid):
        if self.dataset_type in ("A", "A+"):
            lid = None
            if self.dataset_type == "A+":
                lid = random.choice([0, 1, 2])
            suffix = f"_{vid}" if lid is None else f"_{vid}_{lid}"

            image_path = os.path.join(self.dir_images, f"{subject}{suffix}.png")
            if not os.path.exists(image_path):
                image_path = os.path.join(self.dir_images, f"{subject}{suffix}.jpg")
            image = Image.open(image_path).convert("RGB")

            mask_path = os.path.join(self.dir_masks, f"{subject}{suffix}_mask.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(self.dir_masks, f"{subject}{suffix}_mask.jpg")
        else:
            image_path = os.path.join(self.dir_images, subject, "{}.png".format(vid))
            if not os.path.exists(image_path):
                image_path = os.path.join(self.dir_images, subject, "{}.jpg".format(vid))
            image = Image.open(image_path).convert("RGB")

            mask_path = os.path.join(self.dir_masks, subject, "{}.png".format(vid))
            if not os.path.exists(mask_path):
                mask_path = os.path.join(self.dir_masks, subject, "{}.jpg".format(vid))

        mask = Image.open(mask_path).convert("RGB")
        return image, mask

    def _load_projection(self, subject, vid):
        if self.dataset_type in ("A", "A+"):
            K = self.cam_params[subject + "_" + str(vid)]["K"]
            R = self.cam_params[subject + "_" + str(vid)]["R"]
            t = self.cam_params[subject + "_" + str(vid)]["t"]
            intrinsic = np.array(K[0])
            extrinsic = np.zeros([3, 4])
            extrinsic[:, :3] = R
            extrinsic[:, 3] = t
        else:
            intrinsic_path = os.path.join(self.dir_params_root, subject, f"{vid}_intrinsic.npy")
            extrinsic_path = os.path.join(self.dir_params_root, subject, f"{vid}_extrinsic.npy")
            intrinsic = np.load(intrinsic_path)
            extrinsic = np.load(extrinsic_path)

        return intrinsic, extrinsic

    def _scale_render(self, intrinsic, render, roi):
        w, h = render[0].size
        intrinsic[0, :] = intrinsic[0, :] * (roi / w)
        intrinsic[1, :] = intrinsic[1, :] * (roi / h)

        for i, img in enumerate(render):
            render[i] = img.resize((roi, roi), Image.BILINEAR)

        return intrinsic, render

    def _crop_render(self, intrinsic, render, bbox):
        bbox = tuple(bbox)
        for i, img in enumerate(render):
            # PIL的crop会做padding
            render[i] = img.crop(bbox)

        intrinsic[0, -1] = intrinsic[0, -1] - bbox[0]
        intrinsic[1, -1] = intrinsic[1, -1] - bbox[1]
        return intrinsic, render

    def set_augmentations(self, **kwargs):
        """可插拔的增强配置入口，kwargs 中的键覆盖 self.aug_cfg。"""
        for key, value in kwargs.items():
            if key in self.aug_cfg:
                self.aug_cfg[key] = value

    def _maybe_color_jitter(self, image):
        cfg = self.aug_cfg["color_jitter"]
        if random.random() > cfg["p"]:
            return image
        b = 1 + random.uniform(-cfg["brightness"], cfg["brightness"])
        c = 1 + random.uniform(-cfg["contrast"], cfg["contrast"])
        s = 1 + random.uniform(-cfg["saturation"], cfg["saturation"])
        image = ImageEnhance.Brightness(image).enhance(b)
        image = ImageEnhance.Contrast(image).enhance(c)
        image = ImageEnhance.Color(image).enhance(s)
        return image

    def _maybe_gaussian_blur(self, image):
        cfg = self.aug_cfg["gaussian_blur"]
        if random.random() > cfg["p"]:
            return image
        radius = random.uniform(0.1, cfg["radius"])
        return image.filter(ImageFilter.GaussianBlur(radius))

    def _maybe_gaussian_noise(self, image):
        cfg = self.aug_cfg["gaussian_noise"]
        if random.random() > cfg["p"]:
            return image
        arr = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(0.0, cfg["std"], arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    def _maybe_mask_drop(self, mask):
        cfg = self.aug_cfg["random_mask_drop"]
        if random.random() > cfg["p"]:
            return mask
        w, h = mask.size
        drop_w = int(w * random.uniform(0.05, cfg["max_ratio"]))
        drop_h = int(h * random.uniform(0.05, cfg["max_ratio"]))
        if drop_w <= 0 or drop_h <= 0:
            return mask
        x0 = random.randint(0, max(0, w - drop_w))
        y0 = random.randint(0, max(0, h - drop_h))
        mask_np = np.array(mask)
        mask_np[y0 : y0 + drop_h, x0 : x0 + drop_w, :] = 0
        return Image.fromarray(mask_np)

    def _augment_render(self, image, mask, intrinsic=None):
        """按开关式配置对 image/mask/内参做可选增强，仅在训练模式启用。"""
        if not self.is_train or not self.aug_cfg.get("enable", False):
            return image, mask, intrinsic

        intrinsic_input = intrinsic
        if intrinsic is None:
            intrinsic = np.eye(3)

        # 几何抖动（pad/scale/trans + 同步内参）
        image, mask, intrinsic = self._maybe_geom_jitter(image, mask, intrinsic)

        # 光学类增强
        image = self._maybe_color_jitter(image)
        image = self._maybe_gaussian_blur(image)
        image = self._maybe_gaussian_noise(image)
        mask = self._maybe_mask_drop(mask)

        if intrinsic_input is None:
            intrinsic = None
        return image, mask, intrinsic

    def _maybe_geom_jitter(self, image, mask, intrinsic):
        """几何抖动：pad + 随机缩放 + 随机平移 + 中心裁剪，配合内参同步调整。"""
        cfg = self.aug_cfg["geom_jitter"]
        if not cfg.get("enable", False):
            return image, mask, intrinsic

        tw,th = image.size
        pad_ratio = cfg.get("pad_ratio", 0.1)
        scale_min = cfg.get("scale_min", 0.9)
        scale_max = cfg.get("scale_max", 1.1)
        trans_ratio = cfg.get("trans_ratio", 0.1)
        pad_p = cfg.get("pad_p", 1.0)
        scale_p = cfg.get("scale_p", 1.0)
        trans_p = cfg.get("trans_p", 1.0)

        pad_size = int(pad_ratio * tw)
        imgs = [image, mask]
        if pad_size > 0 and random.random() <= pad_p:
            for idx, img in enumerate(imgs):
                imgs[idx] = ImageOps.expand(img, pad_size, fill=0)
            # pad 会平移主点
            intrinsic = intrinsic.copy()
            intrinsic[0, 2] += pad_size
            intrinsic[1, 2] += pad_size

        w, h = imgs[0].size
        rand_scale = 1.0
        if random.random() <= scale_p:
            rand_scale = random.uniform(scale_min, scale_max)
            w = int(rand_scale * w)
            h = int(rand_scale * h)
            for idx, img in enumerate(imgs):
                imgs[idx] = img.resize((w, h), Image.BILINEAR)

            intrinsic = intrinsic.copy()
            intrinsic[0, :] *= rand_scale
            intrinsic[1, :] *= rand_scale

        if trans_ratio > 0 and random.random() <= trans_p:
            dx_lim = int(round((w - tw) * trans_ratio))
            dy_lim = int(round((h - th) * trans_ratio))
            dx = random.randint(-dx_lim, dx_lim) if dx_lim > 0 else 0
            dy = random.randint(-dy_lim, dy_lim) if dy_lim > 0 else 0
        else:
            dx = dy = 0


        x1 = int(round((w - tw) / 2.0)) + dx
        y1 = int(round((h - th) / 2.0)) + dy
        for idx, img in enumerate(imgs):
            imgs[idx] = img.crop((x1, y1, x1 + tw, y1 + th))

        intrinsic[0, 2] += -x1
        intrinsic[1, 2] += -y1

        return imgs[0], imgs[1], intrinsic

    def _load_renders(self, subject, cropper, yid=0):
        projection_list = []
        image_list = []

        # The ids are an even distribution of nov around view_id
        view_ids = [
            self.yaw_list[(yid + len(self.yaw_list) // self.nov * offset) % len(self.yaw_list)]
            for offset in range(self.nov)
        ]
        for vid in view_ids:
            image, mask = self._load_images(subject, vid)
            intrinsic, extrinsic = self._load_projection(subject, vid)

            render = [image, mask]
            intrinsic, render = self._scale_render(intrinsic, render, self.roi)

            #HW参考系在右上角。
            intrinsic[1, :] *= -1.0
            intrinsic[1, 2] += self.roi

            if cropper != None :
                bbox = cropper(intrinsic, extrinsic, subject, render)
                intrinsic, render = self._crop_render(intrinsic, render, bbox)

            image, mask = render
            image, mask, intrinsic = self._augment_render(image, mask, intrinsic)
            # 小心 PIL 类型
            w, h = mask.size

            mask = torch.sum(torch.FloatTensor((np.array(mask).reshape((h, w, -1)))), dim=2) / 255
            mask[mask > 0.1] = 1.0  # 识别背景全黑区域，前景并非需要rgb之和大于255

            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1) * mask.reshape(1, h, w)

            image_list.append(image)
            projection_list.append(torch.Tensor(np.matmul(intrinsic, extrinsic)).float())

        return {"images": image_list, "projection": torch.stack(projection_list, dim=0)}

    @property
    def _num_subjects(self):
        return len(self.subjects)

    def _get_subject(self, sid):
        return self.subjects[sid]

    def _get_smplx(self, subject):
        smplx_path = os.path.join(self.dir_smplxs, subject, "mesh_smplx.obj")
        return trimesh.load(smplx_path)

    def _get_mesh_path(self, sid):
        subject = self._get_subject(sid)
        return os.path.join(self.dir_meshes, subject, subject + ".obj")
