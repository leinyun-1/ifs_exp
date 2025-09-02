import os
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset
from PIL import Image
import trimesh


class BaseIFS(Dataset):
    def __init__(self, path, split, roi, nov, yaw_list):
        super(BaseIFS, self).__init__()

        self.is_train = split == "train"

        self.roi = roi
        self.nov = nov
        self.yaw_list = yaw_list

        [path, grid_sample_path] = path.split(' ')

        # Path setup
        self.dir_render = os.path.join(path, "Res2048", '0418')
        self.dir_images = os.path.join(self.dir_render, "image")
        self.dir_params = os.path.join(self.dir_render, "ProjParams.pkl")
        self.dir_points = os.path.join(self.dir_render, "pts")
        self.dir_depths = os.path.join(self.dir_render, "depth")
        self.dir_normal = os.path.join(self.dir_render, "normal")
        self.dir_masks = os.path.join(self.dir_render, "mask")
        self.dir_texture = os.path.join(self.dir_render, "texture")
        self.dir_colors = os.path.join(self.dir_render, "color")

        self.dir_meshes = os.path.join(path, "mesh")
        self.dir_smplxs = os.path.join(path, "smplx")

        self.dir_samples = grid_sample_path

        if self.is_train:
            dir_subjects = os.path.join(self.dir_render, "train.txt")
        else:
            #dir_subjects = os.path.join(self.dir_render, "test.txt") 
            dir_subjects = os.path.join(self.dir_render, "test1.txt") # 测试特殊样例
        self.subjects = sorted(list(set(np.loadtxt(dir_subjects, dtype=str))))

        self.cam_params = pickle.load(open(self.dir_params, "rb"), encoding="iso-8859-1")

    def _load_images(self, subject, vid):
        image_path = os.path.join(self.dir_images, subject + "_{}.png".format(vid))
        image = Image.open(image_path).convert("RGB")

        mask_path = os.path.join(self.dir_masks, subject + "_{}_mask.png".format(vid))
        mask = Image.open(mask_path).convert("RGB")
        return image, mask

    def _load_projection(self, subject, vid):
        K = self.cam_params[subject + "_" + str(vid)]["K"]
        R = self.cam_params[subject + "_" + str(vid)]["R"]
        t = self.cam_params[subject + "_" + str(vid)]["t"]
        intrinsic = np.array(K[0])
        extrinsic = np.zeros([3, 4])
        extrinsic[:, :3] = R
        extrinsic[:, 3] = t

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

            # HW参考系在右上角。
            intrinsic[1, :] *= -1.0
            intrinsic[1, 2] += self.roi

            if cropper != None :
                bbox = cropper(intrinsic, extrinsic, subject, render)
                intrinsic, render = self._crop_render(intrinsic, render, bbox)

            image, mask = render
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
