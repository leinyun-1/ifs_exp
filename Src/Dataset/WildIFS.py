'''
file:BaseIFS.py #file:ScenaroIFS.py 阅读两个dataset文件，现有数据集的组织形式为：
dataset/
    Res2048/
        0418/
            image/
            ProjParams.pkl
            pts/
            depth/
            normal/
            mask/
            texture/
            color/
        train.txt
        test.txt
    mesh/
    smplx/
grid_samples_32_24/
现在我需要你新建立一个dataset类，专门负责测试in the wild 输入。
输入数据只有8视角图片和对应的相机参数，也就是有image和ProjParams.pkl以及test.txt，没有对应的mesh和smplx。 
输入数据的限制，导致bbox只能固定为-1到1的立方体。同时load_renders的输入中没有cropper参数，因为不需要crop，直接用全图。
新建立的dataset命名为WildIFS，可以不用继承BaseIFS（因为有些东西写死了）
'''

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
import trimesh

class WildIFS(Dataset):
    def __init__(self, path, split, roi, nov, yaw_list):
        super(WildIFS, self).__init__()

        self.is_train = split == "train"

        self.roi = roi
        self.nov = nov
        self.yaw_list = yaw_list

        # Path setup
        self.dir_render = os.path.join(path, 'render')
        self.dir_images = os.path.join(self.dir_render, "image")
        self.dir_params = os.path.join(self.dir_render, "ProjParams.pkl")

        self.dir_smplxs = os.path.join(path, "smplx")

        if self.is_train:
            dir_subjects = os.path.join(self.dir_render, "train.txt")
        else:
            dir_subjects = os.path.join(self.dir_render, "test.txt")
        self.subjects = sorted(list(set(np.loadtxt(dir_subjects, dtype=str))))

        self.cam_params = pickle.load(open(self.dir_params, "rb"), encoding="iso-8859-1")

    def _load_images(self, subject, vid):
        image_path = os.path.join(self.dir_images, subject + "_{}.png".format(vid))
        image = Image.open(image_path).convert("RGB")
        return image

    def _load_projection(self, subject, vid):
        K = self.cam_params[subject + "_" + str(vid)]["K"]
        R = self.cam_params[subject + "_" + str(vid)]["R"]
        t = self.cam_params[subject + "_" + str(vid)]["t"]
        intrinsic = np.array(K[0])
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = np.array(R)
        extrinsic[:3, 3] = np.array(t).squeeze()
        return intrinsic, extrinsic
    
    def _get_smplx(self, subject):
        smplx_path = os.path.join(self.dir_smplxs, subject, "mesh_smplx.obj")
        return trimesh.load(smplx_path)

    def __len__(self):
        return len(self.subjects) * len(self.yaw_list)

    def _get_ids(self, index):
        sid = index % len(self.subjects)
        yid = (index // len(self.subjects)) % len(self.yaw_list)
        return sid, yid
    
    def __getitem__(self, index):
        sid, yid = self._get_ids(index)
        subject = self.subjects[sid]
        yaw = self.yaw_list[yid]

        images = []
        projections = []

        for vid in range(self.nov):
            image = self._load_images(subject, vid)
            intrinsic, extrinsic = self._load_projection(subject, vid)
            images.append(image)
            projections.append((intrinsic, extrinsic))

        bmin = np.array([-1, -1, -1])
        bmax = np.array([1, 1, 1])

        smplx = self._get_smplx(subject)

        return {
            "subject": subject,
            "images": images,
            "projection": projections,
            "bmin": bmin,
            "bmax": bmax,
            "smplx": smplx
        }
    
