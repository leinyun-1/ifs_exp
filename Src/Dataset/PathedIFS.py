import os
import numpy as np

import torch

from Dataset.BaseIFS import BaseIFS


class PathedIFS(BaseIFS):
    def __init__(self, path, split, roi, nov, yaw_list):
        super(PathedIFS, self).__init__(path, split, roi, nov, yaw_list)

        self.num_slides = np.loadtxt(os.path.join(self.dir_samples, "num_slides.txt"), dtype=int)
        self.som = np.loadtxt(os.path.join(self.dir_samples, "som.txt"), dtype=int)

        # todo: 这个数值是硬算的，看起来比较随便
        self.patch_size = self.roi // self.som * 2
        self.patch_size = (self.patch_size // 32) * 32  # 

    def _load_samples(self, subject, pid):
        file_name = os.path.join(self.dir_samples, subject, str(pid) + ".npz")
        data = np.load(file_name)

        samples = data["samples"]
        center = data["center"]
        occs = data["occs"]
        sdf = data["sdf"]

        sdf[sdf > 0.2] = 0.2
        sdf[sdf < -0.2] = -0.2
        # center = (np.max(samples, axis=0) + np.min(samples, axis=0)) / 2.0

        weights = np.ones(occs.shape[0])
        weights[abs(sdf) < 0.005] = 8

        return {
            "samples": torch.from_numpy(samples.T),
            "occs": torch.from_numpy(occs),
            "sdfs": torch.from_numpy(sdf),
            "weights": torch.from_numpy(weights),
        }, center[:, np.newaxis]

    def __len__(self):
        return self._num_subjects * self.num_slides

    def _get_ids(self, index):
        sid = (index // self.num_slides) % self._num_subjects
        pid = index % self.num_slides
        yid = (index // (self._num_subjects * self.num_slides)) % len(self.yaw_list)

        return sid, pid, yid

    class Cropper:
        def __init__(self, som, center, patch_size):
            self.som = som
            self.center = center
            self.patch_size = patch_size

        def __call__(self, intrinsic, extrinsic, subject, render):
            V = np.mgrid[-1:2:2, -1:2:2, -1:2:2].reshape(3, -1)
            V = V / self.som + self.center
            V = np.vstack((V, np.ones((1, 8))))

            v = np.matmul(np.matmul(intrinsic, extrinsic), V)
            v = v[:2, :] / v[2, :]

            bbox = [0, 0, 0, 0]
            bbox[0], bbox[1] = np.min(v, axis=1)
            bbox[2], bbox[3] = np.max(v, axis=1)

            bbox[0] = (bbox[0] + bbox[2]) // 2 - self.patch_size // 2
            bbox[1] = (bbox[1] + bbox[3]) // 2 - self.patch_size // 2
            bbox[2] = bbox[0] + self.patch_size
            bbox[3] = bbox[1] + self.patch_size
            return bbox

    def __getitem__(self, index):
        sid, pid, yid = self._get_ids(index)
        subject = self._get_subject(sid)

        result = {"sid": sid, "yid": yid, "pid": pid}

        sample, center = self._load_samples(subject, pid)
        result.update(sample)

        render = self._load_renders(subject, PathedIFS.Cropper(self.som, center, self.patch_size), yid=yid)
        result.update(render)

        images = result["images"]
        result["images"] = torch.cat([image.unsqueeze(0) for image in images], dim=0)
        return result


def ifs_pack(device, batch):
    for key in batch:
        if torch.is_tensor(batch[key]):
            batch[key] = batch[key].to(device, dtype=torch.float32)
    return batch


def get_dataloader(path, is_train=False, roi=1024, nov=8, yaw_list=[0, 6, 12, 18, 24, 30, 36, 42]):
    split = "train" if is_train else "test"
    dataset = PathedIFS(path=path, split=split, roi=roi, nov=nov, yaw_list=yaw_list)
    print(f"Loaded {split} data: {len(dataset)}")
    return dataset
