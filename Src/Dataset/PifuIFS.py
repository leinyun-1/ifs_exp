import os
import numpy as np

import torch
import trimesh

from Dataset.BaseIFS import BaseIFS


class PifuIFS(BaseIFS):
    def __init__(self, path, split, roi, nov, yaw_list):
        super(PifuIFS, self).__init__(path, split, roi, nov, yaw_list)

        self.num_slides = 1

        self.sigma = 0.02
        self.num_sample_inout = 2000

    def _load_samples(self, subject):
        file_name = os.path.join(self.dir_meshes, subject, subject+'.obj')
        mesh = trimesh.load(file_name)

        radius_list = [self.sigma / 3, self.sigma, self.sigma * 2]
        surface_points = np.zeros((6 * self.num_sample_inout, 3))
        sample_points = np.zeros((6 * self.num_sample_inout, 3))
        for i in range(3):
            d = 2 * self.num_sample_inout
            surface_points[i * d:(i + 1) * d, :], _ = trimesh.sample.sample_surface(mesh,2 * self.num_sample_inout)
            sample_points[i * d:(i + 1) * d, :] = surface_points[i * d:(i + 1) * d, :] + np.random.normal(
                scale=radius_list[i], size=(2 * self.num_sample_inout, 3))

        # add random points within image space
        b_min = np.array([-1,-1,-1])
        b_max = np.array([1,1,1])
        length = b_max - b_min
        random_points = np.random.rand(self.num_sample_inout, 3) * length + b_min
        sample_points = np.concatenate([sample_points, random_points], 0)
        np.random.shuffle(sample_points)
        sample_points = sample_points[:8192]
        inside = mesh.contains(sample_points) #这句话卡住了
        weights = np.ones(inside.shape[0])

        return {
            "samples": torch.from_numpy(sample_points.T),
            "occs": torch.from_numpy(inside),
            "weights": torch.from_numpy(weights),
        }

    def __len__(self):
        return self._num_subjects * self.num_slides

    def _get_ids(self, index):
        sid = (index // self.num_slides) % self._num_subjects
        pid = index % self.num_slides
        yid = (index // (self._num_subjects * self.num_slides)) % len(self.yaw_list)

        return sid, pid, yid


    def __getitem__(self, index):
        sid, pid, yid = self._get_ids(index)
        subject = self._get_subject(sid)

        result = {"sid": sid, "yid": yid, "pid": pid}

        sample = self._load_samples(subject)
        result.update(sample)

        render = self._load_renders(subject, None, yid=yid)
        result.update(render)

        images = result["images"]
        result["images"] = torch.cat([image.unsqueeze(0) for image in images], dim=0)

        result.update({
            "bmin": np.array([-1,-1,-1]),
            "bmax": np.array([1,1,1])
        })
        smplx = self._get_smplx(subject)
        subject = self._get_subject(sid)
        result.update({
            "smplx": smplx,
            "subject": subject
        })
        return result


def ifs_pack(device, batch):
    for key in batch:
        if torch.is_tensor(batch[key]):
            batch[key] = batch[key].to(device, dtype=torch.float32)
    return batch


def get_dataloader(path, is_train=False, roi=1024, nov=8, yaw_list=[0, 6, 12, 18, 24, 30, 36, 42]):
    split = "train" if is_train else "test"
    dataset = PifuIFS(path=path, split=split, roi=roi, nov=nov, yaw_list=yaw_list)
    print(f"Loaded {split} data: {len(dataset)}")
    return dataset