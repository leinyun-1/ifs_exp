import numpy as np
import math

from Utils.Geometry import create_grid
from Dataset.BaseIFS import BaseIFS


class ScenaroIFS(BaseIFS):
    def __init__(self, path, split, ratio, roi, nov, yaw_list, dataset_type):
        super(ScenaroIFS, self).__init__(path, split, roi, nov, yaw_list, dataset_type)

        self.ratio = ratio

    def _get_scene_bbox(self, sid):
        subject = self._get_subject(sid)
        mesh = self._get_smplx(subject)

        b_min = np.min(mesh.vertices, axis=0)
        b_max = np.max(mesh.vertices, axis=0)

        center = (b_min + b_max) / 2
        scale = np.min(1.0 / (b_max - b_min)) * 0.9
        b_min = center - 0.5 / scale
        b_max = center + 0.5 / scale

        return b_min, b_max

    def get_subject_bbox(self, subject):
        mesh = self._get_smplx(subject)
        b_min = np.min(mesh.vertices, axis=0) - 0.25
        b_max = np.max(mesh.vertices, axis=0) + 0.25

        return b_min, b_max

    def __len__(self):
        return self._num_subjects

    def _get_ids(self, index):
        sid = index % self._num_subjects
        yid = (index // self._num_subjects) % len(self.yaw_list)

        return sid, yid

    class Cropper:
        def __init__(self, dataset, ratio):
            self.dataset = dataset
            self.ratio = ratio

        def __call__(self, intrinsic, extrinsic, subject, render):
            bmin, bmax = self.dataset.get_subject_bbox(subject)
            V = create_grid(2, 2, 2, bmin, bmax)[0].reshape(3, -1)
            V = np.vstack((V, np.ones((1, 8))))

            v = np.matmul(np.matmul(intrinsic, extrinsic), V)
            v = v[:2, :] / v[2, :]

            bbox = [0, 0, 0, 0]
            bbox[0], bbox[1] = np.min(v, axis=1)
            bbox[2], bbox[3] = np.max(v, axis=1)

            w = math.ceil((bbox[2] - bbox[0]) / self.ratio) * self.ratio
            h = math.ceil((bbox[3] - bbox[1]) / self.ratio) * self.ratio

            bbox[0] = (bbox[0] + bbox[2]) // 2 - w // 2
            bbox[1] = (bbox[1] + bbox[3]) // 2 - h // 2
            bbox[2] = bbox[0] + w
            bbox[3] = bbox[1] + h

            return bbox

    def __getitem__(self, index):
        sid, yid = self._get_ids(index)

        subject = self._get_subject(sid)
        bmin, bmax = self._get_scene_bbox(sid)
        smplx = self._get_smplx(subject)

        result = {"sid": sid, "subject": subject, "bmin": bmin, "bmax": bmax, "smplx": smplx}
        result.update(self._load_renders(subject, ScenaroIFS.Cropper(self, self.ratio), yid=yid))
        #result.update(self._load_renders(subject, None, yid=yid))

        return result
