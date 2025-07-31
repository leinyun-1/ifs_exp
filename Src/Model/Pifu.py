import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from NotRefactored.net_util import init_net, CostRegNet
from Utils.Geometry import index
from Trainer.ModelInvoker import ModelInvoker, default_pack
from .HourGlass import HGFilter
from .SurfaceClassifier import SurfaceClassifier,TwoLayerMLP

def perspective4(points, calibrations):
    """
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [BxVx3x4] Tensor of projection matrix
    :return: xy: [BxVx2xN] Tensor of xy coordinates in the image plane
    """
    points = points.unsqueeze(1)
    B, _, _, N = points.shape

    device = points.device
    points = torch.cat([points, torch.ones((B, 1, 1, N), device=device)], dim=2)
    points = calibrations @ points
    points[:, :, :2, :] /= points[:, :, 2:, :]
    return points[:, :, :2, :], points[:, :, 2:, :]

class IFSNet(ModelInvoker):
    def __init__(
        self,
        tm,
        metric="loss",
        pack=default_pack,
        selection=None,
        ################################################################
        nov=8,
        rov=48,
        frozen_encoder=False,
        **kwargs,
    ):
        super().__init__(tm, metric, pack, selection)

        self.nov = nov
        self.rov = rov
        self.frozen_encoder = frozen_encoder

        config = {
        "num_stack": 4,
        "num_hourglass": 2,
        "hourglass_dim": 256,
        "hg_down": 'ave_pool',
        "norm": 'group'
        }
        config = SimpleNamespace(**config)

        self.encoder = HGFilter(config)
        self.view_ebd = TwoLayerMLP(257,512,256)
        self.decoder = SurfaceClassifier([256*8, 1024, 512, 256, 128, 1],last_op=nn.Sigmoid())
        init_net(self)

    @classmethod
    def ratio(cls):
        return 16

    def encode(self, images):
        """
        encode the input images
        :param images: V * [B, 3, H, W]
        return: V * [B, C(L), H, W] 'channel here may differ from the channel of input images'
        """
        if not isinstance(images, list):
            return self.encoder(images)

        mvfeature = []
        for view in range(self.nov):
            mvfeature.append(self.encoder(images[view]))
        return mvfeature

    def sample(self, mvfeature, points2d):
        """
        use points to index feature
        points: [B, V, 2, N]
        mvfeature: V * [B, C(L), H, W] or [B * V, C(L), H, W]
        return: [B, V, C(L), N]
        """
        if not isinstance(mvfeature, list):
            B, V, _, N = points2d.shape
            sfeature = index(mvfeature, points2d.view(B * V, 2, N))
            return sfeature.view(B, V, -1, N)

        feature2d = []
        for view in range(self.nov):
            feature2d.append(index(mvfeature[view], points2d[:, view, :, :]).unsqueeze(1))
        return torch.cat(feature2d, dim=1)

    def fusion(self,sfeature):
        #return torch.mean(sfeature,dim=1)
        return sfeature.flatten(1,2)
    

    def decode(self, ffeature, rov):
        B, C, sps = ffeature.shape

        result = self.decoder(ffeature)
        return result


    def query(self, mvfeature, samples, projection, rov):
        """
        Given 3D samples, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param samples: [B, 3, N] world space coordinates of samples
        :param projection: [B, 3, 4] calibration matrices for each image
        :return: [B, Res, N] predictions for each point
        """

        # 2d part
        samples2d,z = perspective4(samples, projection)
        sfeature = self.sample(mvfeature, samples2d)  # B*V*C*N

        sfeature = torch.cat([sfeature,z],dim = 2) # 每个视角叠加对应视角的z值
        sfeature = self.view_ebd(sfeature) # f1(single_view_feat,single_view_z) = phi.  f2(phi_1,,,,phi_n) = s 

        # 3d part
        ffeature = self.fusion(sfeature)
        return self.decode(ffeature, rov).squeeze(1)

    def forward(self, epoch, bidx, data, milestone=False):
        images, projection = data["images"], data["projection"]
        samples, occs = data["samples"], data["occs"]
        weights = data['weights']

        B, V, _, H, W = images.shape
        images = images.view(B * V, 3, H, W)

        mvfeature = self.encode(images)
        mvfeature = [f.detach() for f in mvfeature] if self.frozen_encoder else mvfeature

        result = self.query(mvfeature, samples, projection, self.rov)
        error = torch.mean(F.mse_loss(result, occs, reduction='none') * weights)

        return result, error, None
