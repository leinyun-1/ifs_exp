import torch
import torch.nn as nn
import torch.nn.functional as F

from NotRefactored.net_util import init_net, CostRegNet
from Utils.Geometry import index, perspective4
from Trainer.ModelInvoker import ModelInvoker, default_pack
from .HourGlass import HGFilter
from .SurfaceClassifier import SurfaceClassifier


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

        self.normlayer = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.normlayer(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, concat=True):
        super(Up, self).__init__()
        self.concat = concat

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
        )

        if self.concat:
            self.conv2 = BasicConv2d(out_channels * 2, out_channels, kernel_size=3, stride=1)
        else:
            self.conv2 = BasicConv2d(out_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x, y):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1, inplace=True)

        assert x.size() == y.size()

        if self.concat:
            x = torch.cat((x, y), 1)
        else:
            x = x + y

        return self.conv2(x)


class Encoder(nn.Module):
    """Height and width need to be divided by 48, downsampled by 1/3"""

    def __init__(self, in_channel, channels, out_channels):
        super(Encoder, self).__init__()
        self.inc = nn.Sequential(
            BasicConv2d(in_channel, channels[0], kernel_size=3, stride=1),
            BasicConv2d(channels[0], channels[0], kernel_size=5, stride=1),
            BasicConv2d(channels[0], channels[0], kernel_size=3, stride=1),
        )

        self.down1 = BasicConv2d(channels[0], channels[1], kernel_size=3, stride=2)
        self.down2 = BasicConv2d(channels[1], channels[2], kernel_size=3, stride=2)
        self.down3 = BasicConv2d(channels[2], channels[3], kernel_size=3, stride=2)
        self.down4 = BasicConv2d(channels[3], channels[4], kernel_size=3, stride=2)

        self.up4 = Up(channels[4], channels[3], concat=True)
        self.up3 = Up(channels[3], channels[2], concat=True)
        self.up2 = Up(channels[2], channels[1], concat=True)
        self.up1 = Up(channels[1], channels[0], concat=True)

        self.out3 = nn.Sequential(
            nn.Conv2d(channels[3], out_channels[3], 3, padding=1), nn.BatchNorm2d(out_channels[3]), nn.ReLU()
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(channels[2], out_channels[2], 3, padding=1), nn.BatchNorm2d(out_channels[2]), nn.ReLU()
        )
        self.out1 = nn.Sequential(
            nn.Conv2d(channels[1], out_channels[1], 3, padding=1), nn.BatchNorm2d(out_channels[1]), nn.ReLU()
        )
        self.out0 = nn.Sequential(
            nn.Conv2d(channels[0], out_channels[0], 3, padding=1), nn.BatchNorm2d(out_channels[0]), nn.ReLU()
        )

        self._out_channel = sum(out_channels)

    @property
    def out_channel(self):
        return self._out_channel

    @classmethod
    def ratio(cls):
        return 16

    def forward(self, x):
        _, _, h, w = x.shape

        e0 = self.inc(x)

        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)

        d3 = self.up4(e4, e3)
        d2 = self.up3(d3, e2)
        d1 = self.up2(d2, e1)
        d0 = self.up1(d1, e0)

        result = []
        result.append(self.out0(d0))
        result.append(F.interpolate(self.out1(d1), (h, w), mode="bilinear", align_corners=False))
        result.append(F.interpolate(self.out2(d2), (h, w), mode="bilinear", align_corners=False))
        result.append(F.interpolate(self.out3(d3), (h, w), mode="bilinear", align_corners=False))

        return torch.cat(result, dim=1)


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
        fusion='wheel',
        **kwargs,
    ):
        super().__init__(tm, metric, pack, selection)

        self.nov = nov
        self.rov = rov
        self.frozen_encoder = frozen_encoder
        self.fusion_type = fusion 

        self.encoder = Encoder(in_channel=3, channels=[32, 48, 64, 96, 128], out_channels=[32, 24, 16, 8])
        self.decoder = CostRegNet(self.encoder.out_channel * 2, 128)
        init_net(self)

    @classmethod
    def ratio(cls):
        return Encoder.ratio()

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

    def fusion(self, sfeature):
        B, V, C, N = sfeature.shape
        avg_feat = sfeature.clone()

        wheel_feat = torch.zeros_like(sfeature)
        wheel_feat1 = torch.zeros_like(sfeature)
        wheel_feat[:, 1:, :, :] = sfeature[:, :-1, :, :].clone()
        wheel_feat[:, 0, :, :] = sfeature[:, V - 1, :, :].clone()

        num_levels = 10

        feat0 = sfeature * wheel_feat  # B*V*C*N
        feat0 = feat0.reshape(B, V, C // num_levels, num_levels, N)
        feat0 = torch.sum(feat0, dim=2)  # B*V*l*N

        wheel_feat1[:, :-1, :, :] = sfeature[:, 1:, :, :].clone()
        wheel_feat1[:, V - 1, :, :] = sfeature[:, 0, :, :].clone()

        feat1 = sfeature * wheel_feat1
        feat1 = feat1.reshape(B, V, C // num_levels, num_levels, N)
        feat1 = torch.sum(feat1, dim=2)  # B*V*l*N

        ffeature = (feat0 + feat1) / 2

        weights = torch.sum(ffeature, dim=2)  # B*V*N
        weights = F.softmax(weights, dim=1)
        avg_feat = weights.unsqueeze(2).repeat(1, 1, C, 1) * avg_feat
        avg_feat = torch.sum(avg_feat, dim=1)  # B*C*N

        ffeature = ffeature.reshape(B, V * num_levels, N)
        ffeature = torch.cat([ffeature, avg_feat], dim=1)

        return ffeature


    def fusion_avg(self,sfeature):
        return torch.mean(sfeature,dim=1)
    
    def fusion_var(self,sfeature):
        return torch.var(sfeature,dim=1)
    
    def fusion_avg_var(self,sfeature):
        return torch.cat([torch.mean(sfeature,dim=1),torch.var(sfeature,dim=1)],dim=1)

    def decode(self, ffeature, rov):
        B, C, sps = ffeature.shape

        if rov == -1:
            result = self.decoder(ffeature)
            return result

        else:
            spv = rov**3
            vps = sps // spv

            ffeature = ffeature.reshape(B, C, vps, spv).permute(2, 0, 1, 3).reshape(vps * B, C, rov, rov, rov)

            result = self.decoder(ffeature)
            result = result.reshape(vps, B, 1, spv).permute(1, 2, 0, 3).reshape(B, 1, sps)
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
        samples2d = perspective4(samples, projection)
        sfeature = self.sample(mvfeature, samples2d)  # L*B*V*C*N

        # 3d part
        if self.fusion_type == 'avg_var':
            ffeature = self.fusion_avg_var(sfeature)
        else:
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
