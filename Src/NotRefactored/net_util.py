import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import functools


def check_tensor(t, name="test"):
    if torch.sum(torch.isnan(t) + torch.isinf(t)) > 0:
        print(name, torch.sum(torch.isnan(t) + torch.isinf(t)))
        exit(0)


def reshape_multiview_tensors(image_tensor, calib_tensor):
    # Careful here! Because we put single view and multiview together,
    # the returned tensor.shape is 5-dim: [B, num_views, C, W, H]
    # So we need to convert it back to 4-dim [B*num_views, C, W, H]
    # Don't worry classifier will handle multi-view cases
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1],
        image_tensor.shape[2],
        image_tensor.shape[3],
        image_tensor.shape[4],
    )
    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1], calib_tensor.shape[2], calib_tensor.shape[3]
    )

    return image_tensor, calib_tensor


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1], sample_tensor.shape[2], sample_tensor.shape[3]
    )
    return sample_tensor


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return lr


def compute_acc(pred, gt, thresh=0.5):
    """
    return:
        IOU, precision, and recall
    """
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias)


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def imageSpaceRotation(xy, rot):
    """
    args:
        xy: (B, 2, N) input
        rot: (B, 2) x,y axis rotation angles

    rotation center will be always image center (other rotation center can be represented by additional z translation)
    """
    disp = rot.unsqueeze(2).sin().expand_as(xy)
    return (disp * xy).sum(dim=1)


def cal_gradient_penalty(netD, real_data, fake_data, device, type="mixed", constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == "real":  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == "fake":
            interpolatesv = fake_data
        elif type == "mixed":
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = (
                alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0])
                .contiguous()
                .view(*real_data.shape)
            )
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError("{} not implemented".format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolatesv,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (
            ((gradients + 1e-16).norm(2, dim=1) - constant) ** 2
        ).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, 32)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm="batch", base=64):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if norm == "batch":
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == "group":
            self.bn1 = nn.GroupNorm(base // 2, in_planes)
            self.bn2 = nn.GroupNorm(base // 2, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(base // 2, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(base // 2, in_planes)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4, nn.ReLU(True), nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class Conv_for_res(nn.Module):
    def __init__(self, in_planes, out_planes, norm="group", base=2):
        super(Conv_for_res, self).__init__()
        self.conv1 = ConvBlock(in_planes, out_planes, norm=norm, base=base)
        self.conv2 = ConvBlock(in_planes, out_planes, norm=norm, base=base)
        self.conv3 = ConvBlock(in_planes, out_planes, norm=norm, base=base)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        return out3


class Conv_for_fineDepth(nn.Module):
    def __init__(self, in_c=128 + 1, out_c=1, norm="group", base=2) -> None:
        super(Conv_for_fineDepth, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes=in_c, out_planes=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(base // 2, 64),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes=64, out_planes=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(base // 2, 32),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_planes=32, out_planes=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(base // 2, 8),
            nn.LeakyReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_planes=8, out_planes=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(base // 2, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)

        return out4


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        relu=True,
        bn=True,
        bn_momentum=0.1,
        init_method="xavier",
        **kwargs
    ):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        relu=True,
        bn=True,
        bn_momentum=0.1,
        init_method="xavier",
        **kwargs
    ):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs
        )
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class CostRegNet_small(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet_small, self).__init__()

        self.conv0 = Conv3d(in_channels, base_channels, padding=1)
        self.conv1 = Conv3d(base_channels, base_channels, padding=1)

        self.conv2 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv3 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv4 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv5 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv6 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv7 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)
        #
        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv1 = self.conv1(self.conv0(x))
        conv3 = self.conv3(self.conv2(conv1))
        x = self.conv5(self.conv4(conv3))
        x = conv3 + self.conv6(x)
        x = conv1 + self.conv7(x)
        x = self.prob(x)
        return x


class CostRegNet(nn.Module):
    def __init__(self, in_channel, base_channel):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channel, base_channel)

        self.conv1 = ConvBnReLU3D(base_channel, base_channel * 2, stride=2)
        self.conv2 = ConvBnReLU3D(base_channel * 2, base_channel * 2)

        self.conv3 = ConvBnReLU3D(base_channel * 2, base_channel * 4, stride=2)
        self.conv4 = ConvBnReLU3D(base_channel * 4, base_channel * 4)

        self.conv5 = ConvBnReLU3D(base_channel * 4, base_channel * 8, stride=2)
        self.conv6 = ConvBnReLU3D(base_channel * 8, base_channel * 8)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 8,
                base_channel * 4,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(base_channel * 4),
            nn.ReLU(inplace=True),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 4,
                base_channel * 2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(base_channel * 2),
            nn.ReLU(inplace=True),
        )

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 2,
                base_channel,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True),
        )

        # self.prob = nn.Conv3d(base_channel, 1, 3, stride=1, padding=1)
        self.prob = nn.Sequential(
            Conv3d(base_channel, base_channel // 2, 1),
            Conv3d(base_channel // 2, base_channel // 4, 1),
            Conv3d(base_channel // 4, base_channel // 8, 1),
            nn.Conv3d(base_channel // 8, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


class CostRegNet_1(nn.Module):
    def __init__(self, in_channel, base_channel):
        super(CostRegNet_1, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channel, base_channel)

        self.conv1 = ConvBnReLU3D(base_channel, base_channel * 2, stride=2)
        self.conv2 = ConvBnReLU3D(base_channel * 2, base_channel * 2)

        self.conv3 = ConvBnReLU3D(base_channel * 2, base_channel * 4, stride=2)
        self.conv4 = ConvBnReLU3D(base_channel * 4, base_channel * 4)

        self.conv5 = ConvBnReLU3D(base_channel * 4, base_channel * 8, stride=2)
        self.conv6 = ConvBnReLU3D(base_channel * 8, base_channel * 8)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 8,
                base_channel * 4,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(base_channel * 4),
            nn.ReLU(inplace=True),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 4,
                base_channel * 2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(base_channel * 2),
            nn.ReLU(inplace=True),
        )

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 2,
                base_channel,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True),
        )

        # self.prob = nn.Conv3d(base_channel, 1, 3, stride=1, padding=1)
        self.prob = nn.Sequential(
            Conv3d(base_channel, base_channel // 2, 1),
            Conv3d(base_channel // 2, base_channel // 4, 1),
            Conv3d(base_channel // 4, base_channel // 8, 1),
            nn.Conv3d(base_channel // 8, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        ind = x
        x = self.prob(x)
        return x, ind


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class AggreCon3d(nn.Module):
    def __init__(self):
        super(AggreCon3d, self).__init__()
        self.conv0 = ConvBnReLU3D(256, 128)

        self.conv1 = ConvBnReLU3D(128, 64, stride=2)
        self.conv2 = ConvBnReLU3D(64, 64)

        self.conv3 = ConvBnReLU3D(64, 128, stride=2)
        self.conv4 = ConvBnReLU3D(128, 128)

        self.conv5 = ConvBnReLU3D(128, 256, stride=2)
        self.conv6 = ConvBnReLU3D(256, 256)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # self.conv11 = nn.Sequential(
        #     nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(64, 32, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        # x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


class Small_con3d(nn.Module):
    def __init__(self, in_channel=64, out_channel=32, mid_channel=32) -> None:
        super(Small_con3d, self).__init__()
        self.con0 = ConvBnReLU3D(in_channel, mid_channel)
        self.con1 = ConvBnReLU3D(mid_channel, mid_channel // 2)
        self.con2 = ConvBnReLU3D(mid_channel // 2 + mid_channel, out_channel, pad=0)

    def forward(self, x):
        con0 = self.con0(x)
        con1 = self.con1(con0)
        x = torch.cat([con0, con1], dim=1)
        x = self.con2(x)
        return x


class Conv_for_local_axis(nn.Module):
    def __init__(self, in_channel=400, out_channel=512, mid_channel=[256, 320]) -> None:
        super(Conv_for_local_axis, self).__init__()

        self.con0 = nn.Conv1d(in_channel, mid_channel[0], 2)
        self.con1 = nn.Conv1d(mid_channel[0], mid_channel[1], 3)
        self.con2 = nn.Conv1d(mid_channel[1], out_channel, 2)

    def forward(self, x):
        x = F.leaky_relu(self.con0(x))
        x = F.leaky_relu(self.con1(x))
        x = F.leaky_relu(self.con2(x))
        return x


class CostRegNet_small_for_local(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet_small_for_local, self).__init__()

        self.conv0 = Conv3d(in_channels, base_channels, padding=1)
        self.conv1 = Conv3d(base_channels, base_channels, padding=1)

        self.conv2 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv3 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv4 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv5 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv6 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv7 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)
        #
        self.prob = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            Conv3d(base_channels, base_channels // 2, 1),
            Conv3d(base_channels // 2, base_channels // 4, 1),
            Conv3d(base_channels // 4, base_channels // 8, 1),
            nn.Conv3d(base_channels // 8, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        conv1 = self.conv1(self.conv0(x))
        conv3 = self.conv3(self.conv2(conv1))
        x = self.conv5(self.conv4(conv3))
        x = conv3 + self.conv6(x)
        x = conv1 + self.conv7(x)
        x = self.prob(x)
        return x


class UNet(nn.Module):
    """Height and width need to be divided by 48, downsampled by 1/3"""

    def __init__(self, in_channel, channels, norm="batch"):
        super(UNet, self).__init__()
        self.conv_start = nn.Sequential(
            BasicConv2d(in_channel, channels[0], kernel_size=3, norm=norm),
            BasicConv2d(channels[0], channels[0], kernel_size=5, stride=1, norm=norm),
            BasicConv2d(channels[0], channels[0], kernel_size=3, norm=norm),
        )

        self.conv1a = BasicConv2d(channels[0], channels[1], kernel_size=3, stride=2, norm=norm)
        self.conv2a = BasicConv2d(channels[1], channels[2], kernel_size=3, stride=2, norm=norm)

        self.conv3a = BasicConv2d(channels[2], channels[3], kernel_size=3, stride=2, norm=norm)
        self.conv4a = BasicConv2d(channels[3], channels[4], kernel_size=3, stride=2, norm=norm)

        self.deconv4a = Conv2x(channels[4], channels[3], norm=norm, deconv=True)
        self.deconv3a = Conv2x(channels[3], channels[2], norm=norm, deconv=True)
        self.deconv2a = Conv2x(channels[2], channels[1], norm=norm, deconv=True)
        self.deconv1a = Conv2x(channels[1], channels[0], norm=norm, deconv=True)

    def forward(self, x):
        _, _, h, w = x.shape

        rem0 = self.conv_start(x)
        rem1 = self.conv1a(rem0)
        rem2 = self.conv2a(rem1)
        rem3 = self.conv3a(rem2)
        rem4 = self.conv4a(rem3)

        rem3d = self.deconv4a(rem4, rem3)
        rem2d = self.deconv3a(rem3d, rem2)
        rem1d = self.deconv2a(rem2d, rem1)
        rem0d = self.deconv1a(rem1d, rem0)

        result = []
        result.append(F.interpolate(rem3d, (h, w), mode="bilinear", align_corners=False))
        result.append(F.interpolate(rem2d, (h, w), mode="bilinear", align_corners=False))
        result.append(F.interpolate(rem1d, (h, w), mode="bilinear", align_corners=False))
        result.append(rem0d)

        return result


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        norm="instance",
        deconv=False,
        relu=True,
        **kwargs
    ):
        super(BasicConv2d, self).__init__()
        self.relu = relu

        if deconv:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True, **kwargs
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
                **kwargs
            )

        self.norm_fn = norm
        if self.norm_fn == "group":
            num_groups = out_channels // 8
            self.normlayer = nn.GroupNorm(num_groups, out_channels)
        elif self.norm_fn == "batch":
            self.normlayer = nn.BatchNorm2d(out_channels)
        elif self.norm_fn == "instance":
            self.normlayer = nn.InstanceNorm2d(out_channels)
        elif self.norm_fn == "none":
            self.norm_fn = None
        else:
            raise NotImplemented

    def forward(self, x):
        x = self.conv(x)
        if self.norm_fn:
            x = self.normlayer(x)
        if self.relu:
            x = F.leaky_relu(x, negative_slope=0.1, inplace=True)
        return x


class Conv2x(nn.Module):
    def __init__(self, in_channels, out_channels, norm="instance", deconv=False, concat=True):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv:
            kernel = 4
        else:
            kernel = 3

        self.conv1 = BasicConv2d(
            in_channels, out_channels, kernel_size=kernel, stride=2, norm=norm, deconv=deconv
        )

        if self.concat:
            self.conv2 = BasicConv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, norm=norm)
        else:
            self.conv2 = BasicConv2d(out_channels, out_channels, kernel_size=3, stride=1, norm=norm)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert x.size() == rem.size()
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(
                channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(channels * 2),
        )

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(
                channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(channels),
        )

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class LocalEncoderDeepls(nn.Module):
    def __init__(self, in_channels, channels) -> None:
        super().__init__()
        self.conv = ConvBnReLU3D(in_channels, channels, kernel_size=1, stride=1, pad=0)

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=2, stride=2, pad=0)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=2, stride=2, pad=0)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.conv3a = ConvBnReLU3D(channels * 4, channels * 8, kernel_size=2, stride=2, pad=0)
        self.conv3b = ConvBnReLU3D(channels * 8, channels * 8, kernel_size=3, stride=1, pad=1)

        self.conv4 = ConvBnReLU3D(channels * 8, channels * 8, kernel_size=2, stride=1, pad=0)

    def forward(self, x):
        x = self.conv(x)
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        conv3 = self.conv3b(self.conv3a(conv2))
        # print("conv3 shape: ",conv3.shape)
        conv4 = self.conv4(conv3)
        return conv4


class FPNEncoder(nn.Module):
    def __init__(self, feat_chs, norm_type="BN"):
        super(FPNEncoder, self).__init__()
        self.conv00 = Conv2d(3, feat_chs[0], 7, 1, padding=3, norm_type=norm_type)
        self.conv01 = Conv2d(feat_chs[0], feat_chs[0], 5, 1, padding=2, norm_type=norm_type)

        self.downsample1 = Conv2d(feat_chs[0], feat_chs[1], 5, stride=2, padding=2, norm_type=norm_type)
        self.conv10 = Conv2d(feat_chs[1], feat_chs[1], 3, 1, padding=1, norm_type=norm_type)
        self.conv11 = Conv2d(feat_chs[1], feat_chs[1], 3, 1, padding=1, norm_type=norm_type)

        self.downsample2 = Conv2d(feat_chs[1], feat_chs[2], 5, stride=2, padding=2, norm_type=norm_type)
        self.conv20 = Conv2d(feat_chs[2], feat_chs[2], 3, 1, padding=1, norm_type=norm_type)
        self.conv21 = Conv2d(feat_chs[2], feat_chs[2], 3, 1, padding=1, norm_type=norm_type)

        self.downsample3 = Conv2d(feat_chs[2], feat_chs[3], 3, stride=2, padding=1, norm_type=norm_type)
        self.conv30 = Conv2d(feat_chs[3], feat_chs[3], 3, 1, padding=1, norm_type=norm_type)
        self.conv31 = Conv2d(feat_chs[3], feat_chs[3], 3, 1, padding=1, norm_type=norm_type)

    def forward(self, x):
        conv00 = self.conv00(x)
        conv01 = self.conv01(conv00)
        down_conv0 = self.downsample1(conv01)
        conv10 = self.conv10(down_conv0)
        conv11 = self.conv11(conv10)
        down_conv1 = self.downsample2(conv11)
        conv20 = self.conv20(down_conv1)
        conv21 = self.conv21(conv20)
        down_conv2 = self.downsample3(conv21)
        conv30 = self.conv30(down_conv2)
        conv31 = self.conv31(conv30)

        return [conv01, conv11, conv21, conv31]


class FPNDecoder(nn.Module):
    def __init__(self, feat_chs):
        super(FPNDecoder, self).__init__()
        final_ch = feat_chs[-1]
        self.out0 = nn.Sequential(
            nn.Conv2d(final_ch, feat_chs[3], kernel_size=1), nn.BatchNorm2d(feat_chs[3]), Swish()
        )

        self.inner1 = nn.Conv2d(feat_chs[2], final_ch, 1)
        self.out1 = nn.Sequential(
            nn.Conv2d(final_ch, feat_chs[2], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[2]), Swish()
        )

        self.inner2 = nn.Conv2d(feat_chs[1], final_ch, 1)
        self.out2 = nn.Sequential(
            nn.Conv2d(final_ch, feat_chs[1], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[1]), Swish()
        )

        self.inner3 = nn.Conv2d(feat_chs[0], final_ch, 1)
        self.out3 = nn.Sequential(
            nn.Conv2d(final_ch, feat_chs[0], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[0]), Swish()
        )

    def forward(self, conv01, conv11, conv21, conv31):
        intra_feat = conv31
        out0 = self.out0(intra_feat)

        intra_feat = F.interpolate(
            intra_feat.to(torch.float32), scale_factor=2, mode="bilinear", align_corners=True
        ) + self.inner1(conv21)
        out1 = self.out1(intra_feat)

        intra_feat = F.interpolate(
            intra_feat.to(torch.float32), scale_factor=2, mode="bilinear", align_corners=True
        ) + self.inner2(conv11)
        out2 = self.out2(intra_feat)

        intra_feat = F.interpolate(
            intra_feat.to(torch.float32), scale_factor=2, mode="bilinear", align_corners=True
        ) + self.inner3(conv01)
        out3 = self.out3(intra_feat)

        return [out0, out1, out2, out3]


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        relu=True,
        bn=True,
        bn_momentum=0.1,
        norm_type="IN",
        **kwargs
    ):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        if norm_type == "IN":
            self.bn = nn.InstanceNorm2d(out_channels, momentum=bn_momentum) if bn else None
        elif norm_type == "BN":
            self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu:
            y = F.leaky_relu(y, 0.1, inplace=True)
        return y

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def torch_init_model(model, total_dict, key, rank=0):
    if key in total_dict:
        state_dict = total_dict[key]
    else:
        state_dict = total_dict
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=True,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix="")

    if rank == 0:
        print("missing keys:{}".format(missing_keys))
        print("unexpected keys:{}".format(unexpected_keys))
        print("error msgs:{}".format(error_msgs))
