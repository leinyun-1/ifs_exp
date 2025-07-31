import torch
import torch.nn as nn
import torch.nn.functional as F


class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=None):
        super(SurfaceClassifier, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature, xy=None, ret_list=False):
        '''

        :param feature: list of [BxC_inxN] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''
        y = feature
        tmpy = feature
        y_list = []
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            y_list.append(y)
        if self.last_op:
            y = self.last_op(y)
        if ret_list:
            return y_list
        else:
            return y


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),   # 归一化
            nn.ReLU(inplace=True),        # 激活函数
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),      # 归一化
            nn.ReLU(inplace=True)         # 激活函数
        )

    def forward(self, x): # b v c n 
        b,v,c,n = x.shape
        x = x.permute(0,1,3,2).reshape(-1,c) # (b v n) c 
        return self.net(x).reshape(b,v,n,c-1).permute(0,1,3,2) # b v c n  