from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
from layers_package.resample2d_package.resample2d import Resample2d
from layers_package.channelnorm_package.channelnorm import ChannelNorm
from networks.DispNetC import DispNetC
from networks.DispNetRes import DispNetRes
from networks.submodules import *


class SA_Module(nn.Module):
    """
    Note: simple but effective spatial attention module.
    """

    def __init__(self, input_nc, output_nc=1, ndf=16):
        super(SA_Module, self).__init__()
        self.attention_value = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, output_nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_value = self.attention_value(x)

        return attention_value


class FADNet(nn.Module):

    def __init__(self, batchNorm=True, lastRelu=False, resBlock=True, maxdisp=-1, input_channel=3, attention=False,
                 combine=False):
        super(FADNet, self).__init__()
        self.input_channel = input_channel
        self.batchNorm = batchNorm
        self.lastRelu = lastRelu
        self.maxdisp = maxdisp
        self.resBlock = resBlock
        self.combine=combine
        # First Block (DispNetC)
        self.dispnetc = DispNetC(self.batchNorm, maxdisp=self.maxdisp, input_channel=input_channel,
                                 combine=self.combine)

        # warp layer and channelnorm layer
        self.channelnorm = ChannelNorm()
        self.resample1 = Resample2d()
        self.attention = attention

        # Second Block (DispNetRes), input is 11 channels(img0, img1, img1->img0, flow, diff-mag)
        in_planes = 3 * 3 + 1 + 1
        self.dispnetres = DispNetRes(in_planes, self.batchNorm, lastRelu=self.lastRelu, maxdisp=self.maxdisp,
                                     input_channel=input_channel,attention=attention)

        self.relu = nn.ReLU(inplace=False)

        # # parameter initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         if m.bias is not None:
        #             init.uniform(m.bias)
        #         init.xavier_uniform(m.weight)

        #     if isinstance(m, nn.ConvTranspose2d):
        #         if m.bias is not None:
        #             init.uniform(m.bias)
        #         init.xavier_uniform(m.weight)

    def forward(self, inputs):

        # split left image and right image
        # inputs = inputs_target[0]
        # target = inputs_target[1]
        imgs = torch.chunk(inputs, 2, dim=1)
        img_left = imgs[0]
        img_right = imgs[1]

        # dispnetc
        dispnetc_flows = self.dispnetc(inputs)
        dispnetc_final_flow = dispnetc_flows[0]

        # warp img1 to img0; magnitude of diff between img0 and warped_img1,

        resampled_img1 = warp_right_to_left(inputs[:, self.input_channel:, :, :], -dispnetc_final_flow)

        diff_img0 = inputs[:, :self.input_channel, :, :] - resampled_img1

        norm_diff_img0 = channel_length(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-img
        # lowest scale
        inputs_net2 = torch.cat((inputs, resampled_img1, dispnetc_final_flow, norm_diff_img0), dim=1)

        dispnetres_flows = self.dispnetres([inputs_net2, dispnetc_flows])
        # dispnetres

        index = 0
        # print('Index: ', index)
        dispnetres_final_flow = dispnetres_flows[index]

        if self.training:
            return dispnetc_flows, dispnetres_flows
        else:
            return dispnetc_final_flow, dispnetres_final_flow  # , inputs[:, :3, :, :], inputs[:, 3:, :, :], resampled_img1

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]
