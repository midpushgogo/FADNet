from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
#from correlation_package.modules.corr import Correlation1d # from PWC-Net
from networks.submodules import *

class DispNetC(nn.Module):

    def __init__(self, batchNorm=False, lastRelu=True, resBlock=True, maxdisp=-1, input_channel=3, get_features = False,attention=False,combine=False):
        super(DispNetC, self).__init__()
        
        self.batchNorm = batchNorm
        self.input_channel = input_channel
        self.maxdisp = maxdisp
        self.get_features = get_features
        self.relu = nn.ReLU(inplace=False)
        self.combine=combine
        if self.combine:
            self.compress = ResBlock(256, 32)
            self.conv3d = nn.Sequential(
                nn.Conv3d(64, 32, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv3d(32, 32, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.attention=attention
        if self.attention:
            self.SA0 = SA_Module(input_nc=10)
            self.SA1 = SA_Module(input_nc=10)
            self.SA2 = SA_Module(input_nc=10)
            self.SA3 = SA_Module(input_nc=10)
            self.SA4 = SA_Module(input_nc=10)
            self.SA5 = SA_Module(input_nc=10)
            self.SA6 = SA_Module(input_nc=10)

        # shrink and extract features
        self.conv1   = conv(self.input_channel, 64, 7, 2)
        if resBlock:
            self.conv2   = ResBlock(64, 128, stride=2)
            self.conv3   = ResBlock(128, 256, stride=2)
            self.conv_redir = ResBlock(256, 32, stride=1)
        else:
            self.conv2   = conv(64, 128, stride=2)
            self.conv3   = conv(128, 256, stride=2)
            self.conv_redir = conv(256, 32, stride=1)

        # start corr from conv3, output channel is 32 + (max_disp * 2 / 2 + 1) 
        #self.conv_redir = ResBlock(256, 32, stride=1)
        #self.corr = Correlation1d(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        #self.corr = Correlation1d(pad_size=20, kernel_size=3, max_displacement=20, stride1=1, stride2=1, corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)

        if resBlock:
            self.conv3_1 = ResBlock(72, 256) if not self.combine else ResBlock(112, 256)
            self.conv4   = ResBlock(256, 512, stride=2)
            self.conv4_1 = ResBlock(512, 512)
            self.conv5   = ResBlock(512, 512, stride=2)
            self.conv5_1 = ResBlock(512, 512)
            self.conv6   = ResBlock(512, 1024, stride=2)
            self.conv6_1 = ResBlock(1024, 1024)
        else:
            self.conv3_1 = conv(72, 256) if not self.combine else conv(112, 256)
            self.conv4   = conv(256, 512, stride=2)
            self.conv4_1 = conv(512, 512)
            self.conv5   = conv(512, 512, stride=2)
            self.conv5_1 = conv(512, 512)
            self.conv6   = conv(512, 1024, stride=2)
            self.conv6_1 = conv(1024, 1024)

        self.pred_flow6 = predict_flow(1024)

        # # iconv with resblock
        # self.iconv5 = ResBlock(1025, 512, 1)
        # self.iconv4 = ResBlock(769, 256, 1)
        # self.iconv3 = ResBlock(385, 128, 1)
        # self.iconv2 = ResBlock(193, 64, 1)
        # self.iconv1 = ResBlock(97, 32, 1)
        # self.iconv0 = ResBlock(20, 16, 1)

        # iconv with deconv
        self.iconv5 = nn.ConvTranspose2d(1025, 512, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d(769, 256, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(385, 128, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(193, 64, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(97, 32, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(17+self.input_channel, 16, 3, 1, 1)

        # # original iconv with conv
        # self.iconv5 = conv(self.batchNorm, 1025, 512, 3, 1)
        # self.iconv4 = conv(self.batchNorm, 769, 256, 3, 1)
        # self.iconv3 = conv(self.batchNorm, 385, 128, 3, 1)
        # self.iconv2 = conv(self.batchNorm, 193, 64, 3, 1)
        # self.iconv1 = conv(self.batchNorm, 97, 32, 3, 1)
        # self.iconv0 = conv(self.batchNorm, 20, 16, 3, 1)
        
        # expand and produce disparity
        self.upconv5 = deconv(1024, 512)
        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow5 = predict_flow(512)

        self.upconv4 = deconv(512, 256)
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow4 = predict_flow(256)

        self.upconv3 = deconv(256, 128)
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow3 = predict_flow(128)

        self.upconv2 = deconv(128, 64)
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow2 = predict_flow(64)

        self.upconv1 = deconv(64, 32)
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow1 = predict_flow(32)

        self.upconv0 = deconv(32, 16)
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        if self.maxdisp == -1:
            self.pred_flow0 = predict_flow(16)
        else:
            self.disp_expand = ResBlock(16, self.maxdisp)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, 0.02 / n)
                # m.weight.data.normal_(0, 0.02)
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        #self.freeze()
        
    def forward(self, input):

        # split left image and right image
        imgs = torch.chunk(input, 2, dim = 1)
        img_left = imgs[0]
        img_right = imgs[1]

        conv1_l = self.conv1(img_left)
        conv2_l = self.conv2(conv1_l)
        conv3a_l = self.conv3(conv2_l)

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3a_r = self.conv3(conv2_r)

        # Correlate corr3a_l and corr3a_r
        #out_corr = self.corr(conv3a_l, conv3a_r)

        if self.combine:
            compress_left_feature = self.compress(conv3a_l)
            compress_right_feature = self.compress(conv3a_r)
            out_corr = build_comb(conv3a_l, conv3a_r, compress_left_feature, compress_right_feature, max_disp=40, comb=self.conv3d)
        else:
            out_corr = build_corr(conv3a_l, conv3a_r, max_disp=40)
        out_corr = self.corr_activation(out_corr)
        out_conv3a_redir = self.conv_redir(conv3a_l)
        in_conv3b = torch.cat((out_conv3a_redir, out_corr), 1)

        conv3b = self.conv3_1(in_conv3b)
        conv4a = self.conv4(conv3b)
        conv4b = self.conv4_1(conv4a)
        conv5a = self.conv5(conv4b)
        conv5b = self.conv5_1(conv5a)
        conv6a = self.conv6(conv5b)
        conv6b = self.conv6_1(conv6a)

        pr6 = self.pred_flow6(conv6b)
        upconv5 = self.upconv5(conv6b)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, conv5b), 1)
        iconv5 = self.iconv5(concat5)

        if self.attention:
            left_small=F.interpolate(img_left, size=(pr6.size()[-2]*2,pr6.size()[-1]*2))
            right_small = F.interpolate(img_right, size=(pr6.size()[-2]*2, pr6.size()[-1]*2))
            pr6_=F.interpolate(pr6,scale_factor=2)
            pr6_=pr6_/(img_left.size()[2]/left_small.size()[2])
            r_right = warp_right_to_left(right_small, -pr6_)

            error = left_small - r_right
            attention_map=self.SA6(torch.cat([left_small,right_small,error,pr6_],dim=1))
            pr5 = self.pred_flow5(iconv5*attention_map)
            pr5=pr6_+pr5
        else:
            pr5 = self.pred_flow5(iconv5)

        upconv4 = self.upconv4(iconv5)
        upflow5 = self.upflow5to4(pr5)
        concat4 = torch.cat((upconv4, upflow5, conv4b), 1)
        iconv4 = self.iconv4(concat4)
        
        if self.attention:
            left_small=F.interpolate(img_left, size=(pr5.size()[-2]*2,pr5.size()[-1]*2))
            right_small = F.interpolate(img_right, size=(pr5.size()[-2]*2, pr5.size()[-1]*2))
            pr5_=F.interpolate(pr5,scale_factor=2)
            pr5_=pr5_/(img_left.size()[2]/left_small.size()[2])
            r_right = warp_right_to_left(right_small, -pr5_)

            error = left_small - r_right
            attention_map=self.SA5(torch.cat([left_small,right_small,error,pr5_],dim=1))
            pr4 = self.pred_flow4(iconv4*attention_map)
            pr4 = pr5_ + pr4
        else:
            pr4 = self.pred_flow4(iconv4)
        upconv3 = self.upconv3(iconv4)
        upflow4 = self.upflow4to3(pr4)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)

        if self.attention:
            left_small=F.interpolate(img_left, size=(pr4.size()[-2]*2,pr4.size()[-1]*2))
            right_small = F.interpolate(img_right, size=(pr4.size()[-2]*2, pr4.size()[-1]*2))
            pr4_=F.interpolate(pr4,scale_factor=2)
            pr4_=pr4_/(img_left.size()[2]/left_small.size()[2])
            r_right = warp_right_to_left(right_small, -pr4_)

            error = left_small - r_right
            attention_map=self.SA4(torch.cat([left_small,right_small,error,pr4_],dim=1))
            pr3 = self.pred_flow3(iconv3*attention_map)
            pr3 = pr4_ + pr3
        else:
            pr3 = self.pred_flow3(iconv3)
        upconv2 = self.upconv2(iconv3)
        upflow3 = self.upflow3to2(pr3)
        concat2 = torch.cat((upconv2, upflow3, conv2_l), 1)
        iconv2 = self.iconv2(concat2)

        if self.attention:
            left_small=F.interpolate(img_left, size=(pr3.size()[-2]*2,pr3.size()[-1]*2))
            right_small = F.interpolate(img_right, size=(pr3.size()[-2]*2, pr3.size()[-1]*2))
            pr3_=F.interpolate(pr3,scale_factor=2)
            pr3_=pr3_/(img_left.size()[2]/left_small.size()[2])
            r_right = warp_right_to_left(right_small, -pr3_)

            error = left_small - r_right
            attention_map=self.SA3(torch.cat([left_small,right_small,error,pr3_],dim=1))
            pr2 = self.pred_flow2(iconv2*attention_map)
            pr2 = pr3_ + pr2
        else:
            pr2 = self.pred_flow2(iconv2)
        upconv1 = self.upconv1(iconv2)
        upflow2 = self.upflow2to1(pr2)
        concat1 = torch.cat((upconv1, upflow2, conv1_l), 1)
        iconv1 = self.iconv1(concat1)

        if self.attention:
            left_small=F.interpolate(img_left, size=(pr2.size()[-2]*2,pr2.size()[-1]*2))
            right_small = F.interpolate(img_right, size=(pr2.size()[-2]*2, pr2.size()[-1]*2))
            pr2_=F.interpolate(pr2,scale_factor=2)
            pr2_=pr2_/(img_left.size()[2]/left_small.size()[2])
            r_right = warp_right_to_left(right_small, -pr2_)

            error = left_small - r_right
            attention_map=self.SA2(torch.cat([left_small,right_small,error,pr2_],dim=1))
            pr1 = self.pred_flow1(iconv1*attention_map)
            pr1 = pr2_ + pr1
        else:
            pr1 = self.pred_flow1(iconv1)
        upconv0 = self.upconv0(iconv1)
        upflow1 = self.upflow1to0(pr1)
        concat0 = torch.cat((upconv0, upflow1, img_left), 1)
        iconv0 = self.iconv0(concat0)

        # predict flow
        if self.maxdisp == -1:
            if self.attention:
                left_small = F.interpolate(img_left, size=(pr1.size()[-2]*2, pr1.size()[-1]*2))
                right_small = F.interpolate(img_right, size=(pr1.size()[-2]*2, pr1.size()[-1]*2))
                pr1_ = F.interpolate(pr1, scale_factor=2)
                pr1_ = pr1_ / (img_left.size()[2] / left_small.size()[2])
                r_right = warp_right_to_left(right_small, -pr1_)

                error = left_small - r_right
                attention_map = self.SA1(torch.cat([left_small, right_small, error, pr1_], dim=1))
                pr0 = self.pred_flow0(iconv0 * attention_map)
                pr0 = pr1_ + pr0
            else:
                pr0 = self.pred_flow0(iconv0)
            pr0 = self.relu(pr0)
        else:
            if self.attention:
                left_small = F.interpolate(img_left, size=(pr1.size()[-2]*2, pr1.size()[-1]*2))
                right_small = F.interpolate(img_right, size=(pr1.size()[-2]*2, pr1.size()[-1]*2))
                pr1_ = F.interpolate(pr1, scale_factor=2)
                pr1_ = pr1_ / (img_left.size()[2] / left_small.size()[2])
                r_right = warp_right_to_left(right_small, -pr1_)

                error = left_small - r_right
                attention_map = self.SA1(torch.cat([left_small, right_small, error, pr1_], dim=1))
                pr0 = self.disp_expand(iconv0* attention_map)
                pr0 = pr1_ + pr0
            else:
                pr0 = self.disp_expand(iconv0)
            pr0 = F.softmax(pr0, dim=1)
            pr0 = disparity_regression(pr0, self.maxdisp)

        # predict flow from dropout output
        # pr6 = self.pred_flow6(F.dropout2d(conv6b))
        # pr5 = self.pred_flow5(F.dropout2d(iconv5))
        # pr4 = self.pred_flow4(F.dropout2d(iconv4))
        # pr3 = self.pred_flow3(F.dropout2d(iconv3))
        # pr2 = self.pred_flow2(F.dropout2d(iconv2))
        # pr1 = self.pred_flow1(F.dropout2d(iconv1))
        # pr0 = self.pred_flow0(F.dropout2d(iconv0))

        # if self.training:
        #     # print("finish forwarding.")
        #     return pr0, pr1, pr2, pr3, pr4, pr5, pr6
        # else:
        #     return pr0

        disps = (pr0, pr1, pr2, pr3, pr4, pr5, pr6)


        # can be chosen outside
        if self.get_features:
            features = (iconv5, iconv4, iconv3, iconv2, iconv1, iconv0)
            return disps, features
        else:
            return disps
 
    def freeze(self):
        for name, param in self.named_parameters():
            if ('weight' in name) or ('bias' in name):
                param.requires_grad = False

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


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
