from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
import gc
import time

#   当前改动：
#   backbone降到1/8
#   hourglass的attention去除
class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)         # [b,64,1/4,1/4]
        l3 = self.layer3(l2)        # [b,128,1/8,1/8]
        l4 = self.layer4(l3)        # [b,128,1/8,1/8]

        gwc_feature = torch.cat((l3, l4), dim=1)    # [b,256,1/4,1/4]

        return {"gwc_feature": gwc_feature,'corr_feature':l2}

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        # self.attention_block = attention_block(channels_3d=in_channels * 4, num_heads=16, block=(4, 4, 4))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # conv4 = self.attention_block(conv4)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6

class origin_agg(nn.Module):
    def __init__(self):
        super(origin_agg, self).__init__()
        self.conv3_1_o = ResBlock(48, 128)
        self.conv4_o = ResBlock(128, 256, stride=2)           # 1/16
        self.conv4_1_o = ResBlock(256, 256)
        self.conv5_o = ResBlock(256, 256, stride=2)           # 1/32
        self.conv5_1_o = ResBlock(256, 256)
        self.conv6_o = ResBlock(256, 512, stride=2)          # 1/64
        self.conv6_1_o = ResBlock(512, 512)
        self.iconv5_o = nn.ConvTranspose2d(512, 256, 3, 1, 1)
        self.iconv4_o = nn.ConvTranspose2d(384, 128, 3, 1, 1)
        self.iconv3_o = nn.ConvTranspose2d(192, 64, 3, 1, 1)
        self.upconv5_o = deconv(512, 256)
        self.upconv4_o = deconv(256, 128)
        self.upconv3_o = deconv(128, 64)
        self.disp3_o = nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self,out_corr):

        conv3b = self.conv3_1_o(out_corr)    # 128
        conv4a = self.conv4_o(conv3b)
        conv4b = self.conv4_1_o(conv4a)       # 256 1/16
        conv5a = self.conv5_o(conv4b)
        conv5b = self.conv5_1_o(conv5a)       # 256 1/32
        conv6a = self.conv6_o(conv5b)
        conv6b = self.conv6_1_o(conv6a)       # 512 1/64

        upconv5 = self.upconv5_o(conv6b)      # 256 1/32
        concat5 = torch.cat((upconv5, conv5b), dim=1)   # 512 1/32
        iconv5 = self.iconv5_o(concat5)       # 256

        upconv4 = self.upconv4_o(iconv5)      # 128 1/16
        concat4 = torch.cat((upconv4, conv4b), dim=1)   # 384 1/16
        iconv4 = self.iconv4_o(concat4)       # 128 1/16

        upconv3 = self.upconv3_o(iconv4)      # 64 1/8
        concat3 = torch.cat((upconv3, conv3b), dim=1)    # 192 1/8
        iconv3 = self.iconv3_o(concat3)       # 64
        pr3_o = self.disp3_o(iconv3)
        return pr3_o

class ACVNet(nn.Module):
    def __init__(self, maxdisp=192):
        super(ACVNet, self).__init__()
        self.maxdisp = maxdisp

        self.num_groups = 32
        self.concat_channels = 32
        self.feature_extraction = feature_extraction()
        self.concatconv = nn.Sequential(convbn(128, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, self.concat_channels, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

        self.patch = nn.Conv3d(32, 32, kernel_size=(1,3,3), stride=1, dilation=1, groups=32, padding=(0,1,1), bias=False)
        self.patch_l1 = nn.Conv3d(8, 8, kernel_size=(1,3,3), stride=1, dilation=1, groups=8, padding=(0,1,1), bias=False)
        self.patch_l2 = nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=1, dilation=2, groups=16, padding=(0,2,2), bias=False)
        self.patch_l3 = nn.Conv3d(8, 8, kernel_size=(1,3,3), stride=1, dilation=3, groups=8, padding=(0,3,3), bias=False)

        self.dres1_att_ = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
        self.dres2_att_ = hourglass(32)
        self.classif_att_ = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))


        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.agg2d_0=origin_agg()
        self.agg2d_1 = origin_agg()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):


        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)
        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 8, self.num_groups)
        gwc_volume = self.patch(gwc_volume)
        patch_l1 = self.patch_l1(gwc_volume[:, :8])
        patch_l2 = self.patch_l2(gwc_volume[:, 8:24])
        patch_l3 = self.patch_l3(gwc_volume[:, 24:32])
        patch_volume = torch.cat((patch_l1,patch_l2,patch_l3), dim=1)
        cost_attention = self.dres1_att_(patch_volume)
        cost_attention = self.dres2_att_(cost_attention)
        att_weights = self.classif_att_(cost_attention)     # [b,1,24,1/8,1/8]


        corr_feature_left = features_left["corr_feature"]
        corr_feature_right = features_right["corr_feature"]


        corr_volume=build_corr(corr_feature_left,corr_feature_right,self.maxdisp // 4)

        ac_volume = F.softmax(F.upsample(att_weights,scale_factor=2,mode='trilinear').squeeze(dim=1), dim=1) * corr_volume

        cost0 = self.agg2d_0(ac_volume)
        #cost0 = self.agg2d_1(cost0) + cost0


        if self.training:

            cost_attention = F.upsample(att_weights, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost_attention = torch.squeeze(cost_attention, 1)
            pred_attention = F.softmax(cost_attention, dim=1)
            pred_attention = disparity_regression(pred_attention, self.maxdisp)



            cost0 = F.upsample(cost0.unsqueeze(1), [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            return [pred_attention, pred0]


        else:

            cost0 = self.classif0(cost0)

            cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            return [pred0]

def acv(d):
    return ACVNet(d)


if __name__=='__main__':
    right=torch.randn([1,3,128,128])
    left=torch.randn([1,3,128,128])
    net=ACVNet()
    start=time.time()
    for _ in range(10):
        out=net(left,right)
    print(time.time()-start) #4.1