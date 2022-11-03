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

        return {"gwc_feature": gwc_feature,'concat_feature':l2}

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

        self.dres0 = nn.Sequential(convbn_3d(self.concat_channels * 4, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))



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

        up_cost=F.upsample(att_weights,scale_factor=2,mode='trilinear')
        up_cost=torch.squeeze(up_cost,dim=1)  # [b,48,1/4,1/4]
        up_cost=F.softmax(up_cost,dim=1)
        topk,disp_range=torch.sort(up_cost,dim=1,descending=True)
        disp_range=disp_range[:,:6]     # [b,k,1/4,1/4]
        topk=topk[:,:6]                 # [b,k,1/4,1/4]

        concat_feature_left = features_left["concat_feature"]
        concat_feature_right = features_right["concat_feature"]

        concat_volume=cost_from_disp(concat_feature_left,concat_feature_right,disp_range_samples=disp_range,mode='concat')

        ac_volume = topk * concat_volume   ### ac_volume = att_weights * concat_volume
        cost0 = self.dres0(ac_volume)
        cost0 = self.dres1(cost0) + cost0


        if self.training:

            cost_attention = F.upsample(att_weights, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost_attention = torch.squeeze(cost_attention, 1)
            pred_attention = F.softmax(cost_attention, dim=1)
            pred_attention = disparity_regression(pred_attention, self.maxdisp)

            cost0 = self.classif0(cost0)

            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = torch.sum(pred0 * disp_range*4, 1, keepdim=True)

            pred0=F.upsample(pred0,[left.size()[2], left.size()[3]],mode='bilinear')
            pred0=torch.squeeze(pred0,dim=1)
            return [pred_attention, pred0]


        else:

            cost0 = self.classif0(cost0)

            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = torch.sum(pred0 * disp_range*4, 1, keepdim=True)
            pred0 = F.upsample(pred0, [left.size()[2], left.size()[3]], mode='bilinear')
            pred0 = torch.squeeze(pred0, dim=1)
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
    print(time.time()-start) #2.92