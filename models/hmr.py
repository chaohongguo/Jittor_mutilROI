import math

import jittor as jt
import jittor.nn as nn
import numpy as np
from jittor.models.resnet import *
from utils.geometry import rot6d_to_rotmat
import jittor
import os
from .multi_ROI.mutil_resnet50_all import HMR_sim
from collections import OrderedDict


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.Relu()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class HMR(nn.Module):
    def __init__(self, block, layers, smpl_mean_params):
        super(HMR, self).__init__()
        npose = 24 * 6
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.Relu()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 10 + 3, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)

        self.init_weights()

        mean_params = np.load(smpl_mean_params)
        self.init_pose = jt.unsqueeze(jt.float32(mean_params['pose'][:]), 0)  # [1,144]
        self.init_shape = jt.unsqueeze(jt.float32(mean_params['shape'])[:], 0)  # [10]
        self.init_cam = jt.unsqueeze(jt.float32(mean_params['cam']), 0)  # [3]

    def init_weights(self):
        # Xavier 初始化 (对应 nn.init.xavier_uniform_)
        gain = 0.01
        jt.init.xavier_uniform_(self.decpose.weight, gain=gain)
        jt.init.xavier_uniform_(self.decshape.weight, gain=gain)
        jt.init.xavier_uniform_(self.deccam.weight, gain=gain)

        for m in self.modules():
            if isinstance(m, nn.Conv):
                n = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                std = math.sqrt(2. / n)
                m.weight = jt.init.gauss_(m.weight, mean=0, std=std)

            elif isinstance(m, nn.BatchNorm):
                m.weight = jt.ones_like(m.weight)
                m.bias = jt.zeros_like(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def execute(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = jt.misc.expand(self.init_pose, [batch_size, -1])  # [batch, 144]
        if init_shape is None:
            init_shape = jt.misc.expand(self.init_shape, [batch_size, -1])  # [batch, 10]
        if init_cam is None:
            init_cam = jt.misc.expand(self.init_cam, [batch_size, -1])  # [batch, 3]

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x1 = self.layer1(x)  # [batch,64*4,56,56]
        x2 = self.layer2(x1)  # [batch,128*4,28,28]
        x3 = self.layer3(x2)  # [batch,256*4,14,14]
        x4 = self.layer4(x3)  # [batch,512*4,7,7]

        xf = self.avgpool(x4)  # [batch,512*4,1,1]
        xf = xf.view(xf.size(0), -1)  # [batch,2048]

        pred_pose = init_pose  # [batch,144]
        pred_shape = init_shape  # [batch,10]
        pred_cam = init_cam  # [batch,3]
        for i in range(n_iter):
            xc = jt.concat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        pre_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        return pre_rotmat, pred_shape, pred_cam


def hmr(smpl_mean_params, pretrained=True, **kwargs):
    model = HMR(Bottleneck, [3, 4, 6, 3], smpl_mean_params, **kwargs)
    if pretrained:
        resnet_coco = jittor.load(os.path.realpath('./models/PreTrainBackbones/pose_resnet.pth'))
        old_dict = resnet_coco['state_dict']
        new_dict = OrderedDict([(k.replace('backbone.', ''), v) for k, v in old_dict.items()])
        # feature.load_state_dict(new_dict, strict=False) # old
        model.load_state_dict(new_dict)
    return model


def build_model(smpl_mean_params, pretrained=True, backbone='resnet',
                model_name="mutilROI", option=None, **kwargs):
    if backbone == 'resnet':

        if model_name == 'mutilROI':
            # model = MutilROI_resnet50(smpl_mean_params,
            #                           n_extra_views=option.n_views - 1,
            #                           is_fuse=option.is_fuse,
            #                           is_pos_enc=option.is_pos_enc)
            model = HMR_sim(Bottleneck, [3, 4, 6, 3], smpl_mean_params,
                            n_extraviews=option.n_views - 1,
                            wo_fuse=False,
                            wo_enc=False)
    if pretrained:
        print('Loading pretrained weights for ResNet backbone...')
        resnet_coco = jittor.load('models/backbones/pretrained/pose_resnet.pth')
        old_dict = resnet_coco['state_dict']
        from collections import OrderedDict

        new_dict = OrderedDict([
            (k.replace('backbone.', ''), v)
            for k, v in old_dict.items()
            if not any(param in k for param in [
                'data_preprocessor.mean',
                'data_preprocessor.std',
                'head.deconv_layers.0.weight',
                'head.deconv_layers.1.weight',
                'head.deconv_layers.1.bias',
                'head.deconv_layers.1.running_mean',
                'head.deconv_layers.1.running_var',
                'head.deconv_layers.3.weight',
                'head.deconv_layers.4.weight',
                'head.deconv_layers.4.bias',
                'head.deconv_layers.4.running_mean',
                'head.deconv_layers.4.running_var',
                'head.deconv_layers.6.weight',
                'head.deconv_layers.7.weight',
                'head.deconv_layers.7.bias',
                'head.deconv_layers.7.running_mean',
                'head.deconv_layers.7.running_var',
                'head.final_layer.weight',
                'head.final_layer.bias'
            ])
        ])
        model.load_state_dict(new_dict)


    return model
