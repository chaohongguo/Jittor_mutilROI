import math
import time

import numpy as np
import jittor
import jittor.nn as nn

# from .vit import ViT
from utils.geometry import rot6d_to_rotmat
from .pos_enc import PositionalEncoding


class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
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
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class HMR_sim(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params, n_extraviews=4, bbox_type='rect', wo_enc=False, wo_fuse=False,
                 encoder=None):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        self.inplanes = 64
        super(HMR_sim, self).__init__()
        npose = 24 * 6
        nbbox = 3
        self.n_extraviews = n_extraviews
        self.d_in = 3
        self.wo_enc = wo_enc
        self.wo_fuse = wo_fuse
        self.pos_enc = PositionalEncoding(input_dim=self.d_in)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        if bbox_type == 'square':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        elif bbox_type == 'rect':
            self.avgpool = nn.AvgPool2d((8, 6), stride=1)
        self.fc1 = nn.Linear((512 * block.expansion + nbbox * (self.n_extraviews + 1)) + npose + 13, 2048)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        self.fuse_fc = nn.Linear(512 * block.expansion + self.pos_enc.output_dim, 256)
        self.attention = nn.Sequential(
            nn.Linear(256 * (self.n_extraviews + 1), 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.n_extraviews + 1),
            nn.Tanh()
        )
        self.proj_head = nn.Sequential(
            nn.Linear(512 * block.expansion, 512 * block.expansion),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(),
            nn.Linear(512 * block.expansion, 512 * block.expansion, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
        )
        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                jittor.init.gauss_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight = jittor.ones_like(m.weight)
                m.bias = jittor.zeros_like(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                print(m)
                m.weight = jittor.ones_like(m.weight)
                m.bias = jittor.zeros_like(m.bias)

        mean_params = np.load(smpl_mean_params)
        self.init_pose = jittor.float32(mean_params['pose'][:]).unsqueeze(0)
        self.init_shape = jittor.float32(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.init_cam = jittor.float32(mean_params['cam']).unsqueeze(0)
        # self.register_buffer('init_pose', init_pose)
        # self.register_buffer('init_shape', init_shape)
        # self.register_buffer('init_cam', init_cam)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def execute(self, x_all, bbox_info_all, init_pose=None, init_shape=None, init_cam=None, n_iter=3, n_extraviews=4):
        # print(x_all.shape)

        batch_size, _, h, w = x_all.shape  # (B, 5*c, h, w) #(B, 5*c)
        # print(batch_size, _, h, w)
        n_views = self.n_extraviews + 1
        # print('n_views', n_views)

        if init_pose is None:
            init_pose = self.init_pose.expand(n_views * batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(n_views * batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(n_views * batch_size, -1)

        # start = time.time()

        x = x_all.view(-1, 3, h, w)  # (5*B, c, h, w)
        # print('x', x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        # print(xf.shape)
        xf = xf.view(batch_size, n_views, -1)  # (B, 5, 2048)

        xf_ = self.relu(xf.view(n_views * batch_size, -1))
        alpha = self.sigmoid(xf_)
        xf_hidden = jittor.multiply(xf.view(n_views * batch_size, -1), alpha)
        # print(xf_.shape, alpha.shape, xf_hidden.shape)
        # print(torch.mean(alpha, dim=-1)[0])
        xf_g = self.proj_head(xf_hidden)

        # print('xf', xf.shape)
        bbox_info_all = bbox_info_all.view(batch_size, n_views, 3)  # (B, 5, 3)

        if not self.wo_fuse:
            extra_inds = jittor.arange(n_views).unsqueeze(-1).repeat(1, n_views).view(-1, )
            main_inds = jittor.arange(n_views).unsqueeze(0).repeat(n_views, 1).view(-1, )
            # print(extra_inds, main_inds)
            bbox_trans = bbox_info_all[:, extra_inds, :self.d_in] - bbox_info_all[:, main_inds,
                                                                    :self.d_in]  # (B, 25, 3)
            if not self.wo_enc:
                bbox_trans_emb = self.pos_enc(bbox_trans.view(-1, self.d_in))
                xf = xf.repeat(1, n_views, 1).view(n_views * batch_size, n_views, -1)  # (5*B, 5, 2048)
                xf_cat = jittor.concat([xf, bbox_trans_emb.view(n_views * batch_size, n_views, -1)],
                                   -1)  # (5*B, 5, 2048+195)
                # print('bbox_trans_emb',bbox_trans_emb.shape) # (25*B, 195)
            else:
                # print("Not using relative encodings !!")
                xf = xf.repeat(1, n_views, 1).view(n_views * batch_size, n_views, -1)  # (5*B, 5, 2048)
                xf_cat = xf  # (5*B, 5, 2048)

            # xf_cat = torch.cat([xf, bbox_trans[:, :5]], -1) # (B, 5, 2050)
            # xf_cat = torch.cat([xf, bbox_trans_emb.view(batch_size, n_views*n_views, -1)[:, :5]], -1) # (B, 5, 2048+195)
            # print('xf_cat', xf_cat.shape)

            xf_attention = self.fuse_fc(xf_cat).view(n_views * batch_size, -1)
            # print('xf_cat', xf_attention.shape)
            xf_attention = self.attention(xf_attention)
            xf_attention = self.softmax(xf_attention)
            # print('attention', xf_attention.shape)
            xf_out = jittor.multiply(xf, xf_attention[:, :, None])
            # print(xf_out.shape)
            xf_out = jittor.sum(xf_out, dim=1)
            # print('xf_out', xf_out.shape)

            # xf = xf.view(xf.size(0), -1) #(5*B, 2048)
            # xf_cat = torch.cat([xf, bbox_info_all.view(-1, 3)], 1) #(5*B, 2051)
            # xf_cat = xf_cat.view(batch_size, n_views, -1)
            # print('xf_cat', xf_cat.shape)
            # end = time.time()
            # print("Parallel Time:", end - start)

        else:
            # print("Not using fusion module !!")
            xf_out = xf.view(n_views * batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        # start = time.time()
        for i in range(n_iter):
            xc = jittor.concat(
                [xf_out, bbox_info_all.repeat(1, n_views, 1).view(n_views * batch_size, -1), pred_pose, pred_shape,
                 pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(n_views * batch_size, 24, 3, 3)
        end = time.time()
        # print("Regression Time:", end - start)

        return pred_rotmat, pred_shape, pred_cam, xf_g
