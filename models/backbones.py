from ast import Raise
from turtle import forward
from numpy.lib.function_base import copy
import torch
import torch.nn as nn
from thop import profile
# from torchviz import make_dot
from torchsummary import summary
import os

from copy import deepcopy

import models.R3D as R3D
import models.I3D as I3D
from ._mmbackbones2 import create_mm_backbones
from config import ckpt_root

BACKBONE_TO_OUTDIM = {
    'C3D': 512,
    'C3D_new': 512,
    'C3D_clean': 512,
    'R3D18': 512,
    'R3D18_clean': 512,
    'R3D18_new': 512,
    'R3D34': 512,
    'R3D34_clean': 512,
    'R3D34_new': 512,
    'R3D50': 2048,
    'R3D50_new': 2048,
    'R3D50_clean': 2048,
    'ircsn152': 2048,
    'poseC3D_pretrained': 512,
    'poseC3D': 512,
    'poseC3D_clean': 512,
    'lstm': 128
}

class C3DDecoder(nn.Module):
    def __init__(self) -> None:  # 1, 8, 8 -> 16 224 224
        super(C3DDecoder, self).__init__()
        self.up_sample = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 0, 0)), # 1 6 6
            # nn.Upsample(size=(4, 7, 7)),  # 1 6 6 -> 1 7 7
            nn.ConstantPad3d((1, 0, 1, 0, 0, 0), 0),  # 1 6 6 -> 1 7 7
            nn.Upsample(scale_factor=(2, 2, 2)), # 2 14 14
            # nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            
            nn.Upsample(scale_factor=(2, 2, 2)), # 4 28 28
            # nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(512, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),

            nn.Upsample(scale_factor=(2, 2, 2)), # 8 56 56
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),

            nn.Upsample(scale_factor=(2, 2, 2)), # 16 112 112
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),

            nn.Upsample(scale_factor=(1, 2, 2)), # 16 224 224
            nn.Conv3d(64, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

        self._init_weight()
    
    def forward(self, x):
        x = self.up_sample(x)
        # print(x.size())
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)

class PoseC3DDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2)),  # 8 6 6 -> 8 12 12
            # nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)), # 8 6 6 -> 8 6 6
            # nn.ConstantPad3d((1, 0, 1, 0, 0, 0), 0),  # 1 6 6 -> 1 7 7
            # nn.Upsample(scale_factor=(2, 2, 2)), # 2 14 14
            # nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(512, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            
            nn.Upsample(scale_factor=(1, 2, 2)), # 8 24 24
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),

            nn.Upsample(scale_factor=(2, 2, 2)), # 16 48 48
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(64, 17, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

    def forward(self, x):
        return self.up_sample(x)


class RNNEncoder(nn.Module):
    def __init__(self, in_dim=4, h_dim=128, cell='lstm') -> None:
        super(RNNEncoder, self).__init__()
        if cell == 'lstm':
            self.encoder = nn.LSTM(batch_first=True, input_size=in_dim, hidden_size=h_dim)
        else:
            raise NotImplementedError('Ilegal cell type')

    def forward(self, x):
        # x: B, T, C
        # print(x.size())
        self.encoder.flatten_parameters()
        _, (h, c) = self.encoder(x)  # h: 1, b, c  c: 1, b, c

        return h[0]

class CNN_RNNEncoder(nn.Module):
    def __init__(self, h_dim=128, cell='lstm', cnn_name='vgg16') -> None:
        super(CNN_RNNEncoder, self).__init__()
        self.cnn_backbone = create_backbone(cnn_name)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_dim = 512
        if cell == 'lstm':
            self.encoder = nn.LSTM(batch_first=True, input_size=in_dim, hidden_size=h_dim)
        else:
            raise NotImplementedError('Ilegal cell type')
        
    def forward(self, x):
        # x: B, C, T, H, W
        obs_len = x.size(2)
        featmaps = []
        for i in range(obs_len):
            f = self.cnn_backbone(x[:, :, i])
            f = self.pool(f)  # B, C, 1, 1
            f = f.view(f.size(0), f.size(1))  # B, C
            featmaps.append(f)
        featmaps = torch.stack(featmaps, dim=1)  # B, T, C
        _, (h, c) = self.encoder(featmaps)

        return h[0]

class SegC2D(nn.Module):
    def __init__(self):
        super(SegC2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3a = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv3b = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        return x

class SegC3D(nn.Module):
    def __init__(self):
        super(SegC3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn3 = nn.BatchNorm3d(256)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        return x

class SkeletonConv2D(nn.Module):
    def __init__(self):
        super(SkeletonConv2D, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=(1, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=(1, 1))
        # self.bn4 = nn.BatchNorm2d(128)
        self._initialize_weights()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        # print('C3D output', torch.mean(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class C3DPose(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, bn=True):
        super(C3DPose, self).__init__()

        self.conv1 = nn.Conv3d(17, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(64)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn4 = nn.BatchNorm3d(512)

        self.conv5 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.last_channel = 512
        self.__init_weight()

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.bn1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.bn3(x)

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.bn4(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv5b(x))
        feat = self.pool5(x)

        return feat

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class C3D_backbone(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, pretrained=False, t_downsample='new'):
        super(C3D_backbone, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 16 112 112

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        if t_downsample == 'ori':
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        
        self.last_channel = 512
        self.__init_weight()

        self.pretrained_path = os.path.join(ckpt_root, 'c3d-pretrained.pth')
        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        feat = self.pool5(x)

        # x = x.view(-1, 8192)
        # x = self.relu(self.fc6(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc7(x))
        # x = self.dropout(x)
        #
        # logits = self.fc8(x)

        return feat

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias"
                        }

        p_dict = torch.load(self.pretrained_path)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)
        print('Total layer num : {}, update layer num: {}'.format(len(p_dict.keys()), len(s_dict.keys())))
        # print('Total layer: ', p_dict.keys())
        # print('Update layer: ', s_dict.keys())

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class C3D_full(nn.Module):
    """
    The C3D network.
    """
    def __init__(self, last_dim, pretrained=False):
        super(C3D_full, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))  # 

        # self.fc6 = nn.Linear(8192, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        # self.fc8 = nn.Linear(4096, last_dim)

        # self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.pretrained_path = os.path.join(ckpt_root, 'c3d-pretrained.pth')
        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        # import pdb;pdb.set_trace()
        logits = x.reshape(-1, 8192)
        # x = self.relu(self.fc6(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc7(x))
        # x = self.dropout(x)

        # logits = self.fc8(x)

        return logits

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        # "classifier.0.weight": "fc6.weight",
                        # "classifier.0.bias": "fc6.bias",
                        # # fc7
                        # "classifier.3.weight": "fc7.weight",
                        # "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(self.pretrained_path)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class BackboneOnly(nn.Module):
    def __init__(self, backbone_name):
        super(BackboneOnly, self).__init__()
        self.backbone = create_backbone(backbone_name)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        if 'C3D' or 'R3D' in backbone_name:
            self.fc = nn.Linear(512, 2)
        else:
            self.fc = nn.Linear(512, 2)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def create_backbone(backbone_name, lstm_h_dim=128, lstm_input_dim=4, last_dim=487):
    if backbone_name == 'C3D_new':  # 3, 16, 224, 224 -> 512, 1, 8, 8
        backbone = C3D_backbone(pretrained=True, t_downsample='new')
    elif backbone_name == 'C3D':
        backbone = C3D_backbone(pretrained=True, t_downsample='ori')
    elif backbone_name == 'C3D_clean':
        backbone = C3D_backbone(pretrained=False, t_downsample='ori')
    elif backbone_name == 'C3D_full':
        backbone = C3D_full(last_dim=last_dim, pretrained=True)
    elif backbone_name == 'R3D18':  # 3, 16, 224, 224 -> 512, 1, 7, 7
        pretrained_path = os.path.join(ckpt_root, 'r3d18_KM_200ep.pth')
        backbone = R3D.generate_model(18)
        pretrained = torch.load(pretrained_path, map_location='cpu')
        p_dict = pretrained['state_dict']  # keys必须完全一致
        for name in list(p_dict.keys()):
            if name[:2] == 'fc':
                p_dict.pop(name)
        backbone.load_state_dict(p_dict)
    elif backbone_name == 'R3D18_clean':
        backbone = R3D.generate_model(18)
    # elif backbone_name == 'R3D34':
    #     backbone = _R3D.generate_model(34, pretrained=True)
    # elif backbone_name == 'R3D34_new':
    #     backbone = _R3D.generate_model(34, t_downsample=False)
    elif backbone_name == 'R3D50':  # 3, 16, 224, 224 -> 2048 1 7 7
        pretrained_path = os.path.join(ckpt_root, 'r3d50_KMS_200ep.pth')
        backbone = R3D.generate_model(50)
        pretrained = torch.load(pretrained_path, map_location='cpu')
        p_dict = pretrained['state_dict']  # keys必须完全一致
        for name in list(p_dict.keys()):
            if name[:2] == 'fc':
                p_dict.pop(name)
        backbone.load_state_dict(p_dict)
    elif backbone_name == 'R3D50_no_max':  # 3, 16, 224, 224 -> 2048 1 7 7
        pretrained_path = os.path.join(ckpt_root, 'r3d50_KMS_200ep.pth')
        backbone = R3D.generate_model(50, no_max_pool=True)
        pretrained = torch.load(pretrained_path, map_location='cpu')
        p_dict = pretrained['state_dict']  # keys必须完全一致
        for name in list(p_dict.keys()):
            if name[:2] == 'fc':
                p_dict.pop(name)
        backbone.load_state_dict(p_dict)
    elif backbone_name == 'R3D50_clean':
        backbone = R3D.generate_model(50, pretrained=False)
    elif backbone_name == 'R3D50_new':
        pretrained_path = os.path.join(ckpt_root, 'r3d50_KMS_200ep.pth')
        backbone = R3D.generate_model(50, t_downsample=False)
        pretrained = torch.load(pretrained_path, 
                                map_location='cpu')
        p_dict = pretrained['state_dict']  # keys必须完全一致
        for name in list(p_dict.keys()):
            if name[:2] == 'fc':
                p_dict.pop(name)
        backbone.load_state_dict(p_dict)
    elif backbone_name == 'I3D':
        backbone = I3D.I3D_backbone()
        p_dict = torch.load(os.path.join(ckpt_root, 'i3d_model_rgb.pth'))
        p_dict.pop('conv3d_0c_1x1.conv3d.weight')
        p_dict.pop('conv3d_0c_1x1.conv3d.bias')
        backbone.load_state_dict(p_dict)
    elif backbone_name == 'I3D_clean':
        backbone = I3D.I3D_backbone()
    elif backbone_name == 'SK':
        backbone = SkeletonConv2D()
    elif backbone_name == 'segC3D':
        backbone = SegC3D()
    elif backbone_name == 'segC2D':
        backbone = SegC2D()
    elif backbone_name == 'cnn_lstm':
        backbone = CNN_RNNEncoder(h_dim=lstm_h_dim)
    elif backbone_name == 'lstm':
        backbone = RNNEncoder(h_dim=lstm_h_dim, in_dim=lstm_input_dim)
    elif backbone_name == 'C3Dpose':
        backbone = C3DPose()
    elif backbone_name == 'poseC3D':  # (17, 15/16, 48, 48) -> (512, 8, 6, 6)
        backbone = create_mm_backbones(backbone_name, pretrain=True)
    elif backbone_name == 'poseC3D_pretrained':  # (17, 15/16, 48, 48) -> (512, 8, 6, 6)
        backbone = create_mm_backbones(backbone_name, pretrain=True)
    elif backbone_name == 'poseC3D_clean':  # (17, 15/16, 48, 48) -> (512, 8, 6, 6)
        backbone = create_mm_backbones(backbone_name, pretrain=False)
    elif backbone_name == 'ircsn152':
        backbone = create_mm_backbones(backbone_name, pretrain=True)
    elif backbone_name == 'poseC3Ddecoder':
        backbone = PoseC3DDecoder()
    elif backbone_name == 'C3Ddecoder':
        backbone = C3DDecoder()
    else:
        raise ValueError(backbone_name)
    return backbone

def record_conv3d_info(model):
    kernel_size_list = []
    stride_list = []
    padding_list = []
    for name, m in model.named_modules():
        if (isinstance(m, nn.Conv3d) or isinstance(m, nn.MaxPool3d)) and 'downsample' not in name:
            kernel_size_list.append(m.kernel_size)
            stride_list.append(m.stride)
            padding_list.append(m.padding)
    return kernel_size_list, stride_list, padding_list

def record_conv2d_info(model):
    kernel_size_list = []
    stride_list = []
    padding_list = []
    for name, m in model.named_modules():
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d)) and 'downsample' not in name:
            kernel_size_list.append(m.kernel_size)
            stride_list.append(m.stride)
            padding_list.append(m.padding)
    return kernel_size_list, stride_list, padding_list

def record_sp_conv3d_info_w(conv_info):
    k_list = []
    s_list = []
    p_list = []
    for i in range(len(conv_info[0])):
        cur_k = conv_info[0][i]
        cur_s = conv_info[1][i]
        cur_p = conv_info[2][i]
        k_list.append(cur_k if isinstance(cur_k, int) else cur_k[-1])
        s_list.append(cur_s if isinstance(cur_s, int) else cur_s[-1])
        p_list.append(cur_p if isinstance(cur_p, int) else cur_p[-1])
    return [k_list, s_list, p_list]

def record_sp_conv3d_info_h(conv_info):
    k_list = []
    s_list = []
    p_list = []
    for i in range(len(conv_info[0])):
        cur_k = conv_info[0][i]
        cur_s = conv_info[1][i]
        cur_p = conv_info[2][i]
        k_list.append(cur_k if isinstance(cur_k, int) else cur_k[-2])
        s_list.append(cur_s if isinstance(cur_s, int) else cur_s[-2])
        p_list.append(cur_p if isinstance(cur_p, int) else cur_p[-2])
    return [k_list, s_list, p_list]

def record_sp_conv2d_info_h(conv_info):
    k_list = []
    s_list = []
    p_list = []
    for i in range(len(conv_info[0])):
        cur_k = conv_info[0][i]
        cur_s = conv_info[1][i]
        cur_p = conv_info[2][i]
        k_list.append(cur_k if isinstance(cur_k, int) else cur_k[-2])
        s_list.append(cur_s if isinstance(cur_s, int) else cur_s[-2])
        p_list.append(cur_p if isinstance(cur_p, int) else cur_p[-2])
    return [k_list, s_list, p_list]

def record_sp_conv2d_info_w(conv_info):
    k_list = []
    s_list = []
    p_list = []
    for i in range(len(conv_info[0])):
        cur_k = conv_info[0][i]
        cur_s = conv_info[1][i]
        cur_p = conv_info[2][i]
        k_list.append(cur_k if isinstance(cur_k, int) else cur_k[-1])
        s_list.append(cur_s if isinstance(cur_s, int) else cur_s[-1])
        p_list.append(cur_p if isinstance(cur_p, int) else cur_p[-1])
    return [k_list, s_list, p_list]

def record_t_conv3d_info(conv_info):
    k_list = []
    s_list = []
    p_list = []
    for i in range(len(conv_info[0])):
        cur_k = conv_info[0][i]
        cur_s = conv_info[1][i]
        cur_p = conv_info[2][i]
        k_list.append(cur_k if isinstance(cur_k, int) else cur_k[0])
        s_list.append(cur_s if isinstance(cur_s, int) else cur_s[0])
        p_list.append(cur_p if isinstance(cur_p, int) else cur_p[0])
    return [k_list, s_list, p_list]

def load_pretrained_resnet(model, path):
    pass

if __name__ == "__main__":
    inputs = torch.ones(1, 3, 16, 224, 224)  # B, C, T, H, W
    # net = C3D_backbone(pretrained=True)
    net = create_backbone('R3D50')
    print(net)
    # print(net)
    # for m in net.modules():
    #     print(m)
    # for name, para in net.named_parameters():
    #     print(name, ':')

    summary(net, input_size=[(3, 16, 224, 224)], batch_size=1, device="cpu")
    # flops, paras = profile(model=net, inputs=(inputs,))
    # print('flops:', flops)
    # print('params:', paras)
    # import pdb; pdb.set_trace()

    outputs = net.forward(inputs)
    print(outputs.size())
    # vise = make_dot(outputs, params=dict(net.named_parameters()))
    # vise.render(filename='wrong_forw', view=False, format='pdf')
    
    # conv_info = record_conv3d_info(net)
    # sp_k_list, sp_s_list, sp_p_list = record_sp_conv3d_info_w(conv_info)
    # t_k_list, t_s_list, t_p_list = record_t_conv3d_info(conv_info)
    # from receptive_field import compute_proto_layer_rf_info_v2
    # sp_proto_layer_rf_info = compute_proto_layer_rf_info_v2(input_size=224,
    #                                                      layer_filter_sizes=sp_k_list,
    #                                                      layer_strides=sp_s_list,
    #                                                      layer_paddings=sp_p_list,
    #                                                      prototype_kernel_size=1)
    # t_proto_layer_rf_info = compute_proto_layer_rf_info_v2(input_size=16,
    #                                                      layer_filter_sizes=t_k_list,
    #                                                      layer_strides=t_s_list,
    #                                                      layer_paddings=t_p_list,
    #                                                      prototype_kernel_size=1)
    # print(sp_k_list, sp_s_list, sp_p_list)
    # print(sp_proto_layer_rf_info)
    # print(t_proto_layer_rf_info)

    
    # import pdb;pdb.set_trace()