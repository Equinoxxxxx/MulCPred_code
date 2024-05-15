import torch

import argparse
import os
import os.path as osp
import warnings

import mmcv
import torch
from mmaction.apis import inference_recognizer, init_recognizer
from pathlib import Path
from typing import List, Optional, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import track_iter_progress

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample

def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)

def create_mm_backbones(backbone_name='poseC3D', pretrain=True):
    # parser = argparse.ArgumentParser(
    #     description='MMAction2 test (and eval) a model')
    # parser.add_argument(
    #     '--cfg-options',
    #     nargs='+',
    #     action=DictAction,
    #     default={},
    #     help='override some settings in the used config, the key-value pair '
    #     'in xxx=yyy format will be merged into config file. For example, '
    #     "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    if 'poseC3D' in backbone_name:
        # ckpt_path = '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint-cae8aa4a.pth'
        ckpt_path = '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint_20220815-9972260d.pth'
        cfg_path = '/home/y_feng/workspace6/work/open-mmlab/mmaction2/configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint.py'
        # ckpt_path = ''
        # cfg_path = ''
        # config = mmengine.Config.fromfile(cfg_path)
        # model = MODELS.build(config.model)
        # if pretrain:
        #     load_checkpoint(model, ckpt_path, map_location='cpu')
        model = init_recognizer(cfg_path, ckpt_path, device='cpu')

        return model.backbone
    
    elif backbone_name == 'ircsn152':
        ckpt_path = '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb_20220811-c7a3cc5b.pth'
        cfg_path = '/home/y_feng/workspace6/work/mmaction2/mmaction2/configs/recognition/csn/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb.py'
        model = init_recognizer(cfg_path, ckpt_path, device='cpu')

        return model.backbone



if __name__ == '__main__':
    backbone = create_mm_backbones('ircsn152')
    d = torch.rand(size=(1, 3, 10, 224, 224))
    print(type(backbone))
    # for n, p in backbone.named_parameters():
    #     print(n, p.size())
    output = backbone(d)
    print(output.size())
