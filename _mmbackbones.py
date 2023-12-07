import torch

import argparse
import os
import os.path as osp
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import (build_ddp, build_dp, default_device,
                            register_module_hooks, setup_multi_processes)

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
        cfg = Config.fromfile(cfg_path)
        cfg.merge_from_dict({})
        turn_off_pretrained(cfg.model)
        model = build_model(
                            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        if pretrain:
            load_checkpoint(model, ckpt_path, map_location='cpu')

        return model.backbone



if __name__ == '__main__':
    backbone = create_mm_backbones()
    d = torch.rand(size=(1, 17, 16, 384, 288))
    print(type(backbone))
    for n, p in backbone.named_parameters():
        print(n, p.size())
    
    output = backbone(d)
    print(output.size())
    from _backbones import create_backbone
    backbone2 = create_backbone('C3Dpose')
    output2 = backbone2(d)
    print(output2.size())