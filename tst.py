from pickletools import optimize
from re import A
import sys
# sys.path.append('../../simple_CAM/torch-cam')
import torchcam
import pandas
import os
import pdb

from audioop import cross
from tkinter import Y
from turtle import pd
from numpy.core.fromnumeric import shape
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
# from _backbones import C3D_backbone, SkeletonConv2D, create_backbone, record_conv3d_info
from tqdm import tqdm
import pickle

print(torch.__version__)

from tools.datasets.PIE_JAAD import PIEDataset
from tools.datasets.pie_data import PIE
from tools.datasets.jaad_data import JAAD
from _backbones import create_backbone
import utils
from utils import TITANclip_txt2list, idx2onehot, cls_weights
from tools.datasets.TITAN import TITAN_dataset

import matplotlib.pyplot as plt
from _plot import vis_weight
from tools.datasets.TITAN import ATOM_ACTION_LABEL
from tools.plot import vis_ego_sample

from tools.transforms import RandomResizedCrop, RandomHorizontalFlip
from utils import seed_all
from torchviz import make_dot

# from torchviz import make_dot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pie_opts = {'normalize_bbox': False,
                         'fstride': 1,
                         'sample_type': 'all',
                         'height_rng': [0, float('inf')],
                         'squarify_ratio': 0,
                         'data_split_type': 'default',  # kfold, random, default. default: set03 for test
                         'seq_type': 'crossing',  # crossing , intention
                         'min_track_size': 0,  # discard tracks that are shorter
                         'max_size_observe': 16,  # number of observation frames
                         'max_size_predict': 16,  # number of prediction frames
                         'seq_overlap_rate': 0.6,  # how much consecutive sequences overlap
                         'balance': False,  # balance the training and testing samples
                         'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                         'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                         'encoder_input_type': ['bbox', 'obd_speed'],
                         'decoder_input_type': [],
                         'output_type': ['intention_binary', 'bbox']
                         }
jaad_opts = {'fstride': 1,
             'sample_type': 'all',  
	         'subset': 'default',
             'data_split_type': 'default',
             'seq_type': 'intention',
	         'height_rng': [0, float('inf')],
	         'squarify_ratio': 0,
             'min_track_size': 0,
             'random_params': {'ratios': None,
                               'val_data': True,
                               'regen_data': True},
             'kfold_params': {'num_folds': 5, 'fold': 1}}


img_setting = {'backbone_name':'C3D',
            'separate_backbone':1,
            'conditioned_proto':1,
            'proto_generator_name':'C3D',
            'num_explain':5,
            'conditioned_relevance':1,
            'relevance_generator_name':'C3D',
            'num_proto':10,
            'proto_dim':512,
            'simi_func':'dot',
            'freeze_base':0,
            'freeze_proto':0,
            'freeze_relev':0}

ckpt_path = '../work_dirs/models/i3d_model_rgb.pth'
heatmap_path = '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/skeletons/even_padded/288w_by_384h/1/0/heatmaps.pkl'
coord_path = '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/sk_coords/even_padded/288w_by_384h/1/0/000006.pkl'
p_heatmap_path = '/home/y_feng/workspace6/datasets/PIE_dataset/sk_pseudo_heatmaps/even_padded/48w_by_48h/1_1_1/01013.pkl'

csv1_path = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset/clip_1.csv'
csv2_path = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset/clip_786/synced_sensors.csv'
default_train_path = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset/train_set.txt'
default_val_path = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset/val_set.txt'
default_test_path = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset/test_set.txt'


img_path = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset/images_anonymized/clip_1/images/000012.png'

model_path = '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models_1/backbone_img/17Nov2022-20h02m49s/ckpt/2_0.8322.pth'

# titan = TITAN_dataset(sub_set='default_test', norm_traj=1,
#                                       obs_len=16, pred_len=1, overlap_ratio=1, 
#                                       required_labels=[
#                                                         'atomic_actions', 
#                                                         'simple_context', 
#                                                         'complex_context', 
#                                                         'communicative', 
#                                                         'transporting',
#                                                         'age'
#                                                         ], 
#                                       multi_label_cross=1,  
#                                     #   use_cross=1,
#                                       use_atomic=1, 
#                                       use_complex=1, 
#                                       use_communicative=1, 
#                                       use_transporting=1, 
#                                       use_age=1,
#                                       tte=None,
#                                       small_set=0,
#                                       use_img=1,
#                                       use_skeleton=1,
#                                       use_ctx=0, ctx_mode='ori_local',
#                                       use_traj=1,
#                                       use_ego=0,
#                                       augment_mode='random_crop_hflip'
#                                       )


# pie = PIEDataset(dataset_name='PIE', seq_type='crossing',
#                                     obs_len=15, pred_len=1, obs_interval=1,
#                                     do_balance=False, subset='train', bbox_size=(224, 224), 
#                                     img_norm_mode='torch', color_order='BGR',
#                                     resize_mode='even_padded', 
#                                     use_img=1, 
#                                     use_skeleton=0, skeleton_mode='heatmap',
#                                     use_context=0, ctx_mode='ori_local', 
#                                     use_traj=1, traj_mode='ltrb0-1',
#                                     use_ego=1,
#                                     small_set=0,
#                                     overlap_retio=0.6,
#                                     tte=None,
#                                     recog_act=0,
#                                     normalize_pos=0,
#                                     ego_accel=1,
#                                     speed_unit='m/s')

# torch.manual_seed(42)

# class Model(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.m1 = nn.Sequential(
#             nn.Linear(2, 4),
#             nn.ReLU(),
#         )
#         self.m2 = nn.Linear(4, 4)

#     def forward(self,x):
#         f = self.m1(x)
#         res = self.m2(f)

#         return res, f


# x = torch.randn(4, 2).cuda()
# x.requires_grad = True
# y = torch.arange(4).cuda()

# model = Model().cuda()
# model_p = nn.parallel.DistributedDataParallel(model)

# res, f = model_p(x)
# loss_single = F.cross_entropy(res, y)

# print(torch.autograd.grad(loss_single, x, retain_graph=True,)[0])
# print(torch.autograd.grad(loss_single, f, retain_graph=True,)[0])

# 2.2369382 33.393887 -1.8985002 -66.65366

a = [0.2722, 0.1018, 0.3365, 0.6753, 0.5544, 0.2775]
print(np.mean(a))