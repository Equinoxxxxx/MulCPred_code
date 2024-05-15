from pickletools import optimize
from re import A
import sys
# sys.path.append('../../simple_CAM/torch-cam')
import torchcam
import pandas

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
from models.backbones import C3D_backbone, SkeletonConv2D, create_backbone, record_conv3d_info
from _proto_model import ImagePNet, SkeletonPNet, MultiBackbone
from _train_test import joint, warm_only, last_only
import pdb
import argparse
import os
import pickle
from tqdm import tqdm
import csv
import sklearn

print(torch.__version__)

from _datasets import PIEDataset
from pie_data import PIE
from jaad_data import JAAD
from _SLENN import MultiSLE, SLEseq, SLE3D
import utils
from utils import TITANclip_txt2list, idx2onehot, cls_weights
from _TITAN_dataset import TITAN_dataset

import matplotlib.pyplot as plt
from _plot import vis_weight
from _TITAN_dataset import ATOM_ACTION_LABEL
from _TED import Transformer
from _TEO import TransformerClassifier

from torchvision.transforms import transforms
from tools.transforms import RandomResizedCrop, RandomHorizontalFlip
from utils import seed_all, draw_boxes_on_img, vid_id_int2str, img_nm_int2str
from torchvision.transforms import functional as tvf
import copy

def main():
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


    dataset_name = 'TITAN'

    if dataset_name == 'TITAN':
        dataset = TITAN_dataset(sub_set='default_test', norm_traj=1,
                                        obs_len=15, pred_len=1, overlap_ratio=1, 
                                        required_labels=[
                                                            'atomic_actions', 
                                                            'simple_context', 
                                                            'complex_context', 
                                                            'communicative', 
                                                            'transporting',
                                                            'age'
                                                            ], 
                                        multi_label_cross=1,  
                                        #   use_cross=1,
                                        use_atomic=1, use_complex=1, use_communicative=1, use_transporting=1, use_age=1,
                                        with_neighbors=0,
                                        tte=None,
                                        small_set=0,
                                        use_img=1,
                                        use_skeleton=1,
                                        use_ctx=0, ctx_mode='ori_local',
                                        use_traj=1,
                                        use_ego=0,
                                        augment_mode='random_crop_hflip',
                                        color_order='RGB'
                                        )

    else:
        dataset = PIEDataset(dataset_name='PIE', seq_type='crossing',
                                        obs_len=15, pred_len=1, obs_interval=1,
                                        do_balance=False, subset='train', bbox_size=(224, 224), 
                                        img_norm_mode='torch', color_order='BGR',
                                        resize_mode='even_padded', 
                                        use_img=1, 
                                        use_skeleton=0, skeleton_mode='heatmap',
                                        use_context=0, ctx_mode='ori_local', 
                                        use_traj=1, traj_mode='ltrb0-1',
                                        use_ego=1,
                                        small_set=0,
                                        overlap_retio=0.6,
                                        tte=None,
                                        recog_act=0,
                                        normalize_pos=0,)
    # root path for background
    if dataset_name == 'PIE':
        root_path = '/home/y_feng/workspace6/datasets/PIE_dataset'
        img_root_path = os.path.join(root_path, 'images')
    elif dataset_name == 'JAAD':
        root_path = '/home/y_feng/workspace6/datasets/JAAD'
        img_root_path = os.path.join(root_path, 'images')
    elif dataset_name == 'TITAN':
        root_path = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset'
        img_root_path = os.path.join(root_path, 'images_anonymized')

    # save path
    root = './hflip_check'
    if not os.path.exists(root):
        os.mkdir(root)
    if dataset_name == 'TITAN':
        cache_path = os.path.join(root, 'titan')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
    else:
        cache_path = os.path.join(root, 'pie')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
    
    num_flip = 0
    for _d in dataset:
        flip_flag = _d['hflip_flag']
        print('----')
        print(dataset.transforms['hflip'].flag)
        print(flip_flag)  # 只有false true, 没有 true false
        
        if dataset.transforms['hflip'].flag ^ flip_flag:
            num_flip += 1
            unnormed_traj = copy.deepcopy(_d['obs_bboxes_unnormed'].cpu().numpy())

            # get background
            if dataset_name in ('PIE', 'JAAD'):
                vid_id_int = _d['vid_id_int'].item()
                vid_nm = vid_id_int2str(vid_id_int)
                img_nm_int = _d['img_nm_int'].item()
                img_nm = img_nm_int2str(img_nm_int, dataset_name=dataset_name)
                if dataset_name == 'PIE':
                    set_id_int = _d['set_id_int'].item()
                    set_nm = 'set0' + str(set_id_int)
                    bg_path = os.path.join(img_root_path, set_nm, vid_nm, img_nm)
                else:
                    bg_path = os.path.join(img_root_path, vid_nm, img_nm)
            elif dataset_name == 'TITAN':
                vid_id_int = _d['clip_id_int'].item()
                img_nm_int = _d['img_nm_int'][-1].item()
                vid_nm = 'clip_' + str(vid_id_int)
                img_nm = img_nm_int2str(img_nm_int, dataset_name=dataset_name)
                bg_path = os.path.join(img_root_path, vid_nm, 'images', img_nm)
            background = cv2.imread(filename=bg_path)

            ori_background = copy.deepcopy(background)
            background = torch.tensor(background).permute(2, 0, 1)  # 3HW
            background = tvf.hflip(background).permute(1, 2, 0).numpy()

            ped_id = _d['ped_id_int'].item()
            img_id = img_nm.replace('.png', '')
            save_path = os.path.join(cache_path, f'{vid_nm}_{img_id}_{ped_id}.png')
            save_path_ori = os.path.join(cache_path, f'{vid_nm}_{img_id}_{ped_id}_ori.png')
            print(num_flip)
            print(save_path)
            # print(f'bb shape {bbs.shape}')
            # print(f'flip {flip_flag}')
            # print(f'img {imgs.shape}')
            
            # vis box on img
            img = draw_boxes_on_img(background, unnormed_traj)
            ori_img = draw_boxes_on_img(ori_background, unnormed_traj)
            
            cv2.imwrite(save_path, img)
            cv2.imwrite(save_path_ori, ori_img)
            print('----')


if __name__ == "__main__":
    main()