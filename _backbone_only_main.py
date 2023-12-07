import os
import pickle
import shutil
import time
from turtle import resizemode

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re
import copy
import numpy as np

from _datasets import PIEDataset
from helpers import makedir, draw_curves
from _proto_model import ImagePNet, SkeletonPNet, MultiPNet, MultiBackbone, NonlocalMultiPNet
from _backbones import create_backbone, record_conv3d_info, record_conv2d_info, record_sp_conv3d_info_w, record_t_conv3d_info, record_sp_conv2d_info_h, record_sp_conv2d_info_w, BackboneOnly
import _project_prototypes
import prune
import _multi_train_test as tnt
import save
from log import create_logger
from preprocess import preprocess_input_function
from receptive_field import compute_proto_layer_rf_info_v2
from utils import draw_proto_info_curves

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
    parser = argparse.ArgumentParser()

    # optimizer setting
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--project_batch_size', type=int, default=64)
    parser.add_argument('--warm_epochs', type=int, default=5)
    parser.add_argument('--push_start', type=int, default=10)
    parser.add_argument('--test_every', type=int, default=5)
    parser.add_argument('--linear_epochs', type=int, default=10)
    parser.add_argument('--save_proto_every_epoch', type=int, default=1)
    parser.add_argument('--backbone_lr', type=float, default=0.0001)
    parser.add_argument('--add_on_lr', type=float, default=0.003)
    parser.add_argument('--last_lr', type=float, default=0.0001)
    parser.add_argument('--joint_last', type=int, default=0)
    parser.add_argument('--warm_last', type=int, default=0)
    parser.add_argument('--lr_step_size', type=int, default=10)
    parser.add_argument('--lr_step_gamma', type=float, default=0.5)

    parser.add_argument('--clst_eff', type=float, default=0.8)
    parser.add_argument('--sep_eff', type=float, default=-0.08)
    parser.add_argument('--l1_eff', type=float, default=1e-4)
    parser.add_argument('--orth_type', type=int, default=0)
    parser.add_argument('--orth_eff', type=float, default=0.01)

    parser.add_argument('--gpuid', type=str, default='0') # python3 main.py -gpuid=0,1,2,3
    
    # data setting
    parser.add_argument('--small_set', type=float, default=0)
    parser.add_argument('--bbox_type', type=str, default='default')
    parser.add_argument('--ctx_shape_type', type=str, default='default')
    parser.add_argument('--obs_len', type=int, default=16)
    parser.add_argument('--num_classes', type=int, default=2)

    parser.add_argument('--dataset_name', type=str, default='PIE')
    parser.add_argument('--cross_dataset_name', type=str, default='JAAD')
    parser.add_argument('--cross_dataset', type=int, default=0)
    parser.add_argument('--balance_train', type=int, default=1)
    parser.add_argument('--balance_val', type=int, default=0)
    parser.add_argument('--balance_test', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--normalize_img_mode', type=str, default='torch')
    parser.add_argument('--resize_mode', type=str, default='even_padded')
    parser.add_argument('--max_occ', type=int, default=2)
    parser.add_argument('--test_max_occ', type=int, default=2)
    parser.add_argument('--data_split_type', type=str, default='default')
    parser.add_argument('--min_w', type=int, default=30)
    parser.add_argument('--min_h', type=int, default=40)
    parser.add_argument('--test_min_w', type=int, default=0)
    parser.add_argument('--test_min_h', type=int, default=0)
    parser.add_argument('--overlap', type=float, default=0.6)
    parser.add_argument('--dataloader_workers', type=int, default=8)

    # model setting
    parser.add_argument('--is_prototype_model', type=int, default=1)
    parser.add_argument('--last_pool', type=str, default='avg')
    parser.add_argument('--last_nonlinear', type=int, default=0)  # 0: linear  2: linear+bn+relu  3: bn+relu+ conv 1by1 +bn+relu + flatten + linear
    parser.add_argument('--simi_func', type=str, default='log')
    # traj model setting
    parser.add_argument('--use_traj', type=int, default=0)
    parser.add_argument('--traj_backbone_name', type=str, default='lstm')
    parser.add_argument('--traj_p_per_cls', type=int, default=10)
    parser.add_argument('--traj_prototype_dim', type=int, default=128)
    parser.add_argument('--traj_prototype_activation_function', type=str, default='dot_product')
    parser.add_argument('--traj_add_on_activation', type=str, default=None)
    # img model setting
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--img_backbone_name', type=str, default='C3D')
    parser.add_argument('--img_p_per_cls', type=int, default=10)
    parser.add_argument('--img_prototype_dim', type=int, default=512)
    parser.add_argument('--img_add_on_activation', type=str, default='sigmoid')
    # skeleton model setting
    parser.add_argument('--use_skeleton', type=int, default=1)
    parser.add_argument('--sk_backbone_name', type=str, default='SK')
    parser.add_argument('--sk_p_per_cls', type=int, default=10)
    parser.add_argument('--sk_prototype_dim', type=int, default=512)
    parser.add_argument('--sk_add_on_activation', type=str, default='sigmoid')
    # context model setting
    parser.add_argument('--use_context', type=int, default=1)
    parser.add_argument('--ctx_mode', type=str, default='mask_ped')
    parser.add_argument('--seg_class_set', type=int, default=1)
    parser.add_argument('--ctx_backbone_name', type=str, default='C3D')
    parser.add_argument('--ctx_p_per_cls', type=int, default=10)
    parser.add_argument('--ctx_prototype_dim', type=int, default=512)
    parser.add_argument('--ctx_add_on_activation', type=str, default='sigmoid')
    # single img model setting
    parser.add_argument('--use_single_img', type=int, default=0)
    parser.add_argument('--single_img_backbone_name', type=str, default='vgg16')
    parser.add_argument('--single_img_p_per_cls', type=int, default=10)
    parser.add_argument('--single_img_prototype_dim', type=int, default=512)
    parser.add_argument('--single_img_add_on_activation', type=str, default='sigmoid')

    # test only setting
    parser.add_argument('--test_only', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='../work_dirs/models/multi_img_skeleton_context/27Feb2022-22h56m17s')
    parser.add_argument('--config_path', type=str, default=None)

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    project_batch_size = args.project_batch_size
    warm_epochs = args.warm_epochs
    push_start = args.push_start
    push_epochs = [i for i in range(epochs) if i % push_start == 0]
    test_every = args.test_every
    linear_epochs = args.linear_epochs
    save_proto_every_epoch = args.save_proto_every_epoch
    backbone_lr = args.backbone_lr
    add_on_lr = args.add_on_lr
    last_lr = args.last_lr
    joint_last = args.joint_last
    warm_last = args.warm_last
    lr_step_size = args.lr_step_size
    lr_step_gamma = args.lr_step_gamma
    coefs = {
        'crs_ent': 1,
        'clst': args.clst_eff,
        'sep': args.sep_eff,
        'l1': args.l1_eff,
        'orth': args.orth_eff
    }
    orth_type = args.orth_type
    # data setting
    small_set = args.small_set
    ped_img_size = (224, 224)
    if args.bbox_type == 'max':
        ped_img_size = (375, 688)
    
    ctx_shape = (224, 224)
    if args.ctx_shape_type == 'keep_ratio':
        ctx_shape = (270, 480)
    obs_len = args.obs_len
    ped_vid_size = [obs_len, ped_img_size[0], ped_img_size[1]]
    ctx_vid_size = [obs_len, ctx_shape[0], ctx_shape[1]]
    num_classes = args.num_classes

    dataset_name = args.dataset_name
    cross_dataset_name = args.cross_dataset_name
    cross_dataset = args.cross_dataset
    balance_train = args.balance_train
    balance_val = args.balance_val
    balance_test = args.balance_test
    shuffle = args.shuffle
    normalize_img_mode = args.normalize_img_mode
    resize_mode = args.resize_mode
    max_occ = args.max_occ
    test_max_occ = args.test_max_occ
    min_wh = None
    test_min_wh = None
    if args.min_w > 0 and args.min_h > 0:
        min_wh = (args.min_w, args.min_h)
    if args.test_min_w > 0 and args.test_min_h > 0:
        test_min_wh = (args.test_min_w, args.test_min_h)
    overlap = args.overlap
    dataloader_workers = args.dataloader_workers

    is_prototype_model = args.is_prototype_model
    last_pool = args.last_pool
    last_nonlinear = args.last_nonlinear
    
    data_types = []
    use_traj = args.use_traj
    use_img = args.use_img
    use_skeleton = args.use_skeleton
    use_context = args.use_context

    img_backbone_name = args.img_backbone_name
    sk_backbone_name = args.sk_backbone_name
    ctx_backbone_name = args.ctx_backbone_name
    use_single_img = args.use_single_img
    if use_traj:
        data_types.append('traj')
    if use_img:
        data_types.append('img')
    if use_skeleton:
        data_types.append('skeleton')
    if use_context:
        data_types.append('context')
    if use_single_img:
        data_types.append('single_img')

        test_only = args.test_only
    model_path = args.model_path
    config_path = args.config_path
    config = {'warm_epochs': warm_epochs,
              'is_prototype_model': is_prototype_model,

              'ped_img_size': ped_img_size,
              'obs_len': obs_len,

              'balance_train': balance_train,
              'balance_test': balance_test,
              'shuffle': shuffle,
              'normalize_img_mode': normalize_img_mode,
              'resize_mode': resize_mode,
              'max_occ': max_occ
             }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    print('Default device: ', os.environ['CUDA_VISIBLE_DEVICES'])

    if test_only:
        model_name = model_path.split('/')[-1]
        work_dir = model_path.replace(model_name, '')
        model_dir = os.path.join(work_dir, 'test_only', model_name)
        makedir(model_dir)
        
        log, logclose = create_logger(log_filename=os.path.join(model_dir, 'test.log'))
        img_dir = os.path.join(model_dir, 'img')
        sk_dir = os.path.join(model_dir, 'skeleton')
        ctx_dir = os.path.join(model_dir, 'context')
        proto_value_info_dir = os.path.join(model_dir, 'proto_value_info')
        makedir(img_dir)
        makedir(sk_dir)
        makedir(ctx_dir)
        makedir(proto_value_info_dir)

        # load the data
        log('----------------------------Load data-----------------------------')
        # test set
        test_dataset = PIEDataset(dataset_name=dataset_name, obs_len=obs_len, do_balance=balance_test, subset='test', bbox_size=ped_img_size, 
                                    img_norm_mode=normalize_img_mode, resize_mode=resize_mode, max_occ=test_max_occ, min_wh=test_min_wh,
                                    use_img=use_img, use_skeleton=use_skeleton, use_context=use_context, ctx_mode=args.ctx_mode,
                                    small_set=small_set)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers=dataloader_workers , pin_memory=False)
        
        # construct model
        log('----------------------------Construct model-----------------------------')
        if is_prototype_model > 0:  # ----------TBD-----------
            return
        else:
            model = torch.load(model_path)
            model = model.to(device)
            ppnet_multi = torch.nn.DataParallel(model)
            test_res = tnt._train_or_test(model=ppnet_multi, dataloader=test_loader, optimizer=None, 
                                                class_specific=True, use_l1_mask=True,
                                                coefs=coefs, log=log, device=device, is_prototype_model=is_prototype_model, 
                                                data_types=data_types,
                                                orth_type=orth_type,
                                                vis_path=img_dir)
            logclose()
            return
