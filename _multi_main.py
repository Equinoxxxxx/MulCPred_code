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
import _project_sk_prototypes
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
    parser.add_argument('--seq_type', type=str, default='crossing')
    parser.add_argument('--apply_tte', type=int, default=1)
    parser.add_argument('--test_apply_tte', type=int, default=1)


    parser.add_argument('--dataset_name', type=str, default='PIE')
    parser.add_argument('--cross_dataset_name', type=str, default='JAAD')
    parser.add_argument('--cross_dataset', type=int, default=0)
    parser.add_argument('--balance_train', type=int, default=0)
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
    parser.add_argument('--overlap', type=float, default=0.9)
    parser.add_argument('--dataloader_workers', type=int, default=8)

    # model setting
    parser.add_argument('--is_prototype_model', type=int, default=1)
    parser.add_argument('--last_pool', type=str, default='avg')
    parser.add_argument('--fusion', type=int, default=1)
    parser.add_argument('--last_nonlinear', type=int, default=0)  # 0: linear  2: linear+bn+relu  3: bn+relu+ conv 1by1 +bn+relu + flatten + linear
    parser.add_argument('--simi_func', type=str, default='log')
    parser.add_argument('--update_proto', type=int, default=1)
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
    parser.add_argument('--skeleton_mode', type=str, default='coord')
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
    parser.add_argument('--model_path', type=str, default='../work_dirs/models/multi_img/27Feb2022-20h48m16s/78nopush0.8241.pth')
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
    seq_type = args.seq_type
    apply_tte = args.apply_tte
    tte = None
    test_tte = None
    if apply_tte:
        tte = [30, 60]
    test_apply_tte = args.test_apply_tte
    if test_apply_tte:
        test_tte = [30, 60]

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
    fusion = args.fusion
    last_pool = args.last_pool
    last_nonlinear = args.last_nonlinear
    update_proto = args.update_proto
    
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
    if is_prototype_model == 1:
        img_model_setting = {
                    'backbone': None,
                    'vid_size': ped_vid_size,
                    'p_per_cls': args.img_p_per_cls,
                    'prototype_dim': args.img_prototype_dim,
                    'sp_proto_layer_rf_info': None,
                    't_proto_layer_rf_info': None,
                    'num_classes': args.num_classes,
                    'init_weights': True,
                    'prototype_activation_function': args.simi_func,
                    'add_on_activation': args.img_add_on_activation,
                    'last_nonlinear': last_nonlinear
                }
        sk_model_setting = {
                    'backbone': None,
                    'skeleton_mode': args.skeleton_mode,
                    'skeleton_seq_shape': (2, args.obs_len, 17),
                    'p_per_cls': args.sk_p_per_cls,
                    'prototype_dim': args.sk_prototype_dim,
                    'sp_proto_layer_rf_info': None,
                    'num_classes': args.num_classes,
                    'init_weights': True,
                    'prototype_activation_function': args.simi_func,
                    'add_on_activation': args.sk_add_on_activation,
                    'last_nonlinear': last_nonlinear
                }
        seg_class_idx = [24, 26, 19, 20]
        if args.seg_class_set == 2:  # TBD
            pass
        ctx_model_setting = {
                    'backbone': None,
                    'backbone_name': args.ctx_backbone_name,
                    'vid_size': ctx_vid_size,
                    'p_per_cls': args.ctx_p_per_cls,
                    'prototype_dim': args.ctx_prototype_dim,
                    'sp_proto_layer_rf_info': None,
                    't_proto_layer_rf_info': None,
                    'num_classes': args.num_classes,
                    'init_weights': True,
                    'prototype_activation_function': args.simi_func,
                    'add_on_activation': args.ctx_add_on_activation,
                    'ctx_mode': args.ctx_mode,
                    'seg_class_idx': seg_class_idx,
                    'last_nonlinear': last_nonlinear
                }
        single_img_model_setting = {
                    'backbone': None,
                    'p_per_cls': args.sk_p_per_cls,
                    'prototype_dim': args.sk_prototype_dim,
                    'sp_proto_layer_rf_info': None,
                    'num_classes': args.num_classes,
                    'init_weights': True,
                    'prototype_activation_function': 'log',
                    'add_on_activation': 'sigmoid',
                }
    elif is_prototype_model == 2:
        traj_model_setting = {
                'backbone': None,
                'p_per_cls': 10,
                'prototype_dim': 128,
                'num_classes': 2,
                'init_weights': True,
                'prototype_activation_function': 'dot_product',
                'add_on_activation': 'linear_bn',
                'name': 'multi'
        }
        img_model_setting = {
                'backbone': None,
                'p_per_cls': 10,
                'prototype_dim': 128,
                'num_classes': 2,
                'init_weights': True,
                'prototype_activation_function': 'dot_product',
                'add_on_activation': 'linear_bn',
                'name': 'multi'
        }
        ctx_model_setting = {
                'backbone': None,
                'p_per_cls': 10,
                'prototype_dim': 128,
                'num_classes': 2,
                'init_weights': True,
                'prototype_activation_function': 'dot_product',
                'add_on_activation': 'linear_bn',
                'name': 'multi',
                'ctx_mode': 'local'
        }

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
        vis_dir = os.path.join(model_dir, 'vis_backbone')
        makedir(vis_dir)
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
                                    skeleton_mode=args.skeleton_mode,
                                    small_set=small_set,
                                    tte=test_tte,
                                    seq_type=seq_type)
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

    exp_id = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    work_dir = '../work_dirs/models'
    model_type = 'multi'
    for d in data_types:
        model_type += '_' + d
    model_dir = os.path.join(work_dir, model_type, exp_id)
    print('Save dir of current exp: ', model_dir)
    makedir(model_dir)
    ckpt_dir = os.path.join(model_dir, 'ckpt')
    makedir(ckpt_dir)
    config_path = os.path.join(model_dir, 'config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    log('--------args----------')
    for k in list(vars(args).keys()):
        log(str(k)+': '+str(vars(args)[k]))
    log('--------args----------\n')

    vis_dir = os.path.join(model_dir, 'vis_backbone')
    makedir(vis_dir)
    traj_dir = os.path.join(model_dir, 'traj')
    img_dir = os.path.join(model_dir, 'img')
    sk_dir = os.path.join(model_dir, 'skeleton')
    ctx_dir = os.path.join(model_dir, 'context')
    single_img_dir = os.path.join(model_dir, 'single_img')
    proto_value_info_dir = os.path.join(model_dir, 'proto_value_info')
    makedir(traj_dir)
    makedir(img_dir)
    makedir(sk_dir)
    makedir(ctx_dir)
    makedir(single_img_dir)
    makedir(proto_value_info_dir)
    if args.ctx_mode == 'seg_multi':
        ctx_dirs = []
        for i in range(len(seg_class_idx)):
            d = os.path.join(ctx_dir, str(i)+'th_class')
            makedir(d)
            ctx_dirs.append(d)
    log('config' + str(config))

    # # load the data
    log('----------------------------Load data-----------------------------')
    # train set
    train_dataset = PIEDataset(dataset_name=dataset_name, obs_len=obs_len, do_balance=balance_train, subset='train', bbox_size=ped_img_size, 
                                img_norm_mode=normalize_img_mode, resize_mode=resize_mode, max_occ=max_occ, min_wh=min_wh,
                                use_img=use_img, 
                                use_skeleton=use_skeleton, skeleton_mode=args.skeleton_mode,
                                use_context=use_context, ctx_mode=args.ctx_mode,
                                use_single_img=use_single_img,
                                small_set=small_set,
                                overlap_retio=overlap,
                                tte=tte,
                                seq_type=seq_type)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                                num_workers=dataloader_workers , pin_memory=False)
    # push set
    # push_set = PIEDataset(dataset_name=dataset_name, obs_len=obs_len, do_balance=balance_train, subset='train', bbox_size=ped_img_size, 
    #                             normalize_img_mode=normalize_img_mode, resize_mode=resize_mode, max_occ=max_occ, min_wh=min_wh,
    #                             use_img=use_img, use_skeleton=use_skeleton, use_context=use_context, ctx_mode=args.ctx_mode,
    #                             use_single_img=use_single_img,
    #                             small_set=small_set,
    #                             overlap_retio=overlap)
    train_push_loader = torch.utils.data.DataLoader(train_dataset, batch_size=project_batch_size, shuffle=shuffle,
                                                num_workers=dataloader_workers , pin_memory=False)
    # test set
    test_dataset = PIEDataset(dataset_name=dataset_name, obs_len=obs_len, do_balance=balance_test, subset='test', bbox_size=ped_img_size, 
                                img_norm_mode=normalize_img_mode, resize_mode=resize_mode, max_occ=test_max_occ, min_wh=test_min_wh,
                                use_img=use_img, 
                                use_skeleton=use_skeleton, skeleton_mode=args.skeleton_mode,
                                use_context=use_context, ctx_mode=args.ctx_mode,
                                use_single_img=use_single_img,
                                small_set=small_set,
                                tte=test_tte,
                                seq_type=seq_type)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=dataloader_workers , pin_memory=False)
    # cross set
    if cross_dataset:
        JAAD_dataset = PIEDataset(dataset_name=cross_dataset_name, obs_len=obs_len, do_balance=balance_test, subset='test', bbox_size=ped_img_size, 
                                img_norm_mode=normalize_img_mode, resize_mode=resize_mode, max_occ=test_max_occ, min_wh=test_min_wh,
                                use_img=use_img, 
                                use_skeleton=use_skeleton, skeleton_mode=args.skeleton_mode,
                                use_context=use_context, ctx_mode=args.ctx_mode,
                                use_single_img=use_single_img,
                                small_set=small_set,
                                tte=test_tte,
                                seq_type=seq_type)
        cross_loader = torch.utils.data.DataLoader(JAAD_dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=8, pin_memory=False)
    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(batch_size))

    # construct the model
    log('----------------------------Construct model-----------------------------')
    if is_prototype_model == 1:
        if use_img:
            img_backbone = create_backbone(args.img_backbone_name)
            conv_info = record_conv3d_info(img_backbone)
            # log('conv info: ' + str(conv_info))
            spw_k_list, spw_s_list, spw_p_list = record_sp_conv2d_info_w(conv_info)
            sph_k_list, sph_s_list, sph_p_list = record_sp_conv2d_info_h(conv_info)
            t_k_list, t_s_list, t_p_list = record_t_conv3d_info(conv_info)
            sph_proto_rf_info = compute_proto_layer_rf_info_v2(input_size=ped_vid_size[-2],
                                                                layer_filter_sizes=sph_k_list,
                                                                layer_strides=sph_s_list,
                                                                layer_paddings=sph_p_list,
                                                                prototype_kernel_size=1)
            spw_proto_rf_info = compute_proto_layer_rf_info_v2(input_size=ped_vid_size[-1],
                                                                layer_filter_sizes=spw_k_list,
                                                                layer_strides=spw_s_list,
                                                                layer_paddings=spw_p_list,
                                                                prototype_kernel_size=1)
            t_proto_rf_info = compute_proto_layer_rf_info_v2(input_size=ped_vid_size[0],
                                                                layer_filter_sizes=t_k_list,
                                                                layer_strides=t_s_list,
                                                                layer_paddings=t_p_list,
                                                                prototype_kernel_size=1)
            log('spatial h receiptive field info [n, j, r, start]: ' + str(sph_proto_rf_info))    # [n, j, r, start]
            log('spatial w receiptive field info: ' + str(spw_proto_rf_info))
            log('temporal receiptive field info: ' + str(t_proto_rf_info))
            img_model_setting['backbone'] = img_backbone
            img_model_setting['sp_proto_layer_rf_info'] = (sph_proto_rf_info, spw_proto_rf_info)
            img_model_setting['t_proto_layer_rf_info'] = t_proto_rf_info
        if use_skeleton:
            sk_backbone = create_backbone(args.sk_backbone_name)
            if sk_model_setting['skeleton_mode'] == 'coord':
                conv_info = record_conv2d_info(sk_backbone)
                # log('conv info: ' + str(conv_info))
                spw_k_list, spw_s_list, spw_p_list = record_sp_conv2d_info_w(conv_info)
                sph_k_list, sph_s_list, sph_p_list = record_sp_conv2d_info_h(conv_info)
                sph_proto_rf_info = compute_proto_layer_rf_info_v2(input_size=obs_len,
                                                                    layer_filter_sizes=sph_k_list,
                                                                    layer_strides=sph_s_list,
                                                                    layer_paddings=sph_p_list,
                                                                    prototype_kernel_size=1)
                spw_proto_rf_info = compute_proto_layer_rf_info_v2(input_size=17,
                                                                    layer_filter_sizes=spw_k_list,
                                                                    layer_strides=spw_s_list,
                                                                    layer_paddings=spw_p_list,
                                                                    prototype_kernel_size=1)
                log('spatial h receiptive field info: ' + str(sph_proto_rf_info))
                log('spatial w receiptive field info: ' + str(spw_proto_rf_info))
                sk_model_setting['backbone'] = sk_backbone
                sk_model_setting['sp_proto_layer_rf_info'] = (sph_proto_rf_info, spw_proto_rf_info)
            elif sk_model_setting['skeleton_mode'] == 'heatmap':
                conv_info = record_conv3d_info(sk_backbone)
                # log('conv info: ' + str(conv_info))
                spw_k_list, spw_s_list, spw_p_list = record_sp_conv2d_info_w(conv_info)
                sph_k_list, sph_s_list, sph_p_list = record_sp_conv2d_info_h(conv_info)
                t_k_list, t_s_list, t_p_list = record_t_conv3d_info(conv_info)
                sph_proto_rf_info = compute_proto_layer_rf_info_v2(input_size=ped_vid_size[-2],
                                                                    layer_filter_sizes=sph_k_list,
                                                                    layer_strides=sph_s_list,
                                                                    layer_paddings=sph_p_list,
                                                                    prototype_kernel_size=1)
                spw_proto_rf_info = compute_proto_layer_rf_info_v2(input_size=ped_vid_size[-1],
                                                                    layer_filter_sizes=spw_k_list,
                                                                    layer_strides=spw_s_list,
                                                                    layer_paddings=spw_p_list,
                                                                    prototype_kernel_size=1)
                t_proto_rf_info = compute_proto_layer_rf_info_v2(input_size=ped_vid_size[0],
                                                                    layer_filter_sizes=t_k_list,
                                                                    layer_strides=t_s_list,
                                                                    layer_paddings=t_p_list,
                                                                    prototype_kernel_size=1)
                log('spatial h receiptive field info [n, j, r, start]: ' + str(sph_proto_rf_info))    # [n, j, r, start]
                log('spatial w receiptive field info: ' + str(spw_proto_rf_info))
                log('temporal receiptive field info: ' + str(t_proto_rf_info))
                sk_model_setting['backbone'] = sk_backbone
                sk_model_setting['sp_proto_layer_rf_info'] = (sph_proto_rf_info, spw_proto_rf_info)
                sk_model_setting['t_proto_layer_rf_info'] = t_proto_rf_info
        if use_context:
            ctx_backbone = create_backbone(args.ctx_backbone_name)
            conv_info = record_conv3d_info(ctx_backbone)
            # log('conv info: ' + str(conv_info))
            spw_k_list, spw_s_list, spw_p_list = record_sp_conv2d_info_w(conv_info)
            sph_k_list, sph_s_list, sph_p_list = record_sp_conv2d_info_h(conv_info)
            t_k_list, t_s_list, t_p_list = record_t_conv3d_info(conv_info)
            sph_proto_rf_info = compute_proto_layer_rf_info_v2(input_size=ctx_vid_size[-2],
                                                                layer_filter_sizes=sph_k_list,
                                                                layer_strides=sph_s_list,
                                                                layer_paddings=sph_p_list,
                                                                prototype_kernel_size=1)
            spw_proto_rf_info = compute_proto_layer_rf_info_v2(input_size=ctx_vid_size[-1],
                                                                layer_filter_sizes=spw_k_list,
                                                                layer_strides=spw_s_list,
                                                                layer_paddings=spw_p_list,
                                                                prototype_kernel_size=1)
            t_proto_rf_info = compute_proto_layer_rf_info_v2(input_size=ctx_vid_size[0],
                                                                layer_filter_sizes=t_k_list,
                                                                layer_strides=t_s_list,
                                                                layer_paddings=t_p_list,
                                                                prototype_kernel_size=1)
            log('ctx spatial h receiptive field info: ' + str(sph_proto_rf_info))
            log('ctx spatial w receiptive field info: ' + str(spw_proto_rf_info))
            log('ctx temporal receiptive field info: ' + str(t_proto_rf_info))
            ctx_model_setting['backbone'] = ctx_backbone
            ctx_model_setting['sp_proto_layer_rf_info'] = (sph_proto_rf_info, spw_proto_rf_info)
            ctx_model_setting['t_proto_layer_rf_info'] = t_proto_rf_info
        if use_single_img:
            single_img_backbone = create_backbone(args.single_img_backbone_name)
            conv_info = record_conv2d_info(single_img_backbone)
            # log('conv info: ' + str(conv_info))
            spw_k_list, spw_s_list, spw_p_list = record_sp_conv2d_info_w(conv_info)
            sph_k_list, sph_s_list, sph_p_list = record_sp_conv2d_info_h(conv_info)
            sph_proto_rf_info = compute_proto_layer_rf_info_v2(input_size=ped_img_size[1],
                                                                layer_filter_sizes=sph_k_list,
                                                                layer_strides=sph_s_list,
                                                                layer_paddings=sph_p_list,
                                                                prototype_kernel_size=1)
            spw_proto_rf_info = compute_proto_layer_rf_info_v2(input_size=ped_img_size[0],
                                                                layer_filter_sizes=spw_k_list,
                                                                layer_strides=spw_s_list,
                                                                layer_paddings=spw_p_list,
                                                                prototype_kernel_size=1)
            log('spatial h receiptive field info: ' + str(sph_proto_rf_info))
            log('spatial w receiptive field info: ' + str(spw_proto_rf_info))
            single_img_model_setting['backbone'] = single_img_backbone
            single_img_model_setting['sp_proto_layer_rf_info'] = (sph_proto_rf_info, spw_proto_rf_info)
        ppnet = MultiPNet(data_types=data_types, 
                            img_model_settings=img_model_setting, 
                            skeleton_model_settings=sk_model_setting,
                            context_model_settings=ctx_model_setting,
                            single_img_model_settings=single_img_model_setting)
        if ppnet.use_img:
            img_model_parallel = torch.nn.DataParallel(ppnet.img_model)
        if ppnet.use_skeleton:
            sk_model_parallel = torch.nn.DataParallel(ppnet.skeleton_model)
        if ppnet.use_ctx:
            if args.ctx_mode == 'seg_multi':
                ctx_model_parallel = []
                for i in range(len(seg_class_idx)):
                    ctx_model_parallel.append(torch.nn.DataParallel(ppnet.context_model[i]))
            else:
                ctx_model_parallel = torch.nn.DataParallel(ppnet.context_model)
        if ppnet.use_single_img:
            single_img_model_parallel = torch.nn.DataParallel(ppnet.single_img_model)
    elif is_prototype_model == 2:
        if use_traj:
            traj_backbone = create_backbone('lstm')
            traj_model_setting['backbone'] = traj_backbone
        if use_img:
            img_backbone = create_backbone('cnn_lstm')
            img_model_setting['backbone'] = img_backbone
        if use_context:
            context_backbone = create_backbone('cnn_lstm')
            ctx_model_setting['backbone'] = context_backbone
        ppnet = NonlocalMultiPNet(data_types=data_types,
                                  traj_model_settings=traj_model_setting,
                                  img_model_settings=img_model_setting,
                                  context_model_settings=ctx_model_setting)
        if use_traj:
            traj_model_parallel = torch.nn.DataParallel(ppnet.traj_model)
        if use_img:
            img_model_parallel = torch.nn.DataParallel(ppnet.img_model)
        if use_context:
            ctx_model_parallel = torch.nn.DataParallel(ppnet.context_model)
    else:
        ppnet = MultiBackbone(use_img=use_img, use_skeleton=use_skeleton, use_context=use_context,
                               img_backbone_name=img_backbone_name, sk_backbone_name=sk_backbone_name, ctx_backbone_name=ctx_backbone_name,
                               last_pool=last_pool, fusion=fusion)
    #if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.to(device)
    ppnet_multi = torch.nn.DataParallel(ppnet)

    class_specific = True
    # log('Model info')
    # log(str(ppnet))
    # define optimizer
    log('----------------------------Construct optimizer-----------------------------')
    from settings import joint_optimizer_lrs, joint_lr_step_size, warm_optimizer_lrs, last_layer_optimizer_lr
    if is_prototype_model == 1:
        joint_optimizer_specs = []
        warm_optimizer_specs = []
        last_layer_optimizer_specs = []
        if 'img' in data_types:
            joint_optimizer_specs += [{'params': ppnet.img_model.backbone.parameters(), 'lr': backbone_lr, 'weight_decay': 1e-3}, # bias are now also being regularized
                                    {'params': ppnet.img_model.add_on_layers.parameters(), 'lr': add_on_lr, 'weight_decay': 1e-3},
                                    {'params': ppnet.img_model.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
                                    ]
            warm_optimizer_specs += [{'params': ppnet.img_model.add_on_layers.parameters(), 'lr': add_on_lr, 'weight_decay': 1e-3},
                                    {'params': ppnet.img_model.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
                                    ]
            
            last_layer_optimizer_specs += [{'params': ppnet.img_model.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
        if 'skeleton' in data_types:
            joint_optimizer_specs += [{'params': ppnet.skeleton_model.backbone.parameters(), 'lr': backbone_lr, 'weight_decay': 1e-3}, # bias are now also being regularized
                                    {'params': ppnet.skeleton_model.add_on_layers.parameters(), 'lr': add_on_lr, 'weight_decay': 1e-3},
                                    {'params': ppnet.skeleton_model.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
                                    ]
            
            warm_optimizer_specs += [{'params': ppnet.skeleton_model.add_on_layers.parameters(), 'lr': add_on_lr, 'weight_decay': 1e-3},
                                    {'params': ppnet.skeleton_model.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
                                    ]
            last_layer_optimizer_specs += [{'params': ppnet.skeleton_model.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
        if 'context' in data_types:
            if train_dataset.ctx_mode == 'seg_multi':
                for i in range(len(seg_class_idx)):
                    joint_optimizer_specs += [{'params': ppnet.context_model[i].backbone.parameters(), 'lr': backbone_lr, 'weight_decay': 1e-3}, # bias are now also being regularized
                                            {'params': ppnet.context_model[i].add_on_layers.parameters(), 'lr': add_on_lr, 'weight_decay': 1e-3},
                                            {'params': ppnet.context_model[i].prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
                                            ]
                    
                    warm_optimizer_specs += [{'params': ppnet.context_model[i].add_on_layers.parameters(), 'lr': add_on_lr, 'weight_decay': 1e-3},
                                            {'params': ppnet.context_model[i].prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
                                            ]
                    
                    last_layer_optimizer_specs += [{'params': ppnet.context_model[i].last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
            else:
                joint_optimizer_specs += [{'params': ppnet.context_model.backbone.parameters(), 'lr': backbone_lr, 'weight_decay': 1e-3}, # bias are now also being regularized
                                            {'params': ppnet.context_model.add_on_layers.parameters(), 'lr': add_on_lr, 'weight_decay': 1e-3},
                                            {'params': ppnet.context_model.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
                                            ]
                    
                warm_optimizer_specs += [{'params': ppnet.context_model.add_on_layers.parameters(), 'lr': add_on_lr, 'weight_decay': 1e-3},
                                        {'params': ppnet.context_model.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
                                        ]
                
                last_layer_optimizer_specs += [{'params': ppnet.context_model.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
        if 'single_img' in data_types:
            joint_optimizer_specs += [{'params': ppnet.single_img_model.backbone.parameters(), 'lr': backbone_lr, 'weight_decay': 1e-3}, # bias are now also being regularized
                                    {'params': ppnet.single_img_model.add_on_layers.parameters(), 'lr': add_on_lr, 'weight_decay': 1e-3},
                                    {'params': ppnet.single_img_model.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
                                    ]
            
            warm_optimizer_specs += [{'params': ppnet.single_img_model.add_on_layers.parameters(), 'lr': add_on_lr, 'weight_decay': 1e-3},
                                    {'params': ppnet.single_img_model.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
                                    ]
            last_layer_optimizer_specs += [{'params': ppnet.single_img_model.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
        if joint_last:
            joint_optimizer_specs += last_layer_optimizer_specs
        if warm_last:
            warm_optimizer_specs += last_layer_optimizer_specs
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
        
        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=lr_step_size, gamma=lr_step_gamma)

        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
    if is_prototype_model == 2:
        joint_optimizer_specs = []
        warm_optimizer_specs = []
        last_layer_optimizer_specs = []
        if 'traj' in data_types:
            joint_optimizer_specs += [{'params': ppnet.traj_model.backbone.parameters(), 'lr': backbone_lr, 'weight_decay': 1e-3}, # bias are now also being regularized
                                    {'params': ppnet.traj_model.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
                                    ]
            if traj_model_setting['add_on_activation'] is not None:
                joint_optimizer_specs += [{'params': ppnet.traj_model.add_on_layers.parameters(), 'lr': add_on_lr, 'weight_decay': 1e-3}]
            
            last_layer_optimizer_specs += [{'params': ppnet.traj_model.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
        if 'img' in data_types:
            joint_optimizer_specs += [{'params': ppnet.img_model.backbone.parameters(), 'lr': backbone_lr, 'weight_decay': 1e-3}, # bias are now also being regularized
                                    {'params': ppnet.img_model.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
                                    ]
            if img_model_setting['add_on_activation'] is not None:
                joint_optimizer_specs += [{'params': ppnet.img_model.add_on_layers.parameters(), 'lr': add_on_lr, 'weight_decay': 1e-3}]
            
            last_layer_optimizer_specs += [{'params': ppnet.img_model.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
        if 'context' in data_types:
            joint_optimizer_specs += [{'params': ppnet.context_model.backbone.parameters(), 'lr': backbone_lr, 'weight_decay': 1e-3}, # bias are now also being regularized
                                    {'params': ppnet.context_model.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
                                    ]
            if ctx_model_setting['add_on_activation'] is not None:
                joint_optimizer_specs += [{'params': ppnet.context_model.add_on_layers.parameters(), 'lr': add_on_lr, 'weight_decay': 1e-3}]
            
            last_layer_optimizer_specs += [{'params': ppnet.context_model.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=lr_step_size, gamma=lr_step_gamma)

        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
    else:
        joint_optimizer_specs = [{'params': ppnet.parameters(), 'lr': backbone_lr, 'weight_decay': 1e-3}, # bias are now also being regularized
                                ]
        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=lr_step_size, gamma=lr_step_gamma)
    # weighting of different training losses
    # from settings import coefs

    # train the model
    log('----------------------------Start training-----------------------------')
    acc_curves_train = []
    acc_curves_test = []
    ce_curves_train = []
    ce_curves_test = []
    clst_curves_train = []
    clst_curves_test = []
    sep_curves_train = []
    sep_curves_test = []
    l1_curves_train = []
    l1_curves_test = []
    orth_curves_train = []
    orth_curves_test = []
    dist_curves_train = []
    dist_curves_test = []

    p_infos = []
    for epoch in range(1, epochs+1):
        display_p_corr = False
        if epoch % 5 == 0:
            display_p_corr = True
        log('epoch: \t{0}'.format(epoch))
        ppnet_multi.train()
        if is_prototype_model > 0:
            if epoch <= warm_epochs:
                # fix住backbone，只训练backbone以后的部分(默认5个epoch)
                tnt.warm_only_multi(model=ppnet_multi, log=log)
                train_res = tnt._train_or_test(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer, 
                                                class_specific=class_specific, use_l1_mask=True,
                                                coefs=coefs, log=log, device=device, is_prototype_model=is_prototype_model, 
                                                data_types=data_types,
                                                orth_type=orth_type,
                                                display_p_corr=display_p_corr)
                if last_nonlinear == 1:
                    if use_traj:
                        print('\tgrad to linear', torch.mean(ppnet_multi.module.traj_model.last_layer.weight.grad))
                    if use_img:
                        print('\tgrad to linear', torch.mean(ppnet_multi.module.img_model.last_layer.weight.grad))
                    if use_skeleton:
                        print('\tgrad to linear', torch.mean(ppnet_multi.module.skeleton_model.last_layer.weight.grad))
                    if use_context:
                        print('\tgrad to linear', torch.mean(ppnet_multi.module.context_model.last_layer.weight.grad))
                    if use_single_img:
                        print('\tgrad to linear', torch.mean(ppnet_multi.module.single_img_model.last_layer.weight.grad))
            else:
                # 训练整个模型
                tnt.joint_multi(model=ppnet_multi, log=log)  # 模型中的参数全部设置为可训练
            
                train_res = tnt._train_or_test(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer, 
                                                class_specific=class_specific, use_l1_mask=True,
                                                coefs=coefs, log=log, device=device, is_prototype_model=is_prototype_model, 
                                                data_types=data_types,
                                                orth_type=orth_type,
                                                display_p_corr=display_p_corr)
                print('\tcur lr: ', joint_optimizer.param_groups[0]['lr'])
                if is_prototype_model == 1:
                    if use_traj:
                        print('\tgrad to linear', torch.mean(ppnet_multi.module.traj_model.last_layer.weight.grad))
                    if use_img:
                        print('\tgrad to input', torch.mean(ppnet_multi.module.img_model.backbone.conv1.weight.grad))
                        # print('\tgrad to backbone', torch.mean(ppnet_multi.module.img_model.backbone.conv5b.weight.grad))
                        print('\tgrad to add on', torch.mean(list(ppnet_multi.module.img_model.add_on_layers.parameters())[-1].grad))
                    if use_skeleton:
                        print('\tgrad to input', torch.mean(ppnet_multi.module.skeleton_model.backbone.conv1.weight.grad))
                        print('\tgrad to backbone', torch.mean(ppnet_multi.module.skeleton_model.backbone.conv3.weight.grad))
                        print('\tgrad to add on', torch.mean(list(ppnet_multi.module.skeleton_model.add_on_layers.parameters())[-1].grad))
                    if use_context:
                        if train_dataset.ctx_mode == 'seg_multi':
                            print('\tgrad to input', torch.mean(ppnet_multi.module.context_model[0].backbone.conv1.weight.grad))
                            print('\tgrad to backbone', torch.mean(ppnet_multi.module.context_model[0].backbone.conv5b.weight.grad))
                            print('\tgrad to add on', torch.mean(list(ppnet_multi.module.context_model[0].add_on_layers.parameters())[-1].grad))
                        else:
                            print('\tgrad to input', torch.mean(ppnet_multi.module.context_model.backbone.conv1.weight.grad))
                            # print('\tgrad to backbone', torch.mean(ppnet_multi.module.context_model.backbone.conv5b.weight.grad))
                            print('\tgrad to add on', torch.mean(list(ppnet_multi.module.context_model.add_on_layers.parameters())[-1].grad))
                    if use_single_img:
                        print('\tgrad to add on', torch.mean(list(ppnet_multi.module.single_img_model.add_on_layers.parameters())[-1].grad))
                joint_lr_scheduler.step()
        else:
            train_res = tnt._train_or_test(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer, 
                                                class_specific=class_specific, use_l1_mask=True,
                                                coefs=coefs, log=log, device=device, is_prototype_model=is_prototype_model, 
                                                data_types=data_types,
                                                orth_type=orth_type)
            print('\tcur lr: ', joint_optimizer.param_groups[0]['lr'])
            if use_img:
                print('\tgrad to input', torch.mean(ppnet_multi.module.img_backbone.conv1.weight.grad))
                # print('\tgrad to backbone', torch.mean(ppnet_multi.module.img_model.backbone.conv5b.weight.grad))
            if use_skeleton:
                print('\tgrad to input', torch.mean(ppnet_multi.module.skeleton_backbone.conv1.weight.grad))
                # print('\tgrad to backbone', torch.mean(ppnet_multi.module.skeleton_model.backbone.conv3.weight.grad))
            if use_context:
                if train_dataset.ctx_mode == 'seg_multi':
                    print('\tgrad to input', torch.mean(ppnet_multi.module.context_model[0].backbone.conv1.weight.grad))      # TBD
                    # print('\tgrad to backbone', torch.mean(ppnet_multi.module.context_model[0].backbone.conv5b.weight.grad))
                else:
                    print('\tgrad to input', torch.mean(ppnet_multi.module.context_backbone.conv1.weight.grad))
                    # print('\tgrad to backbone', torch.mean(ppnet_multi.module.context_model.backbone.conv5b.weight.grad))
            if use_single_img:
                print('\tgrad to input', torch.mean(ppnet_multi.module.single_img_backbone.conv1.weight.grad))
                # print('\tgrad to backbone', torch.mean(ppnet_multi.module.skeleton_model.backbone.conv3.weight.grad))
            joint_lr_scheduler.step()
        if epoch%test_every == 0:
            log('Testing')
            ppnet_multi.eval()
            test_res = tnt._train_or_test(model=ppnet_multi, dataloader=test_loader, optimizer=None, 
                                                class_specific=class_specific, use_l1_mask=True,
                                                coefs=coefs, log=log, device=device, is_prototype_model=is_prototype_model, 
                                                data_types=data_types,
                                                orth_type=orth_type,
                                                vis_path=vis_dir)
            if cross_dataset:
                log('Cross testing')
                cross_res = tnt._train_or_test(model=ppnet_multi, dataloader=cross_loader, optimizer=None, 
                                                class_specific=class_specific, use_l1_mask=True,
                                                coefs=coefs, log=log, device=device, is_prototype_model=is_prototype_model, 
                                                data_types=data_types,
                                                orth_type=orth_type)
            acc_train, ce_train = train_res[:2]
            acc_test, ce_test = test_res[:2]
            acc_curves_train.append(acc_train)
            acc_curves_test.append(acc_test)
            ce_curves_train.append(ce_train)
            ce_curves_test.append(ce_test)
            draw_curves(path=os.path.join(model_dir, '_acc.png'), train_curve=acc_curves_train, test_curve=acc_curves_test, metric_type='acc', test_every=test_every)
            draw_curves(path=os.path.join(model_dir, '_ce.png'), train_curve=ce_curves_train, test_curve=ce_curves_test, test_every=test_every)
            if is_prototype_model > 0:
                acc_train, ce_train, clst_train, l1_train, orth_train, dist_train = train_res[:6]
                acc_test, ce_test, clst_test, l1_test, orth_test, dist_test = test_res[:6]
                clst_curves_train.append(clst_train)
                clst_curves_test.append(clst_test)
                l1_curves_train.append(l1_train)
                l1_curves_test.append(l1_test)
                orth_curves_train.append(orth_train)
                orth_curves_test.append(orth_test)
                draw_curves(path=os.path.join(model_dir, '_clst.png'), train_curve=clst_curves_train, test_curve=clst_curves_test, test_every=test_every)
                draw_curves(path=os.path.join(model_dir, '_l1.png'), train_curve=l1_curves_train, test_curve=l1_curves_test, test_every=test_every)
                draw_curves(path=os.path.join(model_dir, '_orth.png'), train_curve=orth_curves_train, test_curve=orth_curves_test, test_every=test_every)
                if class_specific:
                    sep_train, avg_sep_train = train_res[-2:]
                    sep_test, avg_sep_test= test_res[-2:]
                    sep_curves_train.append(sep_train)
                    sep_curves_test.append(sep_test)
                    draw_curves(path=os.path.join(model_dir, '_sep.png'), train_curve=sep_curves_train, test_curve=sep_curves_test, test_every=test_every)

            log('Epoch' + str(epoch) + 'done. Test results: ' + str(acc_test))
            save.save_model_w_condition(model=ppnet, model_dir=ckpt_dir, model_name=str(epoch) + 'nopush', accu=test_res[0],
                                        target_accu=0.30, log=log)
        else:
            save.save_model_w_condition(model=ppnet, model_dir=ckpt_dir, model_name=str(epoch) + 'nopush', accu=train_res[0],
                                        target_accu=0.30, log=log)
        
        if epoch >= push_start and (epoch)%push_start == 0 and is_prototype_model > 0:
            log('Project feat to ptototypes')
            if is_prototype_model == 1:
                if use_img:
                    log('push img')
                    _project_prototypes.push_prototypes(
                        train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                        prototype_network_parallel=img_model_parallel, # pytorch network with prototype_vectors
                        class_specific=class_specific,
                        preprocess_input_function=None, # normalize if needed ``````````
                        prototype_layer_stride=1,
                        prototype_info_dir=img_dir,
                        epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                        log=log,
                        save_every_epoch=save_proto_every_epoch,
                        update_proto=update_proto)
                if use_skeleton:
                    log('push skeleton')
                    if args.skeleton_mode == 'coord':
                        _project_sk_prototypes.push_sk_protos(
                            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                            prototype_network_parallel=sk_model_parallel, # pytorch network with prototype_vectors
                            class_specific=class_specific,
                            prototype_layer_stride=1,
                            prototype_info_dir=sk_dir,
                            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                            log=log,
                            save_every_epoch=save_proto_every_epoch,
                            update_proto=update_proto)
                    elif args.skeleton_mode == 'heatmap':
                        _project_sk_prototypes.push_sk_heatmap_prototypes(
                            dataloader=train_push_loader,
                            prototype_network_parallel=sk_model_parallel,
                            class_specific=class_specific,
                            prototype_layer_stride=1,
                            prototype_info_dir=sk_dir,
                            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                            log=log,
                            save_every_epoch=save_proto_every_epoch,
                            img_size=(384, 288),
                            update_proto=update_proto
                        )
                    else:
                        raise NotImplementedError(args.skeleton_mode)
                if use_context:
                    log('push ctx')
                    if args.ctx_mode == 'seg_multi':
                        for i in range(len(seg_class_idx)):
                            _project_prototypes.push_ctx_protos(
                                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                                prototype_network_parallel=ctx_model_parallel[i], # pytorch network with prototype_vectors
                                class_specific=class_specific,
                                preprocess_input_function=None, # normalize if needed ``````````
                                prototype_layer_stride=1,
                                prototype_info_dir=ctx_dirs[i],
                                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                                log=log,
                                save_every_epoch=save_proto_every_epoch,
                                seg_class_i=i,
                                update_proto=update_proto)
                    else:
                        _project_prototypes.push_ctx_protos(
                            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                            prototype_network_parallel=ctx_model_parallel, # pytorch network with prototype_vectors
                            class_specific=class_specific,
                            preprocess_input_function=None, # normalize if needed ``````````
                            prototype_layer_stride=1,
                            prototype_info_dir=ctx_dir,
                            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                            log=log,
                            save_every_epoch=save_proto_every_epoch,
                            update_proto=update_proto)
                if use_single_img:
                    log('push single img')
                    _project_prototypes.push_single_img_protos(
                        train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                        prototype_network_parallel=single_img_model_parallel, # pytorch network with prototype_vectors
                        class_specific=class_specific,
                        prototype_layer_stride=1,
                        prototype_info_dir=single_img_dir,
                        epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                        log=log,
                        save_every_epoch=save_proto_every_epoch,
                        update_proto=update_proto)
            if is_prototype_model == 2:
                if use_traj:
                    log('push traj')
                    _project_prototypes.push_nonlocal_prototypes(
                        train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                        prototype_network_parallel=traj_model_parallel, # pytorch network with prototype_vectors
                        class_specific=class_specific,
                        prototype_info_dir=traj_dir,
                        epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                        log=log,
                        save_every_epoch=save_proto_every_epoch,
                        data_type='traj',
                        update_proto=update_proto)
                if use_img:
                    log('push img')
                    _project_prototypes.push_nonlocal_prototypes(
                        train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                        prototype_network_parallel=img_model_parallel, # pytorch network with prototype_vectors
                        class_specific=class_specific,
                        prototype_info_dir=img_dir,
                        epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                        log=log,
                        save_every_epoch=save_proto_every_epoch,
                        data_type='img',
                        update_proto=update_proto)
                if use_context:
                    log('push context')
                    _project_prototypes.push_nonlocal_prototypes(
                        train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                        prototype_network_parallel=ctx_model_parallel, # pytorch network with prototype_vectors
                        class_specific=class_specific,
                        prototype_info_dir=ctx_dir,
                        epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                        log=log,
                        save_every_epoch=save_proto_every_epoch,
                        data_type='context',
                        update_proto=update_proto)
                pass
            log('Testing')
            ppnet_multi.eval()
            res_test = tnt._train_or_test(model=ppnet_multi, dataloader=test_loader, optimizer=None, 
                                                class_specific=class_specific, use_l1_mask=True,
                                                coefs=coefs, log=log, device=device, is_prototype_model=is_prototype_model, 
                                                data_types=data_types,
                                                orth_type=orth_type,
                                                display_p_corr=True)
            acc_test = res_test[0]
            save.save_model_w_condition(model=ppnet, model_dir=ckpt_dir, model_name=str(epoch) + 'push', accu=acc_test,
                                        target_accu=0.30, log=log)
            p_infos.append(res_test[-1])  # T NP 3
            p_info_array = np.stack(p_infos, axis=0)
            for i in range(p_info_array.shape[1]):
                path = os.path.join(proto_value_info_dir, str(i)+'.png')
                info = p_info_array[:, i]  # T 3
                draw_proto_info_curves(path, info, draw_every=push_start)
            log('Project done. Test results: acc' + str(acc_test))

            # last only
            tnt.last_only_multi(model=ppnet_multi, log=log)
            log('Train linear only')
            ppnet_multi.train()
            for i in range(1, linear_epochs+1):
                log('iteration: \t{0}'.format(i))
                res_train = tnt._train_or_test(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer, 
                                                class_specific=class_specific, use_l1_mask=True,
                                                coefs=coefs, log=log, device=device, is_prototype_model=is_prototype_model, 
                                                data_types=data_types,
                                                orth_type=orth_type)
                if i%test_every == 0:
                    log('Testing')
                    ppnet_multi.eval()
                    res_test = tnt._train_or_test(model=ppnet_multi, dataloader=test_loader, optimizer=None, 
                                                class_specific=class_specific, use_l1_mask=True,
                                                coefs=coefs, log=log, device=device, is_prototype_model=is_prototype_model, 
                                                data_types=data_types,
                                                orth_type=orth_type)
                save.save_model_w_condition(model=ppnet, model_dir=ckpt_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=res_train[0],
                                            target_accu=0.3, log=log)
            if last_nonlinear == 0:
                if ppnet.use_img:
                    log('\timg last layer param:' + str(list(ppnet.img_model.last_layer.parameters())))  # generator -> list
                if ppnet.use_skeleton:
                    log('\tskeleton last layer param:' + str(list(ppnet.skeleton_model.last_layer.parameters())))
                if ppnet.use_ctx:
                    if train_dataset.ctx_mode == 'seg_multi':
                        for m in ppnet.context_model:
                            log('\tcontext last layer param:' + str(list(m.last_layer.parameters())))
                    else: 
                        log('\tcontext last layer param:' + str(list(ppnet.context_model.last_layer.parameters())))
                if ppnet.use_single_img:
                    log('\tsingle img last layer param:' + str(list(ppnet.single_img_model.last_layer.parameters())))
            log('Linear layer training done.')
                
    logclose()

if __name__ == '__main__':
    main()