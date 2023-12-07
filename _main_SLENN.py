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
import torch.multiprocessing
import pytorch_warmup as warmup
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import re
import copy
import numpy as np

from tools.datasets.PIE_JAAD import PIEDataset
from tools.datasets.TITAN import TITAN_dataset
from tools.datasets.TITAN import LABEL2DICT, NUM_CLS_ATOMIC, \
    NUM_CLS_COMPLEX, NUM_CLS_COMMUNICATIVE, NUM_CLS_TRANSPORTING, NUM_CLS_AGE
from helpers import makedir, draw_curves
from _SLENN import MultiSLE
from _baselines import PCPA, BackBones, TEO
from _SENN import MultiSENN
from _train_test_SLE2 import train_test2
from _train_test_SENN import train_test_SENN
from _SLE_explain import SLE_explaine
from _SENN_explain import SENN_explaine
import save
from log import create_logger
from utils import draw_proto_info_curves, save_model, freeze, cls_weights, seed_all
from tools.plot import vis_weight_single_cls, draw_logits_histogram, \
    draw_multi_task_curve, draw_train_test_curve, draw_train_val_test_curve
from tools.gpu_mem_track import MemTracker

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False

def main():
    seed_all(42)
    parser = argparse.ArgumentParser()
    print('start')
    # general setting
    parser.add_argument('--model_name', type=str, default='SLE')
    parser.add_argument('--pool', type=str, default='avg')
    parser.add_argument('--q_modality', type=str, default='ego')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--warm_strategy', type=str, default='backbone_later')
    parser.add_argument('--warm_step_type', type=str, default='epoch')
    parser.add_argument('--warm_step', type=int, default=3)
    parser.add_argument('--test_every', type=int, default=10)
    parser.add_argument('--explain_every', type=int, default=10)
    parser.add_argument('--vis_every', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--backbone_lr', type=float, default=1e-7)
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--t_max', type=int, default=5)
    parser.add_argument('--lr_step_size', type=int, default=10)
    parser.add_argument('--lr_step_gamma', type=float, default=0.5)
    parser.add_argument('--loss_func', type=str, default='weighted_ce')
    parser.add_argument('--loss_weight', type=str, default='sklearn')
    parser.add_argument('--loss_weight_batch', type=int, default=0)
    parser.add_argument('--orth_type', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--one_logit', type=int, default=0)
    parser.add_argument('--key_metric', type=str, default='f1')

    parser.add_argument('--gpuid', type=str, default='0') # python3 main.py -gpuid=0,1,2,3

    # data setting
    parser.add_argument('--seq_type', type=str, default='trajectory')
    parser.add_argument('--small_set', type=float, default=0)
    parser.add_argument('--bbox_type', type=str, default='default')
    parser.add_argument('--ctx_shape_type', type=str, default='default')
    parser.add_argument('--obs_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--apply_tte', type=int, default=0)
    parser.add_argument('--test_apply_tte', type=int, default=0)
    parser.add_argument('--apply_sampler', type=int, default=0)
    parser.add_argument('--recog_act', type=int, default=0)
    parser.add_argument('--norm_pos', type=int, default=0)
    parser.add_argument('--obs_interval', type=int, default=0)
    parser.add_argument('--augment_mode', type=str, default='none')

    parser.add_argument('--dataset_name', type=str, default='JAAD')
    parser.add_argument('--cross_dataset_name', type=str, default='PIE')
    parser.add_argument('--cross_dataset', type=int, default=0)
    parser.add_argument('--balance_train', type=int, default=1)
    parser.add_argument('--balance_val', type=int, default=0)
    parser.add_argument('--balance_test', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('--img_norm_mode', type=str, default='torch')
    parser.add_argument('--color_order', type=str, default='BGR')
    parser.add_argument('--resize_mode', type=str, default='even_padded')
    parser.add_argument('--max_occ', type=int, default=2)
    parser.add_argument('--test_max_occ', type=int, default=2)
    parser.add_argument('--data_split_type', type=str, default='default')
    parser.add_argument('--min_h', type=int, default=0)
    parser.add_argument('--min_w', type=int, default=0)
    parser.add_argument('--test_min_w', type=int, default=0)
    parser.add_argument('--test_min_h', type=int, default=0)
    parser.add_argument('--overlap', type=float, default=0.6)
    parser.add_argument('--test_overlap', type=float, default=0.6)
    parser.add_argument('--dataloader_workers', type=int, default=8)
    parser.add_argument('--pop_occl_track', type=int, default=0)

    # model setting
    parser.add_argument('--fusion_mode', type=int, default=1)
    parser.add_argument('--separate_backbone', type=int, default=1)
    parser.add_argument('--conditioned_proto', type=int, default=1)
    parser.add_argument('--conditioned_relevance', type=int, default=1)
    parser.add_argument('--num_explain', type=int, default=5)
    parser.add_argument('--num_proto_per_modality', type=int, default=5)
    parser.add_argument('--proto_dim', type=int, default=256)
    parser.add_argument('--simi_func', type=str, default='dot')
    parser.add_argument('--pred_traj', type=int, default=0)
    parser.add_argument('--freeze_base', type=int, default=0)
    parser.add_argument('--freeze_proto', type=int, default=0)
    parser.add_argument('--freeze_relev', type=int, default=0)
    parser.add_argument('--softmax_t', type=str, default='1')
    parser.add_argument('--proto_activate', type=str, default='softmax')
    parser.add_argument('--use_atomic', type=int, default=0)
    parser.add_argument('--use_complex', type=int, default=0)
    parser.add_argument('--use_communicative', type=int, default=0)
    parser.add_argument('--use_transporting', type=int, default=0)
    parser.add_argument('--use_age', type=int, default=0)
    parser.add_argument('--use_cross', type=int, default=1)
    parser.add_argument('--multi_label_cross', type=int, default=0)
    parser.add_argument('--lambda1', type=float, default=0.01)
    parser.add_argument('--lambda2', type=float, default=1.)
    parser.add_argument('--lambda3', type=float, default=0.1)
    parser.add_argument('--lambda_contrast', type=float, default=0.1)
    parser.add_argument('--contrast_mode', type=str, default='barlow_twins')
    parser.add_argument('--backbone_add_on', type=int, default=1)
    parser.add_argument('--score_sum_linear', type=int, default=1)

    # img setting
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--img_backbone_name', type=str, default='C3D')
    # sk setting
    parser.add_argument('--use_skeleton', type=int, default=0)
    parser.add_argument('--sk_mode', type=str, default='heatmap')
    parser.add_argument('--sk_backbone_name', type=str, default='poseC3D_pretrained')
    # ctx setting
    parser.add_argument('--use_context', type=int, default=1)
    parser.add_argument('--ctx_mode', type=str, default='ori_local')
    parser.add_argument('--seg_mode', type=int, default=0)
    parser.add_argument('--ctx_backbone_name', type=str, default='C3D')
    # traj setting
    parser.add_argument('--use_traj', type=int, default=1)
    parser.add_argument('--traj_mode', type=str, default='ltrb')
    parser.add_argument('--traj_backbone_name', type=str, default='lstm')
    # ego setting
    parser.add_argument('--use_ego', type=int, default=0)
    parser.add_argument('--ego_backbone_name', type=str, default='lstm')

    # visualize setting
    parser.add_argument('--vis_feat_mode', type=str, default='mean')
    
    # test only setting
    parser.add_argument('--test_only', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='../work_dirs/models/multi_img/27Feb2022-20h48m16s/78nopush0.8241.pth')
    parser.add_argument('--config_path', type=str, default=None)

    # SENN setting
    parser.add_argument('--use_robust', type=int, default=1)

    args = parser.parse_args()
    
    # general setting
    model_name = args.model_name
    pool = args.pool
    q_modality = args.q_modality
    epochs = args.epochs
    batch_size = args.batch_size
    warm_strategy = args.warm_strategy
    warm_step_type = args.warm_step_type
    warm_step = args.warm_step
    test_every = args.test_every
    explain_every = args.explain_every
    vis_every = args.vis_every
    lr = args.lr
    backbone_lr = args.backbone_lr
    scheduler = args.scheduler
    t_max = args.t_max
    lr_step_size = args.lr_step_size
    lr_step_gamma = args.lr_step_gamma
    loss_func = args.loss_func
    loss_weight = args.loss_weight
    if loss_weight == 'trainable':
        trainable_weights = 1
    else:
        trainable_weights = 0
    loss_weight_batch = args.loss_weight_batch
    orth_type = args.orth_type
    weight_decay = args.weight_decay
    one_logit = args.one_logit

    # data setting
    seq_type = args.seq_type
    small_set = args.small_set
    ped_img_size = (224, 224)
    if args.bbox_type == 'max':
        ped_img_size = (375, 688)
    ctx_shape = (224, 224)
    if args.ctx_shape_type == 'keep_ratio':
        ctx_shape = (270, 480)
    obs_len = args.obs_len
    pred_len = args.pred_len
    num_classes = args.num_classes
    apply_tte = args.apply_tte
    test_apply_tte = args.test_apply_tte
    tte = None
    test_tte = None
    if apply_tte:
        tte = [0, 60]
    if test_apply_tte:
        test_tte = [0, 60]
    apply_sampler = args.apply_sampler
    recog_act = args.recog_act
    norm_pos = args.norm_pos
    obs_interval = args.obs_interval
    augment_mode = args.augment_mode

    ped_vid_size = [obs_len, ped_img_size[0], ped_img_size[1]]
    ctx_vid_size = [obs_len, ctx_shape[0], ctx_shape[1]]
    dataset_name = args.dataset_name
    cross_dataset_name = args.cross_dataset_name
    cross_dataset = args.cross_dataset
    balance_train = args.balance_train
    if loss_func != 'ce':
        balance_train = False
    balance_val = args.balance_val
    balance_test = args.balance_test
    shuffle = args.shuffle
    img_norm_mode = args.img_norm_mode
    color_order = args.color_order
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
    test_overlap = args.test_overlap
    dataloader_workers = args.dataloader_workers
    pop_occl_track = args.pop_occl_track

    # model setting
    fusion_mode = args.fusion_mode
    separate_backbone = args.separate_backbone
    conditioned_proto = args.conditioned_proto
    conditioned_relevance = args.conditioned_relevance
    num_explain = args.num_explain
    num_proto_per_modality = args.num_proto_per_modality
    proto_dim = args.proto_dim
    simi_func = args.simi_func
    pred_traj = args.pred_traj
    freeze_base = args.freeze_base
    freeze_proto = args.freeze_proto
    freeze_relev = args.freeze_relev
    temperature = 1
    softmax_t = args.softmax_t
    proto_activate = args.proto_activate
    use_atomic = args.use_atomic
    use_complex = args.use_complex
    use_communicative = args.use_communicative
    use_transporting = args.use_transporting
    use_age = args.use_age
    use_cross = args.use_cross
    multi_label_cross = args.multi_label_cross
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    lambda3 = args.lambda3
    lambda_contrast = args.lambda_contrast
    backbone_add_on = args.backbone_add_on
    score_sum_linear = args.score_sum_linear

    use_img = args.use_img
    img_backbone_name = args.img_backbone_name

    use_skeleton = args.use_skeleton
    sk_backbone_name = args.sk_backbone_name
    sk_mode = args.sk_mode

    use_context = args.use_context
    ctx_backbone_name = args.ctx_backbone_name
    ctx_mode = args.ctx_mode
    seg_mode = args.seg_mode

    use_traj = args.use_traj
    traj_mode = args.traj_mode
    traj_backbone_name = args.traj_backbone_name

    use_ego = args.use_ego
    ego_backbone_name = args.ego_backbone_name
    
    # calc input len
    if obs_interval == 0:
        input_len = obs_len
    else:
        input_len = obs_len // (obs_interval + 1)
    
    # vis setting
    vis_feat_mode = args.vis_feat_mode

    use_robust = args.use_robust

    # conditioned config
    if 'R3D' in img_backbone_name or 'csn' in img_backbone_name\
        or 'R3D' in ctx_backbone_name or 'csn' in ctx_backbone_name:
        img_norm_mode = 'kinetics'
    if img_norm_mode in ('kinetics', '0.5', 'activitynet'):
        color_order = 'RGB'
    else:
        color_order = 'BGR'
    if 'C3D' in ctx_backbone_name:
        proto_dim = 512
    elif 'R3D18' in ctx_backbone_name:
        proto_dim = 512
    elif 'R3D50' in ctx_backbone_name:
        proto_dim = 2048

    if softmax_t == 'transformer':
        temperature = proto_dim ** 0.5
    else:
        temperature = float(softmax_t)

    if not pred_traj:
        pred_len = 1

    if model_name != 'SLE':
        warm_strategy = ''
        lambda_contrast = 0
        lambda1 = 0
        lambda3 = 0
    if model_name == 'PCPA':
        use_img = 0
        # ctx_backbone_name = 'C3D'
        scheduler = 'step'
        # lr_step_gamma = 1
        sk_mode = 'coord'
        ctx_mode = 'ori_local'
        traj_mode = 'ltrb'
    elif model_name == 'backbone':
        # lr_step_gamma = 1
        use_img = 1
        use_traj = 0
        use_ego = 0
        use_skeleton = 0
        use_context = 0
    elif model_name == 'TEO':
        lr = 1e-4
        use_img = 0
        use_traj = 1
        use_ego = 0
        use_skeleton = 0
        use_context = 0
        norm_pos = 0
    elif model_name == 'SENN':
        norm_pos = 0

    if seg_mode == 0:
        seg_cls_idx = [24, 26, 19, 20]
    elif seg_mode == 1:
        seg_cls_idx = [24, 26, 19, 20, 8]
    else:
        raise NotImplementedError(seg_mode)
    
    if sk_mode == 'img+heatmap':
        sk_backbone_name = 'C3D'
    
    if dataset_name != 'TITAN':
        use_atomic = use_complex = use_communicative = use_transporting = use_age = 0
    required_labels=['simple_context']
    if use_atomic:
        required_labels.append('atomic_actions')
    if use_complex:
        required_labels.append('complex_context')
    if use_communicative:
        required_labels.append('communicative')
    if use_transporting:
        required_labels.append('transporting')
    if use_age:
        required_labels.append('age')
    
    if multi_label_cross and dataset_name == 'TITAN':
        num_classes = 13
    elif 'bce' in loss_func:
        num_classes = 1
    else:
        num_classes = 2

    pred_k = 'final'
    if dataset_name == 'TITAN' and use_atomic and not use_cross:
        num_classes = NUM_CLS_ATOMIC
        pred_k = 'atomic'

    if apply_sampler:
        loss_func = 'ce'

    data_types = []
    if use_traj:
        data_types.append('traj')
    if use_ego:
        data_types.append('ego')
    if use_img:
        data_types.append('img')
    if use_skeleton:
        data_types.append('skeleton')
    if use_context:
        data_types.append('context')
    if pred_traj:
        data_types.append('pred_traj')

    img_setting = {'backbone_name':img_backbone_name,
                'separate_backbone':separate_backbone,
                'conditioned_proto':conditioned_proto,
                'proto_generator_name':img_backbone_name,
                'num_explain':num_explain,
                'conditioned_relevance':conditioned_relevance,
                'relevance_generator_name':img_backbone_name,
                'num_proto':num_proto_per_modality,
                'proto_dim':proto_dim,
                'simi_func':simi_func,
                'freeze_base':freeze_base,
                'freeze_proto':freeze_proto,
                'freeze_relev':freeze_relev,
                'temperature': temperature,
                'proto_activate': proto_activate,
                'backbone_add_on': backbone_add_on,
                'score_sum_linear': score_sum_linear}
    
    sk_setting = {'backbone_name':sk_backbone_name,
                'sk_mode': sk_mode,
                'separate_backbone':separate_backbone,
                'conditioned_proto':conditioned_proto,
                'proto_generator_name':sk_backbone_name,
                'num_explain':num_explain,
                'conditioned_relevance':conditioned_relevance,
                'relevance_generator_name':sk_backbone_name,
                'num_proto':num_proto_per_modality,
                'proto_dim':proto_dim,
                'simi_func':simi_func,
                'freeze_base':0,
                'freeze_proto':0,
                'freeze_relev':0,
                'temperature': temperature,
                'proto_activate': proto_activate,
                'backbone_add_on': backbone_add_on,
                'score_sum_linear': score_sum_linear}
    
    ctx_setting = {'backbone_name':ctx_backbone_name,
                'ctx_mode': ctx_mode,
                'separate_backbone':separate_backbone,
                'conditioned_proto':conditioned_proto,
                'proto_generator_name':ctx_backbone_name,
                'num_explain':num_explain,
                'conditioned_relevance':conditioned_relevance,
                'relevance_generator_name':ctx_backbone_name,
                'num_proto':num_proto_per_modality,
                'proto_dim':proto_dim,
                'simi_func':simi_func,
                'freeze_base':freeze_base,
                'freeze_proto':freeze_proto,
                'freeze_relev':freeze_relev,
                'temperature': temperature,
                'seg_cls_idx': seg_cls_idx,
                'proto_activate': proto_activate,
                'backbone_add_on': backbone_add_on,
                'score_sum_linear': score_sum_linear}
    if 'seg_multi' in ctx_mode:
        ctx_setting['num_proto'] = 3

    traj_setting = {'backbone_name':traj_backbone_name,
                'separate_backbone':separate_backbone,
                'conditioned_proto':conditioned_proto,
                'proto_generator_name':traj_backbone_name,
                'num_explain':num_explain,
                'conditioned_relevance':conditioned_relevance,
                'relevance_generator_name':traj_backbone_name,
                'num_proto':num_proto_per_modality,
                'proto_dim':proto_dim,
                'in_dim': 4,
                'simi_func':simi_func,
                'temperature': temperature,
                'proto_activate': proto_activate,
                'backbone_add_on': backbone_add_on,
                'score_sum_linear': score_sum_linear}
    
    ego_setting = {'backbone_name':ego_backbone_name,
                'separate_backbone':separate_backbone,
                'conditioned_proto':conditioned_proto,
                'proto_generator_name':ego_backbone_name,
                'num_explain':num_explain,
                'conditioned_relevance':conditioned_relevance,
                'relevance_generator_name':ego_backbone_name,
                'num_proto':num_proto_per_modality,
                'proto_dim':proto_dim,
                'in_dim': 2,
                'simi_func':simi_func,
                'temperature': temperature,
                'proto_activate': proto_activate,
                'backbone_add_on': backbone_add_on,
                'score_sum_linear': score_sum_linear}
    # if dataset_name == 'TITAN':
    #     ego_setting['in_dim'] = 2
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    print('Default device: ', os.environ['CUDA_VISIBLE_DEVICES'])

    # create dirs
    exp_id = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    work_dir = '../work_dirs/models_1'
    makedir(work_dir)
    model_type = model_name
    for d in data_types:
        model_type += '_' + d
    model_dir = os.path.join(work_dir, model_type, exp_id)
    print('Save dir of current exp: ', model_dir)
    makedir(model_dir)
    multi_task_dir = os.path.join(model_dir, 'multi_task')
    makedir(multi_task_dir)
    ckpt_dir = os.path.join(model_dir, 'ckpt')
    makedir(ckpt_dir)
    
    last_weights_path = os.path.join(model_dir, 'last_weight.txt')
    last_weights_vis_path = os.path.join(model_dir, 'last_weithts')
    makedir(last_weights_vis_path)
    if use_atomic:
        atomic_weights_path = os.path.join(model_dir, 'atomic_weight.txt')
    if use_complex:
        complex_weights_path = os.path.join(model_dir, 'complex_weight.txt')
    if use_communicative:
        communicative_weights_path = os.path.join(model_dir, 'communicative_weight.txt')
    if use_transporting:
        transporting_weights_path = os.path.join(model_dir, 'transporting_weight.txt')
    
    logits_hist_dir = os.path.join(model_dir, 'logits_histogram')
    makedir(logits_hist_dir)
    # config_path = os.path.join(model_dir, 'config.pkl')
    # with open(config_path, 'wb') as f:
    #     pickle.dump(config, f)

    # logger
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    log('--------args----------')
    for k in list(vars(args).keys()):
        log(str(k)+': '+str(vars(args)[k]))
    log('--------args----------\n')
    args_dir = os.path.join(model_dir, 'args.pkl')
    with open(args_dir, 'wb') as f:
        pickle.dump(args, f)

    # vis dir
    proto_dir = os.path.join(model_dir, 'proto')
    makedir(proto_dir)
    # img_dir = os.path.join(model_dir, 'img')
    # sk_dir = os.path.join(model_dir, 'skeleton')
    # ctx_dir = os.path.join(model_dir, 'context')
    # traj_dir = os.path.join(model_dir, 'traj')
    # ego_dir = os.path.join(model_dir, 'ego')
    # makedir(traj_dir)
    # makedir(img_dir)
    # makedir(sk_dir)
    # makedir(ctx_dir)
    # makedir(ego_dir)

    # load the data
    log('----------------------------Load data-----------------------------')
    # train set
    if dataset_name in ('PIE', 'JAAD'):
        train_dataset = PIEDataset(dataset_name=dataset_name, seq_type=seq_type,
                                    obs_len=obs_len, pred_len=pred_len, do_balance=balance_train, subset='train', bbox_size=ped_img_size, 
                                    img_norm_mode=img_norm_mode, color_order=color_order,
                                    resize_mode=resize_mode, max_occ=max_occ, min_wh=min_wh,
                                    use_img=use_img,
                                    use_skeleton=use_skeleton, skeleton_mode=sk_mode,
                                    use_context=use_context, ctx_mode=ctx_mode, seg_cls_idx=seg_cls_idx,
                                    use_traj=use_traj, traj_mode=traj_mode,
                                    use_ego=use_ego,
                                    small_set=small_set,
                                    overlap_retio=overlap,
                                    tte=tte,
                                    recog_act=recog_act,
                                    normalize_pos=norm_pos,
                                    obs_interval=obs_interval,
                                    augment_mode=augment_mode,
                                    )
        # explain set
        explain_dataset = PIEDataset(dataset_name=dataset_name, seq_type=seq_type,
                                    obs_len=obs_len, pred_len=pred_len, do_balance=balance_train, subset='train', bbox_size=ped_img_size, 
                                    img_norm_mode=img_norm_mode, color_order=color_order,
                                    resize_mode=resize_mode, max_occ=max_occ, min_wh=min_wh,
                                    use_img=use_img,
                                    use_skeleton=use_skeleton, skeleton_mode=sk_mode,
                                    use_context=use_context, ctx_mode=ctx_mode, seg_cls_idx=seg_cls_idx,
                                    use_traj=use_traj, traj_mode=traj_mode,
                                    use_ego=use_ego,
                                    small_set=small_set,
                                    overlap_retio=overlap,
                                    tte=tte,
                                    recog_act=recog_act,
                                    normalize_pos=norm_pos,
                                    obs_interval=obs_interval,
                                    augment_mode='none',
                                    )
        # val set
        val_dataset = PIEDataset(dataset_name=dataset_name, seq_type=seq_type,
                                    obs_len=obs_len, pred_len=pred_len, 
                                    do_balance=balance_test, 
                                    subset='val', 
                                    bbox_size=ped_img_size, 
                                    img_norm_mode=img_norm_mode, color_order=color_order,
                                    resize_mode=resize_mode, 
                                    max_occ=test_max_occ, min_wh=test_min_wh,
                                    use_img=use_img, 
                                    use_skeleton=use_skeleton, skeleton_mode=sk_mode,
                                    use_context=use_context, ctx_mode=ctx_mode, 
                                    seg_cls_idx=seg_cls_idx,
                                    use_traj=use_traj, traj_mode=traj_mode,
                                    use_ego=use_ego,
                                    small_set=small_set,
                                    overlap_retio=overlap,
                                    tte=test_tte,
                                    recog_act=recog_act,
                                    normalize_pos=norm_pos,
                                    obs_interval=obs_interval)
        # test set
        test_dataset = PIEDataset(dataset_name=dataset_name, seq_type=seq_type,
                                    obs_len=obs_len, pred_len=pred_len, do_balance=balance_test, subset='test', bbox_size=ped_img_size, 
                                    img_norm_mode=img_norm_mode, color_order=color_order,
                                    resize_mode=resize_mode, max_occ=test_max_occ, min_wh=test_min_wh,
                                    use_img=use_img, 
                                    use_skeleton=use_skeleton, skeleton_mode=sk_mode,
                                    use_context=use_context, ctx_mode=ctx_mode, seg_cls_idx=seg_cls_idx,
                                    use_traj=use_traj, traj_mode=traj_mode,
                                    use_ego=use_ego,
                                    small_set=small_set,
                                    overlap_retio=overlap,
                                    tte=test_tte,
                                    recog_act=recog_act,
                                    normalize_pos=norm_pos,
                                    obs_interval=obs_interval)
    elif dataset_name == 'TITAN':
        train_dataset = TITAN_dataset(sub_set='default_train', norm_traj=norm_pos,
                                      img_norm_mode=img_norm_mode, color_order=color_order,
                                      obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap, 
                                      recog_act=recog_act,
                                      required_labels=required_labels, multi_label_cross=multi_label_cross, 
                                      use_atomic=use_atomic, use_complex=use_complex, use_communicative=use_communicative, use_transporting=use_transporting, use_age=use_age,
                                      loss_weight=loss_weight,
                                      tte=None,
                                      small_set=small_set,
                                      use_img=use_img, img_mode=resize_mode, 
                                      use_skeleton=use_skeleton, sk_mode=sk_mode,
                                      use_ctx=use_context, ctx_mode=ctx_mode,
                                      use_traj=use_traj, traj_mode=traj_mode,
                                      use_ego=use_ego,
                                      augment_mode=augment_mode,
                                      pop_occl_track=pop_occl_track,
                                      )
        explain_dataset = TITAN_dataset(sub_set='default_train', norm_traj=norm_pos,
                                      img_norm_mode=img_norm_mode, color_order=color_order,
                                      obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap, 
                                      recog_act=recog_act,
                                      required_labels=required_labels, multi_label_cross=multi_label_cross, 
                                      use_atomic=use_atomic, use_complex=use_complex, use_communicative=use_communicative, use_transporting=use_transporting, use_age=use_age,
                                      loss_weight=loss_weight,
                                      tte=None,
                                      small_set=small_set,
                                      use_img=use_img, img_mode=resize_mode, 
                                      use_skeleton=use_skeleton, sk_mode=sk_mode,
                                      use_ctx=use_context, ctx_mode=ctx_mode,
                                      use_traj=use_traj, traj_mode=traj_mode,
                                      use_ego=use_ego,
                                      augment_mode='none',
                                      pop_occl_track=pop_occl_track,
                                      )
        val_dataset = TITAN_dataset(sub_set='default_val', norm_traj=norm_pos,
                                      img_norm_mode=img_norm_mode, color_order=color_order,
                                      obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap, 
                                      recog_act=recog_act,
                                      required_labels=required_labels, multi_label_cross=multi_label_cross,  
                                      use_atomic=use_atomic, use_complex=use_complex, use_communicative=use_communicative, use_transporting=use_transporting, use_age=use_age,
                                      loss_weight=loss_weight,
                                      tte=None,
                                      small_set=small_set,
                                      use_img=use_img, img_mode=resize_mode, 
                                      use_skeleton=use_skeleton, sk_mode=sk_mode,
                                      use_ctx=use_context, ctx_mode=ctx_mode,
                                      use_traj=use_traj, traj_mode=traj_mode,
                                      use_ego=use_ego,
                                      pop_occl_track=pop_occl_track,
                                      )
        
        test_dataset = TITAN_dataset(sub_set='default_test', norm_traj=norm_pos,
                                      img_norm_mode=img_norm_mode, color_order=color_order,
                                      obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap, 
                                      recog_act=recog_act,
                                      required_labels=required_labels, multi_label_cross=multi_label_cross,  
                                      use_atomic=use_atomic, use_complex=use_complex, use_communicative=use_communicative, use_transporting=use_transporting, use_age=use_age,
                                      loss_weight=loss_weight,
                                      tte=None,
                                      small_set=small_set,
                                      use_img=use_img, img_mode=resize_mode, 
                                      use_skeleton=use_skeleton, sk_mode=sk_mode,
                                      use_ctx=use_context, ctx_mode=ctx_mode,
                                      use_traj=use_traj, traj_mode=traj_mode,
                                      use_ego=use_ego,
                                      pop_occl_track=pop_occl_track,
                                      )
    if apply_sampler:
        # n_nc = train_dataset.n_nc
        # n_c = train_dataset.n_c
        n = train_dataset.num_samples
        # _sampler_weight = [n / n_nc, n / n_c]
        _sampler_weight = cls_weights(train_dataset.num_samples_complex, loss_weight, 'cpu')
        # sampler_weight = np.array([_sampler_weight[int(gt[0])] for gt in train_dataset.samples['target']])
        sampler_weight = np.array([_sampler_weight[int(gt[-1])] for gt in train_dataset.samples['pred']['complex_context']])
        sampler_weight = torch.from_numpy(sampler_weight).double()
        sampler = torch.utils.data.WeightedRandomSampler(weights=sampler_weight, num_samples=n)
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=False,
                                                    num_workers=dataloader_workers, 
                                                    pin_memory=False,
                                                    sampler=sampler,
                                                    drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=1,
                                                    num_workers=dataloader_workers , 
                                                    pin_memory=False,
                                                drop_last=True)
    explain_loader = torch.utils.data.DataLoader(explain_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=False,
                                                num_workers=dataloader_workers, 
                                                pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=0,
                                                num_workers=dataloader_workers, 
                                                pin_memory=False,
                                                drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=0,
                                                num_workers=dataloader_workers, 
                                                pin_memory=False,
                                                drop_last=True)
    
    # construct the model
    log('----------------------------Construct model-----------------------------')
    if model_name == 'SLE':
        model = MultiSLE(num_classes=num_classes,
                        use_atomic=use_atomic, 
                        use_complex=use_complex, 
                        use_communicative=use_communicative, 
                        use_transporting=use_transporting, 
                        use_age=use_age,
                        use_img=use_img, img_setting=img_setting,
                        use_skeleton=use_skeleton, sk_setitng=sk_setting,
                        use_context=use_context, ctx_setting=ctx_setting,
                        use_traj=use_traj, traj_setting=traj_setting,
                        use_ego=use_ego, ego_setting=ego_setting,
                        pred_traj=pred_traj, pred_len=pred_len,
                        fusion_mode=fusion_mode,
                        trainable_weights=trainable_weights,
                        init_class_weights=train_dataset.class_weights)
    elif model_name == 'PCPA':
        model = PCPA(h_dim=proto_dim, q_modality=q_modality, num_classes=num_classes,
                    use_cross=use_cross,
                    use_atomic=use_atomic, use_complex=use_complex, use_communicative=use_communicative, use_transporting=use_transporting, use_age=use_age,
                    trainable_weights=trainable_weights,
                    init_class_weights=train_dataset.class_weights)
    elif model_name == 'backbone':
        model = BackBones(backbone_name=img_backbone_name, num_classes=num_classes,
                            use_cross=use_cross,
                            use_atomic=use_atomic, use_complex=use_complex, use_communicative=use_communicative, use_transporting=use_transporting, use_age=use_age,
                            pool=pool,
                            trainable_weights=trainable_weights,
                            init_class_weights=train_dataset.class_weights)
    elif model_name == 'TEO':
        model = TEO(num_layers= 4, d_model=128,
                              d_input=4, num_heads=8, 
                              dff=256, maximum_position_encoding= obs_len, device=device,
                              num_classes=num_classes,
                              use_cross=use_cross,
                              use_atomic=use_atomic, use_complex=use_complex, use_communicative=use_communicative, use_transporting=use_transporting, use_age=use_age,
                              trainable_weights=trainable_weights,
                              init_class_weights=train_dataset.class_weights)
    elif model_name == 'SENN':
        model = MultiSENN(use_traj=use_traj, 
                          use_ego=use_ego, 
                          use_img=use_img, 
                          use_sk=use_skeleton, 
                          use_ctx=use_context,
                          num_classes=num_classes,
                          pred_k=pred_k)
    model = model.to(device)
    model_parallel = torch.nn.DataParallel(model)
    
    # if use_skeleton:
    #     print('sk model params require grad')
    #     for n, p in model.sk_model.backbone.named_parameters():
    #         print(n, p.requires_grad)
    # define optimizer
    log('----------------------------Construct optimizer-----------------------------')
    if model_name in ('PCPA', 'backbone', 'TEO', 'SENN'):
        opt_specs = [{'params': model.parameters(), 'weight_decay': weight_decay},
                    #  {'params': model.last_fc.weight, 'weight_decay': 1e-3}
                     ]
        optimizer = torch.optim.Adam(opt_specs, lr=lr)
    elif backbone_lr != lr:
        backbone_paras = []
        other_paras = []
        for name, p in model.named_parameters():
            if 'backbone' in name and ('ctx' in name or 'img' in name):
                backbone_paras += [p]
            elif sk_mode == 'pseudo_heatmap' and \
                ('backbone' in name) and ('sk_model' in name):
                backbone_paras += [p]
            else:
                other_paras += [p]
        opt_specs = [{'params': backbone_paras, 
                      'lr': backbone_lr, 'weight_decay': weight_decay},
                     {'params': other_paras, 'weight_decay': weight_decay}]
        optimizer = torch.optim.Adam(opt_specs, lr=lr)
    else:
        opt_specs = [{'params': model.parameters(), 'weight_decay': weight_decay}]
        optimizer = torch.optim.Adam(opt_specs, lr=lr, betas=(0.9, 0.98), eps=1e-9)
    
    if scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                       step_size=lr_step_size, 
                                                       gamma=lr_step_gamma)
    elif scheduler == 'cosine':
        lr_scheduler = \
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                       T_max=t_max, 
                                                       eta_min=1e-7, 
                                                       verbose=True)
    else:
        raise ValueError(scheduler)
    # warm up scheduler
    if warm_step > 0 and warm_strategy == 'backbone_later':
        warm_specs = [{'params': backbone_paras, 
                            'lr': 0, 'weight_decay': weight_decay},
                            {'params': other_paras, 'weight_decay': weight_decay}]
        warm_optimizer = torch.optim.Adam(warm_specs, lr=lr)
        warm_scheduler = \
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=warm_optimizer, 
                                                       T_max=t_max, 
                                                       eta_min=1e-7, 
                                                       verbose=True)
        warm_warmer = warmup.LinearWarmup(warm_optimizer, 
                                           warmup_period=warm_step)
    scheduler_warmer = warmup.LinearWarmup(optimizer, 
                                           warmup_period=warm_step)

    # init the curves of results
    best_val_res = {'acc':0, 'auc':0, 'mAP':0, 'f1':0}
    best_test_res = {'acc':0, 'auc':0, 'mAP':0, 'f1':0}
    best_e = -1
    task = 'cross' if use_cross else 'atomic'
    key_metric = 'f1'
    res_curves = {
        'train':{},
        'val':{},
        'test':{},
    }
    for set_ in res_curves:
        res_curves[set_] = {
            'cross': {'acc': [],'f1': [],'f1b': [],'mAP': [], 'auc':[]},
            'atomic':{'acc': [],'f1': [],'f1b': [],'mAP': [], 'auc':[]},
            'complex': {'acc': [],'f1': [],'f1b': [],'mAP': [], 'auc':[]},
            'communicative': {'acc': [],'f1': [],'f1b': [],'mAP': [], 'auc':[]},
            'transporting': {'acc': [],'f1': [],'f1b': [],'mAP': [], 'auc':[]},
        }
    loss_curves = {
        'train': {},
        'val':{},
        'test': {}
    }
    for k in loss_curves:
        loss_curves[k] = {
            'balance': [],
            'orth': [],
            'contrast': []
        }

    # train the model
    log('----------------------------Start training-----------------------------')
    for e in range(1, epochs+1):
        log('epoch: \t{0}'.format(e))
        if apply_sampler:
            log('sampler weight: ' + str(sampler_weight))
        model_parallel.train()
        cur_optimizer = optimizer
        cur_scheduler = lr_scheduler
        cur_warmer = scheduler_warmer
        if model_name == 'SLE' and warm_strategy == 'backbone_later' \
            and e < warm_step:
            cur_optimizer = warm_optimizer
            cur_scheduler = warm_scheduler
            cur_warmer = warm_warmer
        if model_name == 'SENN':
            train_res=train_test_SENN(model=model_parallel, 
                                        dataloader=train_loader, 
                                        optimizer=cur_optimizer,
                                        log=log, 
                                        device=device,
                                        data_types=data_types,
                                        display_logits=True,
                                        num_classes=num_classes,
                                        multi_label_cross=multi_label_cross, 
                                        use_cross=use_cross,
                                        use_atomic=use_atomic, 
                                        use_complex=use_complex, 
                                        use_communicative=use_communicative, 
                                        use_transporting=use_transporting, 
                                        use_age=use_age,
                                        use_robust=use_robust,
                                        mask=None,
                                        pred_k=pred_k
                                        )
        else:
            train_res = train_test2(model=model_parallel,
                                dataloader=train_loader,
                                optimizer=cur_optimizer,
                                loss_func=loss_func,
                                loss_weight=loss_weight,
                                loss_weight_batch=loss_weight_batch,
                                log=log,
                                device=device,
                                data_types=data_types,
                                num_classes=num_classes,
                                ctx_mode=ctx_mode,
                                orth_type=orth_type,
                                use_cross=use_cross,
                                multi_label_cross=multi_label_cross, 
                                use_atomic=use_atomic, 
                                use_complex=use_complex, 
                                use_communicative=use_communicative, 
                                use_transporting=use_transporting, 
                                use_age=use_age,
                                model_name=model_name,
                                lambda1=lambda1,
                                lambda2=lambda2,
                                lambda3=lambda3,
                                lambda_contrast=lambda_contrast)
        log('\tcur lr: ')
        for dict in cur_optimizer.state_dict()['param_groups']:
            log(str(dict['lr']))
        with cur_warmer.dampening():
            cur_scheduler.step()
        log('Epoch' + str(e) + 'done.')
        # test
        if e%test_every == 0:
            model_parallel.eval()
            if model_name == 'SENN':
                log('\nValidating')
                val_res = train_test_SENN(model=model_parallel, 
                                            dataloader=val_loader, 
                                            optimizer=None,
                                            log=log, 
                                            device=device,
                                            data_types=data_types,
                                            display_logits=True,
                                            num_classes=num_classes,
                                            multi_label_cross=multi_label_cross, 
                                            use_cross=use_cross,
                                            use_atomic=use_atomic, 
                                            use_complex=use_complex, 
                                            use_communicative=use_communicative, 
                                            use_transporting=use_transporting, 
                                            use_age=use_age,
                                            use_robust=use_robust,
                                            mask=None,
                                            pred_k=pred_k
                                            )
                log('\nTesting')
                test_res = train_test_SENN(model=model_parallel, 
                                            dataloader=test_loader, 
                                            optimizer=None,
                                            log=log, 
                                            device=device,
                                            data_types=data_types,
                                            display_logits=True,
                                            num_classes=num_classes,
                                            multi_label_cross=multi_label_cross, 
                                            use_cross=use_cross,
                                            use_atomic=use_atomic, 
                                            use_complex=use_complex, 
                                            use_communicative=use_communicative, 
                                            use_transporting=use_transporting, 
                                            use_age=use_age,
                                            use_robust=use_robust,
                                            mask=None,
                                            pred_k=pred_k
                                            )
            else:
                log('\nValidating')
                val_res = train_test2(model=model_parallel,
                                dataloader=val_loader,
                                optimizer=None,
                                loss_func=loss_func,
                                loss_weight=loss_weight,
                                loss_weight_batch=loss_weight_batch,
                                log=log,
                                device=device,
                                data_types=data_types,
                                num_classes=num_classes,
                                ctx_mode=ctx_mode,
                                orth_type=orth_type,
                                use_cross=use_cross,
                                multi_label_cross=multi_label_cross, 
                                use_atomic=use_atomic, use_complex=use_complex, use_communicative=use_communicative, use_transporting=use_transporting, use_age=use_age,
                                model_name=model_name,
                                lambda1=lambda1,
                                lambda2=lambda2,
                                lambda3=lambda3,
                                lambda_contrast=lambda_contrast)
                log('\nTesting')
                test_res = train_test2(model=model_parallel,
                                dataloader=test_loader,
                                optimizer=None,
                                loss_func=loss_func,
                                loss_weight=loss_weight,
                                loss_weight_batch=loss_weight_batch,
                                log=log,
                                device=device,
                                data_types=data_types,
                                num_classes=num_classes,
                                ctx_mode=ctx_mode,
                                orth_type=orth_type,
                                use_cross=use_cross,
                                multi_label_cross=multi_label_cross, 
                                use_atomic=use_atomic, use_complex=use_complex, use_communicative=use_communicative, use_transporting=use_transporting, use_age=use_age,
                                model_name=model_name,
                                lambda1=lambda1,
                                lambda2=lambda2,
                                lambda3=lambda3,
                                lambda_contrast=lambda_contrast)
                for k in test_res['loss']:
                    loss_curves['train'][k].append(train_res['loss'][k])
                    loss_curves['val'][k].append(val_res['loss'][k])
                    loss_curves['test'][k].append(test_res['loss'][k])
                    path = os.path.join(model_dir, k+'_loss.png')
                    draw_train_val_test_curve(loss_curves['train'][k], 
                                              loss_curves['val'][k],
                                                loss_curves['test'][k],
                                                test_every=test_every,
                                                path=path)
            
            # draw results of crossing classification
            if use_cross:
                acc_train, mAP_train, auc_train, f1_train, logits_train = \
                    train_res['cross'][:5]
                acc_val, mAP_val, auc_val, f1_val, logits_val = \
                    val_res['cross'][:5]
                acc_test, mAP_test, auc_test, f1_test, logits_test = \
                    test_res['cross'][:5]

                res_curves['train']['cross']['acc'].append(acc_train)
                res_curves['train']['cross']['mAP'].append(mAP_train)
                res_curves['train']['cross']['auc'].append(auc_train)
                res_curves['train']['cross']['f1'].append(f1_train)

                res_curves['val']['cross']['acc'].append(acc_val)
                res_curves['val']['cross']['mAP'].append(mAP_val)
                res_curves['val']['cross']['auc'].append(auc_val)
                res_curves['val']['cross']['f1'].append(f1_val)

                res_curves['test']['cross']['acc'].append(acc_test)
                res_curves['test']['cross']['mAP'].append(mAP_test)
                res_curves['test']['cross']['auc'].append(auc_test)
                res_curves['test']['cross']['f1'].append(f1_test)

                for metric in res_curves['train']['cross']:
                    draw_train_val_test_curve(res_curves['train']['cross'][metric],
                                              res_curves['val']['cross'][metric],
                                              res_curves['test']['cross'][metric],
                                              test_every=test_every,
                                              path=os.path.join(model_dir, 
                                                                '_'+metric+'.png'))
                # draw logits histogram
                for i in range(logits_train.size(1)):
                    draw_logits_histogram(logits_train[:, i], 
                                          path=os.path.join(logits_hist_dir, 'cross_'+str(i)+'_train.png'))
                    draw_logits_histogram(logits_val[:, i], 
                                          path=os.path.join(logits_hist_dir, 'cross_'+str(i)+'_val.png'))
                    draw_logits_histogram(logits_test[:, i], 
                                          path=os.path.join(logits_hist_dir, 'cross_'+str(i)+'_test.png'))
            # draw results of other tasks
            for k in train_res:
                if k == 'cross' or k == 'loss' or k == 'pred_traj':
                    continue
                acc_train, mAP_train, auc_train, f1_train, logits_train = \
                    train_res[k][:5]
                acc_val, mAP_val, auc_val, f1_val, logits_val = \
                    val_res[k][:5]
                acc_test, mAP_test, auc_test, f1_test, logits_test = \
                    test_res[k][:5]

                res_curves['train'][k]['acc'].append(acc_train)
                res_curves['train'][k]['mAP'].append(mAP_train)
                res_curves['train'][k]['auc'].append(auc_train)
                res_curves['train'][k]['f1'].append(f1_train)

                res_curves['val'][k]['acc'].append(acc_val)
                res_curves['val'][k]['mAP'].append(mAP_val)
                res_curves['val'][k]['auc'].append(auc_val)
                res_curves['val'][k]['f1'].append(f1_val)

                res_curves['test'][k]['acc'].append(acc_test)
                res_curves['test'][k]['mAP'].append(mAP_test)
                res_curves['test'][k]['auc'].append(auc_test)
                res_curves['test'][k]['f1'].append(f1_test)

                for metric in res_curves['train'][k]:
                    draw_train_val_test_curve(res_curves['train'][k][metric],
                                              res_curves['val'][k][metric],
                                              res_curves['test'][k][metric],
                                              test_every=test_every,
                                              path=os.path.join(multi_task_dir, 
                                                                k+'_'+metric+'.png'))
                # draw logits histogram
                for i in range(logits_train.size(1)):
                    draw_logits_histogram(logits_train[:, i], 
                                          path=os.path.join(logits_hist_dir, 
                                                            '_'.join([k,str(i),'train.png'])))
                    draw_logits_histogram(logits_val[:, i], 
                                          path=os.path.join(logits_hist_dir, 
                                                            '_'.join([k,str(i),'val.png'])))
                    draw_logits_histogram(logits_test[:, i], 
                                          path=os.path.join(logits_hist_dir, 
                                                            '_'.join([k,str(i),'test.png'])))

            # save the model
            save_model(model=model, model_dir=ckpt_dir, 
                        model_name=str(e) + '_',
                        log=log)
            
            cur_res = res_curves['val'][task][key_metric][-1]
            if cur_res > best_val_res[key_metric]:
                best_e = e
                for k in best_val_res:
                    best_val_res[k] = res_curves['val'][task][k][-1]
                    best_test_res[k] = res_curves['test'][task][k][-1]
            log(f'Best val results: epoch {best_e}\n {best_val_res}')
            log(f'Test results: {best_test_res}')
        if e%explain_every == 0 and model_name == 'SLE':
            SLE_explaine(model=model_parallel,
                         dataloader=explain_loader,
                         device=device,
                         use_img=use_img,
                         use_context=use_context,
                         ctx_mode=ctx_mode,
                         seg_cls_idx=seg_cls_idx,
                         use_skeleton=use_skeleton,
                         use_traj=use_traj,
                         use_ego=use_ego,
                         log=log,
                         epoch_number=e,
                         num_explain=num_explain,
                         save_dir=proto_dir,
                         vis_feat_mode=vis_feat_mode,
                         norm_traj=norm_pos,
                         )
            if fusion_mode == 1:
                # log('last layer' + str(model.last_layer.weight))
                last_vis_path_e = os.path.join(last_weights_vis_path, 'epoch_'+str(e))
                makedir(last_vis_path_e)
                with open(last_weights_path, 'w') as f:
                    f.write(str(model.last_layer.weight))
                for i in range(model.last_layer.weight.size(0)):
                    vis_weight_single_cls(model.last_layer.weight[i], 
                                          path=os.path.join(last_vis_path_e, 
                                                            str(i)+'.png'))
                if use_atomic:
                    with open(atomic_weights_path, 'w') as f:
                        f.write(str(model.atomic_layer.weight))
                    for i in range(model.atomic_layer.weight.size(0)):
                        vis_weight_single_cls(model.atomic_layer.weight[i], 
                                              path=os.path.join(last_vis_path_e, 
                                                                'atomic' + str(i) +'.png'))
                if use_complex:
                    with open(complex_weights_path, 'w') as f:
                        f.write(str(model.complex_layer.weight))
                    for i in range(model.complex_layer.weight.size(0)):
                        vis_weight_single_cls(model.complex_layer.weight[i], 
                                              path=os.path.join(last_vis_path_e, 
                                                                'complex' + str(i) +'.png'))
                if use_communicative:
                    with open(communicative_weights_path, 'w') as f:
                        f.write(str(model.communicative_layer.weight))
                    for i in range(model.communicative_layer.weight.size(0)):
                        vis_weight_single_cls(model.communicative_layer.weight[i], 
                                              path=os.path.join(last_vis_path_e, 
                                                                'communicative' + str(i) +'.png'))
                if use_transporting:
                    with open(transporting_weights_path, 'w') as f:
                        f.write(str(model.transporting_layer.weight))
                    for i in range(model.transporting_layer.weight.size(0)):
                        vis_weight_single_cls(model.transporting_layer.weight[i], 
                                              path=os.path.join(last_vis_path_e, 
                                                                'transporting' + str(i) +'.png'))

            if simi_func in ('fix_proto1', 'fix_proto2'):
                if use_traj:
                    log('traj proto vec: ' + str(torch.squeeze(model.traj_model.proto_vec.detach())))
                    log('traj proto vec mean: ' + str(torch.mean(model.traj_model.proto_vec.detach())))
                if use_ego:
                    log('ego proto vec: ' + str(torch.squeeze(model.ego_model.proto_vec.detach())))
                    log('ego proto vec mean: ' + str(torch.mean(model.ego_model.proto_vec.detach())))
                if use_img:
                    log('img proto vec: ' + str(torch.squeeze(model.img_model.proto_vec.detach())))
                    log('img proto vec mean: ' + str(torch.mean(model.img_model.proto_vec.detach())))
                if use_skeleton:
                    log('sk proto vec: ' + str(torch.squeeze(model.sk_model.proto_vec.detach())))
                    log('sk proto vec mean: ' + str(torch.mean(model.sk_model.proto_vec.detach())))
                if use_context:
                    log('ctx proto vec: ' + str(torch.squeeze(model.ctx_model.proto_vec.detach())))
                    log('ctx proto vec mean: ' + str(torch.mean(model.ctx_model.proto_vec.detach())))
        elif e%explain_every == 0 and model_name == 'SENN':
            SENN_explaine(model=model_parallel,
                         dataloader=train_loader,
                         device=device,
                         use_img=use_img,
                         use_context=use_context,
                         ctx_mode=ctx_mode,
                         use_skeleton=use_skeleton,
                         use_traj=use_traj,
                         use_ego=use_ego,
                         log=log,
                         epoch_number=e,
                         num_explain=num_explain,
                         save_dir=proto_dir,
                         norm_traj=norm_pos,
                         )
    log(f'Best val results: epoch {best_e}\n {best_val_res}')
    log(f'Test results: {best_test_res}')
    logclose()


if __name__ == '__main__':
    main()