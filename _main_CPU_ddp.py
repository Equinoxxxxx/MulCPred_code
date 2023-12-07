import os
import pickle
import shutil
import time
from turtle import resizemode

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tools.distributed_parallel import ddp_setup
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import copy
import numpy as np

from _datasets import PIEDataset
from _TITAN_dataset import TITAN_dataset
from helpers import makedir, draw_curves
from _baselines import PCPA, BackBones, TEO
from _CPU import CPU
import save
from _train_test_CPU import contrast_epoch, train_test_epoch
from log import create_logger
from utils import draw_proto_info_curves, save_model, freeze, cls_weights, seed_all
from tools.plot import vis_weight_single_cls, draw_logits_histogram, draw_multi_task_curve, draw_curves2

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def main(rank):
    seed_all(42)
    parser = argparse.ArgumentParser()
    print(f'start {rank}')
    # device
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    # data
    parser.add_argument('--pre_dataset_name', type=str, default='PIE')
    parser.add_argument('--dataset_name', type=str, default='TITAN')
    parser.add_argument('--small_set', type=float, default=0)
    parser.add_argument('--obs_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=6)
    parser.add_argument('--obs_interval', type=int, default=0)
    parser.add_argument('--apply_tte', type=int, default=0)
    parser.add_argument('--test_apply_tte', type=int, default=0)
    parser.add_argument('--augment_mode', type=str, default='none')
    parser.add_argument('--img_norm_mode', type=str, default='torch')
    parser.add_argument('--color_order', type=str, default='BGR')
    parser.add_argument('--resize_mode', type=str, default='even_padded')
    parser.add_argument('--overlap', type=float, default=0.6)
    parser.add_argument('--test_overlap', type=float, default=0.6)
    parser.add_argument('--dataloader_workers', type=int, default=8)
    parser.add_argument('--shuffle', type=int, default=1)

    # train
    parser.add_argument('--p_epochs', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_every', type=int, default=2)
    parser.add_argument('--explain_every', type=int, default=10)
    parser.add_argument('--vis_every', type=int, default=2)
    parser.add_argument('--best_res_k', type=str, default='f1')
    parser.add_argument('--p_lr', type=float, default=0.1)
    parser.add_argument('--p_backbone_lr', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--backbone_lr', type=float, default=0.001)
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--lr_step_size', type=int, default=5)
    parser.add_argument('--lr_step_gamma', type=float, default=0.5)
    parser.add_argument('--t_max', type=float, default=10)
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # loss
    parser.add_argument('--loss_func', type=str, default='weighted_ce')
    parser.add_argument('--contrast_eff', type=float, default=0.0)
    parser.add_argument('--logsig_thresh', type=float, default=100)
    parser.add_argument('--logsig_loss_eff', type=float, default=0.1)
    parser.add_argument('--logsig_loss', type=str, default='kl')

    # model
    parser.add_argument('--model_name', type=str, default='CPU')
    parser.add_argument('--concept_mode', type=str, default='fix_proto')
    parser.add_argument('--contrast_mode', type=str, default='proto_bridge')
    parser.add_argument('--bridge_m', type=str, default='sk')
    parser.add_argument('--n_proto', type=int, default=20)
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--pool', type=str, default='max')
    parser.add_argument('--n_layer_proj', type=int, default=3)
    parser.add_argument('--norm', type=str, default='ln')
    parser.add_argument('--uncertainty', type=str, default='guassian')
    parser.add_argument('--n_sampling', type=int, default=3)

    # modality
    # img settingf
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--img_backbone_name', type=str, default='R3D50_clean')
    # sk setting
    parser.add_argument('--use_skeleton', type=int, default=1)
    parser.add_argument('--sk_mode', type=str, default='pseudo_heatmap')
    parser.add_argument('--sk_backbone_name', type=str, default='poseC3D_clean')
    # ctx setting
    parser.add_argument('--use_context', type=int, default=0)
    parser.add_argument('--ctx_mode', type=str, default='ori_local')
    parser.add_argument('--seg_cls', type=str, default='person,vehicles,roads,traffic_lights')
    parser.add_argument('--fuse_mode', type=str, default='transformer')
    parser.add_argument('--ctx_backbone_name', type=str, default='R3D50')
    # traj setting
    parser.add_argument('--use_traj', type=int, default=1)
    parser.add_argument('--traj_mode', type=str, default='ltrb')
    parser.add_argument('--traj_backbone_name', type=str, default='lstm')
    # ego setting
    parser.add_argument('--use_ego', type=int, default=1)
    parser.add_argument('--ego_backbone_name', type=str, default='lstm')

    args = parser.parse_args()
    # device
    local_rank = rank
    # data
    pre_dataset_name = args.pre_dataset_name
    dataset_name = args.dataset_name
    small_set = args.small_set
    obs_len = args.obs_len
    pred_len = args.pred_len
    obs_interval = args.obs_interval
    apply_tte = args.apply_tte
    test_apply_tte = args.test_apply_tte
    augment_mode = args.augment_mode
    img_norm_mode = args.img_norm_mode
    color_order = args.color_order
    resize_mode = args.resize_mode
    overlap = args.overlap
    test_overlap = args.test_overlap
    dataloader_workers = args.dataloader_workers
    shuffle = args.shuffle
    # train
    p_epochs = args.p_epochs
    epochs = args.epochs
    batch_size = args.batch_size
    test_every = args.test_every
    explain_every = args.explain_every
    vis_every = args.vis_every
    best_res_k = args.best_res_k
    lr = args.lr
    backbone_lr = args.backbone_lr
    p_lr = args.p_lr
    p_backbone_lr = args.p_backbone_lr
    scheduler = args.scheduler
    lr_step_size = args.lr_step_size
    lr_step_gamma = args.lr_step_gamma
    t_max = args.t_max
    optim = args.optim
    wd = args.weight_decay
    # loss
    loss_func = args.loss_func
    contrast_eff = args.contrast_eff
    logsig_thresh = args.logsig_thresh
    logsig_loss_eff = args.logsig_loss_eff
    logsig_loss_func = args.logsig_loss
    # model
    model_name = args.model_name
    concept_mode = args.concept_mode
    contrast_mode = args.contrast_mode
    bridge_m = args.bridge_m
    n_proto = args.n_proto
    proj_dim = args.proj_dim
    pool = args.pool
    n_layer_proj = args.n_layer_proj
    norm = args.norm
    uncertainty = args.uncertainty
    n_sampling = args.n_sampling
    
    use_img = args.use_img
    img_backbone_name = args.img_backbone_name
    use_skeleton = args.use_skeleton
    sk_mode = args.sk_mode
    sk_backbone_name = args.sk_backbone_name
    use_context = args.use_context
    ctx_mode = args.ctx_mode
    seg_cls = args.seg_cls
    fuse_mode = args.fuse_mode
    ctx_backbone_name = args.ctx_backbone_name
    use_traj = args.use_traj
    traj_mode = args.traj_mode
    traj_backbone_name = args.traj_backbone_name
    use_ego = args.use_ego
    ego_backbone_name = args.ego_backbone_name
    
    if uncertainty != 'gaussian':
        logsig_loss_eff = 0

    # modality settings
    modalities = []
    if use_img:
        modalities.append('img')
    if use_skeleton:
        modalities.append('sk')
    if use_context:
        modalities.append('ctx')
    if use_traj:
        modalities.append('traj')
    if use_ego:
        modalities.append('ego')

    img_setting = {
        'modality': 'img',
        'mode': resize_mode,
        'backbone_name': img_backbone_name,
        'pool': pool + '3d',
        'n_layer_proj': n_layer_proj,
        'norm': norm,
        'proj_dim': proj_dim,
        'uncertainty': uncertainty
    }

    sk_setting = {
        'modality': 'sk',
        'mode': sk_mode,
        'backbone_name': sk_backbone_name,
        'pool': pool + '3d',
        'n_layer_proj': n_layer_proj,
        'norm': norm,
        'proj_dim': proj_dim,
        'uncertainty': uncertainty
    }

    ctx_setting = {
        'modality': 'ctx',
        'mode': ctx_mode,
        'seg_cls': seg_cls.split(','),
        'fuse_mode': fuse_mode,
        'backbone_name': ctx_backbone_name,
        'pool': pool + '3d',
        'n_layer_proj': n_layer_proj,
        'norm': norm,
        'proj_dim': proj_dim,
        'uncertainty': uncertainty
    }

    traj_setting = {
        'modality': 'traj',
        'mode': traj_mode,
        'backbone_name': traj_backbone_name,
        'pool': 'none',
        'n_layer_proj': n_layer_proj,
        'norm': norm,
        'proj_dim': proj_dim,
        'uncertainty': uncertainty
    }

    ego_setting = {
        'modality': 'ego',
        'mode': 'none',
        'backbone_name': ego_backbone_name,
        'pool': 'none',
        'n_layer_proj': n_layer_proj,
        'norm': norm,
        'proj_dim': proj_dim,
        'uncertainty': uncertainty,
    }
    modality_to_setting = {
        'img': img_setting,
        'sk': sk_setting,
        'ctx': ctx_setting,
        'traj': traj_setting,
        'ego': ego_setting,
    }
    m_settings = {m: modality_to_setting[m] for m in modalities}
    # mutual setting
    for m in m_settings:
        m_settings[m]['n_sampling'] = n_sampling

    
    # create dirs
    exp_id = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    work_dir = '../work_dirs/contrast'
    makedir(work_dir)
    model_type = model_name
    for m in modalities:
        model_type += '_' + m
    model_dir = os.path.join(work_dir, model_type, exp_id)
    print('Save dir of current exp: ', model_dir)
    makedir(model_dir)
    ckpt_dir = os.path.join(model_dir, 'ckpt')
    makedir(ckpt_dir)
    plot_dir = os.path.join(model_dir, 'plot')
    makedir(plot_dir)
    pretrain_plot_dir = os.path.join(plot_dir, 'pretrain')
    makedir(pretrain_plot_dir)
    train_test_plot_dir = os.path.join(plot_dir, 'train_test')
    makedir(train_test_plot_dir)
    # logger
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    log('--------args----------')
    for k in list(vars(args).keys()):
        log(str(k)+': '+str(vars(args)[k]))
    log('--------args----------\n')
    args_dir = os.path.join(model_dir, 'args.pkl')
    with open(args_dir, 'wb') as f:
        pickle.dump(args, f)
    
    # device
    ddp_setup(local_rank, world_size=torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device=torch.device("cuda", local_rank)
    # load the data
    log('----------------------------Load data-----------------------------')
    pre_datasets = []
    train_datasets = []
    used_datasets = pre_dataset_name + dataset_name
    if 'TITAN' in used_datasets:
        titan_train = TITAN_dataset(sub_set='default_train', norm_traj=False,
                                      img_norm_mode=img_norm_mode, color_order=color_order,
                                      obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap, 
                                      recog_act=False,
                                      multi_label_cross=False, 
                                      use_atomic=False, use_complex=False, use_communicative=False, use_transporting=False, use_age=False,
                                      loss_weight='sklearn',
                                      tte=None,
                                      small_set=small_set,
                                      use_img=use_img, img_mode=resize_mode, 
                                      use_skeleton=use_skeleton, sk_mode=sk_mode,
                                      use_ctx=use_context, ctx_mode=ctx_mode,
                                      use_traj=use_traj, traj_mode=traj_mode,
                                      use_ego=use_ego,
                                      augment_mode=augment_mode
                                      )
    if 'PIE' in used_datasets:
        pie_train = PIEDataset(dataset_name='PIE', seq_type='crossing',
                                    obs_len=obs_len, pred_len=pred_len, do_balance=False, subset='train', bbox_size=(224, 224), 
                                    img_norm_mode=img_norm_mode, color_order=color_order,
                                    resize_mode=resize_mode,
                                    use_img=use_img,
                                    use_skeleton=use_skeleton, skeleton_mode=sk_mode,
                                    use_context=use_context, ctx_mode=ctx_mode,
                                    use_traj=use_traj, traj_mode=traj_mode,
                                    use_ego=use_ego,
                                    small_set=small_set,
                                    overlap_retio=overlap,
                                    tte=[0, 60],
                                    recog_act=False,
                                    normalize_pos=False,
                                    obs_interval=5,
                                    augment_mode=augment_mode)
    if 'JAAD' in used_datasets:
        jaad_train = PIEDataset(dataset_name='JAAD', seq_type='crossing',
                                    obs_len=obs_len, pred_len=pred_len, do_balance=False, subset='train', bbox_size=(224, 224), 
                                    img_norm_mode=img_norm_mode, color_order=color_order,
                                    resize_mode=resize_mode,
                                    use_img=use_img,
                                    use_skeleton=use_skeleton, skeleton_mode=sk_mode,
                                    use_context=use_context, ctx_mode=ctx_mode,
                                    use_traj=use_traj, traj_mode=traj_mode,
                                    use_ego=use_ego,
                                    small_set=small_set,
                                    overlap_retio=overlap,
                                    tte=[0, 60],
                                    recog_act=False,
                                    normalize_pos=False,
                                    obs_interval=5,
                                    augment_mode=augment_mode)
    titan_test = TITAN_dataset(sub_set='default_test', norm_traj=False,
                                      img_norm_mode=img_norm_mode, color_order=color_order,
                                      obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap, 
                                      recog_act=False,
                                      multi_label_cross=False, 
                                      use_atomic=False, use_complex=False, use_communicative=False, use_transporting=False, use_age=False,
                                      loss_weight='sklearn',
                                      tte=None,
                                      small_set=small_set,
                                      use_img=use_img, img_mode=resize_mode, 
                                      use_skeleton=use_skeleton, sk_mode=sk_mode,
                                      use_ctx=use_context, ctx_mode=ctx_mode,
                                      use_traj=use_traj, traj_mode=traj_mode,
                                      use_ego=use_ego,
                                      augment_mode=augment_mode
                                      )
    pie_test = PIEDataset(dataset_name='PIE', seq_type='crossing',
                                    obs_len=obs_len, pred_len=pred_len, do_balance=False, subset='test', bbox_size=(224, 224), 
                                    img_norm_mode=img_norm_mode, color_order=color_order,
                                    resize_mode=resize_mode,
                                    use_img=use_img,
                                    use_skeleton=use_skeleton, skeleton_mode=sk_mode,
                                    use_context=use_context, ctx_mode=ctx_mode,
                                    use_traj=use_traj, traj_mode=traj_mode,
                                    use_ego=use_ego,
                                    small_set=small_set,
                                    overlap_retio=overlap,
                                    tte=[0, 60],
                                    recog_act=False,
                                    normalize_pos=False,
                                    obs_interval=5,
                                    augment_mode=augment_mode)
    jaad_test = PIEDataset(dataset_name='JAAD', seq_type='crossing',
                                    obs_len=obs_len, pred_len=pred_len, do_balance=False, subset='test', bbox_size=(224, 224), 
                                    img_norm_mode=img_norm_mode, color_order=color_order,
                                    resize_mode=resize_mode,
                                    use_img=use_img,
                                    use_skeleton=use_skeleton, skeleton_mode=sk_mode,
                                    use_context=use_context, ctx_mode=ctx_mode,
                                    use_traj=use_traj, traj_mode=traj_mode,
                                    use_ego=use_ego,
                                    small_set=small_set,
                                    overlap_retio=overlap,
                                    tte=[0, 60],
                                    recog_act=False,
                                    normalize_pos=False,
                                    obs_interval=5,
                                    augment_mode=augment_mode)
    test_datasets = [titan_test, pie_test, jaad_test]
    if 'TITAN' in pre_dataset_name:
        pre_datasets.append(titan_train)
    if 'PIE' in pre_dataset_name:
        pre_datasets.append(pie_train)
    if 'JAAD' in pre_dataset_name:
        pre_datasets.append(jaad_train)
    if 'TITAN' in dataset_name:
        train_datasets.append(titan_train)
    if 'PIE' in dataset_name:
        train_datasets.append(pie_train)
    if 'JAAD' in dataset_name:
        train_datasets.append(jaad_train)
    pre_loaders = []

    for d in pre_datasets:
        sampler = DistributedSampler(d)
        pre_loaders.append(torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=False, 
                                                       num_workers=dataloader_workers, pin_memory=True, sampler=sampler))
    train_loaders = []
    for d in train_datasets:
        sampler = DistributedSampler(d)
        train_loaders.append(torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=False, 
                                                         num_workers=dataloader_workers, pin_memory=True, sampler=sampler))
    test_loaders = []
    for d in test_datasets:
        sampler = DistributedSampler(d)
        test_loaders.append(torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=False, 
                                                        num_workers=dataloader_workers, pin_memory=True, sampler=sampler))
    
    # construct the model
    log('----------------------------Construct model-----------------------------')
    if model_name == 'CPU':
        model = CPU(m_settings, 
                    modalities=modalities, 
                    concept_mode=concept_mode, 
                    n_proto=n_proto, 
                    proto_dim=proj_dim, 
                    contrast_mode=contrast_mode,
                    bridge_m=bridge_m,
                    )
    model = model.float().to(device)
    num_gpus = torch.cuda.device_count()
    model_parallel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],output_device=local_rank)
    log('----------------------------Construct optimizer-----------------------------')
    backbone_paras = []
    other_paras = []
    for n, p in model.named_parameters():
        if 'backbone' in n and ('ctx' in n or 'img' in n):
            backbone_paras += [p]
        elif sk_mode == 'pseudo_heatmap' and ('backbone' in n) and ('sk_model' in n):
            backbone_paras += [p]
        else:
            other_paras += [p]
    opt_specs = [{'params': backbone_paras, 'lr': p_backbone_lr},
                     {'params': other_paras, 'lr':p_lr}]
    if optim == 'sgd':
        optimizer = torch.optim.SGD(opt_specs, lr=p_lr, weight_decay=wd)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(opt_specs, lr=p_lr, weight_decay=wd, eps=1e-4)
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW(opt_specs, lr=p_lr, weight_decay=wd, eps=1e-4)
    else:
        raise NotImplementedError(optim)
    if scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_step_gamma)
    elif scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=t_max, eta_min=0)
    else:
        raise NotImplementedError(scheduler)

    # train the model
    log('----------------------------Start training-----------------------------')
    curve_dict_dataset = {
        'pre': {
            'contrast_loss':[],
            'logsig_loss':[]
        },
        'train':{
            'acc':[],
            'auc':[],
            'f1':[],
            'map':[],
            'contrast_loss':[],
            'logsig_loss':[]
        },
        'test':{
            'acc':[],
            'auc':[],
            'f1':[],
            'map':[],
        }
    }

    curve_dict = {
        'TITAN': curve_dict_dataset,
        'PIE': copy.deepcopy(curve_dict_dataset),
        'JAAD': copy.deepcopy(curve_dict_dataset),
    }

    # pretrain
    for e in range(1, p_epochs+1):
        log(f'Pretrain {e} epoch')
        model_parallel.train()
        for loader in pre_loaders:
            cur_dataset = loader.dataset.dataset_name
            cur_lr = optimizer.state_dict()['param_groups'][1]['lr']
            log(cur_dataset+f'  lr: {cur_lr}')
            pre_res = contrast_epoch(model_parallel,
                                     e,
                                     loader, 
                                    optimizer=optimizer,
                                    log=log, 
                                    device=device,
                                    modalities=modalities,
                                    logsig_thresh=logsig_thresh,
                                    logsig_loss_eff=logsig_loss_eff,
                                    logsig_loss_func=logsig_loss_func,
                                    exp_path=exp_id
                                    )
            for k in pre_res:
                curve_dict[loader.dataset.dataset_name]['pre'][k].append(pre_res[k])
        lr_scheduler.step()
        if e%vis_every == 0:
            for loader in pre_loaders:
                cur_dataset = loader.dataset.dataset_name
                draw_curves2(path=os.path.join(pretrain_plot_dir, cur_dataset+'contrast_and_logsig_loss.png'), 
                            val_lists=[curve_dict[cur_dataset]['pre']['contrast_loss'],curve_dict[cur_dataset]['pre']['logsig_loss']],
                            labels=['contrast loss', 'logsig loss'],
                            colors=['r', 'b'],
                            vis_every=vis_every)
    # change optimizer
    opt_specs = [{'params': backbone_paras, 'lr': backbone_lr},
                     {'params': other_paras, 'lr': lr}]
    if optim == 'sgd':
        optimizer = torch.optim.SGD(opt_specs, lr=p_lr, weight_decay=wd)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(opt_specs, lr=p_lr, weight_decay=wd, eps=1e-4)
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW(opt_specs, lr=p_lr, weight_decay=wd, eps=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_step_gamma)
    # fine tune
    best_res = {
        'acc': 0,
        'map': 0,
        'f1': 0,
        'auc': 0,
    }
    for e in range(1, epochs+1):
        log(f'Fine tune {e} epoch')
        model_parallel.train()
        for loader in train_loaders:
            cur_dataset = loader.dataset.dataset_name
            cur_lr = optimizer.state_dict()['param_groups'][1]['lr']
            log(cur_dataset+f'  cur lr: {cur_lr}')
            train_res = train_test_epoch(
                                        model_parallel, 
                                        e,
                                        loader,
                                        loss_func='weighted_ce',
                                        optimizer=optimizer,
                                        log=log, 
                                        device=device,
                                        modalities=modalities,
                                        contrast_eff=contrast_eff,
                                        logsig_thresh=logsig_thresh,
                                        logsig_loss_eff=logsig_loss_eff,
                                        train_or_test='train',
                                        logsig_loss_func=logsig_loss_func,
                                        )
            for k in curve_dict[cur_dataset]['train']:
                curve_dict[cur_dataset]['train'][k].append(train_res['final'][k])
        lr_scheduler.step()
        if e%test_every == 0:
            log(f'Test')
            model_parallel.eval()
            for loader in test_loaders:
                cur_dataset = loader.dataset.dataset_name
                log(cur_dataset)
                test_res = train_test_epoch(
                                            model_parallel, 
                                            e,
                                            loader,
                                            loss_func='weighted_ce',
                                            optimizer=optimizer,
                                            log=log, 
                                            device=device,
                                            modalities=modalities,
                                            contrast_eff=contrast_eff,
                                            logsig_thresh=logsig_thresh,
                                            logsig_loss_eff=logsig_loss_eff,
                                            train_or_test='test',
                                            logsig_loss_func=logsig_loss_func
                                            )
                
                for k in curve_dict[cur_dataset]['test']:
                    curve_dict[cur_dataset]['test'][k].append(test_res['final'][k])
                    val_lists = [curve_dict[cur_dataset]['test'][k]]
                    if cur_dataset in dataset_name:
                        val_lists = [curve_dict[cur_dataset]['train'][k],curve_dict[cur_dataset]['test'][k]]
                    draw_curves2(path=os.path.join(train_test_plot_dir, cur_dataset+k+'.png'), 
                                val_lists=val_lists,
                                labels=['train', 'test'],
                                colors=['r', 'b'],
                                vis_every=vis_every)
            if test_res['final'][best_res_k] > best_res[best_res_k]:
                for k in best_res:
                    best_res[k] = test_res['final'][k]
            if local_rank == 0:
                save_model(model=model, model_dir=ckpt_dir, 
                            model_name=str(e) + '_',
                            log=log)
            log(f'Best res: {best_res}')
    logclose()
    destroy_process_group()
    
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, nprocs=world_size)