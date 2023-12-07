from turtle import update
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time
import scipy
import argparse
import pickle

from tqdm import tqdm
from torchvision.transforms import functional as tvf

from _SLE_explain import SLE_explaine
from tools.datasets.PIE_JAAD import PIEDataset
from tools.datasets.TITAN import TITAN_dataset
from log import create_logger
from helpers import makedir
from utils import ped_id_int2str, seg_context_batch3d, visualize_featmap3d_simple, draw_traj_on_img, draw_boxes_on_img, ltrb2xywh, vid_id_int2str, img_nm_int2str, write_info_txt
from tools.data._img_mean_std import img_mean_std
from tools.plot import vis_weight_single_cls


def main():
    parser = argparse.ArgumentParser()
    print('start')
    # general setting
    parser.add_argument('--ckpt_path', type=str, 
                        default=\
                        '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models_1/SLE_traj_ego_img_skeleton_context/21Oct2023-00h09m20s/ckpt/24__0.pth'
                        )
    parser.add_argument('--args_path', type=str, 
                        default=\
                        '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models_1/SLE_traj_ego_img_skeleton_context/21Oct2023-00h09m20s/args.pkl'
                            )
    
    test_args = parser.parse_args()
    ckpt_path = test_args.ckpt_path
    args_path = test_args.args_path

    with open(args_path, 'rb') as f:
        args = pickle.load(f)
    
    # read args
    model_name = args.model_name
    q_modality = args.q_modality
    epochs = args.epochs
    batch_size = args.batch_size
    test_every = args.test_every
    explain_every = args.explain_every
    vis_every = args.vis_every
    lr = args.lr
    backbone_lr = args.backbone_lr
    lr_step_size = args.lr_step_size
    lr_step_gamma = args.lr_step_gamma
    loss_func = args.loss_func
    loss_weight = args.loss_weight
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
    use_cross = args.use_cross
    use_atomic = args.use_atomic
    use_complex = args.use_complex
    use_communicative = args.use_communicative
    use_transporting = args.use_transporting
    use_age = args.use_age
    multi_label_cross = args.multi_label_cross

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
    if model_name == 'PCPA':
        lr_step_gamma = 1
        sk_mode = 'coord'
        ctx_mode = 'ori_local'
        traj_mode = 'ltrb'
    elif model_name == 'backbone':
        lr_step_gamma = 1
        use_img = 1
        use_traj = 0
        use_ego = 0
        use_skeleton = 0
        use_context = 0

    if seg_mode == 0:
        seg_cls_idx = [24, 26, 19, 20]
    elif seg_mode == 1:
        seg_cls_idx = [24, 26, 19, 20, 8]
    else:
        raise NotImplementedError(seg_mode)
    
    if sk_mode == 'img+heatmap':
        sk_backbone_name = 'C3D'
    
    if dataset_name != 'TITAN':
        use_atomic = use_complex = use_communicative = use_transporting = 0
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

    modalities = []
    if use_traj:
        modalities.append('traj')
    if use_ego:
        modalities.append('ego')
    if use_img:
        modalities.append('img')
    if use_skeleton:
        modalities.append('skeleton')
    if use_context:
        modalities.append('context')
    if pred_traj:
        modalities.append('pred_traj')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    print('Default device: ', os.environ['CUDA_VISIBLE_DEVICES'])

    # dirs
    model_dir = os.path.join('/', *ckpt_path.split('/')[:-2])
    test_id = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    test_dir = os.path.join(model_dir, 'explain_spec_ckpt', test_id)
    makedir(test_dir)

    # logger
    log, logclose = create_logger(log_filename=os.path.join(test_dir, 
                                                            'test.log'))
    log('--------test_args----------')
    for k in list(vars(test_args).keys()):
        log(str(k)+': '+str(vars(test_args)[k]))
    log('--------test_args----------\n')
    log('--------args----------')
    for k in list(vars(args).keys()):
        log(str(k)+': '+str(vars(args)[k]))
    log('--------args----------\n')

    # load the data
    log('----------------------------Load data-----------------------------')
    # test set
    if dataset_name in ('PIE', 'JAAD'):
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
    elif dataset_name == 'TITAN':
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
                                      )
    explain_loader = torch.utils.data.DataLoader(explain_dataset, 
                                               batch_size=16, 
                                               shuffle=False,
                                                num_workers=dataloader_workers, 
                                                pin_memory=False)
    
    # construct the model
    log('----------------------------Construct model-----------------------------')
    model = torch.load(ckpt_path)
    model = model.to(device)
    model_parallel = torch.nn.DataParallel(model)
    model_parallel.eval()

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
                epoch_number=None,
                num_explain=num_explain,
                save_dir=test_dir,
                vis_feat_mode=vis_feat_mode,
                norm_traj=norm_pos,
                )
    # visualize the weights of the linear layer
    if fusion_mode == 1:
        weight_vis_dir = os.path.join(test_dir, 'last_weight')
        makedir(weight_vis_dir)
        if use_cross:
            with open(os.path.join(weight_vis_dir, 'weight.txt'), 'w') as f:
                f.write(str(model.last_layer.weight))
            for i in range(model.last_layer.weight.size(0)):
                vis_weight_single_cls(model.last_layer.weight[i], 
                                        path=os.path.join(weight_vis_dir, 
                                                          'cls_'+str(i)+'.png'))
        if use_atomic:
            with open(os.path.join(weight_vis_dir, 'weight.txt'), 'w') as f:
                f.write(str(model.atomic_layer.weight))
            for i in range(model.atomic_layer.weight.size(0)):
                vis_weight_single_cls(model.atomic_layer.weight[i], 
                                        path=os.path.join(weight_vis_dir, 
                                                        'atomic_cls_'+str(i)+'.png'))
    log(f'Results saved in {test_dir}')

if __name__ == '__main__':
    main()