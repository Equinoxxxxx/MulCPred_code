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
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import re
import copy
import numpy as np
from tqdm import tqdm

from tools.datasets.PIE_JAAD import PIEDataset
from tools.datasets.TITAN import TITAN_dataset
from helpers import makedir, draw_curves
from _SLENN import MultiSLE
from models.baselines import PCPA, BackBones
import _multi_train_test as tnt
from _SLE_explain import SLE_explaine
import save
from log import create_logger
from utils import draw_proto_info_curves, save_model, freeze, draw_multi_task_curve
from tools.metrics import *
from tools.plot import draw_morf



def test():
    parser = argparse.ArgumentParser()
    print('start')
    # general setting
    parser.add_argument('--ckpt_path', 
                        type=str, 
                        default=\
                            '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models_1/SENN_traj_ego_img_skeleton/15Jan2023-15h54m07s/ckpt/34_0.0000.pth'
                            )
    parser.add_argument('--args_path', 
                        type=str, 
                        default=\
                            '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models_1/SENN_traj_ego_img_skeleton/15Jan2023-15h54m07s/args.pkl'
                            )

    # data
    parser.add_argument('--dataset_name', type=str, default='TITAN')

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
    small_set = 0.1
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
    # cross_dataset = args.cross_dataset
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
    
    if use_cross:
        num_classes = 2
    elif use_atomic and dataset_name == 'TITAN':
        num_classes = 6
    if multi_label_cross and dataset_name == 'TITAN':
        num_classes = 13

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
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    print('Default device: ', os.environ['CUDA_VISIBLE_DEVICES'])

    # dirs
    model_dir = os.path.join('/', *ckpt_path.split('/')[:-2])
    test_id = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    test_dir = os.path.join(model_dir, 'test', test_id)
    makedir(test_dir)

    # logger
    log, logclose = create_logger(log_filename=os.path.join(test_dir, 'test.log'))
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
        # test set
        test_dataset = PIEDataset(dataset_name=dataset_name, seq_type=seq_type,
                                    obs_len=obs_len, pred_len=pred_len, do_balance=balance_test, subset='test', bbox_size=ped_img_size, 
                                    img_norm_mode=img_norm_mode, resize_mode=resize_mode, max_occ=test_max_occ, min_wh=test_min_wh,
                                    use_img=use_img, 
                                    use_skeleton=use_skeleton, skeleton_mode=sk_mode,
                                    use_context=use_context, ctx_mode=ctx_mode, seg_cls_idx=seg_cls_idx,
                                    use_traj=use_traj, traj_mode=traj_mode,
                                    use_ego=use_ego,
                                    small_set=0,
                                    overlap_retio=0,
                                    tte=test_tte,
                                    recog_act=recog_act,
                                    normalize_pos=norm_pos,
                                    obs_interval=obs_interval)
    elif dataset_name == 'TITAN':
        test_dataset = TITAN_dataset(sub_set='default_test', norm_traj=norm_pos,
                                      obs_len=obs_len, pred_len=pred_len, overlap_ratio=0, 
                                      required_labels=required_labels, multi_label_cross=multi_label_cross,  
                                      use_atomic=use_atomic, use_complex=use_complex, use_communicative=use_communicative, use_transporting=use_transporting, use_age=use_age,
                                      tte=None,
                                      small_set=0,
                                      use_img=use_img, img_mode=resize_mode, 
                                      use_skeleton=use_skeleton, sk_mode=sk_mode,
                                      use_ctx=use_context, ctx_mode=ctx_mode,
                                      use_traj=use_traj,
                                      use_ego=use_ego,
                                      )
    test_loader1 = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False,
                                                num_workers=dataloader_workers , pin_memory=False)
    test_loader2 = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                num_workers=dataloader_workers , pin_memory=False)
    # construct the model
    log('----------------------------Construct model-----------------------------')
    model = torch.load(ckpt_path)
    model = model.to(device)
    model_parallel = torch.nn.DataParallel(model)

    # test
    model_parallel.eval()
    
    # get relevs
    if model_name in ('SENN', 'PCPA'):
        # SENN: B, nc, np  PCPA: B, m
        _, _relevs = test_epoch(model_parallel, test_loader1, model_name, device, num_classes=num_classes,
                                data_types=data_types,
                                # use_atomic=use_atomic,
                                # use_complex=use_complex,
                                # use_communicative=use_communicative,
                                # use_transporting=use_transporting,
                                # use_age=use_age,
                                use_cross=use_cross,
                                multi_label_cross=0,
                                mask=None,
                                log=log,
                                )
        if model_name == 'SENN':
            relevs = torch.zeros_like(_relevs)
            relevs[:, 0] = copy.deepcopy(_relevs[:, 0]) - copy.deepcopy(_relevs[:, 1])
            relevs[:, 1] = copy.deepcopy(_relevs[:, 1]) - copy.deepcopy(relevs[:, 0])
        else:
            relevs = _relevs
    elif model_name == 'SLE':
        # c, np
        if use_cross:
            _relevs = model.last_layer.weight[:, :model.total_num_proto].detach()
        elif use_atomic:
            _relevs = model.atomic_layer.weight[:, :model.total_num_proto].detach()
        relevs = torch.zeros_like(_relevs)
        relevs[0] = copy.deepcopy(_relevs[0]) - copy.deepcopy(_relevs[1])
        relevs[1] = copy.deepcopy(_relevs[1]) - copy.deepcopy(relevs[0])
        test_epoch(model_parallel, test_loader1, model_name, device, num_classes=num_classes,
                                data_types=data_types,
                                use_atomic=use_atomic,
                                # use_complex=use_complex,
                                # use_communicative=use_communicative,
                                # use_transporting=use_transporting,
                                # use_age=use_age,
                                use_cross=use_cross,
                                multi_label_cross=0,
                                mask=None,
                                log=log,
                                )
    else:
        test_epoch(model_parallel, test_loader1, model_name, device, num_classes=num_classes,
                                data_types=data_types,
                                # use_atomic=use_atomic,
                                # use_complex=use_complex,
                                # use_communicative=use_communicative,
                                # use_transporting=use_transporting,
                                # use_age=use_age,
                                use_cross=use_cross,
                                multi_label_cross=0,
                                mask=None,
                                log=log,
                                )
    
    # morf iters
    print(f'max relev: {torch.max(relevs)}, min relev: {torch.min(relevs)}')
    nf = relevs.size(-1)
    morf_logits = torch.zeros(size=(num_classes, test_dataset.num_samples, nf+1))  # c B nf
    print(morf_logits.size())
    if use_cross:
        class_iters = 2 if model_name in ('SENN', 'SLE') else 1
    elif use_atomic:
        class_iters = 6 if model_name in ('SENN', 'SLE') else 1

    # class iter
    for c in range(class_iters):
        # B(1), np
        if model_name == 'SENN':
            relev_c = relevs[:, c]  # B, np
        elif model_name == 'SLE':
            relev_c = relevs[c].unsqueeze(0)
        elif model_name == 'PCPA':
            relev_c = relevs

        # get masks for cur class
        morf_masks = get_batch_morf_masks(relev_c).to(device)  # B, nf+1, nf

        # nf iter
        for f in range(nf+1):
            print(f)
            for i, data in enumerate(tqdm(test_loader2, miniters=1)):
                # load inputs
                inputs = {}
                if 'img' in data_types:
                    inputs['img'] = data['ped_imgs'].to(device)
                if 'skeleton' in data_types:
                    inputs['skeleton'] = data['obs_skeletons'].to(device)
                if 'context' in data_types:
                    inputs['context'] = data['obs_context'].to(device)
                if 'traj' in data_types:
                    inputs['traj'] = data['obs_bboxes'].to(device)
                if 'ego' in data_types:
                    inputs['ego'] = data['obs_ego'].to(device)
                # load gt
                targets = {}
                targets['final'] = data['pred_intent'].to(device).view(-1) # idx, not one hot
                # forward
                with torch.no_grad():
                    mask = morf_masks[0, f] if model_name == 'SLE' else morf_masks[i, f]
                    output = model(inputs, mask=mask)
                    if use_cross:
                        logit = output[0]['final'].detach()  # 1, 2
                    elif use_atomic:
                        logit = output[0]['atomic'].detach()  # 1, 2
                    logit = F.softmax(logit, dim=-1)
                    if model_name in ('SENN', 'SLE'):
                        morf_logits[c, i, f] = logit[0, c]
                    else:
                        morf_logits[:, i, f] = logit[0]
            print(torch.mean(morf_logits, dim=1))            

    morf_logits = morf_logits.mean(1)  # c, nf
    morf_logits_path = os.path.join(test_dir, model_name+'.pkl')
    with open(morf_logits_path, 'wb') as f:
        pickle.dump(morf_logits, f)
    # normalize
    # morf_logits = F.softmax(morf_logits, dim=0)
    # morf_logits -= torch.min(morf_logits)
    # morf_logits /= torch.max(morf_logits)
    # print(morf_logits)
    # print(torch.mean(relevs, dim=0))
    # print(morf_masks)
    auc_morf = torch.zeros(num_classes)
    auc_morf_max_norm = torch.zeros(num_classes)
    auc_morf_max_min_norm = torch.zeros(num_classes)
    for c in range(morf_logits.size(0)):
        curve_path = os.path.join(test_dir, model_name+'_morf_curve_cls'+str(c)+'.png')
        draw_morf(morf_logits[c], curve_path)
        auc_morf[c] = calc_auc_morf(morf_logits[c])
        print(torch.max(morf_logits[c]), torch.min(morf_logits[c]))
        auc_morf_max_norm[c] = auc_morf[c] / torch.max(morf_logits[c])
        # auc_morf_norm[c] = (auc_morf[c] - torch.min(morf_logits[c])) / (torch.max(morf_logits[c]) - torch.min(morf_logits[c]))
    log(f'\t{model_name} auc-morf: {auc_morf} \tauc-morf max norm: {auc_morf_max_norm}')
    log(f'Res saved in {test_dir}')
    logclose()


def test_epoch(model, dataloader, model_name, device, num_classes=2,
            data_types=['img'],
            use_cross=1,
            use_atomic=0,
            multi_label_cross=0,
            mask=None,
            log=print,
            ):
    start = time.time()
    # targets and logits for whole epoch
    targets_e = {}
    logits_e = {}
    relevs_e = []

    # start iteration
    b_end = time.time()
    tbar = tqdm(dataloader, miniters=1)
    for iter, data in enumerate(tbar):
        # load inputs
        inputs = {}
        if 'img' in data_types:
            inputs['img'] = data['ped_imgs'].to(device)
        if 'skeleton' in data_types:
            inputs['skeleton'] = data['obs_skeletons'].to(device)
        if 'context' in data_types:
            inputs['context'] = data['obs_context'].to(device)
        if 'traj' in data_types:
            inputs['traj'] = data['obs_bboxes'].to(device)
        if 'ego' in data_types:
            inputs['ego'] = data['obs_ego'].to(device)

        # load gt
        targets = {}
        targets['final'] = data['pred_intent'].to(device).view(-1) # idx, not one hot
        if dataloader.dataset.dataset_name == 'TITAN':
            targets['atomic'] = data['atomic_actions'].to(device).view(-1)
            targets['complex'] = data['complex_context'].to(device).view(-1)
            targets['communicative'] = data['communicative'].to(device).view(-1)
            targets['transporting'] = data['transporting'].to(device).view(-1)
            targets['age'] = data['age'].to(device).view(-1)

        # forward
        b_start = time.time()
        # torch.enable_grad() has no effect outside of no_grad()
        with torch.no_grad():
            mse_loss = 0
            if mask is not None:
                output = model(inputs, mask=mask)  # b, num classes
            else:
                output = model(inputs)  # b, num classes
            
            if model_name == 'SLE':
                logits, _, _ = output
            elif model_name == 'SENN':
                logits, multi_protos, _relevs, recons = output  # _relevs: dict{b,nc,np}
                relevs = []
                for k in _relevs:
                    relevs.append(_relevs[k].detach())
                relevs = torch.cat(relevs, dim=2)
                relevs_e.append(relevs)
            elif model_name == 'PCPA':
                logits, m_scores = output  # m_scores: b,m
                relevs_e.append(m_scores.detach())
            else:
                logits = output
            
            # collect targets and logits in batch
            for k in logits:
                if iter == 0:
                    targets_e[k] = targets[k].detach()
                    logits_e[k] = logits[k].detach()
                else:
                    targets_e[k] = torch.cat((targets_e[k], targets[k].detach()), dim=0)
                    logits_e[k] = torch.cat((logits_e[k], logits[k].detach()), dim=0)

            # display
            data_prepare_time = b_start - b_end
            b_end = time.time()
            computing_time = b_end - b_start
            display_dict = {'data': data_prepare_time, 
                            'compute': computing_time
                            }
            if use_cross:
                mean_logit = torch.mean(logits['final'].detach(), dim=0)
                if num_classes == 2:
                    display_dict['logit'] = [round(logits['final'][0, 0].item(), 4), round(logits['final'][0, 1].item(), 4)]
                    display_dict['avg logit'] = [round(mean_logit[0].item(), 4), round(mean_logit[1].item(), 4)]
                elif num_classes == 1:
                    display_dict['logit'] = [round(logits['final'][0, 0].item(), 4)]
                    display_dict['avg logit'] = [round(mean_logit[0].item(), 4)]
            tbar.set_postfix(display_dict)
        del inputs
        torch.cuda.empty_cache()
    tbar.close()
    end = time.time()

    # logits_softmax = F.softmax(logits_e['final'], dim=-1)

    # calc metric
    acc_e = {}
    f1_e = {}
    mAP_e = {}
    prec_e = {}
    rec_e = {}
    for k in logits_e:
        acc_e[k] = calc_acc(logits_e[k], targets_e[k])
        if k == 'final' and (not multi_label_cross):
            f1_e[k] = calc_f1(logits_e[k], targets_e[k], 'binary')
        else:
            f1_e[k] = calc_f1(logits_e[k], targets_e[k])
        mAP_e[k] = calc_mAP(logits_e[k], targets_e[k])
        prec_e[k] = calc_precision(logits_e[k], targets_e[k])
        rec_e[k] = calc_recall(logits_e[k], targets_e[k])
    
    if 'final' in acc_e:
        auc_final = None
        # auc_final = calc_auc(logits_e['final'], targets_e['final'])
        conf_mat = calc_confusion_matrix(logits_e['final'], targets_e['final'])
        conf_mat_norm = calc_confusion_matrix(logits_e['final'], targets_e['final'], norm='true')
    
    # return res
    res = {}
    for k in logits_e:
        if k == 'final':
            res['base'] = [acc_e[k], mAP_e[k], auc_final, f1_e[k], logits_e['final']]
        else:
            res[k] = [acc_e[k], mAP_e[k], logits_e[k]]
    
    # log res
    log('\n')
    for k in acc_e:
        if k == 'final':
            log(f'\tacc: {acc_e[k]}\t mAP: {mAP_e[k]}\t f1: {f1_e[k]}\t AUC: {auc_final}')
            log(f'\tprecision: {prec_e[k]}')
            log(f'\tconf mat: {conf_mat}')
            log(f'\tconf mat norm: {conf_mat_norm}')
            log(f'\t logits: {torch.mean(logits_e["final"], dim=0)}')
        else:
            log(f'\t{k} acc: {acc_e[k]}\t {k} mAP: {mAP_e[k]}\t {k} f1: {f1_e[k]}')
            log(f'\t{k} recall: {rec_e[k]}')
            log(f'\t{k} precision: {prec_e[k]}')
    log('\n')
    
    # relevs
    if model_name in ('PCPA', 'SENN'):
        relevs_e = torch.cat(relevs_e, dim=0)
    return res, relevs_e

def get_one_feat_masks(n_feat):
    '''
    n_feat: int, num of features
    '''
    return torch.eye(n_feat)


def get_batch_morf_masks(relevs):
    '''
    relevs: torch.tensor(b, n_feat)
    return:
        masks: torch.tensor(b, n_feat+1, n_feat)
    '''
    b_size = relevs.size(0)
    n_feat = relevs.size(1)
    neg_masks = (relevs<0)  # b, n_feat
    importances = torch.abs(relevs)
    idcs = torch.argsort(importances, dim=-1, descending=True)  # b, n_feat
    # nn_idcs = idcs.repeat(n_feat, 1, 1).permute(1, 0, 2)  # b, n_feat, n_feat
    # nn = torch.arange(n_feat).repeat(n_feat)  # n_feat, n_feat
    # tril_idcs = torch.tril(nn_idcs, diagonal=0)  # b, n_feat, n_feat
    # tril_idcs += torch.triu(-1*torch.ones(size=(n_feat, n_feat)), diagonal=1)  # lower triangle with rest part -1

    masks = torch.ones(size=(b_size, n_feat+1, n_feat))
    for i in range(b_size):
        for j in range(n_feat):
            masks[i, j+1, idcs[i, :j+1]] = 0
    # turn over masks of neg relev
    masks = masks.permute(0, 2, 1)
    masks[neg_masks] -= 1
    masks[neg_masks] *= -1

    return masks.permute(0, 2, 1)


if __name__ == '__main__':
    test()