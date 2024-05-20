import os
import pickle
import time

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import copy
import numpy as np
from tqdm import tqdm

from tools.datasets.PIE_JAAD import PIEDataset
from tools.datasets.TITAN import TITAN_dataset
from tools.utils import makedir, draw_curves, img_nm_int2str, vid_id_int2str, ped_id_int2str, \
    draw_boxes_on_img, visualize_featmap3d_simple
from tools.log import create_logger
from tools.metrics import *
from tools.plot import vis_ego_sample, vis_weight_single_cls, EGO_RANGE
from train_test import train_test2
from config import dataset_root


BG_ROOT = {
    'PIE': os.path.join(dataset_root, 'PIE_dataset/images'),
    'JAAD': os.path.join(dataset_root, 'JAAD/images'),
    'TITAN': os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset/images_anonymized')
}

IMG_ROOT = {
    'PIE': os.path.join(dataset_root, 'PIE_dataset/cropped_images/even_padded/224w_by_224h'),
    'JAAD': os.path.join(dataset_root, 'JAAD/cropped_images/even_padded/224w_by_224h'),
    'TITAN': os.path.join(dataset_root, 'TITAN/TITAN_extra/cropped_images/even_padded/224w_by_224h/ped/')
}

SK_ROOT = {
    'PIE': os.path.join(dataset_root, 'PIE_dataset/sk_vis/even_padded/288w_by_384h'),
    'JAAD': os.path.join(dataset_root, 'JAAD/sk_vis/even_padded/288w_by_384h'),
    'TITAN': os.path.join(dataset_root, 'TITAN/TITAN_extra/sk_vis/even_padded/288w_by_384h/')
}

CTX_ROOT = {
    'PIE': os.path.join(dataset_root, 'PIE_dataset/context/ori_local/224w_by_224h'),
    'JAAD': os.path.join(dataset_root, 'JAAD/context/ori_local/224w_by_224h'),
    'TITAN': os.path.join(dataset_root, 'TITAN/TITAN_extra/context/ori_local/224w_by_224h/ped'),
}

def main():
    parser = argparse.ArgumentParser()
    print('start')
    # general setting
    parser.add_argument('--ckpt_path', type=str, 
                        default=\
                        '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models_1/SLE_traj_ego_img_skeleton_context/08Oct2023-20h15m26s/ckpt/44__0.pth'
                        )
    parser.add_argument('--args_path', type=str, 
                        default=\
                        '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models_1/SLE_traj_ego_img_skeleton_context/08Oct2023-20h15m26s/args.pkl'
                            )
    parser.add_argument('--exp_type', type=str, 
                        default='custom_mask',
                        help='custom_mask or select_sample or morf')
    parser.add_argument('--exp_type', type=int,
                        help='concept indeces to keep, split by space')
    parser.add_argument('--cross_dataset_exp', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='TITAN')
    parser.add_argument('--split', type=str, default='test')
    INDICES_TO_KEEP = []
    # data
    # parser.add_argument('--dataset_name', type=str, default='TITAN')

    test_args = parser.parse_args()
    ckpt_path = test_args.ckpt_path
    args_path = test_args.args_path
    exp_type = test_args.exp_type
    cross_dataset_exp = test_args.cross_dataset_exp
    split = test_args.split

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
    if cross_dataset_exp:
        dataset_name = test_args.dataset_name
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
    test_dir = os.path.join(model_dir, 'test', test_id)
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
    print('dataset_name', dataset_name)
    if dataset_name in ('PIE', 'JAAD'):
        test_dataset = PIEDataset(dataset_name=dataset_name, 
                                  seq_type=seq_type,
                                    obs_len=obs_len, pred_len=pred_len, 
                                    do_balance=balance_test, 
                                    subset=split, 
                                    bbox_size=ped_img_size, 
                                    img_norm_mode=img_norm_mode, color_order=color_order,
                                    resize_mode=resize_mode, 
                                    max_occ=test_max_occ, min_wh=test_min_wh,
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
                                    obs_interval=obs_interval,
                                    augment_mode='none',
                                    )
    elif dataset_name == 'TITAN':
        test_dataset = TITAN_dataset(sub_set='default_'+split, norm_traj=norm_pos,
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
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                               batch_size=16, 
                                               shuffle=False,
                                                num_workers=dataloader_workers, 
                                                pin_memory=False)
    
    # construct the model
    log('----------------------------Construct model-----------------------------')
    model = torch.load(ckpt_path)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    # mask the concepts
    with torch.no_grad():
        if model_name == 'SLE':
            if use_cross:
                weight = model.module.last_layer.weight
                ori_weight = copy.deepcopy(model.module.last_layer.weight)
            else:
                weight = model.module.atomic_layer.weight
                ori_weight = copy.deepcopy(model.module.atomic_layer.weight)
            if len(INDICES_TO_KEEP) == 0:
                INDICES_TO_KEEP = list(range(ori_weight.size(1)))
            for i in range(weight.size(1)):
                if i not in INDICES_TO_KEEP:
                    weight[:, i] = 0 * weight[:, i]
            if use_cross:
                model.module.last_layer.weight = weight
            else:
                model.module.atomic_layer.weight = weight
    
    # test
    log('\nTesting')
    if exp_type == 'custom_mask':
        test_res = train_test2(model=model,
                        dataloader=test_loader,
                        optimizer=None,
                        loss_func=loss_func,
                        loss_weight=loss_weight,
                        loss_weight_batch=loss_weight_batch,
                        log=log,
                        device=device,
                        data_types=modalities,
                        num_classes=num_classes,
                        ctx_mode=ctx_mode,
                        orth_type=orth_type,
                        use_cross=use_cross,
                        multi_label_cross=multi_label_cross, 
                        use_atomic=use_atomic, use_complex=use_complex, use_communicative=use_communicative, use_transporting=use_transporting, use_age=use_age,
                        model_name=model_name,
                        )
    elif exp_type == 'select_sample':
        # initialize result dict
        max_n_sample = 5
        
        cls_to_correct = {}
        cls_to_wrong = {}
        for cls in range(ori_weight.size(0)):
            cls_to_correct[cls] = []
            cls_to_wrong[cls] = []
            for _ in range(max_n_sample):
                cls_to_correct[cls].append({
                    'set_id_int': None,  # txt
                    'vid_id_int': None,  # txt
                    'ped_id_int': None,  # txt
                    'img_nm_int': None,  # txt
                    'logits_norm': None,  # txt
                    'label': None,  # txt
                    'activation_score': None,  # txt
                    'ori_inputs': {
                        'traj': None,  # txt
                        'ego': None,  # txt
                        'img': None,
                        'skeleton': None,
                        'context': None,
                    },
                    'channel_weights': {
                        'img': None,  # n_proto, C
                        'skeleton': None,
                        'context': None
                    },
                    'ori_feats': {
                        'img': None,
                        'skeleton': None,
                        'context': None
                    },
                })
                cls_to_wrong[cls].append({
                    'set_id_int': None,
                    'vid_id_int': None,
                    'ped_id_int': None,
                    'img_nm_int': None,
                    'logits_norm': None,
                    'label': None,
                    'activation_score': None,
                    'ori_inputs': {
                        'traj': None,
                        'ego': None,
                        'img': None,
                        'skeleton': None,
                        'context': None,
                    },
                    'channel_weights': {
                        'img': None,
                        'skeleton': None,
                        'context': None
                    },
                    'ori_feats': {
                        'img': None,
                        'skeleton': None,
                        'context': None
                    },
                })
        concept_to_sample = {}
        for p in INDICES_TO_KEEP:
            concept_to_sample[p] = []
            for _ in range(max_n_sample):
                concept_to_sample[p].append({
                    'set_id_int': None,
                    'vid_id_int': None,
                    'ped_id_int': None,
                    'img_nm_int': None,
                    'logits_norm': None,
                    'label': None,
                    'activation_score': None,
                    'ori_inputs': {
                        'traj': None,
                        'ego': None,
                        'img': None,
                        'skeleton': None,
                        'context': None,
                    },
                    'channel_weights': {
                        'img': None,
                        'skeleton': None,
                        'context': None
                    },
                    'ori_feats': {
                        'img': None,
                        'skeleton': None,
                        'context': None
                    },
                })

        # run
        for iter, data in enumerate(tqdm(test_loader)):
            # get inputs and labels
            labels = {}
            target = data['pred_intent'].view(-1) # idx, not one hot
            labels['target'] = target.to(device)
            if dataset_name == 'TITAN':
                labels['atomic_actions'] = data['atomic_actions'].to(device).view(-1)
                labels['simple_context'] = data['simple_context'].to(device).view(-1)
                labels['complex_context'] = data['complex_context'].to(device).view(-1)
                labels['communicative'] = data['communicative'].to(device).view(-1)
                labels['transporting'] = data['transporting'].to(device).view(-1)
                labels['age'] = data['age'].to(device).view(-1)
            inputs = {}
            inputs['ped_id_int'] = data['ped_id_int']
            inputs['img_nm_int'] = data['img_nm_int']
            
            if dataset_name == 'PIE':
                inputs['set_id_int'] = data['set_id_int']
                inputs['vid_id_int'] = data['vid_id_int']
            elif dataset_name == 'JAAD':
                inputs['vid_id_int'] = data['vid_id_int']
            elif dataset_name == 'TITAN':
                inputs['vid_id_int'] = data['clip_id_int']

            if use_img:
                inputs['img'] = data['ped_imgs'].to(device)
                inputs['img_ijhw'] = data['img_ijhw'].to(device)
            if use_skeleton:
                inputs['skeleton'] = data['obs_skeletons'].to(device)
                inputs['sk_ijhw'] = data['sk_ijhw'].to(device)
            if use_context:
                inputs['context'] = data['obs_context'].to(device)
                inputs['ctx_ijhw'] = data['ctx_ijhw'].to(device)
            if use_traj:
                inputs['traj'] = data['obs_bboxes'].to(device)
                inputs['traj_unnormed'] = data['obs_bboxes_unnormed'].to(device)
            if use_ego:
                inputs['ego'] = data['obs_ego'].to(device)
            inputs['hflip_flag'] = data['hflip_flag'].to(device)

            # get outputs
            ori_feats = {}
            protos = {}
            scores = []
            for k in ('img', 'skeleton', 'context'):
                if k in modalities:
                    ori_feats[k] = None
            with torch.no_grad():
                if 'traj' in modalities:
                    traj_simi, _, _, traj_protos = \
                        model.module.traj_model(inputs['traj'])
                    scores.append(traj_simi)  # activition scores: B n_concept
                if 'ego' in modalities:
                    ego_simi, _, _, ego_protos = \
                        model.module.ego_model(inputs['ego'])
                    scores.append(ego_simi)  # B n_concept
                if 'img' in modalities:
                    img_simi, _, _, img_protos = \
                        model.module.img_model(inputs['img'])
                    scores.append(img_simi)  # B n_concept
                    img_ori_feats = \
                        model.module.img_model.backbone(inputs['img'])
                    ori_feats['img'] = img_ori_feats
                    protos['img'] = img_protos
                if 'skeleton' in modalities:
                    sk_simi, _, _, sk_protos = \
                        model.module.sk_model(inputs['skeleton'])
                    scores.append(sk_simi)  # B n_concept
                    sk_ori_feats = \
                        model.module.sk_model.backbone(inputs['skeleton'])
                    ori_feats['skeleton'] = sk_ori_feats
                    protos['skeleton'] = sk_protos
                if 'context' in modalities:
                    ctx_simi, _, _, ctx_protos = \
                        model.module.ctx_model(inputs['context'])
                    scores.append(ctx_simi)  # B n_concept
                    ctx_ori_feats = \
                        model.module.ctx_model.backbone(inputs['context'])
                    ori_feats['context'] = ctx_ori_feats
                    protos['context'] = ctx_protos  # B n_proto C
                scores = torch.concat(scores, dim=1)  # B, totao_n_proto
                assert scores.size(1) == ori_weight.size(1), \
                    (scores.size(), ori_weight.size())
                
                # get logits
                if use_cross:
                    logits = model.module.last_layer(scores)  # B, n_cls
                    gt = labels['target'].view(-1)
                elif use_atomic:
                    logits = model.module.atomic_layer(scores)
                    gt = labels['atomic_actions'].view(-1)
                logits_norm = F.softmax(logits, dim=1)  # B, n_cls
                
                # divide correct and wrong predictions
                pred = torch.argmax(logits_norm, dim=1)  # B,
                correct = pred == gt
                wrong = ~correct
                
                # sort logits
                logits_cor = logits_norm[correct]  # n_cor, n_cls
                logits_wro = logits_norm[wrong]  # n_wro, n_cls
                for cls in range(ori_weight.size(0)):
                    # correct samples
                    if len(logits_cor) > 0:
                        max_idx = torch.argmax(logits_cor[:, cls])
                        for i in range(max_n_sample):
                            if cls_to_correct[cls][i]['logits_norm'] is None \
                                or logits_cor[max_idx, cls] > \
                                    cls_to_correct[cls][i]['logits_norm'][cls]:
                                # # 所有样本后移一位
                                # for j in range(max_n_sample-i-1):
                                #     cls_to_correct[cls][max_n_sample-1-j] = \
                                #         cls_to_correct[cls][max_n_sample-2-j]
                                # insert current sample
                                cls_to_correct[cls][i]['logits_norm'] = \
                                    logits_cor[max_idx].detach().cpu().numpy()  # n_cls,
                                if dataset_name == 'PIE':
                                    cls_to_correct[cls][i]['set_id_int'] = \
                                        inputs['set_id_int']\
                                            [correct][max_idx].detach().cpu().numpy()
                                cls_to_correct[cls][i]['vid_id_int'] = \
                                    inputs['vid_id_int']\
                                        [correct][max_idx].detach().cpu().numpy()
                                cls_to_correct[cls][i]['ped_id_int'] = \
                                    inputs['ped_id_int']\
                                        [correct][max_idx].detach().cpu().numpy()
                                cls_to_correct[cls][i]['img_nm_int'] = \
                                    inputs['img_nm_int']\
                                        [correct][max_idx].detach().cpu().numpy()
                                cls_to_correct[cls][i]['label'] = \
                                    gt[correct][max_idx].detach().cpu().numpy()
                                cls_to_correct[cls][i]['activation_score'] = \
                                    scores[correct][max_idx].detach().cpu().numpy()
                                if 'traj' in modalities:
                                    cls_to_correct[cls][i]['ori_inputs']['traj']=\
                                        inputs['traj_unnormed']\
                                            [correct][max_idx].detach().cpu().numpy()
                                if 'ego' in modalities:
                                    cls_to_correct[cls][i]['ori_inputs']['ego']=\
                                        inputs['ego']\
                                            [correct][max_idx].detach().cpu().numpy()
                                if 'img' in modalities:
                                    cls_to_correct[cls][i]['ori_feats']['img']=\
                                        img_ori_feats\
                                            [correct][max_idx].detach().cpu().numpy()
                                    cls_to_correct[cls][i]['channel_weights']['img']=\
                                        img_protos\
                                            [correct][max_idx].detach().cpu().numpy()
                                if 'skeleton' in modalities:
                                    cls_to_correct[cls][i]['ori_feats']['skeleton']=\
                                        sk_ori_feats\
                                            [correct][max_idx].detach().cpu().numpy()
                                    cls_to_correct[cls][i]['channel_weights']['skeleton']=\
                                        sk_protos\
                                            [correct][max_idx].detach().cpu().numpy()
                                if 'context' in modalities:
                                    cls_to_correct[cls][i]['ori_feats']['context']=\
                                        ctx_ori_feats\
                                            [correct][max_idx].detach().cpu().numpy()
                                    cls_to_correct[cls][i]['channel_weights']['context']=\
                                        ctx_protos\
                                            [correct][max_idx].detach().cpu().numpy()
                                break

                    # wrong samples
                    if len(logits_wro) > 0:
                        max_idx = torch.argmax(logits_wro[:, cls])
                        for i in range(max_n_sample):
                            if cls_to_wrong[cls][i]['logits_norm'] is None \
                                or logits_wro[max_idx, cls] > \
                                    cls_to_wrong[cls][i]['logits_norm'][cls]:
                                # # 所有样本后移一位
                                # for j in range(max_n_sample-i-1):
                                #     cls_to_wrong[cls][max_n_sample-1-j] = \
                                #         cls_to_wrong[cls][max_n_sample-2-j]
                                # insert current sample
                                cls_to_wrong[cls][i]['logits_norm'] = \
                                    logits_wro[max_idx].detach().cpu().numpy()  # n_cls,
                                if dataset_name == 'PIE':
                                    cls_to_wrong[cls][i]['set_id_int'] = \
                                        inputs['set_id_int']\
                                            [wrong][max_idx].detach().cpu().numpy()
                                cls_to_wrong[cls][i]['vid_id_int'] = \
                                    inputs['vid_id_int']\
                                        [wrong][max_idx].detach().cpu().numpy()
                                cls_to_wrong[cls][i]['ped_id_int'] = \
                                    inputs['ped_id_int']\
                                        [wrong][max_idx].detach().cpu().numpy()
                                cls_to_wrong[cls][i]['img_nm_int'] = \
                                    inputs['img_nm_int']\
                                        [wrong][max_idx].detach().cpu().numpy()
                                cls_to_wrong[cls][i]['label'] = \
                                    gt[wrong][max_idx].detach().cpu().numpy()
                                cls_to_wrong[cls][i]['activation_score'] = \
                                    scores[wrong][max_idx].detach().cpu().numpy()
                                if 'traj' in modalities:
                                    cls_to_wrong[cls][i]['ori_inputs']['traj']=\
                                        inputs['traj_unnormed']\
                                            [wrong][max_idx].detach().cpu().numpy()
                                if 'ego' in modalities:
                                    cls_to_wrong[cls][i]['ori_inputs']['ego']=\
                                        inputs['ego']\
                                            [wrong][max_idx].detach().cpu().numpy()
                                if 'img' in modalities:
                                    cls_to_wrong[cls][i]['ori_feats']['img']=\
                                        img_ori_feats\
                                            [wrong][max_idx].detach().cpu().numpy()
                                    
                                    cls_to_wrong[cls][i]['channel_weights']['img']=\
                                        img_protos\
                                            [wrong][max_idx].detach().cpu().numpy()
                                    if not hasattr(model.module.img_model, 'score_sum_linear') or \
                                        model.module.img_model.score_sum_linear:
                                        cls_to_wrong[cls][i]['channel_weights']['img'] *= \
                                            torch.squeeze(model.module.img_model.sum_linear.weight).cpu().numpy()
                                if 'skeleton' in modalities:
                                    cls_to_wrong[cls][i]['ori_feats']['skeleton']=\
                                        sk_ori_feats\
                                            [wrong][max_idx].detach().cpu().numpy()
                                    cls_to_wrong[cls][i]['channel_weights']['skeleton']=\
                                        sk_protos\
                                            [wrong][max_idx].detach().cpu().numpy()
                                    if not hasattr(model.module.sk_model, 'score_sum_linear') or \
                                        model.module.sk_model.score_sum_linear:
                                        cls_to_wrong[cls][i]['channel_weights']['skeleton'] *= \
                                            torch.squeeze(model.module.sk_model.sum_linear.weight).cpu().numpy()
                                if 'context' in modalities:
                                    cls_to_wrong[cls][i]['ori_feats']['context']=\
                                        ctx_ori_feats\
                                            [wrong][max_idx].detach().cpu().numpy()
                                    cls_to_wrong[cls][i]['channel_weights']['context']=\
                                        ctx_protos\
                                            [wrong][max_idx].detach().cpu().numpy()
                                    if not hasattr(model.module.ctx_model, 'score_sum_linear') or \
                                        model.module.ctx_model.score_sum_linear:
                                        cls_to_wrong[cls][i]['channel_weights']['context'] *= \
                                            torch.squeeze(model.module.ctx_model.sum_linear.weight).cpu().numpy()
                                break
                    
                # sort activation scores
                for p in INDICES_TO_KEEP:
                    max_idx = torch.argmax(scores[:, p])
                    for i in range(max_n_sample):
                        if concept_to_sample[p][i]['activation_score'] is None \
                            or scores[max_idx, p] > \
                                concept_to_sample[p][i]['activation_score'][p]:
                            # # 所有样本后移一位
                            # for j in range(max_n_sample-i-1):
                            #     concept_to_sample[p][max_n_sample-1-j] = \
                            #         copy.deepcopy(concept_to_sample[p][max_n_sample-2-j])
                            # insert current sample
                            concept_to_sample[p][i]['logits_norm'] = \
                                logits_norm[max_idx].detach().cpu().numpy()
                            if dataset_name == 'PIE':
                                concept_to_sample[p][i]['set_id_int'] = \
                                    inputs['set_id_int']\
                                        [max_idx].detach().cpu().numpy()
                            concept_to_sample[p][i]['vid_id_int'] = \
                                inputs['vid_id_int']\
                                    [max_idx].detach().cpu().numpy()
                            concept_to_sample[p][i]['ped_id_int'] = \
                                inputs['ped_id_int']\
                                    [max_idx].detach().cpu().numpy()
                            concept_to_sample[p][i]['img_nm_int'] = \
                                inputs['img_nm_int']\
                                    [max_idx].detach().cpu().numpy()
                            concept_to_sample[p][i]['label'] = \
                                gt[max_idx].detach().cpu().numpy()
                            concept_to_sample[p][i]['activation_score'] = \
                                scores[max_idx].detach().cpu().numpy()
                            if 'traj' in modalities:
                                concept_to_sample[p][i]['ori_inputs']['traj'] = \
                                    inputs['traj_unnormed']\
                                        [max_idx].detach().cpu().numpy()
                            if 'ego' in modalities:
                                concept_to_sample[p][i]['ori_inputs']['ego'] = \
                                    inputs['ego']\
                                        [max_idx].detach().cpu().numpy()
                            if 'img' in modalities:
                                concept_to_sample[p][i]['ori_feats']['img'] = \
                                    img_ori_feats\
                                        [max_idx].detach().cpu().numpy()
                                concept_to_sample[p][i]['channel_weights']['img']=\
                                    img_protos\
                                        [max_idx].detach().cpu().numpy()
                                if not hasattr(model.module.img_model, 'score_sum_linear') or \
                                        model.module.img_model.score_sum_linear:
                                        concept_to_sample[p][i]['channel_weights']['img'] *= \
                                            torch.squeeze(model.module.img_model.sum_linear.weight).cpu().numpy()
                            if 'skeleton' in modalities:
                                concept_to_sample[p][i]['ori_feats']['skeleton'] = \
                                    sk_ori_feats\
                                        [max_idx].detach().cpu().numpy()
                                concept_to_sample[p][i]['channel_weights']['skeleton'] = \
                                    sk_protos\
                                        [max_idx].detach().cpu().numpy()
                                if not hasattr(model.module.sk_model, 'score_sum_linear') or \
                                        model.module.sk_model.score_sum_linear:
                                        concept_to_sample[p][i]['channel_weights']['skeleton'] *= \
                                            torch.squeeze(model.module.sk_model.sum_linear.weight).cpu().numpy()
                            if 'context' in modalities:
                                concept_to_sample[p][i]['ori_feats']['context'] = \
                                    ctx_ori_feats\
                                        [max_idx].detach().cpu().numpy()
                                concept_to_sample[p][i]['channel_weights']['context']=\
                                    ctx_protos\
                                        [max_idx].detach().cpu().numpy()
                                if not hasattr(model.module.ctx_model, 'score_sum_linear') or \
                                        model.module.ctx_model.score_sum_linear:
                                        concept_to_sample[p][i]['channel_weights']['context'] *= \
                                            torch.squeeze(model.module.ctx_model.sum_linear.weight).cpu().numpy()
                            break
                
        # save and visualize
        # correct samples
        cor_sample_dir = os.path.join(test_dir, 'correct_logit')
        makedir(cor_sample_dir)
        with open(os.path.join(cor_sample_dir, 'dict.pkl'), 'wb') as f:
            pickle.dump(cls_to_correct, f)
        visualize_samples(model,
                          cls_to_correct,
                          cor_sample_dir,
                          dataset_name,
                          modalities,
                          max_n_sample,
                          prefix='cls')
        # wrong samples
        wro_sample_dir = os.path.join(test_dir, 'wrong_logits')
        makedir(wro_sample_dir)
        with open(os.path.join(wro_sample_dir, 'dict.pkl'), 'wb') as f:
            pickle.dump(cls_to_wrong, f)
        visualize_samples(model,
                          cls_to_wrong,
                          wro_sample_dir,
                          dataset_name,
                          modalities,
                          max_n_sample,
                          prefix='cls')
        # samples of concepts
        concept_sample_dir = os.path.join(test_dir, 'concept_sample')
        makedir(concept_sample_dir)
        with open(os.path.join(concept_sample_dir, 'dict.pkl'), 'wb') as f:
            pickle.dump(concept_to_sample, f)
        visualize_samples(model,
                          concept_to_sample,
                          concept_sample_dir,
                          dataset_name,
                          modalities,
                          max_n_sample,
                          prefix='concept')
        

    else:
        raise ValueError(exp_type)
    log(f'Test dir: {test_dir}')
    log(f'Concept indices: {INDICES_TO_KEEP}')


def visualize_samples(model,
                      sample_dict, 
                      save_dir, 
                      dataset_name,
                      modalities,
                      max_n_sample,
                      prefix='cls',  # cls or concept

                      ):
    for cls in tqdm(sample_dict):
        cls_dir = os.path.join(save_dir, f'{prefix}_{cls}')
        makedir(cls_dir)
        for i in range(max_n_sample):
            # correct samples
            cls_dir_i = os.path.join(cls_dir, str(i))
            makedir(cls_dir_i)
            set_id_int = sample_dict[cls][i]['set_id_int']
            vid_id_int = sample_dict[cls][i]['vid_id_int']
            ped_id_int = sample_dict[cls][i]['ped_id_int']
            img_nm_int = sample_dict[cls][i]['img_nm_int']
            info_cor = \
                ['\nlabel: '+str(sample_dict[cls][i]['label'])]+\
                ['\nset id int: '+str(set_id_int)]+\
                ['\nvid id int: '+str(vid_id_int)]+\
                ['\nped id int: '+str(ped_id_int)]+\
                ['\nimg nm int: '+str(img_nm_int)]+\
                ['\nlogits norm: '+str(sample_dict[cls][i]['logits_norm'])]+\
                ['\nego: '+str(sample_dict[cls][i]['ori_inputs']['ego'])]+\
                ['\nactivation score: '+str(sample_dict[cls][i]['activation_score'])]
            txt_path = os.path.join(cls_dir_i, 'info.txt')
            with open(txt_path, 'w') as f:
                f.writelines(info_cor)
            
            # visualize activation score
            act_score_path = os.path.join(cls_dir_i, 'activation_score.png')
            try:
                vis_weight_single_cls(sample_dict[cls][i]['activation_score'],
                                    path=act_score_path)
            except:
                import pdb;pdb.set_trace()

            # visualize ori inputs and heatmaps
            concept_vis_dir = os.path.join(cls_dir_i, 'concept_vis')
            makedir(concept_vis_dir)
            for k in ('img', 'skeleton', 'context', 'traj', 'ego'):
                if k not in modalities:
                    continue
                ori_input_dir = os.path.join(cls_dir_i, 'ori_input_'+k)
                makedir(ori_input_dir)
                # traj
                if k == 'traj':
                    # get background
                    img_nm = img_nm_int2str(img_nm_int[-1], 
                                            dataset_name=dataset_name)
                    if dataset_name in ('PIE', 'JAAD'):
                        vid_nm = vid_id_int2str(vid_id_int[0])
                        if dataset_name == 'PIE':
                            set_nm = 'set0' + str(set_id_int[0])
                            bg_path = os.path.join(BG_ROOT[dataset_name], 
                                                    set_nm, 
                                                    vid_nm, 
                                                    img_nm)
                        else:
                            bg_path = os.path.join(BG_ROOT[dataset_name], 
                                                    vid_nm, 
                                                    img_nm)
                    elif dataset_name == 'TITAN':
                        vid_nm = 'clip_'+str(vid_id_int)
                        bg_path = os.path.join(BG_ROOT[dataset_name], 
                                                vid_nm, 
                                                'images', 
                                                img_nm)
                    background = cv2.imread(filename=bg_path)
                    blank_bg = np.zeros(background.shape) + 255
                    traj = sample_dict[cls][i]['ori_inputs']['traj']
                    traj_img_w_bg = draw_boxes_on_img(copy.deepcopy(background), traj)
                    traj_img = draw_boxes_on_img(blank_bg, traj)
                    cv2.imwrite(filename=os.path.join(ori_input_dir, 
                                                        'traj_w_bg.png'), 
                                img=traj_img_w_bg)
                    cv2.imwrite(filename=os.path.join(ori_input_dir, 
                                                        'traj.png'), 
                                img=traj_img)
                    cv2.imwrite(filename=os.path.join(ori_input_dir, 
                                                        'bg.png'),
                                img=background)
                # ego
                if k == 'ego':
                    ego = sample_dict[cls][i]['ori_inputs']['ego']
                    if len(ego.shape) == 2:
                        ego = ego[:, 0]
                    lim = EGO_RANGE[dataset_name]
                    vis_ego_sample(ego,
                                    lim=lim,
                                    path=os.path.join(ori_input_dir, 'ego.png'))
                # img
                elif k in ('img', 'context', 'skeleton'):
                    if k == 'img':
                        root_dict = IMG_ROOT
                    elif k == 'skeleton':
                        root_dict = SK_ROOT
                    else:
                        root_dict = CTX_ROOT
                    img_nms = []
                    img_paths = []
                    imgs = []
                    for int_ in img_nm_int:
                        img_nms.append(img_nm_int2str(int_, dataset_name))
                    if dataset_name in ('PIE', 'JAAD'):
                        if ped_id_int[2] >= 0:
                            ped_nm = str(ped_id_int[0])+\
                                str(ped_id_int[1])+str(ped_id_int[2])
                        else:
                            ped_nm = str(ped_id_int[0])+\
                                str(ped_id_int[1])+str(-ped_id_int[2])+'b'
                        for img_nm in img_nms:
                            img_paths.append(
                                os.path.join(root_dict[dataset_name],
                                            ped_nm,
                                            img_nm)
                                            )
                    elif dataset_name == 'TITAN':
                        vid_nm = str(vid_id_int)

                        for img_nm in img_nms:
                            img_paths.append(
                                os.path.join(root_dict[dataset_name],
                                            vid_nm,
                                            str(ped_id_int),
                                            img_nm)
                                            )
                    # ori inputs
                    for t in range(len(img_paths)):
                        img_path = img_paths[t]
                        img = cv2.imread(img_path)
                        imgs.append(img)
                        try:
                            cv2.imwrite(filename=os.path.join(ori_input_dir,
                                                            f'ori_img_{t}.png'),
                                        img=img)
                        except:
                            print(img_path)
                            import pdb;pdb.set_trace()
                            raise ValueError

                    # heatmaps for each concept
                    imgs = np.stack(imgs, axis=0)  # THWC
                    feat = sample_dict[cls][i]['ori_feats'][k]  # CTHW
                    try:
                        assert len(feat.shape) == 4, feat.shape
                    except:
                        import pdb;pdb.set_trace()
                    feat = np.transpose(feat, axes=(1, 2, 3, 0))  # T H W C
                    c_weights = sample_dict[cls][i]['channel_weights'][k]  # n_proto C
                    assert len(c_weights.shape) == 2, c_weights.shape
                    
                    vis_modal_dir = os.path.join(concept_vis_dir, k)
                    makedir(vis_modal_dir)
                    for p in range(c_weights.shape[0]):
                        vis_dir = os.path.join(vis_modal_dir, 
                                               f'concept_{p}')
                        makedir(vis_dir)
                        cur_weight = c_weights[p]  # C,
                        visualize_featmap3d_simple(feat,
                                                   imgs,
                                                   mode='weighted',
                                                   channel_weights=cur_weight,
                                                   save_dir=vis_dir,
                                                   )


if __name__ == '__main__':
    main()