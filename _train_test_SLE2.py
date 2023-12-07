from pickletools import optimize
from socket import TIPC_ADDR_ID
import time
from pandas import MultiIndex
from sklearn.preprocessing import label_binarize
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2
import os

from helpers import list_of_distances, make_one_hot
from utils import idx2onehot, cls_weights, get_cls_weights_multi, calc_n_samples_cls
from tools.metrics import *
from _loss_fuctions import FocalLoss, FocalLoss2, \
    WeightedCrossEntropy, FocalLoss3, \
        calc_orth_loss, calc_orth_loss_fix, calc_balance_loss, focal_tversky, \
        dimension_wise_contrast
from helpers import makedir
from tools.datasets.TITAN import NUM_CLS_ATOMIC, \
    NUM_CLS_COMPLEX, NUM_CLS_COMMUNICATIVE, NUM_CLS_TRANSPORTING, NUM_CLS_AGE
from tools.gpu_mem_track import MemTracker

def train_test2(model, dataloader, optimizer=None, model_name='SLE',
                loss_func='weighted_ce',
                loss_weight='switch',
                loss_weight_batch=0,
                log=print, device='cuda:0',
                data_types=['img'],
                check_grad=False, orth_type=0, vis_path=None, display_logits=True,
                num_classes=2,
                ctx_mode='local',
                multi_label_cross=0, 
                use_atomic=0,
                use_complex=0,
                use_communicative=0,
                use_transporting=0,
                use_age=0,
                use_cross=1,
                mask=None,
                lambda1=0.01,
                lambda2=1,
                lambda3=0.1,
                lambda_contrast=0.5,
                ):
    # gpu_tracker = MemTracker()
    start = time.time()
    d_time = 0
    c_time = 0
    is_train = optimizer is not None
    total_ce = 0
    total_mse = 0
    total_orth = 0
    total_contrast = 0
    total_balance = 0
    feat_statis = {}
    simi_mat_statis = {
        'img':{'min':[], 'max':[]},
        'skeleton':{'min':[], 'max':[]},
        'context':{'min':[], 'max':[]},
        'traj':{'min':[], 'max':[]},
        'ego':{'min':[], 'max':[]},
    }

    
    # get class weights
    cls_weights_multi = get_cls_weights_multi(model=model,
                                                dataloader=dataloader,
                                                loss_weight=loss_weight,
                                                device=device,
                                                multi_label_cross=multi_label_cross,
                                                use_cross=use_cross,
                                                use_atomic=use_atomic,
                                                use_complex=use_complex,
                                                use_communicative=use_communicative,
                                                use_transporting=use_transporting,
                                                use_age=use_age,
                                                )
    for k in cls_weights_multi:
        if cls_weights_multi[k] is not None:
            log(k + ' class weights: ' + str(cls_weights_multi[k]))

    # log task weights
    if loss_func == 'm_task_ce':
        if use_cross:
            log('logs2: ' + str(model.module.logs2))
        if use_atomic:
            log('atomic logs2: ' + str(model.module.atomic_logs2))
        if use_complex:
            log('complex logs2: ' + str(model.module.complex_logs2))
        if use_communicative:
            log('communicative logs2: ' + str(model.module.communicative_logs2))
        if use_transporting:
            log('transporting logs2: ' + str(model.module.transporting_logs2))
    # init loss func
    mse = torch.nn.MSELoss()
    if loss_func == 'focal1':
        focal_loss = FocalLoss()
    elif loss_func == 'focal2':
        focal_loss = FocalLoss2()
    elif loss_func == 'focal3':
        focal_loss = FocalLoss3(gamma=3)
    elif loss_func == 'weighted_ce' and loss_weight == 'trainable':
        weighted_ce = WeightedCrossEntropy()
    print('loss func: ',loss_func)
    
    # targets and logits for whole epoch
    targets_e = {}
    logits_e = {}

    # start iteration
    b_end = time.time()
    tbar = tqdm(dataloader, miniters=1)
    for iter, data in enumerate(tbar):
        # update trainable weights
        if loss_weight == 'trainable' and i > 0:
            if use_cross:
                cls_weights_multi['final'] = F.softmax(model.module.class_weights, 
                                                       dim=-1)
            if use_atomic:
                cls_weights_multi['atomic'] = F.softmax(model.module.atomic_weights, 
                                                        dim=-1)
            if use_complex:
                cls_weights_multi['complex'] = F.softmax(model.module.complex_weights, 
                                                         dim=-1)
            if use_communicative:
                cls_weights_multi['communicative'] = F.softmax(model.module.communicative_weights,
                                                                dim=-1)
            if use_transporting:
                cls_weights_multi['transporting'] = F.softmax(model.module.transporting_weights, 
                                                              dim=-1)
            if use_age:
                cls_weights_multi['age'] = F.softmax(model.module.age_weights, 
                                                     dim=-1)

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
        gt_traj = None

        # forward
        b_start = time.time()
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            orth_loss = 0
            balance_loss = 0
            mse_loss = 0
            if mask is not None:
                output = model(inputs, mask=mask)  # b, num classes
            else:
                output = model(inputs)  # b, num classes
            
            if model_name == 'SLE':
                logits, multi_protos, feats = output
            elif model_name == 'PCPA':
                logits, _ = output
            else:
                logits = output
            # print(logits.keys())
            # collect targets and logits in batch
            for k in logits:
                if iter == 0:
                    targets_e[k] = targets[k].detach()
                    logits_e[k] = logits[k].detach()
                else:
                    targets_e[k] = torch.cat((targets_e[k], targets[k].detach()), dim=0)
                    logits_e[k] = torch.cat((logits_e[k], logits[k].detach()), dim=0)
            # calc orth loss
            if orth_type > 0 and lambda1 > 0 and model_name == 'SLE':
                if model.module.sk_model.simi_func not in ('fix_proto1', 'fix_proto2'):
                    for k in data_types:
                        if k == 'context' and ('seg_multi' in ctx_mode):
                            for i in range(len(model.module.ctx_setting['seg_cls_idx'])):
                                protos = multi_protos[k][i]
                                orth_loss += calc_orth_loss(protos, orth_type)
                        else:
                            protos = multi_protos[k]  # B n_p C
                            orth_loss += calc_orth_loss(protos, orth_type)
                else:
                    if model.module.use_traj:
                        protos = model.module.traj_model.proto_vec
                        orth_loss += calc_orth_loss_fix(protos, orth_type)
                    if model.module.use_ego:
                        protos = model.module.ego_model.proto_vec
                        orth_loss += calc_orth_loss_fix(protos, orth_type)
                    if model.module.use_img:
                        protos = model.module.img_model.proto_vec
                        orth_loss += calc_orth_loss_fix(protos, orth_type)
                    if model.module.use_skeleton:
                        protos = model.module.sk_model.proto_vec
                        orth_loss += calc_orth_loss_fix(protos, orth_type)
                    if model.module.use_context:
                        protos = model.module.ctx_model.proto_vec
                        orth_loss += calc_orth_loss_fix(protos, orth_type)
                total_orth += orth_loss.item()

            # calc contrastive loss
            d_contrast_loss = 0
            if lambda_contrast > 0 and model_name == 'SLE':
                for k in feats:
                    _d_contrast_loss, simi_mat  = \
                        dimension_wise_contrast(feats[k])
                    d_contrast_loss = d_contrast_loss + _d_contrast_loss
                    simi_mat_statis[k]['max'].append(torch.max(simi_mat.detach()).cpu().item())
                    simi_mat_statis[k]['min'].append(torch.min(simi_mat.detach()).cpu().item())
                total_contrast += d_contrast_loss.item()

            # calc balance loss
            if lambda3 > 0:
                if use_cross:
                    balance_loss += calc_balance_loss(model.module.last_layer.weight)
                if use_atomic:
                    balance_loss += calc_balance_loss(model.module.atomic_layer.weight)
                total_balance += balance_loss.item()

            # calc cross entropy
            ce_losses = {}
            if loss_func == 'focal3':
                if use_cross:
                    ce_losses['final'] = focal_loss(logits['final'], targets['final'], weight=cls_weights_multi['final'])
                if use_atomic:
                    ce_losses['atomic'] = focal_loss(logits['atomic'], targets['atomic'], weight=cls_weights_multi['atomic'])
                if use_complex:
                    ce_losses['complex'] = focal_loss(logits['complex'], targets['complex'], weight=cls_weights_multi['complex'])
                if use_communicative:
                    ce_losses['communicative'] = focal_loss(logits['communicative'], targets['communicative'], weight=cls_weights_multi['communicative'])
                if use_transporting:
                    ce_losses['transporting'] = focal_loss(logits['transporting'], targets['transporting'], weight=cls_weights_multi['transporting'])
                if use_age:
                    ce_losses['age'] = focal_loss(logits['age'], targets['age'], weight=cls_weights_multi['age'])
            elif loss_func == 'weighted_ce' or loss_func == 'm_task_ce':
                if use_cross:
                    if loss_weight_batch:
                        n_cls = 13 if multi_label_cross else 2
                        num_samples_cls = calc_n_samples_cls(targets['final'], n_cls=n_cls)
                        cls_weights_multi['final'] = cls_weights(num_samples_cls, loss_weight, device=device)
                    ce_losses['final'] = F.cross_entropy(logits['final'], targets['final'], weight=cls_weights_multi['final'])
                    if loss_func == 'm_task_ce':
                        ce_losses['final'] = ce_losses['final'] * torch.exp(-model.module.logs2) + 0.5 * model.module.logs2
                if use_atomic:
                    if loss_weight_batch:
                        n_cls = NUM_CLS_ATOMIC
                        num_samples_cls = calc_n_samples_cls(targets['atomic'], n_cls=n_cls)
                        cls_weights_multi['atomic'] = cls_weights(num_samples_cls, loss_weight, device=device)
                    ce_losses['atomic'] = F.cross_entropy(logits['atomic'], targets['atomic'], weight=cls_weights_multi['atomic'])
                    if loss_func == 'm_task_ce':
                        ce_losses['atomic'] = ce_losses['atomic'] * torch.exp(-model.module.atomic_logs2) + 0.5 * model.module.atomic_logs2
                if use_complex:
                    if loss_weight_batch:
                        n_cls = NUM_CLS_COMPLEX
                        num_samples_cls = calc_n_samples_cls(targets['complex'], n_cls=n_cls)
                        cls_weights_multi['complex'] = cls_weights(num_samples_cls, loss_weight, device=device)
                    ce_losses['complex'] = F.cross_entropy(logits['complex'], targets['complex'], weight=cls_weights_multi['complex'])
                    if loss_func == 'm_task_ce':
                        ce_losses['complex'] = ce_losses['complex'] * torch.exp(-model.module.complex_logs2) + 0.5 * model.module.complex_logs2
                if use_communicative:
                    if loss_weight_batch:
                        n_cls = NUM_CLS_COMMUNICATIVE
                        num_samples_cls = calc_n_samples_cls(targets['communicative'], n_cls=n_cls)
                        cls_weights_multi['communicative'] = cls_weights(num_samples_cls, loss_weight, device=device)
                    ce_losses['communicative'] = F.cross_entropy(logits['communicative'], targets['communicative'], weight=cls_weights_multi['communicative'])
                    if loss_func == 'm_task_ce':
                        ce_losses['communicative'] = ce_losses['communicative'] * torch.exp(-model.module.communicative_logs2) + 0.5 * model.module.communicative_logs2
                if use_transporting:
                    if loss_weight_batch:
                        n_cls = NUM_CLS_TRANSPORTING
                        num_samples_cls = calc_n_samples_cls(targets['transporting'], n_cls=n_cls)
                        cls_weights_multi['transporting'] = cls_weights(num_samples_cls, loss_weight, device=device)
                    ce_losses['transporting'] = F.cross_entropy(logits['transporting'], targets['transporting'], weight=cls_weights_multi['transporting'])
                    if loss_func == 'm_task_ce':
                        ce_losses['transporting'] = ce_losses['transporting'] * torch.exp(-model.module.transporting_logs2) + 0.5 * model.module.transporting_logs2
                if use_age:
                    if loss_weight_batch:
                        n_cls = NUM_CLS_AGE
                        num_samples_cls = calc_n_samples_cls(targets['age'], n_cls=n_cls)
                        cls_weights_multi['age'] = cls_weights(num_samples_cls, loss_weight, device=device)
                    ce_losses['age'] = F.cross_entropy(logits['age'], targets['age'], weight=cls_weights_multi['age'])
                    if loss_func == 'm_task_ce':
                        ce_losses['age'] = ce_losses['age'] * torch.exp(-model.module.age_logs2) + 0.5 * model.module.age_logs2
            elif loss_func == 'bcewithlogits':
                pos_weight=torch.tensor([dataloader.dataset.n_nc / dataloader.dataset.n_c]).float().to(device)
                target_onehot = idx2onehot(targets['final'], logits['final'].size(1))
                ce_losses['final'] = F.binary_cross_entropy_with_logits(logits['final'], target_onehot.float(), pos_weight=pos_weight)
            elif loss_func == 'focal_tversky':
                for k in logits:
                    ce_losses[k] = focal_tversky(logits[k], targets[k])
            elif loss_func == 'focal_tversky+ce':
                for k in logits:
                    ce_losses[k] = focal_tversky(logits[k], targets[k]) + F.cross_entropy(logits[k], targets[k], weight=cls_weights_multi[k])
            else:
                ce_losses['final'] = 0
                if use_cross:
                    ce_losses['final'] = F.cross_entropy(logits['final'], targets['final'])
                if use_atomic:
                    ce_losses['atomic'] = F.cross_entropy(logits['atomic'], targets['atomic'])
                if use_complex:
                    ce_losses['complex'] = F.cross_entropy(logits['complex'], targets['complex'])
                if use_communicative:
                    ce_losses['communicative'] = F.cross_entropy(logits['communicative'], targets['communicative'])
                if use_transporting:
                    ce_losses['transporting'] = F.cross_entropy(logits['transporting'], targets['transporting'])

            # collect loss in batch
            if 'pred_traj' in data_types:
                total_mse += mse_loss.item()
            if use_cross:
                total_ce += ce_losses['final'].item()

            # calc losses and backward
            if is_train:
                loss = mse_loss + \
                    lambda1*((1-lambda_contrast)*orth_loss+lambda_contrast*d_contrast_loss) + \
                        lambda3*balance_loss
                for k in ce_losses:
                    # loss = loss + ce_losses[k]
                    if k == 'final':
                        loss = loss + ce_losses[k]
                    else:
                        loss = loss + lambda2*ce_losses[k]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # display
            data_prepare_time = b_start - b_end
            b_end = time.time()
            computing_time = b_end - b_start
            d_time += data_prepare_time
            c_time += computing_time
            display_dict = {'data': data_prepare_time, 
                            'compute': computing_time,
                            'd all': d_time,
                            'c all': c_time
                            }
            if use_cross:
                with torch.no_grad():
                    mean_logit = torch.mean(logits['final'].detach(), dim=0)
                if display_logits:
                    if num_classes == 2:
                        display_dict['logit'] = [round(logits['final'][0, 0].item(), 4), round(logits['final'][0, 1].item(), 4)]
                        display_dict['avg logit'] = [round(mean_logit[0].item(), 4), round(mean_logit[1].item(), 4)]
                    elif num_classes == 1:
                        display_dict['logit'] = [round(logits['final'][0, 0].item(), 4)]
                        display_dict['avg logit'] = [round(mean_logit[0].item(), 4)]
            tbar.set_postfix(display_dict)
        del inputs
        if is_train:
            del loss
        torch.cuda.empty_cache()
        # gpu_tracker.track()
    tbar.close()
    end = time.time()

    # calc metric
    acc_e = {}
    f1_e = {}
    f1b_e = {}
    mAP_e = {}
    auc_e = {}
    prec_e = {}
    rec_e = {}
    for k in logits_e:
        acc_e[k] = calc_acc(logits_e[k], targets_e[k])
        f1_e[k] = calc_f1(logits_e[k], targets_e[k])
        f1b_e[k] = f1_e[k]
        if use_cross and k == 'final' and (not multi_label_cross):
            # import pdb;pdb.set_trace()
            f1b_e[k] = calc_f1(logits_e[k], targets_e[k], 'binary')
        mAP_e[k] = calc_mAP(logits_e[k], targets_e[k])
        auc_e[k] = calc_auc(logits_e[k], targets_e[k])
        prec_e[k] = calc_precision(logits_e[k], targets_e[k])
        rec_e[k] = calc_recall(logits_e[k], targets_e[k])
    
    if 'final' in acc_e:
        auc_final = calc_auc(logits_e['final'], targets_e['final'])
        if use_cross:
            conf_mat = calc_confusion_matrix(logits_e['final'], targets_e['final'])
            conf_mat_norm = calc_confusion_matrix(logits_e['final'], targets_e['final'], norm='true')
    
    # return res
    res = {}
    res['loss'] = {'balance': total_balance / (iter + 1),
                   'orth': total_orth / (iter + 1),
                   'contrast': total_contrast / (iter + 1)}

    for k in logits_e:
        if k == 'final':
            res['cross'] = [acc_e[k], mAP_e[k], auc_final, f1_e[k], logits_e['final'],]
        else:
            res[k] = [acc_e[k], mAP_e[k], auc_e[k], f1_e[k], logits_e[k]]
    
    # log res
    log('\n')
    for k in acc_e:
        if k == 'final':
            log(f'\tacc: {acc_e[k]}\t mAP: {mAP_e[k]}\t f1: {f1_e[k]}\t f1b: {f1b_e[k]}\t AUC: {auc_final}')
            log(f'\tprecision: {prec_e[k]}')
            if use_cross:
                log(f'\tconf mat: {conf_mat}')
                log(f'\tconf mat norm: {conf_mat_norm}')
            log(f'\tcontrast loss: '+str(res['loss']['contrast']))
        else:
            log(f'\t{k} acc: {acc_e[k]}\t {k} mAP: {mAP_e[k]}\t {k} f1: {f1_e[k]}')
            log(f'\t{k} recall: {rec_e[k]}')
            log(f'\t{k} precision: {prec_e[k]}')
    if lambda_contrast > 0:
        log('\tsimi mat statistics')
        for k in simi_mat_statis:
            log(f'\t {k} min: '+\
                str(simi_mat_statis[k]['min'][:3])+\
                    str(simi_mat_statis[k]['min'][-3:]))
            log(f'\t{k} max: '+\
                str(simi_mat_statis[k]['max'][:3])+\
                    str(simi_mat_statis[k]['max'][-3:]))
    log('\n')
    del total_contrast, total_orth, total_balance, total_ce, total_mse
    return res