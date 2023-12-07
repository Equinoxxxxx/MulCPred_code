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
from _loss_fuctions import SENN_robustness_loss, recon_loss, l1_sparsity_loss
from helpers import makedir
from tools.datasets.TITAN import NUM_CLS_ATOMIC, NUM_CLS_COMPLEX, NUM_CLS_COMMUNICATIVE, NUM_CLS_TRANSPORTING, NUM_CLS_AGE
from tools.gpu_mem_track import MemTracker

def train_test_SENN(model, 
                    dataloader, 
                    optimizer=None,
                    log=print, 
                    device='cuda:0',
                    data_types=['img'],
                    display_logits=True,
                    num_classes=2,
                    multi_label_cross=0, 
                    use_cross=1,
                    use_atomic=0,
                    use_complex=0,
                    use_communicative=0,
                    use_transporting=0,
                    use_age=0,
                    use_robust=1,
                    mask=None,
                    pred_k='final',
                    ):
    # print('0')
    # gpu_tracker = MemTracker()
    start = time.time()
    d_time = 0
    c_time = 0
    is_train = optimizer is not None
    total_ce = 0
    total_recon_loss = 0
    
    # get class weights
    cls_weights_multi = get_cls_weights_multi(model=model,
                                                dataloader=dataloader,
                                                loss_weight='sklearn',
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
    
    # targets and logits for whole epoch
    targets_e = {}
    logits_e = {}
    # print('1')
    # start iteration
    b_end = time.time()
    tbar = tqdm(dataloader, miniters=1)
    for iter, data in enumerate(tbar):
        # print('2')
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
        if use_robust:
            for k in inputs:
                inputs[k].requires_grad_(True)
        
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
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            mse_loss = 0
            if mask is not None:
                output = model(inputs, mask=mask)  # b, num classes
            else:
                output = model(inputs)  # b, num classes
            logits, multi_protos, relevs, recons = output
            # collect targets and logits in batch
            for k in logits:
                if iter == 0:
                    targets_e[k] = targets[k].detach()
                    logits_e[k] = logits[k].detach()
                else:
                    targets_e[k] = torch.cat((targets_e[k], targets[k].detach()), dim=0)
                    logits_e[k] = torch.cat((logits_e[k], logits[k].detach()), dim=0)

            weights = torch.tensor([dataloader.dataset.num_samples/dataloader.dataset.n_nc, 
                                    dataloader.dataset.num_samples/dataloader.dataset.n_c]).float().to(device)
            ce_loss = F.cross_entropy(logits[pred_k], 
                                      targets[pred_k], 
                                      weight=cls_weights_multi[pred_k])
            total_ce += ce_loss.item()

            # calc losses and backward
            if is_train:
                l1 = 0
                mse = 0
                robust_loss = 0
                for k in multi_protos:
                    # print(k)
                    proto = multi_protos[k]
                    relev = relevs[k]
                    recon = recons[k]
                    x = inputs[k]
                    l1 += l1_sparsity_loss(proto)
                    mse += recon_loss(recon, x)
                    if iter%20 == 0 and use_robust:
                        robust_loss += SENN_robustness_loss(x, logits[pred_k], proto, relev)

                loss = ce_loss + 2e-3*robust_loss + mse + 2e-5*l1
                total_recon_loss += mse.item()
                optimizer.zero_grad()
                loss.backward()
                # gpu_tracker.track()
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
            del loss, robust_loss
        torch.cuda.empty_cache()
        
    tbar.close()
    end = time.time()

    # calc metric
    acc_e = {}
    f1_e = {}
    f1b_e = {}
    auc_e = {}
    mAP_e = {}
    prec_e = {}
    rec_e = {}
    for k in logits_e:
        acc_e[k] = calc_acc(logits_e[k], targets_e[k])
        if k == 'final' and (not multi_label_cross):
            f1b_e[k] = calc_f1(logits_e[k], targets_e[k], 'binary')
        f1_e[k] = calc_f1(logits_e[k], targets_e[k])
        auc_e[k] = calc_auc(logits_e[k], targets_e[k])
        mAP_e[k] = calc_mAP(logits_e[k], targets_e[k])
        prec_e[k] = calc_precision(logits_e[k], targets_e[k])
        rec_e[k] = calc_recall(logits_e[k], targets_e[k])
    
    if 'final' in acc_e:
        auc_final = calc_auc(logits_e['final'], targets_e['final'])
        conf_mat = calc_confusion_matrix(logits_e['final'], targets_e['final'])
        conf_mat_norm = calc_confusion_matrix(logits_e['final'], targets_e['final'], norm='true')
    
    # return res
    res = {}
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
            log(f'\tconf mat: {conf_mat}')
            log(f'\tconf mat norm: {conf_mat_norm}')
        else:
            log(f'\t{k} acc: {acc_e[k]}\t {k} mAP: {mAP_e[k]}\t {k} f1: {f1_e[k]}')
            log(f'\t{k} recall: {rec_e[k]}')
            log(f'\t{k} precision: {prec_e[k]}')
    log('\n')

    return res

