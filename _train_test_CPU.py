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
from tools.loss import calc_logsig_loss, margin_entropy_loss, kl_divergence
from tools.contrastive_loss import calc_batch_simi, calc_contrast_loss
from _loss_fuctions import FocalLoss, FocalLoss2, WeightedCrossEntropy, FocalLoss3, calc_orth_loss, calc_orth_loss_fix, calc_balance_loss, focal_tversky
from helpers import makedir

from torchviz import make_dot

def contrast_epoch(model, 
                   epoch,
                    loader, 
                    optimizer=None,
                    log=print, 
                    device='cuda:0',
                    modalities=[],
                    logsig_thresh=100,
                    logsig_loss_eff=0.01,
                    logsig_loss_func='margin',
                    exp_path='',
                    ):
    # torch.autograd.set_detect_anomaly(True)
    start = time.time()
    d_time = 0
    c_time = 0

    # start iteration
    b_end = time.time()
    total_contrast_loss = 0
    total_logsig_loss = 0
    #   intermediate res
    logit_scales = []
    simi_vars_1st_pair = []
    simi_vars_2nd_pair = []
    z_var_mean = {m:[] for m in modalities}
    pool_fs = {m:[] for m in modalities}
    pool_f_grads_min = {m:[] for m in modalities}
    pool_f_grads_max = {m:[] for m in modalities}
    zs = {m:[] for m in modalities}
    z_grads = {m:[] for m in modalities}
    weight_grads_min = []
    weight_grads_max = []
    proj_grads_min = []
    proj_grads_max = []
    tbar = tqdm(loader, miniters=1)
    for iter, data in enumerate(tbar):
        optimizer.zero_grad()
        nan_grads = []
        # load inputs
        inputs = {}
        if 'img' in modalities:
            inputs['img'] = data['ped_imgs'].to(device)
        if 'sk' in modalities:
            inputs['sk'] = data['obs_skeletons'].to(device)
        if 'ctx' in modalities:
            inputs['ctx'] = data['obs_context'].to(device)
        if 'traj' in modalities:
            inputs['traj'] = data['obs_bboxes'].to(device)
        if 'ego' in modalities:
            inputs['ego'] = data['obs_ego'].to(device)
        
        # forward
        b_start = time.time()
        with torch.enable_grad():
            z_dict, mu_dict, logsig_dict, pool_f_dict = model(x_dict=inputs, mode='contrast', log=log)
            protos = model.module.protos if model.module.concept_mode == 'fix_proto' else None
            simi_mats = calc_batch_simi(z_dict, model.module.logit_scale, protos, model.module.bridge_m, model.module.contrast_mode)
            # intermediate vals to check per batch
            logit_scales.append(model.module.logit_scale.detach().item())
            #   simi vars
            simi_vars_1st_pair.append(torch.var(simi_mats[0], dim=1).mean().detach().item())
            simi_vars_2nd_pair.append(torch.var(simi_mats[1], dim=1).mean().detach().item())
            #   z vars
            for m in modalities:
                z_var_mean[m].append(torch.var(z_dict[m][0], dim=0).mean().detach().item())
            # calc loss
            contrast_loss = calc_contrast_loss(simi_mats, 
                                               contrast_mode=model.module.contrast_mode)
            logsig_loss = 0
            for m in logsig_dict:
                if logsig_loss_func == 'margin':
                    logsig_loss += margin_entropy_loss(margin=logsig_thresh, logsigma=logsig_dict[m])
                elif logsig_loss_func == 'kl':
                    logsig_loss += kl_divergence(mu=mu_dict[m], logsigma=logsig_dict[m])
            loss = contrast_loss + logsig_loss_eff * logsig_loss
            
            # grad
            for m in pool_f_dict:
                pool_fs[m].append(pool_f_dict[m].mean().detach().item())
                if torch.cuda.device_count() <= 1:
                    pool_f_grads_min[m].append(torch.autograd.grad(loss, pool_f_dict[m], retain_graph=True)[0].min().detach().item())
                    pool_f_grads_max[m].append(torch.autograd.grad(loss, pool_f_dict[m], retain_graph=True)[0].max().detach().item())
            # back prop
            retain_graph = torch.cuda.device_count() <= 1
            loss.backward(retain_graph=retain_graph)
            
            # # check grads
            # if torch.cuda.device_count() <= 1:
            #     if iter%1 == 0:
            #         cur_lr = optimizer.state_dict()['param_groups'][1]['lr']
            #         log(f'cur iter {iter} cur lr {cur_lr}')
            #         nan_grad_params = []
            #         none_grad_params = []
            #         nan_grad_mus = []
            #         nan_grad_zs = []
            #         nan_grad_simi_rows = []
                    
            #         for m in mu_dict:
            #             cur_grad = torch.autograd.grad(loss, mu_dict[m], retain_graph=True)[0]
            #             for i in range(len(mu_dict[m])):
            #                 has_nan = torch.isnan(cur_grad[i]).any()
            #                 log(f'{m} {i}th sample mu grad: has nan? {has_nan}')
            #                 log(f'\tmu range: {mu_dict[m][i].min().detach().item()}~{mu_dict[m][i].max().detach().item()}')
            #                 log(f'\tgrad range: {cur_grad[i].min().detach().item()}~{cur_grad[i].max().detach().item()}')
            #                 log(f'{m} {i}th sample mu: {mu_dict[m][i].detach()}')
            #                 if has_nan:
            #                     log(f'\tgrad:{cur_grad[i].detach()}')
            #         if model.module.model_dict['img'].uncertainty == 'gaussian':
            #             for m in mu_dict:
            #                 cur_grad = torch.autograd.grad(loss, logsig_dict[m], retain_graph=True)[0]
            #                 for i in range(len(logsig_dict[m])):
            #                     has_nan = torch.isnan(cur_grad[i]).any()
            #                     log(f'{m} {i}th sample logsig grad: has nan? {has_nan}')
            #                     log(f'{m} {i}th sample logsig: {logsig_dict[m][i].detach()}')
            #                     log(f'\tlogsig range: {logsig_dict[m][i].min().detach().item()}~{logsig_dict[m][i].max().detach().item()}')
            #                     log(f'\tgrad range: {cur_grad[i].min().detach().item()}~{cur_grad[i].max().detach().item()}')

            #                     if has_nan:
            #                         log(f'\tgrad:{cur_grad[i].detach()}')
            #         # for m in z_dict:
            #         #     for i in range(len(z_dict[m])):
            #         #         cur_grad = torch.autograd.grad(loss, z_dict[m][i], retain_graph=True)[0]
            #         #         print(cur_grad.shape)
            #         #         for j in range(len(z_dict[m][i])):
            #         #             has_nan = torch.isnan(cur_grad[j]).any()
            #         #             log(f'{m} {j}th sample {i}th z grad: has nan? {has_nan}')
            #         #             log(f'{m} {j}th sample {i}th z: {z_dict[m][i][j].detach()}')
            #         #             log(f'\tgrad: {cur_grad[j].min().detach().item()}~{cur_grad[j].max().detach().item()}')
            #         #             log(f'\tparam: {z_dict[m][i][j].min().detach().item()}~{z_dict[m][i][j].max().detach().item()}')
            #         #             if has_nan:
            #         #                 nan_grad_zs.append(cur_grad[j])
            #         #                 log(f'\tgrad: {cur_grad[j].detach()}')
            #         # log(f'num nan grad zs {len(nan_grad_zs)}')
            #         if model.module.concept_mode == 'fix_proto':
            #             log(f'protos value range: {model.module.protos.min().detach()}~{model.module.protos.max().detach()}')
            #             log(f'protos values: {model.module.protos.detach()}')
            #             log(f'protos grads: {model.module.protos.grad.detach()}')
            #         for i in range(len(simi_mats)):
            #             cur_grad = torch.autograd.grad(loss, simi_mats[i], retain_graph=True)[0].detach()
            #             for j in range(len(cur_grad)):
            #                 has_nan = torch.isnan(cur_grad[j]).any()
            #                 log(f'simi mats {i}th pair {j}th row grad has nan? {has_nan}  Value: {cur_grad[j].min().cpu().numpy()}~{cur_grad[j].max().cpu().numpy()}')
            #                 log(f'\tsimi row {simi_mats[i][j].detach().cpu()}')
            #                 log(f'\tsimi row grad {cur_grad[j].detach().cpu()}')
            #                 if has_nan:
            #                     nan_grad_simi_rows.append(cur_grad[j])
            #         log(f'num nan grad simi rows {len(nan_grad_simi_rows)}')
            #         for n, p in model.module.named_parameters():
            #             if p.grad is not None:
            #                 has_nan = torch.isnan(p.grad).any()
            #                 log(f'{n} grad has nan? {has_nan} Value {p.grad.min().detach().cpu().numpy()}~{p.grad.max().detach().cpu().numpy()} param {p.min().detach().cpu().numpy()}~{p.max().detach().cpu().numpy()}')
            #                 if has_nan:
            #                     nan_grad_params.append(n)
            #             else:
            #                 none_grad_params.append(n)
            #         log(f'num nan grad params {len(nan_grad_params)}')
            #     if iter == 10:
            #         log(exp_path)
            #         raise ValueError

            nn.utils.clip_grad_norm_(model.module.parameters(), norm_type=2., max_norm=5.)
            optimizer.step()
            # weight grad
            weight_grads_min.append(model.module.model_dict['img'].backbone.conv1.weight.grad.detach().min().item())
            weight_grads_max.append(model.module.model_dict['img'].backbone.conv1.weight.grad.detach().max().item())
            # proj grad
            proj_grads_min.append(model.module.model_dict['img'].proj[0].weight.grad.detach().min().item())
            proj_grads_max.append(model.module.model_dict['img'].proj[0].weight.grad.detach().max().item())
            # check nan value
            for name, param in model.module.named_parameters():
                # print('\n',name)
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    nan_grads.append(name)
            # log(f'Num nan grad params: {len(nan_grads)}')

            total_contrast_loss += contrast_loss.detach().cpu()
            if logsig_loss_eff > 0:
                total_logsig_loss += logsig_loss_eff * logsig_loss.detach().cpu()
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
            tbar.set_postfix(display_dict)
        # del inputs
        # del loss
        torch.cuda.empty_cache()
    tbar.close()
    end = time.time()
    # log res
    log('\n')
    log(f'\t {loader.dataset.dataset_name} contrast loss: {total_contrast_loss}\t logsig loss: {total_logsig_loss}')
    #   log intermediate res
    #       logit scale
    log(f'Logit scale per batch: {logit_scales[:3], logit_scales[-1]}')
    #       pool feat per batch
    for m in pool_f_dict:
        log(f'pool feat mean \n {m}: {pool_fs[m][:3], pool_fs[m][-1]}')
    #       pool feat grads per batch
    if torch.cuda.device_count() <= 1:
        for m in pool_f_dict:
            log(f'pool feat mean grads min \n {m}: {pool_f_grads_min[m][:3], pool_f_grads_min[m][-1]}')
            log(f'pool feat mean grads max \n {m}: {pool_f_grads_max[m][:3], pool_f_grads_max[m][-1]}')
    #       var
    # log(f'\nsimi mat row var 1st pair per batch {simi_vars_1st_pair[:3]}')
    # for m in modalities:
    #     log(f'\nz var per batch {m} {z_var_mean[m][0], z_var_mean[m][1], z_var_mean[m][-1]}')
    # simi mat
    log(f'simi mat shape: {len(simi_mats), simi_mats[0].shape})')
    log(f'Simi mats last batch 1st 2 pairs 3 sample: {simi_mats[0][:3, :3].detach().cpu().numpy()} \n{simi_mats[1][:3, :3].detach().cpu().numpy()}')
    log(f'conv weight grad min\n {weight_grads_min[:3], weight_grads_min[-1]}')
    log(f'conv weight grad max\n {weight_grads_max[:3], weight_grads_max[-1]}')
    log(f'proj weight grad min\n {proj_grads_min[:3], proj_grads_min[-1]}')
    log(f'proj weight grad max\n {proj_grads_max[:3], proj_grads_max[-1]}')
    log('\n')
    # return
    res = {
        'contrast_loss': total_contrast_loss,
        'logsig_loss': total_logsig_loss,
    }
    return res

def train_test_epoch(
                    model, 
                    epoch,
                    loader,
                    loss_func='weighted_ce',
                    optimizer=None,
                    log=print, 
                    device='cuda:0',
                    modalities=[],
                    contrast_eff=0,
                    logsig_thresh=100,
                    logsig_loss_eff=0.01,
                    train_or_test='train',
                    logsig_loss_func='margin',
                    exp_path='',
                    ):
    start = time.time()
    d_time = 0
    c_time = 0
    total_ce = 0
    # get class weights
    cls_weights_multi = get_cls_weights_multi(model=model,
                                                    dataloader=loader,
                                                    loss_weight='sklearn',
                                                    device=device,
                                                    use_cross=True,
                                                    )
    for k in cls_weights_multi:
        if cls_weights_multi[k] is not None:
            log(k + ' class weights: ' + str(cls_weights_multi[k]))
    focal = FocalLoss3()
    # targets and logits for whole epoch
    targets_e = {}
    logits_e = {}
    # start iteration
    b_end = time.time()
    total_contrast_loss = 0
    total_logsig_loss = 0
    pool_fs = {m:[] for m in modalities}
    pool_f_grads_min = {m:[] for m in modalities}
    pool_f_grads_max = {m:[] for m in modalities}
    zs = {m:[] for m in modalities}
    z_grads = {m:[] for m in modalities}
    weight_grads_min = []
    weight_grads_max = []
    proj_grads_min = []
    proj_grads_max = []
    simi_vars_1st_pair = []
    simi_vars_2nd_pair = []
    z_var_mean = {m:[] for m in modalities}
    tbar = tqdm(loader, miniters=1)
    # loader.sampler.set_epoch(epoch)
    for iter, data in enumerate(tbar):
        # load inputs
        inputs = {}
        if 'img' in modalities:
            inputs['img'] = data['ped_imgs'].to(device)
        if 'sk' in modalities:
            inputs['sk'] = data['obs_skeletons'].to(device)
        if 'ctx' in modalities:
            inputs['ctx'] = data['obs_context'].to(device)
        if 'traj' in modalities:
            inputs['traj'] = data['obs_bboxes'].to(device)
        if 'ego' in modalities:
            inputs['ego'] = data['obs_ego'].to(device)
        
        # load gt
        targets = {}
        targets['final'] = data['pred_intent'].to(device).view(-1) # idx, not one hot
        if loader.dataset.dataset_name == 'TITAN':
            targets['atomic'] = data['atomic_actions'].to(device).view(-1)
            targets['complex'] = data['complex_context'].to(device).view(-1)
            targets['communicative'] = data['communicative'].to(device).view(-1)
            targets['transporting'] = data['transporting'].to(device).view(-1)
            targets['age'] = data['age'].to(device).view(-1)
        
        # forward
        b_start = time.time()
        grad_req = torch.enable_grad() if train_or_test == 'train' else torch.no_grad()
        with grad_req:
            if contrast_eff > 0:
                logits, simis, z_dict, mu_dict, logsig_dict, pool_f_dict = model(x_dict=inputs, mode='train', log=log)
                protos = model.module.protos if model.module.concept_mode == 'fix_proto' else None
                simi_mats = calc_batch_simi(z_dict, model.module.logit_scale, protos, model.module.bridge_m, model.module.contrast_mode)
            else:
                logits, simis, z_dict, mu_dict, logsig_dict, pool_f_dict = model(x_dict=inputs, mode='train', log=log)
            # collect targets and logits in batch
            for k in logits:
                if iter == 0:
                    targets_e[k] = targets[k].detach()
                    logits_e[k] = logits[k].detach()
                else:
                    targets_e[k] = torch.cat((targets_e[k], targets[k].detach()), dim=0)
                    logits_e[k] = torch.cat((logits_e[k], logits[k].detach()), dim=0)
            if train_or_test == 'train':
                # contrast loss
                contrast_loss = 0
                if train_or_test == 'train' and contrast_eff > 0:
                    simi_mats = model.module.contrast(z_dict)
                    contrast_loss = calc_contrast_loss(simi_mats)
                # logsig loss
                logsig_loss = 0
                if contrast_eff > 0 and logsig_loss_eff > 0:
                    for m in logsig_dict:
                        if logsig_loss_func == 'margin':
                            logsig_loss += margin_entropy_loss(margin=logsig_thresh, logsigma=logsig_dict[m])
                        elif logsig_loss_func == 'kl':
                            logsig_loss += kl_divergence(mu=mu_dict[m], logsigma=logsig_dict[m])
                        else:
                            raise NotImplementedError(logsig_loss)
                # ce loss
                ce_dict = {}
                if loss_func == 'weighted_ce':
                    for k in logits:
                        ce_dict[k] = F.cross_entropy(logits[k], targets[k], weight=cls_weights_multi[k])
                        ce_dict[k] = focal(logits[k], targets[k], weight=cls_weights_multi[k])
                # combine
                loss = contrast_eff*contrast_loss + logsig_loss_eff*logsig_loss
                for k in ce_dict:
                    # print(k)
                    loss += ce_dict[k]
                
                # #
                # dot = make_dot(z_dict['img'].mean(), dict(model.module.model_dict['img'].proj.named_parameters()))
                # dot.render(filename="backward_graph", format="png")
                # raise NotImplementedError()
                # grad
                for m in pool_f_dict:
                    # zs[m].append(z_dict[m][0].detach().mean().item())
                    pool_fs[m].append(pool_f_dict[m].detach().mean().item())
                    if torch.cuda.device_count() <= 1:
                        pool_f_grads_min[m].append(torch.autograd.grad(loss, pool_f_dict[m], retain_graph=True)[0].min().detach().item())
                        pool_f_grads_max[m].append(torch.autograd.grad(loss, pool_f_dict[m], retain_graph=True)[0].max().detach().item())
                optimizer.zero_grad()
                retain_graph = torch.cuda.device_count() <= 1
                loss.backward(retain_graph=retain_graph)
                # check grads
                if torch.cuda.device_count() <= 1:
                    if iter%1 == 0:
                        cur_lr = optimizer.state_dict()['param_groups'][1]['lr']
                        log(f'cur iter {iter} cur lr {cur_lr}')
                        nan_grad_params = []
                        none_grad_params = []
                        nan_grad_mus = []
                        nan_grad_zs = []
                        nan_grad_simi_rows = []
                        for n, p in model.module.named_parameters():
                            if p.grad is not None:
                                has_nan = torch.isnan(p.grad).any()
                                log(f'{n} grad has nan? {has_nan} Value {p.grad.min().detach().cpu().numpy()}~{p.grad.max().detach().cpu().numpy()} param {p.min().detach().cpu().numpy()}~{p.max().detach().cpu().numpy()}')
                                if has_nan:
                                    nan_grad_params.append(n)
                            else:
                                none_grad_params.append(n)
                        log(f'num nan grad params {len(nan_grad_params)}')
                        for m in mu_dict:
                            cur_grad = torch.autograd.grad(loss, mu_dict[m], retain_graph=True)[0]
                            for i in range(len(mu_dict[m])):
                                has_nan = torch.isnan(cur_grad[i]).any()
                                log(f'{m} {i}th sample mu grad: has nan? {has_nan}')
                                log(f'{m} {i}th sample mu: {mu_dict[m][i].detach()}')
                                log(f'\tgrad: {cur_grad[i].min().detach().item()}~{cur_grad[i].max().detach().item()}')
                                log(f'\tparam: {mu_dict[m][i].min().detach().item()}~{mu_dict[m][i].max().detach().item()}')
                                if has_nan:
                                    log(f'\tgrad:{cur_grad[i].detach()}')
                        if model.module.model_dict['img'].uncertainty == 'gaussian':
                            for m in mu_dict:
                                cur_grad = torch.autograd.grad(loss, logsig_dict[m], retain_graph=True)[0]
                                for i in range(len(logsig_dict[m])):
                                    has_nan = torch.isnan(cur_grad[i]).any()
                                    log(f'{m} {i}th sample logsig grad: has nan? {has_nan}')
                                    log(f'{m} {i}th sample logsig: {logsig_dict[m][i].detach()}')
                                    log(f'\tgrad: {cur_grad[i].min().detach().item()}~{cur_grad[i].max().detach().item()}')
                                    log(f'\tparam: {logsig_dict[m][i].min().detach().item()}~{logsig_dict[m][i].max().detach().item()}')
                                    if has_nan:
                                        log(f'\tgrad:{cur_grad[i].detach()}')
                        # for m in z_dict:
                        #     for i in range(len(z_dict[m])):
                        #         cur_grad = torch.autograd.grad(loss, z_dict[m][i], retain_graph=True)[0]
                        #         print(cur_grad.shape)
                        #         for j in range(len(z_dict[m][i])):
                        #             has_nan = torch.isnan(cur_grad[j]).any()
                        #             log(f'{m} {j}th sample {i}th z grad: has nan? {has_nan}')
                        #             log(f'{m} {j}th sample {i}th z: {z_dict[m][i][j].detach()}')
                        #             log(f'\tgrad: {cur_grad[j].min().detach().item()}~{cur_grad[j].max().detach().item()}')
                        #             log(f'\tparam: {z_dict[m][i][j].min().detach().item()}~{z_dict[m][i][j].max().detach().item()}')
                        #             if has_nan:
                        #                 nan_grad_zs.append(cur_grad[j])
                        #                 log(f'\tgrad: {cur_grad[j].detach()}')
                        # log(f'num nan grad zs {len(nan_grad_zs)}')
                        if model.module.concept_mode == 'fix_proto':
                            log(f'protos value range: {model.module.protos.min().detach()}~{model.module.protos.max().detach()}')
                            log(f'protos values: {model.module.protos.detach()}')
                            log(f'protos grads: {model.module.protos.grad.detach()}')
                        # for i in range(len(simi_mats)):
                        #     cur_grad = torch.autograd.grad(loss, simi_mats[i], retain_graph=True)[0].detach()
                        #     for j in range(len(cur_grad)):
                        #         has_nan = torch.isnan(cur_grad[j]).any()
                        #         log(f'simi mats {i}th pair {j}th row grad has nan? {has_nan}  Value: {cur_grad[j].min().cpu().numpy()}~{cur_grad[j].max().cpu().numpy()}')
                        #         log(f'\tsimi row {simi_mats[i][j].detach().cpu()}')
                        #         log(f'\tsimi row grad {cur_grad[j].detach().cpu()}')
                        #         if has_nan:
                        #             nan_grad_simi_rows.append(cur_grad[j])
                        # log(f'num nan grad simi rows {len(nan_grad_simi_rows)}')
                    if iter == 10:
                        log(exp_path)
                        raise ValueError
                nn.utils.clip_grad_norm_(model.module.parameters(), norm_type=2., max_norm=10.)
                optimizer.step()
                total_contrast_loss += contrast_loss
                total_logsig_loss += logsig_loss
                # weight grad
                weight_grads_min.append(model.module.model_dict['img'].backbone.conv1.weight.grad.detach().min().item())
                weight_grads_max.append(model.module.model_dict['img'].backbone.conv1.weight.grad.detach().max().item())
                # proj grad
                proj_grads_min.append(model.module.model_dict['img'].proj[0].weight.grad.detach().min().item())
                proj_grads_max.append(model.module.model_dict['img'].proj[0].weight.grad.detach().max().item())
            # display
            data_prepare_time = b_start - b_end
            b_end = time.time()
            computing_time = b_end - b_start
            d_time += data_prepare_time
            c_time += computing_time
            display_dict = {'data': data_prepare_time, 
                            'compute': computing_time,
                            'd all': d_time,
                            'c all': c_time,
                            }
            if 'final' in logits:
                with torch.no_grad():
                    mean_logit = torch.mean(logits['final'].detach(), dim=0)
                    display_dict['logit'] = [round(logits['final'][0, 0].item(), 4), round(logits['final'][0, 1].item(), 4)]
                    display_dict['avg logit'] = [round(mean_logit[0].item(), 4), round(mean_logit[1].item(), 4)]
            tbar.set_postfix(display_dict)
        del inputs
        if train_or_test == 'train':
            del loss
        torch.cuda.empty_cache()
    tbar.close()
    end = time.time()

    # calc metric
    acc_e = {}
    f1_e = {}
    f1b_e = {}
    mAP_e = {}
    prec_e = {}
    rec_e = {}
    for k in logits_e:
        acc_e[k] = calc_acc(logits_e[k], targets_e[k])
        if k == 'final':
            f1b_e[k] = calc_f1(logits_e[k], targets_e[k], 'binary')
        f1_e[k] = calc_f1(logits_e[k], targets_e[k])
        mAP_e[k] = calc_mAP(logits_e[k], targets_e[k])
        prec_e[k] = calc_precision(logits_e[k], targets_e[k])
        rec_e[k] = calc_recall(logits_e[k], targets_e[k])
    if 'final' in acc_e:
        auc_final = calc_auc(logits_e['final'], targets_e['final'])
        conf_mat = calc_confusion_matrix(logits_e['final'], targets_e['final'])
        conf_mat_norm = calc_confusion_matrix(logits_e['final'], targets_e['final'], norm='true')
    # return res
    res = {k:{} for k in logits_e}
    for k in logits_e:
        if k == 'final':
            res['final'] = {
                'acc': acc_e[k],
                'map': mAP_e[k],
                'f1': f1_e[k],
                'auc': auc_final,
                'logits': logits_e['final'].detach().cpu().numpy(),
                'contrast_loss': total_contrast_loss,
                'logsig_loss': total_logsig_loss,
            }
        else:
            res[k] = {
                'acc': acc_e[k],
                'map': mAP_e[k],
                'logits': logits_e[k]
            }
    # log res
    log(f'\n {loader.dataset.dataset_name}')
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
    if 'proto' in model.module.concept_mode:
        log(f'\t1st 2 proto 10 dim: {model.module.protos[:2, :10]}')
    if train_or_test == 'train':
        for m in pool_f_dict:
            log(f'pool feat mean \n {m}: {pool_fs[m][:3], pool_fs[m][-1]}')
        #       pool feat grads per batch
        if torch.cuda.device_count() <= 1:
            for m in pool_f_dict:
                log(f'pool feat mean grads min\n {m}: {pool_f_grads_min[m][:3], pool_f_grads_min[m][-1]}')
                log(f'pool feat mean grads max\n {m}: {pool_f_grads_max[m][:3], pool_f_grads_max[m][-1]}')
        #       weight grads per batch
        log(f'conv weight grad min\n {weight_grads_min[:3], weight_grads_min[-1]}')
        log(f'conv weight grad max\n {weight_grads_max[:3], weight_grads_max[-1]}')
        log(f'proj weight grad min\n {proj_grads_min[:3], proj_grads_min[-1]}')
        log(f'proj weight grad max\n {proj_grads_max[:3], proj_grads_max[-1]}')
    log('\n')
    return res