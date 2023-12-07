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
from utils import seg_context_batch3d, idx2onehot, calc_auc, calc_f1, calc_acc, calc_recall, calc_mAP, cls_weights, calc_n_samples_cls
from _loss_fuctions import FocalLoss, FocalLoss2, WeightedCrossEntropy, FocalLoss3
from helpers import makedir
from .tools.datasets.TITAN import NUM_CLS_ATOMIC, NUM_CLS_COMPLEX, NUM_CLS_COMMUNICATIVE, NUM_CLS_TRANSPORTING, NUM_CLS_AGE


def train_test(model, dataloader, optimizer=None, model_name='SLE',
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
                mask=None
                ):
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_c_correct = 0
    n_nc_correct = 0
    n_c_pred = 0
    n_c_gt = 0
    n_nc_pred = 0
    n_nc_gt = 0
    n_batches = 0
    total_ce = 0
    
    total_mse = 0
    logits_epoch = []
    pred_idx_epoch = []
    label_idx_epoch = []

    total_atomic_ce = 0
    atomic_logits_epoch = []
    atomic_pred_idx_epoch = []
    atomic_label_idx_epoch = []

    total_complex_ce = 0
    complex_logits_epoch = []
    complex_pred_idx_epoch = []
    complex_label_idx_epoch = []

    total_communicative_ce = 0
    communicative_logits_epoch = []
    communicative_pred_idx_epoch = []
    communicative_label_idx_epoch = []

    total_transporting_ce = 0
    transporting_logits_epoch = []
    transporting_pred_idx_epoch = []
    transporting_label_idx_epoch = []

    total_age_ce = 0
    age_logits_epoch = []
    age_pred_idx_epoch = []
    age_label_idx_epoch = []
    
    n_all = dataloader.dataset.num_samples
    n_c = dataloader.dataset.n_c
    n_nc = dataloader.dataset.n_nc

    # calc class weights
    if loss_weight == 'trainable':
        weight = F.softmax(model.module.class_weights, dim=-1)
        log('class weight: '+ str(weight))
        if use_atomic:
            atomic_weight = F.softmax(model.module.atomic_weights, dim=-1)
            log('atomic weight: '+ str(atomic_weight))
        if use_complex:
            complex_weight = F.softmax(model.module.complex_weights, dim=-1)
            log('complex weight: '+ str(complex_weight))
        if use_communicative:
            communicative_weight = F.softmax(model.module.communicative_weights, dim=-1)
            log('communicative weight: '+ str(communicative_weight))
        if use_transporting:
            transporting_weight = F.softmax(model.module.transporting_weights, dim=-1)
            log('transporting weight: '+ str(transporting_weight))
        if use_age:
            age_weight = F.softmax(model.module.age_weights, dim=-1)
            log('age weight: '+ str(age_weight))
    else:
        weight = cls_weights([n_nc, n_c], loss_weight, device=device)
        if multi_label_cross and dataloader.dataset.dataset_name == 'TITAN':
            num_samples_cls = dataloader.dataset.num_samples_cls
            weight = cls_weights(num_samples_cls=num_samples_cls, loss_weight=loss_weight, device=device)
        log('class weight: '+ str(weight))
        if use_atomic and dataloader.dataset.dataset_name == 'TITAN':
            atomic_weight = cls_weights(dataloader.dataset.num_samples_atomic, loss_weight, device=device)
            log('atomic weight: '+ str(atomic_weight))
        if use_complex and dataloader.dataset.dataset_name == 'TITAN':
            complex_weight = cls_weights(dataloader.dataset.num_samples_complex, loss_weight, device=device)
            log('complex weight: '+ str(complex_weight))
        if use_communicative and dataloader.dataset.dataset_name == 'TITAN':
            communicative_weight = cls_weights(dataloader.dataset.num_samples_communicative, loss_weight, device=device)
            log('communicative weight: '+ str(communicative_weight))
        if use_transporting and dataloader.dataset.dataset_name == 'TITAN':
            transporting_weight = cls_weights(dataloader.dataset.num_samples_transporting, loss_weight, device=device)
            log('transporting weight: '+ str(transporting_weight))
        if use_age and dataloader.dataset.dataset_name == 'TITAN':
            age_weight = cls_weights(dataloader.dataset.num_samples_age, loss_weight, device=device)
            log('age weight: '+ str(age_weight))
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
        focal_loss = FocalLoss3()
    elif loss_func == 'weighted_ce' and loss_weight == 'trainable':
        weighted_ce = WeightedCrossEntropy()
    print(loss_func)
    # start iteration
    b_end = time.time()
    tbar = tqdm(dataloader, miniters=1)
    for i, data in enumerate(tbar):
        # trainable class weights
        if loss_weight == 'trainable' and i > 0:
            weight = F.softmax(model.module.class_weights, dim=-1)
            if use_atomic:
                atomic_weight = F.softmax(model.module.atomic_weights, dim=-1)
            if use_complex:
                complex_weight = F.softmax(model.module.complex_weights, dim=-1)
            if use_communicative:
                communicative_weight = F.softmax(model.module.communicative_weights, dim=-1)
            if use_transporting:
                transporting_weight = F.softmax(model.module.transporting_weights, dim=-1)
            if use_age:
                age_weight = F.softmax(model.module.age_weights, dim=-1)

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
        gt_intent = data['pred_intent'].view(-1) # idx, not one hot
        target = gt_intent.to(device)
        target_onehot = idx2onehot(target, num_cls=num_classes).to(device)  # b, num_classes
        if dataloader.dataset.dataset_name == 'TITAN':
            atomic_target = data['atomic_actions'].to(device).view(-1)
            atomic_target_onehot = idx2onehot(atomic_target, num_cls=NUM_CLS_ATOMIC).to(device)
            complex_target = data['complex_context'].to(device).view(-1)
            complex_target_onehot = idx2onehot(complex_target, num_cls=NUM_CLS_COMPLEX).to(device)
            communicative_target = data['communicative'].to(device).view(-1)
            communicative_target_onehot = idx2onehot(communicative_target, num_cls=NUM_CLS_COMMUNICATIVE).to(device)
            transporting_target = data['transporting'].to(device).view(-1)
            transporting_target_onehot = idx2onehot(transporting_target, num_cls=NUM_CLS_TRANSPORTING).to(device)
            age_target = data['age'].to(device).view(-1)
            age_target_onehot = idx2onehot(age_target, num_cls=NUM_CLS_AGE).to(device)
        gt_traj = None
        
        # forward
        b_start = time.time()
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            orth_loss = 0
            mse_loss = 0
            # get predictions
            if model_name == 'SLE':
                if 'pred_traj' in data_types:
                    gt_traj = data['pred_bboxes'].to(device)
                    if mask is not None:
                        logits, multi_protos, traj_pred = model(inputs, mask=mask)
                    else:
                        logits, multi_protos, traj_pred = model(inputs)
                    mse_loss = mse(traj_pred, gt_traj)
                else:
                    if mask is not None:
                        logits, multi_protos = model(inputs, mask=mask)  # b, num classes
                    else:
                        logits, multi_protos = model(inputs)  # b, num classes
            else:
                if 'pred_traj' in data_types:
                    gt_traj = data['pred_bboxes'].to(device)
                    logits = model(inputs)
                    mse_loss = mse(traj_pred, gt_traj)
                else:
                    logits = model(inputs)  # b, num classes
                    
            if use_cross:
                final_logits = logits['final']  # b, num classes
            if use_atomic:
                atomic_logits = logits['atomic']
            if use_complex:
                complex_logits = logits['complex']
                # complex_logits_norm = F.softmax(complex_logits, dim=-1)
                # print('complex logits size:', complex_logits.size(), complex_logits_norm.size())
            if use_communicative:
                communicative_logits = logits['communicative']
            if use_transporting:
                transporting_logits = logits['transporting']
            if use_age:
                age_logits = logits['age']

            # calc orth loss
            if orth_type > 0 and model_name == 'SLE':
                if model.module.traj_model.simi_func not in ('fix_proto1', 'fix_proto2'):
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
            
            # calc cross entropy
            if loss_func == 'focal2':
                if use_cross:
                    ce_loss = focal_loss(final_logits, target)
                if use_atomic:
                    atomic_ce_loss = focal_loss(atomic_logits, atomic_target)
                if use_complex:
                    complex_ce_loss = focal_loss(complex_logits, complex_target)
                if use_communicative:
                    communicative_ce_loss = focal_loss(communicative_logits, communicative_target)
                if use_transporting:
                    transporting_ce_loss = focal_loss(transporting_logits, transporting_target)
                if use_age:
                    age_ce_loss = focal_loss(age_logits, age_target)
            elif loss_func == 'focal3':
                if use_cross:
                    ce_loss = focal_loss(final_logits, target_onehot, weight=weight)
                if use_atomic:
                    atomic_ce_loss = focal_loss(atomic_logits, atomic_target_onehot, weight=atomic_weight)
                if use_complex:
                    complex_ce_loss = focal_loss(complex_logits, complex_target_onehot, weight=complex_weight)
                if use_communicative:
                    communicative_ce_loss = focal_loss(communicative_logits, communicative_target_onehot, weight=communicative_weight)
                if use_transporting:
                    transporting_ce_loss = focal_loss(transporting_logits, transporting_target_onehot, weight=transporting_weight)
                if use_age:
                    age_ce_loss = focal_loss(age_logits, age_target_onehot, weight=age_weight)
            elif loss_func == 'weighted_ce' or loss_func == 'm_task_ce':
                if loss_weight == 'trainable':
                    if use_cross:
                        ce_loss = weighted_ce(final_logits, target, weight=weight)
                    if use_atomic:
                        atomic_ce_loss = weighted_ce(atomic_logits, atomic_target, weight=atomic_weight)
                    if use_complex:
                        complex_ce_loss = weighted_ce(complex_logits, complex_target, weight=complex_weight)
                    if use_communicative:
                        communicative_ce_loss = weighted_ce(communicative_logits, communicative_target, weight=communicative_weight)
                    if use_transporting:
                        transporting_ce_loss = weighted_ce(transporting_logits, transporting_target, weight=transporting_weight)
                    if use_age:
                        age_ce_loss = weighted_ce(age_logits, age_target, weight=age_weight)
                else:
                    if use_cross:
                        if loss_weight_batch:
                            n_cls = 13 if multi_label_cross else 2
                            num_samples_cls = calc_n_samples_cls(target, n_cls=n_cls)
                            weight = cls_weights(num_samples_cls, loss_weight, device=device)
                        ce_loss = F.cross_entropy(final_logits, target, weight=weight)
                        if loss_func == 'm_task_ce':
                            ce_loss = ce_loss * torch.exp(-model.module.logs2) + 0.5 * model.module.logs2
                    if use_atomic:
                        if loss_weight_batch:
                            n_cls = NUM_CLS_ATOMIC
                            num_samples_cls = calc_n_samples_cls(atomic_target, n_cls=n_cls)
                            atomic_weight = cls_weights(num_samples_cls, loss_weight, device=device)
                        atomic_ce_loss = F.cross_entropy(atomic_logits, atomic_target, weight=atomic_weight)
                        if loss_func == 'm_task_ce':
                            atomic_ce_loss = atomic_ce_loss * torch.exp(-model.module.atomic_logs2) + 0.5 * model.module.atomic_logs2
                    if use_complex:
                        if loss_weight_batch:
                            n_cls = NUM_CLS_COMPLEX
                            num_samples_cls = calc_n_samples_cls(complex_target, n_cls=n_cls)
                            complex_weight = cls_weights(num_samples_cls, loss_weight, device=device)
                        complex_ce_loss = F.cross_entropy(complex_logits, complex_target, weight=complex_weight)
                        if loss_func == 'm_task_ce':
                            complex_ce_loss = complex_ce_loss * torch.exp(-model.module.complex_logs2) + 0.5 * model.module.complex_logs2
                    if use_communicative:
                        if loss_weight_batch:
                            n_cls = NUM_CLS_COMMUNICATIVE
                            num_samples_cls = calc_n_samples_cls(communicative_target, n_cls=n_cls)
                            communicative_weight = cls_weights(num_samples_cls, loss_weight, device=device)
                        communicative_ce_loss = F.cross_entropy(communicative_logits, communicative_target, weight=communicative_weight)
                        if loss_func == 'm_task_ce':
                            communicative_ce_loss = communicative_ce_loss * torch.exp(-model.module.communicative_logs2) + 0.5 * model.module.communicative_logs2
                    if use_transporting:
                        if loss_weight_batch:
                            n_cls = NUM_CLS_TRANSPORTING
                            num_samples_cls = calc_n_samples_cls(transporting_target, n_cls=n_cls)
                            transporting_weight = cls_weights(num_samples_cls, loss_weight, device=device)
                        transporting_ce_loss = F.cross_entropy(transporting_logits, transporting_target, weight=transporting_weight)
                        if loss_func == 'm_task_ce':
                            transporting_ce_loss = transporting_ce_loss * torch.exp(-model.module.transporting_logs2) + 0.5 * model.module.transporting_logs2
                    if use_age:
                        if loss_weight_batch:
                            n_cls = NUM_CLS_AGE
                            num_samples_cls = calc_n_samples_cls(age_target, n_cls=n_cls)
                            age_weight = cls_weights(num_samples_cls, loss_weight, device=device)
                        age_ce_loss = F.cross_entropy(age_logits, age_target, weight=age_weight)
                        if loss_func == 'm_task_ce':
                            age_ce_loss = age_ce_loss * torch.exp(-model.module.age_logs2) + 0.5 * model.module.age_logs2
            elif loss_func == 'bcewithlogits':
                pos_weight=torch.tensor([n_nc / n_c]).float().to(device)
                ce_loss = F.binary_cross_entropy_with_logits(final_logits, target_onehot.float(), pos_weight=None)
            elif loss_func == 'bce':
                pos_weight=torch.tensor([n_nc / n_c]).float().to(device)
                ce_loss = F.binary_cross_entropy(final_logits, target_onehot.float())
            else:
                ce_loss = 0
                if use_cross:
                    ce_loss = F.cross_entropy(final_logits, target)
                if use_atomic:
                    atomic_ce_loss = F.cross_entropy(atomic_logits, atomic_target)
                if use_complex:
                    complex_ce_loss = F.cross_entropy(complex_logits, complex_target)
                if use_communicative:
                    communicative_ce_loss = F.cross_entropy(communicative_logits, communicative_target)
                if use_transporting:
                    transporting_ce_loss = F.cross_entropy(transporting_logits, transporting_target)
            
            # turn logits into idx
            if use_cross:
                if num_classes > 1:
                    _, predicted = torch.max(final_logits.detach(), 1)  # 
                else:
                    if loss_func == 'bce':
                        predicted = final_logits.detach() > 0.5
                    else:
                        predicted = F.sigmoid(final_logits.detach()) > 0.5
                    predicted = predicted.int().squeeze(1)
                    # print('predicted', predicted)
            if use_atomic:
                _, atomic_predicted = torch.max(atomic_logits.detach(), 1)
            if use_complex:
                _, complex_predicted = torch.max(complex_logits.detach(), 1)
            if use_communicative:
                _, communicative_predicted = torch.max(communicative_logits.detach(), 1)
            if use_transporting:
                _, transporting_predicted = torch.max(transporting_logits.detach(), 1)
            if use_age:
                _, age_predicted = torch.max(age_logits.detach(), 1)

            # calc correct
            n_examples += target.size(0)
            n_batches += 1
            if use_cross:
                if multi_label_cross:
                    for j in range(len(target)):
                        if (predicted[j] == 0 or predicted[j] == 1) and (target[j] == 0 or target[j] == 1):
                            n_c_correct += 1
                            n_correct += 1
                        elif (predicted[j] != 0 and predicted[j] != 1) and (target[j] != 0 and target[j] != 1):
                            n_nc_correct += 1
                            n_correct += 1
                    n_c_pred += ((predicted == 0) | (predicted == 1)).sum().item()
                    n_nc_pred += ((predicted != 0) & (predicted != 1)).sum().item()
                    n_c_gt += ((target == 0) | (target == 1)).sum().item()
                    n_nc_gt += ((target != 0) & (target != 1)).sum().item()
                else:
                    correct = predicted == target
                    for j in range(len(target)):
                        if correct[j] and (target[j] == 1):
                            n_c_correct += 1
                    n_correct += correct.sum().item()
                    n_c_pred += (predicted == 1).sum().item()
                    n_nc_pred += (predicted == 0).sum().item()
                    n_c_gt += (target == 1).sum().item()
                    n_nc_gt += (target == 0).sum().item()
            
            
            if 'pred_traj' in data_types:
                total_mse += mse_loss.item()

            if use_cross:
                total_ce += ce_loss.item()
                logits_epoch.append(final_logits.detach())
                pred_idx_epoch.append(predicted)
                label_idx_epoch.append(gt_intent)
            if use_atomic:
                total_atomic_ce += atomic_ce_loss.item()
                atomic_logits_epoch.append(atomic_logits.detach())
                atomic_pred_idx_epoch.append(atomic_predicted)
                atomic_label_idx_epoch.append(atomic_target.cpu())
            if use_complex:
                total_complex_ce += complex_ce_loss.item()
                complex_logits_epoch.append(complex_logits.detach())
                complex_pred_idx_epoch.append(complex_predicted)
                complex_label_idx_epoch.append(complex_target.cpu())
            if use_communicative:
                total_communicative_ce += communicative_ce_loss.item()
                communicative_logits_epoch.append(communicative_logits.detach())
                communicative_pred_idx_epoch.append(communicative_predicted)
                communicative_label_idx_epoch.append(communicative_target.cpu())
            if use_transporting:
                total_transporting_ce += transporting_ce_loss.item()
                transporting_logits_epoch.append(transporting_logits.detach())
                transporting_pred_idx_epoch.append(transporting_predicted)
                transporting_label_idx_epoch.append(transporting_target.cpu())
            if use_age:
                total_age_ce += age_ce_loss.item()
                age_logits_epoch.append(age_logits.detach())
                age_pred_idx_epoch.append(age_predicted)
                age_label_idx_epoch.append(age_target.cpu())
        
        # compute gradient and backward
        if is_train:
            loss = mse_loss + 0.01*orth_loss
            if use_cross:
                loss += ce_loss
            if use_atomic:
                loss += atomic_ce_loss
            if use_complex:
                loss += complex_ce_loss
            if use_communicative:
                loss += communicative_ce_loss
            if use_transporting:
                loss += transporting_ce_loss
            if use_age:
                loss += age_ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        data_prepare_time = b_start - b_end
        b_end = time.time()
        computing_time = b_end - b_start
        display_dict = {
                        'data': data_prepare_time, 
                        'compute': computing_time}
        if use_cross:
            with torch.no_grad():
                mean_logit = torch.mean(final_logits, dim=0)
            if display_logits:
                if num_classes == 2:
                    display_dict['logit'] = [round(final_logits[0, 0].item(), 4), round(final_logits[0, 1].item(), 4)]
                    display_dict['avg logit'] = [round(mean_logit[0].item(), 4), round(mean_logit[1].item(), 4)]
                elif num_classes == 1:
                    display_dict['logit'] = [round(final_logits[0, 0].item(), 4)]
                    display_dict['avg logit'] = [round(mean_logit[0].item(), 4)]
        tbar.set_postfix(display_dict)

        del data
        del inputs

    tbar.close()
    end = time.time()

    if use_cross:
        pred_idx_epoch = torch.concat(pred_idx_epoch, dim=0)  # output idx: B,
        logits_epoch = torch.concat(logits_epoch, dim=0)  # output one hot: B, num_class
        label_idx_epoch = torch.concat(label_idx_epoch, dim=0)  # gt idx: B,
        label_onehot_epoch = idx2onehot(label_idx_epoch, num_classes)  # gt one hot: B, num_class
        if multi_label_cross:
            f1_average = 'macro'
            auc_average = 'macro'
        else:
            f1_average = 'binary'
            auc_average = 'binary'

        auc = calc_auc(logits_epoch, label_onehot_epoch, average=auc_average)
        f1 = calc_f1(pred_idx_epoch, label_idx_epoch, average=f1_average)
        recall = calc_recall(pred_idx_epoch, label_idx_epoch)  # 
        mAP = calc_mAP(logits_epoch, label_onehot_epoch)
    if use_atomic:
        atomic_pred_idx_epoch = torch.concat(atomic_pred_idx_epoch, dim=0)  # output idx
        atomic_logits_epoch = torch.concat(atomic_logits_epoch, dim=0)  # output one hot
        atomic_label_idx_epoch = torch.concat(atomic_label_idx_epoch, dim=0)  # gt idx
        atomic_label_onehot_epoch = idx2onehot(atomic_label_idx_epoch, NUM_CLS_ATOMIC)  # gt one hot
        atomic_acc = calc_acc(atomic_pred_idx_epoch, atomic_label_idx_epoch)
        atomic_rc = calc_recall(atomic_pred_idx_epoch, atomic_label_idx_epoch)
        atomic_f1 = calc_f1(atomic_pred_idx_epoch, atomic_label_idx_epoch)
        atomic_mAP = calc_mAP(atomic_logits_epoch, atomic_label_onehot_epoch)
    if use_complex:
        complex_pred_idx_epoch = torch.concat(complex_pred_idx_epoch, dim=0)  # output idx
        complex_logits_epoch = torch.concat(complex_logits_epoch, dim=0)  # output one hot
        complex_label_idx_epoch = torch.concat(complex_label_idx_epoch, dim=0)  # gt idx
        complex_label_onehot_epoch = idx2onehot(complex_label_idx_epoch, NUM_CLS_COMPLEX)  # gt one hot
        complex_acc = calc_acc(complex_pred_idx_epoch, complex_label_idx_epoch)
        complex_rc = calc_recall(complex_pred_idx_epoch, complex_label_idx_epoch)
        complex_f1 = calc_f1(complex_pred_idx_epoch, complex_label_idx_epoch)
        complex_mAP = calc_mAP(complex_logits_epoch, complex_label_onehot_epoch)
    if use_communicative:
        communicative_pred_idx_epoch = torch.concat(communicative_pred_idx_epoch, dim=0)  # output idx
        communicative_logits_epoch = torch.concat(communicative_logits_epoch, dim=0)  # output one hot
        communicative_label_idx_epoch = torch.concat(communicative_label_idx_epoch, dim=0)  # gt idx
        communicative_label_onehot_epoch = idx2onehot(communicative_label_idx_epoch, NUM_CLS_COMMUNICATIVE)  # gt one hot
        communicative_acc = calc_acc(communicative_pred_idx_epoch, communicative_label_idx_epoch)
        communicative_rc = calc_recall(communicative_pred_idx_epoch, communicative_label_idx_epoch)
        communicative_f1 = calc_f1(communicative_pred_idx_epoch, communicative_label_idx_epoch)
        communicative_mAP = calc_mAP(communicative_logits_epoch, communicative_label_onehot_epoch)
    if use_transporting:
        transporting_pred_idx_epoch = torch.concat(transporting_pred_idx_epoch, dim=0)  # output idx
        transporting_logits_epoch = torch.concat(transporting_logits_epoch, dim=0)  # output one hot
        transporting_label_idx_epoch = torch.concat(transporting_label_idx_epoch, dim=0)  # gt idx
        transporting_label_onehot_epoch = idx2onehot(transporting_label_idx_epoch, NUM_CLS_TRANSPORTING)  # gt one hot
        transporting_acc = calc_acc(transporting_pred_idx_epoch, transporting_label_idx_epoch)
        transporting_rc = calc_recall(transporting_pred_idx_epoch, transporting_label_idx_epoch)
        transporting_f1 = calc_f1(transporting_pred_idx_epoch, transporting_label_idx_epoch)
        transporting_mAP = calc_mAP(transporting_logits_epoch, transporting_label_onehot_epoch)
    if use_age:
        age_pred_idx_epoch = torch.concat(age_pred_idx_epoch, dim=0)  # output idx
        age_logits_epoch = torch.concat(age_logits_epoch, dim=0)  # output one hot
        age_label_idx_epoch = torch.concat(age_label_idx_epoch, dim=0)  # gt idx
        age_label_onehot_epoch = idx2onehot(age_label_idx_epoch, NUM_CLS_AGE)  # gt one hot
        age_acc = calc_acc(age_pred_idx_epoch, age_label_idx_epoch)
        age_rc = calc_recall(age_pred_idx_epoch, age_label_idx_epoch)
        age_f1 = calc_f1(age_pred_idx_epoch, age_label_idx_epoch)
        age_mAP = calc_mAP(age_logits_epoch, age_label_onehot_epoch)
    
    
    # with torch.no_grad():
    #     mean_logits = torch.mean(torch.stack(mean_logits, dim=0), dim=0)
    # log('\twhole set mean logits: \t' + str((mean_logits[0].item(), mean_logits[1].item())))
    
    log('\ttime: \t{0}'.format(end -  start))
    res = {}
    if use_cross:
        n_nc_correct = n_correct - n_c_correct
        try:
            log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100) + '\tnc recall: ' + str(n_nc_correct / n_nc_gt * 100) + '\tc recall: ' + str(n_c_correct / n_c_gt * 100))
            log('\tmulti-class recall: \t' + str(recall))
            log('\tauc: \t' + str(auc) + '\tf1: \t' + str(f1))
            log('\tmAP: \t' + str(mAP)) 
        except:
            pass
        log('\tcross pred:' + str(n_c_pred) + '  not cross pred:' + str(n_nc_pred) + '  cross gt:' + str(n_c_gt) + '  not cross gt:' + str(n_nc_gt))
        log('\tcross ent: \t{0}'.format(total_ce / n_batches))
        res['base'] = [n_correct / n_examples, mAP, auc, f1]
    if use_atomic:
        log('\tatomic accc: ' + str(atomic_acc * 100))
        log('\tatomic recall: ' + str(atomic_rc))
        log('\tatomic f1: ' + str(atomic_f1))
        log('\tatomic mAP: ' + str(atomic_mAP))
        res['atomic'] = [atomic_acc * 100, atomic_mAP]
    if use_complex:
        log('\tcomplex accc: ' + str(complex_acc * 100))
        log('\tcomplex recall: ' + str(complex_rc))
        log('\tcomplex f1: ' + str(complex_f1))
        log('\tcomplex mAP: ' + str(complex_mAP))
        res['complex'] = [complex_acc * 100, complex_mAP]
    if use_communicative:
        log('\tcommunicative accc: ' + str(communicative_acc * 100))
        log('\tcommunicative recall: ' + str(communicative_rc))
        log('\tcommunicative f1: ' + str(communicative_f1))
        log('\tcommunicative mAP: ' + str(communicative_mAP))
        res['communicative'] = [communicative_acc * 100, communicative_mAP]
    if use_transporting:
        log('\ttransporting accc: ' + str(transporting_acc * 100))
        log('\ttransporting recall: ' + str(transporting_rc))
        log('\ttransporting f1: ' + str(transporting_f1))
        log('\ttransporting mAP: ' + str(transporting_mAP))
        res['transporting'] = [transporting_acc * 100, transporting_mAP]
    if use_age:
        log('\tage accc: ' + str(age_acc * 100))
        log('\tage recall: ' + str(age_rc))
        log('\tage f1: ' + str(age_f1))
        log('\tage mAP: ' + str(age_mAP))
        res['age'] = [age_acc * 100, age_mAP]

    if 'pred_traj' in data_types:
        res['pred_traj'] = total_mse / n_batches

    return res


def calc_orth_loss(protos, orth_type, threshold=512):
    '''
    protos: tensor B n_p proto_dim
    '''
    orth_loss = 0
    b_size = protos.size(0)
    if orth_type == 1:  # only diversity
        _mask = 1 - torch.unsqueeze(torch.eye(protos.size(1)), dim=0).cuda()  # 1 n_p n_p
        # mask = _mask.repeat(b_size, 1)  # B n_p n_p
        product = torch.matmul(protos, protos.permute(0, 2, 1))  # B n_p n_p
        orth_loss = torch.mean(torch.norm(_mask * product, dim=(1, 2)))
    elif orth_type == 2:  # diversity and orthoganality
        _mask = torch.unsqueeze(torch.eye(protos.size(1)), dim=0).cuda()  # 1 n_p n_p
        product = torch.matmul(protos, protos.permute(0, 2, 1))  # B n_p n_p
        orth_loss = torch.mean(torch.norm(product - _mask))
    elif orth_type == 3:
        protos_ = F.normalize(protos, dim=-1)
        l2 = ((protos_.unsqueeze(-2) - protos_.unsqueeze(-1)) ** 2).sum(-1)  # B np np
        neg_dis = threshold - l2
        mask = neg_dis>0
        neg_dis *= mask.float()
        neg_dis = torch.triu(neg_dis, diagonal=1)  # upper triangle
        orth_loss = neg_dis.sum(1).sum(1).mean()

    return orth_loss

def calc_orth_loss_fix(protos, orth_type, threshold=1):
    '''
    protos: tensor n_p proto_dim
    '''
    orth_loss = 0
    # print(protos.size())
    protos = protos.reshape(protos.size(0), -1)
    if orth_type == 1:  # only diversity
        _mask = 1 - torch.eye(protos.size(0)).cuda()  # n_p n_p
        product = torch.matmul(protos, protos.permute(1, 0))  # n_p n_p
        orth_loss = torch.norm(_mask * product, dim=(0, 1))
    elif orth_type == 2:  # diversity and orthoganality
        _mask = torch.eye(protos.size(0)).cuda()  # n_p n_p
        product = torch.matmul(protos, protos.permute(1, 0))  # n_p n_p
        orth_loss = torch.norm(product - _mask)
    elif orth_type == 3:
        protos_ = F.normalize(protos, dim=-1)
        l2 = ((protos_.unsqueeze(-2) - protos_.unsqueeze(-1)) ** 2).sum(-1)  # np np
        neg_dis = threshold - l2
        mask = neg_dis>0
        neg_dis *= mask.float()
        neg_dis = torch.triu(neg_dis, diagonal=1)  # upper triangle
        orth_loss = neg_dis.sum(0).sum(0)
    return orth_loss


if __name__ == '__main__':
    pass