from pickletools import optimize
from socket import TIPC_ADDR_ID
import time
from pandas import MultiIndex
import torch
from tqdm import tqdm
import numpy as np
import cv2
import os

from helpers import list_of_distances, make_one_hot
from utils import seg_context_batch3d
from helpers import makedir

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, device='cuda:0', is_prototype_model=1, data_types=['img'],
                   check_grad=False, orth_type=0, vis_path=None, display_logits=True, display_p_corr=False):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_c_correct = 0
    n_c_pred = 0
    n_c_gt = 0
    n_nc_pred = 0
    n_nc_gt = 0
    n_batches = 0
    n_all = dataloader.dataset.num_samples
    n_c = dataloader.dataset.n_c
    n_nc = dataloader.dataset.n_nc
    weight = [n_c / n_all, n_nc / n_all]
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_orth = 0
    mean_logits = []
    min_dist_whole_set = []
    b_end = time.time()
    tbar = tqdm(dataloader, miniters=1)
    # print('-----------------------loading data---------------------')
    for i, data in enumerate(tbar):
        # print('-----------------------data loaded---------------------')
        inputs = {}
        if 'traj' in data_types:
            inputs['traj'] = data['obs_bboxes'].to(device)
        if 'img' in data_types:
            inputs['img'] = data['ped_imgs'].to(device)
        if 'skeleton' in data_types:
            inputs['skeleton'] = data['obs_skeletons'].to(device)
        if 'context' in data_types:
            if dataloader.dataset.ctx_mode=='seg_multi' and False:
                context = data['obs_context']
                seg = data['obs_seg']
                seg = seg_context_batch3d(seg, context, seg_class_idx=dataloader.dataset.seg_class_idx, tgt_size=dataloader.dataset.ctx_size)
                inputs['context'] = seg.to(device)
            else:
                inputs['context'] = data['obs_context'].to(device)
        if 'single_img' in data_types:
            inputs['single_img'] = data['ped_single_imgs'].to(device)
        pred_intent = data['pred_intent'].view(-1) # real numbers, not one hot
        target = pred_intent.to(device)
        b_start = time.time()
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            if is_prototype_model > 0:
                output, min_distances = model(inputs)
                with torch.no_grad():
                    min_dist_whole_set.append(torch.mean(min_distances, dim=0))
                # compute loss
                # print(inputs['img'].size(), output.size(), target.size())
                cross_entropy = torch.nn.functional.cross_entropy(output, target, weight=torch.tensor(weight).float().to(device))
                if class_specific:
                    cluster_cost = 0
                    separation_cost = 0
                    avg_separation_cost = 0
                    l1 = 0
                    orth = 0

                    if model.module.use_img:
                        img_branch = model.module.img_model
                        max_dist = img_branch.prototype_dim  # C * 1 * 1 * 1 = C  (results of add_on_layer are sigmoid-ed)
                        img_min_dist = min_distances[:, :img_branch.num_prototypes]
                        img_cluster_cost, img_separation_cost, img_avg_separation_cost, img_l1, img_orth = calc_regular_terms(model=img_branch, 
                                                                                                                            pred_intent=pred_intent, 
                                                                                                                            min_distances=img_min_dist, 
                                                                                                                            max_dist=max_dist,
                                                                                                                            use_l1_mask=use_l1_mask,
                                                                                                                            orth_type=orth_type,
                                                                                                                            is_prototype_model=is_prototype_model)
                        cluster_cost += img_cluster_cost
                        separation_cost += img_separation_cost
                        avg_separation_cost += img_avg_separation_cost
                        l1 += img_l1
                        orth += img_orth
                    if model.module.use_skeleton:
                        sk_branch = model.module.skeleton_model
                        max_dist = sk_branch.prototype_dim
                        sk_dist_start = 0
                        if model.module.use_img:
                            sk_dist_start += img_branch.num_prototypes
                        sk_min_dist = min_distances[:, sk_dist_start:sk_dist_start+sk_branch.num_prototypes]
                        sk_cluster_cost, sk_separation_cost, sk_avg_separation_cost, sk_l1, sk_orth = calc_regular_terms(model=sk_branch, 
                                                                                                            pred_intent=pred_intent, 
                                                                                                            min_distances=sk_min_dist, 
                                                                                                            max_dist=max_dist,
                                                                                                            use_l1_mask=use_l1_mask,
                                                                                                            orth_type=orth_type,
                                                                                                            is_prototype_model=is_prototype_model)
                        cluster_cost += sk_cluster_cost
                        separation_cost += sk_separation_cost
                        avg_separation_cost += sk_avg_separation_cost
                        l1 += sk_l1
                        orth += sk_orth
                    if model.module.use_ctx:
                        if model.module.context_model_settings['ctx_mode'] == 'seg_multi':
                            ctx_branch = model.module.context_model
                            max_dist = ctx_branch[0].prototype_dim
                            ctx_dist_start = 0
                            if model.module.use_img:
                                ctx_dist_start += img_branch.num_prototypes
                            if model.module.use_skeleton:
                                ctx_dist_start += sk_branch.num_prototypes
                            ctx_min_dist = min_distances[:, ctx_dist_start:ctx_dist_start+ctx_branch[0].num_prototypes * len(ctx_branch)]
                            ctx_cluster_cost, ctx_separation_cost, ctx_avg_separation_cost, ctx_l1, ctx_orth = 0, 0, 0, 0, 0
                            for m in range(len(ctx_branch)):
                                _ctx_cluster_cost, _ctx_separation_cost, _ctx_avg_separation_cost, _ctx_l1, _ctx_orth = calc_regular_terms(model=ctx_branch[m], 
                                                                                                                            pred_intent=pred_intent, 
                                                                                                                            min_distances=ctx_min_dist[:, m*ctx_branch[0].num_prototypes: (m+1)*ctx_branch[0].num_prototypes], 
                                                                                                                            max_dist=max_dist,
                                                                                                                            use_l1_mask=use_l1_mask,
                                                                                                                            orth_type=orth_type,
                                                                                                                            is_prototype_model=is_prototype_model)
                                cluster_cost += _ctx_cluster_cost
                                separation_cost += _ctx_separation_cost
                                avg_separation_cost += _ctx_avg_separation_cost
                                l1 += _ctx_l1
                                orth += _ctx_orth
                        else:
                            ctx_branch = model.module.context_model
                            max_dist = ctx_branch.prototype_dim
                            ctx_dist_start = 0
                            if model.module.use_img:
                                ctx_dist_start += img_branch.num_prototypes
                            if model.module.use_skeleton:
                                ctx_dist_start += sk_branch.num_prototypes
                            ctx_min_dist = min_distances[:, ctx_dist_start:ctx_dist_start+ctx_branch.num_prototypes]
                            ctx_cluster_cost, ctx_separation_cost, ctx_avg_separation_cost, ctx_l1, ctx_orth = calc_regular_terms(model=ctx_branch, 
                                                                                                                pred_intent=pred_intent, 
                                                                                                                min_distances=ctx_min_dist, 
                                                                                                                max_dist=max_dist,
                                                                                                                use_l1_mask=use_l1_mask,
                                                                                                                orth_type=orth_type,
                                                                                                                is_prototype_model=is_prototype_model)
                            cluster_cost += ctx_cluster_cost
                            separation_cost += ctx_separation_cost
                            avg_separation_cost += ctx_avg_separation_cost
                            l1 += ctx_l1
                            orth += ctx_orth
                    if model.module.use_single_img:
                        single_img_branch = model.module.single_img_model
                        max_dist = single_img_branch.prototype_dim
                        single_img_dist_start = 0
                        if model.module.use_img:
                            single_img_dist_start += img_branch.num_prototypes
                        if model.module.use_skeleton:
                            single_img_dist_start += sk_branch.num_prototypes
                        if model.module.use_ctx:
                            if model.module.context_model_settings['ctx_mode'] == 'seg_multi':
                                for m in range(len(ctx_branch)):
                                    single_img_dist_start += ctx_branch[m].num_prototypes
                            else:
                                single_img_dist_start += ctx_branch.num_prototypes
                        single_img_min_dist = min_distances[:, single_img_dist_start:single_img_dist_start+single_img_branch.num_prototypes]
                        single_img_cluster_cost, single_img_separation_cost, single_img_avg_separation_cost, single_img_l1, single_img_orth = calc_regular_terms(model=single_img_branch, 
                                                                                                            pred_intent=pred_intent, 
                                                                                                            min_distances=single_img_min_dist, 
                                                                                                            max_dist=max_dist,
                                                                                                            use_l1_mask=use_l1_mask,
                                                                                                            orth_type=orth_type,
                                                                                                            is_prototype_model=is_prototype_model)
                        cluster_cost += single_img_cluster_cost
                        separation_cost += single_img_separation_cost
                        avg_separation_cost += single_img_avg_separation_cost
                        l1 += single_img_l1
                        orth += single_img_orth
                else:
                    min_min_distance, _ = torch.min(min_distances, dim=1)
                    cluster_cost = torch.mean(min_min_distance)
                    l1 = model.module.last_layer.weight.norm(p=1)

                # evaluation statistics
                total_cluster_cost += cluster_cost.item()
                total_separation_cost += separation_cost.item()
                total_avg_separation_cost += avg_separation_cost.item()
                if orth_type > 0:
                    total_orth += orth.item()  
            else:
                output = model(inputs)
                # compute loss
                cross_entropy = torch.nn.functional.cross_entropy(output, target, )
                if vis_path is not None and ('img' in data_types or 'context' in data_types):
                    batch_feat = model.module.calc_feat(inputs)
                    vis_dir = os.path.join(vis_path, str(i))
                    if 'img' in data_types:
                        visualize_batch(inputs['img'], target, output, batch_feat[0], i, os.path.join(vis_dir, 'img'))
                    if 'context' in data_types:
                        visualize_batch(inputs['context'], target, output, batch_feat[0], i, os.path.join(vis_dir, 'context'))
                    
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            res = predicted == target
            n_correct += res.sum().item()
            for j in range(len(target)):
                if res[j] and target[j] == 1:
                    n_c_correct += 1
            n_c_pred += (predicted == 1).sum().item()
            n_nc_pred += (predicted == 0).sum().item()
            n_c_gt += (target == 1).sum().item()
            n_nc_gt += (target == 0).sum().item()
            n_batches += 1
            total_cross_entropy += cross_entropy.item()

        # compute gradient and do SGD step
        if is_train:
            if is_prototype_model > 0:
                if class_specific:
                    if coefs is not None:
                        loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            + coefs['l1'] * l1
                            + coefs['orth'] * orth)
                    else:
                        loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
                else:
                    if coefs is not None:
                        loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['l1'] * l1
                            + coefs['orth'] * orth)
                    else:
                        loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            else:
                loss = cross_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        data_prepare_time = b_start - b_end
        b_end = time.time()
        computing_time = b_end - b_start
        display_dict = {'acc': n_correct / n_examples, 
                        # 'ce_loss': cross_entropy.item(), 
                        'data': data_prepare_time, 
                        'compute': computing_time
                          }
        with torch.no_grad():
            mean_logit = torch.mean(output, dim=0)
        mean_logits.append(mean_logit)
        if display_logits:
            display_dict['logit'] = [round(output[0, 0].item(), 4), round(output[0, 1].item(), 4)]
            display_dict['avg logit'] = [round(mean_logit[0].item(), 4), round(mean_logit[1].item(), 4)]
        tbar.set_postfix(display_dict)
        
        del data
        del inputs
        del target
        del output
        del predicted
        if is_prototype_model:
            del min_distances
    
    tbar.close()
    end = time.time()
    with torch.no_grad():
        mean_logits = torch.mean(torch.stack(mean_logits, dim=0), dim=0)
    log('\twhole set mean logits: \t' + str((mean_logits[0].item(), mean_logits[1].item())))
    n_nc_correct = n_correct - n_c_correct
    log('\ttime: \t{0}'.format(end -  start))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100) + '\tc recall: ' + str(n_c_correct / n_c_gt * 100) + '\tnc recall: ' + str(n_nc_correct / n_nc_gt * 100))
    log('\tcross pred:' + str(n_c_pred) + '  not cross pred:' + str(n_nc_pred) + '  cross gt:' + str(n_c_gt) + '  not cross gt:' + str(n_nc_gt))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    res = [n_correct / n_examples, total_cross_entropy / n_batches]

    if is_prototype_model:
        p_means = []
        p_maxs = []
        p_mins = []
        log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
        if isinstance(l1, int):
            l1_loss = l1
        else:
            l1_loss = l1.item()
        log('\tl1: \t\t{0}'.format(l1_loss))
        if orth_type > 0:
            log('\torth: \t' + str(total_orth / n_batches))
        log('\tmin dist for each proto--avg along whole set: \t' + str(torch.mean(torch.stack(min_dist_whole_set, dim=0), dim=0)))
        img_p_pair_dist = 0
        sk_p_pair_dist = 0
        ctx_p_pair_dist = 0
        with torch.no_grad():
            if model.module.use_img:
                p = img_branch.prototype_vectors.view(img_branch.num_prototypes, -1).cpu()  # np, C
                p_avg = torch.mean(p, dim=1)
                p_max, _ = torch.max(p, dim=1)
                p_min, _ = torch.min(p, dim=1)
                p_means.append(p_avg.numpy())
                p_maxs.append(p_max.numpy())
                p_mins.append(p_min.numpy())
                p_1st_value = p[:, 0]
                img_p_dist_map = np.array(list_of_distances(p, p))  # np, np  两两p之间l2距离
                if display_p_corr:
                    # log('\tp img: \t' + str(p[0]))
                    # log('\tp img dist map: \t' + str(img_p_dist_map))
                    log('\tp img avg: \t' + str(p_avg))
                    log('\tp img max: \t' + str(p_max))
                    log('\tp img 1st value: \t' + str(p_1st_value))
                img_p_pair_dist = np.mean(img_p_dist_map)
                log('\tavg p img dist: \t{0}'.format(img_p_pair_dist))
            if model.module.use_skeleton:
                p = sk_branch.prototype_vectors.view(sk_branch.num_prototypes, -1).cpu()
                p_avg = torch.mean(p, dim=1)
                p_max, _ = torch.max(p, dim=1)
                p_min, _ = torch.min(p, dim=1)
                p_means.append(p_avg.numpy())
                p_maxs.append(p_max.numpy())
                p_mins.append(p_min.numpy())
                p_1st_value = p[:, 0]
                sk_p_dist_map = np.array(list_of_distances(p, p))
                if display_p_corr:
                    # log('\tp sk: \t' + str(p[0]))
                    # log('\tp sk dist map: \t' + str(sk_p_dist_map))
                    log('\tp sk avg: \t' + str(p_avg))
                    log('\tp sk max: \t' + str(p_max))
                    log('\tp sk 1st value: \t' + str(p_1st_value))
                sk_p_pair_dist = np.mean(sk_p_dist_map)
                log('\tavg p sk dist pair: \t{0}'.format(sk_p_pair_dist))
            if model.module.use_ctx:
                if dataloader.dataset.ctx_mode == 'seg_multi':
                    ctx_p_pair_dist = 0
                    for j in range(len(ctx_branch)):
                        p = ctx_branch[j].prototype_vectors.view(ctx_branch[j].num_prototypes, -1).cpu()
                        p_avg = torch.mean(p, dim=1)
                        p_max, _ = torch.max(p, dim=1)
                        p_min, _ = torch.min(p, dim=1)
                        p_means.append(p_avg.numpy())
                        p_maxs.append(p_max.numpy())
                        p_mins.append(p_min.numpy())
                        p_1st_value = p[:, 0]
                        ctx_p_dist_map = np.array(list_of_distances(p, p))
                        if display_p_corr:
                            # log('\tp ctx: \t' + str(p[0]))
                            # log('\tp ctx dist map: \t' + str(ctx_p_dist_map))
                            log('\tp ctx avg: \t' + str(p_avg))
                            log('\tp ctx max: \t' + str(p_max))
                            log('\tp ctx 1st value: \t' + str(p_1st_value))
                        ctx_p_pair_dist = np.mean(ctx_p_dist_map)
                        ctx_p_pair_dist += np.mean(ctx_p_dist_map)
                else:
                    p = ctx_branch.prototype_vectors.view(ctx_branch.num_prototypes, -1).cpu()
                    p_avg = torch.mean(p, dim=1)
                    p_max, _ = torch.max(p, dim=1)
                    p_min, _ = torch.min(p, dim=1)
                    p_means.append(p_avg.numpy())
                    p_maxs.append(p_max.numpy())
                    p_mins.append(p_min.numpy())
                    p_1st_value = p[:, 0]
                    ctx_p_dist_map = np.array(list_of_distances(p, p))
                    if display_p_corr:
                        # log('\tp ctx: \t' + str(p[0]))
                        # log('\tp ctx dist map: \t' + str(ctx_p_dist_map))
                        log('\tp ctx avg: \t' + str(p_avg))
                        log('\tp ctx max: \t' + str(p_max))
                        log('\tp ctx 1st value: \t' + str(p_1st_value))
                    ctx_p_pair_dist = np.mean(ctx_p_dist_map)
                    log('\tavg p ctx dist pair: \t{0}'.format(ctx_p_pair_dist))
            if model.module.use_single_img:
                p = single_img_branch.prototype_vectors.view(single_img_branch.num_prototypes, -1).cpu()
                p_avg = torch.mean(p, dim=1)
                p_max, _ = torch.max(p, dim=1)
                p_min, _ = torch.min(p, dim=1)
                p_means.append(p_avg.numpy())
                p_maxs.append(p_max.numpy())
                p_mins.append(p_min.numpy())
                p_1st_value = p[:, 0]
                single_img_p_dist_map = np.array(list_of_distances(p, p))
                if display_p_corr:
                    # log('\tp single_img: \t' + str(p[0]))
                    # log('\tp single_img dist map: \t' + str(single_img_p_dist_map))
                    log('\tp single img avg: \t' + str(p_avg))
                    log('\tp single img max: \t' + str(p_max))
                    log('\tp single img 1st value: \t' + str(p_1st_value))
                single_img_p_pair_dist = np.mean(single_img_p_dist_map)
                log('\tavg p single_img dist pair: \t{0}'.format(single_img_p_pair_dist))
        p_avg_pair_dist = [img_p_pair_dist, sk_p_pair_dist, ctx_p_pair_dist]
        res = [n_correct / n_examples, total_cross_entropy / n_batches, total_cluster_cost / n_batches, l1_loss, total_orth / n_batches, p_avg_pair_dist]
        if class_specific:
            log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
            # log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
            res += [total_separation_cost / n_batches, total_avg_separation_cost / n_batches]
        p_means = np.concatenate(p_means, axis=0)
        p_maxs = np.concatenate(p_maxs, axis=0)
        p_mins = np.concatenate(p_mins, axis=0)
        p_info = np.stack([p_means, p_maxs, p_mins], axis=1)  # NP, 3
        res += [p_info]


    return res


def calc_regular_terms(model, pred_intent, min_distances, max_dist, use_l1_mask=True, orth_type=0, is_prototype_model=1):
    max_dist = model.prototype_dim  # C * 1 * 1 * 1 = C  (results of add_on_layer are sigmoid-ed)

    # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
    # calculate cluster cost
    prototypes_of_correct_class = torch.t(model.prototype_class_identity[:,pred_intent]).cuda()  # (np, B) ->transpose-> (B, np)
    if is_prototype_model == 1:
        inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
        cluster_cost = torch.mean(max_dist - inverted_distances)
    elif is_prototype_model == 2:
        cluster_cost = -torch.mean(torch.max(min_distances, dim=1)[0])  # max similarity

    # calculate separation cost
    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
    if is_prototype_model == 1:
        inverted_distances_to_nontarget_prototypes, _ = torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
        separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)  # negative sign is in the loss weight
    elif is_prototype_model == 2:
        separation_cost = torch.mean(torch.max(min_distances * prototypes_of_wrong_class, dim=1)[0])  # max simi of wrong cls

    # calculate avg cluster cost
    avg_separation_cost = torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
    avg_separation_cost = torch.mean(avg_separation_cost)

    # l1 cost
    if use_l1_mask:
        l1_mask = 1 - torch.t(model.prototype_class_identity).cuda()
        if model.last_nonlinear == 2:
            l1 = (model.last_layer[2].weight * l1_mask).norm(p=1)
        elif model.last_nonlinear <= 1:
            l1 = (model.last_layer.weight * l1_mask).norm(p=1)
        else:
            l1 = 0
    else:
        l1 = model.last_layer.weight.norm(p=1)
    
    # orth cost
    orth_cost = calc_orth_loss(model, orth_type=orth_type)

    return cluster_cost, separation_cost, avg_separation_cost, l1, orth_cost

def calc_orth_loss(model, orth_type):
    orth_loss = 0
    p = model.prototype_vectors.view(model.num_prototypes, -1)
    if orth_type == 1:
        mask = 1 - torch.eye(model.num_prototypes).cuda()
        orth_loss = torch.norm(mask * torch.mm(p,
                                               torch.t(p)))
    elif orth_type == 2:
        mask = torch.eye(model.num_prototypes).cuda()
        orth_loss = torch.norm(torch.mm(p,
                                        torch.t(p))
                                - mask)
    elif orth_type == 3:
        p_dist_map = list_of_distances(p, p)
        orth_loss = torch.mean(torch.mean(p_dist_map))
    # print('orth ', orth_loss, 'orth type', orth_type)
    return orth_loss


def last_only_multi(model, log=print):
    if model.module.use_img:
        last_only(model.module.img_model)
        log('\timg last layer')
    if model.module.use_skeleton:
        last_only(model.module.skeleton_model)
        log('\tskeleton last layer')
    if model.module.use_ctx:
        if model.module.context_model_settings['ctx_mode'] == 'seg_multi':
            for m in model.module.context_model:
                last_only(m)
        else:
            last_only(model.module.context_model)
        log('\tcontext last layer')

def warm_only_multi(model, log=print):
    if model.module.use_img:
        warm_only(model.module.img_model)
        log('\timg warm')
    if model.module.use_skeleton:
        warm_only(model.module.skeleton_model)
        log('\tskeleton warm')
    if model.module.use_ctx:
        if model.module.context_model_settings['ctx_mode'] == 'seg_multi':
            for m in model.module.context_model:
                warm_only(m)
        else:
            warm_only(model.module.context_model)
        log('\tcontext warm')

def joint_multi(model, log=print):
    if model.module.use_img:
        joint(model.module.img_model)
        log('\timg joint')
    if model.module.use_skeleton:
        joint(model.module.skeleton_model)
        log('\tskeleton joint')
    if model.module.use_ctx:
        if model.module.context_model_settings['ctx_mode'] == 'seg_multi':
            for m in model.module.context_model:
                joint(m)
        else:
            joint(model.module.context_model)
        log('\tcontext joint')

def last_only(model):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.last_layer.parameters():
        p.requires_grad = True

def warm_only(model, log=print):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')

def joint(model, log=print):
    for p in model.backbone.parameters():
        p.requires_grad = True
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')

def visualize_batch(batch_input, batch_gt, batch_pred, batch_feat, batch_idx, batch_dir):
    c_pred = batch_pred[:, 1]
    nc_pred = batch_pred[:, 0]
    c_pred -= nc_pred
    TP_idx = None
    FP_idx = None
    TN_idx = None
    FN_idx = None
    # print('gt size', batch_gt.size())
    # print('gt', batch_gt, batch_gt==1)
    # print('c pred size', c_pred[batch_gt==1].size(0), c_pred[batch_gt==0].size(0))
    # print('c pred 0', c_pred[batch_gt==0])
    if c_pred[batch_gt==1].size(0) > 0:
        _, TP_idx = torch.max(c_pred[batch_gt==1], 0)
        _, FN_idx = torch.min(c_pred[batch_gt==1], 0)
    if c_pred[batch_gt==0].size(0) > 0:
        _, FP_idx = torch.max(c_pred[batch_gt==0], 0)
        _, TN_idx = torch.min(c_pred[batch_gt==0], 0)
    
    # print(TP_idx, FP_idx, TN_idx, FN_idx)
    if TP_idx is not None:
        visualize_featmap3d(featmap=batch_feat[TP_idx], ori_input=batch_input[TP_idx], save_dir=os.path.join(batch_dir, 'TP'))
    if FN_idx is not None:
        visualize_featmap3d(featmap=batch_feat[FN_idx], ori_input=batch_input[FN_idx], save_dir=os.path.join(batch_dir, 'FN'))
    if TN_idx is not None:
        visualize_featmap3d(featmap=batch_feat[TN_idx], ori_input=batch_input[TN_idx], save_dir=os.path.join(batch_dir, 'TN'))
    if FP_idx is not None:
        visualize_featmap3d(featmap=batch_feat[FP_idx], ori_input=batch_input[FP_idx], save_dir=os.path.join(batch_dir, 'FP'))

def visualize_featmap3d(featmap, ori_input, color_order='BGR', img_norm_mode='torch', save_dir=''):
    '''
    featmap: Cthw
    ori_input: 3THW
    '''
    makedir(save_dir)
    # tgt_size = [1, ori_input.size(1), ori_input.size(2), ori_input.size(3)]

    featmap = torch.mean(featmap, dim=0, keepdim=True).view(1, 1, featmap.size(1), featmap.size(2), featmap.size(3))  # 1 1 thw
    featmap = torch.nn.functional.interpolate(featmap, size=(ori_input.size(1), ori_input.size(2), ori_input.size(3)), mode='trilinear')  # 1 1 THW
    featmap = featmap.view(1, ori_input.size(1), ori_input.size(2), ori_input.size(3)).permute(1, 2, 3, 0).cpu().numpy()  # THW 1
    featmap -= np.amin(featmap)
    featmap /= np.amax(featmap)
    ori_input = ori_input.permute(1, 2, 3, 0).cpu().numpy()  # THW 3
    if color_order == 'RGB':
        ori_input = ori_input[:, :, :, ::-1]
    if img_norm_mode == 'tf':
        ori_input += 1.
        ori_input *= 127.5
    elif img_norm_mode == 'torch':
        ori_input[:,:,:,0] *= 0.225
        ori_input[:,:,:,1] *= 0.224
        ori_input[:,:,:,2] *= 0.229
        ori_input[:,:,:,0] += 0.406
        ori_input[:,:,:,1] += 0.456
        ori_input[:,:,:,2] += 0.485
        ori_input *= 255.
    for i in range(ori_input.shape[0]):
        img = ori_input[i]
        heatmap = cv2.applyColorMap(np.uint8(255*featmap[i]), cv2.COLORMAP_JET)
        img = 0.3 * heatmap + 0.6 * img
        cv2.imwrite(os.path.join(save_dir, str(i)+'.png'), img)



if __name__ == '__main__':
    from _proto_model import ImagePNet, MultiPNet, SkeletonPNet
    from _backbones import create_backbone
    from torchviz import make_dot
    from thop import profile

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    init_const = 1.

