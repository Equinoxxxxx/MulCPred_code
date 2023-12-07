import time
import torch
from tqdm import tqdm

from helpers import list_of_distances, make_one_hot

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, device='cuda:0', is_prototype_model=True, v_data_type='ped_imgs',
                   check_grad=False):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_c_pred = 0
    n_c_gt = 0
    n_nc_pred = 0
    n_nc_gt = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    b_end = time.time()
    tbar = tqdm(dataloader)
    for i, data in enumerate(tbar):
        b_start = time.time()
        # print('Data loading time per batch: ' + str(b_start-b_end))
        # BTHWC -> BCTHW
        v_inputs = data[v_data_type]
        inputs = v_inputs.to(device)
        pred_intent = data['pred_intent'].view(-1) # real numbers, not one hot
        target = pred_intent.to(device)

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            if is_prototype_model:
                output, min_distances = model(v_inputs)

                # compute loss
                cross_entropy = torch.nn.functional.cross_entropy(output, target)
                if class_specific:
                    max_dist = (model.module.prototype_shape[1]
                                * model.module.prototype_shape[2]
                                * model.module.prototype_shape[3])  # C * 1 * 1 * 1 = C  (results of add_on_layer are sigmoid-ed)

                    # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                    # calculate cluster cost
                    prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,pred_intent]).cuda()  # (np, B) ->transpose-> (B, np)
                    inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                    cluster_cost = torch.mean(max_dist - inverted_distances)

                    # calculate separation cost
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    inverted_distances_to_nontarget_prototypes, _ = torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                    separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)  # negative sign is in the loss weight

                    # calculate avg cluster cost
                    avg_separation_cost = torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                    avg_separation_cost = torch.mean(avg_separation_cost)
                    
                    if use_l1_mask:
                        l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                        l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                    else:
                        l1 = model.module.last_layer.weight.norm(p=1) 

                else:
                    min_distance, _ = torch.min(min_distances, dim=1)
                    cluster_cost = torch.mean(min_distance)
                    l1 = model.module.last_layer.weight.norm(p=1)

                # evaluation statistics
                total_cluster_cost += cluster_cost.item()
                total_separation_cost += separation_cost.item()
                total_avg_separation_cost += avg_separation_cost.item()
            else:
                output = model(v_inputs)

                # compute loss
                cross_entropy = torch.nn.functional.cross_entropy(output, target)
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            n_c_pred += (predicted == 1).sum().item()
            n_nc_pred += (predicted == 0).sum().item()
            n_c_gt += (target == 1).sum().item()
            n_nc_gt += (target == 0).sum().item()
            n_batches += 1
            total_cross_entropy += cross_entropy.item()

        # compute gradient and do SGD step
        if is_train:
            if is_prototype_model:
                if class_specific:
                    if coefs is not None:
                        loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            + coefs['l1'] * l1)
                    else:
                        loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
                else:
                    if coefs is not None:
                        loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['l1'] * l1)
                    else:
                        loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            else:
                loss = cross_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tbar.set_postfix({'acc': n_correct / n_examples, 'ce_loss': cross_entropy.item()})
        del data
        del inputs
        del target
        del output
        del predicted
        if is_prototype_model:
            del min_distances
        b_end = time.time()
        # print('Computating time: ' + str(b_end - b_start))

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('cross pred:' + str(n_c_pred) + '  not cross pred:' + str(n_nc_pred) + '  cross gt:' + str(n_c_gt) + '  not cross gt:' + str(n_nc_gt))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    res = [n_correct / n_examples, total_cross_entropy / n_batches]

    if is_prototype_model:
        log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
        l1_loss = model.module.last_layer.weight.norm(p=1).item()
        log('\tl1: \t\t{0}'.format(l1_loss))
        p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
        with torch.no_grad():
            p_avg_pair_dist = torch.mean(list_of_distances(p, p))
        log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))
        res = [n_correct / n_examples, total_cross_entropy / n_batches, total_cluster_cost / n_batches, l1_loss, p_avg_pair_dist.item()]
        if class_specific:
            log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
            log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
            res += [total_separation_cost / n_batches, total_avg_separation_cost / n_batches]


    return res


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print, device='cuda:0', is_prototype_model=True,
            v_data_type='ped_imgs', check_grad=False):
    assert(optimizer is not None)
    
    log('train')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, device=device, is_prototype_model=is_prototype_model,
                          v_data_type=v_data_type, check_grad=check_grad)


def test(model, dataloader, class_specific=False, log=print, device='cuda:0', is_prototype_model=True,
            v_data_type='ped_imgs'):
    log('--------------------------test--------------------------')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log, device=device, is_prototype_model=is_prototype_model,
                          v_data_type=v_data_type)


def last_only(model, log=print):
    for p in model.module.backbone.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.backbone.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.backbone.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
