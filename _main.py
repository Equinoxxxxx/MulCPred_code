import os
import pickle
import shutil
import time

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re
from _datasets import PIEDataset, JAADDataset

from helpers import makedir, draw_curves
from _proto_model import ImagePNet
from _backbones import create_backbone, record_conv3d_info, record_sp_conv3d_info_w, record_t_conv3d_info, BackboneOnly
import _project_prototypes
import prune
import _train_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from receptive_field import compute_proto_layer_rf_info_v2


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--project_batch_size', type=int, default=64)
    parser.add_argument('--warm_epochs', type=int, default=5)
    parser.add_argument('--push_start', type=int, default=10)
    parser.add_argument('--test_every', type=int, default=5)
    parser.add_argument('--linear_epochs', type=int, default=10)
    parser.add_argument('--save_proto_every_epoch', type=int, default=1)
    parser.add_argument('--cross_dataset', type=int, default=0)
    parser.add_argument('--joint_lr', type=float, default=0.0001)
    parser.add_argument('--last_lr', type=float, default=0.0001)
    parser.add_argument('--lr_step_size', type=int, default=5)

    parser.add_argument('--gpuid', type=str, default='0') # python3 main.py -gpuid=0,1,2,3

    parser.add_argument('--is_prototype_model', type=int, default=1)

    parser.add_argument('--backbone_name', type=str, default='C3D')
    parser.add_argument('--bbox_type', type=str, default='default')
    parser.add_argument('--obs_len', type=int, default=16)
    parser.add_argument('--p_per_cls', type=int, default=20)
    parser.add_argument('--prototype_dim', type=int, default=128)
    parser.add_argument('--prototype_activation_function', type=str, default='log')
    parser.add_argument('--add_on_activation', type=str, default=None)

    parser.add_argument('--balance_train', type=int, default=1)
    parser.add_argument('--balance_val', type=int, default=0)
    parser.add_argument('--balance_test', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--normalize_img_mode', type=str, default='torch')
    parser.add_argument('--resize_mode', type=str, default='padded')
    parser.add_argument('--max_occ', type=int, default=2)
    parser.add_argument('--data_split_type', type=str, default='default')
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_skeleton', type=int, default=0)

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    project_batch_size = args.project_batch_size
    warm_epochs = args.warm_epochs
    
    push_start = args.push_start
    push_epochs = [i for i in range(epochs) if i % push_start == 0]
    test_every = args.test_every
    linear_epochs = args.linear_epochs
    save_proto_every_epoch = args.save_proto_every_epoch
    cross_dataset = args.cross_dataset
    is_prototype_model = args.is_prototype_model


    backbone_name = args.backbone_name
    print('backbone: ', backbone_name)
    ped_img_size = (224, 224)
    if args.bbox_type == 'max':
        ped_img_size = (375, 688)
    obs_len = args.obs_len
    ped_vid_size = [obs_len, ped_img_size[0], ped_img_size[1]]
    p_per_cls = args.p_per_cls
    prototype_dim = args.prototype_dim
    prototype_activation_function = args.prototype_activation_function
    add_on_activation = args.add_on_activation

    balance_train = args.balance_train
    balance_val = args.balance_val
    balance_test = args.balance_test
    shuffle = args.shuffle
    normalize_img_mode = args.normalize_img_mode
    resize_mode = args.resize_mode
    max_occ = args.max_occ

    config = {'warm_epochs': warm_epochs,
              'is_prototype_model': is_prototype_model,

              'ped_img_size': ped_img_size,
              'obs_len': obs_len,

              'balance_train': balance_train,
              'balance_test': balance_test,
              'shuffle': shuffle,
              'normalize_img_mode': normalize_img_mode,
              'resize_mode': resize_mode,
              'max_occ': max_occ
             }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    print('Default device: ', os.environ['CUDA_VISIBLE_DEVICES'])


    exp_id = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    work_dir = '../work_dirs/models'
    model_dir = os.path.join(work_dir, backbone_name, exp_id)
    print('Save dir of current exp: ', model_dir)
    makedir(model_dir)
    config_path = os.path.join(model_dir, 'config.pll')
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    # shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    # shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
    # shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
    # shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
    # shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    log('config' + str(config))

    # # load the data
    log('----------------------------Load data-----------------------------')
    # train set
    train_dataset = PIEDataset(obs_len=obs_len, do_balance=balance_train, subset='train', bbox_size=ped_img_size, 
                                img_norm_mode=normalize_img_mode, resize_mode=resize_mode, max_occ=max_occ)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                                num_workers=4, pin_memory=False)
    # push set
    train_push_loader = torch.utils.data.DataLoader(train_dataset, batch_size=project_batch_size, shuffle=shuffle,
                                                num_workers=4, pin_memory=False)
    # test set
    test_dataset = PIEDataset(obs_len=obs_len, do_balance=balance_test, subset='test', bbox_size=ped_img_size, 
                                img_norm_mode=normalize_img_mode, resize_mode=resize_mode, max_occ=max_occ)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=4, pin_memory=False)
    # cross set
    if cross_dataset:
        JAAD_dataset = JAADDataset(obs_len=obs_len, do_balance=balance_test, subset='test', bbox_size=ped_img_size, 
                                normalize_img_mode=normalize_img_mode, resize_mode='padded', max_occ=max_occ)
        cross_loader = torch.utils.data.DataLoader(JAAD_dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=4, pin_memory=False)
    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(batch_size))

    # construct the model
    log('----------------------------Construct model-----------------------------')
    if is_prototype_model:
        backbone = create_backbone(backbone_name)
        conv_info = record_conv3d_info(backbone)
        log('conv info: ' + str(conv_info))
        sp_k_list, sp_s_list, sp_p_list = record_sp_conv3d_info_w(conv_info)
        t_k_list, t_s_list, t_p_list = record_t_conv3d_info(conv_info)
        sp_proto_layer_rf_info = compute_proto_layer_rf_info_v2(input_size=ped_vid_size[-1],
                                                            layer_filter_sizes=sp_k_list,
                                                            layer_strides=sp_s_list,
                                                            layer_paddings=sp_p_list,
                                                            prototype_kernel_size=1)
        t_proto_layer_rf_info = compute_proto_layer_rf_info_v2(input_size=ped_vid_size[0],
                                                            layer_filter_sizes=t_k_list,
                                                            layer_strides=t_s_list,
                                                            layer_paddings=t_p_list,
                                                            prototype_kernel_size=1)
        log('spatial receiptive field info: ' + str(sp_proto_layer_rf_info))
        log('temporal receiptive field info: ' + str(t_proto_layer_rf_info))
        ppnet = ImagePNet(backbone=backbone, 
                            vid_size=ped_vid_size, 
                            p_per_cls=p_per_cls, 
                            prototype_dim=prototype_dim, 
                            sp_proto_layer_rf_info=sp_proto_layer_rf_info,
                            t_proto_layer_rf_info=t_proto_layer_rf_info,
                            prototype_activation_function=prototype_activation_function,
                            add_on_activation=add_on_activation)
    else:
        ppnet = BackboneOnly(backbone_name)
    #if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.to(device)
    ppnet_multi = torch.nn.DataParallel(ppnet)

    class_specific = True  # 是否加入sep loss
    log('Model info')
    log(str(ppnet))
    # define optimizer
    log('----------------------------Construct optimizer-----------------------------')
    from settings import joint_optimizer_lrs, joint_lr_step_size
    if is_prototype_model:
        joint_optimizer_specs = [{'params': ppnet.backbone.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
                                {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
                                {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
                                ]
        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

        from settings import warm_optimizer_lrs
        warm_optimizer_specs = [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
                                {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
                                ]
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

        from settings import last_layer_optimizer_lr
        last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    else:
        joint_optimizer_specs = [{'params': ppnet.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
                                ]
        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)
    # weighting of different training losses
    from settings import coefs

    # train the model
    log('----------------------------Start training-----------------------------')
    import copy
    acc_curves_train = []
    acc_curves_test = []
    ce_curves_train = []
    ce_curves_test = []
    clst_curves_train = []
    clst_curves_test = []
    sep_curves_train = []
    sep_curves_test = []
    l1_curves_train = []
    l1_curves_test = []
    dist_curves_train = []
    dist_curves_test = []
    for epoch in range(1, epochs+1):
        log('epoch: \t{0}'.format(epoch))

        if is_prototype_model:
            if epoch <= warm_epochs:
                # fix住backbone，只训练backbone以后的部分(默认5个epoch)
                tnt.warm_only(model=ppnet_multi, log=log)
                train_res = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log, is_prototype_model=is_prototype_model)
            else:
                # 训练整个模型
                tnt.joint(model=ppnet_multi, log=log)  # 模型中的参数全部设置为可训练
            
                train_res = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log, is_prototype_model=is_prototype_model)
                joint_lr_scheduler.step()
        else:
            train_res = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log, is_prototype_model=is_prototype_model)
            joint_lr_scheduler.step()
        
        if epoch%test_every == 0:
            log('Testing')
            test_res = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log, is_prototype_model=is_prototype_model)
            if cross_dataset:
                log('Cross testing')
                cross_res = tnt.test(model=ppnet_multi, dataloader=cross_loader,
                            class_specific=class_specific, log=log, is_prototype_model=is_prototype_model)
            acc_train, ce_train = train_res[:2]
            acc_test, ce_test = test_res[:2]
            acc_curves_train.append(acc_train)
            acc_curves_test.append(acc_test)
            ce_curves_train.append(ce_train)
            ce_curves_test.append(ce_test)
            draw_curves(path=os.path.join(model_dir, '_acc.png'), train_curve=acc_curves_train, test_curve=acc_curves_test, metric_type='acc', test_every=test_every)
            draw_curves(path=os.path.join(model_dir, '_ce.png'), train_curve=ce_curves_train, test_curve=ce_curves_test, test_every=test_every)
            if is_prototype_model:
                acc_train, ce_train, clst_train, l1_train, dist_train = train_res[:5]
                acc_test, ce_test, clst_test, l1_test, dist_test = test_res[:5]
                clst_curves_train.append(clst_train)
                clst_curves_test.append(clst_test)
                l1_curves_train.append(l1_train)
                l1_curves_test.append(l1_test)
                draw_curves(path=os.path.join(model_dir, '_clst.png'), train_curve=clst_curves_train, test_curve=clst_curves_test, test_every=test_every)
                draw_curves(path=os.path.join(model_dir, '_l1.png'), train_curve=l1_curves_train, test_curve=l1_curves_test, test_every=test_every)
                if class_specific:
                    sep_train, avg_sep_train = train_res[-2:]
                    sep_test, avg_sep_test= test_res[-2:]
                    sep_curves_train.append(sep_train)
                    sep_curves_test.append(sep_test)
                    draw_curves(path=os.path.join(model_dir, '_sep.png'), train_curve=sep_curves_train, test_curve=sep_curves_test, test_every=test_every)

            log('Epoch' + str(epoch) + 'done. Test results: ' + str(acc_test))
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=test_res[0],
                                        target_accu=0.30, log=log)
        else:
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=train_res[0],
                                        target_accu=0.30, log=log)
        
        if epoch >= push_start and (epoch)%push_start == 0 and is_prototype_model:
            log('Project feat to ptototypes')
            _project_prototypes.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=None, # normalize if needed ``````````
                prototype_layer_stride=1,
                prototype_info_dir=img_dir,
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                log=log,
                save_every_epoch=save_proto_every_epoch)
            res_test = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            acc_test = res_test[0]
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=acc_test,
                                        target_accu=0.30, log=log)
            log('Project done. Test results: acc' + str(acc_test))
            if prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)
                log('Train linear only')
                for i in range(1, linear_epochs+1):
                    log('iteration: \t{0}'.format(i))
                    res_train = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                class_specific=class_specific, coefs=coefs, log=log)
                    if i%test_every == 0:
                        res_test = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                        class_specific=class_specific, log=log)
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=res_train[0],
                                                target_accu=0.3, log=log)
                log('last layer param:' + str(list(ppnet.last_layer.parameters())))  # generator -> list
                log('Linear layer training done.')
                
    logclose()

if __name__ == '__main__':
    main()