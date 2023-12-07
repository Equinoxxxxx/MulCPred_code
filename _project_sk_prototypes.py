from turtle import update
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time
import scipy
from tqdm import tqdm
import pdb

from receptive_field import compute_rf_loc_spatial, compute_rf_loc_spatiotemporal
from helpers import find_high_activation_crop_spatiotemporal, makedir, find_high_activation_crop
from utils import seg_context_batch3d, visualize_featmap3d_simple

# push each prototype to the nearest joints in the training set
def push_sk_protos(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    prototype_layer_stride=1,
                    prototype_info_dir='../work_dirs/latest_exp_sk_proto', # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    log=print,
                    prototype_activation_function_in_numpy=None,
                    save_every_epoch=True,
                    update_proto=True):
    
    makedir(prototype_info_dir)
    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)  # the min dist value of each proto
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1], # C
         prototype_shape[2], # H
         prototype_shape[3]] # W
         )

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: sample index in the entire dataset
    1: time start idx
    2: time end idx
    3: joint start index
    4: joint end index
    5: (optional) class identity
    '''
    proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                fill_value=-1)
    highly_act_boxes = np.full(shape=[n_prototypes, 6],
                                        fill_value=-1)

    if save_every_epoch and epoch_number is not None:
        proto_epoch_dir = os.path.join(prototype_info_dir, 'epoch-'+str(epoch_number))
        makedir(proto_epoch_dir)
    else:
        proto_epoch_dir = prototype_info_dir

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes
    dataset_name = dataloader.dataset.dataset_name
    for push_iter, data in enumerate(tqdm(dataloader, miniters=1)):
        
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        search_y = data['pred_intent'].view(-1)
        ped_id_int = data['ped_id_int']
        img_nm_int = data['img_nm_int']
        search_batch_input = data['obs_skeletons']
        start_index_of_search_batch = push_iter * search_batch_size
        update_sk_proto_on_batch(search_batch_input,
                                   ped_id_int=ped_id_int,
                                   img_nm_int=img_nm_int,
                                   start_index_of_search_batch=start_index_of_search_batch,
                                   prototype_network_parallel=prototype_network_parallel,
                                   global_min_proto_dist=global_min_proto_dist,
                                   global_min_fmap_patches=global_min_fmap_patches,
                                   proto_rf_boxes=proto_rf_boxes,
                                   highly_act_boxes=highly_act_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   prototype_layer_stride=prototype_layer_stride,
                                   save_dir=proto_epoch_dir,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
                                   dataset_name=dataset_name)

    np.save(os.path.join(proto_epoch_dir, 'prototype_info.npy'),
            proto_rf_boxes)
    np.save(os.path.join(proto_epoch_dir, 'highly_activated_info.npy'),  # *****************************
            highly_act_boxes)
    if update_proto:
        log('\tExecuting push ...')
        prototype_update = np.reshape(global_min_fmap_patches,
                                    tuple(prototype_shape))
        prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())  # UPDATE THE PROTOTYEPS
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))

# update each prototype for current search batch
def update_sk_proto_on_batch(search_batch_input,  # input of cur batch
                               ped_id_int,
                               img_nm_int,
                               start_index_of_search_batch,  # idx of first sample in cur batch
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated, the min dist of each proto among the whole epoch  (np,)
                               global_min_fmap_patches, # this will be updated  (np, C, (1), 1, 1)
                               proto_rf_boxes, # this will be updated
                               highly_act_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               save_dir=None,
                               prototype_activation_function_in_numpy=None,
                               color_order='BGR',
                               img_norm_mode=None,
                               dataset_name='PIE'):
    t1 = time.time()
    prototype_network_parallel.eval()

    search_batch = search_batch_input  # B 2 seqlen nj

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())  # B, C, H, W
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())  # B, np, H, W

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
    t2 = time.time()
    # print('compute time:', t2-t1)
    for j in range(n_prototypes):
        #if n_prototypes_per_class != None:
        if class_specific:  # 只project正确的样本到proto上
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()  # cls idx of cur proto
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]  # nsample of cur cls, H, W
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:,j,:,:]  # nsample of cur cls, H, W

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            closest_feat_loc = list(np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape))  # [n sample idx, h idx, w idx] for feat map
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                closest_feat_loc[0] = class_to_img_index_dict[target_class][closest_feat_loc[0]]  # n sample idx -> b idx

            # retrieve the corresponding feature map patch
            sample_index_in_batch = closest_feat_loc[0]

            fmap_height_start_index = closest_feat_loc[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = closest_feat_loc[2] * prototype_layer_stride  # *************************** ``
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[sample_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]  # C, 1, 1

            global_min_proto_dist[j] = copy.deepcopy(batch_min_proto_dist_j)
            global_min_fmap_patches[j] = copy.deepcopy(batch_min_fmap_patch_j)  # value to update the prototypes
            # push done, visualize start

            # Find the patches that generate the update of prototypes
            sp_protoL_rf_info = prototype_network_parallel.module.sp_rf_info  # [n, j, r, start]

            rf_prototype_j = compute_rf_loc_spatial(img_size=(search_batch.size(2), search_batch.size(3)),
                                                                prototype_patch_index=closest_feat_loc,
                                                                sp_rf_info=sp_protoL_rf_info)  # [sample idx, h start, h end, w start, w end]
            
            # get the whole image
            original_sample_j_ = search_batch_input[rf_prototype_j[0]]  # CHW
            original_sample_j = copy.deepcopy(original_sample_j_.numpy())
            original_sample_j = np.transpose(original_sample_j, (1, 2, 0))  # CHW -> HWC

            original_nj = original_sample_j.shape[1]
            original_seq_len = original_sample_j.shape[0]

            # crop out the receptive field
            rf_patch_j = original_sample_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4],
                                        :]  # h w 2
            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]

            proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            t_start = rf_prototype_j[1]
            t_end = rf_prototype_j[2]
            # Save the patches closest to prototypes, using cv2(BGR) instead of plt(RGB)
            proto_vis_dir = os.path.join(save_dir,  str(j) + 'th_proto')
            makedir(proto_vis_dir)

            if dataset_name == 'PIE':
                sk_source_img_path = '/home/y_feng/workspace6/datasets/PIE_dataset/skeletons/even_padded/288w_by_384h'
            elif dataset_name == 'JAAD':
                sk_source_img_path = '/home/y_feng/workspace6/datasets/JAAD/skeletons/even_padded/288w_by_384h'
            else:
                raise ValueError('illegal dataset name!')
            # save img name of the sample
            t3 = time.time()
            _ped_id_int = ped_id_int[sample_index_in_batch]
            if _ped_id_int[-1] >= 0:
                ped_id = str(_ped_id_int[0].item()) + '_' + str(_ped_id_int[1].item()) + '_' + str(_ped_id_int[2].item())
            else:
                ped_id = str(_ped_id_int[0].item()) + '_' + str(_ped_id_int[1].item()) + '_' + str(-_ped_id_int[2].item()) + 'b'
            _img_nm_int = img_nm_int[sample_index_in_batch]
            img_nm = []
            for nm_int in _img_nm_int:
                nm = str(nm_int.item())
                while len(nm) < 5:
                    nm = '0' + nm
                nm += '.png'
                img_nm.append(nm)
            content = ['ped_id: '+ped_id+'\n']+['img_nm:\n']+img_nm+['closest patch info:\n']+[str(rf_prototype_j)+'\n']
            sample_info_path = os.path.join(proto_vis_dir, '_sample_info.txt')
            with open(sample_info_path, 'w') as f:
                f.writelines(content)  # overwrite
            t4 = time.time()

            sk_img_paths = [os.path.join(sk_source_img_path, ped_id, nm) for nm in img_nm]
            
            source_imgs = [cv2.imread(path) for path in sk_img_paths]
            cur_proto_source_imgs = source_imgs[t_start: t_end]

            for i in range(len(source_imgs)):
                img = source_imgs[i]
                cv2.imwrite(filename=os.path.join(proto_vis_dir, 'ori_img_'+str(i)+'.png'), img=img)

            for i in range(t_end - t_start):
                img = cur_proto_source_imgs[i]
                joints = rf_patch_j[i] # w, 2
                for j in joints:
                    img = cv2.circle(img, (int(j[1]), int(j[0])), radius=6, color=(0,0,255), thickness=3) 
                cv2.imwrite(filename=os.path.join(proto_vis_dir, '_proto_joints_'+str(i)+'.png'), img=img)

    del protoL_input_, proto_dist_           
    if class_specific:
        del class_to_img_index_dict

def push_sk_heatmap_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    prototype_layer_stride=1,
                    prototype_info_dir='../work_dirs/latest_exp_proto', # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    log=print,
                    save_every_epoch=False,
                    img_size=(384, 288),
                    update_proto=True):
    
    makedir(prototype_info_dir)
    prototype_network_parallel.eval()
    log('push')
    # pdb.set_trace()
    # start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)  # the min dist value of each proto
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3],
         prototype_shape[4]])  # np, C, 1, 1, 1

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: time start idx
    2: time end idx
    3: height start index
    4: height end index
    5: width start index
    6: width end index
    7: (optional) class identity
    '''
    proto_rf_boxes = np.full(shape=[n_prototypes, 8],    # *************************
                                fill_value=-1)
    highly_act_boxes = np.full(shape=[n_prototypes, 8],
                                        fill_value=-1)

    if save_every_epoch and epoch_number is not None:
        proto_epoch_dir = os.path.join(prototype_info_dir, 'epoch-'+str(epoch_number))
        makedir(proto_epoch_dir)
    else:
        proto_epoch_dir = prototype_info_dir

    search_batch_size = dataloader.batch_size
    num_classes = prototype_network_parallel.module.num_classes
    dataset_name = dataloader.dataset.dataset_name
    for push_iter, data in enumerate(tqdm(dataloader, miniters=1)):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        search_y = data['pred_intent'].view(-1)
        ped_id_int = data['ped_id_int']
        img_nm_int = data['img_nm_int']
        search_batch_input = data['obs_skeletons']
        start_index_of_search_batch = push_iter * search_batch_size
        update_sk_heatmap_protos_on_batch(search_batch_input,
                                   ped_id_int=ped_id_int,
                                   img_nm_int=img_nm_int,
                                   start_index_of_search_batch=start_index_of_search_batch,
                                   prototype_network_parallel=prototype_network_parallel,
                                   global_min_proto_dist=global_min_proto_dist,
                                   global_min_fmap_patches=global_min_fmap_patches,
                                   proto_rf_boxes=proto_rf_boxes,
                                   highly_act_boxes=highly_act_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   prototype_layer_stride=prototype_layer_stride,
                                   save_dir=proto_epoch_dir,
                                   dataset_name=dataset_name,
                                   img_size=img_size)
        # time0 = time.time()
    np.save(os.path.join(proto_epoch_dir, 'prototype_info.npy'),
            proto_rf_boxes)
    np.save(os.path.join(proto_epoch_dir, 'highly_activated_info.npy'),  # *****************************
            highly_act_boxes)
    if update_proto:
        log('\tExecuting push ...')
        prototype_update = np.reshape(global_min_fmap_patches,
                                    tuple(prototype_shape))
        prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())  # UPDATE THE PROTOTYEPS
    # prototype_network_parallel.cuda()
    # end = time.time()
    # log('\tpush time: \t{0}'.format(end -  start))
    del data

# update each prototype for current search batch
def update_sk_heatmap_protos_on_batch(search_batch_input,  # input of cur batch
                               ped_id_int,
                               img_nm_int,
                               start_index_of_search_batch,  # idx of first sample in cur batch
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated, the min dist of each proto among the whole epoch  (np,)
                               global_min_fmap_patches, # this will be updated  (np, C, (1), 1, 1)
                               proto_rf_boxes, # this will be updated
                               highly_act_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               prototype_layer_stride=1,
                               save_dir=None,
                               prototype_activation_function_in_numpy=None,
                               color_order='BGR',
                               img_norm_mode=None,
                               dataset_name='PIE',
                               img_size=(384, 288)  # H, W
                               ):

    # pdb.set_trace()
    prototype_network_parallel.eval()

    heatmap_size = (search_batch_input.size(-2), search_batch_input.size(-1))  # H, W
    heatmap_ratio = (img_size[0] / heatmap_size[0], img_size[1] / heatmap_size[1])
    search_batch = search_batch_input  # BCTHW

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        _features, _dists = prototype_network_parallel.module.push_forward(search_batch)

    features = np.copy(_features.detach().cpu().numpy())  # B, C, T, H, W
    dists = np.copy(_dists.detach().cpu().numpy())  # B, np, T, H, W

    del _features, _dists

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_t = prototype_shape[2]
    proto_h = prototype_shape[3]
    proto_w = prototype_shape[4]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3] * prototype_shape[4]  # ************************* ``

    for j in range(n_prototypes):

        #if n_prototypes_per_class != None:
        if class_specific:  # 只project正确的样本到proto上
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()  # cls idx of cur proto
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = dists[class_to_img_index_dict[target_class]][:,j,:,:,:]  # nsample of cur cls, T, H, W ******************** ``
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = dists[:,j,:,:,:]

        min_dist_j = np.amin(proto_dist_j)
        if min_dist_j < global_min_proto_dist[j]:
            closest_feat_loc = list(np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape))  # [n sample idx, t idx, h idx, w idx] for feat map
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                closest_feat_loc[0] = class_to_img_index_dict[target_class][closest_feat_loc[0]]  # n sample idx -> b idx

            # retrieve the corresponding feature map patch
            sample_index_in_batch = closest_feat_loc[0]
            fmap_time_start_index = closest_feat_loc[1] * prototype_layer_stride
            fmap_time_end_index = fmap_time_start_index + proto_t
            fmap_height_start_index = closest_feat_loc[2] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = closest_feat_loc[3] * prototype_layer_stride  # *************************** ``
            fmap_width_end_index = fmap_width_start_index + proto_w

            closest_feat_patch_j = features[sample_index_in_batch,
                                                   :,
                                                   fmap_time_start_index:fmap_time_end_index,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]  # C, 1, 1, 1

            feat_j = copy.deepcopy(features[sample_index_in_batch])  # C T H W
            feat_j = np.transpose(feat_j, axes=(1, 2, 3, 0))  # T H W C

            global_min_proto_dist[j] = copy.deepcopy(min_dist_j)
            global_min_fmap_patches[j] = copy.deepcopy(closest_feat_patch_j)  # value to update the prototypes
            
            # Find the patches that generate the update of prototypes
            sp_protoL_rf_info = prototype_network_parallel.module.sp_rf_info  # [n, j, r, start]  ************************** ``
            t_protoL_rf_info = prototype_network_parallel.module.t_rf_info
            rf_prototype_j = compute_rf_loc_spatiotemporal(img_size=(search_batch.size(3), search_batch.size(4)),
                                                                seq_len=search_batch.size(2),
                                                                prototype_patch_index=closest_feat_loc,
                                                                sp_rf_info=sp_protoL_rf_info,
                                                                t_rf_info=t_protoL_rf_info)  # [b idx, tstart, tend, t, b, l, r]
            # get the whole image
            original_heatmap_j_ = search_batch_input[rf_prototype_j[0]]  # CTHW
            original_heatmap_j = copy.deepcopy(original_heatmap_j_.numpy())
            original_heatmap_j = np.transpose(original_heatmap_j, (1, 2, 3, 0))  # THWC

            original_heatmap_h = original_heatmap_j.shape[1]
            original_heatmap_w = original_heatmap_j.shape[2]
            original_seq_len = original_heatmap_j.shape[0]

            # crop out the receptive field
            rf_vid_j = original_heatmap_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4],
                                        rf_prototype_j[5]:rf_prototype_j[6],
                                        :]  # ************************* ``
            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            proto_rf_boxes[j, 5] = rf_prototype_j[5]
            proto_rf_boxes[j, 6] = rf_prototype_j[6]

            proto_rf_boxes[j, 7] = search_y[rf_prototype_j[0]].item()

            t_start = rf_prototype_j[1]
            t_end = rf_prototype_j[2]

            # create saving dir
            proto_vis_dir = os.path.join(save_dir,  str(j) + 'th_proto')
            makedir(proto_vis_dir)
            if dataset_name == 'PIE':
                sk_source_img_path = '/home/y_feng/workspace6/datasets/PIE_dataset/skeletons/even_padded/288w_by_384h'
            elif dataset_name == 'JAAD':
                sk_source_img_path = '/home/y_feng/workspace6/datasets/JAAD/skeletons/even_padded/288w_by_384h'
            else:
                raise NotImplementedError('illegal dataset name!')

            # get ped id, image name, then ori images
            _ped_id_int = ped_id_int[sample_index_in_batch]
            if _ped_id_int[-1] >= 0:
                ped_id = str(_ped_id_int[0].item()) + '_' + str(_ped_id_int[1].item()) + '_' + str(_ped_id_int[2].item())
            else:
                ped_id = str(_ped_id_int[0].item()) + '_' + str(_ped_id_int[1].item()) + '_' + str(-_ped_id_int[2].item()) + 'b'
            _img_nm_int = img_nm_int[sample_index_in_batch]
            img_nm = []
            for nm_int in _img_nm_int:
                nm = str(nm_int.item())
                while len(nm) < 5:
                    nm = '0' + nm
                nm += '.png'
                img_nm.append(nm)
            sk_img_paths = [os.path.join(sk_source_img_path, ped_id, nm) for nm in img_nm]

            source_imgs = [cv2.imread(path) for path in sk_img_paths]
            source_imgs = np.stack(source_imgs, axis=0)
            cur_proto_source_imgs = source_imgs[t_start: t_end]


            # save ori imgs
            for i in range(len(source_imgs)):
                img = source_imgs[i]
                cv2.imwrite(filename=os.path.join(proto_vis_dir, 'ori_img_'+str(i)+'.png'), img=img)


            # crop and save rf from ori imgs
            t_start_ori = rf_prototype_j[1]
            t_end_ori = rf_prototype_j[2]
            h_start_ori = int(rf_prototype_j[3] * heatmap_ratio[0])
            h_end_ori = int(rf_prototype_j[4] * heatmap_ratio[0])
            w_start_ori = int(rf_prototype_j[5] * heatmap_ratio[1])
            w_end_ori = int(rf_prototype_j[6] * heatmap_ratio[1])

            src_img_rf = source_imgs[t_start_ori: t_end_ori,
                                     h_start_ori: h_end_ori,
                                     w_start_ori: w_end_ori]
            for i in range(src_img_rf.shape[0]):
                img_j = src_img_rf[i]
                cv2.imwrite(os.path.join(proto_vis_dir,
                                        'closest_patch_' + str(i) + '.png'),
                            img_j)

            # find the highly activated region of the original image
            min_dist_map_j = dists[sample_index_in_batch, j, :, :, :]  # thw
            if prototype_network_parallel.module.prototype_activation_function == 'log':
                act_min_dist_map_j = np.log((min_dist_map_j + 1) / (min_dist_map_j + prototype_network_parallel.module.epsilon))
            elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                act_min_dist_map_j = max_dist - min_dist_map_j
            elif prototype_network_parallel.module.prototype_activation_function == 'cos':
                act_min_dist_map_j = -min_dist_map_j
            else:
                raise NotImplementedError(prototype_network_parallel.module.prototype_activation_function)
            # save the numpy array of activated feat map
            np.save(os.path.join(proto_vis_dir,
                                 'activated_dist_map_of_closest_sample.npy'),
                    act_min_dist_map_j)


            # resize dist map
            resized_sample_act_j = scipy.ndimage.zoom(act_min_dist_map_j, zoom=[original_seq_len / act_min_dist_map_j.shape[0],
                                                                                    img_size[0] / act_min_dist_map_j.shape[1],
                                                                                    img_size[1] / act_min_dist_map_j.shape[2]],
                                                                                    order=2)  # thw -> THW
            assert resized_sample_act_j.shape == (original_seq_len, img_size[0], img_size[1]), ((original_seq_len, img_size[0], img_size[1]), resized_sample_act_j.shape)
            highly_act_info_j = find_high_activation_crop_spatiotemporal(resized_sample_act_j)  # s, e, t, b, l, r
            
            # # convert highly act region to heatmap scale
            # highly_act_info_j[2] = int(highly_act_info_j[2] / heatmap_ratio[0])
            # highly_act_info_j[3] = int(highly_act_info_j[3] / heatmap_ratio[0])
            # highly_act_info_j[4] = int(highly_act_info_j[4] / heatmap_ratio[1])
            # highly_act_info_j[5] = int(highly_act_info_j[5] / heatmap_ratio[1])
            # crop out the image patch with high activation as prototype image
            highly_act_patch_j = source_imgs[highly_act_info_j[0]:highly_act_info_j[1],
                                         highly_act_info_j[2]:highly_act_info_j[3],
                                         highly_act_info_j[4]:highly_act_info_j[5],
                                          :]  # patch on ori img
            # save the prototype boundary (rectangular boundary of highly activated region)
            highly_act_boxes[j, 0] = proto_rf_boxes[j, 0]
            highly_act_boxes[j, 1] = highly_act_info_j[0]
            highly_act_boxes[j, 2] = highly_act_info_j[1]
            highly_act_boxes[j, 3] = highly_act_info_j[2]
            highly_act_boxes[j, 4] = highly_act_info_j[3]
            highly_act_boxes[j, 5] = search_y[rf_prototype_j[0]].item()


            # save highly activated patches
            for i in range(highly_act_patch_j.shape[0]):
                patch = highly_act_patch_j[i]
                # import pdb; pdb.set_trace()
                cv2.imwrite(os.path.join(proto_vis_dir, 
                                        'highly_act_patch'+str(i)+'.png'),
                            patch)

            # overlay (upsampled) self activation on original image and save the result
            normed_sample_act_j = resized_sample_act_j - np.amin(resized_sample_act_j)
            normed_sample_act_j = normed_sample_act_j / np.amax(normed_sample_act_j)
            for i in range(normed_sample_act_j.shape[0]):  # idx in original seq len
                mask = normed_sample_act_j[i]
                heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
                heatmap = 0.3*heatmap + 0.5*source_imgs[i]
                cv2.imwrite(os.path.join(proto_vis_dir,
                                         '_dist_heatmap'+str(i)+'.png'),
                            heatmap)
        
            # ori heatmap
            original_heatmap_j = torch.from_numpy(original_heatmap_j).permute(3, 0, 1, 2).contiguous()  # T, 96, 72, 17 -> 17, T, 96, 72
            original_heatmap_j = torch.unsqueeze(original_heatmap_j, 0)  # 17, T, 96, 72 -> 1, 17, T, 96, 72
            scaled_ori_heatmap = torch.nn.functional.interpolate(original_heatmap_j, size=(original_seq_len, img_size[0], img_size[1]), mode='trilinear')  # 1, 17 T 96 72 -> 1, 17 T 384 288
            scaled_ori_heatmap = torch.squeeze(scaled_ori_heatmap, 0).permute(1, 2, 3, 0).numpy()  # 1, 17 T 384 288 -> T 384 288 17
            # scaled_ori_heatmap = scipy.ndimage.zoom(original_heatmap_j, zoom=[1,
            #                                                                   img_size[0] / original_heatmap_j.shape[1],
            #                                                                   img_size[1] / original_heatmap_j.shape[2],
            #                                                                   1], 
            #                                                                   order=1)  # T, 96, 72, 17 -> T 384 288 17   SLOW
            
            scaled_ori_heatmap = np.sum(scaled_ori_heatmap, axis=3, keepdims=False)
            scaled_ori_heatmap = scaled_ori_heatmap - np.amin(scaled_ori_heatmap)
            scaled_ori_heatmap = scaled_ori_heatmap / np.amax(scaled_ori_heatmap)
            for i in range(scaled_ori_heatmap.shape[0]):  # idx in original seq len
                mask = scaled_ori_heatmap[i]
                heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
                heatmap = 0.3*heatmap + 0.5*source_imgs[i]
                cv2.imwrite(os.path.join(proto_vis_dir,
                                         '_ori_heatmap'+str(i)+'.png'),
                            heatmap)
            t0 = time.time()
            # save proto info
            feat_mean, feat_max, feat_min = visualize_featmap3d_simple(feat_j, source_imgs, mode='mean', save_dir=proto_vis_dir)
            t1 = time.time()
            feat_info = {'mean': feat_mean,
                         'max': feat_max,
                         'min': feat_min,
                         'ped_id': ped_id,
                         'img_nm': img_nm,
                         'closest_path_info': rf_prototype_j,
                         'highly_act': highly_act_info_j}
            # content = ['ped_id: '+ped_id+'\n']+['img_nm:\n']+img_nm+['closest patch info:\n']+[str(rf_prototype_j)+'\n']+['highly act info:\n']+[str(highly_act_info_j)+'\n']
            feat_info_path = os.path.join(proto_vis_dir, '_feat_info.txt')
            with open(feat_info_path, 'w') as f:
                f.writelines(str(feat_info))  # overwrite
            t2 = time.time()
            # tqdm.write('  '.join(list(map(str, ['time', t1-t0, t2-t1]))))
    del features, dists
    if class_specific:
        del class_to_img_index_dict
