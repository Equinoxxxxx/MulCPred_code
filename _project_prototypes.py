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

from receptive_field import compute_rf_loc_spatial, compute_rf_loc_spatiotemporal
from helpers import find_high_activation_crop_spatiotemporal, makedir, find_high_activation_crop
from utils import seg_context_batch3d, visualize_featmap3d_simple

# push each prototype to the nearest patch in the training set
def push_nonlocal_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    prototype_info_dir='../work_dirs/latest_exp_proto', # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    log=print,
                    prototype_activation_function_in_numpy=None,
                    save_every_epoch=False,
                    data_type='traj',
                    update_proto=True):
    
    makedir(prototype_info_dir)
    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    # saves the closest distance seen so far
    global_max_simi = np.zeros(n_prototypes)  # the max simi value of each proto
    # saves the patch representation that gives the current smallest distance
    global_closest_feat = np.zeros(
        [n_prototypes,
         prototype_shape[1]])  # np, C

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: sample index in the entire dataset
    7: (optional) class identity
    '''
    proto_info = np.full(shape=[n_prototypes, 2],    # *************************
                                fill_value=-1)

    if save_every_epoch and epoch_number is not None:
        proto_epoch_dir = os.path.join(prototype_info_dir, 'epoch-'+str(epoch_number))
        makedir(proto_epoch_dir)
    else:
        proto_epoch_dir = prototype_info_dir

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes

    for push_iter, data in enumerate(tqdm(dataloader)):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        search_y = data['pred_intent'].view(-1)
        ped_id_int = data['ped_id_int']
        img_nm_int = data['img_nm_int']
        if data_type == 'img':
            search_batch_input = data['ped_imgs']
        elif data_type == 'traj':
            search_batch_input = data['obs_bboxes']
        elif data_type == 'context':
            search_batch_input = data['obs_context']
        start_index_of_search_batch = push_iter * search_batch_size
        color_order = dataloader.dataset.color_order
        img_norm_mode = dataloader.dataset.normalize_img_mode
        update_nonlocal_prototypes_on_batch(search_batch_input,
                                   ped_id_int=ped_id_int,
                                   img_nm_int=img_nm_int,
                                   start_index_of_search_batch=start_index_of_search_batch,
                                   prototype_network_parallel=prototype_network_parallel,
                                   global_max_simi=global_max_simi,
                                   global_closest_feat=global_closest_feat,
                                   proto_info=proto_info,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   save_dir=proto_epoch_dir,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
                                   color_order=color_order,
                                   img_norm_mode=img_norm_mode,
                                   data_type=data_type)

    np.save(os.path.join(proto_epoch_dir, 'prototype_info.npy'),
            proto_info)

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_closest_feat,
                                  tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())  # UPDATE THE PROTOTYEPS
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))
    del data

def update_nonlocal_prototypes_on_batch(search_batch_input,  # input of cur batch
                               ped_id_int,
                               img_nm_int,
                               start_index_of_search_batch,  # idx of first sample in cur batch
                               prototype_network_parallel,
                               global_max_simi, # this will be updated, the min dist of each proto among the whole epoch  (np,)
                               global_closest_feat, # this will be updated  (np, C, (1), 1, 1)
                               proto_info, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               prototype_layer_stride=1,
                               save_dir=None,
                               prototype_activation_function_in_numpy=None,
                               color_order='BGR',
                               img_norm_mode=None,
                               data_type='img'):
    t1 = time.time()
    prototype_network_parallel.eval()

    search_batch = search_batch_input  # BCTHW

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        _features, _simi = prototype_network_parallel.module.push_forward(search_batch)

    features = np.copy(_features.detach().cpu().numpy())  # B, C, (T, H, W)
    simi = np.copy(_simi.detach().cpu().numpy())  # B, np

    del _features, _simi

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
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
            simi_j = simi[class_to_img_index_dict[target_class]][:,j]  # (nsample of cur cls, )
        else:
            # if it is not class specific, then we will search through
            # every example
            simi_j = simi[:,j]

        max_simi_j_cur_batch = np.amax(simi_j)
        if max_simi_j_cur_batch > global_max_simi[j]:
            closest_feat_loc = list(np.unravel_index(np.argmax(simi_j, axis=None), simi_j.shape))  # [n sample idx] for feat map
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                closest_feat_loc[0] = class_to_img_index_dict[target_class][closest_feat_loc[0]]  # n sample idx -> b idx

            # retrieve the corresponding feature map patch
            sample_index_in_batch = closest_feat_loc[0]

            closest_feat_j_cur_batch = features[sample_index_in_batch,
                                                   :]  # C, 1, 1, 1

            global_max_simi[j] = copy.deepcopy(max_simi_j_cur_batch)
            global_closest_feat[j] = copy.deepcopy(closest_feat_j_cur_batch)  # value to update the prototypes
            
            # get the whole image
            original_input_j_ = search_batch_input[sample_index_in_batch]  # CTHW / TC
            original_input_j = copy.deepcopy(original_input_j_.numpy())
            if data_type == 'img' or data_type == 'context':
                original_input_j = np.transpose(original_input_j, (1, 2, 3, 0))  # THWC
                # RGB -> BGR
                if color_order == 'RGB':
                    original_input_j = original_input_j[:, :, :, ::-1]
                # De-normalize
                if img_norm_mode == 'tf':
                    original_input_j += 1.
                    original_input_j *= 127.5
                elif img_norm_mode == 'torch':
                    original_input_j[:,:,:,0] *= 0.225
                    original_input_j[:,:,:,1] *= 0.224
                    original_input_j[:,:,:,2] *= 0.229
                    original_input_j[:,:,:,0] += 0.406
                    original_input_j[:,:,:,1] += 0.456
                    original_input_j[:,:,:,2] += 0.485
                    original_input_j *= 255.
                original_seq_len = original_input_j.shape[0]
            
            # save the prototype receptive field information
            proto_info[j, 0] = sample_index_in_batch + start_index_of_search_batch

            proto_info[j, 1] = search_y[sample_index_in_batch].item()

            # visualize
            proto_vis_dir = os.path.join(save_dir,  str(j) + 'th_proto')
            makedir(proto_vis_dir)

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
                nm += '\n'
                img_nm.append(nm)
            content = ['ped_id: '+ped_id+'\n']+['img_nm:\n']+img_nm
            sample_info_path = os.path.join(proto_vis_dir, '_sample_info.txt')
            with open(sample_info_path, 'w') as f:
                f.writelines(content)  # overwrite
            t4 = time.time()
            # print('writing txt time:', t4-t3)
            # save the whole image containing the prototype as png
            if data_type == 'img' or data_type == 'context':
                for i in range(original_seq_len):
                    ori_img = original_input_j[i]
                    cv2.imwrite(os.path.join(proto_vis_dir, 
                                            'ori_sample_img'+str(i) + '.png'),
                                ori_img)
            if data_type == 'traj':
                plt.close()
                fig = plt.figure()
                for i in range(original_seq_len):
                    l, t, r, b = original_input_j[i]
                    ax = fig.add_subplot(111)
                    plt.gca().add_patch(plt.Rectangle((l, b), r-l, b-t))
                    plt.savefig(os.path.join(proto_vis_dir, 
                                            'ori_traj'+str(i) + '.png'))
                plt.close()
                    

    del features, simi           
    if class_specific:
        del class_to_img_index_dict

def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    prototype_info_dir='../work_dirs/latest_exp_proto', # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    log=print,
                    prototype_activation_function_in_numpy=None,
                    save_every_epoch=False,
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
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3],
         prototype_shape[4]])  # ****************** ``

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

    for push_iter, data in enumerate(tqdm(dataloader)):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        search_y = data['pred_intent'].view(-1)
        ped_id_int = data['ped_id_int']
        img_nm_int = data['img_nm_int']
        search_batch_input = data['ped_imgs']
        start_index_of_search_batch = push_iter * search_batch_size
        color_order = dataloader.dataset.color_order
        img_norm_mode = dataloader.dataset.normalize_img_mode
        update_prototypes_on_batch(search_batch_input,
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
                                   color_order=color_order,
                                   img_norm_mode=img_norm_mode)

    np.save(os.path.join(proto_epoch_dir, 'prototype_info.npy'),
            proto_rf_boxes)
    np.save(os.path.join(proto_epoch_dir, 'highly_activated_info.npy'),  # *****************************
            highly_act_boxes)

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())  # UPDATE THE PROTOTYEPS
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))
    del data

# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,  # input of cur batch
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
                               img_norm_mode=None):
    t1 = time.time()
    prototype_network_parallel.eval()

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
            proto_dist_j = dists[class_to_img_index_dict[target_class]][:,j,:,:,:]  # nsample of cur cls, T, H, W ******************** ``
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = dists[:,j,:,:,:]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
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

            batch_min_fmap_patch_j = features[sample_index_in_batch,
                                                   :,
                                                   fmap_time_start_index:fmap_time_end_index,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]  # C, 1, 1, 1

            feat_j = copy.deepcopy(features[sample_index_in_batch])  # C T H W
            feat_j = np.transpose(feat_j, axes=(1, 2, 3, 0))  # T H W C

            global_min_proto_dist[j] = copy.deepcopy(batch_min_proto_dist_j)
            global_min_fmap_patches[j] = copy.deepcopy(batch_min_fmap_patch_j)  # value to update the prototypes
            

            # Find the patches that generate the update of prototypes
            sp_protoL_rf_info = prototype_network_parallel.module.sp_rf_info  # [n, j, r, start]  ************************** ``
            t_protoL_rf_info = prototype_network_parallel.module.t_rf_info
            rf_prototype_j = compute_rf_loc_spatiotemporal(img_size=(search_batch.size(3), search_batch.size(4)),
                                                                seq_len=search_batch.size(2),
                                                                prototype_patch_index=closest_feat_loc,
                                                                sp_rf_info=sp_protoL_rf_info,
                                                                t_rf_info=t_protoL_rf_info)  # [b idx, tstart, tend, t, b, l, r] ******************** ``
            
            # get the whole image
            original_vid_j_ = search_batch_input[rf_prototype_j[0]]  # CTHW
            original_vid_j = copy.deepcopy(original_vid_j_.numpy())
            original_vid_j = np.transpose(original_vid_j, (1, 2, 3, 0))  # THWC ************************* ``
            # RGB -> BGR
            if color_order == 'RGB':
                original_vid_j = original_vid_j[:, :, :, ::-1]
            # De-normalize
            if img_norm_mode == 'tf':
                original_vid_j += 1.
                original_vid_j *= 127.5
            elif img_norm_mode == 'torch':
                original_vid_j[:,:,:,0] *= 0.225
                original_vid_j[:,:,:,1] *= 0.224
                original_vid_j[:,:,:,2] *= 0.229
                original_vid_j[:,:,:,0] += 0.406
                original_vid_j[:,:,:,1] += 0.456
                original_vid_j[:,:,:,2] += 0.485
                original_vid_j *= 255.
            original_img_h = original_vid_j.shape[1]
            original_img_w = original_vid_j.shape[2]
            original_seq_len = original_vid_j.shape[0]

            # crop out the receptive field
            rf_vid_j = original_vid_j[rf_prototype_j[1]:rf_prototype_j[2],
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

            # Save the patches closest to prototypes, using cv2(BGR) instead of plt(RGB)
            proto_vis_dir = os.path.join(save_dir,  str(j) + 'th_proto')
            makedir(proto_vis_dir)
            for i in range(rf_vid_j.shape[0]):
                img_j = rf_vid_j[i]
                ori_idx = i + rf_prototype_j[1]
                cv2.imwrite(os.path.join(proto_vis_dir,
                                        'closest_patch_' + str(i) + '.png'),
                            img_j)

            # find the highly activated region of the original image
            proto_j_sample_dist = dists[sample_index_in_batch, j, :, :, :]  # THW
            if prototype_network_parallel.module.prototype_activation_function == 'log':
                proto_j_sample_dist_act = np.log((proto_j_sample_dist + 1) / (proto_j_sample_dist + prototype_network_parallel.module.epsilon))
            elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                proto_j_sample_dist_act = max_dist - proto_j_sample_dist
            elif prototype_network_parallel.module.prototype_activation_function == 'cos':
                proto_j_sample_dist_act = -proto_j_sample_dist
            else:
                proto_j_sample_dist_act = prototype_activation_function_in_numpy(proto_j_sample_dist)
            
            resized_sample_act_j = scipy.ndimage.zoom(proto_j_sample_dist_act, zoom=[original_seq_len / proto_j_sample_dist_act.shape[0],
                                                                                    original_img_h / proto_j_sample_dist_act.shape[1],
                                                                                    original_img_w / proto_j_sample_dist_act.shape[2]])
            assert resized_sample_act_j.shape == original_vid_j.shape[:3], [resized_sample_act_j.shape, original_vid_j.shape]
            highly_act_info_j = find_high_activation_crop_spatiotemporal(resized_sample_act_j)  # s, e, t, b, l, r ************************* ``

            # crop out the image patch with high activation as prototype image
            highly_act_patch_j = original_vid_j[highly_act_info_j[0]:highly_act_info_j[1],
                                         highly_act_info_j[2]:highly_act_info_j[3],
                                         highly_act_info_j[4]:highly_act_info_j[5],
                                          :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            highly_act_boxes[j, 0] = proto_rf_boxes[j, 0]
            highly_act_boxes[j, 1] = highly_act_info_j[0]
            highly_act_boxes[j, 2] = highly_act_info_j[1]
            highly_act_boxes[j, 3] = highly_act_info_j[2]
            highly_act_boxes[j, 4] = highly_act_info_j[3]
            highly_act_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # save the numpy array of activated feat map
            np.save(os.path.join(proto_vis_dir,
                                 'activated_dist_map_of_closest_sample.npy'),
                    proto_j_sample_dist_act)
            
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
                nm += '\n'
                img_nm.append(nm)
            content = ['ped_id: '+ped_id+'\n']+['img_nm:\n']+img_nm+['closest patch info:\n']+[str(rf_prototype_j)+'\n']+['highly act info:\n']+[str(highly_act_info_j)+'\n']
            sample_info_path = os.path.join(proto_vis_dir, '_sample_info.txt')
            with open(sample_info_path, 'w') as f:
                f.writelines(content)  # overwrite
            t4 = time.time()
            # print('writing txt time:', t4-t3)
            # save the whole image containing the prototype as png
            for i in range(original_seq_len):
                ori_img = original_vid_j[i]
                cv2.imwrite(os.path.join(proto_vis_dir, 
                                         'ori_sample_img'+str(i) + '.png'),
                            ori_img)
            # save highly activated patches
            for i in range(highly_act_patch_j.shape[0]):
                ori_idx = i + highly_act_info_j[0]
                patch = highly_act_patch_j[i]
                cv2.imwrite(os.path.join(proto_vis_dir, 
                                         'highly_act_patch'+str(i)+'.png'),
                            patch)

            # overlay (upsampled) self activation on original image and save the result
            normed_sample_act_j = resized_sample_act_j - np.amin(resized_sample_act_j)
            normed_sample_act_j = normed_sample_act_j / np.amax(normed_sample_act_j)

            for i in range(normed_sample_act_j.shape[0]):
                mask = normed_sample_act_j[i]
                heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
                heatmap = 0.3*heatmap + 0.5*original_vid_j[i]
                cv2.imwrite(os.path.join(proto_vis_dir,
                                         '_heatmap'+str(i)+'.png'),
                            heatmap)
            
            # visualize heatmap of feature itself
            feat_mean, feat_max, feat_min = visualize_featmap3d_simple(feat_j, original_vid_j, mode='mean', save_dir=proto_vis_dir)
            feat_info = {'mean': feat_mean,
                         'max': feat_max,
                         'min': feat_min}
            feat_info_path = os.path.join(proto_vis_dir, '_feat_info.txt')
            with open(feat_info_path, 'w') as f:
                f.writelines(str(feat_info))  # overwrite

    del features, dists           
    if class_specific:
        del class_to_img_index_dict


def push_ctx_protos(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    prototype_info_dir='../work_dirs/latest_exp_proto', # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    log=print,
                    prototype_activation_function_in_numpy=None,
                    save_every_epoch=False,
                    seg_class_i=-1,
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
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3],
         prototype_shape[4]])  # ****************** ``

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

    for push_iter, data in enumerate(tqdm(dataloader)):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        search_y = data['pred_intent'].view(-1)
        ped_id_int = data['ped_id_int']
        img_nm_int = data['img_nm_int']
        if dataloader.dataset.ctx_mode == 'seg_multi' and False:
            context = data['obs_context']
            seg = data['obs_seg']
            search_batch_input = seg_context_batch3d(seg, context, seg_class_idx=dataloader.dataset.seg_class_idx, tgt_size=dataloader.dataset.ctx_size)
        else:
            search_batch_input = data['obs_context']
        if seg_class_i >= 0:
            search_batch_input = search_batch_input[:, :, :, :, :, seg_class_i]
        start_index_of_search_batch = push_iter * search_batch_size
        color_order = dataloader.dataset.color_order
        img_norm_mode = dataloader.dataset.normalize_img_mode
        update_prototypes_on_batch(search_batch_input,
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
                                   color_order=color_order,
                                   img_norm_mode=img_norm_mode)

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

def push_single_img_protos(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
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
    1: h start idx
    2: h end idx
    3: w start index
    4: w end index
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

    for push_iter, data in enumerate(tqdm(dataloader)):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        search_y = data['pred_intent'].view(-1)
        ped_id_int = data['ped_id_int']
        img_nm_int = data['img_nm_int']
        search_batch_input = data['ped_single_imgs']
        start_index_of_search_batch = push_iter * search_batch_size
        color_order = dataloader.dataset.color_order
        img_norm_mode = dataloader.dataset.normalize_img_mode
        update_2d_protos_on_batch(search_batch_input,
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
                                   color_order=color_order,
                                   img_norm_mode=img_norm_mode)

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


def update_2d_protos_on_batch(search_batch_input,  # input of cur batch
                               ped_id_int,
                               img_nm_int,
                               start_index_of_search_batch,  # idx of first sample in cur batch
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated, the min dist of each proto among the whole epoch  (np,)
                               global_min_fmap_patches, # this will be updated  (np, C, 1, 1)
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
                               img_norm_mode=None):
    t1 = time.time()
    prototype_network_parallel.eval()

    search_batch = search_batch_input  # BCHW

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
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]  # nsample of cur cls, T, H, W ******************** ``
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:,j,:,:]

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

            # Find the patches that generate the update of prototypes
            sp_protoL_rf_info = prototype_network_parallel.module.sp_rf_info  # [n, j, r, start]  ************************** ``
            rf_prototype_j = compute_rf_loc_spatial(img_size=(search_batch.size(2), search_batch.size(3)),
                                                                prototype_patch_index=closest_feat_loc,
                                                                sp_rf_info=sp_protoL_rf_info)  # [sample idx, h start, h end, w start, w end]
            
            # get the whole image
            original_img_j_ = search_batch_input[rf_prototype_j[0]]  # CHW
            original_img_j = copy.deepcopy(original_img_j_.numpy())
            original_img_j = np.transpose(original_img_j, (1, 2, 0))  # HWC ************************* ``
            # RGB -> BGR
            if color_order == 'RGB':
                original_img_j = original_img_j[:, :, ::-1]
            # De-normalize
            if img_norm_mode == 'tf':
                original_img_j += 1.
                original_img_j *= 127.5
            elif img_norm_mode == 'torch':
                original_img_j[:,:,0] *= 0.225
                original_img_j[:,:,1] *= 0.224
                original_img_j[:,:,2] *= 0.229
                original_img_j[:,:,0] += 0.406
                original_img_j[:,:,1] += 0.456
                original_img_j[:,:,2] += 0.485
                original_img_j *= 255.
            original_img_h = original_img_j.shape[0]
            original_img_w = original_img_j.shape[1]

            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4],
                                        :]  # ************************* ``
            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]

            proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # Save the patches closest to prototypes, using cv2(BGR) instead of plt(RGB)
            proto_vis_dir = os.path.join(save_dir,  str(j) + 'th_proto')
            makedir(proto_vis_dir)
            cv2.imwrite(os.path.join(proto_vis_dir,
                                    'closest_patch_' + '.png'),
                        rf_img_j)


            # find the highly activated region of the original image
            proto_j_sample_dist = proto_dist_[sample_index_in_batch, j, :, :]  # HW
            if prototype_network_parallel.module.prototype_activation_function == 'log':
                proto_j_sample_dist_act = np.log((proto_j_sample_dist + 1) / (proto_j_sample_dist + prototype_network_parallel.module.epsilon))
            elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                proto_j_sample_dist_act = max_dist - proto_j_sample_dist
            else:
                proto_j_sample_dist_act = prototype_activation_function_in_numpy(proto_j_sample_dist)
            
            resized_sample_act_j = scipy.ndimage.zoom(proto_j_sample_dist_act, zoom=[original_img_h / proto_j_sample_dist_act.shape[0],
                                                                                    original_img_w / proto_j_sample_dist_act.shape[1]])
            assert resized_sample_act_j.shape == original_img_j.shape[:2], [resized_sample_act_j.shape, original_img_j.shape]
            highly_act_info_j = find_high_activation_crop(resized_sample_act_j)  # s, e, t, b, l, r ************************* ``

            # crop out the image patch with high activation as prototype image
            highly_act_patch_j = original_img_j[highly_act_info_j[0]:highly_act_info_j[1],
                                         highly_act_info_j[2]:highly_act_info_j[3],
                                          :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            highly_act_boxes[j, 0] = proto_rf_boxes[j, 0]
            highly_act_boxes[j, 1] = highly_act_info_j[0]
            highly_act_boxes[j, 2] = highly_act_info_j[1]
            highly_act_boxes[j, 3] = highly_act_info_j[2]
            highly_act_boxes[j, 4] = highly_act_info_j[3]
            highly_act_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # save the numpy array of activated feat map
            np.save(os.path.join(proto_vis_dir,
                                 'activated_dist_map_of_closest_sample.npy'),
                    proto_j_sample_dist_act)
            
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
                nm += '\n'
                img_nm.append(nm)
            content = ['ped_id: '+ped_id+'\n']+['img_nm:\n']+img_nm+['closest patch info:\n']+[str(rf_prototype_j)+'\n']+['highly act info:\n']+[str(highly_act_info_j)+'\n']
            sample_info_path = os.path.join(proto_vis_dir, '_sample_info.txt')
            with open(sample_info_path, 'w') as f:
                f.writelines(content)  # overwrite
            t4 = time.time()
            # print('writing txt time:', t4-t3)
            # save the whole image containing the prototype as png
            cv2.imwrite(os.path.join(proto_vis_dir, 
                                        'ori_sample_img' + '.png'),
                        original_img_j)
            # save highly activated patches
            cv2.imwrite(os.path.join(proto_vis_dir, 
                                        'highly_act_patch'+'.png'),
                        highly_act_patch_j)

            # overlay (upsampled) self activation on original image and save the result
            normed_sample_act_j = resized_sample_act_j - np.amin(resized_sample_act_j)
            normed_sample_act_j = normed_sample_act_j / np.amax(normed_sample_act_j)

            img = normed_sample_act_j
            heatmap = cv2.applyColorMap(np.uint8(255*img), cv2.COLORMAP_JET)
            heatmap = 0.3*heatmap + 0.5*original_img_j
            cv2.imwrite(os.path.join(proto_vis_dir,
                                        '_heatmap'+'.png'),
                        heatmap)
            

    del protoL_input_, proto_dist_           
    if class_specific:
        del class_to_img_index_dict