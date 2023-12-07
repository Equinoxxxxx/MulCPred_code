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
from torchvision.transforms import functional as tvf

from helpers import makedir
from utils import ped_id_int2str, seg_context_batch3d, visualize_featmap3d_simple, draw_traj_on_img, draw_boxes_on_img, ltrb2xywh, vid_id_int2str, img_nm_int2str, write_info_txt
from tools.data._img_mean_std import img_mean_std
from tools.plot import vis_ego_sample
from tools.plot import EGO_RANGE

def SLE_explaine(model,
                 dataloader,
                 device,
                 use_img,
                 use_skeleton,
                 use_context,
                 ctx_mode,
                 seg_cls_idx,
                 use_traj,
                 use_ego,
                 log=print,
                 save_dir='',
                 epoch_number=None,
                 num_explain=5,
                 vis_feat_mode='mean',
                 norm_traj=1,
                 required_labels=['atomic_actions', 'simple_context']
                 ):
    makedir(save_dir)
    model.eval()
    log('explain')
    proto_dim = model.module.proto_dim
    total_num_proto = model.module.total_num_proto
    global_max_simi = {}
    if use_img:
        global_max_simi['img'] = np.full(shape=[model.module.img_model.num_proto, num_explain], fill_value=-float('inf'))
    if use_skeleton:
        global_max_simi['skeleton'] = np.full(shape=[model.module.sk_model.num_proto, num_explain], fill_value=-float('inf'))
    if use_context:
        if ctx_mode == 'seg_multi' or ctx_mode == 'local_seg_multi':
            global_max_simi['context'] = []
            for i in range(len(seg_cls_idx)):
                global_max_simi['context'].append(np.full(shape=[model.module.ctx_model[i].num_proto, num_explain], fill_value=-float('inf')))
        else:
            global_max_simi['context'] = np.full(shape=[model.module.ctx_model.num_proto, num_explain], fill_value=-float('inf'))
    if use_traj:
        global_max_simi['traj'] = np.full(shape=[model.module.traj_model.num_proto, num_explain], fill_value=-float('inf'))
    if use_ego:
        global_max_simi['ego'] = np.full(shape=[model.module.ego_model.num_proto, num_explain], fill_value=-float('inf'))
    
    if epoch_number is not None:
        proto_epoch_dir = os.path.join(save_dir, 
                                    'epoch-'+str(epoch_number))
    else:
        proto_epoch_dir = save_dir
    makedir(proto_epoch_dir)
    batch_size = dataloader.batch_size
    
    dataset_name = dataloader.dataset.dataset_name

    for i, data in enumerate(tqdm(dataloader)):
        labels = {}
        target = data['pred_intent'].view(-1) # idx, not one hot
        labels['target'] = target
        if dataset_name == 'TITAN':
            labels['atomic_actions'] = data['atomic_actions']
            labels['simple_context'] = data['simple_context']
            labels['complex_context'] = data['complex_context']
            labels['communicative'] = data['communicative']
            labels['transporting'] = data['transporting']
            labels['age'] = data['age']
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

        batch_start_idx = i * batch_size
        # dataloader.dataset.color_order = dataloader.dataset.color_order
        # dataloader.dataset.img_norm_mode = dataloader.dataset.img_norm_mode

        # cls_to_batch_idx = {key: [] for key in range(model.module.num_classes)}
        # for img_index, img_y in enumerate(target):
        #     img_label = img_y.item()
        #     cls_to_batch_idx[img_label].append(img_index)

        with torch.no_grad():
            if use_traj:
                if model.module.traj_model.simi_func in ('ego_gen_channel_att+linear', 'traj_gen_channel_att+linear', 'channel_att+linear', 'channel_att+mlp'):
                    traj_ori_feats = model.module.traj_model.backbone(inputs['traj'])
                    traj_simi, _, _, traj_protos = model.module.traj_model(inputs['traj'])
                elif model.module.traj_model.simi_func in ('fix_proto1', 'fix_proto2'):
                    traj_simi = model.module.traj_model(inputs['traj'])
                    traj_ori_feats = None
                    traj_protos = None
                else:
                    traj_ori_feats = model.module.traj_model.proto_backbone(inputs['traj'])
                    traj_simi, _, _, traj_protos = model.module.traj_model(inputs['traj'])
                traj_modal_dir = os.path.join(proto_epoch_dir, 'traj')
                explain_batch_modality(model=model.module.traj_model,
                                       inputs=inputs,
                                       ori_feats=traj_ori_feats,
                                       simis=traj_simi,
                                       protos=traj_protos,
                                       labels=labels,
                                       ctx_mode=ctx_mode,
                                       global_max_simi_modal=global_max_simi['traj'],
                                       modal_dir=traj_modal_dir,
                                    #    cls_to_batch_idx=cls_to_batch_idx,
                                       num_proto=model.module.traj_model.num_proto,
                                       modality='traj',
                                       color_order=dataloader.dataset.color_order,
                                       img_norm_mode=dataloader.dataset.img_norm_mode,
                                       simi_func=model.module.traj_model.simi_func,
                                       log=log,
                                       dataset_name=dataset_name,
                                       norm_traj=norm_traj
                                       )
            if use_ego:
                if model.module.ego_model.simi_func in ('ego_gen_channel_att+linear', 'traj_gen_channel_att+linear', 'channel_att+linear', 'channel_att+mlp'):
                    ego_ori_feats = model.module.ego_model.backbone(inputs['ego'])
                    ego_simi, _, _, ego_protos = model.module.ego_model(inputs['ego'])
                elif model.module.ego_model.simi_func in ('fix_proto1', 'fix_proto2'):
                    ego_simi = model.module.ego_model(inputs['ego'])
                    ego_ori_feats = None
                    ego_protos = None
                else:
                    ego_ori_feats = model.module.ego_model.proto_backbone(inputs['ego'])
                    ego_simi, _, _, ego_protos = model.module.ego_model(inputs['ego'])
                ego_modal_dir = os.path.join(proto_epoch_dir, 'ego')
                explain_batch_modality(model=model.module.ego_model,
                                        inputs=inputs,
                                       ori_feats=ego_ori_feats,
                                       simis=ego_simi,
                                       protos=ego_protos,
                                       labels=labels,
                                        ctx_mode=ctx_mode,
                                       global_max_simi_modal=global_max_simi['ego'],
                                       modal_dir=ego_modal_dir,
                                    #    cls_to_batch_idx=cls_to_batch_idx,
                                       num_proto=model.module.ego_model.num_proto,
                                       modality='ego',
                                       color_order=dataloader.dataset.color_order,
                                       img_norm_mode=dataloader.dataset.img_norm_mode,
                                       simi_func=model.module.ego_model.simi_func,
                                       log=log,
                                       dataset_name=dataset_name,
                                       norm_traj=norm_traj
                                       )
            if use_img:
                extra_prior = None
                if model.module.img_model.simi_func == 'ego_gen_channel_att+linear':
                    extra_prior = ego_ori_feats
                    img_ori_feats = model.module.img_model.backbone(inputs['img'])
                    img_simi, _, _, img_protos = model.module.img_model(inputs['img'], extra_prior=extra_prior)  # B num_p
                elif model.module.img_model.simi_func == 'traj_gen_channel_att+linear':
                    extra_prior = traj_ori_feats
                    img_ori_feats = model.module.img_model.backbone(inputs['img'])
                    img_simi, _, _, img_protos = model.module.img_model(inputs['img'], extra_prior=extra_prior)  # B num_p
                elif model.module.img_model.simi_func in ('channel_att+linear', 'channel_att+mlp'):
                    img_ori_feats = model.module.img_model.backbone(inputs['img'])
                    img_simi, _, _, img_protos = model.module.img_model(inputs['img'], 
                                                                        extra_prior=extra_prior)  # B num_p
                elif model.module.img_model.simi_func in ('fix_proto1', 'fix_proto2'):
                    img_simi, img_protos, img_att_map = model.module.img_model(inputs['img'], 
                                                                               extra_prior=extra_prior)  # B num_p
                    img_ori_feats = img_att_map
                else:
                    img_ori_feats = model.module.img_model.proto_backbone(inputs['img'])
                    img_simi, _, _, img_protos = model.module.img_model(inputs['img'])  # B num_p
                img_modal_dir = os.path.join(proto_epoch_dir, 'img')
                explain_batch_modality(model=model.module.img_model,
                                       inputs=inputs,
                                       ori_feats=img_ori_feats,
                                       simis=img_simi,
                                       protos=img_protos,
                                       labels=labels,
                                        ctx_mode=ctx_mode,
                                       global_max_simi_modal=global_max_simi['img'],
                                       modal_dir=img_modal_dir,
                                    #    cls_to_batch_idx=cls_to_batch_idx,
                                       num_proto=model.module.img_model.num_proto,
                                       modality='img',
                                       color_order=dataloader.dataset.color_order,
                                       img_norm_mode=dataloader.dataset.img_norm_mode,
                                       vis_feat_mode=vis_feat_mode,
                                       simi_func=model.module.img_model.simi_func,
                                       log=log,
                                       dataset_name=dataset_name,
                                       norm_traj=norm_traj
                                       )
            if use_skeleton:
                if model.module.sk_model.simi_func in ('ego_gen_channel_att+linear', 'traj_gen_channel_att+linear', 'channel_att+linear', 'channel_att+mlp'):
                    sk_ori_feats = model.module.sk_model.backbone(inputs['skeleton'])
                    sk_simi, _, _, sk_protos = model.module.sk_model(inputs['skeleton'])
                elif model.module.sk_model.simi_func in ('fix_proto1', 'fix_proto2'):
                    sk_simi, sk_protos, sk_att_map = model.module.sk_model(inputs['skeleton'])  # B num_p
                    sk_ori_feats = sk_att_map
                else:
                    sk_ori_feats = model.module.sk_model.proto_backbone(inputs['skeleton'])
                    sk_simi, _, _, sk_protos = model.module.sk_model(inputs['skeleton'])
                sk_modal_dir = os.path.join(proto_epoch_dir, 'skeleton')
                explain_batch_modality(model=model.module.sk_model,
                                       inputs=inputs,
                                       ori_feats=sk_ori_feats,
                                       simis=sk_simi,
                                       protos=sk_protos,
                                       labels=labels,
                                        ctx_mode=ctx_mode,
                                       global_max_simi_modal=global_max_simi['skeleton'],
                                       modal_dir=sk_modal_dir,
                                    #    cls_to_batch_idx=cls_to_batch_idx,
                                       num_proto=model.module.sk_model.num_proto,
                                       modality='skeleton',
                                       color_order=dataloader.dataset.color_order,
                                       img_norm_mode=dataloader.dataset.img_norm_mode,
                                       simi_func=model.module.sk_model.simi_func,
                                       log=log,
                                       dataset_name=dataset_name,
                                       norm_traj=norm_traj
                                       )
            if use_context:
                extra_prior = None
                if model.module.ctx_model.simi_func == 'ego_gen_channel_att+linear':
                    extra_prior = ego_ori_feats
                elif model.module.ctx_model.simi_func == 'traj_gen_channel_att+linear':
                    extra_prior = traj_ori_feats

                if ctx_mode == 'seg_multi' or ctx_mode == 'local_seg_multi':
                    for i in range(len(seg_cls_idx)):
                        cur_input = inputs['context'][:, :, :, :, :, i]
                        if model.module.ctx_model.simi_func in ('ego_gen_channel_att+linear', 'traj_gen_channel_att+linear', 'channel_att+linear', 'channel_att+mlp'):
                            ctx_ori_feats = model.module.ctx_model[i].backbone(cur_input)
                        else:
                            ctx_ori_feats = model.module.ctx_model[i].proto_backbone(cur_input)
                        ctx_simi, _, _, ctx_protos = model.module.ctx_model[i](cur_input, extra_prior=extra_prior)
                        ctx_modal_dir = os.path.join(proto_epoch_dir, 'context', str(i)+'th_class')
                        explain_batch_modality(model=model.module.ctx_model,
                                               inputs=inputs,
                                        ori_feats=ctx_ori_feats,
                                        simis=ctx_simi,
                                        protos=ctx_protos,
                                        labels=target,
                                        ctx_mode=ctx_mode,
                                        global_max_simi_modal=global_max_simi['context'][i],
                                        modal_dir=ctx_modal_dir,
                                        # cls_to_batch_idx=cls_to_batch_idx,
                                        num_proto=model.module.ctx_model[i].num_proto,
                                        modality='context',
                                        color_order=dataloader.dataset.color_order,
                                        img_norm_mode=dataloader.dataset.img_norm_mode,
                                        vis_feat_mode=vis_feat_mode,
                                        simi_func=model.module.ctx_model[i].simi_func,
                                        log=log,
                                        dataset_name=dataset_name,
                                        seg_idx=i,
                                        norm_traj=norm_traj
                                        )
                else:
                    if model.module.ctx_model.simi_func in ('ego_gen_channel_att+linear', 'traj_gen_channel_att+linear', 'channel_att+linear', 'channel_att+mlp'):
                        ctx_ori_feats = model.module.ctx_model.backbone(inputs['context'])
                        ctx_simi, _, _, ctx_protos = model.module.ctx_model(inputs['context'], extra_prior=extra_prior)
                    elif model.module.ctx_model.simi_func in ('fix_proto1', 'fix_proto2'):
                        ctx_simi, ctx_protos, ctx_att_map = model.module.ctx_model(inputs['context'])  # B num_p
                        ctx_ori_feats = ctx_att_map
                    else:
                        ctx_ori_feats = model.module.ctx_model.proto_backbone(inputs['context'])
                        ctx_simi, _, _, ctx_protos = model.module.ctx_model(inputs['context'], extra_prior=extra_prior)
                    ctx_modal_dir = os.path.join(proto_epoch_dir, 'context')
                    explain_batch_modality(model=model.module.ctx_model,
                                           inputs=inputs,
                                        ori_feats=ctx_ori_feats,
                                        simis=ctx_simi,
                                        protos=ctx_protos,
                                        labels=labels,
                                        ctx_mode=ctx_mode,
                                        global_max_simi_modal=global_max_simi['context'],
                                        modal_dir=ctx_modal_dir,
                                        # cls_to_batch_idx=cls_to_batch_idx,
                                        num_proto=model.module.ctx_model.num_proto,
                                        modality='context',
                                        color_order=dataloader.dataset.color_order,
                                        img_norm_mode=dataloader.dataset.img_norm_mode,
                                        vis_feat_mode=vis_feat_mode,
                                        simi_func=model.module.ctx_model.simi_func,
                                        log=log,
                                        dataset_name=dataset_name,
                                        norm_traj=norm_traj
                                        )


def explain_batch_modality(model,
                            inputs,
                           ori_feats,
                           simis,
                           protos,  # B n_p C
                           labels,
                           ctx_mode,
                           global_max_simi_modal,
                           modal_dir,
                        #    cls_to_batch_idx,
                           num_proto,
                           modality='context',
                           num_classes=2,
                           cls_spc=False,
                           color_order='BGR',
                           img_norm_mode='torch',
                           vis_feat_mode='mean',
                           simi_func='dot',
                           log=print,
                           dataset_name='PIE',
                           seg_idx=-1,
                           norm_traj=1
                           ):
    img_mean, img_std = img_mean_std(img_norm_mode)  # BGR
    if simi_func in ('ego_gen_channel_att+linear', 'traj_gen_channel_att+linear', 'channel_att+linear', 'channel_att+mlp'):
        vis_feat_mode = 'weighted'
    
    # traverse all protos
    for j in range(num_proto):
        if cls_spc:
            raise NotImplementedError('cls spc')
        else:
            simis_j = simis[:, j]  # B,
        simis_j = simis_j.cpu().numpy()
        cur_max_simi_j = np.max(simis_j)

        # update most similar sample
        if cur_max_simi_j > global_max_simi_modal[j, -1]:
            explain_idx = -1
            for i in range(len(global_max_simi_modal[j])):
                if cur_max_simi_j > global_max_simi_modal[j, i]:
                    explain_idx = i
                    break
            
            global_max_simi_modal[j, explain_idx] = copy.deepcopy(cur_max_simi_j)
            highest_simi_loc = list(np.unravel_index(np.argmax(simis_j, axis=None), simis_j.shape))
            sample_idx_in_batch = highest_simi_loc[0]

            # augmentation paras of cur sample
            hflip_flag = copy.deepcopy(inputs['hflip_flag'][sample_idx_in_batch].cpu().numpy())
            sk_ijhw = None
            img_ijhw = None
            ctx_ijhw = None
            if modality == 'skeleton':
                sk_ijhw = copy.deepcopy(inputs['sk_ijhw'][sample_idx_in_batch].cpu().numpy())
            elif modality == 'img':
                img_ijhw = copy.deepcopy(inputs['img_ijhw'][sample_idx_in_batch].cpu().numpy())
            elif modality == 'context':
                ctx_ijhw = copy.deepcopy(inputs['ctx_ijhw'][sample_idx_in_batch].cpu().numpy())
            
            # choose corresponding feat and proto
            if modality == 'img' or modality == 'context' or modality == 'skeleton':
                if simi_func not in ('fix_proto1', 'fix_proto2'):
                    feat_j = copy.deepcopy(ori_feats[sample_idx_in_batch].cpu().numpy())  # C T H W
                    feat_j = np.transpose(feat_j, axes=(1, 2, 3, 0))  # T H W C
                    proto_j = copy.deepcopy(protos[sample_idx_in_batch, j].cpu().numpy())  # C
                else:
                    feat_j = None
                    proto_j = None
                    simi_map_j = copy.deepcopy(protos[sample_idx_in_batch, j].cpu().numpy())  # T H W
                    if ori_feats is not None:
                        att_map_j = copy.deepcopy(ori_feats[sample_idx_in_batch, j].cpu().numpy())  # T H W

            if modality == 'context' and ('seg_multi' in ctx_mode):
                original_input_j = copy.deepcopy(inputs[modality][sample_idx_in_batch, :, :, :, :, seg_idx].cpu().numpy())
            else:
                original_input_j = copy.deepcopy(inputs[modality][sample_idx_in_batch].cpu().numpy())  # C T H W
            original_seq_len = original_input_j.shape[1]  # C T H W
            
            # de-process skeleton heatmap
            if modality == 'skeleton':
                original_input_j = np.transpose(original_input_j, (1, 2, 3, 0))  # CTHW -> THWC
            # de-process orininal img
            if modality == 'img' or modality == 'context':
                original_input_j = np.transpose(original_input_j, (1, 2, 3, 0))  # CTHW -> THWC
                # RGB -> BGR
                if color_order == 'RGB':
                    original_input_j = original_input_j[:, :, :, ::-1]
                # De-normalize
                if img_norm_mode != 'ori':
                    original_input_j[:,:,:,0] *= img_std[0]
                    original_input_j[:,:,:,1] *= img_std[1]
                    original_input_j[:,:,:,2] *= img_std[2]
                    original_input_j[:,:,:,0] += img_mean[0]
                    original_input_j[:,:,:,1] += img_mean[1]
                    original_input_j[:,:,:,2] += img_mean[2]
                    original_input_j *= 255.
            
            proto_vis_dir = os.path.join(modal_dir,  
                                         str(j) + 'th_proto', 
                                         str(explain_idx))
            makedir(proto_vis_dir)
            if not hasattr(model, 'score_sum_linear') or model.score_sum_linear:
                score_linear_vis_dir = os.path.join(proto_vis_dir, 
                                                    'score_linear')
                makedir(score_linear_vis_dir)

            # save img name of the sample
            labels_cur_sample = {}
            for k in labels.keys():
                labels_cur_sample[k] = labels[k][sample_idx_in_batch].cpu().numpy()
            _vid_id_int = inputs['vid_id_int'][sample_idx_in_batch]
            _ped_id_int = inputs['ped_id_int'][sample_idx_in_batch]
            if dataset_name in ('PIE', 'JAAD'):
                if _ped_id_int[-1] >= 0:
                    ped_id = str(_ped_id_int[0].item()) + '_' + str(_ped_id_int[1].item()) + '_' + str(_ped_id_int[2].item())
                else:
                    ped_id = str(_ped_id_int[0].item()) + '_' + str(_ped_id_int[1].item()) + '_' + str(- _ped_id_int[2].item()) + 'b'
            elif dataset_name == 'TITAN':
                ped_id = str(_ped_id_int)
            _img_nm_int = inputs['img_nm_int'][sample_idx_in_batch]
            img_nm = []
            for nm_int in _img_nm_int:
                nm = int(nm_int.item())
                img_nm.append(img_nm_int2str(nm))
            content = ['vid_id: '+str(_vid_id_int)+'\n']+['ped_id: '+ped_id+'\n']+['img_nm:\n']+img_nm+['\n'+str(labels_cur_sample)]+['\nhflip:'+str(hflip_flag)]+['\nsk_ijhw:'+str(sk_ijhw)] +\
                ['\nimg_ijhw:'+str(img_ijhw)] + ['\nctx_ijhw:'+str(ctx_ijhw)] + ['\ncur proto simi value:'+str(cur_max_simi_j)]
            if modality == 'ego':
                content.append(f'\nori input: {original_input_j}')
            sample_info_path = os.path.join(proto_vis_dir, '_sample_info.txt')
            with open(sample_info_path, 'w') as f:
                f.writelines(content)  # overwrite

            # root path for background
            if dataset_name == 'PIE':
                root_path = '/home/y_feng/workspace6/datasets/PIE_dataset'
                img_root_path = os.path.join(root_path, 'images')
            elif dataset_name == 'JAAD':
                root_path = '/home/y_feng/workspace6/datasets/JAAD'
                img_root_path = os.path.join(root_path, 'images')
            elif dataset_name == 'TITAN':
                root_path = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset'
                img_root_path = os.path.join(root_path, 'images_anonymized')
            
            # Visualize
            print_flag = False
            # visualize img or ctx
            if modality == 'img' or modality == 'context':
                # visualize ori img
                for i in range(original_seq_len):
                    ori_img = original_input_j[i]
                    cv2.imwrite(os.path.join(proto_vis_dir, 
                                            'ori_sample_img'+str(i) + '.png'),
                                ori_img)
                # visualize heatmap of feature itself
                if j == 0:
                    print_flag = True
                
                if print_flag and False:
                    log(str(j) + str(modality))
                    log('feat j t0 c0')
                    log(str(feat_j[0, :, :, 0]))
                    log('feat j t0 c-1')
                    log(str(feat_j[0, :, :, -1]))
                    mean_feat = np.mean(feat_j, axis=(0, 1, 2))
                    log('max channel idx' + str(np.argmax(mean_feat)))
                if simi_func not in ('fix_proto1', 'fix_proto2'):
                    feat_mean, feat_max, feat_min = visualize_featmap3d_simple(feat_j, original_input_j, 
                                                                            mode=vis_feat_mode, 
                                                                            channel_weights=proto_j, 
                                                                            save_dir=proto_vis_dir, 
                                                                            print_flag=print_flag,
                                                                            log=log)
                    write_info_txt(content_list=[str({'mean':feat_mean, 'max':feat_max, 'min':feat_min})], 
                                    dir=os.path.join(proto_vis_dir, '_feat_info.txt'))
                    mean_feat_vis_dir = os.path.join(proto_vis_dir, 'mean_feat')
                    min_feat_vis_dir = os.path.join(proto_vis_dir, 'min')
                    max_feat_vis_dir = os.path.join(proto_vis_dir, 'max')
                    makedir(mean_feat_vis_dir)
                    makedir(min_feat_vis_dir)
                    makedir(max_feat_vis_dir)
                    mean_mean, mean_max, mean_min = visualize_featmap3d_simple(feat_j, original_input_j, 
                                                                            mode='mean', 
                                                                            channel_weights=proto_j, 
                                                                            save_dir=mean_feat_vis_dir, 
                                                                            print_flag=print_flag,
                                                                            log=log)
                    write_info_txt(content_list=[str({'mean':mean_mean, 'max':mean_max, 'min':mean_min})], 
                                    dir=os.path.join(mean_feat_vis_dir, '_feat_info.txt'))
                    min_mean, min_max, min_min = visualize_featmap3d_simple(feat_j, original_input_j, 
                                                                            mode='min', 
                                                                            channel_weights=proto_j, 
                                                                            save_dir=min_feat_vis_dir, 
                                                                            print_flag=print_flag,
                                                                            log=log)
                    write_info_txt(content_list=[str({'mean':min_mean, 'max':min_max, 'min':min_min})], 
                                    dir=os.path.join(min_feat_vis_dir, '_feat_info.txt'))
                    max_mean, max_max, max_min = visualize_featmap3d_simple(feat_j, original_input_j, 
                                                                            mode='max', 
                                                                            channel_weights=proto_j, 
                                                                            save_dir=max_feat_vis_dir, 
                                                                            print_flag=print_flag,
                                                                            log=log)
                    write_info_txt(content_list=[str({'mean':max_mean, 'max':max_max, 'min':max_min})], 
                                    dir=os.path.join(max_feat_vis_dir, '_feat_info.txt'))
                    if not hasattr(model, 'score_sum_linear') or model.score_sum_linear:
                        channel_weights = proto_j * \
                            torch.squeeze(model.sum_linear.weight).cpu().numpy()
                        feat_mean, feat_max, feat_min = \
                            visualize_featmap3d_simple(feat_j, original_input_j, 
                                                        mode=vis_feat_mode, 
                                                        channel_weights=channel_weights, 
                                                        save_dir=score_linear_vis_dir, 
                                                        print_flag=print_flag,
                                                        log=log)
                        write_info_txt(content_list=[str({'mean':feat_mean, 
                                                          'max':feat_max, 
                                                          'min':feat_min})], 
                                        dir=os.path.join(score_linear_vis_dir, 
                                                        '_feat_info.txt'))
                    if modality == 'context' and ('seg_multi' in ctx_mode):
                        pass
                else:
                    feat_mean, feat_max, feat_min = visualize_featmap3d_simple(simi_map_j, original_input_j, 
                                                                            mode='fix_proto', 
                                                                            channel_weights=proto_j, 
                                                                            save_dir=proto_vis_dir, 
                                                                            print_flag=print_flag,
                                                                            log=log)
                    write_info_txt(content_list=[str({'mean':feat_mean, 'max':feat_max, 'min':feat_min})], 
                                    dir=os.path.join(proto_vis_dir, '_simi_info.txt'))
                    if simi_func == 'fix_proto2':
                        sp_att_map_dir = os.path.join(proto_vis_dir, 'sp_att_map')
                        makedir(sp_att_map_dir)
                        feat_mean, feat_max, feat_min = visualize_featmap3d_simple(att_map_j, original_input_j, 
                                                                                mode='fix_proto', 
                                                                                channel_weights=proto_j, 
                                                                                save_dir=sp_att_map_dir, 
                                                                                print_flag=print_flag,
                                                                                log=log)
                        write_info_txt(content_list=[str({'mean':feat_mean, 'max':feat_max, 'min':feat_min})], 
                                        dir=os.path.join(sp_att_map_dir, '_att_info.txt'))
            # visualize skeleton heatmap
            if modality == 'skeleton':
                if dataset_name in ('PIE', 'JAAD'):
                    ori_img_root = os.path.join(root_path, 'cropped_images', 'even_padded', '288w_by_384h')
                    sk_img_root = os.path.join(root_path, 'sk_vis', 'even_padded', '288w_by_384h')
                    ped_id_int = inputs['ped_id_int'][sample_idx_in_batch]
                    ped_id = ped_id_int2str(ped_id_int=ped_id_int.cpu().numpy())
                    img_nm_int = inputs['img_nm_int'][sample_idx_in_batch]
                    imgs = []
                    sk_imgs = []
                    for i in range(original_seq_len):
                        img_nm = img_nm_int2str(img_nm_int[i].cpu().numpy())
                        bg_path = os.path.join(ori_img_root, ped_id, img_nm)
                        img = cv2.imread(bg_path)
                        imgs.append(img)  # H W 3
                        sk_bg_path = os.path.join(sk_img_root, ped_id, img_nm)
                        sk_img = cv2.imread(sk_bg_path)
                        sk_imgs.append(sk_img)
                elif dataset_name == 'TITAN':
                    ori_img_root = '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/cropped_images/even_padded/288w_by_384h/ped/'
                    sk_img_root = '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/sk_vis/even_padded/288w_by_384h/'
                    ped_id = str(int(inputs['ped_id_int'][sample_idx_in_batch].cpu().numpy()))
                    clip_id = str(int(inputs['vid_id_int'][sample_idx_in_batch].cpu().numpy()))
                    img_nm_int = inputs['img_nm_int'][sample_idx_in_batch]
                    imgs = []
                    sk_imgs = []
                    for i in range(original_seq_len):
                        img_nm = img_nm_int2str(img_nm_int[i].cpu().numpy(), dataset_name=dataset_name)
                        bg_path = os.path.join(ori_img_root, clip_id, ped_id, img_nm)
                        img = cv2.imread(bg_path)
                        imgs.append(img)  # H W 3
                        sk_bg_path = os.path.join(sk_img_root, clip_id, ped_id, img_nm)
                        sk_img = cv2.imread(sk_bg_path)
                        sk_imgs.append(sk_img)
                imgs = np.stack(imgs, axis=0)  # THW3
                sk_imgs = np.stack(sk_imgs, axis=0)  # THW3

                # reverse augmentation for background
                if hflip_flag:
                    imgs = torch.tensor(imgs).permute(3, 0, 1, 2)  # 3THW
                    sk_imgs = torch.tensor(sk_imgs).permute(3, 0, 1, 2)  # 3THW
                    imgs = tvf.hflip(imgs).permute(1, 2, 3, 0).numpy()
                    sk_imgs = tvf.hflip(sk_imgs).permute(1, 2, 3, 0).numpy()
                if sk_ijhw[0] >= 0:
                    imgs = torch.tensor(imgs).permute(3, 0, 1, 2)  # 3THW
                    sk_imgs = torch.tensor(sk_imgs).permute(3, 0, 1, 2)  # 3THW
                    imgs = tvf.resized_crop(imgs, sk_ijhw[0] * 4, sk_ijhw[1] * 4, sk_ijhw[2] * 4, sk_ijhw[3] * 4, size=[imgs.size(2), imgs.size(3)]).permute(1, 2, 3, 0).numpy()
                    sk_imgs = tvf.resized_crop(sk_imgs, sk_ijhw[0] * 8, sk_ijhw[1] * 6, sk_ijhw[2] * 8, sk_ijhw[3] * 6, size=[sk_imgs.size(2), sk_imgs.size(3)]).permute(1, 2, 3, 0).numpy()
                
                # overlay
                if simi_func not in ('fix_proto1', 'fix_proto2'):
                    feat_mean, feat_max, feat_min = \
                        visualize_featmap3d_simple(feat_j, sk_imgs, 
                                                    mode=vis_feat_mode, 
                                                    channel_weights=proto_j, 
                                                    save_dir=proto_vis_dir,
                                                    log=log)
                    write_info_txt(content_list=[str({'mean':feat_mean, 'max':feat_max, 'min':feat_min})], 
                                    dir=os.path.join(proto_vis_dir, '_feat_info.txt'))

                    vis_on_ori_dir = os.path.join(proto_vis_dir, 'vis_on_ori')
                    makedir(vis_on_ori_dir)
                    feat_mean_, feat_max_, feat_min_ = visualize_featmap3d_simple(feat_j, imgs, 
                                                                            mode=vis_feat_mode, 
                                                                            channel_weights=proto_j, 
                                                                            save_dir=vis_on_ori_dir,
                                                                            log=log)

                    input_heatmap_dir = os.path.join(proto_vis_dir, 'input_heatmap')
                    makedir(input_heatmap_dir)
                    feat_mean_, feat_max_, feat_min_ = visualize_featmap3d_simple(original_input_j, sk_imgs, 
                                                                            mode='mean', 
                                                                            channel_weights=proto_j, 
                                                                            save_dir=input_heatmap_dir,
                                                                            log=log)

                    mean_feat_vis_dir = os.path.join(proto_vis_dir, 'mean_feat')
                    min_feat_vis_dir = os.path.join(proto_vis_dir, 'min')
                    max_feat_vis_dir = os.path.join(proto_vis_dir, 'max')
                    makedir(min_feat_vis_dir)
                    makedir(max_feat_vis_dir)
                    makedir(mean_feat_vis_dir)
                    mean_mean, mean_max, mean_min = \
                        visualize_featmap3d_simple(feat_j, 
                                                   imgs, 
                                                    mode='mean', 
                                                    channel_weights=proto_j, 
                                                    save_dir=mean_feat_vis_dir, 
                                                    print_flag=print_flag,
                                                    log=log)
                    write_info_txt(content_list=[str({'mean':mean_mean, 'max':mean_max, 'min':mean_min})], 
                                    dir=os.path.join(mean_feat_vis_dir, '_feat_info.txt'))
                    min_mean, min_max, min_min = visualize_featmap3d_simple(feat_j, imgs, 
                                                                            mode='min', 
                                                                            channel_weights=proto_j, 
                                                                            save_dir=min_feat_vis_dir, 
                                                                            print_flag=print_flag,
                                                                            log=log)
                    write_info_txt(content_list=[str({'mean':min_mean, 'max':min_max, 'min':min_min})], 
                                    dir=os.path.join(min_feat_vis_dir, '_feat_info.txt'))
                    max_mean, max_max, max_min = visualize_featmap3d_simple(feat_j, imgs, 
                                                                            mode='max', 
                                                                            channel_weights=proto_j, 
                                                                            save_dir=max_feat_vis_dir, 
                                                                            print_flag=print_flag,
                                                                            log=log)
                    write_info_txt(content_list=[str({'mean':max_mean, 'max':max_max, 'min':max_min})], 
                                    dir=os.path.join(max_feat_vis_dir, '_feat_info.txt'))
                    if not hasattr(model, 'score_sum_linear') or model.score_sum_linear:
                        channel_weights = proto_j * \
                            torch.squeeze(model.sum_linear.weight).cpu().numpy()
                        feat_mean, feat_max, feat_min = \
                        visualize_featmap3d_simple(feat_j, sk_imgs, 
                                                    mode=vis_feat_mode, 
                                                    channel_weights=channel_weights, 
                                                    save_dir=score_linear_vis_dir,
                                                    log=log)
                        write_info_txt(content_list=[str({'mean':feat_mean, 
                                                        'max':feat_max, 
                                                        'min':feat_min})], 
                                        dir=os.path.join(score_linear_vis_dir, 
                                                        '_feat_info.txt'))

                else:
                    feat_mean, feat_max, feat_min = visualize_featmap3d_simple(simi_map_j, sk_imgs, 
                                                                            mode='fix_proto', 
                                                                            channel_weights=proto_j, 
                                                                            save_dir=proto_vis_dir,
                                                                            log=log)
                    write_info_txt(content_list=[str({'mean':feat_mean, 'max':feat_max, 'min':feat_min})], 
                                    dir=os.path.join(proto_vis_dir, '_simi_info.txt'))

                    vis_on_ori_dir = os.path.join(proto_vis_dir, 'vis_on_ori')
                    makedir(vis_on_ori_dir)
                    feat_mean_, feat_max_, feat_min_ = visualize_featmap3d_simple(simi_map_j, imgs, 
                                                                            mode='fix_proto', 
                                                                            channel_weights=proto_j, 
                                                                            save_dir=vis_on_ori_dir,
                                                                            log=log)
                    
                    input_heatmap_dir = os.path.join(proto_vis_dir, 'input_heatmap')
                    makedir(input_heatmap_dir)
                    feat_mean_, feat_max_, feat_min_ = visualize_featmap3d_simple(original_input_j, sk_imgs, 
                                                                            mode='mean', 
                                                                            channel_weights=proto_j, 
                                                                            save_dir=input_heatmap_dir,
                                                                            log=log)

                    if simi_func == 'fix_proto2':
                        sp_att_map_dir = os.path.join(proto_vis_dir, 'sp_att_map')
                        makedir(sp_att_map_dir)
                        feat_mean, feat_max, feat_min = visualize_featmap3d_simple(att_map_j, sk_imgs, 
                                                                                mode='fix_proto', 
                                                                                channel_weights=proto_j, 
                                                                                save_dir=sp_att_map_dir, 
                                                                                print_flag=print_flag,
                                                                                log=log)
                        write_info_txt(content_list=[str({'mean':feat_mean, 'max':feat_max, 'min':feat_min})], 
                                        dir=os.path.join(sp_att_map_dir, '_att_info.txt'))
            if modality == 'traj':
                # get background
                if dataset_name in ('PIE', 'JAAD'):
                    vid_id_int = inputs['vid_id_int'][sample_idx_in_batch][0].item()
                    vid_nm = vid_id_int2str(vid_id_int)
                    img_nm_int = inputs['img_nm_int'][sample_idx_in_batch][-1].item()
                    img_nm = img_nm_int2str(img_nm_int, dataset_name=dataset_name)
                    if dataset_name == 'PIE':
                        set_id_int = inputs['set_id_int'][sample_idx_in_batch][0].item()
                        set_nm = 'set0' + str(set_id_int)
                        bg_path = os.path.join(img_root_path, set_nm, vid_nm, img_nm)
                    else:
                        bg_path = os.path.join(img_root_path, vid_nm, img_nm)
                elif dataset_name == 'TITAN':
                    vid_id_int = inputs['vid_id_int'][sample_idx_in_batch].item()
                    img_nm_int = inputs['img_nm_int'][sample_idx_in_batch][-1].item()
                    vid_nm = 'clip_' + str(vid_id_int)
                    img_nm = img_nm_int2str(img_nm_int, dataset_name=dataset_name)
                    bg_path = os.path.join(img_root_path, vid_nm, 'images', img_nm)
                background = cv2.imread(filename=bg_path)
                blank_bg = np.zeros(background.shape) + 255
                # reverse augmentation for background
                if hflip_flag:
                    background = torch.tensor(background).permute(2, 0, 1)  # 3HW
                    background = tvf.hflip(background).permute(1, 2, 0).numpy()

                unnormed_traj = copy.deepcopy(inputs['traj_unnormed'][sample_idx_in_batch].cpu().numpy())
                traj_img = draw_boxes_on_img(background, unnormed_traj)
                blank_traj_img = draw_boxes_on_img(blank_bg, unnormed_traj)
                # print(background)
                # print(bg_path)
                # print(unnormed_traj)
                # print(img.shape)
                cv2.imwrite(filename=os.path.join(proto_vis_dir, 'traj.png'), 
                            img=traj_img)
                cv2.imwrite(filename=os.path.join(proto_vis_dir, 'traj_blank_bg.png'), 
                            img=blank_traj_img)
            if modality == 'ego':
                ego = inputs['ego'][sample_idx_in_batch].cpu().numpy()
                if len(ego.shape) == 2:
                    ego = ego[:, 0]
                lim = EGO_RANGE[dataset_name]
                vis_ego_sample(ego,
                               lim=lim,
                               path=os.path.join(proto_vis_dir, 'ego.png'))