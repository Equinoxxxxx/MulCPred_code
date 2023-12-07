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

def SENN_explaine(model,
                 dataloader,
                 device,
                 use_img,
                 use_skeleton,
                 use_context,
                 ctx_mode,
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
    global_max_simi = {}
    if use_img:
        global_max_simi['img'] = np.full(shape=[model.module.img_model.num_proto, num_explain], fill_value=-float('inf'))
    if use_skeleton:
        global_max_simi['skeleton'] = np.full(shape=[model.module.sk_model.num_proto, num_explain], fill_value=-float('inf'))
    if use_context:
        global_max_simi['context'] = np.full(shape=[model.module.ctx_model.num_proto, num_explain], fill_value=-float('inf'))
    if use_traj:
        global_max_simi['traj'] = np.full(shape=[model.module.traj_model.num_proto, num_explain], fill_value=-float('inf'))
    if use_ego:
        global_max_simi['ego'] = np.full(shape=[model.module.ego_model.num_proto, num_explain], fill_value=-float('inf'))
    
    proto_epoch_dir = os.path.join(save_dir, 'epoch-'+str(epoch_number))
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

        cls_to_batch_idx = {key: [] for key in range(model.module.num_classes)}
        for img_index, img_y in enumerate(target):
            img_label = img_y.item()
            cls_to_batch_idx[img_label].append(img_index)

        with torch.no_grad():
            if use_traj:
                traj_simi, _, _ = model.module.traj_model(inputs['traj'])
                traj_modal_dir = os.path.join(proto_epoch_dir, 'traj')
                SENN_explain_batch_modality(inputs=inputs,
                                       simis=traj_simi,
                                       labels=labels,
                                       ctx_mode=ctx_mode,
                                       global_max_simi_modal=global_max_simi['traj'],
                                       modal_dir=traj_modal_dir,
                                       num_proto=model.module.traj_model.num_proto,
                                       modality='traj',
                                       color_order=dataloader.dataset.color_order,
                                       img_norm_mode=dataloader.dataset.img_norm_mode,
                                       log=log,
                                       dataset_name=dataset_name,
                                       norm_traj=norm_traj
                                       )
            if use_ego:
                ego_simi, _, _ = model.module.ego_model(inputs['ego'])
                ego_modal_dir = os.path.join(proto_epoch_dir, 'ego')
                SENN_explain_batch_modality(inputs=inputs,
                                       simis=ego_simi,
                                       labels=labels,
                                        ctx_mode=ctx_mode,
                                       global_max_simi_modal=global_max_simi['ego'],
                                       modal_dir=ego_modal_dir,
                                       num_proto=model.module.ego_model.num_proto,
                                       modality='ego',
                                       color_order=dataloader.dataset.color_order,
                                       img_norm_mode=dataloader.dataset.img_norm_mode,
                                       log=log,
                                       dataset_name=dataset_name,
                                       norm_traj=norm_traj
                                       )
            if use_img:
                img_simi, _, _ = model.module.img_model(inputs['img'])  # B num_p
                img_modal_dir = os.path.join(proto_epoch_dir, 'img')
                SENN_explain_batch_modality(inputs=inputs,
                                       simis=img_simi,
                                       labels=labels,
                                        ctx_mode=ctx_mode,
                                       global_max_simi_modal=global_max_simi['img'],
                                       modal_dir=img_modal_dir,
                                       num_proto=model.module.img_model.num_proto,
                                       modality='img',
                                       color_order=dataloader.dataset.color_order,
                                       img_norm_mode=dataloader.dataset.img_norm_mode,
                                       log=log,
                                       dataset_name=dataset_name,
                                       norm_traj=norm_traj
                                       )
            if use_skeleton:
                sk_simi, _, _ = model.module.sk_model(inputs['skeleton'])
                sk_modal_dir = os.path.join(proto_epoch_dir, 'skeleton')
                SENN_explain_batch_modality(inputs=inputs,
                                       simis=sk_simi,
                                       labels=labels,
                                        ctx_mode=ctx_mode,
                                       global_max_simi_modal=global_max_simi['skeleton'],
                                       modal_dir=sk_modal_dir,
                                       num_proto=model.module.sk_model.num_proto,
                                       modality='skeleton',
                                       color_order=dataloader.dataset.color_order,
                                       img_norm_mode=dataloader.dataset.img_norm_mode,
                                       log=log,
                                       dataset_name=dataset_name,
                                       norm_traj=norm_traj
                                       )
            if use_context:
                ctx_simi, _, _ = model.module.ctx_model(inputs['context'])
                ctx_modal_dir = os.path.join(proto_epoch_dir, 'context')
                SENN_explain_batch_modality(inputs=inputs,
                                    simis=ctx_simi,
                                    labels=labels,
                                    ctx_mode=ctx_mode,
                                    global_max_simi_modal=global_max_simi['context'],
                                    modal_dir=ctx_modal_dir,
                                    num_proto=model.module.ctx_model.num_proto,
                                    modality='context',
                                    color_order=dataloader.dataset.color_order,
                                    img_norm_mode=dataloader.dataset.img_norm_mode,
                                    log=log,
                                    dataset_name=dataset_name,
                                    norm_traj=norm_traj
                                    )


def SENN_explain_batch_modality(inputs,
                           simis,  # B np
                           labels,
                           ctx_mode,
                           global_max_simi_modal,
                           modal_dir,
                           num_proto,
                           modality='context',
                           color_order='BGR',
                           img_norm_mode='torch',
                           log=print,
                           dataset_name='PIE',
                           norm_traj=1
                           ):
    img_mean, img_std = img_mean_std(img_norm_mode)  # BGR
    
    # traverse all protos
    for j in range(num_proto):
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

            # input sample
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
            
            proto_vis_dir = os.path.join(modal_dir,  str(j) + 'th_proto', str(explain_idx))
            makedir(proto_vis_dir)

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
                nm = nm_int.item()
                img_nm.append(img_nm_int2str(nm))
            content = ['ped_id: '+ped_id+'\n']+['img_nm:\n']+img_nm+['\n'+str(labels_cur_sample)]+['\nhflip:'+str(hflip_flag)]+['\nsk_ijhw:'+str(sk_ijhw)] +\
                ['\nimg_ijhw:'+str(img_ijhw)] + ['\nctx_ijhw:'+str(ctx_ijhw)] + ['\ncur proto simi value:'+str(cur_max_simi_j)]
            if modality == 'ego':
                content.append(f'\ninput: {original_input_j}')
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
            
            # visualize skeleton heatmap
            if modality == 'skeleton':
                if dataset_name in ('PIE', 'JAAD'):
                    ori_img_root = os.path.join(root_path, 'cropped_images', 'even_padded', '288w_by_384h')
                    sk_img_root = os.path.join(root_path, 'skeletons', 'even_padded', '288w_by_384h')
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
                    sk_img_root = '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/skeletons/even_padded/288w_by_384h/'
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
                input_heatmap_dir = os.path.join(proto_vis_dir, 'input_heatmap')
                makedir(input_heatmap_dir)
                feat_mean_, feat_max_, feat_min_ = visualize_featmap3d_simple(original_input_j, sk_imgs, 
                                                                        mode='mean', 
                                                                        channel_weights=None, 
                                                                        save_dir=input_heatmap_dir,
                                                                        log=log)

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

                # reverse augmentation for background
                if hflip_flag:
                    background = torch.tensor(background).permute(2, 0, 1)  # 3HW
                    background = tvf.hflip(background).permute(1, 2, 0).numpy()

                unnormed_traj = copy.deepcopy(inputs['traj_unnormed'][sample_idx_in_batch].cpu().numpy())
                img = draw_boxes_on_img(background, unnormed_traj)
                # print(background)
                # print(bg_path)
                # print(unnormed_traj)
                # print(img.shape)
                cv2.imwrite(filename=os.path.join(proto_vis_dir, 'traj.png'), img=img)
