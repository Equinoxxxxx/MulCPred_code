import cv2
import numpy as np
import os
import pickle
import json
from tqdm import tqdm
from ..utils import makedir
from nuscenes.nuscenes import NuScenes
from .coord_transform import nusc_3dbbox_to_2dbbox
from ..datasets.pie_data import PIE
from ..datasets.jaad_data import JAAD
from config import dataset_root


def crop_img(img, bbox, resize_mode, target_size=(224, 224)):
    l, t, r, b = list(map(int, bbox))
    cropped = img[t:b, l:r]
    if resize_mode == 'ori':
        resized = cropped
    elif resize_mode == 'resized':
        resized = cv2.resize(cropped, target_size)
    elif resize_mode == 'even_padded':
        h = b-t
        w = r-l
        if h < 0 or w < 0:
            raise ValueError('Box size < 0', h, w)
        if h == 0 or w == 0:
            return None
        if  h > 0 and w > 0 and float(w) / h > float(target_size[0]) / target_size[1]:
            ratio = float(target_size[0]) / w
        else:
            ratio = float(target_size[1]) / h
        new_size = (int(w*ratio), int(h*ratio))
        # print(cropped.shape, l, t, r, b, new_size)

        cropped = cv2.resize(cropped, new_size)
        w_pad = target_size[0] - new_size[0]
        h_pad = target_size[1] - new_size[1]
        l_pad = w_pad // 2
        r_pad = w_pad - l_pad
        t_pad = h_pad // 2
        b_pad = h_pad - t_pad
        resized = cv2.copyMakeBorder(cropped,t_pad,b_pad,l_pad,r_pad,cv2.BORDER_CONSTANT,value=(0, 0, 0))  # t, b, l, r
        assert (resized.shape[1], resized.shape[0]) == target_size
    
    return resized

def crop_ctx(img, 
             bbox, 
             mode, 
             target_size=(224, 224),
             padding_value=127):
    ori_H, ori_W = img.shape[:2]
    l, t, r, b = list(map(int, bbox))
    # crop local context
    x = (l+r) // 2
    y = (t+b) // 2
    h = b-t
    w = r-l
    if h == 0 or w == 0:
        return None
    crop_h = h*2
    crop_w = h*2
    crop_l = max(x-h, 0)
    crop_r = min(x+h, ori_W)
    crop_t = max(y-h, 0)
    crop_b = min(y+h, ori_W)
    if mode == 'local':
        # mask target pedestrian
        rect = np.array([[l, t], [r, t], [r, b], [l, b]])
        masked = cv2.fillConvexPoly(img, rect, (127, 127, 127))
        cropped = masked[crop_t:crop_b, crop_l:crop_r]
    elif mode == 'ori_local':
        cropped = img[crop_t:crop_b, crop_l:crop_r]
    l_pad = max(h-x, 0)
    r_pad = max(x+h-ori_W, 0)
    t_pad = max(h-y, 0)
    b_pad = max(y+h-ori_H, 0)
    cropped = cv2.copyMakeBorder(cropped, 
                                 t_pad, b_pad, l_pad, r_pad, 
                                 cv2.BORDER_CONSTANT, 
                                 value=(padding_value, padding_value, padding_value))
    assert cropped.shape[0] == crop_h and cropped.shape[1] == crop_w, (cropped.shape, (crop_h, crop_w))
    # print(cropped.shape, cropped.dtype, img.shape)
    resized = cv2.resize(np.array(cropped, dtype='uint8'), target_size)

    return resized

def crop_img_PIE_JAAD(resize_mode='even_padded', 
                      target_size=(224, 224), 
                      dataset_name='PIE', 
                      ):
    import os
    if dataset_name == 'PIE':
        pie_jaad_root = os.path.join(dataset_root, 'PIE_dataset')
        data_base = PIE(data_path=pie_jaad_root)
        data_opts = {'normalize_bbox': False,
                         'fstride': 1,
                         'sample_type': 'all',
                         'height_rng': [0, float('inf')],
                         'squarify_ratio': 0,
                         'data_split_type': 'default',  # kfold, random, default. default: set03 for test
                         'seq_type': 'intention',  # crossing , intention
                         'min_track_size': 0,  # discard tracks that are shorter
                         'max_size_observe': 1,  # number of observation frames
                         'max_size_predict': 1,  # number of prediction frames
                         'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
                         'balance': False,  # balance the training and testing samples
                         'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                         'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                         'seq_type': 'trajectory',
                         'encoder_input_type': ['bbox', 'obd_speed'],
                         'decoder_input_type': [],
                         'output_type': ['intention_binary', 'bbox']
                         }
    else:
        pie_jaad_root = os.path.join(dataset_root, 'JAAD')
        data_opts = {'fstride': 1,
             'sample_type': 'all',  
	         'subset': 'high_visibility',
             'data_split_type': 'default',
             'seq_type': 'trajectory',
	         'height_rng': [0, float('inf')],
	         'squarify_ratio': 0,
             'min_track_size': 0,
             'random_params': {'ratios': None,
                               'val_data': True,
                               'regen_data': True},
             'kfold_params': {'num_folds': 5, 'fold': 1}}
        data_base = JAAD(data_path=pie_jaad_root)
    cropped_root = os.path.join(pie_jaad_root, 'cropped_images')
    data_dir = os.path.join(cropped_root, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h')
    makedir(data_dir)

    tracks = data_base.generate_data_trajectory_sequence(image_set='all', 
                                                         **data_opts)  # all: 1842  train 882 
    # 'image', 'ped_id', 'bbox', 'center', 'occlusion', 'obd_speed', 'gps_speed', 'heading_angle', 'gps_coord', 'yrp', 'intention_prob', 'intention_binary'
    num_tracks = len(tracks['image'])
    # ids = []
    # for track in tracks['ped_id']:
    #     ids += track
    # id_set = np.unique(ids)
    # print(len(id_set))

    for i in range(num_tracks):
        cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
        ped_dir = os.path.join(data_dir, cur_pid)
        makedir(ped_dir)
        track_len = len(tracks['ped_id'][i])
        for j in range(track_len):
            img_path = tracks['image'][i][j]
            target_path = os.path.join(ped_dir, img_path.split('/')[-1])
            img = cv2.imread(img_path)
            l, t, r, b = tracks['bbox'][i][j]  # l t r b
            l, t, r, b = map(int, [l, t, r, b])
            cropped = img[t:b, l:r]
            if resize_mode == 'ori':
                resized = cropped
            elif resize_mode == 'resized':
                resized = cv2.resize(cropped, target_size)
            elif resize_mode == 'even_padded':
                h = b-t
                w = r-l
                if  float(w) / h > float(target_size[0]) / target_size[1]:
                    ratio = float(target_size[0]) / w
                else:
                    ratio = float(target_size[1]) / h
                new_size = (int(w*ratio), int(h*ratio))
                cropped = cv2.resize(cropped, new_size)
                w_pad = target_size[0] - new_size[0]
                h_pad = target_size[1] - new_size[1]
                l_pad = w_pad // 2
                r_pad = w_pad - l_pad
                t_pad = h_pad // 2
                b_pad = h_pad - t_pad
                resized = cv2.copyMakeBorder(cropped,
                                             t_pad,b_pad,l_pad,r_pad,
                                             cv2.BORDER_CONSTANT,
                                             value=(0, 0, 0))  # t, b, l, r
                assert (resized.shape[1], resized.shape[0]) == target_size
            else:
                h = b-t
                w = r-l
                if  float(w) / h > float(target_size[0]) / target_size[1]:
                    ratio = float(target_size[0]) / w
                else:
                    ratio = float(target_size[1]) / h
                new_size = (int(w*ratio), int(h*ratio))
                cropped = cv2.resize(cropped, new_size)
                w_pad = target_size[0] - new_size[0]
                h_pad = target_size[1] - new_size[1]
                resized = cv2.copyMakeBorder(cropped,0,h_pad,0,w_pad,cv2.BORDER_CONSTANT,value=(0, 0, 0))  # t, b, l, r
                assert (resized.shape[1], resized.shape[0]) == target_size
            cv2.imwrite(target_path, resized)
        print(i, ped_dir, 'done')

def crop_ctx_PIE_JAAD(mode='ori_local', target_size=(224, 224), dataset_name='PIE'):
    if dataset_name == 'PIE':
        pie_jaad_root = os.path.join(dataset_root, 'PIE_dataset')
        data_base = PIE(data_path=pie_jaad_root)
        data_opts = {'normalize_bbox': False,
                         'fstride': 1,
                         'sample_type': 'all',
                         'height_rng': [0, float('inf')],
                         'squarify_ratio': 0,
                         'data_split_type': 'default',  # kfold, random, default. default: set03 for test
                         'seq_type': 'intention',  # crossing , intention
                         'min_track_size': 0,  # discard tracks that are shorter
                         'max_size_observe': 1,  # number of observation frames
                         'max_size_predict': 1,  # number of prediction frames
                         'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
                         'balance': False,  # balance the training and testing samples
                         'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                         'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                         'seq_type': 'trajectory',
                         'encoder_input_type': ['bbox', 'obd_speed'],
                         'decoder_input_type': [],
                         'output_type': ['intention_binary', 'bbox']
                         }
    else:
        pie_jaad_root = os.path.join(dataset_root, 'JAAD')
        data_opts = {'fstride': 1,
             'sample_type': 'all',  
	         'subset': 'high_visibility',
             'data_split_type': 'default',
             'seq_type': 'trajectory',
	         'height_rng': [0, float('inf')],
	         'squarify_ratio': 0,
             'min_track_size': 0,
             'random_params': {'ratios': None,
                               'val_data': True,
                               'regen_data': True},
             'kfold_params': {'num_folds': 5, 'fold': 1}}
        data_base = JAAD(data_path=pie_jaad_root)
    context_root = os.path.join(pie_jaad_root, 'context')
    makedir(context_root)
    data_dir = os.path.join(context_root, mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h')
    makedir(data_dir)

    tracks = data_base.generate_data_trajectory_sequence(image_set='all', **data_opts)
    num_tracks = len(tracks['image'])
    mask_value = (127, 127, 127)

    if mode == 'mask_ped':
        for i in range(num_tracks):
            cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
            ped_dir = os.path.join(data_dir, cur_pid)
            makedir(ped_dir)
            track_len = len(tracks['ped_id'][i])
            for j in range(track_len):
                img_path = tracks['image'][i][j]
                target_path = os.path.join(ped_dir, img_path.split('/')[-1])
                img = cv2.imread(img_path)
                l, t, r, b = tracks['bbox'][i][j]  # l t r b
                l, t, r, b = map(int, [l, t, r, b])
                # if mode == 'mask_ped':
                rect = np.array([[l, t], [r, t], [r, b], [l, b]])
                masked = cv2.fillConvexPoly(img, rect, mask_value)
                resized = cv2.resize(masked, target_size)
                cv2.imwrite(target_path, resized)
            print(i, ped_dir, 'done')

    elif mode == 'ori':
        for i in range(num_tracks):
            cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
            ped_dir = os.path.join(data_dir, cur_pid)
            makedir(ped_dir)
            track_len = len(tracks['ped_id'][i])
            for j in range(track_len):
                img_path = tracks['image'][i][j]
                target_path = os.path.join(ped_dir, img_path.split('/')[-1])
                img = cv2.imread(img_path)
                resized = cv2.resize(img, target_size)
                cv2.imwrite(target_path, resized)
            print(i, ped_dir, 'done')

    elif mode == 'ori_local':
        for i in range(num_tracks):
            cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
            ped_dir = os.path.join(data_dir, cur_pid)
            makedir(ped_dir)
            track_len = len(tracks['ped_id'][i])
            for j in range(track_len):
                img_path = tracks['image'][i][j]
                target_path = os.path.join(ped_dir, img_path.split('/')[-1])
                img = cv2.imread(img_path)
                ori_H, ori_W = img.shape[0], img.shape[1]
                l, t, r, b = tracks['bbox'][i][j]  # l t r b
                l, t, r, b = map(int, [l, t, r, b])
                # crop local context
                x = (l+r) // 2
                y = (t+b) // 2
                h = b-t
                w = r-l
                crop_h = h*2
                crop_w = h*2
                crop_l = max(x-h, 0)
                crop_r = min(x+h, ori_W)
                crop_t = max(y-h, 0)
                crop_b = min(y+h, ori_W)
                cropped = img[crop_t:crop_b, crop_l:crop_r]
                l_pad = max(h-x, 0)
                r_pad = max(x+h-ori_W, 0)
                t_pad = max(h-y, 0)
                b_pad = max(y+h-ori_H, 0)
                cropped = cv2.copyMakeBorder(cropped, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=mask_value)
                assert cropped.shape[0] == crop_h and cropped.shape[1] == crop_w, (cropped.shape, (crop_h, crop_w))
                resized = cv2.resize(cropped, target_size)
                cv2.imwrite(target_path, resized)
            print(i, ped_dir, 'done')
    elif mode == 'local':
        for i in range(num_tracks):
            cur_pid = tracks['ped_id'][i][0][0]  # [[id], [id], ...]
            ped_dir = os.path.join(data_dir, cur_pid)
            makedir(ped_dir)
            track_len = len(tracks['ped_id'][i])
            for j in range(track_len):
                img_path = tracks['image'][i][j]
                target_path = os.path.join(ped_dir, img_path.split('/')[-1])
                img = cv2.imread(img_path)
                ori_H, ori_W = img.shape[0], img.shape[1]
                l, t, r, b = tracks['bbox'][i][j]  # l t r b
                l, t, r, b = map(int, [l, t, r, b])
                # mask target pedestrian
                rect = np.array([[l, t], [r, t], [r, b], [l, b]])
                masked = cv2.fillConvexPoly(img, rect, mask_value)
                # crop local context
                x = (l+r) // 2
                y = (t+b) // 2
                h = b-t
                w = r-l
                crop_h = h*2
                crop_w = h*2
                crop_l = max(x-h, 0)
                crop_r = min(x+h, ori_W)
                crop_t = max(y-h, 0)
                crop_b = min(y+h, ori_W)
                cropped = masked[crop_t:crop_b, crop_l:crop_r]
                l_pad = max(h-x, 0)
                r_pad = max(x+h-ori_W, 0)
                t_pad = max(h-y, 0)
                b_pad = max(y+h-ori_H, 0)
                cropped = cv2.copyMakeBorder(cropped, 
                                             t_pad, b_pad, l_pad, r_pad, 
                                             cv2.BORDER_CONSTANT, 
                                             value=mask_value)
                assert cropped.shape[0] == crop_h and \
                    cropped.shape[1] == crop_w, \
                    (cropped.shape, (crop_h, crop_w))
                resized = cv2.resize(cropped, target_size)
                cv2.imwrite(target_path, resized)
            print(i, ped_dir, 'done')

def crop_img_TITAN(tracks, resize_mode='even_padded', target_size=(224, 224), obj_type='p'):
    crop_root = os.path.join(dataset_root, 'TITAN/TITAN_extra/cropped_images')
    makedir(crop_root)
    data_root = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset')
    if obj_type == 'p':
        crop_obj_path = os.path.join(crop_root, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', 'ped')
        makedir(crop_obj_path)
    else:
        crop_obj_path = os.path.join(crop_root, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', 'veh')
        makedir(crop_obj_path)
    for i in tqdm(range(len(tracks['clip_id']))):
        cid = int(tracks['clip_id'][i][0])
        oid = int(float(tracks['obj_id'][i][0]))
        cur_clip_path = os.path.join(crop_obj_path, str(cid))
        makedir(cur_clip_path)
        cur_obj_path = os.path.join(cur_clip_path, str(oid))
        makedir(cur_obj_path)
        
        for j in range(len(tracks['clip_id'][i])):
            img_nm = tracks['img_nm'][i][j]
            l, t, r, b = list(map(int, tracks['bbox'][i][j]))
            img_path = os.path.join(data_root, 'images_anonymized', 'clip_'+str(cid), 'images', img_nm)
            tgt_path = os.path.join(cur_obj_path, img_nm)
            img = cv2.imread(img_path)
            cropped = img[t:b, l:r]
            if resize_mode == 'ori':
                resized = cropped
            elif resize_mode == 'resized':
                resized = cv2.resize(cropped, target_size)
            elif resize_mode == 'even_padded':
                h = b-t
                w = r-l
                if  float(w) / h > float(target_size[0]) / target_size[1]:
                    ratio = float(target_size[0]) / w
                else:
                    ratio = float(target_size[1]) / h
                new_size = (int(w*ratio), int(h*ratio))
                cropped = cv2.resize(cropped, new_size)
                w_pad = target_size[0] - new_size[0]
                h_pad = target_size[1] - new_size[1]
                l_pad = w_pad // 2
                r_pad = w_pad - l_pad
                t_pad = h_pad // 2
                b_pad = h_pad - t_pad
                resized = cv2.copyMakeBorder(cropped,t_pad,b_pad,l_pad,r_pad,cv2.BORDER_CONSTANT,value=(0, 0, 0))  # t, b, l, r
                assert (resized.shape[1], resized.shape[0]) == target_size
            else:
                raise NotImplementedError(resize_mode)
            cv2.imwrite(tgt_path, resized)
        print(i, cid, cur_obj_path, 'done')

def crop_ctx_TITAN(tracks, mode='ori_local', target_size=(224, 224), obj_type='p'):
    ori_H, ori_W = 1520, 2704
    crop_root = os.path.join(dataset_root, 'TITAN/TITAN_extra/context')
    makedir(crop_root)
    data_root = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset')
    if obj_type == 'p':
        crop_obj_path = os.path.join(crop_root, mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', 'ped')
        makedir(crop_obj_path)
    else:
        crop_obj_path = os.path.join(crop_root, mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', 'veh')
        makedir(crop_obj_path)
    
    if mode == 'local':
        for i in range(len(tracks['clip_id'])):  # tracks
            cid = int(tracks['clip_id'][i][0])
            oid = int(float(tracks['obj_id'][i][0]))
            cur_clip_path = os.path.join(crop_obj_path, str(cid))
            makedir(cur_clip_path)
            cur_obj_path = os.path.join(cur_clip_path, str(oid))
            makedir(cur_obj_path)
            for j in range(len(tracks['clip_id'][i])):  # time steps in each track
                img_nm = tracks['img_nm'][i][j]
                l, t, r, b = list(map(int, tracks['bbox'][i][j]))
                img_path = os.path.join(data_root, 
                                        'images_anonymized', 
                                        'clip_'+str(cid), 
                                        'images', 
                                        img_nm)
                tgt_path = os.path.join(cur_obj_path, 
                                        img_nm)
                img = cv2.imread(img_path)
                # mask target pedestrian
                rect = np.array([[l, t], [r, t], [r, b], [l, b]])
                masked = cv2.fillConvexPoly(img, rect, (127, 127, 127))
                # crop local context
                x = (l+r) // 2
                y = (t+b) // 2
                h = b-t
                w = r-l
                crop_h = h*2
                crop_w = h*2
                crop_l = max(x-h, 0)
                crop_r = min(x+h, ori_W)
                crop_t = max(y-h, 0)
                crop_b = min(y+h, ori_W)
                cropped = masked[crop_t:crop_b, crop_l:crop_r]
                l_pad = max(h-x, 0)
                r_pad = max(x+h-ori_W, 0)
                t_pad = max(h-y, 0)
                b_pad = max(y+h-ori_H, 0)
                cropped = cv2.copyMakeBorder(cropped, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
                assert cropped.shape[0] == crop_h and cropped.shape[1] == crop_w, (cropped.shape, (crop_h, crop_w))
                resized = cv2.resize(cropped, target_size)
                cv2.imwrite(tgt_path, resized)
            print(i, cid, oid, cur_obj_path, 'done')
    elif mode == 'ori_local':
        for i in range(len(tracks['clip_id'])):  # tracks
            cid = int(tracks['clip_id'][i][0])
            oid = int(float(tracks['obj_id'][i][0]))
            cur_clip_path = os.path.join(crop_obj_path, str(cid))
            makedir(cur_clip_path)
            cur_obj_path = os.path.join(cur_clip_path, str(oid))
            makedir(cur_obj_path)
            for j in range(len(tracks['clip_id'][i])):  # time steps in each track
                img_nm = tracks['img_nm'][i][j]
                l, t, r, b = list(map(int, tracks['bbox'][i][j]))
                img_path = os.path.join(data_root, 'images_anonymized', 'clip_'+str(cid), 'images', img_nm)
                tgt_path = os.path.join(cur_obj_path, img_nm)
                img = cv2.imread(img_path)
                # crop local context
                x = (l+r) // 2
                y = (t+b) // 2
                h = b-t
                w = r-l
                crop_h = h*2
                crop_w = h*2
                crop_l = max(x-h, 0)
                crop_r = min(x+h, ori_W)
                crop_t = max(y-h, 0)
                crop_b = min(y+h, ori_W)
                cropped = img[crop_t:crop_b, crop_l:crop_r]
                l_pad = max(h-x, 0)
                r_pad = max(x+h-ori_W, 0)
                t_pad = max(h-y, 0)
                b_pad = max(y+h-ori_H, 0)
                cropped = cv2.copyMakeBorder(cropped, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
                assert cropped.shape[0] == crop_h and cropped.shape[1] == crop_w, (cropped.shape, (crop_h, crop_w))
                resized = cv2.resize(cropped, target_size)
                cv2.imwrite(tgt_path, resized)
            print(i, cid, oid, cur_obj_path, 'done')
    else:
        raise NotImplementedError(mode)

if __name__ == '__main__':
    pass