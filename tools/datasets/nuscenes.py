from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from ..data.nusc_split import TRAIN_SC, VAL_SC
from ..data.coord_transform import nusc_3dbbox_to_2dbbox
from ..visualize import draw_box, draw_boxes_on_img
from ..utils import makedir
from ..data.preprocess import crop_img, crop_ctx

import numpy as np
import pickle
import os
import torch
from tqdm import tqdm
from pyquaternion import Quaternion
import cv2
import copy
import pdb

NUSC_ROOT = '/home/y_feng/workspace6/datasets/nusc/'

class NuscDataset(torch.utils.data.Dataset):
    def __init__(self,
                 subset='train',
                 obs_len=2, pred_len=2, overlap_ratio=0.5, recog_act=0,
                 tte=None,
                 norm_traj=True,
                 min_h=72,
                 min_w=36,
                 min_vis_level=3,
                 sensor='CAM_FRONT',
                 interval=0,
                 ego_accel=True,
                 ):
        super().__init__()
        self.subset = subset
        self.dataset_name = 'nusc'
        self.obs_len = obs_len
        self.pred_len = pred_len

        self.tte = tte
        self.overlap_ratio = overlap_ratio
        self.norm_traj = norm_traj
        self.recog_act = recog_act
        self.min_h = min_h
        self.min_w = min_w
        self.min_vis_level = min_vis_level
        self.sensor = sensor
        self.interval = interval
        # sequence length considering interval
        self._obs_len = self.obs_len * (self.interval + 1)
        self._pred_len = self.pred_len * (self.interval + 1)
        self.ego_accel = ego_accel
        self.ego_motion_key = 'ego_accel' if self.ego_accel else 'ego_speed'
        if self.subset == 'train':
            self.sce_names = TRAIN_SC
        elif self.subset == 'val':
            self.sce_names = VAL_SC
        else:
            raise ValueError(self.subset)

        self._load_tk_id_dicts()

        self.nusc_root = NUSC_ROOT
        self.extra_root = os.path.join(NUSC_ROOT, 'extra')
        self.nusc = NuScenes(version='v1.0-trainval', 
                             dataroot=NUSC_ROOT, 
                             verbose=True)
        self.imgnm_to_objid_path = os.path.join(self.extra_root, 
                                                self.subset+'_imgnm_to_objid_to_ann.pkl')

        # get vehicle tracks and pedestrian tracks
        self.p_tracks = self.get_obj_tracks(obj_type='ped')
        self.v_tracks = self.get_obj_tracks(obj_type='veh')

        # add the acceleration to the pedestrian tracks
        self.p_tracks = self._get_accel(self.p_tracks)

        # get cid to img name to obj id dict
        if not os.path.exists(self.imgnm_to_objid_path):
            self.imgnm_to_objid = \
                self.get_imgnm_to_objid(self.p_tracks, 
                                        self.v_tracks, 
                                        self.imgnm_to_objid_path)
        else:
            with open(self.imgnm_to_objid_path, 'rb') as f:
                self.imgnm_to_objid = pickle.load(f)
        
        # convert tracks into samples
        self.samples = self.tracks_to_samples(self.p_tracks)


    def _load_tk_id_dicts(self):
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_scene_id_to_token.pkl', 'rb')
        self.scene_id_to_token = pickle.load(f)
        f.close()
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_token_to_scene_id.pkl', 'rb')
        self.token_to_scene_id = pickle.load(f)
        f.close()
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_sample_id_to_token.pkl', 'rb')
        self.sample_id_to_token = pickle.load(f)
        f.close()
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_token_to_sample_id.pkl', 'rb')
        self.token_to_sample_id = pickle.load(f)
        f.close()
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_instance_id_to_token.pkl', 'rb')
        self.instance_id_to_token = pickle.load(f)
        f.close()
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_token_to_instance_id.pkl', 'rb')
        self.token_to_instance_id = pickle.load(f)
        f.close()
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_ins_to_ann_to_id.pkl', 'rb')
        self.instk_to_anntk_to_id = pickle.load(f)
        f.close()
        f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_ins_to_id_to_ann.pkl', 'rb')
        self.instk_to_id_to_anntk = pickle.load(f)
        f.close()
    
    def get_obj_tracks(self, obj_type='ped'):
        '''
        track structure:
        {
            'sce_id': [[int, ...], ...],
            'ins_id': [[int, ...], ...],
            'sam_id': [[int, ...], ...],
            'img_nm': [[str, ...], ...],
            'bbox_3d': [[list, ...], ...],
            'bbox_2d': [[list, ...], ...],
            'ego_speed': [[float, ...], ...]
        }
            
        '''
        assert obj_type in ('ped', 'veh'), obj_type
        obj_tracks = {'sce_id': [],
                    'ins_id': [],
                    'sam_id': [],
                    'img_nm': [],
                    'bbox_3d': [],
                    'bbox_2d': [],
                    'ego_speed': []}
        # discard the last frame
        min_track_len = self.obs_len + self.pred_len + 1  
        anns_path = '_'.join(['anns', 
                              self.subset, 
                              obj_type, 
                              self.sensor]
                              ) + '.pkl'
        anns_path = os.path.join(self.extra_root, anns_path)
        with open(anns_path, 'rb') as f:
            instk_to_anntk = pickle.load(f)
        print('Getting pedestrian tracks')
        for instk in tqdm(instk_to_anntk):
            ann_seqs = instk_to_anntk[instk]
            for seq in ann_seqs:
                # skip too short track
                if len(seq) < min_track_len:
                    continue
                
                processing = False
                # discard the last frame for each track to calc speed
                for i in range(len(seq)-1):
                    ann = self.nusc.get('sample_annotation', seq[i])
                    bbox, corners3d = \
                        nusc_3dbbox_to_2dbbox(self.nusc, ann['token'])
                    vis_l = int(ann['visibility_token'])
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    # end current track
                    if w < self.min_w or \
                        h < self.min_h or \
                            vis_l < self.min_vis_level:
                        # check if length >= min length
                        if processing and \
                            len(obj_tracks['ego_speed'][-1]) < min_track_len:
                            for k in obj_tracks:
                                # delete the recent ended seq
                                obj_tracks[k].pop(-1)
                        processing = False
                        continue
                    next_ann = self.nusc.get('sample_annotation', seq[i+1])
                    sam = self.nusc.get('sample', ann['sample_token'])
                    sce = self.nusc.get('scene', sam['scene_token'])
                    ego_vel = self.calc_ego_velocity(ann['token'], 
                                                     next_ann['token'])
                    sen_data = self.nusc.get('sample_data', 
                                             sam['data'][self.sensor])
                    img_nm = sen_data['filename']
                    sce_id = self.token_to_scene_id[sce['token']]
                    sam_id = self.token_to_sample_id[sam['token']]
                    ins_id = self.token_to_instance_id[instk]
                    if processing:
                        obj_tracks['sce_id'][-1].append(int(sce_id))
                        obj_tracks['sam_id'][-1].append(int(sam_id))
                        obj_tracks['ins_id'][-1].append(int(ins_id))
                        obj_tracks['bbox_2d'][-1].append(bbox)
                        obj_tracks['bbox_3d'][-1].append(corners3d)
                        obj_tracks['img_nm'][-1].append(img_nm)
                        obj_tracks['ego_speed'][-1].append(ego_vel)
                    else:
                        processing = True
                        obj_tracks['sce_id'].append([int(sce_id)])
                        obj_tracks['sam_id'].append([int(sam_id)])
                        obj_tracks['ins_id'].append([int(ins_id)])
                        obj_tracks['bbox_2d'].append([bbox])
                        obj_tracks['bbox_3d'].append([corners3d])
                        obj_tracks['img_nm'].append([img_nm])
                        obj_tracks['ego_speed'].append([ego_vel])
        return obj_tracks    
    
    def _get_accel(self, p_tracks):
        new_tracks = {'sce_id': [],
                    'ins_id': [],
                    'sam_id': [],
                    'img_nm': [],
                    'bbox_3d': [],
                    'bbox_2d': [],
                    'ego_speed': [],
                    'ego_accel': [],
                    }
        # calculate acceleration and record the idx of tracks to remove
        idx_to_remove = []
        for i in range(len(p_tracks['ego_speed'])):
            speed_track = p_tracks['ego_speed'][i]
            if len(speed_track) < 2:
                idx_to_remove.append(i)
                continue
            new_tracks['ego_accel'].append([])
            for j in range(len(speed_track)-1):
                accel = (speed_track[j+1] - speed_track[j]) * 2  # 2 fps
                new_tracks['ego_accel'][-1].append(accel)
        # add the rest keys
        for k in p_tracks:
            for i in range(len(p_tracks[k])):
                if i in idx_to_remove:
                    continue
                cur_track = p_tracks[k][i]
                new_tracks[k].append(cur_track[1:])

        # check if the lengths of all keys confront
        for k in new_tracks:
            assert len(new_tracks[k]) == len(new_tracks['ego_accel']), \
                (len(new_tracks[k]), len(new_tracks['ego_accel']))
            for i in range(len(new_tracks[k])):
                assert len(new_tracks[k][i]) == len(new_tracks['ego_accel'][i]), \
                    (len(new_tracks[k][i]), len(new_tracks['ego_accel'][i]))
        
        return new_tracks

    def calc_ego_velocity(self, cur_anntk, next_anntk):
        cur_ann = self.nusc.get('sample_annotation', cur_anntk)
        next_ann = self.nusc.get('sample_annotation', next_anntk)
        cur_sam = self.nusc.get('sample', cur_ann['sample_token'])
        next_sam = self.nusc.get('sample', next_ann['sample_token'])
        cur_sen_data = self.nusc.get('sample_data', cur_sam['data'][self.sensor])
        next_sen_data = self.nusc.get('sample_data', next_sam['data'][self.sensor])
        cur_pose = self.nusc.get('ego_pose', cur_sen_data['ego_pose_token'])
        next_pose = self.nusc.get('ego_pose', next_sen_data['ego_pose_token'])

        cur_loc = np.array(cur_pose['translation'])
        next_loc = np.array(next_pose['translation'])

        cur_time = 1e-6 * cur_pose['timestamp']
        next_time = 1e-6 * next_pose['timestamp']

        ego_vel = (next_loc - cur_loc) / (next_time - cur_time)

        return np.sqrt(ego_vel[0]**2 + ego_vel[1]**2) 

    def get_imgnm_to_objid(self, p_tracks, v_tracks, save_path):
        imgnm_to_oid_to_info = {}
        # pedestrian
        tracks = p_tracks
        n_tracks = len(tracks['bbox_2d'])
        print('Saving imgnm to objid to obj info of pedestrians in nuScenes')
        for i in range(n_tracks):
            ins_id = str(tracks['ins_id'][i][0])
            sam_ids = tracks['sam_id'][i]
            for j in range(len(sam_ids)):
                sam_id = str(sam_ids[j])
                # initialize img dict
                if sam_id not in imgnm_to_oid_to_info:
                    imgnm_to_oid_to_info[sam_id] = {}
                    imgnm_to_oid_to_info[sam_id]['ped'] = {}
                    imgnm_to_oid_to_info[sam_id]['veh'] = {}
                # initialize obj dict
                imgnm_to_oid_to_info[sam_id]['ped'][ins_id] = {}
                imgnm_to_oid_to_info[sam_id]['ped'][ins_id]['bbox_2d'] = \
                    tracks['bbox_2d'][i][j]
                imgnm_to_oid_to_info[sam_id]['ped'][ins_id]['bbox_3d'] = \
                    tracks['bbox_3d'][i][j]
        # vehicle
        tracks = v_tracks
        n_tracks = len(tracks['bbox_2d'])
        print('Saving imgnm to objid to obj info of vehicles in nuScenes')
        for i in range(n_tracks):
            ins_id = str(tracks['ins_id'][i][0])
            sam_ids = tracks['sam_id'][i]
            for j in range(len(sam_ids)):
                sam_id = str(sam_ids[j])
                # initialize img dict
                if sam_id not in imgnm_to_oid_to_info:
                    imgnm_to_oid_to_info[sam_id] = {}
                    imgnm_to_oid_to_info[sam_id]['ped'] = {}
                    imgnm_to_oid_to_info[sam_id]['veh'] = {}
                # initialize obj dict
                imgnm_to_oid_to_info[sam_id]['veh'][ins_id] = {}
                imgnm_to_oid_to_info[sam_id]['veh'][ins_id]['bbox_2d'] = \
                    tracks['bbox_2d'][i][j]
                imgnm_to_oid_to_info[sam_id]['veh'][ins_id]['bbox_3d'] = \
                    tracks['bbox_3d'][i][j]
        with open(save_path, 'wb') as f:
            pickle.dump(imgnm_to_oid_to_info, f)
        return imgnm_to_oid_to_info


    def tracks_to_samples(self, tracks):
        seq_len = self.obs_len + self.pred_len
        overlap_s = self._obs_len if self.overlap_ratio == 0 \
            else int((1 - self.overlap_ratio) * self._obs_len)
        overlap_s = 1 if overlap_s < 1 else overlap_s
        samples = {}
        for dt in tracks.keys():
            try:
                samples[dt] = tracks[dt]
            except KeyError:
                raise ('Wrong data type is selected %s' % dt)
        # split tracks to fixed length samples
        print('---------------Split tracks to samples---------------')
        print(samples.keys())
        for k in tqdm(samples.keys()):
            _samples = []
            for track in samples[k]:
                if len(track) < seq_len:
                    continue
                if self.tte is not None:
                    raise NotImplementedError()
                else:
                    _samples.extend([track[i: i+seq_len] \
                                        for i in range(0,
                                                    len(track)-seq_len+1, 
                                                    overlap_s)])
            samples[k] = _samples

        # normalize tracks by subtracting bbox/center of the 1st frame
        print('---------------Normalize traj---------------')
        bboxes_2d_norm = copy.deepcopy(samples['bbox_2d'])
        bboxes_3d_norm = copy.deepcopy(samples['bbox_3d'])
        if self.norm_traj:
            for i in range(len(bboxes_2d_norm)):
                bboxes_2d_norm[i] = np.subtract(bboxes_2d_norm[i][:], 
                                                bboxes_2d_norm[i][0]).tolist()
                bboxes_3d_norm[i] = np.subtract(bboxes_3d_norm[i][:], 
                                                bboxes_3d_norm[i][0]).tolist()
        samples['bbox_2d_normed'] = bboxes_2d_norm
        samples['bbox_3d_normed'] = bboxes_3d_norm

        # choose ego motion quantity
        samples['ego_motion'] = samples[self.ego_motion_key]
        
        # split obs and pred
        print('---------------Split obs and pred---------------')
        obs_slices = {}
        pred_slices = {}
        for k in samples.keys():
            obs_slices[k] = []
            pred_slices[k] = []
            obs_slices[k].extend([d[0:self._obs_len] for d in samples[k]])
            pred_slices[k].extend([d[self._obs_len:] for d in samples[k]])

        all_samples = {
            'obs': obs_slices,
            'pred': pred_slices
        }

        return all_samples

    def get_neighbors(self, samids):
        
        pass


def save_scene_token_dict():
    scene_id_to_token = {}
    token_to_scene_id = {}
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    for i in range(len(nusc.scene)):
        cur_t = nusc.scene[i]['token']
        scene_id_to_token[str(i)] = cur_t
        token_to_scene_id[cur_t] = str(i)
        print(f'scene {i} done')
    save_path = '/home/y_feng/workspace6/datasets/nusc/extra'
    with open(os.path.join(save_path, 'trainval_scene_id_to_token.pkl'), 'wb') as f:
        pickle.dump(scene_id_to_token, f)
    with open(os.path.join(save_path, 'trainval_token_to_scene_id.pkl'), 'wb') as f:
        pickle.dump(token_to_scene_id, f)

def save_sample_token_dict():
    sample_id_to_token = {}
    token_to_sample_id = {}
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    for i in range(len(nusc.sample)):
        cur_t = nusc.sample[i]['token']
        sample_id_to_token[str(i)] = cur_t
        token_to_sample_id[cur_t] = str(i)
        print(f'sample {i} done')
    save_path = '/home/y_feng/workspace6/datasets/nusc/extra'
    with open(os.path.join(save_path, 'trainval_sample_id_to_token.pkl'), 'wb') as f:
        pickle.dump(sample_id_to_token, f)
    with open(os.path.join(save_path, 'trainval_token_to_sample_id.pkl'), 'wb') as f:
        pickle.dump(token_to_sample_id, f)

def save_instance_token_dict():
    instance_id_to_token = {}
    token_to_instance_id = {}
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    for i in range(len(nusc.instance)):
        cur_t = nusc.instance[i]['token']
        instance_id_to_token[str(i)] = cur_t
        token_to_instance_id[cur_t] = str(i)
        print(f'instance {i} done')
    save_path = '/home/y_feng/workspace6/datasets/nusc/extra'
    with open(os.path.join(save_path, 'trainval_instance_id_to_token.pkl'), 'wb') as f:
        pickle.dump(instance_id_to_token, f)
    with open(os.path.join(save_path, 'trainval_token_to_instance_id.pkl'), 'wb') as f:
        pickle.dump(token_to_instance_id, f)

def _save_ins_to_ann_to_dict():
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    instk_to_anntk_to_id = {}
    instk_to_id_to_anntk = {}
    for ins in tqdm(nusc.instance):
        instk_to_anntk_to_id[ins['token']] = {}
        instk_to_id_to_anntk[ins['token']] = {}
        cur_ann_tk = ins['first_annotation_token']
        i = 0
        while cur_ann_tk != '':
            instk_to_anntk_to_id[ins['token']][cur_ann_tk] = str(i)
            instk_to_id_to_anntk[ins['token']][str(i)] = cur_ann_tk
            cur_ann_tk = nusc.get('sample_annotation', cur_ann_tk)['next']
    save_path = '/home/y_feng/workspace6/datasets/nusc/extra/token_id'
    with open(os.path.join(save_path, 'trainval_ins_to_ann_to_id.pkl'), 'wb') as f:
        pickle.dump(instk_to_anntk_to_id, f)
    with open(os.path.join(save_path, 'trainval_ins_to_id_to_ann.pkl'), 'wb') as f:
        pickle.dump(instk_to_id_to_anntk, f)

def save_ins_tokens(subset, cate):
    '''
    save instance tokens of specific subset and category
    '''
    tokens_to_save = []
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    for ins in nusc.instance:
        category = nusc.get('category', ins['category_token'])
        first_ann = nusc.get('sample_annotation', ins['first_annotation_token'])
        first_sam = nusc.get('sample', first_ann['sample_token'])
        cur_sce = nusc.get('scene', first_sam['scene_token'])
        if subset == 'train':
            scenes_cur_subset = TRAIN_SC
        elif subset == 'val':
            scenes_cur_subset = VAL_SC
        if cur_sce['name'] not in scenes_cur_subset or category['name'].split('.')[0] != cate:
            print('ins '+ins['token']+' not saved')
            continue
        tokens_to_save.append(ins['token'])
        print(f'ins {tokens_to_save[-1]} saved')
    save_path = os.path.join(NUSC_ROOT, 'extra', subset+'_'+cate+'_ins_token.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(tokens_to_save, f)

def save_anns_in_sensor(ins_tokens_path, sensor='CAM_FRONT'):
    '''
    save annotation sequences observed by specific sensor
    '''
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    ins_tokens_nm = ins_tokens_path.split('/')[-1]
    subset, cate, _, _ = ins_tokens_nm.split('_')
    with open(ins_tokens_path, 'rb') as f:
        ins_tokens = pickle.load(f)
    ins_tk_to_ann_tks = {}
    print(f'Processing {ins_tokens_nm}')
    for ins_tk in tqdm(ins_tokens):
        cur_ins = nusc.get('instance', ins_tk)
        ann_tk_seqs = []
        cur_ann_tk = cur_ins['first_annotation_token']
        processing = False
        while cur_ann_tk != '':
            cur_ann = nusc.get('sample_annotation', cur_ann_tk)
            is_observed = is_observed_by_sensor(nusc, cur_ann_tk, sensor)
            if not is_observed:
                processing = False
            elif processing:
                ann_tk_seqs[-1].append(cur_ann_tk)
            else:
                ann_tk_seqs.append([cur_ann_tk])
                processing = True
            cur_ann_tk = cur_ann['next']
        if len(ann_tk_seqs) > 1:
            print(f'ins {ins_tk} has {len(ann_tk_seqs)} seqs')
        ins_tk_to_ann_tks[ins_tk] = ann_tk_seqs
    file_nm = 'anns_' + subset + '_' + cate + '_' + sensor + '.pkl'
    save_path = ins_tokens_path.replace(ins_tokens_nm, file_nm)
    with open(save_path, 'wb') as f:
        pickle.dump(ins_tk_to_ann_tks, f)
    
def is_observed_by_sensor(nusc, ann_tk, sensor):
    ann = nusc.get('sample_annotation', ann_tk)
    sam = nusc.get('sample', ann['sample_token'])
    _, bbs, _ = nusc.get_sample_data(sam['data'][sensor])
    for bb in bbs:
        if bb.token == ann['token']:
            return True
    return False

def check_3d_to_2d():
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    anns_path = '/home/y_feng/workspace6/datasets/nusc/extra/anns_train_human_CAM_FRONT.pkl'
    res_path = './module_test'
    makedir(res_path)
    with open(anns_path, 'rb') as f:
        ins_to_anns = pickle.load(f)
    instk = list(ins_to_anns.keys())[1]
    for i in range(len(ins_to_anns[instk][0])):
        anntk = ins_to_anns[instk][0][i]
        ann = nusc.get('sample_annotation', anntk)
        attr = nusc.get('attribute', ann['attribute_tokens'][0])
        print(f'\nann {ann}')
        print(f'attr {attr}')
        
        is_obs = is_observed_by_sensor(nusc, anntk, 'CAM_FRONT')
        print(f'is obs {is_obs}')
        sam = nusc.get('sample', ann['sample_token'])
        cam_front_data = nusc.get('sample_data', sam['data']['CAM_FRONT'])
        cali_sensor = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
        camera_intrinsic = np.array(cali_sensor['camera_intrinsic'])
        ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
        img_path = os.path.join(NUSC_ROOT, cam_front_data['filename'])
        img = cv2.imread(img_path)

        # Get the annotation box
        box = nusc.get_box(ann['token'])
        
        # Convert coords from map to veh to cam
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        box.translate(-np.array(cali_sensor['translation']))
        box.rotate(Quaternion(cali_sensor['rotation']).inverse)

        corners_3d = box.corners()  # 3, 8 corners in camera coord

        # Project to image plane
        view = np.eye(4)
        view[:3, :3] = np.array(camera_intrinsic)
        in_front = corners_3d[2, :] > 0.1  # ensure z > 0.1
        assert all(in_front), corners_3d
        corners = view_points(corners_3d, view, normalize=True)[:2, :]  # 2, 8
        # Get the 2D bounding box coordinates
        x1 = min(corners[0])
        x2 = max(corners[0])
        y1 = min(corners[1])
        y2 = max(corners[1])
        box2d = [x1, y1, x2, y2]  # ltrb
        print(box2d)
        img = draw_box(img, box2d)
        cv2.imwrite(os.path.join(res_path, '2dbbox'+str(i)+'.png'), img)

def save_cropped_imgs(obj_type='ped', sensor='CAM_FRONT', modality='img', resize_mode='even_padded', target_size=(224, 224)):
    print(f'{obj_type}, {sensor}, {modality}, {resize_mode}')
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    # f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_scene_id_to_token.pkl', 'rb')
    # scene_id_to_token = pickle.load(f)
    # f.close()
    # f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_token_to_scene_id.pkl', 'rb')
    # token_to_scene_id = pickle.load(f)
    # f.close()
    # f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_sample_id_to_token.pkl', 'rb')
    # sample_id_to_token = pickle.load(f)
    # f.close()
    f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_token_to_sample_id.pkl', 'rb')
    token_to_sample_id = pickle.load(f)
    f.close()
    # f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_instance_id_to_token.pkl', 'rb')
    # instance_id_to_token = pickle.load(f)
    # f.close()
    f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_token_to_instance_id.pkl', 'rb')
    token_to_instance_id = pickle.load(f)
    f.close()
    # f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_ins_to_ann_to_id.pkl', 'rb')
    # instk_to_anntk_to_id = pickle.load(f)
    # f.close()
    # f = open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_ins_to_id_to_ann.pkl', 'rb')
    # instk_to_id_to_anntk = pickle.load(f)
    # f.close()

    f = open(os.path.join('/home/y_feng/workspace6/datasets/nusc/extra', '_'.join(['anns_train', obj_type, sensor])+'.pkl'), 'rb')
    instk_to_anntks = pickle.load(f)
    f.close()
    f = open(os.path.join('/home/y_feng/workspace6/datasets/nusc/extra', '_'.join(['anns_val', obj_type, sensor])+'.pkl'), 'rb')
    val_instk_to_anntks = pickle.load(f)
    f.close()
    instk_to_anntks.update(val_instk_to_anntks)

    if modality == 'img':
        root = '/home/y_feng/workspace6/datasets/nusc/extra/cropped_images'
        root = os.path.join(root, sensor, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', obj_type)
        makedir(root)
    elif modality == 'ctx':
        root = '/home/y_feng/workspace6/datasets/nusc/extra/context'
        root = os.path.join(root, sensor, resize_mode, str(target_size[0])+'w_by_'+str(target_size[1])+'h', obj_type)
        makedir(root)

    for instk in tqdm(instk_to_anntks):
        insid = token_to_instance_id[instk]
        ins_path = os.path.join(root, insid)
        makedir(ins_path)
        anntk_seqs = instk_to_anntks[instk]
        seq_all = []
        for seq in anntk_seqs:
            seq_all += seq
        for anntk in seq_all:
            bbox = nusc_3dbbox_to_2dbbox(nusc, anntk)[0]  # ltrb
            l, t, r, b = bbox
            if r-l<2 or b-t<2:
                continue
            ann = nusc.get('sample_annotation', anntk)
            samtk = nusc.get('sample', ann['sample_token'])['token']
            sam = nusc.get('sample', samtk)
            samid = token_to_sample_id[samtk]
            sen_data = nusc.get('sample_data', sam['data'][sensor])
            img_path = os.path.join(NUSC_ROOT, sen_data['filename'])
            img = cv2.imread(img_path)
            if modality == 'img':
                cropped_img = crop_img(img, bbox, resize_mode, target_size)
            elif modality == 'ctx':
                cropped_img = crop_ctx(img, bbox, resize_mode, target_size)
            if cropped_img is None:
                continue
            save_path = os.path.join(ins_path, samid+'.png')
            cv2.imwrite(save_path, cropped_img)


if __name__ == '__main__':
    # save_cropped_imgs(obj_type='human', sensor='CAM_FRONT', modality='img', resize_mode='even_padded', target_size=(224, 224))
    # save_cropped_imgs(obj_type='human', sensor='CAM_FRONT', modality='img', resize_mode='even_padded', target_size=(288, 384))
    # save_cropped_imgs(obj_type='human', sensor='CAM_FRONT', modality='ctx', resize_mode='local', target_size=(224, 224))
    # save_cropped_imgs(obj_type='human', sensor='CAM_FRONT', modality='ctx', resize_mode='ori_local', target_size=(224, 224))
    # save_cropped_imgs(obj_type='vehicle', sensor='CAM_FRONT', modality='img', resize_mode='even_padded', target_size=(224, 224))
    # ins_tk_paths = ['/home/y_feng/workspace6/datasets/nusc/extra/train_human_ins_token.pkl',
    #                 '/home/y_feng/workspace6/datasets/nusc/extra/train_vehicle_ins_token.pkl',
    #                 '/home/y_feng/workspace6/datasets/nusc/extra/val_human_ins_token.pkl',
    #                 '/home/y_feng/workspace6/datasets/nusc/extra/val_vehicle_ins_token.pkl']
    # for path in ins_tk_paths:
    #     save_anns_in_sensor(path, sensor='CAM_FRONT')
    # save_ins_tokens(subset='train', cate='human')
    # save_ins_tokens(subset='val', cate='human')
    # save_ins_tokens(subset='train', cate='vehicle')
    # save_ins_tokens(subset='val', cate='vehicle')
    # save_scene_token_dict()
    # save_sample_token_dict()
    # save_instance_token_dict()
    # sensors = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)

    ins = nusc.instance[1]
    print(f'\ninstance {ins}')
    ann = nusc.get('sample_annotation', ins['first_annotation_token'])
    print(f'\nannotation {ann}')
    attr = nusc.get('attribute', ann['attribute_tokens'][0])
    print(f'\nattr {attr}')
    sam = nusc.get('sample', ann['sample_token'])
    print(f'\nsample {sam}')
    sce = nusc.get('scene', sam['scene_token'])
    print(f'\nscene {sce}')
    cam_front_data = nusc.get('sample_data', 
                              sam['data']['CAM_FRONT'])
    cam_front_cali = nusc.get('calibrated_sensor', 
                              cam_front_data['calibrated_sensor_token'])
    cam_back_data = nusc.get('sample_data', 
                             sam['data']['CAM_BACK'])
    print(f'\ncam front data {cam_front_data}')
    print(f'\ncam front cali: {cam_front_cali}')
    pdb.set_trace()
    # ego_pose_front = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
    # print(f'\nego pose {ego_pose_front}')
    # sam_next = nusc.get('sample', sam['next'])
    # cam_front_data_next = nusc.get('sample_data', sam_next['data']['CAM_FRONT'])


    # print(len(cam_front_data_[1]))
    # print(type(cam_front_data_[1][0]))
    # print(cam_front_data_[1])
    # for bb in cam_front_data_[1]:
    #     print(bb.token)
    
    # for i in range(len(nusc.instance)):
    #     ins = nusc.instance[i]
    #     cur_ann_token = ins['first_annotation_token']
    #     # ann_tokens = nusc.field2token('sample_annotation', 'instance_token', ins['token'])
    #     j = 0
    #     while cur_ann_token != '':
    #         # ann = nusc.get('sample_annotation', ann_tokens[j])
    #         ann = nusc.get('sample_annotation', cur_ann_token)
    #         sam = nusc.get('sample', ann['sample_token'])
    #         cur_sensors = []
    #         for sensor in sensors:
    #             _, bbs, _ = nusc.get_sample_data(sam['data'][sensor])
    #             for bb in bbs:
    #                 if bb.token == ann['token']:
    #                     cur_sensors.append(sensor)
    #                     break
    #         print(f'instance {i} ann {j} is observed by {cur_sensors}')
    #         cur_ann_token = ann['next']
    #         j += 1
    # res_path = '/home/y_feng/workspace6/work/ProtoPNet/ProtoPNet/module_test/check_divided_track'
    # makedir(res_path)
    # nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=True)
    # anns_path = '/home/y_feng/workspace6/datasets/nusc/extra/anns_train_human_CAM_FRONT.pkl'
    # n_ann = 0
    # with open(anns_path, 'rb') as f:
    #     instk_to_anntk = pickle.load(f)
    # for instk in instk_to_anntk:
    #     anntk_seqs = instk_to_anntk[instk]
    #     for seq in anntk_seqs:
    #         n_ann += len(seq)
    # print(n_ann)
    #     if len(anntk_seqs) > 1:
    #         print([len(anns) for anns in anntk_seqs])
    #         cur_ins_path = os.path.join(res_path, instk)
    #         makedir(cur_ins_path)
    #         for i in range(len(anntk_seqs)):
    #             for j in range(len(anntk_seqs[i])):
    #                 img_nm = f'{i}_{j}.png'
    #                 anntk = anntk_seqs[i][j]
    #                 box = nusc_3dbbox_to_2dbbox(nusc, anntk)
    #                 ann = nusc.get('sample_annotation', anntk)
    #                 sam = nusc.get('sample', ann['sample_token'])
    #                 cam_front_data = nusc.get('sample_data', sam['data']['CAM_FRONT'])
    #                 img_path = os.path.join(NUSC_ROOT, cam_front_data['filename'])
    #                 img = cv2.imread(img_path)
    #                 img = draw_box(img, box)
    #                 save_path = os.path.join(cur_ins_path, img_nm)
    #                 cv2.imwrite(save_path, img)