import cv2
from audioop import reverse
from hashlib import new
import pickle
from re import T
from turtle import resizemode
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
import time
import copy
import tqdm
import os
from tqdm import tqdm
import pdb
import csv

from .pie_data import PIE
from .jaad_data import JAAD
from ..utils import makedir
from ..utils import mapping_20, ltrb2xywh, coord2pseudo_heatmap, TITANclip_txt2list, cls_weights
from ..data._img_mean_std import img_mean_std
from ..transforms import RandomHorizontalFlip, RandomResizedCrop, crop_local_ctx
from torchvision.transforms import functional as TVF
from config import dataset_root


ATOM_ACTION_LABEL_ORI = {  # 3 no samples; 7, 8 no test samples
    'standing': 0,
    'running': 1,
    'bending': 2,
    'kneeling': 3,
    'walking': 4,
    'sitting': 5,
    'squatting': 6,
    'jumping': 7,
    'laying down': 8,
    'none of the above': 9,
    # '': 9
}

ATOM_ACTION_LABEL_CORRECTED1 = {  # combine kneel, jump, lay down and none of the above
    'standing': 0,
    'running': 1,
    'bending': 2,
    'walking': 3,
    'sitting': 4,
    'squatting': 5,

    'kneeling': 6,
    'jumping': 6,
    'laying down': 6,
    'none of the above': 6,
    # '': 6
}

ATOM_ACTION_LABEL_CORRECTED2 = {  # remove kneel
    'standing': 0,
    'running': 1,
    'bending': 2,

    'walking': 3,
    'sitting': 4,
    'squatting': 5,
    'jumping': 6,
    'laying down': 7,
    'none of the above': 8,
}

ATOM_ACTION_LABEL_CHOSEN = {  # combine kneel, jump, lay down, squat and none of the above
    'standing': 0,
    'running': 1,
    'bending': 2,
    'walking': 3,
    'sitting': 4,

    'squatting': 5,
    'kneeling': 5,
    'jumping': 5,
    'laying down': 5,
    'none of the above': 5,
    # '': 6
}

ATOM_ACTION_LABEL_CHOSEN2 = {  # combine kneel, jump, lay down, squat and none of the above
    'standing': 0,
    'running': 1,
    'bending': 2,
    'walking': 3,
    'sitting': 4,

    # 'squatting': 5,
    # 'kneeling': 5,
    # 'jumping': 5,
    # 'laying down': 5,
    # 'none of the above': 5,
    # '': 6
}

SIMPLE_CONTEXTUAL_LABEL = {
    'crossing a street at pedestrian crossing': 0,
    'jaywalking (illegally crossing NOT at pedestrian crossing)': 1,
    'waiting to cross street': 2,
    'motorcycling': 3,
    'biking': 4,
    'walking along the side of the road': 5,
    'walking on the road': 6,
    'cleaning an object': 7,
    'closing': 8,
    'opening': 9,
    'exiting a building': 10,
    'entering a building': 11,
    'none of the above': 12,
    # '': 12
}

COMPLEX_CONTEXTUAL_LABEL = {
    'unloading': 0,
    'loading': 1,
    'getting in 4 wheel vehicle': 2,
    'getting out of 4 wheel vehicle': 3,
    'getting on 2 wheel vehicle': 4,
    'getting off 2 wheel vehicle': 5,
    'none of the above': 6,
    # '': 6
}

COMMUNICATIVE_LABEL = {
    'looking into phone': 0,
    'talking on phone': 1,
    'talking in group': 2,
    'none of the above': 3,
    # '': 3
}

TRANSPORTIVE_LABEL = {
    'pushing': 0,
    'carrying with both hands': 1,
    'pulling': 2,
    'none of the above': 3,
    # '': 3
}

MOTOIN_STATUS_LABEL = {
    'stopped': 0,
    'moving': 1,
    'parked': 2,
    'none of the above': 3,
    # '': 3
}

AGE_LABEL = {
    'child': 0,
    'adult': 1,
    'senior over 65': 2,
    # '': 3
}

LABEL2COLUMN = {
    'img_nm': 0,
    'obj_type': 1,
    'obj_id': 2,
    'trunk_open': 7,
    'motion_status': 8,
    'doors_open': 9,
    'communicative': 10,
    'complex_context': 11,
    'atomic_actions': 12,
    'simple_context': 13,
    'transporting': 14,
    'age': 15
}



ATOM_ACTION_LABEL = ATOM_ACTION_LABEL_CHOSEN

LABEL2DICT = {
    'atomic_actions': ATOM_ACTION_LABEL,
    'simple_context': SIMPLE_CONTEXTUAL_LABEL,
    'complex_context': COMPLEX_CONTEXTUAL_LABEL,
    'communicative': COMMUNICATIVE_LABEL,
    'transporting': TRANSPORTIVE_LABEL,
    'age': AGE_LABEL,
}

NUM_CLS_ATOMIC = max([ATOM_ACTION_LABEL[k] for k in ATOM_ACTION_LABEL]) + 1
NUM_CLS_SIMPLE = max([SIMPLE_CONTEXTUAL_LABEL[k] for k in SIMPLE_CONTEXTUAL_LABEL]) + 1
NUM_CLS_COMPLEX = max([COMPLEX_CONTEXTUAL_LABEL[k] for k in COMPLEX_CONTEXTUAL_LABEL]) + 1
NUM_CLS_COMMUNICATIVE = max([COMMUNICATIVE_LABEL[k] for k in COMMUNICATIVE_LABEL]) + 1
NUM_CLS_TRANSPORTING = max([TRANSPORTIVE_LABEL[k] for k in TRANSPORTIVE_LABEL]) + 1
NUM_CLS_AGE = max([AGE_LABEL[k] for k in AGE_LABEL]) + 1

OCC_NUM = 0

LABEL_2_KEY = {
    'crossing': 'cross',
    'atomic_actions': 'atomic',
    'complex_context': 'complex',
    'communicative': 'communicative',
    'transporting': 'transporting',
}

KEY_2_LABEL = {
    'cross': 'crossing',
    'atomic': 'atomic_actions',
    'complex': 'complex_context',
    'communicative': 'communicative',
    'transporting': 'transporting',
}

LABEL_2_IMBALANCE_CLS = {
    'crossing': [1],
    'atomic_actions': [1, 2, 4, 5],
    'complex_context': [0, 1, 2, 3, 4, 5],
    'communicative': [0, 1, 2],
    'transporting': [0, 1, 2],
}

class TITAN_dataset(Dataset):
    def __init__(self,
                 sub_set='default_train',
                 track_save_path='',
                 norm_traj=True,
                 neighbor_mode='last_frame',
                 obs_len=6, pred_len=1, overlap_ratio=0.5, recog_act=0,
                 obs_interval=0,
                 color_order='BGR', img_norm_mode='torch',
                 required_labels=['atomic_actions', 'simple_context'], 
                 multi_label_cross=0, 
                 use_atomic=0, use_complex=0, use_communicative=0, 
                 use_transporting=0, use_age=0,
                 loss_weight='sklearn',
                 tte=None,
                 small_set=0,
                 use_img=1, img_mode='even_padded', img_size=(224, 224),
                 use_ctx=1, ctx_mode='local', ctx_size=(224, 224),
                 use_skeleton=0, sk_mode='pseudo_heatmap',
                 use_traj=1, traj_mode='ltrb',
                 use_ego=1,
                 augment_mode='none',
                 seg_cls=['person', 'vehicles', 'roads', 'traffic_lights'],
                 pop_occl_track=1,
                 ) -> None:
        super(Dataset, self).__init__()
        self.dataset_name = 'TITAN'
        self.sub_set = sub_set
        self.norm_traj = norm_traj
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.obs_interval = obs_interval
        self.overlap_ratio = overlap_ratio
        self.recog_act = recog_act
        self.obs_or_pred = 'obs' if self.recog_act else 'pred'
        self.color_order = color_order
        self.img_norm_mode = img_norm_mode

        self.img_mean, self.img_std = img_mean_std(self.img_norm_mode)

        # sequence length considering interval
        self._obs_len = self.obs_len * (self.obs_interval + 1)
        self._pred_len = self.pred_len * (self.obs_interval + 1)

        self.use_img = use_img
        self.img_mode = img_mode
        self.img_size = img_size
        self.use_ctx = use_ctx
        self.ctx_mode = ctx_mode
        self.ctx_size = ctx_size
        self.use_skeleton = use_skeleton
        self.sk_mode = sk_mode
        self.use_traj = use_traj
        self.traj_mode = traj_mode
        self.use_ego = use_ego
        self.track_save_path = track_save_path
        self.required_labels = required_labels
        self.multi_label_cross = multi_label_cross
        self.use_atomic = use_atomic
        self.use_complex = use_complex
        self.use_communicative = use_communicative
        self.use_transporting = use_transporting
        self.use_age = use_age
        self.loss_weight = loss_weight
        self.neighbor_mode = neighbor_mode
        self.tte = tte
        self.small_set = small_set
        self.augment_mode = augment_mode
        self.seg_cls = seg_cls
        self.pop_occl_track = pop_occl_track
        self.transforms = {'random': 0,
                            'balance': 0,
                            'hflip': None,
                            'resized_crop': {'img': None,
                                            'ctx': None,
                                            'sk': None}}

        self.ori_data_root = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset')
        self.extra_data_root = os.path.join(dataset_root, 'TITAN/TITAN_extra')
        self.cropped_img_root = os.path.join(self.extra_data_root,
                                             'cropped_images', 
                                             self.resize_mode, 
                                             str(img_size[1])+'w_by_'\
                                                +str(img_size[0])+'h')
        if self.ctx_format == 'ped_graph':
            ctx_format_dir = 'ori_local'
        else:
            ctx_format_dir = self.ctx_format
        self.ctx_root = os.path.join(self.extra_data_root, 
                                     'context', 
                                     ctx_format_dir, 
                                     str(ctx_size[1])+'w_by_'\
                                        +str(ctx_size[0])+'h')
        self.ped_ori_local_root = os.path.join(dataset_root, 'TITAN/TITAN_extra/context/ori_local/224w_by_224h/ped')
        self.sk_vis_path = os.path.join(dataset_root, 'TITAN/TITAN_extra/sk_vis/even_padded/288w_by_384h/')
        self.sk_coord_path = os.path.join(dataset_root, 'TITAN/TITAN_extra/sk_coords/even_padded/288w_by_384h/')
        self.sk_heatmap_path = os.path.join(dataset_root, 'TITAN/TITAN_extra/sk_heatmaps/even_padded/288w_by_384h/')
        self.sk_p_heatmap_path = os.path.join(dataset_root, 'TITAN/TITAN_extra/sk_pseudo_heatmaps/even_padded/48w_by_48h/')
        self.seg_root = os.path.join(dataset_root, 'TITAN/TITAN_extra/seg_sam')
        if self.sub_set == 'default_train':
            clip_txt_path = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset/train_set.txt')
        elif self.sub_set == 'default_val':
            clip_txt_path = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset/val_set.txt')
        elif self.sub_set == 'default_test':
            clip_txt_path = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/dataset/test_set.txt')
        elif self.sub_set == 'all':
            clip_txt_path = os.path.join(dataset_root, 'TITAN/honda_titan_dataset/titan_clips.txt')
        else:
            raise NotImplementedError(self.sub_set)
        self.imgnm_to_objid_path = os.path.join(self.extra_data_root, 
                                                'imgnm_to_objid_to_ann.pkl')

        # load clip ids
        self.clip_id_list = TITANclip_txt2list(clip_txt_path)  # list of str
        # load tracks info
        if os.path.exists(self.track_save_path):
            with open(self.track_save_path, 'rb') as f:
                track_info = pickle.load(f)
            self.ids = track_info['ids']
            self.p_tracks = track_info['p_tracks']
            self.num_p_tracks = track_info['num_p_tracks']
            self.v_tracks = track_info['v_tracks']
            self.num_v_tracks = track_info['num_v_tracks']
            if self.neighbor_mode == 'last_frame':
                pass
        else:
            annos, self.ids = self.add_cid()
            self.p_tracks, self.num_p_tracks = self.get_p_tracks(annos)
            self.v_tracks, self.num_v_tracks = self.get_v_tracks(annos)
            track_info = {
                'ids': self.ids,
                'p_tracks': self.p_tracks,
                'num_p_tracks': self.num_p_tracks,
                'v_tracks': self.v_tracks,
                'num_v_tracks': self.num_v_tracks
            }

            track_f_nm = 'neighbors.pkl'
            if self.neighbor_mode:
                track_f_nm = 'w_' + track_f_nm
            else:
                track_f_nm = 'wo_' + track_f_nm
            track_save_path = os.path.join(self.extra_data_root, 
                                           'saved_tracks', 
                                           self.sub_set, 
                                           track_f_nm)
            with open(track_save_path, 'wb') as f:
                pickle.dump(track_info, f)

        # crop_imgs(self.p_tracks, resize_mode=img_mode, target_size=img_size, obj_type='p')
        # print('Crop done')
        # return
        # get cid to img name to obj id dict
        if not os.path.exists(self.imgnm_to_objid_path):
            self.imgnm_to_objid = \
                self.get_imgnm_to_objid(self.p_tracks, 
                                        self.v_tracks, 
                                        self.imgnm_to_objid_path)
        else:
            with open(self.imgnm_to_objid_path, 'rb') as f:
                self.imgnm_to_objid = pickle.load(f)

        # filter short tracks
        self.p_tracks_filtered, self.num_p_tracks = \
            self.filter_short_tracks(self.p_tracks, 
                                     self._obs_len+self._pred_len)

        self.samples = self.track2sample(self.p_tracks_filtered)

        # convert samples to ndarray ?

        # num samples
        self.num_samples = len(self.samples['obs']['img_nm'])

        # apply interval
        if self.obs_interval > 0:
            self.downsample_seq()
            
        # small set
        if small_set > 0:
            small_set_size = int(self.num_samples * small_set)
            for k in self.samples['obs'].keys():
                self.samples['obs'][k] = self.samples['obs'][k]\
                    [:small_set_size]
            for k in self.samples['pred'].keys():
                self.samples['pred'][k] = self.samples['pred'][k]\
                    [:small_set_size]
            self.num_samples = small_set_size

        # cross or not
        obs_cross_labels = self.multi2binary(self.samples['obs']\
                                             ['simple_context'], 
                                             [0, 1])
        self.samples['obs']['crossing'] = obs_cross_labels
        pred_cross_labels = self.multi2binary(self.samples['pred']\
                                              ['simple_context'], 
                                              [0, 1])
        self.samples['pred']['crossing'] = pred_cross_labels
        print('num samples: ', self.num_samples)

        # augmentation
        self.samples = self._add_augment(self.samples)

        # class count
        print(self.sub_set, 
              'pred crossing', 
              len(self.samples[self.obs_or_pred]['crossing']), 
              self.num_samples, 
              self.samples[self.obs_or_pred]['crossing'][-1])
        self.n_c = np.sum(np.array(self.samples[self.obs_or_pred]['crossing'])\
                          [:, -1])
        print('self.n_c', self.n_c)
        self.n_nc = self.num_samples - self.n_c
        self.num_samples_cls = [self.n_nc, self.n_c]
        self.class_weights = {}
        if self.multi_label_cross:
            labels = np.squeeze(self.samples[self.obs_or_pred]\
                                ['simple_context'])
            # print(labels.shape, labels)
            self.num_samples_cls = []
            for i in range(13):
                n_cur_cls = sum(labels == i)
                self.num_samples_cls.append(n_cur_cls)
            
            print('label distr', self.num_samples, self.num_samples_cls)
        print('self.num_samples_cls', self.num_samples_cls)
        self.class_weights['cross'] = cls_weights(self.num_samples_cls, 
                                                  'sklearn')

        if self.use_atomic:
            labels = np.squeeze(self.samples['pred']['atomic_actions'])
            self.num_samples_atomic = []
            for i in range(NUM_CLS_ATOMIC):
                n_cur_cls = sum(labels == i)
                self.num_samples_atomic.append(n_cur_cls)
            print('atomic label distr', 
                  self.num_samples, 
                  self.num_samples_atomic)
            self.class_weights['atomic'] = cls_weights(self.num_samples_atomic, 
                                                       'sklearn')
        if self.use_complex:
            labels = np.squeeze(self.samples['pred']['complex_context'])
            self.num_samples_complex = []
            for i in range(NUM_CLS_COMPLEX):
                n_cur_cls = sum(labels == i)
                self.num_samples_complex.append(n_cur_cls)
            assert sum(self.num_samples_complex) == self.num_samples, \
                sum(self.num_samples_complex)
            print('complex label distr', 
                  self.num_samples, 
                  self.num_samples_complex)
            self.class_weights['complex'] = \
                cls_weights(self.num_samples_complex, 'sklearn')
        if self.use_communicative:
            labels = np.squeeze(self.samples['pred']['communicative'])
            self.num_samples_communicative = []
            for i in range(NUM_CLS_COMMUNICATIVE):
                n_cur_cls = sum(labels == i)
                self.num_samples_communicative.append(n_cur_cls)
            print('communicative label distr', 
                  self.num_samples, 
                  self.num_samples_communicative)
            self.class_weights['communicative'] = \
                cls_weights(self.num_samples_communicative, 'sklearn')
        if self.use_transporting:
            labels = np.squeeze(self.samples['pred']['transporting'])
            self.num_samples_transporting = []
            for i in range(NUM_CLS_TRANSPORTING):
                n_cur_cls = sum(labels == i)
                self.num_samples_transporting.append(n_cur_cls)
            print('transporting label distr', 
                  self.num_samples, 
                  self.num_samples_transporting)
            self.class_weights['transporting'] = \
                cls_weights(self.num_samples_transporting, 'sklearn')
        if self.use_age:
            labels = np.squeeze(self.samples['pred']['age'])
            self.num_samples_age = []
            for i in range(NUM_CLS_AGE):
                n_cur_cls = sum(labels == i)
                self.num_samples_age.append(n_cur_cls)
            self.class_weights['age'] = \
                cls_weights(self.num_samples_age, 'sklearn')
        
        # apply interval
        if self.obs_interval > 0:
            self.donwsample_seq()
            print('Applied interval')
            print('cur input len', len(self.samples['obs_img_nm_int']))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        obs_bbox = torch.tensor(self.samples['obs']['bbox_normed'][idx]).float()
        obs_bbox_unnormed = torch.tensor(self.samples['obs']['bbox'][idx]).float()
        pred_bbox = torch.tensor(self.samples['pred']['bbox_normed'][idx]).float()
        obs_ego = torch.tensor(self.samples['obs']['ego_motion'][idx]).float()
        clip_id_int = torch.tensor(int(self.samples['obs']['clip_id'][idx][0]))
        ped_id_int = torch.tensor(int(float(self.samples['obs']['obj_id'][idx][0])))
        img_nm_int = torch.tensor(self.samples['obs']['img_nm_int'][idx])

        # squeeze the coords
        if '0-1' in self.traj_mode:
            obs_bbox[:, 0] /= 2704
            obs_bbox[:, 2] /= 2704
            obs_bbox[:, 1] /= 1520
            obs_bbox[:, 3] /= 1520
        # act labels
        if self.multi_label_cross:
            pred_intent = torch.tensor([self.samples[self.obs_or_pred]\
                                        ['simple_context'][idx][-1]])  # int
        else:
            pred_intent = torch.tensor([self.samples[self.obs_or_pred]\
                                        ['crossing'][idx][-1]])  # int
        simple_context = torch.tensor([self.samples[self.obs_or_pred]\
                                       ['simple_context'][idx][-1]])
        atomic_action = torch.tensor([self.samples[self.obs_or_pred]\
                                      ['atomic_actions'][idx][-1]])
        complex_context = torch.tensor([self.samples[self.obs_or_pred]\
                                        ['complex_context'][idx][-1]])
        communicative = torch.tensor([self.samples[self.obs_or_pred]\
                                      ['communicative'][idx][-1]])
        transporting = torch.tensor([self.samples[self.obs_or_pred]\
                                     ['transporting'][idx][-1]])
        age = torch.tensor(self.samples[self.obs_or_pred]['age'][idx])
        sample = {'ped_id_int': ped_id_int,
                  'clip_id_int': clip_id_int,
                  'img_nm_int': img_nm_int,
                  'obs_bboxes': obs_bbox,
                  'obs_bboxes_unnormed': obs_bbox_unnormed,
                  'obs_ego': obs_ego,
                  'pred_intent': pred_intent,
                  'atomic_actions': atomic_action,
                  'simple_context': simple_context,
                  'complex_context': complex_context,  # (1,)
                  'communicative': communicative,
                  'transporting': transporting,
                  'age': age,
                  'hflip_flag': torch.tensor(0),
                  'img_ijhw': torch.tensor([-1, -1, -1, -1]),
                  'ctx_ijhw': torch.tensor([-1, -1, -1, -1]),
                  'sk_ijhw': torch.tensor([-1, -1, -1, -1]),
                  }
        if self.use_img:
            imgs = []
            for img_nm in self.samples['obs']['img_nm'][idx]:
                img_path = os.path.join(self.cropped_img_root, 
                                        'ped', 
                                        self.samples['obs']['clip_id']\
                                            [idx][0], 
                                        str(int(float(self.samples['obs']\
                                                      ['obj_id'][idx][0]))), 
                                        img_nm)
                imgs.append(cv2.imread(img_path))
            imgs = np.stack(imgs, axis=0)
            # (T, H, W, C) -> (C, T, H, W)
            ped_imgs = torch.from_numpy(imgs).float().permute(3, 0, 1, 2)
            # normalize img
            if self.img_norm_mode != 'ori':
                ped_imgs /= 255.
                ped_imgs[0, :, :, :] -= self.img_mean[0]
                ped_imgs[1, :, :, :] -= self.img_mean[1]
                ped_imgs[2, :, :, :] -= self.img_mean[2]
                ped_imgs[0, :, :, :] /= self.img_std[0]
                ped_imgs[1, :, :, :] /= self.img_std[1]
                ped_imgs[2, :, :, :] /= self.img_std[2]
            # BGR -> RGB
            if self.color_order == 'RGB':
                ped_imgs = torch.from_numpy(
                    np.ascontiguousarray(ped_imgs.numpy()[::-1, :, :, :]))
            sample['ped_imgs'] = ped_imgs
        if self.use_ctx:
            if self.ctx_mode in ('local', 'ori_local', 'mask_ped', 'ori'):
                ctx_imgs = []
                for img_nm in self.samples['obs']['img_nm'][idx]:
                    img_path = os.path.join(self.ctx_root, 
                                            'ped', 
                                            self.samples['obs']['clip_id']\
                                                [idx][0], 
                                            str(int(float(self.samples\
                                                          ['obs']['obj_id']\
                                                            [idx][0]))), 
                                            img_nm)
                    ctx_imgs.append(cv2.imread(img_path))
                ctx_imgs = np.stack(ctx_imgs, axis=0)
                # (T, H, W, C) -> (C, T, H, W)
                ctx_imgs = torch.from_numpy(ctx_imgs).float().\
                    permute(3, 0, 1, 2)
                # normalize img
                if self.img_norm_mode != 'ori':
                    ctx_imgs /= 255.
                    ctx_imgs[0, :, :, :] -= self.img_mean[0]
                    ctx_imgs[1, :, :, :] -= self.img_mean[1]
                    ctx_imgs[2, :, :, :] -= self.img_mean[2]
                    ctx_imgs[0, :, :, :] /= self.img_std[0]
                    ctx_imgs[1, :, :, :] /= self.img_std[1]
                    ctx_imgs[2, :, :, :] /= self.img_std[2]
                # RGB -> BGR
                if self.color_order == 'RGB':
                    ctx_imgs = torch.from_numpy(\
                        np.ascontiguousarray(ctx_imgs.numpy()[::-1, :, :, :]))
                sample['obs_context'] = ctx_imgs  # shape [3, obs_len, H, W]
            
            elif self.ctx_mode == 'seg_ori_local' \
                or self.ctx_mode == 'seg_local':
                # load imgs
                ctx_imgs = []
                for img_nm in self.samples['obs']['img_nm'][idx]:
                    img_path = os.path.join(self.ped_ori_local_root, 
                                            self.samples['obs']['clip_id']\
                                                [idx][0], 
                                            str(int(float(self.samples['obs']\
                                                          ['obj_id'][idx][0]))), 
                                                          img_nm)
                    ctx_imgs.append(cv2.imread(img_path))
                ctx_imgs = np.stack(ctx_imgs, axis=0)
                # (T, H, W, C) -> (C, T, H, W)
                ctx_imgs = torch.from_numpy(ctx_imgs).float().permute(3, 0, 1, 2)
                # normalize img
                if self.img_norm_mode != 'ori':
                    ctx_imgs /= 255.
                    ctx_imgs[0, :, :, :] -= self.img_mean[0]
                    ctx_imgs[1, :, :, :] -= self.img_mean[1]
                    ctx_imgs[2, :, :, :] -= self.img_mean[2]
                    ctx_imgs[0, :, :, :] /= self.img_std[0]
                    ctx_imgs[1, :, :, :] /= self.img_std[1]
                    ctx_imgs[2, :, :, :] /= self.img_std[2]
                # RGB -> BGR
                if self.color_order == 'RGB':
                    ctx_imgs = torch.from_numpy(\
                        np.ascontiguousarray(ctx_imgs.numpy()[::-1, :, :, :]))  # 3THW
                # load segs
                ctx_segs = {c:[] for c in self.seg_cls}
                for c in self.seg_cls:
                    for img_nm in self.samples['obs']['img_nm'][idx]:
                        c_id = self.samples['obs']['clip_id'][idx][0]
                        f_nm = img_nm.replace('png', 'pkl')
                        seg_path = os.path.join(self.seg_root, c, c_id, f_nm)
                        with open(seg_path, 'rb') as f:
                            seg = pickle.load(f)
                        ctx_segs[c].append(torch.from_numpy(seg))
                for c in self.seg_cls:
                    ctx_segs[c] = torch.stack(ctx_segs[c], dim=0)  # THW
                # crop seg
                crop_segs = {c:[] for c in self.seg_cls}
                for i in range(ctx_imgs.size(1)):  # T
                    for c in self.seg_cls:
                        crop_seg = crop_local_ctx(
                            torch.unsqueeze(ctx_segs[c][i], dim=0), 
                            obs_bbox_unnormed[i], 
                            self.ctx_size, 
                            interpo='nearest')  # 1 h w
                        crop_segs[c].append(crop_seg)
                all_seg = []
                for c in self.seg_cls:
                    all_seg.append(torch.stack(crop_segs[c], dim=1))  # 1Thw
                all_seg = torch.stack(all_seg, dim=4)  # 1Thw n_cls
                sample['obs_context'] = \
                    all_seg * torch.unsqueeze(ctx_imgs, 
                dim=-1)  # 3Thw n_cls
                
        if self.use_skeleton:
            if self.sk_mode == 'pseudo_heatmap':
                cid = str(int(float(self.samples['obs']['clip_id'][idx][0])))
                pid = str(int(float(self.samples['obs']['obj_id'][idx][0])))
                heatmaps = []
                for img_nm in self.samples['obs']['img_nm'][idx]:
                    heatmap_nm = img_nm.replace('.png', '.pkl')
                    heatmap_path = os.path.join(self.sk_p_heatmap_path, cid, pid, heatmap_nm)
                    with open(heatmap_path, 'rb') as f:
                        heatmap = pickle.load(f)
                    heatmaps.append(heatmap)
                heatmaps = np.stack(heatmaps, axis=0)  # T C H W
                # T C H W -> C T H W
                obs_skeletons = torch.from_numpy(heatmaps).float().permute(1, 0, 2, 3)  # shape: (17, seq_len, 48, 48)
            elif self.sk_mode == 'coord':
                cid = str(int(float(self.samples['obs']['clip_id'][idx][0])))
                pid = str(int(float(self.samples['obs']['obj_id'][idx][0])))
                coords = []
                for img_nm in self.samples['obs']['img_nm'][idx]:
                    coord_nm = img_nm.replace('.png', '.pkl')
                    coord_path = os.path.join(self.sk_coord_path, cid, pid, coord_nm)
                    with open(coord_path, 'rb') as f:
                        coord = pickle.load(f)  # nj, 3
                    coords.append(coord[:, :2])  # nj, 2
                coords = np.stack(coords, axis=0)  # T, nj, 2
                try:
                    obs_skeletons = torch.from_numpy(coords).float().permute(2, 0, 1)  # shape: (2, T, nj)
                except:
                    print('coords shape',coords.shape)
                    import pdb;pdb.set_trace()
                    raise NotImplementedError()
            else:
                raise NotImplementedError(self.sk_mode)
            sample['obs_skeletons'] = obs_skeletons

        # augmentation
        if self.augment_mode != 'none':
            if self.transforms['random']:
                sample = self._random_augment(sample)
            elif self.transforms['balance']:
                sample['hflip_flag'] = torch.tensor(self.samples[self.obs_or_pred]['hflip_flag'][idx])
                sample = self._augment(sample)

        return sample

    def _augment(self, sample):
        # flip
        if sample['hflip_flag']:
            if self.use_img:
                sample['ped_imgs'] = TVF.hflip(sample['ped_imgs'])
            if self.use_ctx:
                sample['obs_context'] = TVF.hflip(sample['obs_context'])
            if self.use_skeleton and ('heatmap' in self.sk_mode):
                sample['obs_skeletons'] = TVF.hflip(sample['obs_skeletons'])
            if self.use_traj:
                sample['obs_bboxes_unnormed'][:, 0], sample['obs_bboxes_unnormed'][:, 2] = \
                    2704 - sample['obs_bboxes_unnormed'][:, 2], 2704 - sample['obs_bboxes_unnormed'][:, 0]
                if '0-1' in self.traj_mode:
                    sample['obs_bboxes'][:, 0], sample['obs_bboxes'][:, 2] =\
                         1 - sample['obs_bboxes'][:, 2], 1 - sample['obs_bboxes'][:, 0]
                else:
                    sample['obs_bboxes'][:, 0], sample['obs_bboxes'][:, 2] =\
                         2704 - sample['obs_bboxes'][:, 2], 2704 - sample['obs_bboxes'][:, 0]
            if self.use_ego:
                sample['obs_ego'][:, -1] = -sample['obs_ego'][:, -1]
        # resized crop
        if self.transforms['resized_crop']['img'] is not None:
            sample['ped_imgs'], ijhw = self.transforms['resized_crop']['img'](sample['ped_imgs'])
            self.transforms['resized_crop']['img'].randomize_parameters()
            sample['img_ijhw'] = torch.tensor(ijhw)
        if self.transforms['resized_crop']['ctx'] is not None:
            sample['obs_context'], ijhw = self.transforms['resized_crop']['ctx'](sample['obs_context'])
            self.transforms['resized_crop']['ctx'].randomize_parameters()
            sample['ctx_ijhw'] = torch.tensor(ijhw)
        if self.transforms['resized_crop']['sk'] is not None:
            sample['obs_skeletons'], ijhw = self.transforms['resized_crop']['sk'](sample['obs_skeletons'])
            self.transforms['resized_crop']['sk'].randomize_parameters()
            sample['sk_ijhw'] = torch.tensor(ijhw)
        return sample

    def _random_augment(self, sample):
        # flip
        if self.transforms['hflip'] is not None:
            self.transforms['hflip'].randomize_parameters()
            sample['hflip_flag'] = torch.tensor(self.transforms['hflip'].flag)
            # print('before aug', self.transforms['hflip'].flag, sample['hflip_flag'], self.transforms['hflip'].random_p)
            if self.use_img:
                sample['ped_imgs'] = self.transforms['hflip'](sample['ped_imgs'])
            # print('-1', self.transforms['hflip'].flag, sample['hflip_flag'], self.transforms['hflip'].random_p)
            if self.use_ctx:
                if self.ctx_mode == 'seg_ori_local' or self.ctx_mode == 'seg_local':
                    sample['obs_context'] = self.transforms['hflip'](sample['obs_context'].permute(4, 0, 1, 2, 3)).permute(1, 2, 3, 4, 0)
                sample['obs_context'] = self.transforms['hflip'](sample['obs_context'])
            if self.use_skeleton and ('heatmap' in self.sk_mode):
                sample['obs_skeletons'] = self.transforms['hflip'](sample['obs_skeletons'])
            if self.use_traj and self.transforms['hflip'].flag:
                sample['obs_bboxes_unnormed'][:, 0], sample['obs_bboxes_unnormed'][:, 2] = \
                    2704 - sample['obs_bboxes_unnormed'][:, 2], 2704 - sample['obs_bboxes_unnormed'][:, 0]
                if '0-1' in self.traj_mode:
                    sample['obs_bboxes'][:, 0], sample['obs_bboxes'][:, 2] =\
                         1 - sample['obs_bboxes'][:, 2], 1 - sample['obs_bboxes'][:, 0]
                else:
                    sample['obs_bboxes'][:, 0], sample['obs_bboxes'][:, 2] =\
                         2704 - sample['obs_bboxes'][:, 2], 2704 - sample['obs_bboxes'][:, 0]
            if self.use_ego and self.transforms['hflip'].flag:
                sample['obs_ego'][:, -1] = -sample['obs_ego'][:, -1]
            
        # resized crop
        if self.transforms['resized_crop']['img'] is not None:
            self.transforms['resized_crop']['img'].randomize_parameters()
            sample['ped_imgs'], ijhw = self.transforms['resized_crop']['img'](sample['ped_imgs'])
            sample['img_ijhw'] = torch.tensor(ijhw)
        if self.transforms['resized_crop']['ctx'] is not None:
            self.transforms['resized_crop']['ctx'].randomize_parameters()
            sample['obs_context'], ijhw = self.transforms['resized_crop']['ctx'](sample['obs_context'])
            sample['ctx_ijhw'] = torch.tensor(ijhw)
        if self.transforms['resized_crop']['sk'] is not None:
            self.transforms['resized_crop']['sk'].randomize_parameters()
            sample['obs_skeletons'], ijhw = self.transforms['resized_crop']['sk'](sample['obs_skeletons'])
            sample['sk_ijhw'] = torch.tensor(ijhw)
        return sample

    def add_cid(self):
        annos = {}
        ids = {}
        n_clip = 0
        for cid in self.clip_id_list:
            n_clip += 1
            ids[cid] = {'pid':set(), 'vid':set()}
            csv_path = os.path.join(self.ori_data_root, 'clip_'+cid+'.csv')
            # print(f'cid {cid} n clips {n_clip}')
            clip_obj_info = self.read_obj_csv(csv_path)
            for i in range(len(clip_obj_info)):
                line = clip_obj_info[i]
                assert len(line) == 16
                if line[1] == 'person':
                    ids[cid]['pid'].add((cid, line[2]))
                else:
                    ids[cid]['vid'].add((cid, line[2]))
                clip_obj_info[i].append(cid)
            annos[cid] = self.str2ndarray(clip_obj_info)

        return annos, ids
    
    def _add_augment(self, data):
        '''
        data: self.samples, dict of lists(num samples, ...)
        transforms: torchvision.transforms
        '''
        if 'crop' in self.augment_mode:
            if self.use_img:
                self.transforms['resized_crop']['img'] = \
                    RandomResizedCrop(size=self.img_size, # (h, w)
                                    scale=(0.75, 1), 
                                    ratio=(1., 1.))  # w / h
            if self.use_ctx:
                self.transforms['resized_crop']['ctx'] = \
                    RandomResizedCrop(size=self.ctx_size, # (h, w)
                                      scale=(0.75, 1), 
                                      ratio=(self.ctx_size[1]/self.ctx_size[0], 
                                             self.ctx_size[1]/self.ctx_size[0]))  # w / h
            if self.use_skeleton and self.sk_mode == 'pseudo_heatmap':
                self.transforms['resized_crop']['sk'] = \
                    RandomResizedCrop(size=(48, 48), # (h, w)
                                        scale=(0.75, 1), 
                                        ratio=(1, 1))  # w / h
        if 'hflip' in self.augment_mode:
            if 'random' in self.augment_mode:
                self.transforms['random'] = 1
                self.transforms['balance'] = 0
                self.transforms['hflip'] = RandomHorizontalFlip(p=0.5)
            elif 'balance' in self.augment_mode:
                print(f'Num samples before flip: {self.num_samples}')
                self.transforms['random'] = 0
                self.transforms['balance'] = 1
                imbalance_sets = []

                # init extra samples
                h_flip_samples = {
                    'obs':{},
                    'pred':{}
                }
                for k in data['obs']:
                    h_flip_samples['obs'][k] = []
                    h_flip_samples['pred'][k] = []

                # keys to check
                for k in KEY_2_LABEL:
                    if k in self.augment_mode:
                        imbalance_sets.append(KEY_2_LABEL[k])
                # duplicate samples
                for i in range(len(data['obs']['img_nm'])):
                    for label in imbalance_sets:
                        if data[self.obs_or_pred][label][i][-1] \
                            in LABEL_2_IMBALANCE_CLS[label]:
                            for k in data['obs']:
                                h_flip_samples['obs'][k].append(
                                    copy.deepcopy(data['obs'][k][i]))
                                h_flip_samples['pred'][k].append(
                                    copy.deepcopy(data['pred'][k][i]))
                        break
                h_flip_samples['obs']['hflip_flag'] = \
                    [True for i in range(len(h_flip_samples['obs']['img_nm']))]
                h_flip_samples['pred']['hflip_flag'] = \
                    [True for i in range(len(h_flip_samples['pred']['img_nm']))]
                data['obs']['hflip_flag'] = \
                    [False for i in range(len(data['obs']['img_nm']))]
                data['pred']['hflip_flag'] = \
                    [False for i in range(len(data['pred']['img_nm']))]

                # concat
                for k in data['obs']:
                    data['obs'][k].extend(h_flip_samples['obs'][k])
                    data['pred'][k].extend(h_flip_samples['pred'][k])
            self.num_samples = len(data['obs']['img_nm'])
            print(f'Num samples after flip: {self.num_samples}')
        return data

    def get_p_tracks(self, annos):
        p_tracks = {'clip_id': [],
                    'img_nm': [],
                    'img_nm_int': [],
                    'obj_id': [],
                    'bbox': [],
                    # 'motion_status': [],
                    'communicative': [],
                    'complex_context': [],
                    'atomic_actions': [],
                    'simple_context': [],
                    'transporting': [],
                    'age': [],
                    'ego_motion': []}
        for cid in self.ids.keys():
            # load ego motion
            ego_v_path = os.path.join(self.ori_data_root, 'clip_'+cid, 'synced_sensors.csv')
            ego_v_info = self.read_ego_csv(ego_v_path)  # dict {'img_nm': [info]}
            clip_annos = annos[cid]
            for _, pid in self.ids[cid]['pid']:
                # init new track
                for k in p_tracks.keys():
                    p_tracks[k].append([])
                # filter lines
                lines = clip_annos[(clip_annos[:, 2] == pid) & (clip_annos[:, 1] == 'person')]
                for line in lines:
                    # check if required labels exist
                    flg = 0
                    for label in LABEL2DICT:
                        idx = LABEL2COLUMN[label]
                        # print(line[idx], type(line[idx]))
                        cur_s = line[idx]
                        if cur_s == '':
                            flg = 1
                            OCC_NUM += 1
                            print('occlusion', OCC_NUM)
                            break
                        elif (label in LABEL2DICT) and \
                            (cur_s not in LABEL2DICT[label]):
                            flg = 1
                            # print('Class not to recognize: ', line[idx])
                            break
                    if flg == 1:
                        # pop current track
                        if self.pop_occl_track:
                            for k in p_tracks:
                                p_tracks[k].pop(-1)
                            break
                        # start a new track
                        else:
                            # init new track
                            for k in p_tracks.keys():
                                p_tracks[k].append([])
                            continue
                    p_tracks['clip_id'][-1].append(cid)  # str
                    p_tracks['obj_id'][-1].append(str(int(float(pid))))  # str
                    p_tracks['img_nm'][-1].append(line[0])  # str
                    p_tracks['img_nm_int'][-1].append(int(line[0].replace('.png', '')))
                    tlhw = list(map(float, line[3: 7]))
                    ltrb = [tlhw[1], tlhw[0], tlhw[1]+tlhw[3], tlhw[0]+tlhw[2]]
                    p_tracks['bbox'][-1].append(ltrb)
                    p_tracks['communicative'][-1].append(COMMUNICATIVE_LABEL[line[10]])
                    p_tracks['complex_context'][-1].append(COMPLEX_CONTEXTUAL_LABEL[line[11]])
                    p_tracks['atomic_actions'][-1].append(ATOM_ACTION_LABEL[line[12]])
                    p_tracks['simple_context'][-1].append(SIMPLE_CONTEXTUAL_LABEL[line[13]])
                    p_tracks['transporting'][-1].append(TRANSPORTIVE_LABEL[line[14]])
                    p_tracks['age'][-1].append(AGE_LABEL[line[15]])
                    ego_motion = ego_v_info[line[0].replace('.png', '')]
                    p_tracks['ego_motion'][-1].append(list(map(float, ego_motion)))
        
        num_tracks = len(p_tracks['clip_id'])
        for k in p_tracks.keys():
            assert len(p_tracks[k]) == num_tracks, (k, len(p_tracks[k]), num_tracks)

        return p_tracks, num_tracks

    def get_v_tracks(self, annos):  # TBD
        v_tracks = {'clip_id': [],
                    'img_nm': [],
                    'img_nm_int': [],
                    'obj_type': [],
                    'obj_id': [],
                    'bbox': [],
                    'motion_status': [],
                    'trunk_open': [],
                    'doors_open': [],
                    'ego_motion': []}
        for cid in self.ids.keys():
            # load ego motion
            ego_v_path = os.path.join(self.ori_data_root, 'clip_'+cid, 'synced_sensors.csv')
            ego_v_info = self.read_ego_csv(ego_v_path)  # dict {'img_nm': [info]}
            clip_annos = annos[cid]
            for _, vid in self.ids[cid]['vid']:
                # init new track
                for k in v_tracks.keys():
                    v_tracks[k].append([])
                # filter lines
                lines = clip_annos[(clip_annos[:, 2] == vid) & (clip_annos[:, 1] != 'person')]
                for line in lines:
                    v_tracks['clip_id'][-1].append(cid)
                    v_tracks['obj_id'][-1].append(str(int(float(vid))))
                    v_tracks['img_nm'][-1].append(line[0])
                    v_tracks['img_nm_int'][-1].append(int(line[0].replace('.png', '')))
                    v_tracks['obj_type'][-1].append(line[1])
                    tlhw = list(map(float, line[3: 7]))
                    ltrb = [tlhw[1], tlhw[0], tlhw[1]+tlhw[3], tlhw[0]+tlhw[2]]
                    v_tracks['bbox'][-1].append(ltrb)
                    v_tracks['motion_status'][-1].append(MOTOIN_STATUS_LABEL[line[8]])
                    v_tracks['trunk_open'][-1].append(line[7])
                    v_tracks['doors_open'][-1].append(line[9])
                    ego_motion = ego_v_info[line[0].replace('.png', '')]
                    v_tracks['ego_motion'][-1].append(list(map(float, ego_motion)))

        num_tracks = len(v_tracks['clip_id'])
        for k in v_tracks.keys():
            assert len(v_tracks[k]) == num_tracks, (k, len(v_tracks[k]), num_tracks)

        return v_tracks, num_tracks

    def track2sample(self, tracks):
        seq_len = self._obs_len + self._pred_len
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
                if self.tte is not None:
                    start_idx = len(track) - seq_len - self.tte[1]
                    end_idx = len(track) - seq_len - self.tte[0]
                    _samples.extend(
                        [track[i:i + seq_len] for i in range(start_idx, 
                                                             end_idx + 1, 
                                                             overlap_s)])
                else:
                    _samples.extend(
                        [track[i: i+seq_len] for i in range(0, 
                                                             len(track) - seq_len + 1, 
                                                             overlap_s)])
            samples[k] = _samples

        #  Normalize tracks by subtracting bbox/center at first time step from the rest
        print('---------------Normalize traj---------------')
        bbox_normed = copy.deepcopy(samples['bbox'])
        if self.norm_traj:
            for i in range(len(bbox_normed)):
                bbox_normed[i] = np.subtract(bbox_normed[i][:], bbox_normed[i][0]).tolist()
        samples['bbox_normed'] = bbox_normed

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

    def get_imgnm_to_objid(self, p_tracks, v_tracks, save_path):
        # cid_to_imgnm_to_oid_to_info: cid -> img name -> obj type (ped/veh) -> obj id -> bbox/ego motion
        imgnm_to_oid_to_info = {}
        n_p_tracks = len(p_tracks['bbox'])
        print('Saving imgnm to objid to obj info of pedestrians in TITAN')
        for i in range(n_p_tracks):
            cid = p_tracks['clip_id'][i][0]
            oid = p_tracks['obj_id'][i][0]
            if cid not in imgnm_to_oid_to_info:
                imgnm_to_oid_to_info[cid] = {}
            for j in range(len(p_tracks['img_nm'][i])):
                imgnm = p_tracks['img_nm'][i][j]
                # initialize the dict of the img
                if imgnm not in imgnm_to_oid_to_info[cid]:
                    imgnm_to_oid_to_info[cid][imgnm] = {}
                    imgnm_to_oid_to_info[cid][imgnm]['ped'] = {}
                    imgnm_to_oid_to_info[cid][imgnm]['veh'] = {}
                # initialize the dict of the obj
                bbox = p_tracks['bbox'][i][j]
                ego_motion = p_tracks['ego_motion'][i][j]
                imgnm_to_oid_to_info[cid][imgnm]['ped'][oid] = {}
                imgnm_to_oid_to_info[cid][imgnm]['ped'][oid]['bbox'] = bbox
        print('Saving imgnm to objid to obj info of vehicles in TITAN')
        n_v_tracks = len(v_tracks['bbox'])
        for i in range(n_v_tracks):
            cid = v_tracks['clip_id'][i][0]
            oid = v_tracks['obj_id'][i][0]
            if cid not in imgnm_to_oid_to_info:
                imgnm_to_oid_to_info[cid] = {}
            for j in range(len(v_tracks['img_nm'][i])):
                imgnm = v_tracks['img_nm'][i][j]
                # initialize the dict of the img
                if imgnm not in imgnm_to_oid_to_info[cid]:
                    imgnm_to_oid_to_info[cid][imgnm] = {}
                    imgnm_to_oid_to_info[cid][imgnm]['ped'] = {}
                    imgnm_to_oid_to_info[cid][imgnm]['veh'] = {}
                # initialize the dict of the obj
                bbox = v_tracks['bbox'][i][j]
                imgnm_to_oid_to_info[cid][imgnm]['veh'][oid] = {}
                imgnm_to_oid_to_info[cid][imgnm]['veh'][oid]['bbox'] = bbox
        
        with open(save_path, 'wb') as f:
            pickle.dump(imgnm_to_oid_to_info, f)
        
        return imgnm_to_oid_to_info

    def _get_neighbors(self, sample):
        
        pass

    def filter_short_tracks(self, tracks, min_len):
        '''
        tracks: dict
        '''
        idx = []
        _tracks = copy.deepcopy(tracks)
        n_tracks = len(_tracks['img_nm'])
        for i in range(n_tracks):
            if len(_tracks['img_nm'][i]) < min_len:
                idx.append(i)
        # print('short track to remove',len(idx), idx)
        # for i in idx:
        #     print(_tracks['clip_id'][i], _tracks['obj_id'][i])
        for i in reversed(idx):
            for k in _tracks.keys():
                _tracks[k].pop(i)
        
        return _tracks, len(_tracks['img_nm'])

    def multi2binary(self, labels, idxs):
        '''
        labels: list (n_samples, seq_len)
        idxs: list (int,...)
        '''
        bi_labels = []
        for s in labels:
            bi_labels.append([])
            for t in s:
                if t in idxs:
                    bi_labels[-1].append(1)
                else:
                    bi_labels[-1].append(0)
        return bi_labels

    def get_neighbors(self):  # TBD
        pass
    
    def str2ndarray(self, anno_list):
        return np.array(anno_list)

    def read_obj_csv(self, csv_path):
        res = []
        with open (csv_path, 'r') as f:
            reader = csv.reader(f)
            for item in reader:
                if reader.line_num == 1:
                    continue
                res.append(item)
        
        return res
    
    def read_ego_csv(self, csv_path):
        res = {}
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                img_nm = line[1].split('/')[-1].replace('.png', '')
                res[img_nm] = [line[3], line[5]]
            
        return res

    def downsample_seq(self):
        for k in self.samples['obs']:
            if len(self.samples['obs'][k][0]) == self._obs_len:
                new_k = []
                for s in range(len(self.samples['obs'][k])):
                    ori_seq = self.samples['obs'][k][s]
                    new_seq = []
                    for i in range(0, self._obs_len, self.obs_interval+1):
                        new_seq.append(ori_seq[i])
                    new_k.append(new_seq)
                    assert len(new_k[s]) == self.obs_len, (k, len(new_k), self.obs_len)
                new_k = np.array(new_k)
                self.samples['obs'][k] = new_k
        for k in self.samples['pred']:
            if len(self.samples['pred'][k][0]) == self._pred_len:
                new_k = []
                for s in range(len(self.samples['pred'][k])):
                    ori_seq = self.samples['pred'][k][s]
                    new_seq = []
                    for i in range(0, self._pred_len, self.obs_interval+1):
                        new_seq.append(ori_seq[i])
                    new_k.append(new_seq)
                    assert len(new_k[s]) == self.pred_len, (k, len(new_k), self.pred_len)
                new_k = np.array(new_k)
                self.samples['pred'][k] = new_k

def check_labels():
    not_matched = set()
    ori_data_root = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset'
    for d in os.listdir(ori_data_root):
        if 'clip_' in d and '.csv' in d:
            csv_path = os.path.join(ori_data_root, d)
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for line in reader:
                    if reader.line_num == 1:
                        continue
                    if line[8] not in MOTOIN_STATUS_LABEL.keys():
                        not_matched.add(line[8])
                    if line[10] not in COMMUNICATIVE_LABEL.keys():
                        not_matched.add(line[10])
                    if line[11] not in COMPLEX_CONTEXTUAL_LABEL.keys():
                        not_matched.add(line[11])
                    if line[12] not in ATOM_ACTION_LABEL.keys():
                        not_matched.add(line[12])
                    if line[13] not in SIMPLE_CONTEXTUAL_LABEL.keys():
                        not_matched.add(line[13])
                    if line[14] not in TRANSPORTIVE_LABEL.keys():
                        not_matched.add(line[14])
                    if line[15] not in AGE_LABEL.keys():
                        not_matched.add(line[15])
            print(d, ' done')
    print(not_matched)

def crop_imgs(tracks, resize_mode='even_padded', target_size=(224, 224), obj_type='p'):
    crop_root = '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/cropped_images'
    makedir(crop_root)
    data_root = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset'
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

def save_context_imgs(tracks, mode='local', target_size=(224, 224), obj_type='p'):
    ori_H, ori_W = 1520, 2704
    crop_root = '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/context'
    makedir(crop_root)
    data_root = '/home/y_feng/workspace6/datasets/TITAN/honda_titan_dataset/dataset'
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
                img_path = os.path.join(data_root, 'images_anonymized', 'clip_'+str(cid), 'images', img_nm)
                tgt_path = os.path.join(cur_obj_path, img_nm)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='crop')

    parser.add_argument('--h', type=int, default=224)
    parser.add_argument('--w', type=int, default=224)
    # crop args
    parser.add_argument('--resize_mode', type=str, default='even_padded')
    parser.add_argument('--subset', type=str, default='train')
    # context args
    parser.add_argument('--ctx_mode', type=str, default='local')
    args = parser.parse_args()

    if args.action == 'crop':
        # all_set = TITAN_dataset(sub_set='all')
        # crop_imgs(all_set.p_tracks, resize_mode=args.resize_mode, target_size=(args.w, args.h), obj_type='p')
        if args.subset == 'train':
            dataset = TITAN_dataset(sub_set='default_train', )
            crop_imgs(dataset.p_tracks, resize_mode=args.resize_mode, target_size=(args.w, args.h), obj_type='p')
        elif args.subset == 'val':
            dataset = TITAN_dataset(sub_set='default_val', )
            crop_imgs(dataset.p_tracks, resize_mode=args.resize_mode, target_size=(args.w, args.h), obj_type='p')
        elif args.subset == 'test':
            dataset = TITAN_dataset(sub_set='default_test', )
            crop_imgs(dataset.p_tracks, resize_mode=args.resize_mode, target_size=(args.w, args.h), obj_type='p')
        
        # crop_imgs(val_set.p_tracks, resize_mode=args.resize_mode, target_size=(args.w, args.h), obj_type='p')
        # crop_imgs(test_set.p_tracks, resize_mode=args.resize_mode, target_size=(args.w, args.h), obj_type='p')
        # crop_imgs(train_set.v_tracks, resize_mode=args.resize_mode, target_size=(args.w, args.h), obj_type='v')
        # crop_imgs(val_set.v_tracks, resize_mode=args.resize_mode, target_size=(args.w, args.h), obj_type='v')
        # crop_imgs(test_set.v_tracks, resize_mode=args.resize_mode, target_size=(args.w, args.h), obj_type='v')
    elif args.action == 'context':
        train_set = TITAN_dataset(sub_set='default_train')
        val_set = TITAN_dataset(sub_set='default_val')
        test_set = TITAN_dataset(sub_set='default_test')
        save_context_imgs(train_set.p_tracks, mode=args.ctx_mode, target_size=(args.w, args.h), obj_type='p')
        save_context_imgs(val_set.p_tracks, mode=args.ctx_mode, target_size=(args.w, args.h), obj_type='p')
        save_context_imgs(test_set.p_tracks, mode=args.ctx_mode, target_size=(args.w, args.h), obj_type='p')
        save_context_imgs(train_set.v_tracks, mode=args.ctx_mode, target_size=(args.w, args.h), obj_type='v')
        save_context_imgs(val_set.v_tracks, mode=args.ctx_mode, target_size=(args.w, args.h), obj_type='v')
        save_context_imgs(test_set.v_tracks, mode=args.ctx_mode, target_size=(args.w, args.h), obj_type='v')
    pass