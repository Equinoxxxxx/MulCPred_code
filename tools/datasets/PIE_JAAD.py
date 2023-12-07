from hashlib import new
import pickle
from re import T
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import functional as TVF
import numpy as np
import cv2
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
import time
import copy
import tqdm
import os
from tqdm import tqdm
import pdb

from .pie_data import PIE
from .jaad_data import JAAD
from ..utils import mapping_20, makedir, ltrb2xywh, coord2pseudo_heatmap, cls_weights
from ..transforms import RandomHorizontalFlip, RandomResizedCrop, crop_local_ctx
from ..data._img_mean_std import img_mean_std
from ..general import HiddenPrints


class PIEDataset(Dataset):
    def __init__(self, 
                 dataset_name='PIE', 
                 normalize_pos=True, 
                 img_norm_mode='torch', 
                 data_split_type='default',
                 seq_type='crossing', 
                 obs_len=16, pred_len=1, overlap_retio=0.5,
                 do_balance=True, balance_label='target', 
                 subset='train', 
                 bbox_size=(224, 224), ctx_size=(224, 224), 
                 color_order='BGR', crop_from_ori=False, load_all_img=False, 
                 resize_mode='padded', min_wh=None, max_occ=2, 
                 use_skeleton=0, skeleton_mode='coord', 
                 use_img=0, 
                 use_context=0, ctx_mode='mask_ped', 
                 seg_cls_idx=[24, 26, 19, 20],
                 use_traj=0, traj_mode='ltrb', 
                 use_ego=0,
                 use_single_img=False,
                 pred_img=0,
                 pred_context=0, pred_context_mode='ori',
                 small_set=0, seg_class_set=1,
                 tte=[30, 60],
                 recog_act=0,
                 obs_interval=0.,
                 augment_mode='none',
                 seg_cls=['person', 'vehicles', 'roads', 'traffic_lights'],
                 ego_accel=True,
                 speed_unit='m/s',
                 ):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name
        self.seq_type = seq_type
        self.tte = tte
        self.recog_act = recog_act
        self.subset = subset
        self.normalize_pos = normalize_pos
        self.img_norm_mode = img_norm_mode
        self.color_order = color_order
        self.img_mean, self.img_std = img_mean_std(self.img_norm_mode)  # BGR
        # self.img_mean = [0.406, 0.456, 0.485]  # BGR
        # self.img_std = [0.225, 0.224, 0.229]
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.overlap_ratio = overlap_retio
        self.bbox_size = bbox_size
        self.obs_interval = obs_interval

        # interval
        self._obs_len = self.obs_len * (self.obs_interval + 1)
        self._pred_len = self.pred_len * (self.obs_interval + 1)

        self.crop_from_ori = crop_from_ori
        self.load_all_img = load_all_img
        self.resize_mode = resize_mode
        self.use_skeleton = use_skeleton
        self.skeleton_mode = skeleton_mode
        self.use_img = use_img
        self.use_context = use_context
        self.ctx_mode = ctx_mode
        self.ctx_size = ctx_size
        self._seg_class_idx = seg_cls_idx # person, car, light, sign
        self.seg_class_idx = [11, 13, 6, 7]  # person, car, light, sign
        self.seg_class_idx = []
        for i in self._seg_class_idx:
            self.seg_class_idx.append(mapping_20[i] - 1)
        self.use_single_img = use_single_img
        self.pred_img = pred_img
        self.pred_context = pred_context
        self.pred_context_mode = pred_context_mode
        self.use_traj = use_traj
        self.traj_mode = traj_mode
        self.use_ego = use_ego
        self.augment_mode = augment_mode
        self.seg_cls = seg_cls
        self.ego_accel = ego_accel
        self.ego_motion_key = 'ego_accel' if self.ego_accel else 'obd_speed'
        if self.dataset_name == 'JAAD':
            self.ego_motion_key = 'vehicle_act'
        self.speed_unit = speed_unit
        self.transforms = {'random': 0,
                            'hflip': None,
                            'resized_crop': {'img': None,
                                            'ctx': None,
                                            'sk': None}}

        # data opts
        if dataset_name == 'PIE':
            self.root_path = '/home/y_feng/workspace6/datasets/PIE_dataset'
            self.sk_vis_path = '/home/y_feng/workspace6/datasets/PIE_dataset/sk_vis/even_padded/288w_by_384h/'
            self.sk_coord_path = '/home/y_feng/workspace6/datasets/PIE_dataset/sk_coords/even_padded/288w_by_384h/'
            self.sk_heatmap_path = '/home/y_feng/workspace6/datasets/PIE_dataset/sk_heatmaps/even_padded/288w_by_384h/'
            self.sk_p_heatmap_path = '/home/y_feng/workspace6/datasets/PIE_dataset/sk_pseudo_heatmaps/even_padded/48w_by_48h/'
            self.sam_seg_root = '/home/y_feng/workspace6/datasets/PIE_dataset/seg_sam'
            self.data_opts = {'normalize_bbox': normalize_pos,
                         'fstride': 1,
                         'sample_type': 'all',
                         'height_rng': [0, float('inf')],
                         'squarify_ratio': 0,
                         # kfold, random, default. default: set03 for test
                         'data_split_type': data_split_type,  
                         # crossing , intention
                         'seq_type': seq_type,  
                         'min_track_size': self._obs_len + self._pred_len,  # discard tracks that are shorter
                         'max_size_observe': self._obs_len,  # number of observation frames
                         'max_size_predict': self._pred_len,  # number of prediction frames
                         'seq_overlap_rate': overlap_retio,  # how much consecutive sequences overlap
                         'balance': do_balance,  # balance the training and testing samples
                         'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                         'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                         'seq_type': seq_type,
                         'encoder_input_type': ['bbox', 'obd_speed'],
                         'decoder_input_type': [],
                         'output_type': ['intention_binary', 'bbox']
                         }
            with HiddenPrints():
                data_base = PIE(data_path=self.root_path)
        else:
            self.root_path = '/home/y_feng/workspace6/datasets/JAAD'
            self.sk_vis_path = '/home/y_feng/workspace6/datasets/JAAD/sk_vis/even_padded/288w_by_384h/'
            self.sk_coord_path = '/home/y_feng/workspace6/datasets/JAAD/sk_coords/even_padded/288w_by_384h/'
            self.sk_heatmap_path = '/home/y_feng/workspace6/datasets/JAAD/sk_heatmaps/even_padded/288w_by_384h/'
            self.sk_p_heatmap_path = '/home/y_feng/workspace6/datasets/JAAD/sk_pseudo_heatmaps/even_padded/48w_by_48h/'
            self.sam_seg_root = '/home/y_feng/workspace6/datasets/JAAD/seg_sam'
            self.data_opts = {'fstride': 1,
                            'sample_type': 'all',  
                            'subset': 'default',
                            'data_split_type': data_split_type,
                            'seq_type': seq_type,
                            # 'min_track_size': self._obs_len + self._pred_len,  # discard tracks that are shorter
                            'min_track_size': 1,
                            'height_rng': [0, float('inf')],
                            'squarify_ratio': 0,
                            'random_params': {'ratios': None,
                                            'val_data': True,
                                            'regen_data': True},
                            'kfold_params': {'num_folds': 5, 'fold': 1}}
            with HiddenPrints():
                data_base = JAAD(data_path=self.root_path)
        self.veh_info_path = os.path.join(self.root_path, 'veh_tracks.pkl')
        self.imgnm_to_objid_path = os.path.join(self.root_path, 
                                                'imgnm_to_objid_to_ann.pkl')
        
        if self.tte is not None:

            self.data_opts['min_track_size'] += self.tte[1]

        self.cropped_path = os.path.join(self.root_path, 
                                         'cropped_images', 
                                         resize_mode, 
                                         str(bbox_size[0])+'w_by_'\
                                            +str(bbox_size[1])+'h')
        
        # get pedestrian tracks
        # all data: {'image', 'ped_id', 'bbox', 'center', 'occlusion', 'vehicle_act', 'intention_binary', 'activities', 'image_dimension'}
        with HiddenPrints():
            self.p_tracks = \
                data_base.generate_data_trajectory_sequence(image_set=self.subset, 
                                                            **self.data_opts)
        # remove the image dimension key
        self.p_tracks.pop('image_dimension', None)
        # remove 720x1280 videos
        if self.dataset_name == 'JAAD':
            self.p_tracks = self.rm_720x1280(self.p_tracks)
        
        # convert speed unit
        if self.speed_unit == 'm/s':
            if self.dataset_name == 'JAAD' and self.use_ego:
                raise ValueError('Cannot turn ego motion in JAAD into m/s.')
            elif self.dataset_name == 'PIE':
                self.p_tracks['obd_speed'] = \
                    self._convert_speed_unit(self.p_tracks['obd_speed'])
        # calc acceleration
        if self.dataset_name == 'PIE':
            if self.ego_accel:
                self.p_tracks = self._calc_accel(self.p_tracks)

        # get vehicle tracks
        with open(self.veh_info_path, 'rb') as f:
            self.v_tracks = pickle.load(f)
        
        # get img name to obj id dict
        if self.dataset_name == 'PIE':
            if not os.path.exists(self.imgnm_to_objid_path):
                self.imgnm_to_objid = \
                    self.get_imgnm_to_objid(self.p_tracks, 
                                            self.v_tracks, 
                                            self.imgnm_to_objid_path)
            else:
                with open(self.imgnm_to_objid_path, 'rb') as f:
                    self.imgnm_to_objid = pickle.load(f)

        pids = [track[0][0] for track in self.p_tracks['ped_id']]

        self.samples = self.split_to_samples(self.p_tracks)
        print('-----------Convert samples to ndarray----------')
        for k in tqdm(self.samples.keys()):
            self.samples[k] = np.array(self.samples[k])
        self.num_samples = len(self.samples['obs_image_paths'])
        # remove small bbox
        if min_wh is not None:
            self.samples = self.rm_small_bb(self.samples, min_wh)
        # remove occlusion
        if max_occ < 2:
            self.samples = self.rm_occluded(self.samples, max_occ)

        # apply interval
        if self.obs_interval > 0:
            self.downsample_seq()
            print('Applied interval')
            print('cur input len', len(self.samples['obs_img_nm_int'][0]))

        # small set
        assert small_set <= 1, small_set
        if small_set > 0:
            small_set_size = int(self.num_samples * small_set)
            for k in self.samples.keys():
                self.samples[k] = self.samples[k][:small_set_size]
            self.num_samples = small_set_size
        # balance
        if do_balance:
            self.samples = self.balance(balance_label='target', 
                                        all_samples=self.samples)
            print('num samples before balance:', self.num_samples)
            self.num_samples = len(self.samples['obs_image_paths'])
            print('num samples after balance:', self.num_samples)
        
        # add seg maps in samples
        if self.use_context:
            self.ctx_path = os.path.join(self.root_path, 
                                         'context', 
                                         self.ctx_mode, 
                                         str(ctx_size[0])+'w_by_'\
                                            +str(ctx_size[1])+'h')
            if self.ctx_mode == 'seg_multi' \
                or self.ctx_mode == 'seg_single' \
                    or self.ctx_mode == 'local_seg_multi':
                self.ctx_path = os.path.join(self.root_path, 'seg')
                seg_paths = []
                for s in range(self.num_samples):
                    img_path_seq = self.samples['obs_image_paths'][s]
                    seg_path_seq = []
                    for img_path in img_path_seq:
                        seg_path = img_path.replace('images', 'seg')\
                            .replace('.png', '.pkl')
                        seg_path_seq.append(seg_path)
                    seg_paths.append(seg_path_seq)
                self.samples['obs_seg_paths'] = seg_paths

        if self.pred_context:
            self.pred_context_path = \
                os.path.join(self.root_path, 
                             'context', 
                             self.pred_context_mode, 
                             str(ctx_size[0])+'w_by_'+str(ctx_size[1])+'h')
        
        # augmentation
        self.samples = self._add_augment(self.samples)

        print('Total samples: ', self.num_samples)
        # check num samles
        for k in self.samples:
            # print(k)
            assert len(self.samples[k]) == self.num_samples, \
                (k, len(self.samples[k]), self.num_samples)

        # class count
        self.n_c = sum(np.squeeze(self.samples['target']))
        self.n_nc = self.num_samples - self.n_c
        self.num_samples_cls = [self.n_nc, self.n_c]
        self.class_weights = {}
        self.class_weights['cross'] = cls_weights(self.num_samples_cls, 'sklearn')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # print('-----------getting item-----------')
        # import pdb; pdb.set_trace()
        obs_bboxes = torch.tensor(self.samples['obs_bbox_normed'][idx]).float()  # ltrb
        obs_bboxes_unnormed = torch.tensor(self.samples['obs_bbox'][idx]).float()  # ltrb
        pred_bboxes = torch.tensor(self.samples['pred_bbox_normed'][idx]).float()
        target = torch.tensor(self.samples['target'][idx][0])
        obs_ego = \
            torch.tensor(self.samples['obs_ego'][idx]).float().reshape(-1, 1)
        if obs_ego.size(-1) == 1:
            obs_ego = torch.concat([obs_ego, torch.zeros(obs_ego.size())], 
                                   dim=-1)
        # obs_ego = torch.cat([obs_ego, torch.zeros(obs_ego.size())], dim=-1)

        # squeeze the coords
        if '0-1' in self.traj_mode:
            obs_bboxes[:, 0] /= 1920
            obs_bboxes[:, 2] /= 1920
            obs_bboxes[:, 1] /= 1080
            obs_bboxes[:, 3] /= 1080

        sample = {'ped_id_int': self.samples['obs_ped_id_int'][idx],
                'img_nm_int': self.samples['obs_img_nm_int'][idx],
                'vid_id_int': self.samples['obs_vid_id_int'][idx],
                'obs_bboxes':obs_bboxes, # obslen, 4
                'obs_bboxes_unnormed': obs_bboxes_unnormed,
                'pred_intent': target,   # int
                'pred_bboxes': pred_bboxes,   # shape: [pred_len, 4]
                'obs_ego': obs_ego,  # shape: obs len, 1
                'hflip_flag': torch.tensor(0),
                'img_ijhw': torch.tensor([-1, -1, -1, -1]),
                'ctx_ijhw': torch.tensor([-1, -1, -1, -1]),
                'sk_ijhw': torch.tensor([-1, -1, -1, -1]),
                }
        if self.dataset_name == 'PIE':
            sample['set_id_int'] = self.samples['obs_set_id_int'][idx]

        if self.use_skeleton:
            if self.skeleton_mode == 'coord':
                pid = self.samples['obs_pid'][idx][0][0]
                coords = []
                ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                for path in ori_obs_img_paths:
                    img_nm = path.split('/')[-1]
                    coord_nm = img_nm.replace('.png', '.pkl')
                    coord_path = os.path.join(self.sk_coord_path, pid, coord_nm)
                    with open(coord_path, 'rb') as f:
                        coord = pickle.load(f)  # 17, 3
                    coords.append(coord[:, :2])  # 17, 2
                coords = np.stack(coords, axis=0)  # T 17, 2
                # (T, N, C) -> (C, T, N)
                try:
                    obs_skeletons = torch.from_numpy(coords).float().permute(2, 0, 1)  # 2, T, 17
                except:
                    print('coords shape',coords.shape)
                    import pdb;pdb.set_trace()
                    raise NotImplementedError()
            elif self.skeleton_mode == 'heatmap':
                pid = self.samples['obs_pid'][idx][0][0]
                heatmaps = []
                ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                for path in ori_obs_img_paths:
                    img_nm = path.split('/')[-1]
                    heatmap_nm = img_nm.replace('.png', '.pkl')
                    heatmap_path = os.path.join(self.sk_heatmap_path, pid, heatmap_nm)
                    with open(heatmap_path, 'rb') as f:
                        heatmap = pickle.load(f)
                    heatmaps.append(heatmap)
                heatmaps = np.stack(heatmaps, axis=0)  # T C H W
                # T C H W -> C T H W
                obs_skeletons = torch.from_numpy(heatmaps).float().permute(1, 0, 2, 3)  # shape: (17, seq_len, 96, 72)
            elif self.skeleton_mode == 'pseudo_heatmap':
                pid = self.samples['obs_pid'][idx][0][0]
                heatmaps = []
                ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                for path in ori_obs_img_paths:
                    img_nm = path.split('/')[-1]
                    heatmap_nm = img_nm.replace('.png', '.pkl')
                    heatmap_path = os.path.join(self.sk_p_heatmap_path, pid, heatmap_nm)
                    with open(heatmap_path, 'rb') as f:
                        heatmap = pickle.load(f)
                    heatmaps.append(heatmap)
                heatmaps = np.stack(heatmaps, axis=0)  # T C H W
                # T C H W -> C T H W
                obs_skeletons = torch.from_numpy(heatmaps).float().permute(1, 0, 2, 3)  # shape: (17, seq_len, 48, 48)
            elif self.skeleton_mode == 'img+heatmap':
                if not self.crop_from_ori:
                    pid = self.samples['obs_pid'][idx][0][0]
                    imgs = []
                    ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                    for path in ori_obs_img_paths:
                        img_nm = path.split('/')[-1]
                        img_path = os.path.join(self.cropped_path, pid, img_nm)
                        imgs.append(cv2.imread(img_path))
                else:
                    obs_img_paths = self.samples['obs_image_paths'][idx]
                    obs_bboxes = self.samples['obs_bbox'][idx]
                    imgs = []
                    for path, bbox in zip(obs_img_paths, obs_bboxes):
                        imgs.append(self.load_cropped_image(path, bbox))
                # (T, H, W, C) -> (C, T, H, W)
                imgs = np.stack(imgs, axis=0)
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
                    ped_imgs = torch.from_numpy(np.ascontiguousarray(ped_imgs.numpy()[::-1, :, :, :]))  # 3 T H W
                # load heatmaps
                pid = self.samples['obs_pid'][idx][0][0]
                heatmaps = []
                ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                for path in ori_obs_img_paths:
                    img_nm = path.split('/')[-1]
                    heatmap_nm = img_nm.replace('.png', '.pkl')
                    heatmap_path = os.path.join(self.sk_heatmap_path, pid, heatmap_nm)
                    with open(heatmap_path, 'rb') as f:
                        heatmap = pickle.load(f)
                    heatmaps.append(heatmap)
                heatmaps = np.stack(heatmaps, axis=0)  # T C H W
                # T C H W -> C T H W
                heatmaps = torch.from_numpy(heatmaps).float().permute(1, 0, 2, 3)  # shape: (17, seq_len, 96, 72)
                heatmaps = torch.sum(heatmaps, dim=0, keepdim=True)  # shape: (1, seq_len, 96, 72)
                # normalize heatmaps
                heatmaps = heatmaps - torch.amin(heatmaps, dim=(2, 3), keepdim=True)
                heatmaps = heatmaps / torch.amax(heatmaps, dim=(2, 3), keepdim=True)
                # interpolate
                heatmaps = F.interpolate(heatmaps, size=(ped_imgs.size(2),
                                                         ped_imgs.size(3)), mode='bilinear', align_corners=True)  # 1 T H W
                obs_skeletons = heatmaps * ped_imgs

            else:
                raise NotImplementedError(self.skeleton_mode)
            sample['obs_skeletons'] = obs_skeletons
        
        if self.use_img or self.use_single_img:
            # print('-----------getting img-----------')
            if not self.crop_from_ori:
                pid = self.samples['obs_pid'][idx][0][0]
                imgs = []
                ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                for path in ori_obs_img_paths:
                    img_nm = path.split('/')[-1]
                    img_path = os.path.join(self.cropped_path, pid, img_nm)
                    imgs.append(cv2.imread(img_path))
            else:
                obs_img_paths = self.samples['obs_image_paths'][idx]
                obs_bboxes = self.samples['obs_bbox'][idx]
                imgs = []
                for path, bbox in zip(obs_img_paths, obs_bboxes):
                    imgs.append(self.load_cropped_image(path, bbox))
            # (T, H, W, C) -> (C, T, H, W)
            imgs = np.stack(imgs, axis=0)
            try:
                ped_imgs = torch.from_numpy(imgs).float().permute(3, 0, 1, 2)
            except:
                import pdb;pdb.set_trace()

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
                ped_imgs = torch.from_numpy(np.ascontiguousarray(ped_imgs.numpy()[::-1, :, :, :]))
            if self.use_img:
                sample['ped_imgs'] = ped_imgs  # shape [3, obs_len, H, W]
            if self.use_single_img:
                sample['ped_single_imgs'] = ped_imgs[:, -1]
        
        if self.use_context:
            # print('-----------getting ctx-----------')
            if self.ctx_mode == 'mask_ped' or self.ctx_mode == 'local' or self.ctx_mode == 'ori_local' or self.ctx_mode == 'ori':
                pid = self.samples['obs_pid'][idx][0][0]
                ctx_imgs = []
                ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                for path in ori_obs_img_paths:
                    img_nm = path.split('/')[-1]
                    img_path = os.path.join(self.ctx_path, pid, img_nm)
                    ctx_imgs.append(cv2.imread(img_path))
                ctx_imgs = np.stack(ctx_imgs, axis=0)
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
                # BGR -> RGB
                if self.color_order == 'RGB':
                    ctx_imgs = torch.from_numpy(np.ascontiguousarray(ctx_imgs.numpy()[::-1, :, :, :]))
                sample['obs_context'] = ctx_imgs  # shape [3, obs_len, H, W]
            elif self.ctx_mode == 'seg_ori_local' or self.ctx_mode == 'seg_local':
                # load imgs
                ori_obs_img_paths = self.samples['obs_image_paths'][idx]
                ctx_imgs = []
                for path in ori_obs_img_paths:
                    ctx_imgs.append(cv2.imread(path))
                assert ctx_imgs[0].shape == [1080, 1920, 3], ctx_imgs.shape
                ctx_imgs = np.stack(ctx_imgs, axis=0)  # THWC
                # permute
                ctx_imgs = torch.from_numpy(ctx_imgs).float().permute(3, 0, 1, 2)  # CTHW
                # norm imgs
                if self.img_norm_mode != 'ori':
                    ctx_imgs /= 255.
                    ctx_imgs[0, :, :, :] -= self.img_mean[0]
                    ctx_imgs[1, :, :, :] -= self.img_mean[1]
                    ctx_imgs[2, :, :, :] -= self.img_mean[2]
                    ctx_imgs[0, :, :, :] /= self.img_std[0]
                    ctx_imgs[1, :, :, :] /= self.img_std[1]
                    ctx_imgs[2, :, :, :] /= self.img_std[2]
                # BGR -> RGB
                if self.color_order == 'RGB':
                    ctx_imgs = torch.from_numpy(np.ascontiguousarray(ctx_imgs.numpy()[::-1, :, :, :]))  # CTHW
                # load seg maps
                ctx_segs = {c:[] for c in self.seg_cls}
                if self.dataset_name == 'PIE':
                    for pth in ori_obs_img_paths:
                        s_nm, v_nm, i_nm = pth.split('/')[-3:]
                        s_id, v_id, f_nm = str(int(s_nm.replace('set0', ''))), str(int(v_nm.replace('video_', ''))), i_nm.replace('png', 'pkl')
                        for c in self.seg_cls:
                            seg_path = os.path.join(self.sam_seg_root, c, s_id, v_id, f_nm)
                            with open(seg_path, 'rb') as f:
                                seg = pickle.load(seg_path)
                            ctx_segs[c].append(torch.from_numpy(seg))
                elif self.dataset_name == 'JAAD':
                    for pth in ori_obs_img_paths:
                        v_nm, i_nm = pth.split('/')[-2:]
                        v_id, f_nm = str(int(v_nm.replace('video_', ''))), i_nm.replace('png', 'pkl')
                        for c in self.seg_cls:
                            seg_path = os.path.join(self.sam_seg_root, c, v_id, f_nm)
                            with open(seg_path, 'rb') as f:
                                seg = pickle.load(seg_path)
                            ctx_segs[c].append(torch.from_numpy(seg))
                for c in self.seg_cls:
                    ctx_segs[c] = torch.stack(ctx_segs[c], dim=0)  # THW
                # crop
                crop_imgs = []
                crop_segs = {c:[] for c in self.seg_cls}
                for i in range(ctx_imgs.size(1)):  # T
                    crop_img = crop_local_ctx(ctx_imgs[:, i], obs_bboxes_unnormed[i], self.ctx_size)  # 3 h w
                    crop_imgs.append(crop_img)
                    for c in self.seg_cls:
                        crop_seg = crop_local_ctx(torch.unsqueeze(ctx_segs[c][i], dim=0), obs_bboxes_unnormed[i], self.ctx_size)  # 1 h w
                        crop_segs[c].append(crop_seg)
                crop_imgs = torch.stack(crop_imgs, dim=1)  # 3Thw
                all_seg = []
                for c in self.seg_cls:
                    all_seg.append(torch.stack(crop_segs[c], dim=1))  # 1Thw
                all_seg = torch.stack(all_seg, dim=4)  # 1Thw n_cls
                sample['obs_context'] = all_seg * torch.unsqueeze(crop_imgs, dim=-1)  # 3Thw n_cls
                
            # elif self.ctx_mode == 'seg_multi' or self.ctx_mode == 'local_seg_multi':
            #     pid = self.samples['obs_pid'][idx][0][0]
            #     ctx_imgs = []
            #     ctx_segs = []
            #     ori_obs_img_paths = self.samples['obs_image_paths'][idx]
            #     seg_paths = self.samples['obs_seg_paths'][idx]
            #     for path in ori_obs_img_paths:
            #         ctx_imgs.append(cv2.imread(path))
            #     for path in seg_paths:
            #         with open(path, 'rb') as f:
            #             seg = pickle.load(f)
            #             ctx_segs.append(seg)
            #     ctx_imgs = np.stack(ctx_imgs, axis=0)  # THWC
            #     ctx_segs = np.stack(ctx_segs, axis=0)  # THW
            #     # permute
            #     ctx_imgs = torch.from_numpy(ctx_imgs).float().permute(3, 0, 1, 2)  # CTHW
            #     # normalize img
            #     if self.img_norm_mode != 'ori':
            #         ctx_imgs /= 255.
            #         ctx_imgs[0, :, :, :] -= self.img_mean[0]
            #         ctx_imgs[1, :, :, :] -= self.img_mean[1]
            #         ctx_imgs[2, :, :, :] -= self.img_mean[2]
            #         ctx_imgs[0, :, :, :] /= self.img_std[0]
            #         ctx_imgs[1, :, :, :] /= self.img_std[1]
            #         ctx_imgs[2, :, :, :] /= self.img_std[2]
            #     # BGR -> RGB
            #     if self.color_order == 'RGB':
            #         ctx_imgs = torch.from_numpy(np.ascontiguousarray(ctx_imgs.numpy()[::-1, :, :, :]))  # CTHW
            #     # sample['obs_context'] = ctx_imgs  # CTHW
            #     # sample['obs_seg'] = torch.from_numpy(ctx_segs).float()  # THW
            #     ctx_segs = torch.from_numpy(ctx_segs).float().view(ctx_segs.shape[0], ctx_segs.shape[1], ctx_segs.shape[2], 1).permute(3, 0, 1, 2) # 1THW
                
            #     if self.ctx_mode == 'seg_multi':
            #         ctx_segs = F.interpolate(ctx_segs, size=(self.ctx_size[1], self.ctx_size[0])) # ctx_size: (W, H)
            #         ctx_imgs = F.interpolate(ctx_imgs, size=(self.ctx_size[1], self.ctx_size[0])) # ctx_size: (W, H)
            #         ctx_imgs = ctx_imgs.view(ctx_imgs.size(0), ctx_imgs.size(1), ctx_imgs.size(2), ctx_imgs.size(3), 1)  # CTHW 1
            #         ctx_segs = F.one_hot(ctx_segs.long(), num_classes=20)  # CTHW 19
            #         ctx_segs = ctx_segs * ctx_imgs
            #         sample['obs_context'] = ctx_segs[:, :, :, :, self.seg_class_idx]  # 3 T H W num_seg
            #     else:
            #         ctx_segs = F.interpolate(ctx_segs, size=(1080, 1920)) # ctx_size: (W, H)  1THW
            #         ctx_imgs = F.interpolate(ctx_imgs, size=(1080, 1920)) # ctx_size: (W, H)  CTHW
            #         cropped_segs = []
            #         cropped_imgs = []
            #         if 'ltrb' in self.traj_mode:
            #             for i in range(obs_bboxes.size(0)):
            #                 l, t, r, b = list(map(int, obs_bboxes[i]))
            #                 # crop local context
            #                 x = (l+r) // 2
            #                 y = (t+b) // 2
            #                 h = b-t
            #                 w = r-l
            #                 crop_h = h*2
            #                 crop_w = h*2
            #                 crop_l = max(x-h, 0)
            #                 crop_r = min(x+h, 1920)
            #                 crop_t = max(y-h, 0)
            #                 crop_b = min(y+h, 1920)
            #                 cropped = ctx_imgs[:, i, crop_t:crop_b, crop_l:crop_r]  # Chw
            #                 cropped_s = ctx_segs[:, i, crop_t:crop_b, crop_l:crop_r]  # 1hw
            #                 l_pad = max(h-x, 0)
            #                 r_pad = max(x+h-1920, 0)
            #                 t_pad = max(h-y, 0)
            #                 b_pad = max(y+h-1080, 0)
            #                 cropped = F.pad(cropped, (l_pad, r_pad, t_pad, b_pad), 'constant', 0)
            #                 cropped_s = F.pad(cropped_s, (l_pad, r_pad, t_pad, b_pad), 'constant', 0)
            #                 assert cropped.size(1) == crop_h and cropped.size(2) == crop_w, (cropped.shape, (crop_h, crop_w))
            #                 cropped = F.interpolate(torch.unsqueeze(cropped, dim=0), size=self.ctx_size, mode='bilinear', align_corners=True)  # 1 3 h w
            #                 cropped_s = F.interpolate(torch.unsqueeze(cropped_s, dim=0), size=self.ctx_size)  # 1 1 h w
            #                 cropped_imgs.append(cropped)
            #                 cropped_segs.append(cropped_s)
            #             cropped_imgs = torch.squeeze(torch.stack(cropped_imgs, dim=2), dim=0)  # 3Thw
            #             cropped_segs = torch.squeeze(torch.stack(cropped_segs, dim=2), dim=0)  # 1Thw
            #             cropped_segs = F.one_hot(cropped_segs.long(), num_classes=20)  # 1THW 19
            #             # print(cropped_segs.size(), cropped_imgs.size())
            #             cropped_segs = cropped_segs * torch.unsqueeze(cropped_imgs, dim=-1)
            #             sample['obs_context'] = cropped_segs[:, :, :, :, self.seg_class_idx]  # 3 T H W num_seg

        if self.pred_img:
            pass
        
        if self.pred_context:
            if self.pred_context_mode == 'ori':
                pid = self.samples['obs_pid'][idx][0][0]
                ctx_imgs = []
                ori_obs_img_paths = self.samples['pred_image_paths'][idx]
                for path in ori_obs_img_paths:
                    img_nm = path.split('/')[-1]
                    img_path = os.path.join(self.ctx_path, pid, img_nm)
                    ctx_imgs.append(cv2.imread(img_path))
                ctx_imgs = np.stack(ctx_imgs, axis=0)
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
                # BGR -> RGB
                if self.color_order == 'RGB':
                    ctx_imgs = torch.from_numpy(np.ascontiguousarray(ctx_imgs.numpy()[::-1, :, :, :]))
                sample['pred_context'] = ctx_imgs  # shape [3, obs_len, H, W]

        # augmentation
        if self.augment_mode != 'none':
            if self.transforms['random']:
                sample = self._ranndom_augment(sample)
            elif self.transforms['balance']:
                sample['hflip_flag'] = torch.tensor(self.samples['hflip_flag'][idx])
                sample = self._augment(sample)

        return sample

    def _augment(self, sample):
        # flip
        if sample['hflip_flag']:
            if self.use_img:
                sample['ped_imgs'] = TVF.hflip(sample['ped_imgs'])
            if self.use_context:
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

    def _ranndom_augment(self, sample):
        # flip
        if self.transforms['hflip'] is not None:
            sample['hflip_flag'] = torch.tensor(self.transforms['hflip'].flag)
            if self.use_img:
                sample['ped_imgs'] = self.transforms['hflip'](sample['ped_imgs'])
            if self.use_context:
                sample['obs_context'] = self.transforms['hflip'](sample['obs_context'])
            if self.use_skeleton and ('heatmap' in self.skeleton_mode):
                sample['obs_skeletons'] = self.transforms['hflip'](sample['obs_skeletons'])
            if self.use_traj and self.transforms['hflip'].flag:
                sample['obs_bboxes_unnormed'][:, 0], sample['obs_bboxes_unnormed'][:, 2] = \
                    1920 - sample['obs_bboxes_unnormed'][:, 2], 1920 - sample['obs_bboxes_unnormed'][:, 0]
                if '0-1' in self.traj_mode:
                    sample['obs_bboxes'][:, 0], sample['obs_bboxes'][:, 2] =\
                         1 - sample['obs_bboxes'][:, 2], 1 - sample['obs_bboxes'][:, 0]
                else:
                    sample['obs_bboxes'][:, 0], sample['obs_bboxes'][:, 2] =\
                         1920 - sample['obs_bboxes'][:, 2], 1920 - sample['obs_bboxes'][:, 0]
            self.transforms['hflip'].randomize_parameters()
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
    
    def _convert_speed_unit(self, ego_speed_tracks):
        '''
        Convert speed unit from km/h to m/s
        '''
        new_tracks = []
        for track in ego_speed_tracks:
            new_tracks.append([])
            for speed in track:
                speed = speed[0]
                new_tracks[-1].append(speed / 3.6)
        return new_tracks
    
    def _calc_accel(self, p_tracks):
        '''
        Add acceleration to the keys of tracks.
        The lengths of all tracks will be substracted by 1.
        '''
        new_tracks = {'ego_accel':[]}
        for speed_track in p_tracks['obd_speed']:
            new_tracks['ego_accel'].append([])
            for i in range(len(speed_track) - 1):
                accel = (speed_track[i+1] - speed_track[i]) * 30  # 30 fps
                if self.speed_unit == 'km/h':
                    accel = accel / 3.6
                new_tracks['ego_accel'][-1].append(accel)

        # rest keys
        # pop the last frame for every track
        for k in p_tracks:
            new_tracks[k] = []
            for i in range(len(p_tracks[k])):
                new_tracks[k].append(p_tracks[k][i][:-1])
                assert len(new_tracks[k][i]) == len(new_tracks['ego_accel'][i]), \
                    (len(new_tracks[k][i]), len(new_tracks['ego_accel'][i]))
        return new_tracks

    def _get_neighbors(self, sample):
        '''
        Get the neighbors' info of the target pedestrian during 
        the observation length
        '''
        
        pass
    
    def get_imgnm_to_objid(self, p_tracks, v_tracks, save_path):
        # cid_to_imgnm_to_oid_to_info: cid -> img name -> obj type (ped/veh) 
        # -> obj id -> bbox/ego motion
        imgnm_to_oid_to_info = {}
        
        if self.dataset_name == 'PIE':
            # pedestrian tracks
            print(f'Saving imgnm to objid to obj info of pedestrians in \
                {self.dataset_name}')
            tracks = p_tracks
            n_tracks = len(tracks['image'])
            # track loop
            for i in range(n_tracks):
                img_paths = tracks['image'][i]
                ped_id = tracks['ped_id'][i][0][0]
                set_id, vid_id, oid = ped_id.split('_')
                # initialize set and video dict
                if set_id not in imgnm_to_oid_to_info:
                    imgnm_to_oid_to_info[set_id] = {}
                if vid_id not in imgnm_to_oid_to_info[set_id]:
                    imgnm_to_oid_to_info[set_id][vid_id] = {}
                # image loop
                for j in range(len(img_paths)):
                    img_path = img_paths[j]
                    img_nm = img_path.split('/')[-1]
                    # initialize img dict
                    if img_nm not in imgnm_to_oid_to_info[set_id][vid_id]:
                        imgnm_to_oid_to_info[set_id][vid_id][img_nm] = {}
                        imgnm_to_oid_to_info[set_id][vid_id][img_nm]['ped'] = {}
                        imgnm_to_oid_to_info[set_id][vid_id][img_nm]['veh'] = {}
                    # initialize obj dict
                    imgnm_to_oid_to_info[set_id][vid_id]\
                        [img_nm]['ped'][oid] = {}
                    # add obj info
                    bbox = tracks['bbox'][i][j]  # ltrb float
                    bbox = list(map(int, bbox))  # int
                    imgnm_to_oid_to_info[set_id][vid_id][img_nm]['ped']\
                        [oid]['bbox'] = bbox
            # vehicle tracks
            print(f'Saving imgnm to objid to obj info of vehicles in \
                {self.dataset_name}')
            tracks = v_tracks
            for set_id in v_tracks:
                if set_id not in imgnm_to_oid_to_info:
                    imgnm_to_oid_to_info[set_id] = {}
                for vid_id in v_tracks[set_id]:
                    if vid_id not in imgnm_to_oid_to_info[set_id]:
                        imgnm_to_oid_to_info[set_id][vid_id] = {}
                    for oid in v_tracks[set_id][vid_id]:
                        # one vehcle might have multiple tracks
                        for i in range(len(v_tracks[set_id]\
                                           [vid_id][oid]['img_nm'])):
                            img_nms = v_tracks[set_id][vid_id][oid]['img_nm'][i]
                            bboxes = v_tracks[set_id][vid_id][oid]['bbox'][i]
                            assert len(img_nms) == len(bboxes), \
                                (len(img_nms), len(bboxes))
                            for j in range(len(img_nms)):
                                img_nm = img_nms[j]
                                bbox = bboxes[j]
                                bbox = list(map(int, bbox))  # int
                                if img_nm not in imgnm_to_oid_to_info[set_id]\
                                    [vid_id]:
                                    imgnm_to_oid_to_info[set_id]\
                                        [vid_id][img_nm] = {}
                                    imgnm_to_oid_to_info[set_id]\
                                        [vid_id][img_nm]['ped'] = {}
                                    imgnm_to_oid_to_info[set_id]\
                                        [vid_id][img_nm]['veh'] = {}
                                # initialize obj dict
                                imgnm_to_oid_to_info[set_id]\
                                    [vid_id][img_nm]['veh'][oid] = {}
                                # add obj info
                                imgnm_to_oid_to_info[set_id][vid_id]\
                                    [img_nm]['veh'][oid]['bbox'] = bbox
        elif self.dataset_name == 'JAAD':
            raise ValueError('JAAD is discarded')
        # save the file
        with open(save_path, 'wb') as f:
            pickle.dump(imgnm_to_oid_to_info, f)
        
        return imgnm_to_oid_to_info
        
    def split_to_samples(self, data):
        seq_length = self._obs_len + self._pred_len
        overlap_stride = self._obs_len if self.overlap_ratio == 0 \
            else int((1 - self.overlap_ratio) * self._obs_len)
        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        #  Check data types
        data.pop('image_dimension', None)
        samples = {}
        for dt in data.keys():
            try:
                samples[dt] = data[dt]
            except KeyError:
                raise ('Wrong data type is selected %s' % dt)

        samples['image'] = data['image']
        samples['ped_id'] = data['ped_id']

        #  Sample tracks into samples
        print('---------------Split tracks to samples---------------')
        print(samples.keys())
        for k in tqdm(samples.keys()):
            # import pdb;pdb.set_trace()
            tracks = []
            for track in samples[k]:
                # skip too short 
                if len(track) < seq_length:
                    continue
                if self.tte is not None:
                    if len(track) < self.tte[1]:
                        continue
                    start_idx = len(track) - self.tte[1]
                    end_idx = len(track) - self.tte[0] - seq_length
                    tracks.extend([track[i:i + seq_length] for i in range(start_idx, end_idx, overlap_stride)])
                else:
                    tracks.extend([track[i:i + seq_length] for i in range(0, len(track) - seq_length, overlap_stride)])
            samples[k] = tracks

        #  Normalize tracks by subtracting bbox/center at first time step from the rest
        print('---------------Normalize traj---------------')
        normed_samples = copy.deepcopy(samples)
        
        if self.normalize_pos:
            for i in range(len(normed_samples['bbox'])):
                normed_samples['bbox'][i] = np.subtract(normed_samples['bbox'][i][:],
                                                        normed_samples['bbox'][i][0]).tolist()
            for i in range(len(normed_samples['center'])):
                normed_samples['center'][i] = np.subtract(normed_samples['center'][i][:],
                                                            normed_samples['center'][i][0]).tolist()
        if self.traj_mode == 'xywh':
            for i in range(len(normed_samples['bbox'])):
                normed_samples['bbox'][i] = ltrb2xywh(normed_samples['bbox'][i])

        obs_slices = {}
        pred_slices = {}
        obs_bbox_normed = []
        pred_bbox_normed = []
        obs_center_normed = []
        pred_center_normed = []

        #  Split obs and pred
        print('---------------Split obs and pred---------------')
        for k in samples.keys():
            obs_slices[k] = []
            pred_slices[k] = []
            obs_slices[k].extend([d[0:self._obs_len] for d in samples[k]])
            pred_slices[k].extend([d[self._obs_len:] for d in samples[k]])
            if k == 'bbox':
                obs_bbox_normed.extend([d[0:self._obs_len] for d in normed_samples[k]])
                pred_bbox_normed.extend([d[self._obs_len:] for d in normed_samples[k]])

        if self.seq_type == 'crossing':
            # if self.subset == 'test':
            #     for sample in obs_slices['activities']:
            #         print(len(sample))
            #     pdb.set_trace()
            obs_slices['activities'] = np.array(obs_slices['activities'])[:, -1]
            pred_slices['activities'] = np.array(pred_slices['activities'])[:, -1]
        else:
            obs_slices['intention_binary'] = np.array(obs_slices['intention_binary'])[:, -1]
            pred_slices['intention_binary'] = np.array(pred_slices['intention_binary'])[:, -1]

        # record ped id, img id
        obs_img_nm_int = []
        ped_id_int = []
        vid_id_int = []
        set_id_int = []
        for i in range(len(obs_slices['image'])):
            pid = obs_slices['ped_id'][i][0][0]
            str_list = pid.split('_')
            int_pid = []
            for s in str_list:
                if 'b' in s:
                    int_pid.append(-int(s.replace('b', '')))
                else:
                    int_pid.append(int(s))
            ped_id_int.append(int_pid)
            int_img_nm_seq = []
            int_vid_nm_seq = []
            int_set_nm_seq = []
            for path in obs_slices['image'][i]:
                nm = path.split('/')[-1].split('.')[0]
                vid_nm = path.split('/')[-2].replace('video_', '')
                int_img_nm_seq.append(int(nm))
                int_vid_nm_seq.append(int(vid_nm))
                if self.dataset_name == 'PIE':
                    set_nm = path.split('/')[-3].replace('set', '')
                    int_set_nm_seq.append(int(set_nm))
                else:
                    int_set_nm_seq.append(0)
            obs_img_nm_int.append(int_img_nm_seq)
            vid_id_int.append(int_vid_nm_seq)
            set_id_int.append(int_set_nm_seq)
        obs_img_nm_int = np.array(obs_img_nm_int)
        ped_id_int = np.array(ped_id_int)
        vid_id_int = np.array(vid_id_int)

        if self.dataset_name == 'PIE':
            set_id_int = np.array(set_id_int)
            all_samples = {'obs_img_nm_int': obs_img_nm_int,
                    'obs_ped_id_int': ped_id_int,
                    'obs_vid_id_int': vid_id_int,
                    'obs_set_id_int': set_id_int,
                    'obs_image_paths': obs_slices['image'],
                    'obs_bbox_normed': obs_bbox_normed,
                    'obs_bbox': obs_slices['bbox'],
                    # 'obs_intent': obs_slices['intention_binary'],
                    'obs_pid': obs_slices['ped_id'],
                    'obs_occ': obs_slices['occlusion'],
                    'obs_ego': obs_slices[self.ego_motion_key],
                    'pred_image_paths': pred_slices['image'],
                    'pred_bbox_normed': pred_bbox_normed,
                    'pred_bbox': pred_slices['bbox'],
                    'pred_ego': pred_slices[self.ego_motion_key],
                    # 'pred_intent': pred_slices['intention_binary'],
                    'pred_pid': pred_slices['ped_id'],
                    'pred_occ': pred_slices['occlusion']}
        elif self.dataset_name == 'JAAD':
            all_samples = {'obs_img_nm_int': obs_img_nm_int,
                'obs_ped_id_int': ped_id_int,
                'obs_vid_id_int': vid_id_int,
                'obs_image_paths': obs_slices['image'],
                'obs_bbox_normed': obs_bbox_normed,
                'obs_bbox': obs_slices['bbox'],
                # 'obs_intent': obs_slices['intention_binary'],
                'obs_pid': obs_slices['ped_id'],
                'obs_occ': obs_slices['occlusion'],
                'obs_ego': obs_slices['vehicle_act'],
                'pred_image_paths': pred_slices['image'],
                'pred_bbox_normed': pred_bbox_normed,
                'pred_bbox': pred_slices['bbox'],
                'pred_ego': pred_slices['vehicle_act'],
                # 'pred_intent': pred_slices['intention_binary'],
                'pred_pid': pred_slices['ped_id'],
                'pred_occ': pred_slices['occlusion']}
        # if self.use_skeleton and self.skeleton_mode == 'coord':
        #     all_samples['obs_skeletons'] = np.array(obs_slices['skeletons'])
        #     all_samples['pred_skeletons'] = np.array(pred_slices['skeletons'])
        #     import pdb;pdb.set_trace()
        #     assert all_samples['obs_skeletons'].shape[0] == all_samples['obs_img_nm_int'].shape[0], (all_samples['obs_skeletons'].shape, all_samples['obs_img_nm_int'].shape)
        if self.seq_type == 'crossing':
            all_samples['target'] = pred_slices['activities']
            if self.recog_act:
                all_samples['target'] = obs_slices['activities']
        else:
            all_samples['target'] = obs_slices['intention_binary']
        # pdb.set_trace()
        return all_samples

    def _add_augment(self, data):
        '''
        data: self.samples, dict of lists(num samples, ...)
        transforms: torchvision.transforms
        '''
        if 'crop' in self.augment_mode:
            if self.use_img:
                self.transforms['resized_crop']['img'] = \
                    RandomResizedCrop(size=self.bbox_size, # (h, w)
                                        scale=(0.75, 1), 
                                        ratio=(self.bbox_size[1]/self.bbox_size[0], 
                                                self.bbox_size[1]/self.bbox_size[0]))  # w / h
            if self.use_context:
                self.transforms['resized_crop']['ctx'] = RandomResizedCrop(size=self.ctx_size, # (h, w)
                                                                        scale=(0.75, 1), 
                                                                        ratio=(self.ctx_size[1]/self.ctx_size[0], self.ctx_size[1]/self.ctx_size[0]))  # w / h
            if self.use_skeleton and self.skeleton_mode == 'pseudo_heatmap':
                self.transforms['resized_crop']['sk'] = RandomResizedCrop(size=(48, 48), # (h, w)
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

                # init extra samples
                h_flip_samples = {}
                for k in data:
                    h_flip_samples[k] = []

                # duplicate samples
                for i in range(len(data['obs_image_paths'])):
                    if data['target'][i][0] == 1:
                        for k in data:
                            h_flip_samples[k].append(copy.deepcopy(data[k][i]))
                h_flip_samples['hflip_flag'] = [True] * len(h_flip_samples['obs_image_paths'])
                data['hflip_flag'] = np.array([False] * len(data['obs_image_paths']))
                # concat
                for k in data:
                    data[k] = np.concatenate([data[k], np.array(h_flip_samples[k])], axis=0)

            self.num_samples = len(data['obs_image_paths'])
            print(f'Num samples after flip: {self.num_samples}')
        return data

    def downsample_seq(self):
        new_samples = {}
        for k in self.samples:
            if 'obs' in k and len(self.samples[k][0]) == self._obs_len:
                new_samples[k] = []
                for s in range(len(self.samples[k])):
                    ori_seq = self.samples[k][s]
                    new_seq = []
                    for i in range(0, self._obs_len, self.obs_interval+1):
                        new_seq.append(ori_seq[i])
                    new_samples[k].append(new_seq)
                    assert len(new_samples[k][s]) == self.obs_len, (k, len(new_samples[k]), self.obs_len)
                new_samples[k] = np.array(new_samples[k])

            if 'pred' in k and len(self.samples[k][0]) == self._pred_len:
                new_samples[k] = []
                for s in range(len(self.samples[k])):
                    ori_seq = self.samples[k][s]
                    new_seq = []
                    for i in range(0, self._pred_len, self.obs_interval+1):
                        new_seq.append(ori_seq[i])
                    new_samples[k].append(new_seq)
                    assert len(new_samples[k][s]) == self.pred_len, (k, len(new_samples[k]), self.pred_len)
                new_samples[k] = np.array(new_samples[k])
        for k in new_samples:
            # print('new k', k)
            # print(new_samples[k].shape)
            self.samples[k] = new_samples[k]

    def load_cropped_image(self, 
                           img_path, 
                           bbox, 
                           target_size=(224, 224), 
                           resize_mode='resize'):
        '''

        :param img_path:
        :param bbox: l, t, r, d
        :param target_size: W, H
        :param resize_mode:
        :return:
            resized: np array (H, W, 3)
        '''
        t1 = time.time()
        img = cv2.imread(img_path)
        t2 = time.time()
        print('load time', t2-t1)
        l, t, r, b = map(int, bbox)
        cropped = img[t:b, l:r]
        h, w = cropped.shape[0], cropped.shape[1]
        if resize_mode == 'resize':
            resized = cv2.resize(cropped, target_size)
        else:
            if float(target_size[0]) / target_size[1] < float(w) / h:
                ratio = float(target_size[0]) / w
            else:
                ratio = float(target_size[1]) / h
            new_size = (int(w*ratio), int(h*ratio))
            cropped = cv2.resize(cropped, new_size)
            w_pad = target_size[0] - new_size[0]
            h_pad = target_size[1] - new_size[1]
            resized = cv2.copyMakeBorder(cropped,
                                         0,h_pad,0,w_pad,
                                         cv2.BORDER_CONSTANT,value=(0, 0, 0))  # t, b, l, r
        t3 = time.time()
        print('resize time', t3-t2)
        return resized  # H, W, C

    def load_skeletons(self, pids, skeleton_mode='coord'):
        skeleton_tracks = []
        print('------------Loading skeleton data-----------')
        if skeleton_mode == 'coord':
            tbar = tqdm(pids)
            for pid in tbar:
                cur_track_path = os.path.join(self.sk_vis_path, pid, 'skeletons.pkl')
                with open(cur_track_path, 'rb') as f:
                    track = pickle.load(f)  # shape: (track_len, n_j, 3)
                skeleton_tracks.append(track[:, :, :2])
            tbar.close()
        # elif skeleton_mode == 'heatmap':
        #     tbar = tqdm(pids)
        #     for pid in tbar:
        #         # import pdb;pdb.set_trace()
        #         cur_track_path = os.path.join(self.skeleton_path, pid, 'heatmaps.pkl')
        #         with open(cur_track_path, 'rb') as f:
        #             track = pickle.load(f)  # shape: (track_len, 17, 96, 72)
        #         skeleton_tracks.append(track)
        print('------------Done loading skeleton data-----------')
        return skeleton_tracks

    def max_min_bbox_size(self):
        obs_bboxes = np.array(self.samples['obs_bbox'])  # ltrb
        pred_bboxes = np.array(self.samples['pred_bbox'])
        bboxes = np.concatenate([obs_bboxes, pred_bboxes], axis=1)
        h = bboxes[:, :, 3] - bboxes[:, :, 1]
        w = bboxes[:, :, 2] - bboxes[:, :, 0]

        # PIE: 688, 375, 10, 3, 124, 40, 301, 109
        # JAAD: 635 329 18 7 101 43 296 129
        return np.max(h), np.max(w), np.min(h), np.min(w), np.percentile(h, 50), np.percentile(w, 50), np.percentile(h, 90), np.percentile(w, 90)
    
    def rm_small_bb(self, data, min_size):
        print('----------Remove small bb-----------')
        min_w, min_h = min_size
        idx = list(range(self.num_samples))
        new_idx = list(range(self.num_samples))

        bboxes = np.array(data['obs_bbox'])  # ltrb
        hws = np.stack([bboxes[:, :, 3] - bboxes[:, :, 1], bboxes[:, :, 2] - bboxes[:, :, 0]], axis=2)  # mean: 134, 46
        print('hws shape: ', hws.shape)
        print('mean h: ', np.mean(hws[:, :, 0]))
        print('mean w: ', np.mean(hws[:, :, 1]))
        for i in idx:
            for hw in hws[i]:
                if hw[0] < min_h or hw[1] < min_w:
                    new_idx.remove(i)
                    break
        
        for k in data.keys():
            data[k] = data[k][new_idx]
        print('n samples before removing small bb', self.num_samples)
        self.num_samples = len(new_idx)
        print('n samples after removing small bb', self.num_samples)

        return data

    def rm_occluded(self, data, max_occ):
        print('----------Remove occluded---------')
        idx = list(range(self.num_samples))
        new_idx = list(range(self.num_samples))

        for i in idx:
            for occ in data['obs_occ'][i]:
                if occ > max_occ:
                    new_idx.remove(i)
                    break
        
        for k in data.keys():
            data[k] = data[k][new_idx]
        print('n samples before removing occluded', self.num_samples)
        self.num_samples = len(new_idx)
        print('n samples after removing occluded', self.num_samples)

        return data
    
    def rm_720x1280(self, all_data):
        # video to remove: 61~70
        rm_list = ['video_00'+str(i) for i in range(61, 71)]
        n_track = len(all_data['image'])
        idxs = list(range(n_track))
        new_idxs = list(range(n_track))
        print('Removing video 61~70')
        print(f'Num video before removing: {n_track}')
        for i in idxs:
            video = all_data['image'][i][0].split('/')[-2]
            vid = int(video.replace('video_', ''))
            if vid >= 61 and vid <= 70:
                new_idxs.remove(i)
        all_data.pop('image_dimension', None)
        for k in all_data:
            # print(np.array(all_data[k]).shape, k)
            all_data[k] = np.array(all_data[k])[new_idxs]
        print('Num video after removing: ', len(all_data['image']))
        return all_data



    def balance(self, balance_label, all_samples, random_seed=42):
        for lbl in all_samples[balance_label]:
            for i in lbl:  # 每条track
                if i not in [0, 1]:
                    raise Exception("The label values used for balancing must be"
                                    " either 0 or 1")
        print('---------------------------------------------------------')
        print("Balancing the number of positive and negative intention samples")
        gt_labels = [gt[0] for gt in all_samples[balance_label]]  # 每条track一个值
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
            return all_samples
        else:
            print('Unbalanced: \t Positive: {} \t \
                  Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
                n_rm_index = np.where(np.array(gt_labels) == 1)[0]
                num_keep = num_pos_samples
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]
                n_rm_index = np.where(np.array(gt_labels) == 0)[0]
                num_keep = num_neg_samples
            np.random.seed(random_seed)
            np.random.shuffle(rm_index)
            kp_index = rm_index[:num_keep]
            new_index = list(set(n_rm_index) | set(kp_index))

            for k in all_samples.keys():
                all_samples[k] = all_samples[k][new_index]
                assert len(all_samples[k]) == 2 * num_keep
            return all_samples


def save_cropped_imgs(resize_mode='padded', 
                      target_size=(224, 224), 
                      dataset_name='PIE', 
                      ):
    import os
    if dataset_name == 'PIE':
        cropped_root = '/home/y_feng/workspace6/datasets/PIE_dataset/cropped_images'
        dataset_root = '/home/y_feng/workspace6/datasets/PIE_dataset'
        data_base = PIE(data_path=dataset_root)
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
        cropped_root = '/home/y_feng/workspace6/datasets/JAAD/cropped_images'
        dataset_root = '/home/y_feng/workspace6/datasets/JAAD'
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
        data_base = JAAD(data_path='/home/y_feng/workspace6/datasets/JAAD')
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

def save_context_imgs(mode='mask_ped', target_size=(224, 224), dataset_name='PIE'):
    import os
    if dataset_name == 'PIE':
        context_root = '/home/y_feng/workspace6/datasets/PIE_dataset/context'
        dataset_root = '/home/y_feng/workspace6/datasets/PIE_dataset'
        data_base = PIE(data_path=dataset_root)
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
        context_root = '/home/y_feng/workspace6/datasets/JAAD/context'
        dataset_root = '/home/y_feng/workspace6/datasets/JAAD'
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
        data_base = JAAD(data_path=dataset_root)
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

def track_vehicles():
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='crop')

    parser.add_argument('--h', type=int, default=224)
    parser.add_argument('--w', type=int, default=224)
    parser.add_argument('--dataset_name', type=str, default='PIE')
    # crop args
    parser.add_argument('--resize_mode', type=str, default='padded')
    # context args
    parser.add_argument('--context_mode', type=str, default='mask_ped')

    args = parser.parse_args()

    if args.action == 'crop':
        save_cropped_imgs(resize_mode=args.resize_mode, 
                          target_size=(args.w, args.h), 
                          dataset_name=args.dataset_name)
    elif args.action == 'context':
        save_context_imgs(mode=args.context_mode, 
                          target_size=(args.w, args.h), 
                          dataset_name=args.dataset_name)
    elif args.action == 'pseudo_heatmap':
        coord2pseudo_heatmap(h=args.h, 
                             w=args.w, 
                             dataset_name=args.dataset_name)