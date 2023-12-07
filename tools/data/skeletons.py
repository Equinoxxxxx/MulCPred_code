import imghdr
import pickle
import torch
import os
import scipy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import random

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..utils import makedir

def generate_one_pseudo_heatmap(img_h, img_w, centers, max_values, sigma=0.6, eps=1e-4):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The max values of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for center, max_value in zip(centers, max_values):
            mu_x, mu_y = center[0], center[1]
            if max_value < eps:
                continue

            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            heatmap[st_y:ed_y,
                    st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x],
                                            patch)

        return heatmap


def coord2pseudo_heatmap(dataset_name,
                         h=48,
                         w=48,
                         ):
    if dataset_name == 'PIE':
        coord_root = '/home/y_feng/workspace6/datasets/PIE_dataset/sk_coords/even_padded/288w_by_384h'
        tgt_root = os.path.join('/home/y_feng/workspace6/datasets/PIE_dataset/sk_pseudo_heatmaps/even_padded', str(w)+'w_by_'+str(h)+'h')
    elif dataset_name == 'JAAD':
        coord_root = '/home/y_feng/workspace6/datasets/JAAD/sk_coords/even_padded/288w_by_384h'
        tgt_root = os.path.join('/home/y_feng/workspace6/datasets/JAAD/sk_pseudo_heatmaps/even_padded', str(w)+'w_by_'+str(h)+'h')
    elif dataset_name == 'TITAN':
        coord_root = '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/sk_coords/even_padded/288w_by_384h'
        tgt_root = os.path.join('/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/sk_pseudo_heatmaps/even_padded', str(w)+'w_by_'+str(h)+'h')
    else:
        raise NotImplementedError(dataset_name)
    
    makedir(tgt_root)
    if coord_root == '/home/y_feng/workspace6/datasets/PIE_dataset/sk_coords/even_padded/288w_by_384h' \
        or coord_root == '/home/y_feng/workspace6/datasets/JAAD/sk_coords/even_padded/288w_by_384h' \
        or coord_root == '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/sk_coords/even_padded/288w_by_384h':
        ori_h = 384
        ori_w = 288
    else:
        raise NotImplementedError(coord_root)
    
    h_ratio = h / ori_h
    w_ratio = w / ori_w

    if dataset_name in ('PIE', 'JAAD'):
        for pid in os.listdir(coord_root):
            pid_path = os.path.join(coord_root, pid)
            tgt_pid_path = os.path.join(tgt_root, pid)
            makedir(tgt_pid_path)
            for file in os.listdir(pid_path):
                img_nm = file.replace('.pkl', '')
                src_path = os.path.join(pid_path, file)
                tgt_path = os.path.join(tgt_pid_path, file)
                with open(src_path, 'rb') as f:
                    coords = pickle.load(f)
                tgt_heatmaps = []
                for coord in coords:
                    tgt_h = int(coord[0] * h_ratio)
                    tgt_w = int(coord[1] * w_ratio)
                    tgt_coord = (tgt_w, tgt_h)
                    tgt_heatmap = generate_one_pseudo_heatmap(img_h=h,
                                                            img_w=w,
                                                            centers=[tgt_coord],
                                                            max_values=[coord[-1]])
                    tgt_heatmaps.append(tgt_heatmap)
                tgt_heatmaps = np.stack(tgt_heatmaps, axis=0)
                assert tgt_heatmaps.shape == (17, h, w), tgt_heatmaps.shape
                with open(tgt_path, 'wb') as f:
                    pickle.dump(tgt_heatmaps, f)
                print(tgt_path, '  done')
    elif dataset_name == 'TITAN':
        for cid in os.listdir(coord_root):
            cid_path = os.path.join(coord_root, cid)
            tgt_cid_path = os.path.join(tgt_root, cid)
            makedir(tgt_cid_path)
            for pid in os.listdir(cid_path):
                pid_path = os.path.join(cid_path, pid)
                tgt_pid_path = os.path.join(tgt_cid_path, pid)
                makedir(tgt_pid_path)
                for file in os.listdir(pid_path):
                    img_nm = file.replace('.pkl', '')
                    src_path = os.path.join(pid_path, file)
                    tgt_path = os.path.join(tgt_pid_path, file)
                    with open(src_path, 'rb') as f:
                        coords = pickle.load(f)
                    tgt_heatmaps = []
                    for coord in coords:
                        tgt_h = int(coord[0] * h_ratio)
                        tgt_w = int(coord[1] * w_ratio)
                        tgt_coord = (tgt_w, tgt_h)
                        tgt_heatmap = generate_one_pseudo_heatmap(img_h=h,
                                                                img_w=w,
                                                                centers=[tgt_coord],
                                                                max_values=[coord[-1]])
                        tgt_heatmaps.append(tgt_heatmap)
                    tgt_heatmaps = np.stack(tgt_heatmaps, axis=0)
                    assert tgt_heatmaps.shape == (17, h, w), tgt_heatmaps.shape
                    with open(tgt_path, 'wb') as f:
                        pickle.dump(tgt_heatmaps, f)
                    print(tgt_path, '  done')