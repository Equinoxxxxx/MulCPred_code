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


def find_highly_act_sample(dataloader,
                           model_parallel,
                           log=print,
                           save_dir='',
                           num_epoch=0):
    makedir(save_dir)
    model_parallel.eval()
    log('\t explain')
    start = time.time()
    proto_epoch_dir = os.path.join(save_dir, 'epoch-'+str(num_epoch))
    makedir(proto_epoch_dir)
    search_batch_size = dataloader.batch_size
    for i, data in enumerate(tqdm(dataloader)):
        start_index_of_search_batch = i * search_batch_size
        color_order = dataloader.dataset.color_order
        img_norm_mode = dataloader.dataset.normalize_img_mode

        pass


def find_in_batch(data,
                  ped_id_int,
                  img_nm_int,
                  sample_idx,
                  model_parallel,
                  color_order,
                  img_norm_mode
                  ):
    
    
    pass