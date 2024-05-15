import cv2
import numpy as np
import os
import pickle
import json
from tqdm import tqdm
from ..utils import makedir
from .crop_images import crop_ctx_PIE_JAAD, crop_img_PIE_JAAD, crop_img_TITAN, crop_ctx_TITAN
from .get_skeletons import get_skeletons
from config import dataset_root


def prepare_data(datasets):
    if 'PIE' in datasets:
        crop_img_PIE_JAAD(dataset_name='PIE')
        crop_ctx_PIE_JAAD(dataset_name='PIE')
    if 'JAAD' in datasets:
        crop_img_PIE_JAAD(dataset_name='JAAD')
        crop_ctx_PIE_JAAD(dataset_name='JAAD')
    if 'TITAN' in datasets:
        crop_img_TITAN()
        crop_ctx_TITAN()
    get_skeletons(datasets=datasets)


if __name__ == '__main__':
    prepare_data()