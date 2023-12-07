import imghdr
import pickle
import torch
import os
import scipy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def vis_weight(weights, group_gap=0.4, path=''):
    '''
    weights: tensor (n_cls, n_protos)
    '''
    # plt.close()
    plt.figure(figsize=(15, 10))
    n_c = weights.size(0)
    n_p = weights.size(1)
    # tensor to ndarray
    _weights = weights.cpu().detach().numpy()  # Can't call numpy() on Tensor that requires grad

    # calc width
    group_width = 1 - group_gap
    bar_width = group_width / n_c

    # x coordinates
    x = np.arange(n_p)
    x0 = x - group_width / 2
    x_labels = np.arange(n_p).astype(dtype=np.str)
    for i in range(n_c):
        row = _weights[i]
        plt.bar(x0 + i*bar_width, row, bar_width)
        
    plt.xticks(x, x_labels)
    plt.ylabel('Score')
    plt.xlabel('Proto index')
    plt.savefig(path)
    plt.close()