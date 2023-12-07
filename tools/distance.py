import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from .utils import idx2onehot
import math


def cartesian_similarity(feats1:torch.Tensor, 
                         feats2:torch.Tensor,
                         mode='l2',
                         ):
    '''
    feats1: b, k1, d
    feats2: b, k2, d
    return:
        simi_mats: k1*k1, b, b
    '''
    b1, k1, d1 = feats1.size()
    b2, k2, d2 = feats2.size()
    assert b1 == b2 and d1 == d2, (b1, b2, d1, d2)
    assert k1%k2 == 0, (k1, k2)
    b = b1
    tensor1 = feats1.repeat(b, 1, 1)  # b*b, k1, d
    tensor2 = feats2.repeat_interleave(b, dim=0)  # b*b, k2, d
    tensor1 = tensor1.unsqueeze(1)  # b*b, 1, k1, d
    tensor2 = tensor2.unsqueeze(2)  # b*b, k2, 1, d

    if mode == 'simi':
        simi_matis = torch.sum(tensor1 * tensor2, dim=-1, keepdim=False)  # b*b, k2, k1