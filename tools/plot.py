import imghdr
import pickle
import torch
import os
import scipy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sklearn

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

EGO_RANGE = {
    'PIE': (-67, 35),
    'JAAD': (-67, 35),
    'TITAN': (-2, 2.5)
}


def draw_multi_task_curve(acc_curves_train, acc_curves_test, ce_curves_train, ce_curves_test, train_res, test_res, model_dir, test_every, set_nm='atomic'):
    acc_train, ce_train = train_res[0], train_res[1]
    acc_test, ce_test = test_res[0], train_res[1]
    acc_curves_train.append(acc_train)
    acc_curves_test.append(acc_test)
    ce_curves_train.append(ce_train)
    ce_curves_test.append(ce_test)
    draw_curves(path=os.path.join(model_dir, '_' + set_nm + '_acc.png'), train_curve=acc_curves_train, test_curve=acc_curves_test, test_every=test_every)
    draw_curves(path=os.path.join(model_dir, '_' + set_nm + '_mAP.png'), train_curve=ce_curves_train, test_curve=ce_curves_test, test_every=test_every)

    return acc_curves_train, acc_curves_test, ce_curves_train, ce_curves_test


def draw_curves(path, train_curve, test_curve, metric_type='loss', test_every=1):
    plt.figure(dpi=300, figsize=(15, 10))
    plt.plot(train_curve, color='r', label='train')
    if test_curve is not None:
        plt.plot(test_curve, color='b', label='test')
    plt.xlabel('epoch / '+str(test_every))
    metric_type = path.split('/')[-1].replace('.png', '')
    plt.ylabel(metric_type)
    plt.legend()
    plt.savefig(path)
    plt.close()


def draw_curves2(path, val_lists, labels, colors=['r', 'b'], vis_every=1):
    # down sample
    lengths = [len(l) for l in val_lists]
    min_len = min(lengths)
    plt.figure(dpi=300, figsize=(15, 10))
    for i in range(len(val_lists)):
        curlist = val_lists[i].cpu().numpy() if not isinstance(val_lists[i], list) else np.array(val_lists[i])
        assert len(curlist)%min_len == 0
        if len(curlist) > min_len:
            ratio = len(curlist) // min_len
            idx = list(range(ratio-1, len(curlist), ratio))
            curlist = curlist[idx]
        assert len(curlist) == min_len, (len(curlist), min_len, val_lists)
        plt.plot(curlist, color=colors[i], label=labels[i])
    plt.xlabel('epoch / '+str(vis_every))
    plt.legend()
    plt.savefig(path)
    plt.close()


def draw_train_test_curve(train_curve, test_curve, test_every, path):
    plt.figure(dpi=300, figsize=(15, 10))
    plt.plot(train_curve, color='r', label='train')
    if test_curve is not None:
        plt.plot(test_curve, color='b', label='test')
    plt.xlabel('epoch / '+str(test_every))
    metric_type = path.split('/')[-1].replace('.png', '')
    plt.ylabel(metric_type)
    plt.legend()
    plt.savefig(path)
    plt.close()

def draw_train_val_test_curve(train_curve, val_curve, test_curve, 
                              test_every, path):
    if len(train_curve) == 0 or len(val_curve) == 0\
        or len(test_curve) == 0:
        return
    assert len(train_curve) == len(val_curve) == len(test_curve)
    plt.figure(dpi=300, figsize=(15, 10))
    plt.plot(train_curve, color='r', label='train')
    if val_curve is not None:
        plt.plot(val_curve, color='g', label='val')
    if test_curve is not None:
        plt.plot(test_curve, color='b', label='test')
    plt.xlabel('epoch / '+str(test_every))
    metric_type = path.split('/')[-1].replace('.png', '')
    plt.ylabel(metric_type)
    plt.legend()
    plt.savefig(path)
    plt.close()


def vis_ego_sample(ego, lim, path):
    plt.plot(ego, color='r', linewidth=2)
    plt.ylim(lim)
    plt.axhline(0, color='black', linewidth=.5)
    plt.savefig(path)
    plt.close()
    

def vis_weight_multi_cls(weights, group_gap=0.4, path=''):
    '''
    weights: tensor (n_cls, n_protos)
    '''
    # plt.close()
    plt.figure(dpi=300, figsize=(15, 10))
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
    plt.xlabel('Concept index')
    plt.savefig(path)
    plt.close()

def vis_weight_single_cls(weights, group_gap=0.4, path=''):
    '''
    weights: tensor (n_protos,)
    '''
    # plt.close()
    plt.figure(dpi=300, figsize=(30, 15))
    n_c = 1
    n_p = weights.shape[0]
    # tensor to ndarray
    if isinstance(weights, torch.Tensor):
        _weights = weights.cpu().detach().numpy()  # Can't call numpy() on Tensor that requires grad
    else:
        _weights = weights
    # calc width
    group_width = 1 - group_gap
    bar_width = group_width / n_c

    # x coordinates
    x = np.arange(n_p)
    x0 = x - group_width / 2
    x_labels = np.arange(n_p).astype(dtype=np.str)
    plt.bar(x0, _weights, bar_width)
        
    plt.xticks(x, x_labels)
    plt.ylabel('Score')
    plt.xlabel('Concept index')
    plt.grid(axis='x')
    plt.savefig(path)
    plt.close()

def draw_logits_histogram(logits, path):
    '''
    logits: tensor(n,)
    '''
    logits = torch.softmax(logits, dim=-1)
    logits = logits.detach().cpu().numpy()
    plt.figure(dpi=300)
    plt.hist(logits, bins=40, facecolor="blue", edgecolor="black",)
    plt.xlabel("logits value")
    plt.ylabel("num samples")
    plt.savefig(path)
    plt.close()

def draw_morf(logits, path):
    '''
    logits: torch.tensor(n,) in descending order
    '''
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    logits = logits.detach().cpu().numpy()
    plt.plot(logits)
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Prediction')
    plt.savefig(path)
    plt.close()

def draw_morfs(logits_list, path):
    '''
    logits: list [torch.tensor(n,) in descending order]
    '''
    
    plt.figure(dpi=300)
    for i in range(len(logits_list)):
        logits = logits_list[i]
        logits = logits.detach().cpu().numpy()
        if i == 0:
            plt.plot(logits, color='b', label='Ours')
            print(logits.shape)
        else:
            plt.plot(logits, color='r', label='SENN')
            print(logits.shape)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Prediction')
    plt.legend()
    plt.savefig(path)
    plt.close()

def draw_morfs_both(logits_list, path, labels=None):
    '''
    logits: list [torch.tensor(n,) in descending order]
    '''
    
    plt.figure(dpi=300)
    for i in range(len(logits_list)):
        logits = logits_list[i]
        logits = logits.detach().cpu().numpy()
        plt.plot(logits, label=labels[i])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Prediction')
    plt.legend()
    plt.savefig(path)
    plt.close()