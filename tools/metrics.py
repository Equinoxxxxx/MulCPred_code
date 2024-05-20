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
import sys
sys.path.append("..")
from tools.utils import idx2onehot


def calc_acc(preds, labels):
    '''
    preds: tensor(b,) idx (or tensor(b, n_cls), logits)
    labels: tensor(b), idx
    '''
    with torch.no_grad():
        # logits 2 idx
        if len(preds.size()) == 2:
            preds = torch.max(preds.detach(), 1)[1]  # (b,) idx

        preds = preds.detach().int().cpu().numpy()
        labels = labels.int().cpu().numpy()
        acc = sklearn.metrics.accuracy_score(labels, preds)

    return acc

def calc_recall(preds, labels):
    '''
    preds: tensor(b,) idx (or tensor(b, n_cls), logits)
    labels: tensor(b), idx
    '''
    with torch.no_grad():
    # logits 2 idx
        if len(preds.size()) == 2:
            preds = torch.max(preds.detach(), 1)[1]  # (b,) idx

        preds = preds.detach().int().cpu().numpy()
        labels = labels.int().cpu().numpy()
        recall = sklearn.metrics.recall_score(labels, preds, average=None)

    return recall

def calc_precision(preds, labels):
    '''
    preds: tensor(b,) idx (or tensor(b, n_cls), logits)
    labels: tensor(b), idx
    '''
    with torch.no_grad():
        # logits 2 idx
        if len(preds.size()) == 2:
            preds = torch.max(preds.detach(), 1)[1]  # (b,) idx

        preds = preds.detach().int().cpu().numpy()
        labels = labels.int().cpu().numpy()
        precision = sklearn.metrics.precision_score(labels, preds, average=None)

    return precision

def calc_confusion_matrix(preds, labels, norm=None):
    '''
    preds: tensor(b,) idx (or tensor(b, n_cls), logits)
    labels: tensor(b), idx
    '''
    with torch.no_grad():
    # logits 2 idx
        if len(preds.size()) == 2:
            preds = torch.max(preds.detach(), 1)[1]  # (b,) idx

        preds = preds.detach().int().cpu().numpy()
        labels = labels.int().cpu().numpy()

        conf_mat = sklearn.metrics.confusion_matrix(labels, preds, normalize=norm)
    return conf_mat


def calc_auc(preds, labels, average='macro'):
    '''
    preds: tensor(b, n_cls)  logits
    labels: tensor(b, n_cls) onehot (or tensor(b,) idx)
    '''
    with torch.no_grad():
        # idx 2 onehot
        if len(labels.size()) == 1:
            labels = idx2onehot(labels, preds.size(1))

        preds = preds.detach().cpu().numpy()
        labels = labels.int().cpu().numpy()
        try:
            if average == 'binary':
                if preds.shape[-1] > 1:
                    auc = sklearn.metrics.roc_auc_score(labels, preds, average=None)[-1]
                else:
                    auc = sklearn.metrics.roc_auc_score(labels, preds, average=None)
            else:
                auc = sklearn.metrics.roc_auc_score(labels, preds, average=average)
        except ValueError:
            auc = 0

    return auc

def calc_f1(preds, labels, average='macro'):
    '''
    preds: tensor(b,) idx (or tensor(b, n_cls), logits)
    labels: tensor(b), idx
    '''
    with torch.no_grad():
        # logits 2 idx
        if len(preds.size()) == 2:
            preds = torch.max(preds.detach(), 1)[1]  # (b,) idx

        preds = preds.detach().int().cpu().numpy()
        labels = labels.int().cpu().numpy()
        f1 = sklearn.metrics.f1_score(labels, preds, average=average)

    return f1


def calc_mAP(preds, labels):
    '''
    preds: tensor(b, n_cls)  logits
    labels: tensor(b, n_cls) onehot (or tensor(b,) idx)
    '''
    with torch.no_grad():
        # idx 2 onehot
        if len(labels.size()) == 1:
            labels = idx2onehot(labels, preds.size(1))

        preds = preds.detach().cpu().numpy()
        labels = labels.int().cpu().numpy()
        try:
            mAP = sklearn.metrics.average_precision_score(labels, preds, average='macro')
        except ValueError:
            mAP = 0
    return mAP


def calc_auc_morf(logits):
    '''
    logits: torch.tensor(n, )
    '''
    with torch.no_grad():
        n = logits.size(0)
        res = 0
        for i in range(n-1):
            res += (logits[i] + logits[i+1]) / 2
        res = res / n
    return res.detach().cpu()




if __name__ == '__main__':
    pass