import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)  # np, np

def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(str, file):
    print(str)
    file.write(str + '\n')

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1  # tblr

def find_high_activation_crop_spatiotemporal(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)  # THW
    mask[activation_map < threshold] = 0
    s, e, t, b, l, r = 0, 0, 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            s = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            e = i+1
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            t = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            b = j+1
            break
    for j in range(mask.shape[2]):
        if np.amax(mask[:, :, j]) > 0.5:
            l = j
            break
    for j in reversed(range(mask.shape[2])):
        if np.amax(mask[:, :, j]) > 0.5:
            r = j+1
            break
    return [s, e, t, b, l, r]

def draw_curves(path, train_curve, test_curve, metric_type='loss', test_every=1):
    plt.close()
    plt.plot(train_curve, color='r', label='train')
    if test_curve is not None:
        plt.plot(test_curve, color='b', label='test')
    plt.xlabel('epoch / '+str(test_every))
    if metric_type == 'acc':
        plt.ylabel('acc')
    else:
        plt.ylabel('loss')
    plt.legend()
    plt.savefig(path)
