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

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

mapping_20 = { 
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 1,
        8: 2,
        9: 0,
        10: 0,
        11: 3,
        12: 4,
        13: 5,
        14: 0,
        15: 0,
        16: 0,
        17: 6,
        18: 0,
        19: 7,
        20: 8,
        21: 9,
        22: 10,
        23: 11,
        24: 12,
        25: 13,
        26: 14,
        27: 15,
        28: 16,
        29: 0,
        30: 0,
        31: 17,
        32: 18,
        33: 19,
        -1: 0
    }


class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors

        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1

        Returns:
            return one hot format labels in shape [batchsize, classes]
        """

        one_hot = torch.zeros(labels.size(0), classes)

        #labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth

        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / (length - 1)

        return one_hot.to(target.device)

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                    .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                    .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                    .format(x.size()))


        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)

        if self.reduction == 'none':
            return loss

        elif self.reduction == 'sum':
            return torch.sum(loss)

        elif self.reduction == 'mean':
            return torch.mean(loss)

        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def seg_context_batch3d(seg, context, seg_class_idx, tgt_size):
    # seg: BTHW
    # context: BCTHW

    # resize
    seg = torch.nn.functional.interpolate(seg, size=(tgt_size[1], tgt_size[0]))
    context = torch.nn.functional.interpolate(context, size=(context.size(2), tgt_size[1], tgt_size[0]))

    # mask
    seg = seg.view(seg.size(0), 1, seg.size(1), seg.size(2), seg.size(3))  # B 1 THW
    seg = torch.nn.functional.one_hot(seg.long(), num_classes=20)  # B 1 THW 20
    context = context.view(context.size(0), context.size(1), context.size(2),context.size(3), context.size(4), 1)  # BCTHW 1

    seg = context * seg # BCTHW 20
    seg = seg[:, :, :, :, :, seg_class_idx]

    return seg

def visualize_featmap3d_simple(featmap, ori_input, mode='mean', channel_weights=None, save_dir='', print_flag=False, log=print):
    '''
    when mode != fix_proto:
        featmap: ndarray T1 H1 W1 C1
        ori_input: ndarray T2 H2 W2 C2
        channel_weights: ndarray C1
    when mode == fix_proto:
        featmap: ndarray T1 H1 W1
        ori_input: ndarray T2 H2 W2 C2
    '''
    assert len(ori_input.shape) == 4, (ori_input.shape)
    if mode in ('mean', 'min', 'max', 'fix_proto'):
        if mode == 'mean':
            featmap = np.mean(featmap, axis=3, keepdims=True)  # T1 H1 W1 1
        elif mode == 'min':
            featmap = np.amin(featmap, axis=3, keepdims=True)  # T1 H1 W1 1
        elif mode == 'max':
            featmap = np.amax(featmap, axis=3, keepdims=True)  # T1 H1 W1 1
        elif mode == 'fix_proto':
            featmap = np.expand_dims(featmap, axis=3)  # T1 H1 W1 1
        featmap = torch.from_numpy(featmap).permute(3, 0, 1, 2).contiguous()  # 1 T1 H1 W1
        featmap = torch.unsqueeze(featmap, 0)  # 1 1 T1 H1 W1
        mask = F.interpolate(featmap, size=(ori_input.shape[0], # T
                                            ori_input.shape[1], # H
                                            ori_input.shape[2]  # W
                                            ), mode='trilinear', align_corners=True)  # 1 1 T2 H2 W2
        mask = torch.squeeze(mask, 0).permute(1, 2, 3, 0).numpy()  # T2 H2 W2 1
        assert mask.shape[:3] == ori_input.shape[:3], (mask.shape, ori_input.shape)
        feat_max = np.amax(mask)
        feat_min = np.amin(mask)
        feat_mean = np.mean(mask)
        mask = mask - np.amin(mask)
        mask = mask / (np.amax(mask) + 1e-8)
        for i in range(ori_input.shape[0]):
            img = ori_input[i]
            heatmap = cv2.applyColorMap(np.uint8(255*mask[i]), cv2.COLORMAP_JET)
            heatmap = 0.3*heatmap + 0.5*img
            cv2.imwrite(os.path.join(save_dir,
                                     'feat_heatmap' + str(i) + '.png'),
                        heatmap)
        return feat_mean, feat_max, feat_min
    elif mode == 'separate':
        T1, H1, W1, C1 = featmap.shape
        featmap = torch.from_numpy(featmap).permute(3, 0, 1, 2).contiguous()  # C T1 H1 W1
        featmap = torch.unsqueeze(featmap, 0)  # 1 C T1 H1 W1
        mask = torch.nn.functional.interpolate(featmap, size=(ori_input.shape[0], # T
                                                              ori_input.shape[1], # H
                                                              ori_input.shape[2]  # W
                                                              ), mode='trilinear')  # 1 C T2 H2 W2
        mask = mask.permute(2, 3, 4, 1, 0).numpy()  # T2 H2 W2 C 1
        feat_max = np.amax(mask)
        feat_min = np.amin(mask)
        feat_mean = np.mean(mask)
        mask = mask - np.amin(mask)
        mask = mask / (np.amax(mask) + 1e-8)
        for i in range(ori_input.shape[0]):
            img = ori_input[i]
            save_dir_t = os.path.join(save_dir, str(i))
            makedir(save_dir_t)
            for c in range(mask.shape[3]):
                heatmap = cv2.applyColorMap(np.uint8(255*mask[i, :, :, c]), cv2.COLORMAP_JET)
                heatmap = 0.3*heatmap + 0.5*img
                cv2.imwrite(os.path.join(save_dir_t,
                                        'feat_heatmap_channel' + str(c) + '.png'),
                            heatmap)
        return feat_mean, feat_max, feat_min
    elif mode == 'weighted':
        T1, H1, W1, C1 = featmap.shape
        channel_weights = np.expand_dims(channel_weights, axis=(0, 1, 2))  # 1 1 1 C1
        featmap_ = np.mean(featmap * channel_weights, axis=3, keepdims=True)  # T1 H1 W1 1
        featmap_ = torch.from_numpy(featmap_).permute(3, 0, 1, 2).contiguous()  # 1 T1 H1 W1
        featmap_ = torch.unsqueeze(featmap_, 0)  # 1 1 T1 H1 W1
        mask = F.interpolate(featmap_, size=(ori_input.shape[0], # T
                                                              ori_input.shape[1], # H
                                                              ori_input.shape[2]  # W
                                                              ), mode='trilinear', align_corners=True)  # 1 1 T2 H2 W2
        mask = torch.squeeze(mask, 0).permute(1, 2, 3, 0).numpy()  # T2 H2 W2 1
        # mask = scipy.ndimage.zoom(featmap, zoom=[ori_input.shape[0] / featmap.shape[0], 
        #                                          ori_input.shape[1] / featmap.shape[1],
        #                                          ori_input.shape[2] / featmap.shape[2],
        #                                          1]
        #                           )  # T2 H2 W2 1
        assert mask.shape[:3] == ori_input.shape[:3], (mask.shape, ori_input.shape)
        feat_max = np.amax(mask)
        feat_min = np.amin(mask)
        feat_mean = np.mean(mask)
        mask = mask - np.amin(mask)
        mask = mask / (np.amax(mask) + 1e-8)
        for i in range(ori_input.shape[0]):
            img = ori_input[i]
            heatmap = cv2.applyColorMap(np.uint8(255*mask[i]), cv2.COLORMAP_JET)
            overlay = 0.3*heatmap + 0.5*img
            cv2.imwrite(os.path.join(save_dir,
                                     'feat_heatmap' + str(i) + '.png'),
                        overlay)
        channel_weights_path = os.path.join(save_dir, 'channel_weights.txt')
        max_idx = np.argmax(channel_weights[0, 0, 0])
        content = [str(max_idx), str(channel_weights[0, 0, 0, max_idx]), str(channel_weights)]
        with open(channel_weights_path, 'w') as f:
            f.writelines(str(content))
        mask_info_path = os.path.join(save_dir, mode + '_mask_info.txt')
        content = ['max', str(feat_max), ' min', str(feat_min), ' mean', feat_mean, ' ori shape', str([T1, H1, W1, C1])]
        with open(mask_info_path, 'w') as f:
            f.writelines(str(content))
        if print_flag and False:
            # log('channel weight' + str(channel_weights))
            max_idx = np.argmax(channel_weights[0, 0, 0])
            log('max channel idx' + str(max_idx))
            log('max channel weight' + str(channel_weights[0, 0, 0, max_idx]))
            log('weights sum' + str(np.sum(channel_weights)))
            log('ori featmap t0' + str(featmap_.shape))
            log(str(featmap_[0, 0, 0]))
            print(feat_mean, feat_max, feat_min)
        return feat_mean, feat_max, feat_min
    else:
        raise NotImplementedError(mode)

def visualize_featmap3d(featmap, ori_input, color_order='BGR', img_norm_mode='torch', save_dir='', print_flag=False):
    '''
    featmap: tensor Cthw
    ori_input: tensor 3THW
    '''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # tgt_size = [1, ori_input.size(1), ori_input.size(2), ori_input.size(3)]

    featmap = torch.mean(featmap, dim=0, keepdim=True).view(1, 1, featmap.size(1), featmap.size(2), featmap.size(3))  # 1 1 thw
    featmap = torch.nn.functional.interpolate(featmap, size=(ori_input.size(1), ori_input.size(2), ori_input.size(3)), mode='bilinear')  # 1 1 thw -> 1 1 THW
    featmap = featmap.view(1, ori_input.size(1), ori_input.size(2), ori_input.size(3)).permute(1, 2, 3, 0).cpu().numpy()  # THW 1
    featmap -= np.amin(featmap)
    featmap /= np.amax(featmap)
    ori_input = ori_input.permute(1, 2, 3, 0).cpu().numpy()  # THW 3
    if color_order == 'RGB':
        ori_input = ori_input[:, :, :, ::-1]
    if img_norm_mode == 'tf':
        ori_input += 1.
        ori_input *= 127.5
    elif img_norm_mode == 'torch':
        ori_input[:,:,:,0] *= 0.225
        ori_input[:,:,:,1] *= 0.224
        ori_input[:,:,:,2] *= 0.229
        ori_input[:,:,:,0] += 0.406
        ori_input[:,:,:,1] += 0.456
        ori_input[:,:,:,2] += 0.485
        ori_input *= 255.
    for i in range(ori_input.shape[0]):
        img = ori_input[i]
        heatmap = cv2.applyColorMap(np.uint8(255*featmap[i]), cv2.COLORMAP_JET)
        img = 0.3 * heatmap + 0.7 * img
        cv2.imwrite(os.path.join(save_dir, str(i)+'.png'), img)

def draw_traj_on_img(img, traj_seq):
    '''
    img: ndarray H W 3
    traj_seq: ndarray T 2
    '''
    seq_len = traj_seq.shape[0]
    for i in range(seq_len - 1):
        img = cv2.line(img, traj_seq[i], traj_seq[i+1], color=(255, 0, 0), thickness=4)

    return img

def draw_boxes_on_img(img, traj_seq):
    '''
    img: ndarray H W 3
    traj_seq: ndarray T 4 (ltrb)
    '''
    seq_len = traj_seq.shape[0]
    for i in range(seq_len-1, 0, -4):
        r = i / seq_len
        # print('traj type:', type(traj_seq))
        img = cv2.rectangle(img=img, pt1=(int(traj_seq[i, 0]), int(traj_seq[i, 1])), pt2=(int(traj_seq[i, 2]), int(traj_seq[i, 3])), color=(0, 0, 255 * r), thickness=4)

    return img

def draw_proto_info_curves(path, info, draw_every=1):
    '''
    info: ndarray T, 3
    '''
    mean_curve = info[:, 0].tolist()
    max_curve = info[:, 1].tolist()
    min_curve = info[:, 2].tolist()
    plt.close()
    plt.plot(mean_curve, color='r', label='mean')
    plt.plot(max_curve, color='b', label='max')
    plt.plot(min_curve, color='black', label='min')
    plt.xlabel('epoch / '+str(draw_every))
    plt.ylabel('value')
    plt.legend()
    plt.savefig(path)

def last_conv_channel(module):
    return [i for i in module if isinstance(i, nn.Conv2d) or isinstance(i, nn.Conv3d)][-1].out_channels

def last_lstm_channel(module):
    return [i for i in module if isinstance(i, nn.LSTM)][-1].hidden_size

def save_model(model, model_dir, model_name, res=0, log=print):
    '''
    model: this is not the multigpu model
    '''
    # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
    torch.save(obj=model, f=os.path.join(model_dir, model_name + f'_{res}.pth'))
    log(f'Model saved in {model_dir}')

def ltrb2xywh(bbox_seq):
    '''
    bbox_seq: ndarray: seq len, 4
    '''
    new_seq = np.zeros(shape=bbox_seq.shape)
    new_seq[:, 0] = (bbox_seq[:, 0] + bbox_seq[:, 2]) / 2
    new_seq[:, 1] = (bbox_seq[:, 1] + bbox_seq[:, 3]) / 2
    new_seq[:, 2] = bbox_seq[:, 2] - bbox_seq[:, 0]
    new_seq[:, 3] = bbox_seq[:, 3] - bbox_seq[:, 1]

    return new_seq

def vid_id_int2str(vid_id_int):
    '''
    vid_id: int
    '''
    nm = "%04d" % vid_id_int
    
    return 'video_' + nm

def img_nm_int2str(img_nm_int, dataset_name='PIE'):
    '''
    img_nm_int: int
    '''
    if dataset_name == 'PIE' or dataset_name == 'JAAD':
        nm = "%05d" % img_nm_int
    elif dataset_name == 'TITAN':
        nm = "%06d" % img_nm_int
    
    return nm + '.png'

def ped_id_int2str(ped_id_int):
    '''
    ped_id_int: ndarray or list (3,)
    '''
    if ped_id_int[-1] >= 0:
        ped_id = str(ped_id_int[0]) + '_' + str(ped_id_int[1]) + '_' + str(ped_id_int[2])
    else:
        ped_id = str(ped_id_int[0]) + '_' + str(ped_id_int[1]) + '_' + str(- ped_id_int[2]) + 'b'
    
    return ped_id

def idx2onehot(idx_batch, num_cls):
    '''
    idx_batch: (b,)
    num_cls: int
    onehot_batch: (b. num_cls)
    '''
    if num_cls > 1:
        onehot_batch = torch.nn.functional.one_hot(idx_batch, num_classes=num_cls)
    else:
        onehot_batch = torch.unsqueeze(idx_batch, dim=1)

    return onehot_batch


def calc_auc(logits, labels, average='macro'):
    '''
    logits: tensor(b, n_cls)
    labels: tensor(b, n_cls)
    '''
    logits = logits.cpu().numpy()
    labels = labels.int().cpu().numpy()
    if average == 'binary':
        if logits.shape[-1] > 1:
            auc = sklearn.metrics.roc_auc_score(labels, logits, average=None)[-1]
        else:
            auc = sklearn.metrics.roc_auc_score(labels, logits, average=None)
    else:
        auc = sklearn.metrics.roc_auc_score(labels, logits, average=average)

    return auc

def calc_f1(preds, labels, average='macro'):
    '''
    preds: tensor(b)
    labels: tensor(b)
    '''
    preds = preds.int().cpu().numpy()
    labels = labels.int().cpu().numpy()
    f1 = sklearn.metrics.f1_score(labels, preds, average=average)

    return f1

def calc_acc(preds, labels):
    '''
    preds: tensor(b, n_cls), logits
    labels: tensor(b), idx
    '''
    # # logits 2 idx
    # preds = torch.max(preds.detach(), 1)[1]

    preds = preds.int().cpu().numpy()
    labels = labels.int().cpu().numpy()
    acc = sklearn.metrics.accuracy_score(labels, preds)

    return acc


def calc_recall(preds, labels):
    '''
    preds: tensor(b), idx
    labels: tensor(b), idx
    '''
    preds = preds.int().cpu().numpy()
    labels = labels.int().cpu().numpy()
    recall = sklearn.metrics.recall_score(labels, preds, average=None)

    return recall

def calc_mAP(logits, labels):
    '''
    logits: tensor(b, n_cls)
    labels: tensor(b, n_cls) one hot
    '''
    logits = logits.cpu().numpy()
    labels = labels.int().cpu().numpy()
    mAP = sklearn.metrics.average_precision_score(labels, logits, average='macro')

    return mAP

def enhance_last_attention(h_seq, att_w, h_dim, out_dim):
    '''
    h_seq: B, T, D
    att_w: linear layer
    h_dim: int D
    '''
    seq_len = h_seq.size(1)
    q = h_seq[:, -1]  # B, D
    att1 = att_w(h_seq)  # B, T, D
    q = q.view(-1, 1, h_dim).contiguous().expand(-1, seq_len, -1)  # B T D
    att2 = torch.matmul(att1.view(-1, 1, h_dim), q.view(-1, h_dim, 1))  # B*T 1 1
    att2 = att2.view(-1, seq_len).contiguous()
    score = nn.functional.softmax(att2, dim=1)

    return score


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def calc_orth_loss(protos, orth_type):
    orth_loss = 0
    if orth_type == 1:
        pass
    return orth_loss


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

def TITANclip_txt2list(path):
    clip_id_list = []
    with open(path, 'r') as f:
        s = f.readlines()
    for s_ in s:
        clip_id_list.append(s_.replace('clip_', '').replace('\n', ''))
    
    return clip_id_list  # list of str


def cls_weights(num_samples_cls, loss_weight='reciprocal', device='cpu'):
    '''
    num_samples_cls: list (n_cls,)
    '''
    n_all = sum(num_samples_cls)
    cls_weight = []
    if loss_weight == 'switch':
        raise ValueError(loss_weight)
    elif loss_weight == 'reciprocal':
        for i in num_samples_cls:
            if i > 0:
                cls_weight.append(n_all / i)
            else:
                cls_weight.append(0.)
    elif loss_weight == 'divide_self':
        for i in num_samples_cls:
            if i > 0:
                cls_weight.append(1. / i)
            else:
                cls_weight.append(0.)
    elif loss_weight == 'sklearn':
        n_classes = len(num_samples_cls)
        for i in num_samples_cls:
            if i > 0:
                cls_weight.append(n_all / (i * n_classes))
            else:
                cls_weight.append(0.)
    elif loss_weight == 'no':
        return None
    else:
        raise NotImplementedError(loss_weight)
    
    if device is not None:
        cls_weight = torch.tensor(cls_weight).float().to(device)
    return cls_weight

def get_cls_weights_multi(model,
                            dataloader,
                            loss_weight,
                            device='cpu',
                            multi_label_cross=0, 
                            use_atomic=0,
                            use_complex=0,
                            use_communicative=0,
                            use_transporting=0,
                            use_age=0,
                            use_cross=1):
    '''
    num_samples_cls: list (n_cls,)
    return:
        multi_weights: dict {tensor: (n_cls,)}
    '''
    weight = None
    atomic_weight = None
    complex_weight = None
    communicative_weight = None
    transporting_weight = None
    age_weight = None

    if loss_weight == 'trainable':
        if use_cross:
            weight = F.softmax(model.module.class_weights, dim=-1)
        if use_atomic:
            atomic_weight = F.softmax(model.module.atomic_weights, dim=-1)
        if use_complex:
            complex_weight = F.softmax(model.module.complex_weights, dim=-1)
        if use_communicative:
            communicative_weight = F.softmax(model.module.communicative_weights, dim=-1)
        if use_transporting:
            transporting_weight = F.softmax(model.module.transporting_weights, dim=-1)
        if use_age:
            age_weight = F.softmax(model.module.age_weights, dim=-1)
    else:
        if use_cross:
            weight = cls_weights([dataloader.dataset.n_nc, dataloader.dataset.n_c], loss_weight, device=device)
            if multi_label_cross:
                num_samples_cls = dataloader.dataset.num_samples_cls
                weight = cls_weights(num_samples_cls=num_samples_cls, loss_weight=loss_weight, device=device)
        if use_atomic:
            atomic_weight = cls_weights(dataloader.dataset.num_samples_atomic, loss_weight, device=device)
        if use_complex:
            complex_weight = cls_weights(dataloader.dataset.num_samples_complex, loss_weight, device=device)
        if use_communicative:
            communicative_weight = cls_weights(dataloader.dataset.num_samples_communicative, loss_weight, device=device)
        if use_transporting:
            transporting_weight = cls_weights(dataloader.dataset.num_samples_transporting, loss_weight, device=device)
        if use_age:
            age_weight = cls_weights(dataloader.dataset.num_samples_age, loss_weight, device=device)
    
    multi_weights = {'final': weight,
                    'atomic': atomic_weight,
                    'complex': complex_weight,
                    'communicative': communicative_weight,
                    'transporting': transporting_weight,
                    'age': age_weight}
    return multi_weights

def calc_n_samples_cls(labels, n_cls):
    '''
    labels: tensor (B, 1)
    n_cls: int
    '''
    labels = np.squeeze(labels.cpu().numpy())
    n_sample_cls = []
    for i in range(n_cls):
        n_cur_cls = sum(labels==i)
        n_sample_cls.append(n_cur_cls)
    
    return n_sample_cls

def cls_weight2sample_weight(cls_weights, labels):
    '''
    cls_weights: list (n_cls,)
    labels: tensor (n_samples, 1)
    return:
        sample_weights: list (n_samples)
    '''
    sample_weights = [cls_weights[int(gt[0])] for gt in labels]
    
    return sample_weights

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
    plt.close()
    plt.plot(train_curve, color='r', label='train')
    plt.plot(test_curve, color='b', label='test')
    plt.xlabel('epoch / '+str(test_every))
    metric_type = path.split('/')[-1].replace('.png', '')
    plt.ylabel(metric_type)
    plt.legend()
    plt.savefig(path)
    plt.close()

def write_info_txt(content_list, dir):
    with open(dir, 'w') as f:
        f.writelines(content_list)
