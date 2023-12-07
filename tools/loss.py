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


def kl_divergence(mu, logsigma):
    # print(f'mu shape {mu.shape}, logsig shape {logsigma.shape}')
    return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum()

def margin_entropy_loss(margin, logsigma):
    feat_dim = logsigma.shape[-1]
    entropy = float(feat_dim / 2 * (np.log(2 * np.pi) + 1)) + torch.sum(logsigma, -1) / 2
    zero = torch.zeros_like(entropy)
    loss = torch.max(margin - entropy, zero)
    loss = torch.mean(loss)
    return loss

def L2_contrast_loss():
    pass

def calc_logsig_loss(logsig, thresh):
    return torch.max(0, torch.sum(thresh-torch.sum(logsig, dim=1)))



class CCE(nn.Module):
    def __init__(self, device, balancing_factor=1):
        super(CCE, self).__init__()
        self.nll_loss = nn.NLLLoss()
        self.device = device # {'cpu', 'cuda:0', 'cuda:1', ...}
        self.balancing_factor = balancing_factor

    def forward(self, yHat, y):
        # Note: yHat.shape[1] <=> number of classes
        batch_size = len(y)
        # cross entropy
        cross_entropy = self.nll_loss(F.log_softmax(yHat, dim=1), y)
        # complement entropy
        yHat = F.softmax(yHat, dim=1)
        Yg = yHat.gather(dim=1, index=torch.unsqueeze(y, 1))
        Px = yHat / (1 - Yg) + 1e-7
        Px_log = torch.log(Px + 1e-10)
        y_zerohot = torch.ones(batch_size, yHat.shape[1]).scatter_(
            1, y.view(batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.to(device=self.device)
        complement_entropy = torch.sum(output) / (float(batch_size) * float(yHat.shape[1]))

        return cross_entropy - self.balancing_factor * complement_entropy


class FocalLoss(nn.Module):  # converge faster

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
        probs = torch.sigmoid(logits)  # B, C (H, W)
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


class FocalLoss2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -alpha(1-yi)**gamma *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[alpha, 1-alpha, 1-alpha, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss2,self).__init__()
        self.size_average = size_average
        # if isinstance(alpha,list):
        #     assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
        #     print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
        #     self.alpha = torch.Tensor(alpha)
        # else:
        #     assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
        #     print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
        #     self.alpha = torch.zeros(num_classes)
        #     self.alpha[0] += alpha
        #     self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds, labels, weight=None):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        weight: tensor (C,)
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        # self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        # print(preds.size(), labels.size())
        # logits of the right class
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))  # B, 1
        # self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        # loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLoss3(nn.Module):
    def __init__(self,gamma=2):
        super(FocalLoss3,self).__init__()
        self.gamma=gamma
    def forward(self, preds, labels, weight=None):
        """
		preds: tensor (B, C)
        labels: tensor (B)
        weights: tensor (C,)
		"""
        eps=1e-7
        # idx 2 one hot
        labels = idx2onehot(labels, preds.size(1))
        
        probs = F.softmax(preds, dim=-1)
        log_probs = F.log_softmax(preds, dim=-1)
        ce = -1 * log_probs * labels
        if weight is not None:
            weight = weight.view(1, -1) # 1, C
            ce = weight * ce
        loss = torch.pow((1 - probs), self.gamma) * ce

        return loss.sum()


class WeightedCrossEntropy(nn.Module):
    def __init__(self, reduction='mean') -> None:
        super(WeightedCrossEntropy, self).__init__()
        self.reduction = reduction
    
    def forward(self, preds, labels, weight=None):
        '''
        preds: tensor (B, C)
        labels: tensor (B,)
        weights: tensor (C,)
        '''
        weight = weight.view(1, -1)  # 1, C
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax

        # multiply weights
        if weight is not None:
            preds_logsoft = weight * preds_logsoft

        # select logits of the right class
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))

        loss = -preds_logsoft
        if self.reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    

def calc_orth_loss(protos, orth_type, threshold=1):
    '''
    protos: tensor B n_p proto_dim
    '''
    orth_loss = 0
    b_size = protos.size(0)
    if orth_type == 1:  # only diversity
        _mask = 1 - torch.unsqueeze(torch.eye(protos.size(1)), dim=0).cuda()  # 1 n_p n_p
        # mask = _mask.repeat(b_size, 1)  # B n_p n_p
        product = torch.matmul(protos, protos.permute(0, 2, 1))  # B n_p n_p
        orth_loss = torch.mean(torch.norm(_mask * product, dim=(1, 2)))
    elif orth_type == 2:  # diversity and orthoganality
        _mask = torch.unsqueeze(torch.eye(protos.size(1)), dim=0).cuda()  # 1 n_p n_p
        product = torch.matmul(protos, protos.permute(0, 2, 1))  # B n_p n_p
        orth_loss = torch.mean(torch.norm(product - _mask))
    elif orth_type == 3:
        protos_ = F.normalize(protos, dim=-1)
        l2 = ((protos_.unsqueeze(-2) - protos_.unsqueeze(-1)) ** 2).sum(-1)  # B np np
        neg_dis = math.sqrt(threshold) - l2
        mask = neg_dis>0
        neg_dis *= mask.float()
        neg_dis = torch.triu(neg_dis, diagonal=1)  # upper triangle
        orth_loss = neg_dis.sum(1).sum(1).mean()

    return orth_loss

def calc_orth_loss_fix(protos, orth_type, threshold=1):
    '''
    protos: tensor n_p proto_dim
    '''
    orth_loss = 0
    # print(protos.size())
    protos = protos.reshape(protos.size(0), -1)
    if orth_type == 1:  # only diversity
        _mask = 1 - torch.eye(protos.size(0)).cuda()  # n_p n_p
        product = torch.matmul(protos, protos.permute(1, 0))  # n_p n_p
        orth_loss = torch.norm(_mask * product, dim=(0, 1))
    elif orth_type == 2:  # diversity and orthoganality
        _mask = torch.eye(protos.size(0)).cuda()  # n_p n_p
        product = torch.matmul(protos, protos.permute(1, 0))  # n_p n_p
        orth_loss = torch.norm(product - _mask)
    elif orth_type == 3:
        protos_ = F.normalize(protos, dim=-1)
        l2 = ((protos_.unsqueeze(-2) - protos_.unsqueeze(-1)) ** 2).sum(-1)  # np np
        neg_dis = threshold - l2
        mask = neg_dis>0
        neg_dis *= mask.float()
        neg_dis = torch.triu(neg_dis, diagonal=1)  # upper triangle
        orth_loss = neg_dis.sum(0).sum(0)
    return orth_loss


def tversky(y_pred, y_true):
    '''
    y_pred: tensor(b, n_cls) normed logits
    y_true: tensor(b, n_cls) onehot
    '''
    true_pos = torch.sum(y_pred * y_true, dim=0)
    false_pos = torch.sum(y_pred * (1 - y_true), dim=0)
    false_neg = torch.sum((1 - y_pred) * y_true, dim=0)
    alpha = 0.7
    res = (true_pos + 1.)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + 1.)

    return torch.mean(res)

    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_pred, y_true):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_pred, y_true):
    '''
    y_pred: tensor(b, n_cls) logits
    y_true: tensor(b, n_cls) onehot (or tensor(b,) idx)
    '''
    # idx 2 onehot
    if len(y_true.size()) == 1:
        y_true = idx2onehot(y_true, num_cls=y_pred.size(1))
    
    # norm logits
    y_pred = F.softmax(y_pred, dim=-1)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return torch.pow((1-pt_1), gamma)


def recon_loss(recons, ori_inputs):
    '''
    recons: tensor
    '''
    return F.mse_loss(recons, ori_inputs)

def l1_sparsity_loss(feat):
    '''
    feat: torch.tensor
    '''
    return torch.abs(feat).sum()

def SENN_robustness_loss(x, aggregates, concepts, relevances):
    """Computes Robustness Loss for MNIST data
    
    Formulated by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]
    The loss formulation is specific to the data format
    The concept dimension is always 1 for this project by design

    Parameters
    ----------
    x            : torch.tensor
                 Input as (batch_size x num_features)
    aggregates   : torch.tensor
                 Aggregates from SENN as (batch_size x num_classes)
    concepts     : torch.tensor
                 Concepts from Conceptizer as (batch_size x num_concepts)
    relevances   : torch.tensor
                 Relevances from Parameterizer as (batch_size x num_concepts x num_classes)
   
    Returns
    -------
    robustness_loss  : torch.tensor
        Robustness loss as frobenius norm of (batch_size x num_classes x num_features)
    """
    # concept_dim is always 1
    # concepts = concepts.squeeze(-1)  # b, np
    # aggregates = aggregates.squeeze(-1)  # b, nc

    batch_size = x.size(0)
    num_concepts = concepts.size(1)
    num_classes = aggregates.size(1)

    # Jacobian of aggregates wrt x
    jacobians = []
    for i in range(num_classes):
        grad_tensor = torch.zeros(batch_size, num_classes).to(x.device)
        grad_tensor[:, i] = 1.
        j_yx = torch.autograd.grad(outputs=aggregates, inputs=x, \
                                   grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]  # b (x size)
        # bs x 1 x 28 x 28 -> bs x 784 x 1
        jacobians.append(j_yx.view(batch_size, -1).unsqueeze(-1))
    # bs x num_features x num_classes (bs x 784 x 10)
    J_yx = torch.cat(jacobians, dim=2)

    # Jacobian of concepts wrt x
    jacobians = []
    for i in range(num_concepts):
        grad_tensor = torch.zeros(batch_size, num_concepts).to(x.device)
        grad_tensor[:, i] = 1.
        j_hx = torch.autograd.grad(outputs=concepts, inputs=x, \
                                   grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
        # bs x 1 x 28 x 28 -> bs x 784 x 1
        jacobians.append(j_hx.view(batch_size, -1).unsqueeze(-1))  # 8 64 1
    # bs x num_features x num_concepts
    J_hx = torch.cat(jacobians, dim=2)

    # bs x num_features x num_classes
    # print(x.size(), J_yx.size(), J_hx.size(), relevances.size())  # 8 64 2  8 64 10  8 2 10
    robustness_loss = J_yx - torch.bmm(J_hx, relevances.permute(0, 2, 1))

    return robustness_loss.norm(p='fro')



def calc_balance_loss(weights, threshold=5.):
    '''
    weights: torch.tensor(nc, np)
    '''
    l_b = 0
    l_k = 0
    for c in range(weights.size(0)):
        l_b += torch.abs(torch.sum(weights[c]))
        cur_l_k = threshold - torch.sum(torch.abs(weights[c]))
        l_k += cur_l_k if cur_l_k > 0 else 0

    return l_b + l_k