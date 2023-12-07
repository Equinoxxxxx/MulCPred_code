from os import kill
from pickle import NONE
from turtle import forward
from matplotlib import use
import numpy as np
from numpy import indices
import torch
import torch.nn as nn
import torch.nn.functional as F

from _backbones import create_backbone, BACKBONE_TO_OUTDIM
from _backbones import C3D_backbone
from receptive_field import compute_proto_layer_rf_info_v2
from utils import last_conv_channel, last_lstm_channel, freeze

POOL_DICT = {
    'avg3d': nn.AdaptiveAvgPool3d,
    'max3d': nn.AdaptiveMaxPool3d,
    'avg2d': nn.AdaptiveAvgPool2d,
    'max2d': nn.AdaptiveMaxPool2d,
}


def update_gaussian(mu1, mu2, logsig1, logsig2):
    _eps = 1e-5
    sig1, sig2 = torch.exp(logsig1), torch.exp(logsig2)
    mu = (mu1*sig2 + mu2*sig1) / (sig1 + sig2)
    sig = sig1 * sig2 / (sig1 + sig2 + _eps)
    logsig = torch.log(sig + _eps)
    return mu, logsig

def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


class CPU(nn.Module):
    def __init__(self, 
                #  img_setting,
                #  ctx_setting,
                #  sk_setting,
                #  traj_setting,
                #  ego_setting,
                 m_settings,
                 modalities=[],
                 concept_mode='fix_proto',
                 n_proto=20,
                 proto_dim=128,
                 n_cls=2,
                 contrast_mode='pair_wise',
                 bridge_m='img',
                 ) -> None:
        super().__init__()
        # self.img_setting = img_setting
        # self.sk_setting = sk_setting
        # self.ctx_setting = ctx_setting
        # self.traj_setting = traj_setting
        # self.ego_setting = ego_setting
        
        self.modalities = modalities
        self.concept_mode = concept_mode
        self.m_settings = m_settings
        self.contrast_mode = contrast_mode
        self.bridge_m = bridge_m
        # init contrast scale factor
        self.logit_scale = nn.parameter.Parameter(
            torch.ones([]) * np.log(1 / 0.07))

        model_dict = {}
        for m in self.modalities:
            model_dict[m] = SingleBranch(m_settings[m])
        if 'ctx' in self.modalities and 'seg' in m_settings['ctx']['mode']:
            model_dict['ctx'] = SegBranch(m_settings['ctx'])

        self.model_dict = nn.ModuleDict(model_dict)

        if self.concept_mode == 'fix_proto':
            self.protos = nn.parameter.Parameter(
                torch.rand((n_proto, proto_dim)), 
                requires_grad=True)
            self.classifier = nn.Linear(n_proto, n_cls, bias=False)
        elif self.concept_mode == 'mlp_fuse':
            self.classifier = nn.Sequential(
                nn.Linear(proto_dim*len(self.modalities), 
                          proto_dim, 
                          bias=False),
                nn.BatchNorm1d(proto_dim),
                # nn.ReLU(),
                # nn.Sigmoid(),
                nn.LeakyReLU(),
                # nn.GELU(),
                nn.Linear(proto_dim, n_cls, bias=False)
            )

    def forward(self, x_dict, mode='train', log=print):
        z_dict, mu_dict, logsig_dict, pool_f_dict = self.get_mm_feat(x_dict)

        if mode == 'contrast':
            return z_dict, mu_dict, logsig_dict, pool_f_dict
        elif mode == 'train':
            res, simis, z_dict, mu_dict, logsig_dict, pool_f_dict = self._predict(pool_f_dict, mu_dict, logsig_dict, z_dict)
            return res, simis, z_dict, mu_dict, logsig_dict, pool_f_dict
        else:
            raise NotImplementedError(mode)
    
    def get_mm_feat(self, x_dict, log=print):
        mu_dict = {}
        logsig_dict = {}
        z_dict = {}
        pool_f_dict = {}
        for m in x_dict:
            z_dict[m], mu_dict[m], logsig_dict[m], pool_f_dict[m] = self.model_dict[m](x_dict[m], log)
        
        return z_dict, mu_dict, logsig_dict, pool_f_dict
    
    def _predict(self, pool_f_dict, mu_dict, logsig_dict, z_dict):
        if self.concept_mode == 'mlp_fuse':
            zs = [z_dict[m][0] for m in z_dict]
            # print([z.shape for z in zs])
            zs = torch.cat(zs, dim=1)
            # print(zs.shape)
            res = {}
            res['final'] = self.classifier(zs)
            return res, None, z_dict, mu_dict, logsig_dict, pool_f_dict

        elif self.concept_mode == 'fix_proto':
            mus = [mu_dict[m] for m in mu_dict]
            logsigs = [logsig_dict[m] for m in logsig_dict]
            mu, logsig = mus[0], logsigs[0]
            for i in range(len(mus) - 1):
                mu, logsig = update_gaussian(mu, mus[i+1], logsig, logsigs[i+1])  # mu: b, proj_dim(=proto_dim)
            # cosine similarity
            joint_mu_norm = mu / mu.norm(dim=1, keepdim=True)
            proto_norm = self.protos / self.protos.norm(dim=1, keepdim=True)  # n_p, proto_dim
            simis = joint_mu_norm @ proto_norm.permute(1, 0)  # simis: b, n_proto
            res = {}
            res['final'] = self.classifier(simis)
            return res, simis, z_dict, mu_dict, logsig_dict, pool_f_dict

        else:
            raise NotImplementedError(self.concept_mode)

    def _contrast(self, z_dict):
        eps = 1e-4
        logit_scale = self.logit_scale.exp()
        simi_mats = []
        n_m = len(self.modalities)
        if self.contrast_mode == 'pair_wise':
            for i in range(n_m):
                mi = self.modalities[i]
                for j in range(i+1, n_m):
                    mj = self.modalities[j]
                    for k in range(len(z_dict[mj])):
                        zi = z_dict[mi][k]  # b, proj_dim
                        zj = z_dict[mj][k]  # b, proj_dim
                        # cosine simi
                        zi_norm, zj_norm = zi / zi.norm(dim=1, keepdim=True), zj / zj.norm(dim=1, keepdim=True)
                        simi_mat1 = logit_scale * zi_norm @ zj_norm.t() + eps
                        simi_mat2 = simi_mat1.t()
                        simi_mats += [simi_mat1, simi_mat2]
        elif self.contrast_mode == 'pair_wise_norm_orth':
            for i in range(n_m):
                mi = self.modalities[i]
                for j in range(i+1, n_m):
                    mj = self.modalities[j]
                    for k in range(len(z_dict[mj])):
                        zi = z_dict[mi][k]  # b, proj_dim
                        zj = z_dict[mj][k]  # b, proj_dim
                        # cosine simi
                        zi_norm, zj_norm = \
                            zi/(zi.norm(dim=1, keepdim=True)+eps), zj/(zj.norm(dim=1, keepdim=True)+eps)
                        simi_mat1 = logit_scale * zi_norm @ zj_norm.t() + eps
                        simi_mats += [simi_mat1]
        elif self.contrast_mode == 'pair_wise_orth':
            for i in range(n_m):
                mi = self.modalities[i]
                for j in range(i+1, n_m):
                    mj = self.modalities[j]
                    for k in range(len(z_dict[mj])):
                        zi = z_dict[mi][k]  # b, proj_dim
                        zj = z_dict[mj][k]  # b, proj_dim
                        simi_mat1 = logit_scale * zi @ zj.t() + eps
                        simi_mats += [simi_mat1]
        elif self.contrast_mode == 'proto_pair_wise':
            code_dict = {m:[] for m in self.modalities}
            for m in z_dict:
                for k in range(len(z_dict[m])):
                    code_dict[m].append(z_dict[m][k] @ self.protos.t())  # b, n_p
            for i in range(n_m):
                mi = self.modalities[i]
                for j in range(i+1, n_m):
                    mj = self.modalities[j]
                    for k in range(len(code_dict[mj])):
                        code_i = code_dict[mi][k]  # b, n_p
                        code_j = code_dict[mj][k]  # b, n_p
                        # cosine simi
                        code_i_norm, code_j_norm = code_i / code_i.norm(dim=1, keepdim=True), code_j / code_j.norm(dim=1, keepdim=True)
                        simi_mat1 = logit_scale * code_i_norm @ code_j_norm.t() + eps
                        simi_mat2 = simi_mat1.t()
                        simi_mats += [simi_mat1, simi_mat2]
        elif self.contrast_mode == 'bridge':
            for m in z_dict:
                if m != self.bridge_m:
                    for k in range(len(z_dict[m])):
                        zb = z_dict[self.bridge_m][k]
                        zi = z_dict[m][k]
                        zb_norm, zi_norm = zb / zb.norm(dim=1, keepdim=True), zi / zi.norm(dim=1, keepdim=True)
                        simi_mat1 = logit_scale * zb_norm @ zi_norm.t() + eps
                        simi_mat2 = simi_mat1.t()
                        simi_mats += [simi_mat1, simi_mat2]
        elif self.contrast_mode == 'proto_bridge':
            code_dict = {m:[] for m in self.modalities}
            for m in z_dict:
                for k in range(len(z_dict[m])):
                    code_dict[m].append(z_dict[m][k] @ self.protos.t())  # b, n_p
            for m in code_dict:
                if m != self.bridge_m:
                    for k in range(len(code_dict[m])):
                        code_b = code_dict[self.bridge_m][k]
                        code_i = code_dict[m][k]
                        code_b_norm, code_i_norm = code_b / code_b.norm(dim=1, keepdim=True), code_i / code_i.norm(dim=1, keepdim=True)
                        simi_mat1 = logit_scale * code_b_norm @ code_i_norm.t() + eps
                        simi_mat2 = simi_mat1.t()
                        simi_mats += [simi_mat1, simi_mat2]
        else:
            raise NotImplementedError(self.contrast_mode)
        
        return simi_mats  # list: n_pair*[b, b]


class SingleBranch(nn.Module):
    def __init__(self,
                 setting) -> None:
        '''
        setting: modality, mode, backbone_name, pool, n_layer_proj, bn, proj_dim, uncertainty
        '''
        super().__init__()
        self.modality = setting['modality']
        self.backbone_name = setting['backbone_name']
        self.n_layer_proj = setting['n_layer_proj']
        self.norm = setting['norm']
        self.proj_dim = setting['proj_dim']
        self.uncertainty = setting['uncertainty']
        self.n_sampling = setting['n_sampling']

        lstm_in_dim = 4 if self.modality == 'traj' else 1
        self.backbone = create_backbone(setting['backbone_name'], lstm_input_dim=lstm_in_dim)
        if setting['pool'] != 'none':
            self.pool = POOL_DICT[setting['pool']](1)
        else:
            self.pool = None

        # mu projector
        proj = []
        in_dim = BACKBONE_TO_OUTDIM[self.backbone_name]
        for _ in range(self.n_layer_proj - 1):
            proj.append(nn.Linear(in_dim, in_dim, 
                                #   bias=False
                                  ))
            if self.norm == 'ln':
                proj.append(nn.LayerNorm(in_dim))
            elif self.norm == 'bn':
                proj.append(nn.BatchNorm1d(in_dim))
            # proj.append(nn.ReLU())
            # proj.append(nn.Sigmoid())
            proj.append(nn.LeakyReLU())
            # proj.append(nn.GELU())
        proj.append(nn.Linear(in_dim, self.proj_dim, 
                            #   bias=False
                              ))
        # if self.norm == 'ln':
        #     proj.append(nn.LayerNorm(self.proj_dim))
        # elif self.norm == 'bn':
        #     proj.append(nn.BatchNorm1d(self.proj_dim))
        self.proj = nn.Sequential(*proj)

        # log sigma projector
        if self.uncertainty == 'gaussian':
            logsig_proj = []
            for _ in range(self.n_layer_proj - 1):
                logsig_proj.append(nn.Linear(in_dim, in_dim, bias=False))
                if self.norm == 'ln':
                    logsig_proj.append(nn.LayerNorm(in_dim))
                elif self.norm == 'bn':
                    logsig_proj.append(nn.BatchNorm1d(in_dim))
                # logsig_proj.append(nn.ReLU())
                # logsig_proj.append(nn.Sigmoid())
                logsig_proj.append(nn.LeakyReLU())
                # logsig_proj.append(nn.GELU())
            logsig_proj.append(nn.Linear(in_dim, self.proj_dim, bias=False))
            if self.norm == 'ln':
                logsig_proj.append(nn.LayerNorm(self.proj_dim))
            elif self.norm == 'bn':
                logsig_proj.append(nn.BatchNorm1d(self.proj_dim))
            self.logsig_proj = nn.Sequential(*logsig_proj)
        
        # init params
        self._init_params()

    def forward(self, x, log=print):
        # ego motion只取acceleration
        if self.modality == 'ego' and x.size(-1) == 2:
            x = x[:, :, 0].reshape(x.size(0), x.size(1), 1)
        x = self.backbone(x)
        b, d = x.size(0), x.size(1)
        if self.pool is not None:
            x = self.pool(x).reshape(b, d)  # b, c
        mu = self.proj(x)
        # calc sigma
        zs = [mu]
        logsig = mu
        if self.uncertainty == 'gaussian':
            zs = []
            logsig = self.logsig_proj(x)
            # sample from gaussian
            for i in range(self.n_sampling):
                eps = torch.randn(mu.shape[0], mu.shape[1], device=mu.device)
                eps = torch.clamp(eps, min=-10., max=10.)
                zs.append(mu + eps*torch.exp(logsig))
            # for z in zs:
            #     log(f'z diff {z - zs[0]}')
                
        return zs, mu, logsig, x
    
    def _init_params(self):
        for m in self.proj.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.uncertainty == 'gaussian':
            for m in self.logsig_proj.modules():
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        if 'clean' in self.backbone_name:
            for m in self.backbone.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


class SegBranch(nn.Module):
    def __init__(self,
                 setting) -> None:
        '''
        setting: modality, mode, seg_cls, fuse_mode, backbone_name, pool, n_layer_proj, bn, proj_dim, uncertainty
        '''
        super().__init__()
        self.backbone_name = setting['backbone_name']
        self.mode = setting['mode']
        self.seg_cls = setting['seg_cls']
        self.fuse_mode = setting['fuse_mode']
        self.n_layer_proj = setting['n_layer_proj']
        self.norm = setting['norm']
        self.proj_dim = setting['proj_dim']
        self.uncertainty = setting['uncertainty']
        self.n_sampling = setting['n_sampling']

        # create backbones
        self.backbone = create_backbone(self.backbone_name)
        # pooling layer
        if setting['pool'] != 'none':
            self.pool = POOL_DICT[setting['pool']](1)
        else:
            self.pool = None
        
        in_dim = BACKBONE_TO_OUTDIM[self.backbone_name]
        # fusing layers
        if self.fuse_mode == 'cat':
            self.fuse_layer = nn.Linear(in_dim * len(self.seg_cls), in_dim, bias=False)
        elif self.fuse_mode == 'transformer':
            self.cls_token = nn.parameter.Parameter(torch.randn(1, 1, in_dim))
            tf = nn.TransformerEncoderLayer(in_dim, 4, self.proj_dim, batch_first=True)
            self.fuse_layer = nn.TransformerEncoder(tf, num_layers=2)
        # mu projector
        proj = []
        for _ in range(self.n_layer_proj - 1):
            proj.append(nn.Linear(in_dim, in_dim, bias=False))
            if self.norm == 'ln':
                proj.append(nn.LayerNorm(in_dim))
            elif self.norm == 'bn':
                proj.append(nn.BatchNorm1d(in_dim))
            # proj.append(nn.ReLU())
            proj.append(nn.LeakyReLU())
            # proj.append(nn.GELU())
        proj.append(nn.Linear(in_dim, self.proj_dim), bias=False)
        # if self.norm == 'ln':
        #     proj.append(nn.LayerNorm(self.proj_dim))
        # elif self.norm == 'bn':
        #     proj.append(nn.BatchNorm1d(self.proj_dim))
        self.proj = nn.Sequential(*proj)
        # log sigma projector
        if self.uncertainty == 'gaussian':
            logsig_proj = []
            for _ in range(self.n_layer_proj - 1):
                logsig_proj.append(nn.Linear(in_dim, in_dim, bias=False))
                if self.norm == 'ln':
                    logsig_proj.append(nn.LayerNorm(in_dim))
                elif self.norm == 'bn':
                    logsig_proj.append(nn.BatchNorm1d(in_dim))
                # logsig_proj.append(nn.ReLU())
                logsig_proj.append(nn.LeakyReLU())
                # logsig_proj.append(nn.GELU())
            logsig_proj.append(nn.Linear(in_dim, self.proj_dim), bias=False)
            if self.norm == 'ln':
                logsig_proj.append(nn.LayerNorm(self.proj_dim))
            elif self.norm == 'bn':
                logsig_proj.append(nn.BatchNorm1d(self.proj_dim))
            self.logsig_proj = nn.Sequential(*logsig_proj)
        
        # init params
        self._init_params()
    
    def forward(self, x, log=print):
        '''
        x: tensor b3Thw n_cls
        '''
        x = x.permute(5, 0, 1, 2, 3, 4)
        nc, b, c, t, h, w = x.size()
        x = x.reshape(nc*b, c, t, h, w)
        feats = self.backbone(x)
        if self.pool is not None:
            feats = self.pool(feats)
        feats = feats.reshape(nc, b, feats.size(1)).permute(1, 0, 2)  # b nc c
        if self.fuse_mode == 'cat':
            feats = self.fuse_layer(feats)  # b c
        elif self.fuse_mode == 'transformer':
            cls_tokens = self.cls_token.repeat(b, 1, 1)
            feats = torch.cat([cls_tokens, feats], dim=1)  # b nc+1 c
            feats = self.fuse_layer(feats)[:, 0]  # b c
        mu = self.proj(feats)
        # calc sigma
        zs = [mu]
        logsig = mu
        if self.uncertainty == 'gaussian':
            logsig = self.logsig_proj(feats)
            # sample from gaussian
            for i in range(self.n_sampling):
                eps = torch.randn(mu.shape[0], mu.shape[1], device=mu.device)
                eps = torch.clamp(eps, min=-10., max=10.)
                zs.append(mu + eps*torch.exp(logsig))

        return zs, mu, logsig, None
    
    def _init_params(self):
        for m in self.fuse_layer.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.proj.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.uncertainty == 'gaussian':
            for m in self.logsig_proj.modules():
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        if 'clean' in self.backbone_name:
            for m in self.backbone.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)



class InteractionBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()