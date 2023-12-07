from os import kill
from pickle import NONE
from turtle import forward
from matplotlib import use
from numpy import indices
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from _backbones import create_backbone
from _backbones import C3D_backbone
from receptive_field import compute_proto_layer_rf_info_v2
from utils import last_conv_channel, last_lstm_channel, freeze

from tools.datasets.TITAN import NUM_CLS_ATOMIC, NUM_CLS_COMPLEX, NUM_CLS_COMMUNICATIVE, NUM_CLS_TRANSPORTING, NUM_CLS_AGE

class MultiSLE(nn.Module):
    def __init__(self,
                 use_img, img_setting,
                 use_skeleton, sk_setitng,
                 use_context, ctx_setting,
                 use_traj, traj_setting,
                 use_ego, ego_setting,
                 fusion_mode=0,
                 pred_traj=0, pred_len=16,
                 num_classes=2,
                 use_atomic=0, 
                 use_complex=0, 
                 use_communicative=0, 
                 use_transporting=0, 
                 use_age=0,
                 trainable_weights=0,
                 m_task_weights=0,
                 init_class_weights=None
                 ) -> None:
        super(MultiSLE, self).__init__()
        self.use_img = use_img
        self.use_skeleton = use_skeleton
        self.use_context = use_context
        self.use_traj = use_traj
        self.use_ego = use_ego

        self.img_setting = img_setting
        self.sk_setting = sk_setitng
        self.ctx_setting = ctx_setting
        self.traj_setting = traj_setting
        self.ego_setting = ego_setting
        self.fusion_mode = fusion_mode
        self.pred_traj = pred_traj
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.use_atomic = use_atomic
        self.use_complex = use_complex
        self.use_communicative = use_communicative
        self.use_transporting = use_transporting
        self.use_age = use_age
        self.trainable_weights = trainable_weights
        self.m_task_weights = m_task_weights
        self.init_class_weights = init_class_weights

        self.total_num_proto = 0

        # init contrast scale factor
        self.logit_scale = nn.parameter.Parameter(
            torch.ones([]) * np.log(1 / 0.07))
        # create encoder
        if self.use_traj:
            print('traj model setting', self.traj_setting)
            self.traj_model = SLEseq(**self.traj_setting)
            self.total_num_proto += self.traj_model.num_proto
            self.proto_dim = self.traj_model.proto_dim
        if self.use_ego:
            print('ego model setting', self.ego_setting)
            self.ego_model = SLEseq(**self.ego_setting)
            self.total_num_proto += self.ego_model.num_proto
            self.proto_dim = self.ego_model.proto_dim
        if self.use_img:
            print('img model setting', self.img_setting)
            self.img_model = SLE3D(**self.img_setting)
            self.total_num_proto += self.img_model.num_proto
            self.proto_dim = self.img_model.proto_dim
        if self.use_skeleton:
            print('sk model setting', self.sk_setting)
            if self.sk_setting['simi_func'] in ('ego_gen_channel_att+linear', 
                                                'traj_gen_channel_att+linear', 
                                                'channel_att+linear', 
                                                'channel_att+mlp'):
                self.sk_setting['simi_func'] = 'channel_att+linear'
            if self.sk_setting['sk_mode'] == 'heatmap' \
                or self.sk_setting['sk_mode'] == 'pseudo_heatmap' \
                    or self.sk_setting['sk_mode'] == 'img+heatmap':
                self.sk_model = SLE3D(**self.sk_setting)
                self.total_num_proto += self.sk_model.num_proto
                self.proto_dim = self.sk_model.proto_dim
            else:
                raise NotImplementedError(self.sk_setting['sk_mode'])
        if self.use_context:
            print('ctx model setting', self.ctx_setting)
            if self.ctx_setting['ctx_mode'] in ('mask_ped', 
                                                'ori', 
                                                'local', 
                                                'ori_local'):
                self.ctx_model = SLE3D(**self.ctx_setting)
                self.total_num_proto += self.ctx_model.num_proto
                self.proto_dim = self.ctx_model.proto_dim
            elif 'seg_multi' in self.ctx_setting['ctx_mode']:
                models = []
                for i in self.ctx_setting['seg_cls_idx']:
                    models.append(SLE3D(**self.ctx_setting))
                    self.total_num_proto += self.ctx_setting['num_proto']
                self.ctx_model = torch.nn.ModuleList(models)
            else:
                raise NotImplementedError(self.ctx_setting['ctx_mode'])
        
        # create last layer
        if self.fusion_mode == 0:
            self.last_layer = nn.Sequential(nn.Linear(self.total_num_proto, 
                                                      self.total_num_proto),
                                            nn.ReLU(),
                                            nn.Linear(self.total_num_proto, 
                                                      self.num_classes))
        elif self.fusion_mode == 1:
            last_in_dim = self.total_num_proto
            if self.use_atomic:
                self.atomic_layer = nn.Linear(self.total_num_proto, 
                                              NUM_CLS_ATOMIC, 
                                              bias=False)
                if self.use_atomic == 2:
                    last_in_dim += NUM_CLS_ATOMIC
            if self.use_complex:
                self.complex_layer = nn.Linear(self.total_num_proto, 
                                               NUM_CLS_COMPLEX, 
                                               bias=False)
                if self.use_complex == 2:
                    last_in_dim += NUM_CLS_COMPLEX
            if self.use_communicative:
                self.communicative_layer = nn.Linear(self.total_num_proto, 
                                                     NUM_CLS_COMMUNICATIVE, 
                                                     bias=False)
                if self.use_communicative == 2:
                    last_in_dim += NUM_CLS_COMMUNICATIVE
            if self.use_transporting:
                self.transporting_layer = nn.Linear(self.total_num_proto, 
                                                    NUM_CLS_TRANSPORTING, 
                                                    bias=False)
                if self.use_transporting == 2:
                    last_in_dim += NUM_CLS_TRANSPORTING
            if self.use_age:
                self.age_layer = nn.Linear(self.total_num_proto, NUM_CLS_AGE)
                if self.use_age == 2:
                    last_in_dim += NUM_CLS_AGE

            self.last_layer = nn.Linear(last_in_dim, self.num_classes)
        else:
            raise NotImplementedError('fusion mode', self.fusion_mode)
            
        # create traj decoder
        if self.pred_traj:
            self.h_embedder = nn.Linear(self.total_num_proto, 
                                        self.traj_setting['proto_dim'])
            self.decoder = nn.LSTM(self.traj_setting['in_dim'], 
                                   self.traj_setting['proto_dim'], 
                                   batch_first=True)
            self.decoder_fc = nn.Linear(self.traj_setting['proto_dim'], 
                                        self.traj_setting['in_dim'])

        # create class weights
        if self.trainable_weights:
            self.class_weights = nn.Parameter(torch.tensor(self.init_class_weights['cross']), 
                                              requires_grad=True)
            if use_atomic:
                self.atomic_weights = nn.Parameter(torch.tensor(self.init_class_weights['atomic']), requires_grad=True)
            if use_complex:
                self.complex_weights = nn.Parameter(torch.tensor(self.init_class_weights['complex']), requires_grad=True)
            if use_communicative:
                self.communicative_weights = nn.Parameter(torch.tensor(self.init_class_weights['communicative']), requires_grad=True)
            if use_transporting:
                self.transporting_weights = nn.Parameter(torch.tensor(self.init_class_weights['transporting']), requires_grad=True)
        
        # create task weights
        self.logs2 = nn.Parameter(torch.rand(()), requires_grad=True)
        if self.use_atomic:
            self.atomic_logs2 = nn.Parameter(torch.rand(()), requires_grad=True)
        if self.use_complex:
            self.complex_logs2 = nn.Parameter(torch.rand(()), requires_grad=True)
        if self.use_communicative:
            self.communicative_logs2 = nn.Parameter(torch.rand(()), requires_grad=True)
        if self.use_transporting:
            self.transporting_logs2 = nn.Parameter(torch.rand(()), requires_grad=True)
        if self.use_age:
            self.age_logs2 = nn.Parameter(torch.rand(()), requires_grad=True)

    def decode(self, x, h, c):
        '''
        x: B, 1, in_dim
        h, c: 1, B, h_dim
        '''
        preds = []
        for i in range(self.pred_len):
            x, (h, c) = self.decoder(x, (h, c))
            x = self.decoder_fc(x)
            # h = torch.unsqueeze(h, dim=0)
            # c = torch.unsqueeze(c, dim=0)
            preds.append(x)
        
        preds = torch.concat(preds, dim=1)

        return preds
    
    def forward(self, x, mask=None):
        '''
        x: dict{
            'img': CTHW,
            'skeleton': CTHW,
            'context': CTHW,
            'traj': CT,
        }
        mask: tensor(total_n_p,) or None
        '''
        weighted_simis = []
        feats = {}
        relevances = []
        protos = {}
        if self.use_traj:
            if self.traj_model.simi_func in ('ego_gen_channel_att+linear', 
                                             'traj_gen_channel_att+linear', 
                                             'channel_att+linear', 
                                             'channel_att+mlp', 
                                             'dot', 
                                             'protos_only'):
                traj_weighted_simi, traj_feat, traj_relevances, traj_protos = \
                    self.traj_model(x['traj'])
                weighted_simis.append(traj_weighted_simi)
                feats['traj'] = traj_feat
                relevances.append(traj_relevances)
                protos['traj'] = traj_protos
            elif self.traj_model.simi_func in ('fix_proto1', 'fix_proto2'):
                traj_weighted_simi = self.traj_model(x['traj'])
                weighted_simis.append(traj_weighted_simi)
        if self.use_ego:
            if self.ego_model.simi_func in ('ego_gen_channel_att+linear', 
                                            'traj_gen_channel_att+linear', 
                                            'channel_att+linear', 
                                            'channel_att+mlp', 
                                            'dot', 
                                            'protos_only'):
                ego_weighted_simi, ego_feat, ego_relevances, ego_protos = \
                    self.ego_model(x['ego'])
                weighted_simis.append(ego_weighted_simi)
                feats['ego'] = ego_feat
                relevances.append(ego_relevances)
                protos['ego'] = ego_protos
            elif self.ego_model.simi_func in ('fix_proto1', 'fix_proto2'):
                ego_weighted_simi = self.ego_model(x['ego'])
                weighted_simis.append(ego_weighted_simi)
        if self.use_img:
            if self.img_model.simi_func in ('ego_gen_channel_att+linear', 
                                            'traj_gen_channel_att+linear', 
                                            'channel_att+linear', 
                                            'channel_att+mlp', 
                                            'dot', 
                                            'protos_only'):
                extra_prior = None
                if self.img_model.simi_func == 'ego_gen_channel_att+linear':
                    extra_prior = ego_feat
                elif self.img_model.simi_func == 'traj_gen_channel_att+linear':
                    extra_prior = traj_feat
                img_weighted_simi, img_feat, img_relevances, img_protos = \
                    self.img_model(x['img'], extra_prior=extra_prior)
                weighted_simis.append(img_weighted_simi)
                feats['img'] = img_feat
                relevances.append(img_relevances)
                protos['img'] = img_protos
            elif self.img_model.simi_func in ('fix_proto1', 'fix_proto2'):
                scores, simi_map, att_map = self.img_model(x['img'])
                weighted_simis.append(scores)
                protos['img'] = simi_map
        if self.use_skeleton:
            if self.sk_model.simi_func in ('ego_gen_channel_att+linear', 
                                           'traj_gen_channel_att+linear', 
                                           'channel_att+linear', 
                                           'channel_att+mlp', 
                                           'dot', 
                                           'protos_only'):
                sk_weighted_simi, sk_feat, sk_relevances, sk_protos = \
                    self.sk_model(x['skeleton'])
                weighted_simis.append(sk_weighted_simi)
                feats['skeleton'] = sk_feat
                relevances.append(sk_relevances)
                protos['skeleton'] = sk_protos
            elif self.sk_model.simi_func in ('fix_proto1', 'fix_proto2'):
                scores, simi_map, att_map = self.sk_model(x['skeleton'])
                weighted_simis.append(scores)
                protos['skeleton'] = simi_map
        if self.use_context:
            if self.ctx_model.simi_func in ('ego_gen_channel_att+linear', 
                                            'traj_gen_channel_att+linear', 
                                            'channel_att+linear', 
                                            'channel_att+mlp', 
                                            'dot', 
                                            'protos_only'):
                extra_prior = None
                if self.ctx_model.simi_func == 'ego_gen_channel_att+linear':
                    extra_prior = ego_feat
                elif self.ctx_model.simi_func == 'traj_gen_channel_att+linear':
                    extra_prior = traj_feat
                if 'seg_multi' in self.ctx_setting['ctx_mode']:
                    protos['context'] = []
                    for i in range(len(self.ctx_model)):
                        m = self.ctx_model[i]
                        ctx_weighted_simi, ctx_feat, ctx_relevances, ctx_protos = \
                            m(x['context'][:, :, :, :, :, i], 
                              extra_prior=extra_prior)
                        weighted_simis.append(ctx_weighted_simi)
                        relevances.append(ctx_relevances)
                        protos['context'].append(ctx_protos)
                else:
                    ctx_weighted_simi, ctx_feat, ctx_relevances, ctx_protos = \
                        self.ctx_model(x['context'], extra_prior=extra_prior)
                    weighted_simis.append(ctx_weighted_simi)
                    feats['context'] = ctx_feat
                    relevances.append(ctx_relevances)
                    protos['context'] = ctx_protos
            elif self.ctx_model.simi_func in ('fix_proto1', 'fix_proto2'):
                scores, simi_map, att_map = self.ctx_model(x['context'])
                weighted_simis.append(scores)
                protos['context'] = simi_map
        weighted_simis = torch.concat(weighted_simis, dim=1)  # b, total_n_p

        # mask certain protos
        if mask is not None:
            weighted_simis *= mask.unsqueeze(0)

        if self.fusion_mode == 0 or self.fusion_mode == 1:
            _logits = [weighted_simis]
            logits = {}
            if self.use_atomic:
                atomic_logits = self.atomic_layer(weighted_simis)
                logits['atomic'] = atomic_logits
                if self.use_atomic == 2:
                    _logits.append(atomic_logits)
            if self.use_complex:
                complex_logits = self.complex_layer(weighted_simis)
                logits['complex'] = complex_logits
                if self.use_complex == 2:
                    _logits.append(complex_logits)
            if self.use_communicative:
                communicative_logits = self.communicative_layer(weighted_simis)
                logits['communicative'] = communicative_logits
                if self.use_communicative == 2:
                    _logits.append(communicative_logits)
            if self.use_transporting:
                transporting_logits = self.transporting_layer(weighted_simis)
                logits['transporting'] = transporting_logits
                if self.use_transporting == 2:
                    _logits.append(transporting_logits)
            if self.use_age:
                age_logits = self.age_layer(weighted_simis)
                logits['age'] = age_logits
                if self.use_age == 2:
                    _logits.append(age_logits)
            final_logits = self.last_layer(torch.concat(_logits, dim=1))
            logits['final'] = final_logits
        else:
            raise NotImplementedError('fusion mode', self.fusion_mode)
        
        if self.pred_traj:
            loc = torch.unsqueeze(x['traj'][:, -1], dim=1)
            _h = self.h_embedder(weighted_simis)
            h0 = torch.unsqueeze(_h, 0)  # 1, B, total_num_proto
            c0 = torch.unsqueeze(_h, 0)
            pred_traj = self.decode(loc, h0, c0)
            return logits, protos, pred_traj, feats
        else:
            # print(logits.keys(), self.use_atomic)
            return logits, protos, feats
        

class SLE3D(nn.Module):  # sample level explain
    '''
    input: image patch sequence (CTHW)

    '''
    def __init__(self,
                 backbone_name,
                 separate_backbone=1,
                 conditioned_proto=1,
                 proto_generator_name=None,
                 num_explain=5,
                 conditioned_relevance=1,
                 relevance_generator_name=None,
                 num_proto=10,
                 proto_dim=512,
                 simi_func='dot',
                 freeze_base=False,
                 freeze_proto=False,
                 freeze_relev=False,
                 class_specific=False,
                 temperature=1,
                 proto_activate='softmax',
                 backbone_add_on=0,
                 score_sum_linear=1,
                 **model_opts):
        super(SLE3D, self).__init__()
        self.backbone_name = backbone_name
        self.separate_backbone = separate_backbone
        self.proto_generator_name = proto_generator_name
        self.conditioned_proto = conditioned_proto
        self.num_explain = num_explain
        self.conditioned_relevance = conditioned_relevance
        self.relevance_generator_name = relevance_generator_name
        self.proto_dim = proto_dim
        self.num_proto = num_proto
        # self.proto_mode = proto_mode
        self.simi_func = simi_func
        self.freeze_base = freeze_base
        self.freeze_proto = freeze_proto
        self.freeze_relev = freeze_relev
        self.class_specific = class_specific
        self.temperature = temperature
        self.proto_activate = proto_activate
        self.backbone_add_on = backbone_add_on
        self.score_sum_linear = score_sum_linear

        self.cos_func = nn.CosineSimilarity(dim=1)
        self.epsilon = 1e-4

        # create backbone
        self.backbone = create_backbone(self.backbone_name)
        self.global_pool1 = nn.AdaptiveMaxPool3d((1, 1, 1))
        feat_last_channel = last_conv_channel(self.backbone.modules())
        self.add_on_layer = nn.BatchNorm1d(feat_last_channel)
        if self.backbone_add_on == 1:
            self.add_on_layer = nn.Sequential(
                nn.Linear(feat_last_channel, feat_last_channel//2),
                nn.ReLU(),
                nn.Linear(feat_last_channel//2, feat_last_channel),
                nn.BatchNorm1d(feat_last_channel)
            )

        if self.freeze_base:
            freeze(self.backbone)
        if self.simi_func in ('ego_gen_channel_att+linear', 
                              'traj_gen_channel_att+linear', 
                              'channel_att+linear', 
                              'channel_att+mlp'):
            # self.proto_fc = nn.Linear(feat_last_channel + proto_dim, proto_dim * self.num_proto)
            self.flatten1 = nn.Flatten()
            if self.simi_func == 'channel_att+linear':
                self.channel_att_fc = \
                    nn.Linear(feat_last_channel, 
                                feat_last_channel * self.num_proto, 
                                bias=False)
            elif self.simi_func == 'channel_att+mlp':
                self.channel_att_fc = nn.Sequential(nn.Linear(feat_last_channel, 
                                                              feat_last_channel),
                                                    nn.ReLU(),
                                                    nn.Linear(feat_last_channel, 
                                                              feat_last_channel * self.num_proto),
                                                    nn.ReLU(),
                                                    nn.Linear(feat_last_channel * self.num_proto, 
                                                              feat_last_channel * self.num_proto),
                                                    nn.Sigmoid()
                                                    )
            else:
                self.channel_att_fc = nn.Linear(feat_last_channel + proto_dim, 
                                                feat_last_channel * self.num_proto, 
                                                bias=False)
            self.proto_layernorm = nn.LayerNorm(normalized_shape=feat_last_channel)
            self.sum_linear = nn.Linear(feat_last_channel, 1, bias=False)

        elif self.simi_func == 'fixed_channel_att+linear':
            self.protos = nn.Parameter(torch.rand(size=(self.num_proto, feat_last_channel)))

        elif self.simi_func in ('dot', 'protos_only'):
            self.feat_fc = nn.Linear(feat_last_channel, proto_dim, bias=False)
            self.feat_generator = nn.Sequential(self.backbone,
                                                self.global_pool1,
                                                nn.Flatten(),
                                                self.feat_fc)
            if self.separate_backbone:
                if self.conditioned_proto:
                    self.proto_backbone = create_backbone(self.proto_generator_name)
                    self.global_pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
                    self.proto_fc = nn.Linear(feat_last_channel, self.proto_dim * self.num_proto, bias=False)
                    self.proto_generator = nn.Sequential(self.proto_backbone,
                                                        self.global_pool2,
                                                        nn.Flatten(),
                                                        self.proto_fc)
                    if self.freeze_proto:
                        freeze(self.proto_backbone)
                if self.conditioned_relevance:
                    self.relevance_backbone = create_backbone(self.relevance_generator_name)
                    self.global_pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))
                    self.relevance_fc = nn.Linear(feat_last_channel, self.num_proto, bias=False)
                    self.relevance_generator = nn.Sequential(self.relevance_backbone,
                                                            self.global_pool3,
                                                            nn.Flatten(),
                                                            self.relevance_fc)
                    if self.freeze_relev:
                        freeze(self.relevance_backbone)
            else:
                raise NotImplementedError('separate backbone', self.separate_backbone)
        
        elif self.simi_func == 'fix_proto1':
            self.prototype_shape = [self.num_proto, feat_last_channel, 1, 1, 1]
            self.proto_vec = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
            self.global_max_pool1 = nn.AdaptiveMaxPool3d((1, 1, 1))
        elif self.simi_func == 'fix_proto2':
            self.prototype_shape = [self.num_proto, feat_last_channel, 1, 1, 1]
            self.proto_vec = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
            self.spatial_att = nn.Conv3d(feat_last_channel, self.num_proto, 1)
            self.global_max_pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            raise NotImplementedError(self.simi_func)

    def forward(self, x, extra_prior=None):
        if self.simi_func in ('ego_gen_channel_att+linear', 'traj_gen_channel_att+linear', 'channel_att+linear', 'channel_att+mlp'):
            feat = self.backbone(x)
            feat = self.flatten1(self.global_pool1(feat) ) # B C1
            feat = self.add_on_layer(feat)
            c1 = feat.size(-1)
            # print(c1)
            if self.simi_func in ('ego_gen_channel_att+linear', 'traj_gen_channel_att+linear'):
                feat_prior = torch.concat([feat, extra_prior], dim=1) # B C1 + C2
            else:
                feat_prior = feat
            channel_weights = self.channel_att_fc(feat_prior)  # B n_proto * C1
            channel_weights = channel_weights.reshape(-1, self.num_proto, c1)
            
            
            if self.proto_activate == 'softmax':
                # channel_weights = self.proto_layernorm(channel_weights)
                channel_weights = channel_weights / self.temperature
                channel_distr = F.softmax(channel_weights, dim=-1)  # B n_proto C
                channel_weights = F.normalize(channel_weights, dim=-1)
            elif self.proto_activate == 'sigmoid':
                channel_weights = self.proto_layernorm(channel_weights)
                channel_distr = torch.sigmoid(channel_weights)  # B n_proto C
            elif self.proto_activate == 'norm':
                channel_distr = F.normalize(channel_weights, dim=-1)  # B n_proto C  unit vector
            elif self.proto_activate == 'avg':
                channel_weights = torch.ones(size=channel_weights.size()).cuda()
                channel_distr = torch.ones(size=channel_weights.size()).cuda()
            else:
                raise NotImplementedError()
            feat = feat.reshape(-1, 1, c1)  # B 1 C
            scores = feat * channel_distr  # B n_proto C
            if not hasattr(self, 'score_sum_linear') or self.score_sum_linear:
                scores = F.relu(torch.squeeze(self.sum_linear(scores), -1))  # B n_proto
            else:
                scores = F.relu(torch.sum(scores, dim=-1, keepdim=False))  # B n_proto
            
            return scores, torch.squeeze(feat, 1), channel_weights, channel_distr
        elif self.simi_func in ('protos_only', 'dot'):
            if self.simi_func != 'protos_only':
                feat = self.feat_generator(x)
            else:
                feat = None
            protos = self.proto_generator(x)
            relevances = self.relevance_generator(x)

            protos = protos.view(-1, self.num_proto, self.proto_dim)

            if self.simi_func == 'dot':
                simis = torch.squeeze(torch.matmul(protos, torch.unsqueeze(feat, -1)), -1)  # B, np
            elif self.simi_func == 'protos_only':
                simis = torch.mean(protos, dim=-1)
            else:
                raise NotImplementedError('simi func', self.simi_func)

            x = F.relu(relevances * simis)

            return x, feat, relevances, protos
        elif self.simi_func == 'fix_proto1':
            feat = self.backbone(x)  # B C T H W
            if self.proto_activate == 'norm':
                simi_map = F.relu(F.conv3d(input=feat, 
                                        weight=F.normalize(self.proto_vec, dim=1)))  # B np T H W
            scores = self.global_max_pool1(simi_map).view(-1, self.num_proto)

            return scores, simi_map, None
        elif self.simi_func == 'fix_proto2':
            feat = self.backbone(x)  # B C T H W
            if self.proto_activate == 'norm':
                simi_map = F.relu(F.conv3d(input=feat, 
                                        weight=F.normalize(self.proto_vec, dim=1)))  # B np T H W

            sp_att_map = self.spatial_att(feat)  # B np T H W
            B_, np_, T_, H_, W_ = sp_att_map.size()
            sp_att_map = F.softmax(sp_att_map.view(B_, np_, -1), dim=2)
            sp_att_map = sp_att_map.view(B_, np_, T_, H_, W_)
            
            simi_map = simi_map * sp_att_map
            scores = self.global_max_pool1(simi_map).view(-1, self.num_proto)

            return scores, simi_map, sp_att_map

class SLEseq(nn.Module):  # sample level explain
    '''
    input: sequence (CT)

    '''
    def __init__(self,
                 backbone_name,
                 separate_backbone=1,
                 conditioned_proto=1,
                 proto_generator_name=None,
                 num_explain=5,
                 conditioned_relevance=0,
                 relevance_generator_name=None,
                 num_proto=10,
                 proto_dim=512,
                 in_dim=4,
                 simi_func='dot',
                 base_feat=1,
                 freeze_base=False,
                 freeze_proto=False,
                 freeze_relev=False,
                 temperature=1,
                 proto_activate='softmax',
                 backbone_add_on=0,
                 score_sum_linear=1,
                 **model_opts):
        super(SLEseq, self).__init__()
        self.backbone_name = backbone_name
        self.separate_backbone = separate_backbone
        self.proto_generator_name = proto_generator_name
        self.conditioned_proto = conditioned_proto
        self.num_explain = num_explain
        self.conditioned_relevance = conditioned_relevance
        self.relevance_generator_name = relevance_generator_name
        self.proto_dim = proto_dim
        self.in_dim = in_dim
        self.num_proto = num_proto
        self.simi_func = simi_func
        self.base_feat = base_feat
        self.freeze_base = freeze_base
        self.freeze_proto = freeze_proto
        self.freeze_relev = freeze_relev
        self.temperature = temperature
        self.proto_activate = proto_activate
        self.backbone_add_on = backbone_add_on
        self.score_sum_linear = score_sum_linear

        self.cos_func = nn.CosineSimilarity(dim=1)
        self.epsilon = 1e-4

        # create backbone
        self.backbone = create_backbone(self.backbone_name, lstm_h_dim=self.proto_dim, lstm_input_dim=self.in_dim)
        feat_last_channel = last_lstm_channel(self.backbone.modules())
        self.add_on_layer = nn.BatchNorm1d(feat_last_channel)
        if self.backbone_add_on:
            self.add_on_layer = nn.Sequential(
                nn.Linear(feat_last_channel, feat_last_channel//2),
                nn.ReLU(),
                nn.Linear(feat_last_channel//2, feat_last_channel),
                nn.BatchNorm1d(feat_last_channel)
            )
        if self.freeze_base:
            freeze(self.backbone)
        if self.simi_func in ('ego_gen_channel_att+linear', 'traj_gen_channel_att+linear', 'channel_att+linear', 'channel_att+mlp'):
            # self.proto_fc = nn.Linear(feat_last_channel + proto_dim, proto_dim * self.num_proto)
            self.flatten1 = nn.Flatten()
            if self.simi_func == 'channel_att+mlp':
                self.channel_att_fc = nn.Sequential(nn.Linear(feat_last_channel, feat_last_channel),
                                                    nn.ReLU(),
                                                    nn.Linear(feat_last_channel, feat_last_channel * self.num_proto),
                                                    nn.ReLU(),
                                                    nn.Linear(feat_last_channel * self.num_proto, feat_last_channel * self.num_proto),
                                                    nn.Sigmoid()
                                                    )
            else:
                self.channel_att_fc = nn.Linear(feat_last_channel, feat_last_channel * self.num_proto)

            self.sum_linear = nn.Linear(feat_last_channel, 1)
        
        elif self.simi_func in ('dot', 'protos_only'):
            self.feat_fc = nn.Linear(feat_last_channel, proto_dim)
            self.feat_generator = nn.Sequential(self.backbone,
                                                self.feat_fc)
            if self.separate_backbone:
                if self.conditioned_proto:
                    self.proto_backbone = create_backbone(self.proto_generator_name, lstm_h_dim=self.proto_dim, lstm_input_dim=self.in_dim)
                    self.proto_fc = nn.Linear(feat_last_channel, self.proto_dim * self.num_proto, bias=False)
                    self.proto_generator = nn.Sequential(self.proto_backbone,
                                                        self.proto_fc)
                    if self.freeze_proto:
                        freeze(self.proto_backbone)
                    # freeze(self.proto_backbone)
                if self.conditioned_relevance:
                    self.relevance_backbone = create_backbone(self.relevance_generator_name, lstm_h_dim=self.proto_dim, lstm_input_dim=self.in_dim)
                    self.relevance_fc = nn.Linear(feat_last_channel, self.num_proto, bias=False)
                    self.relevance_generator = nn.Sequential(self.relevance_backbone,
                                                            self.relevance_fc)
                    if self.freeze_relev:
                        freeze(self.relevance_backbone)
            else:
                raise NotImplementedError('separate backbone', self.separate_backbone)
        elif self.simi_func in ('fix_proto1', 'fix_proto2'):
            self.prototype_shape = [self.num_proto, feat_last_channel]
            self.proto_vec = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        else:
            raise NotImplementedError(self.simi_func)

    def forward(self, x):
        # pad ego motion with 0
        if x.size(-1) == 1:
            x = torch.concat([x, torch.zeros(x.size()).cuda()], dim=-1)
        if self.simi_func in ('ego_gen_channel_att+linear', 'traj_gen_channel_att+linear', 'channel_att+linear', 'channel_att+mlp'):
            feat = self.backbone(x) # B C
            feat = self.add_on_layer(feat)
            n_channel = feat.size(-1)
            channel_weights = self.channel_att_fc(feat)  # B n_proto * C
            channel_weights = channel_weights.reshape(-1, self.num_proto, n_channel)
            if self.proto_activate == 'softmax':
                channel_weights = channel_weights / self.temperature
                channel_distr = F.softmax(channel_weights, dim=-1)  # B n_proto C
            elif self.proto_activate == 'sigmoid':
                channel_distr = torch.sigmoid(channel_weights)  # B n_proto C
            elif self.proto_activate == 'norm':
                channel_distr = F.normalize(channel_weights, dim=-1)
            elif self.proto_activate == 'avg':
                channel_weights = torch.ones(size=channel_weights.size()).cuda()
                channel_distr = torch.ones(size=channel_weights.size()).cuda()
            feat = feat.reshape(-1, 1, n_channel)
            scores = feat * channel_distr  # B n_proto C
            if not hasattr(self, 'score_sum_linear') or self.score_sum_linear:
                scores = F.relu(torch.squeeze(self.sum_linear(scores), -1))  # B n_proto
            else:
                scores = F.relu(torch.sum(scores, dim=-1, keepdim=False))  # B n_proto
            
            return scores, torch.squeeze(feat, 1), channel_weights, channel_distr
        
        elif self.simi_func in ('protos_only', 'dot'):
            feat = self.feat_generator(x)
            protos = self.proto_generator(x)
            relevances = self.relevance_generator(x)

            protos = protos.view(-1, self.num_proto, self.proto_dim)  # B num_p d

            if self.simi_func == 'dot':
                simis = torch.squeeze(torch.matmul(protos, torch.unsqueeze(feat, -1)), -1)  # B, np
            elif self.simi_func == 'protos_only':
                simis = torch.mean(protos, dim=-1)
            else:
                raise NotImplementedError('simi func', self.simi_func)

            x = relevances * simis

            return x, feat, relevances, protos
        elif self.simi_func in ('fix_proto1', 'fix_proto2'):
            feat = self.backbone(x) # B C
            # print(self.simi_func)
            if self.proto_activate == 'norm':
                self.proto_vec = nn.Parameter(F.normalize(self.proto_vec, dim=1), requires_grad=True)
            scores = F.relu(F.linear(feat, weight=self.proto_vec))
            
            return scores
