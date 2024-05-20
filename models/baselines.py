from os import kill
from pickle import NONE
from turtle import forward
from matplotlib import use
from numpy import indices
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import create_backbone
from tools.datasets.TITAN import NUM_CLS_ATOMIC, NUM_CLS_COMPLEX, NUM_CLS_COMMUNICATIVE, NUM_CLS_TRANSPORTING, NUM_CLS_AGE
from config import ckpt_root


FLATTEN_DIM = {
    'R3D18': 25088,
    'R3D18_clean': 25088,
    'R3D50': 100352,
    'I3D': 100352,
    'I3D_clean': 100352,
    'C3D':32768,
    'C3D_clean':32768
}

LAST_CHANNEL = {
    'R3D18': 512,
    'R3D18_clean':512,
    'R3D50': 2048,
    'I3D': 1024,
    'I3D_clean': 1024,
    'C3D': 512,
    'C3D_clean': 512,
    'C3D_new': 512,
}

class PCPA(nn.Module):
    def __init__(self, 
                 h_dim=256,
                 q_modality='ego',
                 num_classes=2,
                 use_atomic=0, 
                 use_complex=0, 
                 use_communicative=0, 
                 use_transporting=0, 
                 use_age=0,
                 use_cross=1,
                 dataset_name='TITAN',
                 trainable_weights=0,
                 m_task_weights=0,
                 init_class_weights=None) -> None:
        super(PCPA, self).__init__()
        self.h_dim = h_dim
        self.q_modality = q_modality
        self.num_classes = num_classes
        self.use_atomic = use_atomic
        self.use_complex = use_complex
        self.use_communicative = use_communicative
        self.use_transporting = use_transporting
        self.use_age = use_age
        self.use_cross = use_cross
        self.dataset_name = dataset_name
        self.trainable_weights = trainable_weights
        self.init_class_weights = init_class_weights
        self.m_task_weights = m_task_weights

        self.traj_encoder = nn.GRU(4, self.h_dim, batch_first=True)
        self.traj_att_w = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.traj_att_out = nn.Linear(self.h_dim * 2, self.h_dim, bias=False)
        self.traj_dropout = nn.Dropout(0.5)

        self.sk_encoder = nn.GRU(34, self.h_dim, batch_first=True)
        self.sk_att_w = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.sk_att_out = nn.Linear(self.h_dim * 2, self.h_dim, bias=False)
        self.sk_dropout = nn.Dropout(0.5)

        if self.dataset_name == 'TITAN':
            self.ego_encoder = nn.GRU(2, self.h_dim, batch_first=True)
        else:
            self.ego_encoder = nn.GRU(1, self.h_dim, batch_first=True)
        self.ego_att_w = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.ego_att_out = nn.Linear(self.h_dim * 2, self.h_dim, bias=False)
        self.ego_dropout = nn.Dropout(0.5)

        self.ctx_encoder = create_backbone(backbone_name='C3D_full', last_dim=487)
        self.ctx_embedder = nn.Sequential(nn.Linear(8192, self.h_dim, bias=False),
                                          nn.Sigmoid())
        
        self.modal_att_w = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.modal_att_out = nn.Linear(self.h_dim * 2, self.h_dim, bias=False)

        # self.dropout = nn.Dropout(0.5)
        # self.last_fc = nn.Linear(self.h_dim, self.num_classes, bias=False)
        # self.param_except_last = nn.ModuleList([self.traj_encoder,
        #                                         self.traj_att_w,
        #                                         self.traj_att_out,
        #                                         self.sk_encoder,
        #                                         self.sk_att_w,
        #                                         self.sk_att_out,
        #                                         self.ego_encoder,
        #                                         self.ego_att_w,
        #                                         self.ego_att_out,
        #                                         self.ctx_encoder,
        #                                         self.ctx_embedder,
        #                                         self.modal_att_w,
        #                                         self.modal_att_out
        #                                         ])
        feat_channel = self.h_dim
        # last layers
        last_in_dim = feat_channel
        if self.use_atomic:
            self.atomic_layer = nn.Linear(feat_channel, NUM_CLS_ATOMIC, bias=False)
            if self.use_atomic == 2:
                last_in_dim += NUM_CLS_ATOMIC
        if self.use_complex:
            self.complex_layer = nn.Linear(feat_channel, NUM_CLS_COMPLEX, bias=False)
            if self.use_complex == 2:
                last_in_dim += NUM_CLS_COMPLEX
        if self.use_communicative:
            self.communicative_layer = nn.Linear(feat_channel, NUM_CLS_COMMUNICATIVE, bias=False)
            if self.use_communicative == 2:
                last_in_dim += NUM_CLS_COMMUNICATIVE
        if self.use_transporting:
            self.transporting_layer = nn.Linear(feat_channel, NUM_CLS_TRANSPORTING, bias=False)
            if self.use_transporting == 2:
                last_in_dim += NUM_CLS_TRANSPORTING
        if self.use_age:
            self.age_layer = nn.Linear(feat_channel, NUM_CLS_AGE)
            if self.use_age == 2:
                last_in_dim += NUM_CLS_AGE
        self.last_layer = nn.Linear(last_in_dim, self.num_classes)
        
        # create class weights
        if self.trainable_weights:
            self.class_weights = nn.Parameter(torch.tensor(self.init_class_weights['cross']), requires_grad=True)
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

    def attention(self, h_seq, att_w, out_w, h_dim=256, mask=None):
        '''
        h_seq: B, T, D
        att_w: linear layer
        h_dim: int D
        mask: torch.tensor(num modality,) or None
        '''
        # import pdb;pdb.set_trace()
        seq_len = h_seq.size(1)
        q = h_seq[:, -1]  # B, D
        att1 = att_w(h_seq)  # B, T, D
        q_expand = q.view(-1, 1, h_dim).contiguous().expand(-1, seq_len, -1)  # B T D
        att2 = torch.matmul(att1.reshape(-1, 1, h_dim), q_expand.reshape(-1, h_dim, 1))  # B*T 1 1
        att2 = att2.reshape(-1, seq_len)
        score = nn.functional.softmax(att2, dim=1)
        score = score.reshape(-1, seq_len, 1)  # B T 1

        # remove modalities
        if mask is not None:
            mask = mask.reshape(-1, seq_len, 1)
            score = score * mask

        res1 = torch.sum(score * h_seq, dim=1)  # B D
        res = torch.concat([res1, q], dim=1)  # B 2D
        res = out_w(res)  # B D
        res = torch.tanh(res)

        return res, score
    
    def forward(self, x, mask=None):
        '''
        x: dict
        mask: torch.tensor(num modality,) or None
        '''
        if 'ego' not in x:
            if 'traj' in x:
                self.q_modality = 'traj'
            else:
                self.q_modality = 'skeleton'
        self.traj_encoder.flatten_parameters()
        self.ego_encoder.flatten_parameters()
        self.sk_encoder.flatten_parameters()
        q_feat = None
        feats = []
        if 'context' in x.keys():
            # import pdb;pdb.set_trace()
            obs_len = x['context'].size(2)
            try:
                ctx = nn.functional.interpolate(x['context'], size=(obs_len, 112, 112))  # B 3 T 112 112
            except:
                print(x['context'].shape)
                raise NotImplementedError()
            img_feat = self.ctx_encoder(ctx)
            img_feat = self.ctx_embedder(img_feat)
            if self.q_modality == 'context':
                q_feat = img_feat
            else:
                feats.append(img_feat)
        if 'traj' in x.keys():
            traj_feat, _ = self.traj_encoder(x['traj'])
            traj_feat, _ = self.attention(traj_feat, self.traj_att_w, self.traj_att_out, self.h_dim)
            traj_feat = self.traj_dropout(traj_feat)
            if self.q_modality == 'traj':
                q_feat = traj_feat
            else:
                feats.append(traj_feat)
        if 'skeleton' in x.keys():
            sk = torch.flatten(x['skeleton'].permute(0, 2, 3, 1), start_dim=2)  # B 2 T N -> B T N 2 -> B T 2N
            sk_feat, _ = self.sk_encoder(sk)
            sk_feat, _ = self.attention(sk_feat, self.sk_att_w, self.sk_att_out, self.h_dim)
            sk_feat = self.sk_dropout(sk_feat)
            if self.q_modality == 'skeleton':
                q_feat = sk_feat
            else:
                feats.append(sk_feat)
        if 'ego' in x.keys():
            ego_feat, _ = self.ego_encoder(x['ego'])
            ego_feat, _ = self.attention(ego_feat, self.ego_att_w, self.ego_att_out, self.h_dim)
            ego_feat = self.ego_dropout(ego_feat)
            if self.q_modality == 'ego':
                q_feat = ego_feat
            else:
                feats.append(ego_feat)
        feats.append(q_feat)
        # import pdb; pdb.set_trace()
        feats = torch.stack(feats, dim=1)  # B M D
        
        feat, m_scores = self.attention(feats, self.modal_att_w, self.modal_att_out, self.h_dim, mask=mask)

        _logits = [feat]
        logits = {}
        if self.use_atomic:
            atomic_logits = self.atomic_layer(feat)
            logits['atomic'] = atomic_logits
            if self.use_atomic == 2:
                _logits.append(atomic_logits)
        if self.use_complex:
            complex_logits = self.complex_layer(feat)
            logits['complex'] = complex_logits
            if self.use_complex == 2:
                _logits.append(complex_logits)
        if self.use_communicative:
            communicative_logits = self.communicative_layer(feat)
            logits['communicative'] = communicative_logits
            if self.use_communicative == 2:
                _logits.append(communicative_logits)
        if self.use_transporting:
            transporting_logits = self.transporting_layer(feat)
            logits['transporting'] = transporting_logits
            if self.use_transporting == 2:
                _logits.append(transporting_logits)
        if self.use_age:
            age_logits = self.age_layer(feat)
            logits['age'] = age_logits
            if self.use_age == 2:
                _logits.append(age_logits)
        if self.use_cross:
            final_logits = self.last_layer(torch.concat(_logits, dim=1))
            logits['final'] = final_logits

        return logits, m_scores.squeeze(-1)

class BackBones(nn.Module):
    def __init__(self, backbone_name,
                 num_classes=2,
                 use_atomic=0, 
                 use_complex=0, 
                 use_communicative=0, 
                 use_transporting=0, 
                 use_age=0,
                 use_cross=1,
                 pool='avg',
                 trainable_weights=0,
                 m_task_weights=0,
                 init_class_weights=None) -> None:
        super(BackBones, self).__init__()
        self.num_classes = num_classes
        self.use_atomic = use_atomic
        self.use_complex = use_complex
        self.use_communicative = use_communicative
        self.use_transporting = use_transporting
        self.use_age = use_age
        self.use_cross = use_cross
        self.pool = pool
        self.trainable_weights = trainable_weights
        self.init_class_weights = init_class_weights
        self.m_task_weights = m_task_weights

        self.backbone = create_backbone(backbone_name=backbone_name)

        if self.pool != 'flatten':
            feat_channel = LAST_CHANNEL[backbone_name]
            self.final_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            feat_channel = FLATTEN_DIM[backbone_name]

        # last layers
        last_in_dim = feat_channel
        if self.use_atomic:
            self.atomic_layer = nn.Linear(feat_channel, NUM_CLS_ATOMIC, bias=False)
            if self.use_atomic == 2:
                last_in_dim += NUM_CLS_ATOMIC
        if self.use_complex:
            self.complex_layer = nn.Linear(feat_channel, NUM_CLS_COMPLEX, bias=False)
            if self.use_complex == 2:
                last_in_dim += NUM_CLS_COMPLEX
        if self.use_communicative:
            self.communicative_layer = nn.Linear(feat_channel, NUM_CLS_COMMUNICATIVE, bias=False)
            if self.use_communicative == 2:
                last_in_dim += NUM_CLS_COMMUNICATIVE
        if self.use_transporting:
            self.transporting_layer = nn.Linear(feat_channel, NUM_CLS_TRANSPORTING, bias=False)
            if self.use_transporting == 2:
                last_in_dim += NUM_CLS_TRANSPORTING
        if self.use_age:
            self.age_layer = nn.Linear(feat_channel, NUM_CLS_AGE)
            if self.use_age == 2:
                last_in_dim += NUM_CLS_AGE
        self.last_layer = nn.Linear(last_in_dim, self.num_classes)

        # create class weights
        if self.trainable_weights:
            self.class_weights = nn.Parameter(torch.tensor(self.init_class_weights['cross']), requires_grad=True)
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
    
    def forward(self, x):
        m_logits = {}
        x = x['img']
        x = self.backbone(x)
        if self.pool != 'flatten':
            x = self.final_pool(x)
        x = x.view(x.size(0), -1)

        _logits = [x]
        logits = {}
        if self.use_atomic:
            atomic_logits = self.atomic_layer(x)
            logits['atomic'] = atomic_logits
            if self.use_atomic == 2:
                _logits.append(atomic_logits)
        if self.use_complex:
            complex_logits = self.complex_layer(x)
            logits['complex'] = complex_logits
            if self.use_complex == 2:
                _logits.append(complex_logits)
        if self.use_communicative:
            communicative_logits = self.communicative_layer(x)
            logits['communicative'] = communicative_logits
            if self.use_communicative == 2:
                _logits.append(communicative_logits)
        if self.use_transporting:
            transporting_logits = self.transporting_layer(x)
            logits['transporting'] = transporting_logits
            if self.use_transporting == 2:
                _logits.append(transporting_logits)
        if self.use_age:
            age_logits = self.age_layer(x)
            logits['age'] = age_logits
            if self.use_age == 2:
                _logits.append(age_logits)
        if self.use_cross:
            final_logits = self.last_layer(torch.concat(_logits, dim=1))
            logits['final'] = final_logits
        
        return logits


if __name__ == '__main__':
    model = PCPA()
    # from torchstat import stat
    from thop import profile
    inputs = {}
    ctx = torch.randn(1, 3, 16, 112, 112)
    traj = torch.randn(1, 16, 4)
    ego = torch.randn(1, 16, 1)
    sk = torch.randn(1, 2, 16, 17)
    inputs['context'] = ctx
    inputs['traj'] = traj
    inputs['ego'] = ego
    inputs['skeleton'] = sk

    flops, paras = profile(model=model, inputs=(inputs,))
    print('flops:', flops)
    print('params:', paras)
    for name, para in model.named_parameters():
        print(name, ':', para.shape)
