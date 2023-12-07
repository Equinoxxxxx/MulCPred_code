from os import kill
from pickle import NONE
from turtle import forward
from matplotlib import use
from numpy import indices
import torch
import torch.nn as nn
import torch.nn.functional as F

from _backbones import create_backbone
from _backbones import C3D_backbone
from receptive_field import compute_proto_layer_rf_info_v2
from utils import last_conv_channel, last_lstm_channel, freeze

from tools.datasets.TITAN import NUM_CLS_ATOMIC, NUM_CLS_COMPLEX, NUM_CLS_COMMUNICATIVE, NUM_CLS_TRANSPORTING, NUM_CLS_AGE


class C3DEncDec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = create_backbone('C3D')
        self.decoder = create_backbone('C3Ddecoder')
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class PoseEncDec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = create_backbone('poseC3D_pretrained')
        self.decoder = create_backbone('poseC3Ddecoder')
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class MLPEncDec(nn.Module):
    def __init__(self, in_dim=4, h_dim=128, obs_len=15) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.obs_len = obs_len
        self.pred_len = obs_len
        self.encoder = nn.Linear(self.in_dim*self.obs_len, self.h_dim)
        self.decoder = nn.Linear(self.h_dim, self.in_dim*self.obs_len)
    
    def encode(self,x):
        return self.encoder(x.view(x.size(0), -1))
    
    def decode(self,x):
        return self.decoder(x).reshape(x.size(0), self.pred_len, self.in_dim)
    
    def forward(self,x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class SeqEncDec(nn.Module):
    def __init__(self, in_dim=4, h_dim=128, obs_len=15) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = obs_len
        self.encoder = nn.LSTM(batch_first=True, input_size=in_dim, hidden_size=h_dim)
        self.decoder = nn.LSTM(batch_first=True, input_size=in_dim, hidden_size=h_dim)
        self.out_fc = nn.Linear(h_dim, in_dim)

    def encode(self, x):
        self.encoder.flatten_parameters()
        _, (h, c) = self.encoder(x)  # h, c: (1, b, c)
        return h[0]

    def decode(self, x0, h):
        '''
        x0: tensor b, 1, c
        h: tensor 1, b, c
        '''
        self.decoder.flatten_parameters()
        preds = []
        c = h
        for i in range(self.pred_len):
            _, (h, c) = self.decoder(x0, (h, c))
            x0 = self.out_fc(h[0]).unsqueeze(1)  # b, 1, in_dim
            preds.append(x0)
        preds = torch.cat(preds, dim=1)  # b, t, in_dim
        return preds
    
    def forward(self, x):
        '''
        x: b, t, c
        '''
        h = self.encode(x)
        pred = self.decode(x[:, -1].unsqueeze(1), h.unsqueeze(0))
        return pred

class ConvSENN(nn.Module):
    def __init__(self, modality='img', num_proto=10, num_classes=2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_proto = num_proto
        if modality == 'img' or modality == 'context':
            self.h = C3DEncDec()
            self.theta = create_backbone('C3D')
        elif modality == 'skeleton':
            self.h = PoseEncDec()
            self.theta = self.encoder = create_backbone('poseC3D_pretrained')
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        last_c = 512
        self.h_fc = nn.Linear(last_c, num_proto)
        self.theta_fc = nn.Linear(last_c, num_proto*num_classes)

    def forward(self, x):
        '''
        x: tensor b, c, t, h, w
        '''
        # h
        feat_h = self.h.encode(x)
        recon = self.h.decode(feat_h)
        feat_h = self.pool(feat_h).view(feat_h.size(0), feat_h.size(1))  # b c
        concepts = self.h_fc(feat_h)  # b np
        # concepts = F.relu(concepts)

        # theta
        feat_t = self.theta(x)
        feat_t = self.pool(feat_t).view(feat_t.size(0), feat_t.size(1))  # b c
        relevances = self.theta_fc(feat_t)  # b np*nc

        return concepts, relevances, recon

class SeqSENN(nn.Module):
    def __init__(self, modality='traj', num_protos=10, num_classes=2, obs_len=16) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.num_proto = num_protos
        if modality == 'traj':
            self.in_dim = 4
        else:
            self.in_dim = 2
        self.h = SeqEncDec(in_dim=self.in_dim, obs_len=self.obs_len)
        self.theta = create_backbone('lstm', lstm_h_dim=128, lstm_input_dim=self.in_dim)
        last_c = 128
        self.h_fc = nn.Linear(last_c, num_protos)
        self.theta_fc = nn.Linear(last_c, num_protos*num_classes)

    def forward(self, x):
        '''
        x: tensor b, t, indim
        '''
        # h
        feat_h = self.h.encode(x)  # b c
        recon = self.h.decode(x0=x[:, -1].unsqueeze(1), h=feat_h.unsqueeze(0))  # b t indim
        concepts = self.h_fc(feat_h)  # b np
        # concepts = F.relu(concepts)

        # theta
        # print(x.size())
        relevances = self.theta_fc(self.theta(x))  # b np*nc

        return concepts, relevances, recon

class MLPSENN(nn.Module):
    def __init__(self, modality='traj', num_protos=10, num_classes=2, obs_len=16) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.num_proto = num_protos
        self.num_classes = num_classes
        if modality == 'traj':
            self.in_dim = 4
        else:
            self.in_dim = 2
        self.h = MLPEncDec(in_dim=self.in_dim, h_dim=128, obs_len=self.obs_len)
        self.theta = nn.Linear(self.in_dim*self.obs_len, 128)
        self.h_fc = nn.Linear(128, self.num_proto)
        self.theta_fc = nn.Linear(128, self.num_proto*self.num_classes)
    
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        feat_h = self.h.encode(x)
        recon = self.h.decode(feat_h).reshape(feat_h.size(0), self.obs_len, self.in_dim)
        concepts = self.h_fc(feat_h)
        # concepts = F.relu(concepts)
        relevs = self.theta_fc(self.theta(x))

        return concepts, relevs, recon


class MultiSENN(nn.Module):
    def __init__(self, use_traj=1, use_ego=1, use_img=1, use_sk=1, use_ctx=1,
                 num_classes=2,
                 pred_k='final',
                 ) -> None:
        super().__init__()
        self.use_traj = use_traj
        self.use_ego = use_ego
        self.use_img = use_img
        self.use_sk = use_sk
        self.use_ctx = use_ctx
        self.num_classes = num_classes
        self.pred_k = pred_k

        if self.use_traj:
            self.traj_model = MLPSENN(modality='traj', 
                                      num_classes=self.num_classes, 
                                      num_protos=10, 
                                      obs_len=16)
        if self.use_ego:
            self.ego_model = MLPSENN(modality='ego', 
                                     num_classes=self.num_classes, 
                                     num_protos=10, 
                                     obs_len=16)
        if self.use_img:
            self.img_model = ConvSENN(modality='img', 
                                      num_classes=self.num_classes, 
                                      num_proto=10)
        if self.use_sk:
            self.sk_model = ConvSENN(modality='skeleton', 
                                     num_classes=self.num_classes, 
                                     num_proto=10)
        if self.use_ctx:
            self.ctx_model = ConvSENN(modality='context', 
                                      num_classes=self.num_classes, 
                                      num_proto=10)
        
    def forward(self, x, mask=None):
        '''
        x: dict
        mask: torch.tensor(m*np,) or None
        '''
        recons = {}
        logits = {}
        concepts = {}
        relevs = {}
        _concepts = []
        _relevs = []

        if self.use_traj:
            concept, relev, recon = self.traj_model(x['traj'])
            relev = relev.view(relev.size(0), self.num_classes, self.traj_model.num_proto)  # b nc np
            concepts['traj'] = concept  # b np
            relevs['traj'] = relev
            recons['traj'] = recon
            # logits['final'] += torch.bmm(relev, concept.unsqueeze(2)).squeeze(2)  # b nc
            _concepts.append(concept)
            _relevs.append(relev)
        if self.use_ego:
            concept, relev, recon = self.ego_model(x['ego'])
            relev = relev.view(relev.size(0), self.num_classes, self.ego_model.num_proto)
            concepts['ego'] = concept
            relevs['ego'] = relev
            recons['ego'] = recon
            # logits['final'] += torch.bmm(relev, concept.unsqueeze(2)).squeeze(2)
            _concepts.append(concept)
            _relevs.append(relev)
        if self.use_img:
            concept, relev, recon = self.img_model(x['img'])
            relev = relev.view(relev.size(0), self.num_classes, self.img_model.num_proto)
            concepts['img'] = concept
            relevs['img'] = relev
            recons['img'] = recon
            # logits['final'] += torch.bmm(relev, concept.unsqueeze(2)).squeeze(2)
            _concepts.append(concept)
            _relevs.append(relev)
        if self.use_sk:
            concept, relev, recon = self.sk_model(x['skeleton'])
            relev = relev.view(relev.size(0), self.num_classes, self.sk_model.num_proto)
            concepts['skeleton'] = concept
            relevs['skeleton'] = relev
            recons['skeleton'] = recon
            # logits['final'] += torch.bmm(relev, concept.unsqueeze(2)).squeeze(2)
            _concepts.append(concept)
            _relevs.append(relev)
        if self.use_ctx:
            concept, relev, recon = self.ctx_model(x['context'])
            relev = relev.view(relev.size(0), self.num_classes, self.ctx_model.num_proto)
            concepts['context'] = concept
            relevs['context'] = relev
            recons['context'] = recon
            # logits['final'] += torch.bmm(relev, concept.unsqueeze(2)).squeeze(2)
            _concepts.append(concept)
            _relevs.append(relev)

        
        _concepts = torch.cat(_concepts, dim=1).unsqueeze(2)  # b m*np 1
        _relevs = torch.cat(_relevs, dim=2)  # b nc m*np
        if mask is not None:
            _concepts = _concepts * mask.reshape(1, mask.size(0), 1)
        logits[self.pred_k] = torch.bmm(_relevs, _concepts).squeeze(2)  # b nc
        
        return logits, concepts, relevs, recons


if __name__ == '__main__':
    a = torch.ones([1, 15, 4])
    model = SeqEncDec()

    b = model(a)
    print(b.size())