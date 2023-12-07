from os import kill
from pickle import NONE
from turtle import forward
from matplotlib import use
import torch
import torch.nn as nn
import torch.nn.functional as F

from _backbones import create_backbone
from _backbones import C3D_backbone
from receptive_field import compute_proto_layer_rf_info_v2

def hook_backbone(grad):
    print('grad to backbone output', torch.mean(grad))

def hook_add_on(grad):
    print('grad to add on layer output', torch.mean(grad))

def hook_dist(grad):
    print('grad to distances', torch.mean(grad))

class NonlocalMultiPNet(nn.Module):
    def __init__(self, data_types=['traj', 'img', 'context', 'skeleton'],
                 traj_model_settings=None,
                 img_model_settings=None, 
                 skeleton_model_settings=None, 
                 context_model_settings=None,) -> None:
        super(NonlocalMultiPNet, self).__init__()
        self.data_types = data_types
        self.total_p_per_cls = 0
        self.num_cls = 2
        self.use_traj = False
        self.use_img = False
        self.use_skeleton = False
        self.use_ctx = False
        self.use_single_img = False
        if len(data_types) == 0:
            raise RuntimeError('Must define at least one modality.')
        if 'traj' in self.data_types:
            self.use_traj = True
            self.traj_model_settings = {
                'backbone': None,
                'p_per_cls': 10,
                'prototype_dim': 128,
                'num_classes': 2,
                'init_weights': True,
                'prototype_activation_function': 'dot_product',
                'add_on_activation': None,
                'name': 'multi'
            }
            for k in traj_model_settings.keys():
                self.traj_model_settings[k] = traj_model_settings[k]
            print('traj model settings:', self.traj_model_settings)
            self.traj_model = NonlocalRNNPNet(**self.traj_model_settings)
            self.total_p_per_cls += self.traj_model_settings['p_per_cls']
            self.num_cls = self.traj_model_settings['num_classes']
        if 'img' in self.data_types:
            self.use_img = True
            self.img_model_settings = {
                'backbone': None,
                'p_per_cls': 10,
                'prototype_dim': 128,
                'num_classes': 2,
                'init_weights': True,
                'prototype_activation_function': 'dot_product',
                'add_on_activation': None,
                'name': 'multi'
            }
            for k in img_model_settings.keys():
                self.img_model_settings[k] = img_model_settings[k]
            print('img model settings:', self.img_model_settings)
            self.img_model = NonlocalRNNPNet(**self.img_model_settings)
            self.total_p_per_cls += self.img_model_settings['p_per_cls']
            self.num_cls = self.img_model_settings['num_classes']
        if 'context' in data_types:
            self.use_ctx = True
            self.context_model_settings = {
                'backbone': None,
                'p_per_cls': 10,
                'prototype_dim': 128,
                'num_classes': 2,
                'init_weights': True,
                'prototype_activation_function': 'dot_product',
                'add_on_activation': None,
                'name': 'multi'
            }
            for k in context_model_settings.keys():
                self.context_model_settings[k] = context_model_settings[k]
            print('context model settings:', self.context_model_settings)
            self.context_model = NonlocalRNNPNet(**self.context_model_settings)
            self.total_p_per_cls += self.context_model_settings['p_per_cls']
            self.num_cls = self.context_model_settings['num_classes']

    def forward(self, x):
        res = None
        simis = []
        if 'traj' in x.keys():
            logit, min_dists = self.img_model(x['img'])
            simis.append(min_dists)
            if res is None:
                res = logit
            else:
                res += logit
        if 'img' in x.keys():
            logit, min_dists = self.img_model(x['img'])
            simis.append(min_dists)
            if res is None:
                res = logit
            else:
                res += logit

        if 'skeleton' in x.keys():
            logit, min_dists = self.skeleton_model(x['skeleton'])
            simis.append(min_dists)
            if res is None:
                res = logit
            else:
                res += logit
        
        if 'context' in x.keys():
            logit = 0
            if self.context_model_settings['ctx_mode'] == 'seg_multi':
                for i in range(len(self.context_model_settings['seg_class_idx'])):
                    _logit, min_dists = self.context_model[i](x['context'][:, :, :, :, :, i])
                    simis.append(min_dists)
                    logit += _logit
            else:
                logit, min_dists = self.context_model(x['context'])
                simis.append(min_dists)
            if res is None:
                res = logit
            else:
                res += logit

        simis = torch.concat(simis, dim=1)  # B, m * np

        return res, simis

class NonlocalRNNPNet(nn.Module):
    def __init__(self, backbone, p_per_cls=10, prototype_dim=128,
                 num_classes=2, init_weights=True, prototype_activation_function='dot_product',
                 add_on_activation=None, last_nonlinear=0, name='multi',
                 ctx_mode='local') -> None:
        super(NonlocalRNNPNet, self).__init__()
        self.backbone = backbone
        self.p_per_cls = p_per_cls
        self.prototype_dim = prototype_dim
        self.num_classes = num_classes
        self.init_weights = init_weights
        self.prototype_activation_function = prototype_activation_function
        self.add_on_activation = add_on_activation
        self.last_nonlinear = last_nonlinear

        self.num_prototypes = p_per_cls * num_classes
        self.prototype_shape = [self.num_prototypes, prototype_dim]
        self.epsilon = 1e-4

        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.p_per_cls] = 1

        input_dim = [i for i in self.backbone.modules() if isinstance(i, nn.LSTM)][-1].hidden_size
        if self.add_on_activation == 'linear_bn':
            self.add_on_layers = nn.Sequential(
                nn.Linear(input_dim, self.prototype_dim),
                # nn.BatchNorm1d(self.prototype_dim)
            )
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)
        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.prototype_ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)
        if self.last_nonlinear == 0:
            self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                        bias=False) # do not use bias
        if init_weights and add_on_activation is not None:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Linear):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5, layer=self.last_layer)

    def set_last_layer_incorrect_connection(self, layer, incorrect_strength=-0.5):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def calc_featmap(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.backbone(x)
        # print('backbone out', self.name, torch.mean(x))
        if self.add_on_activation is not None:
            x = self.add_on_layers(x)

        return x  # B, D

    def calc_similarity(self, x):
        # x: B, D
        if self.prototype_activation_function == 'dot_product':
            # print(x.shape, self.prototype_vectors.T.shape)
            x = torch.matmul(x, self.prototype_vectors.T)  # (B, D) (D, np)
            # print('simi shape', x.shape)
        else:
            x = torch.matmul(x, self.prototype_vectors.T)

        return x  # B, np
    
    def push_forward(self, x):
        x = self.calc_featmap(x)
        # print('feat shape', x.shape)
        simi = self.calc_similarity(x)  # B, np
        
        return x, simi
    
    def forward(self, x):
        _, simi = self.push_forward(x)
        # print('simi shape', simi.shape)
        # print('x shape', x.shape)
        x = self.last_layer(simi)

        return x, simi

class MultiBackbone(nn.Module):
    def __init__(self, use_img=True, use_skeleton=True, use_context=True, use_single_img=False,
                 img_backbone_name='C3D', sk_backbone_name='SK', ctx_backbone_name='C3D', single_img_backbone_name='segC2D',
                 last_pool='avg', num_classes=2, fusion=2) -> None:
        super(MultiBackbone, self).__init__()
        self.use_img = use_img
        self.use_skeleton = use_skeleton
        self.use_context = use_context
        self.img_backbone_name = img_backbone_name
        self.sk_backbone_name = sk_backbone_name
        self.ctx_backbone_name = ctx_backbone_name
        self.last_pool = last_pool
        self.total_channels = 0
        self.num_classes = num_classes
        self.fusion = fusion
        if fusion == 1:
            if use_img:
                self.img_backbone = create_backbone(img_backbone_name)
                self.total_channels += [i for i in self.img_backbone.modules() if isinstance(i, nn.Conv3d)][-1].out_channels
            if use_skeleton:
                self.skeleton_backbone = create_backbone(sk_backbone_name)
                self.total_channels += [i for i in self.skeleton_backbone.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
            if use_context:
                self.context_backbone = create_backbone(ctx_backbone_name)
                self.total_channels += [i for i in self.context_backbone.modules() if isinstance(i, nn.Conv3d)][-1].out_channels
            if last_pool == 'avg':
                self.pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
                self.pool2d = nn.AdaptiveAvgPool2d((1, 1))
            else:
                self.pool3d = nn.AdaptiveMaxPool3d((1, 1, 1))
                self.pool2d = nn.AdaptiveMaxPool2d((1, 1))
            
            self.linear = nn.Linear(self.total_channels, num_classes, bias=False)
        elif fusion == 2:
            if use_img:
                self.img_backbone = create_backbone(img_backbone_name)
                self.img_last_channel = [i for i in self.img_backbone.modules() if isinstance(i, nn.Conv3d)][-1].out_channels
                self.img_linear = nn.Linear(self.img_last_channel, self.num_classes)
                self.img_model = nn.Sequential(
                    self.img_backbone,
                    nn.AdaptiveMaxPool3d((1, 1, 1)),
                    nn.Flatten(),
                    self.img_linear
                )
            if use_skeleton:
                self.skeleton_backbone = create_backbone(sk_backbone_name)
                self.sk_last_channel= [i for i in self.skeleton_backbone.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
                self.sk_linear = nn.Linear(self.sk_last_channel, self.num_classes)
                self.skeleton_model = nn.Sequential(
                    self.skeleton_backbone,
                    nn.AdaptiveMaxPool3d((1, 1, 1)),
                    nn.Flatten(),
                    self.sk_linear
                )
            if use_context:
                self.context_backbone = create_backbone(ctx_backbone_name)
                self.ctx_last_channel = [i for i in self.context_backbone.modules() if isinstance(i, nn.Conv3d)][-1].out_channels
                self.ctx_linear = nn.Linear(self.ctx_last_channel, self.num_classes)
                self.context_model = nn.Sequential(
                    self.context_backbone,
                    nn.AdaptiveMaxPool3d((1, 1, 1)),
                    nn.Flatten(),
                    self.ctx_linear
                )
        else:
            raise NotImplementedError('fusion=' + str(fusion))
    def calc_feat(self, x):
        res = []
        if self.use_img:
            img_rep = self.img_backbone(x['img'])
            res.append(img_rep)
        if self.use_skeleton:
            sk_rep = self.skeleton_backbone(x['skeleton'])
            res.append(sk_rep)
        if self.use_context:
            ctx_rep = self.context_backbone(x['context'])
            res.append(ctx_rep)
        return res

    def forward(self, x):
        # import pdb;pdb.set_trace()
        if self.fusion == 1:
            res = []
            if self.use_img:
                rep = self.img_backbone(x['img'])
                rep = self.pool3d(rep)
                rep = rep.view(rep.size(0), -1)
                res.append(rep)
            if self.use_skeleton:
                rep = self.skeleton_backbone(x['skeleton'])
                rep = self.pool2d(rep)
                rep = rep.view(rep.size(0), -1)
                res.append(rep)
            if self.use_context:
                rep = self.context_backbone(x['context'])
                rep = self.pool3d(rep)
                rep = rep.view(rep.size(0), -1)
                res.append(rep)
            
            assert res != [], 'No modality chosen'

            res = torch.concat(res, dim=1)
            res = self.linear(res)
        elif self.fusion == 2:
            res = 0
            if self.use_img:
                res += self.img_model(x['img'])
            if self.use_skeleton:
                res += self.skeleton_model(x['skeleton'])
            if self.use_context:
                res += self.context_model(x['context'])

        return res
        
class MultiPNet(nn.Module):
    def __init__(self, data_types=('img', 'skeleton', 'context'),
                 img_model_settings={}, 
                 skeleton_model_settings={}, 
                 context_model_settings={},
                 single_img_model_settings={}
                 ):
        super(MultiPNet, self).__init__()
        self.data_types = data_types
        self.total_p_per_cls = 0
        self.num_cls = 2
        self.use_img = False
        self.use_skeleton = False
        self.use_ctx = False
        self.use_single_img = False
        if len(data_types) == 0:
            raise RuntimeError('Must define at least one modality.')
        if 'img' in data_types:
            self.use_img = True
            self.img_model_settings = {
                'backbone': None,
                'vid_size': (16, 224, 224),
                'p_per_cls': 20,
                'prototype_dim': 128,
                'sp_proto_layer_rf_info': None,
                't_proto_layer_rf_info': None,
                'num_classes': 2,
                'init_weights': True,
                'prototype_activation_function': 'log',
                'add_on_activation': 'sigmoid',
                'name': 'multi'
            }
            for k in img_model_settings.keys():
                self.img_model_settings[k] = img_model_settings[k]
            print('img model settings:', self.img_model_settings)
            self.img_model = ImagePNet(**self.img_model_settings)
            self.total_p_per_cls += self.img_model_settings['p_per_cls']
            self.num_cls = self.img_model_settings['num_classes']
        if 'skeleton' in data_types:
            self.use_skeleton = True
            self.skeleton_model_settings = {
                'backbone': None,
                'skeleton_mode': 'coord',
                'skeleton_seq_shape': (2, 16, 17),
                'p_per_cls': 20,
                'prototype_dim': 128,
                'sp_proto_layer_rf_info': None,
                'num_classes': 2,
                'init_weights': True,
                'prototype_activation_function': 'log',
                'add_on_activation': 'sigmoid',
            }
            for k in skeleton_model_settings.keys():
                self.skeleton_model_settings[k] = skeleton_model_settings[k]
            print('skeleton model settings:', self.skeleton_model_settings)
            if self.skeleton_model_settings['skeleton_mode'] == 'coord':
                self.skeleton_model = SkeletonPNet(**self.skeleton_model_settings)
            elif self.skeleton_model_settings['skeleton_mode'] == 'heatmap':
                # convert args
                self.heatmap_skeleton_model_settings = {}
                sk_to_img = {
                    'backbone':'backbone',
                    'p_per_cls': 'p_per_cls',
                    'prototype_dim': 'prototype_dim',
                    'sp_proto_layer_rf_info': 'sp_proto_layer_rf_info',
                    't_proto_layer_rf_info': 't_proto_layer_rf_info',
                    'num_classes': 'num_classes',
                    'init_weights': 'init_weights',
                    'prototype_activation_function': 'prototype_activation_function',
                    'add_on_activation': 'add_on_activation'
                }
                for k in sk_to_img.keys():
                    self.heatmap_skeleton_model_settings[sk_to_img[k]] = self.skeleton_model_settings[k]
                self.skeleton_model = ImagePNet(**self.heatmap_skeleton_model_settings)
            self.total_p_per_cls += self.skeleton_model_settings['p_per_cls']
            self.num_cls = self.skeleton_model_settings['num_classes']
        if 'context' in data_types:
            self.use_ctx = True
            self.context_model_settings = {
                'backbone': None,
                'backbone_name': 'segC3D',
                'vid_size': (16, 224, 224),
                'p_per_cls': 20,
                'prototype_dim': 128,
                'sp_proto_layer_rf_info': None,
                't_proto_layer_rf_info': None,
                'num_classes': 2,
                'init_weights': True,
                'prototype_activation_function': 'log',
                'add_on_activation': 'sigmoid',
                'name': 'multi',
                'ctx_mode': 'mask_ped',
                'seg_class_idx': [24, 26, 19, 20]
            }
            for k in context_model_settings.keys():
                self.context_model_settings[k] = context_model_settings[k]
            print('context model settings:', self.context_model_settings)
            if self.context_model_settings['ctx_mode'] == 'seg_multi':
                self.context_model_settings.pop('backbone')
                self.context_model = []
                for i in range(len(self.context_model_settings['seg_class_idx'])):
                    backbone = create_backbone(self.context_model_settings['backbone_name'])
                    self.context_model.append(ImagePNet(backbone=backbone, **self.context_model_settings))
                    self.total_p_per_cls += self.context_model_settings['p_per_cls']
                self.context_model = torch.nn.ModuleList(self.context_model)
            else:
                self.context_model = ImagePNet(**self.context_model_settings)
                self.total_p_per_cls += self.context_model_settings['p_per_cls']
                self.num_cls = self.context_model_settings['num_classes']
        if 'single_img' in data_types:
            self.use_single_img = True
            self.single_img_model_settings = {
                'backbone': None,
                'skeleton_seq_shape': (3, 224, 224),
                'p_per_cls': 20,
                'prototype_dim': 128,
                'sp_proto_layer_rf_info': None,
                'num_classes': 2,
                'init_weights': True,
                'prototype_activation_function': 'log',
                'add_on_activation': 'sigmoid',
            }
            for k in single_img_model_settings.keys():
                self.single_img_model_settings[k] = single_img_model_settings[k]
            self.single_img_model = SkeletonPNet(**self.single_img_model_settings)
            self.total_p_per_cls += self.single_img_model_settings['p_per_cls']
            self.num_cls = self.single_img_model_settings['num_classes']
            
        self.num_prototypes = self.total_p_per_cls * self.num_cls
        # self.linear = nn.Linear(self.num_prototypes, self.num_cls,
        #                             bias=False) # do not use bias
    
    def push_forwad(self, x):
        dists = []
        if 'img' in x.keys():
            _, dist = self.img_model.push_forward(x['img'])
            dists.append(dist)
        if 'skeleton' in x.keys():
            _, dist = self.skeleton_model.push_forward(x['skeleton'])
            dists.append(dist)
        if 'context' in x.keys():
            if self.context_model_settings['ctx_mode'] == 'seg_multi':
                for i in range(len(self.context_model_settings['seg_class_idx'])):
                    _, dist = self.context_model[i].push_forward(x['context'][:, :, :, :, :, i])
                    dists.append(dist)
            else:
                _, dist = self.context_model.push_forward(x['context'])
                dists.append(dist)
        if 'single_img' in x.keys():
            _, dist = self.single_img_model.push_forward(x['single_img'])
            dists.append(dist)
        dists = torch.concat(dists, dim=1)

        return dists

    def forward(self, x):
        res = None
        dists = []
        if 'img' in x.keys():
            logit, min_dists = self.img_model(x['img'])
            dists.append(min_dists)
            if res is None:
                res = logit
            else:
                res += logit

        if 'skeleton' in x.keys():
            logit, min_dists = self.skeleton_model(x['skeleton'])
            dists.append(min_dists)
            if res is None:
                res = logit
            else:
                res += logit
        
        if 'context' in x.keys():
            logit = 0
            if self.context_model_settings['ctx_mode'] == 'seg_multi':
                for i in range(len(self.context_model_settings['seg_class_idx'])):
                    _logit, min_dists = self.context_model[i](x['context'][:, :, :, :, :, i])
                    dists.append(min_dists)
                    logit += _logit
            else:
                logit, min_dists = self.context_model(x['context'])
                dists.append(min_dists)
            if res is None:
                res = logit
            else:
                res += logit
        if 'single_img' in x.keys():
            logit, min_dists = self.single_img_model(x['single_img'])
            dists.append(min_dists)
            if res is None:
                res = logit
            else:
                res += logit

        dists = torch.concat(dists, dim=1)

        return res, dists

class ImagePNet(nn.Module):
    def __init__(self, backbone, vid_size=(16, 224, 224), p_per_cls=20, prototype_dim=128,
                 sp_proto_layer_rf_info=None, t_proto_layer_rf_info=None, num_classes=2, init_weights=True,
                 prototype_activation_function='log', add_on_activation='sigmoid', name='single', 
                 last_nonlinear=0, last_inter_dim=128, **model_opts):
        super(ImagePNet, self).__init__()
        self.name = name
        self.vid_size = vid_size
        self.p_per_cls = p_per_cls
        self.num_classes = num_classes
        self.num_prototypes = p_per_cls * num_classes
        self.prototype_dim = prototype_dim
        self.last_nonlinear = last_nonlinear
        self.last_inter_dim = last_inter_dim
        self.add_on_activation = add_on_activation
        self.cos_func = nn.CosineSimilarity(dim=1)

        self.prototype_shape = [self.num_prototypes, prototype_dim, 1, 1, 1]
        
        self.epsilon = 1e-4
        
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        
        self.backbone = backbone
        
        # a onehot indication matrix for each prototype's class identity. [0~9, :] for class 0, [10~19, :] for class 1, ...
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.p_per_cls] = 1

        self.sp_rf_info = sp_proto_layer_rf_info
        self.t_rf_info = t_proto_layer_rf_info


        # import pdb; pdb.set_trace()
        first_add_on_layer_in_channels = [i for i in self.backbone.modules() if isinstance(i, nn.Conv3d)][-1].out_channels  # channel of the last conv in backbone

        if add_on_activation == 'conv':
            self.add_on_layers = nn.Sequential(
                nn.Conv3d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=self.prototype_dim, out_channels=self.prototype_dim, kernel_size=1)
                )
        elif add_on_activation == 'conv_bn':
            self.add_on_layers = nn.Sequential(
                nn.Conv3d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_dim, kernel_size=1),
                nn.BatchNorm3d(self.prototype_dim)
            )
        elif add_on_activation == 'bn':
            assert first_add_on_layer_in_channels == self.prototype_dim, (first_add_on_layer_in_channels, self.prototype_dim)
            self.add_on_layers = nn.Sequential(
                nn.BatchNorm3d(self.prototype_dim)
            )
        elif add_on_activation == 'sigmoid':
            self.add_on_layers = nn.Sequential(
                nn.Conv3d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=self.prototype_dim, out_channels=self.prototype_dim, kernel_size=1),
                nn.Sigmoid()
                )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.prototype_ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)
        if self.last_nonlinear == 0:
            self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                        bias=False) # do not use bias
        elif self.last_nonlinear == 1:
            self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                        bias=False) # do not use bias  random initialize
        elif self.last_nonlinear == 2:
            self.linear = nn.Linear(self.num_prototypes, self.num_classes,
                                        bias=False)
            self.last_layer = nn.Sequential(
                nn.LayerNorm(self.num_prototypes),
                nn.ReLU(),
                self.linear,
                
            )
        elif self.last_nonlinear == 3:
            self.last_layer = nn.Sequential(
                nn.BatchNorm3d(self.num_prototypes),
                nn.ReLU(),
                nn.Conv3d(self.num_prototypes, self.num_classes, kernel_size=1),
                nn.BatchNorm3d(self.num_classes),
                nn.ReLU(),
                nn.Flatten(),  # 2(n cls)*4(T)*8*8=256
                nn.Linear(512, self.num_classes)
            )
        if init_weights:
            self._initialize_weights()
    
    def calc_featmap(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.backbone(x)
        # print('backbone out', self.name, torch.mean(x))
        x = self.add_on_layers(x)

        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2  # B * C * T * H * W
        x2_patch_sum = F.conv3d(input=x2, weight=self.prototype_ones)  # B * np * T * H * W

        p2 = self.prototype_vectors ** 2  # np * C * 1 * 1 * 1
        p2 = torch.sum(p2, dim=(1, 2, 3, 4))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1, 1)
        p2_reshape = p2.view(-1, 1, 1, 1)  # np, 1, 1, 1

        xp = F.conv3d(input=x, weight=self.prototype_vectors)  # dot product result, B, np, T, H, W
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)  # B, np, T, H, W

        return distances
    
    def cosine_simi(self, x):
        '''
        x: B C T H W
        '''
        simis = []
        # proto: np C 1 1 1
        for i in range(self.num_prototypes):
            proto = self.prototype_vectors[i].view(1, self.prototype_dim, 1, 1, 1)  # 1 C 1 1 1
            simi = self.cos_func(x, proto)  # B T H W
            simis.append(simi)
        simis = torch.stack(simis, dim=1)  # B np T H W

        return simis

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''

        x = self.calc_featmap(x)
        if self.prototype_activation_function == 'cos':
            x = -self.cosine_simi(x)  # convert similarity to distance
        else:
            x = self._l2_convolution(x)

        return x

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        elif self.prototype_activation_function == 'cos':
            return -distances
        else:
            raise NotImplementedError(self.prototype_activation_function)

    def forward(self, x):
        x = self.prototype_distances(x)  # B, np, t, h, w
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        if self.last_nonlinear <= 2:
            # global min pooling  B, np, 1, 1, 1
            min_distances = -F.max_pool3d(-x,
                                kernel_size=(x.size()[2],
                                            x.size()[3],
                                            x.size()[4]))
            min_distances = min_distances.view(-1, self.num_prototypes)  # B, np
            prototype_activations = self.distance_2_similarity(min_distances)
            # print('simi shape', prototype_activations.shape)
            logits = self.last_layer(prototype_activations)
        elif self.last_nonlinear == 3:
            # global min pooling  B, np, 1, 1, 1
            min_distances = -F.max_pool3d(-x,
                                kernel_size=(x.size()[2],
                                            x.size()[3],
                                            x.size()[4]))
            min_distances = min_distances.view(-1, self.num_prototypes)  # B, np
            prototype_activations = self.distance_2_similarity(x)
            logits = self.last_layer(prototype_activations)
        return logits, min_distances

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        x = self.calc_featmap(x)
        distances = self._l2_convolution(x)  # B, np, T, H, W
        return x, distances

    def repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.backbone,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, layer, incorrect_strength=-0.5):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv3d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.last_nonlinear == 0:
            self.set_last_layer_incorrect_connection(incorrect_strength=-0.5, layer=self.last_layer)
        if self.last_nonlinear == 2:
            self.set_last_layer_incorrect_connection(incorrect_strength=-0.5, layer=self.last_layer[2])
        if self.last_nonlinear == 3:
            for m in self.last_layer.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
                    # every init technique has an underscore _ in the name
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

class SkeletonPNet(nn.Module):
    def __init__(self, backbone, skeleton_seq_shape=(2, 16, 17), p_per_cls=20, prototype_dim=128,
                 sp_proto_layer_rf_info=None, num_classes=2, init_weights=True,
                 prototype_activation_function='log', add_on_activation='sigmoid', last_nonlinear=0, **model_opts):

        super(SkeletonPNet, self).__init__()

        self.skeleton_seq_shape = skeleton_seq_shape
        self.p_per_cls = p_per_cls
        self.num_classes = num_classes
        self.num_prototypes = p_per_cls * num_classes
        self.prototype_dim = prototype_dim
        self.last_nonlinear = last_nonlinear
        self.cos_func = nn.CosineSimilarity(dim=1)

        self.prototype_shape = [self.num_prototypes, prototype_dim, 1, 1]
        
        self.epsilon = 1e-4
        
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        
        self.backbone = backbone
        
        # a onehot indication matrix for each prototype's class identity. [0~9, :] for class 0, [10~19, :] for class 1, ...
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.p_per_cls] = 1

        self.sp_rf_info = sp_proto_layer_rf_info

        first_add_on_layer_in_channels = [i for i in self.backbone.modules() if isinstance(i, nn.Conv2d)][-1].out_channels  # channel of the last conv in backbone

        if add_on_activation is None:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_dim, kernel_size=1),
                nn.BatchNorm2d(self.prototype_dim),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_dim, out_channels=self.prototype_dim, kernel_size=1),
                )
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_dim, kernel_size=1),
                nn.BatchNorm2d(self.prototype_dim),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_dim, out_channels=self.prototype_dim, kernel_size=1),
                nn.BatchNorm2d(self.prototype_dim),
                nn.Sigmoid()
                )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.prototype_ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias

        if init_weights:
            self._initialize_weights()
    
    def calc_featmap(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.backbone(x)
        # x.register_hook(hook_backbone)  # not leaf
        x = self.add_on_layers(x)
        # x.register_hook(hook_add_on)  # not leaf
        return x

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x

        x2 - 2xp + p2   B, np, H, W
        '''
        x2 = x ** 2  # B * C * H * W
        # convert (B, C, H, W) to shape (B, np, H, W)
        x2_patch_sum = F.conv2d(input=x2, weight=self.prototype_ones)  # B * np * H * W

        p2 = self.prototype_vectors ** 2  # np * C * 1 * 1
        # convert p2 to shape (np, 1, 1)
        p2 = torch.sum(p2, dim=(1, 2, 3))  # (np,) 
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)  # np, 1, 1

        xp = F.conv2d(input=x, weight=self.prototype_vectors)  # dot product result, B, np, H, W
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)  # B, np, H, W

        # distances.register_hook(hook_dist)  # not leaf

        return distances

    def cosine_simi(self, x):
        '''
        x: B C H W
        '''
        simis = []
        # proto: np C 1 1
        for i in range(self.num_prototypes):
            proto = self.prototype_vectors[i].view(1, self.prototype_dim, 1, 1)  # 1 C 1 1 1
            simi = self.cos_func(x, proto)  # B H W
            simis.append(simi)
        simis = torch.stack(simis, dim=1)  # B np H W

        return simis

    def prototype_distances(self, x):
        '''
        x is the raw input

        return: l2 distance (B, np, H, W)
        '''
        x = self.calc_featmap(x)
        if self.prototype_activation_function == 'cos':
            x = -self.cosine_simi(x)  # convert similarity to distance
        else:
            x = self._l2_convolution(x)
        return x

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        elif self.prototype_activation_function == 'cos':
            return -distances
        else:
            return self.prototype_activation_function(distances)
    
    def forward(self, x):
        x = self.prototype_distances(x)  # B, np, H, W
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling  B, np, 1, 1, 1
        min_distances = -F.max_pool2d(-x,
                            kernel_size=(x.size()[2],
                                        x.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)  # B, np
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)

        return logits, min_distances

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        x = self.calc_featmap(x)
        distances = self._l2_convolution(x)  # B, np, H, W
        return x, distances

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.last_nonlinear == 0:
            self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


if __name__ == '__main__':
    from thop import profile
    from torchviz import make_dot
    from torchsummary import summary
    from _backbones import create_backbone
    backbone = create_backbone(backbone_name='C3D')
    model = ImagePNet(backbone)
    summary(model, input_size=[(3, 16, 224, 224)], batch_size=1, device="cpu")
    # for m in model.modules():
    #         print(m)
    # for name, para in model.named_parameters():
    #     print(name, ':')
    model = torch.nn.DataParallel(model)
    print('add_on_layers:  ', model.module.add_on_layers)
    # inputs = torch.ones(1, 3, 16, 224, 224)
    # flops, paras = profile(model=model, inputs=(inputs,))
    # print('flops:', flops)
    # print('params:', paras)
