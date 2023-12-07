import os
import pickle
import shutil
import time
from turtle import resizemode
from tqdm import tqdm
import cv2

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re
import copy
import numpy as np

from _datasets import PIEDataset
from helpers import makedir, draw_curves
from _proto_model import ImagePNet, SkeletonPNet, MultiPNet, MultiBackbone, NonlocalMultiPNet
from _backbones import BackboneOnly, C3DEncoderDecoder
from _backbones import create_backbone, record_conv3d_info, record_conv2d_info, record_sp_conv3d_info_w, record_t_conv3d_info, record_sp_conv2d_info_h, record_sp_conv2d_info_w
import _project_prototypes
import _project_sk_prototypes
import prune
import _multi_train_test as tnt
import save
from log import create_logger
from preprocess import preprocess_input_function
from receptive_field import compute_proto_layer_rf_info_v2
from utils import draw_proto_info_curves

def main():
    parser = argparse.ArgumentParser()

    # optimizer setting
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dataset_name', type=str, default='PIE')
    parser.add_argument('--cross_dataset_name', type=str, default='JAAD')
    parser.add_argument('--cross_dataset', type=int, default=0)
    parser.add_argument('--small_set', type=float, default=0)
    parser.add_argument('--test_every', type=int, default=10)

    parser.add_argument('--obs_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=16)

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    dataset_name = args.dataset_name
    cross_dataset = args.cross_dataset
    cross_dataset_name = args.cross_dataset_name
    small_set = args.small_set
    test_every = args.test_every

    obs_len = args.obs_len
    pred_len = args.pred_len

    # default device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('Default device: ', os.environ['CUDA_VISIBLE_DEVICES'])

    # exp dir
    exp_id = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    work_dir = '../work_dirs/models/pretrain'
    model_type = 'vid_pred'
    model_dir = os.path.join(work_dir, model_type, exp_id)
    print('Save dir of current exp: ', model_dir)
    makedir(model_dir)
    ckpt_dir = os.path.join(model_dir, 'ckpt')
    makedir(ckpt_dir)

    vis_dir = os.path.join(model_dir, 'vis_res')
    makedir(vis_dir)
    
    # logging
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    log('--------args----------')
    for k in list(vars(args).keys()):
        log(str(k)+': '+str(vars(args)[k]))
    log('--------args----------\n')

    
    # dataset
    # # load the data
    print('----------------------------Load data-----------------------------')
    # train set
    train_dataset = PIEDataset(dataset_name=dataset_name, obs_len=16, pred_len=16, do_balance=False, subset='train',
                                use_img=0, 
                                use_skeleton=0,
                                use_context=1, ctx_mode='ori',
                                pred_context=1, pred_context_mode='ori',
                                use_single_img=0,
                                small_set=small_set,
                                overlap_retio=0.5)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=1,
                                                num_workers=8, pin_memory=False)
    # test set
    test_dataset = PIEDataset(dataset_name=dataset_name, obs_len=16, pred_len=16, do_balance=False, subset='test',
                                use_img=0, 
                                use_skeleton=0,
                                use_context=1, ctx_mode='ori',
                                pred_context=1, pred_context_mode='ori',
                                use_single_img=0,
                                small_set=small_set,
                                overlap_retio=0.5)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                                shuffle=1,
                                                num_workers=8, pin_memory=False)
    # cross set
    if cross_dataset:
        cross_dataset = PIEDataset(dataset_name=cross_dataset_name, obs_len=16, pred_len=16, do_balance=False, subset='test',
                                    use_img=0, 
                                    use_skeleton=0,
                                    use_context=1, ctx_mode='ori',
                                    pred_context=1, pred_context_mode='ori',
                                    use_single_img=0,
                                    small_set=small_set,
                                    overlap_retio=0.5)
        cross_loader = torch.utils.data.DataLoader(cross_dataset, batch_size=batch_size, 
                                                    shuffle=1,
                                                    num_workers=8, pin_memory=False)
        
    # construct the model
    log('----------------------------Construct model-----------------------------')
    model = C3DEncoderDecoder()
    model = model.to(device)
    model_parallel = torch.nn.DataParallel(model)

    # define optimizer
    log('----------------------------Construct optimizer-----------------------------')
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # train the model
    log('----------------------------Start training-----------------------------')

    for e in range(1, epochs+1):
        total_test_loss = 0
        total_train_loss = 0
        # train
        log('train')
        tbar = tqdm(train_loader, miniters=1)
        for i, data in enumerate(tbar):
            inputs = data['obs_context'].to(device)
            gts = data['pred_context'].to(device)
            output = model_parallel(inputs)
            loss = mse(output, gts)
            total_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            display_dict = {'loss': loss.item(),
                            'total loss': total_train_loss
                            }
            tbar.set_postfix(display_dict)
        log('train loss: ' + str(total_train_loss))
        tbar.close()
        # test
        
        if e % test_every == 0:
            log('test')
            cur_vis_dir = os.path.join(vis_dir, 'epoch'+str(e))
            makedir(cur_vis_dir)

            tbar = tqdm(test_loader, miniters=1)
            for i, data in enumerate(tbar):
                inputs = data['obs_context'].to(device)
                gts = data['pred_context'].to(device)
                output = model_parallel(inputs)
                loss = mse(output, gts)
                total_test_loss += loss.item()
                display_dict = {'loss': loss.item(),
                                'total loss': total_test_loss
                                }
                tbar.set_postfix(display_dict)
                visualize_vid(output_tensor=output[0], vis_dir=cur_vis_dir)
            log('saving model ' + str(total_test_loss))
            torch.save(obj=model, f=os.path.join(ckpt_dir, str(e)+'_'+str(total_test_loss)+'.pth'))

            tbar.close()
            log('test loss: ' + str(total_test_loss))


def visualize_vid(output_tensor, norm_mode='torch', vis_dir='', color_order='BGR'):
    # output_tensor: 3THW -> THW3
    output_tensor = output_tensor.permute(1, 2, 3, 0).cpu().detach().numpy()
    if color_order == 'RGB':
        output_tensor = output_tensor[:, :, :, ::-1]
    if norm_mode == 'tf':
        output_tensor += 1.
        output_tensor *= 127.5
    elif norm_mode == 'torch':
        output_tensor[:,:,:,0] *= 0.225
        output_tensor[:,:,:,1] *= 0.224
        output_tensor[:,:,:,2] *= 0.229
        output_tensor[:,:,:,0] += 0.406
        output_tensor[:,:,:,1] += 0.456
        output_tensor[:,:,:,2] += 0.485
        output_tensor *= 255.
    for i in range(output_tensor.shape[0]):
        img = output_tensor[i]
        cv2.imwrite(os.path.join(vis_dir, str(i)+'.png'), img)

if __name__ == '__main__':
    main()