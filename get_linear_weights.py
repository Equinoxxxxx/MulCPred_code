import os
import torch
import argparse
import pickle

from tools.utils import makedir
from models.SLENN import MultiSLE


def main():
    ckpt_path = '/home/y_feng/workspace6/work/MulCPred/results/mulcpred/lam_0/TITAN_atomic/ckpt/ckpt.pth'
    args_path = ckpt_path.replace('ckpt/ckpt.pth', 'args.pkl')
    with open(args_path, 'rb') as f:
        args = pickle.load(f)


    
    concept_dir = ckpt_path.replace('ckpt/ckpt.pth', 'concepts')
    concept_dir = os.path.join(concept_dir, os.listdir(concept_dir)[0])
    weight_dir = os.path.join(concept_dir, 'last_weight')
    makedir(weight_dir)
    weight_path = os.path.join(weight_dir, 'last_weight.txt')

    model = torch.load(ckpt_path)
    if hasattr(model, 'last_layer'):
        weight = model.last_layer.weight
    else:
        weight = model.atomic_layer.weight
    print(weight.size())
    with open(weight_path, 'w') as f:
        f.write(str(weight))


if __name__ == '__main__':
    main()