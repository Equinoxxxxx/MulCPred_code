import os
import pickle
import numpy as np
from helpers import makedir


def split_skeletons(dataset_name, mode='heatmap'):
    if dataset_name == 'PIE':
        root = '/home/y_feng/workspace6/datasets/PIE_dataset/skeletons/even_padded/288w_by_384h/'
        if mode == 'heatmap':
            tgt_root = '/home/y_feng/workspace6/datasets/PIE_dataset/sk_heatmaps/even_padded/288w_by_384h/'
        else:
            tgt_root = '/home/y_feng/workspace6/datasets/PIE_dataset/sk_coords/even_padded/288w_by_384h/'
    elif dataset_name == 'JAAD':
        root = '/home/y_feng/workspace6/datasets/JAAD/skeletons/even_padded/288w_by_384h/'
        if mode == 'heatmap':
            tgt_root = '/home/y_feng/workspace6/datasets/JAAD/sk_heatmaps/even_padded/288w_by_384h/'
        else:
            tgt_root = '/home/y_feng/workspace6/datasets/JAAD/sk_coords/even_padded/288w_by_384h/'
    elif dataset_name == 'TITAN':
        root = '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/skeletons/even_padded/288w_by_384h'
        if mode == 'heatmap':
            tgt_root = '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/sk_heatmaps/even_padded/288w_by_384h/'
        else:
            tgt_root = '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/sk_coords/even_padded/288w_by_384h/'
    else:
        raise NotImplementedError(dataset_name)
        
    if not os.path.exists(tgt_root):
        os.mkdir(tgt_root)
    
    if dataset_name in ('PIE', 'JAAD'):
        for d in os.listdir(root):
            dir = os.path.join(root, d)
            img_nm_list = []
            for n in os.listdir(dir):
                if '.png' in n:
                    img_nm_list.append(n.replace('.png', '.pkl'))
            tgt_dir = os.path.join(tgt_root, d)
            if not os.path.exists(tgt_dir):
                os.mkdir(tgt_dir)
            if mode == 'heatmap':
                src_f = os.path.join(dir, 'heatmaps.pkl')  # seq len, 17, 96, 72
            else:
                src_f = os.path.join(dir, 'skeletons.pkl')  # seq len, 17, 3(x,y,conf)
            with open(src_f, 'rb') as f:
                skeletons = pickle.load(f)
            assert skeletons.shape[0] == len(img_nm_list), (skeletons.shape, len(img_nm_list))
            for i in range(len(img_nm_list)):
                nm = img_nm_list[i]
                tgt_f = os.path.join(tgt_dir, nm)
                heatmap = skeletons[i]
                with open(tgt_f, 'wb') as f:
                    pickle.dump(heatmap, f)
            print(tgt_dir, ' done')
    elif dataset_name == 'TITAN':
        for cid in os.listdir(root):
            cid_dir = os.path.join(root, cid)
            tgt_cid_dir = os.path.join(tgt_root, cid)
            makedir(tgt_cid_dir)
            for pid in os.listdir(cid_dir):
                pid_dir = os.path.join(cid_dir, pid)
                tgt_pid_dir = os.path.join(tgt_cid_dir, pid)
                makedir(tgt_pid_dir)
                img_nm_list = []
                for n in os.listdir(pid_dir):
                    if '.png' in n:
                        img_nm_list.append(n.replace('.png', '.pkl'))
                if mode == 'heatmap':
                    src_f = os.path.join(pid_dir, 'heatmaps.pkl')  # seq len, 17, 96, 72
                else:
                    src_f = os.path.join(pid_dir, 'skeletons.pkl')  # seq len, 17, 3(x,y,conf)
                with open(src_f, 'rb') as f:
                    skeletons = pickle.load(f)
                assert skeletons.shape[0] == len(img_nm_list), (skeletons.shape, len(img_nm_list))
                for i in range(len(img_nm_list)):
                    nm = img_nm_list[i]
                    tgt_f = os.path.join(tgt_pid_dir, nm)
                    data = skeletons[i]
                    with open(tgt_f, 'wb') as f:
                        pickle.dump(data, f)
                print(tgt_pid_dir, ' done')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='PIE')
    parser.add_argument('--mode', type=str, default='heatmap')
    args = parser.parse_args()
    split_skeletons(args.dataset_name, args.mode)