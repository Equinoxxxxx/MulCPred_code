import torch
import pickle
import os
from tools.plot import *
from tools.metrics import calc_auc_morf

def draw_morf_from_logits():
    # # TITAN SLE SENN
    # logits_path1 = '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models_1/SLE_traj_ego_img_skeleton_context/26Jan2023-02h41m18s/test/22Feb2023-15h47m13s/SLE.pkl'  # 0.23 0.03
    # logits_path2 = '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models_1/SENN_traj_ego_img_skeleton_context/16Jan2023-14h19m52s/test/23Feb2023-20h43m02s/SENN.pkl'  # 0.86 0.22
    # logits_path3 = ''
    # PIE SLE SENN
    logits_path1 = '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models_1/SLE_traj_ego_img_skeleton_context/06Feb2023-04h30m29s/test/07Feb2023-00h12m57s/SLE.pkl'  # 0.17 0.05
    logits_path2 = ''  # 0.45 0.34
    logits_path3 = ''
    labels = ['Ours-not crossing', 'Ours-crossing']
    # # TITAN PCPA
    # logits_path1 = '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models_1/PCPA_traj_ego_img_skeleton_context/31Jan2023-12h07m21s/test/23Feb2023-21h57m09s/PCPA.pkl'  # 0.36 0.43
    # logits_path2 = ''
    # logits_path3 = ''
    # # PIE PCPA
    # logits_path1 = '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models_1/PCPA_traj_ego_img_skeleton_context/01Feb2023-05h07m42s/test/23Feb2023-22h00m00s/PCPA.pkl'  # 0.46 0.33
    # logits_path2 = ''
    # logits_path3 = ''
    # labels = ['PCPA-not crossing', 'PCPA-crossing']
    test_dir = '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models_1/SLE_traj_ego_img_skeleton_context/26Jan2023-02h41m18s/test/'
    with open(logits_path1, 'rb') as f:
        morf_logits1 = pickle.load(f)
    # normalize
    morf_logits1 -= torch.min(morf_logits1)
    morf_logits1 /= torch.max(morf_logits1)
    if logits_path2:
        with open(logits_path2, 'rb') as f:
            morf_logits2 = pickle.load(f)
            # normalize
            morf_logits2 -= torch.min(morf_logits2)
            morf_logits2 /= torch.max(morf_logits2)
    if logits_path3:
        with open(logits_path3, 'rb') as f:
            morf_logits3 = pickle.load(f)
            # normalize
            morf_logits3 -= torch.min(morf_logits3)
            morf_logits3 /= torch.max(morf_logits3)
    num_classes = morf_logits1.size(0)

    auc_morf = torch.zeros(num_classes)
    auc_morf_norm = torch.zeros(num_classes)
    morf_logit_list_both = []
    for c in range(morf_logits1.size(0)):
        morf_logit_list = [morf_logits1[c]]
        morf_logit_list_both.append(morf_logits1[c])
        if logits_path2:
            morf_logit_list.append(morf_logits2[c])
            morf_logit_list_both.append(morf_logits2[c])
        if logits_path3:
            morf_logit_list.append(morf_logits3[c])
            morf_logit_list_both.append(morf_logits3[c])
        curve_path = os.path.join(test_dir, '_morf_curve_cls'+str(c)+'.png')
        # draw_morf(morf_logits1[c], curve_path)
        draw_morfs(morf_logit_list, curve_path)
        auc_morf[c] = calc_auc_morf(morf_logits1[c])
        print(torch.max(morf_logits1[c]), torch.min(morf_logits1[c]))
        auc_morf_norm[c] = (auc_morf[c] - torch.min(morf_logits1[c])) / (torch.max(morf_logits1[c]) - torch.min(morf_logits1[c]))
    draw_morfs_both(morf_logit_list_both, os.path.join(test_dir, 'morf_curve_both.png'), labels=labels)
    print(f'\tauc-morf: {auc_morf} \tauc-morf norm: {auc_morf_norm}')
    print(f'Res saved in {test_dir}')


if __name__ == '__main__':
    draw_morf_from_logits()