import torch

if __name__ == '__main__':
    model_path = '/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models_1/SLE_traj_ego_img_skeleton_context/17Jan2023-12h43m49s/ckpt/64_0.0000.pth'
    model = torch.load(model_path)
    abs_weights = model.last_layer.weight[1, :50].abs()  # np
    sorted, idcs = torch.sort(abs_weights, descending=True)
    print(sorted)
    print(idcs)