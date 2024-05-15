import cv2
import numpy as np
import os
import pickle
import json
from tqdm import tqdm
from ..utils import makedir
from config import dataset_root, cktp_root

from simpleHRNet.misc.visualization import draw_points_and_skeleton, joints_dict
from simpleHRNet.SimpleHRNet import SimpleHRNet


# skeletons
def generate_one_pseudo_heatmap(img_h, img_w, centers, max_values, sigma=0.6, eps=1e-4):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The max values of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for center, max_value in zip(centers, max_values):
            mu_x, mu_y = center[0], center[1]
            if max_value < eps:
                continue

            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            heatmap[st_y:ed_y,
                    st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x],
                                            patch)

        return heatmap

def extract_single(model, img_path, dataset='coco'):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # print(image.shape)
    heatmaps, _joints = model.predict(image)
    joints = _joints[0]
    heatmap_img = draw_points_and_skeleton(image, joints, skeleton=joints_dict()[dataset]['skeleton'], 
                                            points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                             points_palette_samples=17, confidence_threshold=0.)
    # pdb.set_trace()
    return heatmap_img, joints, heatmaps[0]

def get_skeletons_PIE():
    imgroot = os.path.join(dataset_root,
                           '')


def get_skeletons_nuscenes(img_root=os.path.join(dataset_root, 
                                                 'nusc/extra/cropped_images/CAM_FRONT/even_padded/288w_by_384h/human'),
                            tgt_root=os.path.join(dataset_root, 'nusc/extra')):
    format = 'coco'
    pseudo_h = 48
    pseudo_w = 48
    sk_vis_root = os.path.join(tgt_root, 
                               'sk_vis',
                               'even_padded',
                               '288w_by_384h')
    sk_pseudo_root = os.path.join(tgt_root, 
                               'sk_pseudo_heatmaps',
                               'even_padded',
                               '48w_by_48h')
    sk_heatmap_root = os.path.join(tgt_root, 
                               'sk_heatmaps',
                               'even_padded',
                               '288w_by_384h')
    sk_coord_root = os.path.join(tgt_root, 
                               'sk_coords',
                               'even_padded',
                               '288w_by_384h')
    # with open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_instance_id_to_token.pkl', 'rb') as f:
    #     insid_to_token = pickle.load(f)
    # with open('/home/y_feng/workspace6/datasets/nusc/extra/token_id/trainval_sample_id_to_token.pkl', 'rb') as f:
    #     samid_to_token = pickle.load(f)
    
    model = SimpleHRNet(c=48, nof_joints=17, 
                        checkpoint_path=os.path.join(cktp_root, "pose_hrnet_w48_384x288.pth"), 
                        multiperson=False, return_heatmaps=True, resolution=(384, 288))
    for insid in tqdm(os.listdir(img_root)):
        img_ins_path = os.path.join(img_root, insid)
        vis_ins_path = os.path.join(sk_vis_root, insid)
        makedir(vis_ins_path)
        pseudo_hm_ins_path = os.path.join(sk_pseudo_root, insid)
        makedir(pseudo_hm_ins_path)
        hm_ins_path = os.path.join(sk_heatmap_root, insid)
        makedir(hm_ins_path)
        coord_ins_path = os.path.join(sk_coord_root, insid)
        makedir(coord_ins_path)
        for img_nm in os.listdir(img_ins_path):
            img_path = os.path.join(img_ins_path, img_nm)
            samid = img_nm.replace('.png', '')
            # get sk_img, coords, heatmap
            skeleton_img, coords, heatmaps = extract_single(model=model, img_path=img_path, 
                                          dataset=format)
            assert skeleton_img.shape == (384, 288, 3), skeleton_img.shape
            # get pseudo heatmap
            ori_h, ori_w = skeleton_img.shape[:2]
            h_ratio = pseudo_h / ori_h
            w_ratio = pseudo_w / ori_w
            pseudo_heatmaps = []
            for coord in coords:  # x, y, confidence
                tgt_h = int(coord[0] * h_ratio)
                tgt_w = int(coord[1] * w_ratio)
                tgt_coord = (tgt_w, tgt_h)
                pseudo_heatmap = generate_one_pseudo_heatmap(img_h=pseudo_h,
                                                            img_w=pseudo_w,
                                                            centers=[tgt_coord],
                                                            max_values=[coord[-1]])
                pseudo_heatmaps.append(pseudo_heatmap)
            pseudo_heatmaps = np.stack(pseudo_heatmaps, axis=0)
            assert pseudo_heatmaps.shape == (17, pseudo_h, pseudo_w), pseudo_heatmaps.shape
            # save pseudo heatmaps, heatmaps, coords, sk img
            f_nm = samid + '.pkl'
            pseudo_hm_f_path = os.path.join(pseudo_hm_ins_path, f_nm)
            hm_f_path = os.path.join(hm_ins_path, f_nm)
            coord_f_path = os.path.join(coord_ins_path, f_nm)
            vis_f_path = os.path.join(vis_ins_path, img_nm)
            with open(pseudo_hm_f_path, 'wb') as f:
                pickle.dump(pseudo_heatmaps, f)
            with open(hm_f_path, 'wb') as f:
                pickle.dump(heatmaps, f)
            with open(coord_f_path, 'wb') as f:
                pickle.dump(coords, f)
            cv2.imwrite(vis_f_path, skeleton_img)
        
    # print saved content
    print(f'pseudo haetmap: shape{pseudo_heatmap.shape} {pseudo_heatmap}' )
    print(f'coords: {coords}')
    print(f'heatmap: shape {heatmaps.shape}')
    print(f'sk img: shape {skeleton_img.shape} {skeleton_img[0, 0, 0]}')

def get_skeleton_bdd100k(img_root=os.path.join(dataset_root, 
                                               'BDD100k/bdd100k/extra/cropped_images/even_padded/288w_by_384h/ped'),
                         tgt_root=os.path.join(dataset_root, 
                                               'BDD100k/bdd100k/extra/')):
    format = 'coco'
    pseudo_h = 48
    pseudo_w = 48
    sk_vis_root = os.path.join(tgt_root, 
                               'sk_vis',
                               'even_padded',
                               '288w_by_384h')
    sk_pseudo_root = os.path.join(tgt_root, 
                               'sk_pseudo_heatmaps',
                               'even_padded',
                               '48w_by_48h')
    sk_heatmap_root = os.path.join(tgt_root, 
                               'sk_heatmaps',
                               'even_padded',
                               '288w_by_384h')
    sk_coord_root = os.path.join(tgt_root, 
                               'sk_coords',
                               'even_padded',
                               '288w_by_384h')
    model = SimpleHRNet(c=48, nof_joints=17, 
                        checkpoint_path=os.path.join(cktp_root, "pose_hrnet_w48_384x288.pth"), 
                        multiperson=False, return_heatmaps=True, resolution=(384, 288))
    for oid in tqdm(os.listdir(img_root)):
        img_oid_dir = os.path.join(img_root, oid)
        makedir(os.path.join(sk_pseudo_root,
                                         oid))
        makedir(os.path.join(sk_heatmap_root,
                                        oid))
        makedir(os.path.join(sk_coord_root,
                                      oid))
        makedir(os.path.join(sk_vis_root,
                                      oid))
        for img_nm in os.listdir(img_oid_dir):
            img_id = img_nm.replace('.png', '')
            img_path = os.path.join(img_oid_dir, img_nm)
            # get sk_img, coords, heatmap
            skeleton_img, coords, heatmaps = extract_single(model=model, img_path=img_path, 
                                          dataset=format)
            assert skeleton_img.shape == (384, 288, 3), skeleton_img.shape
            # get pseudo heatmap
            ori_h, ori_w = skeleton_img.shape[:2]
            h_ratio = pseudo_h / ori_h
            w_ratio = pseudo_w / ori_w
            pseudo_heatmaps = []
            for coord in coords:  # x, y, confidence
                tgt_h = int(coord[0] * h_ratio)
                tgt_w = int(coord[1] * w_ratio)
                tgt_coord = (tgt_w, tgt_h)
                pseudo_heatmap = generate_one_pseudo_heatmap(img_h=pseudo_h,
                                                            img_w=pseudo_w,
                                                            centers=[tgt_coord],
                                                            max_values=[coord[-1]])
                pseudo_heatmaps.append(pseudo_heatmap)
            pseudo_heatmaps = np.stack(pseudo_heatmaps, axis=0)
            assert pseudo_heatmaps.shape == (17, pseudo_h, pseudo_w), pseudo_heatmaps.shape
            # save pseudo heatmaps, heatmaps, coords, sk img
            f_nm = img_id + '.pkl'
            pseudo_f_path = os.path.join(sk_pseudo_root,
                                         oid,
                                         f_nm)
            with open(pseudo_f_path, 'wb') as f:
                pickle.dump(pseudo_heatmaps, f)
            heatmap_path = os.path.join(sk_heatmap_root,
                                        oid,
                                        f_nm)
            with open(heatmap_path, 'wb') as f:
                pickle.dump(heatmaps, f)
            coord_path = os.path.join(sk_coord_root,
                                      oid,
                                      f_nm)
            with open(coord_path, 'wb') as f:
                pickle.dump(coords, f)
            vis_path = os.path.join(sk_vis_root,
                                    oid,
                                    img_nm)
            cv2.imwrite(vis_path, skeleton_img)


def get_skeletons(datasets):
    format = 'coco'
    pseudo_h = 48
    pseudo_w = 48
    model = SimpleHRNet(c=48, nof_joints=17, 
                        checkpoint_path=os.path.join(cktp_root, "pose_hrnet_w48_384x288.pth"), 
                        multiperson=False, return_heatmaps=True, resolution=(384, 288))
    print('Estimating pose')
    if 'PIE' in datasets:
        print('PIE')
        img_root=os.path.join(dataset_root, 
                                'PIE_dataset/cropped_images/even_padded/288w_by_384h'),
        tgt_root=os.path.join(dataset_root, 'PIE_dataset')
        sk_vis_root = os.path.join(tgt_root, 
                               'sk_vis',
                               'even_padded',
                               '288w_by_384h')
        sk_pseudo_root = os.path.join(tgt_root, 
                                'sk_pseudo_heatmaps',
                                'even_padded',
                                '48w_by_48h')
        sk_heatmap_root = os.path.join(tgt_root, 
                                'sk_heatmaps',
                                'even_padded',
                                '288w_by_384h')
        sk_coord_root = os.path.join(tgt_root, 
                                'sk_coords',
                                'even_padded',
                                '288w_by_384h')
        for pid in tqdm(os.listdir(img_root)):
            sid, vid, oid = pid.split('_')
            img_pid_dir = os.path.join(img_root, pid)
            for img_nm in os.listdir(img_pid_dir):
                img_id = img_nm.replace('.png', '')
                img_path = os.path.join(img_pid_dir, img_nm)
                # get sk_img, coords, heatmap
                skeleton_img, coords, heatmaps = extract_single(model=model, img_path=img_path, 
                                            dataset=format)
                assert skeleton_img.shape == (384, 288, 3), skeleton_img.shape
                # get pseudo heatmap
                ori_h, ori_w = skeleton_img.shape[:2]
                h_ratio = pseudo_h / ori_h
                w_ratio = pseudo_w / ori_w
                pseudo_heatmaps = []
                for coord in coords:  # x, y, confidence
                    tgt_h = int(coord[0] * h_ratio)
                    tgt_w = int(coord[1] * w_ratio)
                    tgt_coord = (tgt_w, tgt_h)
                    pseudo_heatmap = generate_one_pseudo_heatmap(img_h=pseudo_h,
                                                                img_w=pseudo_w,
                                                                centers=[tgt_coord],
                                                                max_values=[coord[-1]])
                    pseudo_heatmaps.append(pseudo_heatmap)
                pseudo_heatmaps = np.stack(pseudo_heatmaps, axis=0)
                assert pseudo_heatmaps.shape == (17, pseudo_h, pseudo_w), pseudo_heatmaps.shape
                # save pseudo heatmaps, heatmaps, coords, sk img
                f_nm = img_id + '.pkl'
                pseudo_f_path = os.path.join(sk_pseudo_root,
                                            pid,
                                            f_nm)
                with open(pseudo_f_path, 'wb') as f:
                    pickle.dump(pseudo_heatmaps, f)
                heatmap_path = os.path.join(sk_heatmap_root,
                                            pid,
                                            f_nm)
                with open(heatmap_path, 'wb') as f:
                    pickle.dump(heatmaps, f)
                coord_path = os.path.join(sk_coord_root,
                                        pid,
                                        f_nm)
                with open(coord_path, 'wb') as f:
                    pickle.dump(coords, f)
                vis_path = os.path.join(sk_vis_root,
                                        pid,
                                        img_nm)
                cv2.imwrite(vis_path, skeleton_img)
    if 'JAAD' in datasets:
        print('JAAD')
        img_root=os.path.join(dataset_root, 
                                'JAAD/cropped_images/even_padded/288w_by_384h'),
        tgt_root=os.path.join(dataset_root, 'JAAD')
        sk_vis_root = os.path.join(tgt_root, 
                               'sk_vis',
                               'even_padded',
                               '288w_by_384h')
        sk_pseudo_root = os.path.join(tgt_root, 
                                'sk_pseudo_heatmaps',
                                'even_padded',
                                '48w_by_48h')
        sk_heatmap_root = os.path.join(tgt_root, 
                                'sk_heatmaps',
                                'even_padded',
                                '288w_by_384h')
        sk_coord_root = os.path.join(tgt_root, 
                                'sk_coords',
                                'even_padded',
                                '288w_by_384h')
        for pid in tqdm(os.listdir(img_root)):
            sid, vid, oid = pid.split('_')
            img_pid_dir = os.path.join(img_root, pid)
            for img_nm in os.listdir(img_pid_dir):
                img_id = img_nm.replace('.png', '')
                img_path = os.path.join(img_pid_dir, img_nm)
                # get sk_img, coords, heatmap
                skeleton_img, coords, heatmaps = extract_single(model=model, img_path=img_path, 
                                            dataset=format)
                assert skeleton_img.shape == (384, 288, 3), skeleton_img.shape
                # get pseudo heatmap
                ori_h, ori_w = skeleton_img.shape[:2]
                h_ratio = pseudo_h / ori_h
                w_ratio = pseudo_w / ori_w
                pseudo_heatmaps = []
                for coord in coords:  # x, y, confidence
                    tgt_h = int(coord[0] * h_ratio)
                    tgt_w = int(coord[1] * w_ratio)
                    tgt_coord = (tgt_w, tgt_h)
                    pseudo_heatmap = generate_one_pseudo_heatmap(img_h=pseudo_h,
                                                                img_w=pseudo_w,
                                                                centers=[tgt_coord],
                                                                max_values=[coord[-1]])
                    pseudo_heatmaps.append(pseudo_heatmap)
                pseudo_heatmaps = np.stack(pseudo_heatmaps, axis=0)
                assert pseudo_heatmaps.shape == (17, pseudo_h, pseudo_w), pseudo_heatmaps.shape
                # save pseudo heatmaps, heatmaps, coords, sk img
                f_nm = img_id + '.pkl'
                pseudo_f_path = os.path.join(sk_pseudo_root,
                                            pid,
                                            f_nm)
                with open(pseudo_f_path, 'wb') as f:
                    pickle.dump(pseudo_heatmaps, f)
                heatmap_path = os.path.join(sk_heatmap_root,
                                            pid,
                                            f_nm)
                with open(heatmap_path, 'wb') as f:
                    pickle.dump(heatmaps, f)
                coord_path = os.path.join(sk_coord_root,
                                        pid,
                                        f_nm)
                with open(coord_path, 'wb') as f:
                    pickle.dump(coords, f)
                vis_path = os.path.join(sk_vis_root,
                                        pid,
                                        img_nm)
                cv2.imwrite(vis_path, skeleton_img)
    if 'TITAN' in datasets:
        print('TITAN')
        img_root=os.path.join(dataset_root, 
                                'TITAN/TITAN_extra/cropped_images/even_padded/288w_by_384h/ped'),
        tgt_root=os.path.join(dataset_root, 'TITAN/TITAN_extra')
        sk_vis_root = os.path.join(tgt_root, 
                               'sk_vis',
                               'even_padded',
                               '288w_by_384h')
        sk_pseudo_root = os.path.join(tgt_root, 
                                'sk_pseudo_heatmaps',
                                'even_padded',
                                '48w_by_48h')
        sk_heatmap_root = os.path.join(tgt_root, 
                                'sk_heatmaps',
                                'even_padded',
                                '288w_by_384h')
        sk_coord_root = os.path.join(tgt_root, 
                                'sk_coords',
                                'even_padded',
                                '288w_by_384h')
        for vid in os.listdir(img_root):
            vdir = os.path.join(img_root, vid)
            for oid in os.listdir(vdir):
                odir = os.path.join(vdir, oid)
                for img_nm in os.listdir(odir):
                    img_id = img_nm.replace('.png', '')
                    img_path = os.path.join(odir, img_nm)
                    # get sk_img, coords, heatmap
                    skeleton_img, coords, heatmaps = extract_single(model=model, img_path=img_path, 
                                                dataset=format)
                    assert skeleton_img.shape == (384, 288, 3), skeleton_img.shape
                    # get pseudo heatmap
                    ori_h, ori_w = skeleton_img.shape[:2]
                    h_ratio = pseudo_h / ori_h
                    w_ratio = pseudo_w / ori_w
                    pseudo_heatmaps = []
                    for coord in coords:  # x, y, confidence
                        tgt_h = int(coord[0] * h_ratio)
                        tgt_w = int(coord[1] * w_ratio)
                        tgt_coord = (tgt_w, tgt_h)
                        pseudo_heatmap = generate_one_pseudo_heatmap(img_h=pseudo_h,
                                                                    img_w=pseudo_w,
                                                                    centers=[tgt_coord],
                                                                    max_values=[coord[-1]])
                        pseudo_heatmaps.append(pseudo_heatmap)
                    pseudo_heatmaps = np.stack(pseudo_heatmaps, axis=0)
                    assert pseudo_heatmaps.shape == (17, pseudo_h, pseudo_w), pseudo_heatmaps.shape
                    # save pseudo heatmaps, heatmaps, coords, sk img
                    f_nm = img_id + '.pkl'
                    pseudo_f_path = os.path.join(sk_pseudo_root,
                                                vid,
                                                oid,
                                                f_nm)
                    with open(pseudo_f_path, 'wb') as f:
                        pickle.dump(pseudo_heatmaps, f)
                    heatmap_path = os.path.join(sk_heatmap_root,
                                                vid,
                                                oid,
                                                f_nm)
                    with open(heatmap_path, 'wb') as f:
                        pickle.dump(heatmaps, f)
                    coord_path = os.path.join(sk_coord_root,
                                            vid,
                                            oid,
                                            f_nm)
                    with open(coord_path, 'wb') as f:
                        pickle.dump(coords, f)
                    vis_path = os.path.join(sk_vis_root,
                                            vid,
                                            oid,
                                            img_nm)
                    cv2.imwrite(vis_path, skeleton_img)


    if 'nuscenes' in datasets:
        print('nuscenes')
        img_root=os.path.join(dataset_root, 
                                'nusc/extra/cropped_images/CAM_FRONT/even_padded/288w_by_384h/human'),
        tgt_root=os.path.join(dataset_root, 'nusc/extra')
        sk_vis_root = os.path.join(tgt_root, 
                               'sk_vis',
                               'even_padded',
                               '288w_by_384h')
        sk_pseudo_root = os.path.join(tgt_root, 
                                'sk_pseudo_heatmaps',
                                'even_padded',
                                '48w_by_48h')
        sk_heatmap_root = os.path.join(tgt_root, 
                                'sk_heatmaps',
                                'even_padded',
                                '288w_by_384h')
        sk_coord_root = os.path.join(tgt_root, 
                                'sk_coords',
                                'even_padded',
                                '288w_by_384h')
        for insid in tqdm(os.listdir(img_root)):
            img_ins_path = os.path.join(img_root, insid)
            vis_ins_path = os.path.join(sk_vis_root, insid)
            makedir(vis_ins_path)
            pseudo_hm_ins_path = os.path.join(sk_pseudo_root, insid)
            makedir(pseudo_hm_ins_path)
            hm_ins_path = os.path.join(sk_heatmap_root, insid)
            makedir(hm_ins_path)
            coord_ins_path = os.path.join(sk_coord_root, insid)
            makedir(coord_ins_path)
            for img_nm in os.listdir(img_ins_path):
                img_path = os.path.join(img_ins_path, img_nm)
                samid = img_nm.replace('.png', '')
                # get sk_img, coords, heatmap
                skeleton_img, coords, heatmaps = extract_single(model=model, img_path=img_path, 
                                            dataset=format)
                assert skeleton_img.shape == (384, 288, 3), skeleton_img.shape
                # get pseudo heatmap
                ori_h, ori_w = skeleton_img.shape[:2]
                h_ratio = pseudo_h / ori_h
                w_ratio = pseudo_w / ori_w
                pseudo_heatmaps = []
                for coord in coords:  # x, y, confidence
                    tgt_h = int(coord[0] * h_ratio)
                    tgt_w = int(coord[1] * w_ratio)
                    tgt_coord = (tgt_w, tgt_h)
                    pseudo_heatmap = generate_one_pseudo_heatmap(img_h=pseudo_h,
                                                                img_w=pseudo_w,
                                                                centers=[tgt_coord],
                                                                max_values=[coord[-1]])
                    pseudo_heatmaps.append(pseudo_heatmap)
                pseudo_heatmaps = np.stack(pseudo_heatmaps, axis=0)
                assert pseudo_heatmaps.shape == (17, pseudo_h, pseudo_w), pseudo_heatmaps.shape
                # save pseudo heatmaps, heatmaps, coords, sk img
                f_nm = samid + '.pkl'
                pseudo_hm_f_path = os.path.join(pseudo_hm_ins_path, f_nm)
                hm_f_path = os.path.join(hm_ins_path, f_nm)
                coord_f_path = os.path.join(coord_ins_path, f_nm)
                vis_f_path = os.path.join(vis_ins_path, img_nm)
                with open(pseudo_hm_f_path, 'wb') as f:
                    pickle.dump(pseudo_heatmaps, f)
                with open(hm_f_path, 'wb') as f:
                    pickle.dump(heatmaps, f)
                with open(coord_f_path, 'wb') as f:
                    pickle.dump(coords, f)
                cv2.imwrite(vis_f_path, skeleton_img)
            
        # print saved content
        print(f'pseudo haetmap: shape{pseudo_heatmap.shape} {pseudo_heatmap}' )
        print(f'coords: {coords}')
        print(f'heatmap: shape {heatmaps.shape}')
        print(f'sk img: shape {skeleton_img.shape} {skeleton_img[0, 0, 0]}')
    if 'bdd100k' in datasets:
        print('bdd100k')
        img_root=os.path.join(dataset_root, 
                            'BDD100k/bdd100k/extra/cropped_images/even_padded/288w_by_384h/ped'),
        tgt_root=os.path.join(dataset_root, 
                            'BDD100k/bdd100k/extra/')
        sk_vis_root = os.path.join(tgt_root, 
                               'sk_vis',
                               'even_padded',
                               '288w_by_384h')
        sk_pseudo_root = os.path.join(tgt_root, 
                                'sk_pseudo_heatmaps',
                                'even_padded',
                                '48w_by_48h')
        sk_heatmap_root = os.path.join(tgt_root, 
                                'sk_heatmaps',
                                'even_padded',
                                '288w_by_384h')
        sk_coord_root = os.path.join(tgt_root, 
                                'sk_coords',
                                'even_padded',
                                '288w_by_384h')
        for oid in tqdm(os.listdir(img_root)):
            img_oid_dir = os.path.join(img_root, oid)
            makedir(os.path.join(sk_pseudo_root,
                                            oid))
            makedir(os.path.join(sk_heatmap_root,
                                            oid))
            makedir(os.path.join(sk_coord_root,
                                        oid))
            makedir(os.path.join(sk_vis_root,
                                        oid))
            for img_nm in os.listdir(img_oid_dir):
                img_id = img_nm.replace('.png', '')
                img_path = os.path.join(img_oid_dir, img_nm)
                # get sk_img, coords, heatmap
                skeleton_img, coords, heatmaps = extract_single(model=model, img_path=img_path, 
                                            dataset=format)
                assert skeleton_img.shape == (384, 288, 3), skeleton_img.shape
                # get pseudo heatmap
                ori_h, ori_w = skeleton_img.shape[:2]
                h_ratio = pseudo_h / ori_h
                w_ratio = pseudo_w / ori_w
                pseudo_heatmaps = []
                for coord in coords:  # x, y, confidence
                    tgt_h = int(coord[0] * h_ratio)
                    tgt_w = int(coord[1] * w_ratio)
                    tgt_coord = (tgt_w, tgt_h)
                    pseudo_heatmap = generate_one_pseudo_heatmap(img_h=pseudo_h,
                                                                img_w=pseudo_w,
                                                                centers=[tgt_coord],
                                                                max_values=[coord[-1]])
                    pseudo_heatmaps.append(pseudo_heatmap)
                pseudo_heatmaps = np.stack(pseudo_heatmaps, axis=0)
                assert pseudo_heatmaps.shape == (17, pseudo_h, pseudo_w), pseudo_heatmaps.shape
                # save pseudo heatmaps, heatmaps, coords, sk img
                f_nm = img_id + '.pkl'
                pseudo_f_path = os.path.join(sk_pseudo_root,
                                            oid,
                                            f_nm)
                with open(pseudo_f_path, 'wb') as f:
                    pickle.dump(pseudo_heatmaps, f)
                heatmap_path = os.path.join(sk_heatmap_root,
                                            oid,
                                            f_nm)
                with open(heatmap_path, 'wb') as f:
                    pickle.dump(heatmaps, f)
                coord_path = os.path.join(sk_coord_root,
                                        oid,
                                        f_nm)
                with open(coord_path, 'wb') as f:
                    pickle.dump(coords, f)
                vis_path = os.path.join(sk_vis_root,
                                        oid,
                                        img_nm)
                cv2.imwrite(vis_path, skeleton_img)