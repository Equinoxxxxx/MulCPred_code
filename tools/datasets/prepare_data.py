import pickle
import os
from tqdm import tqdm

from .TITAN import TITAN_dataset
from .PIE_JAAD import PIEDataset
from .nuscenes import NuscDataset


def save_imgnm_to_objid_to_info(dataset_name):
    if dataset_name == 'TITAN':
        TITAN_dataset(sub_set='all',)
    elif dataset_name == 'PIE':
        PIEDataset(dataset_name='PIE',
                   subset='all')
    elif dataset_name == 'JAAD':
        PIEDataset(dataset_name='JAAD',
                   subset='all')
    elif dataset_name == 'nuscenes':
        # files for train set and val set are separate
        NuscDataset(subset='train')
        NuscDataset(subset='val')

    return

def check_imgnm_to_objid(dataset_name):
    if dataset_name == 'TITAN':
        crop_root = \
            '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/cropped_images/even_padded/224w_by_224h'
        imgnm_to_objid_path = \
            '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/imgnm_to_objid_to_ann.pkl'
        with open(imgnm_to_objid_path, 'rb') as f:
            imgnm_to_objid_to_info = pickle.load(f)
        n_img = 0
        for cid in tqdm(imgnm_to_objid_to_info):
            for imgnm in imgnm_to_objid_to_info[cid]:
                n_img += 1
                for obj_type in imgnm_to_objid_to_info[cid][imgnm]:
                    for oid in imgnm_to_objid_to_info[cid][imgnm][obj_type]:
                        if not os.path.exists(
                            os.path.join(crop_root, 
                                        obj_type, 
                                        cid, 
                                        oid, 
                                        imgnm)
                        ):
                            print(
                                'img does not exist: ' + \
                                    (obj_type, 
                                     cid, 
                                     oid, 
                                     imgnm, 
                                     imgnm_to_objid_to_info[cid][imgnm][obj_type][oid])
                                )

        print(f'num img {n_img}')
    elif dataset_name == 'PIE':
        # pedestrian
        crop_root_ped = \
            '/home/y_feng/workspace6/datasets/PIE_dataset/cropped_images/even_padded/224w_by_224h'
        crop_root_veh = \
            '/home/y_feng/workspace6/datasets/PIE_dataset/cropped_images_veh/even_padded/224w_by_224h'
            
        imgnm_to_objid_path = \
            '//home/y_feng/workspace6/datasets/PIE_dataset/imgnm_to_objid_to_ann.pkl'
        with open(imgnm_to_objid_path, 'rb') as f:
            imgnm_to_objid_to_info = pickle.load(f)
        n_img = 0
        for set_id in tqdm(imgnm_to_objid_to_info):
            for vid_id in imgnm_to_objid_to_info[set_id]:
                for imgnm in imgnm_to_objid_to_info[set_id][vid_id]:
                    n_img += 1
                    for obj_type in imgnm_to_objid_to_info[set_id][vid_id][imgnm]:
                        for oid in imgnm_to_objid_to_info\
                            [set_id][vid_id][imgnm][obj_type]:
                            if obj_type == 'ped':
                                img_path = os.path.join(
                                    crop_root_ped,
                                    '_'.join(set_id, vid_id, oid),
                                    imgnm
                                )
                            elif obj_type == 'veh':
                                img_path = os.path.join(
                                    crop_root_veh,
                                    set_id,
                                    vid_id,
                                    oid,
                                    imgnm
                                )
                            else:
                                raise ValueError(obj_type)
                            if not os.path.exists(img_path):
                                print(f'img does not exist: \
                                      {obj_type, set_id, vid_id, oid, imgnm}')
    elif dataset_name == 'nuscenes':
        crop_root = '/home/y_feng/workspace6/datasets/nusc/extra/cropped_images/CAM_FRONT/even_padded/224w_by_224h'
        imgnm_to_objid_paths = ('/home/y_feng/workspace6/datasets/nusc/extra/train_imgnm_to_objid_to_ann.pkl',
                                '/home/y_feng/workspace6/datasets/nusc/extra/val_imgnm_to_objid_to_ann.pkl')
        for dict_path in imgnm_to_objid_paths:
            with open(dict_path, 'rb') as f:
                imgnm_to_objid_to_info = pickle.load(f)
            n_img = 0
            for sam_id in tqdm(imgnm_to_objid_to_info):
                n_img += 1
                img_nm = sam_id + '.png'
                for obj_type in imgnm_to_objid_to_info[sam_id]:
                    for ins_id in imgnm_to_objid_to_info[sam_id][obj_type]:
                        img_path = os.path.join(crop_root,
                                                obj_type,
                                                ins_id,
                                                img_nm,
                                                )
                        if not os.path.exists(img_path):
                            print('img does not exist' + \
                                  (obj_type, ins_id, sam_id))
        print(f'num img {n_img}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='TITAN')
    args = parser.parse_args()
    save_imgnm_to_objid_to_info(args.dataset_name)
    check_imgnm_to_objid(args.dataset_name)