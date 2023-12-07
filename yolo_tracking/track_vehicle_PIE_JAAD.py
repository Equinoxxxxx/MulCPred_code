# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
import sys
import argparse
from functools import partial
from pathlib import Path
import os
import pickle
import cv2
from tqdm import tqdm

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from examples.detectors import get_yolo_inferer

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

from examples.utils import write_mot_results
from tools.utils import makedir
from tools.data.preprocess import crop_img

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):
    # import pdb;pdb.set_trace()
    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        stream=True,
        device=DEVICE,
        show_conf=args.show_conf,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=[2, 5, 7],
        imgsz=args.imgsz
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if 'yolov8' not in str(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args

    for frame_idx, r in enumerate(results):
        if r.boxes.data.shape[1] == 7:
            import pdb;pdb.set_trace()
            if yolo.predictor.source_type.webcam \
                or args.source.endswith(VID_FORMATS):
                p = yolo.predictor.save_dir / 'mot' / (args.source + '.txt')
                yolo.predictor.mot_txt_path = p
            elif 'MOT16' or 'MOT17' or 'MOT20' in args.source:
                p = yolo.predictor.save_dir / 'mot' / (Path(args.source).parent.name + '.txt')
                yolo.predictor.mot_txt_path = p

            if args.save_mot:
                write_mot_results(
                    yolo.predictor.mot_txt_path,
                    r,
                    frame_idx,
                )

            if args.save_id_crops:
                for d in r.boxes:
                    print('args.save_id_crops', d.data)
                    save_one_box(
                        d.xyxy,
                        r.orig_img.copy(),
                        file=(
                            yolo.predictor.save_dir / 
                            'crops' /
                            str(int(d.cls.cpu().numpy().item())) /
                            str(int(d.id.cpu().numpy().item())) / 
                            f'{frame_idx}.png'
                        ),
                        BGR=True
                    )

    if args.save_mot:
        print(f'MOT results saved to {yolo.predictor.mot_txt_path}')

@torch.no_grad()
def track_PIE_JAAD_veh():
    args = parse_opt()
    v_classes = [2, 5, 7]
    # crop size
    if isinstance(args.crop_size, int):
        crop_size = (args.crop_size, args.crop_size)
    elif isinstance(args.crop_size, list):
        assert len(args.crop_size) == 2, args.crop_size
        crop_size = args.crop_size
    # init model
    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    # track vehicles in PIE
    if args.dataset_name == 'PIE':
        # (set id) --> vid id --> obj id --> bbox seq & img nm seq
        cid_to_oid = {}
        track_info_path = \
            '/home/y_feng/workspace6/datasets/PIE_dataset/veh_tracks.pkl'
        img_root = \
            '/home/y_feng/workspace6/datasets/PIE_dataset/images'
        crop_veh_root = \
            os.path.join('/home/y_feng/workspace6/datasets/PIE_dataset/cropped_images_veh', 
                            args.resize_mode, 
                            f'{crop_size[0]}w_by_{crop_size[1]}h'
                            )
        makedir(crop_veh_root)
        start_id = 0
        print('Tracking vehicles in PIE')
        for set_nm in tqdm(os.listdir(img_root), desc='set loop'):
            set_path = os.path.join(img_root, set_nm)
            setid = set_nm.replace('set0', '')
            crop_set_path = os.path.join(crop_veh_root, setid)
            makedir(crop_set_path)
            cid_to_oid[setid] = {}
            for vid_nm in tqdm(os.listdir(set_path), desc='video loop'):
                # refresh max id
                max_id = 0
                # get source img path and crop save path
                vid_path = os.path.join(set_path, vid_nm)
                vidid = str(int(vid_nm.replace('video_', '')))
                crop_vid_path = os.path.join(crop_set_path, vidid)
                makedir(crop_vid_path)
                # initialize the dict and set
                cid_to_oid[setid][vidid] = {}
                proc_oids = set()

                # get frame idx of img
                f_idx_to_img_nm = {}
                img_nms = list(os.listdir(vid_path))
                for i in range(len(img_nms)):
                    img_nm = img_nms[i]
                    f_idx_to_img_nm[str(i)] = img_nm

                # track
                results = yolo.track(
                                    source=vid_path,
                                    conf=args.conf,
                                    iou=args.iou,
                                    show=False,
                                    stream=True,
                                    device=DEVICE,
                                    show_conf=args.show_conf,
                                    show_labels=args.show_labels,
                                    save=args.save,
                                    verbose=False,
                                    exist_ok=args.exist_ok,
                                    project=args.project,
                                    name=args.name,
                                    classes=v_classes,
                                    imgsz=args.imgsz
                                )

                yolo.add_callback('on_predict_start', 
                                  partial(on_predict_start, persist=True))
                if 'yolov8' not in str(args.yolo_model):
                    # replace yolov8 model
                    m = get_yolo_inferer(args.yolo_model)
                    model = m(
                        model=args.yolo_model,
                        device=yolo.predictor.device,
                        args=yolo.predictor.args
                    )
                    yolo.predictor.model = model

                # store custom args in predictor
                yolo.predictor.custom_args = args

                # traverse all frames
                for frame_idx, r in enumerate(results):
                    # when there are objects
                    if r.boxes.data.shape[1] == 7:
                        n_obj = r.boxes.data.shape[0]
                        cur_frame_oids = []
                        for obj_idx in range(n_obj):
                            # get info of cur obj
                            cur_obj = r.boxes[obj_idx]
                            # check the cls
                            cur_cls = int(cur_obj.cls)
                            if cur_cls not in v_classes:
                                continue
                            cur_oid = int(cur_obj.id)
                            ltrb = list(map(int, cur_obj.xyxy[0].numpy()))
                            cur_img_nm = f_idx_to_img_nm[str(frame_idx)]
                            # update max id
                            max_id = max(max_id, cur_oid)
                            # add start id
                            cur_oid = str(cur_oid + start_id)
                            cur_frame_oids.append(cur_oid)
                            # write the cropped img
                            img = cv2.imread(os.path.join(img_root, 
                                                          set_nm, 
                                                          vid_nm, 
                                                          cur_img_nm))
                            cropped = crop_img(img, 
                                               bbox=ltrb, 
                                               resize_mode=args.resize_mode, 
                                               target_size=crop_size)
                            crop_obj_path = os.path.join(crop_vid_path, cur_oid)
                            makedir(crop_obj_path)
                            cv2.imwrite(os.path.join(crop_obj_path, cur_img_nm), 
                                        cropped)

                            # save to the dict
                            if cur_oid not in proc_oids and \
                                cur_oid not in cid_to_oid[setid][vidid]:
                                # init the list of tracks
                                cid_to_oid[setid][vidid][cur_oid] = {
                                    'img_nm': [[cur_img_nm]],
                                    'bbox': [[ltrb]],
                                }
                            elif cur_oid not in proc_oids and \
                                cur_oid in cid_to_oid[setid][vidid]:
                                # append a new track
                                cid_to_oid[setid]\
                                            [vidid]\
                                                [cur_oid]\
                                                    ['img_nm'].append([cur_img_nm])
                                cid_to_oid[setid]\
                                            [vidid]\
                                                [cur_oid]\
                                                    ['bbox'].append([ltrb])
                            elif cur_oid in proc_oids and \
                                cur_oid in cid_to_oid[setid][vidid]:
                                cid_to_oid[setid]\
                                            [vidid]\
                                                [cur_oid]\
                                                    ['img_nm']\
                                                        [-1].append(cur_img_nm)
                                cid_to_oid[setid]\
                                            [vidid]\
                                                [cur_oid]\
                                                    ['bbox']\
                                                        [-1].append(ltrb)
                            proc_oids.add(cur_oid)
                        # check if any observed obj missed in cur frame
                        rm_oids = set()
                        for oid in proc_oids:
                            if oid not in cur_frame_oids:
                                rm_oids.add(oid)
                        for oid in rm_oids:
                            proc_oids.remove(oid)

                # check the number of frames
                assert frame_idx + 1 == len(img_nms), (frame_idx + 1, len(img_nms))

                # accumulate start id
                start_id += max_id

                # check the consistency
                for setid in cid_to_oid:
                    for vidid in cid_to_oid[setid]:
                        for oid in cid_to_oid[setid][vidid]:
                            if len(cid_to_oid[setid][vidid][oid]['img_nm']) > 1:
                                print(f'set {setid} vid {vidid} obj {oid} n_track' 
                                      + 
                                      str(len(cid_to_oid[setid]\
                                                            [vidid]\
                                                                [oid]\
                                                                    ['img_nm'])))
                                # for track in cid_to_oid[setid][vidid][oid]['img_nm']:
                                #     print(track)
        # save the results
        with open(track_info_path, 'wb') as f:
            pickle.dump(cid_to_oid, f)
    ############################################################################
    # track vehicles in JAAD
    elif args.dataset_name == 'JAAD':
        cid_to_oid = {}  # vid id --> obj id --> bbox seq & img nm seq
        track_info_path = '/home/y_feng/workspace6/datasets/JAAD/veh_tracks.pkl'
        img_root = '/home/y_feng/workspace6/datasets/JAAD/images'
        crop_veh_root = \
            os.path.join('/home/y_feng/workspace6/datasets/JAAD/cropped_images_veh', 
                                    args.resize_mode, 
                                    f'{crop_size[0]}w_by_{crop_size[1]}h'
                                    )
        makedir(crop_veh_root)
        start_id = 0
        print('Tracking vehicles in JAAD')
        for vid_nm in tqdm(os.listdir(img_root), desc='video loop'):
            # refresh max id
            max_id = 0
            # get source img path and crop save path
            vid_path = os.path.join(img_root, vid_nm)
            vidid = str(int(vid_nm.replace('video_', '')))
            crop_vid_path = os.path.join(crop_veh_root, vidid)

            # initialize the dict and set
            cid_to_oid[vidid] = {}
            proc_oids = set()

            # get frame idx of img
            f_idx_to_img_nm = {}
            img_nms = list(os.listdir(vid_path))
            for i in range(len(img_nms)):
                img_nm = img_nms[i]
                f_idx_to_img_nm[str(i)] = img_nm

            # track
            results = yolo.track(
                                source=vid_path,
                                conf=args.conf,
                                iou=args.iou,
                                show=False,
                                stream=True,
                                device=DEVICE,
                                show_conf=args.show_conf,
                                show_labels=args.show_labels,
                                save=args.save,
                                verbose=False,
                                exist_ok=args.exist_ok,
                                project=args.project,
                                name=args.name,
                                classes=[2, 3, 4, 6, 8],
                                imgsz=args.imgsz
                            )

            yolo.add_callback('on_predict_start', 
                              partial(on_predict_start, persist=True))
            if 'yolov8' not in str(args.yolo_model):
                # replace yolov8 model
                m = get_yolo_inferer(args.yolo_model)
                model = m(
                    model=args.yolo_model,
                    device=yolo.predictor.device,
                    args=yolo.predictor.args
                )
                yolo.predictor.model = model

            # store custom args in predictor
            yolo.predictor.custom_args = args

            # traverse all frames
            for frame_idx, r in enumerate(results):
                # when there are objects
                if r.boxes.data.shape[1] == 7:
                    n_obj = r.boxes.data.shape[0]
                    cur_frame_oids = []
                    for obj_idx in range(n_obj):
                        # get info of cur obj
                        cur_obj = r.boxes[obj_idx]
                        cur_oid = int(cur_obj.id)
                        ltrb = list(map(int, cur_obj.xyxy[0].numpy()))
                        cur_img_nm = f_idx_to_img_nm[str(frame_idx)]
                        # update max id
                        max_id = max(max_id, cur_oid)
                        # add start id
                        cur_oid = str(cur_oid + start_id)
                        cur_frame_oids.append(cur_oid)
                        # write the cropped img
                        img = cv2.imread(os.path.join(img_root, 
                                                      vid_nm, 
                                                      cur_img_nm))
                        cropped = crop_img(img, 
                                           bbox=ltrb, 
                                           resize_mode=args.resize_mode, 
                                           target_size=crop_size)
                        crop_obj_path = os.path.join(crop_vid_path, cur_oid)
                        makedir(crop_obj_path)
                        cv2.imwrite(os.path.join(crop_obj_path, cur_img_nm), 
                                    cropped)

                        # save to the dict
                        if cur_oid not in proc_oids and \
                            cur_oid not in cid_to_oid[vidid]:
                            # init the list of tracks
                            cid_to_oid[vidid][cur_oid] = {
                                'img_nm': [[cur_img_nm]],
                                'bbox': [[ltrb]],
                            }
                        elif cur_oid not in proc_oids and \
                            cur_oid in cid_to_oid[vidid]:
                            # append a new track
                            cid_to_oid[vidid]\
                                        [cur_oid]\
                                            ['img_nm'].append([cur_img_nm])
                            cid_to_oid[vidid]\
                                        [cur_oid]\
                                            ['bbox'].append([ltrb])
                        elif cur_oid in proc_oids and cur_oid in cid_to_oid[vidid]:
                            cid_to_oid[vidid]\
                                        [cur_oid]\
                                            ['img_nm']\
                                                [-1].append(cur_img_nm)
                            cid_to_oid[vidid]\
                                        [cur_oid]\
                                            ['bbox']\
                                                [-1].append(ltrb)
                        proc_oids.add(cur_oid)
                    
                    # check if any observed obj missed in cur frame
                    rm_oids = set()
                    for oid in proc_oids:
                        if oid not in cur_frame_oids:
                            rm_oids.add(oid)
                    for oid in rm_oids:
                        proc_oids.remove(oid)

            # check the number of frames
            assert frame_idx + 1 == len(img_nms), (frame_idx + 1, len(img_nms))
            
            # accumulate start id
            start_id += max_id

            # check the consistency
            for vidid in cid_to_oid:
                for oid in cid_to_oid[vidid]:
                    if len(cid_to_oid[vidid][oid]['img_nm']) > 1:
                        print(f'vid {vidid} obj {oid} n_track' + 
                                str(len(cid_to_oid[vidid][oid]['img_nm'])))
                        # for track in cid_to_oid[vidid][oid]['img_nm']:
                        #     print(track)
        # save the results
        with open(track_info_path, 'wb') as f:
            pickle.dump(cid_to_oid, f)
    else:
        raise ValueError(args.dataset_name)

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='PIE')
    parser.add_argument('--crop_size', type=int, default=224, nargs='+',
                        help='size of cropped images')
    parser.add_argument('--resize_mode', type=str, default='even_padded')
    
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolo_nas_l',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, 
                        default='/home/y_feng/workspace6/work/ProtoPNet/ProtoPNet/yolo_tracking/test_images',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1080, 1920],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    # 2 3 4 6 8
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    track_PIE_JAAD_veh()
    # args = parse_opt()
    # run(args=args)