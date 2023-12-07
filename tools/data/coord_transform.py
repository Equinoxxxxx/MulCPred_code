import os.path

import cv2
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from nuscenes import NuScenes


def nusc_3dbbox_to_2dbbox(nusc, anntk):
    ann = nusc.get('sample_annotation', anntk)
    
    sam = nusc.get('sample', ann['sample_token'])
    cam_front_data = nusc.get('sample_data', sam['data']['CAM_FRONT'])
    cali_sensor = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
    camera_intrinsic = np.array(cali_sensor['camera_intrinsic'])
    ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])

    # Get the annotation box
    box = nusc.get_box(ann['token'])
    
    # 从世界坐标系->车身坐标系
    box.translate(-np.array(ego_pose['translation']))
    box.rotate(Quaternion(ego_pose['rotation']).inverse)

    # 从车身坐标系->相机坐标系
    box.translate(-np.array(cali_sensor['translation']))
    box.rotate(Quaternion(cali_sensor['rotation']).inverse)

    corners_3d = box.corners()  # 3, 8 corners in camera coord

    # 从相机坐标系->像素坐标系
    view = np.eye(4)
    view[:3, :3] = np.array(camera_intrinsic)
    in_front = corners_3d[2, :] > 0.1  # ensure z > 0.1
    assert all(in_front), corners_3d
    corners = view_points(corners_3d, view, normalize=True)[:2, :]  # 2, 8
    # Get the 2D bounding box coordinates
    x1 = min(corners[0])
    x2 = max(corners[0])
    y1 = min(corners[1])
    y2 = max(corners[1])
    
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, 1600)
    y2 = min(y2, 900)
    box2d = [int(x1), int(y1), int(x2), int(y2)]  # ltrb

    return box2d, corners