import numpy as np
import cv2

def draw_box(img, box):
    img = cv2.rectangle(img=img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0, 0, 255), thickness=2)
    return img

def draw_boxes_on_img(img, traj_seq):
    '''
    img: ndarray H W 3
    traj_seq: ndarray T 4 (ltrb)
    '''
    seq_len = traj_seq.shape[0]
    for i in range(seq_len-1, 0, -4):
        r = i / seq_len
        # print('traj type:', type(traj_seq))
        img = cv2.rectangle(img=img, pt1=(int(traj_seq[i, 0]), int(traj_seq[i, 1])), pt2=(int(traj_seq[i, 2]), int(traj_seq[i, 3])), color=(255 * r, 0, 0), thickness=2)

    return img

def visualize_input_traj():
    pass