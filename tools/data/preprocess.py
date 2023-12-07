import cv2
import numpy as np


def crop_img(img, bbox, resize_mode, target_size=(224, 224)):
    l, t, r, b = list(map(int, bbox))
    cropped = img[t:b, l:r]
    if resize_mode == 'ori':
        resized = cropped
    elif resize_mode == 'resized':
        resized = cv2.resize(cropped, target_size)
    elif resize_mode == 'even_padded':
        h = b-t
        w = r-l
        if  float(w) / h > float(target_size[0]) / target_size[1]:
            ratio = float(target_size[0]) / w
        else:
            ratio = float(target_size[1]) / h
        new_size = (int(w*ratio), int(h*ratio))
        # print(cropped.shape, l, t, r, b, new_size)
        for l in new_size:
            if l == 0:
                return None
        cropped = cv2.resize(cropped, new_size)
        w_pad = target_size[0] - new_size[0]
        h_pad = target_size[1] - new_size[1]
        l_pad = w_pad // 2
        r_pad = w_pad - l_pad
        t_pad = h_pad // 2
        b_pad = h_pad - t_pad
        resized = cv2.copyMakeBorder(cropped,t_pad,b_pad,l_pad,r_pad,cv2.BORDER_CONSTANT,value=(0, 0, 0))  # t, b, l, r
        assert (resized.shape[1], resized.shape[0]) == target_size
    
    return resized

def crop_ctx(img, bbox, mode, target_size=(224, 224)):
    ori_H, ori_W = img.shape[:2]
    l, t, r, b = list(map(int, bbox))
    # crop local context
    x = (l+r) // 2
    y = (t+b) // 2
    h = b-t
    w = r-l
    crop_h = h*2
    crop_w = h*2
    crop_l = max(x-h, 0)
    crop_r = min(x+h, ori_W)
    crop_t = max(y-h, 0)
    crop_b = min(y+h, ori_W)
    if mode == 'local':
        # mask target pedestrian
        rect = np.array([[l, t], [r, t], [r, b], [l, b]])
        masked = cv2.fillConvexPoly(img, rect, (127, 127, 127))
        cropped = masked[crop_t:crop_b, crop_l:crop_r]
    elif mode == 'ori_local':
        cropped = img[crop_t:crop_b, crop_l:crop_r]
    l_pad = max(h-x, 0)
    r_pad = max(x+h-ori_W, 0)
    t_pad = max(h-y, 0)
    b_pad = max(y+h-ori_H, 0)
    cropped = cv2.copyMakeBorder(cropped, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
    assert cropped.shape[0] == crop_h and cropped.shape[1] == crop_w, (cropped.shape, (crop_h, crop_w))
    resized = cv2.resize(cropped, target_size)

    return resized


