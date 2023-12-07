import numpy as np
import random


def ltrb2xywh(ltrb):
    l, t, r, b = ltrb
    x = (l+r) // 2
    y = (t+b) // 2
    w = r-l
    h = b-t
    return np.array([x, y, w, h])

def xywh2ltrb(xywh):
    x, y, w, h = xywh
    l = int(x - w/2)
    r = int(l + w)
    t = int(y - h/2)
    b = int(t + h)
    return np.array([l, t, r, b])

def ltrb2xywh_seq(ltrb_seq):
    '''
    lrtb_seq: ndarray(T, 4)
    '''
    ltrb_seq = np.array(ltrb_seq)
    xs = (ltrb_seq[:, 0]+ltrb_seq[:, 2]) / 2
    ys = (ltrb_seq[:, 1]+ltrb_seq[:, 3]) / 2
    ws = ltrb_seq[:, 2] - ltrb_seq[:, 0]
    hs = ltrb_seq[:, 3] - ltrb_seq[:, 1]
    xywh_seq = np.stack([xs, ys, ws, hs], axis=1)
    assert xywh_seq.shape == ltrb_seq.shape
    return xywh_seq.astype(int)

def xywh2ltrb_seq(xywh_seq):
    '''
    xywh_seq: ndarray(T, 4)
    '''
    xywh_seq = np.array(xywh_seq)
    ls = xywh_seq[:, 0] - xywh_seq[:, 2] / 2
    rs = ls + xywh_seq[:, 2]
    ts = xywh_seq[:, 1] - xywh_seq[:, 3] / 2
    bs = ts + xywh_seq[:, 3]
    ltrb_seq = np.stack([ls, ts, rs, bs], axis=1)
    assert ltrb_seq.shape == xywh_seq.shape
    return ltrb_seq.astype(int)


def bbox2d_relation(bb1, bb2, rela_func='bbox_reg'):
    '''
    bb1, bb2: xywb
    '''
    eps = 1e-5
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    relation = np.array([(x2-x1)/(w1+eps),
                        (y2-y1)/(h1+eps),
                        np.log(w2/(w1+eps) + eps),
                        np.log(h2/(h1+eps) + eps)])
    if rela_func == 'log_bbox_reg':
        relation[:2] = np.log(np.abs(relation[:2]) + eps)
    return relation

def bbox2d_relation_multi_seq(seq, 
                              neighbor_seqs, 
                              rela_func='bbox_reg'):
    '''
    seq: target seq, ndarray(T, 4) xywh
    neighbor_seqs: ndarray(n_neighbor, T, 4) xywh

    return
        relations: ndarray(n_neighbor, T, 4)
    '''
    eps = 1e-5
    xx1 = np.expand_dims(seq[:, 0], axis=0)  # 1, T
    yy1 = np.expand_dims(seq[:, 1], axis=0)
    ww1 = np.expand_dims(seq[:, 2], axis=0)
    hh1 = np.expand_dims(seq[:, 3], axis=0)
    xx2 = neighbor_seqs[:, :, 0]  # K, T
    yy2 = neighbor_seqs[:, :, 1]
    ww2 = neighbor_seqs[:, :, 2]
    hh2 = neighbor_seqs[:, :, 3]
    relations = [(xx2-xx1)/(ww1+eps),
                (yy2-yy1)/(hh1+eps),
                np.log(ww2/(ww1+eps) + eps),
                np.log(hh2/(hh1+eps) + eps)]
    relations = np.stack(relations, axis=2)
    if rela_func == 'log_bbox_reg':
        relations[:, :, :2] = np.log(np.abs(relations[:, :, :2]) + eps)
    assert relations.shape == neighbor_seqs.shape, \
        (relations.shape, neighbor_seqs.shape)  # K, T, 4
    return relations


def closest_neighbor_relation(seq, 
                              neighbor_seqs, 
                              max_n_neighbor, 
                              rela_func='bbox_reg'):
    '''
    seq: target seq, ndarray(T, 4) xywh
    neighbor_seqs: ndarray(n_neighbor, T, 4) xywh
    max_n_neighbor: int
    '''
    n_neighbor = neighbor_seqs.shape[0]
    # pad neighbor sequences with 0
    if n_neighbor < max_n_neighbor:
        pad = np.zeros([max_n_neighbor-n_neighbor, 
                        neighbor_seqs.shape[1], 
                        neighbor_seqs.shape[2]])
        neighbor_seqs = np.concatenate([neighbor_seqs, pad], axis=0)

    # n_neighbor, T, 4
    neighbor_relations = bbox2d_relation_multi_seq(seq, 
                                                   neighbor_seqs, 
                                                   rela_func)
    

def get_neighbor_idx(bbox_relations,
                     max_n_neighbor, 
                     mode='random'):
    '''
    bbox_relations: ndarray(n_neighbor, T, 4)
    '''
    dist = np.abs(bbox_relations)
    if mode == 'random':
        idx = list(range(bbox_relations.shape[0]))
        random.shuffle(idx)
        idx = idx[:max_n_neighbor]
    else:
        raise ValueError(mode)
    return idx


if __name__ == '__main__':
    import pickle
    titan_path = '/home/y_feng/workspace6/datasets/TITAN/TITAN_extra/imgnm_to_objid_to_ann.pkl'
    pie_path = ''
    nusc_train_path = '/home/y_feng/workspace6/datasets/nusc/extra/train_imgnm_to_objid_to_ann.pkl'
    nusc_val_path = '/home/y_feng/workspace6/datasets/nusc/extra/val_imgnm_to_objid_to_ann.pkl'
    with open(titan_path, 'rb') as f:
        titan_dict = pickle.load(f)
    max_n_obj = 0
    max_n_ped = 0
    max_n_veh = 0
    print('-----------------TITAN-------------------')  # 49 46 20
    for vidid in titan_dict:
        for imgnm in titan_dict[vidid]:
            n_veh = len(titan_dict[vidid][imgnm]['veh'])
            n_ped = len(titan_dict[vidid][imgnm]['ped'])
            n_obj = n_veh + n_ped
            max_n_obj = max(max_n_obj, n_obj)
            max_n_ped = max(max_n_ped, n_ped)
            max_n_veh = max(max_n_veh, n_veh)
            print(f'TITAN {imgnm} | n obj: {n_obj} max n obj: {max_n_obj}| n ped {n_ped} max n ped {max_n_ped}| n veh {n_veh} max n veh {max_n_veh}')
    print('------------------nusc------------------')  # 16 14 13
    with open(nusc_train_path, 'rb') as f:
        nusc_dict = pickle.load(f)
    with open(nusc_val_path, 'rb') as f :
        nusc_dict.update(pickle.load(f))
    max_n_obj = 0
    max_n_ped = 0
    max_n_veh = 0
    for samid in nusc_dict:
        n_veh = len(nusc_dict[samid]['veh'])
        n_ped = len(nusc_dict[samid]['ped'])
        n_obj = n_veh + n_ped
        max_n_obj = max(max_n_obj, n_obj)
        max_n_ped = max(max_n_ped, n_ped)
        max_n_veh = max(max_n_veh, n_veh)
        print(f'nusc {samid} | n obj: {n_obj} max n obj: {max_n_obj}| n ped {n_ped} max n ped {max_n_ped}| n veh {n_veh} max n veh {max_n_veh}')
