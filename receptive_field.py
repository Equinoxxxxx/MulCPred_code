import math


def compute_layer_rf_info(layer_filter_size, layer_stride, layer_padding,
                          previous_layer_rf_info):
    n_in = previous_layer_rf_info[0] # input feat map size
    j_in = previous_layer_rf_info[1] # receptive field jump of input layer
    r_in = previous_layer_rf_info[2] # receptive field size of input layer
    start_in = previous_layer_rf_info[3] # center of receptive field of input layer

    if layer_padding == 'SAME':
        n_out = math.ceil(float(n_in) / float(layer_stride))
        if (n_in % layer_stride == 0):
            pad = max(layer_filter_size - layer_stride, 0)
        else:
            pad = max(layer_filter_size - (n_in % layer_stride), 0)
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    elif layer_padding == 'VALID':
        n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
        pad = 0
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    else:
        # layer_padding is an int that is the amount of padding on one side
        pad = layer_padding * 2
        n_out = math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1
        # print(n_out, layer_filter_size, layer_stride, layer_padding, previous_layer_rf_info)
    pL = math.floor(pad/2)

    j_out = j_in * layer_stride
    r_out = r_in + (layer_filter_size - 1)*j_in
    start_out = start_in + ((layer_filter_size - 1)/2 - pL)*j_in
    return [n_out, j_out, r_out, start_out]

def compute_rf_protoL_at_spatial_location(img_size, height_index, width_index, protoL_rf_info):
    n_h = protoL_rf_info[0][0]
    j_h = protoL_rf_info[0][1]
    r_h = protoL_rf_info[0][2]
    start_h = protoL_rf_info[0][3]
    n_w = protoL_rf_info[1][0]
    j_w = protoL_rf_info[1][1]
    r_w = protoL_rf_info[1][2]
    start_w = protoL_rf_info[1][3]
    assert height_index < n_h, ('h idx', height_index, 'n', n_h)
    assert width_index < n_w, ('w idx', width_index, 'n', n_w)

    center_h = start_h + (height_index*j_h)

    rf_start_height_index = max(int(center_h - (r_h/2)), 0)
    rf_end_height_index = min(int(center_h + (r_h/2)), img_size[0])

    center_w = start_w + (width_index*j_w)

    rf_start_width_index = max(int(center_w - (r_w/2)), 0)
    rf_end_width_index = min(int(center_w + (r_w/2)), img_size[1])

    return [rf_start_height_index, rf_end_height_index,
            rf_start_width_index, rf_end_width_index]

def compute_rf_protoL_at_temporal_location(seq_len, t_index, protoL_rf_info):
    n = protoL_rf_info[0]
    j = protoL_rf_info[1]
    r = protoL_rf_info[2]
    start = protoL_rf_info[3]
    assert(t_index < n)

    center_t = start + (t_index*j)

    rf_start_t_index = max(int(center_t - (r/2)), 0)
    rf_end_t_index = min(int(center_t + (r/2)), seq_len)

    return [rf_start_t_index, rf_end_t_index]

def compute_rf_loc_spatial(img_size, prototype_patch_index, sp_rf_info):
    img_index = prototype_patch_index[0]
    height_index = prototype_patch_index[1]
    width_index = prototype_patch_index[2]
    rf_indices = compute_rf_protoL_at_spatial_location(img_size,
                                                       height_index,
                                                       width_index,
                                                       sp_rf_info)
    return [img_index, rf_indices[0], rf_indices[1],
            rf_indices[2], rf_indices[3]]  # sample idx, h start, h end, w start, w end

def compute_rf_loc_spatiotemporal(img_size, seq_len, prototype_patch_index, sp_rf_info, t_rf_info):
    img_index = prototype_patch_index[0]
    t_index = prototype_patch_index[1]
    height_index = prototype_patch_index[2]
    width_index = prototype_patch_index[3]
    sp_rf_indices = compute_rf_protoL_at_spatial_location(img_size,
                                                       height_index,
                                                       width_index,
                                                       sp_rf_info)
    t_rf_indices = compute_rf_protoL_at_temporal_location(seq_len=seq_len,
                                                          t_index=t_index,
                                                          protoL_rf_info=t_rf_info)
    return [img_index, t_rf_indices[0], t_rf_indices[1], sp_rf_indices[0], sp_rf_indices[1],
            sp_rf_indices[2], sp_rf_indices[3]]

def compute_rf_prototypes(img_size, prototype_patch_indices, protoL_rf_info):
    rf_prototypes = []
    for prototype_patch_index in prototype_patch_indices:
        img_index = prototype_patch_index[0]
        height_index = prototype_patch_index[1]
        width_index = prototype_patch_index[2]
        rf_indices = compute_rf_protoL_at_spatial_location(img_size,
                                                           height_index,
                                                           width_index,
                                                           protoL_rf_info)
        rf_prototypes.append([img_index, rf_indices[0], rf_indices[1],
                              rf_indices[2], rf_indices[3]])
    return rf_prototypes

def compute_proto_layer_rf_info(img_size, cfg, prototype_kernel_size):
    rf_info = [img_size, 1, 1, 0.5]

    for v in cfg:
        if v == 'M':
            rf_info = compute_layer_rf_info(layer_filter_size=2,
                                            layer_stride=2,
                                            layer_padding='SAME',
                                            previous_layer_rf_info=rf_info)
        else:
            rf_info = compute_layer_rf_info(layer_filter_size=3,
                                            layer_stride=1,
                                            layer_padding='SAME',
                                            previous_layer_rf_info=rf_info)

    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=prototype_kernel_size,
                                                layer_stride=1,
                                                layer_padding='VALID',
                                                previous_layer_rf_info=rf_info)

    return proto_layer_rf_info

def compute_proto_layer_rf_info_v2(input_size, layer_filter_sizes, layer_strides, layer_paddings, prototype_kernel_size):

    assert(len(layer_filter_sizes) == len(layer_strides))
    assert(len(layer_filter_sizes) == len(layer_paddings))

    rf_info = [input_size, 1, 1, 0.5]
    print('calc receptive field info')
    # print(rf_info)
    # calculate receive field of last layer in backbone
    for i in range(len(layer_filter_sizes)):
        filter_size = layer_filter_sizes[i]
        stride_size = layer_strides[i]
        padding_size = layer_paddings[i]

        rf_info = compute_layer_rf_info(layer_filter_size=filter_size,
                                layer_stride=stride_size,
                                layer_padding=padding_size,
                                previous_layer_rf_info=rf_info)
        # print('layer', i, filter_size, stride_size, padding_size, rf_info)
    # calculate receive field of each prototype
    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=prototype_kernel_size,
                                                layer_stride=1,
                                                layer_padding='VALID',
                                                previous_layer_rf_info=rf_info)

    return proto_layer_rf_info  # [n, j, r, start]

if __name__ == '__main__':
    from _backbones import SkeletonConv2D, record_conv2d_info, record_sp_conv2d_info_w, record_sp_conv2d_info_h
    from _proto_model import SkeletonPNet

    backbone = SkeletonConv2D()
    conv_info = record_conv2d_info(backbone)

    spw_k_list, spw_s_list, spw_p_list = record_sp_conv2d_info_w(conv_info)
    sph_k_list, sph_s_list, sph_p_list = record_sp_conv2d_info_h(conv_info)
    sph_proto_layer_rf_info = compute_proto_layer_rf_info_v2(input_size=16,
                                                        layer_filter_sizes=sph_k_list,
                                                        layer_strides=sph_s_list,
                                                        layer_paddings=sph_p_list,
                                                        prototype_kernel_size=1)
    spw_proto_layer_rf_info = compute_proto_layer_rf_info_v2(input_size=17,
                                                        layer_filter_sizes=spw_k_list,
                                                        layer_strides=spw_s_list,
                                                        layer_paddings=spw_p_list,
                                                        prototype_kernel_size=1)
    # print('h rf info', sph_proto_layer_rf_info)
    # print('w rf info', spw_proto_layer_rf_info)
    ppnet = SkeletonPNet(backbone=backbone, 
                        skeleton_seq_shape=(2, 16, 17), 
                        p_per_cls=20, 
                        prototype_dim=128, 
                        sp_proto_layer_rf_info=(sph_proto_layer_rf_info, spw_proto_layer_rf_info))
    
    featmap_loc = [0, 3, 3]  # b, h, w
    # print('featmap loc', featmap_loc)

    rf_prototype_j = compute_rf_loc_spatial(img_size=(16, 17), 
                                    prototype_patch_index=featmap_loc, 
                                    sp_rf_info=(sph_proto_layer_rf_info, spw_proto_layer_rf_info))  # [sample idx, h start, h end, w start, w end]
    # print(rf_prototype_j)