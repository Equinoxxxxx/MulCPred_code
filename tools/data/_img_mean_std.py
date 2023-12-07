def img_mean_std(norm_mode):
    # BGR order
    if norm_mode == 'activitynet':
        # mean = [0.4477, 0.4209, 0.3906]
        # std = [0.2767, 0.2695, 0.2714]
        mean = [0.3906, 0.4209, 0.4477]
        std = [0.2714, 0.2695, 0.2767]
    elif norm_mode == 'kinetics':
        # mean = [0.4345, 0.4051, 0.3775]
        # std = [0.2768, 0.2713, 0.2737]
        mean = [0.3775, 0.4051, 0.4345]
        std = [0.2737, 0.2713, 0.2768]
    elif norm_mode == '0.5' or norm_mode == 'tf':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif norm_mode == 'torch':
        mean = [0.406, 0.456, 0.485]  # BGR
        std = [0.225, 0.224, 0.229]
    
    elif norm_mode == 'ori':
        mean = None
        std = None
    
    return mean, std