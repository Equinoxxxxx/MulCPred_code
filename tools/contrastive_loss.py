import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def calc_contrast_loss(simi_mats, contrast_mode):
    '''
    simi_mats: list [tensor(b, b)]
    '''
    if contrast_mode == 'pair_wise_norm_orth' or\
        contrast_mode == 'pair_wise_orth':
        loss = 0
        simi_mats = torch.stack(simi_mats, dim=0)  # n_pair, b, b
        I_ = torch.unsqueeze(torch.eye(simi_mats.size(1)), dim=0).to(simi_mats.device)  # 1, b, b
        orth_loss = torch.mean(torch.norm(simi_mats - I_))
        return orth_loss
    else:
        assert contrast_mode in ('pair_wise', 
                                 'proto_pair_wise', 
                                 'bridge',
                                 'proto_bridge')
        ce = 0
        n_pairs = len(simi_mats)
        # print('n pairs', n_pairs)
        label = torch.arange(simi_mats[0].size(0), device=simi_mats[0].device, dtype=torch.long)  # b
        i=0
        for mat in simi_mats:  # b, b
            # print('cur pair', i)
            # print('mat', mat)
            # print('label', label)
            ce = ce + F.cross_entropy(mat, label)
            i+=1
    
        return ce / len(simi_mats)

def calc_batch_simi(z_dict, log_logit_scale, protos, bridge_m, contrast_mode):
    eps = 1e-5
    logit_scale = log_logit_scale.exp()
    simi_mats = []
    modalities = list(z_dict.keys())
    if protos is not None:
        assert protos.requires_grad, protos
    n_m = len(modalities)
    if contrast_mode == 'pair_wise':
        for i in range(n_m):
            mi = modalities[i]
            for j in range(i+1, n_m):
                mj = modalities[j]
                for k in range(len(z_dict[mj])):
                    zi = z_dict[mi][k]  # b, proj_dim
                    zj = z_dict[mj][k]  # b, proj_dim
                    # cosine simi
                    zi_norm, zj_norm = \
                        zi/(zi.norm(dim=1, keepdim=True)+eps), zj/(zj.norm(dim=1, keepdim=True)+eps)
                    simi_mat1 = logit_scale * zi_norm @ zj_norm.t() + eps
                    simi_mat2 = simi_mat1.t()
                    simi_mats += [simi_mat1, simi_mat2]
    elif contrast_mode == 'pair_wise_norm_orth':
        for i in range(n_m):
            mi = modalities[i]
            for j in range(i+1, n_m):
                mj = modalities[j]
                for k in range(len(z_dict[mj])):
                    zi = z_dict[mi][k]  # b, proj_dim
                    zj = z_dict[mj][k]  # b, proj_dim
                    # cosine simi
                    zi_norm, zj_norm = \
                        zi/(zi.norm(dim=1, keepdim=True)+eps), zj/(zj.norm(dim=1, keepdim=True)+eps)
                    simi_mat1 = logit_scale * zi_norm @ zj_norm.t() + eps
                    simi_mats += [simi_mat1]
    elif contrast_mode == 'pair_wise_orth':
        for i in range(n_m):
            mi = modalities[i]
            for j in range(i+1, n_m):
                mj = modalities[j]
                for k in range(len(z_dict[mj])):
                    zi = z_dict[mi][k]  # b, proj_dim
                    zj = z_dict[mj][k]  # b, proj_dim
                    simi_mat1 = logit_scale * zi @ zj.t() + eps
                    simi_mats += [simi_mat1]
    elif contrast_mode == 'proto_pair_wise':
        code_dict = {m:[] for m in modalities}
        for m in z_dict:
            for k in range(len(z_dict[m])):
                code_dict[m].append(z_dict[m][k] @ protos.t())  # b, n_p
        for i in range(n_m):
            mi = modalities[i]
            for j in range(i+1, n_m):
                mj = modalities[j]
                for k in range(len(code_dict[mj])):
                    code_i = code_dict[mi][k]  # b, n_p
                    code_j = code_dict[mj][k]  # b, n_p
                    # cosine simi
                    code_i_norm, code_j_norm = \
                        code_i/(code_i.norm(dim=1, keepdim=True)+eps), code_j/(code_j.norm(dim=1, keepdim=True)+eps)
                    simi_mat1 = logit_scale * code_i_norm @ code_j_norm.t() + eps
                    simi_mat2 = simi_mat1.t()
                    simi_mats += [simi_mat1, simi_mat2]
    elif contrast_mode == 'bridge':
        for m in z_dict:
            if m != bridge_m:
                for k in range(len(z_dict[m])):
                    zb = z_dict[bridge_m][k]
                    zi = z_dict[m][k]
                    zb_norm, zi_norm = \
                        zb / (zb.norm(dim=1, keepdim=True)+eps), zi / (zi.norm(dim=1, keepdim=True)+eps)
                    simi_mat1 = logit_scale * zb_norm @ zi_norm.t() + eps
                    simi_mat2 = simi_mat1.t()
                    simi_mats += [simi_mat1, simi_mat2]
    elif contrast_mode == 'proto_bridge':
        code_dict = {m:[] for m in modalities}
        for m in z_dict:
            for k in range(len(z_dict[m])):
                code_dict[m].append(z_dict[m][k] @ protos.t())  # b, n_p
        for m in code_dict:
            if m != bridge_m:
                for k in range(len(code_dict[m])):
                    code_b = code_dict[bridge_m][k]
                    code_i = code_dict[m][k]
                    code_b_norm, code_i_norm = \
                        code_b / (code_b.norm(dim=1, keepdim=True)+eps), code_i / (code_i.norm(dim=1, keepdim=True)+eps)
                    simi_mat1 = logit_scale * code_b_norm @ code_i_norm.t() + eps
                    simi_mat2 = simi_mat1.t()
                    simi_mats += [simi_mat1, simi_mat2]
    else:
        raise NotImplementedError(contrast_mode)
    
    return simi_mats  # list: n_pair*[b, b]


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss