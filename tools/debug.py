import torch

def print_param_grad(model):
    nan_grad_params = []
    none_grad_params = []
    for n, p in model.named_parameters():
        if p.grad is not None:
            print(n, torch.isnan(p.grad).all().detach().cpu(), f'grad {p.grad.min().detach().cpu().numpy()} ~ {p.grad.max().detach().cpu().numpy()} param {p.min().detach().cpu().numpy()} ~ {p.max().detach().cpu().numpy()}')
            if torch.isnan(p.grad).all():
                nan_grad_params.append(n)
        else:
            none_grad_params.append(n)
    print('num nan grad params', len(nan_grad_params))