import os
import torch

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        log('Model saved in ' + model_dir)

def save_best(model, model_dir, model_name, acc, best_acc, log=print):
    if acc > best_acc:
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '.pth')))
        best_acc = acc
        log('Model saved in ' + model_dir)
    return best_acc