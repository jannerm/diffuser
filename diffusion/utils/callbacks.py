import os
import torch

def save_callback(savepath):
    def _fn(epoch, model):
        fullpath = os.path.join(savepath, f'state_{epoch}.pt')
        state = model.state_dict()
        print(f'[ callback ] Saving state to {fullpath}')
        torch.save(state, fullpath)
    return _fn