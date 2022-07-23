import os
import pickle
import glob
import torch
import pdb

def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(loadpath):
    states = glob.glob1(loadpath, 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_model(loadpath, epoch='latest', device='cuda:0'):
    conf_path = os.path.join(loadpath, 'conf.pkl')

    if epoch is 'latest':
        epoch = get_latest_epoch(loadpath)

    state_path = os.path.join(loadpath, f'state_{epoch}.pt')

    conf = pickle.load(open(conf_path, 'rb'))
    state = torch.load(state_path)

    model = conf.make()
    model.load_state_dict(state, strict=True)
    model.to(device)

    print(
        f'\n[ utils/serialization ] Loaded configuration\n'
        f'    {conf.model_class}\n'
        f'    {conf_path}\n'
        f'    epoch {epoch}\n'
    )
    print(conf)

    return model, conf, epoch