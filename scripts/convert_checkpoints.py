import os
import pickle
import shutil
import pdb

####
import sys
sys.path.append('/home/janner/mount/diffusion')
import diffusion
####

import diffuser.datasets as datasets
import diffuser.models as models
import diffuser.utils as utils


CONFIG_NAMES = ['dataset', 'render', 'model', 'diffusion', 'trainer']

CLASS_MAP = {
    diffusion.datasets.mujoco.MuJoCoDataset: datasets.GoalDataset,
    diffusion.utils.rendering.Maze2dRenderer: utils.Maze2dRenderer,
    diffusion.models.temporal.TemporalMixerUnet: models.TemporalMixerUnet,
    diffusion.models.diffusion.GaussianDiffusion: models.GaussianDiffusion,
    diffusion.utils.training.Trainer: utils.Trainer,
}

CHECKPOINTS = {
    'maze2d-large-v1': {
        'logbase': 'logs/maze2d',
        'diffusion_loadpath': (
            'diffusion/cond_H384_T256'
        )
    }
}

def convert_checkpoint(*loadpath):

    for config_name in CONFIG_NAMES:
        fullpath = os.path.join(*loadpath, f'{config_name}_config.pkl')
        oldpath = os.path.join(*loadpath, f'old_{config_name}_config.pkl')

        config = utils.load_config(fullpath)

        old_class = config._class
        if old_class in CLASS_MAP:
            new_class = CLASS_MAP[old_class]
            print(f'Updating class: {old_class} --> {new_class}')
            config._class = new_class

        if config_name == 'diffusion' and 'transition_dim' in config._dict:
            del config._dict['transition_dim']

        if config_name == 'diffusion' and 'observation_weight' in config._dict:
            del config._dict['observation_weight']

        if config_name == 'diffusion' and 'timesteps' in config._dict:
            config._dict['n_timesteps'] = config._dict['timesteps']
            del config._dict['timesteps']

        shutil.copyfile(fullpath, oldpath)
        pickle.dump(config, open(fullpath, 'wb'))


for dataset, config in CHECKPOINTS.items():

    diffusion_experiment = convert_checkpoint(
        config['logbase'], dataset, config['diffusion_loadpath'])
