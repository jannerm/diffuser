import numpy as np
import torch
import pdb
import os

import gym
import d4rl

from denoising_diffusion_pytorch.datasets.mujoco import RewardMuJoCoDataset
from reward import Classifier, RewardTrainer
# import denoising_diffusion_pytorch.utils as utils
import environments


def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False


#### dataset
H = 128
env_name = 'hopper-medium-v2'
logdir = f'./logs/{env_name}/discount_reward_value_hopper_{H}'
mkdir(logdir)
torch.cuda.set_device(0)

env = gym.make(env_name)
channel_num = 1
dataset = RewardMuJoCoDataset(channel_num, env, H)

## dimensions
obs_dim = dataset.obs_dim

#### model
model = Classifier(
    width = H,
    dim = 32,
    dim_mults = (1, 2, 4, 4, 8),
    channels = channel_num,
).cuda()

#### test
print('testing forward')
x, reward = dataset[0]
x = x.view(1, 1, H, obs_dim).cuda()
loss = model(x)
# loss.backward()
# print('done')
# pdb.set_trace()
####

trainer = RewardTrainer(
    model,
    dataset,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                     # turn on mixed precision training with apex
    results_folder = logdir,
)

trainer.train()
