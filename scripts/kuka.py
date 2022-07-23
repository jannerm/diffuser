import numpy as np
import torch
import pdb

import gym
import d4rl

from denoising_diffusion_pytorch.datasets.tamp import KukaDataset
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.mixer import MixerUnet
# from denoising_diffusion_pytorch.temporal import TemporalMixerUnet
from denoising_diffusion_pytorch.temporal_attention import TemporalUnet
from denoising_diffusion_pytorch.utils.rendering import KukaRenderer
import environments
import sys
sys.path.append('/data/vision/billf/scratch/yilundu/pddlstream')

#### dataset
H = 128
env = gym.make('hopper-medium-v2')
dataset = KukaDataset(H)
renderer = KukaRenderer()

## dimensions
obs_dim = dataset.obs_dim

#### model
# model = Unet(
#     width = H,
#     dim = 32,
#     dim_mults = (1, 2, 4, 8),
#     channels = 2,
#     out_dim = 1,
# ).cuda()

diffusion_path = f'logs/multiple_cube_kuka_conv_new_real2_{H}'
diffusion_epoch = 0

# model = MixerUnet(
#     dim = 32,
#     image_size = (H, obs_dim),
#     dim_mults = (1, 2, 4, 8),
#     channels = 2,
#     out_dim = 1,
# ).cuda()

# model = MixerUnet(
#     horizon = H,
#     transition_dim = obs_dim,
#     cond_dim = H,
#     dim = 32,
#     dim_mults = (1, 2, 4, 8),
# ).cuda()

model = TemporalUnet(
    horizon = H,
    transition_dim = obs_dim,
    cond_dim = H,
    dim = 128,
    dim_mults = (1, 2, 4, 8),
).cuda()


diffusion = GaussianDiffusion(
    model,
    channels = 1,
    image_size = (H, obs_dim),
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

#### test
print('testing forward')
x = dataset[0][0].view(1, H, obs_dim).cuda()
mask = torch.zeros(1, H).cuda()

loss = diffusion(x, mask)
loss.backward()
print('done')
# pdb.set_trace()
####

trainer = Trainer(
    diffusion,
    dataset,
    renderer,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                     # turn on mixed precision training with apex
    results_folder = f'./logs/multiple_cube_kuka_convnew_real2_{H}',
)

trainer.train()
