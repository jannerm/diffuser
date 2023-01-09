import os
import numpy as np
import torch
import pdb
import pybullet as p
import os.path as osp

import gym
import d4rl

from diffusion.denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch import Trainer
from denoising_diffusion_pytorch.datasets.tamp import KukaDataset
from denoising_diffusion_pytorch.mixer import MixerUnet
from denoising_diffusion_pytorch.temporal_attention import TemporalUnet
from denoising_diffusion_pytorch.utils.rendering import KukaRenderer
import diffusion.utils as utils
import environments
from imageio import get_writer
import torch.nn as nn

from diffusion.models.mlp import TimeConditionedMLP
from diffusion.models import Config

from denoising_diffusion_pytorch.utils.pybullet_utils import get_bodies, sample_placement, pairwise_collision, \
    RED, GREEN, BLUE, BLACK, WHITE, BROWN, TAN, GREY, connect, get_movable_joints, set_joint_position, set_pose, add_fixed_constraint, remove_fixed_constraint, set_velocity, get_joint_positions, get_pose, enable_gravity

from gym_stacking.env import StackEnv
from tqdm import tqdm


def get_env_state(robot, cubes, attachments):
    joints = get_movable_joints(robot)
    joint_pos = get_joint_positions(robot, joints)

    for cube in cubes:
        pos, rot = get_pose(cube)
        pos, rot = np.array(pos), np.array(rot)

        if cube in attachments:
            attach = np.ones(1)
        else:
            attach = np.zeros(1)

        joint_pos = np.concatenate([joint_pos, pos, rot, attach], axis=0)

    return joint_pos


def execute(samples, env, idx=0):
    postprocess_samples = []
    robot = env.robot
    joints = get_movable_joints(robot)
    gains = np.ones(len(joints))

    cubes = env.cubes
    link = 8

    near = 0.001
    far = 4.0
    projectionMatrix = p.computeProjectionMatrixFOV(60., 1.0, near, far)

    location = np.array([0.1, 0.1, 2.0])
    end = np.array([0.0, 0.0, 1.0])
    viewMatrix = p.computeViewMatrix(location, end, [0, 0, 1])

    attachments = set()

    states = [get_env_state(robot, cubes, attachments)]
    rewards = 0
    ims = []

    for sample in samples[1:]:
        p.setJointMotorControlArray(bodyIndex=robot, jointIndices=joints, controlMode=p.POSITION_CONTROL,
                targetPositions=sample[:7], positionGains=gains)

        attachments = set()
        # Add constraints of objects
        for j in range(4):
            contact = sample[14+j*8]

            if contact > 0.5:
                add_fixed_constraint(cubes[j], robot, link)
                attachments.add(cubes[j])
                env.attachments[j] = 1
            else:
                remove_fixed_constraint(cubes[j], robot, link)
                set_velocity(cubes[j], linear=[0, 0, 0], angular=[0, 0, 0, 0])
                env.attachments[j] = 0


        for i in range(10):
            p.stepSimulation()

        states.append(get_env_state(robot, cubes, attachments))

        _, _, im, _, seg = p.getCameraImage(width=256, height=256, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix)
        im = np.array(im)
        im = im.reshape((256, 256, 4))

        state = env.get_state()
        # print(state)
        reward = env.compute_reward()

        rewards = rewards + reward
        ims.append(im)
        # writer.append_data(im)

    attachments = {}
    env.attachments[:] = 0
    env.get_state()
    reward = env.compute_reward()
    rewards = rewards + reward
    state = get_env_state(robot, cubes, attachments)

    # writer.close()

    return state, states, ims, rewards


def eval_episode(model, env, dataset, idx=0):
    state = env.reset()

    samples_full_list = []
    obs_dim = dataset.obs_dim

    samples = torch.Tensor(state[..., :-4])
    samples = (samples - dataset.mins) / (dataset.maxs - dataset.mins + 1e-8)
    samples = samples[None, None, None].cuda()
    samples = (samples - 0.5) * 2

    conditions = [
           (0, obs_dim, samples),
    ]

    rewards = 0
    frames = []

    counter = 0
    map_tuple = {}
    for i in range(4):
        for j in range(4):
            if i == j:
                continue

            map_tuple[(i, j)] = counter
            counter = counter + 1

    total_samples = []

    for i in range(3):
        stack = env.goal[env.progress+1]
        place = env.goal[env.progress]
        cond_idx = map_tuple[(stack, place)]
        samples = samples_orig = trainer.ema_model.guided_conditional_sample(model, 1, conditions, cond_idx, stack, place)

        samples = torch.clamp(samples, -1, 1)
        samples_unscale = (samples + 1) * 0.5
        samples = dataset.unnormalize(samples_unscale)

        samples = to_np(samples.squeeze(0).squeeze(0))

        samples, samples_list, frames_new, reward = execute(samples, env, idx=i)
        frames.extend(frames_new)

        total_samples.extend(samples_list)

        samples = (samples - dataset.mins) / (dataset.maxs - dataset.mins + 1e-8)
        samples = torch.Tensor(samples[None, None, None]).to(samples_orig.device)
        samples = (samples - 0.5) * 2


        conditions = [
               (0, obs_dim, samples),
        ]

        samples_list.append(samples)

        rewards = rewards + reward

    if not osp.exists("cond_samples/"):
        os.makedirs("cond_samples/")

    writer = get_writer("cond_samples/cond_video_writer{}.mp4".format(idx))

    for frame in frames:
        writer.append_data(frame)

    np.save("cond_samples/cond_sample_{}.npy".format(idx), np.array(total_samples))
    # writer = get_writer("video_writer.mp4")

    # for frame in frames:
    #     writer.append_data(frame)


    return rewards


class PosGuide(nn.Module):
    def __init__(self, cube, cube_other):
        super().__init__()
        self.cube = cube
        self.cube_other = cube_other

    def forward(self, x, t):
        cube_one = x[..., 64:, 7+self.cube*8: 7+self.cube*8]
        cube_two = x[..., 64:, 7+self.cube_other*8:7+self.cube_other*8]

        pred = -100 * torch.pow(cube_one - cube_two, 2).sum(dim=-1)
        return pred



def to_np(x):
    return x.detach().cpu().numpy()

def pad_obs(obs, val=0):
    state = np.concatenate([np.ones(1)*val, obs])
    return state

def set_obs(env, obs):
    state = pad_obs(obs)
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])

#### dataset
H = 128
dataset = KukaDataset(H)

env_name = "multiple_cube_kuka_temporal_convnew_real2_128"
H = 128
T = 1000

diffusion_path = f'logs/{env_name}/'
diffusion_epoch = 650

dataset = KukaDataset(H)
weighted = 5.0
trial = 0

savepath = f'logs/{env_name}/plans_weighted{weighted}_{H}_{T}/{trial}'
utils.mkdir(savepath)

## dimensions
obs_dim = dataset.obs_dim
act_dim = 0

#### model
# model = MixerUnet(
#     dim = 32,
#     image_size = (H, obs_dim),
#     dim_mults = (1, 2, 4, 8),
#     channels = 2,
#     out_dim = 1,
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
    channels = 2,
    image_size = (H, obs_dim),
    timesteps = T,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

#### load reward and value functions
# reward_model, *_ = utils.load_model(reward_path, reward_epoch)
# value_model, *_ = utils.load_model(value_path, value_epoch)
# value_guide = guides.ValueGuide(reward_model, value_model, discount)
env = StackEnv(conditional=True)

trainer = Trainer(
    diffusion,
    dataset,
    env,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                     # turn on mixed precision training with apex
    results_folder = diffusion_path,
)


print(f'Loading: {diffusion_epoch}')
trainer.load(diffusion_epoch)
render_kwargs = {
    'trackbodyid': 2,
    'distance': 10,
    'lookat': [10, 2, 0.5],
    'elevation': 0
}

x = dataset[0][0].view(1, 1, H, obs_dim).cuda()
conditions = [
       (0, obs_dim, x[:, :, :1]),
]
trainer.ema_model.eval()
hidden_dims = [128, 128, 128]


config = Config(
    model_class=TimeConditionedMLP,
    time_dim=128,
    input_dim=obs_dim,
    hidden_dims=hidden_dims,
    output_dim=12,
    savepath="",
)

device = torch.device('cuda')
model = config.make()
model.to(device)


ckpt_path = "/data/vision/billf/scratch/yilundu/denoising-diffusion/logs/kuka_cube_stack_classifier_new3/value_0.99/state_80.pt"
ckpt = torch.load(ckpt_path)

model.load_state_dict(ckpt)


samples_list = []
frames = []

# models = [PosGuide(1, 3), PosGuide(1, 4), PosGuide(1, 2)]

counter = 0
map_tuple = {}
for i in range(4):
    for j in range(4):
        if i == j:
            continue

        map_tuple[(i, j)] = counter
        counter = counter + 1


# Red = block 0
# Green = block 1
# Blue = block 2
# Yellow block 3


rewards =  []

for i in tqdm(range(100)):
    reward = eval_episode(model, env, dataset, idx=i)
    rewards.append(reward)
    print("rewards mean: ", np.mean(rewards))
    print("rewards std: ", np.std(rewards) / len(rewards) ** 0.5)



samples_full_list = np.array(samples_full_list)
np.save("execution.npy", samples_full_list)

# writer = get_writer("full_execution.mp4")

for frame in frames:
    writer.append_data(frame)

writer.close()
import pdb
pdb.set_trace()
assert False

# samples_next = trainer.ema_model.guided_conditional_sample(model, 1, conditions)
# samples_next = trainer.ema_model.conditional_sample(1, conditions)
samples = torch.cat(samples_list, dim=-2)


# samples = trainer.ema_model.conditional_sample(1, conditions)
samples = torch.clamp(samples, -1, 1)
samples_unscale = (samples + 1) * 0.5
samples = dataset.unnormalize(samples_unscale)



# x = x = (x + 1) * 0.5
# x = dataset.unnormalize(x)

samples = to_np(samples.squeeze(0).squeeze(0))
postprocess(samples, renderer)


savepath = "execute_sim_11.mp4"

savepath = savepath.replace('.png', '.mp4')
writer = get_writer(savepath)

for img in imgs:
    writer.append_data(img)

writer.close()
