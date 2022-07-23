import numpy as np
import gym
import d4rl
import pdb

def pad_obs(obs, val=0):
    state = np.concatenate([np.ones(1)*val, obs])
    return state

def set_state_qpos(env, qpos, qvel):
    env.set_state(qpos, qvel)

def set_state_obs(env, obs):
    state = pad_obs(obs)
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])

env = gym.make('hopper-expert-v2')
env.reset()

dataset = env.get_dataset()
for ind in [0, 304, 305, 306]:
    obs = dataset['observations'][ind]
    act = dataset['actions'][ind]
    reference = dataset['next_observations'][ind]

    qpos = dataset['infos/qpos'][ind]
    qvel = dataset['infos/qvel'][ind]

    # set_state_qpos(env, qpos, qvel)
    set_state_obs(env, obs)

    next_obs, rew, term, _ = env.step(act)
    error = ((next_obs - reference)**2).sum()
    print(error)