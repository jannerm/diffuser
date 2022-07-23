import numpy as np
import gym
import d4rl
import pdb

from denoising_diffusion_pytorch.utils.rendering import set_state

def pad_obs(obs, val=0):
    state = np.concatenate([np.ones(1)*val, obs])
    return state

def set_state_qpos(env, qpos, qvel):
    env.set_state(qpos, qvel)

def set_state_obs(env, obs):
    state = pad_obs(obs)
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])

env = gym.make('hopper-expert-v2').env
env.reset()
dataset = env.get_dataset()

observations = dataset['observations']
# qstate = np.concatenate([dataset['infos/qpos'], dataset['infos/qvel']], axis=-1)

# print((observations == qstate[:, 1:]).all())

# set_state_qpos(env, dataset['infos/qpos'][0], dataset['infos/qvel'][0])
# x0 = np.concatenate([np.zeros(1), dataset['observations'][0]])
set_state_obs(env, observations[0])
# env.sim.forward()

total_reward = 0
for t in range(1000):
	act = dataset['actions'][t]
	next_obs, rew, term, _ = env.step(act)
	total_reward += rew

	# reference = qstate[t+1,1:]
	# reference = dataset['observations'][t+1]
	reference = dataset['next_observations'][t]
	error = ((next_obs - reference)**2).sum()
	print(t, error, rew, total_reward, term)

	if term:
		break

pdb.set_trace()
