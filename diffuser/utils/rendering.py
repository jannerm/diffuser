import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gym
import mujoco_py as mjc
import warnings
import pdb

from .arrays import to_np
from .video import save_video, save_videos

from diffuser.datasets.d4rl import load_environment

# -----------------------------------------------------------------------------#
# ------------------------------- helper structs ------------------------------#
# -----------------------------------------------------------------------------#


def env_map(env_name):
    '''
        map D4RL dataset names to custom fully-observed
        variants for rendering
    '''
    if 'halfcheetah' in env_name:
        return 'HalfCheetahFullObs-v2'
    elif 'hopper' in env_name:
        return 'HopperFullObs-v2'
    elif 'walker2d' in env_name:
        return 'Walker2dFullObs-v2'
    else:
        return env_name

# -----------------------------------------------------------------------------#
# ------------------------------ helper functions -----------------------------#
# -----------------------------------------------------------------------------#


def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask


def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

# -----------------------------------------------------------------------------#
# ---------------------------------- renderers --------------------------------#
# -----------------------------------------------------------------------------#


class MuJoCoRenderer:
    '''
        default mujoco renderer
    '''

    def __init__(self, env):
        if type(env) is str:
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env
        # - 1 because the envs in renderer are fully-observed
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
        except:
            print(
                '[ utils/rendering ] Warning: could not initialize offscreen renderer')
            self.viewer = None

    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate([
            xpos[:, None],
            observations,
        ], axis=-1)
        return states

    def pad_observations_hand(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        qvel_dim = self.env.sim.data.qvel.size
        qpos_pred = observations[..., :30]
        pads = np.zeros(
            (qpos_pred.shape[0], qpos_pred.shape[1], qpos_dim + qvel_dim - qpos_pred.shape[2]))
        states = np.concatenate([qpos_pred, pads], axis=-1)
        states_dict = {'qpos': states[..., :qpos_dim],
                       'qvel': states[..., qpos_dim:]}
        return states_dict

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None, qpos=None, qvel_value=None, states_dict=None, actions=None):

        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        if qpos is not None and qvel_value is not None:
            self.env.set_state(qpos, qvel_value)
        elif actions is not None:
            self.env.reset()
            self.env.step(actions)
            self.env.set_env_state(self.env.get_env_state())
        else:
            set_state(self.env, state)

        self.viewer.render(*dim, camera_id=1)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        images = []
        qpos = kwargs.get('qpos', None)
        qvel_values = kwargs.get('qvel_value', None)
        actions = kwargs.get('actions', None)
        for i, observation in enumerate(observations):
            if qpos is not None and qvel_values is not None:
                kwargs['qpos'] = qpos[i]
                kwargs['qvel_value'] = qvel_values[i]
            elif actions is not None:
                kwargs['actions'] = actions[i]

            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):

        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 2, 0.5],
            'elevation': 0,
        }
        images = []
        qpos = kwargs.get('qpos', None)
        qvel_values = kwargs.get('qvel_value', None)
        actions = kwargs.get('actions', None)
        for i, path in enumerate(paths):
            # [ H x obs_dim ]
            path = atmost_2d(path)
            if qpos is not None and qvel_values is not None:
                kwargs['qpos'] = qpos[i]
                kwargs['qvel_value'] = qvel_values[i]
            elif actions is not None:
                kwargs['actions'] = actions[i]

            img = self.renders(to_np(path), dim=dim, partial=True,
                               qvel=True, render_kwargs=render_kwargs, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images

    def render_rollout(self, savepath, states,rollout_qpos, rollout_qvel, **video_kwargs):
        if type(states) is list:
            states = np.array(states)
        images = self._renders(
            states, qpos=rollout_qpos, qvel_value=rollout_qvel, partial=True)
        save_video(savepath, images, **video_kwargs)

    def render_plan(self, savepath, actions, observations_pred, state, states_pred, adroit, fps=30):

        # [ batch_size x horizon x observation_dim ]
        if adroit:
            observations_real, states_real = rollouts_from_state_with_dict(
                self.env, state, actions)
            # there will be one more state in `observations_real`
            # than in `observations_pred` because the last action
            # does not have an associated next_state in the sampled trajectory
            observations_real = observations_real[:, :-1]
            states_real = states_real[:, :-1]
            qpos_real, qvel_real = states_real[...,
                                               :self.env.sim.data.qpos.size], states_real[..., self.env.sim.data.qpos.size:]
        else:
            observations_real = rollouts_from_state(
                self.env, state, actions)
            observations_real = observations_real[:, :-1]
            qpos_real, qvel_real = np.empty(
                shape=observations_real.shape[:-1]), np.empty(shape=observations_real.shape[:-1])

        images_pred = np.stack([
            self._renders(obs_pred, partial=True,
                          qpos=states_pred['qpos'][i], qvel_value=states_pred['qvel'][i])
            for i, obs_pred in enumerate(observations_pred)
        ])

        images_real = np.stack([
            self._renders(obs_real, partial=False,
                          qpos=qpos_real[i], qvel_value=qvel_real[i])
            for i, obs_real in enumerate(observations_real)
        ])

        # [ batch_size x horizon x H x W x C ]
        images = np.concatenate([images_pred, images_real], axis=-2)
        save_videos(savepath, *images)

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        '''
            diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
        '''
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [10, 2, 0.5],
            'elevation': 0,
        }

        diffusion_path = to_np(diffusion_path)

        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')

            # [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[
                :, :, :self.observation_dim]

            frame = []
            for states in states_l:
                img = self.composite(None, states, dim=(
                    1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        save_video(savepath, frames, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

# -----------------------------------------------------------------------------#
# ---------------------------------- rollouts ---------------------------------#
# -----------------------------------------------------------------------------#


def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')

        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])


def rollouts_from_state_with_dict(env, state, actions_l):
    rollouts = []
    rollouts_states = []
    for actions in actions_l:
        obs, states = rollout_from_state_with_dict(env, state, actions)
        rollouts.append(obs)
        rollouts_states.append(states)
    return np.stack(rollouts), np.stack(rollouts_states)


def rollouts_from_state(env, state, actions_l):
    rollouts = []
    for actions in actions_l:
        obs = rollout_from_state(env, state, actions)
        rollouts.append(obs)
    return np.stack(rollouts)


def rollout_from_state_with_dict(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    get_obs_method = getattr(env, '_get_obs', env.get_obs)
    observations = [get_obs_method()]
    states = [state]
    env.reset()
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        st = env.get_env_state()
        states.append(np.concatenate([st['qpos'], st['qvel']]))
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        # if terminated early, pad with zeros
        observations.append(np.zeros(obs.size))
    return np.stack(observations), np.stack(states)


def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    get_obs_method = getattr(env, '_get_obs', env.get_obs)
    observations = [get_obs_method()]
    env.reset()
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        # if terminated early, pad with zeros
        observations.append(np.zeros(obs.size))
    return np.stack(observations)
