import collections
import numpy as np
import pdb

import torch
from torch.utils.data import Dataset

def sequence_dataset(env, dataset=None, **kwargs):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1
####

def to_tensor(x, dtype=torch.float, device='cpu'):
    return torch.tensor(x, dtype=dtype, device=device)

class MuJoCoDataset(Dataset):

    def __init__(self, env, H, max_path_length=1000, max_n_episodes=4000):
        itr = sequence_dataset(env)
        obs_dim = np.prod(env.observation_space.shape)
        act_dim = np.prod(env.action_space.shape)
        qstates = np.zeros((max_n_episodes, max_path_length, obs_dim + act_dim))
        rewards = np.zeros((max_n_episodes, max_path_length, 1))
        path_lengths = np.zeros(max_n_episodes, dtype=np.int)
        print('Loading qstates')
        for i, episode in enumerate(itr):
            # qstate = np.concatenate([episode['infos/qpos'], episode['infos/qvel'], episode['actions']], axis=-1)
            qstate = np.concatenate([episode['observations'], episode['actions']], axis=-1)
            reward = episode['rewards'][:, None]
            path_length = len(qstate)
            assert path_length <= max_path_length
            qstates[i, :path_length] = qstate
            rewards[i, :path_length] = reward
            path_lengths[i] = path_length
        qstates = qstates[:i+1]
        path_lengths = path_lengths[:i+1]

        indices = self.make_indices(path_lengths, H)

        self.env = env
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.qstates = qstates
        self.rewards = rewards
        self.path_lengths = path_lengths
        self.indices = indices

        self.normalize_all()

        print(f'[ MuJoCoDataset ] qstates: {qstates.shape}')

    def make_indices(self, path_lengths, horizon):
        indices = []
        for i, path_length in enumerate(path_lengths):
            for start in range(path_length - horizon + 1):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def normalize_all(self):
        '''
            normalizes to [-1, 1]
        '''
        dataset = self.env.get_dataset()
        # X = np.concatenate([dataset['infos/qpos'], dataset['infos/qvel'], dataset['actions']], axis=-1)
        X = np.concatenate([dataset['observations'], dataset['actions']], axis=-1)
        mins = self.mins = X.min(axis=0)
        maxs = self.maxs = X.max(axis=0)
        self.qstates_raw = self.qstates.copy()
        self.qstates = self.normalize(self.qstates)
        # ## [ 0, 1 ]
        # self.qstates = (self.qstates - mins) / (maxs - mins)
        # ## [ -1, 1 ]
        # self.qstates = self.qstates * 2 - 1

    def normalize(self, x, dim=None):
        if dim:
            mins = self.mins[:dim]
            maxs = self.maxs[:dim]
        else:
            mins = self.mins
            maxs = self.maxs
        ## [ 0, 1 ]
        x = (x - mins) / (maxs - mins)
        ## [ -1, 1 ]
        x = x * 2 - 1
        return x

    def unnormalize(self, x):
        '''
            x : [ 0, 1 ]
        '''
        if not x.max() <= 1 and x.min() >= 0:
            print(f'x range: ({x.min():.4f}, {x.max():.4f})')
            pdb.set_trace()
        mins = to_tensor(self.mins, dtype=x.dtype, device=x.device)
        maxs = to_tensor(self.maxs, dtype=x.dtype, device=x.device)
        return x * (maxs - mins) + mins

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        qstates = self.qstates[path_ind, start:end]
        assert qstates.max() <= 1.0 + eps and qstates.min() >= -1.0 - eps, f'qstates range: ({qstates.min():.4f}, {qstates.max():.4f})'
        return to_tensor(qstates[None])

def get_dataset(mode, *args, **kwargs):
    dataset_class = {
        'reward': RewardDataset,
        'value': ValueDataset,
    }[mode]
    return dataset_class(*args, **kwargs)

class RewardDataset(MuJoCoDataset):

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, H=None, **kwargs)
        self.discount = discount
        self.max_path_length = self.qstates.shape[1]
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def make_indices(self, path_lengths, *args, **kwargs):
        indices = []
        for i, path_length in enumerate(path_lengths):
            for start in range(path_length):
                indices.append((i, start))
        indices = np.array(indices)
        return indices

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start = self.indices[idx]
        qstate = self.qstates[path_ind, start]
        reward = self.rewards[path_ind, start]

        return qstate, reward

class ValueDataset(RewardDataset):

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start = self.indices[idx]
        qstate = self.qstates[path_ind, start]

        ## [ path_length - start - 1 ]
        rewards_to_go = self.rewards[path_ind, start+1:]
        value = (rewards_to_go * self.discounts[:len(rewards_to_go)]).sum().reshape(1)
        return qstate, value

class NoisyWrapper:

    def __init__(self, dataset, timesteps):
        from diffusion.denoising_diffusion_pytorch import cosine_beta_schedule, extract

        self.dataset = dataset
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1. - alphas_cumprod))
        self.extract = extract

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __getitem__(self, *args, **kwargs):
        x, y = self.dataset.__getitem__(*args, **kwargs)
        x = torch.tensor(x)
        y = torch.tensor(y)

        t = torch.randint(0, self.timesteps, (1,)).long()
        noise = torch.randn_like(x)

        sample = (
            self.extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )
        return sample, t, y


class ConditionalMuJoCoDataset(MuJoCoDataset):

    conditions = [
        ([], 1), ## none
        ([0], 1), ## first
        # ([-1], 1), ## last
        # ([0,-1], 1), ## first and last
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ## make conditions
        conditions_k, conditions_p = zip(*self.conditions)

        self.conditions_k = np.array(conditions_k, dtype=np.object)
        self.conditions_p = np.array(conditions_p) / sum(conditions_p)

    def __getitem__(self, *args, **kwargs):
        x = super().__getitem__(*args, **kwargs)

        cond = np.random.choice(self.conditions_k, p=self.conditions_p)
        mask = torch.zeros_like(x)

        assert x.shape[-1] == self.obs_dim + self.act_dim
        for t in cond:
            mask[:, t, :self.obs_dim] = 1

        # joined = torch.cat([x, mask], axis=0)
        return x, mask
