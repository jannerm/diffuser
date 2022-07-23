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
        obs_dim = 12
        qstates = np.zeros((max_n_episodes, max_path_length, obs_dim))
        path_lengths = np.zeros(max_n_episodes, dtype=np.int)
        print('Loading qstates')
        for i, episode in enumerate(itr):
            qstate = np.concatenate([episode['infos/qpos'], episode['infos/qvel']], axis=-1)
            path_length = len(qstate)
            assert path_length <= max_path_length
            qstates[i, :path_length] = qstate
            path_lengths[i] = path_length
        qstates = qstates[:i+1]
        path_lengths = path_lengths[:i+1]

        ## make indices
        print('Making indices')
        indices = []
        for i, path_length in enumerate(path_lengths):
            for start in range(path_length - H + 1):
                end = start + H
                indices.append((i, start, end))
        indices = np.array(indices)

        self.env = env
        self.obs_dim = obs_dim
        self.qstates = qstates
        self.path_lengths = path_lengths
        self.indices = indices

        self.normalize()

        print(f'[ MuJoCoDataset ] qstates: {qstates.shape}')

    def normalize(self):
        '''
            normalizes to [-1, 1]
        '''
        dataset = self.env.get_dataset()
        X = np.concatenate([dataset['infos/qpos'], dataset['infos/qvel']], axis=-1)
        mins = self.mins = X.min(axis=0)
        maxs = self.maxs = X.max(axis=0)
        ## [ 0, 1 ]
        self.qstates = (self.qstates - mins) / (maxs - mins)
        ## [ -1, 1 ]
        self.qstates = self.qstates * 2 - 1

    def unnormalize(self, x):
        '''
            x : [ 0, 1 ]
        '''
        assert x.max() <= 1 and x.min() >= 0, f'x range: ({x.min():.4f}, {x.max():.4f})'
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

class ConditionalMuJoCoDataset(MuJoCoDataset):

    conditions = [
        # ([], 1), ## none
        ([0], 1), ## first
        ([-1], 1), ## last
        ([0,-1], 1), ## first and last
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ## make conditions
        conditions_k, conditions_p = zip(*self.conditions)

        self.conditions_k = np.array(conditions_k, dtype=np.object)
        self.conditions_p = np.array(conditions_p) / sum(conditions_p)
         # = [p for (_, p) in self.conditions]
        # self.conditions_k = []
        # pdb.set_trace()

    def __getitem__(self, *args, **kwargs):
        x = super().__getitem__(*args, **kwargs)

        cond = np.random.choice(self.conditions_k, p=self.conditions_p)
        mask = torch.zeros_like(x)
        for t in cond:
            mask[:, t] = 1

        # joined = torch.cat([x, mask], axis=0)
        return x, mask
