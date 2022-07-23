import os
import copy
import numpy as np
import pickle as pkl
import h5py
import pdb
from glob import glob

def write_hdf5(save_path, data_dict):
    f = h5py.File(save_path, 'w')
    for key, val in data_dict.items():
        f.create_dataset(key, data=val)
    f.close()

agent = 'sac'
epoch = 6700
fullobs = True

observation_dim = 47
action_dim = 11

data_dict = {
    'observations': np.zeros((0, observation_dim)),
    'actions': np.zeros((0, action_dim)),
    'rewards': np.zeros(0),
    'terminals': np.zeros(0),
    'timeouts': np.zeros(0),
}

dataset = "/data/vision/billf/scratch/yilundu/pddlstream/output_kuka_stacking_real/*.npy"
paths = glob(dataset)

for path in paths:
    data = np.load(path)
    data = data[::2]
    actions = data[1:] - data[:-1]
    actions = np.concatenate([actions[:, :7], actions[:, 14:15], actions[:, 22:23], actions[:, 30:31], actions[:, 38:39]], axis=-1)
    append_actions = np.zeros((1, actions.shape[-1]))
    actions = np.concatenate([actions, append_actions], axis=0)

    start = data[0]
    end = data[-1]

    diff = np.abs(end[7:] - start[7:]).reshape((-1, 8)).mean(axis=-1)
    cube_idx = np.argmax(diff)

    cube_pos_final = end[7:].reshape((-1, 8))[:, :2]
    sort_idx = np.argsort(np.linalg.norm(cube_pos_final - cube_pos_final[cube_idx:cube_idx+1], axis=-1))
    stack_id = sort_idx[1]

    cube_id = np.eye(4)[cube_idx]
    stack_id = np.eye(4)[stack_id]


    select_state = data[:, 7+cube_idx*8:10+cube_idx*8]
    final_pos = end[7+cube_idx*8:10+cube_idx*8]

    state_dist = np.linalg.norm(select_state - final_pos[None, :], axis=-1)
    state_mask = state_dist == 0
    idx = np.arange(state_dist.shape[0])[state_mask].min()


    observations = data
    path_length = data.shape[0]

    cube_id = np.concatenate([cube_id, stack_id], axis=-1)
    cube_observations = np.tile(cube_id[None, :], (path_length, 1))
    observations = np.concatenate([observations, cube_observations], axis=-1)

    timeouts = np.zeros(path_length)
    timeouts[-1] = 1
    terminals = np.zeros(path_length)
    rewards = np.zeros(path_length)
    rewards[idx] = 1

    path_dict = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'terminals': terminals,
        'timeouts': timeouts,
    }

    ##
    for key in data_dict.keys():
        print(key, data_dict[key].shape, path_dict[key].shape)
        data_dict[key] = np.concatenate([data_dict[key], path_dict[key]], axis=0)

## save with full state
save_path = f'cond_stacking.hdf5'
write_hdf5(save_path, data_dict)

## save with full state except x position
data_dict_partial = copy.deepcopy(data_dict)
data_dict_partial['observations'] = data_dict_partial['observations'][:,:-8]
print(f'partial: {data_dict_partial["observations"].shape}')

save_path = f'uncond_stacking.hdf5'
write_hdf5(save_path, data_dict_partial)

pdb.set_trace()
