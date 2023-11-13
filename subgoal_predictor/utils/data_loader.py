"""
data_loader.py

Implementation of the HDF5 Dataset Class
To load data from stored demonstration (HDF5) format to PyTorch Compatible format
"""

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class HDF5Dataset(Dataset):
    def __init__(self, file_path, horizon, transform=None):
        self.file_path = file_path
        self.file = h5py.File(file_path, 'r')
        self.data_group = self.file['data']
        self.demos = list(self.data_group.keys())
        self.transform = transform
        self.horizon = horizon
        self.state_curr, self.state_goal, self.actions_curr, self.model_file_curr = self.load_all_demos()
    def __len__(self):
        return len(self.demos)

    def load_all_demos(self):  # TODO: fix this as this will consume too much memory
        state_curr = []
        state_goal = []
        actions_curr = []
        model_file_curr = []

        for key in self.data_group.keys():  # Added zero index for each extraction as each demo contains a list which
            # contains all the states
            demo_group = self.data_group[key]
            states = demo_group['states'][:][0]
            # print(states)
            actions = demo_group['actions'][:][0]
            model_file = demo_group.attrs['model_file'][0]

            goal_states = np.zeros_like(states)
            goal_states[:-self.horizon] = states[self.horizon:]
            goal_states[-self.horizon:] = states[-1]

            state_curr.append(states)
            state_goal.append(goal_states)
            actions_curr.append(actions)
            model_file_curr.append(model_file)

        return state_curr, state_goal, actions_curr, model_file_curr

    def __getitem__(self, idx):
        if self.transform:
            state_c = self.transform(self.state_curr[idx])
            state_g = self.transform(self.state_goal[idx])
            action_c = self.transform(self.actions_curr[idx])
        else:
            state_c = self.state_curr[idx]
            state_g = self.state_goal[idx]
            action_c = self.actions_curr[idx]

        model_file_c = self.model_file_curr[idx]

        return state_c, state_g, action_c, model_file_c

    # def __getitem__(self, idx):  # This version if we want to treat each demonstration as a data point
    #     demo_key = self.demos[idx]
    #     demo_group = self.data_group[demo_key]
    #
    #     model_file = demo_group.attrs['model_file']
    #     states = demo_group['states'][:]
    #     actions = demo_group['actions'][:]
    #
    #     # Create goal_states array
    #     goal_states = np.zeros_like(states)
    #     goal_states[:-self.horizon] = states[self.horizon:]
    #     goal_states[-self.horizon:] = states[-1]  # For Last H states, the goal state will be the last state itself
    #
    #     if self.transform:
    #         states = self.transform(states)
    #         actions = self.transform(actions)
    #         goal_states = self.transform(goal_states)
    #
    #     return states, goal_states, actions, model_file

    def close(self):
        self.file.close()


def get_loader(file_path, batch_size=1, shuffle=True, num_workers=4, transform=None):
    dataset = HDF5Dataset(file_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# Example usage
file_path = '/home/dpenmets/robosuite/robosuite/models/assets/demonstrations/1699818905_9049585/demo.hdf5'  # Replace with your file path
train_loader = DataLoader(HDF5Dataset(file_path, horizon=5), batch_size=1, shuffle=True)

for state_c, state_g, action_c, model_file_c in train_loader:
    print(state_c)
    print(state_g)
    break
    pass
