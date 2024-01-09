import os
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import minari
from torch.nn.utils.rnn import pad_sequence
import torch


# def collate_fn(batch):
#
#     import pdb; pdb.set_trace()
#     return {
#         "id": torch.Tensor([x.id for x in batch]),
#         "seed": torch.Tensor([x.seed for x in batch]),
#         "total_timesteps": torch.Tensor([x.total_timesteps for x in batch]),
#         "observations": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.observations) for x in batch],
#             batch_first=True
#         ),
#         "actions": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.actions) for x in batch],
#             batch_first=True
#         ),
#         "rewards": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.rewards) for x in batch],
#             batch_first=True
#         ),
#         "terminations": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.terminations) for x in batch],
#             batch_first=True
#         ),
#         "truncations": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.truncations) for x in batch],
#             batch_first=True
#         )
#     }

#
# def collate_fn(batch):
#     batched_data = {
#         'id': torch.tensor([x.id for x in batch]),
#         'seed': torch.tensor([x.seed for x in batch]),
#         'total_timesteps': torch.tensor([x.total_timesteps for x in batch]),
#         'actions': pad_sequence([torch.tensor(x.actions, dtype=torch.float32) for x in batch], batch_first=True),
#         'rewards': pad_sequence([torch.tensor(x.rewards, dtype=torch.float32) for x in batch], batch_first=True),
#         'terminations': pad_sequence([torch.tensor(x.terminations, dtype=torch.bool) for x in batch], batch_first=True),
#         'truncations': pad_sequence([torch.tensor(x.truncations, dtype=torch.bool) for x in batch], batch_first=True),
#     }
#
#     # Process nested dictionary of observations
#     observation_keys = batch[0].observations.keys()
#     batched_data['observations'] = {
#         key: pad_sequence([torch.tensor(x.observations[key], dtype=torch.float64) for x in batch], batch_first=True) for
#         key in observation_keys}
#
#     return batched_data
#
#
# def collate_fn_no_padding(batch):
#     batched_data = {
#         'id': torch.tensor([x.id for x in batch]),
#         'seed': torch.tensor([x.seed for x in batch]),
#         'total_timesteps': torch.tensor([x.total_timesteps for x in batch]),
#         'actions': [torch.tensor(x.actions, dtype=torch.float32) for x in batch],
#         'rewards': [torch.tensor(x.rewards, dtype=torch.float32) for x in batch],
#         'terminations': [torch.tensor(x.terminations, dtype=torch.bool) for x in batch],
#         'truncations': [torch.tensor(x.truncations, dtype=torch.bool) for x in batch],
#     }
#
#     # Process nested dictionary of observations
#     observation_keys = batch[0].observations.keys()
#     batched_data['observations'] = {
#         key: [torch.tensor(x.observations[key], dtype=torch.float64) for x in batch] for key in observation_keys
#     }
#
#     return batched_data
#
#
# minari_dataset = minari.load_dataset("pointmaze-umaze-v1")
#
# # iterable = minari_dataset.sample_episodes(1)
# #
# # for i in iterable:
# #     print(i)
#
# # import pdb; pdb.set_trace()
#
#
# dataloader = DataLoader(minari_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn_no_padding)
# # dataloader = DataLoader(minari_dataset, batch_size=10, shuffle=True)
#
# # Iterate over the DataLoader
# for batch in dataloader:
#     import pdb;
#
#     pdb.set_trace()
#     print("Sample data batch:", batch['observations']['observation'].shape)
#     # break

import torch
from torch.utils.data import Dataset, DataLoader


class MinariObservationDataset(Dataset):
    def __init__(self, minari_dataset, h):
        self.h = h
        self.pairs = []
        for episode in minari_dataset:
            observations = episode.observations['observation']
            # Skip episodes that are shorter than h timesteps
            if len(observations) > h:
                for i in range(len(observations) - h):
                    self.pairs.append((observations[i], observations[i + h]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s_t, s_g = self.pairs[idx]
        return torch.tensor(s_t, dtype=torch.float32), torch.tensor(s_g, dtype=torch.float32)


def collate_fn(batch):
    s_t = torch.stack([item[0] for item in batch])
    s_g = torch.stack([item[1] for item in batch])
    return s_t, s_g


# Usage
h = 5  # Define your timestep gap
minari_dataset = minari.load_dataset("pointmaze-umaze-v1")
minari_observation_dataset = MinariObservationDataset(minari_dataset, h)
dataloader = DataLoader(minari_observation_dataset, batch_size=32, collate_fn=collate_fn)

# Iterate over the DataLoader
for s_t, s_g in dataloader:
    print("s_t:", s_t)
    print("s_g:", s_g)
    break
