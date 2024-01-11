import minari
import torch
from torch.utils.data import Dataset, DataLoader


class MinariObservationDataset(Dataset):
    def __init__(self, input_minari_dataset, horizon):
        self.h = horizon
        self.pairs = []
        for episode in input_minari_dataset:
            observations = episode.observations['observation']
            # Skip episodes that are shorter than h timesteps
            if len(observations) > horizon:
                for i in range(len(observations) - horizon):
                    self.pairs.append((observations[i], observations[i + horizon]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s_t, s_g = self.pairs[idx]
        return torch.tensor(s_t, dtype=torch.float32), torch.tensor(s_g, dtype=torch.float32)


def collate_fn(batch):
    s_t = torch.stack([item[0] for item in batch])
    s_g = torch.stack([item[1] for item in batch])
    return s_t, s_g

# ------------------------------------------------------------------------------
# Example Usage


h = 5  # Define your timestep gap
minari_dataset = minari.load_dataset("pointmaze-umaze-v1")
minari_observation_dataset = MinariObservationDataset(minari_dataset, h)
dataloader = DataLoader(minari_observation_dataset, batch_size=32, collate_fn=collate_fn)

# Iterate over the DataLoader
for s_t, s_g in dataloader:
    print("s_t:", s_t)
    print("s_g:", s_g)
    break
