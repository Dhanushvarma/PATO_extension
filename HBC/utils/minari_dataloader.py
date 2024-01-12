import minari
import torch
from torch.utils.data import Dataset, DataLoader


class MinariObservationDataset(Dataset):
    def __init__(self, input_minari_dataset, horizon, mode='cVAE'):
        self.h = horizon
        self.mode = mode
        self.pairs = []
        self.sequences = []

        for episode in input_minari_dataset:
            observations = episode.observations['observation']
            actions = episode.observations['action']
            # Skip episodes that are shorter than h timesteps
            if len(observations) > horizon:
                for i in range(len(observations) - horizon):
                    # For cVAE mode
                    self.pairs.append((observations[i], observations[i + horizon]))
                    # For RNN mode
                    self.sequences.append((observations[i:i + horizon + 1], actions[i:i + horizon]))

    def set_mode(self, mode):
        assert mode in ['cVAE', 'RNN']
        self.mode = mode

    def __len__(self):
        if self.mode == 'cVAE':
            return len(self.pairs)
        elif self.mode == 'RNN':
            return len(self.sequences)

    def __getitem__(self, idx):
        if self.mode == 'cVAE':
            s_t, s_g = self.pairs[idx]
            return torch.tensor(s_t, dtype=torch.float32), torch.tensor(s_g, dtype=torch.float32)
        elif self.mode == 'RNN':
            states, actions = self.sequences[idx]
            return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)

def collate_fn(batch):
    if isinstance(batch[0][0], torch.Tensor):
        s_t = torch.stack([item[0] for item in batch])
        s_g = torch.stack([item[1] for item in batch])
        return s_t, s_g
    else:
        states = torch.stack([item[0] for item in batch])
        actions = torch.stack([item[1] for item in batch])
        return states, actions


# ------------------------------------------------------------------------------
# Example Usage
# dataset = MinariObservationDataset(input_minari_dataset, horizon, mode='cVAE')
# dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)


# dataset.set_mode('RNN')
# dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
# ------------------------------------------------------------------------------
