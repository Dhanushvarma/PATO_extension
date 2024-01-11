import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super(Encoder, self).__init__()
        # State_dim*2 as we provide both s_t and s_g
        self.fc1 = nn.Linear(state_dim * 2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)

    def forward(self, state, goal):
        x = torch.cat((state, goal), dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var
