import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.output = nn.Linear(128, output_dim)

    def forward(self, z):
        z = F.leaky_relu(self.fc1(z))
        z = F.leaky_relu(self.fc2(z))
        z = F.leaky_relu(self.fc3(z))
        z = F.leaky_relu(self.fc4(z))
        z = F.leaky_relu(self.fc5(z))
        return self.output(z)
