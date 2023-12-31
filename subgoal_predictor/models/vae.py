import torch
import torch.nn as nn
from .decoder import Decoder
from .encoder import Encoder


class CVAE(nn.Module):
    def __init__(self, state_dim, goal_dim, latent_dim, output_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(state_dim, goal_dim, latent_dim)
        self.decoder = Decoder(latent_dim, state_dim, output_dim)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, state, goal):
        mean, log_var = self.encoder(state, goal)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z, state), mean, log_var

    def generate(self, z, state):
        return self.decoder(z, state)
