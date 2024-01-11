import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class CVAE(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(state_dim * 2, latent_dim)  # Encoder expects combined state and goal dimensions
        self.decoder = Decoder(latent_dim, state_dim)  # Decoder expects latent dimension and state dimension

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, state, goal):
        # Concatenate state and goal for the encoder
        combined = torch.cat((state, goal), dim=1)
        mean, log_var = self.encoder(combined)
        z = self.reparameterize(mean, log_var)
        # Decoder takes the latent vector z and the state
        return self.decoder(z, state), mean, log_var

    def generate(self, z, state):
        # Generate a state given a latent vector z and a state
        return self.decoder(z, state)
