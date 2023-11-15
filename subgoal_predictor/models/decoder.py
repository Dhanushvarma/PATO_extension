import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, latent_dim, state_dim, output_dim):
        """

        :param latent_dim: Size of the latent space which is 128
        :param state_dim: State dimension which is currently 32
        :param output_dim: Output is also a state, so it should also be 32
        """
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + state_dim, 128)  # Adjust input dimension
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.output = nn.Linear(128, output_dim)

    def forward(self, z, state):
        # Concatenate z and state
        combined = torch.cat((z, state), dim=1)
        combined = F.leaky_relu(self.fc1(combined))
        combined = F.leaky_relu(self.fc2(combined))
        combined = F.leaky_relu(self.fc3(combined))
        combined = F.leaky_relu(self.fc4(combined))
        combined = F.leaky_relu(self.fc5(combined))
        return self.output(combined)

# Example usage
# decoder = Decoder(latent_dim=..., state_dim=..., output_dim=...)
# output = decoder(latent_vector, current_state)
