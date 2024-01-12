import torch
import unittest
from HBC.models.vae import CVAE  # Ensure this points to your CVAE model
from torchsummary import summary


class TestCVAE(unittest.TestCase):

    def setUp(self) -> None:
        self.model = CVAE(3, 10)  # Assuming 3 is state_dim and 10 is latent_dim

    def test_summary(self):
        # Assuming the input shape is (3, 64, 64), modify if different
        print(summary(self.model, [(3, 64, 64), (3, 64, 64)], device='cpu'))

    def test_forward(self):
        x = torch.randn(16, 3, 64, 64)  # Random input tensor
        y = self.model(x, x)  # Assuming the same input is used for state and goal
        print("Model Output size:", y[0].size())

    def test_loss(self):
        x = torch.randn(16, 3, 64, 64)
        reconstructed, mean, log_var = self.model(x, x)
        # Implement or import your VAE loss function
        # Example: loss = vae_loss(reconstructed, x, mean, log_var)
        # print("Loss:", loss.item())


if __name__ == '__main__':
    unittest.main()
