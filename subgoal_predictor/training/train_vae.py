import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('/home/dpenmets/PycharmProjects/PATO_extension' )
from torch.utils.data import DataLoader
from subgoal_predictor.models.vae import CVAE
from subgoal_predictor.utils.data_loader import HDF5Dataset
from subgoal_predictor.utils.common_utils import initialize_wandb

# Configuration parameters
config_params = {
    'learning_rate': 0.001,
    'epochs': 1000,
    'batch_size': 32
}


# Initialize WandB
wandb_run = initialize_wandb('pato_extension', 'dpenmets', config_params)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
train_dataset = HDF5Dataset('/home/dpenmets/robosuite/robosuite/models/assets/demonstrations/1699818905_9049585/demo'
                            '.hdf5' , horizon=5)
train_loader = DataLoader(train_dataset, batch_size=config_params['batch_size'], shuffle=True)

# Initialize model
input_dim = train_dataset.input_dim
output_dim = train_dataset.output_dim
latent_dim = 128  # Example latent dimension
model = CVAE(input_dim, input_dim, latent_dim, output_dim).to(device)

# Loss and optimizer
criterion = nn.MSELoss()  # Example criterion
optimizer = optim.Adam(model.parameters(), lr=config_params['learning_rate'])

# Training Loop
for epoch in range(config_params['epochs']):
    model.train()
    running_loss = 0.0

    for i, (state_c, state_g, action_c, model_file_c) in enumerate(train_loader):
        # Move tensors to the configured device
        state_c = state_c.float().to(device)
        state_g = state_g.float().to(device)

        print("state_c type", state_c)
        print("state_g type", state_g)
        # Forward pass
        output, mean, log_var = model(state_c, state_g)
        # Reconstruction loss
        recon_loss = criterion(output, state_g)
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        # Total loss
        loss = recon_loss + kl_loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Log training metrics
    wandb_run.log({'epoch': epoch, 'loss': running_loss / len(train_loader)})

    print(f'Epoch [{epoch+1}/{config_params["epochs"]}], Loss: {running_loss / len(train_loader):.4f}')

# Save the model
torch.save(model.state_dict(), 'cvae_model.pth')
wandb_run.save('cvae_model.pth')


print("Training complete")
