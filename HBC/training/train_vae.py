import argparse
import torch
import torch.optim as optim
import sys
import minari
import wandb
import subgoal_predictor.utils.common_utils as CU
sys.path.append('/home/dpenmets/PycharmProjects/PATO_extension')
from torch.utils.data import DataLoader
from subgoal_predictor.models.vae import CVAE
from subgoal_predictor.utils.minari_dataloader import MinariObservationDataset, collate_fn




def train_cvae(config):
    wandb.init(project=config['wandb_project'], entity=config['wandb_entity'], name=config['wandb_name'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(config['state_dim'], config['latent_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Optional: resume from checkpoint
    if config.get('resume'):
        print(f"Resume checkpoint from: {config['resume']}")
        checkpoint = torch.load(config['resume'], map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    # Setup DataLoader
    h = config['horizon']  # Define your timestep gap
    minari_dataset = minari.load_dataset("pointmaze-umaze-v1")
    minari_observation_dataset = MinariObservationDataset(minari_dataset, h)

    # Split observation_dataset into training and testing sets
    split_index = int(len(minari_observation_dataset) * config['train_split'])
    train_dataset = minari_observation_dataset[:split_index]
    test_dataset = minari_observation_dataset[split_index:]

    # Create DataLoaders for train and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=collate_fn)

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        for batch_idx, (s_t, s_g) in enumerate(train_dataloader):
            s_t, s_g = s_t.to(device), s_g.to(device)

            # Forward pass
            reconstructed, mean, log_var = model(s_t, s_g)

            # Compute loss
            loss, recon_loss, kl_div = CU.vae_loss(reconstructed, s_g, mean, log_var)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Log training batch metrics to wandb
            wandb.log({"train_loss": loss.item(), "train_recon_loss": recon_loss.item(), "train_kl_div": kl_div.item(),
                       "epoch": epoch})

        # Average training loss
        train_loss /= len(train_dataset)
        wandb.log({"average_train_loss": train_loss, "epoch": epoch})

        # Validation loop
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (s_t, s_g) in enumerate(test_dataloader):
                s_t, s_g = s_t.to(device), s_g.to(device)
                reconstructed, mean, log_var = model(s_t, s_g)
                loss, recon_loss, kl_div = CU.vae_loss(reconstructed, s_g, mean, log_var)
                test_loss += loss.item()

                # Optionally log individual batch metrics for testing
                # wandb.log({"test_loss": loss.item(), "epoch": epoch})

        # Average testing loss
        test_loss /= len(test_dataset)
        wandb.log({"average_test_loss": test_loss, "epoch": epoch})

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    # Load configuration from a YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    input_config = load_config(args.config)
    train_cvae(input_config)
