import argparse
import torch
import torch.optim as optim
import sys
import minari
import wandb
import os
import HBC.utils.common_utils as CU
from torch.utils.data import DataLoader
from HBC.models.vae import CVAE
from HBC.utils.minari_dataloader import MinariObservationDataset, collate_fn

sys.path.append('/home/dpenmets/PycharmProjects/PATO_extension')

class cVAE_trainer:

    def __init__(self, cfg):

        self.cfg = cfg # Config from YAML file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CVAE(cfg['state_dim'], cfg['latent_dim']).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg['learning_rate'])

        # Loading the model onto device
        self.model.to(self.device)

        #Dataset
        minari_vae_data, self.env_name = CU.create_minari_dataset("pointmaze-umaze-v1", self.cfg['horizon'], "cVAE") #TODO: improve


        # Split observation_dataset into training and testing sets
        split_index = int(len(minari_vae_data) * cfg['train_split'])
        train_dataset = minari_vae_data[:split_index]
        test_dataset = minari_vae_data[split_index:]

        # Create DataLoaders for train and test sets
        self.train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], collate_fn=collate_fn)
        self.test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], collate_fn=collate_fn)

        #WandB logging
        self.wandb = CU.initialize_wandb(cfg["wandb"]["project_name"], cfg["wandb"]["entity_name"], cfg["wandb"]["cfg_params"])

        # File path for saving stuff (runs_cvae/run_name/ - 1)checkpoints, 2)model, 3)yaml file from cfg)
        self.cpt_path = "./runs_cvae/" + cfg['run_name'] + '/'
        self.checkpoints_dir = os.path.join(self.cpt_path, 'checkpoints/')
        self.model_file = os.path.join(self.cpt_path, 'model.pth')
        self.config_file = os.path.join(self.cpt_path, 'config.yaml')

        # Initialize directories and files based on resume flag
        CU.initialize_directories_and_files(self.cpt_path, self.checkpoints_dir, self.model_file, self.config_file, self.cfg['resume'])


    def train(self):

        # Load the latest checkpoint if resuming
        if self.cfg['resume']:
            start_epoch = self.load_latest_checkpoint()

        for epoch in range(self.cfg['num_epochs']):

            self.model.train()
            train_loss = 0.0
            #---------- Training Code ----------#
            
            for batch_idx, (s_t, s_g) in enumerate(self.train_dataloader):
                s_t, s_g = s_t.to(self.device), s_g.to(self.device)

                # Forward pass
                reconstructed, mean, log_var = self.model(s_t, s_g)

                # Compute loss
                loss, recon_loss, kl_div = CU.vae_loss(reconstructed, s_g, mean, log_var)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                # Log training batch metrics to wandb
                wandb.log({"train_loss": loss.item(), "train_recon_loss": recon_loss.item(), "train_kl_div": kl_div.item(),
                        "epoch": epoch})

            # Average training loss
            train_loss /= len(self.train_dataloader)
            wandb.log({"average_train_loss": train_loss, "epoch": epoch})

            #---------- Testing ----------#
            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_idx, (s_t, s_g) in enumerate(self.test_dataloader):
                    s_t, s_g = s_t.to(self.device), s_g.to(self.device)
                    reconstructed, mean, log_var = self.model(s_t, s_g)
                    loss, recon_loss, kl_div = CU.vae_loss(reconstructed, s_g, mean, log_var)
                    test_loss += loss.item()

                    # Optionally log individual batch metrics for testing
                    wandb.log({"test_loss": loss.item(), "epoch": epoch})

            # Average testing loss
            test_loss /= len(self.test_dataloader)
            wandb.log({"average_test_loss": test_loss, "epoch": epoch})

            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

            # Saving Checkpoint
            if epoch % self.cfg['log_freq'] == 0:
                self.save_checkpoint(epoch)

        
        # Save the final model
        torch.save(self.model.state_dict(), self.model_file)
        print(f"Model saved at {self.model_file}")

    
    def save_checkpoint(self, epoch):

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cfg': self.cfg
        }

        checkpoint_path = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    
    def load_latest_checkpoint(self):

        checkpoints = [ckpt for ckpt in os.listdir(self.checkpoints_dir) if ckpt.endswith('.pth')]
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found for resuming.")

        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(self.checkpoints_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Resumed from checkpoint: {checkpoint_path}")
        return checkpoint['epoch'] + 1
    

    def print_parameter_summary(self):
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

        print("\nOptimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            print(var_name, "\t", self.optimizer.state_dict()[var_name])


    #TODO: write code for hyperparameter tuning
            
        

def main(config_path, resume):
    # Load configuration
    cfg = CU.load_config(config_path)

    # Update resume flag in the configuration
    cfg['resume'] = resume

    # Initialize the trainer
    trainer = cVAE_trainer(cfg)

    # Train
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--resume", type=lambda x: (str(x).lower() == 'true'), required=False, default=False,
                        help="Flag to resume training from the last checkpoint.")
    args = parser.parse_args()

    main(args.config, args.resume)












            