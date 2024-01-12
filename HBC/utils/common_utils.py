import wandb
import yaml
import torch
import torch.nn.functional as F
import minari
from HBC.utils.minari_dataloader import MinariObservationDataset, collate_fn
import gymnasium as gym
import os
import yaml

def initialize_wandb(project_name, entity_name, config_params):
    """
    Initialize a Weights & Biases run.

    :param project_name: str, name of the WandB project.
    :param entity_name: str, your WandB username or team name.
    :param config_params: dict, configuration parameters for the run (like hyperparameters).
    :return: wandb.run object, the initialized WandB run.
    """
    # Initialize the WandB run
    wandb_run = wandb.init(project=project_name, entity=entity_name, config=config_params)

    return wandb_run


def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        return yaml.safe_load(file)


def save_checkpoint(model, optimizer, step, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step}
    checkpoint_path = checkpoint_dir / f"model.ckpt-{step}.pt"
    torch.save(checkpoint_state, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def vae_loss(reconstructed, original, mean, log_var):
    # Reconstruction Loss (using Mean Squared Error)
    recon_loss = F.mse_loss(reconstructed, original, reduction='sum')

    # KL Divergence
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    # Total Loss
    return recon_loss + kl_div, recon_loss, kl_div


def create_minari_dataset(dataset_name, horizon, mode):

    minari_dataset = minari.load_dataset(dataset_name)
    dataset = MinariObservationDataset(minari_dataset, horizon=horizon, mode=mode)

    #Env name retrieval
    env_name = minari_dataset.recover_environment()

    return dataset, env_name


def initialize_directories_and_files(cpt_path, checkpoints_dir, model_file, config_file, cfg, resume):
    # Create the main directory for the run if it doesn't exist
    if not os.path.exists(cpt_path):
        os.makedirs(cpt_path)
        print(f"Created main directory: {cpt_path}")

    # Checkpoint directory handling
    if not os.path.exists(checkpoints_dir):
        if resume:
            raise FileNotFoundError(f"Checkpoint directory {checkpoints_dir} does not exist for resuming.")
        else:
            os.makedirs(checkpoints_dir)
            print(f"Created checkpoint directory: {checkpoints_dir}")

    # Handling model file for resuming
    if resume and not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model file {model_file} not found for resuming.")

    # Handling configuration file
    if not os.path.exists(config_file):
        if resume:
            raise FileNotFoundError(f"Config file {config_file} not found for resuming.")
        else:
            with open(config_file, 'w') as file:
                yaml.dump(cfg, file)
            print(f"Config file created at {config_file}")
    elif not resume:
        # Overwrite the config file in case of a new run
        with open(config_file, 'w') as file:
            yaml.dump(cfg, file)
        print(f"Config file overwritten at {config_file}")
