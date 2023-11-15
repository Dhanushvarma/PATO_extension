import wandb


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
