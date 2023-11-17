import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, TensorDataset
from subgoal_predictor.utils.data_loader import HDF5Dataset
from subgoal_predictor.models.subgoal_policy import AutoregressiveLSTMPolicy

# Configuration parameters
config_params = {
    'learning_rate': 0.001,
    'epochs': 100,
    'sequence_length': 1,
    'steps_L': 10,
    'batch_size': 32,
    'state_dim': 32,
    'goal_dim': 32,
    'action_dim': 32,
    'ensemble_size': 5  # Number of policies in the ensemble
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_sequences(data, sequence_length, step_L):
    """
    Create sequences from the data for LSTM training.
    Each sequence consists of (state, goal_state, action_sequence) for L steps.
    """
    sequences = []
    for i in range(len(data) - sequence_length - step_L + 1):
        sequence = (data['state'][i], data['goal'][i], data['action'][i:i+step_L])
        sequences.append(sequence)
    return sequences


# Initialize ensemble of models
ensemble = [AutoregressiveLSTMPolicy(config_params['state_dim'], config_params['goal_dim'], config_params['action_dim'], config_params['steps_L']).to(device) for _ in range(config_params['ensemble_size'])]

# Loss function and optimizers for each model in the ensemble
criterions = [nn.MSELoss() for _ in range(config_params['ensemble_size'])]
optimizers = [optim.Adam(model.parameters(), lr=config_params['learning_rate']) for model in ensemble]

# Load and preprocess dataset
train_dataset = HDF5Dataset('/path/to/your/demo.hdf5', horizon=config_params['sequence_length'])
all_sequences = []
for demo in train_dataset.demos:
    demo_data = train_dataset[demo]
    demo_sequences = create_sequences(demo_data, config_params['sequence_length'], config_params['steps_L'])
    all_sequences.extend(demo_sequences)

# Training Loop
for epoch in range(config_params['epochs']):
    for model, criterion, optimizer in zip(ensemble, criterions, optimizers):
        model.train()
        total_loss = 0.0

        # Shuffle sequences for each model
        random.shuffle(all_sequences)

        for sequence in all_sequences:
            state, goal, action_sequence = sequence
            state = torch.tensor(state).float().to(device).unsqueeze(0)
            goal = torch.tensor(goal).float().to(device).unsqueeze(0)
            action_sequence = torch.tensor(action_sequence).float().to(device).unsqueeze(0)

            optimizer.zero_grad()

            # Forward pass
            predicted_action_sequence = model(state, goal)

            # Loss calculation
            loss = criterion(predicted_action_sequence, action_sequence)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(all_sequences)
        print(f'Epoch [{epoch+1}/{config_params["epochs"]}], Model {ensemble.index(model) + 1}, Loss: {avg_loss:.4f}')

# Save the models
for idx, model in enumerate(ensemble):
    torch.save(model.state_dict(), f'autoregressive_lstm_policy_{idx}.pth')
