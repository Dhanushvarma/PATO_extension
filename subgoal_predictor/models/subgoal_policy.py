import torch
import torch.nn as nn


class AutoregressiveLSTMPolicy(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, steps_L, hidden_dim=256):
        super(AutoregressiveLSTMPolicy, self).__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.steps_L = steps_L
        self.hidden_dim = hidden_dim

        # Input processing MLP
        self.input_mlp = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU()
        )

        # LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_dim + action_dim, hidden_size=hidden_dim, batch_first=True)

        # Output MLP to generate actions
        self.output_mlp = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, goal):
        # Initial input processing
        combined_input = torch.cat((state, goal), dim=1)
        lstm_input = self.input_mlp(combined_input)

        # Placeholder for actions generated at each step
        actions = []

        hidden = None
        for _ in range(self.steps_L):
            # Reshape for LSTM
            lstm_input = lstm_input.unsqueeze(1)  # Add sequence dimension

            # LSTM forward pass
            lstm_out, hidden = self.lstm(lstm_input, hidden)

            # Predict next action
            next_action = self.output_mlp(lstm_out.squeeze(1))
            actions.append(next_action)

            # Prepare input for the next time step
            lstm_input = torch.cat((lstm_out, next_action), dim=1)

        # Concatenate all actions to form the sequence
        action_sequence = torch.stack(actions, dim=1)

        return action_sequence
