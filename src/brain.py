# src/brain.py
import torch.nn as nn

class ImitationBrain(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImitationBrain, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    def forward(self, x):
        return self.network(x)