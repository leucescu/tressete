import numpy as np
import torch
import torch.nn as nn
from tianshou.data import Batch

class TressetteMLP(nn.Module):
    def __init__(self, input_dim=214, hidden_dim=512, feature_dim=256):
        super().__init__()
        self.input_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01)
        )
        self.block1 = self._make_residual_block(hidden_dim)
        self.block2 = self._make_residual_block(hidden_dim)
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim // 2, feature_dim)
        )

    def _make_residual_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.01),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x):
        x = self.input_net(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        return self.feature_net(x)  # OUTPUT: features vector, NOT logits

class CriticMLP(nn.Module):
    def __init__(self, input_dim=214, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
            self._make_residual_block(hidden_dim),
            self._make_residual_block(hidden_dim),  # Extra block
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim//2, 1)
        )
    
    def _make_residual_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.01),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x):
        if isinstance(x, Batch):
            x = x.obs if hasattr(x, "obs") else x["obs"]
        elif isinstance(x, dict):
            x = x["obs"]

        # Safely convert to tensor if needed
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)

        x = x.to(next(self.parameters()).device)
        return self.net(x)
