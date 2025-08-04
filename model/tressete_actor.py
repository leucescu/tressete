import numpy as np
import torch
import torch.nn as nn
from model.model import TressetteMLP
from tianshou.data import Batch

class TressetteActor(nn.Module):
    def __init__(self, input_dim=214, hidden_dim=1024, action_dim=10):  # 2x wider
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
            self._make_residual_block(hidden_dim),
            self._make_residual_block(hidden_dim),
            self._make_residual_block(hidden_dim),  # Extra block
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.01)
        )
        self.action_head = nn.Linear(hidden_dim//2, action_dim)
    
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
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x = x.to(next(self.parameters()).device)
        features = self.feature_extractor(x)
        logits = self.action_head(features)
        return logits