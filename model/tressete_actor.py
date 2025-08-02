import numpy as np
import torch
import torch.nn as nn
from model.model import TressetteMLP
from tianshou.data import Batch

class TressetteActor(nn.Module):
    def __init__(self, input_dim=214, hidden_dim=512, action_dim=10):
        super().__init__()
        self.feature_extractor = TressetteMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            feature_dim=256
        )
        self.action_head = nn.Linear(256, action_dim)
   
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