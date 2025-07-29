import torch
import numpy as np
from model.model import TressetteMLP
from model.state_encoder import encode_state  # Your encoder

# What the hell does this do ?
class TressetteAIPlayer:
    def __init__(self, model_path, player_index, device='cpu'):
        self.device = device
        self.model = TressetteMLP().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.player_index = player_index

    def act(self, obs, player, valid_actions):
        state = encode_state(obs, player)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(state_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
        
        # Mask invalid actions
        mask = np.zeros_like(probs)
        mask[valid_actions] = 1.0
        probs *= mask
        if probs.sum() == 0:
            action = np.random.choice(valid_actions)
        else:
            probs /= probs.sum()
            action = np.random.choice(len(probs), p=probs)

        return action
