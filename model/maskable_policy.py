import torch
import torch.nn as nn
import numpy as np
from tianshou.data import Batch
from tianshou.policy import PPOPolicy

class MaskableActor(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        
    def forward(self, obs, state=None, info={}):
        # Always extract observation and mask
        if isinstance(obs, (dict, Batch)):
            x = obs['obs']  # State vector
            mask = obs.get('action_mask', None)
        else:
            x = obs
            mask = None
        
        # Convert to tensor
        if not isinstance(x, torch.Tensor):
            device = next(self.parameters()).device
            x = torch.as_tensor(x, device=device, dtype=torch.float32)
        
        # Process through network
        logits = self.net(x)
        
        # Apply mask if available
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                device = next(self.parameters()).device
                mask = torch.as_tensor(mask, device=device, dtype=torch.float32)
            
            # Apply mask with extreme negative value
            neg_inf = torch.finfo(logits.dtype).min
            logits = torch.where(mask.bool(), logits, neg_inf)

        if torch.isnan(logits).any(): 
            raise ValueError("Logits contain NaN values after masking.")

        if torch.isinf(logits).any():
            raise ValueError("Logits contain  Inf values after masking.")
        
        return logits, state
    
    def dist(self, logits):
        return torch.distributions.Categorical(logits=logits)

class MaskablePPOPolicy(PPOPolicy):
    def learn(self, batch: Batch, batch_size: int, repeat: int = 1, **kwargs):
        self.optim.zero_grad()
        result = super().learn(batch, batch_size=batch_size, repeat=repeat, **kwargs)
        
        # Clip gradients before optimizer step
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self._grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self._grad_norm)

        
        self.optim.step()
        return result

    def predict(self, obs, mask=None, deterministic=False):
        self.eval()
        with torch.no_grad():
            output = self.actor(obs)
            logits = output[0] if isinstance(output, tuple) else output

            if mask is not None:
                # Mask invalid actions by setting their logits to a very low value
                logits = logits.masked_fill(~mask, -1e9)

            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()

        return action.cpu().numpy(), None

    def forward(self, batch, state=None):
        # Let the actor handle masking
        logits, state = self.actor(batch.obs, state=state, info=batch.info)
        
        # Calculate value
        if isinstance(batch.obs, (dict, Batch)) and 'obs' in batch.obs:
            state_vector = batch.obs['obs']
        else:
            state_vector = batch.obs
            
        value = self.critic(state_vector)
        
        # Create distribution
        dist = self.dist_fn(logits)
        
        # Sample action
        if self.training:
            action = dist.sample()
        else:
            action = logits.argmax(dim=-1)
        
        return Batch(logits=logits, act=action, state=state, dist=dist, value=value)