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
            logits = self.actor(obs)
            if isinstance(logits, tuple):
                logits = logits[0]  # e.g., (logits, value)

            if mask is not None:
                logits = logits.clone()
                if logits.dim() == 2 and mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                logits[~mask] = -1e9  # mask out invalid actions

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
    
class CustomMaskablePPOPolicy(MaskablePPOPolicy):
    def learn(self, batch, batch_size, repeat, **kwargs):
        # Pre-update metrics
        with torch.no_grad():
            # Compute old value estimates
            old_values = self.critic(batch.obs).flatten()
            returns = batch.returns
            
            # Compute explained variance (old)
            variance = torch.var(returns)
            ev_old = 1 - torch.var(returns - old_values) / (variance + 1e-8) if variance > 1e-8 else torch.nan
            
            # Get old logits from actor
            old_logits, _ = self.actor(batch.obs)

        # Perform the PPO update with gradient clipping
        result = super().learn(batch, batch_size, repeat, **kwargs)
        
        # Post-update metrics
        with torch.no_grad():
            # Compute new value estimates
            new_values = self.critic(batch.obs).flatten()
            
            # Compute explained variance (new)
            variance = torch.var(returns)
            ev_new = 1 - torch.var(returns - new_values) / (variance + 1e-8) if variance > 1e-8 else torch.nan
            
            # Compute KL divergence
            new_logits, _ = self.actor(batch.obs)
            
            # Create distributions
            old_dist = torch.distributions.Categorical(logits=old_logits)
            new_dist = torch.distributions.Categorical(logits=new_logits)
            
            # Calculate KL divergence
            kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean().item()
        
        # Convert to Python floats for logging
        result.update({
            # KL divergence (Kullback-Leibler divergence) measures how much one probability distribution differs from another. In PPO:
            # It quantifies how much the updated policy has changed from the old policy
            #   *Low KL (0.01-0.05) indicates stable learning
            #   *High KL (>0.1) suggests policy is changing too rapidly, risking instability
            #   *Near-zero KL means policy isn't learning
            'kl': kl,
            # Explained variance of the value function before the update
            'ev_old': ev_old,
            # Explained variance of the value function after the update
            'ev_new': ev_new
            # If EV_new < EV_old, the update degraded value predictions
        })
        return result