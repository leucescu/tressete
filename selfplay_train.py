import os
import wandb
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from model.training_config import TrainingConfig
from model.gym_wrapper import TresetteGymWrapper
from model.model import TressetteMLP
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Wrapper for environment creation
def make_env(opponent_model=None, use_heuristic=False, device='cpu'):
    def _init():
        return TresetteGymWrapper(
            opponent_model=None if use_heuristic else opponent_model,
            opponent_policy="heuristic" if use_heuristic else None,
            device=device
        )
    return _init


class CustomMLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        self.model = nn.Sequential(
            TressetteMLP(
                input_dim=214,
                hidden_dim=512,
                output_dim=features_dim,
                dropout_rate=0.0        # Temporarily disabled for testing
            )
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.model(x)

# Learning rate scheduler using cosine decay with warmup
def cosine_schedule(initial_value, min_lr=1e-5, warmup_fraction=0.3):
    def scheduler(progress_remaining):
        if progress_remaining > (1 - warmup_fraction):
            return initial_value * (1 - progress_remaining) / warmup_fraction
        progress = (1 - progress_remaining - warmup_fraction) / (1 - warmup_fraction)
        return min_lr + 0.5 * (initial_value - min_lr) * (1 + np.cos(np.pi * progress))
    return scheduler

def main():
    cfg = TrainingConfig()

    print(f"Using device: {cfg.device}")
    wandb.init(project="tresette-agent", sync_tensorboard=True)

    logger = configure(cfg.log_dir, ["stdout", "csv", "tensorboard"])
    policy_kwargs = dict(
        features_extractor_class=CustomMLPExtractor,
        features_extractor_kwargs=dict(features_dim=512),  # Increased from 256
        net_arch=[dict(pi=[256, 256], vf=[512, 512, 256])],  # Deeper value head
        activation_fn=nn.LeakyReLU,
        ortho_init=True
    )

    model = PPO(
        'MlpPolicy',
        DummyVecEnv([make_env(use_heuristic=True, device=cfg.device)]),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=cfg.tensorboard_log,
        device=cfg.device,
        learning_rate=cosine_schedule(cfg.initial_lr, cfg.min_lr),
        ent_coef=cfg.ent_coef_start,
        vf_coef=cfg.vf_coef,
        normalize_advantage=True,  # Critical for stable updates
        target_kl=cfg.kl_coef,            # Add KL early stopping
        gamma=cfg.gamma,          # Increase from default 0.99
        gae_lambda=cfg.gae_lambda,     # Higher lambda for longer credit assignment
        n_steps=cfg.n_steps, 
    )
    model.set_logger(logger)

    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=cfg.checkpoint_path)

    current_timesteps = 0
    opponent_model = None
    use_heuristic = True

    while current_timesteps < cfg.total_timesteps:
        next_timesteps = min(current_timesteps + cfg.clone_interval, cfg.total_timesteps)

        if current_timesteps >= cfg.heuristic_cutoff and use_heuristic:
            use_heuristic = False
            if os.path.exists(cfg.clone_model_path + ".zip"):
                opponent_model = PPO.load(cfg.clone_model_path, device=cfg.device)
            else:
                print("Clone model not found. Continuing with heuristic.")

        env = DummyVecEnv([make_env(opponent_model, use_heuristic, cfg.device)])
        model.set_env(env)

        model.learn(
            total_timesteps=cfg.clone_interval,
            reset_num_timesteps=False,
            callback=checkpoint_callback
        )
        current_timesteps = next_timesteps

        model.save(cfg.clone_model_path)

    model.save(cfg.final_model_path)
    wandb.finish()

if __name__ == "__main__":
    main()
