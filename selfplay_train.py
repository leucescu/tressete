import os
import wandb
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from model.training_config import TrainingConfig
from model.gym_wrapper import TresetteGymWrapper
from model.model import TressetteMLP


# === Custom Feature Extractor with Dropout ===
class CustomMLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, dropout_rate=0.1):
        super().__init__(observation_space, features_dim)
        self.model = nn.Sequential(
            TressetteMLP(
                input_dim=214,
                hidden_dim=512,
                output_dim=features_dim,
                dropout_rate=dropout_rate
            ),
            nn.Dropout(dropout_rate)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.model(x)


# === Masked Policy ===
class MaskedPolicy(ActorCriticPolicy):
    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        logits = distribution.distribution.logits

        mask = self.get_valid_actions_mask(obs)  # Use mask from last 10 obs dims
        masked_logits = logits.masked_fill(mask == 0, float("-inf"))

        new_distribution = torch.distributions.Categorical(logits=masked_logits)

        if deterministic:
            actions = torch.argmax(masked_logits, dim=1)
        else:
            actions = new_distribution.sample()

        log_prob = new_distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def get_valid_actions_mask(self, obs):
        # obs shape: (batch_size, 214) — last 10 dims are valid action mask
        return obs[:, 204:214]  # shape: (batch_size, 10)


# === Learning rate scheduler ===
def cosine_schedule(initial_value, min_lr=1e-5, warmup_fraction=0.3):
    def scheduler(progress_remaining):
        if progress_remaining > (1 - warmup_fraction):
            return initial_value * (1 - progress_remaining) / warmup_fraction
        progress = (1 - progress_remaining - warmup_fraction) / (1 - warmup_fraction)
        return min_lr + 0.5 * (initial_value - min_lr) * (1 + np.cos(np.pi * progress))
    return scheduler


# === Env Factory ===
def make_env(opponent_model=None, use_heuristic=False, device='cpu'):
    def _init():
        return TresetteGymWrapper(
            opponent_model=None if use_heuristic else opponent_model,
            opponent_policy="heuristic" if use_heuristic else None,
            device=device
        )
    return _init


# === Training ===
def main():
    cfg = TrainingConfig()

    print(f"Using device: {cfg.device}")
    wandb.init(project="tresette-agent", sync_tensorboard=True)
    logger = configure(cfg.log_dir, ["stdout", "csv", "tensorboard"])

    policy_kwargs = dict(
        features_extractor_class=CustomMLPExtractor,
        features_extractor_kwargs=dict(features_dim=512, dropout_rate=0.1),
        net_arch=[dict(pi=[256, 256], vf=[512, 512, 256])],
        activation_fn=nn.LeakyReLU,
        ortho_init=True
    )

    model = PPO(
        policy=MaskedPolicy,
        env=DummyVecEnv([make_env(use_heuristic=True, device=cfg.device)]),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=cfg.tensorboard_log,
        device=cfg.device,
        learning_rate=cosine_schedule(cfg.initial_lr, cfg.min_lr),
        ent_coef=cfg.ent_coef_start,
        vf_coef=cfg.vf_coef,
        normalize_advantage=True,
        target_kl=cfg.kl_coef,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
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
