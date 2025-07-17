import torch
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from model.model import AttentionMLP
from model.gym_wrapper import TresetteGymWrapper

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Wrapper function with cloned opponent injection
def make_env(opponent_model=None, device='cpu'):
    def _init():
        return TresetteGymWrapper(opponent_model=opponent_model, device=device)
    return _init

# Custom features extractor using your AttentionMLP
class CustomMLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.model = AttentionMLP(input_dim=observation_space.shape[0], hidden_dim=256, output_dim=features_dim)

    def forward(self, x):
        return self.model(x)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    wandb.init(project="tresette-agent", sync_tensorboard=True)

    # Initial environment with no opponent model (random opponent)
    env = DummyVecEnv([make_env(None, device)])

    # Logger config
    new_logger = configure("logs", ["stdout", "csv", "tensorboard"])

    policy_kwargs = dict(
        features_extractor_class=CustomMLPExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[128], vf=[128])]
    )

    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log='./runs/tresette', device=device)
    model.set_logger(new_logger)

    # Training parameters
    total_timesteps = 500_000
    clone_interval = 50_000  # timesteps between cloning opponent
    current_timesteps = 0

    while current_timesteps < total_timesteps:
        next_checkpoint = min(current_timesteps + clone_interval, total_timesteps)
        model.learn(total_timesteps=clone_interval, reset_num_timesteps=False)
        current_timesteps = next_checkpoint

        # Clone current model for opponent
        opponent_model = PPO.load("tresette_agent_clone.zip") if current_timesteps > clone_interval else None

        # Save current model
        model.save("tresette_agent_clone")

        # Re-create env with cloned opponent
        env.close()
        env = DummyVecEnv([make_env(opponent_model, device)])
        model.set_env(env)

    # Final save
    model.save("tresette_agent_final")

if __name__ == "__main__":
    main()
