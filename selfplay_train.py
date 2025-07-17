import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from model.gym_wrapper import TresetteGymWrapper
from model.model import AttentionMLP
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

# Linear learning rate schedule
def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize Wandb for experiment tracking, logs will sync with tensorboard
    wandb.init(project="tresette-agent", sync_tensorboard=True)

    # Start environment with no opponent model (random behavior initially)
    env = DummyVecEnv([make_env(None, device)])

    # Logger configuration for stable_baselines3
    new_logger = configure("logs", ["stdout", "csv", "tensorboard"])

    # Define the policy network with your custom Attention MLP extractor
    policy_kwargs = dict(
        features_extractor_class=CustomMLPExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]

    )

    initial_lr = 3e-4  # Initial learning rate

    # Create PPO model, set device to CUDA if available, and apply learning rate schedule
    model = PPO(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log='./runs/tresette',
        device=device,
        learning_rate=linear_schedule(initial_lr)
    )
    model.set_logger(new_logger)

    # Checkpoint callback to save every 10,000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path='./checkpoints/')

    total_timesteps = 300_000
    clone_interval = 50_000  # Clone opponent every 50k timesteps
    current_timesteps = 0
    opponent_model = None  # No opponent at start (random)

    while current_timesteps < total_timesteps:
        next_timesteps = min(current_timesteps + clone_interval, total_timesteps)

        # Train for clone_interval steps with checkpoint saving
        model.learn(total_timesteps=clone_interval, reset_num_timesteps=False, callback=checkpoint_callback)
        current_timesteps = next_timesteps

        # Save current model for cloning
        model.save("tresette_agent_clone")

        # Load cloned model for opponent after first interval
        if current_timesteps >= clone_interval:
            opponent_model = PPO.load("tresette_agent_clone", device=device)

        # Close old env and recreate with new cloned opponent
        env.close()
        env = DummyVecEnv([make_env(opponent_model, device)])
        model.set_env(env)

    # Final save
    model.save("tresette_agent_final")

if __name__ == "__main__":
    main()
