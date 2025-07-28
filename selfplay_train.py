import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from model.gym_wrapper import TresetteGymWrapper
from model.model import AttentionMLP
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


# Custom feature extractor using your attention-based MLP
class CustomMLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.model = AttentionMLP(
            input_dim=observation_space.shape[0],
            hidden_dim=256,
            output_dim=features_dim
        )

    def forward(self, x):
        return self.model(x)

def exp_decay_schedule(initial_value, decay_rate=0.99, min_lr=1e-5):
    def scheduler(progress_remaining):
        current_step = 1 - progress_remaining
        lr = initial_value * (decay_rate ** (current_step * 100))
        return max(min_lr, lr)
    return scheduler

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    wandb.init(project="tresette-agent", sync_tensorboard=True)

    total_timesteps = 3_000_000
    initial_lr = 3e-4
    decay_rate = 0.995  # tweak this for slower or faster decay
    min_lr = 1e-5

    new_logger = configure("logs", ["stdout", "csv", "tensorboard"])
    policy_kwargs = dict(
        features_extractor_class=CustomMLPExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )

    model = PPO(
        'MlpPolicy',
        DummyVecEnv([make_env(use_heuristic=True, device=device)]),  # heuristic start
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log='./runs/tresette',
        device=device,
        learning_rate=exp_decay_schedule(initial_lr, decay_rate, min_lr),
        ent_coef=0.01,
        vf_coef=1.0
    )
    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path='./checkpoints/')

    clone_interval = 50_000
    heuristic_cutoff = 50_000
    current_timesteps = 0
    opponent_model = None
    use_heuristic = True

    while current_timesteps < total_timesteps:
        next_timesteps = min(current_timesteps + clone_interval, total_timesteps)

        # Switch to self-play after 1M
        if current_timesteps >= heuristic_cutoff and use_heuristic:
            use_heuristic = False
            opponent_model = PPO.load("tresette_agent_clone", device=device)

        model.learn(
            total_timesteps=clone_interval,
            reset_num_timesteps=False,
            callback=checkpoint_callback
        )
        current_timesteps = next_timesteps

        model.save("tresette_agent_clone")

        env = DummyVecEnv([make_env(opponent_model, use_heuristic, device)])
        model.set_env(env)
        env.close()

    model.save("tresette_agent_final")


if __name__ == "__main__":
    main()
