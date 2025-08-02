import torch
import numpy as np
import wandb
from copy import deepcopy

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv

from model.model import CriticMLP
from model.training_config import TrainingConfig
from model.gym_wrapper import TresetteGymWrapper
from model.maskable_policy import MaskableActor, MaskablePPOPolicy
from model.tressete_actor import TressetteActor

from env.baseline_policies import AdvancedHeuristicPolicy


def make_env(opponent_model=None, use_heuristic=False, device='cpu'):
    def _init():
        return TresetteGymWrapper(
            opponent_model=None if use_heuristic else opponent_model,
            opponent_policy="heuristic" if use_heuristic else None,
            device=device
        )
    return _init


class WandbLogger:
    def __init__(self, project_name="tressete-training", config=None):
        self.project_name = project_name
        self.config = config

    def log_train_data(self, data: dict, step: int):
        log_data = {f"train/{k}": float(np.mean(v)) if isinstance(v, (list, np.ndarray)) else v
                    for k, v in data.items()}
        wandb.log(log_data, step=step)

    def log_test_data(self, data: dict, step: int):
        log_data = {f"test/{k}": float(np.mean(v)) if isinstance(v, (list, np.ndarray)) else v
                    for k, v in data.items()}
        wandb.log(log_data, step=step)

    def log_update_data(self, data: dict, step: int):
        log_data = {f"update/{k}": float(np.mean(v)) if isinstance(v, (list, np.ndarray)) else v
                    for k, v in data.items()}
        wandb.log(log_data, step=step)

    def save_data(self, data, name, step=None, env_step=None, gradient_step=None):
        # Optional: save checkpoints or models here if needed
        pass


def main():
    cfg = TrainingConfig()
    device = cfg.device

    wandb.init(project="tressete-training", config=vars(cfg))
    wandb_logger = WandbLogger(config=vars(cfg))

    # Initial opponent is heuristic
    opponent_policy = AdvancedHeuristicPolicy()
    use_heuristic = True

    train_envs = DummyVectorEnv([make_env(opponent_policy, use_heuristic=use_heuristic, device=device)
                                 for _ in range(cfg.num_train_envs)])
    test_envs = DummyVectorEnv([make_env(opponent_policy, use_heuristic=use_heuristic, device=device)
                                for _ in range(cfg.num_test_envs)])

    actor_net = TressetteActor(cfg.input_dim, cfg.hidden_dim, cfg.action_dim).to(device)
    critic_net = CriticMLP(cfg.input_dim, cfg.hidden_dim).to(device)
    actor = MaskableActor(actor_net).to(device)

    optim = torch.optim.Adam(list(actor.parameters()) + list(critic_net.parameters()), lr=cfg.lr)
    # Add LR scheduler: decay LR by 0.8 every 50,000 steps
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50_000, gamma=0.8)

    policy = MaskablePPOPolicy(
        actor=actor,
        critic=critic_net,
        optim=optim,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor=cfg.discount_factor,
        max_grad_norm=cfg.max_grad_norm,
        action_space=train_envs.action_space[0]
    ).to(device)

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(cfg.buffer_size, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    print(f"Initial collection of {cfg.initial_collect_step} steps with heuristic opponent...")
    train_collector.collect(n_step=cfg.initial_collect_step)

    total_steps = cfg.initial_collect_step
    use_heuristic = True
    current_opponent = opponent_policy

    while total_steps < cfg.max_total_steps:
        result = train_collector.collect(n_step=cfg.step_per_collect)
        total_steps += cfg.step_per_collect

        # rew_mean = np.mean(result["rew"]) if hasattr(result["rew"], "__len__") else float(result["rew"])
        # print(f"Step {total_steps}: Train reward mean: {rew_mean:.3f}")

        wandb_logger.log_train_data(result, total_steps)

        # Update policy only if enough samples in buffer
        if len(train_collector.buffer) >= cfg.batch_size:
            for _ in range(cfg.repeat_per_collect):
                update_result = policy.update(
                    sample_size=cfg.batch_size * cfg.repeat_per_collect,
                    buffer=train_collector.buffer,
                    batch_size=cfg.batch_size,
                    repeat=1
                )
                wandb_logger.log_update_data(update_result, total_steps)
        else:
            print(f"Buffer size ({len(train_collector.buffer)}) less than batch size ({cfg.batch_size}), skipping update")

        # Step the LR scheduler after updates
        scheduler.step()

        # Log current learning rate
        current_lr = optim.param_groups[0]['lr']
        wandb.log({"lr": current_lr}, step=total_steps)

        test_result = test_collector.collect(n_episode=cfg.episode_per_test)
        rew_test_mean = np.mean(test_result["rew"]) if hasattr(test_result["rew"], "__len__") else float(test_result["rew"])
        wandb_logger.log_test_data({"rew": rew_test_mean}, total_steps)
        print(f"Step {total_steps}: Test reward mean: {rew_test_mean:.3f}")

        # Switch from heuristic to self-play after cutoff_steps
        if use_heuristic and total_steps >= cfg.cutoff_steps:
            print("Switching to self-play: cloning current policy as opponent.")
            use_heuristic = False
            current_opponent = deepcopy(policy)

            torch.save(current_opponent.state_dict(), f"clone_opponent_step_{total_steps}.pth")

            train_envs = DummyVectorEnv([make_env(current_opponent, use_heuristic=use_heuristic, device=device)
                                        for _ in range(cfg.num_train_envs)])
            test_envs = DummyVectorEnv([make_env(current_opponent, use_heuristic=use_heuristic, device=device)
                                       for _ in range(cfg.num_test_envs)])

            train_collector.env = train_envs
            train_collector.reset()

            test_collector.env = test_envs
            test_collector.reset()

        # Periodically update the opponent by cloning current policy
        if not use_heuristic and total_steps % cfg.clone_interval == 0:
            print(f"Step {total_steps}: Cloning policy as new opponent for self-play.")
            current_opponent = deepcopy(policy)

            torch.save(current_opponent.state_dict(), f"clone_opponent_step_{total_steps}.pth")

            train_envs = DummyVectorEnv([make_env(current_opponent, use_heuristic=use_heuristic, device=device)
                                        for _ in range(cfg.num_train_envs)])
            test_envs = DummyVectorEnv([make_env(current_opponent, use_heuristic=use_heuristic, device=device)
                                       for _ in range(cfg.num_test_envs)])

            train_collector.env = train_envs
            train_collector.reset()

            test_collector.env = test_envs
            test_collector.reset()

    wandb.finish()


if __name__ == "__main__":
    main()
