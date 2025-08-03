import torch
import numpy as np
import wandb
import os

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv

from model.model import CriticMLP
from model.training_config import TrainingConfig
from model.gym_wrapper import TresetteGymWrapper
from model.maskable_policy import MaskableActor, CustomMaskablePPOPolicy
from model.tressete_actor import TressetteActor
from env.baseline_policies import AdvancedHeuristicPolicy
from utils.wandb_logger import WandbLogger

def make_env(opponent_model=None, use_heuristic=False, device='cpu'):
    def _init():
        return TresetteGymWrapper(
            opponent_model=None if use_heuristic else opponent_model,
            opponent_policy="heuristic" if use_heuristic else None,
            device=device
        )
    return _init

def clone_policy(original_policy, cfg, device):
    actor_net = TressetteActor(cfg.input_dim, cfg.hidden_dim, cfg.action_dim).to(device)
    critic_net = CriticMLP(cfg.input_dim, cfg.hidden_dim).to(device)
    actor = MaskableActor(actor_net).to(device)

    actor.load_state_dict(original_policy.actor.state_dict())
    critic_net.load_state_dict(original_policy.critic.state_dict())

    cloned_policy = CustomMaskablePPOPolicy(
        actor=actor,
        critic=critic_net,
        optim=None,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor=cfg.discount_factor,
        max_grad_norm=cfg.max_grad_norm,
        action_space=original_policy.action_space
    ).to(device)

    cloned_policy.eval()
    return cloned_policy

def main():
    cfg = TrainingConfig()
    device = cfg.device

    wandb.init(project="tressete-training", config=vars(cfg))
    wandb_logger = WandbLogger(config=vars(cfg))

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
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50_000, gamma=0.8)

    policy = CustomMaskablePPOPolicy(
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

    while total_steps < cfg.max_total_steps:
        # Collect experience
        result = train_collector.collect(n_step=cfg.step_per_collect)
        total_steps += cfg.step_per_collect
        wandb_logger.log_train_data(result, total_steps)

        # Only perform updates if buffer has enough data
        if len(train_collector.buffer) >= cfg.batch_size:
            for update_idx in range(cfg.repeat_per_collect):
                update_result = policy.update(
                    sample_size=cfg.batch_size,
                    buffer=train_collector.buffer,
                    batch_size=cfg.batch_size,
                    repeat=1
                )
                
                # Convert metrics to Python floats (handles NaN properly)
                metrics = {
                    'kl': update_result.get('kl', float('nan')),
                    'ev_old': update_result.get('ev_old', float('nan')),
                    'ev_new': update_result.get('ev_new', float('nan'))
                }
                
                wandb_logger.log_update_data(metrics, total_steps)
                
                # Print metrics every 5,000 steps
                if total_steps % 5000 == 0 and update_idx == 0:
                    print(f"\n--- Step {total_steps} Update Metrics ---")
                    print(f"KL Divergence:      {metrics['kl']:.4f} (Target: 0.01-0.05)")
                    print(f"EV (Before Update): {metrics['ev_old']:.4f} (1.0 = perfect prediction)")
                    print(f"EV (After Update):  {metrics['ev_new']:.4f} (1.0 = perfect prediction)")
                    print(f"Value Improvement:  {metrics['ev_new'] - metrics['ev_old']:+.4f}")

        # These should be OUTSIDE the update if-block but INSIDE the while-loop
        scheduler.step()
        wandb.log({"lr": optim.param_groups[0]['lr']}, step=total_steps)

        # Test performance periodically
        if total_steps % cfg.test_interval == 0:
            test_result = test_collector.collect(n_episode=cfg.episode_per_test)
            rew_test_mean = np.mean(test_result["rew"]) if hasattr(test_result["rew"], "__len__") else float(test_result["rew"])
            wandb_logger.log_test_data({"rew": rew_test_mean}, total_steps)
            print(f"Step {total_steps}: Test reward mean: {rew_test_mean:.3f}")

        # Self-play opponent switching
        if use_heuristic and total_steps >= cfg.cutoff_steps:
            print("Switching to self-play: cloning current policy as opponent.")
            use_heuristic = False
            current_opponent = clone_policy(policy, cfg, device)
            torch.save(current_opponent.state_dict(), f"trained_models/clone_opponent_step_{total_steps}.pth")

            train_envs = DummyVectorEnv([make_env(current_opponent, use_heuristic=use_heuristic, device=device)
                                         for _ in range(cfg.num_train_envs)])
            test_envs = DummyVectorEnv([make_env(current_opponent, use_heuristic=use_heuristic, device=device)
                                        for _ in range(cfg.num_test_envs)])
            train_collector.env = train_envs
            test_collector.env = test_envs

        elif not use_heuristic and total_steps % cfg.clone_interval == 0:
            print(f"Step {total_steps}: Cloning policy as new opponent for self-play.")
            current_opponent = clone_policy(policy, cfg, device)
            torch.save(current_opponent.state_dict(), f"trained_models/clone_opponent_step_{total_steps}.pth")

            train_envs = DummyVectorEnv([make_env(current_opponent, use_heuristic=False, device=device)
                                         for _ in range(cfg.num_train_envs)])
            test_envs = DummyVectorEnv([make_env(current_opponent, use_heuristic=False, device=device)
                                        for _ in range(cfg.num_test_envs)])
            train_collector.env = train_envs
            test_collector.env = test_envs

    wandb.finish()

if __name__ == "__main__":
    main()