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
from env.baseline_policies import AdvancedHeuristicPolicy, SimpleHeuristicPolicy, HighestLeadSuitPolicy, RandomPolicy
from utils.wandb_logger import WandbLogger

def make_env(opponent_model=None, opponent_policy="heuristic", device='cpu'):
    """Create environment with specified opponent policy"""
    def _init():
        return TresetteGymWrapper(
            opponent_model=None if opponent_policy != "model" else opponent_model,
            opponent_policy=opponent_policy,
            device=device
        )
    return _init

def clone_policy(original_policy, cfg, device):
    """Clone the current policy for self-play"""
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

def create_envs(cfg, opponent_type, opponent_model, device):
    """Create train and test environments with specified opponent"""
    train_envs = DummyVectorEnv([
        make_env(opponent_model, opponent_type, device) 
        for _ in range(cfg.num_train_envs)
    ])
    
    test_envs = DummyVectorEnv([
        make_env(opponent_model, opponent_type, device) 
        for _ in range(cfg.num_test_envs)
    ])
    
    return train_envs, test_envs

def update_opponent(opponent_type, opponent_model, train_collector, test_collector, cfg, device):
    """Update the opponent in both train and test environments"""
    train_envs, test_envs = create_envs(cfg, opponent_type, opponent_model, device)
    
    train_collector.env = train_envs
    train_collector.reset()
    train_collector.reset_buffer()
    
    test_collector.env = test_envs
    test_collector.reset()
    
    return train_envs, test_envs

def get_heuristic_policy(stage):
    """Get heuristic policy based on curriculum stage"""
    policies = {
        0: "random",
        1: "highest_lead",
        2: "simple_heuristic",
        3: "advanced_heuristic"
    }
    return policies.get(min(stage, 3), "advanced_heuristic")

def main():
    cfg = TrainingConfig()
    device = cfg.device

    wandb.init(project="tressete-training", config=vars(cfg))
    wandb_logger = WandbLogger(config=vars(cfg))
    
    # Initial curriculum stage
    current_stage = 0
    opponent_type = get_heuristic_policy(current_stage)
    opponent_model = None

    # Create initial environments
    train_envs, test_envs = create_envs(cfg, opponent_type, opponent_model, device)

    # Initialize policy
    actor_net = TressetteActor(cfg.input_dim, cfg.hidden_dim, cfg.action_dim).to(device)
    critic_net = CriticMLP(cfg.input_dim, cfg.hidden_dim).to(device)
    actor = MaskableActor(actor_net).to(device)

    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic_net.parameters()), 
        lr=cfg.lr,
        weight_decay=cfg.weight_decay  # Add weight decay
    )
    
    # Phase 1: Linear Warmup (0-5,000 steps)
        # LR starts at 10% of initial value (0.1 * lr)
        # Gradually increases to full initial LR
        # Helps stabilize early training
    # Phase 2: Cosine Annealing (5,000+ steps)
        # LR decays following a cosine curve
        # Slowly reduces to near-zero by end of training
        # Formula: η_t = η_min + 0.5*(η_max - η_min)*(1 + cos(π*t/T))
        # Where T is total steps after warmup
        # Benefits:
    # Warmup prevents early instability
    # Cosine decay provides smooth reduction
    # Allows more exploration early, more fine-tuning later
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=5000),
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.max_total_steps-5000)
        ],
        milestones=[5000]
    )

    policy = CustomMaskablePPOPolicy(
        actor=actor,
        critic=critic_net,
        optim=optim,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor=cfg.discount_factor,
        max_grad_norm=cfg.max_grad_norm,
        action_space=train_envs.action_space[0]
    ).to(device)

    # Create collectors
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(cfg.buffer_size, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # Initial experience collection
    print(f"Initial collection of {cfg.initial_collect_step} steps with {opponent_type} opponent...")
    train_collector.collect(n_step=cfg.initial_collect_step)
    total_steps = cfg.initial_collect_step

    while total_steps < cfg.max_total_steps:
        # Check for curriculum stage update
        for threshold, stage in cfg.curriculum_stages.items():
            if total_steps >= threshold and stage > current_stage:
                current_stage = stage
                if stage == 4:  # Self-play
                    print("Switching to self-play: cloning current policy as opponent.")
                    opponent_type = "model"
                    opponent_model = clone_policy(policy, cfg, device)
                else:
                    new_policy = get_heuristic_policy(current_stage)
                    print(f"Curriculum update: Switching to {new_policy} at step {total_steps}")
                    opponent_type = new_policy
                    opponent_model = None
                
                # Update environments
                train_envs, test_envs = update_opponent(
                    opponent_type, opponent_model, 
                    train_collector, test_collector, 
                    cfg, device
                )
                break

        # Collect experience
        result = train_collector.collect(n_step=cfg.step_per_collect)
        total_steps += cfg.step_per_collect
        wandb_logger.log_train_data(result, total_steps)

        # Perform updates if buffer has enough data
        if len(train_collector.buffer) >= cfg.batch_size:
            for update_idx in range(cfg.repeat_per_collect):
                update_result = policy.update(
                    sample_size=cfg.batch_size,
                    buffer=train_collector.buffer,
                    batch_size=cfg.batch_size,
                    repeat=1
                )
                
                # Log update metrics
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

        # Update learning rate
        scheduler.step()
        current_lr = optim.param_groups[0]['lr']
        wandb.log({"lr": current_lr}, step=total_steps)
        
        # Log learning rate every 1000 steps
        if total_steps % 5000 == 0:
            print(f"Step {total_steps}: Learning rate = {current_lr:.6f}")

        # Test performance periodically
        if total_steps % cfg.test_interval == 0:
            test_result = test_collector.collect(n_episode=cfg.episode_per_test)
            rew_test_mean = np.mean(test_result["rew"]) if hasattr(test_result["rew"], "__len__") else float(test_result["rew"])
            wandb_logger.log_test_data({"rew": rew_test_mean}, total_steps)
            print(f"Step {total_steps}: Test reward mean: {rew_test_mean:.3f}")

        # Update self-play opponent periodically
        if current_stage == 4 and total_steps % cfg.clone_interval == 0:
            print(f"Step {total_steps}: Cloning policy as new opponent for self-play.")
            opponent_model = clone_policy(policy, cfg, device)
            torch.save(opponent_model.state_dict(), f"trained_models/clone_opponent_step_{total_steps}.pth")
            
            # Update environments with new opponent model
            train_envs, test_envs = update_opponent(
                opponent_type, opponent_model, 
                train_collector, test_collector, 
                cfg, device
            )

    # Save final model
    torch.save(policy.state_dict(), "trained_models/final_policy.pth")
    wandb.finish()

if __name__ == "__main__":
    main()