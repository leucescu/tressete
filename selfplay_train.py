import torch
import numpy as np
import wandb
from collections import deque
import os

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv

from model.model import CriticMLP
from model.training_config import TrainingConfig
from model.gym_wrapper import TresetteGymWrapper
from model.maskable_policy import MaskableActor, CustomMaskablePPOPolicy
from model.tressete_actor import TressetteActor
from utils.wandb_logger import WandbLogger

def make_env(opponent_model=None, opponent_policy="heuristic", difficulty=1.0, device='cpu'):
    """Create environment with specified opponent policy and difficulty"""
    def _init():
        return TresetteGymWrapper(
            opponent_model=None if opponent_policy != "model" else opponent_model,
            opponent_policy=opponent_policy,
            # difficulty=difficulty,
            device=device
        )
    return _init

def clone_policy(original_policy, cfg, device):
    """Clone the current policy for self-play with direct state transfer"""
    # Create new policy with same architecture
    actor_net = TressetteActor(cfg.input_dim, cfg.hidden_dim, cfg.action_dim).to(device)
    critic_net = CriticMLP(cfg.input_dim, cfg.hidden_dim).to(device)
    actor = MaskableActor(actor_net).to(device)

    
    # Direct state transfer
    cloned_policy = CustomMaskablePPOPolicy(
        actor=actor,
        critic=critic_net,
        optim=None,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor=cfg.discount_factor,
        max_grad_norm=cfg.max_grad_norm,
        action_space=original_policy.action_space
    ).to(device)
    
    # Load weights directly from original policy
    cloned_policy.load_state_dict(original_policy.state_dict())
    cloned_policy.eval()
    return cloned_policy

def create_envs(cfg, opponent_type, opponent_model, difficulty, device):
    """Create train and test environments with specified opponent and difficulty"""
    train_envs = DummyVectorEnv([
        make_env(opponent_model, opponent_type, difficulty, device) 
        for _ in range(cfg.num_train_envs)
    ])
    
    test_envs = DummyVectorEnv([
        make_env(opponent_model, opponent_type, difficulty, device) 
        for _ in range(cfg.num_test_envs)
    ])
    
    return train_envs, test_envs

def update_environments(cfg, opponent_type, opponent_model, difficulty, train_collector, test_collector, device):
    """Update environments with new opponent and difficulty"""
    # Clean up old environments
    if hasattr(train_collector.env, 'close'):
        train_collector.env.close()
    if hasattr(test_collector.env, 'close'):
        test_collector.env.close()
    
    # Create new environments
    train_envs, test_envs = create_envs(cfg, opponent_type, opponent_model, difficulty, device)
    
    # Update collectors
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

def evaluate_win_rate(collector, num_episodes=96):
    results = collector.collect(n_episode=num_episodes)
    rews = results.get("rews", [])
    lens = results.get("lens", [])

    if len(rews) == 0 or len(lens) == 0:
        print("Warning: No data collected!")
        return 0.0

    wins = 0
    total_episodes = len(rews)

    for episode_rewards, episode_dones in zip(rews, lens):
        if episode_dones and episode_rewards > 0:
            wins += 1

    return wins / total_episodes

def should_advance_stage(current_stage, total_steps, win_rate, cfg, stage_entered_steps):
    """Check if we should advance to next curriculum stage with stabilization"""
    # Check if we've met minimum stage duration
    stage_duration = total_steps - stage_entered_steps.get(current_stage, total_steps)
    if stage_duration < cfg.min_stage_duration:
        return False
    
    # Check win rate thresholds
    if current_stage == 0 and win_rate >= cfg.random_win_rate:
        return True
    elif current_stage == 1 and win_rate >= cfg.highest_lead_win_rate:
        return True
    elif current_stage == 2 and win_rate >= cfg.simple_heuristic_win_rate:
        return True
    elif current_stage == 3 and win_rate >= cfg.advanced_heuristic_win_rate:
        return True
    
    return False


def main():
    cfg = TrainingConfig()
    device = cfg.device
    
    # Create output directory
    os.makedirs("trained_models", exist_ok=True)
    
    wandb.init(project="tressete-training", config=vars(cfg))
    wandb_logger = WandbLogger(config=vars(cfg))
    
    # Curriculum tracking
    current_stage = 0
    opponent_type = get_heuristic_policy(current_stage)
    opponent_model = None
    difficulty = 0.6
    win_rate_history = deque(maxlen=10)
    stage_entered_steps = {}  # Track when each stage was entered
    stage_entered_steps[current_stage] = 0  # Initialize

    # Create initial environments
    train_envs, test_envs = create_envs(cfg, opponent_type, opponent_model, difficulty, device)

    # Initialize policy
    actor_net = TressetteActor(cfg.input_dim, cfg.hidden_dim, cfg.action_dim).to(device)
    critic_net = CriticMLP(cfg.input_dim, cfg.hidden_dim).to(device)
    actor = MaskableActor(actor_net).to(device)

    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic_net.parameters()), 
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    # Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optim,
    #     milestones=cfg.lr_schedule['milestones'],
    #     gamma=cfg.lr_schedule['gamma']
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.max_total_steps)


    policy = CustomMaskablePPOPolicy(
        actor=actor,
        critic=critic_net,
        optim=optim,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor=cfg.discount_factor,
        max_grad_norm=cfg.max_grad_norm,
        action_space=train_envs.action_space[0],
        vf_coef=cfg.vf_coef,
        eps_clip=cfg.eps_clip,
    ).to(device)

    # Create collectors
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(cfg.buffer_size, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # Initial experience collection
    print(f"Initial collection of {cfg.initial_collect_step} steps with {opponent_type} opponent (difficulty={difficulty:.1f})...")
    train_collector.collect(n_step=cfg.initial_collect_step)
    total_steps = cfg.initial_collect_step
    stage_entered_steps = {0: total_steps}  # Track stage start time

    while total_steps < cfg.max_total_steps:
        # Collect experience
        result = train_collector.collect(n_step=cfg.step_per_collect)
        n_steps = result.get("n/st", cfg.step_per_collect)
        total_steps += n_steps
        wandb_logger.log_train_data(result, total_steps)

        # Perform updates if buffer has enough data
        buffer_len = len(train_collector.buffer)
        if buffer_len >= cfg.min_update_samples:
            for update_idx in range(cfg.repeat_per_collect):
                # Get actual batch size (might be less than full batch)
                actual_batch_size = min(cfg.batch_size, buffer_len)
                
                update_result = policy.update(
                    value_clip=True,  # Add value clipping
                    vf_clip_param=0.5,  # Clip value loss
                    sample_size=actual_batch_size,
                    buffer=train_collector.buffer,
                    batch_size=cfg.mini_batch_size,
                    repeat=1
                )
                
                # Check for NaN values
                nan_detected = False
                for key, value in update_result.items():
                    if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                        print(f"NaN detected in {key} at step {total_steps}!")
                        nan_detected = True
                
                # Log update metrics
                if not nan_detected:
                    loss = update_result.get('loss', [float('nan')])
                    if isinstance(loss, list):
                        loss = np.mean(loss)

                    metrics = {
                        'kl': update_result.get('kl', float('nan')),
                        'ev_old': update_result.get('ev_old', float('nan')),
                        'ev_new': update_result.get('ev_new', float('nan')),
                        'loss': loss,
                        'loss_v': update_result.get('loss/v', float('nan'))
                    }
                    wandb_logger.log_update_data(metrics, total_steps)
                    
                    # Print metrics periodically
                    if total_steps % 5000 == 0 and update_idx == 0:
                        print(f"\n--- Step {total_steps} Update Metrics ---")
                        print(f"KL Divergence:      {metrics['kl']:.6f}")
                        print(f"EV (Before Update): {metrics['ev_old']:.4f}")
                        print(f"EV (After Update):  {metrics['ev_new']:.4f}")
                        print(f"Value Loss:         {metrics['loss_v']:.6f}")
                        print(f"Total Loss:         {metrics['loss']:.6f}")

        # Update learning rate
        scheduler.step()
        current_lr = optim.param_groups[0]['lr']
        wandb.log({"lr": current_lr}, step=total_steps)
        
        # Test performance periodically
        if total_steps % cfg.test_interval == 0:
            test_result = test_collector.collect(n_episode=cfg.episode_per_test)
            rewards = test_result.get("rews", [])
            
            if len(rewards) > 0:
                rew_test_mean = np.mean(rewards)
                rew_test_std = np.std(rewards)
            else:
                rew_test_mean = float('nan')
                rew_test_std = float('nan')
                print("Warning: No rewards collected during test!")
                
            wandb_logger.log_test_data({"rew": rew_test_mean}, total_steps)
            print(f"Step {total_steps}: Test reward - mean: {rew_test_mean:.3f}, std: {rew_test_std:.3f}")

        # Curriculum progression
        if total_steps % 3000 == 0:
            win_rate = evaluate_win_rate(test_collector, cfg.eval_episodes)
            win_rate_history.append(win_rate)
            avg_win_rate = np.mean(win_rate_history) if win_rate_history else 0.0
            wandb.log({
                "win_rate": win_rate,
                "avg_win_rate": avg_win_rate,
                "stage": current_stage,
                "difficulty": difficulty
            }, step=total_steps)
            
            print(f"Step {total_steps}: Win rate = {win_rate:.2f}, "
                  f"Avg win rate = {avg_win_rate:.2f}, "
                  f"Stage = {current_stage}, "
                  f"Difficulty = {difficulty:.1f}")
            
            # Check if we should advance stage
            if should_advance_stage(current_stage, total_steps, avg_win_rate, cfg, stage_entered_steps):
                current_stage += 1
                
                if current_stage == 4:  # Self-play
                    print(f"\n===== ADVANCING TO SELF-PLAY AT STEP {total_steps} =====")
                    opponent_type = "model"
                    opponent_model = clone_policy(policy, cfg, device)
                    difficulty = 1.0
                    torch.save(opponent_model.state_dict(), f"trained_models/initial_self_play_clone_{total_steps}.pth")
                else:
                    opponent_type = get_heuristic_policy(current_stage)
                    opponent_model = None
                    difficulty = min(1.0, difficulty + 0.1)
                    print(f"\n===== ADVANCING TO STAGE {current_stage}: {opponent_type.upper()} (difficulty={difficulty:.1f}) =====")
                
                # Track stage entry
                stage_entered_steps[current_stage] = total_steps
                
                # Update environments
                train_envs, test_envs = update_environments(
                    cfg, opponent_type, opponent_model, difficulty, 
                    train_collector, test_collector, device
                )
                
                # Reset win rate history for new stage
                win_rate_history.clear()

        # Update self-play opponent periodically
        if current_stage == 4 and total_steps % cfg.clone_interval == 0:
            print(f"Step {total_steps}: Cloning policy as new opponent for self-play.")
            opponent_model = clone_policy(policy, cfg, device)
            torch.save(opponent_model.state_dict(), f"trained_models/clone_opponent_step_{total_steps}.pth")
            
            # Update environments with new opponent model
            train_envs, test_envs = update_environments(
                cfg, "model", opponent_model, difficulty, 
                train_collector, test_collector, device
            )

        # Save checkpoint
        if total_steps % cfg.save_interval == 0:
            model_path = f"trained_models/checkpoint_step_{total_steps}.pth"
            torch.save(policy.state_dict(), model_path)
            print(f"Saved checkpoint at step {total_steps} to {model_path}")

    # Save final model
    final_path = "trained_models/final_policy.pth"
    torch.save(policy.state_dict(), final_path)
    print(f"Training complete! Final model saved to {final_path}")
    artifact = wandb.Artifact("final_model", type="model")
    artifact.add_file(final_path)
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()