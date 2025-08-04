import torch
import numpy as np

from model.gym_wrapper import TresetteGymWrapper
from model.maskable_policy import MaskableActor, CustomMaskablePPOPolicy
from model.tressete_actor import TressetteActor
from model.model import CriticMLP
from model.training_config import TrainingConfig


def load_policy_checkpoint(path, cfg, device):
    actor_net = TressetteActor(cfg.input_dim, cfg.hidden_dim, cfg.action_dim).to(device)
    critic_net = CriticMLP(cfg.input_dim, cfg.hidden_dim).to(device)
    actor = MaskableActor(actor_net).to(device)

    policy = CustomMaskablePPOPolicy(
        actor=actor,
        critic=critic_net,
        optim=None,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor=cfg.discount_factor,
        max_grad_norm=cfg.max_grad_norm,
        action_space=None
    ).to(device)

    state_dict = torch.load(path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def play_single_game(agent_policy, opponent_policy=None, use_heuristic_opponent=True, device='cpu'):
    env = TresetteGymWrapper(
        opponent_model=opponent_policy if not use_heuristic_opponent else None,
        opponent_policy="advanced_heuristic" if use_heuristic_opponent else None,
        device=device
    )
    obs_dict, _ = env.reset()
    done = False

    while not done:
        if env.env.current_player == env.agent_index:
            obs = obs_dict["obs"]
            valid_actions = np.where(obs_dict["action_mask"] == 1)[0]

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                out = agent_policy.actor(obs_tensor)
                logits = out[0] if isinstance(out, tuple) else out
                logits_np = logits.cpu().numpy()[0]

                masked_logits = np.full_like(logits_np, -1e9)
                masked_logits[valid_actions] = logits_np[valid_actions]

                action = int(np.argmax(masked_logits))

            if action not in valid_actions:
                print(f"Invalid action chosen by agent: {action}, picking random valid.")
                action = np.random.choice(valid_actions)

            obs_dict, reward, done, truncated, info = env.step(action)
        else:
            obs_dict, reward, done, truncated, info = env.step(None)

    print("\n=== GAME OVER ===")
    for i, player in enumerate(env.env.players):
        print(f"Player {i} points: {player.num_pts}")

    winner = max(range(len(env.env.players)), key=lambda i: env.env.players[i].num_pts)

    # Determine winner name
    if winner == env.agent_index:
        winner_name = "AI"
    else:
        winner_name = "Heuristic" if use_heuristic_opponent else "Opponent AI"

    print(f"Winner is: {winner_name}")


def main():
    device = 'cpu'
    cfg = TrainingConfig()

    agent_policy_path = "trained_models/final_policy.pth"
    agent_policy = load_policy_checkpoint(agent_policy_path, cfg, device)

    dummy_env = TresetteGymWrapper()
    agent_policy.action_space = dummy_env.action_space

    use_heuristic_opponent = True
    opponent_policy = None
    if not use_heuristic_opponent:
        opponent_policy_path = "trained_models/final_policy.pth"
        opponent_policy = load_policy_checkpoint(opponent_policy_path, cfg, device)
        opponent_policy.action_space = dummy_env.action_space

    play_single_game(agent_policy, opponent_policy=opponent_policy,
                     use_heuristic_opponent=use_heuristic_opponent, device=device)


if __name__ == "__main__":
    main()
