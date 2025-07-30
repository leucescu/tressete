import numpy as np
from stable_baselines3 import PPO

from env.tresette_engine import TresetteEngine
from env.baseline_policies import SlightlySmarterHeuristicPolicy, AdvancedHeuristicPolicy
from model.state_encoder import encode_state  # Your 204-dim vector function

# === Load trained model ===
model = PPO.load("tresette_agent_clone.zip")

# === Setup Game ===
engine = TresetteEngine()
obs = engine.reset()

ai_index = 0
heuristic_index = 1

def play_game():
    # === Game Loop ===
    engine.reset()
    while not engine.done:
        player = engine.players[engine.current_player]
        obs = engine._get_obs()
        valid_actions = engine.get_valid_actions()

        if engine.current_player == ai_index:
            encoded = encode_state(obs, player)
            action, _ = model.predict(encoded, deterministic=True)

            if action not in valid_actions:
                print(f"[AI WARNING] Chose invalid action {action}. Picking random valid.")
                action = np.random.choice(valid_actions)

        else:  # Heuristic opponent
            # lead_suit = engine.trick.lead_suit
            action = AdvancedHeuristicPolicy.get_action_index(engine)

        engine.step(action)

    # === Results ===
    print("\n🃏 GAME OVER 🃏")
    for i, p in enumerate(engine.players):
        print(f"Player {i} scored {p.num_pts} points")

    winner = max(range(len(engine.players)), key=lambda i: engine.players[i].num_pts)
    print(f"\n🏆 Winner: Player {winner} 🏆")

def main():
    print("Starting Tresette game with AI vs Heuristic...")
    play_game()

if __name__ == "__main__":
    main()