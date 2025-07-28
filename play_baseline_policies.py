import sys
from env.tresette_engine import TresetteEngine
from env.baseline_policies import (
    RandomPolicy,
    HighestLeadSuitPolicy,
    SimpleHeuristicPolicy,
    SlightlySmarterHeuristicPolicy
)

policy_classes = {
    "random": RandomPolicy,
    "highest-lead": HighestLeadSuitPolicy,
    "heuristic": SlightlySmarterHeuristicPolicy
}


def select_policy(player_id):
    print(f"\nChoose policy for Player {player_id}:")
    for i, key in enumerate(policy_classes.keys(), 1):
        print(f"{i}. {key}")
    while True:
        try:
            choice = int(input("Enter choice number: "))
            if 1 <= choice <= len(policy_classes):
                selected_key = list(policy_classes.keys())[choice - 1]
                return policy_classes[selected_key]()
        except ValueError:
            pass
        print("Invalid choice. Please try again.")


def print_trick(trick):
    print("\nTrick Result:")
    for pid, card in trick:
        print(f"  Player {pid} played {card}")
    print()


def play_game():
    print("Starting Tressette Baseline Game")
    try:
        num_players = int(input("Enter number of players (2–4): "))
        if num_players < 2 or num_players > 4:
            raise ValueError
    except ValueError:
        print("Invalid number. Defaulting to 2 players.")
        num_players = 2

    # Select policy instances
    policies = [select_policy(pid) for pid in range(num_players)]

    env = TresetteEngine(num_players=num_players)
    obs = env.reset()

    while not env.done:
        current_player = env.current_player
        player = env.players[current_player]
        lead_suit = env.trick[0][1].suit if env.trick else None
        trick = env.trick

        policy = policies[current_player]
        action_idx = policy.get_action_index(player, lead_suit, trick)
        chosen_card = player.hand[action_idx]
        print(f"Player {current_player} plays: {chosen_card}")
        _, _,  = env.step(action_idx)

        if len(env.trick) == 0:
            print_trick(env.trick)

    print("\nGame Over")
    print("=" * 30)
    for player in env.players:
        print(f"Player {player.id} scored {player.num_pts:.2f} points")
    print("=" * 30)


if __name__ == "__main__":
    try:
        play_game()
    except KeyboardInterrupt:
        sys.exit("\nGame interrupted.")