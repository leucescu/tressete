import numpy as np
from env.tresette_engine import SUITS, RANKS

# Converter used to convert a card represetation to one hot encoding vector index
def card_to_index(card):
    return SUITS.index(card.suit) * 10 + RANKS.index(card.rank)

def encode_state(obs, player, num_players=2):
    # Initialize the state vector with zeros (162 dimensions)
    state = np.zeros(162, dtype=np.float32)

    # 1. Own hand (40-dim)
    for card in obs['hand']:
        idx = card_to_index(card)
        state[idx] = 1.0

    # 2. Played cards (40-dim)
    for card in obs['trick']:
        idx = card_to_index(card)
        state[40 + idx] = 1.0

    # 3. Known cards of others (40-dim)
    for card in obs['other_players_known_cards']:
        idx = card_to_index(card)
        state[80 + idx] = 1.0

    # 4. Points (2 floats)
    base = 120
    state[base] = player.num_pts
    state[base + 1] = obs.get("opponent_points", 0.0)  # custom, see wrapper

    # 5. Final trick flag (1 float)
    state[base + 2] = 1.0 if obs.get("final_trick", False) else 0.0

# 6. Number of cards left in the deck (normalized float)
    state[161] = obs.get("cards_left_in_deck", 0) / 40.0  # Normalize (max 40 cards in deck)

    return state