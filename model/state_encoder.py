import numpy as np
from env.card_config import SUITS, RANKS

# Converter used to convert a card represetation to one hot encoding vector index
def card_to_index(card):
    return SUITS.index(card.suit) * 10 + RANKS.index(card.rank)

def encode_state(obs, player, num_players=2):
    # Initialize the state vector with zeros (204 dimensions)
    state = np.zeros(204, dtype=np.float32)

    # 1. Own hand (40-dim)
    if player.hand.cards != []:
        for card in player.hand.cards:
            idx = card_to_index(card)
            state[idx] = 1.0

    # 2. Trick cards (40-dim)
    for _, card in obs['trick']:
        if card is not None:
            idx = card_to_index(card)
            state[40 + idx] = 1.0

    # 3. Known cards of others (40-dim)
    for card in obs['other_players_known_cards']:
        if card is not None:
            idx = card_to_index(card)
            state[80 + idx] = 1.0

    # 4. Opponent won cards (40-dim)
    for _, card in obs['opponent_won_cards']:
        if card is not None:
            idx = card_to_index(card)
            state[120 + idx] = 1.0

    # 5. Own won cards (40-dim)
    for _, card in player.won_cards:
        if card is not None:
            idx = card_to_index(card)
            state[160 + idx] = 1.0

    # 6. Points (2 floats)
    state[201] = player.num_pts
    state[202] = obs["opponent_points"]  # custom, see wrapper

    # # 5. Final trick flag (1 float)
    # state[base + 2] = 1.0 if obs.get("final_trick", False) else 0.0

    # 7. Number of cards left in the deck (normalized float)
    state[203] = obs["cards_left_in_deck"] / 40.0  # Normalize (max 40 cards in deck)

    return state