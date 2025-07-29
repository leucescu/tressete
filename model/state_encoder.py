import numpy as np
from env.card_config import SUITS, RANKS
from env.player import Player

# Converter used to convert a card represetation to one hot encoding vector index
def card_to_index(card):
    return SUITS.index(card.suit) * 10 + RANKS.index(card.rank)

def encode_state(obs, player: Player, num_players: int=2):
    # Initialize the state vector with zeros (214 dimensions)
    state = np.zeros(214, dtype=np.float32)

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

    # 7. Number of cards left in the deck (normalized float)
    state[203] = obs["cards_left_in_deck"] / 40.0  # Normalize (max 40 cards in deck)

    # --- 8. Append valid action mask (10 binary flags) ---
    lead_card = obs['trick'][0][1] if obs['trick'] else None
    lead_suit = lead_card.suit if lead_card else None
    valid_actions = player.get_valid_moves(lead_suit)

    # Create a binary mask: 1 if action is valid, 0 otherwise
    valid_action_mask = np.zeros(10, dtype=np.float32)
    valid_action_mask[valid_actions] = 1.0
    state[204:] = valid_action_mask

    return state