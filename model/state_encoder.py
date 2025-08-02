import numpy as np
from env.card_config import SUITS, RANKS

class EncodedState:
    def __init__(self, num_players: int=2):
        self.agent_state = np.zeros(214, dtype=np.float32)
        self.opponent_state = np.zeros(214, dtype=np.float32)

    @staticmethod
    def card_to_index(card):
        return SUITS.index(card.suit) * 10 + RANKS.index(card.rank)

    # Converter used to convert a card represetation to one hot encoding vector index
    @staticmethod
    def encode_state(player_index: int, gym_env):
        # Initialize the state vector with zeros (214 dimensions)
        state = np.zeros(214, dtype=np.float32)

        tressete_env = gym_env.env
        trick = tressete_env.trick
        player = tressete_env.players[player_index]
        other_known_cards = tressete_env.players[(player_index + 1) % tressete_env.num_players].known_cards
        # This may be needed to change in the future if more players are added
        opponent_won_cards = tressete_env.players[(player_index + 1) % tressete_env.num_players].won_cards
        opponent_points = tressete_env.players[(player_index + 1) % tressete_env.num_players].num_pts
        cards_left_in_deck = len(tressete_env.deck.cards)

        # 1. Own hand (40-dim)
        if player.hand.cards != []:
            for card in player.hand.cards:
                idx = EncodedState.card_to_index(card)
                state[idx] = 1.0


        # 2. Trick cards (40-dim)
        for _, card in trick.played_cards:
            if card is not None:
                idx = EncodedState.card_to_index(card)
                state[40 + idx] = 1.0

        # 3. Known cards of others (40-dim)
        for card in other_known_cards:
            if card is not None:
                idx = EncodedState.card_to_index(card)
                state[80 + idx] = 1.0

        # 4. Opponent won cards (40-dim)
        for _, card in opponent_won_cards:
            if card is not None:
                idx = EncodedState.card_to_index(card)
                state[120 + idx] = 1.0

        # 5. Own won cards (40-dim)
        for _, card in player.won_cards:
            if card is not None:
                idx = EncodedState.card_to_index(card)
                state[160 + idx] = 1.0

        # 6. Points (2 floats)
        state[201] = player.num_pts
        state[202] = opponent_points  # custom, see wrapper

        # 7. Number of cards left in the deck (normalized float)
        state[203] = cards_left_in_deck / 40.0  # Normalize (max 40 cards in deck)

        # --- 8. Append valid action mask (10 binary flags) ---
        valid_actions = gym_env.env.get_valid_actions()  # Use env method for consistency
        valid_action_mask = np.zeros(10, dtype=np.float32)
        for i in valid_actions:
            if 0 <= i < 10:
                valid_action_mask[i] = 1.0
        state[204:] = valid_action_mask

        # Verify length before returning
        if len(state) != 214:
            print(f"Invalid state length: {len(state)}")
            return np.zeros(214, dtype=np.float32)
            
        return state

    def update_player_state(self, gym_env):
        encoded_agent = EncodedState.encode_state(gym_env.agent_index, gym_env)
        encoded_opponent = EncodedState.encode_state((gym_env.agent_index + 1) % gym_env.env.num_players, gym_env)

        self.agent_state = encoded_agent
        self.opponent_state = encoded_opponent

