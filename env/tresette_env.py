import math
import random
from typing import List

SUITS = ['bastoni', 'coppe', 'denari', 'spade']
RANKS = ['1', '2', '3', '4', '5', '6', '7', '11', '12', '13']

RANK_ORDER = {
    '3': 10,
    '2': 9,
    '1': 8,
    '13': 7,
    '12': 6,
    '11': 5,
    '7': 4,
    '6': 3,
    '5': 2,
    '4': 1
}

POINT_PER_RANK = {
    '3': 1/3,
    '2': 1/3,
    '1': 1,
    '13': 1/3,
    '12': 1/3,
    '11': 1/3,
    '7': 0,
    '6': 0,
    '5': 0,
    '4': 0
}

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
        self.point_value = POINT_PER_RANK[self.rank]

    def __eq__(self, other):
        return isinstance(other, Card) and self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return hash((self.suit, self.rank))

    def __repr__(self):
        return f"{self.rank} of {self.suit}"

    def value(self):
        return RANK_ORDER[self.rank]

class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in SUITS for rank in RANKS]

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self):
        if self.cards:
            return self.cards.pop(0)  # draw from top
        return None

    def deal(self, num_hands, cards_per_hand):
        hands = []
        for i in range(num_hands):
            hand = []
            for _ in range(cards_per_hand):
                hand.append(self.draw())
            hands.append(hand)
        return hands

    def cards_left(self):
        return len(self.cards)

class Player:
    def __init__(self, player_id):
        self.id = player_id
        self.hand = []
        self.won_cards = []
        self.known_cards = []
        self.num_pts = 0.0 
        self.last_trick_pts = 0.0

    def receive_cards(self, cards):
        self.hand = cards

    def play_card(self, index):
        card = self.hand.pop(index)

        if card in self.known_cards:
            # The first match is removed
            self.known_cards.remove(card)

        return card

    def draw_card(self, deck):
        card = deck.draw()
        if card:
            self.known_cards.append(card)
            self.hand.append(card)
        return card
    
    def collect_trick(self, trick_cards: List[Card]):
        self.won_cards.extend(trick_cards)
        self.update_points_value(trick_cards)

    def get_valid_moves(self, lead_suit=None):
        if lead_suit is None:
            return list(range(len(self.hand)))

        # Must follow suit if possible
        valid_indices = [i for i, card in enumerate(self.hand) if card.suit == lead_suit]
        if valid_indices:
            return valid_indices
        return list(range(len(self.hand)))
    
    def update_points_value(self, trick_cards: List[Card]):
        self.last_trick_pts = 0.0
        for trick_card in trick_cards:
            pts = trick_card.point_value
            self.num_pts += pts
            self.last_trick_pts += pts 

    def update_points_at_the_end(self, bonus_points):
        self.num_pts += bonus_points

class TresetteEnv:
    def __init__(self, num_players=2, initial_hand_size=10):
        self.num_players = num_players
        self.initial_hand_size = initial_hand_size
        self.deck = Deck()
        self.players = [Player(i) for i in range(num_players)]
        self.reset()

    def reset(self):
        self.deck = Deck()
        self.deck.shuffle()
        hands = self.deck.deal(self.num_players, self.initial_hand_size)
        for i, player in enumerate(self.players):
            player.receive_cards(hands[i])
        self.current_player = 0
        self.trick = []  # list of tuples (player_id, card)
        self.done = False
        self.final_trick_winner = None
        return self._get_obs()

    def _get_obs(self):
        current_hand = self.players[self.current_player].hand
        trick_cards = [card for _, card in self.trick]

        other_known_cards = []
        for player in self.players:
            other_known_cards.extend(player.known_cards)

        return {
            "hand": current_hand,
            "trick": trick_cards,
            "current_player": self.current_player,
            "cards_left_in_deck": len(self.deck.cards),
            "other_players_known_cards": other_known_cards
        }

    # Examine this again if it is right
    def step(self, action_idx):
        if self.done:
            raise Exception("Game over. Call reset().")

        player = self.players[self.current_player]
        valid_actions = player.get_valid_moves(self.trick[0][1].suit if self.trick else None)
        if action_idx not in valid_actions:
            raise ValueError(f"Invalid action {action_idx}. Valid actions: {valid_actions}")

        played_card = player.play_card(action_idx)
        self.trick.append((self.current_player, played_card))

        # Next player
        self.current_player = (self.current_player + 1) % self.num_players

        # Not utilized yet
        # info = {}

        # Trick complete
        if len(self.trick) == self.num_players:
            winner = self._resolve_trick()
            trick_cards = [card for _, card in self.trick]
            self.players[winner].collect_trick(trick_cards)  # Add won cards to winner
            self.trick = []
            self.current_player = winner

            # Draw phase: winner draws first
            for i in range(self.num_players):
                draw_player_idx = (winner + i) % self.num_players
                self.players[draw_player_idx].draw_card(self.deck)

                if all(len(p.hand) == 0 for p in self.players):
                    self.final_trick_winner = winner
    
        # Check if game done (no cards in hands and deck empty)
        hands_empty = all(len(p.hand) == 0 for p in self.players)
        deck_empty = len(self.deck.cards) == 0
        self.done = hands_empty and deck_empty

        if self.done:
            bonus_points = 11.0
            for player in self.players:
                bonus_points -= player.num_pts
            self.players[winner].update_points_at_the_end(bonus_points)

        return self._get_obs(), self.done

    def _resolve_trick(self):
        lead_suit = self.trick[0][1].suit
        lead_cards = [(pid, card) for pid, card in self.trick if card.suit == lead_suit]
        winner, _ = max(lead_cards, key=lambda x: x[1].value())
        return winner

    def get_valid_actions(self):
        player = self.players[self.current_player]
        lead_suit = self.trick[0][1].suit if self.trick else None
        return player.get_valid_moves(lead_suit)
    
    def get_reward_value(self):
        pass
