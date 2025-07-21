from env.card import Card
from env.hand import Hand
from env.deck import Deck

from typing import List

class Player:
    def __init__(self, player_id):
        self.id = player_id
        self.reset()

    def reset(self):
        self.hand = Hand()

        self.won_cards: List[Card] = []
        self.known_cards: List[Card] = []

        self.num_pts: float = 0.0
        self.last_trick_pts: float = 0.0


    def draw_card(self, deck: Deck):
        card = deck.draw()
        if card:
            self.known_cards.append(card)
            self.hand.add_single_to_hand(card)
        return card
    
    def play_card(self, index):
        card = self.hand.cards.pop(index)

        if card in self.known_cards:
            # The first match is removed
            self.known_cards.remove(card)

        return card

    def get_valid_moves(self, lead_suit=None):
        if lead_suit is None:
            return list(range(len(self.hand)))

        # Must follow suit if possible
        valid_indices = [i for i, card in enumerate(self.hand.cards) if card.suit == lead_suit]
        if valid_indices:
            return valid_indices
        return list(range(len(self.hand)))

    def collect_trick(self, trick_cards: List[Card], trick_pts: float):
        self.won_cards.extend(trick_cards)
        self.num_pts += trick_pts
        self.last_trick_pts = trick_pts 

    def update_points_at_the_end(self, bonus_points: float):
        self.num_pts += bonus_points