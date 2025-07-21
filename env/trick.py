from env.card import Card

from typing import List

class Trick:
    def __init__(self, trick_size=2):
        self.size = trick_size
        self.reset()

    def reset(self):
        self.lead_suit = None
        self.played_cards: List[Card] = []
    
    def resolve_trick(self):
        lead_cards = [(player_id, card) for player_id, card in self.played_cards if card.suit == self.lead_suit]
        winner, _ = max(lead_cards, key=lambda x: x[1].value())
        points = self.calculate_trick_points()

        # Reset for the next trick
        self.reset()

        return winner, points
    
    def calculate_trick_points(self):
        return sum(card.point_value for _, card in self.played_cards)