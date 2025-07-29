from env.card import Card

from typing import List

class Hand:
    def __init__(self):
        self.cards: List[Card] = []

    def receive_cards(self, cards: List[Card]):
        self.cards = cards

    def add_single_to_hand(self, card: Card):
        self.cards.append(card)

    def suit_counts(self):
        counts = {}
        for card in self.cards:
            counts[card.suit] = counts.get(card.suit, 0) + 1
        return counts