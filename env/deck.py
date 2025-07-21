from env.card_config import RANKS, SUITS
from env.card import Card

from typing import List
import random

class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in SUITS for rank in RANKS]

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self):
        if self.cards:
            return self.cards.pop(0)  # draw from top
        return None

    # This may need to be modified
    def deal(self, num_hands, cards_per_hand):

        hands: List[List[Card]] = []

        for _ in range(num_hands):

            hand: List[Card] = []

            for _ in range(cards_per_hand):
                hand.append(self.draw())
            hands.append(hand)
        return hands

    def cards_left(self):
        return len(self.cards)