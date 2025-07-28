from abc import ABC, abstractmethod
import random
from typing import List, Optional

from env.tresette_engine import Player, Card

class BaselinePolicy(ABC):
    @abstractmethod
    def get_action_index() -> int:
        pass

class RandomPolicy(BaselinePolicy):
    @staticmethod
    def get_action_index(player: Player, lead_suit: str, trick: Optional[List] = None) -> int:
        valid_moves = player.get_valid_moves(lead_suit)
        return random.choice(valid_moves)

class HighestLeadSuitPolicy(BaselinePolicy):
    @staticmethod
    def get_action_index(player: Player, lead_suit: str, trick: Optional[List] = None) -> int:
        valid_moves = player.get_valid_moves(lead_suit)
        if lead_suit is not None:
            lead_suit_cards = [(i, player.hand.cards[i]) for i in valid_moves if player.hand.cards[i].suit == lead_suit]
            if lead_suit_cards:
                i, _ = max(lead_suit_cards, key=lambda x: x[1].value())
                return i
        return random.choice(valid_moves)


class SimpleHeuristicPolicy(BaselinePolicy):
    @staticmethod
    def get_action_index(player: Player, lead_suit: str, trick: Optional[List] = None) -> int:
        valid_moves = player.get_valid_moves(lead_suit)
        if lead_suit is None:
            # Leading: play highest card
            highest_idx = max(valid_moves, key=lambda i: player.hand.cards[i].value())
            return highest_idx

        if trick:
            lead_cards = [(pid, card) for pid, card in trick if card.suit == lead_suit]
            highest_card_value = max(card.value() for _, card in lead_cards)
            beating_cards = [i for i in valid_moves if player.hand.cards[i].suit == lead_suit and player.hand.cards[i].value() > highest_card_value]
            if beating_cards:
                chosen_idx = min(beating_cards, key=lambda i: player.hand.cards[i].value())
                return chosen_idx

        lowest_idx = min(valid_moves, key=lambda i: player.hand.cards[i].value())
        return lowest_idx
