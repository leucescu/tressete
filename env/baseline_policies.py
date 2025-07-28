from abc import ABC, abstractmethod
import random
from typing import List, Optional

from env.tresette_engine import Player, Card, Trick

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

# This may have to be examined
class SimpleHeuristicPolicy(BaselinePolicy):
    @staticmethod
    def get_action_index(player: Player, lead_suit: str, trick: Optional[Trick] = None) -> int:
        valid_moves = player.get_valid_moves(lead_suit)
        if lead_suit is None:
            # Leading: play highest card
            highest_idx = max(valid_moves, key=lambda i: player.hand.cards[i].value())
            return highest_idx

        if trick:
            lead_cards = [(pid, card) for pid, card in trick.played_cards if card.suit == lead_suit]
            highest_card_value = max(card.value() for _, card in lead_cards)
            beating_cards = [i for i in valid_moves if player.hand.cards[i].suit == lead_suit and player.hand.cards[i].value() > highest_card_value]
            if beating_cards:
                chosen_idx = min(beating_cards, key=lambda i: player.hand.cards[i].value())
                return chosen_idx

        lowest_idx = min(valid_moves, key=lambda i: player.hand.cards[i].value())
        return lowest_idx

# Add tests for this one and examine it
class SlightlySmarterHeuristicPolicy(BaselinePolicy):
    @staticmethod
    def get_action_index(player: Player, lead_suit: str, trick: Optional[Trick] = None) -> int:
        valid_moves = player.get_valid_moves(lead_suit)

        if lead_suit is None:
            # Play the highest card from the suit with most cards
            suit_counts = player.hand.suit_counts()
            most_common_suit = max(suit_counts.items(), key=lambda x: x[1])[0]
            candidate_indices = [i for i in valid_moves if player.hand.cards[i].suit == most_common_suit]
            if candidate_indices:
                return max(candidate_indices, key=lambda i: player.hand.cards[i].value())
            return max(valid_moves, key=lambda i: player.hand.cards[i].value())

        if trick:
            lead_cards = [(pid, card) for pid, card in trick.played_cards if card.suit == lead_suit]
            highest_card_value = max(card.point_value for _, card in lead_cards)
            beating_cards = [i for i in valid_moves if player.hand.cards[i].suit == lead_suit and player.hand.cards[i].value() > highest_card_value]
            if beating_cards:
                # Play the weakest card that still beats
                return min(beating_cards, key=lambda i: player.hand.cards[i].value())

        # If cannot beat, discard lowest-value card from a less valuable suit
        suit_counts = player.hand.suit_counts()
        discard_index = min(valid_moves, key=lambda i: (player.hand.cards[i].value(), suit_counts.get(player.hand.cards[i].suit, 0)))
        return discard_index
    