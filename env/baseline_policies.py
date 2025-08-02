from abc import ABC, abstractmethod
import random
from typing import List, Optional

from env.tresette_engine import Player, Card, Trick, TresetteEngine
from env.card_config import RANK_ORDER, RANKS, SUITS, POINT_PER_RANK

class BaselinePolicy(ABC):
    @abstractmethod
    def get_action_index() -> int:
        pass

class RandomPolicy(BaselinePolicy):
    @staticmethod
    def get_action_index(game_env: TresetteEngine) -> int:
        player: Player = game_env.players[game_env.current_player]
        lead_suit: str = game_env.trick.lead_suit
        valid_moves = player.get_valid_moves(lead_suit)
        return random.choice(valid_moves)

class HighestLeadSuitPolicy(BaselinePolicy):
    @staticmethod
    def get_action_index(game_env: TresetteEngine) -> int:
        player: Player = game_env.players[game_env.current_player]
        lead_suit: str = game_env.trick.lead_suit
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
    def get_action_index(game_env: TresetteEngine) -> int:
        player: Player = game_env.players[game_env.current_player]
        lead_suit: str = game_env.trick.lead_suit
        trick = game_env.trick
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


class AdvancedHeuristicPolicy(BaselinePolicy):

    @staticmethod
    def get_action_index(game_env: TresetteEngine) -> int:

        player: Player = game_env.players[game_env.current_player]
        lead_suit: str = game_env.trick.lead_suit


        valid_moves = player.get_valid_moves(lead_suit)

        # This would be a first play in the trick
        if lead_suit is None:
            # Check if there is an ace that can be played safely
            index_to_play = AdvancedHeuristicPolicy.try_playing_ace(player, game_env)
            if index_to_play is not None:
                return index_to_play
            
            # Get indices of cards that can be beaten by the opponent and those that cannot
            winning_indices = []
            unknown_indices = []

            for i, card in enumerate(player.hand.cards):
                if AdvancedHeuristicPolicy.can_opponent_beat_it(card.suit, card.rank, game_env):
                    unknown_indices.append(i)
                else:
                    winning_indices.append(i)


            # It would be wise to avoid playing suits with aces that might be in the opponent's hand
            suits_with_played_aces: List[str] = AdvancedHeuristicPolicy.get_suits_with_played_aces(game_env)
            suits_with_aces_in_hand: List[str] = AdvancedHeuristicPolicy.get_suits_with_aces(player)

            suits_safer_to_play= suits_with_played_aces.copy() 
            suits_safer_to_play.extend(suits_with_aces_in_hand)

            if winning_indices:
                # Check if there are indices of cards that can be played safely (Avoid playing cards that can be used to beat the opponents ace in the future)
                safe_indices = [i for i in winning_indices if player.hand.cards[i].suit in suits_safer_to_play]
                if safe_indices:
                    return AdvancedHeuristicPolicy.get_index_to_play_by_potential_points(safe_indices, player)
                elif unknown_indices:
                    # We will not choose from winning indices, because that would mean playing 2 or 3 when ace is still in play
                    return AdvancedHeuristicPolicy.play_weakest_card(player, unknown_indices)
                else:
                    return AdvancedHeuristicPolicy.play_weakest_card(player, winning_indices)
            else:
                safe_indices = [i for i in unknown_indices if player.hand.cards[i].suit in suits_safer_to_play]
                if safe_indices:
                    return AdvancedHeuristicPolicy.play_weakest_card(player, safe_indices)
                else:
                    return AdvancedHeuristicPolicy.play_weakest_card(player, valid_moves)
        
        # This is a second play in the trick
        else:
            beating_indices = AdvancedHeuristicPolicy.get_indices_of_cards_that_can_beat_the_lead_suit(valid_moves, player, game_env.trick.played_cards[0][1]) 
            ace_index = AdvancedHeuristicPolicy.get_card_index_by_suit_and_rank(player.hand.cards, lead_suit, '1')
            if ace_index is not None and ace_index in beating_indices:
                return ace_index
            else:
                if beating_indices:
                    return AdvancedHeuristicPolicy.get_index_to_play_by_potential_points(beating_indices, player)
                else:
                    # Play the weakest card from the lead suit
                    return AdvancedHeuristicPolicy.play_weakest_card(player, valid_moves)

    @staticmethod
    def get_indices_of_cards_that_can_beat_the_lead_suit(valid_moves: List[int], player: Player, opponent_card: Card):
        # Get all the indices of cards that can beat the lead suit
        beating_indices = []
        for i in valid_moves:
            card = player.hand.cards[i]
            if card.suit == opponent_card.suit and RANK_ORDER[card.rank] > RANK_ORDER[opponent_card.rank]:
                beating_indices.append(i)
        return beating_indices

    @staticmethod
    def get_index_to_play_by_potential_points(safe_indices: List[int], player: Player) -> int:
        # Play the card that has the highest potential points
        max_points = -1
        index_to_play = None
        min_rank = 14
        
        for index in safe_indices:
            card = player.hand.cards[index]
            potential_points = POINT_PER_RANK[card.rank]
            if potential_points > max_points:
                max_points = potential_points
                index_to_play = index
                min_rank = card.rank
            elif potential_points == max_points and RANK_ORDER[card.rank] < RANK_ORDER[min_rank]:
                max_points = potential_points
                index_to_play = index
                min_rank = card.rank

        return index_to_play

    @staticmethod
    def play_weakest_card(player: Player, valid_moves: List[int]) -> int:
        # Get all the indices from valid_moves with minimal rank
        index_to_play = None
        suit_count = player.hand.suit_counts()
        potential_indexes_to_play: List[int] = []
        lowest_rank_order = min(RANK_ORDER[player.hand.cards[i].rank] for i in valid_moves)

        for i in valid_moves:
            if RANK_ORDER[player.hand.cards[i].rank] == lowest_rank_order:
                potential_indexes_to_play.append(i)

        if len(potential_indexes_to_play) == 1:
            return potential_indexes_to_play[0]
        elif len(potential_indexes_to_play) > 1:
            # Play the one with the suit that has the most cards in hand
            index_to_play = max(potential_indexes_to_play, key=lambda i: suit_count.get(player.hand.cards[i].suit, 0))
            if index_to_play is not None:
                return index_to_play
            else:
                raise ValueError("No valid card found to play with the given suit and rank")
        else:
            raise ValueError("No valid moves to play")

    @staticmethod
    def get_suits_with_aces(player: Player) -> List[str]:
        return [card.suit for card in player.hand.cards if card.rank == '1']

    @staticmethod
    def get_suits_with_played_aces(game_env: TresetteEngine) -> List[str]:
        played_suits: List[str] = []
        played_cards: List[Card] = []

        # Add won cards from all players
        for player in game_env.players:
            for card in player.won_cards:
                if card:  # Ensure card is not None
                    played_cards.append(card[1])

        for card in played_cards:
            if card.rank == '1':  # Ace
                played_suits.append(card.suit)
        
        return played_suits

    @staticmethod
    def get_card_index_by_suit_and_rank(cards: List[Card], suit: str, rank: str) -> Optional[int]:
        for i, card in enumerate(cards):
            if card.suit == suit and card.rank == rank:
                return i
        return None

    @staticmethod
    def can_opponent_beat_it(suit: str, rank:str, game_env: TresetteEngine) -> bool:
        opponent_player: Player = game_env.players[(game_env.current_player + 1) % game_env.num_players]
        opponents_cards = opponent_player.known_cards

        # 1.Check if all opponent's cards are known and none of them can beat the designated card
        if len(opponents_cards) == len(opponent_player.hand.cards):
            for card in opponents_cards:
                if card.suit == suit and RANK_ORDER[card.rank] > RANK_ORDER[rank]:
                    return True
    
        # 2.Check if all the cards that can be beat the designated card are either played or in your hand
        cards_to_check: List[Card] = []
        
        # Add won cards from all players
        for player in game_env.players:
            for card in player.won_cards:
                if card:  # Ensure card is not None
                    cards_to_check.append(card[1])
        
        # Add cards from current player's hand
        current_player = game_env.players[game_env.current_player]
        for card in current_player.hand.cards:
            if card:  # Ensure card is not None
                cards_to_check.append(card)

        beating_ranks = AdvancedHeuristicPolicy.get_all_the_ranks_that_can_beat_the_given_rank(rank)

        for beating_rank in beating_ranks:
            if not any(card.suit == suit and card.rank == beating_rank for card in cards_to_check):
                return True
        
        return False

    @staticmethod
    def get_all_the_ranks_that_can_beat_the_given_rank(rank):
        # Returns all the ranks that can beat the given rank
        return [r for r in RANKS if RANK_ORDER[r] > RANK_ORDER[rank]]

    @staticmethod
    def try_playing_ace(player: Player, game_env: TresetteEngine) -> Optional[int]:

        # Check if there is a suit where you have the ace
        suits_with_aces = AdvancedHeuristicPolicy.get_suits_with_aces(player)
        if suits_with_aces:
            possible_aces_to_play = []
            # Check if you can still play the ace and opponent cannot beat it
            for suit in suits_with_aces:
                if not AdvancedHeuristicPolicy.can_opponent_beat_it(suit, '1', game_env):
                    possible_aces_to_play.append(suit)

            if possible_aces_to_play:
                # TODO: Modify this in future, for now play the first ace found
                choosen_suit = possible_aces_to_play[0]
                return AdvancedHeuristicPolicy.get_card_index_by_suit_and_rank(player.hand.cards, choosen_suit, '1')
            else:
                return None
        