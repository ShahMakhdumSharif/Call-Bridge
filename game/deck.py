"""Deck creation, shuffling, and dealing for Call Bridge."""

from __future__ import annotations
import random
from typing import Dict, List, Optional
from game.card import Card, Rank, Suit


class Deck:
    """Standard 52-card deck."""

    NUM_PLAYERS = 4
    CARDS_PER_PLAYER = 13

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self.cards: List[Card] = self.create_standard_deck()

    @staticmethod
    def create_standard_deck() -> List[Card]:
        """Return a fresh list of 52 unique standard cards."""
        return [
            Card(suit=suit, rank=rank)
            for suit in (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
            for rank in Rank
        ]

    def reset(self) -> None:
        """Reset the deck to a full 52-card state."""
        self.cards = self.create_standard_deck()

    def shuffle(self) -> None:
        """Shuffle the deck in place."""
        self._rng.shuffle(self.cards)

    def deal(self, num_players: int = NUM_PLAYERS) -> Dict[int, List[Card]]:
        """Deal 13 cards to each of the 4 players and return sorted hands."""
        if num_players != self.NUM_PLAYERS:
            raise ValueError(f"Call Bridge requires {self.NUM_PLAYERS} players.")
        if len(self.cards) != self.NUM_PLAYERS * self.CARDS_PER_PLAYER:
            raise ValueError("Deck must have exactly 52 cards before dealing.")

        hands: Dict[int, List[Card]] = {pid: [] for pid in range(num_players)}
        for idx, card in enumerate(self.cards):
            player_id = idx % num_players
            hands[player_id].append(card)

        for pid in hands:
            hands[pid].sort()

        return hands
