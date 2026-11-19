"""Card representation for Call Bridge."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict


class Suit(Enum):
    """Four standard playing card suits. Spades is permanently the trump suit."""

    CLUBS = "Clubs"
    DIAMONDS = "Diamonds"
    HEARTS = "Hearts"
    SPADES = "Spades"

    @property
    def symbol(self) -> str:
        symbols: Dict[Suit, str] = {
            Suit.CLUBS: "♣",
            Suit.DIAMONDS: "♦",
            Suit.HEARTS: "♥",
            Suit.SPADES: "♠",
        }
        return symbols[self]

    @property
    def short_code(self) -> str:
        codes: Dict[Suit, str] = {
            Suit.CLUBS: "C",
            Suit.DIAMONDS: "D",
            Suit.HEARTS: "H",
            Suit.SPADES: "S",
        }
        return codes[self]

    @property
    def sort_order(self) -> int:
        order: Dict[Suit, int] = {
            Suit.CLUBS: 0,
            Suit.DIAMONDS: 1,
            Suit.HEARTS: 2,
            Suit.SPADES: 3,
        }
        return order[self]

    @property
    def is_red(self) -> bool:
        return self in (Suit.HEARTS, Suit.DIAMONDS)


class Rank(Enum):
    """Card ranks ordered from 2 (lowest) to Ace (14, highest)."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def label(self) -> str:
        labels: Dict[int, str] = {
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "10",
            11: "J",
            12: "Q",
            13: "K",
            14: "A",
        }
        return labels[self.value]


TRUMP_SUIT: Suit = Suit.SPADES


@dataclass(frozen=True)
class Card:
    """Immutable standard playing card."""

    suit: Suit
    rank: Rank

    def __post_init__(self) -> None:
        if not isinstance(self.suit, Suit):
            raise TypeError(f"Expected Suit enum, got {type(self.suit)}")
        if not isinstance(self.rank, Rank):
            raise TypeError(f"Expected Rank enum, got {type(self.rank)}")

    @property
    def is_trump(self) -> bool:
        return self.suit == TRUMP_SUIT

    @property
    def value(self) -> int:
        return self.rank.value

    @property
    def short_name(self) -> str:
        return f"{self.rank.label}{self.suit.symbol}"

    @property
    def ascii_code(self) -> str:
        return f"{self.rank.label}{self.suit.short_code}"

    def __str__(self) -> str:
        return f"{self.rank.label}{self.suit.symbol}"

    def __repr__(self) -> str:
        return f"Card({self.rank.label}{self.suit.symbol})"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        if self.suit != other.suit:
            return self.suit.sort_order < other.suit.sort_order
        return self.rank.value < other.rank.value

    @classmethod
    def from_str(cls, code: str) -> Card:
        """Create a Card from string like 'AS', '10H', '2♠', 'KD'."""
        cleaned = code.strip().upper()
        if not cleaned:
            raise ValueError("Empty card string")

        suit_char = cleaned[-1]
        rank_str = cleaned[:-1]

        suit_map = {
            "C": Suit.CLUBS,
            "♣": Suit.CLUBS,
            "D": Suit.DIAMONDS,
            "♦": Suit.DIAMONDS,
            "H": Suit.HEARTS,
            "♥": Suit.HEARTS,
            "S": Suit.SPADES,
            "♠": Suit.SPADES,
        }
        rank_map = {
            "2": Rank.TWO,
            "3": Rank.THREE,
            "4": Rank.FOUR,
            "5": Rank.FIVE,
            "6": Rank.SIX,
            "7": Rank.SEVEN,
            "8": Rank.EIGHT,
            "9": Rank.NINE,
            "10": Rank.TEN,
            "T": Rank.TEN,
            "J": Rank.JACK,
            "Q": Rank.QUEEN,
            "K": Rank.KING,
            "A": Rank.ACE,
        }
        if suit_char not in suit_map:
            raise ValueError(f"Invalid suit in card code: {code}")
        if rank_str not in rank_map:
            raise ValueError(f"Invalid rank in card code: {code}")
        return cls(suit=suit_map[suit_char], rank=rank_map[rank_str])
