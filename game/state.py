"""Game state and strict imperfect-information PlayerObservation models."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from game.card import Card, Suit, TRUMP_SUIT
from game.rules import (
    NUM_PLAYERS,
    TRICKS_PER_ROUND,
    determine_trick_winner,
    get_legal_moves,
    is_legal_move,
    next_player,
)
from game.scoring import calculate_round_scores


class GamePhase(Enum):
    """Lifecycle phases of a Call Bridge game."""

    BIDDING = "BIDDING"
    PLAYING = "PLAYING"
    ROUND_OVER = "ROUND_OVER"
    GAME_OVER = "GAME_OVER"


@dataclass(frozen=True)
class TrickRecord:
    """Public record of a completed trick."""

    trick_number: int
    lead_player: int
    lead_suit: Suit
    plays: Tuple[Tuple[int, Card], ...]
    winner: int


@dataclass(frozen=True)
class RoundSummary:
    """Summary of a completed round."""

    round_number: int
    dealer: int
    starting_player: int
    bids: Dict[int, int]
    tricks_won: Dict[int, int]
    round_scores: Dict[int, int]
    cumulative_scores: Dict[int, int]


@dataclass(frozen=True)
class PlayerObservation:
    """Strictly restricted observation provided to a single player or AI agent.

    Contains ONLY information legitimately observable by `player_id`.
    Never holds references to the internal GameState, GameEngine, or opponents' hands.
    """

    player_id: int
    hand: Tuple[Card, ...]
    own_bid: Optional[int]
    public_bids: Dict[int, Optional[int]]
    cards_played: Tuple[Card, ...]
    current_trick: Tuple[Tuple[int, Card], ...]
    lead_suit: Optional[Suit]
    tricks_won: Dict[int, int]
    remaining_card_counts: Dict[int, int]
    trump_suit: Suit
    trick_history: Tuple[TrickRecord, ...]
    current_player: int
    dealer: int
    starting_player: int
    trick_number: int
    round_number: int
    phase: GamePhase
    cumulative_scores: Dict[int, int]
    legal_moves: Tuple[Card, ...]
    known_void_suits: Dict[int, FrozenSet[Suit]]


@dataclass
class SimulationState:
    """Lightweight state used by AI search (MCTS / Max^n) on determinized hands.

    Uses the centralized rules and scoring modules exclusively.
    """

    hands: Dict[int, List[Card]]
    bids: Dict[int, int]
    tricks_won: Dict[int, int]
    current_trick: List[Tuple[int, Card]]
    current_player: int
    trick_number: int
    trump_suit: Suit = TRUMP_SUIT

    def clone(self) -> SimulationState:
        """Create a fast deep copy of the simulation state."""
        return SimulationState(
            hands={pid: list(cards) for pid, cards in self.hands.items()},
            bids=dict(self.bids),
            tricks_won=dict(self.tricks_won),
            current_trick=list(self.current_trick),
            current_player=self.current_player,
            trick_number=self.trick_number,
            trump_suit=self.trump_suit,
        )

    @property
    def lead_suit(self) -> Optional[Suit]:
        if not self.current_trick:
            return None
        return self.current_trick[0][1].suit

    def is_terminal(self) -> bool:
        """Return True if all 13 tricks have been completed."""
        return sum(self.tricks_won.values()) >= TRICKS_PER_ROUND or all(
            len(cards) == 0 for cards in self.hands.values()
        )

    def get_legal_moves(self, player_id: Optional[int] = None) -> List[Card]:
        """Return legal moves for the current (or specified) player using central rules."""
        pid = self.current_player if player_id is None else player_id
        return get_legal_moves(self.hands[pid], self.lead_suit)

    def play_card(self, card: Card) -> Optional[int]:
        """Apply a card play in the simulation using central rules.

        Returns the trick winner player_id if the trick completed, else None.
        """
        pid = self.current_player
        hand = self.hands[pid]
        if not is_legal_move(card, hand, self.lead_suit):
            raise ValueError(f"Illegal move {card} by player {pid} in simulation.")

        hand.remove(card)
        self.current_trick.append((pid, card))

        if len(self.current_trick) == NUM_PLAYERS:
            winner = determine_trick_winner(self.current_trick)
            self.tricks_won[winner] += 1
            self.current_trick.clear()
            self.current_player = winner
            self.trick_number += 1
            return winner

        self.current_player = next_player(pid, NUM_PLAYERS)
        return None

    def get_round_scores(self) -> Dict[int, int]:
        """Compute round scores at terminal state using central scoring."""
        return calculate_round_scores(self.bids, self.tricks_won)


@dataclass
class InternalGameState:
    """Complete internal state maintained privately inside GameEngine."""

    num_players: int = NUM_PLAYERS
    round_number: int = 0
    dealer: int = 0
    starting_player: int = 0
    current_player: int = 0
    phase: GamePhase = GamePhase.BIDDING
    trump_suit: Suit = TRUMP_SUIT
    hands: Dict[int, List[Card]] = field(
        default_factory=lambda: {pid: [] for pid in range(NUM_PLAYERS)}
    )
    bids: Dict[int, Optional[int]] = field(
        default_factory=lambda: {pid: None for pid in range(NUM_PLAYERS)}
    )
    tricks_won: Dict[int, int] = field(
        default_factory=lambda: {pid: 0 for pid in range(NUM_PLAYERS)}
    )
    current_trick: List[Tuple[int, Card]] = field(default_factory=list)
    trick_history: List[TrickRecord] = field(default_factory=list)
    cards_played: List[Card] = field(default_factory=list)
    known_void_suits: Dict[int, Set[Suit]] = field(
        default_factory=lambda: {pid: set() for pid in range(NUM_PLAYERS)}
    )
    trick_number: int = 1
    round_scores: Dict[int, int] = field(
        default_factory=lambda: {pid: 0 for pid in range(NUM_PLAYERS)}
    )
    cumulative_scores: Dict[int, int] = field(
        default_factory=lambda: {pid: 0 for pid in range(NUM_PLAYERS)}
    )
    round_history: List[RoundSummary] = field(default_factory=list)
    winner_ids: List[int] = field(default_factory=list)
    last_trick_winner: Optional[int] = None
    last_completed_trick: List[Tuple[int, Card]] = field(default_factory=list)
