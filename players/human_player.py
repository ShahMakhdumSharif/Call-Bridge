"""Human player representation for Call Bridge."""

from __future__ import annotations
from typing import Optional
from game.card import Card
from game.rules import is_legal_move, validate_bid
from game.state import PlayerObservation
from players.base_player import BasePlayer


class HumanPlayer(BasePlayer):
    """Human player controlled through the graphical interface (or queued inputs in tests)."""

    def __init__(self, player_id: int = 0, name: str = "Player 1 (You)") -> None:
        super().__init__(
            player_id=player_id,
            name=name,
            agent_type="Human",
            is_human=True,
        )
        self._pending_bid: Optional[int] = None
        self._pending_card: Optional[Card] = None

    def set_pending_bid(self, bid: int) -> None:
        """Set the bid chosen by the human user in the UI."""
        validate_bid(bid)
        self._pending_bid = bid

    def set_pending_card(self, card: Card) -> None:
        """Set the card chosen by the human user in the UI."""
        self._pending_card = card

    def choose_bid(self, observation: PlayerObservation) -> int:
        if self._pending_bid is None:
            raise RuntimeError("HumanPlayer has no pending bid set.")
        bid = self._pending_bid
        self._pending_bid = None
        validate_bid(bid)
        return bid

    def choose_card(self, observation: PlayerObservation) -> Card:
        if self._pending_card is None:
            raise RuntimeError("HumanPlayer has no pending card set.")
        card = self._pending_card
        self._pending_card = None
        if not is_legal_move(card, observation.hand, observation.lead_suit):
            raise ValueError(f"Selected card {card} is not a legal move.")
        return card
