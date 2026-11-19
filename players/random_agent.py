"""Random baseline agent for Call Bridge testing and experimental evaluation."""

from __future__ import annotations
import random
from typing import Optional
from game.card import Card
from game.rules import MAX_BID, MIN_BID, get_legal_moves
from game.state import PlayerObservation
from players.base_player import BasePlayer


class RandomAgent(BasePlayer):
    """Agent that chooses uniformly random legal bids (0-13) and legal cards."""

    def __init__(
        self,
        player_id: int,
        name: str = "Random Agent",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            player_id=player_id,
            name=name,
            agent_type="Random Baseline",
            is_human=False,
        )
        self._rng = random.Random(seed)

    def choose_bid(self, observation: PlayerObservation) -> int:
        """Select a random legal bid between 0 and 13 inclusive."""
        return self._rng.randint(MIN_BID, MAX_BID)

    def choose_card(self, observation: PlayerObservation) -> Card:
        """Select a random legal card from the observation's legal moves."""
        legal_moves = list(observation.legal_moves)
        if not legal_moves:
            legal_moves = get_legal_moves(observation.hand, observation.lead_suit)
        if not legal_moves:
            raise RuntimeError(f"No legal moves available for RandomAgent (player {self.player_id}).")
        return self._rng.choice(legal_moves)
