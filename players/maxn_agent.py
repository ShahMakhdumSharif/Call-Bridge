"""Heuristic Max^n Player Agent (Agent 3) for Call Bridge."""

from __future__ import annotations
from typing import Optional
from ai.bidding import choose_strategic_bid
from ai.maxn import MaxNSearch
from game.card import Card
from game.state import PlayerObservation
from players.base_player import BasePlayer


class MaxNAgent(BasePlayer):
    """Intelligent Agent 3 — 4-Player Heuristic Max^n Agent."""

    def __init__(
        self,
        player_id: int = 3,
        name: str = "Player 4 (Max^n)",
        max_depth: int = 4,
        num_determinizations: int = 6,
        bid_risk_offset: float = -0.22,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            player_id=player_id,
            name=name,
            agent_type="Heuristic Max^n",
            is_human=False,
        )
        self.bid_risk_offset = bid_risk_offset
        self.maxn = MaxNSearch(
            max_depth=max_depth,
            num_determinizations=num_determinizations,
            seed=seed,
        )

    @property
    def max_depth(self) -> int:
        return self.maxn.max_depth

    @max_depth.setter
    def max_depth(self, depth: int) -> None:
        self.maxn.set_max_depth(depth)

    @property
    def num_determinizations(self) -> int:
        return self.maxn.num_determinizations

    @num_determinizations.setter
    def num_determinizations(self, count: int) -> None:
        if count <= 0:
            raise ValueError("num_determinizations must be positive.")
        self.maxn.num_determinizations = count

    def choose_bid(self, observation: PlayerObservation) -> int:
        """Estimate realistic tricks from the agent's own hand observation."""
        return choose_strategic_bid(observation, risk_offset=self.bid_risk_offset)

    def choose_card(self, observation: PlayerObservation) -> Card:
        """Run 4-player Max^n search with heuristic evaluation on determinized worlds."""
        return self.maxn.search(observation)
