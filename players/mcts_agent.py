"""MCTS Player Agents (Agent 1 and Agent 2) for Call Bridge."""

from __future__ import annotations
from typing import Optional
from ai.bidding import choose_strategic_bid
from ai.mcts import MCTSSearch
from game.card import Card
from game.state import PlayerObservation
from players.base_player import BasePlayer


class MCTSAgent(BasePlayer):
    """Monte Carlo Tree Search AI agent operating under imperfect information.

    Supports two distinct presets (Agent 1 and Agent 2) as well as custom parameters:
    - Agent 1 (Balanced UCT + Strategic Rollouts): `exploration_constant=1.414`, `rollout_policy="strategic"`
    - Agent 2 (Exploratory UCT + Tactical-Conservative Rollouts): `exploration_constant=0.95`, `rollout_policy="tactical_conservative"`
    """

    def __init__(
        self,
        player_id: int,
        name: str = "MCTS Agent",
        simulation_budget: int = 500,
        exploration_constant: float = 1.414,
        rollout_policy: str = "strategic",
        bid_risk_offset: float = -0.20,
        seed: Optional[int] = None,
        agent_label: str = "MCTS",
    ) -> None:
        super().__init__(
            player_id=player_id,
            name=name,
            agent_type=agent_label,
            is_human=False,
        )
        self.bid_risk_offset = bid_risk_offset
        self.mcts = MCTSSearch(
            simulation_budget=simulation_budget,
            exploration_constant=exploration_constant,
            rollout_policy=rollout_policy,
            seed=seed,
        )

    @property
    def simulation_budget(self) -> int:
        return self.mcts.simulation_budget

    @simulation_budget.setter
    def simulation_budget(self, budget: int) -> None:
        self.mcts.set_simulation_budget(budget)

    @property
    def exploration_constant(self) -> float:
        return self.mcts.exploration_constant

    @property
    def rollout_policy(self) -> str:
        return self.mcts.rollout_policy

    @classmethod
    def create_agent_1(
        cls,
        player_id: int = 1,
        name: str = "Player 2 (MCTS-1)",
        simulation_budget: int = 500,
        seed: Optional[int] = None,
    ) -> MCTSAgent:
        """Factory for Intelligent Agent 1 — MCTS (UCT C=1.414, Strategic rollout)."""
        return cls(
            player_id=player_id,
            name=name,
            simulation_budget=simulation_budget,
            exploration_constant=1.414,
            rollout_policy="strategic",
            bid_risk_offset=-0.18,
            seed=seed,
            agent_label="MCTS-1 (UCT)",
        )

    @classmethod
    def create_agent_2(
        cls,
        player_id: int = 2,
        name: str = "Player 3 (MCTS-2)",
        simulation_budget: int = 500,
        seed: Optional[int] = None,
    ) -> MCTSAgent:
        """Factory for Intelligent Agent 2 — MCTS (Tactical C=0.90, Conservative rollout)."""
        return cls(
            player_id=player_id,
            name=name,
            simulation_budget=simulation_budget,
            exploration_constant=0.90,
            rollout_policy="tactical_conservative",
            bid_risk_offset=-0.28,
            seed=seed,
            agent_label="MCTS-2 (Tactical)",
        )

    def choose_bid(self, observation: PlayerObservation) -> int:
        """Estimate realistic tricks from the agent's own hand observation."""
        return choose_strategic_bid(observation, risk_offset=self.bid_risk_offset)

    def choose_card(self, observation: PlayerObservation) -> Card:
        """Run determinized MCTS simulations from `observation` to choose the best legal card."""
        return self.mcts.search(observation)
