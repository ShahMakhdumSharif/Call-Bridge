"""Configuration model for Call Bridge AI budgets and search depth."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GameConfig:
    """Configurable AI parameters adjustable via manual numeric inputs or CLI flags."""

    mcts_agent1_budget: int = 500
    mcts_agent2_budget: int = 500
    maxn_search_depth: int = 4
