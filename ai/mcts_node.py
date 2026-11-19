"""Tree node representation for Multiplayer Information-Set / Determinized MCTS."""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence
from game.card import Card
from game.rules import NUM_PLAYERS


class MCTSNode:
    """Node in the multiplayer MCTS search tree."""

    def __init__(
        self,
        player_id: int,
        parent: Optional[MCTSNode] = None,
        action: Optional[Card] = None,
    ) -> None:
        self.player_id = player_id
        self.parent = parent
        self.action = action
        self.children: Dict[Card, MCTSNode] = {}
        self.visits: int = 0
        self.availability: int = 0
        self.total_rewards: Dict[int, float] = {pid: 0.0 for pid in range(NUM_PLAYERS)}

    def get_untried_actions(self, legal_moves: Sequence[Card]) -> List[Card]:
        """Return legal moves in the current determinization that have not yet been expanded."""
        return [card for card in legal_moves if card not in self.children]

    def uct_score(
        self,
        acting_player: int,
        exploration_constant: float,
    ) -> float:
        """Compute the multiplayer UCT value of this child node for `acting_player`."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.total_rewards[acting_player] / self.visits
        parent_avail = max(1, self.availability)
        exploration = exploration_constant * math.sqrt(math.log(parent_avail) / self.visits)
        return exploitation + exploration

    def select_best_child(
        self,
        legal_moves: Sequence[Card],
        acting_player: int,
        exploration_constant: float,
    ) -> MCTSNode:
        """Select the legal child node with the highest UCT score for `acting_player`."""
        legal_children = [self.children[card] for card in legal_moves if card in self.children]
        if not legal_children:
            raise ValueError("No legal children available to select.")

        # Update availability counts for all legal children at this decision step
        for child in legal_children:
            child.availability += 1

        return max(
            legal_children,
            key=lambda child: child.uct_score(acting_player, exploration_constant),
        )

    def expand(self, action: Card, next_player_id: int) -> MCTSNode:
        """Create and attach a child node for `action`."""
        child = MCTSNode(
            player_id=next_player_id,
            parent=self,
            action=action,
        )
        child.availability = 1
        self.children[action] = child
        return child

    def update(self, rewards: Dict[int, float]) -> None:
        """Update visit count and accumulated rewards for all players."""
        self.visits += 1
        for pid, val in rewards.items():
            self.total_rewards[pid] += val
