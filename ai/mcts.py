"""Monte Carlo Tree Search (MCTS) engine with determinization for Call Bridge."""

from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple
from ai.determinization import sample_determinization
from ai.mcts_node import MCTSNode
from ai.rollout import run_rollout
from game.card import Card
from game.state import PlayerObservation, SimulationState


class MCTSSearch:
    """Multiplayer Information-Set MCTS with UCT selection and determinization.

    Implements the four core MCTS steps:
    1. Selection (`_select`)
    2. Expansion (`_expand`)
    3. Simulation / Rollout (`_simulate`)
    4. Backpropagation (`_backpropagate`)
    """

    SUPPORTED_BUDGETS = (100, 500, 1000, 5000)

    def __init__(
        self,
        simulation_budget: int = 500,
        exploration_constant: float = 1.414,
        rollout_policy: str = "strategic",
        rollout_epsilon: float = 0.15,
        seed: Optional[int] = None,
    ) -> None:
        if simulation_budget <= 0:
            raise ValueError(f"simulation_budget must be positive, got {simulation_budget}.")
        self.simulation_budget = simulation_budget
        self.exploration_constant = exploration_constant
        self.rollout_policy = rollout_policy
        self.rollout_epsilon = rollout_epsilon
        self._rng = random.Random(seed)
        self.last_action_stats: Dict[Card, Dict[str, float]] = {}

    def set_simulation_budget(self, budget: int) -> None:
        """Update the simulation budget dynamically."""
        if budget <= 0:
            raise ValueError(f"simulation_budget must be positive, got {budget}.")
        self.simulation_budget = budget

    def search(self, observation: PlayerObservation) -> Card:
        """Run MCTS from the player's imperfect-information `observation` and return the best legal card."""
        legal_moves = list(observation.legal_moves)
        if not legal_moves:
            raise ValueError(f"No legal moves in observation for player {observation.player_id}.")
        if len(legal_moves) == 1:
            only_move = legal_moves[0]
            self.last_action_stats = {
                only_move: {"visits": 1.0, "avg_reward": 1.0}
            }
            return only_move

        root = MCTSNode(player_id=observation.current_player)
        cumulative_scores = dict(observation.cumulative_scores)

        # Cache a pool of determinizations so high budgets (1000-5000) run quickly while
        # still sampling diverse hidden card distributions.
        max_distinct_worlds = min(self.simulation_budget, 64)
        determinizations: List[SimulationState] = [
            sample_determinization(observation, self._rng)
            for _ in range(max_distinct_worlds)
        ]

        for sim_idx in range(self.simulation_budget):
            base_world = determinizations[sim_idx % max_distinct_worlds]
            sim_state = base_world.clone()

            # 1. Selection & 2. Expansion
            node = self._select_and_expand(root, sim_state)

            # 3. Simulation / Rollout
            rewards = self._simulate(sim_state, cumulative_scores)

            # 4. Backpropagation
            self._backpropagate(node, rewards)

        # Decision-making based on accumulated simulation results
        best_card = self._choose_best_action(root, legal_moves, observation.player_id)
        return best_card

    def _select_and_expand(self, root: MCTSNode, state: SimulationState) -> MCTSNode:
        """1. Selection & 2. Expansion phases using UCT in the determinized state."""
        node = root
        while not state.is_terminal():
            legal_moves = state.get_legal_moves()
            untried = node.get_untried_actions(legal_moves)

            if untried:
                # 2. Expansion: pick an untried legal move in this determinization
                chosen_action = self._rng.choice(untried)
                acting_pid = state.current_player
                state.play_card(chosen_action)
                child = node.expand(chosen_action, next_player_id=state.current_player)
                child.player_id = acting_pid
                return child

            # 1. Selection: all legal moves in this determinization have been tried
            acting_pid = state.current_player
            child = node.select_best_child(
                legal_moves=legal_moves,
                acting_player=acting_pid,
                exploration_constant=self.exploration_constant,
            )
            assert child.action is not None
            state.play_card(child.action)
            node = child

        return node

    def _simulate(
        self,
        state: SimulationState,
        cumulative_scores: Dict[int, int],
    ) -> Dict[int, float]:
        """3. Simulation / Rollout phase."""
        return run_rollout(
            state=state,
            rng=self._rng,
            policy=self.rollout_policy,
            cumulative_scores=cumulative_scores,
        )

    @staticmethod
    def _backpropagate(node: Optional[MCTSNode], rewards: Dict[int, float]) -> None:
        """4. Backpropagation phase: update visit counts and rewards up to the root."""
        curr = node
        while curr is not None:
            curr.update(rewards)
            curr = curr.parent

    def _choose_best_action(
        self,
        root: MCTSNode,
        legal_moves: List[Card],
        observer_id: int,
    ) -> Card:
        """Choose the best action at the root based on accumulated simulation results."""
        self.last_action_stats = {}
        best_card = legal_moves[0]
        best_score: Tuple[float, int] = (-float("inf"), -1)

        for card in legal_moves:
            child = root.children.get(card)
            if child is None or child.visits == 0:
                self.last_action_stats[card] = {"visits": 0.0, "avg_reward": 0.0}
                continue
            avg_reward = child.total_rewards[observer_id] / child.visits
            self.last_action_stats[card] = {
                "visits": float(child.visits),
                "avg_reward": avg_reward,
            }
            # Combine average reward and visit count for robust decision-making
            candidate_metric = (round(avg_reward, 5), child.visits)
            if candidate_metric > best_score:
                best_score = candidate_metric
                best_card = card

        return best_card
