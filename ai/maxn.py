"""Multiplayer Max^n search algorithm for Call Bridge."""

from __future__ import annotations
import random
from typing import Dict, List, Optional, Sequence, Tuple
from ai.determinization import sample_determinization
from ai.heuristic import evaluate_state
from ai.rollout import card_beats
from game.card import Card, TRUMP_SUIT
from game.rules import NUM_PLAYERS, determine_current_winning_play
from game.state import PlayerObservation, SimulationState

UtilityVector = Tuple[float, float, float, float]


class MaxNSearch:
    """Multiplayer Max^n search engine with configurable depth and determinization.

    Unlike two-player minimax, each player `p in {0, 1, 2, 3}` at an internal node
    chooses the child move that maximizes component `p` of the 4-player utility vector
    `(u_0, u_1, u_2, u_3)`.
    """

    def __init__(
        self,
        max_depth: int = 4,
        num_determinizations: int = 6,
        max_branching: int = 6,
        seed: Optional[int] = None,
    ) -> None:
        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {max_depth}.")
        if num_determinizations <= 0:
            raise ValueError(f"num_determinizations must be positive, got {num_determinizations}.")
        self.max_depth = max_depth
        self.num_determinizations = num_determinizations
        self.max_branching = max_branching
        self._rng = random.Random(seed)
        self.nodes_evaluated: int = 0
        self.last_action_utilities: Dict[Card, float] = {}

    def set_max_depth(self, depth: int) -> None:
        if depth <= 0:
            raise ValueError(f"max_depth must be positive, got {depth}.")
        self.max_depth = depth

    def search(self, observation: PlayerObservation) -> Card:
        """Choose the best legal card for `observation.player_id` using Max^n across determinizations."""
        legal_moves = list(observation.legal_moves)
        if not legal_moves:
            raise ValueError(f"No legal moves available for player {observation.player_id}.")
        if len(legal_moves) == 1:
            only_card = legal_moves[0]
            self.last_action_utilities = {only_card: 0.0}
            return only_card

        observer_id = observation.player_id
        cumulative_scores = dict(observation.cumulative_scores)
        self.nodes_evaluated = 0

        accumulated_utilities: Dict[Card, float] = {card: 0.0 for card in legal_moves}

        for _ in range(self.num_determinizations):
            world = sample_determinization(observation, self._rng)
            for card in legal_moves:
                next_state = world.clone()
                next_state.play_card(card)
                utility_vec = self._maxn(
                    state=next_state,
                    depth=self.max_depth - 1,
                    cumulative_scores=cumulative_scores,
                )
                accumulated_utilities[card] += utility_vec[observer_id]

        self.last_action_utilities = {
            card: total_u / float(self.num_determinizations)
            for card, total_u in accumulated_utilities.items()
        }

        best_card = max(legal_moves, key=lambda c: (round(self.last_action_utilities[c], 6), c.value))
        return best_card

    def _maxn(
        self,
        state: SimulationState,
        depth: int,
        cumulative_scores: Dict[int, int],
    ) -> UtilityVector:
        """Recursive 4-player Max^n evaluation."""
        self.nodes_evaluated += 1

        if depth <= 0 or state.is_terminal():
            return evaluate_state(state, cumulative_scores)

        acting_player = state.current_player
        legal_moves = state.get_legal_moves(acting_player)
        candidate_moves = self._order_and_prune_moves(state, legal_moves)

        best_vector: Optional[UtilityVector] = None
        best_player_utility = -float("inf")

        for card in candidate_moves:
            child_state = state.clone()
            child_state.play_card(card)
            child_vector = self._maxn(
                state=child_state,
                depth=depth - 1,
                cumulative_scores=cumulative_scores,
            )

            if best_vector is None or child_vector[acting_player] > best_player_utility:
                best_player_utility = child_vector[acting_player]
                best_vector = child_vector

        assert best_vector is not None
        return best_vector

    def _order_and_prune_moves(
        self,
        state: SimulationState,
        legal_moves: Sequence[Card],
    ) -> List[Card]:
        """Select strategically distinct representative moves when branching is wide."""
        if len(legal_moves) <= self.max_branching:
            return list(legal_moves)

        pid = state.current_player
        needed = state.bids.get(pid, 3) - state.tricks_won.get(pid, 0)
        sorted_moves = sorted(legal_moves, key=lambda c: (c.suit == TRUMP_SUIT, c.value))

        selected: List[Card] = []
        # Always include lowest and highest cards per available suit
        by_suit: Dict[object, List[Card]] = {}
        for card in sorted_moves:
            by_suit.setdefault(card.suit, []).append(card)

        for cards in by_suit.values():
            selected.append(cards[0])
            if len(cards) > 1:
                selected.append(cards[-1])
            if len(cards) > 2:
                selected.append(cards[len(cards) // 2])

        # Also include the cheapest winning card and highest losing card if following in a trick
        if state.current_trick:
            lead_suit = state.current_trick[0][1].suit
            winning_play = determine_current_winning_play(state.current_trick)
            if winning_play is not None:
                _, best_card = winning_play
                winners = [c for c in sorted_moves if card_beats(c, best_card, lead_suit)]
                losers = [c for c in sorted_moves if not card_beats(c, best_card, lead_suit)]
                if winners:
                    selected.append(winners[0])
                if losers:
                    selected.append(losers[-1])

        # Deduplicate while preserving order
        unique_moves: List[Card] = []
        seen = set()
        for card in selected:
            if card not in seen:
                seen.add(card)
                unique_moves.append(card)

        if needed > 0:
            unique_moves.sort(key=lambda c: (c.suit == TRUMP_SUIT, c.value), reverse=True)
        else:
            unique_moves.sort(key=lambda c: (c.suit == TRUMP_SUIT, c.value))

        return unique_moves[: self.max_branching]
