"""Central Game Engine for Call Bridge."""

from __future__ import annotations
import random
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
from game.card import Card, Suit, TRUMP_SUIT
from game.deck import Deck
from game.rules import (
    NUM_PLAYERS,
    TRICKS_PER_ROUND,
    determine_trick_winner as rules_determine_trick_winner,
    get_legal_moves as rules_get_legal_moves,
    is_legal_move,
    next_player,
    validate_bid,
)
from game.scoring import (
    WINNING_SCORE,
    calculate_round_scores as scoring_calculate_round_scores,
    check_winning_condition,
    update_cumulative_scores as scoring_update_cumulative_scores,
)
from game.state import (
    GamePhase,
    InternalGameState,
    PlayerObservation,
    RoundSummary,
    TrickRecord,
)


class GameEngine:
    """Central authority managing Call Bridge state, rules, scoring, and observations.

    Internal state is stored in a name-mangled private attribute (`__state`) so
    AI agents cannot access opponents' hidden cards.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        winning_score: int = WINNING_SCORE,
        initial_dealer: int = 3,
    ) -> None:
        self._rng = random.Random(seed)
        self._winning_score = winning_score
        self._initial_dealer = initial_dealer % NUM_PLAYERS
        self.__state = InternalGameState()
        self.start_new_game(initial_dealer=self._initial_dealer)

    def start_new_game(self, initial_dealer: Optional[int] = None) -> None:
        """Reset all cumulative scores and start Round 1.

        By default, `initial_dealer=3` so `starting_player = (dealer + 1) % 4 = 0`
        (Player 1 / Human leads Round 1, and dealer rotates clockwise each round).
        """
        if initial_dealer is not None:
            self._initial_dealer = initial_dealer % NUM_PLAYERS

        self.__state = InternalGameState(
            num_players=NUM_PLAYERS,
            round_number=0,
            dealer=self._initial_dealer,
            starting_player=next_player(self._initial_dealer, NUM_PLAYERS),
            current_player=next_player(self._initial_dealer, NUM_PLAYERS),
            phase=GamePhase.BIDDING,
            trump_suit=TRUMP_SUIT,
        )
        self.start_new_round()

    def start_new_round(
        self,
        custom_hands: Optional[Dict[int, List[Card]]] = None,
        starting_player: Optional[int] = None,
    ) -> None:
        """Deal a new 13-card round and transition to the BIDDING phase.

        Raises RuntimeError if the game has already terminated.
        """
        if self.__state.phase == GamePhase.GAME_OVER:
            raise RuntimeError("Cannot start a new round after the game is over.")

        state = self.__state
        if state.round_number == 0:
            state.round_number = 1
            state.dealer = self._initial_dealer
        else:
            state.round_number += 1
            state.dealer = next_player(state.dealer, NUM_PLAYERS)

        if starting_player is not None:
            if not (0 <= starting_player < NUM_PLAYERS):
                raise ValueError(f"Invalid starting_player: {starting_player}")
            state.starting_player = starting_player
        else:
            state.starting_player = next_player(state.dealer, NUM_PLAYERS)

        state.current_player = state.starting_player
        state.phase = GamePhase.BIDDING
        state.trump_suit = TRUMP_SUIT

        if custom_hands is not None:
            self._validate_custom_hands(custom_hands)
            state.hands = {pid: sorted(list(cards)) for pid, cards in custom_hands.items()}
        else:
            deck = Deck(rng=self._rng)
            deck.shuffle()
            state.hands = deck.deal(NUM_PLAYERS)

        state.bids = {pid: None for pid in range(NUM_PLAYERS)}
        state.tricks_won = {pid: 0 for pid in range(NUM_PLAYERS)}
        state.current_trick = []
        state.trick_history = []
        state.cards_played = []
        state.known_void_suits = {pid: set() for pid in range(NUM_PLAYERS)}
        state.trick_number = 1
        state.round_scores = {pid: 0 for pid in range(NUM_PLAYERS)}
        state.last_trick_winner = None
        state.last_completed_trick = []

    @staticmethod
    def _validate_custom_hands(custom_hands: Dict[int, List[Card]]) -> None:
        if set(custom_hands.keys()) != set(range(NUM_PLAYERS)):
            raise ValueError("custom_hands must specify hands for player IDs 0, 1, 2, 3.")
        all_cards: List[Card] = []
        for pid, hand in custom_hands.items():
            if len(hand) != TRICKS_PER_ROUND:
                raise ValueError(f"Player {pid} must receive {TRICKS_PER_ROUND} cards, got {len(hand)}.")
            all_cards.extend(hand)
        if len(set(all_cards)) != NUM_PLAYERS * TRICKS_PER_ROUND:
            raise ValueError("custom_hands must contain 52 unique cards.")

    # ------------------------------------------------------------------
    # Public Read-Only Metadata Properties (No Hidden Hand Access!)
    # ------------------------------------------------------------------

    @property
    def phase(self) -> GamePhase:
        return self.__state.phase

    @property
    def round_number(self) -> int:
        return self.__state.round_number

    @property
    def trick_number(self) -> int:
        return self.__state.trick_number

    @property
    def dealer(self) -> int:
        return self.__state.dealer

    @property
    def starting_player(self) -> int:
        return self.__state.starting_player

    @property
    def current_player(self) -> int:
        return self.__state.current_player

    @property
    def trump_suit(self) -> Suit:
        return self.__state.trump_suit

    @property
    def bids(self) -> Dict[int, Optional[int]]:
        return dict(self.__state.bids)

    @property
    def tricks_won(self) -> Dict[int, int]:
        return dict(self.__state.tricks_won)

    @property
    def round_scores(self) -> Dict[int, int]:
        return dict(self.__state.round_scores)

    @property
    def cumulative_scores(self) -> Dict[int, int]:
        return dict(self.__state.cumulative_scores)

    @property
    def current_trick(self) -> List[Tuple[int, Card]]:
        return list(self.__state.current_trick)

    @property
    def last_completed_trick(self) -> List[Tuple[int, Card]]:
        return list(self.__state.last_completed_trick)

    @property
    def last_trick_winner(self) -> Optional[int]:
        return self.__state.last_trick_winner

    @property
    def trick_history(self) -> List[TrickRecord]:
        return list(self.__state.trick_history)

    @property
    def round_history(self) -> List[RoundSummary]:
        return list(self.__state.round_history)

    @property
    def winner_ids(self) -> List[int]:
        return list(self.__state.winner_ids)

    @property
    def is_game_over(self) -> bool:
        return self.__state.phase == GamePhase.GAME_OVER

    @property
    def winning_score(self) -> int:
        return self._winning_score

    @property
    def lead_suit(self) -> Optional[Suit]:
        if not self.__state.current_trick:
            return None
        return self.__state.current_trick[0][1].suit

    # ------------------------------------------------------------------
    # Core Engine API
    # ------------------------------------------------------------------

    def get_observation(self, player_id: int) -> PlayerObservation:
        """Return a strictly filtered observation for `player_id`.

        Opponents' cards are never included or reachable from this object.
        """
        if not (0 <= player_id < NUM_PLAYERS):
            raise ValueError(f"Invalid player_id: {player_id}")

        state = self.__state
        own_hand = tuple(state.hands[player_id])
        legal = tuple(self.get_legal_moves(player_id)) if (
            state.phase == GamePhase.PLAYING and state.current_player == player_id
        ) else tuple(rules_get_legal_moves(own_hand, self.lead_suit))

        known_voids: Dict[int, FrozenSet[Suit]] = {
            pid: frozenset(suits) for pid, suits in state.known_void_suits.items()
        }

        return PlayerObservation(
            player_id=player_id,
            hand=own_hand,
            own_bid=state.bids[player_id],
            public_bids=dict(state.bids),
            cards_played=tuple(state.cards_played),
            current_trick=tuple(state.current_trick),
            lead_suit=self.lead_suit,
            tricks_won=dict(state.tricks_won),
            remaining_card_counts={pid: len(state.hands[pid]) for pid in range(NUM_PLAYERS)},
            trump_suit=state.trump_suit,
            trick_history=tuple(state.trick_history),
            current_player=state.current_player,
            dealer=state.dealer,
            starting_player=state.starting_player,
            trick_number=state.trick_number,
            round_number=state.round_number,
            phase=state.phase,
            cumulative_scores=dict(state.cumulative_scores),
            legal_moves=legal,
            known_void_suits=known_voids,
        )

    def get_legal_moves(self, player_id: int) -> List[Card]:
        """Return the list of legal cards that `player_id` can play."""
        if not (0 <= player_id < NUM_PLAYERS):
            raise ValueError(f"Invalid player_id: {player_id}")
        hand = self.__state.hands[player_id]
        return rules_get_legal_moves(hand, self.lead_suit)

    def submit_bid(self, player_id: int, bid: int) -> None:
        """Submit a bid for `player_id` during the BIDDING phase."""
        state = self.__state
        if state.phase != GamePhase.BIDDING:
            raise RuntimeError(f"Cannot submit bid in phase {state.phase.value}.")
        if not (0 <= player_id < NUM_PLAYERS):
            raise ValueError(f"Invalid player_id: {player_id}")
        if player_id != state.current_player:
            raise ValueError(
                f"It is player {state.current_player}'s turn to bid, not player {player_id}."
            )
        if state.bids[player_id] is not None:
            raise ValueError(f"Player {player_id} has already submitted a bid.")

        validate_bid(bid)
        state.bids[player_id] = bid

        if all(b is not None for b in state.bids.values()):
            state.phase = GamePhase.PLAYING
            state.current_player = state.starting_player
        else:
            state.current_player = next_player(player_id, NUM_PLAYERS)

    def play_card(self, player_id: int, card: Card) -> Optional[int]:
        """Play `card` for `player_id` during the PLAYING phase.

        Returns:
            The trick winner's `player_id` if this card completed the 4-card trick,
            or `None` if the trick is still in progress.
        """
        state = self.__state
        if state.phase != GamePhase.PLAYING:
            raise RuntimeError(f"Cannot play a card in phase {state.phase.value}.")
        if not (0 <= player_id < NUM_PLAYERS):
            raise ValueError(f"Invalid player_id: {player_id}")
        if player_id != state.current_player:
            raise ValueError(
                f"It is player {state.current_player}'s turn to play, not player {player_id}."
            )

        hand = state.hands[player_id]
        lead_suit = self.lead_suit
        if not is_legal_move(card, hand, lead_suit):
            raise ValueError(
                f"Illegal move: {card} by player {player_id}. Lead suit: {lead_suit}, Hand: {hand}"
            )

        # Record public void-suit knowledge if player failed to follow lead suit
        if lead_suit is not None and card.suit != lead_suit:
            state.known_void_suits[player_id].add(lead_suit)

        hand.remove(card)
        state.current_trick.append((player_id, card))
        state.cards_played.append(card)

        if len(state.current_trick) < NUM_PLAYERS:
            state.current_player = next_player(player_id, NUM_PLAYERS)
            return None

        # Trick completed (4 cards played)
        winner_id = self.determine_trick_winner()
        completed_plays = tuple(state.current_trick)
        lead_player = completed_plays[0][0]
        trick_lead_suit = completed_plays[0][1].suit

        record = TrickRecord(
            trick_number=state.trick_number,
            lead_player=lead_player,
            lead_suit=trick_lead_suit,
            plays=completed_plays,
            winner=winner_id,
        )
        state.trick_history.append(record)
        state.tricks_won[winner_id] += 1
        state.last_trick_winner = winner_id
        state.last_completed_trick = list(completed_plays)
        state.current_trick.clear()

        if len(state.trick_history) == TRICKS_PER_ROUND:
            # All 13 tricks completed: score the round and check winning condition
            self.calculate_round_scores()
            self.update_cumulative_scores()
            summary = RoundSummary(
                round_number=state.round_number,
                dealer=state.dealer,
                starting_player=state.starting_player,
                bids={pid: int(b) for pid, b in state.bids.items() if b is not None},
                tricks_won=dict(state.tricks_won),
                round_scores=dict(state.round_scores),
                cumulative_scores=dict(state.cumulative_scores),
            )
            state.round_history.append(summary)

            game_over, winners = self.check_winner()
            if game_over:
                state.winner_ids = winners
                state.phase = GamePhase.GAME_OVER
            else:
                state.phase = GamePhase.ROUND_OVER
        else:
            state.trick_number += 1
            state.current_player = winner_id

        return winner_id

    def determine_trick_winner(self) -> int:
        """Determine the winner of the current 4-card trick using centralized rules."""
        return rules_determine_trick_winner(self.__state.current_trick)

    def calculate_round_scores(self) -> Dict[int, int]:
        """Calculate and store round scores after all 13 tricks are completed."""
        state = self.__state
        if any(b is None for b in state.bids.values()):
            raise RuntimeError("Cannot calculate scores before all bids are submitted.")
        validated_bids = {pid: int(b) for pid, b in state.bids.items() if b is not None}
        state.round_scores = scoring_calculate_round_scores(validated_bids, state.tricks_won)
        return dict(state.round_scores)

    def update_cumulative_scores(self) -> Dict[int, int]:
        """Add current round scores to cumulative scores and return updated totals."""
        state = self.__state
        state.cumulative_scores = scoring_update_cumulative_scores(
            state.cumulative_scores,
            state.round_scores,
        )
        return dict(state.cumulative_scores)

    def check_winner(self) -> Tuple[bool, List[int]]:
        """Check whether the winning condition (>= 16 points) has been met.

        Only valid after all 13 tricks of the current round have been completed.
        """
        state = self.__state
        if len(state.trick_history) < TRICKS_PER_ROUND:
            return False, []
        return check_winning_condition(state.cumulative_scores, self._winning_score)

    def get_game_results(self) -> Dict[str, Any]:
        """Return public game results summary."""
        state = self.__state
        return {
            "is_game_over": self.is_game_over,
            "phase": state.phase.value,
            "round_number": state.round_number,
            "winner_ids": list(state.winner_ids),
            "is_tie": len(state.winner_ids) > 1,
            "cumulative_scores": dict(state.cumulative_scores),
            "last_round_scores": dict(state.round_scores),
            "last_round_bids": dict(state.bids),
            "last_round_tricks_won": dict(state.tricks_won),
            "round_history": list(state.round_history),
        }
