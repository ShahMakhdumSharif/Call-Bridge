"""Automated tests for Heuristic Max^n Agent (Agent 3)."""

import pytest
from ai.heuristic import evaluate_state
from game.card import Card
from game.engine import GameEngine
from game.state import SimulationState
from players.maxn_agent import MaxNAgent


def test_maxn_configurable_depth() -> None:
    agent = MaxNAgent(player_id=3, max_depth=3, num_determinizations=4, seed=42)
    assert agent.max_depth == 3
    agent.max_depth = 5
    assert agent.max_depth == 5

    with pytest.raises(ValueError):
        agent.max_depth = 0


def test_heuristic_returns_4_player_utility_vector() -> None:
    sim = SimulationState(
        hands={
            0: [Card.from_str("AS"), Card.from_str("KH")],
            1: [Card.from_str("2S"), Card.from_str("3H")],
            2: [Card.from_str("3D"), Card.from_str("4D")],
            3: [Card.from_str("5C"), Card.from_str("6C")],
        },
        bids={0: 3, 1: 2, 2: 3, 3: 3},
        tricks_won={0: 2, 1: 2, 2: 4, 3: 3},
        current_trick=[],
        current_player=0,
        trick_number=12,
    )
    utilities = evaluate_state(sim, cumulative_scores={0: 10, 1: 8, 2: 5, 3: 12})
    assert len(utilities) == 4
    # Player 0 holds AS and KH and needs 1 trick to hit bid=3 (+3), whereas Player 1 holds tiny cards
    assert utilities[0] > utilities[2]  # Player 2 already overtricked (4 vs bid 3 -> 0 or negative)


def test_maxn_agent_makes_valid_bids_and_legal_moves() -> None:
    engine = GameEngine(seed=123)
    agents = {
        pid: MaxNAgent(player_id=pid, max_depth=3, num_determinizations=3, seed=10 + pid)
        for pid in range(4)
    }

    for _ in range(4):
        pid = engine.current_player
        obs = engine.get_observation(pid)
        bid = agents[pid].choose_bid(obs)
        assert isinstance(bid, int)
        assert 0 <= bid <= 13
        engine.submit_bid(pid, bid)

    # Play 3 full tricks and verify legal moves and search evaluations
    for _ in range(3):
        for _ in range(4):
            pid = engine.current_player
            obs = engine.get_observation(pid)
            card = agents[pid].choose_card(obs)
            assert card in obs.legal_moves
            assert agents[pid].maxn.nodes_evaluated >= 0
            engine.play_card(pid, card)


def test_maxn_maximizes_acting_player_own_utility_and_avoids_overtrick_penalty() -> None:
    """Verify Max^n avoids winning an unwanted trick that would push tricks_won > bid + 1."""
    engine = GameEngine(seed=88, initial_dealer=3)
    custom_hands = {
        0: [Card.from_str(c) for c in ["KH", "2H", "3D", "4D", "5D", "6D", "7D", "8D", "2C", "3C", "4C", "5C", "6C"]],
        1: [Card.from_str(c) for c in ["QH", "5H", "6H", "7H", "9D", "10D", "JD", "QD", "7C", "8C", "9C", "10C", "JC"]],
        2: [Card.from_str(c) for c in ["10H", "8H", "JS", "10S", "9S", "8S", "7S", "6S", "5S", "QC", "KC", "AC", "KD"]],
        3: [Card.from_str(c) for c in ["9H", "JH", "AH", "AS", "KS", "QS", "4S", "3S", "2S", "AD", "2D", "3H", "4H"]],
    }
    engine._GameEngine__state.round_number = 0
    engine.start_new_round(custom_hands=custom_hands, starting_player=1)

    engine.submit_bid(1, 3)
    engine.submit_bid(2, 3)
    engine.submit_bid(3, 6)
    engine.submit_bid(0, 0)  # Player 0 bid 0! Player 0 wants 0 tricks (or at most 1).

    # Play Trick 1: 1 leads QH, 2 plays 10H, 3 plays 9H
    engine.play_card(1, Card.from_str("QH"))
    engine.play_card(2, Card.from_str("10H"))
    engine.play_card(3, Card.from_str("9H"))

    # Player 0 has [2H, KH]. Since Player 0 bid 0, ducking with 2H avoids taking an unnecessary trick
    # and avoids being forced to lead Trick 2!
    obs0 = engine.get_observation(0)
    agent = MaxNAgent(player_id=0, max_depth=4, num_determinizations=4, seed=42)
    chosen = agent.choose_card(obs0)
    assert chosen == Card.from_str("2H")
