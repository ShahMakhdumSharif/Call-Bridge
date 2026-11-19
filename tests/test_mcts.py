"""Automated tests for MCTS Agent 1 and MCTS Agent 2."""

import pytest
from game.card import Card
from game.engine import GameEngine
from players.mcts_agent import MCTSAgent


def test_mcts_agent1_and_agent2_are_independent_instances() -> None:
    agent1 = MCTSAgent.create_agent_1(player_id=1, simulation_budget=100, seed=1)
    agent2 = MCTSAgent.create_agent_2(player_id=2, simulation_budget=500, seed=2)

    assert agent1.mcts is not agent2.mcts
    assert agent1.simulation_budget == 100
    assert agent2.simulation_budget == 500
    assert agent1.exploration_constant != agent2.exploration_constant
    assert agent1.rollout_policy != agent2.rollout_policy


def test_mcts_configurable_budgets() -> None:
    agent = MCTSAgent.create_agent_1(player_id=1, simulation_budget=100, seed=42)
    for budget in (100, 500, 1000, 5000):
        agent.simulation_budget = budget
        assert agent.simulation_budget == budget

    with pytest.raises(ValueError):
        agent.simulation_budget = 0


def test_mcts_agents_make_valid_bids_and_legal_moves() -> None:
    engine = GameEngine(seed=77)
    agent1 = MCTSAgent.create_agent_1(player_id=1, simulation_budget=100, seed=10)
    agent2 = MCTSAgent.create_agent_2(player_id=2, simulation_budget=100, seed=20)

    # Test bidding for all 4 seats using MCTS agents
    agents = {
        0: MCTSAgent.create_agent_1(player_id=0, simulation_budget=100, seed=1),
        1: agent1,
        2: agent2,
        3: MCTSAgent.create_agent_2(player_id=3, simulation_budget=100, seed=3),
    }

    for _ in range(4):
        pid = engine.current_player
        obs = engine.get_observation(pid)
        bid = agents[pid].choose_bid(obs)
        assert isinstance(bid, int)
        assert 0 <= bid <= 13
        engine.submit_bid(pid, bid)

    # Play 3 full tricks and verify every move is legal and search populates stats
    for _ in range(3):
        for _ in range(4):
            pid = engine.current_player
            obs = engine.get_observation(pid)
            chosen_card = agents[pid].choose_card(obs)
            assert chosen_card in obs.legal_moves
            assert chosen_card in agents[pid].mcts.last_action_stats
            engine.play_card(pid, chosen_card)


def test_mcts_tactical_winning_move_selection() -> None:
    """Verify MCTS prefers winning a needed trick 4th-in-hand with KH over QH when AH is still out."""
    engine = GameEngine(seed=15, initial_dealer=3)
    # Player 0 has KH and 2H plus only tiny cards (no Spades, no Aces).
    # Winning Trick 1 4th-in-hand with KH over (QH, 10H, 9H) is Player 0's only chance to win a trick!
    custom_hands = {
        0: [Card.from_str(c) for c in ["KH", "2H", "3H", "4H", "2D", "3D", "4D", "5D", "6D", "2C", "3C", "4C", "5C"]],
        1: [Card.from_str(c) for c in ["QH", "5H", "6H", "7H", "7D", "8D", "9D", "10D", "JD", "6C", "7C", "8C", "9C"]],
        2: [Card.from_str(c) for c in ["10H", "8H", "JS", "10S", "9S", "8S", "7S", "6S", "5S", "10C", "JC", "QC", "KC"]],
        3: [Card.from_str(c) for c in ["9H", "JH", "AH", "AS", "KS", "QS", "4S", "3S", "2S", "QD", "KD", "AD", "AC"]],
    }
    engine._GameEngine__state.round_number = 0
    engine.start_new_round(custom_hands=custom_hands, starting_player=1)

    engine.submit_bid(1, 3)
    engine.submit_bid(2, 3)
    engine.submit_bid(3, 6)
    engine.submit_bid(0, 1)  # Player 0 bids 1 and needs 1 trick

    # Players 1, 2, 3 play Hearts: QH, 10H, 9H
    engine.play_card(1, Card.from_str("QH"))
    engine.play_card(2, Card.from_str("10H"))
    engine.play_card(3, Card.from_str("9H"))

    # Player 0 is 4th to act; KH wins immediately whereas 2H/3H/4H loses and AH/trumps dominate later.
    obs0 = engine.get_observation(0)
    agent = MCTSAgent.create_agent_1(player_id=0, simulation_budget=100, seed=42)
    decision = agent.choose_card(obs0)
    assert decision == Card.from_str("KH")
