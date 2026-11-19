"""Main entry point for the Call Bridge Card Game with Intelligent AI Agents.

Launches the HTML5 & CSS3 Web UI (`python main.py`) or runs the experimental
AI evaluation benchmark suite (`python main.py --evaluate`).
"""

from __future__ import annotations
import argparse
import sys
from ai.evaluation import run_budget_comparison_suite
from ui.config import GameConfig
from ui.web_ui import run_web_ui


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Bridge Card Game with Intelligent MCTS & Heuristic Max^n AI Agents (HTML5/CSS3 UI)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface for the web server to bind to. Use 0.0.0.0 inside a container.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the HTML5/CSS3 Web UI server. Default: 8765",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the HTML5/CSS3 server without automatically opening a browser tab.",
    )
    parser.add_argument(
        "--mcts1-budget",
        type=int,
        default=500,
        help="Simulation budget for Player 2 (MCTS Agent 1). Default: 500",
    )
    parser.add_argument(
        "--mcts2-budget",
        type=int,
        default=500,
        help="Simulation budget for Player 3 (MCTS Agent 2). Default: 500",
    )
    parser.add_argument(
        "--maxn-depth",
        type=int,
        default=4,
        help="Search depth for Player 4 (Heuristic Max^n Agent 3). Default: 4",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run automated experimental evaluation comparing Random, MCTS-1, MCTS-2, and Max^n across budgets.",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=2,
        help="Number of complete games per budget in --evaluate mode. Default: 2",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.evaluate:
        reports = run_budget_comparison_suite(
            budgets=(100, 500, 1000),
            num_games=args.eval_games,
            max_rounds_per_game=10,
            base_seed=42,
        )
        for rep in reports:
            print(rep.format_table())
            print()
        return 0

    config = GameConfig(
        mcts_agent1_budget=args.mcts1_budget,
        mcts_agent2_budget=args.mcts2_budget,
        maxn_search_depth=args.maxn_depth,
    )
    run_web_ui(
        config=config,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
