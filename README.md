# Call Bridge Card Game with Intelligent AI Agents (HTML5 + CSS3 + Python 3)

A complete, four-player individual **Call Bridge** card game written in **Python 3** with a responsive **HTML5 & CSS3** user interface. One human player competes individually against three distinct AI agents (**MCTS Agent 1**, **MCTS Agent 2**, and **Heuristic Maxⁿ Agent 3**) under strict imperfect-information constraints.

---

## 1. Key Features

- **Authentic Call Bridge Rules**:
  - 4 individual players (no partnerships), standard 52-card deck, 13 cards dealt per player, 13 tricks per round.
  - **Spades (`♠`)** is permanently the Trump suit.
  - Card rank ordering within each suit: `2 < 3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A`.
  - Anti-clockwise turn rotation around the table (`Bottom → Right → Top → Left`); lead player may play any card, and other players must follow the lead suit if able (or play any card when void).
  - Highest Spade wins the trick; otherwise, highest card of the lead suit wins and leads the next trick.
- **Independent Bidding System**:
  - Every player independently bids an integer in `0..13` (no requirement that the sum of bids equals 13).
  - Floating center bidding panel leaves your 13 cards and all opponents' bids completely unblurred and visible.
- **Centralized Scoring & 16-Point Winning Condition**:
  - Let `b` = player's bid and `t` = actual tricks won:
    - `t < b` → `-b`
    - `t == b` → `+b`
    - `t == b + 1` → `0`
    - `t > b + 1` → `-(t - b - 1)`
  - Scores are calculated only after all 13 tricks of a round are completed and accumulated across rounds.
  - The game immediately terminates after a completed round when one or more players reach or exceed **16 cumulative points** (ties resolved by highest cumulative score, or declared a tie if equal).
- **Four Distinct AI Implementations**:
  1. **Player 2 — MCTS Agent 1 (`MCTS-1`)**: Multiplayer UCT Monte Carlo Tree Search (`C = 1.414`) with goal-directed strategic rollouts and configurable simulation budgets (`100`, `500`, `1000`, `5000`, or any manually typed integer).
  2. **Player 3 — MCTS Agent 2 (`MCTS-2`)**: Independent MCTS instance (`C = 0.90`) with tactical trump-conserving rollouts and independent configurable simulation budgets.
  3. **Player 4 — Heuristic Maxⁿ Agent (`Max^n`)**: True 4-player Maxⁿ search maximizing each acting player's component of a 4-tuple utility vector `(u_0, u_1, u_2, u_3)` with a domain-specific heuristic evaluation function and configurable search depth.
  4. **Random Baseline Agent (`RandomAgent`)**: Uniformly random legal bidding (`0–13`) and legal card selection for automated testing and experimental benchmarking.
- **Strict Imperfect-Information Security**:
  - The internal `GameEngine` state is private (`__state`).
  - AI agents receive only a restricted, immutable `PlayerObservation` containing their own hand and public table history.
  - Determinization (`ai/determinization.py`) samples consistent hidden card distributions respecting public played cards, hand sizes, and revealed void suits without ever accessing real hidden hands.
- **HTML5 & CSS3 User Interface**:
  - Uses exact Unicode suit glyphs **`♠ ♥ ♣ ♦`** scaled proportionally to card dimensions (`--card-w`, `--card-h`).
  - Non-overlapping 13-card human hand row with stationary hover hit-boxes and in-place DOM updates so cards never shake on hover.
  - Manual numeric input boxes (`<input type="number">`) to type any custom MCTS simulation budget or Maxⁿ search depth before or during a match.

---

## 2. Project Structure

```text
Call Bridge/
│
├── main.py
├── requirements.txt
├── pytest.ini
├── README.md
│
├── game/
│   ├── __init__.py
│   ├── card.py
│   ├── deck.py
│   ├── state.py
│   ├── engine.py
│   ├── rules.py
│   └── scoring.py
│
├── players/
│   ├── __init__.py
│   ├── base_player.py
│   ├── human_player.py
│   ├── mcts_agent.py
│   ├── maxn_agent.py
│   └── random_agent.py
│
├── ai/
│   ├── __init__.py
│   ├── mcts.py
│   ├── mcts_node.py
│   ├── determinization.py
│   ├── rollout.py
│   ├── maxn.py
│   ├── heuristic.py
│   ├── bidding.py
│   └── evaluation.py
│
├── ui/
│   ├── __init__.py
│   ├── config.py
│   ├── web_ui.py
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
└── tests/
    ├── test_cards.py
    ├── test_engine.py
    ├── test_rules.py
    ├── test_scoring.py
    ├── test_mcts.py
    ├── test_maxn.py
    └── test_hidden_information.py
```

---

## 3. Running the Game

```bash
.venv/bin/python main.py
```

This starts the local Python game server on `http://127.0.0.1:8765` and opens the HTML5/CSS3 game table in your browser.

### Customizing AI Budgets via Command-Line Flags

```bash
.venv/bin/python main.py --mcts1-budget 1000 --mcts2-budget 500 --maxn-depth 4
```

### Running the Automated Test Suite

```bash
.venv/bin/pytest -v
```

### Running Experimental AI Evaluation & Benchmarks

```bash
.venv/bin/python main.py --evaluate --eval-games 2
```
