let selectedBid = 3;
let bidSuggestedInitialized = false;
let lastRoundNumber = 0;
let lastHandCodesSignature = "";
let lastBidGridSignature = "";
let lastTrickSignature = "";
let lastSeatsSignature = "";
let lastHudSignature = "";

function formatLatex(str) {
  if (!str) return "";
  return String(str).replace(/Max\^n|Maxⁿ/g, '<span class="latex-math">Max<sup>n</sup></span>');
}

function renderCardHTML(card, extraClass = "") {
  const colorClass = card.is_red ? "red" : "black";
  const trumpClass = card.is_trump ? "trump" : "";
  return `
    <div class="playing-card ${colorClass} ${trumpClass} ${extraClass}">
      <div class="card-corner top-left">
        <div class="card-rank">${card.rank}</div>
        <div class="card-corner-suit">${card.symbol}</div>
      </div>
      <div class="card-center-suit">${card.symbol}</div>
      <div class="card-corner bottom-right">
        <div class="card-rank">${card.rank}</div>
        <div class="card-corner-suit">${card.symbol}</div>
      </div>
    </div>
  `;
}

function initHeroCards() {
  const hero = document.getElementById("heroCards");
  const sample = [
    {rank: "A", symbol: "♠", is_red: false, is_trump: true},
    {rank: "A", symbol: "♥", is_red: true, is_trump: false},
    {rank: "A", symbol: "♣", is_red: false, is_trump: false},
    {rank: "A", symbol: "♦", is_red: true, is_trump: false},
  ];
  hero.innerHTML = sample.map(c => renderCardHTML(c)).join("");
}

function buildBidGrid() {
  const sig = String(selectedBid);
  if (sig === lastBidGridSignature) return;
  lastBidGridSignature = sig;
  const grid = document.getElementById("bidGrid");
  let html = "";
  for (let b = 0; b <= 13; b++) {
    const sel = (b === selectedBid) ? "selected" : "";
    html += `<button class="bid-tile ${sel}" onclick="selectBid(${b})">${b}</button>`;
  }
  grid.innerHTML = html;
}

function selectBid(b) {
  selectedBid = b;
  buildBidGrid();
}

function setInputVal(id, val) {
  document.getElementById(id).value = val;
}

function openSettingsModal() {
  document.getElementById("settingsModal").classList.remove("hidden");
}
function closeSettingsModal() {
  document.getElementById("settingsModal").classList.add("hidden");
}
function openRulesModal() {
  document.getElementById("rulesModal").classList.remove("hidden");
}
function closeRulesModal() {
  document.getElementById("rulesModal").classList.add("hidden");
}
function showHomeScreen() {
  document.getElementById("homeOverlay").classList.remove("hidden");
}

async function startMatchFromHome() {
  await fetch("/api/new_game", {method: "POST"});
  bidSuggestedInitialized = false;
  lastHandCodesSignature = "";
  document.getElementById("homeOverlay").classList.add("hidden");
  pollState();
}

async function saveAiSettings() {
  const b1 = parseInt(document.getElementById("inputMcts1").value || "500", 10);
  const b2 = parseInt(document.getElementById("inputMcts2").value || "500", 10);
  const d3 = parseInt(document.getElementById("inputMaxn").value || "4", 10);
  await fetch("/api/settings", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({mcts1_budget: b1, mcts2_budget: b2, maxn_depth: d3})
  });
  closeSettingsModal();
  pollState();
}

async function confirmHumanBid() {
  await fetch("/api/bid", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({bid: selectedBid})
  });
  pollState();
}

async function playHumanCard(code) {
  await fetch("/api/play", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({card: code})
  });
  pollState();
}

async function nextRound() {
  await fetch("/api/next_round", {method: "POST"});
  bidSuggestedInitialized = false;
  lastHandCodesSignature = "";
  pollState();
}

async function startNewGame() {
  await fetch("/api/new_game", {method: "POST"});
  bidSuggestedInitialized = false;
  lastHandCodesSignature = "";
  pollState();
}

function updateHumanHandInPlace(humanHand, isHumanPlay) {
  const handEl = document.getElementById("humanHand");
  const codesSig = humanHand.map(c => c.code).join(",");

  // Only create/remove DOM elements when the set of cards in hand changes
  if (codesSig !== lastHandCodesSignature) {
    lastHandCodesSignature = codesSig;
    handEl.innerHTML = humanHand.map(card => `
      <div class="hand-card-slot" data-code="${card.code}" onclick="playHumanCard('${card.code}')">
        ${renderCardHTML(card)}
      </div>
    `).join("");
  }

  // Update legal / illegal classes in-place on existing DOM nodes without touching DOM structure
  const legalMap = {};
  humanHand.forEach(c => { legalMap[c.code] = c.is_legal; });

  const slots = handEl.querySelectorAll(".hand-card-slot");
  slots.forEach(slot => {
    const code = slot.getAttribute("data-code");
    const isLegal = isHumanPlay ? Boolean(legalMap[code]) : true;
    slot.classList.toggle("legal-slot", isLegal);
    slot.classList.toggle("illegal-slot", !isLegal);
  });
}

function updateUI(state) {
  if (state.round_number !== lastRoundNumber) {
    lastRoundNumber = state.round_number;
    bidSuggestedInitialized = false;
  }

  const cfg = state.config;
  if (document.getElementById("settingsModal").classList.contains("hidden")) {
    document.getElementById("inputMcts1").value = cfg.mcts_agent1_budget;
    document.getElementById("inputMcts2").value = cfg.mcts_agent2_budget;
    document.getElementById("inputMaxn").value = cfg.maxn_search_depth;
  }

  document.getElementById("hudRoundTrick").innerHTML =
    `ROUND ${state.round_number} &bull; TRICK ${state.trick_number}/13`;
  document.getElementById("hudDealer").innerHTML =
    `Dealer: ${formatLatex(state.players[state.dealer].name.split("(")[0].trim())}`;
  document.getElementById("hudLead").innerHTML =
    `1st Lead: ${formatLatex(state.players[state.starting_player].name.split("(")[0].trim())}`;

  const hudSig = JSON.stringify(state.players.map(p => [p.is_turn, p.bid, p.tricks_won, p.cumulative_score]));
  if (hudSig !== lastHudSignature) {
    lastHudSignature = hudSig;
    const hudRows = state.players.map(p => {
      const turnCls = p.is_turn ? "active-turn" : "";
      const bidStr = (p.bid === null) ? "-" : p.bid;
      return `<tr class="${turnCls}">
        <td>${p.is_turn ? "▸ " : ""}${formatLatex(p.name)}</td>
        <td>${bidStr}</td>
        <td>${p.tricks_won}</td>
        <td><b>${p.cumulative_score}</b></td>
      </tr>`;
    }).join("");
    document.getElementById("hudTableBody").innerHTML = hudRows;
  }

  const seatsSig = JSON.stringify(state.players);
  if (seatsSig !== lastSeatsSignature) {
    lastSeatsSignature = seatsSig;
    const codes = ["YOU", "M1", "M2", "MX"];
    state.players.forEach((p, idx) => {
      const el = document.getElementById(`seat-${idx}`);
      el.className = `seat-plaque ${["seat-bottom","seat-right","seat-top","seat-left"][idx]} ${p.is_turn ? "active-turn" : ""} ${p.is_trick_winner ? "trick-winner" : ""}`;
      const bidStr = (p.bid === null) ? "?" : p.bid;
      let badges = "";
      if (p.is_dealer) badges += `<span class="mini-badge dealer">Dealer</span>`;
      if (p.is_starting_player) badges += `<span class="mini-badge lead">1st Lead</span>`;
      if (idx > 0) badges += `<span class="mini-badge cards">🂠 ${p.remaining_cards} cards</span>`;
      if (p.is_turn) badges += `<span class="mini-badge turn">${idx === 0 ? "Your Turn" : "Thinking..."}</span>`;

      el.innerHTML = `
        <div class="plaque-top">
          <div class="avatar-badge">${codes[idx]}</div>
          <div>
            <div class="plaque-name">${formatLatex(p.name)}</div>
            <div class="plaque-meta">${formatLatex(p.ai_meta)}</div>
          </div>
        </div>
        <div class="plaque-stats">
          <span>Bid: ${bidStr}</span>
          <span>Won: ${p.tricks_won}</span>
          <span>Score: ${p.cumulative_score}</span>
        </div>
        <div class="plaque-badges">${badges}</div>
      `;
    });
  }

  const trickSig = JSON.stringify([state.displayed_trick, state.last_trick_winner]);
  if (trickSig !== lastTrickSignature) {
    lastTrickSignature = trickSig;
    const slotNames = ["You", "MCTS-1", "MCTS-2", '<span class="latex-math">Max<sup>n</sup></span>'];
    for (let pid = 0; pid < 4; pid++) {
      const slotEl = document.getElementById(`slot-${pid}`);
      const played = state.displayed_trick.find(item => item.player_id === pid);
      if (played) {
        const winCls = (state.last_trick_winner === pid) ? "winner-glow" : "";
        slotEl.innerHTML = renderCardHTML(played.card, `table-size ${winCls}`);
      } else {
        slotEl.innerHTML = slotNames[pid];
      }
    }
  }

  const banner = document.getElementById("statusBanner");
  if (state.error_message) {
    banner.innerHTML = formatLatex(state.error_message);
    banner.classList.add("error");
  } else {
    banner.innerHTML = formatLatex(state.status_message);
    banner.classList.remove("error");
  }

  const isHumanPlay = (state.phase === "PLAYING" && state.current_player === 0 && !state.holding_trick);
  updateHumanHandInPlace(state.human_hand, isHumanPlay);

  const bidModal = document.getElementById("bidModal");
  if (state.phase === "BIDDING" && state.current_player === 0) {
    if (!bidSuggestedInitialized) {
      selectedBid = state.hand_stats.suggested_bid;
      bidSuggestedInitialized = true;
    }
    buildBidGrid();
    bidModal.classList.remove("hidden");
  } else {
    bidModal.classList.add("hidden");
  }

  const endModal = document.getElementById("endModal");
  if (!state.holding_trick && (state.real_phase === "ROUND_OVER" || state.real_phase === "GAME_OVER")) {
    endModal.classList.remove("hidden");
    const isGameOver = (state.real_phase === "GAME_OVER");
    document.getElementById("endModalTitle").textContent =
      isGameOver ? "GAME OVER — FINAL TOURNAMENT STANDINGS" : `ROUND ${state.round_number} COMPLETE`;

    if (isGameOver) {
      const winners = state.winner_ids.map(id => formatLatex(state.players[id].name)).join(", ");
      const winScore = state.players[state.winner_ids[0]].cumulative_score;
      document.getElementById("endModalBanner").innerHTML =
        state.winner_ids.length > 1
          ? `TIE GAME! Co-Champions: ${winners} (${winScore} Points)`
          : `CHAMPION: ${winners} with ${winScore} Cumulative Points!`;
    } else {
      document.getElementById("endModalBanner").textContent =
        `First player to reach or exceed ${state.winning_score} points wins the game.`;
    }

    document.getElementById("endModalTableBody").innerHTML = state.players.map(p => {
      const scCls = p.round_score > 0 ? "score-pos" : (p.round_score < 0 ? "score-neg" : "");
      const rStr = p.round_score > 0 ? `+${p.round_score}` : `${p.round_score}`;
      return `<tr>
        <td><b>${formatLatex(p.name)}</b></td>
        <td>${formatLatex(p.ai_meta)}</td>
        <td>${p.bid}</td>
        <td>${p.tricks_won}</td>
        <td class="${scCls}">${rStr}</td>
        <td><b>${p.cumulative_score} pts</b></td>
      </tr>`;
    }).join("");

    document.getElementById("endModalButtons").innerHTML = isGameOver
      ? `<button class="btn gold" onclick="startNewGame()">Start New Game</button>
         <button class="btn" onclick="showHomeScreen(); document.getElementById('endModal').classList.add('hidden');">Return to Home Screen</button>`
      : `<button class="btn gold" onclick="nextRound()">Continue to Next Round &rarr;</button>`;
  } else {
    endModal.classList.add("hidden");
  }
}

async function pollState() {
  try {
    const res = await fetch("/api/state");
    const data = await res.json();
    updateUI(data);
  } catch (e) {}
}

initHeroCards();
buildBidGrid();
pollState();
setInterval(pollState, 180);
