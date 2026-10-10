"""Heuristic + Monte Carlo decision engine for 4-player Catan.

Pipeline for ``CatanEngine.recommend``:

1. Enumerate candidate moves for the acting player given the phase
   (robber placement, discards, or main-phase builds / trades / dev cards).
2. Score each candidate with a fast heuristic: the change in
   ``position_score`` (a VP-equivalent evaluation of production, expansion,
   hand progress, dev cards and longest-road / largest-army potential).
3. Re-rank the strongest candidates with Monte Carlo rollouts: hidden hands
   are sampled from the ``HandBelief``, unseen dev cards from the remaining
   deck, then a fast greedy self-play simulator plays the game out.  Common
   random numbers are shared across candidates to cut variance.

All heuristic sub-systems are exposed individually:

* ``settlement_scores``  - pip/diversity/port/expansion node evaluation
* ``road_plans``         - Dijkstra routing toward uncontested high-value spots
* ``robber_options``     - opponent disruption vs. self-block, leader targeting
* ``resource_values`` / ``evaluate_trade`` - goal-driven dynamic resource pricing
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .board import PIPS, RES_INDEX, RESOURCES, Board, Resource
from .state_manager import (
    COSTS, DEV_DECK, NUM_RES, PIECE_LIMITS, GameState, can_afford, fmt_hand,
)

WIN_VP = 10


class AnalysisCancelled(Exception):
    """Raised inside a running analysis when ``should_stop`` reports a newer request."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class NodeScore:
    node: int
    value: float
    pips: int
    production: dict[str, int]
    diversity: float
    port: float
    expansion: float

    def __str__(self) -> str:
        prod = " ".join(f"{k[:2]}{v}" for k, v in self.production.items())
        return f"node {self.node}: {self.value:.2f} ({self.pips} pips: {prod})"


@dataclass
class RoadPlan:
    target: int
    edges: list[int]
    cost: int
    score: float
    contested: str  # "free" | "race" | "behind"


@dataclass
class RobberOption:
    hex: int
    victim: str | None
    score: float
    detail: str


@dataclass
class Move:
    label: str
    events: list[dict[str, Any]] = field(default_factory=list)
    heuristic: float = 0.0
    win_prob: float | None = None
    detail: str = ""
    rank_score: float = 0.0
    kind: str = ""                          # "", "offer", "devcard", "roll", "setup"
    offer: dict[str, list[int]] | None = None


@dataclass
class Recommendation:
    player: str
    phase: str
    moves: list[Move]
    position_score: float
    win_prob: float
    all_win_probs: dict[str, float]
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> Move:
        return self.moves[0]

    def render(self) -> str:
        lines = [f"=== AI advice for {self.player} ({self.phase} phase) ==="]
        if self.moves:
            lines.append(f"Optimal Move: [{self.moves[0].label}]")
        if len(self.moves) > 1:
            lines.append(f"Alternative: [{self.moves[1].label}]")
        lines.append(f"Expected Win % / Position Score: [{self.win_prob * 100:.1f}% / {self.position_score:.2f}]")
        lines.append("Top moves:")
        for i, m in enumerate(self.moves, 1):
            wp = f"win {m.win_prob * 100:5.1f}%" if m.win_prob is not None else "win   n/a"
            lines.append(f"  {i}. {m.label:<44} {wp} | heuristic {m.heuristic:+.2f}"
                         + (f" | {m.detail}" if m.detail else ""))
        if self.all_win_probs:
            lines.append("Table win %: " + ", ".join(f"{c} {p * 100:.1f}%" for c, p in self.all_win_probs.items()))
        if any(m.win_prob is not None for m in self.moves):
            lines.append("ranking: Monte Carlo win %; heuristic breaks ties within half a standard error")
        lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CatanEngine:
    def __init__(self, game: GameState, *, rollouts: int = 80, seed: int = 7,
                 max_rounds: int = 60, mc_top_k: int = 5,
                 should_stop: Callable[[], bool] | None = None):
        self.game = game
        self.should_stop = should_stop
        self.rollouts = rollouts
        self.seed = seed
        self.max_rounds = max_rounds
        self.mc_top_k = mc_top_k
        self._weights = self._resource_weights(game.board)

    # -- static board knowledge ---------------------------------------------
    @staticmethod
    def _resource_weights(board: Board) -> list[float]:
        """Scarcity weights (mean 1.0): scarce resources are worth more per pip."""
        totals = board.resource_total_pips()
        mean = sum(totals.values()) / len(totals)
        raw = [min(1.6, max(0.7, math.sqrt(mean / max(totals[r], 1)))) for r in RESOURCES]
        avg = sum(raw) / len(raw)
        return [w / avg for w in raw]

    def player_production(self, color: str, game: GameState | None = None,
                          include_robber: bool = False) -> list[float]:
        """Expected pips per resource (multiply by 1/36 for cards per roll)."""
        g = game or self.game
        prod = [0.0] * NUM_RES
        p = g.players[color]
        for n in p.buildings():
            mult = 2 if n in p.cities else 1
            for r, pips in g.board.node_production(n, include_robber=include_robber).items():
                prod[RES_INDEX[r]] += mult * pips
        return prod

    # -- 1. settlement scoring ---------------------------------------------
    def node_score(self, node: int, color: str | None = None, game: GameState | None = None) -> NodeScore:
        g = game or self.game
        board = g.board
        prod = board.node_production(node, include_robber=True)
        pips = sum(prod.values())
        base = sum(self._weights[RES_INDEX[r]] * v for r, v in prod.items())
        robbed = board.tiles[board.robber_hex]
        if node in board.topology.hex_nodes[board.robber_hex] and robbed.resource:
            base -= 0.3 * robbed.pips  # temporary block, mild penalty
        own = self.player_production(color, g, include_robber=True) if color else [0.0] * NUM_RES
        diversity = 0.0
        for r in prod:
            diversity += 1.2 if own[RES_INDEX[r]] == 0 else 0.4
        port_bonus = 0.0
        port = board.node_ports.get(node)
        if port is not None:
            if port.resource is None:
                port_bonus = 1.5
            else:
                ri = RES_INDEX[port.resource]
                port_bonus = 0.5 + 0.35 * (own[ri] + prod.get(port.resource, 0))
        expansion = 0.0
        topo = board.topology
        seen = {node}
        for nb in topo.node_neighbors[node]:
            for nb2 in topo.node_neighbors[nb]:
                if nb2 in seen:
                    continue
                seen.add(nb2)
                if g.can_place_settlement(color or "", nb2, require_road=False) and not \
                        any(nb3 == node for nb3 in topo.node_neighbors[nb2]):
                    expansion = max(expansion, 0.15 * board.node_pips(nb2))
        value = base + diversity + port_bonus + expansion
        return NodeScore(node, value, pips, {r.value: v for r, v in prod.items()}, diversity, port_bonus, expansion)

    def settlement_scores(self, color: str, *, require_road: bool = True, top: int | None = None,
                          game: GameState | None = None) -> list[NodeScore]:
        g = game or self.game
        nodes = g.legal_settlement_nodes(color, require_road=require_road)
        scores = sorted((self.node_score(n, color, g) for n in nodes), key=lambda s: -s.value)
        return scores[:top] if top else scores

    # -- 2. road pathing ------------------------------------------------------
    def _network_distances(self, color: str, g: GameState) -> tuple[dict[int, int], dict[int, tuple[int, int]]]:
        topo = g.board.topology
        p = g.players[color]
        occupied = {n: c for c, q in g.players.items() for n in q.buildings()}
        road_owner = {e: c for c, q in g.players.items() for e in q.roads}
        sources = set(p.buildings())
        for e in p.roads:
            sources.update(topo.edge_nodes[e])
        dist: dict[int, int] = {}
        prev: dict[int, tuple[int, int]] = {}
        heap = [(0, n) for n in sources]
        for n in sources:
            dist[n] = 0
        heapq.heapify(heap)
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, 1 << 30):
                continue
            if occupied.get(u, color) != color:
                continue  # cannot build through an opponent's building
            for e in topo.node_edges[u]:
                owner = road_owner.get(e)
                if owner is not None and owner != color:
                    continue
                a, b = topo.edge_nodes[e]
                v = b if a == u else a
                nd = d + (0 if owner == color else 1)
                if nd < dist.get(v, 1 << 30) and nd <= 6:
                    dist[v] = nd
                    prev[v] = (u, e)
                    heapq.heappush(heap, (nd, v))
        return dist, prev

    def road_plans(self, color: str, *, top: int = 3, game: GameState | None = None) -> list[RoadPlan]:
        g = game or self.game
        if not g.players[color].buildings():
            return []
        dist, prev = self._network_distances(color, g)
        opp_dist = [self._network_distances(c, g)[0] for c in g.order if c != color]
        plans = []
        for node, d in dist.items():
            if d == 0 or not g.can_place_settlement(color, node, require_road=False):
                continue
            edges = []
            cur = node
            while cur in prev and dist[cur] > 0:
                u, e = prev[cur]
                if e not in g.players[color].roads:
                    edges.append(e)
                cur = u
            edges.reverse()
            best_opp = min((od.get(node, 99) for od in opp_dist), default=99)
            contested = "free" if best_opp > d else ("race" if best_opp == d else "behind")
            factor = {"free": 1.0, "race": 0.65, "behind": 0.3}[contested]
            value = self.node_score(node, color, g).value
            plans.append(RoadPlan(node, edges, d, value * factor * (0.75 ** (d - 1)), contested))
        plans.sort(key=lambda p: -p.score)
        return plans[:top]

    # -- 3. robber placement -------------------------------------------------
    def _leader_factor(self, color: str, g: GameState) -> float:
        vps = {c: g.public_vp(c) + g.expected_hidden_vp(c) for c in g.order}
        mean = sum(vps.values()) / len(vps)
        return max(0.5, 1.0 + 0.3 * (vps[color] - mean))

    def robber_options(self, color: str, *, top: int = 3, game: GameState | None = None) -> list[RobberOption]:
        g = game or self.game
        board = g.board
        topo = board.topology
        rolls_blocked = 8.0  # robber typically sits ~2 rounds
        # Monopoly-ish producers: share of a resource's total production on the board.
        shares: dict[tuple[str, int], float] = {}
        for c in g.order:
            prod = self.player_production(c, g, include_robber=True)
            for i in range(NUM_RES):
                shares[(c, i)] = prod[i]
        totals = [sum(shares[(c, i)] for c in g.order) or 1.0 for i in range(NUM_RES)]
        options = []
        for h, tile in enumerate(board.tiles):
            if h == board.robber_hex:
                continue
            loss, self_loss = 0.0, 0.0
            victims: dict[str, float] = {}
            for n in topo.hex_nodes[h]:
                owner = g.node_owner(n)
                if not owner:
                    continue
                mult = 2 if owner[1] == "city" else 1
                cards = mult * tile.pips * rolls_blocked / 36.0
                if owner[0] == color:
                    self_loss += cards
                    continue
                f = self._leader_factor(owner[0], g)
                if tile.resource is not None:
                    ri = RES_INDEX[tile.resource]
                    if shares[(owner[0], ri)] / totals[ri] > 0.5:
                        f += 0.3
                    loss += cards * f * self._weights[ri]
                victims[owner[0]] = victims.get(owner[0], 0.0) + f
            best_victim, steal = None, 0.0
            for v, f in victims.items():
                cards = g.hand_total(v)
                if cards == 0:
                    continue
                val = f * (1.0 + (0.3 if cards > 7 else 0.0))
                if val > steal:
                    best_victim, steal = v, val
            score = loss + steal - 1.5 * self_loss
            detail = f"blocks {loss:.1f} wtd cards" + (f", self -{self_loss:.1f}" if self_loss else "")
            options.append(RobberOption(h, best_victim, score, detail))
        options.sort(key=lambda o: -o.score)
        return options[:top]

    # -- 4. trade evaluation ---------------------------------------------------
    def build_goals(self, color: str, game: GameState | None = None) -> list[tuple[str, tuple[int, ...], float]]:
        g = game or self.game
        p = g.players[color]
        goals: list[tuple[str, tuple[int, ...], float]] = []
        if p.settlements and len(p.cities) < PIECE_LIMITS["city"]:
            goals.append(("city", COSTS["city"], 1.3))
        if len(p.settlements) < PIECE_LIMITS["settlement"]:
            ready = bool(g.legal_settlement_nodes(color))
            goals.append(("settlement", COSTS["settlement"], 1.2 if ready else 0.7))
            if not ready:
                goals.append(("road", COSTS["road"], 0.8))
        if g.dev_deck > 0:
            goals.append(("devcard", COSTS["devcard"], 0.6))
        return goals

    def hand_potential(self, color: str, hand: Sequence[float], game: GameState | None = None) -> float:
        """VP-equivalent value of holding ``hand`` toward the player's build goals."""
        progress = sorted(
            (weight * (sum(min(h, c) for h, c in zip(hand, cost)) / sum(cost)) ** 2
             for _, cost, weight in self.build_goals(color, game)), reverse=True)
        # the closest goal dominates; secondary goals still give cards some value
        goal_value = (progress[0] + 0.25 * sum(progress[1:])) if progress else 0.0
        total = sum(hand)
        risk = (1 / 6) * (total / 2) * 0.12 if total > 7 else 0.0  # P(7) * expected discard
        return 0.5 * goal_value + 0.04 * total - risk

    def resource_values(self, color: str, hand: Sequence[float] | None = None,
                        game: GameState | None = None) -> dict[str, float]:
        g = game or self.game
        hand = list(hand if hand is not None else g.expected_hand(color))
        prod = self.player_production(color, g)
        out = {}
        for i, r in enumerate(RESOURCES):
            v = 0.25 * self._weights[i] / (1.0 + prod[i] / 6.0)
            for name, cost, weight in self.build_goals(color, g):
                missing = [max(0, c - h) for c, h in zip(cost, hand)]
                if missing[i] > 0:
                    v += weight / max(1, sum(missing))
            out[r.value] = round(v, 3)
        return out

    def evaluate_trade(self, color: str, give: Sequence[int], get: Sequence[int],
                       partner: str | None = None, game: GameState | None = None) -> float:
        """Net VP-equivalent gain; positive means accept. Penalises helping a leader."""
        g = game or self.game
        hand = g.expected_hand(color)
        after = [h - a + b for h, a, b in zip(hand, give, get)]
        if min(after) < -1e-9:
            return float("-inf")
        net = self.hand_potential(color, after, g) - self.hand_potential(color, hand, g)
        if partner:
            ph = g.expected_hand(partner)
            pa = [h - b + a for h, a, b in zip(ph, give, get)]
            gain = self.hand_potential(partner, pa, g) - self.hand_potential(partner, ph, g)
            net -= 0.6 * max(0.0, gain) * self._leader_factor(partner, g)
        return net

    # -- position evaluation -------------------------------------------------
    def position_score(self, color: str, game: GameState | None = None) -> float:
        g = game or self.game
        p = g.players[color]
        vp = g.public_vp(color) + g.expected_hidden_vp(color)
        prod = self.player_production(color, g)
        prod_value = sum(w * x for w, x in zip(self._weights, prod)) / 36.0
        diversity = sum(1 for x in prod if x > 0)
        plans = self.road_plans(color, top=1, game=g)
        ready = self.settlement_scores(color, top=1, game=g)
        expansion = 0.0
        if len(p.settlements) + len(p.cities) < 9 and len(p.settlements) < PIECE_LIMITS["settlement"]:
            if ready:
                expansion = 0.06 * ready[0].value
            elif plans:
                expansion = 0.05 * plans[0].score
        hand = g.expected_hand(color)
        dev = p.dev_hidden * 0.3 if color != g.perspective else \
            0.3 * (p.dev_hidden - p.dev_known.get("vp", 0))
        army = 0.12 * p.knights_played if g.largest_army_owner != color else 0.0
        road = 0.08 * g.longest_road(color) if g.longest_road_owner != color else 0.0
        return (vp + 4.0 * prod_value + 0.12 * diversity + expansion
                + self.hand_potential(color, hand, g) + dev + army + road)

    def softmax_win(self, game: GameState | None = None) -> dict[str, float]:
        g = game or self.game
        scores = {c: self.position_score(c, g) for c in g.order}
        m = max(scores.values())
        exps = {c: math.exp(1.2 * (s - m)) for c, s in scores.items()}
        z = sum(exps.values())
        return {c: v / z for c, v in exps.items()}

    # -- candidate generation ---------------------------------------------------
    def _acting_hand(self, color: str) -> list[int]:
        return list(self.game.belief.most_likely()[self.game.idx(color)])

    def _candidates(self, color: str) -> list[Move]:
        g = self.game
        p = g.players[color]
        hand = self._acting_hand(color)
        moves: list[Move] = []

        def add_build_moves(h: list[int], prefix: list[dict[str, Any]], prefix_label: str) -> None:
            if can_afford(h, COSTS["city"]) and len(p.cities) < PIECE_LIMITS["city"]:
                for n in p.settlements:
                    moves.append(Move(prefix_label + f"Build City at Node {n}",
                                      prefix + [{"type": "build", "player": color, "item": "city", "target": n}],
                                      detail=f"{g.board.node_pips(n)} pips"))
            if can_afford(h, COSTS["settlement"]) and len(p.settlements) < PIECE_LIMITS["settlement"]:
                for ns in self.settlement_scores(color, top=3):
                    moves.append(Move(prefix_label + f"Build Settlement at Node {ns.node}",
                                      prefix + [{"type": "build", "player": color, "item": "settlement",
                                                 "target": ns.node}], detail=str(ns)))
            if can_afford(h, COSTS["road"]) and len(p.roads) < PIECE_LIMITS["road"]:
                seen = set()
                for plan in self.road_plans(color, top=3):
                    if not plan.edges or plan.edges[0] in seen:
                        continue
                    e = plan.edges[0]
                    seen.add(e)
                    a, b = g.board.topology.edge_nodes[e]
                    moves.append(Move(prefix_label + f"Build Road on Edge {e} ({a}-{b})",
                                      prefix + [{"type": "build", "player": color, "item": "road", "target": e}],
                                      detail=f"toward node {plan.target} ({plan.cost} roads, {plan.contested})"))
            if can_afford(h, COSTS["devcard"]) and g.dev_deck > 0:
                moves.append(Move(prefix_label + "Buy Dev Card",
                                  prefix + [{"type": "build", "player": color, "item": "devcard"}],
                                  kind="devcard"))

        add_build_moves(hand, [], "")
        # Bank/port trades that unlock a build this turn.
        for name, cost, _ in self.build_goals(color):
            if can_afford(hand, cost):
                continue
            missing = [max(0, c - h) for c, h in zip(cost, hand)]
            if sum(missing) != 1:
                continue
            need_i = missing.index(1)
            for i, r in enumerate(RESOURCES):
                rate = g.trade_rate(color, r)
                if i == need_i or hand[i] - cost[i] < rate or g.bank[need_i] == 0:
                    continue
                h2 = list(hand)
                h2[i] -= rate
                h2[need_i] += 1
                trade = {"type": "bank_trade", "player": color, "give": r.value,
                         "get": RESOURCES[need_i].value, "times": 1}
                n_before = len(moves)
                add_build_moves(h2, [trade], f"Trade {rate} {r.value}->1 {RESOURCES[need_i].value}, ")
                moves[n_before:] = [m for m in moves[n_before:] if name.split()[0] in m.label.lower()
                                    or (name == "devcard" and "Dev" in m.label)]
        # Table-trade proposals: 1:1 offers that complete a goal within two cards.
        values = self.resource_values(color, hand)
        offered: set[tuple[int, int]] = set()
        for name, cost, _ in self.build_goals(color):
            missing = [max(0, c - h) for c, h in zip(cost, hand)]
            if not 0 < sum(missing) <= 2:
                continue
            for need_i in (i for i in range(NUM_RES) if missing[i]):
                surplus = [i for i in range(NUM_RES) if hand[i] - cost[i] > 0 and i != need_i]
                if not surplus:
                    continue
                give_i = min(surplus, key=lambda i: values[RESOURCES[i].value])
                if (give_i, need_i) in offered:
                    continue
                offered.add((give_i, need_i))
                give = tuple(1 if i == give_i else 0 for i in range(NUM_RES))
                get = tuple(1 if i == need_i else 0 for i in range(NUM_RES))
                if self.evaluate_trade(color, give, get) <= 0:
                    continue
                moves.append(Move(
                    f"Offer Trade: 1 {RESOURCES[give_i].value} for 1 {RESOURCES[need_i].value}",
                    # modelled as a swap with the bank so the heuristic/MC can price it
                    [{"type": "adjust", "player": color, "cards": give, "sign": -1},
                     {"type": "adjust", "player": color, "cards": get, "sign": 1}],
                    detail=f"toward {name}; partner must accept",
                    kind="offer", offer={"give": list(give), "get": list(get)}))
        # Knight before/after roll if the robber sits on us.
        knights = p.dev_known.get("knight", 0) if color == g.perspective else 0
        if knights:
            own_hexes = {h for n in p.buildings() for h in g.board.topology.node_hexes[n]}
            if g.board.robber_hex in own_hexes or p.knights_played + 1 >= 3:
                moves.append(Move("Play Knight", [{"type": "play_dev", "player": color, "card": "knight"}]
                                  + self._robber_followup(color)))
        moves.append(Move("End Turn (save cards)", [], detail=self._saving_hint(color, hand)))
        return moves

    def _saving_hint(self, color: str, hand: Sequence[int]) -> str:
        goals = self.build_goals(color)
        best = None
        for name, cost, w in goals:
            missing = [max(0, c - h) for c, h in zip(cost, hand)]
            k = sum(missing)
            if best is None or (k, -w) < (best[0], -best[2]):
                best = (k, name, w, missing)
        if not best:
            return ""
        return f"closest goal {best[1]}: need {fmt_hand(best[3])}" if best[0] else f"{best[1]} affordable"

    def _robber_followup(self, color: str, game: GameState | None = None) -> list[dict[str, Any]]:
        opts = self.robber_options(color, top=1, game=game)
        if not opts:
            return []
        o = opts[0]
        ev: dict[str, Any] = {"type": "robber", "hex": o.hex, "thief": color}
        if o.victim:
            ev["victim"] = o.victim
        return [ev]

    def _apply(self, moves: list[dict[str, Any]]) -> GameState:
        g2 = self.game.clone()
        g2.belief.max_states = 256
        for ev in moves:
            g2.apply_event(ev)
        return g2

    # -- recommendation -------------------------------------------------------
    def _setup_moves(self, color: str, kind: str) -> list[Move]:
        g = self.game
        topo = g.board.topology
        moves: list[Move] = []
        if kind == "settlement":
            second = len(g.players[color].buildings()) == 1
            for ns in self.settlement_scores(color, require_road=False, top=8):
                start = ""
                if second:
                    cards = [g.board.tiles[h].resource.value for h in topo.node_hexes[ns.node]
                             if g.board.tiles[h].resource]
                    start = f"; starting cards {', '.join(cards) or 'none'}"
                moves.append(Move(f"Place Settlement at Node {ns.node}",
                                  [{"type": "build", "player": color, "item": "settlement", "target": ns.node}],
                                  heuristic=ns.value, detail=str(ns).split(": ", 1)[1] + start, kind="setup"))
            return moves
        p = g.players[color]
        bare = [n for n in p.buildings() if not any(e in p.roads for e in topo.node_edges[n])]
        for anchor in bare:
            for e in topo.node_edges[anchor]:
                if g.edge_owner(e) is not None:
                    continue
                a, b = topo.edge_nodes[e]
                far = b if a == anchor else a
                spots = [w for w in topo.node_neighbors[far]
                         if w != anchor and g.can_place_settlement(color, w, require_road=False)]
                scored = sorted((self.node_score(w, color) for w in spots), key=lambda x: -x.value)
                value = scored[0].value if scored else 0.0
                detail = f"toward node {scored[0].node} ({scored[0].pips} pips)" if scored else "dead end"
                moves.append(Move(f"Place Road on Edge {e} ({anchor}-{far})",
                                  [{"type": "build", "player": color, "item": "road", "target": e}],
                                  heuristic=value, detail=detail, kind="setup"))
        return sorted(moves, key=lambda m: -m.heuristic)

    def recommend(self, color: str | None = None, *, top_k: int = 3) -> Recommendation:
        g = self.game
        color = color or g.current_color
        notes: list[str] = []
        if g.phase == "setup":
            nxt = g.setup_next()
            own = g.pending_setup(color) if color else None
            if own:                       # any order: advise the player you selected
                nxt = (color, own)
            if nxt:
                color, kind = nxt
                moves = self._setup_moves(color, kind)
                win = self.softmax_win()
                return Recommendation(color, "setup", moves[:top_k], self.position_score(color), win[color], win,
                                      [f"setup: {color} places a {kind}; ranked by placement value "
                                       "(win % starts once every piece is down)"])
        pre_roll = g.phase == "main" and g.last_roll is None and color == g.current_color
        if pre_roll:
            hand = self._acting_hand(color)
            moves = [Move("Roll the dice", [], kind="roll",
                          detail=f"then aim for: {self._saving_hint(color, hand)}")]
            p = g.players[color]
            if color == g.perspective and p.dev_known.get("knight", 0):
                moves.append(Move("Play Knight before rolling",
                                  [{"type": "play_dev", "player": color, "card": "knight"}]
                                  + self._robber_followup(color)))
            notes.append("before the roll: enter the number rolled (only a knight may be played first)")
            win_probs = self.softmax_win()
            if self.rollouts > 0:
                for m in moves:
                    after = self._apply(m.events) if m.events else g
                    res = self.simulate(after, color, start_next_turn=False)
                    m.win_prob = res[color]
                    if m is moves[0]:
                        win_probs = res
                moves.sort(key=lambda m: -(m.win_prob or 0))
            return Recommendation(color, g.phase, moves[:top_k], self.position_score(color),
                                  win_probs[color], win_probs, notes)
        if g.phase in ("robber",) and color == g.current_color:
            moves = []
            for o in self.robber_options(color, top=max(top_k, 4)):
                ev: dict[str, Any] = {"type": "robber", "hex": o.hex, "thief": color}
                label = f"Move Robber to {g.board.hex_label(o.hex)}"
                if o.victim:
                    ev["victim"] = o.victim
                    label += f", steal from {o.victim}"
                moves.append(Move(label, [ev], heuristic=o.score, detail=o.detail))
        elif g.phase == "discard":
            moves = []
            for c, n in g.pending_discards.items():
                cards = self.suggest_discard(c, n)
                moves.append(Move(f"{c} discard {fmt_hand(cards)}",
                                  [{"type": "discard", "player": c, "cards": tuple(cards)}],
                                  detail=f"keeps best progress toward {self._saving_hint(c, self._acting_hand(c))}"))
            notes.append("discard suggestions only; robber advice follows after discards")
            ps = self.position_score(color)
            win = self.softmax_win()
            return Recommendation(color, g.phase, moves[:top_k], ps, win[color], win, notes)
        else:
            if color != g.current_color:
                notes.append(f"advising {color} as if it were their turn")
            moves = self._candidates(color)
        # Heuristic pass.
        base = self.position_score(color)
        for m in moves:
            if m.events and g.phase == "main":
                try:
                    after = self._apply(m.events)
                    m.heuristic = self.position_score(color, after) - base
                except Exception as exc:  # illegal combos are discarded
                    m.heuristic = float("-inf")
                    m.detail = f"illegal: {exc}"
        moves = [m for m in moves if m.heuristic != float("-inf")]
        moves.sort(key=lambda m: -m.heuristic)
        # Monte Carlo pass on the strongest candidates (+ always the pass move).
        win_probs: dict[str, float] = {}
        if self.rollouts > 0 and moves:
            shortlist = moves[: self.mc_top_k]
            if g.phase == "main":
                pass_move = next((m for m in moves if not m.events), None)
                if pass_move and pass_move not in shortlist:
                    shortlist.append(pass_move)
            table: dict[int, dict[str, float]] = {}
            for m in shortlist:
                after = self._apply(m.events) if m.events else self.game
                # the dice for this turn are already rolled: rollouts start with the next player
                table[id(m)] = self.simulate(after, color, start_next_turn=True)
                m.win_prob = table[id(m)][color]
            # A table offer only happens if someone accepts: score it as a coin flip
            # between the accepted outcome and simply passing.
            pass_move = next((m for m in shortlist if not m.events and m.kind != "roll"), None)
            if pass_move is not None:
                for m in shortlist:
                    if m.kind == "offer" and m.win_prob is not None:
                        accepted = m.win_prob
                        m.win_prob = 0.5 * accepted + 0.5 * pass_move.win_prob  # type: ignore[operator]
                        m.detail += f" · {accepted * 100:.0f}% if accepted"
            # Rank by MC win rate; the heuristic may shift a move by at most half a
            # standard error, so it only decides between statistically tied moves.
            se = 0.5 * math.sqrt(0.25 / self.rollouts)
            hs = [m.heuristic for m in shortlist]
            lo, span = min(hs), (max(hs) - min(hs)) or 1.0
            for m in shortlist:
                m.rank_score = m.win_prob + se * (m.heuristic - lo) / span  # type: ignore[operator]
            moves.sort(key=lambda m: -(m.rank_score if m.win_prob is not None else -1.0 + 0.001 * m.heuristic))
            win_probs = table[id(moves[0])]
        else:
            win_probs = self.softmax_win()
        ps = self.position_score(color)
        return Recommendation(color, g.phase, moves[:top_k], ps, win_probs.get(color, 0.0), win_probs, notes)

    def suggest_discard(self, color: str, n: int) -> list[int]:
        hand = self._acting_hand(color)
        out = [0] * NUM_RES
        for _ in range(n):
            best_i, best_v = None, -1e9
            for i in range(NUM_RES):
                if hand[i] == 0:
                    continue
                h2 = list(hand)
                h2[i] -= 1
                v = self.hand_potential(color, h2)
                if v > best_v:
                    best_i, best_v = i, v
            if best_i is None:
                break
            hand[best_i] -= 1
            out[best_i] += 1
        return out

    # -- Monte Carlo ------------------------------------------------------------
    def simulate(self, game: GameState, color: str, *, start_next_turn: bool = True,
                 n: int | None = None) -> dict[str, float]:
        n = n or self.rollouts
        wins = {c: 0.0 for c in game.order}
        sim_static = _SimStatic(game.board)
        for i in range(n):
            if self.should_stop is not None and i % 8 == 0 and self.should_stop():
                raise AnalysisCancelled()
            rng = random.Random(self.seed * 100003 + i)  # common random numbers
            sim = _Sim(sim_static, game, rng)
            start = (game.current + 1) % len(game.order) if start_next_turn else game.current
            for c, w in sim.run(start, self.max_rounds).items():
                wins[game.order[c]] += w
        return {c: w / n for c, w in wins.items()}


# ---------------------------------------------------------------------------
# Fast greedy self-play simulator
# ---------------------------------------------------------------------------


class _SimStatic:
    def __init__(self, board: Board):
        topo = board.topology
        self.topo = topo
        self.n_nodes = topo.num_nodes
        self.roll_hexes: dict[int, list[tuple[int, int, tuple[int, ...]]]] = {}
        for t in board.tiles:
            if t.number and t.resource is not None:
                self.roll_hexes.setdefault(t.number, []).append(
                    (t.id, RES_INDEX[t.resource], topo.hex_nodes[t.id]))
        self.node_pips = [sum(board.tiles[h].pips for h in topo.node_hexes[n]) for n in range(topo.num_nodes)]
        self.hex_pips = [t.pips for t in board.tiles]
        self.node_port: dict[int, int | None] = {
            n: (RES_INDEX[p.resource] if p.resource else None) for n, p in board.node_ports.items()}


class _Sim:
    __slots__ = ("s", "rng", "np", "hands", "owner", "level", "roads", "edge_owner", "vp_hidden",
                 "knights", "knights_ready", "deck", "bank", "robber", "lr_owner", "la_owner",
                 "lr_len", "rates", "n_sett", "n_city", "n_road")

    def __init__(self, st: _SimStatic, game: GameState, rng: random.Random):
        self.s = st
        self.rng = rng
        order = game.order
        self.np = len(order)
        joint = game.belief.sample(rng)
        self.hands = [list(h) for h in joint]
        self.owner = [-1] * st.n_nodes
        self.level = [0] * st.n_nodes
        self.edge_owner = [-1] * st.topo.num_edges
        self.roads: list[set[int]] = [set() for _ in order]
        self.n_sett = [0] * self.np
        self.n_city = [0] * self.np
        self.n_road = [0] * self.np
        for i, c in enumerate(order):
            p = game.players[c]
            for n in p.settlements:
                self.owner[n], self.level[n] = i, 1
            for n in p.cities:
                self.owner[n], self.level[n] = i, 2
            for e in p.roads:
                self.edge_owner[e] = i
                self.roads[i].add(e)
            self.n_sett[i], self.n_city[i], self.n_road[i] = len(p.settlements), len(p.cities), len(p.roads)
        # Unseen dev cards: deck + opponents' hidden cards drawn from the same pool.
        pool: list[str] = []
        for card, k in DEV_DECK.items():
            seen = sum(q.dev_played.get(card, 0) for q in game.players.values())
            if card == "knight":
                seen = max(seen, sum(q.knights_played for q in game.players.values()))
            if card == "vp":
                seen = max(seen, sum(q.vp_cards for q in game.players.values()))
            if game.perspective:
                seen += game.players[game.perspective].dev_known.get(card, 0)
            pool.extend([card] * max(0, k - seen))
        rng.shuffle(pool)
        self.vp_hidden = [0] * self.np
        self.knights = [game.players[c].knights_played for c in order]
        self.knights_ready = [0] * self.np
        for i, c in enumerate(order):
            p = game.players[c]
            cards = [card for card, k in p.dev_known.items() for _ in range(k)] if c == game.perspective else []
            cards += [pool.pop() for _ in range(max(0, p.dev_hidden - len(cards))) if pool]
            for card in cards:
                if card == "vp":
                    self.vp_hidden[i] += 1
                elif card == "knight":
                    self.knights_ready[i] += 1
            self.vp_hidden[i] += p.vp_cards
        self.deck = pool[: game.dev_deck]
        self.bank = list(game.bank)
        self.robber = game.board.robber_hex
        self.lr_owner = order.index(game.longest_road_owner) if game.longest_road_owner else -1
        self.la_owner = order.index(game.largest_army_owner) if game.largest_army_owner else -1
        self.lr_len = [game.longest_road(c) for c in order]
        self.rates = [[4] * NUM_RES for _ in order]
        for n in range(st.n_nodes):
            if self.owner[n] >= 0:
                self._port(self.owner[n], n)

    # -- helpers -----------------------------------------------------------------
    def _port(self, p: int, n: int) -> None:
        if n in self.s.node_port:
            r = self.s.node_port[n]
            if r is None:
                self.rates[p] = [min(x, 3) for x in self.rates[p]]
            else:
                self.rates[p][r] = 2

    def vp(self, p: int) -> int:
        return (self.n_sett[p] + 2 * self.n_city[p] + self.vp_hidden[p]
                + (2 if self.lr_owner == p else 0) + (2 if self.la_owner == p else 0))

    def _spot_ok(self, n: int) -> bool:
        if self.owner[n] >= 0:
            return False
        return all(self.owner[nb] < 0 for nb in self.s.topo.node_neighbors[n])

    def _connected(self, p: int, n: int) -> bool:
        return any(self.edge_owner[e] == p for e in self.s.topo.node_edges[n])

    def _pay(self, p: int, cost: Sequence[int]) -> None:
        h = self.hands[p]
        for i, c in enumerate(cost):
            h[i] -= c
            self.bank[i] += c

    # -- turn ----------------------------------------------------------------------
    def run(self, start: int, max_rounds: int) -> dict[int, float]:
        p = start
        for _ in range(max_rounds * self.np):
            if self.turn(p):
                return {p: 1.0}
            p = (p + 1) % self.np
        vps = [self.vp(i) for i in range(self.np)]
        top = max(vps)
        winners = [i for i, v in enumerate(vps) if v == top]
        return {i: 1.0 / len(winners) for i in winners}

    def turn(self, p: int) -> bool:
        rng = self.rng
        st = self.s
        # Knight before rolling if robber hurts us, or to grab largest army.
        if self.knights_ready[p] and (self._robber_hits(p) or (self.knights[p] >= 2 and self.la_owner != p)):
            self._play_knight(p)
        roll = rng.randint(1, 6) + rng.randint(1, 6)
        if roll == 7:
            for i in range(self.np):
                h = self.hands[i]
                total = sum(h)
                if total > 7:
                    for _ in range(total // 2):
                        j = max(range(NUM_RES), key=lambda k: h[k])
                        h[j] -= 1
                        self.bank[j] += 1
            self._move_robber(p)
        else:
            demand: dict[int, list[int]] = {}
            for h, r, nodes in st.roll_hexes.get(roll, ()):
                if h == self.robber:
                    continue
                for n in nodes:
                    o = self.owner[n]
                    if o >= 0:
                        demand.setdefault(r, [0] * self.np)[o] += self.level[n]
            for r, per in demand.items():
                tot = sum(per)
                if tot <= self.bank[r]:
                    for i, k in enumerate(per):
                        self.hands[i][r] += k
                    self.bank[r] -= tot
        # Build loop (greedy).
        for _ in range(8):
            if not self._act(p):
                break
            if self.vp(p) >= WIN_VP:
                return True
        if self.knights_ready[p] == 0:
            pass
        return self.vp(p) >= WIN_VP

    def _robber_hits(self, p: int) -> bool:
        return any(self.owner[n] == p for n in self.s.topo.hex_nodes[self.robber])

    def _move_robber(self, p: int) -> None:
        st = self.s
        vps = [self.vp(i) for i in range(self.np)]
        best_h, best_v = self.robber, -1e9
        for h in range(len(st.hex_pips)):
            if h == self.robber:
                continue
            v = 0.0
            for n in st.topo.hex_nodes[h]:
                o = self.owner[n]
                if o == p:
                    v -= 100
                elif o >= 0:
                    v += st.hex_pips[h] * self.level[n] * (1 + 0.3 * vps[o])
            if v > best_v:
                best_h, best_v = h, v
        self.robber = best_h
        victims = {self.owner[n] for n in st.topo.hex_nodes[best_h] if self.owner[n] >= 0 and self.owner[n] != p}
        victims = {v for v in victims if sum(self.hands[v]) > 0}
        if victims:
            v = max(victims, key=lambda i: vps[i])
            h = self.hands[v]
            j = self.rng.choices(range(NUM_RES), weights=h, k=1)[0]
            h[j] -= 1
            self.hands[p][j] += 1

    def _play_knight(self, p: int) -> None:
        self.knights_ready[p] -= 1
        self.knights[p] += 1
        if self.knights[p] >= 3 and (self.la_owner < 0 or self.knights[p] > self.knights[self.la_owner]):
            self.la_owner = p
        self._move_robber(p)

    def _act(self, p: int) -> bool:
        h = self.hands[p]
        st = self.s
        topo = st.topo
        # City on best settlement.
        if self.n_city[p] < 4 and can_afford(h, COSTS["city"]):
            setts = [n for n in range(st.n_nodes) if self.owner[n] == p and self.level[n] == 1]
            if setts:
                n = max(setts, key=lambda x: st.node_pips[x])
                self._pay(p, COSTS["city"])
                self.level[n] = 2
                self.n_city[p] += 1
                self.n_sett[p] -= 1
                return True
        spots = [n for n in range(st.n_nodes) if self._connected(p, n) and self._spot_ok(n)] \
            if self.n_sett[p] < 5 else []
        if spots and can_afford(h, COSTS["settlement"]):
            n = max(spots, key=lambda x: st.node_pips[x])
            self._pay(p, COSTS["settlement"])
            self.owner[n], self.level[n] = p, 1
            self.n_sett[p] += 1
            self._port(p, n)
            return True
        if not spots and self.n_road[p] < 15 and self.n_sett[p] < 5 and can_afford(h, COSTS["road"]):
            e = self._best_road(p)
            if e is not None:
                self._pay(p, COSTS["road"])
                self._place_road(p, e)
                return True
        if self.deck and can_afford(h, COSTS["devcard"]):
            self._pay(p, COSTS["devcard"])
            card = self.deck.pop()
            if card == "vp":
                self.vp_hidden[p] += 1
            elif card == "knight":
                self.knights_ready[p] += 1
            elif card == "road_building":
                for _ in range(2):
                    e = self._best_road(p)
                    if e is not None and self.n_road[p] < 15:
                        self._place_road(p, e)
            elif card == "year_of_plenty":
                for _ in range(2):
                    j = min(range(NUM_RES), key=lambda k: h[k])
                    if self.bank[j]:
                        h[j] += 1
                        self.bank[j] -= 1
            elif card == "monopoly":
                j = max(range(NUM_RES), key=lambda k: sum(self.hands[q][k] for q in range(self.np) if q != p))
                for q in range(self.np):
                    if q != p:
                        h[j] += self.hands[q][j]
                        self.hands[q][j] = 0
            return True
        # Maritime trade toward the first goal that is at most two cards short.
        goals = (COSTS["city"] if self.n_city[p] < 4 and self.n_sett[p] else None,
                 COSTS["settlement"] if spots else None,
                 COSTS["road"] if not spots and self.n_sett[p] < 5 and self.n_road[p] < 15 else None,
                 COSTS["devcard"] if self.deck else None)
        for cost in goals:
            if cost is None:
                continue
            missing = [max(0, c - x) for c, x in zip(cost, h)]
            if not 0 < sum(missing) <= 2:
                continue
            need = next(i for i in range(NUM_RES) if missing[i])
            if self.bank[need] == 0:
                continue
            for i in range(NUM_RES):
                rate = self.rates[p][i]
                if i != need and h[i] - cost[i] >= rate:
                    h[i] -= rate
                    self.bank[i] += rate
                    h[need] += 1
                    self.bank[need] -= 1
                    return True
        return False

    def _best_road(self, p: int) -> int | None:
        st = self.s
        topo = st.topo
        best, best_v = None, 0.0
        for e in range(topo.num_edges):
            if self.edge_owner[e] >= 0:
                continue
            a, b = topo.edge_nodes[e]
            for u, v in ((a, b), (b, a)):
                if not (self.owner[u] == p or (self.owner[u] < 0 and self._connected(p, u))):
                    continue
                if self.owner[v] >= 0 and self.owner[v] != p:
                    continue
                if self._spot_ok(v):
                    val = st.node_pips[v] + 3.0
                else:
                    val = max((st.node_pips[w] for w in topo.node_neighbors[v] if self._spot_ok(w)), default=0) * 0.6
                val += self.rng.random() * 0.1
                if val > best_v:
                    best, best_v = e, val
        return best

    def _place_road(self, p: int, e: int) -> None:
        self.edge_owner[e] = p
        self.roads[p].add(e)
        self.n_road[p] += 1
        if self.n_road[p] >= 5:
            self.lr_len[p] = self._longest(p)
            cur = self.lr_owner
            if self.lr_len[p] >= 5 and (cur < 0 or self.lr_len[p] > self.lr_len[cur]):
                self.lr_owner = p

    def _longest(self, p: int) -> int:
        topo = self.s.topo
        adj: dict[int, list[tuple[int, int]]] = {}
        for e in self.roads[p]:
            a, b = topo.edge_nodes[e]
            adj.setdefault(a, []).append((e, b))
            adj.setdefault(b, []).append((e, a))
        best = 0

        def dfs(node: int, used: set[int], length: int) -> None:
            nonlocal best
            if length > best:
                best = length
            if length and self.owner[node] >= 0 and self.owner[node] != p:
                return
            for e, nb in adj[node]:
                if e not in used:
                    used.add(e)
                    dfs(nb, used, length + 1)
                    used.discard(e)

        for n in adj:
            dfs(n, set(), 0)
        return best
