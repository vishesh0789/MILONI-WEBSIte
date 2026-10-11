"""Lean decision engine: bitboards + linear evaluation + pruned expectimax.

Built for small machines (8 GB laptops): no tree search with millions of nodes, no
rollouts.  A full recommendation takes a few milliseconds and well under 10 MB.

Representation
--------------
* The 54 corners and 72 edges are bits in Python ints (arbitrary width, one machine
  word-ish).  Per board we precompute, once:
  ``nb[n]`` neighbour-corner mask, ``ne[n]`` incident-edge mask, ``en[e]`` the two corners
  of an edge, and per robber position the pip vector of every corner.
* A player is three masks (settlements, cities, roads) plus a 5-tuple hand.
  Legal settlement spots are ``ALL & ~(occupied | nb(occupied)) & network`` - a few
  integer operations instead of 54 Python loops.

Evaluation (VP-equivalent units, linear in features)
---------------------------------------------------
``VP``  + 4.0 x expected production (Σ pips/36 × scarcity weight)
        + 0.12 x resource diversity + port synergy (2:1 ports scale with that resource's
          production, 3:1 a flat bonus)
        + expansion potential (best open spot within 0/1/2 roads, discounted 0.85^k)
        + 0.08 x longest-road length (until it is held) + longest road / army in VP
        + dev cards × expected value (VP odds from the unseen deck, knights toward army)
        + hand progress toward the next build − discard risk above the room's limit
        + VP-race term: being closer to the target than the field is worth extra.

Search
------
Candidates are pruned to the top K per kind (cities, settlements, roads, dev card, the
bank trades that unlock one of those, pass).  Each is valued by expectimax:
``V(move) = Σ_roll P(roll) · max_f eval(after move, + expected income from the
three opponent rolls, + my roll's income, then follow-up build f)``.  A 7 applies the
discard rule.  Evaluations are cached by (player, masks, hand, robber).
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from .board import PIPS, RES_INDEX, RESOURCES, Board
from .state_manager import COSTS, DEV_DECK, NUM_RES, PIECE_LIMITS, GameState

ROLL_P = {n: PIPS[n] / 36.0 for n in range(2, 13)}
K_CITY, K_SETTLE, K_ROAD = 3, 4, 3


def bits(mask: int) -> Iterable[int]:
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def popcount(mask: int) -> int:
    return bin(mask).count("1")


class BoardBits:
    """Per-board constants as integer masks (built once, shared by every evaluation)."""

    def __init__(self, board: Board):
        topo = board.topology
        self.n_nodes, self.n_edges = topo.num_nodes, topo.num_edges
        self.all_nodes = (1 << self.n_nodes) - 1
        self.nb = [sum(1 << m for m in topo.node_neighbors[n]) for n in range(self.n_nodes)]
        self.ne = [sum(1 << e for e in topo.node_edges[n]) for n in range(self.n_nodes)]
        self.en = [(1 << a) | (1 << b) for a, b in topo.edge_nodes]
        self.edge_ends = list(topo.edge_nodes)
        self.port = [-1] * self.n_nodes       # -1 none, 5 = 3:1, 0..4 = 2:1 resource
        for n, p in board.node_ports.items():
            self.port[n] = 5 if p.resource is None else RES_INDEX[p.resource]
        self.tiles = [(RES_INDEX[t.resource] if t.resource else -1, t.number or 0, t.pips) for t in board.tiles]
        self.node_hexes = topo.node_hexes
        totals = [0] * NUM_RES
        for r, _, pips in self.tiles:
            if r >= 0:
                totals[r] += pips
        mean = sum(totals) / NUM_RES
        raw = [min(1.6, max(0.7, math.sqrt(mean / max(t, 1)))) for t in totals]
        self.weight = [w * NUM_RES / sum(raw) for w in raw]
        self._prod: dict[int, list[tuple[int, ...]]] = {}
        self._income: dict[int, dict[int, list[tuple[int, ...]]]] = {}

    def node_prod(self, robber: int) -> list[tuple[int, ...]]:
        """Pip vector per corner with the robber's tile switched off."""
        if robber not in self._prod:
            out = []
            for n in range(self.n_nodes):
                v = [0] * NUM_RES
                for h in self.node_hexes[n]:
                    r, _, pips = self.tiles[h]
                    if r >= 0 and h != robber:
                        v[r] += pips
                out.append(tuple(v))
            self._prod[robber] = out
        return self._prod[robber]

    def node_income(self, robber: int) -> dict[int, list[tuple[int, ...]]]:
        """Cards per corner for each dice total (settlement multiplier 1)."""
        if robber not in self._income:
            table: dict[int, list[tuple[int, ...]]] = {}
            for roll in ROLL_P:
                rows = []
                for n in range(self.n_nodes):
                    v = [0] * NUM_RES
                    for h in self.node_hexes[n]:
                        r, num, _ = self.tiles[h]
                        if r >= 0 and num == roll and h != robber:
                            v[r] += 1
                    rows.append(tuple(v))
                table[roll] = rows
            self._income[robber] = table
        return self._income[robber]

    def spot_value(self, n: int, robber: int, own: tuple[float, ...]) -> float:
        prod = self.node_prod(robber)[n]
        v = sum(w * x for w, x in zip(self.weight, prod))
        v += sum(1.2 if own[r] == 0 else 0.4 for r in range(NUM_RES) if prod[r])
        port = self.port[n]
        if port == 5:
            v += 1.5
        elif port >= 0:
            v += 0.5 + 0.35 * (own[port] + prod[port])
        return v


class Pos:
    """One player's position: masks + hand (+ the few counters the evaluation needs)."""

    __slots__ = ("sett", "city", "road", "hand", "dev", "knights", "vp_bonus", "hidden_vp")

    def __init__(self, sett: int, city: int, road: int, hand: tuple[float, ...], dev: int,
                 knights: int, vp_bonus: int, hidden_vp: float):
        self.sett, self.city, self.road, self.hand = sett, city, road, hand
        self.dev, self.knights, self.vp_bonus, self.hidden_vp = dev, knights, vp_bonus, hidden_vp

    def copy(self, **kw: Any) -> "Pos":
        p = Pos(self.sett, self.city, self.road, self.hand, self.dev, self.knights, self.vp_bonus, self.hidden_vp)
        for k, v in kw.items():
            setattr(p, k, v)
        return p

    def key(self) -> tuple:
        return (self.sett, self.city, self.road, tuple(round(x, 2) for x in self.hand), self.dev, self.knights)


_BITS_CACHE: dict[tuple, BoardBits] = {}


def board_bits(board: Board) -> BoardBits:
    key = (tuple((t.terrain.value, t.number) for t in board.tiles), tuple((p.type, p.nodes) for p in board.ports))
    if key not in _BITS_CACHE:
        if len(_BITS_CACHE) > 8:
            _BITS_CACHE.clear()
        _BITS_CACHE[key] = BoardBits(board)
    return _BITS_CACHE[key]


class FastEngine:
    """Linear evaluation + pruned expectimax over a bitboard snapshot of a GameState."""

    def __init__(self, game: GameState):
        self.g = game
        self.bb = board_bits(game.board)
        self.robber = game.board.robber_hex
        self.target = int(game.rules.get("vp_target", 10))
        self.limit = int(game.rules.get("discard_limit", 7))
        self.order = list(game.order)
        joint = game.belief.most_likely()
        self.pos: dict[str, Pos] = {}
        for i, c in enumerate(game.order):
            p = game.players[c]
            sett = sum(1 << n for n in p.settlements)
            city = sum(1 << n for n in p.cities)
            road = sum(1 << e for e in p.roads)
            bonus = p.vp_cards + (2 if game.longest_road_owner == c else 0) + (2 if game.largest_army_owner == c else 0)
            self.pos[c] = Pos(sett, city, road, tuple(float(x) for x in joint[i]), p.dev_hidden,
                              p.knights_played, bonus, game.expected_hidden_vp(c))
        self.dev_deck = game.dev_deck
        self.cache: dict[tuple, float] = {}
        self.evals = 0
        self.temperature = (0.35, 1.6)   # (base, growth with game progress); see calibrate() in tests

    # -- board queries (all mask arithmetic) -------------------------------------
    def occupied(self) -> int:
        m = 0
        for p in self.pos.values():
            m |= p.sett | p.city
        return m

    def all_roads(self) -> int:
        m = 0
        for p in self.pos.values():
            m |= p.road
        return m

    def neighbours(self, mask: int) -> int:
        out = 0
        for n in bits(mask):
            out |= self.bb.nb[n]
        return out

    def road_nodes(self, road: int) -> int:
        out = 0
        for e in bits(road):
            out |= self.bb.en[e]
        return out

    def free_spots(self, occ: int) -> int:
        return self.bb.all_nodes & ~(occ | self.neighbours(occ))

    def network(self, c: str, pos: Pos | None = None, occ: int | None = None) -> int:
        """Corners this player's roads/buildings reach (not through opponents' buildings)."""
        pos = pos or self.pos[c]
        occ = self.occupied() if occ is None else occ
        own = pos.sett | pos.city
        return (self.road_nodes(pos.road) | own) & ~(occ & ~own)

    def legal_settlements(self, c: str, pos: Pos, occ: int) -> int:
        return self.free_spots(occ) & self.road_nodes(pos.road)

    def road_frontier(self, c: str, pos: Pos, occ: int, roads: int) -> int:
        out = 0
        for n in bits(self.network(c, pos, occ)):
            out |= self.bb.ne[n]
        return out & ~roads

    def production(self, pos: Pos) -> list[float]:
        prod = [0.0] * NUM_RES
        table = self.bb.node_prod(self.robber)
        for n in bits(pos.sett):
            for r, x in enumerate(table[n]):
                prod[r] += x
        for n in bits(pos.city):
            for r, x in enumerate(table[n]):
                prod[r] += 2 * x
        return prod

    def income(self, pos: Pos, roll: int) -> list[float]:
        rows = self.bb.node_income(self.robber)[roll]
        out = [0.0] * NUM_RES
        for n in bits(pos.sett):
            for r, x in enumerate(rows[n]):
                out[r] += x
        for n in bits(pos.city):
            for r, x in enumerate(rows[n]):
                out[r] += 2 * x
        return out

    def longest_road(self, c: str, pos: Pos, occ: int) -> int:
        key = ("lr", pos.road, occ & ~(pos.sett | pos.city))
        if key in self.cache:
            return int(self.cache[key])
        blocked = occ & ~(pos.sett | pos.city)
        adj: dict[int, list[tuple[int, int]]] = {}
        for e in bits(pos.road):
            a, b = self.bb.edge_ends[e]
            adj.setdefault(a, []).append((e, b))
            adj.setdefault(b, []).append((e, a))
        best = 0

        def dfs(node: int, used: int, length: int) -> None:
            nonlocal best
            best = max(best, length)
            if length and blocked >> node & 1:
                return
            for e, nxt in adj.get(node, ()):
                if not used >> e & 1:
                    dfs(nxt, used | 1 << e, length + 1)

        for start in adj:
            dfs(start, 0, 0)
        self.cache[key] = best
        return best

    # -- evaluation ------------------------------------------------------------------
    def evaluate(self, c: str, pos: Pos | None = None, occ: int | None = None, roads: int | None = None) -> float:
        pos = pos or self.pos[c]
        occ = self.occupied() if occ is None else occ
        key = ("ev", c, pos.key(), occ, self.robber)
        if key in self.cache:
            return self.cache[key]
        self.evals += 1
        bb = self.bb
        prod = self.production(pos)
        vp = popcount(pos.sett) + 2 * popcount(pos.city) + pos.vp_bonus + pos.hidden_vp
        f_prod = sum(w * x for w, x in zip(bb.weight, prod)) / 36.0
        f_div = sum(1 for x in prod if x > 0)
        f_port = 0.0
        for n in bits(pos.sett | pos.city):
            port = bb.port[n]
            if port == 5:
                f_port += 0.25
            elif port >= 0:
                f_port += 0.06 * prod[port]
        # expansion: best open spot within 0, 1 or 2 roads of the network
        f_exp = 0.0
        if popcount(pos.sett) < PIECE_LIMITS["settlement"]:
            free = self.free_spots(occ)
            reach = self.network(c, pos, occ)
            passable = bb.all_nodes & ~(occ & ~(pos.sett | pos.city))
            for k in range(3):
                spots = free & reach
                if spots:
                    best = max(bb.spot_value(n, self.robber, tuple(prod)) for n in bits(spots))
                    f_exp = max(f_exp, best * (0.85 ** k) * (1.0 if k == 0 else 0.8))
                reach = (reach | self.neighbours(reach)) & passable
        holds_lr = self.g.longest_road_owner == c
        f_lr = 0.0 if holds_lr else 0.08 * self.longest_road(c, pos, occ)
        unseen_vp = DEV_DECK["vp"] - sum(q.vp_cards for q in self.g.players.values())
        unseen = max(1, self.dev_deck + sum(q.dev_hidden for q in self.g.players.values()))
        f_dev = pos.dev * (0.3 + unseen_vp / unseen) + (0.12 * pos.knights if self.g.largest_army_owner != c else 0)
        f_hand = self.hand_value(c, pos)
        race = max(0.0, vp - (self.target - 4)) * 0.35          # the last few VP are worth more
        value = vp + 4.0 * f_prod + 0.12 * f_div + f_port + 0.06 * f_exp + f_lr + f_dev + f_hand + race
        self.cache[key] = value
        return value

    def goals(self, pos: Pos) -> list[tuple[tuple[int, ...], float]]:
        out = []
        if pos.sett and popcount(pos.city) < PIECE_LIMITS["city"]:
            out.append((COSTS["city"], 1.3))
        if popcount(pos.sett) < PIECE_LIMITS["settlement"]:
            out.append((COSTS["settlement"], 1.1))
            out.append((COSTS["road"], 0.6))
        if self.dev_deck:
            out.append((COSTS["devcard"], 0.6))
        return out

    def hand_value(self, c: str, pos: Pos) -> float:
        hand = pos.hand
        prog = sorted((w * (sum(min(h, k) for h, k in zip(hand, cost)) / sum(cost)) ** 2
                       for cost, w in self.goals(pos)), reverse=True)
        goal = (prog[0] + 0.25 * sum(prog[1:])) if prog else 0.0
        total = sum(hand)
        risk = (1 / 6) * (total / 2) * 0.12 if total > self.limit else 0.0
        return 0.5 * goal + 0.04 * total - risk

    # -- moves ---------------------------------------------------------------------------
    @staticmethod
    def afford(hand: tuple[float, ...], cost: tuple[int, ...]) -> bool:
        return all(h + 1e-9 >= k for h, k in zip(hand, cost))

    @staticmethod
    def pay(hand: tuple[float, ...], cost: tuple[int, ...]) -> tuple[float, ...]:
        return tuple(h - k for h, k in zip(hand, cost))

    def moves(self, c: str, pos: Pos, occ: int, roads: int, *, trades: bool = True) -> list[tuple[str, list[dict], Pos]]:
        """(label, events, position after) for the pruned candidate set."""
        bb, out = self.bb, []
        prod = tuple(self.production(pos))
        hand = pos.hand

        def builds(h: tuple[float, ...], prefix: list[dict], tag: str) -> None:
            if self.afford(h, COSTS["city"]) and popcount(pos.city) < PIECE_LIMITS["city"]:
                table = bb.node_prod(self.robber)
                for n in sorted(bits(pos.sett), key=lambda n: -sum(table[n]))[:K_CITY]:
                    out.append((f"{tag}Build City at Node {n}",
                                prefix + [{"type": "build", "player": c, "item": "city", "target": n}],
                                pos.copy(sett=pos.sett & ~(1 << n), city=pos.city | 1 << n,
                                         hand=self.pay(h, COSTS["city"]))))
            if self.afford(h, COSTS["settlement"]) and popcount(pos.sett) < PIECE_LIMITS["settlement"]:
                spots = self.legal_settlements(c, pos, occ)
                for n in sorted(bits(spots), key=lambda n: -bb.spot_value(n, self.robber, prod))[:K_SETTLE]:
                    out.append((f"{tag}Build Settlement at Node {n}",
                                prefix + [{"type": "build", "player": c, "item": "settlement", "target": n}],
                                pos.copy(sett=pos.sett | 1 << n, hand=self.pay(h, COSTS["settlement"]))))
            if self.afford(h, COSTS["road"]) and popcount(pos.road) < PIECE_LIMITS["road"]:
                free = self.free_spots(occ)
                scored = []
                for e in bits(self.road_frontier(c, pos, occ, roads)):
                    far = self.bb.en[e] & ~self.network(c, pos, occ)
                    look = far | self.neighbours(far)
                    cand = free & look
                    val = max((bb.spot_value(n, self.robber, prod) * (1.0 if far >> n & 1 else 0.8)
                               for n in bits(cand)), default=0.0)
                    scored.append((val, e))
                for val, e in sorted(scored, reverse=True)[:K_ROAD]:
                    a, b = bb.edge_ends[e]
                    out.append((f"{tag}Build Road on Edge {e} ({a}-{b})",
                                prefix + [{"type": "build", "player": c, "item": "road", "target": e}],
                                pos.copy(road=pos.road | 1 << e, hand=self.pay(h, COSTS["road"]))))
            if self.afford(h, COSTS["devcard"]) and self.dev_deck:
                out.append((f"{tag}Buy Dev Card", prefix + [{"type": "build", "player": c, "item": "devcard"}],
                            pos.copy(dev=pos.dev + 1, hand=self.pay(h, COSTS["devcard"]))))

        builds(hand, [], "")
        if trades:
            for cost in (COSTS["city"], COSTS["settlement"], COSTS["devcard"], COSTS["road"]):
                missing = [max(0.0, k - h) for k, h in zip(cost, hand)]
                if self.afford(hand, cost) or sum(missing) > 1.0 + 1e-9:
                    continue
                need = max(range(NUM_RES), key=lambda i: missing[i])
                for i, r in enumerate(RESOURCES):
                    rate = self.g.trade_rate(c, r)
                    if i != need and hand[i] - cost[i] >= rate and self.g.bank[need] > 0:
                        h2 = list(hand)
                        h2[i] -= rate
                        h2[need] += 1
                        builds(tuple(h2), [{"type": "bank_trade", "player": c, "give": r.value,
                                            "get": RESOURCES[need].value, "times": 1}],
                               f"Trade {rate} {r.value}->1 {RESOURCES[need].value}, ")
                        break
        out.append(("End Turn (save cards)", [], pos))
        return out

    def expectimax(self, c: str, after: Pos, occ: int, roads: int) -> float:
        """E over my next roll of the best follow-up, after three opponent rolls of income."""
        base_income = [0.0] * NUM_RES
        for roll, p in ROLL_P.items():
            if roll != 7:
                for r, x in enumerate(self.income(after, roll)):
                    base_income[r] += 3 * p * x
        total = 0.0
        for roll, p in ROLL_P.items():
            hand = [h + b for h, b in zip(after.hand, base_income)]
            if roll == 7:
                if sum(hand) > self.limit:                  # discard half, cheapest cards first
                    drop = sum(hand) / 2
                    hand = [h * (1 - drop / sum(hand)) for h in hand]
            else:
                for r, x in enumerate(self.income(after, roll)):
                    hand[r] += x
            pos = after.copy(hand=tuple(hand))
            best = self.evaluate(c, pos, occ)
            for _, _, nxt in self.moves(c, pos, occ, roads, trades=False)[:-1]:
                best = max(best, self.evaluate(c, nxt, occ | nxt.sett | nxt.city))
            total += p * best
        return total

    # -- public API ------------------------------------------------------------------------
    def recommend(self, c: str, top_k: int = 5) -> list[dict[str, Any]]:
        pos = self.pos[c]
        occ, roads = self.occupied(), self.all_roads()
        pass_value = self.expectimax(c, pos, occ, roads)
        out = []
        for label, events, after in self.moves(c, pos, occ, roads):
            occ2, roads2 = occ | after.sett | after.city, roads | after.road
            v = pass_value if not events else self.expectimax(c, after, occ2, roads2)
            out.append({"label": label, "events": events, "value": v, "delta": v - pass_value,
                        "now": self.evaluate(c, after, occ2)})
        out.extend(self.offers(c, pos, occ, roads, pass_value))
        out.sort(key=lambda m: -m["value"])
        return out[:top_k]

    def offers(self, c: str, pos: Pos, occ: int, roads: int, pass_value: float) -> list[dict[str, Any]]:
        """1:1 table offers that complete a build this turn, scored as accepted half the time."""
        hand, out, seen = pos.hand, [], set()
        for goal in ("city", "settlement", "devcard", "road"):
            cost = COSTS[goal]
            missing = [max(0.0, k - h) for k, h in zip(cost, hand)]
            if not 0.5 < sum(missing) <= 1.0 + 1e-9:
                continue
            need = max(range(NUM_RES), key=lambda i: missing[i])
            spare = [i for i in range(NUM_RES) if i != need and hand[i] - cost[i] >= 1]
            if not spare:
                continue
            give = min(spare, key=lambda i: (cost[i] > 0, -(hand[i] - cost[i])))   # a card this goal doesn't use
            if (give, need) in seen:
                continue
            seen.add((give, need))
            h2 = list(hand)
            h2[give] -= 1
            h2[need] += 1
            swapped = pos.copy(hand=tuple(h2))
            accepted = max((self.expectimax(c, nxt, occ | nxt.sett | nxt.city, roads | nxt.road)
                            for _, _, nxt in self.moves(c, swapped, occ, roads, trades=False)[:-1]), default=pass_value)
            if accepted <= pass_value + 1e-9:                    # only offers that help even if accepted
                continue
            v = 0.5 * accepted + 0.5 * pass_value
            g_vec = tuple(1 if i == give else 0 for i in range(NUM_RES))
            n_vec = tuple(1 if i == need else 0 for i in range(NUM_RES))
            out.append({"label": f"Offer Trade: 1 {RESOURCES[give].value} for 1 {RESOURCES[need].value}",
                        "events": [{"type": "adjust", "player": c, "cards": g_vec, "sign": -1},
                                   {"type": "adjust", "player": c, "cards": n_vec, "sign": 1}],
                        "value": v, "delta": v - pass_value, "now": self.evaluate(c, swapped, occ),
                        "offer": {"give": list(g_vec), "get": list(n_vec)}, "goal": goal,
                        "if_accepted": accepted - pass_value})
        return out

    def win_estimate(self, override: dict[str, float] | None = None) -> dict[str, float]:
        """Softmax over position values (VP-race aware); a fast stand-in for win probability."""
        vals = {c: (override or {}).get(c, self.evaluate(c)) for c in self.order}
        top = max(vals.values())
        # early on a lead means little; near the target it decides the game
        lead_vp = max(popcount(p.sett) + 2 * popcount(p.city) + p.vp_bonus for p in self.pos.values())
        sharp = self.temperature[0] + self.temperature[1] * min(1.0, lead_vp / self.target) ** 2
        ex = {c: math.exp(sharp * (v - top)) for c, v in vals.items()}
        z = sum(ex.values())
        return {c: e / z for c, e in ex.items()}


def bench(path: str | None = None, runs: int = 20) -> dict[str, float]:
    """Time and peak memory of a fast recommendation (``python -m catan.fast_engine``)."""
    import resource
    import time
    import tracemalloc
    from pathlib import Path

    src = path or str(Path(__file__).resolve().parent.parent / "examples" / "example_board.json")
    g = GameState.load(src)
    g.execute("g red 2wheat 2ore 1wood 1brick 1sheep")
    g.execute("12")
    t0 = time.perf_counter()
    for _ in range(runs):
        eng = FastEngine(g)
        eng.recommend(g.current_color)
        eng.win_estimate()
    ms = (time.perf_counter() - t0) / runs * 1000
    tracemalloc.start()                      # memory measured separately: tracing slows Python down
    FastEngine(g).recommend(g.current_color)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 * 1024) if rss > 10 ** 7 else rss / 1024  # macOS reports bytes, Linux KiB
    return {"ms_per_recommendation": round(ms, 1), "peak_alloc_mb": round(peak / 1e6, 2),
            "process_max_rss_mb": round(rss_mb, 1), "evaluations_last_run": eng.evals}


if __name__ == "__main__":  # pragma: no cover
    print(bench())
