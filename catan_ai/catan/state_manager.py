"""Game state tracking with a sub-2-second short-command input syntax.

Commands (case-insensitive, players accept colour prefixes ``r b o w``)::

    d <roll>                         dice roll, e.g. ``d 8`` (7 -> discard/robber phase)
    r <hex> [victim] [res]           move robber, optional steal (res only if you saw it)
    x <player> <res...> | <n>        discard on a 7: ``x red 2w b o`` or unknown ``x red 4``
    b <player> <item> [id] [free]    build: settlement|s, city|c, road|r (edge id or ``12-13``),
                                     devcard|dev [type]; ``free`` skips cost/connectivity (setup)
    t [player] <give> <get> [times]  bank/port trade at the best available rate: ``t w o``
    tp <a> <b> <a_gives> <b_gives>   player trade: ``tp red blue 2w o``
    p <player> <card> [args]         play dev: knight | mono <res> | yop <r1> <r2> | rb <e1> <e2> | vp
    g <player> <res...>              manual correction: player gains from bank
    l <player> <res...>              manual correction: player returns to bank
    n                                end turn / next player
    me <player>                      set perspective (your own hand becomes exact)
    s                                show state           u   undo           ai   recommend

Every command is parsed into a plain ``dict`` event (``parse_command``) and
applied with ``GameState.apply_event`` so automated feeds (log scrapers,
websocket bridges) can push the same events as JSON.

Hidden information
------------------
Card *totals* are always exact (every change of hand size is public).  Card
*composition* becomes uncertain after unseen steals and unknown discards.  The
``HandBelief`` keeps a weighted set of joint hand assignments for all players;
public events (builds, trades, discards) prune impossible assignments, so the
tracker re-learns exact hands as players spend.
"""

from __future__ import annotations

import copy
import json
import random
import re
import shlex
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from math import comb
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .board import (
    PLAYER_COLORS, RES_INDEX, RESOURCES, SCHEMA_VERSION, Board, Resource, parse_resource,
)

Hand = tuple[int, int, int, int, int]
Joint = tuple[Hand, ...]
NUM_RES = len(RESOURCES)
BANK_START = 19

COSTS: dict[str, Hand] = {
    "road": (1, 1, 0, 0, 0),
    "settlement": (1, 1, 1, 1, 0),
    "city": (0, 0, 0, 2, 3),
    "devcard": (0, 0, 1, 1, 1),
}
PIECE_LIMITS = {"road": 15, "settlement": 5, "city": 4}
DEV_DECK: dict[str, int] = {
    "knight": 14, "vp": 5, "road_building": 2, "year_of_plenty": 2, "monopoly": 2,
}
DEV_ALIASES = {
    "knight": "knight", "k": "knight", "soldier": "knight",
    "vp": "vp", "victory": "vp", "point": "vp",
    "rb": "road_building", "road_building": "road_building", "roads": "road_building",
    "yop": "year_of_plenty", "year_of_plenty": "year_of_plenty", "plenty": "year_of_plenty",
    "mono": "monopoly", "monopoly": "monopoly", "m": "monopoly",
}
ITEM_ALIASES = {
    "s": "settlement", "set": "settlement", "settle": "settlement", "settlement": "settlement",
    "c": "city", "city": "city",
    "r": "road", "road": "road",
    "d": "devcard", "dev": "devcard", "devcard": "devcard", "card": "devcard",
}


class CommandError(ValueError):
    pass


def hand_add(a: Sequence[int], b: Sequence[int], sign: int = 1) -> Hand:
    return tuple(x + sign * y for x, y in zip(a, b))  # type: ignore[return-value]


def can_afford(hand: Sequence[float], cost: Sequence[int]) -> bool:
    return all(h >= c for h, c in zip(hand, cost))


def fmt_hand(hand: Sequence[float], exact: bool = True) -> str:
    parts = []
    for r, n in zip(RESOURCES, hand):
        if n > 1e-9:
            parts.append(f"{n:g}{r.value[0:2]}" if exact else f"{n:.1f}{r.value[0:2]}")
    return " ".join(parts) or "-"


def parse_resource_list(tokens: Iterable[str]) -> Hand:
    """Parse ``2w b o`` / ``2w+b`` / ``w,w,o`` / ``wood:2`` into a count vector."""
    counts = [0] * NUM_RES
    for tok in tokens:
        for part in re.split(r"[+,]", tok):
            part = part.strip().lower()
            if not part:
                continue
            m = re.fullmatch(r"(\d*)([a-z]+)(?::?(\d+))?", part)
            if not m:
                raise CommandError(f"cannot parse resources '{part}'")
            n = int(m.group(1) or m.group(3) or 1)
            counts[RES_INDEX[parse_resource(m.group(2))]] += n
    return tuple(counts)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Hidden-information hand tracker
# ---------------------------------------------------------------------------


class HandBelief:
    """Weighted set of joint hand assignments for all players."""

    def __init__(self, n_players: int, max_states: int = 2048):
        self.n = n_players
        self.max_states = max_states
        self.states: dict[Joint, float] = {tuple((0,) * NUM_RES for _ in range(n_players)): 1.0}
        self.warnings: list[str] = []

    # -- internals ---------------------------------------------------------
    def _set(self, new: Mapping[Joint, float]) -> None:
        total = sum(new.values())
        items = sorted(new.items(), key=lambda kv: -kv[1])[: self.max_states]
        total = sum(w for _, w in items) or 1.0
        self.states = {s: w / total for s, w in items}

    @staticmethod
    def _replace(state: Joint, p: int, hand: Hand) -> Joint:
        return state[:p] + (hand,) + state[p + 1:]

    def most_likely(self) -> Joint:
        return max(self.states.items(), key=lambda kv: kv[1])[0]

    # -- public mutation ---------------------------------------------------
    def set_hand(self, p: int, hand: Sequence[int]) -> None:
        self._set({self._replace(s, p, tuple(hand)): w for s, w in self.states.items()})  # type: ignore[arg-type]

    def add(self, p: int, counts: Sequence[int]) -> None:
        new: dict[Joint, float] = defaultdict(float)
        for s, w in self.states.items():
            new[self._replace(s, p, hand_add(s[p], counts))] += w
        self._set(new)

    def remove(self, p: int, counts: Sequence[int], *, label: str = "") -> bool:
        """Remove known cards; prunes impossible states. Returns False if inconsistent."""
        new: dict[Joint, float] = defaultdict(float)
        for s, w in self.states.items():
            if can_afford(s[p], counts):
                new[self._replace(s, p, hand_add(s[p], counts, -1))] += w
        if new:
            self._set(new)
            return True
        # Input contradicts the tracker (missed event?). Reconcile on the MAP state.
        best = self.most_likely()
        fixed = tuple(max(0, h - c) for h, c in zip(best[p], counts))
        self.warnings.append(
            f"player {p}: could not afford {fmt_hand(counts)} {label} with tracked hand "
            f"{fmt_hand(best[p])}; assuming missed income")
        self.states = {self._replace(best, p, fixed): 1.0}  # type: ignore[arg-type]
        return False

    def steal(self, thief: int, victim: int, res: int | None = None) -> None:
        if res is not None:
            one = tuple(1 if i == res else 0 for i in range(NUM_RES))
            self.remove(victim, one, label="(steal)")
            self.add(thief, one)
            return
        new: dict[Joint, float] = defaultdict(float)
        for s, w in self.states.items():
            total = sum(s[victim])
            if total == 0:
                new[s] += w
                continue
            for r in range(NUM_RES):
                if s[victim][r] == 0:
                    continue
                one = tuple(1 if i == r else 0 for i in range(NUM_RES))
                s2 = self._replace(s, victim, hand_add(s[victim], one, -1))
                s2 = self._replace(s2, thief, hand_add(s2[thief], one))
                new[s2] += w * s[victim][r] / total
        self._set(new)

    def discard_unknown(self, p: int, n: int) -> None:
        """Player discarded ``n`` cards of unknown composition (hypergeometric branching)."""
        new: dict[Joint, float] = defaultdict(float)
        for s, w in self.states.items():
            hand = s[p]
            total = sum(hand)
            if total < n:
                new[self._replace(s, p, (0,) * NUM_RES)] += w
                continue
            denom = comb(total, n)
            for combo in product(*(range(min(h, n) + 1) for h in hand)):
                if sum(combo) != n:
                    continue
                weight = 1
                for h, c in zip(hand, combo):
                    weight *= comb(h, c)
                new[self._replace(s, p, hand_add(hand, combo, -1))] += w * weight / denom
        self._set(new)

    def monopoly(self, p: int, res: int) -> None:
        new: dict[Joint, float] = defaultdict(float)
        for s, w in self.states.items():
            hands = [list(h) for h in s]
            taken = 0
            for i in range(self.n):
                if i != p:
                    taken += hands[i][res]
                    hands[i][res] = 0
            hands[p][res] += taken
            new[tuple(tuple(h) for h in hands)] += w  # type: ignore[misc]
        self._set(new)

    # -- queries -----------------------------------------------------------
    def total(self, p: int) -> int:
        return sum(self.most_likely()[p])

    def expected(self, p: int) -> list[float]:
        out = [0.0] * NUM_RES
        for s, w in self.states.items():
            for i in range(NUM_RES):
                out[i] += w * s[p][i]
        return out

    def prob_afford(self, p: int, cost: Sequence[int]) -> float:
        return sum(w for s, w in self.states.items() if can_afford(s[p], cost))

    def is_exact(self, p: int) -> bool:
        first = next(iter(self.states))[p]
        return all(s[p] == first for s in self.states)

    def sample(self, rng: random.Random) -> Joint:
        states, weights = zip(*self.states.items())
        return rng.choices(states, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Player & game state
# ---------------------------------------------------------------------------


@dataclass
class PlayerState:
    color: str
    settlements: set[int] = field(default_factory=set)
    cities: set[int] = field(default_factory=set)
    roads: set[int] = field(default_factory=set)
    dev_hidden: int = 0                                   # unplayed, type unknown to observer
    dev_known: dict[str, int] = field(default_factory=dict)  # perspective player's own cards
    dev_played: dict[str, int] = field(default_factory=dict)
    knights_played: int = 0
    vp_cards: int = 0                                     # revealed or known VP cards

    def buildings(self) -> set[int]:
        return self.settlements | self.cities


@dataclass
class GameState:
    board: Board
    order: list[str]
    players: dict[str, PlayerState]
    belief: HandBelief
    bank: list[int] = field(default_factory=lambda: [BANK_START] * NUM_RES)
    dev_deck: int = sum(DEV_DECK.values())
    current: int = 0
    turn_number: int = 1
    phase: str = "main"                  # main | discard | robber
    pending_discards: dict[str, int] = field(default_factory=dict)
    longest_road_owner: str | None = None
    largest_army_owner: str | None = None
    perspective: str | None = None
    log: list[str] = field(default_factory=list)
    last_roll: int | None = None
    _history: list["GameState"] = field(default_factory=list, repr=False)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GameState":
        board = Board.from_dict(data)
        board.validate(strict_composition=data.get("strict_composition", True))
        players_data = data.get("players", [])
        order = data.get("turn_order") or [p["color"] for p in players_data] or list(PLAYER_COLORS)
        game = cls(board=board, order=list(order),
                   players={c: PlayerState(c) for c in order},
                   belief=HandBelief(len(order)))
        topo = board.topology
        for pd in players_data:
            ps = game.players[pd["color"]]
            for n in pd.get("settlements", []):
                game._place_building(ps, int(n), "settlement")
            for n in pd.get("cities", []):
                game._place_building(ps, int(n), "city")
            for e in pd.get("roads", []):
                ps.roads.add(game._parse_edge(e) if isinstance(e, (str, list)) else int(e))
            ps.dev_hidden = int(pd.get("dev_cards", 0))
            ps.knights_played = int(pd.get("knights_played", 0))
            ps.vp_cards = int(pd.get("vp_cards", 0))
            game.dev_deck -= ps.dev_hidden + ps.knights_played + ps.vp_cards
        hands = data.get("hands")
        if hands:
            for color, hand in hands.items():
                vec = parse_resource_list([f"{v}{k}" for k, v in hand.items() if v]) if isinstance(hand, dict) \
                    else tuple(hand)
                game._gain(color, vec)
        elif data.get("starting_resources", "second_settlement") == "second_settlement":
            for pd in players_data:
                setts = list(pd.get("settlements", [])) + list(pd.get("cities", []))
                if len(setts) >= 2:
                    vec = [0] * NUM_RES
                    for h in topo.node_hexes[int(setts[1])]:
                        res = board.tiles[h].resource
                        if res is not None:
                            vec[RES_INDEX[res]] += 1
                    game._gain(pd["color"], vec)
        if "current_player" in data:
            game.current = game.order.index(data["current_player"])
        game.turn_number = int(data.get("turn_number", 1))
        game.perspective = data.get("perspective")
        game._update_longest_road()
        game._update_largest_army()
        return game

    @classmethod
    def load(cls, path: str | Path) -> "GameState":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "source": "state_manager",
                               "layout": {"orientation": "pointy", "radius": self.board.topology.radius}}
        out.update(self.board.to_dict())
        out["turn_order"] = self.order
        out["current_player"] = self.current_color
        out["turn_number"] = self.turn_number
        out["perspective"] = self.perspective
        out["players"] = [
            {"color": c, "settlements": sorted(p.settlements), "cities": sorted(p.cities),
             "roads": sorted(p.roads), "dev_cards": p.dev_hidden,
             "knights_played": p.knights_played, "vp_cards": p.vp_cards}
            for c, p in self.players.items()
        ]
        best = self.belief.most_likely()
        out["hands"] = {c: {r.value: best[i][j] for j, r in enumerate(RESOURCES)}
                        for i, c in enumerate(self.order)}
        out["starting_resources"] = "none"
        return out

    # -- small helpers -------------------------------------------------------
    @property
    def current_color(self) -> str:
        return self.order[self.current]

    def idx(self, color: str) -> int:
        return self.order.index(color)

    def resolve_player(self, token: str) -> str:
        t = token.lower()
        matches = [c for c in self.order if c.startswith(t)]
        if len(matches) != 1:
            raise CommandError(f"unknown/ambiguous player '{token}'")
        return matches[0]

    def is_player_token(self, token: str) -> bool:
        try:
            self.resolve_player(token)
            return True
        except CommandError:
            return False

    def node_owner(self, node: int) -> tuple[str, str] | None:
        for c, p in self.players.items():
            if node in p.settlements:
                return c, "settlement"
            if node in p.cities:
                return c, "city"
        return None

    def edge_owner(self, edge: int) -> str | None:
        for c, p in self.players.items():
            if edge in p.roads:
                return c
        return None

    def expected_hand(self, color: str) -> list[float]:
        return self.belief.expected(self.idx(color))

    def hand_total(self, color: str) -> int:
        return self.belief.total(self.idx(color))

    def player_ports(self, color: str) -> list[Any]:
        return [self.board.node_ports[n] for n in self.players[color].buildings()
                if n in self.board.node_ports]

    def trade_rate(self, color: str, res: Resource) -> int:
        rate = 4
        for port in self.player_ports(color):
            if port.resource is None:
                rate = min(rate, 3)
            elif port.resource == res:
                rate = 2
        return rate

    def public_vp(self, color: str) -> int:
        p = self.players[color]
        vp = len(p.settlements) + 2 * len(p.cities) + p.vp_cards
        vp += 2 if self.longest_road_owner == color else 0
        vp += 2 if self.largest_army_owner == color else 0
        return vp

    def expected_hidden_vp(self, color: str) -> float:
        p = self.players[color]
        if color == self.perspective:
            return float(p.dev_known.get("vp", 0))
        unseen_vp = DEV_DECK["vp"] - sum(q.vp_cards for q in self.players.values())
        unseen = self.dev_deck + sum(q.dev_hidden for q in self.players.values())
        return p.dev_hidden * (unseen_vp / unseen) if unseen else 0.0

    # -- legality ------------------------------------------------------------
    def can_place_settlement(self, color: str, node: int, *, require_road: bool = True) -> bool:
        topo = self.board.topology
        if self.node_owner(node) is not None:
            return False
        if any(self.node_owner(nb) is not None for nb in topo.node_neighbors[node]):
            return False
        if require_road:
            return any(e in self.players[color].roads for e in topo.node_edges[node])
        return True

    def legal_settlement_nodes(self, color: str, *, require_road: bool = True) -> list[int]:
        return [n for n in range(self.board.topology.num_nodes)
                if self.can_place_settlement(color, n, require_road=require_road)]

    def can_place_road(self, color: str, edge: int) -> bool:
        if self.edge_owner(edge) is not None:
            return False
        topo = self.board.topology
        p = self.players[color]
        for n in topo.edge_nodes[edge]:
            owner = self.node_owner(n)
            if owner and owner[0] == color:
                return True
            if owner is None and any(e in p.roads for e in topo.node_edges[n] if e != edge):
                return True
        return False

    def legal_road_edges(self, color: str) -> list[int]:
        return [e for e in range(self.board.topology.num_edges) if self.can_place_road(color, e)]

    # -- longest road / largest army ---------------------------------------
    def longest_road(self, color: str) -> int:
        topo = self.board.topology
        roads = self.players[color].roads
        if not roads:
            return 0
        adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for e in roads:
            a, b = topo.edge_nodes[e]
            adj[a].append((e, b))
            adj[b].append((e, a))
        blocked = {n for c, p in self.players.items() if c != color for n in p.buildings()}
        best = 0

        def dfs(node: int, used: set[int], length: int) -> None:
            nonlocal best
            best = max(best, length)
            if length > 0 and node in blocked:
                return
            for e, nb in adj[node]:
                if e not in used:
                    used.add(e)
                    dfs(nb, used, length + 1)
                    used.discard(e)

        for start in list(adj):
            dfs(start, set(), 0)
        return best

    def _update_longest_road(self) -> None:
        lengths = {c: self.longest_road(c) for c in self.order}
        owner = self.longest_road_owner
        owner_len = lengths.get(owner, 0) if owner else 0
        best = max(lengths.values())
        leaders = [c for c, l in lengths.items() if l == best]
        if owner and owner_len >= 5 and owner_len >= best:
            return
        if best >= 5 and len(leaders) == 1:
            self.longest_road_owner = leaders[0]
        elif owner and (owner_len < 5 or owner_len < best):
            self.longest_road_owner = None

    def _update_largest_army(self) -> None:
        owner = self.largest_army_owner
        owner_k = self.players[owner].knights_played if owner else 2
        for c in self.order:
            k = self.players[c].knights_played
            if k >= 3 and k > owner_k:
                self.largest_army_owner, owner_k = c, k

    # -- resource flows (bank exact, hands via belief) ----------------------
    def _gain(self, color: str, vec: Sequence[int]) -> None:
        self.belief.add(self.idx(color), vec)
        for i, n in enumerate(vec):
            self.bank[i] -= n

    def _ensure(self, color: str, vec: Sequence[int], label: str) -> None:
        """If no tracked hand can cover ``vec``, assume missed income from the bank."""
        i = self.idx(color)
        if self.belief.prob_afford(i, vec) > 0:
            return
        best = self.belief.most_likely()[i]
        deficit = [max(0, c - h) for c, h in zip(vec, best)]
        self.belief.warnings.append(
            f"{color} could not afford {fmt_hand(vec)} {label} with tracked hand {fmt_hand(best)}; "
            f"assuming missed income {fmt_hand(deficit)}")
        self.belief.states = {self.belief.most_likely(): 1.0}
        self._gain(color, deficit)

    def _pay(self, color: str, vec: Sequence[int], label: str = "") -> None:
        self._ensure(color, vec, label)
        self.belief.remove(self.idx(color), vec, label=label)
        for i, n in enumerate(vec):
            self.bank[i] += n

    def _place_building(self, ps: PlayerState, node: int, kind: str) -> None:
        if kind == "city":
            ps.settlements.discard(node)
            ps.cities.add(node)
        else:
            ps.settlements.add(node)

    def _parse_edge(self, tok: Any) -> int:
        topo = self.board.topology
        if isinstance(tok, list):
            return topo.edge_between(int(tok[0]), int(tok[1]))
        tok = str(tok)
        if "-" in tok:
            a, b = tok.split("-")
            return topo.edge_between(int(a), int(b))
        e = int(tok)
        if not 0 <= e < topo.num_edges:
            raise CommandError(f"edge {e} out of range")
        return e

    def clone(self) -> "GameState":
        hist, self._history = self._history, []
        try:
            return copy.deepcopy(self)
        finally:
            self._history = hist

    # -- event application ---------------------------------------------------
    def execute(self, line: str) -> str:
        """Parse and apply one short command; returns a human-readable result."""
        event = parse_command(line, self)
        if event["type"] == "undo":
            return "undone" if self.undo() else "nothing to undo"
        if event["type"] in ("show", "noop"):
            return self.summary() if event["type"] == "show" else ""
        msg = self.apply_events([event])
        self.log.append(f"T{self.turn_number} {line.strip()}")
        return msg

    def undo(self) -> bool:
        if not self._history:
            return False
        prev = self._history.pop()
        hist = self._history
        self.__dict__.update(prev.__dict__)
        self._history = hist
        return True

    def apply_events(self, events: Sequence[Mapping[str, Any]]) -> str:
        """Apply events atomically as one undoable step (rolled back on any error)."""
        snapshot = self.clone()
        n_warn = len(self.belief.warnings)
        history = self._history
        try:
            msg = "; ".join(m for m in (self.apply_event(ev) for ev in events) if m)
        except Exception:
            self.__dict__.update(snapshot.__dict__)  # roll back the partial step
            self._history = history
            raise
        self._history = (history + [snapshot])[-50:]
        new_warn = self.belief.warnings[n_warn:]
        if new_warn:
            msg += "\n  ! " + "\n  ! ".join(new_warn)
        return msg

    def apply_event(self, ev: Mapping[str, Any]) -> str:
        handler = getattr(self, f"_ev_{ev['type']}", None)
        if handler is None:
            raise CommandError(f"unknown event type {ev['type']}")
        return handler(ev)

    def _ev_roll(self, ev: Mapping[str, Any]) -> str:
        roll = int(ev["value"])
        if not 2 <= roll <= 12:
            raise CommandError("roll must be 2..12")
        self.last_roll = roll
        if roll == 7:
            self.pending_discards = {c: self.hand_total(c) // 2 for c in self.order
                                     if self.hand_total(c) > 7}
            self.phase = "discard" if self.pending_discards else "robber"
            if self.pending_discards:
                need = ", ".join(f"{c} {n}" for c, n in self.pending_discards.items())
                return f"rolled 7 - discards needed: {need} (x <player> <cards>), then r <hex> [victim]"
            return "rolled 7 - nobody discards; move robber: r <hex> [victim]"
        demand: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for h in self.board.number_hexes.get(roll, []):
            if h == self.board.robber_hex:
                continue
            res = self.board.tiles[h].resource
            assert res is not None
            for n in self.board.topology.hex_nodes[h]:
                owner = self.node_owner(n)
                if owner:
                    demand[RES_INDEX[res]][owner[0]] += 2 if owner[1] == "city" else 1
        gains: dict[str, list[int]] = defaultdict(lambda: [0] * NUM_RES)
        notes = []
        for ri, per in demand.items():
            total = sum(per.values())
            if total <= self.bank[ri]:
                for c, k in per.items():
                    gains[c][ri] += k
            elif len(per) == 1:
                c = next(iter(per))
                gains[c][ri] += self.bank[ri]
                notes.append(f"bank short on {RESOURCES[ri].value}")
            else:
                notes.append(f"bank short on {RESOURCES[ri].value}: nobody receives it")
        for c, vec in gains.items():
            self._gain(c, vec)
        blocked = [h for h in self.board.number_hexes.get(roll, []) if h == self.board.robber_hex]
        parts = [f"{c} +{fmt_hand(v)}" for c, v in gains.items()] or ["no production"]
        if blocked:
            parts.append(f"(robber blocks hex {blocked[0]})")
        return f"rolled {roll}: " + ", ".join(parts + notes)

    def _ev_discard(self, ev: Mapping[str, Any]) -> str:
        color = ev["player"]
        if "count" in ev:
            n = int(ev["count"])
            self.belief.discard_unknown(self.idx(color), n)
            # composition unknown: re-derive the bank from the most likely joint hand
            best = self.belief.most_likely()
            self.bank = [BANK_START - sum(h[i] for h in best) for i in range(NUM_RES)]
            msg = f"{color} discarded {n} unknown cards"
        else:
            vec = tuple(ev["cards"])
            self._pay(color, vec, "(discard)")
            n = sum(vec)
            msg = f"{color} discarded {fmt_hand(vec)}"
        if color in self.pending_discards:
            if n != self.pending_discards[color]:
                msg += f" (expected {self.pending_discards[color]})"
            del self.pending_discards[color]
        if self.phase == "discard" and not self.pending_discards:
            self.phase = "robber"
            msg += " - now move robber"
        return msg

    def _ev_robber(self, ev: Mapping[str, Any]) -> str:
        h = int(ev["hex"])
        if not 0 <= h < self.board.topology.num_hexes:
            raise CommandError("hex out of range")
        if h == self.board.robber_hex:
            raise CommandError("robber must move to a different hex")
        self.board.robber_hex = h
        thief = ev.get("thief") or self.current_color
        msg = f"robber -> {self.board.hex_label(h)}"
        victim = ev.get("victim")
        if victim:
            adjacent = {self.node_owner(n)[0] for n in self.board.topology.hex_nodes[h]  # type: ignore[index]
                        if self.node_owner(n)}
            if victim not in adjacent:
                msg += f" (warning: {victim} has no building on hex {h})"
            res = ev.get("resource")
            ri = RES_INDEX[Resource(res)] if res else None
            if ri is not None:
                self._ensure(victim, [int(k == ri) for k in range(NUM_RES)], "(steal)")
            self.belief.steal(self.idx(thief), self.idx(victim), ri)
            msg += f"; {thief} stole {'1 ' + res if res else 'a card'} from {victim}"
        if self.pending_discards:
            msg += f" (warning: discards still pending {self.pending_discards})"
        self.phase = "main"
        return msg

    def _ev_build(self, ev: Mapping[str, Any]) -> str:
        color, item, free = ev["player"], ev["item"], bool(ev.get("free"))
        ps = self.players[color]
        if item == "devcard":
            if self.dev_deck <= 0:
                raise CommandError("dev deck is empty")
            if not free:
                self._pay(color, COSTS["devcard"], "(devcard)")
            self.dev_deck -= 1
            ps.dev_hidden += 1
            card = ev.get("card")
            if card:
                ps.dev_known[card] = ps.dev_known.get(card, 0) + 1
            return f"{color} bought a dev card{' (' + card + ')' if card else ''}; deck {self.dev_deck}"
        if item == "road":
            edge = self._parse_edge(ev["target"])
            if not free and not self.can_place_road(color, edge):
                raise CommandError(f"road on edge {edge} is not legal for {color}")
            if self.edge_owner(edge):
                raise CommandError(f"edge {edge} already taken")
            if len(ps.roads) >= PIECE_LIMITS["road"]:
                raise CommandError("no roads left")
            if not free:
                self._pay(color, COSTS["road"], "(road)")
            ps.roads.add(edge)
            self._update_longest_road()
            a, b = self.board.topology.edge_nodes[edge]
            lr = f"; longest road: {self.longest_road_owner}" if self.longest_road_owner else ""
            return f"{color} road on edge {edge} ({a}-{b}){lr}"
        node = int(ev["target"])
        if not 0 <= node < self.board.topology.num_nodes:
            raise CommandError("node out of range")
        if item == "settlement":
            if not self.can_place_settlement(color, node, require_road=not free):
                raise CommandError(f"settlement on node {node} is not legal for {color}")
            if len(ps.settlements) >= PIECE_LIMITS["settlement"]:
                raise CommandError("no settlements left")
            if not free:
                self._pay(color, COSTS["settlement"], "(settlement)")
            ps.settlements.add(node)
            self._update_longest_road()
        elif item == "city":
            if node not in ps.settlements:
                raise CommandError(f"{color} has no settlement on node {node}")
            if len(ps.cities) >= PIECE_LIMITS["city"]:
                raise CommandError("no cities left")
            if not free:
                self._pay(color, COSTS["city"], "(city)")
            self._place_building(ps, node, "city")
        else:
            raise CommandError(f"unknown item {item}")
        return f"{color} {item} on node {node} (VP {self.public_vp(color)})"

    def _ev_bank_trade(self, ev: Mapping[str, Any]) -> str:
        color = ev.get("player") or self.current_color
        give, get = Resource(ev["give"]), Resource(ev["get"])
        times = int(ev.get("times", 1))
        rate = int(ev.get("rate") or self.trade_rate(color, give))
        give_vec = [0] * NUM_RES
        get_vec = [0] * NUM_RES
        give_vec[RES_INDEX[give]] = rate * times
        get_vec[RES_INDEX[get]] = times
        if self.bank[RES_INDEX[get]] < times:
            raise CommandError(f"bank has no {get.value}")
        self._pay(color, give_vec, "(trade)")
        self._gain(color, get_vec)
        return f"{color} traded {rate * times} {give.value} -> {times} {get.value} ({rate}:1)"

    def _ev_player_trade(self, ev: Mapping[str, Any]) -> str:
        a, b = ev["a"], ev["b"]
        ga, gb = tuple(ev["a_gives"]), tuple(ev["b_gives"])
        self._ensure(a, ga, "(trade)")
        self._ensure(b, gb, "(trade)")
        self.belief.remove(self.idx(a), ga, label="(trade)")
        self.belief.remove(self.idx(b), gb, label="(trade)")
        self.belief.add(self.idx(b), ga)
        self.belief.add(self.idx(a), gb)
        return f"{a} gave {fmt_hand(ga)} to {b} for {fmt_hand(gb)}"

    def _ev_play_dev(self, ev: Mapping[str, Any]) -> str:
        color, card = ev["player"], ev["card"]
        ps = self.players[color]
        if ps.dev_hidden <= 0:
            raise CommandError(f"{color} has no unplayed dev cards")
        ps.dev_hidden -= 1
        if ps.dev_known.get(card):
            ps.dev_known[card] -= 1
        ps.dev_played[card] = ps.dev_played.get(card, 0) + 1
        if card == "knight":
            ps.knights_played += 1
            self._update_largest_army()
            self.phase = "robber"
            return f"{color} played knight ({ps.knights_played}); move robber: r <hex> [victim]"
        if card == "vp":
            ps.vp_cards += 1
            return f"{color} revealed a VP card"
        if card == "monopoly":
            ri = RES_INDEX[Resource(ev["resource"])]
            before = self.belief.expected(self.idx(color))[ri]
            self.belief.monopoly(self.idx(color), ri)
            after = self.belief.expected(self.idx(color))[ri]
            return f"{color} monopolised {ev['resource']} (~{after - before:.1f} cards)"
        if card == "year_of_plenty":
            vec = [0] * NUM_RES
            for r in ev["resources"]:
                vec[RES_INDEX[Resource(r)]] += 1
            self._gain(color, vec)
            return f"{color} year of plenty: {fmt_hand(vec)}"
        if card == "road_building":
            msgs = [self._ev_build({"player": color, "item": "road", "target": e, "free": True})
                    for e in ev.get("edges", [])]
            return f"{color} road building: " + "; ".join(msgs)
        raise CommandError(f"unknown dev card {card}")

    def _ev_adjust(self, ev: Mapping[str, Any]) -> str:
        color, vec = ev["player"], tuple(ev["cards"])
        if ev["sign"] > 0:
            self._gain(color, vec)
            return f"{color} +{fmt_hand(vec)} (manual)"
        self._pay(color, vec, "(manual)")
        return f"{color} -{fmt_hand(vec)} (manual)"

    def _ev_next_turn(self, ev: Mapping[str, Any]) -> str:
        self.current = (self.current + 1) % len(self.order)
        if self.current == 0:
            self.turn_number += 1
        self.phase, self.pending_discards, self.last_roll = "main", {}, None
        return f"turn {self.turn_number}: {self.current_color} to roll"

    def _ev_perspective(self, ev: Mapping[str, Any]) -> str:
        self.perspective = ev["player"]
        return f"perspective set to {self.perspective}"

    # -- display ---------------------------------------------------------------
    def summary(self) -> str:
        lines = [f"Turn {self.turn_number} | {self.current_color} to act | phase {self.phase} | "
                 f"robber on {self.board.hex_label(self.board.robber_hex)}",
                 "bank: " + fmt_hand(self.bank) + f" | dev deck {self.dev_deck}"]
        for c in self.order:
            p = self.players[c]
            i = self.idx(c)
            exact = self.belief.is_exact(i)
            hand = self.belief.expected(i)
            tag = "" if exact else " ~"
            extras = []
            if self.longest_road_owner == c:
                extras.append("LR")
            if self.largest_army_owner == c:
                extras.append("LA")
            lines.append(
                f"  {c:<6} VP {self.public_vp(c):>2} | {self.hand_total(c):>2} cards [{fmt_hand(hand, exact)}]{tag}"
                f" | S{sorted(p.settlements)} C{sorted(p.cities)} roads {len(p.roads)}"
                f" (LR len {self.longest_road(c)}) | dev {p.dev_hidden} K{p.knights_played}"
                f"{' ' + '/'.join(extras) if extras else ''}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command parser
# ---------------------------------------------------------------------------


def _is_trade_player(tok: str, game: GameState) -> bool:
    """In ``t`` a leading token is a player only if it cannot also be a resource (w/b/o clash)."""
    if tok in game.order:
        return True
    try:
        parse_resource(re.sub(r"^\d+", "", tok))
        return False
    except ValueError:
        return game.is_player_token(tok)


def parse_command(line: str, game: GameState) -> dict[str, Any]:
    toks = shlex.split(line.strip().lower())
    if not toks:
        return {"type": "noop"}
    cmd, args = toks[0], toks[1:]

    def need(n: int, usage: str) -> None:
        if len(args) < n:
            raise CommandError(f"usage: {usage}")

    if cmd in ("d", "roll"):
        need(1, "d <roll>")
        return {"type": "roll", "value": int(args[0])}
    if cmd in ("r", "robber"):
        need(1, "r <hex> [victim] [res]")
        ev: dict[str, Any] = {"type": "robber", "hex": int(args[0])}
        if len(args) > 1:
            ev["victim"] = game.resolve_player(args[1])
        if len(args) > 2:
            ev["resource"] = parse_resource(args[2]).value
        return ev
    if cmd in ("x", "discard"):
        need(2, "x <player> <cards...|n>")
        player = game.resolve_player(args[0])
        if len(args) == 2 and args[1].isdigit():
            return {"type": "discard", "player": player, "count": int(args[1])}
        return {"type": "discard", "player": player, "cards": parse_resource_list(args[1:])}
    if cmd in ("b", "build", "buy"):
        need(2, "b <player> <item> [id] [free]")
        player = game.resolve_player(args[0])
        item = ITEM_ALIASES.get(args[1])
        if item is None:
            raise CommandError(f"unknown item '{args[1]}'")
        rest = [a for a in args[2:] if a != "free"]
        ev = {"type": "build", "player": player, "item": item, "free": "free" in args[2:]}
        if item == "devcard":
            if rest:
                ev["card"] = DEV_ALIASES.get(rest[0], rest[0])
        else:
            if not rest:
                raise CommandError(f"usage: b <player> {item} <id>")
            ev["target"] = rest[0]
        return ev
    if cmd in ("t", "trade"):
        player = None
        if args and len(args) >= 3 and _is_trade_player(args[0], game):
            player, args = game.resolve_player(args[0]), args[1:]
        need(2, "t [player] <give> <get> [times]")
        m = re.fullmatch(r"(\d*)([a-z]+)", args[0])
        if not m:
            raise CommandError("bad give resource")
        give = parse_resource(m.group(2)).value
        get = parse_resource(re.sub(r"^\d+", "", args[1])).value
        times = int(args[2]) if len(args) > 2 else int(re.match(r"\d*", args[1]).group(0) or 1)  # type: ignore[union-attr]
        ev = {"type": "bank_trade", "player": player, "give": give, "get": get, "times": times}
        if m.group(1):
            ev["rate"] = int(m.group(1)) // times
        return ev
    if cmd == "tp":
        need(4, "tp <a> <b> <a_gives> <b_gives>")
        return {"type": "player_trade", "a": game.resolve_player(args[0]), "b": game.resolve_player(args[1]),
                "a_gives": parse_resource_list([args[2]]), "b_gives": parse_resource_list([args[3]])}
    if cmd in ("p", "play"):
        need(2, "p <player> <card> [args]")
        player = game.resolve_player(args[0])
        card = DEV_ALIASES.get(args[1])
        if card is None:
            raise CommandError(f"unknown dev card '{args[1]}'")
        ev = {"type": "play_dev", "player": player, "card": card}
        if card == "monopoly":
            need(3, "p <player> mono <res>")
            ev["resource"] = parse_resource(args[2]).value
        elif card == "year_of_plenty":
            need(3, "p <player> yop <r1> [r2]")
            ev["resources"] = [parse_resource(a).value for a in args[2:4]]
        elif card == "road_building":
            ev["edges"] = args[2:4]
        return ev
    if cmd in ("g", "gain", "l", "lose"):
        need(2, f"{cmd} <player> <res...>")
        return {"type": "adjust", "player": game.resolve_player(args[0]),
                "cards": parse_resource_list(args[1:]), "sign": 1 if cmd in ("g", "gain") else -1}
    if cmd in ("n", "next", "end"):
        return {"type": "next_turn"}
    if cmd == "me":
        need(1, "me <player>")
        return {"type": "perspective", "player": game.resolve_player(args[0])}
    if cmd in ("s", "show", "state"):
        return {"type": "show"}
    if cmd in ("u", "undo"):
        return {"type": "undo"}
    raise CommandError(f"unknown command '{cmd}' (d r x b t tp p g l n me s u ai)")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


def repl(game: GameState, *, engine_factory: Any = None) -> None:  # pragma: no cover - interactive
    try:
        import readline  # noqa: F401  (history / line editing)
    except ImportError:
        pass
    print(game.summary())
    print(__doc__.split("Every command")[0])
    while True:
        try:
            line = input(f"[{game.current_color} {game.phase}]> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        line = line.strip()
        if line in ("q", "quit", "exit"):
            return
        if line.startswith("save "):
            Path(line[5:].strip()).write_text(json.dumps(game.to_dict(), indent=2))
            print("saved")
            continue
        if line.split()[:1] == ["ai"] and engine_factory:
            parts = line.split()
            player = game.resolve_player(parts[1]) if len(parts) > 1 else None
            print(engine_factory(game).recommend(player).render())
            continue
        try:
            out = game.execute(line)
            if out:
                print(out)
        except (CommandError, KeyError, ValueError) as exc:
            print(f"error: {exc}")


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - interactive
    import argparse

    from .catan_engine import CatanEngine

    ap = argparse.ArgumentParser(description="Catan live state tracker")
    ap.add_argument("state", help="board/game JSON (from vision_extractor or saved state)")
    ap.add_argument("--me", help="your colour (perspective)")
    ap.add_argument("--rollouts", type=int, default=60)
    ns = ap.parse_args(argv)
    game = GameState.load(ns.state)
    if ns.me:
        game.perspective = game.resolve_player(ns.me)
    repl(game, engine_factory=lambda g: CatanEngine(g, rollouts=ns.rollouts))


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
