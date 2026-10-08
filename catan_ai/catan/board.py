"""Hex coordinate system, intersection/edge numbering and the Board schema.

Geometry
--------
Colonist.io draws the classic 19-hex board with *pointy-top* hexes laid out in
rows of 3-4-5-4-3.  Every hex is addressed with cube/axial coordinates
``(q, r, s)`` where ``q + r + s == 0`` and ``max(|q|, |r|, |s|) <= 2``.

* ``r`` grows downward (row index - 2), ``q`` grows to the right along a row.
* Hex ids 0..18 are assigned in *reading order*: sorted by ``r`` then ``q``
  (top-left hex is 0, bottom-right hex is 18).

Pixel centre of a hex of circumradius ``size`` (screen coordinates, y down)::

    x = size * sqrt(3) * (q + r / 2)
    y = size * 1.5 * r

Integer lattice for intersections
---------------------------------
To deduplicate intersections shared by up to three hexes without floating
point error, every corner is expressed on an integer lattice with unit
``sqrt(3)/2 * size`` horizontally and ``size / 2`` vertically::

    X = 2q + r + dx,   Y = 3r + dy

with the six pointy-top corners ``(dx, dy)`` in clockwise order starting at
the top corner::

    N (0,-2)  NE (1,-1)  SE (1,1)  S (0,2)  SW (-1,1)  NW (-1,-1)

Node ids 0..53 are assigned in reading order: sorted by ``(Y, X)``.  Node 0 is
the top corner of hex 0, node 53 is the bottom corner of hex 18.  Edge ids
0..71 are assigned by sorting the ``(min_node, max_node)`` pairs.

Ports occupy 9 of the 30 coastal edges.  Coastal edges are ordered clockwise
starting from the top of the board; the standard slot spacing (3,3,4 repeated)
gives slots ``PORT_SLOT_INDICES``.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "1.0"


class Resource(str, Enum):
    WOOD = "wood"
    BRICK = "brick"
    SHEEP = "sheep"
    WHEAT = "wheat"
    ORE = "ore"


RESOURCES: tuple[Resource, ...] = tuple(Resource)
RES_INDEX: dict[Resource, int] = {r: i for i, r in enumerate(RESOURCES)}


class Terrain(str, Enum):
    FOREST = "forest"
    HILLS = "hills"
    PASTURE = "pasture"
    FIELDS = "fields"
    MOUNTAINS = "mountains"
    DESERT = "desert"


TERRAIN_RESOURCE: dict[Terrain, Resource | None] = {
    Terrain.FOREST: Resource.WOOD,
    Terrain.HILLS: Resource.BRICK,
    Terrain.PASTURE: Resource.SHEEP,
    Terrain.FIELDS: Resource.WHEAT,
    Terrain.MOUNTAINS: Resource.ORE,
    Terrain.DESERT: None,
}
RESOURCE_TERRAIN: dict[Resource, Terrain] = {
    r: t for t, r in TERRAIN_RESOURCE.items() if r is not None
}

# Standard base-game composition (used for validation / constrained decoding).
TERRAIN_COUNTS: dict[Terrain, int] = {
    Terrain.FOREST: 4,
    Terrain.HILLS: 3,
    Terrain.PASTURE: 4,
    Terrain.FIELDS: 4,
    Terrain.MOUNTAINS: 3,
    Terrain.DESERT: 1,
}
NUMBER_COUNTS: dict[int, int] = {2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 1}

# Pips == number of 2d6 combinations producing the roll.
PIPS: dict[int, int] = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def roll_probability(n: int) -> float:
    return PIPS.get(n, 0) / 36.0


PLAYER_COLORS: tuple[str, ...] = ("red", "blue", "orange", "white")

PORT_GENERIC = "3:1"
PORT_TYPES: tuple[str, ...] = (PORT_GENERIC,) + tuple(r.value for r in RESOURCES)
PORT_SLOT_INDICES: tuple[int, ...] = (0, 3, 6, 10, 13, 16, 20, 23, 26)
DEFAULT_PORT_TYPES: tuple[str, ...] = (
    "3:1", "wheat", "ore", "3:1", "sheep", "3:1", "3:1", "brick", "wood",
)

# Pointy-top corner offsets on the integer lattice, clockwise from the top.
CORNER_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1),
)
SQRT3 = math.sqrt(3.0)


def parse_resource(token: str) -> Resource:
    """Parse a resource name or short alias (w b s g o / wh / grain / lumber ...)."""
    t = token.strip().lower()
    aliases = {
        "w": Resource.WOOD, "wo": Resource.WOOD, "wood": Resource.WOOD,
        "lumber": Resource.WOOD, "l": Resource.WOOD,
        "b": Resource.BRICK, "br": Resource.BRICK, "brick": Resource.BRICK,
        "c": Resource.BRICK, "clay": Resource.BRICK,
        "s": Resource.SHEEP, "sh": Resource.SHEEP, "sheep": Resource.SHEEP,
        "wool": Resource.SHEEP,
        "g": Resource.WHEAT, "wh": Resource.WHEAT, "wheat": Resource.WHEAT,
        "grain": Resource.WHEAT, "h": Resource.WHEAT,
        "o": Resource.ORE, "or": Resource.ORE, "ore": Resource.ORE, "stone": Resource.ORE,
    }
    if t not in aliases:
        raise ValueError(f"unknown resource '{token}'")
    return aliases[t]


# ---------------------------------------------------------------------------
# Static topology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Topology:
    """Immutable graph of hexes, intersections (nodes) and borders (edges)."""

    radius: int
    hex_coords: tuple[tuple[int, int], ...]          # hex id -> (q, r)
    hex_nodes: tuple[tuple[int, ...], ...]           # hex id -> 6 node ids (N, NE, SE, S, SW, NW)
    node_lattice: tuple[tuple[int, int], ...]        # node id -> (X, Y) integer lattice
    node_hexes: tuple[tuple[int, ...], ...]          # node id -> adjacent hex ids
    node_neighbors: tuple[tuple[int, ...], ...]      # node id -> adjacent node ids
    node_edges: tuple[tuple[int, ...], ...]          # node id -> incident edge ids
    edge_nodes: tuple[tuple[int, int], ...]          # edge id -> (a, b), a < b
    edge_hexes: tuple[tuple[int, ...], ...]          # edge id -> adjacent hex ids (1 or 2)
    coastal_edges: tuple[int, ...]                   # clockwise from top
    hex_index: Mapping[tuple[int, int], int] = field(repr=False)
    edge_index: Mapping[tuple[int, int], int] = field(repr=False)

    def __deepcopy__(self, memo: dict[int, Any]) -> "Topology":
        return self  # immutable and shared by every board / game clone

    @property
    def num_hexes(self) -> int:
        return len(self.hex_coords)

    @property
    def num_nodes(self) -> int:
        return len(self.node_lattice)

    @property
    def num_edges(self) -> int:
        return len(self.edge_nodes)

    def hex_id(self, q: int, r: int) -> int:
        return self.hex_index[(q, r)]

    def edge_between(self, a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in self.edge_index:
            raise KeyError(f"nodes {a} and {b} are not adjacent")
        return self.edge_index[key]

    def port_slot_edges(self) -> list[int]:
        return [self.coastal_edges[i] for i in PORT_SLOT_INDICES]

    # -- pixel helpers ---------------------------------------------------
    @staticmethod
    def hex_center_unit(q: int, r: int) -> tuple[float, float]:
        """Hex centre in units of the hex circumradius (origin = board centre)."""
        return SQRT3 * (q + r / 2.0), 1.5 * r

    def node_center_unit(self, node: int) -> tuple[float, float]:
        x, y = self.node_lattice[node]
        return x * SQRT3 / 2.0, y / 2.0

    def edge_center_unit(self, edge: int) -> tuple[float, float]:
        a, b = self.edge_nodes[edge]
        ax, ay = self.node_center_unit(a)
        bx, by = self.node_center_unit(b)
        return (ax + bx) / 2.0, (ay + by) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "radius": self.radius,
            "orientation": "pointy",
            "hexes": [
                {"id": i, "q": q, "r": r, "s": -q - r, "nodes": list(self.hex_nodes[i])}
                for i, (q, r) in enumerate(self.hex_coords)
            ],
            "nodes": [
                {"id": n, "lattice": list(self.node_lattice[n]),
                 "hexes": list(self.node_hexes[n]), "neighbors": list(self.node_neighbors[n])}
                for n in range(self.num_nodes)
            ],
            "edges": [
                {"id": e, "nodes": list(self.edge_nodes[e]), "hexes": list(self.edge_hexes[e])}
                for e in range(self.num_edges)
            ],
            "port_slots": self.port_slot_edges(),
        }


@lru_cache(maxsize=4)
def build_topology(radius: int = 2) -> Topology:
    coords = sorted(
        ((q, r) for q in range(-radius, radius + 1) for r in range(-radius, radius + 1)
         if abs(-q - r) <= radius),
        key=lambda c: (c[1], c[0]),
    )
    corner_keys: list[list[tuple[int, int]]] = []
    all_keys: set[tuple[int, int]] = set()
    for q, r in coords:
        keys = [(2 * q + r + dx, 3 * r + dy) for dx, dy in CORNER_OFFSETS]
        corner_keys.append(keys)
        all_keys.update(keys)
    ordered = sorted(all_keys, key=lambda k: (k[1], k[0]))
    node_of = {k: i for i, k in enumerate(ordered)}
    hex_nodes = tuple(tuple(node_of[k] for k in keys) for keys in corner_keys)

    node_hexes: list[list[int]] = [[] for _ in ordered]
    edge_hex_map: dict[tuple[int, int], list[int]] = {}
    for h, nodes in enumerate(hex_nodes):
        for i, n in enumerate(nodes):
            node_hexes[n].append(h)
            a, b = n, nodes[(i + 1) % 6]
            edge_hex_map.setdefault((min(a, b), max(a, b)), []).append(h)

    edge_list = sorted(edge_hex_map)
    edge_index = {e: i for i, e in enumerate(edge_list)}
    node_neighbors: list[set[int]] = [set() for _ in ordered]
    node_edges: list[list[int]] = [[] for _ in ordered]
    for i, (a, b) in enumerate(edge_list):
        node_neighbors[a].add(b)
        node_neighbors[b].add(a)
        node_edges[a].append(i)
        node_edges[b].append(i)

    def edge_angle(e: tuple[int, int]) -> float:
        (ax, ay), (bx, by) = ordered[e[0]], ordered[e[1]]
        x, y = (ax + bx) * SQRT3 / 4.0, (ay + by) / 4.0
        return (math.atan2(y, x) + math.pi / 2.0) % (2 * math.pi)

    coastal = sorted((e for e, hs in edge_hex_map.items() if len(hs) == 1), key=edge_angle)

    return Topology(
        radius=radius,
        hex_coords=tuple(coords),
        hex_nodes=hex_nodes,
        node_lattice=tuple(ordered),
        node_hexes=tuple(tuple(sorted(h)) for h in node_hexes),
        node_neighbors=tuple(tuple(sorted(n)) for n in node_neighbors),
        node_edges=tuple(tuple(sorted(e)) for e in node_edges),
        edge_nodes=tuple(edge_list),
        edge_hexes=tuple(tuple(sorted(edge_hex_map[e])) for e in edge_list),
        coastal_edges=tuple(edge_index[e] for e in coastal),
        hex_index={c: i for i, c in enumerate(coords)},
        edge_index=edge_index,
    )


# ---------------------------------------------------------------------------
# Board (topology + tile contents)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HexTile:
    id: int
    q: int
    r: int
    terrain: Terrain
    number: int | None

    @property
    def s(self) -> int:
        return -self.q - self.r

    @property
    def resource(self) -> Resource | None:
        return TERRAIN_RESOURCE[self.terrain]

    @property
    def pips(self) -> int:
        return PIPS[self.number] if self.number else 0


@dataclass(frozen=True)
class Port:
    type: str          # "3:1" or a resource value ("wood", ...)
    nodes: tuple[int, int]

    @property
    def resource(self) -> Resource | None:
        return None if self.type == PORT_GENERIC else Resource(self.type)

    @property
    def rate(self) -> int:
        return 3 if self.type == PORT_GENERIC else 2


class BoardValidationError(ValueError):
    def __init__(self, errors: Sequence[str]):
        super().__init__("; ".join(errors))
        self.errors = list(errors)


@dataclass
class Board:
    topology: Topology
    tiles: list[HexTile]
    ports: list[Port]
    robber_hex: int

    # cached lookups
    node_ports: dict[int, Port] = field(init=False, repr=False)
    number_hexes: dict[int, list[int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.node_ports = {n: p for p in self.ports for n in p.nodes}
        self.number_hexes = {}
        for t in self.tiles:
            if t.number:
                self.number_hexes.setdefault(t.number, []).append(t.id)

    # -- queries ---------------------------------------------------------
    def node_production(self, node: int, *, include_robber: bool = False) -> dict[Resource, int]:
        """Pips per resource produced by a settlement on ``node``."""
        out: dict[Resource, int] = {}
        for h in self.topology.node_hexes[node]:
            t = self.tiles[h]
            if t.resource is None or (h == self.robber_hex and not include_robber):
                continue
            out[t.resource] = out.get(t.resource, 0) + t.pips
        return out

    def node_pips(self, node: int, *, include_robber: bool = True) -> int:
        return sum(self.node_production(node, include_robber=include_robber).values())

    def resource_total_pips(self) -> dict[Resource, int]:
        out = {r: 0 for r in RESOURCES}
        for t in self.tiles:
            if t.resource:
                out[t.resource] += t.pips
        return out

    def hex_label(self, h: int) -> str:
        t = self.tiles[h]
        return f"hex {h} ({t.terrain.value}{' ' + str(t.number) if t.number else ''})"

    # -- validation ------------------------------------------------------
    def validate(self, *, strict_composition: bool = True) -> list[str]:
        """Return warnings; raise BoardValidationError on hard errors."""
        errors: list[str] = []
        warnings: list[str] = []
        topo = self.topology
        if len(self.tiles) != topo.num_hexes:
            errors.append(f"expected {topo.num_hexes} hexes, got {len(self.tiles)}")
        for t in self.tiles:
            if topo.hex_coords[t.id] != (t.q, t.r):
                errors.append(f"hex {t.id} coords ({t.q},{t.r}) != canonical {topo.hex_coords[t.id]}")
            if t.terrain == Terrain.DESERT and t.number is not None:
                errors.append(f"desert hex {t.id} must not have a number")
            if t.terrain != Terrain.DESERT and t.number not in NUMBER_COUNTS:
                errors.append(f"hex {t.id} has invalid number {t.number}")
        terrain_counts = {k: 0 for k in Terrain}
        for t in self.tiles:
            terrain_counts[t.terrain] += 1
        number_counts: dict[int, int] = {}
        for t in self.tiles:
            if t.number:
                number_counts[t.number] = number_counts.get(t.number, 0) + 1
        comp_msgs = []
        if terrain_counts != TERRAIN_COUNTS:
            comp_msgs.append(f"terrain composition {({k.value: v for k, v in terrain_counts.items()})} "
                             "differs from base game")
        if number_counts != NUMBER_COUNTS:
            comp_msgs.append(f"number tokens {dict(sorted(number_counts.items()))} differ from base game")
        (errors if strict_composition else warnings).extend(comp_msgs)
        if not 0 <= self.robber_hex < topo.num_hexes:
            errors.append(f"robber hex {self.robber_hex} out of range")
        for p in self.ports:
            if p.type not in PORT_TYPES:
                errors.append(f"bad port type {p.type}")
            a, b = p.nodes
            try:
                e = topo.edge_between(a, b)
                if e not in topo.coastal_edges:
                    errors.append(f"port {p.type} on non-coastal edge {e}")
            except KeyError:
                errors.append(f"port nodes {p.nodes} are not adjacent")
        # 6/8 adjacency is legal on Colonist random maps but worth flagging.
        for t in self.tiles:
            if t.number in (6, 8):
                for n in topo.hex_nodes[t.id]:
                    for h2 in topo.node_hexes[n]:
                        if h2 > t.id and self.tiles[h2].number in (6, 8):
                            warnings.append(f"red numbers adjacent: hex {t.id} & hex {h2}")
        if errors:
            raise BoardValidationError(errors)
        return sorted(set(warnings))

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "hexes": [
                {"id": t.id, "q": t.q, "r": t.r, "s": t.s, "terrain": t.terrain.value,
                 "resource": t.resource.value if t.resource else None, "number": t.number}
                for t in self.tiles
            ],
            "ports": [{"type": p.type, "nodes": list(p.nodes),
                       "edge": self.topology.edge_between(*p.nodes)} for p in self.ports],
            "robber_hex": self.robber_hex,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Board":
        topo = build_topology(int(data.get("layout", {}).get("radius", 2)))
        tiles: list[HexTile | None] = [None] * topo.num_hexes
        for h in data["hexes"]:
            if "id" in h:
                hid = int(h["id"])
            else:
                hid = topo.hex_id(int(h["q"]), int(h["r"]))
            q, r = topo.hex_coords[hid]
            if "terrain" in h and h["terrain"]:
                terrain = Terrain(h["terrain"])
            elif h.get("resource"):
                terrain = RESOURCE_TERRAIN[Resource(h["resource"])]
            else:
                terrain = Terrain.DESERT
            tiles[hid] = HexTile(hid, q, r, terrain, h.get("number"))
        if any(t is None for t in tiles):
            raise BoardValidationError(["missing hexes: " + str([i for i, t in enumerate(tiles) if t is None])])
        ports = []
        for p in data.get("ports", []):
            if "nodes" in p:
                a, b = p["nodes"]
            else:
                a, b = topo.edge_nodes[int(p["edge"])]
            ports.append(Port(p["type"], (min(a, b), max(a, b))))
        robber = data.get("robber_hex")
        if robber is None:
            robber = next(t.id for t in tiles if t and t.terrain == Terrain.DESERT)
        return cls(topo, tiles, ports, int(robber))  # type: ignore[arg-type]


def default_ports(types: Iterable[str] = DEFAULT_PORT_TYPES) -> list[Port]:
    topo = build_topology()
    return [Port(t, topo.edge_nodes[e]) for t, e in zip(types, topo.port_slot_edges())]


def random_board_dict(seed: int | None = None) -> dict[str, Any]:
    """A legal random base-game board (no adjacent 6/8) with shuffled port types, no pieces."""
    rng = random.Random(seed)
    topo = build_topology()
    terrains = [t for t, k in TERRAIN_COUNTS.items() for _ in range(k)]
    numbers = [n for n, k in NUMBER_COUNTS.items() for _ in range(k)]
    neighbours = [{h2 for n in topo.hex_nodes[h] for h2 in topo.node_hexes[n] if h2 != h}
                  for h in range(topo.num_hexes)]
    while True:
        rng.shuffle(terrains)
        rng.shuffle(numbers)
        it = iter(numbers)
        nums = [None if t == Terrain.DESERT else next(it) for t in terrains]
        if all(not (nums[h] in (6, 8) and any(nums[o] in (6, 8) for o in neighbours[h]))
               for h in range(topo.num_hexes)):
            break
    port_types = list(DEFAULT_PORT_TYPES)
    rng.shuffle(port_types)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "random",
        "layout": {"orientation": "pointy", "radius": 2},
        "hexes": [{"id": h, "q": q, "r": r, "terrain": terrains[h].value, "number": nums[h]}
                  for h, (q, r) in enumerate(topo.hex_coords)],
        "ports": [{"type": t, "nodes": list(topo.edge_nodes[e])}
                  for t, e in zip(port_types, topo.port_slot_edges())],
        "robber_hex": terrains.index(Terrain.DESERT),
        "turn_order": list(PLAYER_COLORS),
        "players": [],
    }


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
