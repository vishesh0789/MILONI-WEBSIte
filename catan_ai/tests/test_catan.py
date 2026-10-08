"""Tests for topology, state tracking, hidden-hand belief, AI engine and vision."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from catan.board import (
    NUMBER_COUNTS, PORT_SLOT_INDICES, RESOURCES, Board, BoardValidationError, build_topology,
)
from catan.catan_engine import CatanEngine
from catan.state_manager import (
    BANK_START, CommandError, GameState, HandBelief, parse_command, parse_resource_list,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "example_board.json"


@pytest.fixture
def data() -> dict:
    return json.loads(EXAMPLE.read_text())


@pytest.fixture
def game(data: dict) -> GameState:
    return GameState.from_dict(data)


def assert_conservation(g: GameState) -> None:
    """Bank + every possible joint hand assignment sums to 19 of each resource."""
    for state in g.belief.states:
        for i in range(len(RESOURCES)):
            assert g.bank[i] + sum(h[i] for h in state) == BANK_START


# -- topology -----------------------------------------------------------------


def test_topology_counts_and_invariants() -> None:
    t = build_topology()
    assert (t.num_hexes, t.num_nodes, t.num_edges, len(t.coastal_edges)) == (19, 54, 72, 30)
    assert all(q + r + (-q - r) == 0 and max(abs(q), abs(r), abs(q + r)) <= 2 for q, r in t.hex_coords)
    assert t.hex_coords[0] == (0, -2) and t.hex_coords[18] == (0, 2) and t.hex_coords[9] == (0, 0)
    assert t.hex_nodes[0][0] == 0 and t.hex_nodes[18][3] == 53  # top corner of hex 0, bottom of hex 18
    assert all(len(set(ns)) == 6 for ns in t.hex_nodes)
    assert sorted(len(n) for n in t.node_neighbors).count(2) == 18  # coastal corners with degree 2
    assert all(len(h) in (1, 2, 3) for h in t.node_hexes)
    # each hex's consecutive corners are joined by an edge
    for ns in t.hex_nodes:
        for i in range(6):
            t.edge_between(ns[i], ns[(i + 1) % 6])
    slots = [t.edge_nodes[t.coastal_edges[i]] for i in PORT_SLOT_INDICES]
    nodes = [n for pair in slots for n in pair]
    assert len(nodes) == len(set(nodes)) == 18
    # reading order: node ids increase top-to-bottom
    ys = [t.node_lattice[n][1] for n in range(t.num_nodes)]
    assert ys == sorted(ys)


def test_board_validation(data: dict) -> None:
    board = Board.from_dict(data)
    assert board.validate() == [] or all("adjacent" in w for w in board.validate())
    bad = json.loads(json.dumps(data))
    bad["hexes"][0]["number"] = 7
    with pytest.raises(BoardValidationError):
        Board.from_dict(bad).validate()
    bad = json.loads(json.dumps(data))
    bad["hexes"][9]["number"] = 5  # desert with a number
    with pytest.raises(BoardValidationError):
        Board.from_dict(bad).validate()


def test_example_number_distribution(data: dict) -> None:
    counts: dict[int, int] = {}
    for h in data["hexes"]:
        if h["number"]:
            counts[h["number"]] = counts.get(h["number"], 0) + 1
    assert counts == NUMBER_COUNTS


# -- state manager --------------------------------------------------------------


def test_load_grants_second_settlement_resources(game: GameState) -> None:
    # red's second settlement (node 20) touches brick 10 and ore 8 (+ water)
    assert game.expected_hand("red") == [0, 1, 0, 0, 1]
    assert game.public_vp("red") == 2
    assert_conservation(game)


def test_roll_distribution_and_robber_block(game: GameState) -> None:
    out = game.execute("d 8")
    assert "red +2or" in out and "blue +2wo" in out
    assert game.expected_hand("red")[4] == 3
    game.execute("n")
    game.execute("d 7")
    game.execute("r 11 red o")  # mountains 8; steal ore seen by perspective
    game.execute("n")
    before = game.expected_hand("red")
    out = game.execute("d 8")
    assert "robber blocks hex 11" in out
    assert game.expected_hand("red") == before  # ore 8 blocked
    assert_conservation(game)


def test_seven_requires_discards(game: GameState) -> None:
    game.execute("g blue 2w 2b 2s 2g")  # blue now 11 cards
    out = game.execute("d 7")
    assert game.phase == "discard" and game.pending_discards == {"blue": 5}
    assert "blue 5" in out
    game.execute("x blue 2w 2b g")
    assert game.phase == "robber" and game.hand_total("blue") == 6
    game.execute("r 4 white")
    assert game.phase == "main" and game.board.robber_hex == 4
    assert_conservation(game)


def test_unknown_steal_branches_and_build_prunes(game: GameState) -> None:
    game.execute("g blue 1o")            # blue: 2 wood, 1 wheat, 1 ore
    game.execute("d 7")
    game.execute("r 15 blue")            # red (current player) steals an unseen card from blue
    assert not game.belief.is_exact(game.idx("red"))
    assert game.hand_total("red") == 3
    assert game.expected_hand("red")[0] == pytest.approx(0.5)  # 2 of blue's 4 cards were wood
    # red builds a road: needs wood + brick -> stolen card must have been wood
    game.execute("b red road 38")
    assert game.belief.is_exact(game.idx("red")) and game.belief.is_exact(game.idx("blue"))
    assert game.expected_hand("blue") == [1, 0, 0, 1, 1]
    assert_conservation(game)


def test_unknown_discard_branching() -> None:
    b = HandBelief(2)
    b.add(0, (2, 1, 0, 0, 1))
    b.discard_unknown(0, 2)
    assert b.total(0) == 2
    assert abs(sum(b.expected(0)) - 2) < 1e-9
    # hypergeometric expectation: each card kept with prob 2/4
    assert b.expected(0) == pytest.approx([1.0, 0.5, 0.0, 0.0, 0.5])


def test_build_legality_and_costs(game: GameState) -> None:
    with pytest.raises(CommandError):
        game.execute("b red settlement 25")       # adjacent to own settlement 20/31 (distance rule)
    with pytest.raises(CommandError):
        game.execute("b red road 0")              # not connected
    game.execute("g red w b s g")
    game.execute("b red road 38")                 # 26-32, from settlement 20 via road 32 (20-26)
    assert game.can_place_settlement("red", 32)
    game.execute("g red w b")
    game.execute("b red s 32")
    assert game.public_vp("red") == 3
    with pytest.raises(CommandError):
        game.execute("b red city 14")             # orange's settlement
    assert_conservation(game)


def test_port_trade_rates(game: GameState) -> None:
    # red settlement 20 sits on the 2:1 ore port (nodes 20-26)
    from catan.board import Resource
    assert game.trade_rate("red", Resource.ORE) == 2
    assert game.trade_rate("red", Resource.WOOD) == 4
    game.execute("g red o")
    out = game.execute("t o g")
    assert "2 ore -> 1 wheat (2:1)" in out
    assert game.expected_hand("red") == [0, 1, 0, 1, 0]
    assert_conservation(game)


def test_player_trade_and_dev_cards(game: GameState) -> None:
    game.execute("tp red blue o w")
    assert game.expected_hand("red")[0] == 1 and game.expected_hand("blue")[4] == 1
    game.execute("g red s g o")
    game.execute("b red dev knight")
    assert game.players["red"].dev_hidden == 1 and game.dev_deck == 24
    game.execute("p red knight")
    assert game.phase == "robber" and game.players["red"].knights_played == 1
    assert_conservation(game)


def test_inconsistent_input_assumes_missed_income(game: GameState) -> None:
    out = game.execute("b red city 20")  # red holds 1 brick 1 ore: a missed roll is assumed
    assert "assuming missed income" in out
    assert 20 in game.players["red"].cities
    assert_conservation(game)


def test_undo(game: GameState) -> None:
    before = game.expected_hand("blue")
    game.execute("d 8")
    with pytest.raises(CommandError):
        game.execute("b red road 0")  # a failed command must not touch state or undo history
    game.execute("u")
    assert game.expected_hand("blue") == before and game.last_roll is None


def test_parser_aliases(game: GameState) -> None:
    assert parse_resource_list(["2w+b", "ore:2", "g"]) == (2, 1, 0, 1, 2)
    ev = parse_command("t w o", game)            # 'w' must be read as wood, not white
    assert ev["give"] == "wood" and ev["player"] is None
    ev = parse_command("t white 3w o", game)
    assert ev["player"] == "white" and ev["rate"] == 3
    ev = parse_command("b o city 14", game)      # 'o' as a player prefix -> orange
    assert ev["player"] == "orange" and ev["item"] == "city"
    assert parse_command("r 11 bl", game)["victim"] == "blue"


def _road_len_after(game: GameState, edge: int) -> int:
    g2 = game.clone()
    g2.players["red"].roads.add(edge)
    return g2.longest_road("red")


def test_longest_road_award(game: GameState) -> None:
    game.execute("g red 5w 5b")
    for _ in range(5):  # greedily extend the longest road with legal roads
        best = max(game.legal_road_edges("red"), key=lambda e: _road_len_after(game, e))
        game.execute(f"b red road {best}")
        if game.longest_road("red") >= 5:
            break
    assert game.longest_road("red") >= 5
    assert game.longest_road_owner == "red"
    assert game.public_vp("red") == 4


def test_state_roundtrip(game: GameState) -> None:
    game.execute("d 8")
    g2 = GameState.from_dict(game.to_dict())
    assert g2.expected_hand("red") == game.expected_hand("red")
    assert g2.players["blue"].roads == game.players["blue"].roads


# -- engine -----------------------------------------------------------------------


def test_settlement_scores_and_road_plans(game: GameState) -> None:
    eng = CatanEngine(game, rollouts=0)
    scores = eng.settlement_scores("red", require_road=False)
    assert scores == sorted(scores, key=lambda s: -s.value)
    assert all(game.can_place_settlement("red", s.node, require_road=False) for s in scores)
    plans = eng.road_plans("red")
    assert plans and all(p.cost == len(p.edges) for p in plans)
    first = plans[0].edges[0]
    assert game.can_place_road("red", first)


def test_robber_avoids_own_hexes(game: GameState) -> None:
    eng = CatanEngine(game, rollouts=0)
    topo = game.board.topology
    for opt in eng.robber_options("red", top=3):
        own = [n for n in topo.hex_nodes[opt.hex] if game.node_owner(n) and game.node_owner(n)[0] == "red"]
        assert not own
        assert opt.hex != game.board.robber_hex


def test_recommend_top3_with_city(game: GameState) -> None:
    game.execute("g red 2g 2o")  # red: 1 brick, 3 ore, 2 wheat -> city affordable
    rec = CatanEngine(game, rollouts=12, mc_top_k=3).recommend(top_k=3)
    assert len(rec.moves) == 3
    assert "City" in rec.best.label
    text = rec.render()
    assert "Optimal Move: [" in text and "Alternative: [" in text and "Expected Win %" in text
    assert 0.0 <= rec.win_prob <= 1.0
    assert abs(sum(rec.all_win_probs.values()) - 1.0) < 1e-9


def test_recommend_robber_phase(game: GameState) -> None:
    game.execute("d 7")
    rec = CatanEngine(game, rollouts=0).recommend()
    assert rec.phase == "robber" and rec.best.label.startswith("Move Robber")
    ev = rec.best.events[0]
    game.apply_event(ev)
    assert game.phase == "main"


def test_trade_evaluator_prefers_needed_resource(game: GameState) -> None:
    eng = CatanEngine(game, rollouts=0)
    game.execute("g red 2o")  # red: 1 brick 3 ore, no wheat
    good = eng.evaluate_trade("red", [0, 0, 0, 0, 1], [0, 0, 0, 1, 0])   # ore -> wheat
    bad = eng.evaluate_trade("red", [0, 1, 0, 0, 0], [0, 0, 0, 0, 1])    # brick -> ore
    assert good > 0 > bad


def test_simulator_terminates_and_is_seeded(game: GameState) -> None:
    eng = CatanEngine(game, rollouts=10, seed=3)
    a = eng.simulate(game, "red")
    b = eng.simulate(game, "red")
    assert a == b and abs(sum(a.values()) - 1.0) < 1e-9


def test_heuristic_selfplay_smoke(game: GameState) -> None:
    rng = random.Random(5)
    for _ in range(4 * 6):
        roll = rng.randint(1, 6) + rng.randint(1, 6)
        game.execute(f"d {roll}")
        if roll == 7:
            eng = CatanEngine(game, rollouts=0)
            for c, n in list(game.pending_discards.items()):
                game.apply_event({"type": "discard", "player": c, "cards": tuple(eng.suggest_discard(c, n))})
            game.apply_event(CatanEngine(game, rollouts=0).recommend().best.events[0])
        for ev in CatanEngine(game, rollouts=0).recommend().best.events:
            game.apply_event(ev)
        game.execute("n")
        assert_conservation(game)
    assert sum(game.public_vp(c) for c in game.order) > 8


# -- vision -----------------------------------------------------------------------


@pytest.mark.parametrize("size", [40, 64, 96])
def test_vision_roundtrip_on_synthetic_screenshot(data: dict, size: int) -> None:
    pytest.importorskip("cv2")
    from catan.vision_extractor import BoardVisionExtractor, VisionConfig, render_synthetic_board

    data = json.loads(json.dumps(data))
    data["robber_hex"] = 4
    data["players"][0]["cities"] = [data["players"][0]["settlements"].pop()]
    img = render_synthetic_board(data, size=size)
    res = BoardVisionExtractor(VisionConfig(ocr="template")).extract(img)
    out = res.data
    for want, got in zip(data["hexes"], out["hexes"]):
        assert (want["terrain"], want["number"]) == (got["terrain"], got["number"]), got
    assert out["robber_hex"] == 4
    assert sorted((p["type"], tuple(p["nodes"])) for p in out["ports"]) == \
        sorted((p["type"], tuple(p["nodes"])) for p in data["ports"])
    for want, got in zip(data["players"], out["players"]):
        for key in ("settlements", "cities", "roads"):
            assert sorted(want.get(key, [])) == sorted(got[key]), (want["color"], key)
    GameState.from_dict(out)  # extracted JSON loads into the state manager
