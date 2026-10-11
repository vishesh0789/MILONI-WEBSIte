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


def test_pre_roll_advice_is_to_roll(game: GameState) -> None:
    rec = CatanEngine(game, rollouts=6).recommend()
    assert rec.best.label == "Roll the dice" and rec.best.win_prob is not None


def test_recommend_top3_with_city(game: GameState) -> None:
    game.execute("g red 2g 2o")  # red: 1 brick, 3 ore, 2 wheat -> city affordable
    game.execute("d 12")         # nobody sits on the 12
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


# -- GUI server -------------------------------------------------------------------


def test_gui_server_api(data: dict) -> None:
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from catan.gui_server import Session, make_handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Session(GameState.from_dict(data), autosave_path=None)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path: str, body: dict) -> dict:
        req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    try:
        assert b"Catan Analysis Board" in urllib.request.urlopen(base + "/").read()
        state = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert len(state["tiles"]) == 19 and len(state["topology"]["nodes"]) == 54
        res = post("/api/command", {"line": "d 8"})
        assert res["ok"] and res["state"]["last_roll"] == 8
        assert not post("/api/command", {"line": "b red s 0"})["ok"]  # illegal -> error, state intact
        res = post("/api/analyze", {"rollouts": 4, "top_k": 3})
        moves = res["analysis"]["moves"]
        assert res["ok"] and 1 <= len(moves) <= 3 and res["analysis"]["heatmap"]
        res = post("/api/apply", {"events": moves[0]["events"], "label": moves[0]["label"]})
        assert res["ok"]
        assert post("/api/undo", {})["state"]["can_undo"] in (True, False)
    finally:
        server.shutdown()


# -- setup, corrections, persistence ---------------------------------------------


@pytest.fixture
def blank(data: dict) -> GameState:
    d = json.loads(json.dumps(data))
    d["players"] = []
    return GameState.from_dict(d)


def test_setup_snake_draft(blank: GameState) -> None:
    g = blank
    assert g.phase == "setup" and g.setup_next() == ("red", "settlement")
    with pytest.raises(CommandError):
        g.execute("8")  # no dice during setup
    eng = CatanEngine(g, rollouts=0)
    for _ in range(16):
        rec = eng.recommend()
        assert rec.phase == "setup" and rec.moves
        g.apply_events(rec.best.events)
    assert g.phase == "main" and g.current_color == "red" and g.last_roll is None
    assert all(len(p.settlements) == 2 and len(p.roads) == 2 for p in g.players.values())
    # second settlements paid their starting cards (snake: white placed 5th, so white's 2nd is 5th too)
    assert all(g.hand_total(c) > 0 for c in g.order)
    assert_conservation(g)


def test_setup_road_must_touch_new_settlement(blank: GameState) -> None:
    blank.execute("b red s 20")
    with pytest.raises(CommandError):
        blank.execute("b red r 0")
    blank.execute("b red r 20-26")
    assert blank.setup_next() == ("blue", "settlement")


def test_board_edit_commands(game: GameState) -> None:
    game.execute("tile 4 mountains 11")
    assert game.board.tiles[4].terrain.value == "mountains" and game.board.tiles[4].number == 11
    assert game.board_warnings()  # composition no longer standard -> a warning, not an error
    game.execute("tile 4 hills 6")
    game.execute("tile 9 robber")
    with pytest.raises(CommandError):
        game.execute("tile 4 7")
    edge = game.board.topology.coastal_edges[1]
    game.execute(f"port {edge} ore")
    assert any(p.type == "ore" for p in game.board.ports)
    game.execute("piece 0 blue city")
    assert 0 in game.players["blue"].cities and game.public_vp("blue") == 4
    game.execute("piece 0 none")
    game.execute("piece 26-32 white")
    assert game.board.topology.edge_between(26, 32) in game.players["white"].roads
    game.execute("u")
    assert game.board.topology.edge_between(26, 32) not in game.players["white"].roads


def test_set_hand_and_devs(game: GameState) -> None:
    game.execute("hand blue 2w 1o")
    assert game.expected_hand("blue") == [2, 0, 0, 0, 1]
    assert_conservation(game)
    game.execute("hand orange 5")
    assert game.hand_total("orange") == 5 and not game.belief.is_exact(game.idx("orange"))
    deck = game.dev_deck
    game.execute("devs white 2 3")
    assert game.players["white"].dev_hidden == 2 and game.dev_deck == deck - 5
    assert game.largest_army_owner == "white"


def test_save_load_keeps_phase_devs_and_uncertainty(game: GameState) -> None:
    game.execute("8")
    game.execute("devs orange 1 0")
    game.execute("n")
    game.execute("7")
    with pytest.raises(CommandError):
        game.execute("r 16 blue")  # blue is moving the robber: cannot rob itself
    game.execute("r 15 red")       # unknown steal -> uncertain hands
    g2 = GameState.from_dict(json.loads(json.dumps(game.to_dict())))
    assert g2.dev_deck == game.dev_deck and g2.phase == game.phase == "main"
    assert not g2.belief.is_exact(g2.idx("red"))
    assert g2.expected_hand("red") == pytest.approx(game.expected_hand("red"))
    assert g2.bank == game.bank


def test_random_board_is_legal() -> None:
    from catan.board import random_board_dict

    for seed in range(5):
        d = random_board_dict(seed)
        board = Board.from_dict(d)
        assert board.validate() == []  # standard composition and no adjacent 6/8
        assert GameState.from_dict(d).phase == "setup"


def test_offer_is_scored_as_maybe_accepted(game: GameState) -> None:
    game.execute("d 12")
    rec = CatanEngine(game, rollouts=8, mc_top_k=6).recommend(top_k=8)
    for m in rec.moves:
        if m.kind == "offer" and m.win_prob is not None:
            assert "if accepted" in m.detail


def test_analysis_can_be_cancelled(game: GameState) -> None:
    from catan.catan_engine import AnalysisCancelled

    game.execute("d 12")
    with pytest.raises(AnalysisCancelled):
        CatanEngine(game, rollouts=50, should_stop=lambda: True).recommend()


def test_gui_new_game_and_autosave(data: dict, tmp_path: Path) -> None:
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from catan.gui_server import Session, make_handler

    save = tmp_path / "autosave.json"
    session = Session(GameState.from_dict(data), autosave_path=save)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path: str, body: dict) -> dict:
        req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    try:
        res = post("/api/new", {"kind": "random"})
        assert res["ok"] and res["state"]["phase"] == "setup" and res["state"]["setup_next"] == ["red", "settlement"]
        res = post("/api/analyze", {"rollouts": 20})
        assert res["analysis"]["phase"] == "setup" and res["analysis"]["moves"][0]["kind"] == "setup"
        node = res["analysis"]["moves"][0]["events"][0]["target"]
        assert post("/api/command", {"line": f"b red s {node}"})["ok"]
        assert save.exists() and json.loads(save.read_text())["phase"] == "setup"
        res = post("/api/new", {"kind": "example"})
        assert res["state"]["previous"] and res["state"]["previous"]["phase"] == "setup"
        res = post("/api/new", {"kind": "previous"})  # back to the game we just left
        assert res["ok"] and res["state"]["phase"] == "setup"
        assert node in res["state"]["players"][0]["settlements"]
        res = post("/api/new", {"kind": "previous"})  # and forth again
        assert res["state"]["phase"] == "main"
    finally:
        server.shutdown()


# -- real Colonist.io screenshot ------------------------------------------------------

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize("variant", ["original", "small", "large", "jpeg", "padded", "darker", "warm"])
def test_vision_on_real_colonist_screenshot(variant: str) -> None:
    cv2 = pytest.importorskip("cv2")
    from catan.vision_extractor import BoardVisionExtractor

    img = cv2.imread(str(FIXTURES / "colonist_mobile_board.jpg"))
    truth = json.loads((FIXTURES / "colonist_mobile_board.json").read_text())
    if variant == "small":
        img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    elif variant == "large":
        img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    elif variant == "jpeg":
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 35])[1], cv2.IMREAD_COLOR)
    elif variant == "darker":  # photo of a screen / dim capture
        img = (img * 0.65).astype("uint8")
    elif variant == "warm":    # colour cast
        img = (img * [0.85, 0.97, 1.0]).clip(0, 255).astype("uint8")
    elif variant == "padded":  # desktop-style wide canvas
        img = cv2.copyMakeBorder(img, 300, 300, 500, 500, cv2.BORDER_CONSTANT, value=tuple(int(v) for v in img[5, 5]))
    data = BoardVisionExtractor().extract(img).data
    assert [[h["terrain"], h["number"]] for h in data["hexes"]] == truth["hexes"]
    assert data["robber_hex"] == truth["robber_hex"]
    assert sorted([p["type"], p["nodes"]] for p in data["ports"]) == sorted(truth["ports"])
    assert not any(p["settlements"] or p["cities"] or p["roads"] for p in data["players"])  # empty board
    GameState.from_dict(data)


# -- free control: names, order, any setup order, editable log --------------------------


def test_names_order_and_free_setup_order(blank: GameState) -> None:
    g = blank
    g.execute("rules setup=free")       # the snake draft is the default; this room places in any order
    g.execute("name blue Coron")
    assert g.resolve_player("coron") == "blue" and g.resolve_player("#blue") == "blue"
    g.execute("b coron s 0")            # blue first, not red: any order is fine
    g.execute("b blue r 0-3")
    g.execute("b blue s 28")            # blue's second settlement pays starting cards right away
    assert g.hand_total("blue") > 0
    assert g.pending_setup("blue") == "road" and g.pending_setup("red") == "settlement"
    rec = CatanEngine(g, rollouts=0).recommend("orange")
    assert rec.player == "orange" and rec.phase == "setup"
    g.execute("order white orange blue red")
    assert g.order == ["white", "orange", "blue", "red"]
    assert g.hand_total("blue") > 0     # hands follow their players when the order changes
    assert_conservation(g)


def test_devs_with_vp_cards(game: GameState) -> None:
    deck = game.dev_deck
    game.execute("devs red 1 0 2")
    assert game.players["red"].vp_cards == 2 and game.public_vp("red") == 4 and game.dev_deck == deck - 3
    game.execute("devs red 0 0 0")
    assert game.public_vp("red") == 2 and game.dev_deck == deck


def test_session_log_delete_rewind_and_recolor(data: dict) -> None:
    from catan.gui_server import Session

    s = Session(GameState.from_dict(data), autosave_path=None)
    s.run_command("8")                    # red +2 ore, blue +2 wood
    s.run_command("g red 2wheat")
    s.run_command("b red city 20")        # paid by the 8 and the gain
    s.run_command("n")
    assert 20 in s.game.players["red"].cities
    roll = s.actions[0]["id"]
    problems = s.delete(roll)            # the city really happened: kept, with a note about missing cards
    assert any("b red city 20" in p and "missed income" in p for p in problems)
    assert 20 in s.game.players["red"].cities and s.game.current_color == "blue"
    s.undo()                              # undo the end-of-turn
    assert s.game.current_color == "red"
    s.recolor({"orange": "black"})
    assert "black" in s.game.order and "orange" not in s.game.order
    assert s.game.players["black"].settlements == {14, 40}
    s.run_command("b black road 16")
    gain = next(a["id"] for a in s.actions if a.get("line", "").startswith("g red"))
    s.rewind(gain)                        # back to before the gain
    assert all(a["id"] < gain or a["kind"] not in ("command", "apply") for a in s.actions)
    assert s.game.expected_hand("red") == [0, 1, 0, 0, 1]


def test_session_delete_drops_moves_that_no_longer_fit(data: dict) -> None:
    from catan.gui_server import Session

    d = json.loads(json.dumps(data))
    d["players"] = []
    s = Session(GameState.from_dict(d), autosave_path=None)
    s.run_command("b red s 20")
    s.run_command("b red r 20-26")
    problems = s.delete(s.actions[0]["id"])   # remove the settlement: its road has nothing to attach to
    assert problems and "dropped" in problems[0]
    assert not s.game.players["red"].roads and not s.actions


# -- room rules ---------------------------------------------------------------------


def test_room_rules_discard_limit_friendly_robber_and_winner(game: GameState) -> None:
    game.execute("g blue 2w 2b 2s 2g")          # blue: 11 cards
    game.execute("rules discard=11")
    game.execute("7")
    assert game.phase == "robber"                # 11 is not above the limit of 11
    game.execute("u")
    game.execute("rules discard=7 friendly=on")
    game.execute("7")
    assert game.phase == "discard" and game.pending_discards == {"blue": 5}
    game.execute("x blue 5")
    assert game.friendly_blocked("red")          # everyone else has 2 VP: their tiles are protected
    with pytest.raises(CommandError):
        game.execute("r 16 blue")
    opts = CatanEngine(game, rollouts=0).robber_options("red", top=19)
    assert all(o.hex not in game.friendly_blocked("red") for o in opts)
    assert game.winner() is None
    game.execute("rules vp=4")
    game.execute("devs red 0 0 2")               # 2 settlements + 2 VP cards shown = 4
    assert game.winner() == "red"
    g2 = GameState.from_dict(json.loads(json.dumps(game.to_dict())))
    assert g2.rules == {"vp_target": 4, "discard_limit": 7, "friendly_robber": True, "snake_setup": True}


def test_simulator_uses_room_vp_target(game: GameState) -> None:
    import random as _random

    from catan.catan_engine import _Sim, _SimStatic

    game.execute("12")
    game.execute("rules vp=4")
    sim = _Sim(_SimStatic(game.board), game, _random.Random(0))
    assert sim.win_vp == 4
    winner = sim.run(game.current, 60)
    assert len(winner) == 1 and sim.vp(next(iter(winner))) >= 4   # someone reached the room's target


# -- snake draft, narrated events, fast engine ----------------------------------------------


def test_snake_draft_deduces_first_player_and_locks_order(blank: GameState) -> None:
    g = blank
    g.execute("b blue s 0")                      # blue places first -> blue is player 1
    assert g.order == ["blue", "orange", "white", "red"]
    with pytest.raises(CommandError):
        g.execute("b red s 20")                  # blue still owes a road
    plan = [("blue", "r 0-3"), ("orange", "s 40"), ("orange", "r 40-45"), ("white", "s 14"), ("white", "r 14-9"),
            ("red", "s 20"), ("red", "r 20-26"),                       # round 2 starts with the last player again
            ("red", "s 31"), ("red", "r 31-25"), ("white", "s 48"), ("white", "r 48-44"),
            ("orange", "s 17"), ("orange", "r 17-22"), ("blue", "s 28"), ("blue", "r 28-34")]
    for who, what in plan:
        assert g.setup_next()[0] == who
        g.execute(f"b {who} {what}")
    assert g.phase == "main" and g.current_color == "blue" and g.order_locked
    assert g.hand_total("red") == 3              # node 31's three tiles, from the second settlement only
    msg = g.execute("8")
    assert g.current_color == "blue" and "rolled 8" in msg
    g.execute("orange rolls 6")                  # a named roll moves the turn on
    assert g.current_color == "orange"
    assert_conservation(g)


def test_first_player_can_be_set_before_setup(blank: GameState) -> None:
    blank.execute("first white")
    assert blank.order == ["white", "red", "blue", "orange"] and blank.setup_next() == ("white", "settlement")
    with pytest.raises(CommandError):
        blank.execute("b red s 20")


def test_narrated_events(game: GameState) -> None:
    game.execute("name orange Derbe")
    game.execute("red rolls 8")
    assert game.expected_hand("red")[4] == 3
    game.execute("blue rolls 6")
    assert game.current_color == "blue"
    game.execute("derbe rolled 7 -> robber to (2,0) steal from red ore")      # hex 11 by axial coordinate
    assert game.board.robber_hex == 11 and game.current_color == "orange" and game.phase == "main"
    assert game.expected_hand("orange")[4] == 2
    game.execute("orange trades 4 wheat for ore") if game.expected_hand("orange")[3] >= 4 else None
    game.execute("white rolls 4")
    game.execute("g white wood brick")
    game.execute("white builds road 12-17")
    assert game.board.topology.edge_between(12, 17) in game.players["white"].roads
    game.execute("g white sheep wheat ore")
    game.execute("white buys dev_card")
    assert game.players["white"].dev_hidden == 1
    game.execute("white plays knight to hex 6 steal from red")
    assert game.board.robber_hex == 6 and game.players["white"].knights_played == 1
    game.execute("white ends turn")
    assert game.current_color == "red"
    with pytest.raises(CommandError):
        game.execute("red rolls 15")
    with pytest.raises(CommandError):
        game.execute("someone does something rolls")
    assert_conservation(game)


def test_short_commands_still_win_over_narration(game: GameState) -> None:
    game.execute("tile 9 robber")                # contains "robber" but is a short command
    game.execute("rules discard=9")
    assert game.rules["discard_limit"] == 9


def test_fast_engine_speed_memory_and_choice(game: GameState) -> None:
    import time
    import tracemalloc

    from catan.fast_engine import FastEngine

    game.execute("g red 2wheat 2ore")
    game.execute("12")
    t0 = time.perf_counter()
    rec = CatanEngine(game, mode="fast").recommend(top_k=3)
    assert time.perf_counter() - t0 < 1.0                         # sub-second (typically ~20 ms)
    assert rec.engine == "fast" and "City" in rec.best.label
    tracemalloc.start()
    FastEngine(game).recommend("red")
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 20 * 1024 * 1024                                # far under the 500 MB budget
    est = FastEngine(game).win_estimate()
    assert abs(sum(est.values()) - 1) < 1e-9


def test_fast_engine_offers_table_trade_one_card_short(game: GameState) -> None:
    game.execute("12")
    game.execute("hand red 3wheat 2ore")
    rec = CatanEngine(game, mode="fast").recommend("red", top_k=5)
    offers = [m for m in rec.moves if m.kind == "offer"]
    assert offers and offers[0].label == "Offer Trade: 1 wheat for 1 ore"
    assert offers[0].offer == {"give": [0, 0, 0, 1, 0], "get": [0, 0, 0, 0, 1]}
    assert rec.best.kind == "offer"                               # nothing else gets red closer this turn


def test_bitboard_legal_spots_match_rules(game: GameState) -> None:
    from catan.fast_engine import FastEngine, bits

    game.execute("g red 5w 5b")
    for e in (38, 47):
        game.execute(f"b red road {e}")
    fe = FastEngine(game)
    pos = fe.pos["red"]
    assert set(bits(fe.legal_settlements("red", pos, fe.occupied()))) == set(game.legal_settlement_nodes("red"))
