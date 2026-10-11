#!/usr/bin/env python3
"""End-to-end demo / test runner.

1. (optional) Render the example board as a synthetic Colonist-style screenshot and
   extract it back through the vision pipeline.
2. Load the extracted board JSON into the state manager.
3. Process the dice sequence ``d 6``, ``d 8``, ``d 7`` (+ robber) with short commands.
4. Ask the AI engine for the top-3 moves.

Usage::

    python run_demo.py                     # uses examples/example_board.json
    python run_demo.py --vision            # round-trip through the screenshot extractor
    python run_demo.py --rollouts 200      # stronger (slower) Monte Carlo
    python run_demo.py --selfplay 6        # let the heuristic AI play N extra rounds first
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from catan.board import RESOURCES, build_topology
from catan.catan_engine import CatanEngine
from catan.state_manager import GameState

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE / "examples" / "example_board.json"


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def run(game: GameState, cmd: str) -> None:
    print(f"> {cmd}")
    out = game.execute(cmd)
    if out:
        print("  " + out.replace("\n", "\n  "))


def load_board(use_vision: bool) -> dict:
    data = json.loads(EXAMPLE.read_text())
    if not use_vision:
        return data
    import cv2

    from catan.vision_extractor import BoardVisionExtractor, render_synthetic_board

    shot = HERE / "examples" / "synthetic_screenshot.png"
    cv2.imwrite(str(shot), render_synthetic_board(data, size=64))
    t0 = time.time()
    result = BoardVisionExtractor().extract(str(shot))
    print(f"vision: extracted {shot.name} in {time.time() - t0:.2f}s "
          f"(ocr backend: {result.data['meta']['ocr_backend']})")
    mism = [h["id"] for h, g in zip(result.data["hexes"], data["hexes"])
            if (h["terrain"], h["number"]) != (g["terrain"], g["number"])]
    print(f"vision: hex mismatches vs ground truth: {mism or 'none'}")
    for w in result.warnings:
        print(f"vision warning: {w}")
    # A screenshot cannot reveal placement order / whose turn it is; keep those from the example.
    extracted = result.data
    truth = {p["color"]: p for p in data["players"]}
    for p in extracted["players"]:
        if sorted(p["settlements"]) == sorted(truth[p["color"]]["settlements"]):
            p["settlements"] = truth[p["color"]]["settlements"]
    extracted["perspective"] = data.get("perspective")
    return extracted


def selfplay(game: GameState, rounds: int, seed: int = 11) -> None:
    rng = random.Random(seed)
    for _ in range(rounds * len(game.order)):
        roll = rng.randint(1, 6) + rng.randint(1, 6)
        game.execute(f"d {roll}")
        if roll == 7:
            eng = CatanEngine(game, rollouts=0)
            for c, n in list(game.pending_discards.items()):
                cards = eng.suggest_discard(c, n)
                game.execute(f"x {c} " + " ".join(f"{k}{r.value}" for k, r in zip(cards, RESOURCES) if k))
            ev = CatanEngine(game, rollouts=0).recommend().best.events[0]
            game.execute(f"r {ev['hex']} {ev.get('victim', '')}")
        for _ in range(4):
            move = CatanEngine(game, rollouts=0).recommend().best
            if not move.events:
                break
            for ev in move.events:
                game.apply_event(ev)
        game.execute("n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", action="store_true", help="round-trip through the screenshot extractor")
    ap.add_argument("--rollouts", type=int, default=80)
    ap.add_argument("--selfplay", type=int, default=0, help="heuristic self-play rounds before the demo")
    ns = ap.parse_args()

    banner("1. Load extracted board")
    topo = build_topology()
    print(f"topology: {topo.num_hexes} hexes, {topo.num_nodes} nodes, {topo.num_edges} edges, "
          f"{len(topo.coastal_edges)} coastal edges")
    print("hex 0 (q,r,s) =", (*topo.hex_coords[0], -sum(topo.hex_coords[0])), "corners N..NW ->",
          topo.hex_nodes[0])
    game = GameState.from_dict(load_board(ns.vision))
    print(game.summary())

    engine = CatanEngine(game, rollouts=0)
    print("\nTop open settlement spots (setup-style scoring, ignoring roads):")
    for s in engine.settlement_scores("red", require_road=False, top=5):
        print("  ", s)

    if ns.selfplay:
        banner(f"(self-play {ns.selfplay} rounds with the heuristic AI)")
        selfplay(game, ns.selfplay)
        print(game.summary())

    banner("2. Process dice sequence: d 6, d 8, d 7")
    t0 = time.time()
    first = game.current_color
    run(game, "d 6")
    print("\nAI after the 6 (main phase, " + first + "):")
    print(CatanEngine(game, rollouts=ns.rollouts).recommend().render())
    run(game, "n")
    run(game, "d 8")
    run(game, "n")
    run(game, "d 7")
    print(f"\n(commands processed in {time.time() - t0:.2f}s incl. one AI call)")

    banner("3. AI: robber placement after the 7")
    rec = CatanEngine(game, rollouts=ns.rollouts).recommend(top_k=3)
    print(rec.render())
    best = rec.best.events[0]
    run(game, f"r {best['hex']} {best.get('victim', '')}".strip())
    print(game.summary())

    banner("4. AI: top 3 recommended moves (fast expectimax)")
    t0 = time.time()
    print(CatanEngine(game, mode="fast").recommend(top_k=3).render())
    print(f"(engine time {(time.time() - t0) * 1000:.0f} ms)")

    banner("4b. Second opinion: Monte Carlo rollouts")
    t0 = time.time()
    rec = CatanEngine(game, rollouts=ns.rollouts).recommend(top_k=3)
    print(rec.render())
    print(f"(engine time {time.time() - t0:.2f}s, {ns.rollouts} rollouts per candidate)")

    me = game.perspective or "red"
    print(f"\nPerspective player ({me}) planning view:")
    eng = CatanEngine(game, rollouts=0)
    for p in eng.road_plans(me):
        print(f"  road plan -> node {p.target}: build edges {p.edges} ({p.cost} roads, {p.contested}, "
              f"score {p.score:.2f})")
    print("  resource values:", eng.resource_values(me))
    give, get = [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]
    print(f"  trade 1 ore -> 1 wheat with blue: net {eng.evaluate_trade(me, give, get, partner='blue'):+.3f}")
    print("\nWin probability (Monte Carlo, current position):",
          {c: f"{p * 100:.1f}%" for c, p in CatanEngine(game, rollouts=ns.rollouts).simulate(
              game, me, start_next_turn=True).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
