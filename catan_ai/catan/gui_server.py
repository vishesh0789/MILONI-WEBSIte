"""Local analysis-board GUI (Stockfish/lichess style) for the Catan engine.

    python -m catan.gui_server [board.json] [--me red] [--port 8765]

Starts a standard-library HTTP server and opens the browser.  The page in
``catan/gui/index.html`` draws the board, lets you play moves by clicking, and
asks the engine for analysis; everything goes through the same
``GameState`` / ``CatanEngine`` objects the CLI uses.

JSON API (all POST bodies are JSON unless noted)::

    GET  /api/state                      full board + game state
    POST /api/command   {"line": "d 8"}  short command (same syntax as the CLI)
    POST /api/apply     {"events": [...]} apply engine events (e.g. "play this line")
    POST /api/undo
    POST /api/analyze   {"player": null, "rollouts": 80, "top_k": 5}
    POST /api/load      <board/game JSON>
    POST /api/new                        reload the example board
    POST /api/extract   <PNG/JPEG bytes> run the screenshot extractor and load the result
    GET  /api/save                       current game as JSON (loadable again)
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .board import RESOURCES
from .catan_engine import CatanEngine
from .state_manager import CommandError, GameState

HERE = Path(__file__).resolve().parent
PAGE = HERE / "gui" / "index.html"
EXAMPLE = HERE.parent / "examples" / "example_board.json"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return obj


class Session:
    """One game shared by the page; a lock serialises commands and analysis."""

    def __init__(self, game: GameState):
        self.lock = threading.Lock()
        self.game = game
        self.feed: list[dict[str, Any]] = []

    def record(self, line: str, message: str, ok: bool = True) -> None:
        self.feed.append({"turn": self.game.turn_number, "player": self.game.current_color,
                          "line": line, "message": message, "ok": ok})
        self.feed = self.feed[-200:]

    # -- serialisation --------------------------------------------------------
    def state(self) -> dict[str, Any]:
        g = self.game
        topo = g.board.topology
        players = []
        for c in g.order:
            p = g.players[c]
            i = g.idx(c)
            players.append({
                "color": c,
                "settlements": sorted(p.settlements), "cities": sorted(p.cities), "roads": sorted(p.roads),
                "vp": g.public_vp(c), "hidden_vp": round(g.expected_hidden_vp(c), 2),
                "cards": g.hand_total(c), "hand": [round(x, 2) for x in g.belief.expected(i)],
                "exact": g.belief.is_exact(i), "dev": p.dev_hidden, "dev_known": p.dev_known,
                "knights": p.knights_played, "road_length": g.longest_road(c),
                "longest_road": g.longest_road_owner == c, "largest_army": g.largest_army_owner == c,
                "ports": sorted({port.type for port in g.player_ports(c)}),
            })
        return {
            "topology": {
                "hexes": [{"id": h, "q": q, "r": r, "xy": topo.hex_center_unit(q, r)}
                          for h, (q, r) in enumerate(topo.hex_coords)],
                "nodes": [{"id": n, "xy": topo.node_center_unit(n), "hexes": topo.node_hexes[n]}
                          for n in range(topo.num_nodes)],
                "edges": [{"id": e, "nodes": topo.edge_nodes[e]} for e in range(topo.num_edges)],
                "hex_nodes": topo.hex_nodes,
            },
            "tiles": [{"id": t.id, "terrain": t.terrain.value, "number": t.number, "pips": t.pips}
                      for t in g.board.tiles],
            "ports": [{"type": p.type, "nodes": p.nodes, "edge": topo.edge_between(*p.nodes)}
                      for p in g.board.ports],
            "robber": g.board.robber_hex,
            "order": g.order, "current": g.current_color, "phase": g.phase, "turn": g.turn_number,
            "last_roll": g.last_roll, "perspective": g.perspective,
            "pending_discards": g.pending_discards, "bank": g.bank, "dev_deck": g.dev_deck,
            "players": players, "resources": [r.value for r in RESOURCES],
            "can_undo": bool(g._history), "feed": self.feed[-80:],
        }

    def analyze(self, player: str | None, rollouts: int, top_k: int) -> dict[str, Any]:
        g = self.game
        color = player or g.current_color
        t0 = time.time()
        eng = CatanEngine(g, rollouts=rollouts)
        rec = eng.recommend(color, top_k=top_k)
        heur = CatanEngine(g, rollouts=0)
        heat = [{"node": s.node, "value": round(s.value, 2), "pips": s.pips}
                for s in heur.settlement_scores(color, require_road=False)]
        plans = [{"target": p.target, "edges": p.edges, "cost": p.cost, "contested": p.contested,
                  "score": round(p.score, 2)} for p in heur.road_plans(color, top=3)]
        return {
            "player": color, "phase": rec.phase,
            "moves": [{"label": m.label, "events": _jsonable(m.events), "win": m.win_prob,
                       "heuristic": round(m.heuristic, 3), "detail": m.detail} for m in rec.moves],
            "win": rec.win_prob, "table": rec.all_win_probs, "position_score": round(rec.position_score, 2),
            "scores": {c: round(heur.position_score(c), 2) for c in g.order},
            "heatmap": heat, "plans": plans,
            "resource_values": heur.resource_values(color),
            "notes": rec.notes, "rollouts": rollouts, "seconds": round(time.time() - t0, 2),
            "text": rec.render(),
        }


def make_handler(session: Session) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(_jsonable(obj)).encode())

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            elif self.path == "/api/state":
                with session.lock:
                    self._json(session.state())
            elif self.path == "/api/save":
                with session.lock:
                    self._json(session.game.to_dict())
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self) -> None:  # noqa: N802
            try:
                raw = self._body()
                with session.lock:
                    result = self._route(raw)
                self._json(result)
            except (CommandError, KeyError, ValueError) as exc:
                with session.lock:
                    self._json({"ok": False, "error": str(exc), "state": session.state()}, 200)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)

        def _route(self, raw: bytes) -> dict[str, Any]:
            g = session.game
            if self.path == "/api/command":
                line = json.loads(raw)["line"].strip()
                if line in ("u", "undo"):
                    msg = "undone" if g.undo() else "nothing to undo"
                else:
                    try:
                        msg = g.execute(line)
                    except (CommandError, KeyError, ValueError) as exc:
                        session.record(line, str(exc), ok=False)
                        raise
                session.record(line, msg)
                return {"ok": True, "message": msg, "state": session.state()}
            if self.path == "/api/apply":
                events = json.loads(raw)["events"]
                label = json.loads(raw).get("label", "engine move")
                msg = g.apply_events(events) if events else "turn passed (no action)"
                session.record(label, msg)
                return {"ok": True, "message": msg, "state": session.state()}
            if self.path == "/api/undo":
                msg = "undone" if g.undo() else "nothing to undo"
                session.record("u", msg)
                return {"ok": True, "message": msg, "state": session.state()}
            if self.path == "/api/analyze":
                req = json.loads(raw or b"{}")
                return {"ok": True, "analysis": session.analyze(
                    req.get("player"), int(req.get("rollouts", 80)), int(req.get("top_k", 5)))}
            if self.path in ("/api/load", "/api/new"):
                data = json.loads(raw) if self.path == "/api/load" else json.loads(EXAMPLE.read_text())
                perspective = g.perspective
                session.game = GameState.from_dict(data)
                session.game.perspective = session.game.perspective or perspective
                session.feed = []
                session.record("load", "board loaded")
                return {"ok": True, "state": session.state()}
            if self.path == "/api/extract":
                import cv2
                import numpy as np

                from .vision_extractor import BoardVisionExtractor

                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("could not decode the image (use PNG or JPEG)")
                res = BoardVisionExtractor().extract(img)
                perspective = g.perspective
                session.game = GameState.from_dict({**res.data, "strict_composition": False})
                session.game.perspective = perspective
                session.feed = []
                session.record("extract", "board read from screenshot; check tiles and numbers")
                return {"ok": True, "warnings": res.warnings, "state": session.state()}
            raise KeyError(f"unknown endpoint {self.path}")

    return Handler


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Catan analysis board GUI")
    ap.add_argument("state", nargs="?", default=str(EXAMPLE), help="board/game JSON (default: example)")
    ap.add_argument("--me", help="your colour (perspective)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ns = ap.parse_args(argv)
    game = GameState.load(ns.state)
    if ns.me:
        game.perspective = game.resolve_player(ns.me)
    session = Session(game)
    server = ThreadingHTTPServer(("127.0.0.1", ns.port), make_handler(session))
    url = f"http://127.0.0.1:{ns.port}/"
    print(f"Catan analysis board running at {url}  (Ctrl+C to stop)")
    if not ns.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
