"""Local analysis-board GUI (Stockfish/lichess style) for the Catan engine.

    python -m catan.gui_server [board.json] [--me red] [--port 8765] [--resume]

Starts a standard-library HTTP server and opens the browser.  The page in
``catan/gui/index.html`` draws the board, lets you play moves by clicking, and
asks the engine for analysis; everything goes through the same
``GameState`` / ``CatanEngine`` objects the CLI uses.

Analysis runs on a snapshot of the game outside the state lock, so clicks never
wait for the engine; a newer analysis request cancels the running one.  Every
change is autosaved to ``~/.catan_ai/autosave.json`` (``--resume`` or the
"New game" menu brings it back).

JSON API (all POST bodies are JSON unless noted)::

    GET  /api/state                      full board + game state
    POST /api/command   {"line": "d 8"}  short command (same syntax as the CLI)
    POST /api/apply     {"events": [...], "label": "..."}  apply engine events
    POST /api/undo
    POST /api/analyze   {"player": null, "rollouts": 80, "top_k": 5}
    POST /api/load      <board/game JSON>
    POST /api/new       {"kind": "example" | "blank" | "random" | "resume"}
    POST /api/extract   <PNG/JPEG bytes> run the screenshot extractor and load the result
    GET  /api/save                       current game as JSON (loadable again)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .board import RESOURCES, random_board_dict
from .catan_engine import AnalysisCancelled, CatanEngine
from .state_manager import CommandError, GameState

HERE = Path(__file__).resolve().parent
PAGE = HERE / "gui" / "index.html"
EXAMPLE = HERE.parent / "examples" / "example_board.json"
AUTOSAVE = Path.home() / ".catan_ai" / "autosave.json"   # the game in progress
PREVIOUS_NAME = "previous.json"                           # the game before the last New game / Open
USER_ERRORS = (CommandError, KeyError, ValueError)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return obj


def autosave_info(path: Path = AUTOSAVE) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
        return {"turn": data.get("turn_number", 1), "saved_at": data.get("saved_at", ""),
                "phase": data.get("phase", "main")}
    except (OSError, ValueError):
        return None


PLAYER_PALETTE = ("red", "blue", "orange", "white", "black", "green", "purple", "brown", "pink", "gray")
REPLAYED = ("command", "apply")


def _recolor(obj: Any, mapping: dict[str, str]) -> Any:
    """Rename player colours inside nested event data (values and dict keys)."""
    if isinstance(obj, dict):
        return {mapping.get(k, k): _recolor(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recolor(v, mapping) for v in obj]
    if isinstance(obj, str):
        return mapping.get(obj, obj)
    return obj


def _recolor_line(line: str, mapping: dict[str, str]) -> str:
    pattern = re.compile(r"(?<![\w#])(#?)(" + "|".join(map(re.escape, mapping)) + r")(?!\w)")
    return pattern.sub(lambda m: m.group(1) + mapping[m.group(2)], line)


class Session:
    """One game shared by the page; a lock serialises mutations.

    The game is ``base`` (its starting position) plus a list of logged actions.
    Deleting or rewinding log entries rebuilds the game by replaying the rest.
    """

    def __init__(self, game: GameState, autosave_path: Path | None = AUTOSAVE):
        self.lock = threading.Lock()
        self.autosave_path = autosave_path
        self.generation = 0          # bumped by every analysis request (cancels older ones)
        self.version = 0             # bumped by every state change
        self._next_id = 1
        self.start(game)

    # -- the log ------------------------------------------------------------------
    def start(self, game: GameState, actions: list[dict[str, Any]] | None = None) -> list[str]:
        """Begin from ``game``; optionally replay saved actions on top. Returns replay problems."""
        self.base = game.to_dict()
        self.game = game
        self.actions: list[dict[str, Any]] = []
        return self._replay(list(actions or [])) if actions else []

    @property
    def feed(self) -> list[dict[str, Any]]:
        return self.actions

    def _entry(self, kind: str, **kw: Any) -> dict[str, Any]:
        entry = {"id": self._next_id, "kind": kind, "turn": self.game.turn_number,
                 "player": self.game.current_color, **kw}
        self._next_id += 1
        return entry

    def note(self, line: str, message: str, ok: bool = True) -> None:
        """A log line that is not a move (errors, 'new game')."""
        self.actions.append(self._entry("info" if ok else "error", line=line, message=message, ok=ok))
        self.actions = self.actions[-400:]

    def run_command(self, line: str) -> str:
        entry = self._entry("command", line=line)
        msg = self.game.execute(line)
        entry.update(message=msg, ok=True)
        self.actions.append(entry)
        return msg

    def run_events(self, events: list[dict[str, Any]], label: str) -> str:
        entry = self._entry("apply", line=label, events=events)
        msg = self.game.apply_events(events) if events else "no action"
        entry.update(message=msg, ok=True)
        self.actions.append(entry)
        return msg

    def _replay(self, actions: list[dict[str, Any]]) -> list[str]:
        """Rebuild the game from ``base`` + ``actions``; moves that no longer apply are dropped."""
        self.game = GameState.from_dict(self.base)
        self.actions, problems = [], []
        for a in actions:
            if a.get("kind") not in REPLAYED:
                self.actions.append(a)
                continue
            a = dict(a, turn=self.game.turn_number, player=self.game.current_color)
            try:
                if a["kind"] == "command":
                    a["message"] = self.game.execute(a["line"])
                else:
                    a["message"] = self.game.apply_events(a["events"]) if a.get("events") else "no action"
                self.actions.append(a)
                if "assuming missed income" in a["message"]:
                    problems.append(f"'{a.get('line')}' now needs cards nobody received - assumed missed income")
            except USER_ERRORS as exc:
                problems.append(f"dropped '{a.get('line')}': {str(exc).strip(chr(39) + chr(34))}")
        self._next_id = max([self._next_id] + [int(a.get("id", 0)) + 1 for a in self.actions])
        return problems

    def delete(self, entry_id: int) -> list[str]:
        if not any(a["id"] == entry_id for a in self.actions):
            raise ValueError("that log entry no longer exists")
        return self._replay([a for a in self.actions if a["id"] != entry_id])

    def rewind(self, entry_id: int) -> list[str]:
        """Go back to just before ``entry_id`` (it and everything after are removed)."""
        ids = [a["id"] for a in self.actions]
        if entry_id not in ids:
            raise ValueError("that log entry no longer exists")
        return self._replay(self.actions[:ids.index(entry_id)])

    def undo(self) -> str:
        for a in reversed(self.actions):
            if a["kind"] in REPLAYED:
                self.delete(a["id"])
                return f"undid '{a.get('line')}'"
        return "nothing to undo"

    def recolor(self, mapping: dict[str, str]) -> list[str]:
        """Change player colours everywhere: starting position and every logged move."""
        mapping = {old: new for old, new in mapping.items() if old != new}
        if not mapping:
            return []
        current = list(self.game.order)
        final = [mapping.get(c, c) for c in current]
        bad = [c for c in final if c not in PLAYER_PALETTE]
        if bad or len(set(final)) != len(final):
            raise ValueError(f"each player needs a different colour from: {', '.join(PLAYER_PALETTE)}")
        base = _recolor(self.base, mapping)
        base["players"] = [dict(p, color=mapping.get(p["color"], p["color"])) for p in self.base.get("players", [])]
        if not base["players"]:
            base["players"] = [{"color": c} for c in base.get("turn_order") or final]
        self.base = base
        actions = []
        for a in self.actions:
            a = dict(a)
            if a.get("line"):
                a["line"] = _recolor_line(a["line"], mapping)
            if a.get("events"):
                a["events"] = _recolor(a["events"], mapping)
            actions.append(a)
        return self._replay(actions)

    # -- persistence ------------------------------------------------------------------
    def changed(self) -> None:
        self.version += 1
        if self.autosave_path is None:
            return
        try:
            data = self.game.to_dict()
            data["saved_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
            data["gui"] = {"base": self.base, "actions": self.actions[-400:]}
            self.autosave_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.autosave_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(_jsonable(data)))
            tmp.replace(self.autosave_path)
        except OSError:
            pass  # autosave is best effort

    def restore(self, data: dict[str, Any]) -> list[str]:
        """Load a saved game; autosaves carry their log so it stays editable."""
        gui = data.get("gui")
        if gui and gui.get("base"):
            return self.start(GameState.from_dict(gui["base"]), gui.get("actions"))
        self.start(GameState.from_dict(data))
        for f in data.get("gui_feed") or []:          # older autosaves: read-only history
            self.note(f.get("line", ""), f.get("message", ""), f.get("ok", True))
        return []

    @property
    def previous_path(self) -> Path | None:
        return self.autosave_path.with_name(PREVIOUS_NAME) if self.autosave_path else None

    def replace_game(self, game: GameState, keep_previous: bool = True, data: dict[str, Any] | None = None) -> list[str]:
        """Switch games; the outgoing autosaved game is kept as the 'previous game'."""
        if keep_previous and self.autosave_path is not None and self.autosave_path.exists():
            try:
                self.autosave_path.replace(self.previous_path)  # type: ignore[arg-type]
            except OSError:
                pass
        perspective = self.game.perspective
        problems = self.restore(data) if data is not None else self.start(game)
        if not self.game.perspective and perspective in self.game.order:
            self.game.perspective = perspective
            self.base["perspective"] = perspective
        return problems

    # -- serialisation ------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        g = self.game
        topo = g.board.topology
        players = []
        for c in g.order:
            p = g.players[c]
            i = g.idx(c)
            players.append({
                "color": c, "name": g.names.get(c, ""), "pending_setup": g.pending_setup(c) if g.phase == "setup" else None,
                "vp_cards": p.vp_cards,
                "settlements": sorted(p.settlements), "cities": sorted(p.cities), "roads": sorted(p.roads),
                "vp": g.public_vp(c), "hidden_vp": round(g.expected_hidden_vp(c), 2),
                "cards": g.hand_total(c), "hand": [round(x, 2) for x in g.belief.expected(i)],
                "exact": g.belief.is_exact(i), "dev": p.dev_hidden, "dev_known": p.dev_known,
                "knights": p.knights_played, "road_length": g.longest_road(c),
                "longest_road": g.longest_road_owner == c, "largest_army": g.largest_army_owner == c,
                "ports": sorted({port.type for port in g.player_ports(c)}),
                "rates": [g.trade_rate(c, r) for r in RESOURCES],
            })
        nxt = g.setup_next() if g.phase == "setup" else None
        return {
            "topology": {
                "hexes": [{"id": h, "q": q, "r": r, "xy": topo.hex_center_unit(q, r)}
                          for h, (q, r) in enumerate(topo.hex_coords)],
                "nodes": [{"id": n, "xy": topo.node_center_unit(n), "hexes": topo.node_hexes[n]}
                          for n in range(topo.num_nodes)],
                "edges": [{"id": e, "nodes": topo.edge_nodes[e]} for e in range(topo.num_edges)],
                "hex_nodes": topo.hex_nodes,
                "coastal_edges": topo.coastal_edges,
            },
            "tiles": [{"id": t.id, "terrain": t.terrain.value, "number": t.number, "pips": t.pips}
                      for t in g.board.tiles],
            "ports": [{"type": p.type, "nodes": p.nodes, "edge": topo.edge_between(*p.nodes)}
                      for p in g.board.ports],
            "robber": g.board.robber_hex,
            "order": g.order, "current": g.current_color, "phase": g.phase, "turn": g.turn_number,
            "last_roll": g.last_roll, "perspective": g.perspective, "setup_next": nxt,
            "pending_discards": g.pending_discards, "bank": g.bank, "dev_deck": g.dev_deck,
            "players": players, "resources": [r.value for r in RESOURCES],
            "can_undo": any(a["kind"] in REPLAYED for a in self.actions), "feed": self.actions[-150:],
            "version": self.version, "palette": PLAYER_PALETTE,
            "board_warnings": g.board_warnings(),
            # before the first move, autosave.json still holds the last session's game
            "autosave": autosave_info(self.autosave_path) if self.autosave_path and self.version == 0 else None,
            "previous": autosave_info(self.previous_path) if self.previous_path else None,
        }

    def analyze(self, player: str | None, rollouts: int, top_k: int) -> dict[str, Any]:
        """Runs on a snapshot outside the lock; raises AnalysisCancelled if superseded."""
        with self.lock:
            self.generation += 1
            gen = self.generation
            g = self.game.clone()
            version = self.version
        color = player if player in g.order else None
        t0 = time.time()
        stop = lambda: self.generation != gen  # noqa: E731
        eng = CatanEngine(g, rollouts=rollouts, should_stop=stop)
        rec = eng.recommend(color, top_k=top_k)
        color = rec.player
        heur = CatanEngine(g, rollouts=0)
        setup = g.phase == "setup"
        heat = [{"node": s.node, "value": round(s.value, 2), "pips": s.pips}
                for s in heur.settlement_scores(color, require_road=False)]
        plans = [] if setup else [
            {"target": p.target, "edges": p.edges, "cost": p.cost, "contested": p.contested,
             "score": round(p.score, 2)} for p in heur.road_plans(color, top=3)]
        return {
            "player": color, "phase": rec.phase, "version": version,
            "moves": [{"label": m.label, "events": _jsonable(m.events), "win": m.win_prob,
                       "heuristic": round(m.heuristic, 3), "detail": m.detail, "kind": m.kind,
                       "offer": m.offer} for m in rec.moves],
            "win": rec.win_prob, "table": rec.all_win_probs, "position_score": round(rec.position_score, 2),
            "scores": {c: round(heur.position_score(c), 2) for c in g.order},
            "heatmap": heat, "plans": plans,
            "resource_values": heur.resource_values(color),
            "notes": rec.notes, "rollouts": rollouts if any(m.win_prob is not None for m in rec.moves) else 0,
            "seconds": round(time.time() - t0, 2), "text": rec.render(),
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
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page navigated away mid-response

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
                if self.path == "/api/analyze":
                    req = json.loads(raw or b"{}")
                    try:
                        analysis = session.analyze(req.get("player"), int(req.get("rollouts", 80)),
                                                   int(req.get("top_k", 5)))
                    except AnalysisCancelled:
                        return self._json({"ok": False, "cancelled": True})
                    return self._json({"ok": True, "analysis": analysis})
                with session.lock:
                    result = self._route(raw)
                self._json(result)
            except USER_ERRORS as exc:
                with session.lock:
                    self._json({"ok": False, "error": str(exc).strip("'\""), "state": session.state()})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)

        def _route(self, raw: bytes) -> dict[str, Any]:
            g = session.game
            req = json.loads(raw or b"{}") if self.path != "/api/extract" else {}

            def done(msg: str, problems: list[str] | None = None) -> dict[str, Any]:
                session.changed()
                return {"ok": True, "message": msg, "problems": problems or [], "state": session.state()}

            if self.path == "/api/command":
                line = req["line"].strip()
                if line in ("u", "undo"):
                    return done(session.undo())
                if line.split()[:1] in (["s"], ["show"], ["help"], ["h"], ["?"]):
                    return {"ok": True, "message": g.execute(line), "state": session.state()}
                try:
                    msg = session.run_command(line)
                except USER_ERRORS as exc:
                    session.note(line, str(exc).strip("'\""), ok=False)
                    raise
                return done(msg)
            if self.path == "/api/apply":
                return done(session.run_events(req["events"], req.get("label", "engine move")))
            if self.path == "/api/undo":
                return done(session.undo())
            if self.path == "/api/log/delete":
                problems = session.delete(int(req["id"]))
                return done("log entry deleted" + (f"; {len(problems)} later move(s) no longer fit" if problems else ""), problems)
            if self.path == "/api/log/rewind":
                problems = session.rewind(int(req["id"]))
                return done("rewound the game", problems)
            if self.path == "/api/players":
                problems = session.recolor({k: str(v).lower() for k, v in (req.get("colors") or {}).items()})
                for color, name in (req.get("names") or {}).items():
                    if session.game.names.get(color, "") != (name or "").strip():
                        session.run_command(f"name #{color} {name}".rstrip())
                if req.get("order") and list(req["order"]) != session.game.order:
                    session.run_command("order " + " ".join("#" + c for c in req["order"]))
                return done("players updated", problems)
            if self.path == "/api/load":
                problems = session.replace_game(None, data=req)  # type: ignore[arg-type]
                session.note("load", "game loaded")
                return done("game loaded", problems)
            if self.path == "/api/new":
                kind = req.get("kind", "example")
                if kind in ("resume", "previous"):
                    path = session.autosave_path if kind == "resume" else session.previous_path
                    if not path or not path.exists():
                        raise ValueError("there is no saved game to go back to yet")
                    data = json.loads(path.read_text())
                    problems = session.replace_game(None, keep_previous=kind != "resume", data=data)  # type: ignore[arg-type]
                    return done(f"resumed the game saved {data.get('saved_at', '')}", problems)
                if kind == "random":
                    data, msg = random_board_dict(), "random board - setup: place settlements and roads in any order"
                elif kind == "blank":
                    data = json.loads(EXAMPLE.read_text())
                    data["players"], data["hands"] = [], None
                    msg = "example board without pieces - setup: place settlements and roads in any order"
                else:
                    data, msg = json.loads(EXAMPLE.read_text()), "example game loaded"
                session.replace_game(GameState.from_dict(data))
                session.note("new game", msg)
                return done(msg)
            if self.path == "/api/extract":
                try:
                    import cv2
                    import numpy as np
                except ImportError as exc:
                    raise ValueError("reading screenshots needs OpenCV: pip install -r requirements.txt") from exc
                from .vision_extractor import BoardVisionExtractor

                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("could not open that image - use a PNG or JPEG screenshot")
                res = BoardVisionExtractor().extract(img, turn_order=list(g.order))
                res.data["names"] = dict(g.names)
                session.replace_game(GameState.from_dict(res.data))
                session.note("read screenshot", "board read from screenshot - check it with Edit board")
                session.changed()
                return {"ok": True, "warnings": res.warnings, "state": session.state()}
            raise KeyError(f"unknown endpoint {self.path}")

    return Handler


def serve(session: Session, port: int) -> tuple[ThreadingHTTPServer, int]:
    """Bind to ``port`` or the next free one."""
    last: OSError | None = None
    for p in range(port, port + 20):
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), make_handler(session)), p
        except OSError as exc:
            last = exc
    raise SystemExit(f"no free port between {port} and {port + 19}: {last}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Catan analysis board GUI")
    ap.add_argument("state", nargs="?", help="board/game JSON (default: example game)")
    ap.add_argument("--me", help="your colour (perspective)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--resume", action="store_true", help=f"continue the autosaved game ({AUTOSAVE})")
    ap.add_argument("--no-browser", action="store_true")
    ns = ap.parse_args(argv)
    game = GameState.load(ns.state or EXAMPLE)
    if ns.me:
        game.perspective = game.resolve_player(ns.me)
    session = Session(game)
    if ns.resume and AUTOSAVE.exists():
        problems = session.restore(json.loads(AUTOSAVE.read_text()))
        for p in problems:
            print("note:", p)
        if ns.me:
            session.game.perspective = session.game.resolve_player(ns.me)
    server, port = serve(session, ns.port)
    url = f"http://127.0.0.1:{port}/"
    print(f"Catan analysis board: {url}")
    print("Keep this window open while you play. Press Ctrl+C to stop.")
    if not ns.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped - your game is autosaved; continue later with --resume")


if __name__ == "__main__":
    main()
