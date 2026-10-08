# Catan AI: Colonist.io board vision, live state tracker and move advisor

A modular Python system for 4-player Settlers of Catan on the Colonist.io base map:

| Module | File | Role |
|---|---|---|
| Geometry core | `catan/board.py` | Cube/axial hex coordinates, node and edge numbering, `Board` model, validation |
| 1. Vision | `catan/vision_extractor.py` | Screenshot → validated board JSON (tiles, numbers, ports, robber, pieces) |
| 2. State | `catan/state_manager.py` | Short-command input, bank/hand tracking, hidden-information belief |
| 3. AI | `catan/catan_engine.py` | Heuristic evaluation + Monte Carlo rollouts → top-N moves |
| Schema | `catan/board_schema.json` | JSON Schema for the extracted board/state |
| Demo | `run_demo.py`, `examples/example_board.json` | Load → `d 6`, `d 8`, `d 7` → AI top 3 |

```
screenshot ──► vision_extractor ──► board JSON ──► GameState ◄── short commands / JSON events
                                                      │
                                                      ▼
                                                 CatanEngine ──► Optimal Move / Alternative / Win %
```

## Quick start

```bash
cd catan_ai
pip install -r requirements.txt          # numpy + opencv for vision; core runs on stdlib only
python run_demo.py                       # example board, dice sequence, AI advice
python run_demo.py --vision              # also renders a synthetic screenshot and extracts it
python -m pytest                         # 26 tests

# live play
python -m catan.vision_extractor shot.png -o board.json --debug overlay.png
python -m catan.state_manager board.json --me red
```

## Hex coordinate system and numbering

Colonist draws pointy-top hexes in rows of 3-4-5-4-3. Each hex has cube coordinates
`(q, r, s)` with `q + r + s = 0` and `max(|q|,|r|,|s|) ≤ 2`. `r` is the row (top = −2) and
`q` grows to the right. Hex pixel centre for circumradius `size`:
`x = size·√3·(q + r/2)`, `y = size·1.5·r`.

**Intersections (nodes).** Every corner sits on an integer lattice, so corners shared by up to three
hexes deduplicate exactly with no floating-point error:

```
X = 2q + r + dx,  Y = 3r + dy     corners (dx,dy) clockwise from the top:
N (0,-2)  NE (1,-1)  SE (1,1)  S (0,2)  SW (-1,1)  NW (-1,-1)
```

* **Hex ids** 0–18 are in reading order (sorted by `r`, then `q`).
* **Node ids** 0–53 are in reading order (sorted by `Y`, then `X`). Node 0 is the top corner of hex 0
  and node 53 is the bottom corner of hex 18.
* **Edge ids** 0–71 are sorted by `(min_node, max_node)`. Roads can be entered as an edge id
  (`38`) or a node pair (`26-32`).
* **Coastal edges** (30) are ordered clockwise from the top. The 9 standard port slots are coastal
  indices `0,3,6,10,13,16,20,23,26` (spacing 3-3-4).

| hex | (q, r, s) | corner nodes N, NE, SE, S, SW, NW |
|---|---|---|
| 0 | (0, -2, 2) | 0, 4, 8, 12, 7, 3 |
| 1 | (1, -2, 1) | 1, 5, 9, 13, 8, 4 |
| 2 | (2, -2, 0) | 2, 6, 10, 14, 9, 5 |
| 3 | (-1, -1, 2) | 7, 12, 17, 22, 16, 11 |
| 4 | (0, -1, 1) | 8, 13, 18, 23, 17, 12 |
| 5 | (1, -1, 0) | 9, 14, 19, 24, 18, 13 |
| 6 | (2, -1, -1) | 10, 15, 20, 25, 19, 14 |
| 7 | (-2, 0, 2) | 16, 22, 28, 33, 27, 21 |
| 8 | (-1, 0, 1) | 17, 23, 29, 34, 28, 22 |
| 9 | (0, 0, 0) | 18, 24, 30, 35, 29, 23 |
| 10 | (1, 0, -1) | 19, 25, 31, 36, 30, 24 |
| 11 | (2, 0, -2) | 20, 26, 32, 37, 31, 25 |
| 12 | (-2, 1, 1) | 28, 34, 39, 43, 38, 33 |
| 13 | (-1, 1, 0) | 29, 35, 40, 44, 39, 34 |
| 14 | (0, 1, -1) | 30, 36, 41, 45, 40, 35 |
| 15 | (1, 1, -2) | 31, 37, 42, 46, 41, 36 |
| 16 | (-2, 2, 0) | 39, 44, 48, 51, 47, 43 |
| 17 | (-1, 2, -1) | 40, 45, 49, 52, 48, 44 |
| 18 | (0, 2, -2) | 41, 46, 50, 53, 49, 45 |

`python -m catan.vision_extractor shot.png --debug overlay.png` draws every hex, node and edge id
on the screenshot. It's the quickest way to look up an id mid-game.

## Module 1: Vision (`vision_extractor.py`)

1. **Localisation.** Pixels near any terrain prototype (LAB distance) form a land mask. The largest
   blob is hole-filled, which drops port discs and fills tokens and pieces. Its bounding box seeds
   the centre and size (`width = 5√3·size`, `height = 8·size`). A coordinate-descent search then
   maximises the IoU between the mask and the predicted 19-hex polygon.
2. **Terrain.** Each hex is classified from the median LAB colour of an annulus that skips the token
   and the borders, using the nearest prototype. If the counts break the base-game composition
   (4/3/4/4/3/1), a quota-constrained assignment re-labels the least confident hexes.
3. **Numbers.** OCR runs if one is installed (easyocr, then pytesseract). Otherwise digits are
   segmented by connected components and correlated against rendered digit templates. Two priors
   help: the pip-dot count (used only when the dots are cleanly separated) and red ink (6/8). A
   second quota assignment enforces the token distribution.
4. **Ports.** Each of the 30 coastal edges is probed just outside the coast. The 9 strongest
   non-water probes that share no nodes are classified as 3:1 or a 2:1 resource port.
5. **Pieces.** Settlements and cities are colour blobs at node pixels; a city is a noticeably
   larger blob. Roads are colour matches along the middle third of each edge, and the robber is a
   dark blob inside a hex.
6. **Output.** Schema JSON with per-hex confidence. `Board.validate` warnings go in `meta.warnings`.

Captures with a hex radius under 48 px are upscaled automatically. Against synthetic Colonist-style
renders (`render_synthetic_board`), the extractor recovers every tile, number, port, robber,
settlement, city and road exactly for hex radii from 36 to 128 px; the tests cover 40, 64 and 96.
**Real Colonist themes differ.** Call `VisionConfig().calibrate(screenshot, known_board_json)` once
on a labelled screenshot to re-fit the terrain colours. If localisation fails, pass `--size` and
`--center`. A screenshot can't show placement order or whose turn it is, so set
`settlements[1]` (the second settlement, which grants setup income), `current_player` and
`perspective` yourself if they matter.

## Module 2: State tracking (`state_manager.py`)

| Command | Example | Effect |
|---|---|---|
| `d <roll>` | `d 8` | Distributes to every bordering settlement or city, skips the robber hex, applies bank-shortage rules. On a 7 it computes who must discard |
| `r <hex> [victim] [res]` | `r 11 red` | Moves the robber. The steal is unknown unless you saw the card (`r 11 red o`) |
| `x <player> <cards>` / `x <player> <n>` | `x blue 2w b o` | Discard on a 7, known cards or an unknown count |
| `b <player> <item> [id] [free]` | `b red s 24`, `b red r 26-32`, `b blue dev` | Build or buy with legality checks. `free` is for setup placements |
| `t [player] <give> <get> [times]` | `t w o`, `t 4w o` | Bank or port trade at the best rate the player has (4:1 / 3:1 / 2:1) |
| `tp <a> <b> <a_gives> <b_gives>` | `tp red blue 2w o` | Player-to-player trade |
| `p <player> <card> [args]` | `p red knight`, `p red mono w`, `p red yop o o`, `p red rb 12 13` | Play a dev card |
| `g` / `l <player> <cards>` | `g red o` | Manual correction (gain from / return to the bank) |
| `n`, `me <p>`, `s`, `u`, `ai [p]`, `save f.json` | | Next turn, set perspective, show, undo, AI advice, save |

Resource aliases: `w`/wood, `b`/brick, `s`/sheep, `g`/wheat (grain), `o`/ore. Counts can be written
as `2w`, `w:2`, `w+w` or `w,w`. Players accept colour prefixes. In `t`, a lone `w`, `b` or `o` is
always read as a resource.

**Hidden information.** Card totals are always exact, because every change in hand size is public.
The bank is exact for every known-card event. `HandBelief` keeps a weighted set of joint hand
assignments for all four players (pruned to 2048):

* Unseen steals branch by the victim's composition.
* Unknown discards branch hypergeometrically.
* Monopoly applies to each branch.
* Every public spend (build, trade, discard) eliminates branches that couldn't afford it, so the
  tracker re-learns exact hands as players spend.

If an input contradicts every branch (for example, a roll you forgot to log), the tracker assumes
missed income from the bank and prints a warning. Card conservation always holds.

Every command parses to a plain dict event (`parse_command`) applied with `GameState.apply_event`.
A log scraper or websocket bridge can push the same events as JSON:
`{"type": "roll", "value": 8}`, `{"type": "build", "player": "red", "item": "city", "target": 14}`.

## Module 3: AI engine (`catan_engine.py`)

* **Settlement scoring.** Pips weighted by board scarcity, a diversity bonus (resources the player
  lacks count more), port synergy (2:1 port × own production of that resource) and expansion reach
  (best free spot two steps away). A small penalty applies while the robber blocks a hex.
* **Road pathing.** Dijkstra from the player's network: own roads cost 0, opponent roads and
  buildings block. It targets legal spots by `value × 0.75^(roads−1)` and discounts targets that
  opponents reach first (`free` / `race` / `behind`).
* **Robber.** Blocked cards over about 8 rolls, scaled by each victim's VP lead and a near-monopoly
  bonus, plus the steal value (higher if the victim holds more than 7). Own buildings are penalised
  at 1.5×.
* **Trade evaluator.** Hand potential toward the build goals (city, settlement, road, dev card) with
  dynamic per-resource values. Helping a leading trade partner is penalised.
* **Position score.** VP (including expected hidden VP cards) plus production value, diversity,
  expansion, hand potential minus 7-risk, dev cards, and army/road progress.
* **Monte Carlo.** Candidates are shortlisted by heuristic. For each rollout, hidden hands are
  sampled from the belief and unseen dev cards from the remaining pool, then a greedy self-play
  simulator (about 9 ms per game) plays to 10 VP. Common random numbers are shared across
  candidates. Ranking uses MC win rate, and the heuristic can shift a move by at most one standard
  error, so it only breaks statistical ties.

```
=== AI advice for orange (robber phase) ===
Optimal Move: [Move Robber to hex 16 (hills 5), steal from blue]
Alternative: [Move Robber to hex 11 (mountains 8), steal from red]
Expected Win % / Position Score: [63.7% / 5.81]
```

Main-phase candidates are cities, settlements, roads along the best plans, dev cards, port or bank
trades that unlock a build this turn, 1:1 table-trade offers, knights (for your own hand) and
ending the turn.

## Limitations

* Vision has only been tested on synthetic renders. Real screenshots need the one-off
  `calibrate` step, and OCR is recommended for non-Hershey fonts.
* The rollout policy is greedy and simplified: no player-to-player trades, and a basic
  monopoly/YOP policy. Win % is a strength signal between moves, not a calibrated probability.
* Rule enforcement is advisory. For example, the tracker doesn't stop a dev card being played on
  the turn it was bought.
