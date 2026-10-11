# Catan AI: Colonist.io board vision, live state tracker and move advisor

A modular Python system for 4-player Settlers of Catan on the Colonist.io base map:

| Module | File | Role |
|---|---|---|
| Geometry core | `catan/board.py` | Cube/axial hex coordinates, node and edge numbering, `Board` model, validation |
| 1. Vision | `catan/vision_extractor.py` | Screenshot → validated board JSON (tiles, numbers, ports, robber, pieces) |
| 2. State | `catan/state_manager.py` | Snake-draft turn manager, short commands, bank/hand tracking, hidden-information belief |
| 2b. Events | `catan/events.py` | Narrated event lines (`white rolled 7 -> robber to 11 steal from blue`) → events |
| 3. AI | `catan/fast_engine.py` | Default: bitboard linear evaluation + depth-2 expectimax (~15 ms, ~15 MB) |
| 3b. AI | `catan/catan_engine.py` | Advice front end; optional Monte Carlo rollouts for a deeper second opinion |
| Schema | `catan/board_schema.json` | JSON Schema for the extracted board/state |
| GUI | `catan/gui_server.py`, `catan/gui/index.html` | Analysis board in the browser (Stockfish/lichess style) |
| Demo | `run_demo.py`, `examples/example_board.json` | Load → `d 6`, `d 8`, `d 7` → AI top 3 |

```
screenshot ──► vision_extractor ──► board JSON ──► GameState ◄── short commands / JSON events
                                                      │
                                                      ▼
                                                 CatanEngine ──► Optimal Move / Alternative / Win %
```

## Quick start (Mac)

1. Install Python 3.10 or newer once: `brew install python@3.12` (or from python.org).
2. Get the code and open the folder:
   ```bash
   git clone https://github.com/vishesh0789/MILONI-WEBSIte.git
   cd MILONI-WEBSIte && git checkout claude/catan-game-ai-system-2gpqzu && cd catan_ai
   ```
3. **Double-click `start.command`** in Finder, or run `./start.command` in Terminal.
   The first run sets up a private Python environment (`.venv`) and installs what it needs, then
   the analysis board opens in your browser. Keep the Terminal window open while you play.
   * `./start.command --me red` sets your colour.
   * `./start.command --resume` continues the last game.
   * `./start.command --lan` also lets your phone open the board: it prints an address like
     `http://192.168.1.20:8765/` to type into the phone's browser (same Wi-Fi). There's no
     password, so use it only on a network you trust.

If macOS says the file can't be opened, right-click it, choose **Open**, then **Open** again (needed
once). Other ways to run it:

```bash
python -m catan.gui_server --me red      # the analysis board (what start.command runs)
python -m catan.state_manager board.json --me red   # keyboard-only tracker in the terminal
python -m catan.state_manager board.json --engine sim --rollouts 120   # Monte Carlo advice instead
python -m catan.fast_engine              # benchmark the fast engine (time and memory)
python run_demo.py [--vision]            # scripted demo: d 6, d 8, d 7 and AI advice
python -m pytest                         # 57 tests
```

## Analysis board GUI

A local server (standard library only) opens a browser page that drives the same `GameState` and
`CatanEngine` as the CLI. Press **?** on the page for the built-in guide.

* **Any starting point.**
  * *New game ▾* offers a random board or the example board, both starting from setup, or the
    example game with pieces already placed.
  * *Open ▾* reads a Colonist screenshot (or drop one on the board) or loads a saved game.
  * After a screenshot the board opens in **Edit board** mode so you can fix misreads. Click a
    tile to change its terrain, number or the robber; a coastal edge for its port; a corner or
    edge to place or remove pieces of any colour.
* **Setup (snake draft).** Click the row of whoever places first, then their settlement: the turn
  order rotates to start with them and locks. From then on the board follows 1→2→3→4, 4→3→2→1
  and always places for the player whose pick it is. The second settlement pays its starting
  cards, and the engine rings the best spots. Untick **Snake draft** in Players to place in any
  order for any player instead.
* **Each turn.**
  * Click the number rolled (or type it), then click the board for builds: a corner for a
    settlement (your own settlement becomes a city), an edge for a road, a tile for the robber
    and who to steal from.
  * **Trade** covers bank or port trades at your best rate and player trades. **Dev card** covers
    buying (you pick what you drew) and playing knight, road building, VP, monopoly and year of
    plenty.
  * On a 7, the banner lists who must discard, with a card picker and the engine's pick.
  * Click a player row to log moves for that player.
* **Free control.**
  * **Players** sets any colours (red, blue, orange, white, black, green, purple, brown, pink,
    gray), names (usable in commands) and the turn order. Changing a colour rewrites every piece
    and logged move.
  * Click any settlement, city or road to upgrade it, downgrade it, remove it or give it to
    another player.
  * Click a card count to add or remove that card. ✎ sets a whole hand (exact, or just a count),
    unplayed dev cards, knights played and VP cards shown.
  * The selected player stays selected until you change it (in a snake-draft setup the draft
    picks the player).
  * The engine never moves on its own: ▶ play needs a second click to apply.
* **Room rules.** In the Players dialog, set VP to win, the discard limit, friendly robber and snake/free setup to
  match your Colonist room. The tracker enforces them (discards, protected tiles), the engine
  plans and simulates with them, and a banner shows when someone reaches the target.
* **Editable game log.** Every move has **✕ delete** and **↶ back to here**.
  * Deleting replays the game from its start without that move. Later moves that no longer fit
    are dropped and listed. Builds that are now short of cards are kept, with a note that the
    tracker assumed income that wasn't logged.
  * Rewinding removes the move and everything after it.
  * Undo removes the last move the same way.
  * The log is saved with the game, so it stays editable after a restart.
* **Engine.**
  * Lines show the win % after each move, and **▶ play** applies one. A trade offer asks who
    accepted, and buying a dev card asks which card you drew.
  * Before the roll the advice is "Roll the dice", or a knight first when that's better.
  * Trade offers are scored as 50% likely to be accepted.
  * Analysis is progressive: an instant estimate first, then the simulation. It never blocks
    clicks, and a newer position cancels the old analysis.
  * The board shows numbered rings for the top 3 moves, a dashed road plan with a ghost
    settlement, and a heatmap of the best open spots.
  * The win bar shows everyone's chances, yours at the bottom.
* **Safety nets.**
  * Every move is autosaved to `~/.catan_ai/autosave.json`. *New game ▾* offers "Resume last
    session's game" and "Back to the previous game", in case you started a new one by mistake.
  * Undo with the button or Ctrl/⌘+Z.
  * If port 8765 is busy, the server picks the next free port.

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

1. **White balance.** Colonist's token faces and sails are pure white. The brightest near-neutral
   pixels show what white looks like in this capture, and each colour channel is scaled back. This
   undoes dim captures, photos of a screen and colour casts.
2. **Localisation.**
   * Pixels near any terrain colour form a land mask, and the largest blob is hole-filled.
   * Its bounding box seeds the board centre and size (`width = 5√3·size`, `height = 8·size`).
   * A coordinate-descent search then maximises the IoU with the predicted 19-hex polygon.
3. **Terrain.** Each tile's median colour is matched against reference colours for both Colonist's
   illustrated tiles and flat classic art. A quota step enforces the 4/3/4/4/3/1 composition.
4. **Numbers.** The token is found as the white, filled, roughly square blob in each tile. Colonist
   puts it in the lower half, under the terrain picture. The number is then read from cues that
   don't depend on the font:
   * **pip dots** (2/12, 3/11, 4/10, 5/9, 6/8 pairs)
   * **digit count**
   * **enclosed holes** (5 has none, 9 and 6 one, 8 two), with ink picked by Otsu per token
   * **red ink** for 6/8

   Font templates (or OCR, if installed) only break ties, and a quota step enforces the base-game
   token set.
5. **Ports.**
   * Colonist ports are white ship sails in the water.
   * Each sail is tied to a coastal edge by its position along the edge's outward normal, plus
     the wooden piers that join the ship to its two corners.
   * The sail icon ("?", tree, bricks, sheep, wheat, rocks) is matched against reference masks in
     `catan/assets/port_icons.npz`, enforcing four 3:1 ports and one of each 2:1.
   * If no sails are found (classic art), it falls back to colour probes.
6. **Robber.** Colonist's grey statue is found as a compact pale-grey blob on any non-mountain
   tile; classic art uses a dark figure. If neither is found, the robber goes on the desert with a
   warning, which also covers a robber on a mountain, where the grey rock art hides the statue.
7. **Pieces.**
   * Buildings are compact player-coloured blobs at corners. A city is noticeably larger, and tile
     art of a similar colour is rejected because it also fills the wider area around the corner.
   * Roads must cover a bar across the edge line, which rejects the one-sided shading Colonist
     draws along hills tiles.

**Measured on a real Colonist mobile screenshot** (`tests/fixtures/colonist_mobile_board.jpg`,
board crop only):

| Variant | Result |
|---|---|
| Original; 0.6×–2.2× scale; JPEG quality 20–60; board-only crop; wide desktop-style canvas; 25–40% darker; warm or cool tint; slight blur | 19/19 tiles, 9/9 ports (type and corners), robber correct, no phantom pieces |
| Very small capture (hex radius ~28 px) | 17/19 tiles, 9/9 ports |
| Overexposed (+15%, whites clipped) | 16/19 tiles, 9/9 ports |
| Smaller than ~25 px per hex | Not readable; take a bigger screenshot |

Still unverified on real Colonist images:
* **Pieces.** The test screenshot is from setup, so there are no pieces on it yet. Piece colours
  are estimates; an orange player on wheat fields is the case most likely to need calibration.
  Colonist also has colours other than red, blue, orange and white (for example black).
* **Robber on mountains.** It can't be detected there; the app warns and you fix it in Edit board.

A screenshot can't show placement order or whose turn it is, so the board opens in **Edit board**
mode for a quick check. `VisionConfig().calibrate(screenshot, known_board_json)` re-fits the
terrain colours for an unusual theme.

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
| `8` | | A bare number is a dice roll |
| `hand <player> <cards>` / `hand <player> <n>` | `hand blue 2w o`, `hand blue 5` | Set a hand exactly, or as n cards of unknown type (joining mid-game) |
| `devs <player> <unplayed> [knights] [vp]` | `devs blue 2 1 0` | Set dev cards held, knights already played, VP cards shown |
| `name <player> <name>` | `name black PrideNL` | Name a player; names work in every command |
| `order <p1> <p2> <p3> <p4>` | `order red blue black orange` | Set the turn order |
| `rules vp=<n> discard=<n> friendly=on\|off setup=snake\|free` | `rules vp=12 friendly=on` | Room rules; `rules` alone shows them |
| `first <player>` | `first blue` | Who places first in setup (otherwise the first settlement decides) |
| `tile <hex> [terrain] [number] [robber]` | `tile 4 hills 6` | Fix a tile (also `tile 4 robber`) |
| `port <edge> <type>` | `port 20-26 ore`, `port 12 none` | Set or remove a port on a coastal edge |
| `piece <node\|a-b> <player\|none> [city]` | `piece 24 red city` | Place or remove a piece without rules or costs |
| `n`, `me <p>`, `s`, `u`, `ai [p]`, `help`, `save f.json` | | Next turn, set perspective, show, undo, AI advice, all commands, save |

**Setup (turn manager).** Setup is detected automatically while players have fewer than two
settlements and roads; `b <player> s <node>` and `b <player> r <edge>` are then free.

* **First player.** `first blue` sets it, or the first settlement placed deduces it: the order
  rotates to start with that player (`red blue orange white` → `orange white red blue`) and locks.
* **Snake draft.** Round 1 runs P1→P4 (settlement then road each), round 2 runs P4→P1. An
  out-of-turn placement is refused with who is next (`snake draft (round 2): next is White placing
  a road`). `rules setup=free` allows any order instead.
* **Starting resources** come from the second settlement only.
* **After round 2** the order is locked P1→P4, P1 is to roll and turn 1 begins. Dice are refused
  until setup is complete.
* A road must connect to that player's pieces (`piece` places anything anywhere).

**Turns.** A bare roll (`8`) after a roll ends the current turn and rolls for the next player. A
named roll (`white rolls 8`) advances to White, noting anyone skipped. The same player rolling twice
is refused, as is any roll while discards or the robber are pending.

**Narrated events (`events.py`).** Any line that doesn't start with a short command is read as
narration. Parts can be chained with `->`, `;` or `then`, and the subject carries over:

```
white rolls 8
white rolled 7 -> robber to 11 steal from blue          (… steal from blue wood   if you saw it)
white rolled 7 -> robber to (0,-1) steal from blue      (axial (q,r) or cube q,r,s also work)
blue discards 2 wood brick ore      /   blue discards 4  (unknown cards)
blue builds road 38    blue builds road 26-32    blue builds settlement 24    blue upgrades 24
orange buys dev_card   orange buys dev card knight       (type only if it's you)
red plays knight to 11 steal from blue    red plays monopoly wheat    red plays year of plenty ore ore
red plays road building 38 44   red trades 4 wood for ore   red trades 2 wood with blue for ore
red ends turn
```

Players can be named by colour, a unique colour prefix or their name.

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

## Module 3: AI engine

### Fast engine (default, `fast_engine.py`)

Built for an 8 GB laptop: no numpy in the hot path, everything is Python ints used as bitmasks.

* **Bitboards.** 54 nodes and 72 edges are bits. `BoardBits` (cached per board) holds node
  neighbour masks, node↔edge masks, ports per node and, per robber position, each node's
  production vector. Legal settlement spots, the road frontier and occupied nodes are a few
  `&`/`|` operations.
* **Linear evaluation** per player:
  `VP + 4·production(pips × scarcity) + 0.12·diversity + port synergy + 0.06·expansion radius
  (free spots 0/1/2 roads out, ×0.85 per road) + 0.08·longest-road length + dev-card value +
  hand value toward goals + VP race pressure (0.35 per VP above target−4)`. Results are memoised.
* **Search.** Candidates are pruned to the top 3 cities, top 4 settlements, top 3 roads, a dev card,
  trades that unlock a build, and passing. Each is scored by a depth-2 expectimax: opponents take
  their expected income, then a chance node over my 11 dice totals (a 7 halves an oversized hand),
  then the best follow-up build.
* **Win %.** A softmax over position values whose sharpness grows as the leader nears the target.
  On test positions it is within about 8 points of the Monte Carlo win rate on average.
* **Cost.** Roughly 13–20 ms per recommendation, under 0.1 MB allocated per call, about 15 MB process
  RSS (`python -m catan.fast_engine` prints this on your machine).

The GUI's depth menu has **Fast** (default), **Simulate 80** and **Simulate 200**. Simulate shows
the fast answer immediately, then replaces it with the Monte Carlo result. In the terminal use
`--engine sim --rollouts N`.

### Simulation engine (`catan_engine.py`, `mode="sim"`)

* **Settlement scoring.** Pips weighted by board scarcity, a diversity bonus (resources the player
  lacks count more), port synergy (2:1 port × own production of that resource) and expansion reach
  (best free spot two steps away). A small penalty applies while the robber blocks a hex.
* **Road pathing.** Dijkstra from the player's network: own roads cost 0, opponent roads and
  buildings block. It targets legal spots by `value × 0.75^(roads−1)` and discounts targets that
  opponents reach first (`free` / `race` / `behind`).
* **Robber.** Blocked cards over about 8 rolls, scaled by each victim's VP lead and a near-monopoly
  bonus, plus the steal value (higher if the victim holds more than 7). Own buildings are penalised
  at 1.5×. It never recommends robbing only yourself while a harmless tile exists, and it
  respects the friendly robber.
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

Main-phase candidates are:
* cities, settlements, and roads along the best plans
* dev cards
* port or bank trades that unlock a build this turn
* 1:1 table-trade offers, scored as 50% likely to be accepted
* knights (for your own hand)
* ending the turn

During setup the engine ranks settlement spots (for the second settlement it also lists the
starting cards) and the roads that point toward the best free spot. Before the roll it recommends
rolling, or a knight first.

## Hooking the board parser into the engine

```python
from catan.vision_extractor import BoardVisionExtractor
from catan.state_manager import GameState
from catan.catan_engine import CatanEngine
from catan.fast_engine import FastEngine

board_json = BoardVisionExtractor().extract("screenshot.png").data    # tiles, numbers, ports, robber, pieces
game = GameState.from_dict(board_json)                               # setup phase if no pieces yet
game.execute("me red")                                               # your colour
game.execute("first blue")                                           # optional; else deduced
game.execute("blue builds settlement 24")                           # narrate or use short commands
...
print(CatanEngine(game, mode="fast").recommend().render())           # Optimal Move / Alternative / Win %
FastEngine(game).recommend("red", top_k=3)                           # raw ranked moves (label, events, value)
```

The GUI's **Read screenshot** and `python -m catan.state_manager board.json` do exactly this.
Every recommended move carries its `events`, so applying one is `game.apply_events(move["events"])`.

## Limitations

* Before the roll the only advice is to roll (or play a knight first). In Catan you can't build
  before rolling.
* Vision is verified on one real Colonist board (plus synthetic renders). Piece detection has not
  been tested on real Colonist pieces yet; see the vision section.
* The fast engine looks one turn ahead (my move, the dice, my best follow-up); it does not model
  opponents' builds or trades beyond their expected income. Use Simulate for a deeper check.
* The rollout policy is greedy and simplified. Player trades are basic: 1-for-1, accepted half the
  time, and refused to anyone within 2 VP of winning. Monopoly and YOP use a basic policy. Win % is
  a strength signal between moves, not a calibrated probability.
* Only the standard 19-hex base map is supported (not 5-6 player or Seafarers maps).
* Rule enforcement is advisory. For example, the tracker doesn't stop a dev card being played on
  the turn it was bought.
