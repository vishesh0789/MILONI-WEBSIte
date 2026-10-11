"""Natural event lines -> state-manager events.

Accepts the way you'd narrate a Colonist game, one line per thing that happened
(several parts can be chained with ``->``, ``;`` or ``then``)::

    8                                          (bare roll, still handled by the short parser)
    white rolls 8
    white rolled 7 -> robber to 11 steal from blue
    white rolled 7 -> robber to (0,-1) steal from blue wood      (card if you saw it)
    blue discards 2 wood brick ore            /  blue discards 4   (unknown cards)
    blue builds road 38          blue builds road 26-32          blue builds settlement 24
    blue upgrades 24             blue builds city 24
    orange buys dev_card         orange buys dev card knight       (type if it's you)
    red plays knight to 11 steal from blue    red plays monopoly wheat
    red plays year of plenty ore ore          red plays road building 38 44
    red trades 4 wood for ore    red trades 2 wood with blue for ore
    red ends turn

Players can be named by colour (or unique prefix) or by name.  Hexes by id
(``11``, ``hex 11``) or axial / cube coordinate (``(0,-1)``, ``0,-1,1``).
Edges by id or ``a-b`` corner pair.  Short commands (``b red s 24``...) keep working.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .board import RESOURCES, parse_resource

if TYPE_CHECKING:  # pragma: no cover
    from .state_manager import GameState

VERBS = (r"rolls?|rolled|builds?|built|places?|placed|upgrades?|upgraded|buys?|bought|plays?|played|"
         r"robber|steals?|stole|discards?|discarded|trades?|traded|ends?|passes")
_NATURAL = re.compile(rf"\b({VERBS})\b|->")
HEX = r"(\(\s*-?\d+\s*,\s*-?\d+(?:\s*,\s*-?\d+)?\s*\)|-?\d+\s*,\s*-?\d+(?:\s*,\s*-?\d+)?|\d+)"
DEV_NAMES = {
    "knight": "knight", "soldier": "knight", "monopoly": "monopoly", "mono": "monopoly",
    "yearofplenty": "year_of_plenty", "yop": "year_of_plenty", "plenty": "year_of_plenty",
    "roadbuilding": "road_building", "rb": "road_building", "roads": "road_building",
    "vp": "vp", "victorypoint": "vp", "point": "vp",
}


class ParseError(ValueError):
    pass


# first words that always mean a short command (``r 11 blue``, ``tile 9 robber``, ``rules discard=9``...)
SHORT_COMMANDS = {
    "d", "r", "x", "b", "t", "tp", "p", "g", "l", "n", "u", "s", "h", "me", "show", "state", "undo", "help",
    "next", "gain", "lose", "hand", "devs", "dev", "tile", "port", "piece", "name", "order", "rules", "first",
    "robber", "discard", "build", "buy", "trade", "play", "roll",
}


def looks_natural(line: str) -> bool:
    words = line.strip().lower().split()
    if not words or words[0] in SHORT_COMMANDS or words[0].isdigit():
        return False
    return bool(_NATURAL.search(line.lower()))


def parse_natural(line: str, game: "GameState") -> list[dict[str, Any]]:
    """Turn one narrated line into events (raises ParseError if any part is not understood)."""
    text = re.sub(r"\s+", " ", line.strip().lower())
    parts = [p.strip(" ,.") for p in re.split(r"->|;|\bthen\b", text) if p.strip(" ,.")]
    events: list[dict[str, Any]] = []
    subject: str | None = None
    for part in parts:
        who, rest = _subject(part, game)
        subject = who or subject
        got = _parse_part(rest, subject, game)
        if got is None:
            raise ParseError(f"didn't understand '{part}' (type help for examples)")
        if got and got[0].get("type") == "robber" and not got[0].get("thief") and subject:
            got[0]["thief"] = subject
        events.extend(got)
    return events


# -- helpers ---------------------------------------------------------------------


def _subject(part: str, game: "GameState") -> tuple[str | None, str]:
    words = part.split(" ", 1)
    if len(words) == 2 and not _NATURAL.fullmatch(words[0]) and words[0] not in ("robber",):
        try:
            return game.resolve_player(words[0]), words[1]
        except ValueError:
            pass
    return None, part


def _hex(token: str, game: "GameState") -> int:
    nums = [int(x) for x in re.findall(r"-?\d+", token)]
    topo = game.board.topology
    if len(nums) == 1:
        if not 0 <= nums[0] < topo.num_hexes:
            raise ParseError(f"there is no hex {nums[0]} (0-{topo.num_hexes - 1})")
        return nums[0]
    q, r = nums[0], nums[1]
    if len(nums) == 3 and q + r + nums[2] != 0:
        raise ParseError(f"({q},{r},{nums[2]}) is not a cube coordinate (q+r+s must be 0)")
    if (q, r) not in topo.hex_index:
        raise ParseError(f"there is no hex at ({q},{r})")
    return topo.hex_index[(q, r)]


def _robber_tail(text: str, game: "GameState", thief: str | None) -> list[dict[str, Any]] | None:
    m = re.search(rf"\b(?:to|on|at|robber)\s+(?:hex\s+)?{HEX}", text)
    if not m:
        return None
    ev: dict[str, Any] = {"type": "robber", "hex": _hex(m.group(1), game)}
    if thief:
        ev["thief"] = thief
    s = re.search(r"\b(?:steals?|stole|stealing|rob|robs|robbed)\s+(?:an?\s+|1\s+)?(?:(\w+)\s+)?from\s+([\w#]+)"
                  r"(?:\s+\(?(\w+)\)?)?", text)
    if s:
        ev["victim"] = game.resolve_player(s.group(2))
        card = s.group(1) if s.group(1) not in (None, "card", "a") else s.group(3)
        if card:
            try:
                ev["resource"] = parse_resource(card).value
            except ValueError:
                pass
    return [ev]


def _cards(text: str) -> tuple[int, ...]:
    counts = [0] * len(RESOURCES)
    for n, name in re.findall(r"(?:(\d+)\s*)?([a-z]+)", text):
        if name in ("and", "a", "an", "of", "cards", "card"):
            continue
        counts[[r.value for r in RESOURCES].index(parse_resource(name).value)] += int(n or 1)
    return tuple(counts)


def _parse_part(text: str, who: str | None, game: "GameState") -> list[dict[str, Any]] | None:
    from .state_manager import ITEM_ALIASES  # local import: no cycle at module load

    # roll (possibly followed by the robber in the same part)
    m = re.search(r"\b(?:rolls?|rolled)\s+(?:an?\s+)?(\d{1,2})\b", text)
    if m:
        out: list[dict[str, Any]] = [{"type": "roll", "value": int(m.group(1)), **({"player": who} if who else {})}]
        tail = _robber_tail(text[m.end():], game, who)
        return out + (tail or [])

    if re.search(r"\b(?:ends?|ended)\b(?:\s+\w+)?\s+turn\b|\bpasses\b|^end turn$", text):
        return [{"type": "next_turn"}]

    # builds / upgrades
    m = re.search(r"\b(?:builds?|built|places?|placed)\s+(?:an?\s+)?(road|settlement|house|city)\s+"
                  r"(?:on\s+|at\s+|to\s+)?(?:edge\s+|node\s+|corner\s+)?(\d+(?:-\d+)?)", text)
    if m:
        item = ITEM_ALIASES.get(m.group(1), "settlement" if m.group(1) == "house" else m.group(1))
        return [{"type": "build", "player": _need(who, "who built it"), "item": item, "target": m.group(2)}]
    m = re.search(r"\b(?:upgrades?|upgraded)\s+(?:settlement\s+)?(?:at\s+|on\s+)?(?:node\s+|corner\s+)?(\d+)", text)
    if m:
        return [{"type": "build", "player": _need(who, "who upgraded"), "item": "city", "target": m.group(1)}]

    # dev cards
    m = re.search(r"\b(?:buys?|bought)\s+(?:an?\s+)?(?:dev(?:elopment)?[ _-]?(?:card)?|card)\b(?:\s+\(?([a-z ]+?)\)?)?$", text)
    if m:
        ev = {"type": "build", "player": _need(who, "who bought it"), "item": "devcard"}
        if m.group(1):
            key = m.group(1).replace(" ", "").replace("_", "")
            if key in DEV_NAMES:
                ev["card"] = DEV_NAMES[key]
        return [ev]
    m = re.search(r"\b(?:plays?|played)\s+(?:an?\s+|the\s+)?(knight|soldier|monopoly|mono|year[ _-]?of[ _-]?plenty|yop|"
                  r"road[ _-]?building|rb|vp|victory[ _-]?point)\b(.*)", text)
    if m:
        p = _need(who, "who played it")
        card = DEV_NAMES[re.sub(r"[ _-]", "", m.group(1))]
        rest = m.group(2)
        ev: dict[str, Any] = {"type": "play_dev", "player": p, "card": card}
        if card == "knight":
            return [ev] + (_robber_tail(rest, game, p) or [])
        if card == "monopoly":
            ev["resource"] = parse_resource(rest.split()[-1] if rest.split() else "").value
        elif card == "year_of_plenty":
            vec = _cards(re.sub(r"\b(?:for|taking|takes|gets)\b", " ", rest))
            ev["resources"] = [r.value for r, k in zip(RESOURCES, vec) for _ in range(k)]
        elif card == "road_building":
            ev["edges"] = re.findall(r"\d+-\d+|\d+", rest)
        return [ev]

    # robber on its own ("robber to 11 steal from blue", "moves robber to (0,-1)")
    if "robber" in text:
        tail = _robber_tail(text, game, who)
        if tail:
            return tail

    # discards
    m = re.search(r"\b(?:discards?|discarded)\s+(.+)", text)
    if m:
        p = _need(who, "who discarded")
        body = m.group(1).strip()
        if re.fullmatch(r"\d+(?:\s+cards?)?", body):
            return [{"type": "discard", "player": p, "count": int(body.split()[0])}]
        return [{"type": "discard", "player": p, "cards": _cards(body)}]

    # trades
    m = re.search(r"\b(?:trades?|traded)\s+(.+?)\s+with\s+([\w#]+)\s+for\s+(.+)", text)
    if m and m.group(2) not in ("bank", "port", "the"):
        a = _need(who, "who traded")
        return [{"type": "player_trade", "a": a, "b": game.resolve_player(m.group(2)),
                 "a_gives": _cards(m.group(1)), "b_gives": _cards(m.group(3))}]
    m = re.search(r"\b(?:trades?|traded)\s+(\d+)?\s*([a-z]+)\s+for\s+(\d+)?\s*([a-z]+)", text)
    if m:
        times = int(m.group(3) or 1)
        ev = {"type": "bank_trade", "player": _need(who, "who traded"), "give": parse_resource(m.group(2)).value,
              "get": parse_resource(m.group(4)).value, "times": times}
        if m.group(1):
            ev["rate"] = int(m.group(1)) // times
        return [ev]
    return None


def _need(who: str | None, what: str) -> str:
    if not who:
        raise ParseError(f"start the line with a player ({what}?) - e.g. 'blue builds road 38'")
    return who
