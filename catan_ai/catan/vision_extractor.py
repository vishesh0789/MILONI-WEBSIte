"""Board vision: Colonist.io screenshot -> validated board JSON.

Pipeline
--------
1. **Board localisation** - pixels close to any terrain prototype (LAB space)
   form a land mask; the largest blob's bounding box gives the board centre
   and hex circumradius (pointy-top board: width = 5*sqrt(3)*size,
   height = 8*size).  A small grid search over scale/offset then maximises
   classification confidence to snap the hex lattice onto the image.
2. **Terrain classification** - median LAB colour of an annulus inside each hex
   (skipping the number token and borders) -> nearest prototype.  If the raw
   result violates the base-game composition a quota-constrained assignment
   fixes the least confident hexes.
3. **Number tokens** - OCR (easyocr -> pytesseract) when installed, otherwise
   contour/template matching against rendered digits.  Pip dots under the
   number and the red colour of 6/8 constrain candidates; a second
   quota-constrained assignment enforces the token distribution.
4. **Ports** - every coastal edge is probed just outside the coastline; the 9
   strongest non-water probes (no shared nodes) are classified 3:1 or 2:1.
5. **Pieces** - settlements/cities are colour blobs at intersection pixels
   (city = larger blob), roads are colour blobs at edge midpoints; robber is a
   dark blob inside a hex.
6. **Output** - schema dict (see ``board_schema.json``) validated through
   ``Board.validate``; per-element confidences and warnings in ``meta``.

Colour prototypes are tuned for Colonist's default theme.  For a different
theme or capture pipeline call ``VisionConfig.calibrate(image, known_json)``
on one labelled screenshot.

``cv2`` and ``numpy`` are imported lazily so the rest of the package works
without them; OCR engines are optional.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .board import (
    NUMBER_COUNTS, PIPS, PLAYER_COLORS, PORT_GENERIC, RESOURCES, SCHEMA_VERSION, TERRAIN_COUNTS,
    TERRAIN_RESOURCE, Board, Terrain, Topology, build_topology,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

RGB = tuple[int, int, int]


def _cv():
    import cv2  # noqa: WPS433
    import numpy as np  # noqa: WPS433
    return cv2, np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class VisionConfig:
    terrain_rgb: dict[Terrain, list[RGB]] = field(default_factory=lambda: {
        Terrain.FOREST: [(36, 112, 52), (52, 128, 60)],
        Terrain.HILLS: [(186, 88, 48), (170, 80, 50)],
        Terrain.PASTURE: [(146, 204, 84), (128, 190, 72)],
        Terrain.FIELDS: [(236, 192, 58), (226, 178, 50)],
        Terrain.MOUNTAINS: [(122, 132, 140), (140, 148, 156)],
        Terrain.DESERT: [(222, 204, 152), (210, 192, 140)],
    })
    water_rgb: RGB = (48, 128, 196)
    token_rgb: RGB = (246, 232, 200)
    player_rgb: dict[str, RGB] = field(default_factory=lambda: {
        "red": (222, 40, 40), "blue": (40, 90, 222), "orange": (250, 150, 10), "white": (250, 250, 250),
    })
    port_rgb: dict[str, RGB] = field(default_factory=lambda: {
        PORT_GENERIC: (255, 255, 255),
        "wood": (36, 112, 52), "brick": (186, 88, 48), "sheep": (146, 204, 84),
        "wheat": (236, 192, 58), "ore": (122, 132, 140),
    })
    land_threshold: float = 28.0         # LAB distance for the land mask
    piece_threshold: float = 30.0        # LAB distance for player colour matches
    piece_min_fraction: float = 0.55
    road_min_fraction: float = 0.75
    city_area_ratio: float = 1.45        # city blob area vs. settlement blob area
    token_radius: float = 0.34           # relative to hex size
    ocr: str = "auto"                    # auto | easyocr | tesseract | template
    hex_size_px: float | None = None     # manual calibration overrides
    center_px: tuple[float, float] | None = None
    refine: bool = True

    def calibrate(self, image: Any, known: dict[str, Any], *, size: float | None = None,
                  center: tuple[float, float] | None = None) -> "VisionConfig":
        """Learn terrain prototypes from a screenshot whose board JSON is known."""
        cv2, np = _cv()
        ext = BoardVisionExtractor(self)
        img = ext._load(image)
        geom = ext.locate_board(img) if size is None else BoardGeometry(center[0], center[1], size)  # type: ignore[index]
        lab = _to_lab(img)
        board = Board.from_dict(known)
        samples: dict[Terrain, list[Any]] = {}
        for t in board.tiles:
            samples.setdefault(t.terrain, []).append(ext._hex_sample(img, lab, geom, t.id)[1])
        for terrain, cols in samples.items():
            self.terrain_rgb[terrain] = [tuple(int(v) for v in np.median(np.array(cols), axis=0))]  # type: ignore[misc]
        return self


@dataclass
class BoardGeometry:
    cx: float
    cy: float
    size: float

    def hex_px(self, topo: Topology, h: int) -> tuple[float, float]:
        x, y = topo.hex_center_unit(*topo.hex_coords[h])
        return self.cx + x * self.size, self.cy + y * self.size

    def node_px(self, topo: Topology, n: int) -> tuple[float, float]:
        x, y = topo.node_center_unit(n)
        return self.cx + x * self.size, self.cy + y * self.size

    def edge_px(self, topo: Topology, e: int) -> tuple[float, float]:
        x, y = topo.edge_center_unit(e)
        return self.cx + x * self.size, self.cy + y * self.size


@dataclass
class ExtractionResult:
    data: dict[str, Any]
    geometry: BoardGeometry
    warnings: list[str]

    def to_json(self, path: str | Path | None = None) -> str:
        s = json.dumps(self.data, indent=2)
        if path:
            Path(path).write_text(s + "\n", encoding="utf-8")
        return s


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def _to_lab(img_bgr: Any) -> Any:
    cv2, np = _cv()
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _rgb_to_lab(rgb: RGB) -> Any:
    cv2, np = _cv()
    px = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]


def _disc_mask(shape: tuple[int, int], cx: float, cy: float, r_out: float, r_in: float = 0.0) -> Any:
    _, np = _cv()
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (d2 <= r_out ** 2) & (d2 >= r_in ** 2)


def _quota_assign(costs: Sequence[Sequence[float]], labels: Sequence[Any], quotas: dict[Any, int]) -> list[Any]:
    """Assign each row a label minimising total cost subject to label quotas."""
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment  # type: ignore

        cols = [lab for lab in labels for _ in range(quotas.get(lab, 0))]
        mat = np.array([[row[labels.index(c)] for c in cols] for row in costs])
        r, c = linear_sum_assignment(mat)
        out: list[Any] = [None] * len(costs)
        for i, j in zip(r, c):
            out[i] = cols[j]
        return out
    except ImportError:
        pass
    remaining = dict(quotas)
    pairs = sorted((costs[i][j], i, j) for i in range(len(costs)) for j in range(len(labels)))
    out = [None] * len(costs)
    for _, i, j in pairs:
        if out[i] is None and remaining.get(labels[j], 0) > 0:
            out[i] = labels[j]
            remaining[labels[j]] -= 1
    return out


# ---------------------------------------------------------------------------
# Number reading backends
# ---------------------------------------------------------------------------


class _DigitReader:
    def __init__(self, mode: str):
        self.mode = mode
        self._easy = None
        self._tess = None
        if mode in ("auto", "easyocr"):
            try:
                import easyocr  # type: ignore
                self._easy = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception:  # noqa: BLE001 - optional dependency
                self._easy = None
        if self._easy is None and mode in ("auto", "tesseract"):
            try:
                import pytesseract  # type: ignore
                pytesseract.get_tesseract_version()
                self._tess = pytesseract
            except Exception:  # noqa: BLE001
                self._tess = None
        self._templates: dict[str, Any] = {}

    @property
    def backend(self) -> str:
        return "easyocr" if self._easy else "tesseract" if self._tess else "template"

    def ocr(self, crop_gray: Any) -> int | None:
        text = ""
        if self._easy is not None:
            res = self._easy.readtext(crop_gray, allowlist="0123456789", detail=0)
            text = "".join(res)
        elif self._tess is not None:
            text = self._tess.image_to_string(
                crop_gray, config="--psm 7 -c tessedit_char_whitelist=0123456789")
        text = "".join(ch for ch in text if ch.isdigit())
        return int(text) if text and int(text) in NUMBER_COUNTS else None

    def template_scores(self, binary: Any) -> dict[int, float]:
        """Contour number matching: segment digits, correlate each with a rendered template."""
        cv2, np = _cv()
        n_lab, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
        comps = [i for i in range(1, n_lab)]
        if not comps:
            return {n: 0.0 for n in NUMBER_COUNTS}
        biggest = max(stats[i, cv2.CC_STAT_AREA] for i in comps)
        comps = sorted((i for i in comps if stats[i, cv2.CC_STAT_AREA] >= 0.25 * biggest),
                       key=lambda i: stats[i, cv2.CC_STAT_LEFT])
        glyphs = []
        for i in comps:
            x, y, w, h = (stats[i, k] for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                  cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            glyphs.append(binary[y:y + h, x:x + w].astype(np.float32))
        whole = binary.astype(np.float32)
        ys, xs = np.nonzero(whole)
        whole = whole[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        scores: dict[int, float] = {}
        for n in NUMBER_COUNTS:
            text = str(n)
            if len(text) == len(glyphs):
                scores[n] = sum(self._corr(gl, self._template(d)) for gl, d in zip(glyphs, text)) / len(text)
            else:  # touching digits / noise: fall back to whole-number correlation, penalised
                scores[n] = 0.8 * self._corr(whole, self._template(text))
        return scores

    @staticmethod
    def _corr(a: Any, b: Any, height: int = 32) -> float:
        cv2, np = _cv()
        ra = cv2.resize(a, (max(1, round(a.shape[1] * height / a.shape[0])), height), interpolation=cv2.INTER_AREA)
        rb = cv2.resize(b, (max(1, round(b.shape[1] * height / b.shape[0])), height), interpolation=cv2.INTER_AREA)
        width = max(ra.shape[1], rb.shape[1])
        pa = np.zeros((height, width), np.float32)
        pb = np.zeros((height, width), np.float32)
        # centre both glyphs horizontally so aspect-ratio differences cost, not misalign
        oa, ob = (width - ra.shape[1]) // 2, (width - rb.shape[1]) // 2
        pa[:, oa:oa + ra.shape[1]] = ra
        pb[:, ob:ob + rb.shape[1]] = rb
        den = math.sqrt(float((pa * pa).sum()) * float((pb * pb).sum())) or 1.0
        return float((pa * pb).sum()) / den

    def _template(self, text: str) -> Any:
        if text not in self._templates:
            cv2, np = _cv()
            canvas = np.zeros((80, 120), np.uint8)
            cv2.putText(canvas, text, (5, 65), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 5, cv2.LINE_AA)
            ys, xs = np.nonzero(canvas > 127)
            self._templates[text] = (canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 127).astype(np.float32)
        return self._templates[text]


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class BoardVisionExtractor:
    def __init__(self, config: VisionConfig | None = None):
        self.cfg = config or VisionConfig()
        self.topo = build_topology()
        self._digits: _DigitReader | None = None

    # -- entry point ---------------------------------------------------------
    def extract(self, image: Any, *, turn_order: Sequence[str] = PLAYER_COLORS,
                detect_pieces: bool = True) -> ExtractionResult:
        img = self._load(image)
        lab = _to_lab(img)
        geom = self.locate_board(img, lab)
        warnings: list[str] = []
        scale = 1.0
        if geom.size < 48 and not self.cfg.hex_size_px:
            # small captures: upscale so number tokens have enough pixels for OCR/templates
            cv2, _ = _cv()
            f = scale = 60.0 / geom.size
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            lab = _to_lab(img)
            geom = self.locate_board(img, lab)
            warnings.append(f"image upscaled x{f:.2f} (hex radius was {60.0 / f:.0f}px)")

        terrains, t_conf = self.classify_tiles(img, lab, geom)
        numbers, n_conf = self.read_numbers(img, geom, terrains)
        ports = self.detect_ports(img, lab, geom)
        robber = self.detect_robber(img, geom, terrains)
        players: dict[str, dict[str, list[int]]] = {c: {"settlements": [], "cities": [], "roads": []}
                                                    for c in turn_order}
        if detect_pieces:
            players = self.detect_pieces(img, lab, geom, turn_order)

        data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "source": "colonist.io screenshot",
            "layout": {"orientation": "pointy", "radius": 2},
            "hexes": [
                {"id": h, "q": q, "r": r, "s": -q - r, "terrain": terrains[h].value,
                 "resource": (TERRAIN_RESOURCE[terrains[h]].value if TERRAIN_RESOURCE[terrains[h]] else None),
                 "number": numbers[h], "confidence": round(min(t_conf[h], n_conf[h]), 3)}
                for h, (q, r) in enumerate(self.topo.hex_coords)
            ],
            "ports": ports,
            "robber_hex": robber,
            "turn_order": list(turn_order),
            "current_player": turn_order[0],
            "starting_resources": "second_settlement",
            "players": [{"color": c, **players[c]} for c in turn_order],
            "meta": {
                "geometry": {"cx": round(geom.cx / scale, 1), "cy": round(geom.cy / scale, 1),
                             "hex_size": round(geom.size / scale, 2)},
                "ocr_backend": self.digits.backend,
                "note": "settlement order unknown from a screenshot; settlements[1] is used for setup income",
            },
        }
        try:
            warnings += Board.from_dict(data).validate(strict_composition=True)
        except ValueError as exc:
            warnings.append(f"validation: {exc}")
        for h in data["hexes"]:
            if h["confidence"] < 0.5:
                warnings.append(f"low confidence on hex {h['id']} ({h['terrain']} {h['number']})")
        data["meta"]["warnings"] = warnings
        return ExtractionResult(data, geom, warnings)

    @property
    def digits(self) -> _DigitReader:
        if self._digits is None:
            self._digits = _DigitReader(self.cfg.ocr)
        return self._digits

    @staticmethod
    def _load(image: Any) -> Any:
        cv2, np = _cv()
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(image)
            return img
        return np.ascontiguousarray(image)

    # -- 1. localisation -----------------------------------------------------
    def locate_board(self, img: Any, lab: Any | None = None) -> BoardGeometry:
        cv2, np = _cv()
        lab = _to_lab(img) if lab is None else lab
        if self.cfg.hex_size_px and self.cfg.center_px:
            return BoardGeometry(self.cfg.center_px[0], self.cfg.center_px[1], self.cfg.hex_size_px)
        protos = [_rgb_to_lab(c) for cols in self.cfg.terrain_rgb.values() for c in cols]
        dmin = np.min(np.stack([np.linalg.norm(lab - p, axis=2) for p in protos]), axis=0)
        mask = (dmin < self.cfg.land_threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("no board-coloured region found; calibrate VisionConfig")
        # Ports / UI widgets share terrain colours but sit apart on the water: keep the
        # largest blob only, and fill its holes (tokens, pieces, robber).
        board_mask = np.zeros_like(mask)
        cv2.drawContours(board_mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
        geom = self._geom_from_mask(board_mask)
        if self.cfg.refine:
            geom = self._refine(board_mask, geom)
        return geom

    @staticmethod
    def _geom_from_mask(mask: Any) -> BoardGeometry:
        cv2, _ = _cv()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        size = (w / (5 * math.sqrt(3)) + h / 8.0) / 2.0
        return BoardGeometry(x + w / 2.0, y + h / 2.0, size)

    def _board_polygon_mask(self, shape: tuple[int, int], geom: BoardGeometry) -> Any:
        cv2, np = _cv()
        out = np.zeros(shape, np.uint8)
        for h in range(self.topo.num_hexes):
            pts = np.array([geom.node_px(self.topo, n) for n in self.topo.hex_nodes[h]], np.int32)
            cv2.fillPoly(out, [pts], 1)
        return out.astype(bool)

    def _refine(self, mask: Any, geom: BoardGeometry) -> BoardGeometry:
        """Coordinate-descent on IoU between the land mask and the predicted hex lattice."""
        land = mask > 0

        def iou(g: BoardGeometry) -> float:
            pred = self._board_polygon_mask(land.shape, g)
            return float((pred & land).sum()) / float((pred | land).sum() or 1)

        best, best_score = geom, iou(geom)
        for step in (0.04, 0.015, 0.005):
            improved = True
            while improved:
                improved = False
                for dx, dy, ds in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    g = BoardGeometry(best.cx + dx * step * best.size, best.cy + dy * step * best.size,
                                      best.size * (1 + ds * step))
                    sc = iou(g)
                    if sc > best_score + 1e-6:
                        best, best_score, improved = g, sc, True
        return best

    # -- 2. terrain ----------------------------------------------------------
    def _hex_sample(self, img: Any, lab: Any, geom: BoardGeometry, h: int) -> tuple[Any, RGB]:
        _, np = _cv()
        cx, cy = geom.hex_px(self.topo, h)
        r = geom.size
        x0, x1 = int(max(0, cx - r)), int(min(img.shape[1], cx + r))
        y0, y1 = int(max(0, cy - r)), int(min(img.shape[0], cy + r))
        sub_lab = lab[y0:y1, x0:x1]
        mask = _disc_mask(sub_lab.shape[:2], cx - x0, cy - y0, 0.72 * r, (self.cfg.token_radius + 0.08) * r)
        pix = sub_lab[mask]
        if pix.size == 0:
            return np.zeros(3, np.float32), (0, 0, 0)
        med_lab = np.median(pix, axis=0)
        bgr = np.median(img[y0:y1, x0:x1][mask], axis=0)
        return med_lab, (int(bgr[2]), int(bgr[1]), int(bgr[0]))

    def _terrain_distances(self, med_lab: Any) -> dict[Terrain, float]:
        _, np = _cv()
        return {t: min(float(np.linalg.norm(med_lab - _rgb_to_lab(c))) for c in cols)
                for t, cols in self.cfg.terrain_rgb.items()}

    def _classify_one(self, img: Any, lab: Any, geom: BoardGeometry, h: int) -> tuple[Terrain, float, dict[Terrain, float]]:
        med, _ = self._hex_sample(img, lab, geom, h)
        d = self._terrain_distances(med)
        ranked = sorted(d.items(), key=lambda kv: kv[1])
        (t1, d1), (_, d2) = ranked[0], ranked[1]
        conf = max(0.0, min(1.0, (d2 - d1) / (d2 + 1e-6))) * math.exp(-d1 / 60.0)
        return t1, conf, d

    def classify_tiles(self, img: Any, lab: Any, geom: BoardGeometry) -> tuple[list[Terrain], list[float]]:
        results = [self._classify_one(img, lab, geom, h) for h in range(self.topo.num_hexes)]
        terrains = [r[0] for r in results]
        conf = [r[1] for r in results]
        counts = {t: terrains.count(t) for t in Terrain}
        if counts != TERRAIN_COUNTS:
            labels = list(Terrain)
            costs = [[r[2][t] for t in labels] for r in results]
            fixed = _quota_assign(costs, labels, TERRAIN_COUNTS)
            for h, t in enumerate(fixed):
                if t != terrains[h]:
                    terrains[h], conf[h] = t, conf[h] * 0.5
        return terrains, conf

    # -- 3. numbers ------------------------------------------------------------
    def _token_crop(self, img: Any, geom: BoardGeometry, h: int) -> Any:
        cx, cy = geom.hex_px(self.topo, h)
        r = self.cfg.token_radius * geom.size
        x0, x1 = int(max(0, cx - r)), int(min(img.shape[1], cx + r))
        y0, y1 = int(max(0, cy - r)), int(min(img.shape[0], cy + r))
        return img[y0:y1, x0:x1]

    def _read_token(self, crop: Any) -> tuple[dict[int, float], int | None]:
        cv2, np = _cv()
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hgt, wid = crop.shape[:2]
        inner = _disc_mask((hgt, wid), wid / 2, hgt / 2, 0.88 * min(hgt, wid) / 2)
        ink = ((hsv[..., 2] < 110) | ((hsv[..., 1] > 120) & ((hsv[..., 0] < 10) | (hsv[..., 0] > 165))))
        ink &= inner
        red = ((hsv[..., 1] > 120) & ((hsv[..., 0] < 10) | (hsv[..., 0] > 165)) & inner)
        is_red = red.sum() > 0.25 * max(1, ink.sum())
        # Pip dots: small round components in the lower part of the token.
        n_lab, labels, stats, cents = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
        # Pip dots sit in a band below the digits.  The count is only used as a
        # prior when every dot is a separate, roughly round blob (touching dots at
        # low resolution make the count unreliable).
        dots = [i for i in range(1, n_lab) if cents[i][1] > 0.62 * hgt and stats[i, cv2.CC_STAT_AREA] >= 2
                and stats[i, cv2.CC_STAT_HEIGHT] <= 0.2 * hgt]
        separated = all(stats[i, cv2.CC_STAT_WIDTH] <= 1.5 * stats[i, cv2.CC_STAT_HEIGHT] for i in dots)
        pips = len(dots) if separated else 0
        digits_mask = ink.copy()
        digits_mask[int(0.62 * hgt):, :] = False
        scores = self.digits.template_scores(digits_mask)
        ocr_val = None
        if self.digits.backend != "template":
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ocr_val = self.digits.ocr(cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
        for n in scores:
            if ocr_val is not None:
                scores[n] += 1.0 if n == ocr_val else 0.0
            if 1 <= pips <= 5:
                scores[n] += 0.35 if PIPS[n] == pips else -0.35
            scores[n] += 0.25 if (n in (6, 8)) == is_red else -0.25
        return scores, ocr_val

    def read_numbers(self, img: Any, geom: BoardGeometry, terrains: Sequence[Terrain]) -> tuple[list[int | None], list[float]]:
        numbers: list[int | None] = [None] * self.topo.num_hexes
        conf = [1.0] * self.topo.num_hexes
        land = [h for h in range(self.topo.num_hexes) if terrains[h] != Terrain.DESERT]
        all_scores = {}
        for h in land:
            scores, _ = self._read_token(self._token_crop(img, geom, h))
            all_scores[h] = scores
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            numbers[h] = ranked[0][0]
            conf[h] = max(0.0, min(1.0, (ranked[0][1] - ranked[1][1]) / 0.25))
        counts: dict[int, int] = {}
        for h in land:
            counts[numbers[h]] = counts.get(numbers[h], 0) + 1  # type: ignore[index]
        if counts != NUMBER_COUNTS and len(land) == sum(NUMBER_COUNTS.values()):
            labels = sorted(NUMBER_COUNTS)
            costs = [[-all_scores[h][n] for n in labels] for h in land]
            fixed = _quota_assign(costs, labels, NUMBER_COUNTS)
            for h, n in zip(land, fixed):
                if n != numbers[h]:
                    numbers[h], conf[h] = n, conf[h] * 0.5
        return numbers, conf

    # -- 4. ports --------------------------------------------------------------
    def detect_ports(self, img: Any, lab: Any, geom: BoardGeometry) -> list[dict[str, Any]]:
        _, np = _cv()
        water = _rgb_to_lab(self.cfg.water_rgb)
        port_lab = {k: _rgb_to_lab(v) for k, v in self.cfg.port_rgb.items()}
        probes = []
        for e in self.topo.coastal_edges:
            ex, ey = self.topo.edge_center_unit(e)
            norm = math.hypot(ex, ey) or 1.0
            px = geom.cx + (ex + 0.75 * ex / norm) * geom.size
            py = geom.cy + (ey + 0.75 * ey / norm) * geom.size
            mask = _disc_mask(lab.shape[:2], px, py, 0.22 * geom.size)
            pix = lab[mask]
            if pix.size == 0:
                continue
            non_water = np.linalg.norm(pix - water, axis=1) > 25
            frac = float(non_water.mean())
            if frac < 0.3:
                continue
            med = np.median(pix[non_water], axis=0)
            kind, dist = min(((k, float(np.linalg.norm(med - v))) for k, v in port_lab.items()), key=lambda kv: kv[1])
            probes.append((frac, e, kind, dist))
        probes.sort(reverse=True)
        used: set[int] = set()
        ports = []
        for frac, e, kind, dist in probes:
            a, b = self.topo.edge_nodes[e]
            if a in used or b in used or len(ports) == 9:
                continue
            used.update((a, b))
            ports.append({"type": kind, "nodes": [a, b], "edge": e,
                          "confidence": round(frac * math.exp(-dist / 60.0), 3)})
        return sorted(ports, key=lambda p: self.topo.coastal_edges.index(p["edge"]))

    # -- 5. pieces & robber -------------------------------------------------------
    def _color_fraction(self, lab: Any, x: float, y: float, r: float, colors: Sequence[str]) -> dict[str, float]:
        _, np = _cv()
        x0, x1 = int(max(0, x - r)), int(min(lab.shape[1], x + r + 1))
        y0, y1 = int(max(0, y - r)), int(min(lab.shape[0], y + r + 1))
        sub = lab[y0:y1, x0:x1]
        mask = _disc_mask(sub.shape[:2], x - x0, y - y0, r)
        pix = sub[mask]
        if pix.size == 0:
            return {c: 0.0 for c in colors}
        return {c: float((np.linalg.norm(pix - _rgb_to_lab(self.cfg.player_rgb[c]), axis=1)
                          < self.cfg.piece_threshold).mean()) for c in colors}

    def detect_pieces(self, img: Any, lab: Any, geom: BoardGeometry,
                      colors: Sequence[str]) -> dict[str, dict[str, list[int]]]:
        out: dict[str, dict[str, list[int]]] = {c: {"settlements": [], "cities": [], "roads": []} for c in colors}
        s = geom.size
        for n in range(self.topo.num_nodes):
            x, y = geom.node_px(self.topo, n)
            core = self._color_fraction(lab, x, y, 0.14 * s, colors)
            c, f = max(core.items(), key=lambda kv: kv[1])
            if f < self.cfg.piece_min_fraction:
                continue
            outer = self._color_fraction(lab, x, y, 0.34 * s, colors)[c]
            # settlement disc ~ (0.2/0.34)^2 of the probe; a city fills noticeably more
            ratio = outer / ((0.20 / 0.34) ** 2)
            kind = "cities" if ratio > self.cfg.city_area_ratio else "settlements"
            out[c][kind].append(n)
        protos = {c: _rgb_to_lab(self.cfg.player_rgb[c]) for c in colors}
        for e in range(self.topo.num_edges):
            fr = self._line_fraction(lab, geom, e, protos)
            c, f = max(fr.items(), key=lambda kv: kv[1])
            if f >= self.cfg.road_min_fraction:
                out[c]["roads"].append(e)
        return out

    def _line_fraction(self, lab: Any, geom: BoardGeometry, e: int, protos: dict[str, Any]) -> dict[str, float]:
        """Fraction of points along the edge's middle third whose 3x3 neighbourhood (median) matches."""
        _, np = _cv()
        a, b = self.topo.edge_nodes[e]
        (ax, ay), (bx, by) = geom.node_px(self.topo, a), geom.node_px(self.topo, b)
        ts = np.linspace(0.35, 0.65, 9)
        hits = {c: 0 for c in protos}
        h, w = lab.shape[:2]
        for t in ts:
            x, y = int(round(ax + (bx - ax) * t)), int(round(ay + (by - ay) * t))
            patch = lab[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)].reshape(-1, 3)
            if patch.size == 0:
                continue
            for c, proto in protos.items():
                if float(np.median(np.linalg.norm(patch - proto, axis=1))) < self.cfg.piece_threshold:
                    hits[c] += 1
        return {c: k / len(ts) for c, k in hits.items()}

    def detect_robber(self, img: Any, geom: BoardGeometry, terrains: Sequence[Terrain]) -> int:
        cv2, np = _cv()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        best, best_frac = None, 0.0
        for h in range(self.topo.num_hexes):
            cx, cy = geom.hex_px(self.topo, h)
            r = geom.size
            x0, x1 = int(max(0, cx - r)), int(min(img.shape[1], cx + r))
            y0, y1 = int(max(0, cy - r)), int(min(img.shape[0], cy + r))
            sub = hsv[y0:y1, x0:x1]
            mask = _disc_mask(sub.shape[:2], cx - x0, cy - y0, 0.7 * r, (self.cfg.token_radius + 0.05) * r)
            dark = (sub[..., 2] < 75) & (sub[..., 1] < 90)
            frac = float(dark[mask].mean()) if mask.any() else 0.0
            if frac > best_frac:
                best, best_frac = h, frac
        if best is None or best_frac < 0.04:
            return terrains.index(Terrain.DESERT)
        return best

    # -- debugging ----------------------------------------------------------------
    def render_debug(self, image: Any, geom: BoardGeometry | None = None) -> Any:
        """Overlay hex ids, node ids and edge ids to verify the lattice alignment."""
        cv2, _ = _cv()
        img = self._load(image).copy()
        geom = geom or self.locate_board(img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for h in range(self.topo.num_hexes):
            x, y = geom.hex_px(self.topo, h)
            cv2.putText(img, f"H{h}", (int(x) - 12, int(y) - int(0.45 * geom.size)), font, 0.4, (0, 0, 0), 1)
        for n in range(self.topo.num_nodes):
            x, y = geom.node_px(self.topo, n)
            cv2.circle(img, (int(x), int(y)), 3, (255, 0, 255), -1)
            cv2.putText(img, str(n), (int(x) + 3, int(y) - 3), font, 0.33, (80, 0, 80), 1)
        for e in range(self.topo.num_edges):
            x, y = geom.edge_px(self.topo, e)
            cv2.putText(img, str(e), (int(x) - 5, int(y) + 4), font, 0.28, (0, 90, 0), 1)
        return img


# ---------------------------------------------------------------------------
# Synthetic Colonist-style renderer (tests, calibration, demos)
# ---------------------------------------------------------------------------


def render_synthetic_board(data: dict[str, Any], *, size: float = 56.0, margin: float = 2.2,
                           config: VisionConfig | None = None) -> Any:
    """Draw a board JSON as a flat Colonist-like screenshot (BGR ndarray)."""
    cv2, np = _cv()
    cfg = config or VisionConfig()
    topo = build_topology()
    board = Board.from_dict(data)
    w = int(size * (5 * math.sqrt(3) + 2 * margin))
    h = int(size * (8 + 2 * margin))
    bgr = lambda c: (int(c[2]), int(c[1]), int(c[0]))  # noqa: E731
    img = np.full((h, w, 3), bgr(cfg.water_rgb), np.uint8)
    geom = BoardGeometry(w / 2.0, h / 2.0, size)
    for t in board.tiles:
        cx, cy = geom.hex_px(topo, t.id)
        pts = np.array([geom.node_px(topo, n) for n in topo.hex_nodes[t.id]], np.int32)
        cv2.fillPoly(img, [pts], bgr(cfg.terrain_rgb[t.terrain][0]), cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (60, 60, 60), 1, cv2.LINE_AA)
        if t.number:
            r = int(cfg.token_radius * size * 0.95)
            cv2.circle(img, (int(cx), int(cy)), r, bgr(cfg.token_rgb), -1, cv2.LINE_AA)
            ink = (40, 40, 200) if t.number in (6, 8) else (30, 30, 30)
            text = str(t.number)
            scale = size / 75.0
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            cv2.putText(img, text, (int(cx - tw / 2), int(cy + th / 2 - 0.06 * size)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 2, cv2.LINE_AA)
            pips = PIPS[t.number]
            gap = 0.1 * size
            for i in range(pips):
                px = cx + (i - (pips - 1) / 2) * gap
                cv2.circle(img, (int(round(px)), int(cy + 0.21 * size)), max(1, round(0.024 * size)), ink, -1,
                           cv2.LINE_AA)
    if board.robber_hex is not None and board.tiles[board.robber_hex].number:
        cx, cy = geom.hex_px(topo, board.robber_hex)
        cv2.ellipse(img, (int(cx), int(cy - 0.52 * size)), (int(0.13 * size), int(0.17 * size)),
                    0, 0, 360, (45, 45, 45), -1)
    for p in board.ports:
        e = topo.edge_between(*p.nodes)
        ex, ey = topo.edge_center_unit(e)
        norm = math.hypot(ex, ey)
        px = geom.cx + (ex + 0.75 * ex / norm) * size
        py = geom.cy + (ey + 0.75 * ey / norm) * size
        for n in p.nodes:
            nx, ny = geom.node_px(topo, n)
            cv2.line(img, (int(nx), int(ny)), (int(px), int(py)), (200, 220, 230), 2)
        cv2.circle(img, (int(px), int(py)), int(0.27 * size), bgr(cfg.port_rgb[p.type]), -1, cv2.LINE_AA)
    for pd in data.get("players", []):
        col = bgr(cfg.player_rgb[pd["color"]])
        for e in pd.get("roads", []):
            a, b = topo.edge_nodes[int(e)]
            (ax, ay), (bx, by) = geom.node_px(topo, a), geom.node_px(topo, b)
            t0, t1 = 0.22, 0.78
            p0 = (int(ax + (bx - ax) * t0), int(ay + (by - ay) * t0))
            p1 = (int(ax + (bx - ax) * t1), int(ay + (by - ay) * t1))
            cv2.line(img, p0, p1, (20, 20, 20), max(2, int(0.16 * size)), cv2.LINE_AA)
            cv2.line(img, p0, p1, col, max(2, int(0.11 * size)), cv2.LINE_AA)
        for kind, rad in (("settlements", 0.20), ("cities", 0.29)):
            for n in pd.get(kind, []):
                x, y = geom.node_px(topo, int(n))
                cv2.circle(img, (int(x), int(y)), int(rad * size) + 2, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(img, (int(x), int(y)), int(rad * size), col, -1, cv2.LINE_AA)
    return img


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    import argparse

    ap = argparse.ArgumentParser(description="Extract a Catan board JSON from a Colonist.io screenshot")
    ap.add_argument("image")
    ap.add_argument("-o", "--out", help="write JSON here")
    ap.add_argument("--debug", help="write an overlay PNG with hex/node/edge ids")
    ap.add_argument("--ocr", default="auto", choices=["auto", "easyocr", "tesseract", "template"])
    ap.add_argument("--size", type=float, help="manual hex circumradius in px")
    ap.add_argument("--center", type=float, nargs=2, help="manual board centre x y in px")
    ns = ap.parse_args(argv)
    cfg = VisionConfig(ocr=ns.ocr, hex_size_px=ns.size, center_px=tuple(ns.center) if ns.center else None)
    ext = BoardVisionExtractor(cfg)
    res = ext.extract(ns.image)
    print(res.to_json(ns.out))
    for w in res.warnings:
        print("warning:", w)
    if ns.debug:
        import cv2
        cv2.imwrite(ns.debug, ext.render_debug(ns.image, res.geometry))


if __name__ == "__main__":  # pragma: no cover
    main()
