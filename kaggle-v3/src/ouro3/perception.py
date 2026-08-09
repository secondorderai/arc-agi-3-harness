"""General, game-agnostic perception primitives for 64x64 ARC frames."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, deque
from typing import Any, Iterable, Sequence

Grid = Sequence[Sequence[int]]
Cell = tuple[int, int]


def normalize_grid(raw: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    rows: list[tuple[int, ...]] = []
    for raw_row in raw:
        if not isinstance(raw_row, (list, tuple)):
            continue
        row: list[int] = []
        for value in raw_row:
            try:
                row.append(int(value))
            except (TypeError, ValueError):
                row.append(0)
        rows.append(tuple(row))
    return tuple(rows)


def grid_shape(grid: Grid) -> tuple[int, int]:
    return len(grid), max((len(row) for row in grid), default=0)


def _neighbors(
    row: int,
    col: int,
    rows: int,
    cols: int,
    *,
    connectivity: int = 4,
) -> Iterable[Cell]:
    if row:
        yield row - 1, col
    if row + 1 < rows:
        yield row + 1, col
    if col:
        yield row, col - 1
    if col + 1 < cols:
        yield row, col + 1
    if connectivity == 8:
        for row_delta, col_delta in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            other_row = row + row_delta
            other_col = col + col_delta
            if 0 <= other_row < rows and 0 <= other_col < cols:
                yield other_row, other_col
    elif connectivity != 4:
        raise ValueError("connectivity must be 4 or 8")


def _components_from_cells(
    cells: set[Cell],
    rows: int,
    cols: int,
    *,
    connectivity: int = 4,
) -> list[list[Cell]]:
    remaining = set(cells)
    components: list[list[Cell]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue: deque[Cell] = deque([start])
        component: list[Cell] = []
        while queue:
            cell = queue.popleft()
            component.append(cell)
            for neighbor in _neighbors(
                cell[0], cell[1], rows, cols, connectivity=connectivity
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def _bbox(cells: Sequence[Cell]) -> list[int]:
    row_values = [cell[0] for cell in cells]
    col_values = [cell[1] for cell in cells]
    return [min(row_values), min(col_values), max(row_values), max(col_values)]


def _shape_hash(cells: Sequence[Cell]) -> str:
    row0 = min(row for row, _ in cells)
    col0 = min(col for _, col in cells)
    canonical = ";".join(f"{row-row0},{col-col0}" for row, col in sorted(cells))
    return hashlib.sha1(canonical.encode("ascii")).hexdigest()[:12]


def segment_grid(
    grid: Grid,
    *,
    include_background: bool = False,
    connectivity: int = 4,
) -> list[dict[str, Any]]:
    """Return deterministic color components with containment metadata.

    The historical behavior remains the default.  Retrodictive modes can keep
    both four- and eight-connected interpretations alive until transitions
    provide evidence for one of them.
    """

    normalized = normalize_grid(grid)
    rows, cols = grid_shape(normalized)
    if not rows or not cols:
        return []
    counts = Counter(cell for row in normalized for cell in row)
    background = counts.most_common(1)[0][0]
    by_color: dict[int, set[Cell]] = {}
    for row_index, row in enumerate(normalized):
        for col_index, color in enumerate(row):
            if include_background or color != background:
                by_color.setdefault(color, set()).add((row_index, col_index))

    objects: list[dict[str, Any]] = []
    for color in sorted(by_color):
        for cells in _components_from_cells(
            by_color[color], rows, cols, connectivity=connectivity
        ):
            box = _bbox(cells)
            objects.append(
                {
                    "id": len(objects),
                    "color": color,
                    "area": len(cells),
                    "bbox": box,
                    "centroid": [
                        round(sum(row for row, _ in cells) / len(cells), 3),
                        round(sum(col for _, col in cells) / len(cells), 3),
                    ],
                    "height": box[2] - box[0] + 1,
                    "width": box[3] - box[1] + 1,
                    "shape_hash": _shape_hash(cells),
                    "touches_border": bool(
                        box[0] == 0 or box[1] == 0 or box[2] == rows - 1 or box[3] == cols - 1
                    ),
                }
            )
    # Dense checkerboards can create thousands of single-cell components.
    # Keep segmentation linear in that case; pairwise relations would be
    # quadratic and add little useful evidence.
    compute_pairwise_relations = len(objects) <= 256
    for obj in objects:
        inner = obj["bbox"]
        obj["contained_by"] = (
            [
                outer["id"]
                for outer in objects
                if outer["id"] != obj["id"]
                and outer["bbox"][0] <= inner[0]
                and outer["bbox"][1] <= inner[1]
                and outer["bbox"][2] >= inner[2]
                and outer["bbox"][3] >= inner[3]
            ]
            if compute_pairwise_relations
            else []
        )
        obj["adjacent_to"] = (
            [
                other["id"]
                for other in objects
                if other["id"] != obj["id"]
                and _bbox_gap(inner, other["bbox"]) <= 1
            ]
            if compute_pairwise_relations
            else []
        )
    return objects


def segment_grid_multi(grid: Grid) -> dict[str, list[dict[str, Any]]]:
    """Return a bounded, deterministic set of competing object ontologies."""

    return {
        "color-4": segment_grid(grid, connectivity=4),
        "color-8": segment_grid(grid, connectivity=8),
        "color-4-all": segment_grid(
            grid, include_background=True, connectivity=4
        ),
    }


def _bbox_gap(first: Sequence[int], second: Sequence[int]) -> int:
    row_gap = max(0, max(first[0], second[0]) - min(first[2], second[2]) - 1)
    col_gap = max(0, max(first[1], second[1]) - min(first[3], second[3]) - 1)
    return max(row_gap, col_gap)


def changed_regions(before: Grid, after: Grid) -> list[dict[str, Any]]:
    old = normalize_grid(before)
    new = normalize_grid(after)
    rows = max(len(old), len(new))
    cols = max(grid_shape(old)[1], grid_shape(new)[1])
    cells: set[Cell] = set()
    for row in range(rows):
        for col in range(cols):
            old_value = old[row][col] if row < len(old) and col < len(old[row]) else None
            new_value = new[row][col] if row < len(new) and col < len(new[row]) else None
            if old_value != new_value:
                cells.add((row, col))
    regions: list[dict[str, Any]] = []
    strip_rows = max(2, math.ceil(rows * 0.125)) if rows >= 16 else 0
    strip_cols = max(2, math.ceil(cols * 0.125)) if cols >= 16 else 0
    for component in _components_from_cells(cells, rows, cols):
        box = _bbox(component)
        entirely_in_edge_strip = bool(
            (strip_rows and (box[2] < strip_rows or box[0] >= rows - strip_rows))
            or (strip_cols and (box[3] < strip_cols or box[1] >= cols - strip_cols))
        )
        region = {
            "id": len(regions),
            "area": len(component),
            "bbox": box,
            "centroid": [
                round(sum(row for row, _ in component) / len(component), 3),
                round(sum(col for _, col in component) / len(component), 3),
            ],
            "classification": "hud" if entirely_in_edge_strip else "gameplay",
        }
        regions.append(region)
    return regions


def track_objects(
    before: Grid,
    after: Grid,
    *,
    connectivity: int = 4,
) -> list[dict[str, Any]]:
    old_objects = segment_grid(before, connectivity=connectivity)
    new_objects = segment_grid(after, connectivity=connectivity)
    unmatched_new = set(range(len(new_objects)))
    tracks: list[dict[str, Any]] = []
    for old in old_objects:
        candidates = [
            (index, new)
            for index, new in enumerate(new_objects)
            if index in unmatched_new and new["color"] == old["color"]
        ]
        if not candidates:
            tracks.append({"status": "disappeared", "before": old, "after": None})
            continue
        index, matched = min(candidates, key=lambda pair: _object_match_cost(old, pair[1]))
        unmatched_new.remove(index)
        delta = [
            round(matched["centroid"][0] - old["centroid"][0], 3),
            round(matched["centroid"][1] - old["centroid"][1], 3),
        ]
        status = "unchanged"
        if delta != [0.0, 0.0]:
            status = "moved"
        if old["shape_hash"] != matched["shape_hash"] or old["area"] != matched["area"]:
            status = "changed"
        tracks.append(
            {
                "status": status,
                "color": old["color"],
                "delta": delta,
                "before_id": old["id"],
                "after_id": matched["id"],
                "before_bbox": old["bbox"],
                "after_bbox": matched["bbox"],
            }
        )
    for index in sorted(unmatched_new):
        tracks.append({"status": "appeared", "before": None, "after": new_objects[index]})
    return tracks


def _object_match_cost(first: dict[str, Any], second: dict[str, Any]) -> float:
    shape_penalty = 0 if first["shape_hash"] == second["shape_hash"] else 8
    area_penalty = abs(first["area"] - second["area"]) / max(1, first["area"])
    distance = math.dist(first["centroid"], second["centroid"])
    return shape_penalty + area_penalty + distance


def animation_summary(history: Sequence[Grid], *, max_period: int = 6) -> dict[str, Any]:
    frames = [normalize_grid(frame) for frame in history if frame]
    if len(frames) < 2:
        return {"detected": False, "period": None, "changing_pixels": 0}
    signatures = [
        hashlib.sha1(repr(frame).encode("utf-8")).hexdigest()[:12] for frame in frames
    ]
    detected_period: int | None = None
    for period in range(1, min(max_period, len(signatures) // 2) + 1):
        if signatures[-period:] == signatures[-2 * period : -period]:
            detected_period = period
            break
    changed = {
        (row, col)
        for first, second in zip(frames, frames[1:])
        for row in range(min(len(first), len(second)))
        for col in range(min(len(first[row]), len(second[row])))
        if first[row][col] != second[row][col]
    }
    return {
        "detected": detected_period is not None or len(changed) > 0,
        "period": detected_period,
        "changing_pixels": len(changed),
        "frame_count": len(frames),
    }


def analyze_frame(
    grid: Grid,
    *,
    previous_grid: Grid | None = None,
    recent_grids: Sequence[Grid] | None = None,
) -> dict[str, Any]:
    normalized = normalize_grid(grid)
    rows, cols = grid_shape(normalized)
    prior = normalize_grid(previous_grid) if previous_grid is not None else ()
    regions = changed_regions(prior, normalized) if prior else []
    tracks = track_objects(prior, normalized) if prior else []
    return {
        "shape": [rows, cols],
        "segmentation": segment_grid(normalized),
        "changed_regions": regions,
        "object_tracks": tracks,
        "animation_summary": animation_summary([*(recent_grids or []), normalized]),
    }


def analyze_transition(
    before: Grid,
    after: Grid,
    *,
    recent_grids: Sequence[Grid] | None = None,
) -> dict[str, Any]:
    regions = changed_regions(before, after)
    gameplay_regions = [region for region in regions if region["classification"] == "gameplay"]
    hud_regions = [region for region in regions if region["classification"] == "hud"]
    return {
        "board_changed": bool(regions),
        "gameplay_changed": bool(gameplay_regions),
        "hud_changed": bool(hud_regions),
        "changed_regions": regions,
        "object_tracks": track_objects(before, after),
        "animation_summary": animation_summary([*(recent_grids or []), before, after]),
    }


def candidate_sequence_score(
    observations: Sequence[dict[str, Any]],
    *,
    predicted_gameplay_changes: int = 0,
    terminal_penalty: float = 100.0,
) -> float:
    """Generic score used by model-written searches and deterministic fallback."""

    gameplay_changes = sum(bool(item.get("gameplay_changed")) for item in observations)
    level_gains = sum(bool(item.get("level_completed")) for item in observations)
    terminal_failures = sum(bool(item.get("game_over")) for item in observations)
    mismatch = abs(gameplay_changes - predicted_gameplay_changes)
    return level_gains * 1_000.0 + gameplay_changes * 5.0 - mismatch * 7.0 - terminal_failures * terminal_penalty


def shortest_path(
    start: Cell,
    goals: set[Cell],
    *,
    rows: int,
    cols: int,
    blocked: set[Cell] | None = None,
) -> list[Cell]:
    blocked = blocked or set()
    if start in goals:
        return [start]
    queue: deque[Cell] = deque([start])
    parent: dict[Cell, Cell | None] = {start: None}
    end: Cell | None = None
    while queue:
        current = queue.popleft()
        for neighbor in _neighbors(current[0], current[1], rows, cols):
            if neighbor in blocked or neighbor in parent:
                continue
            parent[neighbor] = current
            if neighbor in goals:
                end = neighbor
                queue.clear()
                break
            queue.append(neighbor)
    if end is None:
        return []
    path: list[Cell] = []
    cursor: Cell | None = end
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    return list(reversed(path))
