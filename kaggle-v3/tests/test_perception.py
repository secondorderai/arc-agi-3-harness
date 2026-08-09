from __future__ import annotations

from ouro3.perception import (
    analyze_transition,
    animation_summary,
    candidate_sequence_score,
    segment_grid,
    shortest_path,
    track_objects,
)


def test_synthetic_movement_tracks_object_delta() -> None:
    before = [[0] * 8 for _ in range(8)]
    after = [[0] * 8 for _ in range(8)]
    before[3][2] = 4
    after[3][3] = 4
    tracks = track_objects(before, after)
    moved = next(track for track in tracks if track["status"] == "moved")
    assert moved["delta"] == [0.0, 1.0]
    transition = analyze_transition(before, after)
    assert transition["gameplay_changed"]
    assert not transition["hud_changed"]


def test_synthetic_click_target_segmentation_is_stable() -> None:
    grid = [[0] * 12 for _ in range(12)]
    for row in range(4, 7):
        for col in range(7, 10):
            grid[row][col] = 8
    objects = segment_grid(grid)
    assert len(objects) == 1
    assert objects[0]["area"] == 9
    assert objects[0]["centroid"] == [5.0, 8.0]
    assert len(objects[0]["shape_hash"]) == 12


def test_animation_period_and_candidate_score() -> None:
    first = [[0, 1], [0, 0]]
    second = [[0, 0], [1, 0]]
    summary = animation_summary([first, second, first, second])
    assert summary["detected"]
    assert summary["period"] == 2
    assert summary["changing_pixels"] == 2
    assert candidate_sequence_score(
        [{"gameplay_changed": True}, {"level_completed": True}],
        predicted_gameplay_changes=1,
    ) > 1_000


def test_shortest_path_avoids_blocked_cells() -> None:
    path = shortest_path(
        (0, 0),
        {(2, 2)},
        rows=3,
        cols=3,
        blocked={(0, 1), (1, 1)},
    )
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert not set(path) & {(0, 1), (1, 1)}
