from __future__ import annotations

from ouro3.perception import segment_grid_multi
from ouro3.retrodict import RetrodictiveWorldModel


def _click_grid(columns: tuple[int, ...], recolored: tuple[int, ...] = ()):
    grid = [[0] * 8 for _ in range(5)]
    for col in columns:
        grid[2][col] = 2 if col in recolored else 1
    return tuple(tuple(row) for row in grid)


def test_multi_ontology_keeps_four_and_eight_connected_views() -> None:
    grid = ((1, 0), (0, 1))
    views = segment_grid_multi(grid)
    assert len(views["color-4"]) == 2
    assert len(views["color-8"]) == 1
    assert set(views) == {"color-4", "color-8", "color-4-all"}


def test_click_rule_is_certified_by_full_log_and_generalizes() -> None:
    world = RetrodictiveWorldModel()
    first = _click_grid((1, 3, 5))
    second = _click_grid((1, 3, 5), (1,))
    third = _click_grid((1, 3, 5), (1, 3))
    world.observe(
        level=1,
        action={"action": "MOUSE", "row": 2, "col": 1},
        before=first,
        after=second,
    )
    world.observe(
        level=1,
        action={"action": "MOUSE", "row": 2, "col": 3},
        before=second,
        after=third,
    )
    prediction = world.predict(
        third,
        {"action": "MOUSE", "row": 2, "col": 5},
    )
    assert prediction is not None
    assert prediction.certified
    assert prediction.grid == _click_grid((1, 3, 5), (1, 3, 5))
    click_rules = [rule for rule in world.rules.values() if rule.kind == "click-recolor"]
    assert click_rules
    assert all(not rule.contradictions for rule in click_rules)
    assert any(rule.certified for rule in click_rules)


def test_contradiction_removes_rule_certification() -> None:
    world = RetrodictiveWorldModel()
    states = [
        _click_grid((1, 3, 5)),
        _click_grid((1, 3, 5), (1,)),
        _click_grid((1, 3, 5), (1, 3)),
    ]
    for col, before, after in zip((1, 3), states, states[1:]):
        world.observe(
            level=1,
            action={"action": "MOUSE", "row": 2, "col": col},
            before=before,
            after=after,
        )
    world.observe(
        level=1,
        action={"action": "MOUSE", "row": 2, "col": 5},
        before=states[-1],
        after=states[-1],
    )
    click_rules = [rule for rule in world.rules.values() if rule.kind == "click-recolor"]
    assert click_rules
    assert all(rule.contradictions for rule in click_rules)
    assert not any(rule.certified for rule in click_rules)


def test_contextual_click_noop_generalizes_without_global_mouse_noop() -> None:
    world = RetrodictiveWorldModel()

    def state(marker_col: int) -> tuple[tuple[int, ...], ...]:
        values = [[0] * 8 for _ in range(5)]
        values[1][1] = 2
        values[3][marker_col] = 3
        return tuple(tuple(row) for row in values)

    for marker_col in (3, 4):
        grid = state(marker_col)
        world.observe(
            level=1,
            action={"action": "MOUSE", "row": 1, "col": 1},
            before=grid,
            after=grid,
        )

    prediction = world.predict(
        state(5),
        {"action": "MOUSE", "row": 1, "col": 1},
        level=1,
    )
    assert prediction is not None
    assert prediction.grid == state(5)
    assert any(
        world.rules[rule_id].kind == "click-noop"
        for rule_id in prediction.rule_ids
    )


def test_click_inside_hollow_object_recolors_its_border() -> None:
    def state(
        left: int,
        recolored: tuple[int, ...] = (),
    ) -> tuple[tuple[int, ...], ...]:
        values = [[0] * 12 for _ in range(7)]
        for offset in (0, 4, 8):
            color = 3 if offset in recolored else 2
            for row in range(2, 5):
                for col in range(left + offset, left + offset + 3):
                    if row in (2, 4) or col in (left + offset, left + offset + 2):
                        values[row][col] = color
        return tuple(tuple(row) for row in values)

    world = RetrodictiveWorldModel()
    first = state(1)
    second = state(1, (0,))
    third = state(1, (0, 4))
    world.observe(
        level=1,
        action={"action": "MOUSE", "row": 3, "col": 2},
        before=first,
        after=second,
    )
    world.observe(
        level=1,
        action={"action": "MOUSE", "row": 3, "col": 6},
        before=second,
        after=third,
    )
    prediction = world.predict(
        third,
        {"action": "MOUSE", "row": 3, "col": 10},
        level=1,
    )
    assert prediction is not None
    assert prediction.certified
    assert prediction.grid == state(1, (0, 4, 8))


def test_object_translation_composes_two_edge_counters() -> None:
    def state(anchor_col: int, consumed: int) -> tuple[tuple[int, ...], ...]:
        values = [[5] * 24 for _ in range(10)]
        for row in (4, 5):
            for col in (anchor_col, anchor_col + 1):
                values[row][col] = 10
        for col in range(consumed):
            values[0][col] = 0
            values[-1][-1 - col] = 0
        return tuple(tuple(row) for row in values)

    world = RetrodictiveWorldModel()
    world.observe(level=1, action="RIGHT", before=state(2, 1), after=state(7, 2))
    world.observe(level=1, action="RIGHT", before=state(7, 2), after=state(12, 3))
    world.observe(level=1, action="RIGHT", before=state(12, 3), after=state(17, 4))
    prediction = world.predict(state(17, 4), "RIGHT", level=1)
    assert prediction is not None
    assert prediction.certified
    assert prediction.grid == state(22, 5)


def test_object_translation_composes_nested_pattern_and_border_step() -> None:
    def grid(anchor_col: int, consumed: int) -> tuple[tuple[int, ...], ...]:
        values = [[0 for _ in range(8)] for _ in range(6)]
        for row, col, color in (
            (2, anchor_col, 1),
            (2, anchor_col + 1, 1),
            (3, anchor_col, 1),
            (3, anchor_col + 1, 2),
        ):
            values[row][col] = color
        for row in range(6):
            values[row][7] = 4 if row < consumed else 3
        return tuple(tuple(row) for row in values)

    world = RetrodictiveWorldModel()
    world.observe(
        level=1,
        action="RIGHT",
        before=grid(1, 1),
        after=grid(2, 2),
    )
    world.observe(
        level=1,
        action="RIGHT",
        before=grid(2, 2),
        after=grid(3, 3),
    )
    world.observe(
        level=1,
        action="RIGHT",
        before=grid(3, 3),
        after=grid(4, 4),
    )
    prediction = world.predict(grid(4, 4), "RIGHT", level=1)
    assert prediction is not None
    assert prediction.grid == grid(5, 5)
    assert prediction.certified


def test_blocked_object_motion_is_learned_as_a_separate_noop() -> None:
    def state(col: int) -> tuple[tuple[int, ...], ...]:
        values = [[0] * 4 for _ in range(3)]
        values[1][col] = 2
        return tuple(tuple(row) for row in values)

    world = RetrodictiveWorldModel()
    for col in range(3):
        world.observe(
            level=1,
            action="RIGHT",
            before=state(col),
            after=state(col + 1),
        )
    for _ in range(2):
        world.observe(
            level=1,
            action="RIGHT",
            before=state(3),
            after=state(3),
        )
    prediction = world.predict(state(3), "RIGHT", level=1)
    assert prediction is not None
    assert prediction.grid == state(3)
    assert any(
        rule.kind == "object-blocked-noop" and rule.certified
        for rule in world.rules.values()
    )


def test_exact_replay_plan_and_alias_conflict_are_fail_closed() -> None:
    world = RetrodictiveWorldModel()
    first = ((1, 0, 0),)
    second = ((0, 1, 0),)
    goal = ((0, 0, 1),)
    world.observe(level=1, action="RIGHT", before=first, after=second)
    world.observe(level=1, action="RIGHT", before=first, after=second)
    world.observe(
        level=1,
        action="SPACE",
        before=second,
        after=goal,
        payload={"level_completed": True},
    )
    world.observe(
        level=1,
        action="SPACE",
        before=second,
        after=goal,
        payload={"level_completed": True},
    )
    plan = world.plan(first, level=1, valid_actions=[])
    assert plan is not None
    assert plan.source == "exact-replay"
    assert [action["action"] for action in plan.actions] == ["RIGHT", "SPACE"]

    alternate = ((2, 0, 0),)
    world.observe(level=1, action="RIGHT", before=first, after=alternate)
    assert world.clone_graph.active
    assert world.plan(
        first,
        level=1,
        valid_actions=[],
    ) is None


def test_certified_typed_rule_supports_cpu_search_on_unseen_state() -> None:
    world = RetrodictiveWorldModel()
    world.observe(
        level=1,
        action="RIGHT",
        before=((1, 0, 0, 0),),
        after=((0, 1, 0, 0),),
    )
    world.observe(
        level=1,
        action="RIGHT",
        before=((0, 1, 0, 0),),
        after=((0, 0, 1, 0),),
        payload={"level_completed": True},
    )
    plan = world.plan(
        ((0, 0, 1, 0),),
        level=1,
        valid_actions=["RIGHT"],
    )
    assert plan is not None
    assert plan.source == "typed-certified-search"
    assert plan.actions == ({"action": "RIGHT"},)


def test_probe_selection_reports_model_disagreement_and_novelty() -> None:
    world = RetrodictiveWorldModel()
    state = ((1, 0, 0),)
    world.observe(level=1, action="RIGHT", before=state, after=((0, 1, 0),))
    probe = world.select_probe(state, ["RIGHT", "SPACE"])
    assert probe is not None
    assert probe.action["action"] in {"RIGHT", "SPACE"}
    assert probe.risk == 0.0


def test_identical_screen_actions_are_exactly_scoped_by_level() -> None:
    world = RetrodictiveWorldModel()
    before = ((1, 0),)
    first = ((0, 1),)
    second = ((2, 0),)
    world.observe(level=1, action="RIGHT", before=before, after=first)
    world.observe(level=2, action="RIGHT", before=before, after=second)
    world.observe(level=1, action="RIGHT", before=before, after=first)
    world.observe(level=2, action="RIGHT", before=before, after=second)
    level_one = world.predict(before, "RIGHT", level=1)
    level_two = world.predict(before, "RIGHT", level=2)
    assert level_one is not None and level_one.grid == first
    assert level_two is not None and level_two.grid == second
    assert not world.clone_graph.active
