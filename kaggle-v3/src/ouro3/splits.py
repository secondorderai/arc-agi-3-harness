"""Frozen public-game split shared with earlier Ouroboros ARC work."""

from __future__ import annotations

DEV: tuple[str, ...] = (
    "ft09",
    "m0r0",
    "sp80",
    "s5i5",
    "ls20",
    "lp85",
    "cn04",
    "tr87",
    "sb26",
    "sk48",
    "bp35",
    "r11l",
    "tu93",
)
TEST: tuple[str, ...] = (
    "vc33",
    "lf52",
    "su15",
    "sc25",
    "g50t",
    "wa30",
    "ka59",
    "dc22",
    "tn36",
)
QUARANTINE: tuple[str, ...] = ("ar25", "re86", "cd82")
PUBLIC: tuple[str, ...] = DEV + TEST + QUARANTINE

FOLDS: dict[str, tuple[str, ...]] = {
    "dev": DEV,
    "test": TEST,
    "quarantine": QUARANTINE,
    "public": PUBLIC,
}


def base_game_id(value: str) -> str:
    return str(value).strip().split("-", 1)[0].lower()


def select_fold(name: str) -> tuple[str, ...]:
    try:
        return FOLDS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown fold {name!r}; choose one of {sorted(FOLDS)}") from exc
