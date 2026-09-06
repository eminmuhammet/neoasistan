from __future__ import annotations

__version__ = "0.1.5"


def parse_version(text: str) -> tuple[int, ...]:
    """Turns "1.2.3" into (1, 2, 3) for comparison. Unparseable pieces count
    as 0 so a malformed manifest can never look newer than it is."""
    parts: list[int] = []
    for piece in text.strip().lstrip("vV").split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def is_newer(candidate: str, current: str = __version__) -> bool:
    return parse_version(candidate) > parse_version(current)
