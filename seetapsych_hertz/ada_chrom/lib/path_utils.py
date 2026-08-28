"""Path resolution helpers for delivery entrypoint run defaults."""

from __future__ import annotations

from pathlib import Path


def _is_blank(value: Path | str | None) -> bool:
    return value is None or str(value).strip() == ""


def resolve_required_path(value: Path | str | None, default_value: str, label: str) -> Path:
    selected = value if value is not None else default_value
    if _is_blank(selected):
        raise ValueError(f"{label} is required. Set the script-level string or pass it explicitly.")
    return Path(selected)


def resolve_optional_path(value: Path | str | None, default_value: str) -> Path | None:
    selected = value if value is not None else default_value
    if _is_blank(selected):
        return None
    return Path(selected)


def optional_path_argument(value: Path | str | None) -> Path | None:
    if _is_blank(value):
        return None
    return Path(value)


def required_path_argument(value: Path | str | None, label: str) -> Path:
    if _is_blank(value):
        raise ValueError(f"{label} is required. Pass it explicitly.")
    return Path(value)
