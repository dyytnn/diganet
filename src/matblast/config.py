from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
import yaml


def assert_known_keys(mapping: Mapping[str, Any], allowed: Iterable[str], where: str) -> None:
    """Raise if ``mapping`` contains keys outside ``allowed``.

    Config sections such as ``data`` and ``training`` are not forwarded as keyword
    arguments, so a misspelled key would otherwise be ignored silently. Every new
    nested option block must be validated with this helper.

    Args:
        mapping: The dictionary read from YAML.
        allowed: The complete set of accepted keys.
        where: Dotted config path used in the error message, e.g. ``"data.oversample"``.
    """
    allowed_set = set(allowed)
    unknown = sorted(set(mapping) - allowed_set)
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}; allowed keys are {sorted(allowed_set)}")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and resolve paths relative to the project root."""
    with Path(path).open() as f:
        cfg = yaml.safe_load(f) or {}
    for section in ("data", "model", "loss", "training"):
        cfg.setdefault(section, {})
    return cfg
