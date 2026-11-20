"""Pytest configuration that skips collecting GUI tests unless Qt extras are installed."""

from __future__ import annotations

import importlib.util


def _has_module(name: str) -> bool:
    """Return True when top-level ``name`` can be located without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


collect_ignore_glob: list[str] = []

if not (_has_module("PySide6") and _has_module("pytestqt")):
    collect_ignore_glob.append("*")
