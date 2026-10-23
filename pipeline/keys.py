"""Read API keys from the secrets dir without ever printing them.

Usage: from pipeline.keys import get; key = get("ELEVENLABS_API_KEY")
Accepts lines like KEY=value, export KEY=value, key: "value".
"""
import os
import re
from pathlib import Path

_FILES = ["api-keys.txt", "dashboard.env"]  # claude.env / fallback.env are never read here
_cache: dict[str, str] = {}


def _load() -> None:
    d = Path.home() / ".secrets"
    for name in _FILES:
        p = d / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*[\"']?(.*?)[\"']?\s*$", line)
            if m:
                _cache[m.group(1).upper()] = m.group(2)


def get(name: str) -> str:
    if name in os.environ:
        return os.environ[name]
    if not _cache:
        _load()
    if name.upper() not in _cache:
        raise KeyError(f"{name} not found in the secrets dir ({', '.join(_FILES)})")
    return _cache[name.upper()]


def has(name: str) -> bool:
    try:
        get(name)
        return True
    except KeyError:
        return False
