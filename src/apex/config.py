"""Load config.json and resolve project paths.

All paths and constants come from config.json; nothing else in the codebase
should hardcode them.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("APEX_CONFIG", PROJECT_ROOT / "config.json"))


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Return the parsed config.json (cached)."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def path(key: str) -> Path:
    """Return an absolute path for a key in the "paths" section."""
    return PROJECT_ROOT / load_config()["paths"][key]
