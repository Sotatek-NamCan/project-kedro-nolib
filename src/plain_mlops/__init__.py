"""Package initialisation with automatic .env loading."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DOTENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(_DOTENV_PATH, override=False)
