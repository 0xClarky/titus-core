"""Environment helpers for Titus CLIs."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_env(env_file: str | Path = ".env") -> None:
    """Load .env file if present (idempotent)."""

    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
