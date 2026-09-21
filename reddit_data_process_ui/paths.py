"""Repo-local folders for scraped posts and downloaded models."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
CHROMA_DIR = DATA_DIR / "chroma"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def model_dirname(repo_id: str) -> str:
    return repo_id.strip().replace("/", "--")


def model_path(repo_id: str) -> Path:
    ensure_dirs()
    return MODELS_DIR / model_dirname(repo_id)
