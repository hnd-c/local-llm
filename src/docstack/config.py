"""Load settings from configs/settings.toml (repo root) with env overrides."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

import tomllib


def repo_root() -> Path:
    """local-llm/ (contains pyproject.toml)."""
    here = Path(__file__).resolve()
    for p in [here.parent.parent.parent, here.parent.parent]:
        if (p / "pyproject.toml").exists():
            return p
    return here.parent.parent


def _load_toml() -> dict[str, Any]:
    path = repo_root() / "configs" / "settings.toml"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def default_libreoffice_path() -> str:
    if sys.platform == "win32":
        return r"C:\Program Files\LibreOffice\program\soffice.exe"
    if sys.platform == "darwin":
        return "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    return "soffice"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCSTACK_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    ollama_url: str = "http://127.0.0.1:11434"
    required_models: list[str] = Field(
        default_factory=lambda: ["qwen3:4b", "qwen3:8b"],
    )
    fast_model: str = "qwen3:4b"
    deep_model: str = "qwen3:8b"
    num_ctx: int = 4096
    max_ctx_chars: int = 12000
    retrieval_top_k: int = 8
    retrieval_min_score: float = 0.4
    deep_model_num_gpu: int = 0

    chunk_size: int = 1200
    chunk_overlap: int = 150
    min_chars_per_page: int = 50
    ocr_confidence_threshold: int = 70
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    data_dir: Path = Field(default=Path("data"))
    chroma_dir: Path = Field(default=Path("data/chroma"))
    uploads_dir: Path = Field(default=Path("data/uploads"))
    outputs_dir: Path = Field(default=Path("data/outputs"))
    libreoffice_path: str = ""

    @field_validator("required_models", mode="before")
    @classmethod
    def _ensure_required_models(cls, v: Any) -> list[str]:
        if v is None:
            return ["qwen3:4b", "qwen3:8b"]
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        return ["qwen3:4b", "qwen3:8b"]

    def model_post_init(self, __context: Any) -> None:
        root = repo_root()
        self.data_dir = (root / self.data_dir).resolve()
        self.chroma_dir = (root / self.chroma_dir).resolve()
        self.uploads_dir = (root / self.uploads_dir).resolve()
        self.outputs_dir = (root / self.outputs_dir).resolve()
        if not self.libreoffice_path.strip():
            self.libreoffice_path = default_libreoffice_path()


def _merge_toml_into_settings(s: Settings) -> Settings:
    raw = _load_toml()
    llm = raw.get("llm", {})
    ingest = raw.get("ingest", {})
    paths = raw.get("paths", {})
    updates: dict[str, Any] = {}
    for k, v in llm.items():
        if k in Settings.model_fields:
            updates[k] = v
    mapping_ingest = {
        "chunk_size": "chunk_size",
        "chunk_overlap": "chunk_overlap",
        "min_chars_per_page": "min_chars_per_page",
        "ocr_confidence_threshold": "ocr_confidence_threshold",
        "embedding_model": "embedding_model",
    }
    for tk, sk in mapping_ingest.items():
        if tk in ingest:
            updates[sk] = ingest[tk]
    for k, v in paths.items():
        if k == "data_dir":
            updates["data_dir"] = Path(v)
        elif k == "chroma_dir":
            updates["chroma_dir"] = Path(v)
        elif k == "uploads_dir":
            updates["uploads_dir"] = Path(v)
        elif k == "outputs_dir":
            updates["outputs_dir"] = Path(v)
        elif k == "libreoffice_path":
            updates["libreoffice_path"] = v
    if updates:
        return Settings(**{**s.model_dump(), **updates})
    return s


@lru_cache
def get_settings() -> Settings:
    base = Settings()
    merged = _merge_toml_into_settings(base)
    merged.data_dir.mkdir(parents=True, exist_ok=True)
    merged.chroma_dir.mkdir(parents=True, exist_ok=True)
    merged.uploads_dir.mkdir(parents=True, exist_ok=True)
    merged.outputs_dir.mkdir(parents=True, exist_ok=True)
    return merged
