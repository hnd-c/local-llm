"""CLI for DocStack."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer
import uvicorn

from docstack.chunk.chunker import records_to_chunks
from docstack.config import get_settings
from docstack.index.embedder import clear_embedding_model_cache
from docstack.ingest.router import ingest_path
from docstack.index.store import index_chunks, wipe_collection

app = typer.Typer(no_args_is_help=True, add_completion=False)
models_app = typer.Typer(help="Ollama model images (not installed by pip).")
app.add_typer(models_app, name="models")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the FastAPI RAG middleware."""
    uvicorn.run(
        "docstack.api:app",
        host=host,
        port=port,
        reload=reload,
        factory=False,
    )


@app.command("wipe-index")
def wipe_index() -> None:
    """Delete the Chroma vector store (data/chroma). Edit [ingest] embedding_model first, then re-ingest."""
    settings = get_settings()
    chroma = settings.chroma_dir
    typer.echo(f"Removing {chroma} …")
    if chroma.exists():
        shutil.rmtree(chroma)
    chroma.mkdir(parents=True, exist_ok=True)
    clear_embedding_model_cache()
    get_settings.cache_clear()
    typer.echo("Vector DB wiped and embedding cache cleared. Restart the API, then run ingest again.")


@app.command("ingest-file")
def ingest_file(path: Path) -> None:
    """Ingest a single file into Chroma (replace mode)."""
    settings = get_settings()
    path = path.resolve()
    typer.echo(f"Ingesting {path} …")
    wipe_collection()
    records = ingest_path(path, settings.min_chars_per_page)
    chunks = records_to_chunks(records)
    n = index_chunks(chunks)
    typer.echo(f"Indexed {n} chunks.")


@models_app.command("pull")
def models_pull() -> None:
    """Pull every tag in configs/settings.toml [llm] required_models (ollama on PATH)."""
    settings = get_settings()
    ollama = shutil.which("ollama")
    if not ollama:
        typer.secho(
            "ollama executable not found on PATH. Install Ollama, then retry.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if not settings.required_models:
        typer.secho(
            "[llm] required_models is empty in configs/settings.toml — nothing to pull.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=0)
    for name in settings.required_models:
        typer.echo(f"ollama pull {name}")
        subprocess.run([ollama, "pull", name], check=True)
    typer.echo("Done.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
