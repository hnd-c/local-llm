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


@app.command("wipe-chats")
def wipe_chats(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Clear Open WebUI chat history (webui.db) and uploaded files cache.

    Useful when the UI feels slow or you want a clean slate.
    Does NOT touch the vector index — run wipe-index separately for that.
    """
    import site

    targets: list[Path] = []

    # Open WebUI stores its DB inside the installed package data dir
    for sp in site.getsitepackages():
        candidate = Path(sp) / "open_webui" / "data" / "webui.db"
        if candidate.exists():
            targets.append(candidate)
    # Also check local data/ dir
    local_db = Path("data/webui.db")
    if local_db.exists() and local_db.stat().st_size > 0:
        targets.append(local_db.resolve())
    # Open WebUI's own RAG vector DB (should be disabled, but clean it anyway)
    for sp in site.getsitepackages():
        owui_chroma = Path(sp) / "open_webui" / "data" / "vector_db"
        if owui_chroma.exists():
            targets.append(owui_chroma)
    # Uploaded files cache
    uploads = Path("data/uploads")
    if uploads.exists() and any(uploads.iterdir()):
        targets.append(uploads.resolve())

    if not targets:
        typer.echo("Nothing to clear.")
        return

    typer.echo("The following will be cleared:")
    for t in targets:
        typer.echo(f"  {t}")

    if not yes:
        typer.confirm("\nProceed?", abort=True)

    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
            t.mkdir(parents=True, exist_ok=True)
            typer.echo(f"Cleared directory: {t}")
        else:
            t.unlink()
            typer.echo(f"Deleted: {t}")

    typer.echo("\nDone. Restart ./start.sh to reinitialise Open WebUI with a clean database.")


@app.command("wipe-all")
def wipe_all(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Wipe everything: vector index + chat history + uploads + logs.

    Full clean slate. You will need to re-ingest all documents after this.
    """
    if not yes:
        typer.confirm(
            "This will delete ALL data (vector index, chat history, uploads, logs). Continue?",
            abort=True,
        )
    wipe_index()
    wipe_chats(yes=True)
    for log in [Path("data/docstack.log"), Path("data/openwebui.log")]:
        if log.exists():
            log.write_text("")
            typer.echo(f"Cleared log: {log}")
    typer.echo("\nFull wipe complete.")


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
