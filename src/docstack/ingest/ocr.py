"""OCRmyPDF wrapper — uses the Python API (no system PATH needed)."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def ocrmypdf_available() -> bool:
    """True if the ocrmypdf Python package is importable."""
    try:
        import ocrmypdf  # noqa: F401

        return True
    except ImportError:
        return False


def run_ocrmypdf(input_pdf: Path, output_pdf: Path | None = None) -> Path:
    """Run OCR on a PDF via the ocrmypdf Python API; return path to searchable PDF."""
    if output_pdf is None:
        fd, tmp = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        output_pdf = Path(tmp)

    try:
        import ocrmypdf
    except ImportError as exc:
        raise RuntimeError(
            "ocrmypdf package not installed. Run: pip install ocrmypdf"
        ) from exc

    # Use Nepali + English for Devanagari script; fall back to eng-only if nep pack missing.
    import subprocess

    available_langs = set()
    try:
        r = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, text=True, timeout=10
        )
        available_langs = {ln.strip() for ln in r.stdout.splitlines() + r.stderr.splitlines()}
    except Exception:  # noqa: BLE001
        pass

    langs = ["nep", "eng"] if "nep" in available_langs else ["eng"]
    logger.info("Running OCR (Python API, langs=%s): %s → %s", "+".join(langs), input_pdf, output_pdf)
    result = ocrmypdf.ocr(
        input_pdf,
        output_pdf,
        language=langs,
        skip_text=True,
        optimize=0,
        progress_bar=False,
    )
    if result != ocrmypdf.ExitCode.ok and result != ocrmypdf.ExitCode.already_done_ocr:
        raise RuntimeError(f"ocrmypdf returned exit code: {result}")

    return output_pdf
