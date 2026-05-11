"""OCRmyPDF wrapper and quality heuristics."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def ocrmypdf_available() -> bool:
    return shutil.which("ocrmypdf") is not None


def run_ocrmypdf(input_pdf: Path, output_pdf: Path | None = None) -> Path:
    """Run OCR on a PDF; return path to searchable PDF."""
    if output_pdf is None:
        fd, tmp = tempfile.mkstemp(suffix=".pdf")
        import os

        os.close(fd)
        output_pdf = Path(tmp)
    if not ocrmypdf_available():
        raise RuntimeError("ocrmypdf not found on PATH. Install OCRmyPDF and Tesseract.")
    cmd = [
        "ocrmypdf",
        "--skip-text",
        "--optimize",
        "0",
        str(input_pdf),
        str(output_pdf),
    ]
    logger.info("Running OCR: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        logger.error("ocrmypdf stderr: %s", r.stderr)
        raise RuntimeError(f"ocrmypdf failed (code {r.returncode}): {r.stderr[:500]}")
    return output_pdf
