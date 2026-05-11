"""OCRmyPDF wrapper — uses the Python API (no system PATH needed)."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def tesseract_langs() -> list[str]:
    """Return ['nep', 'eng'] if Nepali tessdata is available, else ['eng']."""
    try:
        r = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, text=True, timeout=10
        )
        available = {ln.strip() for ln in r.stdout.splitlines() + r.stderr.splitlines()}
        return ["nep", "eng"] if "nep" in available else ["eng"]
    except Exception:  # noqa: BLE001
        return ["eng"]


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

    langs = tesseract_langs()
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
