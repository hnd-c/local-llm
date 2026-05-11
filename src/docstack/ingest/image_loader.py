"""Image OCR loader: JPG, PNG, TIFF, BMP, WEBP → text via Tesseract."""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

from docstack.models import DocumentRecord

logger = logging.getLogger(__name__)

# Prefer Nepali + English when nep tessdata is available
def _ocr_langs() -> list[str]:
    try:
        r = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, timeout=10)
        available = {ln.strip() for ln in r.stdout.splitlines() + r.stderr.splitlines()}
        return ["nep", "eng"] if "nep" in available else ["eng"]
    except Exception:
        return ["eng"]


def _doc_id(path: Path) -> str:
    st = path.stat()
    return hashlib.sha256(f"{path.resolve()}:{st.st_mtime_ns}".encode()).hexdigest()[:16]


def extract_image_records(path: Path) -> list[DocumentRecord]:
    """Run Tesseract OCR on an image file and return a single DocumentRecord."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("pytesseract and Pillow are required for image OCR") from exc

    langs = _ocr_langs()
    logger.info("OCR image %s with langs=%s", path.name, "+".join(langs))

    img = Image.open(path)
    text = pytesseract.image_to_string(img, lang="+".join(langs)).strip()

    if not text:
        logger.warning("No text extracted from image %s", path.name)
        return []

    return [
        DocumentRecord(
            doc_id=_doc_id(path),
            source_path=str(path.resolve()),
            filename=path.name,
            mime=path.suffix.lstrip(".").lower(),
            page=1,
            block_type="text",
            text=text,
        )
    ]
