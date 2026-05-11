"""Image OCR loader: JPG, PNG, TIFF, BMP, WEBP → text via Tesseract."""

from __future__ import annotations

import logging
from pathlib import Path

from docstack.ingest.ocr import tesseract_langs
from docstack.models import DocumentRecord, make_doc_id

logger = logging.getLogger(__name__)


def extract_image_records(path: Path) -> list[DocumentRecord]:
    """Run Tesseract OCR on an image file and return a single DocumentRecord."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("pytesseract and Pillow are required for image OCR") from exc

    langs = tesseract_langs()
    logger.info("OCR image %s with langs=%s", path.name, "+".join(langs))

    img = Image.open(path)
    text = pytesseract.image_to_string(img, lang="+".join(langs)).strip()

    if not text:
        logger.warning("No text extracted from image %s", path.name)
        return []

    return [
        DocumentRecord(
            doc_id=make_doc_id(path),
            source_path=str(path.resolve()),
            filename=path.name,
            mime=path.suffix.lstrip(".").lower(),
            page=1,
            block_type="text",
            text=text,
        )
    ]
