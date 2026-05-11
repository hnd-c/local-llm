"""Route files to the appropriate loader based on file extension."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from docstack.config import get_settings
from docstack.ingest.docx_loader import extract_docx_records
from docstack.ingest.image_loader import extract_image_records
from docstack.ingest.ocr import ocrmypdf_available, run_ocrmypdf
from docstack.ingest.office_loader import extract_pptx_records, extract_xlsx_records
from docstack.ingest.pdf_loader import extract_pdf_records, needs_ocr
from docstack.ingest.text_loader import (
    extract_csv_records,
    extract_html_records,
    extract_markdown_records,
    extract_txt_records,
)
from docstack.models import DocumentRecord

logger = logging.getLogger(__name__)

# Extensions handled natively (no LibreOffice needed)
_NATIVE: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "md",
    ".markdown": "md",
    ".csv": "csv",
    ".html": "html",
    ".htm": "html",
    ".xlsx": "xlsx",
    ".pptx": "pptx",
    # images
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".tiff": "image",
    ".tif": "image",
    ".bmp": "image",
    ".webp": "image",
}

# Extensions that LibreOffice can convert to PDF for extraction
_LIBREOFFICE_TO_PDF = {
    ".doc",
    ".xls",
    ".ppt",
    ".odt",
    ".ods",
    ".odp",
    ".rtf",
}

SUPPORTED_EXTENSIONS = sorted(_NATIVE.keys() | _LIBREOFFICE_TO_PDF)


def _soffice() -> str:
    return get_settings().libreoffice_path


def _convert_via_libreoffice(path: Path, out_dir: Path, target: str = "pdf") -> Path:
    soffice = _soffice()
    if not Path(soffice).exists():
        raise FileNotFoundError(
            f"LibreOffice not found at {soffice}. "
            "Install LibreOffice or set libreoffice_path in configs/settings.toml."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [soffice, "--headless", "--convert-to", target, "--outdir", str(out_dir), str(path.resolve())]
    logger.info("LibreOffice convert: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"LibreOffice convert failed: {r.stderr[:800]}")
    expected = out_dir / (path.stem + f".{target}")
    if not expected.exists():
        raise RuntimeError(f"Expected converted file missing: {expected}")
    return expected


def ingest_path(path: Path, min_chars_per_page: int) -> list[DocumentRecord]:
    """Return normalised DocumentRecords for any supported file."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    kind = _NATIVE.get(suffix)

    # ── Native formats ────────────────────────────────────────────────
    if kind == "pdf":
        ocr_tmp: Path | None = None
        try:
            work_pdf = path
            if needs_ocr(path, min_chars_per_page):
                if ocrmypdf_available():
                    ocr_tmp = Path(tempfile.mkstemp(suffix=".pdf")[1])
                    run_ocrmypdf(path, ocr_tmp)
                    work_pdf = ocr_tmp
                else:
                    logger.warning("ocrmypdf not available; indexing low-text PDF as-is.")
            return extract_pdf_records(
                path, min_chars_per_page, ocr_pdf_path=work_pdf if work_pdf != path else None
            )
        finally:
            if ocr_tmp and ocr_tmp.exists():
                try:
                    ocr_tmp.unlink()
                except OSError:
                    pass

    if kind == "docx":
        return extract_docx_records(path)

    if kind == "txt":
        return extract_txt_records(path)

    if kind == "md":
        return extract_markdown_records(path)

    if kind == "csv":
        return extract_csv_records(path)

    if kind == "html":
        return extract_html_records(path)

    if kind == "xlsx":
        return extract_xlsx_records(path)

    if kind == "pptx":
        return extract_pptx_records(path)

    if kind == "image":
        return extract_image_records(path)

    # ── LibreOffice-converted formats (→ PDF then extract) ────────────
    if suffix in _LIBREOFFICE_TO_PDF:
        tmpdir = Path(tempfile.mkdtemp(prefix="docstack_lo_"))
        try:
            # .doc → docx is cleaner for Word files
            if suffix == ".doc":
                converted = _convert_via_libreoffice(path, tmpdir, target="docx")
                return extract_docx_records(converted)
            # everything else → PDF
            pdf_path = _convert_via_libreoffice(path, tmpdir, target="pdf")
            return ingest_path(pdf_path, min_chars_per_page)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    raise ValueError(
        f"Unsupported file type: {suffix!r}. "
        f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )
