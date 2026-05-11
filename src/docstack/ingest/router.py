"""Route files to PDF / DOCX / legacy DOC loaders."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from docstack.config import get_settings
from docstack.ingest.docx_loader import extract_docx_records
from docstack.ingest.ocr import ocrmypdf_available, run_ocrmypdf
from docstack.ingest.pdf_loader import extract_pdf_records, needs_ocr
from docstack.models import DocumentRecord

logger = logging.getLogger(__name__)


def _convert_doc_to_docx(doc_path: Path, out_dir: Path) -> Path:
    settings = get_settings()
    soffice = settings.libreoffice_path
    if not Path(soffice).exists():
        raise FileNotFoundError(
            f"LibreOffice not found at {soffice}. Install LibreOffice or set libreoffice_path in configs/settings.toml."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        soffice,
        "--headless",
        "--convert-to",
        "docx",
        "--outdir",
        str(out_dir),
        str(doc_path.resolve()),
    ]
    logger.info("Converting .doc: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"LibreOffice convert failed: {r.stderr[:800]}")
    expected = out_dir / (doc_path.stem + ".docx")
    if not expected.exists():
        raise RuntimeError(f"Expected converted file missing: {expected}")
    return expected


def ingest_path(path: Path, min_chars_per_page: int) -> list[DocumentRecord]:
    """Return normalised DocumentRecords for a supported file."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
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
            return extract_pdf_records(path, min_chars_per_page, ocr_pdf_path=work_pdf if work_pdf != path else None)
        finally:
            if ocr_tmp and ocr_tmp.exists():
                try:
                    ocr_tmp.unlink()
                except OSError:
                    pass

    if suffix == ".docx":
        return extract_docx_records(path)

    if suffix == ".doc":
        tmpdir = Path(tempfile.mkdtemp(prefix="docstack_doc_"))
        try:
            converted = _convert_doc_to_docx(path, tmpdir)
            return extract_docx_records(converted)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    raise ValueError(f"Unsupported file type: {suffix}")
