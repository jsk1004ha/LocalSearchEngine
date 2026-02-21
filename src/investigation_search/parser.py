from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

from .schema import EvidenceUnit, SourceType


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class ParseConfig:
    parser_version: str = "2.0"
    enable_ocr: bool = True
    ocr_lang: str = "kor+eng"
    text_confidence: float = 0.92
    table_confidence: float = 0.86
    ocr_confidence: float = 0.72
    max_table_cells_per_page: int = 600


class DocumentParser:
    def __init__(self, config: ParseConfig | None = None):
        self.config = config or ParseConfig()

    def parse_document(
        self,
        path: str | Path,
        *,
        doc_id: str | None = None,
        timestamp: str | None = None,
    ) -> List[EvidenceUnit]:
        file_path = Path(path)
        resolved_doc_id = doc_id or file_path.stem
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        ext = file_path.suffix.lower()

        if ext in {".txt", ".md", ".log", ".csv"}:
            return _parse_text_file(file_path, resolved_doc_id, ts, self.config)
        if ext == ".pdf":
            return _parse_pdf(file_path, resolved_doc_id, ts, self.config)
        if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            return _parse_image_ocr(file_path, resolved_doc_id, ts, self.config)

        raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")


def parse_documents(
    paths: Sequence[str | Path] | Iterable[str | Path],
    *,
    parser: DocumentParser | None = None,
    timestamp: str | None = None,
) -> List[EvidenceUnit]:
    parser = parser or DocumentParser()
    units: list[EvidenceUnit] = []
    for path in paths:
        units.extend(parser.parse_document(path, timestamp=timestamp))
    return units


def _parse_text_file(path: Path, doc_id: str, timestamp: str, config: ParseConfig) -> List[EvidenceUnit]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    units: list[EvidenceUnit] = []
    _append_sentences(
        units,
        doc_id=doc_id,
        section_path="body",
        text=text,
        timestamp=timestamp,
        confidence=config.text_confidence,
        source_type=SourceType.TEXT_SENTENCE,
        base_offset=0,
    )
    return units


def _parse_pdf(path: Path, doc_id: str, timestamp: str, config: ParseConfig) -> List[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    try:
        import pdfplumber  # type: ignore[import-not-found]
    except ImportError:
        pdfplumber = None

    if pdfplumber is None:
        _maybe_install_docparse_deps()
        try:
            import pdfplumber  # type: ignore[import-not-found]
        except ImportError:
            pdfplumber = None

    if pdfplumber is not None:
        with pdfplumber.open(path) as pdf:
            offset = 0
            for page_index, page in enumerate(pdf.pages, start=1):
                section_base = f"page/{page_index}"
                page_text = page.extract_text() or ""
                if page_text.strip():
                    _append_sentences(
                        units,
                        doc_id=doc_id,
                        section_path=f"{section_base}/text",
                        text=page_text,
                        timestamp=timestamp,
                        confidence=config.text_confidence,
                        source_type=SourceType.TEXT_SENTENCE,
                        base_offset=offset,
                    )
                    offset += len(page_text) + 1

                table_cells = 0
                for table_index, table in enumerate(page.extract_tables() or [], start=1):
                    for row_index, row in enumerate(table or [], start=1):
                        for col_index, cell in enumerate(row or [], start=1):
                            if not cell:
                                continue
                            content = str(cell).strip()
                            if not content:
                                continue
                            if table_cells >= config.max_table_cells_per_page:
                                break
                            start = offset
                            end = start + len(content)
                            units.append(
                                EvidenceUnit(
                                    doc_id=doc_id,
                                    source_type=SourceType.TABLE_CELL,
                                    content=content,
                                    section_path=f"{section_base}/table/{table_index}/r{row_index}c{col_index}",
                                    char_start=start,
                                    char_end=end,
                                    timestamp=timestamp,
                                    confidence=config.table_confidence,
                                    metadata={"parser": "pdfplumber"},
                                )
                            )
                            offset = end + 1
                            table_cells += 1
                        if table_cells >= config.max_table_cells_per_page:
                            break
                    if table_cells >= config.max_table_cells_per_page:
                        break

                if config.enable_ocr and not page_text.strip():
                    try:
                        image = page.to_image(resolution=200).original
                        ocr_text = _ocr_image_text(image=image, ocr_lang=config.ocr_lang)
                        if ocr_text.strip():
                            _append_sentences(
                                units,
                                doc_id=doc_id,
                                section_path=f"{section_base}/ocr",
                                text=ocr_text,
                                timestamp=timestamp,
                                confidence=config.ocr_confidence,
                                source_type=SourceType.OCR_TEXT,
                                base_offset=offset,
                            )
                            offset += len(ocr_text) + 1
                    except Exception:
                        # OCR dependency/runtime failures should not break parser.
                        continue
        return units

    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as exc:
        _maybe_install_docparse_deps()
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError as exc2:
            raise RuntimeError(
                "PDF 파싱을 위해 `pdfplumber` 또는 `pypdf`가 필요합니다. `pip install -r requirements-docparse.txt` 권장."
            ) from exc2

    reader = PdfReader(str(path))
    offset = 0
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue
        _append_sentences(
            units,
            doc_id=doc_id,
            section_path=f"page/{page_index}/text",
            text=page_text,
            timestamp=timestamp,
            confidence=config.text_confidence,
            source_type=SourceType.TEXT_SENTENCE,
            base_offset=offset,
        )
        offset += len(page_text) + 1
    return units


def _parse_image_ocr(path: Path, doc_id: str, timestamp: str, config: ParseConfig) -> List[EvidenceUnit]:
    if not config.enable_ocr:
        return []
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        _maybe_install_docparse_deps()
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc2:
            raise RuntimeError(
                "이미지 OCR을 위해 `pillow`가 필요합니다. `pip install -r requirements-docparse.txt` 후 재시도하세요."
            ) from exc2

    image = Image.open(path)
    text = _ocr_image_text(image=image, ocr_lang=config.ocr_lang)
    units: list[EvidenceUnit] = []
    if not text.strip():
        return units

    _append_sentences(
        units,
        doc_id=doc_id,
        section_path="image/ocr",
        text=text,
        timestamp=timestamp,
        confidence=config.ocr_confidence,
        source_type=SourceType.OCR_TEXT,
        base_offset=0,
    )
    return units


def _ocr_image_text(image: object, ocr_lang: str) -> str:
    try:
        import pytesseract  # type: ignore[import-not-found]
    except ImportError as exc:
        _maybe_install_docparse_deps()
        try:
            import pytesseract  # type: ignore[import-not-found]
        except ImportError as exc2:
            raise RuntimeError(
                "OCR을 위해 `pytesseract`가 필요합니다. `pip install -r requirements-docparse.txt` 후 재시도하세요."
            ) from exc2

    text = pytesseract.image_to_string(image, lang=ocr_lang)
    return text or ""


def _append_sentences(
    out: list[EvidenceUnit],
    *,
    doc_id: str,
    section_path: str,
    text: str,
    timestamp: str,
    confidence: float,
    source_type: SourceType,
    base_offset: int,
) -> None:
    cursor = base_offset
    for sentence in _split_sentences(text):
        content = sentence.strip()
        if not content:
            continue
        start = cursor
        end = start + len(content)
        out.append(
            EvidenceUnit(
                doc_id=doc_id,
                source_type=source_type,
                content=content,
                section_path=section_path,
                char_start=start,
                char_end=end,
                timestamp=timestamp,
                confidence=confidence,
                metadata={},
            )
        )
        cursor = end + 1


def _split_sentences(text: str) -> List[str]:
    chunks = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part and part.strip()]
    return chunks if chunks else [text.strip()]


def _maybe_install_docparse_deps() -> None:
    from .bootstrap import auto_install_enabled, ensure_installed, requirements_path

    if not auto_install_enabled():
        return
    req = requirements_path("requirements-docparse.txt")
    ensure_installed(
        requirements_files=[req] if req is not None else None,
        packages=("pdfplumber", "pypdf", "pillow", "pytesseract"),
        auto_install=True,
        quiet=False,
    )
