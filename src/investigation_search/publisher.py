from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .library_viewer import export_knowledge_library, list_knowledge_library_sessions, load_knowledge_library_session


def publish_knowledge_library_zip(
    library_dir: str | Path,
    output_zip: str | Path,
    *,
    export_format: str = "html",
    session_ids: Sequence[str] | None = None,
    include_raw: bool = True,
) -> Path:
    lib_dir = Path(library_dir)
    out_path = Path(output_zip)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "bundle"
        stage.mkdir(parents=True, exist_ok=True)
        export_dir = stage / "export"
        export_knowledge_library(
            lib_dir,
            export_dir,
            format=export_format,
            session_ids=session_ids,
            include_raw=include_raw,
        )
        # Also include VERSION-like metadata for debugging.
        (stage / "bundle_meta.json").write_text(
            json.dumps(
                {"export_format": export_format, "include_raw": include_raw, "session_count": _count_sessions(export_dir)},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        _zip_dir(stage, out_path)

    return out_path


def publish_knowledge_library_pdf(
    library_dir: str | Path,
    output_pdf: str | Path,
    *,
    session_ids: Sequence[str] | None = None,
    font_path: str | Path | None = None,
    max_sessions: int = 50,
) -> Path:
    """Publish a simple PDF dossier for sessions.

    Requires optional dependency: fpdf2
    """

    try:
        from fpdf import FPDF  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        from .bootstrap import auto_install_enabled, ensure_installed, requirements_path

        if auto_install_enabled():
            req = requirements_path("requirements-publish.txt")
            ensure_installed(
                requirements_files=[req] if req is not None else None,
                packages=("fpdf2",),
                auto_install=True,
            )
            from fpdf import FPDF  # type: ignore[import-not-found]
        else:
            raise RuntimeError("PDF publish requires `fpdf2`. Install: `pip install -r requirements-publish.txt`") from exc

    lib_dir = Path(library_dir)
    out_path = Path(output_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if session_ids is None:
        session_ids = [s.session_id for s in list_knowledge_library_sessions(lib_dir, limit=max_sessions)]

    selected = list(session_ids)[: max(0, int(max_sessions))]
    if not selected:
        raise ValueError("No sessions selected for PDF publish.")

    font_file = Path(font_path) if font_path is not None else _default_unicode_font()
    if font_file is None or not Path(font_file).exists():
        raise RuntimeError(
            "Unicode font not found for PDF. Pass `font_path=...`.\n"
            "Windows example: C:\\\\Windows\\\\Fonts\\\\malgun.ttf"
        )

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)

    # Register a Unicode font for KO/EN mixed text.
    pdf.add_font("UNIFONT", "", str(font_file), uni=True)
    pdf.add_font("UNIFONT", "B", str(font_file), uni=True)

    pdf.set_font("UNIFONT", size=12)

    for idx, sid in enumerate(selected, start=1):
        record = load_knowledge_library_session(lib_dir, sid)
        _render_session_pdf(pdf, record, title_prefix=f"[{idx}/{len(selected)}] ")

    pdf.output(str(out_path))
    return out_path


def _render_session_pdf(pdf: object, record: Mapping[str, Any], *, title_prefix: str = "") -> None:
    # pdf is fpdf.FPDF
    sid = str(record.get("session_id", "")).strip()
    created_at = str(record.get("created_at", "")).strip()
    mode = str(record.get("mode", "")).strip()
    query = str(record.get("clean_query") or record.get("raw_query") or "").strip()
    filters = record.get("filters") or {}
    result = record.get("result") or {}

    answer = str(result.get("answer", "")).strip()
    evidence = result.get("evidence") or []
    contradictions = result.get("contradictions") or []
    sources = result.get("sources") or []

    pdf.add_page()
    pdf.set_font("UNIFONT", "B", 16)
    pdf.multi_cell(0, 8, f"{title_prefix}Session {sid}")
    pdf.ln(1)

    pdf.set_font("UNIFONT", "", 11)
    pdf.multi_cell(0, 6, f"created_at: {created_at}")
    pdf.multi_cell(0, 6, f"mode: {mode}")
    pdf.multi_cell(0, 6, f"query: {query}")
    pdf.ln(2)

    pdf.set_font("UNIFONT", "B", 13)
    pdf.multi_cell(0, 7, "Answer")
    pdf.set_font("UNIFONT", "", 11)
    pdf.multi_cell(0, 6, answer or "(empty)")
    pdf.ln(2)

    pdf.set_font("UNIFONT", "B", 13)
    pdf.multi_cell(0, 7, "Evidence (top)")
    pdf.set_font("UNIFONT", "", 10)
    for i, item in enumerate(evidence[:12], start=1):
        unit = item.get("evidence") if isinstance(item, Mapping) else None
        content = str((unit or {}).get("content", "")).strip()
        stype = str((unit or {}).get("source_type", "")).strip()
        doc_id = str((unit or {}).get("doc_id", "")).strip()
        score = float(item.get("score", 0.0)) if isinstance(item, Mapping) else 0.0
        verdict = str(item.get("verdict", "")).strip() if isinstance(item, Mapping) else ""
        pdf.multi_cell(0, 5, f"{i}. score={score:.4f} verdict={verdict} type={stype} doc={doc_id}")
        if content:
            pdf.multi_cell(0, 5, "   " + content)
        pdf.ln(1)

    if contradictions:
        pdf.set_font("UNIFONT", "B", 13)
        pdf.multi_cell(0, 7, "Contradictions")
        pdf.set_font("UNIFONT", "", 10)
        for i, item in enumerate(contradictions[:10], start=1):
            unit = item.get("evidence") if isinstance(item, Mapping) else None
            content = str((unit or {}).get("content", "")).strip()
            stype = str((unit or {}).get("source_type", "")).strip()
            doc_id = str((unit or {}).get("doc_id", "")).strip()
            verdict = str(item.get("verdict", "")).strip() if isinstance(item, Mapping) else ""
            pdf.multi_cell(0, 5, f"{i}. verdict={verdict} type={stype} doc={doc_id}")
            if content:
                pdf.multi_cell(0, 5, "   " + content)
            pdf.ln(1)

    pdf.set_font("UNIFONT", "B", 13)
    pdf.multi_cell(0, 7, "Sources")
    pdf.set_font("UNIFONT", "", 9)
    for src in sources[:30]:
        if not isinstance(src, Mapping):
            continue
        citation_id = str(src.get("citation_id", "")).strip()
        doc_id = str(src.get("doc_id", "")).strip()
        section_path = str(src.get("section_path", "")).strip()
        stype = str(src.get("source_type", "")).strip()
        ts = str(src.get("timestamp", "")).strip()
        pdf.multi_cell(0, 5, f"- {citation_id} [{stype}] {doc_id} section={section_path} ts={ts}")

    pdf.ln(1)
    pdf.set_font("UNIFONT", "", 9)
    pdf.multi_cell(0, 5, "filters: " + _compact_json(filters, limit=500))


def _compact_json(payload: object, *, limit: int) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(payload)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _zip_dir(src_dir: Path, out_zip: Path) -> None:
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in src_dir.rglob("*"):
            if not path.is_file():
                continue
            zf.write(path, arcname=str(path.relative_to(src_dir)))


def _count_sessions(export_dir: Path) -> int:
    sessions_dir = export_dir / "sessions"
    if not sessions_dir.exists():
        return 0
    return len(list(sessions_dir.glob("*.*")))


def _default_unicode_font() -> Path | None:
    # Prefer Windows Korean fonts when available.
    candidates: list[str] = []
    if os.name == "nt":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        candidates.extend(
            [
                str(Path(windir) / "Fonts" / "malgun.ttf"),
                str(Path(windir) / "Fonts" / "malgunbd.ttf"),
                str(Path(windir) / "Fonts" / "arialuni.ttf"),
                str(Path(windir) / "Fonts" / "segoeui.ttf"),
            ]
        )
    # Generic fallbacks (Linux/macOS)
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    )
    for cand in candidates:
        p = Path(cand)
        if p.exists() and p.is_file():
            return p
    return None
