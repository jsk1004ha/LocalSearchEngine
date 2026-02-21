from __future__ import annotations

import html
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_SAFE_SLUG_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class LibrarySessionSummary:
    session_id: str
    created_at: str
    mode: str
    raw_query: str
    clean_query: str


def list_knowledge_library_sessions(
    library_dir: str | Path,
    *,
    limit: int | None = None,
    mode: str | None = None,
    query_contains: str | None = None,
) -> list[LibrarySessionSummary]:
    lib_dir = Path(library_dir)
    summaries: list[LibrarySessionSummary] = []

    index_path = lib_dir / "sessions.jsonl"
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_mode = str(payload.get("mode", "")).strip()
            clean_query = str(payload.get("clean_query", "")).strip()
            raw_query = str(payload.get("raw_query", "")).strip()

            if mode and row_mode != mode:
                continue
            if query_contains and query_contains not in clean_query and query_contains not in raw_query:
                continue

            session_id = str(payload.get("session_id", "")).strip()
            created_at = str(payload.get("created_at", "")).strip()
            if not session_id:
                continue
            summaries.append(
                LibrarySessionSummary(
                    session_id=session_id,
                    created_at=created_at,
                    mode=row_mode,
                    raw_query=raw_query,
                    clean_query=clean_query,
                )
            )
    else:
        # Fallback: scan session json files (no fast index).
        sessions_dir = lib_dir / "sessions"
        for path in sorted(sessions_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            session_id = str(payload.get("session_id", "")).strip() or path.stem
            row_mode = str(payload.get("mode", "")).strip()
            clean_query = str(payload.get("clean_query", "")).strip()
            raw_query = str(payload.get("raw_query", "")).strip()
            created_at = str(payload.get("created_at", "")).strip()
            if mode and row_mode != mode:
                continue
            if query_contains and query_contains not in clean_query and query_contains not in raw_query:
                continue
            summaries.append(
                LibrarySessionSummary(
                    session_id=session_id,
                    created_at=created_at,
                    mode=row_mode,
                    raw_query=raw_query,
                    clean_query=clean_query,
                )
            )

    summaries.sort(key=lambda s: s.created_at, reverse=True)
    if limit is None:
        return summaries
    return summaries[: max(0, int(limit))]


def load_knowledge_library_session(library_dir: str | Path, session_id: str) -> dict[str, Any]:
    lib_dir = Path(library_dir)
    path = lib_dir / "sessions" / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def export_knowledge_library(
    library_dir: str | Path,
    output_dir: str | Path,
    *,
    format: str = "html",
    session_ids: Sequence[str] | None = None,
    include_raw: bool = True,
) -> dict[str, object]:
    lib_dir = Path(library_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = str(format).strip().lower()
    if fmt not in {"html", "md", "markdown"}:
        raise ValueError("format must be one of: html, md")

    selected_ids = list(session_ids) if session_ids else _all_session_ids(lib_dir)
    exported = 0

    if fmt == "html":
        (out_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (out_dir / "assets").mkdir(parents=True, exist_ok=True)
        (out_dir / "assets" / "style.css").write_text(_DEFAULT_CSS, encoding="utf-8")
        index_rows: list[dict[str, str]] = []
        for sid in selected_ids:
            try:
                record = load_knowledge_library_session(lib_dir, sid)
            except Exception:
                continue
            index_rows.append(
                {
                    "session_id": str(record.get("session_id", sid)),
                    "created_at": str(record.get("created_at", "")),
                    "mode": str(record.get("mode", "")),
                    "query": str(record.get("clean_query") or record.get("raw_query") or ""),
                    "answer": str(((record.get("result") or {}) or {}).get("answer", "")),
                }
            )
            (out_dir / "sessions" / f"{_safe_slug(sid)}.html").write_text(
                _render_session_html(record),
                encoding="utf-8",
            )
            _export_osint_derivatives(record, out_dir=out_dir)
            exported += 1

        (out_dir / "index.html").write_text(_render_index_html(index_rows), encoding="utf-8")
    else:
        (out_dir / "sessions").mkdir(parents=True, exist_ok=True)
        index_lines = ["# Knowledge Library", ""]
        for sid in selected_ids:
            try:
                record = load_knowledge_library_session(lib_dir, sid)
            except Exception:
                continue
            created_at = str(record.get("created_at", ""))
            mode = str(record.get("mode", ""))
            query = str(record.get("clean_query") or record.get("raw_query") or "")
            index_lines.append(f"- `{sid}` {created_at} [{mode}] {query}")
            index_lines.append(f"  - sessions/{_safe_slug(sid)}.md")
            (out_dir / "sessions" / f"{_safe_slug(sid)}.md").write_text(_render_session_markdown(record), encoding="utf-8")
            _export_osint_derivatives(record, out_dir=out_dir)
            exported += 1
        (out_dir / "index.md").write_text("\n".join(index_lines).rstrip() + "\n", encoding="utf-8")

    if include_raw:
        _copy_raw_library(lib_dir, out_dir / "raw", session_ids=selected_ids)

    return {"format": fmt, "exported_sessions": exported, "output_dir": str(out_dir)}


def _safe_slug(text: str) -> str:
    cleaned = _SAFE_SLUG_RE.sub("_", text.strip())
    return cleaned[:120] or "session"


def _all_session_ids(lib_dir: Path) -> list[str]:
    sessions_dir = lib_dir / "sessions"
    if not sessions_dir.exists():
        return []
    return [p.stem for p in sorted(sessions_dir.glob("*.json"))]


def _copy_raw_library(lib_dir: Path, out_raw_dir: Path, *, session_ids: Sequence[str]) -> None:
    out_raw_dir.mkdir(parents=True, exist_ok=True)
    # Always copy the session index if present.
    index_path = lib_dir / "sessions.jsonl"
    if index_path.exists():
        shutil.copy2(index_path, out_raw_dir / "sessions.jsonl")

    (out_raw_dir / "sessions").mkdir(parents=True, exist_ok=True)
    for sid in session_ids:
        src = lib_dir / "sessions" / f"{sid}.json"
        if src.exists():
            shutil.copy2(src, out_raw_dir / "sessions" / f"{sid}.json")

    evidence_dir = lib_dir / "evidence"
    if evidence_dir.exists():
        shutil.copytree(evidence_dir, out_raw_dir / "evidence", dirs_exist_ok=True)

    osint_dir = lib_dir / "osint"
    if osint_dir.exists():
        shutil.copytree(osint_dir, out_raw_dir / "osint", dirs_exist_ok=True)


def _render_index_html(rows: Sequence[Mapping[str, str]]) -> str:
    body_rows: list[str] = []
    for row in rows:
        sid = _safe_slug(str(row.get("session_id", "")))
        created_at = html.escape(str(row.get("created_at", "")))
        mode = html.escape(str(row.get("mode", "")))
        query = html.escape(str(row.get("query", "")))
        answer = html.escape(_trim(str(row.get("answer", "")), 180))
        body_rows.append(
            "<tr>"
            f"<td><a href='sessions/{sid}.html'><code>{sid}</code></a></td>"
            f"<td>{created_at}</td>"
            f"<td><span class='pill'>{mode}</span></td>"
            f"<td>{query}</td>"
            f"<td class='muted'>{answer}</td>"
            "</tr>"
        )

    return (
        "<!doctype html>"
        "<html lang='en'>"
        "<head>"
        "<meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        "<title>Knowledge Library</title>"
        "<link rel='stylesheet' href='assets/style.css'/>"
        "</head>"
        "<body>"
        "<main class='container'>"
        "<h1>Knowledge Library</h1>"
        "<p class='muted'>Exported view of investigation search sessions.</p>"
        "<div class='toolbar'>"
        "<input id='filter' class='filter' placeholder='Filter (mode / query / answer / session id)'/>"
        "</div>"
        "<table>"
        "<thead><tr><th>Session</th><th>Created</th><th>Mode</th><th>Query</th><th>Answer</th></tr></thead>"
        "<tbody>"
        + "".join(body_rows)
        + "</tbody>"
        "</table>"
        "<script>"
        "(function(){"
        "const input=document.getElementById('filter');"
        "const rows=[...document.querySelectorAll('tbody tr')];"
        "function norm(s){return (s||'').toLowerCase();}"
        "input.addEventListener('input',()=>{"
        "  const q=norm(input.value).trim();"
        "  for(const r of rows){"
        "    const t=norm(r.innerText);"
        "    r.style.display = (!q || t.includes(q)) ? '' : 'none';"
        "  }"
        "});"
        "})();"
        "</script>"
        "</main>"
        "</body>"
        "</html>"
    )


def _render_session_html(record: Mapping[str, Any]) -> str:
    sid = html.escape(str(record.get("session_id", "")))
    created_at = html.escape(str(record.get("created_at", "")))
    mode = html.escape(str(record.get("mode", "")))
    raw_query = html.escape(str(record.get("raw_query", "")))
    clean_query = html.escape(str(record.get("clean_query", "")))
    filters = record.get("filters") or {}
    filters_json = html.escape(json.dumps(filters, ensure_ascii=False, indent=2, sort_keys=True))

    result = record.get("result") or {}
    answer = html.escape(str(result.get("answer", "")))

    answer_sources = _render_sources_list(result.get("answer_sources") or [])
    sources = _render_sources_list(result.get("sources") or [])
    evidence = _render_evidence_table(result.get("evidence") or [])
    contradictions = _render_evidence_table(result.get("contradictions") or [], title="Contradictions")

    osint_block = _render_osint(record)

    return (
        "<!doctype html>"
        "<html lang='en'>"
        "<head>"
        "<meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        f"<title>Session {sid}</title>"
        "<link rel='stylesheet' href='../assets/style.css'/>"
        "</head>"
        "<body>"
        "<main class='container'>"
        "<p><a href='../index.html'>&larr; Back to index</a></p>"
        f"<h1>Session <code>{sid}</code></h1>"
        "<div class='meta'>"
        f"<div><span class='k'>created_at</span> {created_at}</div>"
        f"<div><span class='k'>mode</span> <span class='pill'>{mode}</span></div>"
        "</div>"
        "<h2>Query</h2>"
        f"<p><span class='k'>raw</span> <code>{raw_query}</code></p>"
        f"<p><span class='k'>clean</span> <code>{clean_query}</code></p>"
        "<details><summary>filters</summary>"
        f"<pre>{filters_json}</pre>"
        "</details>"
        "<h2>Answer</h2>"
        f"<div class='answer'>{answer}</div>"
        "<h3>Answer Sources</h3>"
        f"{answer_sources}"
        "<h3>All Sources</h3>"
        f"{sources}"
        "<h2>Evidence</h2>"
        f"{evidence}"
        f"{contradictions}"
        f"{osint_block}"
        "</main>"
        "</body>"
        "</html>"
    )


def _render_sources_list(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "<p class='muted'>none</p>"
    rows: list[str] = ["<ul>"]
    for src in items:
        citation_id = html.escape(str(src.get("citation_id", "")))
        doc_id = html.escape(str(src.get("doc_id", "")))
        section_path = html.escape(str(src.get("section_path", "")))
        stype = html.escape(str(src.get("source_type", "")))
        ts = html.escape(str(src.get("timestamp", "")))
        maybe_link = _as_link(doc_id)
        doc_view = maybe_link or f"<code>{doc_id}</code>"
        rows.append(
            "<li>"
            f"<code>{citation_id}</code> "
            f"<span class='pill'>{stype}</span> "
            f"{doc_view} "
            f"<span class='muted'>section={section_path} ts={ts}</span>"
            "</li>"
        )
    rows.append("</ul>")
    return "".join(rows)


def _render_evidence_table(items: Sequence[Mapping[str, Any]], *, title: str = "Evidence") -> str:
    if not items:
        return f"<p class='muted'>{html.escape(title)}: none</p>"
    rows: list[str] = []
    for idx, item in enumerate(items, start=1):
        score = html.escape(f"{float(item.get('score', 0.0)):.4f}")
        verdict = html.escape(str(item.get("verdict", "")))
        why = html.escape(_trim(str(item.get("why_it_matches", "")), 200))
        unit = item.get("evidence") or {}
        content = html.escape(str(unit.get("content", "")))
        stype = html.escape(str(unit.get("source_type", "")))
        doc_id = str(unit.get("doc_id", ""))
        section_path = html.escape(str(unit.get("section_path", "")))
        ts = html.escape(str(unit.get("timestamp", "")))

        link = _as_link(str((unit.get("metadata") or {}).get("url") or doc_id)) or f"<code>{html.escape(doc_id)}</code>"
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td class='num'>{score}</td>"
            f"<td><span class='pill'>{verdict}</span></td>"
            f"<td><span class='pill'>{stype}</span></td>"
            f"<td>{link}<div class='muted'>section={section_path}<br/>ts={ts}</div></td>"
            f"<td><div class='snippet'>{content}</div><div class='muted'>why={why}</div></td>"
            "</tr>"
        )

    return (
        f"<h3>{html.escape(title)}</h3>"
        "<table>"
        "<thead><tr><th>#</th><th>score</th><th>verdict</th><th>type</th><th>source</th><th>content</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody>"
        "</table>"
    )


def _render_osint(record: Mapping[str, Any]) -> str:
    result = record.get("result") or {}
    diagnostics = result.get("diagnostics") or {}
    osint_data = diagnostics.get("osint")
    if not isinstance(osint_data, Mapping):
        return ""

    timeline = osint_data.get("timeline") if isinstance(osint_data.get("timeline"), list) else []
    graph = osint_data.get("graph") if isinstance(osint_data.get("graph"), Mapping) else {}

    blocks: list[str] = ["<h2>OSINT</h2>"]
    blocks.append("<details open><summary>graph</summary>")
    node_count = html.escape(str(graph.get("node_count", 0)))
    edge_count = html.escape(str(graph.get("edge_count", 0)))
    blocks.append(f"<p class='muted'>nodes={node_count}, edges={edge_count}</p>")
    blocks.append("<pre class='code'>")
    blocks.append(html.escape(json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True))[:12000])
    blocks.append("</pre>")
    blocks.append("</details>")

    blocks.append("<details open><summary>timeline</summary>")
    if timeline:
        blocks.append("<ul>")
        for ev in timeline[:50]:
            date = html.escape(str(ev.get("date", "")))
            title = html.escape(str(ev.get("title", "")))
            url = str(ev.get("url", ""))
            link = _as_link(url) or html.escape(url)
            blocks.append(f"<li><code>{date}</code> {title} ({link})</li>")
        blocks.append("</ul>")
    else:
        blocks.append("<p class='muted'>none</p>")
    blocks.append("</details>")

    sid = str(record.get("session_id", "")).strip()
    if sid:
        dot_path = f"../osint/{_safe_slug(sid)}.dot"
        blocks.append("<p class='muted'>Artifacts:</p>")
        blocks.append(f"<ul><li><a href='{dot_path}'>graphviz .dot</a></li></ul>")

    return "".join(blocks)


def _export_osint_derivatives(record: Mapping[str, Any], *, out_dir: Path) -> None:
    sid = str(record.get("session_id", "")).strip()
    if not sid:
        return
    result = record.get("result") or {}
    diagnostics = result.get("diagnostics") or {}
    osint_data = diagnostics.get("osint")
    if not isinstance(osint_data, Mapping):
        return
    graph = osint_data.get("graph")
    if not isinstance(graph, Mapping):
        return

    # Export graphviz dot for external visualization tooling.
    osint_out = out_dir / "osint"
    osint_out.mkdir(parents=True, exist_ok=True)
    (osint_out / f"{_safe_slug(sid)}.dot").write_text(_graphviz_dot(graph), encoding="utf-8")


def _graphviz_dot(graph: Mapping[str, Any]) -> str:
    def _q(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"')

    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
    out: list[str] = ["digraph osint {", "  rankdir=LR;"]
    for node in nodes[:400]:
        if not isinstance(node, Mapping):
            continue
        node_id = str(node.get("node_id", ""))
        label = str(node.get("label", node_id))
        node_type = str(node.get("node_type", "node"))
        shape = "box" if node_type == "query" else ("ellipse" if node_type == "domain" else "note")
        out.append(f'  "{_q(node_id)}" [label="{_q(label)}", shape={shape}];')
    for edge in edges[:800]:
        if not isinstance(edge, Mapping):
            continue
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        label = str(edge.get("edge_type", "rel"))
        out.append(f'  "{_q(src)}" -> "{_q(tgt)}" [label="{_q(label)}"];')
    out.append("}")
    return "\n".join(out) + "\n"


def _render_session_markdown(record: Mapping[str, Any]) -> str:
    sid = str(record.get("session_id", "")).strip()
    created_at = str(record.get("created_at", "")).strip()
    mode = str(record.get("mode", "")).strip()
    raw_query = str(record.get("raw_query", "")).strip()
    clean_query = str(record.get("clean_query", "")).strip()
    filters = record.get("filters") or {}
    result = record.get("result") or {}

    lines: list[str] = []
    lines.append(f"# Session `{sid}`")
    lines.append("")
    lines.append(f"- created_at: `{created_at}`")
    lines.append(f"- mode: `{mode}`")
    lines.append("")
    lines.append("## Query")
    lines.append("")
    lines.append(f"- raw: `{raw_query}`")
    lines.append(f"- clean: `{clean_query}`")
    lines.append("")
    lines.append("## Filters")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(filters, ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Answer")
    lines.append("")
    lines.append(str(result.get("answer", "")))
    lines.append("")

    lines.append("## Answer Sources")
    lines.append("")
    lines.extend(_sources_markdown(result.get("answer_sources") or []))
    lines.append("")

    lines.append("## Evidence")
    lines.append("")
    lines.extend(_evidence_markdown(result.get("evidence") or []))
    lines.append("")

    lines.append("## Contradictions")
    lines.append("")
    lines.extend(_evidence_markdown(result.get("contradictions") or []))
    lines.append("")

    diagnostics = result.get("diagnostics") or {}
    if "osint" in diagnostics:
        lines.append("## OSINT")
        lines.append("")
        osint_json = json.dumps(diagnostics.get("osint"), ensure_ascii=False, indent=2, sort_keys=True)
        lines.append("```json")
        lines.append(osint_json)
        lines.append("```")
        lines.append("")
        lines.append(f"- graphviz_dot: `../osint/{_safe_slug(sid)}.dot`")
        lines.append("")

    lines.append("## Diagnostics (raw)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _sources_markdown(items: Sequence[Mapping[str, Any]]) -> list[str]:
    if not items:
        return ["- none"]
    lines: list[str] = []
    for src in items:
        citation_id = str(src.get("citation_id", "")).strip()
        doc_id = str(src.get("doc_id", "")).strip()
        section_path = str(src.get("section_path", "")).strip()
        stype = str(src.get("source_type", "")).strip()
        ts = str(src.get("timestamp", "")).strip()
        lines.append(f"- `{citation_id}` [{stype}] doc=`{doc_id}` section=`{section_path}` ts=`{ts}`")
    return lines


def _evidence_markdown(items: Sequence[Mapping[str, Any]]) -> list[str]:
    if not items:
        return ["- none"]
    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        score = float(item.get("score", 0.0))
        verdict = str(item.get("verdict", "")).strip()
        why = str(item.get("why_it_matches", "")).strip()
        unit = item.get("evidence") or {}
        content = str(unit.get("content", "")).strip()
        stype = str(unit.get("source_type", "")).strip()
        doc_id = str(unit.get("doc_id", "")).strip()
        section_path = str(unit.get("section_path", "")).strip()
        ts = str(unit.get("timestamp", "")).strip()
        url = str((unit.get("metadata") or {}).get("url") or "")
        lines.append(f"{idx}. score={score:.4f} verdict={verdict} type={stype}")
        lines.append(f"   - source: doc=`{doc_id}` section=`{section_path}` ts=`{ts}`")
        if url:
            lines.append(f"   - url: {url}")
        if why:
            lines.append(f"   - why: {why}")
        lines.append(f"   - content: {content}")
    return lines


def _trim(text: str, max_len: int) -> str:
    if max_len <= 0 or len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _as_link(value: str) -> str | None:
    url = value.strip()
    if not url:
        return None
    if url.startswith("http://") or url.startswith("https://"):
        esc = html.escape(url)
        return f"<a href='{esc}' target='_blank' rel='noreferrer'>{esc}</a>"
    return None


_DEFAULT_CSS = """
:root {
  --fg: #121212;
  --muted: #6b7280;
  --bg: #ffffff;
  --border: #e5e7eb;
  --pill: #f3f4f6;
  --code: #0b1020;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
pre { background: #fafafa; border: 1px solid var(--border); padding: 12px; overflow: auto; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; }
th, td { border: 1px solid var(--border); padding: 10px; vertical-align: top; }
th { background: #fafafa; text-align: left; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
.muted { color: var(--muted); }
.pill { background: var(--pill); border: 1px solid var(--border); border-radius: 999px; padding: 2px 10px; font-size: 12px; }
.meta { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 12px 0; }
.k { color: var(--muted); margin-right: 6px; }
.answer { border: 1px solid var(--border); border-radius: 10px; padding: 14px; background: #fcfcfc; }
.snippet { white-space: pre-wrap; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
details > summary { cursor: pointer; }
.toolbar { margin: 12px 0; display: flex; gap: 12px; align-items: center; }
.filter { width: 100%; max-width: 520px; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; font: inherit; }
""".strip() + "\n"
