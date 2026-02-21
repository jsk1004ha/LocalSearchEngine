from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .bootstrap import AUTO_INSTALL_ENV, auto_install_enabled, ensure_installed, repo_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="investigation_search", description="Local Investigation Search Engine tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tui = sub.add_parser("tui", help="Run terminal UI (Textual)")
    p_tui.add_argument("--build-dir", type=str, default=None, help="Offline build directory (manifest.json 포함)")
    p_tui.add_argument("--docs", type=str, nargs="*", default=None, help="문서 파일들(.txt/.pdf/이미지 등)")
    p_tui.add_argument("--mode", type=str, default="investigation", help="default mode (investigation/report/fbi/...)")
    p_tui.add_argument("--top-k", type=int, default=5, help="top_k_per_pass")
    p_tui.add_argument("--time-budget", type=int, default=120, help="time budget sec")
    p_tui.add_argument("--knowledge-library-dir", type=str, default=str(Path("artifacts") / "knowledge_library"))
    p_tui.add_argument("--enable-knowledge-library", action="store_true", help="persist searches to knowledge library")
    p_tui.add_argument("--no-web", action="store_true", help="disable web fallback")
    p_tui.add_argument("--no-web-sandbox", action="store_true", help="disable subprocess isolation for web search")
    p_tui.add_argument(
        "--auto-install",
        action="store_true",
        help=f"auto install missing optional deps (also via env {AUTO_INSTALL_ENV}=1)",
    )

    p_lib = sub.add_parser("library", help="Knowledge library utilities")
    lib_sub = p_lib.add_subparsers(dest="library_cmd", required=True)

    p_list = lib_sub.add_parser("list", help="List sessions")
    p_list.add_argument("--library-dir", type=str, default=str(Path("artifacts") / "knowledge_library"))
    p_list.add_argument("--limit", type=int, default=30)
    p_list.add_argument("--mode", type=str, default=None)
    p_list.add_argument("--query-contains", type=str, default=None)

    p_show = lib_sub.add_parser("show", help="Show a session JSON")
    p_show.add_argument("session_id", type=str)
    p_show.add_argument("--library-dir", type=str, default=str(Path("artifacts") / "knowledge_library"))

    p_export = lib_sub.add_parser("export", help="Export knowledge library to HTML/Markdown for viewing")
    p_export.add_argument("--library-dir", type=str, default=str(Path("artifacts") / "knowledge_library"))
    p_export.add_argument("--out-dir", type=str, required=True)
    p_export.add_argument("--format", type=str, default="html", choices=["html", "md"])
    p_export.add_argument("--session", type=str, action="append", default=None, help="session id (repeatable)")
    p_export.add_argument("--no-raw", action="store_true", help="do not copy raw json/jsonl into export")

    p_serve = lib_sub.add_parser("serve", help="Serve an exported directory over HTTP (local)")
    p_serve.add_argument("--dir", type=str, required=True, help="export dir (contains index.html/index.md)")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)

    p_pub = lib_sub.add_parser("publish", help="Publish knowledge library (zip/pdf)")
    p_pub.add_argument("--library-dir", type=str, default=str(Path("artifacts") / "knowledge_library"))
    p_pub.add_argument("--out", type=str, required=True)
    p_pub.add_argument("--format", type=str, default="zip", choices=["zip", "pdf"])
    p_pub.add_argument("--export-format", type=str, default="html", choices=["html", "md"], help="zip export format")
    p_pub.add_argument("--session", type=str, action="append", default=None, help="session id (repeatable)")
    p_pub.add_argument("--font-path", type=str, default=None, help="PDF unicode font path (e.g. malgun.ttf)")
    p_pub.add_argument("--max-sessions", type=int, default=50)
    p_pub.add_argument("--auto-install", action="store_true", help=f"auto install publish deps (also via env {AUTO_INSTALL_ENV}=1)")

    p_del = lib_sub.add_parser("delete", help="Delete ALL knowledge library data (irreversible)")
    p_del.add_argument("--library-dir", type=str, default=str(Path("artifacts") / "knowledge_library"))
    p_del.add_argument("--yes", action="store_true", help="confirm deletion")

    args = parser.parse_args(argv)

    if args.cmd == "tui":
        auto_install = bool(getattr(args, "auto_install", False)) or auto_install_enabled()
        if auto_install:
            root = repo_root()
            reqs = ["requirements-tui.txt"]
            if args.build_dir:
                reqs.insert(0, "requirements-search.txt")
                fallback_pkgs = ("numpy", "sentence-transformers", "hnswlib", "textual")
            else:
                reqs.insert(0, "requirements-core.txt")
                fallback_pkgs = ("numpy", "textual")
            ensure_installed(
                requirements_files=_existing(root, *reqs),
                packages=fallback_pkgs,
                auto_install=True,
                quiet=False,
            )
            if not args.build_dir and _needs_docparse(args.docs or []):
                ensure_installed(
                    requirements_files=_existing(root, "requirements-docparse.txt"),
                    packages=("pdfplumber", "pypdf", "pillow", "pytesseract"),
                    auto_install=True,
                    quiet=False,
                )

        engine = _load_engine_for_tui(args)
        from .tui import run_tui  # lazy import (optional dep: textual)

        run_tui(
            engine,
            default_mode=args.mode,
            top_k_per_pass=args.top_k,
            time_budget_sec=args.time_budget,
        )
        return 0

    if args.cmd == "library":
        lib_dir = Path(args.library_dir)
        if args.library_cmd == "list":
            from .library_viewer import list_knowledge_library_sessions

            sessions = list_knowledge_library_sessions(
                lib_dir,
                limit=args.limit,
                mode=args.mode,
                query_contains=args.query_contains,
            )
            for s in sessions:
                print(f"{s.created_at}\t{s.mode}\t{s.session_id}\t{s.clean_query or s.raw_query}")
            return 0
        if args.library_cmd == "show":
            from .library_viewer import load_knowledge_library_session

            payload = load_knowledge_library_session(lib_dir, args.session_id)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.library_cmd == "export":
            from .library_viewer import export_knowledge_library

            report = export_knowledge_library(
                lib_dir,
                Path(args.out_dir),
                format=args.format,
                session_ids=args.session,
                include_raw=not args.no_raw,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.library_cmd == "serve":
            _serve_dir(Path(args.dir), host=str(args.host), port=int(args.port))
            return 0
        if args.library_cmd == "publish":
            auto_install = bool(getattr(args, "auto_install", False)) or auto_install_enabled()
            if args.format == "pdf" and auto_install:
                root = repo_root()
                ensure_installed(
                    requirements_files=_existing(root, "requirements-publish.txt"),
                    packages=("fpdf2",),
                    auto_install=True,
                    quiet=False,
                )
            if args.format == "zip":
                from .publisher import publish_knowledge_library_zip

                out = publish_knowledge_library_zip(
                    lib_dir,
                    Path(args.out),
                    export_format=args.export_format,
                    session_ids=args.session,
                    include_raw=True,
                )
                print(str(out))
                return 0
            from .publisher import publish_knowledge_library_pdf

            out = publish_knowledge_library_pdf(
                lib_dir,
                Path(args.out),
                session_ids=args.session,
                font_path=args.font_path,
                max_sessions=args.max_sessions,
            )
            print(str(out))
            return 0
        if args.library_cmd == "delete":
            if not args.yes:
                print("Refusing to delete without --yes", file=sys.stderr)
                return 2
            if not lib_dir.exists():
                print("Library directory does not exist.")
                return 0
            from .library import KnowledgeLibrary

            KnowledgeLibrary(lib_dir).delete_all()
            print(f"Deleted: {lib_dir}")
            return 0

    return 2


def _serve_dir(directory: Path, *, host: str, port: int) -> None:
    import http.server
    import socketserver

    directory = directory.resolve()
    if not directory.exists():
        raise FileNotFoundError(f"serve dir not found: {directory}")

    handler = http.server.SimpleHTTPRequestHandler
    # Python 3.11+: handler supports `directory=` kw
    try:
        handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=str(directory), **kwargs)  # type: ignore[assignment]
    except TypeError:
        pass

    with socketserver.TCPServer((host, int(port)), handler) as httpd:
        sa = httpd.socket.getsockname()
        print(f"Serving {directory} on http://{sa[0]}:{sa[1]}/ (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def _load_engine_for_tui(args: argparse.Namespace):
    from .engine import InvestigationEngine

    enable_library = bool(getattr(args, "enable_knowledge_library", False))
    library_dir = getattr(args, "knowledge_library_dir", str(Path("artifacts") / "knowledge_library"))
    enable_web = not bool(getattr(args, "no_web", False))
    enable_web_sandbox = not bool(getattr(args, "no_web_sandbox", False))

    if args.build_dir:
        engine = _load_engine_from_build(
            Path(args.build_dir),
            enable_knowledge_library=enable_library,
            knowledge_library_dir=library_dir,
            enable_web_fallback=enable_web,
            enable_web_sandbox=enable_web_sandbox,
        )
        return engine

    docs = args.docs or []
    if not docs:
        raise SystemExit("tui requires --build-dir or --docs")
    from .parser import DocumentParser, parse_documents

    units = parse_documents(docs, parser=DocumentParser())
    return InvestigationEngine(
        units,
        enable_cache=True,
        enable_web_fallback=enable_web,
        enable_web_sandbox=enable_web_sandbox,
        enable_knowledge_library=enable_library,
        knowledge_library_dir=Path(library_dir),
    )


def _load_engine_from_build(
    build_dir: Path,
    *,
    enable_knowledge_library: bool,
    knowledge_library_dir: str | Path,
    enable_web_fallback: bool,
    enable_web_sandbox: bool,
) :
    from .engine import InvestigationEngine
    from .index_ann import load_index
    from .offline import load_bm25_from_build, load_sharded_bm25_indices

    manifest_path = build_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_count = int(manifest.get("shard_count", 1))
    build_id = manifest.get("knowledge_build_id") or manifest.get("build_id")
    embedding_model = str(manifest.get("embedding_model") or "intfloat/multilingual-e5-small")

    if shard_count > 1 and (build_dir / "shards").exists():
        shard_meta = manifest.get("shards", [])
        ann_shards = []
        all_units: list[Any] = []
        for shard in shard_meta:
            shard_dir = build_dir / shard["path"]
            units = _load_units(shard_dir / "evidence_units.json")
            all_units.extend(units)
            ann_backend = shard.get("ann_backend", "empty")
            ann_meta_path = shard_dir / "ann_index.meta.json"
            if ann_backend == "empty" or not ann_meta_path.exists():
                ann_shards.append(None)
            else:
                ann_suffix = ".npy" if ann_backend == "exact" else ".bin"
                ann_index_path = shard_dir / f"ann_index{ann_suffix}"
                ann_shards.append(load_index(ann_index_path, ann_meta_path))

        bm25_shards = load_sharded_bm25_indices(build_dir)
        return InvestigationEngine(
            all_units,
            build_id=build_id,
            shard_count=shard_count,
            ann_shards=ann_shards,
            bm25_shards=bm25_shards,
            embedding_model=embedding_model,
            enable_web_fallback=enable_web_fallback,
            enable_web_sandbox=enable_web_sandbox,
            enable_knowledge_library=enable_knowledge_library,
            knowledge_library_dir=Path(knowledge_library_dir),
        )

    # Single shard.
    units = _load_units(build_dir / "evidence_units.json")
    bm25 = load_bm25_from_build(build_dir)
    ann_meta_path = build_dir / "ann_index.meta.json"
    ann_backend = manifest.get("ann_backend", "hnsw")
    ann_suffix = ".npy" if ann_backend == "exact" else ".bin"
    ann_index_path = build_dir / f"ann_index{ann_suffix}"
    ann = load_index(ann_index_path, ann_meta_path) if ann_meta_path.exists() else None

    return InvestigationEngine(
        units,
        build_id=build_id,
        shard_count=1,
        ann_index=ann,
        bm25_index=bm25,
        embedding_model=embedding_model,
        enable_web_fallback=enable_web_fallback,
        enable_web_sandbox=enable_web_sandbox,
        enable_knowledge_library=enable_knowledge_library,
        knowledge_library_dir=Path(knowledge_library_dir),
    )


def _load_units(path: Path) -> list[Any]:
    from .schema import EvidenceUnit, SourceType

    raw_units = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvidenceUnit(
            doc_id=row["doc_id"],
            source_type=SourceType(row["source_type"]),
            content=row["content"],
            section_path=row["section_path"],
            char_start=int(row["char_start"]),
            char_end=int(row["char_end"]),
            timestamp=row["timestamp"],
            confidence=float(row["confidence"]),
            metadata=row.get("metadata", {}),
        )
        for row in raw_units
    ]


def _existing(root: Path | None, *names: str) -> list[Path]:
    if root is None:
        return []
    paths: list[Path] = []
    for name in names:
        p = root / name
        if p.exists():
            paths.append(p)
    return paths


def _needs_docparse(docs: Sequence[str]) -> bool:
    for raw in docs:
        ext = Path(raw).suffix.lower()
        if ext in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            return True
    return False
