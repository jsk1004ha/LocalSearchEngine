from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .bootstrap import AUTO_INSTALL_ENV, auto_install_enabled, ensure_installed, repo_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="investigation_search", description="Investigation Search Engine tools (web-only)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tui = sub.add_parser("tui", help="Run terminal UI (Textual)")
    p_tui.add_argument("--mode", type=str, default="investigation", help="default mode (investigation/report/fbi/...)")
    p_tui.add_argument("--top-k", type=int, default=5, help="top_k_per_pass")
    p_tui.add_argument("--time-budget", type=int, default=120, help="time budget sec")
    p_tui.add_argument("--knowledge-library-dir", type=str, default=str(Path("artifacts") / "knowledge_library"))
    p_tui.add_argument("--enable-knowledge-library", action="store_true", help="persist searches to knowledge library")
    p_tui.add_argument("--no-web-sandbox", action="store_true", help="disable subprocess isolation for web search")
    p_tui.add_argument("--web-fetch", action="store_true", help="fetch and parse result pages (HTML/PDF) to create richer evidence")
    p_tui.add_argument("--web-fetch-pages", type=int, default=4, help="max pages to fetch per query")
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
            reqs = ["requirements-core.txt", "requirements-tui.txt"]
            fallback_pkgs = ("numpy", "textual")
            ensure_installed(
                requirements_files=_existing(root, *reqs),
                packages=fallback_pkgs,
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
    enable_web_sandbox = not bool(getattr(args, "no_web_sandbox", False))
    enable_web_fetch = bool(getattr(args, "web_fetch", False))
    web_fetch_pages = int(getattr(args, "web_fetch_pages", 4) or 0)

    return InvestigationEngine(
        [],
        enable_cache=True,
        enable_web_fallback=True,
        enable_web_sandbox=enable_web_sandbox,
        enable_web_fetch=enable_web_fetch,
        web_fetch_max_pages=web_fetch_pages,
        enable_knowledge_library=enable_library,
        knowledge_library_dir=Path(library_dir),
    )


def _existing(root: Path | None, *names: str) -> list[Path]:
    if root is None:
        return []
    paths: list[Path] = []
    for name in names:
        p = root / name
        if p.exists():
            paths.append(p)
    return paths

