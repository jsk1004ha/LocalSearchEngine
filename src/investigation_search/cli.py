from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

from .bootstrap import AUTO_INSTALL_ENV, auto_install_enabled, ensure_installed, module_available, repo_root, requirements_path
from .result_payload import result_to_payload
from .tooling import build_runtime_tooling
from .webfetch import Crawl4AIWebFetchProvider, PlaywrightStealthWebFetchProvider, StdlibWebFetchProvider, SubprocessSandboxWebFetchProvider


class MissingDependencyError(RuntimeError):
    """Raised when a runtime dependency required by the command is missing."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="investigation_search", description="Investigation Search Engine tools (web-only)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tui = sub.add_parser("tui", help="Run terminal UI (Textual)")
    p_tui.add_argument("--mode", type=str, default="investigation", help=_mode_help("default"))
    p_tui.add_argument("--top-k", type=int, default=5, help="top_k_per_pass")
    p_tui.add_argument("--time-budget", type=int, default=120, help="time budget sec")
    _add_runtime_engine_args(p_tui)
    p_tui.add_argument(
        "--auto-install",
        action="store_true",
        help=f"auto install missing optional deps (also via env {AUTO_INSTALL_ENV}=1)",
    )

    p_search = sub.add_parser("search", help="Run one-shot search from CLI")
    p_search.add_argument("query", type=str, help="search query")
    p_search.add_argument("--mode", type=str, default="investigation", help=_mode_help("search"))
    p_search.add_argument("--top-k", type=int, default=5, help="top_k_per_pass")
    p_search.add_argument("--time-budget", type=int, default=120, help="time budget sec")
    p_search.add_argument("--max-items", type=int, default=8, help="max evidence items to print/output")
    p_search.add_argument("--json", action="store_true", help="emit JSON payload")
    p_search.add_argument("--no-diagnostics", action="store_true", help="hide diagnostics from output")
    _add_runtime_engine_args(p_search)
    p_search.add_argument(
        "--auto-install",
        action="store_true",
        help=f"auto install missing optional deps (also via env {AUTO_INSTALL_ENV}=1)",
    )

    p_web = sub.add_parser("web", help="Run browser web UI")
    p_web.add_argument("--host", type=str, default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8787)
    p_web.add_argument("--title", type=str, default="Investigation Search Web")
    p_web.add_argument("--mode", type=str, default="investigation", help=_mode_help("default"))
    p_web.add_argument("--top-k", type=int, default=5, help="default top_k_per_pass")
    p_web.add_argument("--time-budget", type=int, default=120, help="default time budget sec")
    p_web.add_argument("--max-items", type=int, default=8, help="default max evidence cards")
    p_web.add_argument("--diagnostics-default", action="store_true", help="show diagnostics by default in UI")
    _add_runtime_engine_args(p_web)
    p_web.add_argument(
        "--auto-install",
        action="store_true",
        help=f"auto install missing optional deps (also via env {AUTO_INSTALL_ENV}=1)",
    )

    p_build = sub.add_parser("build", help="Build a local evidence/index bundle from documents and optional URLs")
    p_build.add_argument("--output-dir", type=str, required=True, help="directory to write the build artifacts")
    p_build.add_argument("--document", action="append", default=[], help="local document path (repeatable)")
    p_build.add_argument("--url", action="append", default=[], help="URL to crawl with Crawl4AI (repeatable)")
    p_build.add_argument("--corpus-snapshot-id", type=str, default="runtime", help="logical corpus snapshot identifier")
    p_build.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p_build.add_argument("--ann-backend", type=str, default="hnsw", choices=["hnsw", "exact"])
    p_build.add_argument("--shard-count", type=int, default=1)
    p_build.add_argument("--crawl-max-pages", type=int, default=5, help="max pages to ingest from URL inputs")
    p_build.add_argument("--crawl-max-text-chars", type=int, default=180000)
    p_build.add_argument("--crawl-fit-markdown", action="store_true", help="prefer fit_markdown from Crawl4AI")
    p_build.add_argument(
        "--auto-install",
        action="store_true",
        help=f"auto install missing optional deps (also via env {AUTO_INSTALL_ENV}=1)",
    )

    p_env = sub.add_parser("env-check", help="Report runtime dependency status and suggested install steps")
    p_env.add_argument(
        "--command",
        type=str,
        default="search",
        choices=["search", "web", "tui", "build"],
        help="which command profile to check",
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

    if args.cmd in {"tui", "search", "web", "build"}:
        auto_install = bool(getattr(args, "auto_install", False)) or auto_install_enabled()
        if auto_install:
            _ensure_runtime_dependencies(args.cmd)

    if args.cmd == "env-check":
        _print_json(_collect_runtime_environment_report(command=str(args.command)))
        return 0

    if args.cmd == "tui":
        try:
            engine = _load_engine_for_runtime(args)
            from .tui import run_tui  # lazy import (optional dep: textual)

            run_tui(
                engine,
                default_mode=args.mode,
                top_k_per_pass=args.top_k,
                time_budget_sec=args.time_budget,
            )
            return 0
        except MissingDependencyError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if args.cmd == "search":
        try:
            engine = _load_engine_for_runtime(args)
            result = engine.search(
                args.query,
                top_k_per_pass=max(1, int(args.top_k)),
                time_budget_sec=max(1, int(args.time_budget)),
                mode=args.mode,
            )
            include_diagnostics = not bool(args.no_diagnostics)
            if bool(args.json):
                payload = result_to_payload(
                    result,
                    query=args.query,
                    mode=str(args.mode),
                    top_k_per_pass=max(1, int(args.top_k)),
                    time_budget_sec=max(1, int(args.time_budget)),
                    max_items=max(0, int(args.max_items)),
                    include_diagnostics=include_diagnostics,
                )
                session_id = result.diagnostics.get("knowledge_library_session_id")
                if session_id:
                    payload["knowledge_library_session_id"] = session_id
                _print_json(payload)
                return 0

            from .viewer import render_result_text

            print(
                render_result_text(
                    result,
                    query=args.query,
                    max_items=max(0, int(args.max_items)),
                    include_diagnostics=include_diagnostics,
                )
            )
            session_id = result.diagnostics.get("knowledge_library_session_id")
            if session_id:
                print(f"\nSaved session: {session_id}")
            return 0
        except MissingDependencyError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if args.cmd == "web":
        try:
            engine = _load_engine_for_runtime(args)
            from .webapp import WebUiConfig, run_web_ui

            run_web_ui(
                engine,
                host=str(args.host),
                port=max(1, int(args.port)),
                config=WebUiConfig(
                    title=str(args.title),
                    default_mode=str(args.mode),
                    default_top_k=max(1, int(args.top_k)),
                    default_time_budget_sec=max(1, int(args.time_budget)),
                    default_max_items=max(1, int(args.max_items)),
                    default_show_diagnostics=bool(args.diagnostics_default),
                ),
            )
            return 0
        except MissingDependencyError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if args.cmd == "build":
        try:
            return _run_build_command(args)
        except MissingDependencyError as exc:
            print(str(exc), file=sys.stderr)
            return 2

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
            _print_json(payload)
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
            _print_json(report)
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


def _load_engine_for_runtime(args: argparse.Namespace):
    try:
        from .engine import InvestigationEngine
        from .offline import load_bm25_from_build, load_build, load_sharded_bm25_indices, load_sharded_build
    except (ImportError, ModuleNotFoundError) as exc:
        raise _as_missing_dependency_error(exc, command="runtime search")

    enable_library = bool(getattr(args, "enable_knowledge_library", False))
    library_dir = getattr(args, "knowledge_library_dir", str(Path("artifacts") / "knowledge_library"))
    enable_web_sandbox = not bool(getattr(args, "no_web_sandbox", False))
    enable_web_fetch = bool(getattr(args, "web_fetch", False))
    web_fetch_pages = int(getattr(args, "web_fetch_pages", 4) or 0)
    web_fetch_crawl_depth = int(getattr(args, "web_fetch_crawl_depth", 0) or 0)
    web_fetch_workers = int(getattr(args, "web_fetch_workers", 4) or 0)
    tooling = build_runtime_tooling(
        web_fetch_backend=str(getattr(args, "web_fetch_backend", "default") or "default"),
        pii_redaction_backend=str(getattr(args, "pii_redaction", "off") or "off"),
        graph_backend=str(getattr(args, "graph_backend", "off") or "off"),
        neo4j_uri=str(getattr(args, "neo4j_uri", "") or ""),
        neo4j_user=str(getattr(args, "neo4j_user", "") or ""),
        neo4j_password=str(getattr(args, "neo4j_password", "") or ""),
        neo4j_database=str(getattr(args, "neo4j_database", "") or ""),
    )
    web_fetch_provider = _resolve_web_fetch_provider(
        tooling.web_fetch_backend,
        web_fetch_workers=max(1, web_fetch_workers),
    )
    build_dir_raw = str(getattr(args, "build_dir", "") or "").strip()
    evidence_units = []
    build_id = None
    ann_index = None
    ann_shards = None
    bm25_index = None
    bm25_shards = None
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    shard_count = 1
    if build_dir_raw:
        build_dir = Path(build_dir_raw)
        manifest = json.loads((build_dir / "manifest.json").read_text(encoding="utf-8"))
        build_id = manifest.get("knowledge_build_id")
        embedding_model = str(manifest.get("embedding_model") or embedding_model)
        shard_count = max(1, int(manifest.get("shard_count", 1) or 1))
        if shard_count > 1:
            shard_bundles = load_sharded_build(build_dir)
            ann_shards = []
            evidence_units = []
            for units, _embeddings, ann in shard_bundles:
                evidence_units.extend(units)
                ann_shards.append(ann)
            bm25_shards = load_sharded_bm25_indices(build_dir)
        else:
            evidence_units, manifest, _embeddings, ann_index = load_build(build_dir)
            bm25_index = load_bm25_from_build(build_dir)
            build_id = manifest.get("knowledge_build_id", build_id)
            embedding_model = str(manifest.get("embedding_model") or embedding_model)

    return InvestigationEngine(
        evidence_units,
        build_id=build_id,
        ann_index=ann_index,
        ann_shards=ann_shards,
        bm25_index=bm25_index,
        bm25_shards=bm25_shards,
        embedding_model=embedding_model,
        shard_count=shard_count,
        enable_cache=True,
        enable_web_fallback=True,
        enable_web_sandbox=enable_web_sandbox,
        enable_web_fetch=enable_web_fetch,
        web_fetch_max_pages=web_fetch_pages,
        web_fetch_crawl_depth=web_fetch_crawl_depth,
        web_fetch_max_workers=max(1, web_fetch_workers),
        web_fetch_provider=web_fetch_provider,
        enable_knowledge_library=enable_library,
        knowledge_library_dir=Path(library_dir),
        runtime_tooling=tooling,
    )


def _add_runtime_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--knowledge-library-dir", type=str, default=str(Path("artifacts") / "knowledge_library"))
    parser.add_argument("--build-dir", type=str, default="", help="optional offline build directory to load local evidence and indices")
    parser.add_argument("--enable-knowledge-library", action="store_true", help="persist searches to knowledge library")
    parser.add_argument("--no-web-sandbox", action="store_true", help="disable subprocess isolation for web search")
    parser.add_argument("--web-fetch", action="store_true", help="fetch and parse result pages (HTML/PDF) to create richer evidence")
    parser.add_argument("--web-fetch-pages", type=int, default=4, help="max pages to fetch per query")
    parser.add_argument("--web-fetch-crawl-depth", type=int, default=0, help="extra in-site crawl depth from fetched pages (0~2)")
    parser.add_argument("--web-fetch-workers", type=int, default=4, help="parallel workers for web page fetch")
    parser.add_argument(
        "--web-fetch-backend",
        type=str,
        default="default",
        choices=["default", "auto", "stdlib", "sandbox", "playwright", "crawl4ai"],
        help="optional fetch backend; default preserves existing runtime behavior",
    )
    parser.add_argument(
        "--pii-redaction",
        type=str,
        default="off",
        choices=["off", "auto", "presidio"],
        help="optional text redaction backend applied to fetched content",
    )
    parser.add_argument(
        "--graph-backend",
        type=str,
        default="off",
        choices=["off", "auto", "neo4j"],
        help="optional graph export target for diagnostics/analysis metadata",
    )
    parser.add_argument("--neo4j-uri", type=str, default="", help="optional Neo4j URI for graph export")
    parser.add_argument("--neo4j-user", type=str, default="", help="optional Neo4j username")
    parser.add_argument("--neo4j-password", type=str, default="", help="optional Neo4j password")
    parser.add_argument("--neo4j-database", type=str, default="", help="optional Neo4j database")


def _resolve_web_fetch_provider(backend: str, *, web_fetch_workers: int):
    selected = str(backend or "default")
    if selected in {"", "default"}:
        return None
    if selected == "stdlib":
        return StdlibWebFetchProvider(max_workers=max(1, int(web_fetch_workers)))
    if selected == "sandbox":
        return SubprocessSandboxWebFetchProvider(max_workers=max(1, int(web_fetch_workers)))
    if selected == "playwright":
        return PlaywrightStealthWebFetchProvider()
    if selected == "crawl4ai":
        return Crawl4AIWebFetchProvider()
    raise ValueError(f"unsupported web fetch backend: {selected}")


def _ensure_runtime_dependencies(cmd: str) -> None:
    root = repo_root()
    reqs = ["requirements-core.txt"]
    fallback_pkgs = ("numpy",)
    if cmd == "tui":
        reqs.append("requirements-tui.txt")
        fallback_pkgs = ("numpy", "textual")
    elif cmd == "build":
        reqs.append("requirements-search.txt")
        fallback_pkgs = ("numpy",)
    ensure_installed(
        requirements_files=_existing(root, *reqs),
        packages=fallback_pkgs,
        auto_install=True,
        quiet=False,
    )


def _mode_help(prefix: str) -> str:
    return f"{prefix} mode (investigation/reporter/fbi/collection/sniper/rumor/library/llm)"


def _print_json(payload: object) -> None:
    try:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    except UnicodeEncodeError:
        # Windows cp949 등 제한된 콘솔 인코딩에서는 확장 유니코드 출력이 실패할 수 있다.
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


def _existing(root: Path | None, *names: str) -> list[Path]:
    if root is None:
        return []
    paths: list[Path] = []
    for name in names:
        p = root / name
        if p.exists():
            paths.append(p)
    return paths


def _run_build_command(args: argparse.Namespace) -> int:
    try:
        from .crawl_ingest import CrawlIngestConfig, crawl4ai_urls_to_evidence_units
        from .offline import build_manifest, write_build
        from .parser import DocumentParser, parse_documents
    except (ImportError, ModuleNotFoundError) as exc:
        raise _as_missing_dependency_error(exc, command="build")

    documents = [str(path).strip() for path in list(getattr(args, "document", []) or []) if str(path).strip()]
    urls = [str(url).strip() for url in list(getattr(args, "url", []) or []) if str(url).strip()]
    if not documents and not urls:
        print("At least one --document or --url is required", file=sys.stderr)
        return 2

    parser = DocumentParser()
    units = []
    if documents:
        units.extend(parse_documents(documents, parser=parser))
    if urls:
        units.extend(
            crawl4ai_urls_to_evidence_units(
                urls,
                config=CrawlIngestConfig(
                    max_pages=max(1, int(args.crawl_max_pages)),
                    max_text_chars=max(1000, int(args.crawl_max_text_chars)),
                    fit_markdown=bool(args.crawl_fit_markdown),
                ),
            )
        )
    if not units:
        print("No evidence units were produced from the supplied inputs", file=sys.stderr)
        return 2

    manifest = build_manifest(
        units,
        corpus_snapshot_id=str(args.corpus_snapshot_id),
        parser_version=parser.config.parser_version,
        embedding_model=str(args.embedding_model),
    )
    write_build(
        output_dir=Path(args.output_dir),
        evidence_units=units,
        manifest=manifest,
        embedding_model=str(args.embedding_model),
        ann_backend=str(args.ann_backend),
        shard_count=max(1, int(args.shard_count)),
    )
    _print_json(
        {
            "ok": True,
            "output_dir": str(Path(args.output_dir)),
            "evidence_count": len(units),
            "document_count": len(documents),
            "url_count": len(urls),
            "knowledge_build_id": manifest.get("knowledge_build_id"),
        }
    )
    return 0


def _as_missing_dependency_error(exc: BaseException, *, command: str) -> MissingDependencyError:
    report = _collect_runtime_environment_report(command=command)
    guidance = report.get("guidance", [])
    guidance_text = ""
    if guidance:
        guidance_text = "\nSuggested next steps:\n- " + "\n- ".join(str(item) for item in guidance)
    if isinstance(exc, ModuleNotFoundError) and str(getattr(exc, "name", "")) == "numpy":
        return MissingDependencyError(
            f"Missing dependency for {command}: numpy. Install requirements-core.txt (numpy>=1.26) "
            f"or run in the project Python environment before retrying.{guidance_text}"
        )
    name = getattr(exc, "name", None)
    if name:
        return MissingDependencyError(
            f"Missing dependency for {command}: {name}. Install the required package or matching requirements file, then retry.{guidance_text}"
        )
    return MissingDependencyError(
        f"{command} could not start due to an import error: {type(exc).__name__}: {exc}{guidance_text}"
    )


def _collect_runtime_environment_report(*, command: str) -> dict[str, object]:
    root = repo_root()
    req_core = requirements_path("requirements-core.txt")
    req_search = requirements_path("requirements-search.txt")
    req_tui = requirements_path("requirements-tui.txt")
    pip_cmd = shutil.which("pip3") or shutil.which("pip")
    pip_module_available = module_available("pip")
    on_wsl = bool(os.environ.get("WSL_DISTRO_NAME")) or Path("/proc/version").exists() and "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()

    required_modules = ["numpy"]
    if command == "tui":
        required_modules.append("textual")
    missing_modules = [name for name in required_modules if not module_available(name)]

    local_windows_venv = Path(".venv/Scripts/python.exe")
    local_linux_venv = Path(".venv/bin/python")
    guidance: list[str] = []
    if missing_modules:
        can_run_pip = bool(pip_module_available or pip_cmd)
        if pip_module_available and req_core is not None:
            guidance.append(f"{sys.executable} -m pip install -r {req_core.name}")
        elif pip_cmd and req_core is not None:
            guidance.append(f"{pip_cmd} install -r {req_core.name}")
        else:
            guidance.append("Install pip for this Python first (for example: sudo apt-get install python3-pip)")
        if can_run_pip and command in {"search", "web", "build"} and req_search is not None:
            guidance.append(f"{sys.executable} -m pip install -r {req_search.name}")
        if can_run_pip and command == "tui" and req_tui is not None:
            guidance.append(f"{sys.executable} -m pip install -r {req_tui.name}")
        if on_wsl and local_windows_venv.exists() and not local_linux_venv.exists():
            guidance.append("Detected Windows virtualenv under .venv/Scripts; create/use a Linux venv inside WSL instead")

    return {
        "command": command,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "pip_module_available": pip_module_available,
        "pip_command": pip_cmd,
        "repo_root": str(root) if root is not None else None,
        "requirements": {
            "core": str(req_core) if req_core is not None else None,
            "search": str(req_search) if req_search is not None else None,
            "tui": str(req_tui) if req_tui is not None else None,
        },
        "environment": {
            "auto_install_enabled": auto_install_enabled(),
            "on_wsl": on_wsl,
            "windows_venv_python_exists": local_windows_venv.exists(),
            "linux_venv_python_exists": local_linux_venv.exists(),
        },
        "required_modules": required_modules,
        "missing_modules": missing_modules,
        "guidance": guidance,
    }

