from __future__ import annotations

import argparse
import json
import sys

from .webfetch import fetch_urls


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="webfetch_worker")
    p.add_argument("--url", action="append", default=None, help="URL to fetch (repeatable)")
    p.add_argument("--timeout-sec", type=float, default=12.0)
    p.add_argument("--max-bytes", type=int, default=2_500_000)
    p.add_argument("--max-text-chars", type=int, default=180_000)
    p.add_argument("--max-redirects", type=int, default=6)
    p.add_argument("--max-pages", type=int, default=5)
    p.add_argument("--user-agent", type=str, default="Mozilla/5.0 (compatible; InvestigationSearch/0.9)")
    p.add_argument("--allow-pdf", action="store_true", help="Allow PDF parsing via pypdf if installed")
    p.add_argument("--max-workers", type=int, default=4)

    args = p.parse_args(argv)
    urls = args.url or []
    try:
        pages = fetch_urls(
            urls,
            timeout_sec=float(args.timeout_sec),
            user_agent=str(args.user_agent),
            max_bytes=int(args.max_bytes),
            max_text_chars=int(args.max_text_chars),
            max_redirects=int(args.max_redirects),
            max_pages=int(args.max_pages),
            allow_pdf=bool(args.allow_pdf),
            max_workers=int(args.max_workers),
        )
        payload = {
            "ok": True,
            "pages": [
                {
                    "url": page.url,
                    "final_url": page.final_url,
                    "status": page.status,
                    "content_type": page.content_type,
                    "title": page.title,
                    "text": page.text,
                    "bytes_read": page.bytes_read,
                    "truncated": page.truncated,
                    "discovered_links": list(page.discovered_links),
                    "error": page.error,
                }
                for page in pages
            ],
        }
    except Exception as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}

    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

