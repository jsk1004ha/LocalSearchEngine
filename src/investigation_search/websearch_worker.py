from __future__ import annotations

import argparse
import json
import sys

from .websearch import DuckDuckGoSearchProvider, FreeOSINTSearchProvider, SearxNGSearchProvider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="websearch_worker", add_help=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--timeout-sec", type=float, default=12.0)
    parser.add_argument("--max-bytes", type=int, default=1_500_000)
    parser.add_argument("--endpoint", type=str, default="https://duckduckgo.com/html/")
    parser.add_argument("--user-agent", type=str, default="Mozilla/5.0 (compatible; LocalInvestigationSearch/0.1)")
    parser.add_argument("--provider-kind", type=str, default="free-osint")
    args = parser.parse_args(argv)

    if args.provider_kind == "duckduckgo":
        provider = DuckDuckGoSearchProvider(
            endpoint=args.endpoint,
            timeout_sec=max(1.0, float(args.timeout_sec)),
            user_agent=args.user_agent,
            max_bytes=max(0, int(args.max_bytes)),
        )
    elif args.provider_kind == "searxng":
        provider = SearxNGSearchProvider(
            endpoint=args.endpoint,
            timeout_sec=max(1.0, float(args.timeout_sec)),
            user_agent=args.user_agent,
        )
    else:
        provider = FreeOSINTSearchProvider(
            searxng_endpoint=args.endpoint,
            timeout_sec=max(1.0, float(args.timeout_sec)),
            user_agent=args.user_agent,
        )

    try:
        results = provider.search(args.query, max_results=max(0, int(args.max_results)))
        payload = {
            "ok": True,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "rank": r.rank,
                    "provider": r.provider,
                }
                for r in results
            ],
        }
    except Exception as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
