from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence


_TAG_RE = re.compile(r"<[^>]+>")
_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'<(?:a|div)[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</(?:a|div)>',
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    rank: int
    provider: str


class WebSearchProvider(Protocol):
    provider_name: str

    def search(self, query: str, *, max_results: int = 5) -> List[WebSearchResult]:
        """Search web and return ranked snippets."""


@dataclass(frozen=True)
class DuckDuckGoSearchProvider:
    provider_name: str = "duckduckgo"
    endpoint: str = "https://duckduckgo.com/html/"
    timeout_sec: float = 12.0
    user_agent: str = "Mozilla/5.0 (compatible; LocalInvestigationSearch/0.1)"

    def search(self, query: str, *, max_results: int = 5) -> List[WebSearchResult]:
        clean_query = query.strip()
        if not clean_query or max_results <= 0:
            return []

        url = self._build_url(clean_query)
        request = urllib.request.Request(url=url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as resp:
            html_text = resp.read().decode("utf-8", errors="ignore")
        return _parse_duckduckgo_html(html_text, provider=self.provider_name, max_results=max_results)

    def _build_url(self, query: str) -> str:
        params = urllib.parse.urlencode({"q": query, "kl": "us-en"})
        return f"{self.endpoint}?{params}"


def _parse_duckduckgo_html(raw_html: str, *, provider: str, max_results: int) -> List[WebSearchResult]:
    link_rows = list(_RESULT_RE.finditer(raw_html))
    snippets = [m.group("snippet") for m in _SNIPPET_RE.finditer(raw_html)]

    out: list[WebSearchResult] = []
    for idx, match in enumerate(link_rows, start=1):
        if len(out) >= max_results:
            break
        href = _resolve_ddg_url(match.group("href"))
        title = _clean_html(match.group("title"))
        snippet = _clean_html(snippets[idx - 1]) if idx - 1 < len(snippets) else ""
        if not href or not title:
            continue
        out.append(WebSearchResult(title=title, url=href, snippet=snippet, rank=idx, provider=provider))
    return out


def _resolve_ddg_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.path.startswith("/l/"):
        q = urllib.parse.parse_qs(parsed.query)
        uddg = q.get("uddg", [])
        if uddg:
            return urllib.parse.unquote(uddg[0])
    return url


def _clean_html(text: str) -> str:
    clean = _TAG_RE.sub(" ", text)
    clean = html.unescape(clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def to_evidence_content(results: Sequence[WebSearchResult] | Iterable[WebSearchResult]) -> list[str]:
    return [f"{row.title}. {row.snippet}".strip() for row in results]
