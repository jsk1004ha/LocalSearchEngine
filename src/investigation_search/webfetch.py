from __future__ import annotations

import html
import io
import ipaddress
import os
import re
import socket
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence, Tuple


_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(?P<title>.*?)</title>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"[ \\t\\r\\f\\v]+")
_NL_RE = re.compile(r"\\n{3,}")
_BLOCK_TAG_RE = re.compile(r"</?(p|br|li|div|tr|h[1-6]|pre|blockquote)[^>]*>", re.IGNORECASE)


@dataclass(frozen=True)
class WebFetchedPage:
    url: str
    final_url: str
    status: int | None
    content_type: str
    title: str
    text: str
    bytes_read: int
    truncated: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


class WebFetchProvider(Protocol):
    provider_name: str

    def fetch(self, urls: Sequence[str] | Iterable[str], *, max_pages: int = 5) -> List[WebFetchedPage]:
        """Fetch urls and return extracted plain text."""


@dataclass(frozen=True)
class StdlibWebFetchProvider:
    provider_name: str = "stdlib-webfetch"
    timeout_sec: float = 12.0
    user_agent: str = "Mozilla/5.0 (compatible; InvestigationSearch/0.9)"
    max_bytes: int = 2_500_000
    max_text_chars: int = 180_000
    max_redirects: int = 6
    allow_pdf: bool = True

    def fetch(self, urls: Sequence[str] | Iterable[str], *, max_pages: int = 5) -> List[WebFetchedPage]:
        return fetch_urls(
            urls,
            timeout_sec=float(self.timeout_sec),
            user_agent=str(self.user_agent),
            max_bytes=int(self.max_bytes),
            max_text_chars=int(self.max_text_chars),
            max_redirects=int(self.max_redirects),
            max_pages=int(max_pages),
            allow_pdf=bool(self.allow_pdf),
        )


def fetch_urls(
    urls: Sequence[str] | Iterable[str],
    *,
    timeout_sec: float,
    user_agent: str,
    max_bytes: int,
    max_text_chars: int,
    max_redirects: int,
    max_pages: int,
    allow_pdf: bool,
) -> List[WebFetchedPage]:
    seq = [str(u).strip() for u in list(urls)]
    seq = [u for u in seq if u]
    if not seq or max_pages <= 0:
        return []

    out: list[WebFetchedPage] = []
    for raw_url in seq[: max(0, int(max_pages))]:
        out.append(
            _fetch_one(
                raw_url,
                timeout_sec=float(timeout_sec),
                user_agent=str(user_agent),
                max_bytes=int(max_bytes),
                max_text_chars=int(max_text_chars),
                max_redirects=int(max_redirects),
                allow_pdf=bool(allow_pdf),
            )
        )
    return out


def chunk_text(text: str, *, max_chars: int = 900, overlap: int = 120, min_chars: int = 160) -> List[Tuple[str, int, int]]:
    """Return chunks as (chunk, start, end) in the original `text` offsets."""
    clean = (text or "").strip()
    if not clean:
        return []

    max_chars = max(120, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars // 2))
    min_chars = max(0, int(min_chars))

    # Prefer paragraph-ish boundaries; keep offsets by working on the original string.
    # We create a list of spans (start,end) and then pack them.
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(clean)
    # Split on double newlines if present.
    parts = re.split(r"(\\n\\s*\\n+)", clean)
    if len(parts) <= 1:
        spans = [(0, n)]
    else:
        cursor = 0
        for part in parts:
            if not part:
                continue
            seg_start = cursor
            seg_end = cursor + len(part)
            cursor = seg_end
            if part.strip():
                spans.append((seg_start, seg_end))

    chunks: list[tuple[str, int, int]] = []
    buf_start: int | None = None
    buf_end: int | None = None
    for seg_start, seg_end in spans:
        if buf_start is None:
            buf_start, buf_end = seg_start, seg_end
        else:
            # Extend buffer to include this span.
            buf_end = seg_end

        if buf_end is None:
            continue
        if (buf_end - buf_start) >= max_chars:
            end = buf_end
            chunk = clean[buf_start:end].strip()
            if len(chunk) >= min_chars:
                chunks.append((chunk, int(buf_start), int(end)))
            # Start next chunk with overlap.
            next_start = max(0, end - overlap)
            buf_start, buf_end = next_start, next_start

    if buf_start is not None and buf_end is not None and buf_end > buf_start:
        chunk = clean[buf_start:buf_end].strip()
        if len(chunk) >= min_chars:
            chunks.append((chunk, int(buf_start), int(buf_end)))

    if not chunks:
        # Fallback: hard slicing.
        hard: list[tuple[str, int, int]] = []
        step = max_chars - overlap
        for i in range(0, n, max(step, 1)):
            j = min(n, i + max_chars)
            piece = clean[i:j].strip()
            if len(piece) >= min_chars:
                hard.append((piece, int(i), int(j)))
            if j >= n:
                break
        return hard

    return chunks


def _fetch_one(
    raw_url: str,
    *,
    timeout_sec: float,
    user_agent: str,
    max_bytes: int,
    max_text_chars: int,
    max_redirects: int,
    allow_pdf: bool,
) -> WebFetchedPage:
    url = str(raw_url or "").strip()
    if not url:
        return WebFetchedPage(
            url="",
            final_url="",
            status=None,
            content_type="",
            title="",
            text="",
            bytes_read=0,
            truncated=False,
            error="empty_url",
        )

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        parsed = None

    if parsed is None or parsed.scheme.lower() not in {"http", "https"}:
        return WebFetchedPage(
            url=url,
            final_url=url,
            status=None,
            content_type="",
            title="",
            text="",
            bytes_read=0,
            truncated=False,
            error="unsupported_scheme",
        )

    host = (parsed.hostname or "").strip()
    if not host:
        return WebFetchedPage(
            url=url,
            final_url=url,
            status=None,
            content_type="",
            title="",
            text="",
            bytes_read=0,
            truncated=False,
            error="missing_hostname",
        )

    if not _is_public_hostname(host):
        return WebFetchedPage(
            url=url,
            final_url=url,
            status=None,
            content_type="",
            title="",
            text="",
            bytes_read=0,
            truncated=False,
            error="blocked_hostname",
        )

    req = urllib.request.Request(
        url=url,
        method="GET",
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/pdf;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        },
    )

    opener = urllib.request.build_opener(_LimitedRedirectHandler(max_redirects=max_redirects))
    try:
        with opener.open(req, timeout=float(timeout_sec)) as resp:
            final_url = str(getattr(resp, "geturl", lambda: url)())
            status = int(getattr(resp, "status", None) or 0) or None
            content_type = str(resp.headers.get("Content-Type") or "").strip()

            # Re-check redirect target host for SSRF prevention.
            try:
                final_parsed = urllib.parse.urlparse(final_url)
                final_host = (final_parsed.hostname or "").strip()
            except Exception:
                final_host = ""
            if final_host and not _is_public_hostname(final_host):
                raise RuntimeError("redirect_blocked_hostname")

            raw = resp.read(max(0, int(max_bytes)) + 1)
    except Exception as exc:
        return WebFetchedPage(
            url=url,
            final_url=url,
            status=None,
            content_type="",
            title="",
            text="",
            bytes_read=0,
            truncated=False,
            error=f"fetch_error:{type(exc).__name__}",
        )

    truncated = len(raw) > max(0, int(max_bytes))
    if truncated:
        raw = raw[: max(0, int(max_bytes))]

    bytes_read = len(raw)
    ctype_lower = content_type.lower()
    url_lower = final_url.lower()

    title = ""
    extracted = ""

    if ("application/pdf" in ctype_lower or url_lower.endswith(".pdf")) and allow_pdf:
        extracted, title = _extract_pdf_text(raw, max_chars=max_text_chars)
        if not extracted:
            return WebFetchedPage(
                url=url,
                final_url=final_url,
                status=status,
                content_type=content_type,
                title=title,
                text="",
                bytes_read=bytes_read,
                truncated=truncated,
                error="pdf_no_text",
            )
    else:
        decoded = _decode_bytes(raw, content_type=content_type)
        if "<html" in decoded.lower() or "text/html" in ctype_lower:
            extracted, title = _html_to_text(decoded)
        else:
            extracted = _normalize_text(decoded)

    extracted = (extracted or "")[: max(0, int(max_text_chars))]
    return WebFetchedPage(
        url=url,
        final_url=final_url,
        status=status,
        content_type=content_type,
        title=title[:240],
        text=extracted,
        bytes_read=bytes_read,
        truncated=truncated,
        error=None if extracted.strip() else "empty_text",
    )


def _decode_bytes(raw: bytes, *, content_type: str) -> str:
    # Try charset from headers first.
    ct = str(content_type or "")
    charset = ""
    if "charset=" in ct.lower():
        try:
            charset = ct.split("charset=", 1)[1].split(";", 1)[0].strip().strip('"').strip("'")
        except Exception:
            charset = ""
    for enc in [charset, "utf-8", "cp949", "euc-kr", "latin-1"]:
        if not enc:
            continue
        try:
            return raw.decode(enc, errors="ignore")
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def _normalize_text(text: str) -> str:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Collapse long whitespace but preserve paragraph breaks.
    s = _WS_RE.sub(" ", s)
    s = _NL_RE.sub("\n\n", s)
    return s.strip()


def _html_to_text(raw_html: str) -> tuple[str, str]:
    if not raw_html:
        return "", ""
    # Title
    title = ""
    m = _TITLE_RE.search(raw_html)
    if m:
        title = _clean_html(m.group("title"))

    # Remove scripts/styles and mark block tags as line breaks.
    cleaned = _SCRIPT_STYLE_RE.sub(" ", raw_html)
    cleaned = _BLOCK_TAG_RE.sub("\n", cleaned)
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _WS_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[ \\t]*\\n[ \\t]*", "\n", cleaned)
    cleaned = _NL_RE.sub("\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned, title


def _clean_html(text: str) -> str:
    if not text:
        return ""
    clean = _TAG_RE.sub(" ", text)
    clean = html.unescape(clean)
    clean = _WS_RE.sub(" ", clean)
    clean = clean.strip()
    return clean


def _extract_pdf_text(raw: bytes, *, max_chars: int) -> tuple[str, str]:
    # Optional dependency: pypdf
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except Exception:
        return "", ""

    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception:
        return "", ""

    title = ""
    try:
        meta = reader.metadata or {}
        title = str(getattr(meta, "title", "") or meta.get("/Title") or "").strip()
    except Exception:
        title = ""

    texts: list[str] = []
    remaining = max(0, int(max_chars))
    # Limit pages to control runtime.
    for page in list(reader.pages)[:12]:
        if remaining <= 0:
            break
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = _normalize_text(t)
        if not t:
            continue
        if len(t) > remaining:
            t = t[:remaining]
        texts.append(t)
        remaining -= len(t)
    return ("\n\n".join(texts)).strip(), title


class _LimitedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, *, max_redirects: int):
        super().__init__()
        self._max_redirects = max(0, int(max_redirects))
        self._count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        self._count += 1
        if self._count > self._max_redirects:
            raise RuntimeError("too_many_redirects")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _is_public_hostname(host: str) -> bool:
    h = str(host or "").strip().lower()
    if not h:
        return False
    if h in {"localhost", "localhost.localdomain"}:
        return False
    if h.endswith(".local") or h.endswith(".internal"):
        return False

    # Raw IP literal
    try:
        ip = ipaddress.ip_address(h)
        return _is_public_ip(ip)
    except ValueError:
        pass

    # Resolve DNS; block if any resolution is non-public.
    try:
        infos = socket.getaddrinfo(h, None, proto=socket.IPPROTO_TCP)
    except Exception:
        return False
    if not infos:
        return False
    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0] if isinstance(sockaddr, tuple) and sockaddr else ""
        if not ip_str:
            return False
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        if not _is_public_ip(ip):
            return False
    return True


def _is_public_ip(ip: ipaddress._BaseAddress) -> bool:
    # Explicitly block private/internal ranges.
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return False
    # Final gate: `is_global` is a property on modern Python versions.
    try:
        return bool(getattr(ip, "is_global"))
    except Exception:
        return True


@dataclass(frozen=True)
class SubprocessSandboxWebFetchProvider:
    """Fetch + parse web pages in a separate Python process.

    Best-effort isolation:
    - Process boundary
    - URL allowlist (http/https) and SSRF guard (blocks private/loopback IPs)
    - Timeout + max_bytes
    """

    provider_name: str = "webfetch-subprocess"
    timeout_sec: float = 12.0
    user_agent: str = "Mozilla/5.0 (compatible; InvestigationSearch/0.9)"
    max_bytes: int = 2_500_000
    max_text_chars: int = 180_000
    max_redirects: int = 6
    allow_pdf: bool = True

    def fetch(self, urls: Sequence[str] | Iterable[str], *, max_pages: int = 5) -> List[WebFetchedPage]:
        import json
        import subprocess
        import sys

        seq = [str(u).strip() for u in list(urls)]
        seq = [u for u in seq if u]
        if not seq or max_pages <= 0:
            return []

        cmd = [
            sys.executable,
            "-m",
            "investigation_search.webfetch_worker",
            "--timeout-sec",
            str(float(self.timeout_sec)),
            "--max-bytes",
            str(int(self.max_bytes)),
            "--max-text-chars",
            str(int(self.max_text_chars)),
            "--max-redirects",
            str(int(self.max_redirects)),
            "--max-pages",
            str(int(max_pages)),
            "--user-agent",
            str(self.user_agent),
        ]
        if self.allow_pdf:
            cmd.append("--allow-pdf")
        for u in seq[: max(0, int(max_pages))]:
            cmd.extend(["--url", u])

        env = os.environ.copy()
        env.setdefault("PYTHONNOUSERSITE", "1")
        _ensure_worker_pythonpath(env)

        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(1.0, float(self.timeout_sec) + 4.0),
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("webfetch_worker_timeout")

        raw = (proc.stdout or "").strip()
        if not raw:
            err = (proc.stderr or "").strip()
            raise RuntimeError(f"webfetch_worker_no_output rc={proc.returncode} stderr={err[:180]}")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise RuntimeError("webfetch_worker_invalid_json")
        if not isinstance(payload, dict) or not payload.get("ok"):
            raise RuntimeError(str(payload.get("error") or "webfetch_worker_error"))
        pages = payload.get("pages", [])
        if not isinstance(pages, list):
            return []

        out: list[WebFetchedPage] = []
        for row in pages:
            if not isinstance(row, dict):
                continue
            out.append(
                WebFetchedPage(
                    url=str(row.get("url", "")),
                    final_url=str(row.get("final_url", "")),
                    status=int(row.get("status", 0) or 0) or None,
                    content_type=str(row.get("content_type", "")),
                    title=str(row.get("title", "")),
                    text=str(row.get("text", "")),
                    bytes_read=int(row.get("bytes_read", 0) or 0),
                    truncated=bool(row.get("truncated", False)),
                    error=row.get("error"),
                )
            )
        return out


def _ensure_worker_pythonpath(env: dict) -> None:
    # When running from a repo checkout, ensure the subprocess can import from `src`.
    here = os.path.abspath(__file__)
    src_dir = os.path.dirname(os.path.dirname(here))  # .../src
    current = env.get("PYTHONPATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if src_dir not in parts:
        parts.insert(0, src_dir)
        env["PYTHONPATH"] = os.pathsep.join(parts)
