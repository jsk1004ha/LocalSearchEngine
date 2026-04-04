from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Sequence

from .schema import EvidenceUnit, SourceType
from .webfetch import chunk_text


@dataclass(frozen=True)
class CrawlIngestConfig:
    max_pages: int = 5
    max_text_chars: int = 180_000
    chunk_max_chars: int = 900
    chunk_overlap: int = 120
    chunk_min_chars: int = 160
    confidence: float = 0.64
    fit_markdown: bool = True


@dataclass(frozen=True)
class CrawledDocument:
    url: str
    final_url: str
    title: str
    markdown: str
    success: bool
    error: str | None = None


async def _crawl4ai_collect(urls: Sequence[str], config: CrawlIngestConfig) -> list[CrawledDocument]:
    try:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig  # type: ignore[import-not-found]
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("crawl4ai_not_installed") from exc

    out: list[CrawledDocument] = []
    crawl_config = CrawlerRunConfig(markdown_generator=DefaultMarkdownGenerator())
    async with AsyncWebCrawler() as crawler:
        for raw_url in list(urls)[: max(0, int(config.max_pages))]:
            url = str(raw_url or "").strip()
            if not url:
                continue
            try:
                result = await crawler.arun(url=url, config=crawl_config)
                markdown_result = getattr(result, "markdown", "")
                markdown = _extract_markdown(markdown_result, prefer_fit=bool(config.fit_markdown))
                if not markdown:
                    markdown = str(getattr(result, "cleaned_html", "") or "")
                out.append(
                    CrawledDocument(
                        url=url,
                        final_url=str(getattr(result, "url", url) or url),
                        title=str(getattr(result, "title", "") or ""),
                        markdown=str(markdown or "")[: int(config.max_text_chars)],
                        success=bool(getattr(result, "success", False)),
                        error=None if bool(getattr(result, "success", False)) else str(getattr(result, "error_message", "crawl_failed")),
                    )
                )
            except Exception as exc:
                out.append(
                    CrawledDocument(
                        url=url,
                        final_url=url,
                        title="",
                        markdown="",
                        success=False,
                        error=f"crawl4ai_error:{type(exc).__name__}",
                    )
                )
    return out


def crawl4ai_urls_to_documents(
    urls: Sequence[str] | Iterable[str],
    *,
    config: CrawlIngestConfig | None = None,
) -> list[CrawledDocument]:
    cfg = config or CrawlIngestConfig()
    clean_urls = [str(url).strip() for url in list(urls)]
    clean_urls = [url for url in clean_urls if url]
    if not clean_urls:
        return []
    return asyncio.run(_crawl4ai_collect(clean_urls, cfg))


def crawl4ai_urls_to_evidence_units(
    urls: Sequence[str] | Iterable[str],
    *,
    config: CrawlIngestConfig | None = None,
    timestamp: str | None = None,
) -> list[EvidenceUnit]:
    cfg = config or CrawlIngestConfig()
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    documents = crawl4ai_urls_to_documents(urls, config=cfg)
    units: list[EvidenceUnit] = []

    for doc in documents:
        if not doc.success:
            continue
        text = str(doc.markdown or "").strip()
        if not text:
            continue
        page_url = str(doc.final_url or doc.url)
        chunks = chunk_text(
            text,
            max_chars=int(cfg.chunk_max_chars),
            overlap=int(cfg.chunk_overlap),
            min_chars=int(cfg.chunk_min_chars),
        )
        if not chunks:
            chunks = [(text, 0, len(text))]
        title = str(doc.title or "").strip()
        section_base = title[:90] if title else "crawl4ai_page"
        for idx, (chunk, start_off, end_off) in enumerate(chunks, start=1):
            units.append(
                EvidenceUnit(
                    doc_id=page_url,
                    source_type=SourceType.WEB_PAGE_TEXT,
                    content=chunk,
                    section_path=f"{section_base}#chunk{idx}",
                    char_start=int(start_off),
                    char_end=int(end_off),
                    timestamp=ts,
                    confidence=float(cfg.confidence),
                    metadata={
                        "provider": "crawl4ai",
                        "url": page_url,
                        "title": title,
                        "crawl_success": True,
                    },
                )
            )
    return units


def _extract_markdown(markdown_result: object, *, prefer_fit: bool) -> str:
    if isinstance(markdown_result, str):
        return markdown_result
    fit_markdown = getattr(markdown_result, "fit_markdown", "")
    if prefer_fit and isinstance(fit_markdown, str) and fit_markdown.strip():
        return fit_markdown
    raw_markdown = getattr(markdown_result, "raw_markdown", "")
    if isinstance(raw_markdown, str) and raw_markdown.strip():
        return raw_markdown
    markdown = getattr(markdown_result, "markdown", "")
    if isinstance(markdown, str):
        return markdown
    return str(markdown_result or "")

