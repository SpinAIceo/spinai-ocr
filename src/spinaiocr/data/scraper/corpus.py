"""Text-only scrapers for corpus building (synthetic data input).

Supported sources:
    WikipediaScraper    — MediaWiki random/read API (CC-BY-SA)
    KoglNewsScraper     — Korean Open Government Licence news announcements
    RssScraper          — generic RSS/Atom (provide feed URL + license)

We DO NOT scrape Naver News, Daum News, or any paywalled/copyrighted source.
"""
from __future__ import annotations

import re
from typing import AsyncIterator

from spinaiocr.data.scraper.base import AsyncScraper, ScrapeResult, ScraperConfig


_TAG = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    return _TAG.sub(" ", s).strip()


class WikipediaScraper(AsyncScraper):
    """Random Korean Wikipedia articles.

    Uses the MediaWiki REST API. Content is CC-BY-SA 3.0 (attribution required
    in downstream usage; preserved in manifest).
    """

    source_name = "wikipedia_ko"
    license = "cc-by-sa"

    def __init__(self, config: ScraperConfig | None = None, lang: str = "ko", n_pages: int = 10_000) -> None:
        super().__init__(config)
        self.lang = lang
        self.n_pages = n_pages

    async def iter_tasks(self) -> AsyncIterator[str]:
        # Random-article API — returns a JSON list of titles we then fetch.
        import aiohttp  # type: ignore

        list_url = (
            f"https://{self.lang}.wikipedia.org/w/api.php"
            "?action=query&list=random&rnnamespace=0&rnlimit=50&format=json"
        )
        headers = {"User-Agent": self.config.user_agent}
        fetched = 0
        async with aiohttp.ClientSession(headers=headers) as session:
            while fetched < self.n_pages:
                async with session.get(list_url, timeout=self.config.timeout) as resp:
                    data = await resp.json()
                for row in data.get("query", {}).get("random", []):
                    title = row.get("title", "").replace(" ", "_")
                    if not title:
                        continue
                    yield (
                        f"https://{self.lang}.wikipedia.org/api/rest_v1/page/plain/{title}"
                    )
                    fetched += 1
                    if fetched >= self.n_pages:
                        return

    async def fetch(self, session, url: str) -> ScrapeResult | None:
        async with session.get(url, timeout=self.config.timeout) as resp:
            if resp.status != 200:
                return None
            text = await resp.text()
        text = _strip_html(text).strip()
        if len(text) < 50:
            return None
        return ScrapeResult(
            url=url,
            source_name=self.source_name,
            license=self.license,
            content_type="text/plain",
            text=text,
            meta={"lang": self.lang},
        )


class KoglNewsScraper(AsyncScraper):
    """Korean government press-release feed (KOGL Type 1 — free commercial).

    Note: data.go.kr / various ministry sites syndicate via RSS. Because
    specific endpoints change often, this takes a seeds list; the default is
    empty until the user configures one in their local run.
    """

    source_name = "kogl_news"
    license = "kogl-type-1"

    def __init__(self, seeds: list[str] | None = None, config: ScraperConfig | None = None) -> None:
        super().__init__(config)
        self.seeds = list(seeds or [])

    async def iter_tasks(self) -> AsyncIterator[str]:
        for s in self.seeds:
            yield s

    async def fetch(self, session, url: str) -> ScrapeResult | None:
        async with session.get(url, timeout=self.config.timeout) as resp:
            if resp.status != 200:
                return None
            text = await resp.text()
        cleaned = _strip_html(text)
        if len(cleaned) < 30:
            return None
        return ScrapeResult(
            url=url,
            source_name=self.source_name,
            license=self.license,
            content_type="text/html",
            text=cleaned,
            meta={},
        )


class RssScraper(AsyncScraper):
    """Generic RSS/Atom feed scraper. License must be passed in."""

    source_name = "rss"

    def __init__(
        self,
        feed_urls: list[str],
        license: str,
        source_name: str = "rss",
        config: ScraperConfig | None = None,
    ) -> None:
        self.source_name = source_name
        self.license = license  # type: ignore[misc]
        super().__init__(config)
        self.feeds = feed_urls

    async def iter_tasks(self) -> AsyncIterator[str]:
        import xml.etree.ElementTree as ET

        import aiohttp  # type: ignore

        headers = {"User-Agent": self.config.user_agent}
        async with aiohttp.ClientSession(headers=headers) as session:
            for feed in self.feeds:
                try:
                    async with session.get(feed, timeout=self.config.timeout) as resp:
                        body = await resp.text()
                except Exception:  # noqa: BLE001
                    continue
                try:
                    root = ET.fromstring(body)
                except ET.ParseError:
                    continue
                for item in root.iter():
                    if item.tag.endswith("link"):
                        url = (item.text or item.attrib.get("href", "")).strip()
                        if url:
                            yield url

    async def fetch(self, session, url: str) -> ScrapeResult | None:
        async with session.get(url, timeout=self.config.timeout) as resp:
            if resp.status != 200:
                return None
            body = await resp.text()
        cleaned = _strip_html(body)
        if len(cleaned) < 50:
            return None
        return ScrapeResult(
            url=url,
            source_name=self.source_name,
            license=self.license,
            content_type="text/html",
            text=cleaned,
        )
