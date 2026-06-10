"""Scraper foundations: rate limiting, robots.txt, manifest, license filtering.

All scrapers in this package MUST derive from :class:`AsyncScraper` and honor
robots.txt. This is a hard project rule — not optional.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import AsyncIterator, Iterable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

log = logging.getLogger(__name__)


ALLOWED_LICENSES = {
    "cc0",
    "public-domain",
    "cc-by",
    "cc-by-sa",
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "bsd-2-clause",
    "kogl-type-1",  # Korea Open Government License Type 1 (free, incl. commercial)
}


@dataclass
class ScraperConfig:
    user_agent: str = "SPINAI-OCR-Bot/0.1 (+https://github.com/spinai/spinai-ocr)"
    rate_limit_per_host: float = 1.0  # seconds between requests per host
    max_concurrent: int = 4
    timeout: float = 30.0
    respect_robots: bool = True
    allowed_licenses: frozenset[str] = frozenset(ALLOWED_LICENSES)
    output_dir: Path = Path("data/scraped")


@dataclass
class ScrapeResult:
    url: str
    source_name: str
    license: str
    fetched_at: float = field(default_factory=time.time)
    content_type: str = ""
    local_path: str = ""
    text: str = ""
    # for images
    width: int = 0
    height: int = 0
    bytes: int = 0
    # freeform
    meta: dict = field(default_factory=dict)


class _HostLimiter:
    """Per-host polite delay."""

    def __init__(self, per_host_delay: float) -> None:
        self._delay = per_host_delay
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def wait(self, host: str) -> None:
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            last = self._last.get(host, 0.0)
            remaining = self._delay - (time.monotonic() - last)
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last[host] = time.monotonic()


class _RobotsCache:
    def __init__(self, user_agent: str) -> None:
        self._user_agent = user_agent
        self._cache: dict[str, RobotFileParser] = {}

    async def allowed(self, session, url: str) -> bool:
        parts = urlparse(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._cache:
            rp = RobotFileParser()
            try:
                async with session.get(f"{base}/robots.txt", timeout=10) as resp:
                    text = await resp.text()
                rp.parse(text.splitlines())
            except Exception:  # noqa: BLE001
                rp.parse([])  # permissive if no robots.txt
            self._cache[base] = rp
        return self._cache[base].can_fetch(self._user_agent, url)


class AsyncScraper:
    """Base class. Subclasses override :meth:`iter_tasks` and :meth:`fetch`."""

    source_name = "base"
    license = "unknown"

    def __init__(self, config: ScraperConfig | None = None) -> None:
        self.config = config or ScraperConfig()
        if self.license not in self.config.allowed_licenses:
            raise ValueError(
                f"{type(self).__name__} license '{self.license}' is not in the "
                f"allowed list {sorted(self.config.allowed_licenses)}. "
                f"Either add the license to the whitelist or use a different source."
            )
        self._limiter = _HostLimiter(self.config.rate_limit_per_host)
        self._robots: _RobotsCache | None = None
        self._sem = asyncio.Semaphore(self.config.max_concurrent)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = self.config.output_dir / f"{self.source_name}.jsonl"

    # --- overrides -------------------------------------------------------

    async def iter_tasks(self) -> AsyncIterator[str]:  # pragma: no cover - subclass responsibility
        raise NotImplementedError
        yield  # type: ignore[unreachable]

    async def fetch(self, session, url: str) -> ScrapeResult | None:  # pragma: no cover
        raise NotImplementedError

    # --- runtime ---------------------------------------------------------

    async def _run_one(self, session, url: str) -> ScrapeResult | None:
        if self.config.respect_robots:
            assert self._robots is not None
            if not await self._robots.allowed(session, url):
                log.info("robots.txt disallows %s", url)
                return None
        host = urlparse(url).netloc
        await self._limiter.wait(host)
        async with self._sem:
            try:
                return await self.fetch(session, url)
            except Exception as e:  # noqa: BLE001
                log.warning("fetch failed for %s: %s", url, e)
                return None

    async def run(self, limit: int | None = None) -> int:
        try:
            import aiohttp  # type: ignore
        except ImportError as e:
            raise ImportError("aiohttp required. pip install aiohttp") from e

        self._robots = _RobotsCache(self.config.user_agent)
        headers = {"User-Agent": self.config.user_agent}
        count = 0
        async with aiohttp.ClientSession(headers=headers) as session:
            tasks: list[asyncio.Task] = []
            async for url in self.iter_tasks():
                if limit is not None and count >= limit:
                    break
                tasks.append(asyncio.create_task(self._run_one(session, url)))
                count += 1
                if len(tasks) >= self.config.max_concurrent * 4:
                    await self._drain(tasks)
            await self._drain(tasks)
        return count

    async def _drain(self, tasks: list[asyncio.Task]) -> None:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        tasks.clear()
        with self._manifest.open("a", encoding="utf-8") as f:
            for r in results:
                if isinstance(r, ScrapeResult):
                    f.write(json.dumps(asdict(r), ensure_ascii=False, default=str) + "\n")

    # --- helpers for subclasses -----------------------------------------

    def _slug(self, url: str) -> str:
        return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def run_scrapers(scrapers: Iterable[AsyncScraper], limit: int | None = None) -> dict[str, int]:
    """Utility to run multiple scrapers sequentially."""
    import asyncio as _asyncio

    totals: dict[str, int] = {}
    for sc in scrapers:
        totals[sc.source_name] = _asyncio.run(sc.run(limit=limit))
    return totals
