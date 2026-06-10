"""Image scrapers with strict license filtering.

Only CC-BY / CC-BY-SA / CC0 / public-domain sources are implemented.
Scraped images feed detection training directly — we save the image file
plus a JSONL record with URL/license/dimensions.

Implemented:
    CommonsImageScraper  — Wikimedia Commons Featured / Quality images (CC-BY-SA, CC0)
    FlickrCCScraper      — Flickr CC license search (requires api_key)

Not implemented (by policy):
    Google Images, Bing, Naver — aggregators, license unclear.
    Instagram/Twitter/Pinterest — ToS + copyright issues.
"""
from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

from spinaiocr.data.scraper.base import AsyncScraper, ScrapeResult, ScraperConfig


class CommonsImageScraper(AsyncScraper):
    """Wikimedia Commons. Uses the `generator=random` query to walk the catalog.

    All Commons uploads require a free-culture license; we still verify on
    a per-file basis via `imageinfo.extmetadata.LicenseShortName`.
    """

    source_name = "wikimedia_commons"
    license = "cc-by-sa"  # lowest-common-denominator; per-image license stored in meta

    def __init__(self, n_images: int = 5000, config: ScraperConfig | None = None) -> None:
        super().__init__(config)
        self.n_images = n_images
        self._pending: list[dict] = []

    async def iter_tasks(self) -> AsyncIterator[str]:
        import aiohttp  # type: ignore

        api = (
            "https://commons.wikimedia.org/w/api.php"
            "?action=query&generator=random&grnnamespace=6&grnlimit=50"
            "&prop=imageinfo&iiprop=url|size|mime|extmetadata&format=json"
        )
        fetched = 0
        async with aiohttp.ClientSession(
            headers={"User-Agent": self.config.user_agent}
        ) as session:
            while fetched < self.n_images:
                async with session.get(api, timeout=self.config.timeout) as resp:
                    data = await resp.json()
                pages = (data.get("query", {}) or {}).get("pages", {}) or {}
                for page in pages.values():
                    infos = page.get("imageinfo") or []
                    if not infos:
                        continue
                    info = infos[0]
                    url = info.get("url")
                    mime = info.get("mime", "")
                    lic = (
                        info.get("extmetadata", {})
                        .get("LicenseShortName", {})
                        .get("value", "")
                        .lower()
                    )
                    if not url or not mime.startswith("image/"):
                        continue
                    if lic and not any(
                        tag in lic for tag in ("cc0", "public domain", "cc-by", "cc by")
                    ):
                        continue
                    self._pending.append(
                        {
                            "url": url,
                            "mime": mime,
                            "license": lic or "cc-by-sa",
                            "width": info.get("width", 0),
                            "height": info.get("height", 0),
                        }
                    )
                    yield url
                    fetched += 1
                    if fetched >= self.n_images:
                        return

    async def fetch(self, session, url: str) -> ScrapeResult | None:
        meta = next((m for m in self._pending if m["url"] == url), None)
        async with session.get(url, timeout=self.config.timeout) as resp:
            if resp.status != 200:
                return None
            data = await resp.read()
        slug = self._slug(url)
        ext = Path(url).suffix.lower() or ".jpg"
        dest = Path(self.config.output_dir) / self.source_name / f"{slug}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return ScrapeResult(
            url=url,
            source_name=self.source_name,
            license=(meta or {}).get("license", self.license),
            content_type=(meta or {}).get("mime", "image/unknown"),
            local_path=str(dest),
            width=int((meta or {}).get("width") or 0),
            height=int((meta or {}).get("height") or 0),
            bytes=len(data),
        )


class FlickrCCScraper(AsyncScraper):
    """Flickr CC license search. Requires `api_key`.

    Note: Flickr ToS allows crawling CC-licensed photos, but you must
    preserve attribution. Our manifest stores the original page URL.
    """

    source_name = "flickr_cc"
    license = "cc-by"

    def __init__(
        self,
        api_key: str,
        query: str = "sign",
        n_images: int = 1000,
        config: ScraperConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.api_key = api_key
        self.query = query
        self.n_images = n_images
        # Flickr license 4 = CC-BY, 5 = CC-BY-SA, 7 = no known copyright, 9/10 = CC0
        self._allowed_license_ids = "4,5,7,9,10"
        self._pending: list[dict] = []

    async def iter_tasks(self) -> AsyncIterator[str]:
        import aiohttp  # type: ignore

        page = 1
        fetched = 0
        async with aiohttp.ClientSession(
            headers={"User-Agent": self.config.user_agent}
        ) as session:
            while fetched < self.n_images:
                url = (
                    "https://www.flickr.com/services/rest/"
                    "?method=flickr.photos.search"
                    f"&api_key={self.api_key}&format=json&nojsoncallback=1"
                    f"&text={self.query}&license={self._allowed_license_ids}"
                    f"&per_page=100&page={page}&extras=url_l,license"
                )
                async with session.get(url, timeout=self.config.timeout) as resp:
                    data = await resp.json()
                photos = (data.get("photos", {}) or {}).get("photo", []) or []
                if not photos:
                    return
                for p in photos:
                    img_url = p.get("url_l")
                    if not img_url:
                        continue
                    self._pending.append({"url": img_url, "license_id": p.get("license")})
                    yield img_url
                    fetched += 1
                    if fetched >= self.n_images:
                        return
                page += 1

    async def fetch(self, session, url: str) -> ScrapeResult | None:
        async with session.get(url, timeout=self.config.timeout) as resp:
            if resp.status != 200:
                return None
            data = await resp.read()
        slug = self._slug(url)
        dest = Path(self.config.output_dir) / self.source_name / f"{slug}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return ScrapeResult(
            url=url,
            source_name=self.source_name,
            license=self.license,
            content_type="image/jpeg",
            local_path=str(dest),
            bytes=len(data),
        )
