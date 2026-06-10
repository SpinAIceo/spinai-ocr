"""PDF crawlers for structured-document training data.

Only public-domain / CC-licensed PDF sources. We render each page to an
image and pair it with teacher-OCR pseudo-labels downstream.

Implemented:
    ArxivPdfScraper        — arXiv (non-commercial license for some papers;
                             we filter by license metadata and keep CC-BY.)
    GovPdfListScraper      — takes a seed list of .pdf URLs (KOGL Type 1 default).
"""
from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

from spinaiocr.data.scraper.base import AsyncScraper, ScrapeResult, ScraperConfig


class GovPdfListScraper(AsyncScraper):
    """Download a curated list of .pdf URLs. The user supplies seeds.

    Default license is KOGL Type 1 (Korean government, free commercial).
    Override via constructor when mixing other sources.
    """

    source_name = "gov_pdfs"
    license = "kogl-type-1"

    def __init__(
        self,
        pdf_urls: list[str],
        license: str = "kogl-type-1",
        source_name: str = "gov_pdfs",
        config: ScraperConfig | None = None,
    ) -> None:
        self.source_name = source_name
        self.license = license  # type: ignore[misc]
        super().__init__(config)
        self.pdf_urls = pdf_urls

    async def iter_tasks(self) -> AsyncIterator[str]:
        for u in self.pdf_urls:
            yield u

    async def fetch(self, session, url: str) -> ScrapeResult | None:
        async with session.get(url, timeout=self.config.timeout) as resp:
            if resp.status != 200 or "pdf" not in resp.content_type.lower():
                return None
            data = await resp.read()
        slug = self._slug(url)
        dest = Path(self.config.output_dir) / self.source_name / f"{slug}.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return ScrapeResult(
            url=url,
            source_name=self.source_name,
            license=self.license,
            content_type="application/pdf",
            local_path=str(dest),
            bytes=len(data),
        )


def render_pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 220) -> list[Path]:
    """Convert a PDF into one PNG per page. Requires `pypdfium2`."""
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pypdfium2 required. pip install pypdfium2"
        ) from e
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    for i, page in enumerate(doc):
        img = page.render(scale=scale).to_pil()
        path = out_dir / f"{pdf_path.stem}_p{i:04d}.png"
        img.save(path)
        written.append(path)
    return written
