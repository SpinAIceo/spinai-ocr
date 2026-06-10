"""PDF text-layer fallback.

Many PDFs already contain a text layer (born-digital or previously OCR'd).
Running OCR on those wastes compute. This module:

    1. Opens a PDF with `pypdfium2`.
    2. For each page, checks whether extractable text covers enough of the
       page (coverage = sum of text-bbox area / page area).
    3. If coverage ≥ `text_coverage_threshold`, returns the extracted text
       directly. Otherwise, renders the page to an image and calls the
       provided OCR callable.

Returns a unified list of (page_number, source, text) tuples.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class PageResult:
    page: int
    source: str  # "text_layer" | "ocr"
    text: str


def extract(
    pdf_path: str | Path,
    ocr_callable: Callable,
    text_coverage_threshold: float = 0.1,
    dpi: int = 220,
) -> list[PageResult]:
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError as e:
        raise ImportError("pypdfium2 required. pip install pypdfium2") from e

    scale = dpi / 72.0
    results: list[PageResult] = []

    # iter 138: PdfDocument supports the context manager; PdfPage / PdfTextPage
    # / PdfBitmap do not, so we close them manually in finally blocks. Without
    # this a 100-page document leaks 100 page + 100 textpage + ~100 bitmap
    # handles until the doc itself is closed (or the GC runs the finalizer,
    # which is non-deterministic and has caused pdfium internal state issues
    # under concurrent access in other projects).
    with pdfium.PdfDocument(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            try:
                text_page = page.get_textpage()
                try:
                    try:
                        text = text_page.get_text_range().strip()
                    except Exception:  # noqa: BLE001
                        text = ""

                    coverage = 0.0
                    page_w, page_h = page.get_size()
                    page_area = max(page_w * page_h, 1.0)
                    try:
                        for r in text_page.get_rects():
                            left, bottom, right, top = r
                            coverage += (right - left) * (top - bottom)
                    except Exception:  # noqa: BLE001
                        coverage = 0.0
                    # iter 138: clamp at 1.0. Overlapping rect runs (multi-line
                    # text where pdfium reports per-run rects that share
                    # vertical bands) sum to > page_area on dense pages.
                    # The 0.1 default threshold makes this practically harmless
                    # but the >1.0 ratio is misleading to anyone logging it.
                    coverage_ratio = min(coverage / page_area, 1.0)
                finally:
                    text_page.close()

                if text and coverage_ratio >= text_coverage_threshold:
                    results.append(
                        PageResult(page=i, source="text_layer", text=text)
                    )
                else:
                    bitmap = page.render(scale=scale)
                    try:
                        pil = bitmap.to_pil()
                    finally:
                        bitmap.close()
                    ocr = ocr_callable(pil)
                    # ocr_callable is expected to return an object with
                    # `.text` or a str.
                    ocr_text = getattr(ocr, "text", None)
                    if ocr_text is None:
                        ocr_text = str(ocr)
                    results.append(PageResult(page=i, source="ocr", text=ocr_text))
            finally:
                page.close()

    return results
