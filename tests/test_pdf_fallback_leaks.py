"""iter 138 regression: pdf_fallback must close PdfPage / PdfTextPage / PdfBitmap.

Pre-iter-138 the extract() loop iterated `for page in doc:` and called
`page.get_textpage()` + `page.render(...)` per iteration, never closing
either. On a 100-page PDF that leaks 100 page handles + 100 textpage
handles + ~100 bitmap handles until the PdfDocument itself is collected.

We can't test handle counts directly, but pypdfium2's helper objects
expose a `.raw` attribute that is None after `.close()`. We monkeypatch
the close() methods to record calls and assert each helper is closed
exactly once per page.
"""
from __future__ import annotations

import io

import pypdfium2 as pdfium
import pytest

from spinaiocr.inference.pdf_fallback import extract


def _build_blank_pdf(n_pages: int = 3) -> bytes:
    doc = pdfium.PdfDocument.new()
    for _ in range(n_pages):
        doc.new_page(200, 300)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _ocr_stub(pil):  # noqa: ARG001
    class _R:
        text = "stubbed"
    return _R()


def test_extract_closes_page_textpage_bitmap_per_iteration(tmp_path, monkeypatch):
    pdf_bytes = _build_blank_pdf(n_pages=3)
    pdf_path = tmp_path / "blank.pdf"
    pdf_path.write_bytes(pdf_bytes)

    page_closes: list[int] = []
    textpage_closes: list[int] = []
    bitmap_closes: list[int] = []

    orig_page_close = pdfium.PdfPage.close
    orig_textpage_close = pdfium.PdfTextPage.close
    orig_bitmap_close = pdfium.PdfBitmap.close

    def page_close(self):
        page_closes.append(id(self))
        return orig_page_close(self)

    def textpage_close(self):
        textpage_closes.append(id(self))
        return orig_textpage_close(self)

    def bitmap_close(self):
        bitmap_closes.append(id(self))
        return orig_bitmap_close(self)

    monkeypatch.setattr(pdfium.PdfPage, "close", page_close)
    monkeypatch.setattr(pdfium.PdfTextPage, "close", textpage_close)
    monkeypatch.setattr(pdfium.PdfBitmap, "close", bitmap_close)

    results = extract(pdf_path, _ocr_stub, text_coverage_threshold=0.1, dpi=72)

    assert len(results) == 3
    # Blank pages have no text → all 3 take the OCR fallback path → all 3
    # render a bitmap. Page + textpage close once per page.
    assert len(page_closes) == 3
    assert len(textpage_closes) == 3
    assert len(bitmap_closes) == 3
    # All must be distinct objects (no double-close on the same handle).
    assert len(set(page_closes)) == 3
    assert len(set(textpage_closes)) == 3
    assert len(set(bitmap_closes)) == 3


def test_extract_clamps_coverage_ratio(tmp_path):
    """The clamped coverage_ratio doesn't change classification at the
    default threshold, but pre-iter-138 the ratio could exceed 1.0 on
    dense pages with overlapping rect runs. Smoke-test the boundary by
    setting threshold=0.05 on a blank page (coverage=0): we should NOT
    accept the empty-text path.
    """
    pdf_bytes = _build_blank_pdf(n_pages=1)
    pdf_path = tmp_path / "blank.pdf"
    pdf_path.write_bytes(pdf_bytes)

    results = extract(pdf_path, _ocr_stub, text_coverage_threshold=0.05, dpi=72)
    assert len(results) == 1
    # Blank page: text="" or coverage<0.05 → must take OCR path.
    assert results[0].source == "ocr"
    assert results[0].text == "stubbed"
