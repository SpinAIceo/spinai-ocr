"""Web scrapers for OCR training data.

Subpackages:
    base      — AsyncScraper, robots.txt gate, rate limiter, license filter
    corpus    — text-only scrapers (wikipedia, public RSS, gov open data)
    images    — image scrapers (CC-licensed sources only)
    pdfs      — public PDF crawlers (public-domain / CC only)

Design principles:
    * Respect robots.txt. No bypass, no fake UAs.
    * Rate-limit per host (default 1 req / sec).
    * License whitelist: only CC-BY / CC-BY-SA / CC0 / public-domain / MIT / Apache.
    * Store source URL + license in manifest alongside every item.
"""
from spinaiocr.data.scraper.base import AsyncScraper, ScrapeResult, ScraperConfig

__all__ = ["AsyncScraper", "ScrapeResult", "ScraperConfig"]
