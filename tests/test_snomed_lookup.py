"""Smoke tests for the SNOMED-lite PoC."""
from __future__ import annotations

from spinai_ocr.postprocess.snomed_lookup import SnomedLookup, get_lookup


def test_brand_match_korean():
    lookup = get_lookup()
    m = lookup.resolve("타이레놀 500mg 12정")
    assert m is not None
    assert m.snomed_id == "387517004"
    assert m.generic_en == "acetaminophen"


def test_brand_match_english():
    lookup = get_lookup()
    m = lookup.resolve("Rx Only AMOXICILLIN 500 mg Capsules, USP")
    assert m is not None
    assert m.generic_en == "amoxicillin"


def test_long_brand_wins_over_short():
    lookup = get_lookup()
    m = lookup.resolve("바이엘아스피린 100mg 30정")
    # must match "바이엘아스피린" not just "아스피린"
    assert m is not None
    assert m.generic_en == "acetylsalicylic acid"
    assert "바이엘" in m.brand_matched.lower() or "아스피린" == m.brand_matched


def test_no_match_returns_none():
    lookup = get_lookup()
    assert lookup.resolve("처음보는약품 NewDrugXYZ 99mg") is None


def test_dose_extraction():
    lookup = get_lookup()
    out = lookup.resolve_with_dose("타이레놀 500 mg tablet")
    assert out["extracted_dose"] == 500.0
    assert out["extracted_unit"] == "mg"
    assert out["snomed_id"] == "387517004"


def test_dose_unit_conversion_mcg():
    lookup = get_lookup()
    # levothyroxine is dosed in mcg — check normalisation to mg
    out = lookup.resolve_with_dose("씬지로이드 50 mcg")
    assert out["extracted_dose"] == 0.05
    assert out["extracted_unit"] == "mg"
