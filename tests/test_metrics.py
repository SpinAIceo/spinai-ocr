import unicodedata

import pytest

from spinaiocr.benchmark.metrics import (
    compute_bag_f1_char,
    compute_bag_f1_word,
    compute_cer,
    compute_wer,
)


def test_cer_perfect():
    r = compute_cer(["안녕하세요"], ["안녕하세요"])
    assert r.value == 0.0


def test_cer_one_sub():
    r = compute_cer(["안녕하세오"], ["안녕하세요"])
    assert r.value == 1 / 5


def test_wer_word_level():
    r = compute_wer(["hello world foo"], ["hello world bar"])
    assert r.value == 1 / 3


# iter 139: optional Unicode normalization. Default None preserves the
# byte-exact behavior every iter 67-138 number was computed under.

def test_cer_default_treats_nfc_and_nfd_as_different():
    """Without normalize, NFC '가' (U+AC00, 1 char) vs NFD '가' (2 chars)
    differ — Levenshtein on raw codepoints sees them as completely
    different strings. This pins prior behavior."""
    nfc = "가"
    nfd = unicodedata.normalize("NFD", nfc)
    assert len(nfc) == 1
    assert len(nfd) == 2
    r = compute_cer([nfd], [nfc])
    # 1 char ref, 2 chars hyp → 2 substitutions/insertions → 2 edits / 1 char
    assert r.value > 0.0


def test_cer_normalize_nfc_collapses_hangul():
    """With normalize='NFC', the same visual string in either form
    matches exactly."""
    nfc = "안녕하세요"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd
    r = compute_cer([nfd], [nfc], normalize="NFC")
    assert r.value == 0.0


def test_wer_normalize_applies():
    nfc = "안녕 세계"
    nfd = unicodedata.normalize("NFD", nfc)
    r_default = compute_wer([nfd], [nfc])
    r_nfc = compute_wer([nfd], [nfc], normalize="NFC")
    # Both ref and hyp tokenize to 2 words. Default: tokens differ char-wise
    # but `_levenshtein_seq` does sequence-level equality → NFC '안녕' vs
    # NFD '안녕' as full strings are unequal → 2 edits / 2 words = 1.0.
    assert r_default.value == 1.0
    assert r_nfc.value == 0.0


def test_bag_f1_normalize_applies():
    nfc = "안녕 세계"
    nfd = unicodedata.normalize("NFD", nfc)
    r_default = compute_bag_f1_word([nfd], [nfc])
    r_nfc = compute_bag_f1_word([nfd], [nfc], normalize="NFC")
    assert r_default.value == 0.0
    assert r_nfc.value == 1.0
    # char bag F1 also normalizes (and ignores spaces)
    r_char = compute_bag_f1_char([nfd], [nfc], normalize="NFC")
    assert r_char.value == 1.0


def test_invalid_normalize_raises():
    with pytest.raises(ValueError):
        compute_cer(["a"], ["b"], normalize="bogus")
    with pytest.raises(ValueError):
        compute_wer(["a"], ["b"], normalize="utf8")
