"""CER / WER / bag-F1 metrics.

Built-in Levenshtein implementation; no external metric libraries are
imported. iter 139: removed the false "Uses jiwer when available"
docstring claim — there has never been a jiwer code path here.

Unicode normalization (iter 139): every metric accepts an optional
keyword-only `normalize` parameter. Default `None` preserves byte-exact
prior behavior — every number reported across iter 67-138 was computed
without normalization. Pass `normalize="NFC"` (or `"NFKC"`/`"NFD"`/
`"NFKD"`) when comparing against external benchmarks where the GT may
be in a different Hangul normalization form than our model output.

Our `vocab.base.build_ko_en_vocab()` emits precomposed syllables (NFC)
by construction, so internal benchmarks are NFC-internal-consistent
without normalization. The opt-in matters for real-world fixtures.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable

# Allowed forms per unicodedata.normalize. Validated up-front so a typo
# fails loudly instead of silently degrading every per-row metric.
_NORM_FORMS = {"NFC", "NFKC", "NFD", "NFKD"}


def _maybe_norm(s: str, form: str | None) -> str:
    if form is None:
        return s
    return unicodedata.normalize(form, s)


def _levenshtein_seq(a, b) -> int:
    """Levenshtein distance over any sequence whose items support `==`."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr[0] = i
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[len(b)]


def _levenshtein(a: str, b: str) -> int:
    return _levenshtein_seq(a, b)


@dataclass
class MetricResult:
    name: str
    value: float
    n: int

    def __repr__(self) -> str:
        return f"{self.name}={self.value:.4f} (n={self.n})"


def _validate_norm(form: str | None) -> None:
    if form is not None and form not in _NORM_FORMS:
        raise ValueError(
            f"normalize must be one of {sorted(_NORM_FORMS)} or None; got {form!r}"
        )


def compute_cer(
    hyps: Iterable[str],
    refs: Iterable[str],
    *,
    normalize: str | None = None,
) -> MetricResult:
    _validate_norm(normalize)
    total_edits = 0
    total_chars = 0
    n = 0
    for h, r in zip(hyps, refs):
        h = _maybe_norm(h, normalize)
        r = _maybe_norm(r, normalize)
        total_edits += _levenshtein(h, r)
        total_chars += max(len(r), 1)
        n += 1
    return MetricResult(name="CER", value=total_edits / max(total_chars, 1), n=n)


def compute_wer(
    hyps: Iterable[str],
    refs: Iterable[str],
    *,
    normalize: str | None = None,
) -> MetricResult:
    _validate_norm(normalize)
    total_edits = 0
    total_words = 0
    n = 0
    for h, r in zip(hyps, refs):
        h = _maybe_norm(h, normalize)
        r = _maybe_norm(r, normalize)
        h_tokens = h.split()
        r_tokens = r.split()
        total_edits += _levenshtein_seq(h_tokens, r_tokens)
        total_words += max(len(r_tokens), 1)
        n += 1
    return MetricResult(name="WER", value=total_edits / max(total_words, 1), n=n)


def _bag_f1(hyp_items: list, ref_items: list) -> float:
    """Multiset (bag) F1 between two sequences — order-invariant.

    TP = min(count_hyp(x), count_ref(x)) summed over all x
    FP = sum(count_hyp) − TP
    FN = sum(count_ref) − TP
    """
    from collections import Counter

    hc, rc = Counter(hyp_items), Counter(ref_items)
    tp = sum((hc & rc).values())
    fp = sum(hc.values()) - tp
    fn = sum(rc.values()) - tp
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0  # both empty
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_bag_f1_word(
    hyps: Iterable[str],
    refs: Iterable[str],
    *,
    normalize: str | None = None,
) -> MetricResult:
    """Average bag-of-words F1 across (hyp, ref) pairs. Order-invariant —
    better than CER/WER for multi-line page OCR where detection order is
    arbitrary."""
    _validate_norm(normalize)
    n = 0
    total = 0.0
    for h, r in zip(hyps, refs):
        h = _maybe_norm(h, normalize)
        r = _maybe_norm(r, normalize)
        total += _bag_f1(h.split(), r.split())
        n += 1
    return MetricResult(name="bag_word_F1", value=total / max(n, 1), n=n)


def compute_bag_f1_char(
    hyps: Iterable[str],
    refs: Iterable[str],
    *,
    normalize: str | None = None,
) -> MetricResult:
    """Same as word bag-F1 but on individual characters. Ignores spaces."""
    _validate_norm(normalize)
    n = 0
    total = 0.0
    for h, r in zip(hyps, refs):
        h = _maybe_norm(h, normalize)
        r = _maybe_norm(r, normalize)
        total += _bag_f1(
            [c for c in h if not c.isspace()],
            [c for c in r if not c.isspace()],
        )
        n += 1
    return MetricResult(name="bag_char_F1", value=total / max(n, 1), n=n)
