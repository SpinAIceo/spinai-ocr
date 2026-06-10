"""Active-learning review queue.

Given a stream of pseudo-labeled samples, surface the ones most worth a
human QA pass first. We combine two signals:

    1. **Teacher disagreement** — consensus tier (`low` > `mid` > `high`).
    2. **Student uncertainty** — per-line confidence (when available).

Outputs a priority-sorted JSONL + a printable report. A human can then
pull from the top of the queue; accepted labels flow back into the
training set, rejected ones are sidelined.

This is separate from :mod:`spinaiocr.data.mining` which is the fully
automated student-vs-teacher divergence loop. `mining` picks **training**
targets; `active_learning` picks **review** targets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from heapq import nlargest
from pathlib import Path
from typing import Iterable


@dataclass
class ReviewItem:
    image: str
    words: list[dict] = field(default_factory=list)
    priority: float = 0.0
    reasons: list[str] = field(default_factory=list)


def _tier_score(tier: str) -> float:
    return {"high": 0.0, "mid": 0.5, "low": 1.0}.get(tier, 0.7)


def score_pseudo_labels(pseudo_jsonl: Path) -> list[ReviewItem]:
    """Score a consensus pseudo-label file (output of scripts/pseudo_label.py).

    Each line: {"image": ..., "words": [{"tier": "...", "conf": ...}, ...]}
    """
    items: list[ReviewItem] = []
    with pseudo_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            words = rec.get("words", [])
            if not words:
                continue
            # average priority across words; very short docs + disagreement
            # = highest priority
            tier_scores = [_tier_score(w.get("tier", "mid")) for w in words]
            conf_scores = [1.0 - float(w.get("conf", 1.0)) for w in words]
            mean_tier = sum(tier_scores) / len(tier_scores)
            mean_conf = sum(conf_scores) / len(conf_scores)
            priority = 0.6 * mean_tier + 0.4 * mean_conf
            reasons: list[str] = []
            if mean_tier > 0.5:
                reasons.append("low teacher consensus")
            if mean_conf > 0.3:
                reasons.append("high student uncertainty")
            items.append(
                ReviewItem(
                    image=rec["image"],
                    words=words,
                    priority=priority,
                    reasons=reasons,
                )
            )
    return items


def build_review_queue(
    items: Iterable[ReviewItem], top_k: int, out_jsonl: Path
) -> int:
    top = nlargest(top_k, items, key=lambda it: it.priority)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for it in top:
            f.write(
                json.dumps(
                    {
                        "image": it.image,
                        "priority": it.priority,
                        "reasons": it.reasons,
                        "words": it.words,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(top)
