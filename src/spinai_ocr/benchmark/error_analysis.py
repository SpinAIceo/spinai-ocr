"""Detailed error analysis tools.

    * per-character accuracy
    * confusion matrix (top-K confused ref→hyp pairs)
    * CER bucketed by text length, font, language, image brightness
    * deletion/insertion/substitution rates via edit-op alignment
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from spinai_ocr.benchmark.metrics import _levenshtein_seq


def _edit_ops(hyp: str, ref: str) -> list[tuple[str, str, str]]:
    """Return a list of (op, ref_char, hyp_char) tuples using
    standard Levenshtein backtrace."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return [("ins", "", c) for c in hyp]
    if m == 0:
        return [("del", c, "") for c in ref]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            ops.append(("eq", ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
        else:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
    return list(reversed(ops))


@dataclass
class ErrorReport:
    n_pairs: int = 0
    total_chars: int = 0
    subs: int = 0
    ins: int = 0
    dels: int = 0
    per_char_acc: dict = field(default_factory=dict)   # char → {correct, total}
    top_sub_pairs: list = field(default_factory=list)  # [(ref, hyp, count)]
    top_ins_chars: list = field(default_factory=list)  # [(char, count)]
    top_del_chars: list = field(default_factory=list)
    by_length_bucket: dict = field(default_factory=dict)
    by_group: dict = field(default_factory=dict)

    @property
    def cer(self) -> float:
        return (self.subs + self.ins + self.dels) / max(self.total_chars, 1)

    def to_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "total_chars": self.total_chars,
            "cer": self.cer,
            "subs": self.subs,
            "ins": self.ins,
            "dels": self.dels,
            "sub_rate": self.subs / max(self.total_chars, 1),
            "ins_rate": self.ins / max(self.total_chars, 1),
            "del_rate": self.dels / max(self.total_chars, 1),
            "per_char_accuracy": {
                c: {"acc": v["correct"] / max(v["total"], 1), **v}
                for c, v in self.per_char_acc.items()
            },
            "top_confusions": self.top_sub_pairs,
            "top_insertions": self.top_ins_chars,
            "top_deletions": self.top_del_chars,
            "by_length_bucket": self.by_length_bucket,
            "by_group": self.by_group,
        }


def _bucket(n: int) -> str:
    if n <= 5:
        return "short(≤5)"
    if n <= 15:
        return "medium(6-15)"
    if n <= 40:
        return "long(16-40)"
    return "verylong(>40)"


def analyze(
    hyps: list[str],
    refs: list[str],
    groups: list[str] | None = None,
    top_k: int = 30,
) -> ErrorReport:
    """Build a rich ErrorReport. `groups` gives an optional grouping label
    per (hyp, ref) — e.g. font name, source document — used for per-group CER."""
    rep = ErrorReport()
    sub_counter: Counter = Counter()
    ins_counter: Counter = Counter()
    del_counter: Counter = Counter()
    per_char: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    by_len: dict[str, dict] = defaultdict(lambda: {"n": 0, "edits": 0, "chars": 0})
    by_grp: dict[str, dict] = defaultdict(lambda: {"n": 0, "edits": 0, "chars": 0})

    for i, (h, r) in enumerate(zip(hyps, refs)):
        rep.n_pairs += 1
        rep.total_chars += len(r)
        edits = _levenshtein_seq(h, r)
        ops = _edit_ops(h, r)
        for op, rc, hc in ops:
            if op == "eq":
                per_char[rc]["correct"] += 1
                per_char[rc]["total"] += 1
            elif op == "sub":
                sub_counter[(rc, hc)] += 1
                rep.subs += 1
                per_char[rc]["total"] += 1
            elif op == "ins":
                ins_counter[hc] += 1
                rep.ins += 1
            elif op == "del":
                del_counter[rc] += 1
                rep.dels += 1
                per_char[rc]["total"] += 1

        b = _bucket(len(r))
        by_len[b]["n"] += 1
        by_len[b]["edits"] += edits
        by_len[b]["chars"] += max(len(r), 1)
        if groups is not None and i < len(groups):
            g = groups[i]
            by_grp[g]["n"] += 1
            by_grp[g]["edits"] += edits
            by_grp[g]["chars"] += max(len(r), 1)

    for b in by_len:
        by_len[b]["cer"] = by_len[b]["edits"] / max(by_len[b]["chars"], 1)
    for g in by_grp:
        by_grp[g]["cer"] = by_grp[g]["edits"] / max(by_grp[g]["chars"], 1)

    rep.per_char_acc = dict(per_char)
    rep.top_sub_pairs = [
        {"ref": rc, "hyp": hc, "count": n}
        for (rc, hc), n in sub_counter.most_common(top_k)
    ]
    rep.top_ins_chars = [{"char": c, "count": n} for c, n in ins_counter.most_common(top_k)]
    rep.top_del_chars = [{"char": c, "count": n} for c, n in del_counter.most_common(top_k)]
    rep.by_length_bucket = dict(by_len)
    rep.by_group = dict(by_grp)
    return rep
