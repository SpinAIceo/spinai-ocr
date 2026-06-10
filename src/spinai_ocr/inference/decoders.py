"""CTC decoders: greedy, beam search, beam + character bigram LM.

Why not use an external library? Small, deterministic, no extra deps, easy
to tweak for Korean (space handling, CJK bigrams).

All decoders accept `logits: [T, V]` (single sample) and return a string.
A batch helper runs any decoder per-row.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from spinai_ocr.vocab.base import JamoVocab, Vocab, _reassemble_jamo


# ---------------------------------------------------------------------------
# Greedy (for parity)
# ---------------------------------------------------------------------------


def ctc_greedy(logits: np.ndarray, vocab: Vocab, blank_penalty: float = 0.0) -> str:
    if blank_penalty != 0.0:
        logits = logits.copy()
        logits[..., vocab.blank_id] -= blank_penalty
    ids = logits.argmax(-1)
    return vocab.decode(ids, ctc_collapse=True)


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


@dataclass
class BeamConfig:
    beam_width: int = 16
    prune_threshold: float = 1e-3  # drop hypotheses with prob below this
    blank_id: int | None = None  # defaults to vocab.blank_id
    topk_per_step: int = 0  # 0 = consider all V classes; >0 keeps only the
                            # top-K highest-logit classes per time step. Huge
                            # speedup when V >> beam_width (e.g. 11K vocab).


def _log_sum_exp(a: float, b: float) -> float:
    # log1p(exp(-d)) beats log(exp(0)+exp(-d)) by skipping one exp call.
    # ~2x faster per call; hot path has 2M+ calls per inference.
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_beam_search(
    logits: np.ndarray,  # [T, V]
    vocab: Vocab,
    cfg: BeamConfig | None = None,
    lm_fn=None,       # (prefix_str, next_char) -> log_prob bonus
    lm_alpha: float = 0.3,
    lm_beta: float = 1.0,  # length reward
    is_log_probs: bool = False,  # skip internal log-softmax if caller already did it
    blank_penalty: float = 0.0,  # iter 53: subtract from blank logit pre-softmax
) -> str:
    """CTC prefix beam search in log-space.

    Based on Graves & Jaitly (ICML 2014) with optional LM fusion:
        score = acoustic_log_prob + alpha * lm_log_prob + beta * len(prefix)
    """
    cfg = cfg or BeamConfig()
    blank = cfg.blank_id if cfg.blank_id is not None else vocab.blank_id
    if blank_penalty != 0.0:
        # Subtract penalty from blank logit. Renormalize when caller passes
        # log_probs — beam_lm calibration of `lm_alpha` assumes properly
        # normalized acoustic log_probs; skipping the renorm distorts the
        # acoustic vs LM weight ratio (verified iter 53: skipping renorm
        # gave mean 0.3337 vs renormed 0.3184 on n=262).
        if is_log_probs:
            adj = logits.copy()
            adj[..., blank] -= blank_penalty
            m = adj.max(axis=-1, keepdims=True)
            logits = (adj - m) - np.log(np.exp(adj - m).sum(axis=-1, keepdims=True))
        else:
            logits = logits.copy()
            logits[..., blank] -= blank_penalty
    T, V = logits.shape
    if is_log_probs:
        log_probs = logits
    else:
        # Numerically stable log-softmax.
        m = logits.max(axis=-1, keepdims=True)
        log_probs = (logits - m) - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))

    # Each beam maps a prefix string to (log_prob_ending_in_blank, log_prob_ending_in_non_blank)
    beams: dict[str, tuple[float, float]] = {"": (0.0, -math.inf)}

    # Precompute per-step candidate classes if topk_per_step is set.
    # Lists beat sets for pure iteration — hash walk is pure overhead here.
    if cfg.topk_per_step > 0 and cfg.topk_per_step < V:
        # Partition on logits directly (top-K is the LAST K after partition
        # at index V-K) — avoids the O(T*V) negation copy we had before.
        K = cfg.topk_per_step
        per_step_idx = np.argpartition(logits, V - K, axis=-1)[:, V - K:]
        per_step_lists: list[list[int]] | None = []
        for row in per_step_idx:
            ids = row.tolist()
            if blank not in ids:
                ids.append(blank)
            per_step_lists.append(ids)
    else:
        per_step_lists = None

    neg_inf_pair = (-math.inf, -math.inf)
    prune_cutoff = math.log(cfg.prune_threshold) - 20
    skip_tokens = {vocab.sos_token, vocab.eos_token, vocab.pad_token}
    itoc = vocab._itoc  # noqa: SLF001
    lse = _log_sum_exp  # local bind (attribute lookup shaves cycles in hot loop)
    use_lm = lm_fn is not None
    for t in range(T):
        new_beams: dict[str, tuple[float, float]] = {}
        candidate_ids = per_step_lists[t] if per_step_lists is not None else range(V)
        log_probs_t = log_probs[t]
        # Precompute blank prob + non-blank (char, cp) pairs ONCE per step,
        # reused across every beam. Previously each lookup into `itoc` ran
        # per-beam × per-candidate (240× per step for bw=8, top=30). Now 30×.
        blank_cp = float(log_probs_t[blank])
        chars_cps: list[tuple[str, float]] = []
        for c_id in candidate_ids:
            if c_id == blank:
                continue
            ch = itoc.get(int(c_id), "")
            if not ch or ch in skip_tokens:
                continue
            chars_cps.append((ch, float(log_probs_t[c_id])))

        # Precompute the LM bonus table once per step, keyed by
        # (last_char_of_prefix, candidate_ch). Unique last_chars across
        # ≤beam_width beams is typically 3–6, so we save (beams × chars)
        # function calls per step by doing (unique_last_chars × chars) instead.
        # Cache population + lookup is strictly faster than repeated calls.
        lm_table: dict[tuple[str, str], float] = {}
        if use_lm:
            last_chars_seen: set[str] = set()
            for prefix in beams:
                last_chars_seen.add(prefix[-1] if prefix else "")
            for lc in last_chars_seen:
                # logprob expects a prefix, use the single-char form
                pseudo_prefix = lc  # empty string → LM uses <bos>
                for ch, _ in chars_cps:
                    lm_table[(lc, ch)] = lm_alpha * lm_fn(pseudo_prefix, ch) + lm_beta

        for prefix, (pb, pnb) in beams.items():
            prefix_prob = lse(pb, pnb)
            if prefix_prob < prune_cutoff:
                continue
            last_char = prefix[-1] if prefix else ""
            # blank extension
            new_pb, new_pnb = new_beams.get(prefix, neg_inf_pair)
            new_beams[prefix] = (lse(new_pb, prefix_prob + blank_cp), new_pnb)
            # non-blank extensions
            for ch, cp in chars_cps:
                new_prefix = prefix + ch
                lm_bonus = lm_table[(last_char, ch)] if use_lm else 0.0
                if last_char == ch:
                    # same char: collapse via blank path only
                    new_pb, new_pnb = new_beams.get(new_prefix, neg_inf_pair)
                    new_beams[new_prefix] = (new_pb, lse(new_pnb, pb + cp + lm_bonus))
                    # extend current prefix's non-blank path
                    same_pb, same_pnb = new_beams.get(prefix, neg_inf_pair)
                    new_beams[prefix] = (same_pb, lse(same_pnb, pnb + cp))
                else:
                    new_pb, new_pnb = new_beams.get(new_prefix, neg_inf_pair)
                    new_beams[new_prefix] = (
                        new_pb,
                        lse(new_pnb, prefix_prob + cp + lm_bonus),
                    )

        # prune to top beam_width via heap — O(N log K) vs full sort's O(N log N).
        # iter 84: pre-compute the (score, prefix, pair) tuple so heapq sorts
        # by its first element directly — eliminates the per-comparison
        # `key=lambda` callback (3.2M calls/inference profiled). Profile-
        # measured: -28% on ctc_beam_search self-time, -6% on /ocr p50.
        if new_beams:
            if len(new_beams) <= cfg.beam_width:
                beams = new_beams
            else:
                scored = [(lse(p[0], p[1]), pfx, p) for pfx, p in new_beams.items()]
                top = heapq.nlargest(cfg.beam_width, scored)
                beams = {pfx: pair for _, pfx, pair in top}
        # If every prefix was pruned, keep the previous beam set so we can
        # still return a best hypothesis at the end (common with very peaky
        # distributions like TTA-averaged log-probs).

    # final: pick the prefix with highest total prob
    if not beams:
        return ""
    best = max(beams.items(), key=lambda it: _log_sum_exp(it[1][0], it[1][1]))
    return best[0]


# ---------------------------------------------------------------------------
# Bigram character LM (built from a training corpus)
# ---------------------------------------------------------------------------


class CharBigramLM:
    """Simple add-α smoothed char bigram log-probability lookup."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._counts: dict[tuple[str, str], int] = defaultdict(int)
        self._prefix_counts: dict[str, int] = defaultdict(int)
        self._vocab_size = 0
        self._cache: dict[tuple[str, str], float] = {}

    def fit(self, corpus: list[str]) -> None:
        chars = set()
        for line in corpus:
            prev = "<bos>"
            for ch in line:
                self._counts[(prev, ch)] += 1
                self._prefix_counts[prev] += 1
                chars.add(ch)
                prev = ch
            self._counts[(prev, "<eos>")] += 1
            self._prefix_counts[prev] += 1
        self._vocab_size = len(chars) + 2  # +<bos>, +<eos>
        self._cache.clear()

    def logprob(self, prefix: str, next_char: str) -> float:
        prev = prefix[-1] if prefix else "<bos>"
        key = (prev, next_char)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        num = self._counts.get(key, 0) + self.alpha
        den = self._prefix_counts.get(prev, 0) + self.alpha * self._vocab_size
        val = math.log(num / den)
        self._cache[key] = val
        return val

    __call__ = logprob


class CharTrigramLM:
    """Char trigram with stupid-backoff to bigram then unigram.

    P(c | a, b) = add-α smoothed trigram if (a,b) has been seen,
                  else backoff * bigram P(c | b),
                  else backoff^2 * unigram P(c).

    Stupid-backoff isn't a true probability (doesn't sum to 1) but is the
    standard cheap choice for char-LM decoder fusion where only relative
    ranking between candidates matters (Brants et al. 2007). Add-α on the
    trigram level prevents -inf for any (a, b, c) pair seen even once.
    """

    BACKOFF_DEFAULT = 0.4  # log domain: log(0.4) ≈ -0.916 (Brants 2007)

    def __init__(self, alpha: float = 0.5, backoff: float | None = None) -> None:
        self.alpha = alpha
        self.BACKOFF = backoff if backoff is not None else self.BACKOFF_DEFAULT
        self._tri: dict[tuple[str, str, str], int] = defaultdict(int)
        self._tri_prefix: dict[tuple[str, str], int] = defaultdict(int)
        self._bi: dict[tuple[str, str], int] = defaultdict(int)
        self._bi_prefix: dict[str, int] = defaultdict(int)
        self._uni: dict[str, int] = defaultdict(int)
        self._uni_total = 0
        self._vocab_size = 0
        self._cache: dict[tuple[str, str, str], float] = {}
        self._log_backoff = math.log(self.BACKOFF)

    def fit(self, corpus: list[str]) -> None:
        chars: set[str] = set()
        for line in corpus:
            a, b = "<bos>", "<bos>"
            for ch in line:
                self._tri[(a, b, ch)] += 1
                self._tri_prefix[(a, b)] += 1
                self._bi[(b, ch)] += 1
                self._bi_prefix[b] += 1
                self._uni[ch] += 1
                self._uni_total += 1
                chars.add(ch)
                a, b = b, ch
            self._tri[(a, b, "<eos>")] += 1
            self._tri_prefix[(a, b)] += 1
            self._bi[(b, "<eos>")] += 1
            self._bi_prefix[b] += 1
            self._uni["<eos>"] += 1
            self._uni_total += 1
        self._vocab_size = len(chars) + 2
        self._cache.clear()

    def logprob(self, prefix: str, next_char: str) -> float:
        n = len(prefix)
        a = prefix[-2] if n >= 2 else "<bos>"
        b = prefix[-1] if n >= 1 else "<bos>"
        key = (a, b, next_char)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        tri_prefix_count = self._tri_prefix.get((a, b), 0)
        if tri_prefix_count > 0:
            num = self._tri.get(key, 0) + self.alpha
            den = tri_prefix_count + self.alpha * self._vocab_size
            val = math.log(num / den)
        elif self._bi_prefix.get(b, 0) > 0:
            # backoff: log(BACKOFF) + log P_bigram(c | b)
            num = self._bi.get((b, next_char), 0) + self.alpha
            den = self._bi_prefix[b] + self.alpha * self._vocab_size
            val = self._log_backoff + math.log(num / den)
        else:
            # deepest backoff: unigram
            num = self._uni.get(next_char, 0) + self.alpha
            den = self._uni_total + self.alpha * self._vocab_size
            val = 2 * self._log_backoff + math.log(num / max(den, 1))
        self._cache[key] = val
        return val

    __call__ = logprob


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------


def decode_batch(
    logits_batch: np.ndarray,  # [B, T, V]
    vocab: Vocab,
    *,
    mode: str = "greedy",   # "greedy" | "beam" | "beam_lm"
    beam_cfg: BeamConfig | None = None,
    lm: CharBigramLM | None = None,
    lm_alpha: float = 0.3,
    is_log_probs: bool = False,
    blank_penalty: float = 0.0,
    row_T: list[int] | None = None,
) -> list[str]:
    """Decode a [B, T, V] batch row-by-row.

    iter 104: when ``row_T`` is provided (length B, with row_T[i] ≤ T),
    each row is sliced to its actual T_i before decoding. For SVTR with
    width-padded inputs, padding-region steps after T_i correspond to
    the all-white right-pad region of the crop and the model is trained
    to emit blank there — decoding them is wasted work (and occasionally
    introduces padding-region hallucinations). Slicing skips that work.

    JamoVocab: beam / beam_lm modes return raw U+1100 jamo sequences;
    _reassemble_jamo is applied automatically. beam_lm downgrades to beam
    (no LM) because the syllable-level CharBigramLM cannot score jamo
    transitions meaningfully.
    """
    is_jamo = isinstance(vocab, JamoVocab)
    # beam_lm with jamo vocab: LM scores jamo-pair bigrams which are
    # nonsensical (e.g. ᄒ→ᅡ→ᆫ = single syllable 한). Downgrade to beam.
    effective_mode = "beam" if (is_jamo and mode == "beam_lm") else mode
    results: list[str] = []
    for i, row in enumerate(logits_batch):
        if row_T is not None:
            t_i = int(row_T[i])
            if 0 < t_i < row.shape[0]:
                row = row[:t_i]
        if effective_mode == "greedy":
            results.append(ctc_greedy(row, vocab, blank_penalty=blank_penalty))
        elif effective_mode == "beam":
            raw = ctc_beam_search(row, vocab, cfg=beam_cfg,
                                  is_log_probs=is_log_probs,
                                  blank_penalty=blank_penalty)
            results.append(_reassemble_jamo(list(raw)) if is_jamo else raw)
        elif effective_mode == "beam_lm":
            if lm is None:
                raise ValueError("mode=beam_lm requires a CharBigramLM")
            raw = ctc_beam_search(row, vocab, cfg=beam_cfg,
                                  lm_fn=lm, lm_alpha=lm_alpha,
                                  is_log_probs=is_log_probs,
                                  blank_penalty=blank_penalty)
            results.append(raw)
        else:
            raise ValueError(mode)
    return results
