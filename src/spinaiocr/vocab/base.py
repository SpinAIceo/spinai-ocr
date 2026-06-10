"""Vocabulary primitives for recognition head.

Design notes:
- CTC-style: index 0 reserved for <blank>.
- Attention-style: extra <sos>, <eos>, <pad> tokens appended.
- Korean handling: we use precomposed syllables (AC00–D7A3) by default.
  Jamo decomposition is supported as an alternative mode via `jamo=True`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

VOCAB_DIR = Path(__file__).parent / "files"


@dataclass
class Vocab:
    name: str
    chars: list[str]
    blank_token: str = "<blank>"
    unk_token: str = "<unk>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"
    pad_token: str = "<pad>"
    _ctoi: dict[str, int] = field(init=False, repr=False)
    _itoc: dict[int, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        specials = [
            self.blank_token,
            self.unk_token,
            self.sos_token,
            self.eos_token,
            self.pad_token,
        ]
        # deduplicate while preserving order
        seen = set()
        ordered = []
        for tok in specials + list(self.chars):
            if tok not in seen:
                seen.add(tok)
                ordered.append(tok)
        self._ctoi = {c: i for i, c in enumerate(ordered)}
        self._itoc = {i: c for c, i in self._ctoi.items()}

    def __len__(self) -> int:
        return len(self._ctoi)

    @property
    def size(self) -> int:
        return len(self)

    @property
    def blank_id(self) -> int:
        return self._ctoi[self.blank_token]

    @property
    def unk_id(self) -> int:
        return self._ctoi[self.unk_token]

    def encode(self, text: str) -> list[int]:
        return [self._ctoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: Iterable[int], ctc_collapse: bool = True) -> str:
        chars = []
        prev = -1
        for i in ids:
            if ctc_collapse:
                if i == prev:
                    continue
                prev = i
                if i == self.blank_id:
                    continue
            ch = self._itoc.get(i, "")
            if ch in {self.sos_token, self.eos_token, self.pad_token, self.blank_token}:
                continue
            chars.append(ch)
        return "".join(chars)

    def decode_with_unk_stats(self, ids: Iterable[int],
                               ctc_collapse: bool = True) -> tuple[str, int, list[int]]:
        """Like decode, but also returns the count of <unk> tokens and
        their positions in the output string. Medical B2B needs this for
        _53 Tier-2 reject/queue policy — silent <unk> replacement is a
        safety risk when the text is a drug dosage or NDC.
        """
        chars = []
        unk_positions: list[int] = []
        prev = -1
        unk = self.unk_token
        for i in ids:
            if ctc_collapse:
                if i == prev:
                    continue
                prev = i
                if i == self.blank_id:
                    continue
            ch = self._itoc.get(i, "")
            if ch in {self.sos_token, self.eos_token, self.pad_token, self.blank_token}:
                continue
            if ch == unk:
                unk_positions.append(len(chars))
            chars.append(ch)
        text = "".join(chars)
        return text, len(unk_positions), unk_positions

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.chars), encoding="utf-8")


def _korean_syllables() -> list[str]:
    # AC00–D7A3: 11,172 precomposed Hangul syllables
    return [chr(cp) for cp in range(0xAC00, 0xD7A4)]


def _ascii_printable() -> list[str]:
    # ASCII printable without space handled separately
    return [chr(c) for c in range(0x20, 0x7F)]


def build_ko_en_vocab() -> Vocab:
    chars: list[str] = []
    chars.extend(_ascii_printable())
    chars.extend(_korean_syllables())
    # common Korean punctuation / symbols
    extra = list("·…※←→↑↓─━│┃◆◇○●◎■□★☆♠♣♥♦€£¥₩°±×÷≤≥≠≈∞")
    for ch in extra:
        if ch not in chars:
            chars.append(ch)
    return Vocab(name="ko_en_v1", chars=chars)


def _hiragana() -> list[str]:
    return [chr(cp) for cp in range(0x3040, 0x30A0)]


def _katakana() -> list[str]:
    return [chr(cp) for cp in range(0x30A0, 0x3100)]


def _cjk_common() -> list[str]:
    """Top ~3500 most common CJK characters (covers >99% of modern Korean/
    Japanese/Simplified Chinese printed text). Backed by the Unicode
    Basic Multilingual Plane Common set used by MMOCR. We use a conservative
    3500-char subset of CJK Unified Ideographs (U+4E00–U+9FFF)."""
    return [chr(cp) for cp in range(0x4E00, 0x4E00 + 3500)]


def build_multilang_vocab() -> Vocab:
    """ko + en + ja + zh unified vocab (~18k tokens). Suitable for models
    targeting East-Asian documents with mixed scripts."""
    chars: list[str] = []
    chars.extend(_ascii_printable())
    chars.extend(_korean_syllables())
    chars.extend(_hiragana())
    chars.extend(_katakana())
    chars.extend(_cjk_common())
    # common symbols/punctuation
    extra = list("·…※←→↑↓─━│┃◆◇○●◎■□★☆♠♣♥♦€£¥₩°±×÷≤≥≠≈∞「」『』【】〈〉《》、。，．：；？！")
    for ch in extra:
        if ch not in chars:
            chars.append(ch)
    return Vocab(name="multilang_v1", chars=chars)


def build_ko_en_medical_v1_vocab() -> Vocab:
    """Corpus-pruned vocab for the medical B2B pivot (_52 / _53).

    Built from the top ~2000 most-frequent chars across all training data
    (full_ko + diverse_ko + PillScan + KO docs + korean-pair + synth seeds).
    99.63% coverage of observed text, vs 11,309 in the original ko_en_v1.

    Why: the SVTR-lite head Linear(192 → 11309) = 2.17 M params was 61% of
    the entire model. Pruning to a domain-relevant vocab frees ~1.79 M
    params to reinvest in MixingBlocks (dim 192→320, depth 4→6) — same
    total budget, ~2× capacity in the "brain" that actually recognises
    visual features. See _53.
    """
    assets = (Path(__file__).parent.parent / "assets" / "vocabs"
              / "ko_en_medical_v1.txt")
    if not assets.exists():
        raise FileNotFoundError(
            f"{assets} not built yet. Run:\n"
            "  python - <<'PY'\n"
            "  from collections import Counter; from pathlib import Path\n"
            "  # see scripts for full command\n"
            "  PY"
        )
    chars = [ln for ln in assets.read_text(encoding="utf-8").splitlines() if ln]
    return Vocab(name="ko_en_medical_v1", chars=chars)


_BUILDERS = {
    "ko_en_v1": build_ko_en_vocab,
    "multilang_v1": build_multilang_vocab,
    "ko_en_medical_v1": build_ko_en_medical_v1_vocab,
}


def load_vocab(name: str) -> Vocab:
    if name in _BUILDERS:
        return _BUILDERS[name]()
    if name == "jamo_ko_v1":
        return build_jamo_vocab()
    path = VOCAB_DIR / f"{name}.txt"
    if path.exists():
        chars = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln]
        return Vocab(name=name, chars=chars)
    raise KeyError(f"Unknown vocab: {name}")


# ---------------------------------------------------------------------------
# Jamo vocabulary (SNU 2025 "Jamo Is All You Need")
# ---------------------------------------------------------------------------

# Hangul Jamo block (U+1100–U+11FF): unambiguous 1-char-per-jamo
_CHO  = [chr(c) for c in range(0x1100, 0x1113)]  # 19 choseong
_JUNG = [chr(c) for c in range(0x1161, 0x1176)]  # 21 jungseong
_JONG = [chr(c) for c in range(0x11A8, 0x11C3)]  # 27 jongseong (no empty slot)

# Lookup tables for syllable ↔ jamo conversion
_CHO_IDX  = {c: i for i, c in enumerate(_CHO)}
_JUNG_IDX = {c: i for i, c in enumerate(_JUNG)}
_JONG_IDX = {c: i + 1 for i, c in enumerate(_JONG)}  # 1-based; 0 = no jongseong

_SYLL_BASE = 0xAC00
_N_JUNG = 21
_N_JONG = 28  # 0 = no jong, 1–27 = actual jongseong


def _syllable_to_jamo(ch: str) -> list[str]:
    cp = ord(ch) - _SYLL_BASE
    if not (0 <= cp <= 11171):
        return [ch]  # not a Korean syllable — return as-is
    cho_i  = cp // (_N_JUNG * _N_JONG)
    jung_i = (cp % (_N_JUNG * _N_JONG)) // _N_JONG
    jong_i = cp % _N_JONG
    result = [_CHO[cho_i], _JUNG[jung_i]]
    if jong_i:
        result.append(_JONG[jong_i - 1])
    return result


def _jamo_to_syllable(cho: str, jung: str, jong: str | None) -> str:
    ci = _CHO_IDX.get(cho)
    vi = _JUNG_IDX.get(jung)
    if ci is None or vi is None:
        return cho + jung + (jong or "")
    ji = _JONG_IDX.get(jong, 0) if jong else 0
    return chr(_SYLL_BASE + ci * _N_JUNG * _N_JONG + vi * _N_JONG + ji)


class JamoVocab(Vocab):
    """Korean-aware vocab that encodes syllables as jamo sequences.

    encode('한국') → ids for [ㅎ, ㅏ, ㄴ, ㄱ, ㅜ, ㄱ]  (6 tokens, not 2)
    decode(ids)    → reassembles jamo runs into syllables → '한국'

    Vocab size: 5 specials + 19 cho + 21 jung + 27 jong + ASCII = ~118 chars
    vs. ko_en_medical_v1 ~2005 or ko_en_v1 ~11309.
    """

    def encode(self, text: str) -> list[int]:
        jamo_seq: list[str] = []
        for ch in text:
            if '가' <= ch <= '힣':
                jamo_seq.extend(_syllable_to_jamo(ch))
            else:
                jamo_seq.append(ch)
        return [self._ctoi.get(j, self.unk_id) for j in jamo_seq]

    def decode(self, ids: Iterable[int], ctc_collapse: bool = True) -> str:
        raw: list[str] = []
        prev = -1
        for i in ids:
            if ctc_collapse:
                if i == prev:
                    continue
                prev = i
                if i == self.blank_id:
                    continue
            ch = self._itoc.get(i, "")
            if ch in {self.sos_token, self.eos_token, self.pad_token,
                      self.blank_token, self.unk_token}:
                continue
            raw.append(ch)
        return _reassemble_jamo(raw)


def _reassemble_jamo(tokens: list[str]) -> str:
    """Convert a flat jamo token list back to syllables + passthrough chars.

    State machine: accumulate (cho, jung, optional jong) triples.
    A token is cho  if it's in _CHO_IDX.
    A token is jung if it's in _JUNG_IDX.
    A token is jong if it's in _JONG_IDX.
    """
    result = []
    i = 0
    n = len(tokens)
    while i < n:
        ch = tokens[i]
        if ch not in _CHO_IDX:
            result.append(ch)
            i += 1
            continue
        # we have a choseong — must be followed by jungseong
        if i + 1 < n and tokens[i + 1] in _JUNG_IDX:
            cho  = ch
            jung = tokens[i + 1]
            # check for jongseong: next token in _JONG_IDX AND not followed by jungseong
            if (i + 2 < n and tokens[i + 2] in _JONG_IDX
                    and not (i + 3 < n and tokens[i + 3] in _JUNG_IDX)):
                jong = tokens[i + 2]
                result.append(_jamo_to_syllable(cho, jung, jong))
                i += 3
            else:
                result.append(_jamo_to_syllable(cho, jung, None))
                i += 2
        else:
            # lone choseong (e.g. compat jamo in middle of ASCII text)
            result.append(ch)
            i += 1
    return "".join(result)


def build_jamo_vocab() -> "JamoVocab":
    """67-char jamo vocab + ASCII printable + specials."""
    ascii_chars = [chr(c) for c in range(0x20, 0x7F)]  # printable ASCII
    chars = _CHO + _JUNG + _JONG + ascii_chars
    return JamoVocab(name="jamo_ko_v1", chars=chars)
