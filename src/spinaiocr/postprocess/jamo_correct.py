# -*- coding: utf-8 -*-
"""Rule-based Korean jamo post-correction for SVTR OCR output.

SVTR-deep occasionally emits:
  1. Spurious spaces between adjacent Korean syllables/jamo
  2. Decomposed compat-jamo runs that can compose into a valid syllable
"""
from __future__ import annotations
import logging
import re

__all__ = ["correct_jamo"]

log = logging.getLogger("spinaiocr.postprocess.jamo")

_SYLLABLE_START = 0xAC00
_SYLLABLE_END   = 0xD7A3
_COMPAT_CHO_START = 0x3131  # chrono compat-jamo start
_COMPAT_JUNG_END  = 0x3163  # compat-jamo end

_COMPAT_TO_CHO = {
    0x3131:0, 0x3132:1, 0x3134:2, 0x3137:3, 0x3138:4, 0x3139:5,
    0x3141:6, 0x3142:7, 0x3143:8, 0x3145:9, 0x3146:10, 0x3147:11,
    0x3148:12, 0x3149:13, 0x314A:14, 0x314B:15, 0x314C:16, 0x314D:17, 0x314E:18,
}
_COMPAT_TO_JUNG = {
    0x314F:0, 0x3150:1, 0x3151:2, 0x3152:3, 0x3153:4, 0x3154:5,
    0x3155:6, 0x3156:7, 0x3157:8, 0x3158:9, 0x3159:10, 0x315A:11,
    0x315B:12, 0x315C:13, 0x315D:14, 0x315E:15, 0x315F:16, 0x3160:17,
    0x3161:18, 0x3162:19, 0x3163:20,
}
_COMPAT_TO_JONG = {
    0x3131:1, 0x3132:2, 0x3133:3, 0x3134:4, 0x3135:5, 0x3136:6,
    0x3137:7, 0x3139:8, 0x313A:9, 0x313B:10, 0x313C:11, 0x313D:12,
    0x313E:13, 0x313F:14, 0x3140:15, 0x3141:16, 0x3142:17, 0x3144:18,
    0x3145:19, 0x3146:20, 0x3147:21, 0x3148:22, 0x314A:23, 0x314B:24,
    0x314C:25, 0x314D:26, 0x314E:27,
}
_N_CHO = 19; _N_JUNG = 21; _N_JONG = 28

_COMPAT_CHO_RANGE  = frozenset(_COMPAT_TO_CHO.keys())
_COMPAT_JUNG_RANGE = frozenset(_COMPAT_TO_JUNG.keys())
_COMPAT_JONG_RANGE = frozenset(_COMPAT_TO_JONG.keys())

# Build regex pattern at import time using chr() so no literal Korean in source.
_KO_CLASS = '[' + chr(0xAC00) + '-' + chr(0xD7A3) + chr(0x3131) + '-' + chr(0x3163) + ']'
_SPACE_PAT = re.compile('(' + _KO_CLASS + ') (' + _KO_CLASS + ')')


def _remove_spurious_spaces(text: str) -> str:
    """Remove single spaces between adjacent Korean chars/jamo."""
    prev = None
    while prev != text:
        prev = text
        text = _SPACE_PAT.sub(lambda m: m.group(1) + m.group(2), text)
    return text


def _compose_syllable(cho: int, jung: int, jong: int = 0) -> str:
    return chr(_SYLLABLE_START + (cho * _N_JUNG + jung) * _N_JONG + jong)


def _compose_jamo_runs(text: str) -> str:
    """Compose runs of compat-jamo (cho+jung[+jong]) into Hangul syllables."""
    result: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        o = ord(text[i])
        if o in _COMPAT_CHO_RANGE and i + 1 < n and ord(text[i + 1]) in _COMPAT_JUNG_RANGE:
            cho  = _COMPAT_TO_CHO[o]
            jung = _COMPAT_TO_JUNG[ord(text[i + 1])]
            # Optional jongseong: only consume if next-next is NOT a vowel
            if (i + 2 < n and ord(text[i + 2]) in _COMPAT_JONG_RANGE
                    and (i + 3 >= n or ord(text[i + 3]) not in _COMPAT_JUNG_RANGE)):
                jong = _COMPAT_TO_JONG[ord(text[i + 2])]
                result.append(_compose_syllable(cho, jung, jong))
                i += 3
                continue
            result.append(_compose_syllable(cho, jung))
            i += 2
            continue
        result.append(text[i])
        i += 1
    return "".join(result)


def correct_jamo(text: str) -> str:
    """Apply rule-based jamo correction to OCR output text.

    Rules applied in order:
      1. Collapse single spaces between adjacent Korean chars/jamo.
         Example: Å48 SPACE ±55 -> Å48±55
      2. Compose runs of compat jamo (cho+jung[+jong]) into syllables.
         Example: ㅇㅏㄴ -> 한

    Non-Korean text passes through unchanged.
    """
    if not text:
        return text
    # Quick guard: skip if no Korean range chars at all
    if not any(_COMPAT_CHO_START <= ord(c) <= _SYLLABLE_END for c in text):
        return text
    after_space = _remove_spurious_spaces(text)
    after_compose = _compose_jamo_runs(after_space)
    n_space_fixed = len(text) - len(after_space)
    n_jamo_composed = sum(
        1 for a, b in zip(after_space, after_compose) if a != b
    ) if after_space != after_compose else 0
    if n_space_fixed or n_jamo_composed:
        log.debug(
            "jamo_correct.applied n_space_fixed=%d n_jamo_composed=%d "
            "original=%r corrected=%r",
            n_space_fixed, n_jamo_composed, text, after_compose,
            extra={
                "n_space_fixed": n_space_fixed,
                "n_jamo_composed": n_jamo_composed,
                "original": text,
                "corrected": after_compose,
            },
        )
    return after_compose
