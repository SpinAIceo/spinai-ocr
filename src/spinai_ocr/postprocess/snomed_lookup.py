"""SNOMED-lite lookup: OCR text → SNOMED CT concept via Korean drug aliases.

MVP per user suggestion — skip full Snowstorm server for now, use a
curated JSON table of ~40 common Korean medications. Expand via Snowstorm
API later when the PoC proves out.

Behavior:
    resolve("타이레놀 500mg")          → {snomed_id: 387517004, ...}
    resolve("acetaminophen 500 mg")   → same match (generic_en)
    resolve("Tylenol (500 mg)")       → no match (brand not in seed)
    resolve("처음보는약 100mg")          → None
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TABLE_PATH = Path("data/pharma/snomed_ko_mvp.json")


@dataclass(frozen=True)
class SnomedMatch:
    snomed_id: str
    snomed_fsn: str
    generic_en: str
    generic_ko: str
    brand_matched: str | None  # the specific brand string we matched on
    dose_mg: float | None
    form: str | None
    category: str | None


class SnomedLookup:
    """Case-insensitive substring match against brand names + generic names."""

    def __init__(self, table_path: Path = TABLE_PATH):
        self.path = table_path
        self._index: list[tuple[str, dict]] = []  # (lowercased alias, concept)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for concept in data.get("concepts", []):
            aliases = [*concept.get("brands_ko", [])]
            for key in ("generic_en", "generic_ko", "snomed_fsn"):
                v = concept.get(key)
                if v:
                    aliases.append(v)
            for alias in aliases:
                if alias and len(alias) >= 2:
                    self._index.append((alias.lower(), concept))
        # Sort by alias length descending so longer brands match first
        # (e.g. "바이엘아스피린" before "아스피린")
        self._index.sort(key=lambda x: -len(x[0]))

    def resolve(self, text: str) -> SnomedMatch | None:
        """Return first concept whose brand/generic appears in text.

        Case-insensitive. Matches longest alias first.
        """
        if not text:
            return None
        haystack = text.lower()
        for alias, concept in self._index:
            if alias in haystack:
                return SnomedMatch(
                    snomed_id=concept["snomed_id"],
                    snomed_fsn=concept["snomed_fsn"],
                    generic_en=concept.get("generic_en", ""),
                    generic_ko=concept.get("generic_ko", ""),
                    brand_matched=alias,
                    dose_mg=concept.get("dose_mg"),
                    form=concept.get("form"),
                    category=concept.get("category"),
                )
        return None

    def resolve_with_dose(self, text: str) -> dict:
        """Resolve SNOMED + also extract dose from text if a number+unit is present.
        Returns dict with match fields plus extracted_dose_mg / extracted_form.
        """
        m = self.resolve(text)
        out: dict = m.__dict__ if m else {}
        # Extract any "NNN mg" / "NNN mcg" / "NNN ml" from text
        dose_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(mg|mcg|ml|µg|g|iu)\b",
            text, re.IGNORECASE,
        )
        if dose_match:
            val = float(dose_match.group(1))
            unit = dose_match.group(2).lower()
            if unit == "mcg" or unit == "µg":
                val /= 1000.0
                unit = "mg"
            elif unit == "g":
                val *= 1000.0
                unit = "mg"
            out["extracted_dose"] = val
            out["extracted_unit"] = unit
        form_match = re.search(
            r"\b(tablet|capsule|syrup|injection|inhaler|정|캡슐|시럽|주사)\b",
            text, re.IGNORECASE,
        )
        if form_match:
            out["extracted_form_raw"] = form_match.group(1)
        return out


@lru_cache(maxsize=1)
def get_lookup() -> SnomedLookup:
    return SnomedLookup()
