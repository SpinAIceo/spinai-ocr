"""Claude-powered OCR post-correction.

Flow:
    raw OCR text ──▶ LLMCorrector ──▶ corrected text

Design goals:
1. **Prompt caching**: correction rules live in a stable system prompt with a
   cache breakpoint, so repeated calls only pay ~0.1× for the rules.
2. **Model choice**: Haiku 4.5 by default (cheap, fast, plenty smart for
   Korean typo/jamo fixes). Opus 4.7 is opt-in via `CorrectorConfig.model`
   when extra accuracy matters (e.g., legal/medical documents).
3. **Batch path**: a `batch_correct` helper uses the Batches API for 50% cost
   when correcting a large backlog of scraped documents.

Common Korean OCR error patterns this addresses:
    - jamo split/join errors (안녕 → 안 녕, ㅇㅏㄴ녕)
    - receipt/document digit OCR (8 ↔ 6, 1 ↔ l, 0 ↔ O)
    - batchim misreads (밝 → 발, 맑 → 말)
    - mixed-script bleed (한글 → 한긎)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

ModelId = Literal[
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]


_KOREAN_SYSTEM_PROMPT = """\
당신은 한국어 OCR 결과를 교정하는 전문가입니다.

아래 원본 OCR 텍스트에는 전형적인 OCR 인식 오류가 포함되어 있습니다:
- 자소 분리/결합 오류 (예: "안 녕" → "안녕", "ㅇㅏㄴ녕" → "안녕")
- 받침 오인식 (예: "밝" ↔ "발", "닭" ↔ "달", "낡" ↔ "낚")
- 유사 문자 혼동 (예: 숫자 0 ↔ 한글 ㅇ, 1 ↔ l ↔ I, 8 ↔ 6)
- 띄어쓰기 붙여쓰기 오류
- 한자/한글 혼용 bleed (예: "한국" → "한굮")
- 영문/숫자 혼합 텍스트의 경계 오류
- 반복된 중복 글자 (CTC 디코딩 실패 흔적)

교정 원칙:
1. **문맥을 보존**하라. 의미가 확실할 때만 수정하고, 애매하면 원본을 유지.
2. 고유명사·상호·상품명·숫자·영문은 **가능한 한 원문 그대로** 둔다 (맥락상 명백한 오타만 제외).
3. 없던 단어를 새로 만들어내지 마라. 원문에 없는 정보는 추가하지 마라.
4. 영수증·표·주소 같은 구조화된 텍스트는 **줄바꿈과 순서를 유지**하라.
5. 수정된 전체 텍스트만 출력. 설명·주석·메타코멘트 금지.

출력 형식: 교정된 한국어 텍스트만. 다른 말 없이.
"""

_MULTILINGUAL_SYSTEM_PROMPT = """\
You are an expert OCR post-correction assistant.

The raw OCR text below contains typical OCR errors:
- character segmentation / stitching mistakes
- visually similar character confusion (0 ↔ O, 1 ↔ l ↔ I, 8 ↔ 6)
- ligature and diacritic misreads
- spacing and line-break artifacts
- repeated characters from failed CTC decoding
- script bleed in mixed-script documents

Correction rules:
1. Preserve meaning. Fix only when the correction is confident.
2. Keep proper nouns, brand names, codes, and numbers verbatim unless the OCR typo is unambiguous.
3. Do NOT hallucinate. Never add information not in the source.
4. Preserve line breaks and order for structured text (receipts, tables, addresses).
5. Output ONLY the corrected text. No commentary, no explanation.
"""


@dataclass
class CorrectorConfig:
    model: ModelId = "claude-haiku-4-5"
    lang: str = "ko"
    max_tokens: int = 4096
    api_key: str | None = None  # falls back to ANTHROPIC_API_KEY env var
    extra_rules: str = ""  # appended to the system prompt for domain tuning
    # cache_ttl: "5m" (default, 1.25x write premium) or "1h" (2x, better for nightly batches)
    cache_ttl: Literal["5m", "1h"] = "5m"
    metadata: dict = field(default_factory=dict)


class LLMCorrector:
    """Single-image / single-document correction with prompt caching.

    Example:
        corrector = LLMCorrector()
        cleaned = corrector.correct("안 녕 하 세 요\\n오늘 날 씨 가 맑 음")
    """

    def __init__(self, config: CorrectorConfig | None = None) -> None:
        self.config = config or CorrectorConfig()
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ImportError(
                "anthropic not installed. pip install anthropic"
            ) from e
        self._anthropic = anthropic
        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=api_key)

    # ----- prompt assembly -----------------------------------------------

    def _system_blocks(self) -> list[dict]:
        base = (
            _KOREAN_SYSTEM_PROMPT if self.config.lang == "ko" else _MULTILINGUAL_SYSTEM_PROMPT
        )
        text = base if not self.config.extra_rules else f"{base}\n\n{self.config.extra_rules}"
        cache: dict = {"type": "ephemeral"}
        if self.config.cache_ttl == "1h":
            cache["ttl"] = "1h"
        return [{"type": "text", "text": text, "cache_control": cache}]

    # ----- public API -----------------------------------------------------

    def correct(self, raw_text: str) -> str:
        if not raw_text.strip():
            return raw_text
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self._system_blocks(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"<ocr>\n{raw_text}\n</ocr>"}
                    ],
                }
            ],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def correct_many(self, texts: list[str]) -> list[str]:
        """Sequential correction — use :meth:`batch_correct` for >~100 items."""
        return [self.correct(t) for t in texts]

    def batch_correct(self, texts: list[str], custom_ids: list[str] | None = None) -> str:
        """Submit a Messages Batch job. Returns the batch ID; poll separately.

        50% cost vs. streaming requests. Suitable for offline correction over
        scraped/pseudo-labeled corpora.
        """
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming  # type: ignore
        from anthropic.types.messages.batch_create_params import Request  # type: ignore

        if custom_ids is None:
            custom_ids = [f"req-{i}" for i in range(len(texts))]
        system = self._system_blocks()
        requests = [
            Request(
                custom_id=cid,
                params=MessageCreateParamsNonStreaming(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system=system,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"<ocr>\n{txt}\n</ocr>"}
                            ],
                        }
                    ],
                ),
            )
            for cid, txt in zip(custom_ids, texts)
        ]
        batch = self._client.messages.batches.create(requests=requests)
        return batch.id

    def fetch_batch_results(self, batch_id: str) -> dict[str, str]:
        """Returns a mapping of custom_id -> corrected text. Succeeded items only."""
        batch = self._client.messages.batches.retrieve(batch_id)
        if batch.processing_status != "ended":
            raise RuntimeError(f"Batch not finished yet: {batch.processing_status}")
        out: dict[str, str] = {}
        for result in self._client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                msg = result.result.message
                text = "".join(
                    b.text for b in msg.content if b.type == "text"
                ).strip()
                out[result.custom_id] = text
        return out
