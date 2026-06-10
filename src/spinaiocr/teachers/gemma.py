"""Gemma 4 (and Gemma 3 multimodal) OCR teacher.

Strategy
--------
Gemma 4 is a general VLM — unlike PaddleOCR/EasyOCR it does not natively
emit (bbox, text) pairs. We prompt it with a structured request and parse
JSON out of the response. The response schema is:

    {"words": [{"bbox": [[x,y],[x,y],[x,y],[x,y]], "text": "..."}]}

When the model declines or the JSON fails to parse, we fall back to a
**recognition-only** prediction with a single bbox covering the full image.
This still contributes to multi-teacher consensus for recognition and for
**knowledge distillation** (see :mod:`spinaiocr.training.distill`), where
we use the raw free-form transcription as a soft target.

Model loading
-------------
Two paths:
1. **Local** (default): HuggingFace `transformers` with bf16 on GPU. Pass
   `checkpoint="google/gemma-4-9b-it"` or an alternative Korean-tuned model.
2. **API** (optional): pass `endpoint_url=...` to use an OpenAI-compatible
   serving endpoint (vLLM, TGI). Convenient when inference lives on a
   dedicated GPU box.

License note: Gemma weights carry the **Gemma Terms of Use** — not
unrestricted commercial; review https://ai.google.dev/gemma/terms before
shipping distilled weights derived from Gemma outputs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

from spinaiocr.teachers.base import (
    ImageLike,
    OCRTeacher,
    TeacherLine,
    TeacherPrediction,
    _load_image,
    register,
)


_PROMPT_KO = (
    "아래 이미지의 모든 텍스트를 읽고 JSON으로만 응답하세요. 다음 정확한 스키마를 따르세요:\n"
    '{"words":[{"bbox":[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],"text":"..."}]}\n'
    "- 좌표는 픽셀 단위 정수. 좌상단부터 시계방향.\n"
    "- 한 번 읽기 어려운 텍스트는 제외.\n"
    "- 설명·주석·코드펜스 금지. JSON만 출력."
)

_PROMPT_EN = (
    "Read every piece of visible text in the image and respond with JSON ONLY.\n"
    'Schema: {"words":[{"bbox":[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],"text":"..."}]}\n'
    "- integer pixel coords, clockwise from top-left\n"
    "- skip illegible text\n"
    "- no commentary, no code fences, JSON only."
)


_JSON_FALLBACK = re.compile(r"\{[\s\S]*\}")


@dataclass
class GemmaTeacherConfig:
    checkpoint: str = "google/gemma-4-9b-it"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 1024
    # Prefer a vLLM endpoint on WSL2 — PagedAttention + prefix-caching makes
    # the fixed OCR prompt nearly free after the first request. See
    # deploy/vllm/README.md for setup. Set to None to force local transformers.
    endpoint_url: str | None = "http://localhost:8000/v1/chat/completions"
    endpoint_model: str = "gemma-4"


def _parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    # strip code fences if any
    if raw.startswith("```"):
        raw = raw.strip("`")
        # remove leading json marker line
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        m = _JSON_FALLBACK.search(raw)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None


def _coerce_bbox(bbox) -> np.ndarray | None:
    try:
        arr = np.array(bbox, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] < 3:
            return None
        return arr
    except Exception:  # noqa: BLE001
        return None


@register
class GemmaTeacher(OCRTeacher):
    name = "gemma"
    license = "gemma-tou"  # Gemma Terms of Use — review before commercial use
    commercial_ok = False  # conservatively flagged

    def __init__(self, lang: str = "ko", cfg: GemmaTeacherConfig | None = None) -> None:
        super().__init__(lang=lang)
        self.cfg = cfg or GemmaTeacherConfig()
        self._local_model = None
        self._local_processor = None

    # ---- lazy backends --------------------------------------------------

    def _load_local(self) -> None:
        if self._local_model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
        except ImportError as e:
            raise ImportError(
                "transformers + torch required for local Gemma. "
                "pip install 'transformers>=4.50' torch"
            ) from e
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.cfg.dtype]
        self._local_processor = AutoProcessor.from_pretrained(self.cfg.checkpoint)
        self._local_model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.checkpoint, torch_dtype=dtype
        ).to(self.cfg.device)
        self._local_model.eval()

    def _run_local(self, arr: np.ndarray, prompt: str) -> str:
        import torch

        self._load_local()
        assert self._local_model is not None and self._local_processor is not None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(arr)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._local_processor.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
        ).to(self.cfg.device)
        with torch.no_grad():
            out = self._local_model.generate(
                **inputs if isinstance(inputs, dict) else {"input_ids": inputs},
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
            )
        return self._local_processor.batch_decode(out, skip_special_tokens=True)[0]

    def _run_endpoint(self, arr: np.ndarray, prompt: str) -> str:
        import base64
        import io

        import requests

        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        payload = {
            "model": self.cfg.endpoint_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": self.cfg.max_new_tokens,
            "temperature": 0.0,
        }
        resp = requests.post(
            self.cfg.endpoint_url,  # type: ignore[arg-type]
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ---- public API -----------------------------------------------------

    def __call__(self, image: ImageLike) -> TeacherPrediction:
        arr = _load_image(image)
        prompt = _PROMPT_KO if self.lang == "ko" else _PROMPT_EN
        if self.cfg.endpoint_url:
            raw = self._run_endpoint(arr, prompt)
        else:
            raw = self._run_local(arr, prompt)

        data = _parse_json(raw)
        lines: list[TeacherLine] = []
        if isinstance(data, dict) and isinstance(data.get("words"), list):
            for w in data["words"]:
                bbox = _coerce_bbox(w.get("bbox"))
                text = (w.get("text") or "").strip()
                if bbox is None or not text:
                    continue
                lines.append(TeacherLine(text=text, bbox=bbox, confidence=1.0))

        if not lines:
            # recognition-only fallback
            h, w = arr.shape[:2]
            bbox = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            lines = [TeacherLine(text=raw.strip(), bbox=bbox, confidence=0.5)]

        return TeacherPrediction(
            teacher=self.name, lines=lines, lang=self.lang, license=self.license
        )

    @property
    def last_raw(self) -> str:
        """For distillation: access the most recent raw Gemma output."""
        return getattr(self, "_last_raw", "")
