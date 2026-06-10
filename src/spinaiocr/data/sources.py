"""Registry of public OCR data sources.

All entries MUST carry license info. Only sources with licenses permitting
redistribution and commercial use go into the default training mix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SourceKind = Literal["kaggle", "hf", "http", "github", "aihub", "icdar", "synthetic"]
License = Literal[
    "cc0",
    "cc-by",
    "cc-by-sa",
    "cc-by-nc",
    "mit",
    "apache-2.0",
    "bsd",
    "unknown",
    "restricted",
]


@dataclass
class DataSource:
    name: str
    kind: SourceKind
    langs: list[str]
    license: License
    url: str
    identifier: str = ""  # kaggle slug, HF dataset id, etc.
    task: str = "recognition"  # recognition | detection | layout | synthetic
    notes: str = ""
    commercial_ok: bool = False
    tags: list[str] = field(default_factory=list)


# Hand-curated seed registry. Extend via scripts/collect_sources.py which
# queries Kaggle/HF APIs and appends new rows after license verification.
SOURCES: list[DataSource] = [
    # ---------- Korean ----------
    DataSource(
        name="AIHub Korean OCR (printed)",
        kind="aihub",
        langs=["ko"],
        license="restricted",  # AI Hub requires registration + research-only usage
        url="https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100",
        task="recognition",
        commercial_ok=False,
        notes="AI Hub 한국어 인쇄체 OCR. 가입 필요, 상업적 사용 제한. 학습 레시피에는 포함하되 배포 모델과 분리 관리.",
        tags=["korean", "printed"],
    ),
    DataSource(
        name="AIHub Korean OCR (handwriting)",
        kind="aihub",
        langs=["ko"],
        license="restricted",
        url="https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100",
        task="recognition",
        commercial_ok=False,
        notes="한국어 손글씨. 손글씨 모델 전용.",
        tags=["korean", "handwriting"],
    ),
    # ---------- English ----------
    DataSource(
        name="ICDAR 2015 Incidental Scene Text",
        kind="icdar",
        langs=["en"],
        license="cc-by",
        url="https://rrc.cvc.uab.es/?ch=4",
        task="detection",
        commercial_ok=True,
        tags=["english", "scene", "detection"],
    ),
    DataSource(
        name="ICDAR 2019 MLT",
        kind="icdar",
        langs=["en", "ko", "ja", "zh", "ar", "hi", "bn", "it"],
        license="cc-by",
        url="https://rrc.cvc.uab.es/?ch=15",
        task="detection",
        commercial_ok=True,
        notes="9개 언어 multi-lingual scene text. 한국어 포함.",
        tags=["multi", "scene", "detection", "korean"],
    ),
    DataSource(
        name="TextOCR",
        kind="hf",
        identifier="facebook/textocr",
        langs=["en"],
        license="cc-by",
        url="https://textvqa.org/textocr/",
        task="recognition",
        commercial_ok=True,
        tags=["english", "scene"],
    ),
    DataSource(
        name="IIIT 5K-word",
        kind="http",
        langs=["en"],
        license="cc-by",
        url="https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset",
        task="recognition",
        commercial_ok=True,
        tags=["english", "recognition"],
    ),
    DataSource(
        name="SynthText",
        kind="http",
        langs=["en"],
        license="cc-by",
        url="https://www.robots.ox.ac.uk/~vgg/data/scenetext/",
        task="synthetic",
        commercial_ok=True,
        notes="800k synthetic scene text images.",
        tags=["english", "synthetic"],
    ),
    DataSource(
        name="MJSynth / Synth90K",
        kind="http",
        langs=["en"],
        license="cc-by",
        url="https://www.robots.ox.ac.uk/~vgg/data/text/",
        task="synthetic",
        commercial_ok=True,
        tags=["english", "synthetic"],
    ),
    # ---------- Documents ----------
    DataSource(
        name="PubLayNet",
        kind="http",
        langs=["en"],
        license="cc-by-nc",
        url="https://github.com/ibm-aur-nlp/PubLayNet",
        task="layout",
        commercial_ok=False,
        notes="NC: 상업적 사용 불가. 레이아웃 연구용.",
        tags=["layout", "english"],
    ),
    DataSource(
        name="DocBank",
        kind="github",
        identifier="doc-analysis/DocBank",
        langs=["en"],
        license="apache-2.0",
        url="https://github.com/doc-analysis/DocBank",
        task="layout",
        commercial_ok=True,
        tags=["layout", "english"],
    ),
    DataSource(
        name="CORD (receipt parsing)",
        kind="hf",
        identifier="naver-clova-ix/cord-v2",
        langs=["en"],
        license="cc-by",
        url="https://huggingface.co/datasets/naver-clova-ix/cord-v2",
        task="layout",
        commercial_ok=True,
        tags=["receipt", "document"],
    ),
    DataSource(
        name="FUNSD",
        kind="http",
        langs=["en"],
        license="cc-by",
        url="https://guillaumejaume.github.io/FUNSD/",
        task="layout",
        commercial_ok=True,
        tags=["form", "document"],
    ),
    # ---------- Japanese / Chinese ----------
    DataSource(
        name="ICDAR 2019 ReCTS (Chinese)",
        kind="icdar",
        langs=["zh"],
        license="cc-by",
        url="https://rrc.cvc.uab.es/?ch=12",
        task="detection",
        commercial_ok=True,
        tags=["chinese", "scene"],
    ),
    DataSource(
        name="LSVT (Chinese)",
        kind="icdar",
        langs=["zh"],
        license="cc-by",
        url="https://rrc.cvc.uab.es/?ch=16",
        task="detection",
        commercial_ok=True,
        tags=["chinese", "scene"],
    ),
    DataSource(
        name="Japanese Handwriting ETL",
        kind="http",
        langs=["ja"],
        license="cc-by",
        url="http://etlcdb.db.aist.go.jp/",
        task="recognition",
        commercial_ok=True,
        tags=["japanese", "handwriting"],
    ),
    # ---------- Multilingual / Auxiliary ----------
    DataSource(
        name="SynthTIGER",
        kind="github",
        identifier="clovaai/synthtiger",
        langs=["ko", "en", "ja", "zh", "multi"],
        license="mit",
        url="https://github.com/clovaai/synthtiger",
        task="synthetic",
        commercial_ok=True,
        notes="학습 파이프라인에서 직접 호출하여 수천만 장 생성.",
        tags=["synthetic", "multi"],
    ),
    DataSource(
        name="TextRecognitionDataGenerator (trdg)",
        kind="github",
        identifier="Belval/TextRecognitionDataGenerator",
        langs=["multi"],
        license="mit",
        url="https://github.com/Belval/TextRecognitionDataGenerator",
        task="synthetic",
        commercial_ok=True,
        tags=["synthetic", "multi"],
    ),
]


def filter_sources(
    lang: str | None = None,
    task: str | None = None,
    commercial_only: bool = False,
) -> list[DataSource]:
    result = []
    for s in SOURCES:
        if lang and lang not in s.langs and "multi" not in s.langs:
            continue
        if task and s.task != task:
            continue
        if commercial_only and not s.commercial_ok:
            continue
        result.append(s)
    return result
