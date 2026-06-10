"""FastAPI server for SPINAI OCR.

Endpoints:
    GET  /health           — liveness probe
    GET  /version          — version + loaded model info
    POST /ocr              — multipart image → structured OCR result
    POST /ocr/batch        — JSON array of base64 images → array of results
    GET  /errors/recent    — recent WARNING+ log entries for batch error review
    GET  /logs/stream      — SSE stream of live log entries

Run:
    uvicorn spinaiocr.serve.app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import threading
import time as _time
from pathlib import Path
from typing import Annotated, AsyncGenerator

import os

from fastapi import FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from spinaiocr import __version__
from spinaiocr.config import DecodeMode, LangCode, ModelTier, PipelineConfig
from spinaiocr.inference.pipeline import OCRPipeline
from spinaiocr.log import get_logger, setup_logging
from spinaiocr.serve.ratelimit import (
    check_monthly_quota, commit_monthly_call, get_usage,
)
from spinaiocr.serve.hard_negatives import maybe_queue_hard_negative
from spinaiocr.serve.request_log import (
    log_ocr_request, read_recent, aggregate, image_fingerprint,
)

setup_logging()
log = get_logger("spinaiocr.serve.app")

# Hard caps: a single upload larger than this, or a batch larger than this,
# is rejected at the boundary rather than OOM-ing the Railway instance.
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_BATCH_SIZE = 16

app = FastAPI(title="SPINAI OCR", version=__version__)


@app.on_event("startup")
async def _on_startup() -> None:
    log.info(
        "serve.startup version=%s pid=%d",
        __version__, os.getpid(),
        extra={"event": "startup", "version": __version__, "pid": os.getpid()},
    )


# CORS for local dev (harmless in production since we also serve the UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Repo-root web/ is the canonical UI source for local dev; static/ is the
# baked copy that ships in the Docker image. Prefer web/ when running from
# source so edits to web/index.html show up immediately without a sync step
# (caught a divergence in _43 where the two files drifted by hours).
_REPO_WEB_INDEX = Path(__file__).resolve().parents[3] / "web" / "index.html"


@app.get("/")
def index():
    """Serve the SPINAI OCR web UI."""
    if _REPO_WEB_INDEX.exists():
        return FileResponse(_REPO_WEB_INDEX)
    idx = _STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(idx)
    raise HTTPException(404, "UI not installed")

# Key by (lang, tier) so requests with different combos don't silently share
# the first-created pipeline. Previously a `tier=standard` request after a
# `tier=lite` one reused the lite pipeline and lied about the tier in results.
_PIPELINES: dict[tuple[LangCode, ModelTier], OCRPipeline] = {}
_PIPELINE_LOCK = threading.Lock()


def _get_pipeline(lang: LangCode = "ko", tier: ModelTier = "consumer_v1") -> OCRPipeline:
    """Default tier flipped lite → consumer_v1 in _63 after live A/B
    showed 2.4× CER reduction (0.54 → 0.22) on the same 96 samples.
    API callers who need the old/fast path can still pass ?tier=lite."""
    key = (lang, tier)
    pipe = _PIPELINES.get(key)
    if pipe is not None:
        return pipe
    with _PIPELINE_LOCK:
        pipe = _PIPELINES.get(key)
        if pipe is None:
            pipe = OCRPipeline(config=PipelineConfig(lang=lang, tier=tier))
            _PIPELINES[key] = pipe
    return pipe


class Line(BaseModel):
    text: str
    bbox: list[list[float]]
    confidence: float
    # _80 (iter 24): per-step min top-1 softmax prob, calibrated with the
    # same per-tier conf_power as `confidence`. Use for UI signals like
    # "highlight the weakest character in this line" — geometric-mean
    # `confidence` can be high while one single character is wildly
    # uncertain, especially on long receipts / addresses.
    min_char_conf: float = 1.0
    unk_count: int = 0   # _53: OOV char count for medical safety


class OCRResponse(BaseModel):
    lines: list[Line]
    image_width: int
    image_height: int
    status: str = "ok"  # "ok" | "partial" (some unk) | "uncertain" (too many unk)
    total_unk: int = 0


class BatchRequest(BaseModel):
    images: Annotated[list[str], Field(description="base64-encoded images")]
    lang: LangCode = "ko"
    tier: ModelTier = "consumer_v1"
    decode: DecodeMode = "beam_lm"
    single_line: bool | None = None


@app.get("/health")
def health() -> dict:
    """Liveness probe. Kept fast so Railway healthchecks don't stack.
    Use /ready to also verify that model checkpoints are installed."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness probe: app can actually serve /ocr. Checks that required
    checkpoints exist on disk — catches deploys where the checkpoint copy
    step was accidentally skipped (broken Dockerfile, .dockerignore miss).

    iter 136: also check `consumer_v1` (prod default tier per /ocr at
    line 87). Pre-fix only verified `lite` → /ready returned 200 on a
    container missing consumer_v1 ckpts, even though the default /ocr
    call would silently degrade to empty output. Both tiers are required
    for the deploy to be honestly ready."""
    required = [
        Path("checkpoints/lite/ko/rec.pth"),
        Path("checkpoints/lite/ko/det.pth"),
        Path("checkpoints/consumer_v1/ko/rec.pth"),
        Path("checkpoints/consumer_v1/ko/det.pth"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise HTTPException(503, {"status": "not_ready", "missing": missing})
    return {"status": "ready", "pipelines_loaded": len(_PIPELINES)}


@app.get("/env")
def env_diag() -> dict:
    """_130 (iter 76): probe runtime env to characterize prod-vs-local gaps.
    matched-CER differs 0.04 (local CPU) vs 0.06 (prod CPU) on identical
    inputs + identical model SHA. Threading + Docker cache both falsified.
    Hypothesis = lib version / torch wheel diff. This endpoint exposes
    versions so a single GET reveals all candidates without burning /ocr quota."""
    import sys, hashlib, platform
    info: dict = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["torch_interop"] = torch.get_num_interop_threads()
        info["mkl"] = torch.backends.mkl.is_available()
        info["mkldnn"] = torch.backends.mkldnn.is_available()
    except Exception as e:  # noqa: BLE001
        info["torch_err"] = str(e)
    try:
        import PIL
        info["pillow"] = PIL.__version__
    except Exception as e:  # noqa: BLE001
        info["pillow_err"] = str(e)
    try:
        import cv2
        info["cv2"] = cv2.__version__
    except Exception as e:  # noqa: BLE001
        info["cv2_err"] = str(e)
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception as e:  # noqa: BLE001
        info["numpy_err"] = str(e)
    rec = Path("checkpoints/consumer_v1/ko/rec.pth")
    if rec.exists():
        info["consumer_v1_rec_sha"] = hashlib.sha256(rec.read_bytes()).hexdigest()[:10]
        info["consumer_v1_rec_size"] = rec.stat().st_size
    return info


@app.get("/version")
def version() -> dict:
    return {
        "version": __version__,
        "pipelines_loaded": [
            {"lang": l, "tier": t} for (l, t) in _PIPELINES
        ],
    }


# _137 (iter 82) activated SPINAI_HN_QUEUE in deploy. _138 (iter 83) adds
# retrieve mechanism — Railway's container is non-persistent, so without a
# pull endpoint each redeploy wipes the captured failure corpus before any
# value is extracted. Auth = SPINAI_ADMIN_TOKEN env var; X-Admin-Token
# header must match exactly. Both env-var unset and header mismatch return
# 404 (not 401) so an unauthorized caller can't even confirm the endpoint
# exists. List capped at limit=500 to prevent abusively large responses.
def _admin_auth_or_404(token: str | None) -> None:
    expected = os.environ.get("SPINAI_ADMIN_TOKEN", "").strip()
    if not expected or not token or token != expected:
        raise HTTPException(status_code=404, detail="not_found")


@app.get("/admin/hard_negatives/list")
def admin_hn_list(
    limit: int = 200,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict:
    _admin_auth_or_404(x_admin_token)
    queue_root = Path(os.environ.get("SPINAI_HN_DIR", "data/hard_negatives"))
    queue_file = queue_root / "queue.jsonl"
    if not queue_file.exists():
        return {"n": 0, "entries": [], "queue_root": queue_root.as_posix()}
    import json as _json
    entries = []
    with queue_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(_json.loads(line))
            except Exception:
                continue
    entries = entries[-max(1, min(limit, 500)):]
    img_dir = queue_root / "images"
    total_mb = 0.0
    n_imgs = 0
    if img_dir.exists():
        for p in img_dir.iterdir():
            if p.is_file():
                total_mb += p.stat().st_size / (1024 * 1024)
                n_imgs += 1
    return {
        "n": len(entries),
        "n_images_on_disk": n_imgs,
        "total_disk_mb": round(total_mb, 2),
        "queue_root": queue_root.as_posix(),
        "entries": entries,
    }


@app.get("/admin/hard_negatives/image/{uid}")
def admin_hn_image(
    uid: str,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
):
    _admin_auth_or_404(x_admin_token)
    if not uid.isalnum() or len(uid) > 32:
        raise HTTPException(status_code=400, detail="bad_uid")
    queue_root = Path(os.environ.get("SPINAI_HN_DIR", "data/hard_negatives"))
    img_path = queue_root / "images" / f"{uid}.jpg"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="not_found")
    return FileResponse(img_path, media_type="image/jpeg")


@app.post("/warmup")
def warmup(
    lang: LangCode = "ko",
    tier: ModelTier = "consumer_v1",
    decode: DecodeMode = "beam_lm",
) -> dict:
    """Force model load so the first real /ocr call is fast.

    Railway containers idle → cold-start loads DBNet + SVTR + angle classifier
    (~10 s). Calling /warmup once on page load lets the user's real request be
    immediate.

    When `decode=beam_lm` (the default), also eagerly fit the char-bigram LM
    from the bundled corpora. Without this, the first beam_lm /ocr request
    pays the ~380 ms LM-fit cost inline — measured post _34: cold 407 ms →
    warm 25 ms. Fitting here moves that into the explicit warmup budget so
    the first user-facing request is already warm.
    """
    t0 = _time.perf_counter()
    pipe = _get_pipeline(lang, tier)
    pipe._ensure_loaded()
    lm_fit_ms: float | None = None
    if decode == "beam_lm":
        t_lm = _time.perf_counter()
        # _38: pipeline keeps two LMs (single_line w/ korean-pair+wiki,
        # multi_poly w/o). _51: fit them in parallel via threadpool — they
        # are independent and CPU-bound. Was 596 ms sequential; parallel
        # bounds at max(single, multi) ≈ 300 ms.
        from concurrent.futures import ThreadPoolExecutor
        fit_tasks = []
        with ThreadPoolExecutor(max_workers=2) as ex:
            if pipe._lm_single_line is None:
                fit_tasks.append(("single", ex.submit(pipe._build_lm, True)))
            if pipe._lm_multi_poly is None:
                fit_tasks.append(("multi", ex.submit(pipe._build_lm, False)))
            for name, fut in fit_tasks:
                lm = fut.result()
                if name == "single":
                    pipe._lm_single_line = lm
                else:
                    pipe._lm_multi_poly = lm
        lm_fit_ms = round((_time.perf_counter() - t_lm) * 1000, 1)
    # Touch both models with synthetic images so PyTorch's first-inference
    # kernel dispatch (cuDNN / MKL auto-tune) happens here, not on the user's
    # first real request. Two separate shapes to cover both routes:
    #   48×160  — single-line (recognizer only)
    #   480×480 — full detector + recognizer (multi-polygon path)
    # Measured pre-change: first beam_lm /ocr after basic warmup was ~230 ms
    # (single-line) / ~190 ms (multi-poly). After: ~50 ms / ~90 ms.
    dummy_ms: float | None = None
    try:
        import numpy as _np
        t_d = _time.perf_counter()
        pipe(_np.full((48, 160, 3), 255, dtype=_np.uint8),
             single_line=True, decode_mode="greedy")
        pipe(_np.full((480, 480, 3), 255, dtype=_np.uint8),
             single_line=False, decode_mode="greedy")
        dummy_ms = round((_time.perf_counter() - t_d) * 1000, 1)
    except Exception:  # noqa: BLE001 — warmup best-effort
        pass
    return {
        "status": "ready",
        "elapsed_ms": round((_time.perf_counter() - t0) * 1000, 1),
        "lm_fit_ms": lm_fit_ms,
        "dummy_inference_ms": dummy_ms,
        "lang": lang,
        "tier": tier,
        "decode": decode,
    }


def _client_identity(request: Request) -> str:
    """Identify the caller for quota tracking. Prefers X-API-Key header over
    IP (so paid-tier keys are stable across clients). Anonymous IP is the
    default freemium bucket."""
    key = request.headers.get("x-api-key")
    if key:
        return f"key:{key}"
    # X-Forwarded-For is set by Railway / Vercel proxies; fall back to peer.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return f"ip:{fwd.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


@app.get("/quota")
def quota(request: Request) -> dict:
    """Inspect the caller's monthly-quota usage. Free tier = 50 /ocr calls/mo."""
    return get_usage(_client_identity(request))


def _decode_and_preprocess(blob: bytes, preprocess_mode: str | None):
    """Decode JPEG/PNG bytes → PIL.Image (RGB). iter 92: JPEG draft decode
    at 1/2 native scale for photos ≥~2000px wide cuts decode 65→30ms.
    Optional CLAHE preprocess applied if requested. Returns (image, pre_ms).
    iter 106: hoisted from /ocr closure for cleanliness; unchanged behavior."""
    im = Image.open(io.BytesIO(blob))
    if im.format == "JPEG":
        im.draft("JPEG", (1000, 750))
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    pre_taken: float | None = None
    if preprocess_mode == "clahe":
        t = _time.perf_counter()
        import cv2
        import numpy as _np
        arr = _np.asarray(im)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr2 = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]),
                             cv2.COLOR_LAB2RGB)
        im = Image.fromarray(arr2)
        pre_taken = round((_time.perf_counter() - t) * 1000, 1)
    return im, pre_taken


@app.post("/ocr", response_model=OCRResponse)
async def ocr(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    lang: LangCode = "ko",
    tier: ModelTier = "consumer_v1",
    single_line: bool | None = None,
    decode: DecodeMode = "beam_lm",
    preprocess: str | None = None,
) -> OCRResponse:
    t0 = _time.perf_counter()
    # Freemium gate — 50 successful /ocr calls per identity per UTC month.
    # Check first; commit only after successful OCR so 413/400/422 don't
    # burn quota on malformed requests.
    identity = _client_identity(request)
    check_monthly_quota(identity, limit=50)
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"Image too large (>{MAX_IMAGE_BYTES} bytes)")
    sha16 = image_fingerprint(data)
    log.info(
        "ocr.upload lang=%s tier=%s decode=%s bytes=%d sha=%s",
        lang, tier, decode, len(data), sha16,
        extra={"phase": "upload", "lang": lang, "tier": tier, "decode": decode,
               "bytes": len(data), "sha256_16": sha16,
               "content_type": file.content_type or "unknown"},
    )
    # iter 90: PIL decode + optional CLAHE moved to threadpool. Both release
    # GIL during their hot loops, so concurrent requests now overlap instead
    # of serializing on the event loop. iter 87 caveat resolved.
    pre_ms: float | None = None

    try:
        img, pre_ms = await run_in_threadpool(
            _decode_and_preprocess, data, preprocess)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Invalid image: {e}")
    img_w0, img_h0 = img.size
    pipe = _get_pipeline(lang, tier)
    # Offload CPU-heavy OCR to a worker thread so the event loop keeps
    # accepting concurrent requests. Without this every request serialized.
    # decode_mode is passed per-call to stay thread-safe across concurrent
    # requests hitting the same shared pipeline.
    _t_pipe = _time.perf_counter()
    result = await run_in_threadpool(
        pipe, img, single_line=single_line, decode_mode=decode,
    )
    pipe_ms = round((_time.perf_counter() - _t_pipe) * 1000, 1)
    _timing = getattr(result, "timing_ms", {})
    if _timing.get("detect_ms") is not None:
        response.headers["X-Detect-Ms"] = str(_timing["detect_ms"])
    if _timing.get("recognize_ms") is not None:
        response.headers["X-Recognize-Ms"] = str(_timing["recognize_ms"])
    log.info(
        "ocr.pipeline.done lang=%s tier=%s lines=%d pipe_ms=%.1f "
        "detect_ms=%s recognize_ms=%s",
        lang, tier, len(result.lines), pipe_ms,
        _timing.get("detect_ms", "?"), _timing.get("recognize_ms", "?"),
        extra={"phase": "pipeline_done", "lang": lang, "tier": tier,
               "decode": decode, "n_lines": len(result.lines),
               "pipe_ms": pipe_ms,
               "detect_ms": _timing.get("detect_ms"),
               "recognize_ms": _timing.get("recognize_ms"),
               "sha256_16": sha16},
    )
    # _52: hard-negative queue — save low-confidence requests for the
    # data flywheel (periodic re-label via consensus teacher + retrain).
    # Disabled by default; enable in production with SPINAI_HN_QUEUE=1.
    # iter 96: dispatch via run_in_threadpool. When SPINAI_HN_QUEUE=1 (prod),
    # this path runs JPEG encode + JSONL append on the event loop —
    # blocked concurrent /ocr requests on shared disk I/O. Same pattern
    # as iter 90's _decode_and_preprocess fix. Bench (16 conc, c=8,
    # text-rich big-photo, threshold=0.99 forcing every queue): pre-fix
    # sync p50 444ms / req/s 16.41 → post-fix p50 424ms / req/s 16.58
    # (-5% p50, +1% req/s — marginal because pipeline ~80ms still
    # dominates per-request time vs hard-neg ~10ms disk I/O; gain
    # widens with sustained queue write-heavy load).
    try:
        uid = await run_in_threadpool(
            maybe_queue_hard_negative,
            img, result.lines,
            {"lang": lang, "tier": tier, "decode": decode,
             "single_line": single_line, "identity": identity},
        )
        if uid:
            response.headers["X-HN-Queued"] = uid
    except Exception:  # noqa: BLE001 — never break /ocr for telemetry
        pass
    # _53: medical safety — <unk>-based Tier-2 gate. Total OOV chars
    # across all recognised lines.
    total_unk = sum(getattr(l, "unk_count", 0) for l in result.lines)
    total_chars = sum(len(l.text) for l in result.lines) or 1
    unk_ratio = total_unk / total_chars
    response.headers["X-Elapsed-Ms"] = f"{(_time.perf_counter() - t0) * 1000:.1f}"
    if total_unk:
        response.headers["X-Unk-Count"] = str(total_unk)
    # >=20% unk → reject rather than return garbage that could be misread
    # as a drug name / dosage. 422 signals "text present but unsafe to trust".
    if unk_ratio >= 0.20:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "text_uncertain",
                "reason": "high_oov_ratio",
                "unk_count": total_unk,
                "total_chars": total_chars,
                "unk_ratio": round(unk_ratio, 3),
                "suggestion": "retry with a sharper / closer / less-glare image",
            },
        )
    status = "ok" if total_unk == 0 else "partial"
    commit_monthly_call(identity)
    # _60: per-request granular log for model-performance analysis.
    # Never raises. Stored as one JSON per line; `/metrics` aggregates.
    # iter 97 NEG: tried run_in_threadpool dispatch here (parallel to
    # iter 90 / 96 fixes). Bench showed +3-7% p50 latency hit and zero
    # req/s gain on local SSD — the JSONL append is ~1ms total, smaller
    # than starlette's threadpool-dispatch overhead (anyio.to_thread.
    # run_sync + functools.partial cross-thread sync). Reverted.
    # Knowledge: don't blindly threadpool-dispatch every sync I/O path —
    # the offloaded work must be expensive enough to amortize dispatch
    # cost. iter 96 saw a small win because JPEG encode (~10ms) is 10×
    # the dispatch overhead; iter 97's plain JSONL append is below
    # break-even. logs/iter97/ has the A/B numbers.
    try:
        import uuid as _uuid
        confs = [float(l.confidence) for l in result.lines] or [0.0]
        mccs = [float(getattr(l, "min_char_conf", 1.0)) for l in result.lines] or [1.0]
        log_ocr_request(
            req_id=_uuid.uuid4().hex[:12],
            identity=identity,
            params={"lang": lang, "tier": tier, "decode": decode,
                     "single_line": single_line, "preprocess": preprocess},
            image={"w": img_w0, "h": img_h0, "bytes": len(data),
                    "sha256_16": sha16,
                    "content_type": file.content_type or "unknown"},
            result={"status": status, "n_lines": len(result.lines),
                     "total_unk": total_unk, "total_chars": total_chars,
                     "avg_conf": round(sum(confs) / len(confs), 4),
                     "min_conf": round(min(confs), 4),
                     "max_conf": round(max(confs), 4),
                     "min_char_conf": round(min(mccs), 4),
                     "text_preview": (" | ".join(l.text for l in result.lines))[:200]},
            timing_ms={"total": round((_time.perf_counter() - t0) * 1000, 1),
                        "pipeline": pipe_ms,
                        "preprocess": pre_ms},
        )
    except Exception:  # noqa: BLE001
        pass
    return OCRResponse(
        status=status,
        total_unk=total_unk,
        lines=[Line(
            text=l.text,
            bbox=[list(p) for p in l.bbox],
            confidence=l.confidence,
            min_char_conf=getattr(l, "min_char_conf", 1.0),
            unk_count=getattr(l, "unk_count", 0),
        ) for l in result.lines],
        image_width=result.image_width,
        image_height=result.image_height,
    )


@app.get("/metrics")
def metrics(limit: int = 200) -> dict:
    """Aggregated stats for recent /ocr calls — CER proxy (avg_conf),
    latency percentiles, unk rate, grouped by (tier, decode).

    Use for model-performance testing without re-running traffic:
    ``curl .../metrics?limit=500 | jq .``
    """
    entries = read_recent(limit=min(max(limit, 1), 2000))
    return aggregate(entries)


@app.get("/metrics/recent")
def metrics_recent(limit: int = 50) -> list[dict]:
    """Last N raw request log entries. Useful for inspecting individual
    bad cases (low confidence, high unk, long latency)."""
    return read_recent(limit=min(max(limit, 1), 500))


@app.get("/errors/recent")
def errors_recent(
    limit: int = 100,
    lang: str | None = None,
    min_level: str = "WARNING",
) -> dict:
    """Return recent WARNING+ log entries from spinai.jsonl for batch error review.

    Useful for: finding per-language failure patterns, diagnosing confidence
    drops, reviewing skipped sources during training. Filter by lang= to see
    ko/en/multi specific issues.

    Query params:
        limit     — max entries (default 100, max 500)
        lang      — filter by language (ko, en, multi)
        min_level — minimum log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    """
    _LOG_DIR = Path(os.environ.get("SPINAI_LOG_DIR", "logs"))
    # Prefer errors.jsonl (WARNING-only, smaller) when filtering for errors.
    # Fall back to full spinai.jsonl if errors.jsonl doesn't exist yet.
    jsonl_path = _LOG_DIR / "errors.jsonl"
    if not jsonl_path.exists():
        jsonl_path = _LOG_DIR / "spinai.jsonl"
    if not jsonl_path.exists():
        return {"n": 0, "entries": [], "note": "spinai.jsonl not found — run with logging enabled"}
    level_map = {
        "DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50
    }
    min_lvl_num = level_map.get(min_level.upper(), 30)
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            raw_lines = f.readlines()
    except Exception as e:
        return {"n": 0, "entries": [], "error": str(e)}
    entries = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        rec_level = level_map.get(rec.get("level", ""), 0)
        if rec_level < min_lvl_num:
            continue
        if lang:
            extra = rec.get("extra", {})
            if extra.get("lang") != lang and f"lang={lang}" not in rec.get("msg", ""):
                continue
        entries.append(rec)
    entries = entries[-min(max(limit, 1), 500):]
    by_level: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    for e in entries:
        by_level[e.get("level", "?")] = by_level.get(e.get("level", "?"), 0) + 1
        elang = e.get("extra", {}).get("lang", "?")
        by_lang[elang] = by_lang.get(elang, 0) + 1
    return {
        "n": len(entries),
        "by_level": by_level,
        "by_lang": by_lang,
        "filter": {"lang": lang, "min_level": min_level},
        "entries": entries,
    }


# In-memory SSE subscriber queue for /logs/stream
_SSE_SUBSCRIBERS: list[asyncio.Queue] = []
_SSE_LOCK = threading.Lock()


class _SSELogHandler(logging.Handler):
    """Pushes formatted log records to all active SSE subscriber queues."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from spinaiocr.log.formatters import JsonLinesFormatter
            msg = JsonLinesFormatter().format(record)
        except Exception:
            msg = self.format(record)
        with _SSE_LOCK:
            dead = []
            for q in _SSE_SUBSCRIBERS:
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                _SSE_SUBSCRIBERS.remove(q)


_sse_handler = _SSELogHandler()
_sse_handler.setLevel(logging.INFO)
logging.getLogger("spinaiocr").addHandler(_sse_handler)


@app.get("/logs/stream")
async def logs_stream(
    lang: str | None = None,
    min_level: str = "INFO",
) -> StreamingResponse:
    """SSE stream of live SPINAI log entries (INFO+).

    Usage in browser:
        const es = new EventSource('/logs/stream?lang=ko&min_level=WARNING');
        es.onmessage = e => console.log(JSON.parse(e.data));

    Filter params:
        lang      — only emit records whose extra.lang matches
        min_level — minimum log level to emit
    """
    level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
    min_lvl_num = level_map.get(min_level.upper(), 20)
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    with _SSE_LOCK:
        _SSE_SUBSCRIBERS.append(q)

    async def _generate() -> AsyncGenerator[str, None]:
        yield "data: {\"event\":\"connected\"}\n\n"
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(q.get(), timeout=25.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                try:
                    rec = json.loads(raw)
                except Exception:
                    continue
                rec_lvl = level_map.get(rec.get("level", ""), 0)
                if rec_lvl < min_lvl_num:
                    continue
                if lang:
                    if rec.get("extra", {}).get("lang") != lang and \
                            f"lang={lang}" not in rec.get("msg", ""):
                        continue
                yield f"data: {json.dumps(rec, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            with _SSE_LOCK:
                try:
                    _SSE_SUBSCRIBERS.remove(q)
                except ValueError:
                    pass

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _decode_b64_to_pil(b64: str) -> Image.Image:
    """Decode + open in one function so it can ship to the threadpool as a
    single unit (base64 decode + PIL decode are both GIL-releasing on I/O-
    heavy paths, and keeping them together saves an extra thread hop)."""
    try:
        raw = base64.b64decode(b64)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Invalid base64: {e}")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Batch item too large")
    try:
        im = Image.open(io.BytesIO(raw))
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Invalid base64 image: {e}")


@app.post("/ocr/batch", response_model=list[OCRResponse])
async def ocr_batch(request: Request, req: BatchRequest,
                     response: Response) -> list[OCRResponse]:
    t0 = _time.perf_counter()
    if len(req.images) > MAX_BATCH_SIZE:
        raise HTTPException(
            413, f"Batch too large: {len(req.images)} > {MAX_BATCH_SIZE}"
        )
    # Freemium gate — each image in a batch counts as one call. Check up
    # front that the identity has headroom for the whole batch; commit
    # each call only after the batch succeeds.
    identity = _client_identity(request)
    usage = get_usage(identity)
    n = len(req.images)
    if usage["used"] + n > usage["limit"]:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "monthly_quota_exceeded_batch",
                "used": usage["used"], "batch_size": n,
                "limit": usage["limit"], "identity": identity,
            },
        )
    pipe = _get_pipeline(req.lang, req.tier)

    async def _one(b64: str) -> OCRResponse:
        img = await run_in_threadpool(_decode_b64_to_pil, b64)
        result = await run_in_threadpool(
            pipe, img, single_line=req.single_line, decode_mode=req.decode,
        )
        # Parity with /ocr: propagate per-line unk_count and derive per-item
        # status so medical-safety downstream consumers can filter on
        # status == "uncertain" without re-running the gate themselves.
        total_unk = sum(getattr(l, "unk_count", 0) for l in result.lines)
        total_chars = sum(len(l.text) for l in result.lines) or 1
        ratio = total_unk / total_chars
        if ratio >= 0.20:
            status = "uncertain"
        elif total_unk:
            status = "partial"
        else:
            status = "ok"
        return OCRResponse(
            status=status,
            total_unk=total_unk,
            lines=[Line(
                text=l.text,
                bbox=[list(p) for p in l.bbox],
                confidence=l.confidence,
                min_char_conf=getattr(l, "min_char_conf", 1.0),
                unk_count=getattr(l, "unk_count", 0),
            ) for l in result.lines],
            image_width=result.image_width,
            image_height=result.image_height,
        )

    # Parallel: earlier this was `for b64 in images: await …` — each image
    # blocked the next, so a 16-image batch was ~16× single-call latency.
    # With asyncio.gather the threadpool runs them concurrently; GPU/CPU
    # bound work overlaps with I/O decoding for sibling items.
    out = await asyncio.gather(*(_one(b) for b in req.images))
    # iter 135: parity with /ocr 422 path — uncertain items (unk_ratio ≥ 0.20)
    # are free-retry per medical-safety design; do NOT burn quota for them.
    # Pre-fix, batch committed all N items including uncertain → users with
    # mixed batches paid for failed-safety items they could re-submit for free
    # via single /ocr.
    n_committed = sum(1 for r in out if r.status != "uncertain")
    for _ in range(n_committed):
        commit_monthly_call(identity)
    total_unk_batch = sum(r.total_unk for r in out)
    uncertain = sum(1 for r in out if r.status == "uncertain")
    response.headers["X-Elapsed-Ms"] = f"{(_time.perf_counter() - t0) * 1000:.1f}"
    if total_unk_batch:
        response.headers["X-Unk-Count"] = str(total_unk_batch)
    if uncertain:
        response.headers["X-Uncertain-Items"] = str(uncertain)
    return out
