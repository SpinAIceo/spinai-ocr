"""Live smoke test: runs the full pipeline end-to-end, measures timings,
and validates core API. Writes a log file for the /loop iteration."""
import time, json, sys, traceback, os
from pathlib import Path

os.environ.setdefault("SPINAI_LOG_LEVEL", "INFO")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
iter_log = LOG_DIR / "live_test_iter.jsonl"

def log(ev, **kw):
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "ev": ev, **kw}
    with iter_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)

errors = []
timings = {}

def stage(name):
    def deco(fn):
        def wrap(*a, **kw):
            t0 = time.perf_counter()
            try:
                r = fn(*a, **kw)
                timings[name] = time.perf_counter() - t0
                log(f"{name}.ok", dt_ms=round(timings[name]*1000, 1))
                return r
            except Exception as e:
                tb = traceback.format_exc()
                errors.append((name, str(e), tb))
                log(f"{name}.err", err=str(e), dt_ms=round((time.perf_counter()-t0)*1000, 1))
                raise
        return wrap
    return deco

@stage("import_pipeline")
def _import():
    from spinai_ocr.inference.pipeline import OCRPipeline
    return OCRPipeline

@stage("build_image")
def _build_img():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    W, H = 640, 180
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 32)
    except Exception:
        font = ImageFont.load_default()
    d.text((20, 20), "커피 4,500원 케이크 6,000원", fill=(0, 0, 0), font=font)
    d.text((20, 90), "전화번호 010-1234-5678", fill=(0, 0, 0), font=font)
    return np.array(img)

@stage("load_pipeline")
def _load(Pipeline):
    return Pipeline()

@stage("run_once")
def _run(pipe, img):
    return pipe(img)

@stage("run_warm")
def _warm(pipe, img, n=5):
    for _ in range(n):
        out = pipe(img)
    return out

def main():
    try:
        OCRPipeline = _import()
        img = _build_img()
        pipe = _load(OCRPipeline)
        r1 = _run(pipe, img)
        # Warm bench
        t0 = time.perf_counter()
        _warm(pipe, img, 5)
        per_call = (time.perf_counter() - t0) / 5
        log("bench.per_call", per_call_ms=round(per_call*1000, 1))

        # Extract text structure — handle OCRResult, dict, or str
        texts = []
        lines_attr = getattr(r1, "lines", None)
        if lines_attr is not None:
            for ln in lines_attr:
                t = getattr(ln, "text", None)
                if t:
                    texts.append(t)
        elif isinstance(r1, dict):
            for rec in r1.get("records", []) or []:
                if isinstance(rec, dict) and rec.get("text"):
                    texts.append(rec["text"])
        combined = " | ".join(texts) if texts else str(r1)[:200]
        log("output", text=combined, n_records=len(texts))

        # Lightweight CER vs expected GT (synth is known)
        expected = "커피 4,500원 케이크 6,000원 전화번호 010-1234-5678"
        from difflib import SequenceMatcher
        predicted = " ".join(texts)
        sim = SequenceMatcher(None, expected, predicted).ratio() if predicted else 0.0
        log("quality", similarity=round(sim, 3),
            expected_len=len(expected), got_len=len(predicted))

    except Exception as e:
        log("fatal", err=str(e))
    log("done", errors=len(errors), timings_ms={k: round(v*1000,1) for k,v in timings.items()})
    return errors

if __name__ == "__main__":
    errs = main()
    sys.exit(0 if not errs else 1)
