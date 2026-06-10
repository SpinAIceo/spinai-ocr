# SPINAI OCR

> **The fastest, smallest Korean OCR.** ~80 MB models · ~14 ms/line on CPU · fully offline. No GPU, no cloud.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![PyPI](https://img.shields.io/badge/pip-spinai--ocr-blue.svg)](https://pypi.org/project/spinai-ocr/)

A Korean-first OCR engine built to run **on a CPU, offline, in a tiny footprint** —
for on-prem, air-gapped, and edge deployments where you can't ship data to a cloud
API or run a GPU.

**🔴 Live demo:** https://spinai-ocr-production.up.railway.app — try it in your browser, no install.

---

## Why SPINAI OCR?

PaddleOCR and EasyOCR are excellent multilingual giants. Naver CLOVA and PaddleOCR-VL
are more accurate on hard Korean text. **SPINAI is the opposite trade-off:** a small,
fast, fully-offline Korean engine for when accuracy-at-any-cost isn't the constraint —
*data residency, latency, and footprint* are.

| | **SPINAI** | EasyOCR | PaddleOCR | Naver CLOVA |
|---|---|---|---|---|
| Model size | **~80 MB** | ~100 MB+ | ~100 MB+ | cloud |
| CPU latency / line | **~14 ms** | ~50 ms | varies | network RTT |
| Offline / no GPU | **✅** | ✅ | partial | ❌ cloud-only |
| Docker image | **<100 MB** | larger | larger | n/a |
| Korean accuracy | competitive | competitive | higher | highest |

> **Honest about accuracy:** if you need maximum accuracy, use PaddleOCR-VL or Naver
> CLOVA. If you need Korean OCR that runs **offline on a CPU in <100 MB**, that's us.

## Install

```bash
pip install git+https://github.com/SpinAIceo/spinai-ocr.git
# PyPI release coming soon: pip install spinaiocr
```

The Korean model (~80 MB) is downloaded automatically on first run, then everything
runs offline. No manual setup — the model is hosted on HuggingFace (spinaiceo/spinai-ocr-consumer-v1).

## Quickstart

```python
from spinaiocr import OCRPipeline

ocr = OCRPipeline(lang="ko")          # CPU by default, fully offline
result = ocr("your_image.jpg")
for line in result.lines:
    print(f"{line.confidence:.2f}  {line.text}")
```

More in [`examples/`](examples/). REST API + web UI also included (`spinai_ocr.serve`).

## How it works

- **Detection:** DBNet (text-region segmentation)
- **Recognition:** SVTRv2 + CTC (compact, CPU-friendly, no autoregression)
- **Languages:** Korean (한국어) first-class; English supported

## Honest limitations

- Accuracy is **competitive, not #1** — PaddleOCR-VL / CLOVA win on decorative fonts,
  handwriting, and complex scenes.
- Detection can merge regions on cluttered real-world scenes.
- This is the **speed/size/offline** OCR. We publish where we lose, not just where we win.

## License

Apache-2.0. Built with ❤️ for the Korean OCR community. Contributions welcome.
