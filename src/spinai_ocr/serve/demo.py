"""Streamlit demo UI.

Run:
    streamlit run -m spinai_ocr.serve.demo
or:
    streamlit run src/spinai_ocr/serve/demo.py

Shows side-by-side: uploaded image with detection boxes overlaid + recognized
text (raw vs LLM-corrected if the correction toggle is on).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from spinai_ocr.config import PipelineConfig
from spinai_ocr.inference.pipeline import OCRPipeline


def _draw_boxes(pil: Image.Image, result) -> Image.Image:
    out = pil.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for line in result.lines:
        pts = [tuple(p) for p in line.bbox]
        draw.polygon(pts, outline="red", width=2)
        draw.text(pts[0], line.text[:30], fill="red")
    return out


def main() -> None:  # pragma: no cover — UI
    import streamlit as st  # type: ignore

    st.set_page_config(page_title="SPINAI OCR Demo", layout="wide")
    st.title("SPINAI OCR — Demo")
    st.caption("Korean-first open-source OCR. Upload an image to try it.")

    # iter 136: align demo selectors with what actually ships. Pre-fix listed
    # "standard"/"large" tiers (not installed → silent empty output) and
    # omitted consumer_v1 (the prod /ocr default per app.py) + medical.
    # Only "ko" has rec checkpoints; other LangCode values are kept for
    # forward-compat but degrade gracefully via the no-ckpt info banner.
    lang = st.sidebar.selectbox("Language", ["ko", "en", "ja", "zh", "multi"], 0)
    tier = st.sidebar.selectbox("Model tier", ["consumer_v1", "lite", "medical"], 0)
    use_llm = st.sidebar.checkbox("LLM post-correction (Claude Haiku)", value=False)
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Installed checkpoints: lite, consumer_v1, medical (Korean only). "
        "Other (lang, tier) combos fall through with 'no lines'."
    )

    file = st.file_uploader("Image", type=["png", "jpg", "jpeg", "bmp", "tiff", "tif"])
    if file is None:
        return

    pil = Image.open(io.BytesIO(file.read())).convert("RGB")
    cfg = PipelineConfig(lang=lang, tier=tier, use_llm_postprocess=use_llm)  # type: ignore[arg-type]
    pipe = OCRPipeline(config=cfg)
    with st.spinner("Running OCR..."):
        result = pipe(pil)

    left, right = st.columns(2)
    with left:
        st.subheader("Detection")
        st.image(_draw_boxes(pil, result), use_column_width=True)
    with right:
        st.subheader("Recognition")
        if not result.lines:
            st.info("No text detected (or no checkpoints loaded).")
        else:
            for line in result.lines:
                st.write(f"- **{line.text}** ({line.confidence:.2f})")

    with st.expander("Raw JSON result"):
        import json

        st.code(
            json.dumps(
                {
                    "lines": [
                        {"text": l.text, "bbox": l.bbox, "confidence": l.confidence}
                        for l in result.lines
                    ],
                    "image_width": result.image_width,
                    "image_height": result.image_height,
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )


if __name__ == "__main__":
    main()
