"""Convert public OCR dataset formats into SPINAI's internal JSONL schema.

Internal detection JSONL (one record per line):
    {"image": "path/rel/to/root.jpg",
     "words": [{"points": [[x,y],[x,y],[x,y],[x,y]], "text": "..."}]}

Internal recognition TSV (one line):
    image_path\tTEXT
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _write_jsonl(records: Iterable[dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# ICDAR 2015 / 2019 MLT format
# ---------------------------------------------------------------------------


_ICDAR_LINE = re.compile(r"^(.+?)$")


def convert_icdar(images_root: Path, gt_root: Path, out_jsonl: Path) -> int:
    """ICDAR GT: one .txt per image, each line is
       x1,y1,x2,y2,x3,y3,x4,y4,transcription
    Missing/illegible transcriptions are '###' → saved as ignore.
    """
    records = []
    for gt in sorted(gt_root.glob("*.txt")):
        stem = gt.stem.replace("gt_", "", 1)
        img_candidates = list(images_root.glob(f"{stem}.*"))
        if not img_candidates:
            continue
        words = []
        for line in gt.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            parts = line.strip().split(",", 8)
            if len(parts) < 9:
                continue
            coords = [float(x) for x in parts[:8]]
            text = parts[8]
            words.append(
                {
                    "points": [
                        [coords[0], coords[1]],
                        [coords[2], coords[3]],
                        [coords[4], coords[5]],
                        [coords[6], coords[7]],
                    ],
                    "text": text,
                }
            )
        records.append({"image": img_candidates[0].name, "words": words})
    return _write_jsonl(records, out_jsonl)


# ---------------------------------------------------------------------------
# COCO-Text v2 format
# ---------------------------------------------------------------------------


def convert_coco_text(coco_json: Path, out_jsonl: Path) -> int:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data.get("imgs", {}).values()}
    anns_by_img: dict[int, list[dict]] = {}
    for ann in data.get("anns", {}).values():
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    records = []
    for img_id, img in images.items():
        words = []
        for a in anns_by_img.get(img_id, []):
            bbox = a.get("bbox")
            if not bbox:
                continue
            x, y, w, h = bbox
            words.append(
                {
                    "points": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    "text": a.get("utf8_string", ""),
                }
            )
        records.append({"image": img.get("file_name", ""), "words": words})
    return _write_jsonl(records, out_jsonl)


# ---------------------------------------------------------------------------
# AI Hub Korean OCR JSON format (common fields; varies by dataset)
# ---------------------------------------------------------------------------


def convert_aihub_json_dir(json_dir: Path, out_jsonl: Path) -> int:
    """AI Hub typically ships one JSON per image with keys like:
        {"image": {"file_name": "...", "width": ..., "height": ...},
         "annotations": [{"bbox": [x,y,w,h], "text": "...", "illegible": 0}]}
    or sometimes the COCO-ish nested dict. We defensively accept both shapes.
    """
    records = []
    for jp in sorted(json_dir.rglob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        img_info = data.get("image", {}) if isinstance(data.get("image"), dict) else {}
        file_name = img_info.get("file_name") or data.get("images", [{}])[0].get("file_name", "")
        anns_raw = data.get("annotations") or data.get("anns") or []
        words = []
        for a in anns_raw:
            if a.get("illegible"):
                continue
            bbox = a.get("bbox") or a.get("box")
            pts = a.get("points") or a.get("polygon")
            if pts and len(pts) >= 4:
                points = [list(map(float, p)) for p in pts]
            elif bbox and len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            else:
                continue
            words.append({"points": points, "text": a.get("text", "")})
        if file_name:
            records.append({"image": file_name, "words": words})
    return _write_jsonl(records, out_jsonl)


# ---------------------------------------------------------------------------
# Recognition TSV export (crops from detection labels)
# ---------------------------------------------------------------------------


@dataclass
class CropSpec:
    min_height: int = 8
    min_text_length: int = 1


def jsonl_to_recognition_tsv(
    jsonl: Path,
    images_root: Path,
    out_dir: Path,
    cfg: CropSpec | None = None,
) -> int:
    """Crop each polygon to an image file + append to labels.tsv."""
    import cv2
    import numpy as np

    cfg = cfg or CropSpec()
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "crops"
    img_dir.mkdir(exist_ok=True)
    labels = out_dir / "labels.tsv"

    count = 0
    with jsonl.open("r", encoding="utf-8") as fin, labels.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            img_path = images_root / rec["image"]
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            for idx, w in enumerate(rec.get("words", [])):
                text = w.get("text", "").strip()
                if len(text) < cfg.min_text_length or text == "###":
                    continue
                pts = np.array(w["points"], dtype=np.float32)
                widths = np.linalg.norm(pts[0] - pts[1])
                heights = np.linalg.norm(pts[0] - pts[3])
                if heights < cfg.min_height:
                    continue
                dst = np.array(
                    [[0, 0], [widths, 0], [widths, heights], [0, heights]], dtype=np.float32
                )
                M = cv2.getPerspectiveTransform(pts, dst)
                crop = cv2.warpPerspective(img, M, (int(widths), int(heights)))
                name = f"{Path(rec['image']).stem}_{idx:04d}.png"
                cv2.imwrite(str(img_dir / name), crop)
                fout.write(f"crops/{name}\t{text}\n")
                count += 1
    return count
