"""SPINAI OCR — 3-line quickstart.

First run downloads the Korean model (~80 MB) into ./checkpoints. After that it
runs fully offline on CPU. See README for the REST API + web demo.
"""
from spinaiocr import OCRPipeline

ocr = OCRPipeline(lang="ko")          # CPU by default
result = ocr("path/to/your_image.jpg")

for line in result.lines:
    print(f"{line.confidence:.2f}  {line.text}")
