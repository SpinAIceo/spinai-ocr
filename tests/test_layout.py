import numpy as np

from spinaiocr.layout.analyzer import LayoutAnalyzer


def _bbox(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def test_layout_two_lines():
    polys = [_bbox(10, 20, 200, 18), _bbox(10, 60, 200, 18)]
    texts = ["첫 번째 줄", "두 번째 줄"]
    result = LayoutAnalyzer().analyze((200, 400), polys, texts)
    assert len(result.regions) == 2
    assert result.regions[0].text == "첫 번째 줄"
    assert result.regions[1].order == 1


def test_layout_title_detected():
    polys = [_bbox(10, 10, 300, 40), _bbox(10, 80, 300, 15)]
    texts = ["큰 제목", "본문 한 줄"]
    result = LayoutAnalyzer().analyze((400, 400), polys, texts)
    assert result.regions[0].kind == "title"


def test_markdown_output():
    polys = [_bbox(0, 0, 200, 30)]
    texts = ["안녕하세요"]
    md = LayoutAnalyzer().analyze((400, 400), polys, texts).to_markdown()
    assert "안녕하세요" in md
