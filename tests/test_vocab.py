from spinaiocr.vocab.base import load_vocab


def test_ko_en_vocab_roundtrip():
    v = load_vocab("ko_en_v1")
    text = "안녕 Hello 123!"
    ids = v.encode(text)
    # decode without CTC collapse should reproduce the input characters
    # (collapse is only safe when the model emits blanks between repeats)
    decoded = "".join(v._itoc[i] for i in ids)
    assert decoded == text


def test_ko_en_vocab_unk():
    v = load_vocab("ko_en_v1")
    # a character outside precomposed Hangul + ASCII + extra set
    # use an emoji which we don't include
    ids = v.encode("한글🙂")
    assert v.unk_id in ids


def test_ctc_collapse_removes_repeats_and_blanks():
    v = load_vocab("ko_en_v1")
    a_id = v.encode("a")[0]
    b_id = v.encode("b")[0]
    seq = [a_id, a_id, v.blank_id, a_id, b_id, b_id]
    assert v.decode(seq, ctc_collapse=True) == "aab"
