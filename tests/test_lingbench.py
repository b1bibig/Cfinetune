"""언어력 벤치의 데이터 파싱과 통계 (모델·네트워크 없이)."""

import re

from cfinetune import lingbench as lb


def test_parse_kobalt_splits_question_and_ten_choices():
    q = "지문:\n가나다\n\n문제: 고르시오.\n\n" + "\n".join(f"{L}: 보기{L}" for L in lb.LETTERS) + "\n"
    item = lb.parse_kobalt({"Question": q, "Answer": "H", "Class": "Syntax"})
    assert item["question"].endswith("고르시오.")
    assert item["choices"] == [f"보기{L}" for L in lb.LETTERS]
    assert item["answer"] == 7


def test_pairs_file_is_well_formed():
    pairs = lb.load_pairs()
    assert len(pairs) == 210
    cats = {p["cat"] for p in pairs}
    assert len(cats) == 15
    assert all(sum(p["cat"] == c for p in pairs) == 14 for c in cats)
    assert all(p["natural"] != p["translationese"] for p in pairs)


def test_translationese_side_has_its_marker():
    """번역투 문장에는 그 유형의 표지가 실제로 들어 있어야 한다 (자연문보다 많이)."""
    for p in lb.load_pairs():
        pat = lb.MARKERS.get(p["cat"])
        if pat:
            assert len(re.findall(pat, p["translationese"])) > len(re.findall(pat, p["natural"])), p


def test_register_of_endings():
    assert [lb.register_of(w) for w in ("했습니다", "좋아요", "먹었어", "가자", "됩니까", "하세요")] == \
        ["polite", "polite", "casual", "casual", "polite", "polite"]
    assert lb.sentence_endings("안녕! 잘 지냈어? 나는 잘 지내.\n고마워요 😊") == ["안녕", "지냈어", "지내", "고마워요"]


def test_text_stats_counts():
    s = lb.text_stats("그녀는 책을 통해 배웠다. 漢字 ABC 그녀는 책을 통해 배웠다.")
    assert s["markers"]["pronoun"] == 2 and s["markers"]["through"] == 2
    assert 0 < s["han_ratio"] < s["hangul_ratio"]
    assert s["rep4"] > 0


def test_wilson_interval_contains_estimate():
    lo, hi = lb.wilson(70, 100)
    assert lo < 0.7 < hi and 0.6 < lo and hi < 0.8
