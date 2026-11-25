"""Extractor adapter tests: jieba baseline + protocol."""

from __future__ import annotations

import pytest

from haoliang.extractors import (
    JiebaExtractor,
    TermExtractor,
    make_extractor,
    normalize_pos,
)


def test_normalize_pos():
    assert normalize_pos("n") == "noun"
    assert normalize_pos("nr") == "proper_noun"
    assert normalize_pos("nz") == "proper_noun"
    assert normalize_pos("v") == "verb"
    assert normalize_pos("a") == "adj"
    assert normalize_pos("x") == "unknown"
    assert normalize_pos("eng") == "unknown"
    assert normalize_pos("uj") == "other"


def test_make_extractor_returns_protocol():
    pytest.importorskip("jieba")
    extractor = make_extractor("jieba")
    assert isinstance(extractor, TermExtractor)


def test_jieba_extracts_nouns_and_keeps_offsets():
    pytest.importorskip("jieba")
    extractor = JiebaExtractor(user_words=["安全点"])
    text = "safepoint 是 JVM 的安全点，GC 会在这里停顿。"
    candidates = extractor.extract(text)

    assert candidates
    for candidate in candidates:
        assert text[candidate.start : candidate.end] == candidate.text

    terms = {candidate.text for candidate in candidates}
    assert "safepoint" in terms
    assert "JVM" in terms
    assert "安全点" in terms


def test_jieba_drops_verbs_and_punctuation():
    pytest.importorskip("jieba")
    extractor = JiebaExtractor()
    candidates = extractor.extract("我们来看看这个实现。")
    terms = {candidate.text for candidate in candidates}

    assert "看看" not in terms
    assert "。" not in terms
