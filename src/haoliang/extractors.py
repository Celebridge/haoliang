"""Term candidate extraction adapters.

The rest of the system depends on `TermExtractor`, not on jieba.
Add HanLP / spaCy / LLM by implementing the same protocol:

    class HanLPExtractor:
        name = "hanlp"
        def extract(self, text, *, language="zh"): ...

`annotate` only needs `extract()`; swapping the adapter does not change callers.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

CANDIDATE_POS = {"noun", "proper_noun", "unknown"}


@dataclass(frozen=True)
class TermCandidate:
    """A candidate term span.

    `pos` is normalized to a small set so adapters can differ internally:
    noun / proper_noun / verb / adj / unknown / other.
    """

    text: str
    start: int
    end: int
    pos: str
    source: str
    confidence: float = 0.5


@runtime_checkable
class TermExtractor(Protocol):
    name: str

    def extract(self, text: str, *, language: str = "zh") -> list[TermCandidate]:
        ...


def normalize_pos(flag: str) -> str:
    """Map jieba / HanLP / spaCy style tags to a small POS enum."""
    flag = (flag or "").lower()
    if flag in {"nr", "ns", "nt", "nz", "propn", "proper_noun"}:
        return "proper_noun"
    if flag.startswith("n") or flag == "noun":
        return "noun"
    if flag.startswith("v") or flag == "verb":
        return "verb"
    if flag.startswith("a") or flag == "adj":
        return "adj"
    if flag in {"x", "eng", "unk", "unknown"}:
        return "unknown"
    return "other"


class JiebaExtractor:
    """jieba adapter: Chinese segmentation + POS tagging.

    Only nouns, proper nouns, and unknown/English tokens are returned.
    Verbs, adjectives, pronouns, particles, and punctuation are dropped.
    """

    name = "jieba"

    def __init__(self, user_words: Iterable[str] | None = None) -> None:
        try:
            with warnings.catch_warnings():
                # jieba 0.42 uses invalid escape sequences; Python 3.12+ warns.
                warnings.simplefilter("ignore", SyntaxWarning)
                import jieba
                import jieba.posseg as pseg
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "jieba is not installed; run `uv pip install jieba`"
            ) from exc
        self._jieba = jieba
        self._pseg = pseg
        self.add_words(user_words or [])

    def add_words(self, words: Iterable[str]) -> None:
        """Seed jieba's dictionary with known terms/aliases.

        Without this, domain compounds like「安全点」are split into
        「安全 / 点」and never become candidates.
        """
        for word in words:
            token = str(word).strip()
            if not token:
                continue
            self._jieba.add_word(token, tag="n")

    def frequency(self, word: str) -> int | None:
        """jieba 词典里的词频；不在词典里返回 None。"""
        return self._jieba.get_FREQ(word)

    def is_common(self, word: str, threshold: int = 1000) -> bool:
        """高频词当作通用词，默认阈值 1000。"""
        freq = self.frequency(word)
        return freq is not None and freq >= threshold

    def is_compound_prefix(self, prefix: str) -> bool:
        """Whether `prefix` can start a technical compound.

        One jieba token, two CJK chars, and a content-word POS. This filters
        accidental spans like `余可` (two tokens) or `一轮` (measure word).
        """
        if len(prefix) != 2 or not all(
            "\u4e00" <= char <= "\u9fff" for char in prefix
        ):
            return False
        if prefix == "最左":
            return True
        tokens = list(self._pseg.cut(prefix))
        if len(tokens) != 1:
            return False
        flag = tokens[0].flag
        return flag.startswith("n") or flag in {"v", "vn", "a", "an"}

    def extract(self, text: str, *, language: str = "zh") -> list[TermCandidate]:
        if language != "zh":
            raise ValueError("JiebaExtractor supports language='zh' only")

        candidates: list[TermCandidate] = []
        cursor = 0
        for word, flag in self._pseg.cut(text):
            token = word.strip()
            start = text.find(word, cursor)
            if start < 0:
                start = cursor
            end = start + len(word)
            cursor = end

            if not token or token.isspace():
                continue
            if all(not char.isalnum() and char != "_" for char in token):
                continue

            pos = normalize_pos(flag)
            if pos not in CANDIDATE_POS:
                continue

            candidates.append(
                TermCandidate(
                    text=token,
                    start=start,
                    end=end,
                    pos=pos,
                    source=self.name,
                    confidence=0.6 if pos != "unknown" else 0.4,
                )
            )
        return candidates


def make_extractor(name: str = "jieba") -> TermExtractor:
    if name == "jieba":
        return JiebaExtractor()
    raise ValueError(f"unknown extractor: {name}")
