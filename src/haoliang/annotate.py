"""Annotate a draft against the reader ledger.

Pipeline:

    text
      → dictionary match (ledger term + aliases)
      → extractor (jieba / HanLP / ...)
      → drop overlapping spans (prefer ledger / longer span)
      → merge connectors (otel + collector → otel-collector)
      → drop fragments, single CJK, common words
      → match ledger
      → mark known / unknown / not_in_ledger
      → flag undefined terms

This is a linter, not a generator. It does not rewrite the draft.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .concepts import (
    concepts_for_term,
    concepts_path,
    derive_domain_profiles,
    load_concepts,
)
from .extractors import TermCandidate, TermExtractor, make_extractor
from .glossary import (
    glossary_path,
    load_local_terms,
    new_local_term,
    next_local_id,
    save_local_terms,
    upsert_local_term,
)
from .jargon import JargonJudge, sentence_for
from .reader_config import ReaderConfig, load_config
from .term_store import (
    TermIndex,
    edit_terms,
    find_by_name_exact,
    load_terms,
    next_id,
    term_names,
    upsert,
)
from .terms import add_pending_review, familiarity, new_term

CONNECTORS = {"-", "_", "/", "."}
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
FRAGMENT_RE = re.compile(r"^(?:[A-Za-z]|\d+(?:\.\d+)?[A-Za-z%]?)$")
ENGLISH_TOKEN_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9+#]*(?: [A-Za-z][A-Za-z0-9+#]*)*$"
)
# Function words that should never be glued into a multi-word candidate.
# Unlike COMMON_EN, this does not include words like "full" or "new", which
# are legitimate parts of terms such as "Full GC" / "new generation".
ENGLISH_MERGE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "do",
    "does",
    "did",
    "for",
    "from",
    "had",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "not",
    "of",
    "on",
    "or",
    "so",
    "than",
    "that",
    "the",
    "then",
    "this",
    "to",
    "via",
    "was",
    "were",
    "when",
    "where",
    "which",
    "with",
}
COMMON_EN = {
    "full",
    "young",
    "old",
    "major",
    "minor",
    "new",
    "the",
    "and",
    "for",
    "with",
    "unknown",
    "pending",
    "running",
    "ready",
    "true",
    "false",
    "none",
    "null",
    "default",
}
COMMON_ZH = {"内层", "外层", "双向", "时会"}
# Words that belong to the harness, never to an answer shown to a reader.
LEAK_WORDS = {
    "账本": "内部账本",
    "brief": "工具名",
    "coverage": "工具字段",
    "domain_profile": "工具名",
    "familiarity": "内部字段",
    "heard": "内部等级",
    "proposal": "内部状态",
    "候选词": "内部术语",
    "覆盖不足": "元说明",
    "给你的建议": "元说明",
    "按资深": "元说明",
    "不据此判断": "元说明",
}
LEAK_PHRASES = (
    "级证据",
    "只是 heard",
    "known 为",
    "unknown 为",
    "本题命中",
    "本次 brief",
    "下面是正文",
    "不属于正文",
    "给 AI 的说明",
    # 对话式元评论：把读者当成对话对象，而不是读者
    "你背过",
    "你之前",
    "你说不清",
    "你肯定",
    "答这道题时",
    "能显出",
    "面试教练",
)
# 首行只写标签的占位残留，例如 `A.` / `正文` / `回答`
PLACEHOLDER_LINES = {"A.", "A", "B.", "正文", "回答", "答案", "以下是", "正文如下"}
FRAGMENT_EDGE_CHARS = set(
    "否会拿指太中于的了是在和与及或对把被为从到里外内后前时"
    "而就才可能要让给跟向往第每各某这那其此该本"
)
COMPOUND_SUFFIXES = (
    "索引",
    "查询",
    "回收",
    "标记",
    "预测",
    "模型",
    "机制",
    "策略",
    "协议",
    "算法",
    "结构",
    "服务",
    "集群",
    "节点",
    "容器",
    "线程",
    "事务",
    "缓存",
    "队列",
    "延迟",
    "阈值",
    "参数",
    "配置",
    "监控",
    "日志",
    "指标",
    "故障",
    "恢复",
    "扩容",
    "缩容",
    "限流",
    "熔断",
    "降级",
    "重试",
    "前缀",
)
COMPOUND_PREFIX_STOPWORDS = {
    "这个",
    "那个",
    "一个",
    "一种",
    "所有",
    "每个",
    "上述",
    "如下",
    "以下",
    "这种",
    "那种",
    "哪些",
    "什么",
    "怎么",
    "如何",
    "超过",
    "触发",
    "谈调",
    "完成",
    "进行",
    "使用",
    "实现",
    "包括",
    "支持",
    "提供",
    "导致",
    "产生",
    "发生",
    "出现",
    "开始",
    "结束",
    "继续",
    "需要",
    "可以",
    "能够",
    "应该",
    "必须",
    "建议",
    "避免",
    "减少",
    "增加",
    "提高",
    "降低",
    "检查",
    "确认",
    "处理",
    "执行",
    "调用",
    "返回",
    "设置",
    "获取",
    "判断",
    "选择",
    "计算",
    "记录",
    "保存",
    "加载",
    "读取",
    "写入",
    "更新",
    "删除",
    "创建",
    "启动",
    "停止",
    "等待",
    "说明",
    "表示",
    "认为",
    "发现",
    "看到",
    "知道",
    "了解",
    "分析",
    "比较",
    "测试",
    "验证",
    "保证",
    "确保",
}
COMPOUND_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
CODE_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:/@+-"
)


def _is_single_cjk(text: str) -> bool:
    return len(text) == 1 and bool(CJK_RE.search(text))


def _is_fragment(text: str) -> bool:
    if text in CONNECTORS:
        return True
    if FRAGMENT_RE.fullmatch(text):
        return True
    if text in COMMON_ZH:
        return True
    if text and (text[0] in FRAGMENT_EDGE_CHARS or text[-1] in FRAGMENT_EDGE_CHARS):
        return True
    return text.casefold() in COMMON_EN


def _is_english_token(text: str) -> bool:
    return bool(ENGLISH_TOKEN_RE.fullmatch(text))


def _in_code_span(text: str, start: int, end: int) -> bool:
    """True when the candidate sits inside a Markdown `code span`."""
    return text.count("`", 0, start) % 2 == 1 and text.find("`", end) >= 0


def _is_code_fragment(text: str, start: int, end: int) -> bool:
    """True when a candidate is only part of a larger code-like token."""
    if _in_code_span(text, start, end):
        return True
    left = start
    while left > 0 and text[left - 1] in CODE_CHARS:
        left -= 1
    right = end
    while right < len(text) and text[right] in CODE_CHARS:
        right += 1
    token = text[left:right]
    if (left, right) == (start, end):
        return False
    if any(char in token for char in "-@:./_"):
        return True
    if any(char.isdigit() for char in token):
        return True
    if token.isascii() and token != text[start:end]:
        return True
    # Only apply the camelCase test to ASCII tokens. `MySQL索引` is a Latin
    # word glued to Chinese, not a code fragment, and must still be found.
    return (
        token.isascii()
        and token != token.lower()
        and token != token.upper()
    )


def ledger_candidates(
    text: str,
    terms: list[dict],
    *,
    index: TermIndex | None = None,
) -> list[TermCandidate]:
    """Find known terms/aliases directly in the raw text.

    This is the high-precision path: it works even when the tokenizer splits
    a term (e.g.「Spring 事务」or「B+ 树」).

    With a `TermIndex`, matching is one automaton pass over the text instead
    of scanning every card × alias.
    """
    candidates: list[TermCandidate] = []
    if index is not None:
        for start, end, _key in index.iter_matches(text):
            candidates.append(
                TermCandidate(
                    text=text[start:end],
                    start=start,
                    end=end,
                    pos="noun",
                    source="ledger",
                    confidence=1.0,
                )
            )
        return candidates

    lower_text = text.casefold()
    for card in terms:
        names = [str(card.get("term", ""))]
        aliases = card.get("aliases") or []
        if isinstance(aliases, list):
            names.extend(str(alias) for alias in aliases)
        for name in names:
            name = name.strip()
            if not name:
                continue
            needle = name.casefold()
            start = lower_text.find(needle)
            while start >= 0:
                candidates.append(
                    TermCandidate(
                        text=text[start : start + len(name)],
                        start=start,
                        end=start + len(name),
                        pos="noun",
                        source="ledger",
                        confidence=1.0,
                    )
                )
                start = lower_text.find(needle, start + 1)
    return candidates


def _drop_overlaps(candidates: list[TermCandidate]) -> list[TermCandidate]:
    """Prefer ledger spans and longer spans; drop covered fragments."""
    ordered = sorted(
        candidates,
        key=lambda c: (
            c.start,
            -(c.end - c.start),
            0 if c.source == "ledger" else 1,
        ),
    )
    kept: list[TermCandidate] = []
    for candidate in ordered:
        # Starts are sorted and retained intervals never overlap.
        if kept and candidate.start < kept[-1].end:
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda c: c.start)


def merge_candidates(text: str, candidates: list[TermCandidate]) -> list[TermCandidate]:
    """Merge adjacent candidates separated by a connector.

    jieba splits `otel-collector` into `otel` + `collector`; this puts it back.
    It may also emit the connector itself as a token (`REQUIRES` + `_` + `NEW`),
    so connector-only candidates are dropped before merging.
    """
    candidates = [
        candidate
        for candidate in candidates
        if candidate.text.strip() not in CONNECTORS
    ]

    merged: list[TermCandidate] = []
    for candidate in candidates:
        if not merged:
            merged.append(candidate)
            continue

        previous = merged[-1]
        gap = text[previous.end : candidate.start]
        if (
            (previous.source == "ledger" and candidate.source == "ledger")
            or gap.strip() not in CONNECTORS
            or gap != gap.strip()
        ):
            merged.append(candidate)
            continue

        merged[-1] = TermCandidate(
            text=text[previous.start : candidate.end],
            start=previous.start,
            end=candidate.end,
            pos=(
                "unknown"
                if "unknown" in {previous.pos, candidate.pos}
                else previous.pos
            ),
            source="compound",
            confidence=min(previous.confidence, candidate.confidence),
        )
    return merged


def merge_english_phrases(
    text: str,
    candidates: list[TermCandidate],
) -> list[TermCandidate]:
    """Merge adjacent English tokens separated by one space.

    `tree shaking`, `source map`, `ES module`, `Spring AOP`, `Full GC` are one
    term each, but jieba emits them as separate tokens.
    """
    merged: list[TermCandidate] = []
    for candidate in candidates:
        if not merged:
            merged.append(candidate)
            continue

        previous = merged[-1]
        gap = text[previous.end : candidate.start]
        if (
            previous.source == "ledger"
            and candidate.source == "ledger"
        ):
            merged.append(candidate)
            continue
        merged_text = text[previous.start : candidate.end]
        in_code = _in_code_span(text, previous.start, candidate.end)
        if (
            gap == " "
            and _is_english_token(previous.text)
            and _is_english_token(candidate.text)
            and (
                in_code
                or (
                    previous.text.casefold() not in ENGLISH_MERGE_STOPWORDS
                    and candidate.text.casefold() not in ENGLISH_MERGE_STOPWORDS
                    # At most three tokens: `tree shaking`, `ES module`, `Full GC`.
                    and merged_text.count(" ") < 2
                )
            )
        ):
            merged[-1] = TermCandidate(
                text=merged_text,
                start=previous.start,
                end=candidate.end,
                pos="noun",
                source="phrase",
                confidence=min(previous.confidence, candidate.confidence),
            )
        else:
            merged.append(candidate)
    return merged


def compound_candidates(
    text: str,
    prefix_check=None,
) -> list[TermCandidate]:
    """Propose Chinese compounds ending in a technical suffix.

    jieba tags `联合` as a verb and `索引` as a proper noun, so the usual
    noun-only extraction never sees `联合索引`. This is a conservative
    suffix scan: 1-2 CJK chars before a known suffix, skipping function-word
    prefixes. `prefix_check` can reject accidental spans such as `余可`.
    The judge still decides whether the compound matters.
    """
    found: dict[tuple[int, int], str] = {}
    function_chars = set("的了是在和与及或对把被为从到中上下里外内后前时")
    for suffix in COMPOUND_SUFFIXES:
        cursor = 0
        while True:
            index = text.find(suffix, cursor)
            if index < 0:
                break
            for prefix_len in (2, 1):
                begin = index - prefix_len
                if begin < 0:
                    continue
                prefix = text[begin:index]
                if not all(COMPOUND_CJK_RE.match(char) for char in prefix):
                    continue
                if prefix in COMPOUND_PREFIX_STOPWORDS:
                    break
                if any(char in function_chars for char in prefix):
                    break
                # `is_compound_prefix` is a jieba POS check for two-character
                # prefixes; applying it to one character would reject every
                # one-character prefix and make that branch dead.
                if (
                    prefix_check is not None
                    and prefix_len == 2
                    and not prefix_check(prefix)
                ):
                    break
                found[(begin, index + len(suffix))] = text[
                    begin : index + len(suffix)
                ]
                break
            cursor = index + len(suffix)
    return [
        TermCandidate(
            text=token,
            start=start,
            end=end,
            pos="noun",
            source="compound",
            confidence=0.45,
        )
        for (start, end), token in found.items()
    ]


def _starts_definition(text: str) -> bool:
    """Prefix match that rejects `是否`, `指向`, `即使`, `即将`."""
    for marker in (
        "：",
        ":",
        "是指",
        "指的是",
        "就是",
        "也就是",
        "这类",
        "这种",
        "是",
        "指",
        "即",
    ):
        if not text.startswith(marker):
            continue
        nxt = text[len(marker) : len(marker) + 1]
        if marker == "是" and nxt in {"否", "不"}:
            continue
        if marker == "指" and nxt in {"向", "标", "数", "令", "导"}:
            continue
        if marker == "即" and nxt in {"使", "将", "刻", "便", "时"}:
            continue
        return True
    return False


def _tail_is_definition(tail: str) -> bool:
    stripped = tail.lstrip()
    if _starts_definition(stripped):
        return True
    for open_char, close_char in (("（", "）"), ("(", ")")):
        if not stripped.startswith(open_char):
            continue
        end = stripped.find(close_char, 1)
        if end < 0:
            return False
        content = stripped[1:end]
        if CJK_RE.search(content):
            head = content.lstrip()
            return not (
                head
                and head[0].isascii()
                and head[0].isalpha()
                and not any(separator in content for separator in "，,")
            )
        after = stripped[end + 1 :].lstrip()
        if _starts_definition(after):
            return True
    return False


def _inside_parenthetical(text: str, start: int, end: int) -> bool:
    """True when [start, end) is an alias inside a definition parenthesis.

    `ICP（Index Condition Pushdown）` qualifies; `角色可变（Eden / Survivor）`
    and `（默认 8 次，G1MixedGCCountTarget）` do not.
    """
    open_pos = max(text.rfind("（", 0, start), text.rfind("(", 0, start))
    if open_pos < 0:
        return False
    if text.find("）", open_pos, start) >= 0 or text.find(")", open_pos, start) >= 0:
        return False
    close_pos = min(
        (pos for pos in (text.find("）", end), text.find(")", end)) if pos >= 0),
        default=-1,
    )
    if close_pos < 0:
        return False
    content = text[open_pos + 1 : close_pos]
    if len(content) > 200 or "/" in content:
        return False
    if CJK_RE.search(content) and not any(
        separator in content for separator in "，,"
    ):
        return False
    # The alias is the first token: `（Mixed GC）`,
    # `（ICP，Index Condition Pushdown）`, `（InitiatingHeapOccupancyPercent，...）`.
    head = content.lstrip()
    if head and head[0].isascii() and head[0].isalpha():
        return True
    # Or the parenthesis explains a Latin head: `initial-mark（初始标记，...）`.
    head_start = open_pos
    while head_start > 0 and (
        text[head_start - 1].isascii()
        and (text[head_start - 1].isalnum() or text[head_start - 1] in "_-")
    ):
        head_start -= 1
    latin_head = text[head_start:open_pos]
    return bool(latin_head) and any(char.isalpha() for char in latin_head)


def _find_term_occurrences(text: str, term: str):
    if all(char.isascii() for char in term):
        pattern = re.compile(
            r"(?<![A-Za-z0-9])"
            + re.escape(term)
            + r"(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            yield match.start()
        return
    folded_text, folded_term = text.casefold(), term.casefold()
    if len(folded_text) != len(text) or len(folded_term) != len(term):
        # Expanded case folds would make offsets into the source text wrong.
        for match in re.finditer(re.escape(term), text, re.IGNORECASE):
            yield match.start()
        return
    start = folded_text.find(folded_term)
    while start >= 0:
        yield start
        start = folded_text.find(folded_term, start + 1)


def _prose_occurrences(text: str, term: str):
    for start in _find_term_occurrences(text, term):
        if not _is_code_fragment(text, start, start + len(term)):
            yield start


def first_prose_occurrence(text: str, term: str) -> int | None:
    """Return the first occurrence outside inline code and larger identifiers."""
    return next(_prose_occurrences(text, term), None)


def _defined_at(text: str, start: int, end: int) -> bool:
    return _inside_parenthetical(text, start, end) or _tail_is_definition(
        text[end : end + 80]
    )


def _definition_status(text: str, term: str) -> tuple[bool, bool]:
    occurrences = _prose_occurrences(text, term)
    first = next(occurrences, None)
    if first is None:
        return False, False
    if _defined_at(text, first, first + len(term)):
        return True, False
    return False, any(_defined_at(text, start, start + len(term)) for start in occurrences)


def find_leaks(text: str) -> list[dict]:
    """Find harness words that leaked into a reader-facing draft.

    Checking is intentionally cheap and line-oriented: a leak is reported with
    its line number so a caller can show the author exactly where it is.
    """
    leaks: list[dict] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if line.strip() in PLACEHOLDER_LINES:
            leaks.append(
                {
                    "line": index + 1,
                    "hits": ["占位标签"],
                    "text": line.strip(),
                }
            )
        break
    for number, line in enumerate(lines, 1):
        lowered = line.casefold()
        hits = [
            word for word in LEAK_WORDS if word.casefold() in lowered
        ]
        hits.extend(
            phrase for phrase in LEAK_PHRASES if phrase.casefold() in lowered
        )
        if hits:
            leaks.append(
                {
                    "line": number,
                    "hits": sorted(set(hits)),
                    "text": line.strip()[:120],
                }
            )
    return leaks


def term_is_defined(text: str, term: str) -> bool:
    """Check whether a term is explained when first used in prose."""
    start = first_prose_occurrence(text, term)
    return start is not None and _defined_at(text, start, start + len(term))


def annotate(
    store: str | Path,
    text: str,
    *,
    domain: str | None = None,
    extractor: TermExtractor | None = None,
    index: TermIndex | None = None,
    judge: JargonJudge | None = None,
    record_reviews: bool = False,
    record_local_terms: bool = False,
    project: str | None = None,
    glossary_root: str | Path | None = None,
    include_sensitive: bool = False,
    now: datetime | None = None,
    config: ReaderConfig | None = None,
    min_freq: int = 1000,
    max_candidates: int = 50,
    replay_review_terms: frozenset[str] = frozenset(),
) -> dict:
    """Annotate one draft against the ledger.

    `domain` is the draft's context. It is a soft prior: a matching card is
    preferred, but a term known in another domain is still used as fallback.

    `index` can be reused across drafts to avoid rebuilding the automaton.

    `judge` only sees `not_in_ledger` candidates. `record_reviews=True` writes
    a `pending_reviews` entry when the judge returns `review`.
    `record_local_terms=True` writes project/model terms to the local glossary,
    never to the reader ledger.

    `min_freq` is the jieba dictionary frequency above which an unmatched
    candidate is treated as a common word and dropped.
    """
    cfg = config or load_config()
    if index is None:
        terms = load_terms(store)
        index = TermIndex(terms)
    else:
        terms = index.terms
    concept_cards = load_concepts(concepts_path(store))
    visible_terms = (
        terms
        if include_sensitive
        else [
            card
            for card in terms
            if card.get("sensitivity") != "sensitive"
        ]
    )
    domain_profile = None
    if judge is not None and domain:
        profiles = derive_domain_profiles(
            visible_terms,
            config=cfg,
            now=now,
            domain=domain,
            include_sensitive=include_sensitive,
        )
        domain_profile = profiles[0] if profiles else None
    extractor = extractor or make_extractor("jieba")

    # Seed the tokenizer with ledger terms/aliases. This helps segmentation
    # even though ledger_candidates() is the authoritative known-term path.
    # Reusing an extractor across drafts should not re-add 20k words each time.
    if hasattr(extractor, "add_words") and not getattr(
        extractor, "_haoliang_seeded", False
    ):
        vocabulary: list[str] = []
        for card in terms:
            vocabulary.append(str(card.get("term", "")))
            aliases = card.get("aliases") or []
            if isinstance(aliases, list):
                vocabulary.extend(str(alias) for alias in aliases)
        extractor.add_words(vocabulary)
        extractor._haoliang_seeded = True

    if (
        not include_sensitive
        and index is not None
        and len(visible_terms) != len(terms)
    ):
        index = TermIndex(visible_terms)
    candidates = (
        ledger_candidates(text, visible_terms, index=index)
        + extractor.extract(text)
        + compound_candidates(
            text, getattr(extractor, "is_compound_prefix", None)
        )
    )
    candidates = _drop_overlaps(candidates)
    candidates = merge_candidates(text, candidates)
    candidates = merge_english_phrases(text, candidates)
    source_priority = {"ledger": 0, "compound": 1, "phrase": 1}
    candidates = sorted(
        candidates,
        key=lambda candidate: (source_priority.get(candidate.source, 2), candidate.start),
    )

    sensitive_names = {
        name.casefold()
        for card in terms
        if card.get("sensitivity") == "sensitive"
        for name in term_names(card)
    }
    visible_names = {
        name.casefold()
        for card in visible_terms
        for name in term_names(card)
    }
    sensitive_only_names = sensitive_names - visible_names

    results: list[dict] = []
    seen: set[str] = set()
    seen_fragments: set[str] = set()
    for candidate in candidates:
        if len(results) >= max_candidates:
            break

        key = candidate.text.casefold()
        if not include_sensitive and key in sensitive_only_names:
            continue
        if key in seen:
            continue

        fragment = _is_code_fragment(text, candidate.start, candidate.end)
        if fragment:
            if key in seen_fragments:
                continue
            seen_fragments.add(key)
        else:
            seen.add(key)

        matched = None if fragment else index.find(candidate.text, domain)
        # Immediately after annotate writes a pending-only card, revise can
        # replay that draft's judgment. An observation always wins over replay.
        replay_review = (
            matched is not None
            and key in replay_review_terms
            and matched.get("status") == "proposal"
            and not matched.get("evidence")
            and matched.get("pending_reviews")
        )
        archived = (
            None
            if fragment or matched is not None
            else index.is_archived(candidate.text, domain)
        )
        if archived is not None:
            status = "archived"
            term_id = archived["id"]
            view = None
            action = "none"
        elif fragment:
            status = "fragment"
            term_id = None
            view = None
            action = "none"
        elif matched is None or replay_review:
            if not replay_review:
                if _is_fragment(candidate.text) or _is_single_cjk(candidate.text):
                    continue
                if hasattr(extractor, "is_common") and extractor.is_common(candidate.text, min_freq):
                    continue
            status = "not_in_ledger"
            term_id = None
            view = None
            action = "define"
        else:
            view = familiarity(matched, now=now, config=cfg)
            term_id = matched["id"]
            action = view["explain"]
            status = "known" if action in {"none", "reason"} else "unknown"

        defined, defined_later = (
            (False, False) if fragment else _definition_status(text, candidate.text)
        )
        results.append(
            {
                "text": candidate.text,
                "start": candidate.start,
                "end": candidate.end,
                "pos": candidate.pos,
                "source": candidate.source,
                "confidence": candidate.confidence,
                "status": status,
                "term_id": term_id,
                "familiarity": view["id"] if view else None,
                "action": action,
                "defined": defined,
                "defined_later": defined_later,
                "fragment": fragment,
                "needs_explanation": status in {"unknown", "not_in_ledger"},
                "concepts": (
                    [
                        {"id": item["id"], "label": item["label"]}
                        for item in concepts_for_term(concept_cards, term_id)
                    ]
                    if term_id
                    else []
                ),
                "judge": None,
                "judge_confidence": None,
                "judge_reason": "",
            }
        )

    judge_meta: dict = {
        "used": judge is not None,
        "model": None,
        "usage": {},
        "error": "",
        "reviews_written": 0,
        "review_terms_written": [],
        "local_terms_written": 0,
    }
    local_terms: list[dict] = []
    if judge is not None:
        judge_meta["model"] = cfg.jargon.model
        pending = [
            item
            for item in results
            if item["status"] == "not_in_ledger" and not item.get("fragment")
        ]
        contexts = [
            {
                "term": item["text"],
                "sentence": sentence_for(text, item["start"], item["end"]),
                "status": "not_in_ledger",
            }
            for item in pending
        ]
        review_terms_written: list[str] = []
        local_written = 0
        try:
            decisions = judge.judge(
                text,
                contexts,
                domain=domain,
                reader_profile=cfg.jargon.reader_profile,
                domain_profile=domain_profile,
            )
            decision_map = {item.term.casefold(): item for item in decisions}
            judge_meta["usage"] = getattr(judge, "last_usage", {})
            judge_meta["decisions"] = [
                {
                    "term": item.term,
                    "action": item.action,
                    "confidence": item.confidence,
                    "sense": item.sense,
                    "scope": item.scope,
                    "gap": item.gap,
                    "definition": item.definition,
                    "reason": item.reason,
                }
                for item in decisions
            ]
            reviews_to_write = []
            for item in pending:
                decision = decision_map.get(item["text"].casefold())
                if decision is None:
                    item["needs_explanation"] = True
                    item["action"] = "define"
                    continue

                item["judge"] = decision.action
                item["judge_confidence"] = decision.confidence
                item["judge_reason"] = decision.reason
                item["judge_scope"] = decision.scope

                if decision.scope in {"project", "model"}:
                    if decision.action == "define":
                        item["needs_explanation"] = True
                        item["action"] = "define"
                    elif decision.action == "skip":
                        item["needs_explanation"] = False
                        item["action"] = "none"
                    else:
                        item["action"] = "review"
                        item["needs_explanation"] = cfg.jargon.review_as == "define"
                    local_terms.append(
                        {
                            "term": decision.term,
                            "scope": decision.scope,
                            "project": project or domain or "general",
                            "action": decision.action,
                            "confidence": decision.confidence,
                            "definition": decision.definition,
                            "reason": decision.reason,
                        }
                    )
                    continue

                if decision.scope == "common":
                    item["needs_explanation"] = False
                    item["action"] = "none"
                    continue

                if decision.action == "define":
                    item["needs_explanation"] = True
                    item["action"] = "define"
                elif decision.action == "skip":
                    item["needs_explanation"] = False
                    item["action"] = "none"
                else:
                    item["action"] = "review"
                    item["needs_explanation"] = cfg.jargon.review_as == "define"
                    if record_reviews and not item["defined"]:
                        reviews_to_write.append((item["text"], decision))
            if reviews_to_write:
                written = []
                review_domain = domain or "general"
                # The provider call has already finished. Re-resolve each
                # candidate against the latest committed cards under the lock.
                with edit_terms(store) as latest_terms:
                    for term, decision in reviews_to_write:
                        card = find_by_name_exact(latest_terms, term, review_domain)
                        if card is None:
                            card = new_term(
                                next_id(latest_terms), term, review_domain, now=now, config=cfg
                            )
                            upsert(latest_terms, card)
                        elif (
                            card.get("status") == "archived"
                            or card.get("sensitivity") == "sensitive"
                            or card.get("evidence")
                        ):
                            continue
                        before = len(card.get("pending_reviews", []))
                        add_pending_review(
                            card,
                            source="jargon_judge",
                            domain=review_domain,
                            reason=decision.reason or "sense=?, gap=?",
                            confidence=decision.confidence,
                            time=now.isoformat() if now else None,
                        )
                        if len(card["pending_reviews"]) > before:
                            written.append(term)
                review_terms_written = written
            if record_local_terms and local_terms:
                project_name = project or domain or "general"
                path = glossary_path(project_name, glossary_root)
                items = load_local_terms(path)
                for local in local_terms:
                    item = new_local_term(
                        next_local_id(items),
                        local["term"],
                        local["scope"],
                        project_name,
                        definition=local.get("definition", ""),
                        reason=local.get("reason", ""),
                        confidence=local.get("confidence"),
                        source="jargon_judge",
                        now=now,
                    )
                    upsert_local_term(items, item)
                    local_written += 1
                save_local_terms(path, items)
        except Exception as exc:  # noqa: BLE001 - provider failure should not break annotate
            judge_meta["error"] = str(exc)
            local_written = 0
        judge_meta["reviews_written"] = len(review_terms_written)
        judge_meta["review_terms_written"] = review_terms_written
        judge_meta["local_terms_written"] = local_written

    undefined_items = [
        item
        for item in results
        if item["needs_explanation"] and not item["defined"]
    ]
    # Ledger terms first. A term already tracked by the reader model is more
    # likely to matter than a random not_in_ledger noun, and revision only
    # receives the first `max_terms` entries.
    undefined_items.sort(
        key=lambda item: (0 if item["status"] == "unknown" else 1, item["start"])
    )
    undefined = [item["text"] for item in undefined_items]
    known = [item["text"] for item in results if item["status"] == "known"]
    review_items = [
        {
            "term": item["text"],
            "reason": item.get("judge_reason", ""),
            "confidence": item.get("judge_confidence"),
            "scope": item.get("judge_scope"),
        }
        for item in results
        if item.get("judge") == "review"
        and item.get("judge_scope") == "domain"
        and not item["defined"]
    ]

    advice_parts: list[str] = []
    if undefined:
        advice_parts.append("先定义：" + "、".join(undefined))
    if known:
        advice_parts.append("可以直接用：" + "、".join(known))
    advice = "；".join(advice_parts) if advice_parts else "没有需要处理的词。"

    return {
        "text": text,
        "candidates": results,
        "undefined": undefined,
        "review": review_items,
        "leaks": find_leaks(text),
        "local_terms": local_terms,
        "domain_profile": domain_profile,
        "judge": judge_meta,
        "summary": {
            "candidates": len(results),
            "known": len(known),
            "undefined": len(undefined),
        },
        "advice": advice,
    }
