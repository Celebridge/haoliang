"""读者切片：把 term 账本投影成「这次该解释到什么深度」。

输入：query + 可选 domain + domain_mode。
输出：ranked（按分数排序）、known / uncertain / unknown 三组、可选目录。

domain 不是硬过滤器，而是排序先验：

- `soft`（默认）：同名词按 domain 消歧，其它词仍可跨领域命中。
- `hard`：只保留 `domain` 出现在卡片 `domains` 里的候选（用户显式要求时用）。
- `global`：忽略 domain。

别名和合并前的名称同样参与匹配。
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from .concepts import (
    concepts_for_term,
    concepts_path,
    derive_domain_profiles,
    load_concepts,
)
from .reader_config import ReaderConfig, load_config
from .term_store import TermIndex, card_domains, load_terms
from .terms import familiarity

_TOKEN_RE = re.compile(r"[a-z0-9_+#.-]+|[\u4e00-\u9fff]+", re.IGNORECASE)
DOMAIN_MODES = {"soft", "hard", "global"}


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text)}


def _ascii_word_match(needle: str, haystack: str) -> bool:
    if not all(char.isascii() for char in needle):
        return needle in haystack
    pattern = (
        r"(?<![A-Za-z0-9])"
        + re.escape(needle)
        + r"(?![A-Za-z0-9])"
    )
    return re.search(pattern, haystack) is not None


def _match_score(query: str, term: str) -> float:
    """词条和 query 的粗匹配：子串优先，其次 token 重合。"""
    q = query.casefold().strip()
    t = term.casefold().strip()
    if not q or not t:
        return 0.0
    if _ascii_word_match(t, q):
        return 10.0 + min(len(t), 20) / 100
    if _ascii_word_match(q, t):
        return 8.0 + min(len(q), 20) / 100

    q_tokens = _tokens(q)
    t_tokens = _tokens(t)
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = q_tokens & t_tokens
    if not overlap:
        return 0.0
    return 4.0 * len(overlap) / len(t_tokens)


def _best_match(
    query: str, card: dict, index: TermIndex, domain: str | None, include_sensitive: bool
) -> tuple[float, str]:
    """在 term + aliases 里找最像 query 的那个写法。"""
    best_score = 0.0
    best_name = str(card.get("term", ""))
    for name in index.names_for_card(card):
        score = _match_score(query, name)
        if score > best_score:
            if domain and index.find(name, domain, include_sensitive=include_sensitive) is not card:
                continue
            best_score = score
            best_name = name
    return best_score, best_name


def _catalog(cards: list[dict], config: ReaderConfig, now: datetime | None) -> list[dict]:
    """没有匹配时返回的目录：每个领域有多少词、多少未知、几个例子。"""
    by_domain: dict[str, dict] = {}
    for card in cards:
        view = familiarity(card, now=now, config=config)
        for name in card_domains(card):
            entry = by_domain.setdefault(
                name,
                {"name": name, "terms": 0, "unknown": 0, "sample": []},
            )
            entry["terms"] += 1
            if view["explain"] == "define":
                entry["unknown"] += 1
            if len(entry["sample"]) < 5:
                entry["sample"].append(card["term"])
    return sorted(by_domain.values(), key=lambda item: item["name"])


def _advice(
    known: list[dict],
    uncertain: list[dict],
    unknown: list[dict],
    *,
    has_candidates: bool = True,
) -> str:
    repeated = Counter(item["term"] for item in known + uncertain + unknown)

    def label(item: dict) -> str:
        term = item["term"]
        return f"{term}（{item['domain']}）" if repeated[term] > 1 else term

    parts: list[str] = []
    if unknown:
        parts.append("先定义：" + "、".join(map(label, unknown)))
    if uncertain:
        parts.append("给一句解释：" + "、".join(map(label, uncertain)))
    if known:
        usable = [item for item in known if item["action"] == "none"]
        explainable = [item for item in known if item["action"] == "reason"]
        if usable:
            parts.append("可以直接用：" + "、".join(map(label, usable)))
        if explainable:
            parts.append("可以当前提：" + "、".join(map(label, explainable)))
    if parts:
        return "；".join(parts)
    if has_candidates:
        return "有匹配词条，但本次切片预算为 0，未展开；需要时提高 max_terms。"
    return "没有匹配到词条；按一般读者处理，或先问一句。"


def domain_snapshot(
    store: str | Path,
    domain: str,
    *,
    max_terms: int = 12,
    now: datetime | None = None,
    config: ReaderConfig | None = None,
    index: TermIndex | None = None,
    include_sensitive: bool = False,
) -> dict:
    """Return a compact profile of what the reader knows in one domain.

    This is the broad signal to load before writing: it does not depend on a
    query. `brief` remains the query-specific slice.
    """
    cfg = config or load_config()
    if not str(domain).strip():
        raise ValueError("domain must not be empty")
    if max_terms < 0:
        raise ValueError("max_terms must be >= 0")
    if index is None:
        index = TermIndex(load_terms(store))
    cards = [
        card
        for card in index.active_terms
        if domain in card_domains(card)
        and (include_sensitive or card.get("sensitivity") != "sensitive")
    ]

    buckets: dict[str, list[dict]] = {
        "known": [],
        "uncertain": [],
        "unknown": [],
    }
    for card in cards:
        view = familiarity(card, now=now, config=cfg)
        item = {
            "id": card["id"],
            "term": card["term"],
            "familiarity": view["id"],
            "action": view["explain"],
            "knowledge": view["knowledge"],
            "retention": view["retention"],
            "status": card.get("status"),
        }
        if view["explain"] in {"none", "reason"}:
            buckets["known"].append(item)
        elif view["explain"] == "define_brief":
            buckets["uncertain"].append(item)
        else:
            buckets["unknown"].append(item)

    counts = {key: len(items) for key, items in buckets.items()}
    advice = _advice(
        buckets["known"],
        buckets["uncertain"],
        buckets["unknown"],
        has_candidates=bool(cards),
    )
    for items in buckets.values():
        items.sort(key=lambda item: (item["knowledge"], item["term"]))
        del items[max_terms:]

    profile = derive_domain_profiles(
        cards,
        config=cfg,
        now=now,
        domain=domain,
        include_sensitive=include_sensitive,
    )
    return {
        "domain": domain,
        "profile": profile[0] if profile else None,
        "known": buckets["known"],
        "uncertain": buckets["uncertain"],
        "unknown": buckets["unknown"],
        "counts": {
            "terms": len(cards),
            "known": counts["known"],
            "uncertain": counts["uncertain"],
            "unknown": counts["unknown"],
        },
        "advice": advice,
    }


def reader_slice(
    store: str | Path,
    query: str,
    *,
    domain: str | None = None,
    domain_mode: str = "soft",
    max_terms: int = 12,
    include_domain_fallback: bool = True,
    include_sensitive: bool = False,
    now: datetime | None = None,
    config: ReaderConfig | None = None,
    index: TermIndex | None = None,
) -> dict:
    """生成一次读者切片。

    - `domain_mode="soft"`：domain 只加分，不丢候选（默认）。
    - `domain_mode="hard"`：只保留 `domains` 包含该领域的候选（显式约束）。
    - `domain_mode="global"`：忽略 domain。
    - `max_terms` 限制切片大小，默认 12。
    - `include_domain_fallback=False`：只保留与 query 有词面/词元匹配的卡；
      `brief` 用这个模式，避免把同领域的无关 unknown 塞进建议。
    - `include_sensitive=False`：默认不召回 `sensitivity=sensitive` 的卡。
    - `index` 可复用，避免每次查询都加载并扫描全部卡片。
    """
    if domain_mode not in DOMAIN_MODES:
        raise ValueError(f"domain_mode must be one of {sorted(DOMAIN_MODES)}")
    if domain_mode == "hard" and not domain:
        raise ValueError("domain_mode='hard' requires a domain")
    if max_terms < 0:
        raise ValueError("max_terms must be >= 0")

    cfg = config or load_config()
    if index is None:
        all_cards = load_terms(store)
        index = TermIndex(all_cards)
    all_cards = [
        card
        for card in index.active_terms
        if include_sensitive or card.get("sensitivity") != "sensitive"
    ]
    concept_cards = load_concepts(concepts_path(store))
    domain_profile = None
    if domain:
        profiles = derive_domain_profiles(
            all_cards,
            config=cfg,
            now=now,
            domain=domain,
            include_sensitive=include_sensitive,
        )
        domain_profile = profiles[0] if profiles else None

    if domain and domain_mode == "hard":
        cards = index.by_domain.get(domain, [])
    elif domain:
        cards = index.candidate_cards(
            query,
            domain=domain,
            include_domain=include_domain_fallback,
        )
    else:
        cards = index.candidate_cards(query)
    if not include_sensitive:
        cards = [
            card for card in cards if card.get("sensitivity") != "sensitive"
        ]

    soft_domain = domain if domain_mode == "soft" else None
    scored: list[tuple[float, float, dict, dict, str, bool, list[str], float]] = []
    for card in cards:
        view = familiarity(card, now=now, config=cfg)
        score, matched_by = _best_match(query, card, index, soft_domain, include_sensitive)
        domains = card_domains(card)
        domain_match = bool(domain) and domain_mode != "global" and domain in domains
        # Levels are data, not ids: a configured level that maps to `define` or
        # `define_brief` is a level that needs explanation.
        needs_explanation = view["explain"] in {"define", "define_brief"}

        # 没有词面匹配时，只在「有 domain 且该卡属于这个 domain」时补需要解释的词。
        if score == 0 and not (
            include_domain_fallback
            and domain
            and domain_mode != "global"
            and domain_match
            and needs_explanation
            and (
                not soft_domain
                or index.find(card["term"], soft_domain, include_sensitive=include_sensitive)
                is card
            )
        ):
            continue

        # 同分时先讲掌握度低的词：unknown 0.0 → heard 0.25 → usable 0.6 → explainable 0.9
        level_threshold = float(view["min_knowledge"])
        rank_score = score + (cfg.retrieval.domain_boost if domain_match else 0.0)
        scored.append(
            (
                rank_score,
                level_threshold,
                card,
                view,
                matched_by,
                domain_match,
                domains,
                score,
            )
        )

    scored.sort(
        key=lambda row: (-row[0], row[1], str(row[2].get("term", "")))
    )
    selected = scored[:max_terms]

    known: list[dict] = []
    uncertain: list[dict] = []
    unknown: list[dict] = []
    ranked: list[dict] = []
    for rank_score, _, card, view, matched_by, domain_match, domains, lexical_score in selected:
        item = {
            "id": card["id"],
            "term": card["term"],
            "domain": card["domain"],
            "domains": domains,
            "familiarity": view["id"],
            "label": view["label"],
            "action": view["explain"],
            "knowledge": view["knowledge"],
            "retention": view["retention"],
            "status": card.get("status"),
            "provisional": card.get("status") != "canon",
            "stale": view["retention"] < cfg.decay.retention_threshold,
            "evidence_count": len(card.get("evidence", [])),
            "matched_by": matched_by,
            "match": "query" if lexical_score > 0 else "domain",
            "domain_match": domain_match,
            "concepts": [
                {"id": item["id"], "label": item["label"]}
                for item in concepts_for_term(concept_cards, card["id"])
            ],
        }
        ranked.append(item)
        if view["explain"] in {"none", "reason"}:
            known.append(item)
        elif view["explain"] == "define_brief":
            uncertain.append(item)
        else:
            unknown.append(item)

    return {
        "query": query,
        "domain": domain,
        "domain_mode": domain_mode,
        "domain_profile": domain_profile,
        "known": known,
        "uncertain": uncertain,
        "unknown": unknown,
        "ranked": ranked,
        "available_domains": [] if ranked else _catalog(all_cards, cfg, now),
        "summary": {
            "terms": len(ranked),
            "known": len(known),
            "uncertain": len(uncertain),
            "unknown": len(unknown),
            "candidate_terms": len(cards) if include_domain_fallback else len(scored),
            "domain_terms": (
                sum(1 for card in all_cards if domain in card_domains(card))
                if domain
                else len(all_cards)
            ),
        },
        "advice": _advice(
            known,
            uncertain,
            unknown,
            has_candidates=bool(scored),
        ),
    }
