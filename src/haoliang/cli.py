"""Haoliang 命令行入口。

命令结构：
    haoliang build   构建卡片 + 向量 + 图 + SQLite
    haoliang query   检索卡片
    haoliang stats   统计
    haoliang fts     关键词检索
    haoliang persona 编译大肥鱼人设
    haoliang summarize 用模型生成卡片摘要
    haoliang revise  检查草稿并做一次独立改稿
    haoliang review  记录一次拿不准的术语判断

argparse 的用法类似 Java 的 CLI 框架：
- `add_subparsers` 定义子命令；
- `set_defaults(func=...)` 把子命令和具体函数绑定；
- `main()` 里统一调用 `args.func(args)`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .annotate import annotate
from .cards import build_cards
from .concepts import (
    concepts_path,
    load_concepts,
    new_concept,
    save_concepts,
    validate_concept,
    walk_concept,
)
from .embed import embed_cards, make_embedder
from .extractors import make_extractor
from .glossary import (
    find_local_by_id,
    find_local_term_in_root,
    glossary_path,
    iter_local_terms,
    load_local_terms,
    new_local_term,
    next_local_id,
    save_local_terms,
    upsert_local_term,
)
from .graph import build_edges, write_edges
from .index import build_index, search_fts
from .jargon import LLMJargonJudge
from .pending import resolve_pending
from .persona import compile_persona
from .query import query_store, stats
from .reader import reader_slice
from .reader_config import ReaderConfig, load_config
from .revisers import LLMReviser
from .revision import RevisionResult, revise
from .summarize import summarize_store
from .term_store import (
    TermIndex,
    edit_terms,
    find_by_id,
    find_by_name_exact,
    load_terms,
    next_id,
    upsert,
)
from .terms import (
    add_pending_review,
    archive_term,
    can_promote,
    confirm,
    familiarity,
    merge_terms,
    new_term,
    record_evidence,
    restore_term,
    validate_term,
)


def _cmd_build(args: argparse.Namespace) -> int:
    """build 子命令：卡片 → 向量 → 边 → SQLite。"""
    out = Path(args.out)

    # 实体表是可选的；没有就传空列表
    known_entities: list[str] = []
    if args.entities:
        known_entities = json.loads(Path(args.entities).read_text(encoding="utf-8"))

    manifest = build_cards(args.source_md, args.answers, out, known_entities)

    # 重新读一遍 cards.jsonl：build_cards 已经把卡片写到磁盘，
    # 这里用同一个列表更省事，但重读能确保「写出去的」和「算向量的」一致。
    cards = [
        json.loads(line)
        for line in (out / "cards.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]

    embedder = make_embedder(
        args.embedder,
        model=args.model,
        dim=args.dim,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        dimensions=args.dimensions,
        device=args.device,
        batch_size=args.batch_size,
        request_interval=args.request_interval,
    )
    embed_cards(cards, embedder, out)

    # 向量落盘后再读回来构图：
    # 这样图构建和 query 用的是同一份 embeddings.npy，避免内存/磁盘不一致。
    import numpy as np

    emb = np.load(out / "embeddings.npy")

    edges = build_edges(cards, emb, top_k=args.top_k, min_sim=args.min_sim)
    write_edges(edges, out)

    build_index(out)

    # 输出 manifest，方便脚本/CI 检查构建结果
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    """query 子命令。"""
    result = query_store(
        args.store,
        args.query,
        top_k=args.top_k,
        depth=args.depth,
        alpha=args.alpha,
        include_sensitive=args.include_sensitive,
        include_skipped=args.include_skipped,
        explain=args.explain,
        mode="vector" if args.no_graph else args.mode,
        max_hops=args.max_hops,
        min_edge_weight=args.min_edge_weight,
        seed_pool=args.seed_pool,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"query: {result['query']}")

    if result.get("seeds"):
        print(
            "seeds:",
            ", ".join(f"{k}:{v:.3f}" for k, v in result["seeds"]),
        )

    for card in result["cards"]:
        print(f"\n[{card['id']}] {card['score']:.3f}  {card['title']}")
        print(f"  {card['summary']}")
        if card.get("path"):
            print("  path:", " → ".join(card["path"]))

    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    """stats 子命令。"""
    print(json.dumps(stats(args.store), ensure_ascii=False, indent=2))
    return 0


def _cmd_fts(args: argparse.Namespace) -> int:
    """fts 子命令：SQLite FTS5 关键词检索。"""
    print(
        json.dumps(
            search_fts(args.store, args.query, args.limit),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _cmd_persona(args: argparse.Namespace) -> int:
    """persona 子命令：编译大肥鱼人设。"""
    text = compile_persona(args.preset, args.answers)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)

    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    """summarize 子命令：用模型生成卡片摘要，并可重建 store。"""
    summarize_store(
        args.store,
        args.out,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_chars=args.max_chars,
        batch_size=args.batch_size,
        limit=args.limit,
        include_sensitive=args.include_sensitive,
        device=args.device,
        rebuild=not args.no_rebuild,
    )
    return 0


def _reader_config(args: argparse.Namespace) -> ReaderConfig:
    """CLI 只负责选配置文件；阈值和层级都在 config/reader.json。"""
    return load_config(args.config) if getattr(args, "config", None) else load_config()


def _print_term(card: dict, config: ReaderConfig, as_json: bool) -> None:
    view = familiarity(card, config=config)
    check = can_promote(card, config=config)
    payload = {
        "id": card["id"],
        "term": card["term"],
        "domain": card["domain"],
        "status": card["status"],
        "author": card.get("author"),
        "familiarity": view,
        "evidence_count": len(card.get("evidence", [])),
        "pending_reviews": card.get("pending_reviews", []),
        "promotion": check,
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    counts = check["counts"]
    rules = config.promotion
    print(f"{card['id']}  {card['term']} ({card['domain']})  [{card['status']}]")
    print(
        f"  层级: {view['label']} ({view['id']})  "
        f"knowledge={view['knowledge']:.2f}  retention={view['retention']:.2f}"
    )
    print(f"  证据: {payload['evidence_count']} 条")
    pending = payload["pending_reviews"]
    if pending:
        print(f"  未决: {len(pending)} 条")
        for item in pending:
            confidence = item.get("confidence")
            suffix = f"  confidence={confidence}" if confidence is not None else ""
            print(
                f"        {item.get('domain', '-')}  "
                f"{item.get('source', '-')}  {item.get('reason', '-')}{suffix}"
            )
    print(f"  升级: {'可以' if check['ok'] else '未满足'}")
    print(
        f"        正证据 {counts['positive']}/{rules.min_positive_evidence}  "
        f"日期 {counts['distinct_days']}/{rules.min_distinct_days}  "
        f"强证据 {counts['strong']}/{rules.min_strong_evidence}"
    )
    if counts["recent_negative"]:
        print(
            f"        最近 {rules.negative_cooldown_days} 天有 "
            f"{counts['recent_negative']} 条负证据"
        )


def _cmd_observe(args: argparse.Namespace) -> int:
    if args.scope != "domain":
        raise ValueError(
            "observe only records domain terms; use `haoliang glossary` "
            "for project/model terms"
        )
    local = find_local_term_in_root(
        args.term,
        root=args.glossary_root,
    )
    if local is not None:
        raise ValueError(
            f"{args.term} is a local {local.get('scope')} term; "
            "do not write it to the reader ledger"
        )
    config = _reader_config(args)
    with edit_terms(args.store) as terms:
        card = find_by_name_exact(terms, args.term, args.domain)
        if card is not None and card.get("status") == "archived":
            raise ValueError(f"term is archived; restore it first: {card['id']}")
        if card is None:
            card = new_term(next_id(terms), args.term, args.domain, config=config)
            upsert(terms, card)

        record_evidence(
            card,
            args.kind,
            quote=args.quote,
            source=args.source,
            time=args.time,
            weight=args.weight,
            config=config,
        )
        errors = validate_term(card)
        if errors:
            raise ValueError("invalid term: " + "; ".join(errors))
    _print_term(card, config, args.json)
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    """review 子命令：记录一次拿不准的判断，不改掌握度。"""
    local = find_local_term_in_root(args.term, root=args.glossary_root)
    if local is not None:
        raise ValueError(
            f"{args.term} is a local {local.get('scope')} term; "
            "do not write it to the reader ledger"
        )
    config = _reader_config(args)
    with edit_terms(args.store) as terms:
        card = find_by_name_exact(terms, args.term, args.domain)
        if card is not None and card.get("status") == "archived":
            raise ValueError(f"term is archived; restore it first: {card['id']}")
        if card is None:
            card = new_term(next_id(terms), args.term, args.domain, config=config)
            upsert(terms, card)

        add_pending_review(
            card,
            source=args.source,
            domain=args.domain,
            reason=args.reason,
            confidence=args.confidence,
            time=args.time,
        )
        errors = validate_term(card)
        if errors:
            raise ValueError("invalid term: " + "; ".join(errors))
    _print_term(card, config, args.json)
    return 0


def _cmd_pending_list(args: argparse.Namespace) -> int:
    terms = load_terms(args.store)
    items = [
        {
            "id": card["id"],
            "term": card["term"],
            "domain": card["domain"],
            "pending_reviews": card.get("pending_reviews", []),
        }
        for card in terms
        if card.get("pending_reviews")
        and card.get("status") != "archived"
    ]
    if args.json:
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return 0
    for item in items:
        print(f"{item['id']}  {item['term']} ({item['domain']})")
        for pending in item["pending_reviews"]:
            print(
                f"  {pending.get('domain')}  {pending.get('source')}  "
                f"{pending.get('reason')}"
            )
    return 0


def _cmd_pending_resolve(args: argparse.Namespace) -> int:
    config = _reader_config(args)
    with edit_terms(args.store) as terms:
        card = find_by_id(terms, args.term_id)
        if card is None:
            raise ValueError(f"term not found: {args.term_id}")

        resolve_pending(
            card,
            args.action,
            domain=args.domain,
            reason=args.reason,
            note=args.note,
            confirmed_by="user",
            gold=args.gold,
            source="pending:user",
            config=config,
        )
    _print_term(card, config, args.json)
    return 0


def _cmd_archive(args: argparse.Namespace) -> int:
    config = _reader_config(args)
    with edit_terms(args.store) as terms:
        card = find_by_id(terms, args.term_id)
        if card is None:
            raise ValueError(f"term not found: {args.term_id}")
        if args.merged_into:
            card, _ = merge_terms(
                terms,
                card["id"],
                args.merged_into,
                reason=args.reason,
            )
        else:
            archive_term(card, reason=args.reason)
    _print_term(card, config, args.json)
    return 0


def _cmd_restore(args: argparse.Namespace) -> int:
    config = _reader_config(args)
    with edit_terms(args.store) as terms:
        card = find_by_id(terms, args.term_id)
        if card is None:
            raise ValueError(f"term not found: {args.term_id}")
        restore_term(card)
    _print_term(card, config, args.json)
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    config = _reader_config(args)
    with edit_terms(args.store) as terms:
        source, target = merge_terms(
            terms,
            args.source_id,
            args.target_id,
            reason=args.reason,
        )
    if args.json:
        print(
            json.dumps(
                {"source": source, "target": target},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(f"archived {source['id']} -> {target['id']}")
    _print_term(target, config, False)
    return 0


def _cmd_propose(args: argparse.Namespace) -> int:
    local = find_local_term_in_root(args.term, root=args.glossary_root)
    if local is not None:
        raise ValueError(
            f"{args.term} is a local {local.get('scope')} term; "
            "do not write it to the reader ledger"
        )
    config = _reader_config(args)
    with edit_terms(args.store) as terms:
        if args.id and find_by_id(terms, args.id) is not None:
            raise ValueError(f"term id already exists: {args.id}")
        existing = find_by_name_exact(terms, args.term, args.domain)
        if existing is not None:
            if existing.get("status") == "archived":
                raise ValueError(
                    f"term is archived; restore it first: {existing['id']}"
                )
            _print_term(existing, config, args.json)
            return 0

        card = new_term(
            args.id or next_id(terms),
            args.term,
            args.domain,
            author=args.author,
            sensitivity=args.sensitivity,
            config=config,
        )
        errors = validate_term(card)
        if errors:
            raise ValueError("invalid term: " + "; ".join(errors))
        upsert(terms, card)
    _print_term(card, config, args.json)
    return 0


def _cmd_confirm(args: argparse.Namespace) -> int:
    config = _reader_config(args)
    with edit_terms(args.store) as terms:
        card = find_by_id(terms, args.term_id)
        if card is None:
            raise ValueError(f"term not found: {args.term_id}")
        confirm(card, config=config)
    _print_term(card, config, args.json)
    return 0


def _cmd_terms(args: argparse.Namespace) -> int:
    config = _reader_config(args)
    terms = load_terms(args.store)
    if args.domain:
        terms = [card for card in terms if card.get("domain") == args.domain]
    if args.status:
        terms = [card for card in terms if card.get("status") == args.status]

    if args.json:
        payload = [
            {
                "id": card["id"],
                "term": card["term"],
                "domain": card["domain"],
                "status": card["status"],
                "familiarity": familiarity(card, config=config),
            }
            for card in terms
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for card in terms:
        view = familiarity(card, config=config)
        print(
            f"{card['id']}  {card['term']} ({card['domain']})  "
            f"{view['id']}  {card['status']}"
        )
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    config = _reader_config(args)
    terms = load_terms(args.store)
    card = find_by_id(terms, args.term_id)
    if card is None:
        raise ValueError(f"term not found: {args.term_id}")
    _print_term(card, config, args.json)
    return 0


def _cmd_slice(args: argparse.Namespace) -> int:
    config = _reader_config(args)
    result = reader_slice(
        args.store,
        args.query,
        domain=args.domain,
        domain_mode=args.domain_mode,
        max_terms=args.max_terms,
        config=config,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"query: {result['query']}  mode: {result['domain_mode']}")
    for key, title in (("known", "已知"), ("uncertain", "待确认"), ("unknown", "未知")):
        items = result[key]
        if not items:
            continue
        print(f"\n{title}:")
        for item in items:
            flags = []
            if item["provisional"]:
                flags.append("provisional")
            if item["stale"]:
                flags.append("stale")
            suffix = f"  [{', '.join(flags)}]" if flags else ""
            via = ""
            if item["matched_by"] != item["term"]:
                via = f"  via {item['matched_by']}"
            print(
                f"  {item['term']} ({item['domain']})  "
                f"{item['familiarity']}  {item['action']}{via}{suffix}"
            )
    if result["available_domains"]:
        print("\n可用领域:")
        for entry in result["available_domains"]:
            sample = ", ".join(entry["sample"])
            print(
                f"  {entry['name']}  terms={entry['terms']}  "
                f"unknown={entry['unknown']}  e.g. {sample}"
            )
    print(f"\n{result['advice']}")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    extractor = make_extractor(args.extractor)
    if args.term and hasattr(extractor, "add_words"):
        extractor.add_words(args.term)
    candidates = extractor.extract(args.text, language=args.language)
    if args.json:
        print(
            json.dumps(
                [candidate.__dict__ for candidate in candidates],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    for candidate in candidates:
        print(
            f"{candidate.start:>3}-{candidate.end:<3}  "
            f"{candidate.text:<18} {candidate.pos:<12} {candidate.source}"
        )
    return 0


def _cmd_annotate(args: argparse.Namespace) -> int:
    config = _reader_config(args)
    if args.text and args.file:
        raise ValueError("provide text or --file, not both")
    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        raise ValueError("provide text or --file")

    extractor = make_extractor(args.extractor)
    judge = LLMJargonJudge(rules=config.jargon) if args.judge == "llm" else None
    result = annotate(
        args.store,
        text,
        domain=args.domain,
        extractor=extractor,
        judge=judge,
        record_reviews=judge is not None,
        record_local_terms=judge is not None,
        project=args.project,
        glossary_root=args.glossary_root,
        now=None,
        config=config,
        min_freq=args.min_freq,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    for item in result["candidates"]:
        mark = "defined" if item["defined"] else "undefined"
        familiarity_id = item["familiarity"] or "-"
        judge_action = item.get("judge") or "-"
        print(
            f"  {item['text']:<18} {item['status']:<14} "
            f"{familiarity_id:<12} {item['action']:<10} "
            f"{judge_action:<8} {mark}"
        )
    meta = result.get("judge", {})
    if meta.get("used"):
        usage = meta.get("usage") or {}
        print(
            f"\njudge: {meta.get('model')}  "
            f"reviews_written={meta.get('reviews_written', 0)}  "
            f"local_terms_written={meta.get('local_terms_written', 0)}  "
            f"tokens={usage.get('total_tokens', 0)}"
        )
        if meta.get("error"):
            print(f"judge error: {meta['error']}")
    print(f"\n{result['advice']}")
    return 0


def _cmd_revise(args: argparse.Namespace) -> int:
    """revise 子命令：annotate 一次，再让 editor 做一次独立改稿。"""
    config = _reader_config(args)
    if args.text and args.file:
        raise ValueError("provide text or --file, not both")
    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        raise ValueError("provide text or --file")

    extractor = make_extractor(args.extractor)
    judge = LLMJargonJudge(rules=config.jargon) if args.judge == "llm" else None
    annotation = annotate(
        args.store,
        text,
        domain=args.domain,
        extractor=extractor,
        judge=judge,
        record_reviews=judge is not None,
        record_local_terms=judge is not None,
        project=args.project,
        glossary_root=args.glossary_root,
        now=None,
        config=config,
        min_freq=args.min_freq,
    )

    rules = config.revision
    max_terms = rules.max_terms if args.max_terms is None else args.max_terms
    if annotation["undefined"]:
        reviser = LLMReviser(rules=rules)
        result = revise(annotation, reviser, max_terms=max_terms)
    else:
        result = RevisionResult(text=text, notes="nothing to revise")

    if args.json:
        print(
            json.dumps(
                {
                    "draft": text,
                    "undefined": annotation["undefined"],
                    "final": result.text,
                    "edits": [
                        {
                            "start": edit.start,
                            "end": edit.end,
                            "replacement": edit.replacement,
                        }
                        for edit in result.edits
                    ],
                    "notes": result.notes,
                    "usage": result.usage,
                    "judge": annotation.get("judge", {}),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print("undefined:", "、".join(annotation["undefined"]) or "-")
    judge_meta = annotation.get("judge", {})
    if judge_meta.get("used"):
        usage = judge_meta.get("usage") or {}
        print(
            f"judge: {judge_meta.get('model')}  "
            f"reviews_written={judge_meta.get('reviews_written', 0)}  "
            f"tokens={usage.get('total_tokens', 0)}"
        )
        if judge_meta.get("error"):
            print(f"judge error: {judge_meta['error']}")
    if result.notes:
        print("notes:", result.notes)
    if result.usage:
        print(
            "usage:",
            " ".join(f"{key}={value}" for key, value in result.usage.items()),
        )
    print("\n--- draft ---")
    print(text)
    print("\n--- final ---")
    print(result.text)
    return 0


def _concept_path(store: str) -> Path:
    return concepts_path(store)


def _find_concept(concepts: list[dict], concept_id: str) -> dict | None:
    return next((card for card in concepts if card.get("id") == concept_id), None)


def _cmd_concept_add(args: argparse.Namespace) -> int:
    path = _concept_path(args.store)
    concepts = load_concepts(path)
    if _find_concept(concepts, args.concept_id) is not None:
        raise ValueError(f"concept already exists: {args.concept_id}")
    card = new_concept(
        args.concept_id,
        args.concept,
        args.domain,
        args.label,
        args.member,
        summary=args.summary,
        prerequisites=args.prerequisite,
    )
    errors = validate_concept(card)
    if errors:
        raise ValueError("invalid concept: " + "; ".join(errors))
    concepts.append(card)
    save_concepts(path, concepts)
    print(json.dumps(card, ensure_ascii=False, indent=2))
    return 0


def _cmd_concept_list(args: argparse.Namespace) -> int:
    concepts = load_concepts(_concept_path(args.store))
    if args.json:
        print(json.dumps(concepts, ensure_ascii=False, indent=2))
        return 0
    for card in concepts:
        print(
            f"{card['id']}  {card['label']} ({card['domain']})  "
            f"members={len(card.get('members', []))}  {card['status']}"
        )
    return 0


def _cmd_concept_show(args: argparse.Namespace) -> int:
    concepts = load_concepts(_concept_path(args.store))
    card = _find_concept(concepts, args.concept_id)
    if card is None:
        raise ValueError(f"concept not found: {args.concept_id}")
    if args.json:
        print(json.dumps(card, ensure_ascii=False, indent=2))
        return 0

    terms = {item["id"]: item["term"] for item in load_terms(args.store)}
    print(f"{card['id']}  {card['label']} ({card['domain']})  {card['status']}")
    if card.get("summary"):
        print(f"  {card['summary']}")
    print("  members:")
    for term_id in card.get("members", []):
        print(f"    {term_id}  {terms.get(term_id, '-')}")
    if card.get("prerequisites"):
        print("  prerequisites:", ", ".join(card["prerequisites"]))
    for item in card.get("related", []):
        print(f"  {item.get('type')}: {item.get('target')} {item.get('note', '')}")
    return 0


def _cmd_concept_walk(args: argparse.Namespace) -> int:
    concepts = load_concepts(_concept_path(args.store))
    walked = walk_concept(concepts, args.concept_id, max_hops=args.max_hops)
    if not walked:
        raise ValueError(f"concept not found: {args.concept_id}")
    if args.json:
        print(json.dumps(walked, ensure_ascii=False, indent=2))
        return 0

    terms = {item["id"]: item["term"] for item in load_terms(args.store)}
    for node in walked:
        members = ", ".join(terms.get(item, item) for item in node["members"])
        print(
            f"{'  ' * node['depth']}{node['id']}  {node['label']}  "
            f"[{members}]"
        )
    return 0


def _cmd_glossary_add(args: argparse.Namespace) -> int:
    path = glossary_path(args.project, args.root)
    items = load_local_terms(path)
    card = new_local_term(
        next_local_id(items),
        args.term,
        args.scope,
        args.project,
        definition=args.definition,
        reason=args.reason,
        source=args.source,
    )
    upsert_local_term(items, card)
    save_local_terms(path, items)
    print(json.dumps(card, ensure_ascii=False, indent=2))
    return 0


def _cmd_glossary_list(args: argparse.Namespace) -> int:
    if args.project:
        items = load_local_terms(glossary_path(args.project, args.root))
    else:
        items = iter_local_terms(args.root)
    if args.scope:
        items = [item for item in items if item.get("scope") == args.scope]
    if args.json:
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return 0
    for item in items:
        print(
            f"{item['id']}  {item['term']}  [{item['scope']}]  "
            f"{item['project']}  {item.get('status')}  "
            f"{item.get('definition', '')}"
        )
    return 0


def _cmd_glossary_show(args: argparse.Namespace) -> int:
    if args.project:
        items = load_local_terms(glossary_path(args.project, args.root))
    else:
        items = iter_local_terms(args.root)
    item = find_local_by_id(items, args.local_id)
    if item is None:
        raise ValueError(f"local term not found: {args.local_id}")
    print(json.dumps(item, ensure_ascii=False, indent=2))
    return 0


def _cmd_draft(args: argparse.Namespace) -> int:
    """draft：一次真实闭环，slice → annotate → revise → trace。"""
    config = _reader_config(args)
    if args.draft and args.file:
        raise ValueError("provide --draft or --file, not both")
    text = args.draft
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        raise ValueError("provide --draft or --file")

    terms = load_terms(args.store)
    index = TermIndex(terms)
    extractor = make_extractor(args.extractor)
    context = reader_slice(
        args.store,
        args.query,
        domain=args.domain,
        index=index,
        config=config,
    )
    brief = context["advice"]

    judge = LLMJargonJudge(rules=config.jargon) if args.judge == "llm" else None
    annotation = annotate(
        args.store,
        text,
        domain=args.domain,
        extractor=extractor,
        index=index,
        judge=judge,
        record_reviews=judge is not None,
        record_local_terms=judge is not None,
        project=args.project,
        glossary_root=args.glossary_root,
        config=config,
    )

    if annotation["undefined"] and not args.no_revise:
        reviser = LLMReviser(rules=config.revision)
        revision = revise(
            annotation,
            reviser,
            max_terms=config.revision.max_terms,
        )
    else:
        revision = RevisionResult(
            text=text,
            notes=(
                "nothing to revise"
                if not annotation["undefined"]
                else "skipped by --no-revise"
            ),
        )

    record = {
        "query": args.query,
        "domain": args.domain,
        "brief": brief,
        "draft": text,
        "undefined": annotation["undefined"],
        "local_terms": annotation.get("local_terms", []),
        "judge": annotation.get("judge", {}),
        "final": revision.text,
        "revision": {
            "notes": revision.notes,
            "edits": [
                {
                    "start": edit.start,
                    "end": edit.end,
                    "replacement": edit.replacement,
                }
                for edit in revision.edits
            ],
            "usage": revision.usage,
        },
    }
    trace = Path(args.trace)
    trace.parent.mkdir(parents=True, exist_ok=True)
    with trace.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.json:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    print(f"query: {args.query}  domain: {args.domain or '-'}")
    print(f"brief: {brief}")
    print(f"undefined: {'、'.join(annotation['undefined']) or '-'}")
    if annotation.get("local_terms"):
        print(
            "local terms:",
            "、".join(
                f"{item['term']}({item['scope']})"
                for item in annotation["local_terms"]
            ),
        )
    if revision.notes:
        print(f"revision: {revision.notes}")
    print("\n--- final ---")
    print(revision.text)
    print(f"\ntrace: {trace}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    p = argparse.ArgumentParser(
        prog="haoliang",
        description="不替鱼说它快乐。A reader model for one mind.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- build ----
    b = sub.add_parser(
        "build",
        help="build cards, embeddings, graph, and SQLite index",
    )
    b.add_argument("--source-md", required=True)
    b.add_argument("--answers", required=True)
    b.add_argument("--out", default="var/store")
    b.add_argument(
        "--embedder",
        default="hash",
        choices=["hash", "openai", "bge", "local"],
    )
    b.add_argument("--model", default=None)
    b.add_argument("--dim", type=int, default=256)
    b.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (e.g. https://ark.cn-beijing.volces.com/api/v3)",
    )
    b.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="environment variable holding the API key (e.g. ARK_API_KEY)",
    )
    b.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="requested embedding dimensions (OpenAI-compatible APIs)",
    )
    b.add_argument(
        "--device",
        default=None,
        help="BGE device: mps / cpu (default: auto)",
    )
    b.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="texts per embedding request (Ark Agent Plan: <= 10)",
    )
    b.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="seconds to sleep between embedding batches (Ark: 0.5-1.0)",
    )
    b.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="semantic neighbors per card",
    )
    b.add_argument("--min-sim", type=float, default=0.15)
    b.add_argument(
        "--entities",
        default=None,
        help="JSON list of known entity names",
    )
    b.set_defaults(func=_cmd_build)

    # ---- query ----
    q = sub.add_parser("query", help="search the card graph")
    q.add_argument("query")
    q.add_argument("--store", default="var/store")
    q.add_argument("--top-k", type=int, default=10)
    q.add_argument("--depth", type=int, default=2)
    q.add_argument("--alpha", type=float, default=0.85)
    q.add_argument("--json", action="store_true")
    q.add_argument("--explain", action="store_true")
    q.add_argument("--include-sensitive", action="store_true")
    q.add_argument("--include-skipped", action="store_true")
    q.add_argument(
        "--no-graph",
        action="store_true",
        help="disable PPR graph rerank (vector-only baseline)",
    )
    q.add_argument(
        "--mode",
        choices=["graph", "vector", "deep"],
        default="graph",
        help="retrieval mode: graph (PPR), vector (semantic only), deep (bounded non-semantic walk)",
    )
    q.add_argument(
        "--max-hops",
        type=int,
        default=2,
        help="deep: max non-semantic hops from the anchor",
    )
    q.add_argument(
        "--min-edge-weight",
        type=float,
        default=0.6,
        help="deep: ignore edges below this weight",
    )
    q.add_argument(
        "--seed-pool",
        type=int,
        default=8,
        help="deep: pick the anchor from this many vector seeds",
    )
    q.set_defaults(func=_cmd_query)

    # ---- stats ----
    s = sub.add_parser("stats", help="show store stats")
    s.add_argument("--store", default="var/store")
    s.set_defaults(func=_cmd_stats)

    # ---- fts ----
    f = sub.add_parser("fts", help="keyword search via SQLite FTS5")
    f.add_argument("query")
    f.add_argument("--store", default="var/store")
    f.add_argument("--limit", type=int, default=20)
    f.set_defaults(func=_cmd_fts)

    # ---- persona ----
    pe = sub.add_parser(
        "persona",
        help="compile a short persona brief from a preset",
    )
    pe.add_argument("--preset", required=True)
    pe.add_argument("--answers", required=True)
    pe.add_argument("--out", default=None)
    pe.set_defaults(func=_cmd_persona)

    sm = sub.add_parser(
        "summarize",
        help="summarize cards with an LLM and rebuild the store",
    )
    sm.add_argument("--store", default="var/store-links")
    sm.add_argument("--out", default="var/store-summarized")
    sm.add_argument("--model", default="deepseek-chat")
    sm.add_argument("--base-url", default="https://api.deepseek.com")
    sm.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    sm.add_argument("--max-chars", type=int, default=120)
    sm.add_argument("--batch-size", type=int, default=8)
    sm.add_argument("--limit", type=int, default=0)
    sm.add_argument("--include-sensitive", action="store_true")
    sm.add_argument("--device", default="mps")
    sm.add_argument("--no-rebuild", action="store_true")
    sm.set_defaults(func=_cmd_summarize)

    # ---- reader terms ----
    ob = sub.add_parser("observe", help="record evidence for a term card")
    ob.add_argument("term")
    ob.add_argument("--domain", required=True)
    ob.add_argument("--kind", required=True)
    ob.add_argument("--scope", choices=["domain", "project", "model", "common"], default="domain")
    ob.add_argument("--project", default=None)
    ob.add_argument("--glossary-root", default="var/glossary")
    ob.add_argument("--quote", default="")
    ob.add_argument("--source", default="")
    ob.add_argument("--time", default=None)
    ob.add_argument("--weight", type=float, default=None)
    ob.add_argument("--store", default="var/reader")
    ob.add_argument("--config", default=None)
    ob.add_argument("--json", action="store_true")
    ob.set_defaults(func=_cmd_observe)

    re_ = sub.add_parser(
        "review",
        help="record an unresolved jargon judgment (does not change familiarity)",
    )
    re_.add_argument("term")
    re_.add_argument("--domain", required=True)
    re_.add_argument("--reason", required=True)
    re_.add_argument("--source", default="jargon_judge")
    re_.add_argument("--confidence", type=float, default=None)
    re_.add_argument("--time", default=None)
    re_.add_argument("--glossary-root", default="var/glossary")
    re_.add_argument("--store", default="var/reader")
    re_.add_argument("--config", default=None)
    re_.add_argument("--json", action="store_true")
    re_.set_defaults(func=_cmd_review)

    pd = sub.add_parser("pending", help="list and resolve pending reviews")
    pd_sub = pd.add_subparsers(dest="pending_command", required=True)
    pl = pd_sub.add_parser("list", help="list cards with pending reviews")
    pl.add_argument("--store", default="var/reader")
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(func=_cmd_pending_list)
    pr_ = pd_sub.add_parser("resolve", help="resolve one pending review")
    pr_.add_argument("term_id")
    pr_.add_argument("--action", choices=["known", "unknown", "archive"], required=True)
    pr_.add_argument("--domain", default=None)
    pr_.add_argument("--reason", default=None)
    pr_.add_argument("--note", default="")
    pr_.add_argument(
        "--gold",
        action="store_true",
        help="write a gold label (labeled_known/asked), not a self-report",
    )
    pr_.add_argument("--store", default="var/reader")
    pr_.add_argument("--config", default=None)
    pr_.add_argument("--json", action="store_true")
    pr_.set_defaults(func=_cmd_pending_resolve)

    ar = sub.add_parser("archive", help="archive a term or merge it into another")
    ar.add_argument("term_id")
    ar.add_argument("--merged-into", default=None)
    ar.add_argument("--reason", default="")
    ar.add_argument("--store", default="var/reader")
    ar.add_argument("--config", default=None)
    ar.add_argument("--json", action="store_true")
    ar.set_defaults(func=_cmd_archive)

    rs = sub.add_parser("restore", help="restore an archived term")
    rs.add_argument("term_id")
    rs.add_argument("--store", default="var/reader")
    rs.add_argument("--config", default=None)
    rs.add_argument("--json", action="store_true")
    rs.set_defaults(func=_cmd_restore)

    mg = sub.add_parser("merge", help="merge source term into target term")
    mg.add_argument("source_id")
    mg.add_argument("target_id")
    mg.add_argument("--reason", default="")
    mg.add_argument("--store", default="var/reader")
    mg.add_argument("--config", default=None)
    mg.add_argument("--json", action="store_true")
    mg.set_defaults(func=_cmd_merge)

    dr = sub.add_parser("draft", help="slice -> annotate -> revise -> trace")
    dr.add_argument("query")
    dr.add_argument("--draft", default=None)
    dr.add_argument("--file", default=None)
    dr.add_argument("--domain", default=None)
    dr.add_argument("--judge", choices=["off", "llm"], default="off")
    dr.add_argument("--no-revise", action="store_true")
    dr.add_argument("--project", default=None)
    dr.add_argument("--glossary-root", default="var/glossary")
    dr.add_argument("--store", default="var/reader")
    dr.add_argument("--extractor", default="jieba", choices=["jieba"])
    dr.add_argument("--trace", default="var/usage/trace.jsonl")
    dr.add_argument("--config", default=None)
    dr.add_argument("--json", action="store_true")
    dr.set_defaults(func=_cmd_draft)

    pr = sub.add_parser("propose", help="create a proposal term card")
    pr.add_argument("term")
    pr.add_argument("--domain", required=True)
    pr.add_argument("--id", default=None)
    pr.add_argument(
        "--author",
        default="ai-proposed",
        choices=["player", "ai-proposed", "imported"],
    )
    pr.add_argument(
        "--sensitivity",
        default="private",
        choices=["private", "sensitive"],
    )
    pr.add_argument("--glossary-root", default="var/glossary")
    pr.add_argument("--store", default="var/reader")
    pr.add_argument("--config", default=None)
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=_cmd_propose)

    co = sub.add_parser("confirm", help="promote a proposal term card to canon")
    co.add_argument("term_id")
    co.add_argument("--store", default="var/reader")
    co.add_argument("--config", default=None)
    co.add_argument("--json", action="store_true")
    co.set_defaults(func=_cmd_confirm)

    ts = sub.add_parser("terms", help="list term cards")
    ts.add_argument("--domain", default=None)
    ts.add_argument("--status", default=None, choices=["proposal", "canon", "archived"])
    ts.add_argument("--store", default="var/reader")
    ts.add_argument("--config", default=None)
    ts.add_argument("--json", action="store_true")
    ts.set_defaults(func=_cmd_terms)

    sh = sub.add_parser("show", help="show one term card")
    sh.add_argument("term_id")
    sh.add_argument("--store", default="var/reader")
    sh.add_argument("--config", default=None)
    sh.add_argument("--json", action="store_true")
    sh.set_defaults(func=_cmd_show)

    sl = sub.add_parser("slice", help="build a reader slice for a query")
    sl.add_argument("query")
    sl.add_argument("--domain", default=None)
    sl.add_argument(
        "--domain-mode",
        choices=["soft", "hard", "global"],
        default="soft",
        help="soft: domain is a ranking prior; hard: filter by domains; global: ignore domain",
    )
    sl.add_argument("--max-terms", type=int, default=12)
    sl.add_argument("--store", default="var/reader")
    sl.add_argument("--config", default=None)
    sl.add_argument("--json", action="store_true")
    sl.set_defaults(func=_cmd_slice)

    ex = sub.add_parser("extract", help="extract term candidates from text")
    ex.add_argument("text")
    ex.add_argument("--extractor", default="jieba", choices=["jieba"])
    ex.add_argument("--language", default="zh", choices=["zh"])
    ex.add_argument(
        "--term",
        action="append",
        default=[],
        help="known term/alias to seed the tokenizer (repeatable)",
    )
    ex.add_argument("--json", action="store_true")
    ex.set_defaults(func=_cmd_extract)

    an = sub.add_parser("annotate", help="annotate a draft against the reader ledger")
    an.add_argument("text", nargs="?", default=None)
    an.add_argument("--file", default=None)
    an.add_argument("--domain", default=None)
    an.add_argument("--judge", choices=["off", "llm"], default="off")
    an.add_argument("--project", default=None)
    an.add_argument("--glossary-root", default="var/glossary")
    an.add_argument("--extractor", default="jieba", choices=["jieba"])
    an.add_argument("--store", default="var/reader")
    an.add_argument("--min-freq", type=int, default=1000)
    an.add_argument("--config", default=None)
    an.add_argument("--json", action="store_true")
    an.set_defaults(func=_cmd_annotate)

    rv = sub.add_parser(
        "revise",
        help="annotate a draft and run one isolated editor pass",
    )
    rv.add_argument("text", nargs="?", default=None)
    rv.add_argument("--file", default=None)
    rv.add_argument("--domain", default=None)
    rv.add_argument("--judge", choices=["off", "llm"], default="off")
    rv.add_argument("--project", default=None)
    rv.add_argument("--glossary-root", default="var/glossary")
    rv.add_argument("--extractor", default="jieba", choices=["jieba"])
    rv.add_argument("--store", default="var/reader")
    rv.add_argument("--min-freq", type=int, default=1000)
    rv.add_argument(
        "--max-terms",
        type=int,
        default=None,
        help="override config revision.max_terms for this call",
    )
    rv.add_argument("--config", default=None)
    rv.add_argument("--json", action="store_true")
    rv.set_defaults(func=_cmd_revise)

    # ---- concept ----
    cp = sub.add_parser("concept", help="manage concept cards")
    cp_sub = cp.add_subparsers(dest="concept_command", required=True)

    ca = cp_sub.add_parser("add", help="create a concept card")
    ca.add_argument("concept_id")
    ca.add_argument("--concept", required=True)
    ca.add_argument("--domain", required=True)
    ca.add_argument("--label", required=True)
    ca.add_argument("--member", action="append", required=True)
    ca.add_argument("--prerequisite", action="append", default=[])
    ca.add_argument("--summary", default="")
    ca.add_argument("--store", default="var/reader")
    ca.set_defaults(func=_cmd_concept_add)

    cl = cp_sub.add_parser("list", help="list concept cards")
    cl.add_argument("--store", default="var/reader")
    cl.add_argument("--json", action="store_true")
    cl.set_defaults(func=_cmd_concept_list)

    cs = cp_sub.add_parser("show", help="show one concept card")
    cs.add_argument("concept_id")
    cs.add_argument("--store", default="var/reader")
    cs.add_argument("--json", action="store_true")
    cs.set_defaults(func=_cmd_concept_show)

    cw = cp_sub.add_parser("walk", help="walk prerequisites and related concepts")
    cw.add_argument("concept_id")
    cw.add_argument("--max-hops", type=int, default=2)
    cw.add_argument("--store", default="var/reader")
    cw.add_argument("--json", action="store_true")
    cw.set_defaults(func=_cmd_concept_walk)

    # ---- local glossary ----
    gl = sub.add_parser("glossary", help="manage project-local terms")
    gl_sub = gl.add_subparsers(dest="glossary_command", required=True)

    ga = gl_sub.add_parser("add", help="add a project/model local term")
    ga.add_argument("term")
    ga.add_argument("--scope", choices=["project", "model"], required=True)
    ga.add_argument("--project", required=True)
    ga.add_argument("--definition", default="")
    ga.add_argument("--reason", default="")
    ga.add_argument("--source", default="manual")
    ga.add_argument("--root", default="var/glossary")
    ga.set_defaults(func=_cmd_glossary_add)

    gll = gl_sub.add_parser("list", help="list local terms")
    gll.add_argument("--project", default=None)
    gll.add_argument("--scope", choices=["project", "model"], default=None)
    gll.add_argument("--root", default="var/glossary")
    gll.add_argument("--json", action="store_true")
    gll.set_defaults(func=_cmd_glossary_list)

    gls = gl_sub.add_parser("show", help="show one local term")
    gls.add_argument("local_id")
    gls.add_argument("--project", default=None)
    gls.add_argument("--root", default="var/glossary")
    gls.set_defaults(func=_cmd_glossary_show)

    return p


def main(argv: list[str] | None = None) -> int:
    """程序入口：解析参数并分发到子命令。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except BrokenPipeError:
        # 例如 `haoliang query ... | head` 时管道提前关闭，不算错误
        return 0
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - CLI 边界
        # CLI 最外层兜底：不打印 traceback，给用户一行可读错误
        print(f"haoliang: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
