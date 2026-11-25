"""Regressions found in the 2026-11-23 review."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from haoliang.annotate import (
    annotate,
    find_leaks,
    merge_english_phrases,
    term_is_defined,
)
from haoliang.extractors import make_extractor
from haoliang.jargon import JargonDecision
from haoliang.reader import domain_snapshot, reader_slice
from haoliang.reader_config import Level, load_config
from haoliang.reader_tools import ReaderService
from haoliang.term_store import (
    find_by_name_exact,
    save_terms,
)
from haoliang.terms import (
    archive_term,
    familiarity,
    merge_terms,
    new_term,
    record_evidence,
    restore_term,
)

NOW = datetime(2026, 11, 23, tzinfo=timezone.utc)


def _card(term_id: str, term: str, domain: str = "jvm", **kwargs):
    return new_term(term_id, term, domain, config=load_config(), **kwargs)


def test_negative_evidence_does_not_reset_decay():
    config = load_config()
    card = _card("t_0001", "GC")
    record_evidence(card, "self_report", time="2026-09-01", config=config)
    record_evidence(card, "self_report", time="2026-09-02", config=config)

    later = datetime(2026, 11, 1, tzinfo=timezone.utc)
    record_evidence(
        card,
        "self_report_unknown",
        time=later.isoformat(),
        config=config,
    )

    view = familiarity(card, now=later, config=config)
    assert view["retention"] < 0.001
    assert view["id"] == "unknown"


def test_merge_restore_does_not_double_count_evidence():
    source = _card("t_0001", "旧名")
    target = _card("t_0002", "新名")
    record_evidence(source, "explained", time="2026-11-19", config=load_config())
    terms = [source, target]

    merge_terms(terms, "t_0001", "t_0002", reason="改名")
    assert source["evidence"] == []
    assert len(target["evidence"]) == 1

    restore_term(source, now=NOW)
    assert source["evidence"] == []
    assert len(target["evidence"]) == 1


def test_merge_unions_domains_and_keeps_canon():
    source = _card(
        "t_0001",
        "旧名",
        domain="jvm",
        domains=["jvm", "interview"],
    )
    target = _card("t_0002", "新名", domain="jvm")
    record_evidence(source, "explained", time="2026-11-19", config=load_config())
    source["status"] = "canon"
    terms = [source, target]

    merge_terms(terms, "t_0001", "t_0002", reason="改名")

    assert target["domains"] == ["jvm", "interview"]
    assert target["status"] == "canon"


def test_rearchive_keeps_merge_link():
    source = _card("t_0001", "旧名")
    target = _card("t_0002", "新名")
    terms = [source, target]
    merge_terms(terms, "t_0001", "t_0002", reason="改名")

    archive_term(source, reason="再次归档")

    assert source["status"] == "archived"
    assert source["merged_into"] == "t_0002"


def test_write_lookup_follows_merged_name():
    source = _card("t_0001", "旧名")
    target = _card("t_0002", "新名")
    terms = [source, target]
    merge_terms(terms, "t_0001", "t_0002", reason="改名")

    found = find_by_name_exact(terms, "旧名", "jvm")

    assert found is target


def test_domain_fallback_uses_explain_not_level_id(tmp_path):
    config = load_config()
    custom = replace(
        config,
        levels=(
            Level("vague", "模糊", 0.0, "define"),
            Level("solid", "扎实", 0.5, "none"),
        ),
    )
    card = _card("t_0001", "GC")
    record_evidence(card, "self_report", time=NOW.isoformat(), config=config)
    save_terms(tmp_path, [card])

    result = reader_slice(
        tmp_path,
        "safepoint",
        domain="jvm",
        config=custom,
        now=NOW,
    )

    assert [item["term"] for item in result["ranked"]] == ["GC"]


def test_zero_budget_domain_profile_still_reports_advice(tmp_path):
    config = load_config()
    card = _card("t_0001", "GC")
    record_evidence(card, "self_report", time=NOW.isoformat(), config=config)
    save_terms(tmp_path, [card])

    result = domain_snapshot(
        tmp_path,
        "jvm",
        max_terms=0,
        config=config,
        now=NOW,
    )

    assert result["counts"]["terms"] == 1
    assert "GC" in result["advice"]


def test_brief_coverage_counts_query_hits_not_domain_cards(tmp_path):
    config = load_config()
    cards = []
    for index, term in enumerate(("GC", "safepoint", "G1"), 1):
        card = _card(f"t_{index:04d}", term)
        record_evidence(card, "self_report", time=NOW.isoformat(), config=config)
        cards.append(card)
    save_terms(tmp_path, cards)

    result = reader_slice(
        tmp_path,
        "GC",
        domain="jvm",
        include_domain_fallback=False,
        config=config,
        now=NOW,
    )

    assert result["summary"]["candidate_terms"] == 1


def test_glossary_add_returns_persisted_card(tmp_path):
    service = ReaderService(
        tmp_path / "reader",
        allow_write=True,
        glossary_root=tmp_path / "glossary",
    )
    service.glossary_add("Aether", "project", "demo", definition="first")
    second = service.glossary_add("Aether", "project", "demo", definition="second")

    persisted = {item["id"] for item in service.glossary_list("demo")}
    assert second["id"] in persisted
    assert len(persisted) == 1


def test_observe_rejects_self_report_kinds(tmp_path):
    service = ReaderService(tmp_path / "reader", allow_write=True)

    with pytest.raises(ValueError, match="does not accept"):
        service.observe("GC", "jvm", "self_report")
    with pytest.raises(ValueError, match="does not accept"):
        service.observe("GC", "jvm", "labeled_known")


def test_annotate_finds_cjk_term_glued_to_latin(tmp_path):
    config = load_config()
    card = _card("t_0001", "索引", domain="db")
    record_evidence(card, "explained", time=NOW.isoformat(), config=config)
    save_terms(tmp_path, [card])

    result = annotate(
        tmp_path,
        "MySQL索引 会加速查询。",
        domain="db",
        config=config,
        now=NOW,
    )
    by_text = {item["text"]: item for item in result["candidates"]}

    assert by_text["索引"]["status"] == "known"
    assert by_text["索引"]["fragment"] is False


def test_bu_shi_is_not_a_definition():
    assert term_is_defined("GC 是不是会停顿？", "GC") is False
    assert term_is_defined("GC 是否会导致停顿？", "GC") is False


def test_find_leaks_skips_leading_blank_line():
    leaks = find_leaks("\n正文\nGC 会在这里停顿。")

    assert leaks[0]["line"] == 2
    assert leaks[0]["hits"] == ["占位标签"]


def test_english_merge_is_bounded():
    text = "The safepoint is where GC pauses"
    candidates = make_extractor("jieba").extract(text)

    merged = merge_english_phrases(text, candidates)

    assert all(item.text.count(" ") <= 2 for item in merged)
    assert "The safepoint is where GC pauses" not in {
        item.text for item in merged
    }


class _CountingJudge:
    def __init__(self):
        self.calls = 0
        self.last_usage: dict = {}

    def judge(self, text, contexts, **kwargs):
        self.calls += 1
        return [
            JargonDecision(term=item["term"], action="skip", confidence=0.9)
            for item in contexts
        ]


def test_revise_reuses_cached_judge_without_judge_flag(tmp_path):
    seed = _card("t_0001", "GC")
    save_terms(tmp_path, [seed])
    judge = _CountingJudge()
    service = ReaderService(tmp_path, allow_write=True, judge=judge)

    first = service.annotate("C1 会影响停顿。", domain="jvm", judge=True)
    revised = service.revise("C1 会影响停顿。", domain="jvm", judge=False)

    assert first["undefined"] == []
    assert revised["undefined_before"] == []
    assert judge.calls == 1


def test_revise_reuses_judge_after_write_annotate(tmp_path):
    seed = _card("t_0001", "GC")
    save_terms(tmp_path, [seed])
    judge = _CountingJudge()
    service = ReaderService(tmp_path, allow_write=True, judge=judge)

    service.annotate(
        "C1 会影响停顿。",
        domain="jvm",
        judge=True,
        record_reviews=True,
    )
    revised = service.revise("C1 会影响停顿。", domain="jvm", judge=False)

    assert revised["undefined_before"] == []
    assert judge.calls == 1


def test_revise_reports_leaks_introduced_by_editor(tmp_path):
    seed = _card("t_0001", "GC")
    save_terms(tmp_path, [seed])
    service = ReaderService(tmp_path, allow_write=True)

    class _Editor:
        def revise(self, request):
            from haoliang.revision import RevisionResult

            return RevisionResult(
                text=request.draft + " 账本里还有一句。",
                notes="test",
            )

    service._reviser = _Editor()
    result = service.revise("C1 会影响停顿。", domain="jvm")

    assert result["leaks"]
    assert any("账本" in item["hits"] for item in result["leaks"])
