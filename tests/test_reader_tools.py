"""ReaderService: long-lived primitives for MCP and harnesses."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from haoliang.jargon import JargonDecision
from haoliang.reader_tools import ReaderService
from haoliang.revision import RevisionResult
from haoliang.term_store import load_terms, save_terms
from haoliang.terms import add_pending_review, new_term, record_evidence


def _seed(tmp_path):
    store = tmp_path / "reader"
    now = datetime.now(timezone.utc)
    safepoint = new_term(
        "t_0001",
        "safepoint",
        "jvm",
        aliases=["安全点"],
        now=now,
    )
    record_evidence(safepoint, "labeled_known", time=now.isoformat())
    gc = new_term("t_0002", "GC", "jvm", now=now)
    save_terms(store, [safepoint, gc])
    return store, now


def test_domain_profile_groups_terms(tmp_path):
    store, _ = _seed(tmp_path)
    service = ReaderService(store, allow_write=False)

    profile = service.domain_profile("jvm")

    assert [item["term"] for item in profile["known"]] == ["safepoint"]
    assert [item["term"] for item in profile["unknown"]] == ["GC"]
    assert profile["counts"]["terms"] == 2


def test_brief_and_annotate(tmp_path):
    store, _ = _seed(tmp_path)
    service = ReaderService(store, allow_write=True, glossary_root=tmp_path / "glossary")

    brief = service.brief("GC 怎么调")
    result = service.annotate("GC 会在这里停顿。", domain="jvm")

    assert "GC" in [item["term"] for item in brief["unknown"]]
    assert "GC" in result["undefined"]


def test_write_guard(tmp_path):
    store, _ = _seed(tmp_path)
    service = ReaderService(store, allow_write=False)

    with pytest.raises(PermissionError):
        service.observe("GC", "jvm", "used")


def test_observe_and_pending_resolve(tmp_path):
    store, _ = _seed(tmp_path)
    service = ReaderService(store, allow_write=True, glossary_root=tmp_path / "glossary")

    observed = service.observe("GC", "jvm", "used", quote="用了 GC")
    assert observed["evidence_count"] == 1

    terms = load_terms(store)
    gc = next(item for item in terms if item["term"] == "GC")
    add_pending_review(
        gc,
        source="test",
        domain="jvm",
        reason="unsure",
        time="2026-11-19",
    )
    save_terms(store, terms)
    service.runtime.invalidate()

    assert service.pending_list()[0]["term"] == "GC"
    resolved = service.pending_resolve(gc["id"], "known", confirmed_by="user")
    assert resolved["pending_reviews"] == []
    assert resolved["status"] == "proposal"
    assert resolved["familiarity"]["id"] == "usable"
    assert any(
        event["kind"] == "self_report"
        for event in next(
            item for item in load_terms(store) if item["term"] == "GC"
        )["evidence"]
    )


class CountingJudge:
    def __init__(self):
        self.calls = 0
        self.last_usage: dict = {}

    def judge(self, text, contexts, **kwargs):
        self.calls += 1
        return [
            JargonDecision(term=item["term"], action="skip", confidence=0.9)
            for item in contexts
        ]


def test_annotation_cache_reuses_judge_for_same_text(tmp_path):
    store, _ = _seed(tmp_path)
    judge = CountingJudge()
    service = ReaderService(store, allow_write=True, judge=judge)

    first = service.annotate("C1 会影响停顿。", domain="jvm", judge=True)
    second = service.annotate("C1 会影响停顿。", domain="jvm", judge=True)
    revised = service.revise("C1 会影响停顿。", domain="jvm", judge=True)

    assert judge.calls == 1
    assert first["candidates"] == second["candidates"]
    assert first["judge"] == second["judge"]
    assert revised["undefined_before"] == []


def test_annotation_cache_clears_after_write(tmp_path):
    store, _ = _seed(tmp_path)
    judge = CountingJudge()
    service = ReaderService(store, allow_write=True, judge=judge)

    service.annotate("C1 会影响停顿。", domain="jvm", judge=True)
    service.observe("GC", "jvm", "used")
    service.annotate("C1 会影响停顿。", domain="jvm", judge=True)

    assert judge.calls == 2


def test_brief_empty_is_explicit_about_coverage(tmp_path):
    store, _ = _seed(tmp_path)
    service = ReaderService(store, allow_write=False)

    brief = service.brief("Kafka 的 ISR 是什么", domain="jvm")

    assert brief["known"] == []
    assert brief["uncertain"] == []
    assert brief["unknown"] == []
    assert "不要据此推断读者水平" in brief["advice"]
    assert brief["coverage"]["domain_terms"] >= 1


def test_brief_excludes_unrelated_domain_fallback(tmp_path):
    store, _ = _seed(tmp_path)
    terms = load_terms(store)
    terms.append(new_term("t_0003", "C1", "jvm", now=datetime(2026, 11, 19, tzinfo=timezone.utc)))
    save_terms(store, terms)
    service = ReaderService(store, allow_write=False)

    brief = service.brief("safepoint 是什么", domain="jvm")

    assert [item["term"] for item in brief["known"]] == ["safepoint"]
    assert "C1" not in [item["term"] for item in brief["unknown"]]
    assert "C1" not in brief["advice"]


def test_revise_reports_undefined_after(tmp_path):
    store, _ = _seed(tmp_path)
    service = ReaderService(store, allow_write=True)

    class FakeReviser:
        def revise(self, request):
            return RevisionResult(
                text=request.draft.replace("C1", "C1（客户端即时编译器）", 1),
                notes="test",
            )

    service._reviser = FakeReviser()
    result = service.revise("C1 会影响停顿。", domain="jvm")

    assert result["undefined_before"] == ["C1"]
    assert result["undefined_after"] == []
    assert result["final"].startswith("C1（客户端即时编译器）")


def test_revise_notes_list_uncovered_terms(tmp_path):
    store, _ = _seed(tmp_path)
    service = ReaderService(store, allow_write=True)

    result = service.revise("C1 会影响停顿。", domain="jvm", max_terms=0)

    assert result["undefined_after"] == ["C1"]
    assert "未覆盖" in result["notes"]
    assert "C1" in result["notes"]


def test_annotation_cache_clears_after_external_write(tmp_path):
    store, _ = _seed(tmp_path)
    judge = CountingJudge()
    service = ReaderService(store, allow_write=True, judge=judge)

    service.annotate("C1 会影响停顿。", domain="jvm", judge=True)
    terms = load_terms(store)
    terms.append(
        new_term("t_0090", "C1", "jvm", now=datetime(2026, 11, 20, tzinfo=timezone.utc))
    )
    save_terms(store, terms)

    service.annotate("C1 会影响停顿。", domain="jvm", judge=True)

    assert judge.calls == 2


def test_pending_archive_does_not_require_user_confirmation(tmp_path):
    store, _ = _seed(tmp_path)
    service = ReaderService(store, allow_write=True)
    terms = load_terms(store)
    gc = next(item for item in terms if item["term"] == "GC")
    add_pending_review(
        gc,
        source="test",
        domain="jvm",
        reason="unsure",
        time="2026-11-20",
    )
    save_terms(store, terms)
    service.runtime.invalidate()

    result = service.pending_resolve(gc["id"], "archive")

    assert result["status"] == "archived"
    assert result["pending_reviews"] == []


def test_propose_creates_a_card_in_the_requested_domain(tmp_path):
    store, _ = _seed(tmp_path)
    service = ReaderService(store, allow_write=True)

    service.propose("节点", "k8s")
    service.propose("节点", "db")

    domains = [
        item["domain"]
        for item in load_terms(store)
        if item["term"] == "节点"
    ]
    assert set(domains) == {"k8s", "db"}


def test_agent_cannot_certify_pending_known(tmp_path):
    store, _ = _seed(tmp_path)
    service = ReaderService(store, allow_write=True)

    with pytest.raises(ValueError):
        service.pending_resolve("t_0002", "known")


def test_local_glossary_blocks_observe(tmp_path):
    store, _ = _seed(tmp_path)
    service = ReaderService(store, allow_write=True, glossary_root=tmp_path / "glossary")

    service.glossary_add(
        "Aether",
        "project",
        "demo",
        definition="失败补偿层",
    )

    with pytest.raises(ValueError):
        service.observe("Aether", "demo", "asked")


def test_propose_creates_proposal(tmp_path):
    store, _ = _seed(tmp_path)
    service = ReaderService(store, allow_write=True)

    result = service.propose("ZGC", "jvm")

    assert result["status"] == "proposal"
    assert result["term"] == "ZGC"


def test_revise_without_undefined_does_not_call_editor(tmp_path):
    store, _ = _seed(tmp_path)
    service = ReaderService(store, allow_write=True)

    result = service.revise("safepoint 是安全点。", domain="jvm")

    assert result["final"] == "safepoint 是安全点。"
    assert result["notes"] == "nothing to revise"
