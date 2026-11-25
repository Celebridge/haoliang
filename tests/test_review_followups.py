"""Regressions for the reader review follow-ups."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from haoliang import terms
from haoliang.jargon import JargonDecision, LLMJargonJudge
from haoliang.llm import ChatResult
from haoliang.reader_config import DEFAULT_CONFIG_PATH, load_config
from haoliang.reader_tools import ReaderService
from haoliang.term_store import edit_terms, load_terms, save_terms
from haoliang.terms import add_pending_review, merge_terms, new_term, record_evidence

NOW = datetime(2026, 11, 24, tzinfo=timezone.utc)
SRC = Path(__file__).resolve().parents[1] / "src"


class ReviewJudge:
    calls = 0

    def judge(self, draft, candidates, **kwargs):
        self.calls += 1
        return [
            JargonDecision(term=item["term"], action="review", reason="需确认")
            for item in candidates
            if item["term"] == "C1"
        ]


def test_judge_writes_pending_on_latest_snapshot(tmp_path):
    class JudgeWithConcurrentWriter(ReviewJudge):
        def judge(self, draft, candidates, **kwargs):
            subprocess.run(
                [
                    sys.executable, "-m", "haoliang.cli", "observe", "Redis",
                    "--domain", "db", "--kind", "used", "--store", str(tmp_path),
                    "--glossary-root", str(tmp_path / "glossary"),
                ],
                env={**os.environ, "PYTHONPATH": str(SRC)},
                check=True,
                capture_output=True,
                timeout=15,
            )
            return super().judge(draft, candidates, **kwargs)

    judge = JudgeWithConcurrentWriter()
    service = ReaderService(tmp_path, allow_write=True, judge=judge)

    result = service.annotate("C1 会影响停顿。", domain="jvm", judge=True, record_reviews=True)

    assert not result["judge"]["error"]
    assert result["judge"]["reviews_written"] == 1
    cards = load_terms(tmp_path)
    assert [(card["id"], card["term"]) for card in cards] == [
        ("t_0001", "Redis"), ("t_0002", "C1"),
    ]
    assert cards[0]["evidence"][0]["kind"] == "used"
    assert cards[1]["pending_reviews"][0]["reason"] == "需确认"

    revised = service.revise("C1 会影响停顿。", domain="jvm", max_terms=0)
    assert not revised["undefined_before"]
    assert [item["term"] for item in revised["review"]] == ["C1"]
    assert judge.calls == 1


def test_failed_ledger_edit_does_not_commit(tmp_path):
    with pytest.raises(RuntimeError), edit_terms(tmp_path) as cards:
        cards.append(new_term("t_0001", "GC", "jvm"))
        raise RuntimeError("abort update")

    assert load_terms(tmp_path) == []
    assert ReaderService(tmp_path, allow_write=True).propose("GC", "jvm")["id"] == "t_0001"


@pytest.mark.parametrize("older_merge", [False, True])
def test_merge_keeps_pending_details_sensitive(tmp_path, older_merge):
    source = new_term("t_0001", "旧名称", "db", sensitivity="sensitive")
    target = new_term("t_0002", "目标词", "db")
    add_pending_review(source, source="test", domain="db", reason="待确认的敏感说明")
    cards = [source, target]
    merge_terms(cards, "t_0001", "t_0002")
    assert target["sensitivity"] == "sensitive"
    if older_merge:
        target["sensitivity"] = "private"
    save_terms(tmp_path, cards)

    service = ReaderService(tmp_path)

    assert service.pending_list() == []
    assert service.brief("目标词", domain="db")["coverage"]["terms"] == 0


def test_brief_uses_domain_for_same_name_and_follows_merged_name(tmp_path, monkeypatch):
    monkeypatch.setattr(terms, "_now", lambda: NOW)
    old = new_term("t_0001", "树节点", "db", now=NOW)
    known = new_term("t_0002", "节点", "db", now=NOW)
    unknown = new_term("t_0003", "节点", "k8s", now=NOW)
    record_evidence(known, "explained", time=NOW.isoformat())
    cards = [old, known, unknown]
    merge_terms(cards, "t_0001", "t_0002", now=NOW)
    save_terms(tmp_path, cards)
    service = ReaderService(tmp_path)

    for name in ("节点", "树节点"):
        result = service.brief(f"解释{name}", domain="db")
        assert [item["id"] for item in result["known"]] == ["t_0002"]
        assert not result["unknown"]
    assert service.brief("节点", domain="k8s")["unknown"][0]["id"] == "t_0003"
    assert service.brief("树节点", domain="k8s")["known"][0]["id"] == "t_0002"


def test_cached_judge_does_not_freeze_familiarity(tmp_path, monkeypatch):
    card = new_term("t_0001", "GC", "jvm", now=NOW)
    record_evidence(card, "explained", time=NOW.isoformat())
    save_terms(tmp_path, [card])
    judge = ReviewJudge()
    service = ReaderService(tmp_path, judge=judge)
    draft = "GC 会停顿。C1 会影响停顿。"

    monkeypatch.setattr(terms, "_now", lambda: NOW)
    assert "GC" not in service.annotate(draft, domain="jvm", judge=True)["undefined"]
    monkeypatch.setattr(terms, "_now", lambda: NOW + timedelta(days=3))
    result = service.annotate(draft, domain="jvm", judge=True)

    assert "GC" in result["undefined"]
    assert [item["term"] for item in result["review"]] == ["C1"]
    assert judge.calls == 1


def test_malformed_judge_output_can_retry_same_draft(tmp_path):
    class Client:
        calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            response = "truncated json" if self.calls == 1 else '{"decisions":[]}'
            return ChatResult(text=response, usage={})

    client = Client()
    service = ReaderService(tmp_path, judge=LLMJargonJudge(client=client))

    assert service.annotate("C1 会停顿。", judge=True)["judge"]["error"]
    assert not service.annotate("C1 会停顿。", judge=True)["judge"]["error"]
    assert not service.annotate("C1 会停顿。", judge=True)["judge"]["error"]
    assert client.calls == 2


def test_hot_reload_rebuilds_default_model_adapters(tmp_path):
    config_path = tmp_path / "reader.json"
    data = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config_path.write_text(json.dumps(data), encoding="utf-8")
    service = ReaderService(tmp_path, config_path=config_path)
    service.brief("GC")
    judge, reviser = service._jargon_judge(), service._revision_engine()

    data["jargon"]["confidence_threshold"] = 0.99
    data["revision"]["max_growth_chars"] = 0
    before = config_path.stat().st_mtime_ns
    config_path.write_text(json.dumps(data), encoding="utf-8")
    os.utime(config_path, ns=(before + 1_000_000_000, before + 1_000_000_000))

    service.brief("GC")

    assert service._jargon_judge() is not judge
    assert service._jargon_judge().rules.confidence_threshold == 0.99
    assert service._revision_engine() is not reviser
    assert service._revision_engine().rules.max_growth_chars == 0
    assert load_config(config_path) == service.runtime.config
