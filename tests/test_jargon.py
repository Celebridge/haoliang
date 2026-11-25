"""LLM jargon judge: only not_in_ledger, review writes pending."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from haoliang.annotate import annotate
from haoliang.glossary import glossary_path, load_local_terms
from haoliang.jargon import JargonDecision, LLMJargonJudge
from haoliang.llm import ChatResult
from haoliang.reader_config import JargonRules, load_config
from haoliang.term_store import load_terms, save_terms
from haoliang.terms import new_term, record_evidence


class FakeClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.messages: list[dict[str, str]] = []

    def complete(self, messages, **kwargs) -> ChatResult:
        self.messages = messages
        return ChatResult(text=json.dumps(self.payload), usage={"total_tokens": 10})


class BoomClient:
    def complete(self, messages, **kwargs) -> ChatResult:
        raise RuntimeError("provider down")


def _seed(tmp_path):
    store = tmp_path / "reader"
    now = datetime(2026, 11, 19, tzinfo=timezone.utc)
    safepoint = new_term("t_0001", "safepoint", "jvm", now=now)
    record_evidence(safepoint, "labeled_known", time=now.isoformat())
    gc = new_term("t_0002", "GC", "jvm", now=now)
    record_evidence(gc, "asked", time=now.isoformat())
    save_terms(store, [safepoint, gc])
    return store, now


def _payload(action: str, term: str = "容器", confidence: float = 0.9):
    return {
        "decisions": [
            {
                "term": term,
                "sense": "term",
                "gap": "unknown",
                "action": action,
                "confidence": confidence,
                "reason": "K8s 语境",
            }
        ]
    }


def test_judge_limits_terms_and_parses_decision():
    client = FakeClient(_payload("define"))
    judge = LLMJargonJudge(
        rules=JargonRules(max_candidates=1),
        client=client,
    )

    decisions = judge.judge(
        "容器 负责隔离。",
        [
            {"term": "容器", "sentence": "容器 负责隔离。", "status": "not_in_ledger"},
            {"term": "Pod", "sentence": "Pod 是调度单元。", "status": "not_in_ledger"},
        ],
        domain="k8s",
        reader_profile="K8s 弱项",
    )

    assert decisions == [
        JargonDecision(
            term="容器",
            action="define",
            confidence=0.9,
            sense="term",
            gap="unknown",
            reason="K8s 语境",
        )
    ]


def test_low_confidence_skip_becomes_review():
    client = FakeClient(_payload("skip", confidence=0.2))
    judge = LLMJargonJudge(
        rules=JargonRules(confidence_threshold=0.6),
        client=client,
    )

    decisions = judge.judge(
        "容器 负责隔离。",
        [{"term": "容器", "sentence": "容器 负责隔离。", "status": "not_in_ledger"}],
        domain="k8s",
    )

    assert decisions[0].action == "review"


def test_judge_skips_malformed_confidence_without_losing_batch():
    client = FakeClient(
        {
            "decisions": [
                {"term": "容器", "action": "define", "confidence": "high"},
                {"term": "Pod", "action": "skip", "confidence": 0.9},
            ]
        }
    )
    judge = LLMJargonJudge(rules=JargonRules(), client=client)

    decisions = judge.judge(
        "容器 和 Pod。",
        [
            {"term": "容器", "sentence": "容器。", "status": "not_in_ledger"},
            {"term": "Pod", "sentence": "Pod。", "status": "not_in_ledger"},
        ],
        domain="k8s",
    )

    assert [item.term for item in decisions] == ["Pod"]


def test_low_confidence_define_becomes_review():
    client = FakeClient(_payload("define", confidence=0.3))
    judge = LLMJargonJudge(
        rules=JargonRules(confidence_threshold=0.6),
        client=client,
    )

    decisions = judge.judge(
        "容器 负责隔离。",
        [{"term": "容器", "sentence": "容器 负责隔离。", "status": "not_in_ledger"}],
        domain="k8s",
    )

    assert decisions[0].action == "review"


def test_annotate_sends_only_not_in_ledger_to_judge(tmp_path):
    store, now = _seed(tmp_path)
    client = FakeClient(_payload("define"))
    judge = LLMJargonJudge(rules=JargonRules(), client=client)

    result = annotate(
        store,
        "safepoint 是安全点。GC 会停顿。容器 负责隔离。",
        domain="jvm",
        judge=judge,
        record_reviews=False,
        now=now,
    )

    prompt = client.messages[1]["content"]
    assert "- 词：容器" in prompt
    assert "- 词：GC" not in prompt
    assert "- 词：safepoint" not in prompt
    assert result["judge"]["used"] is True
    assert "容器" in result["undefined"]


def test_review_writes_pending_and_is_skipped_by_default(tmp_path):
    store, now = _seed(tmp_path)
    judge = LLMJargonJudge(
        rules=JargonRules(review_as="skip"),
        client=FakeClient(_payload("review", confidence=0.4)),
    )

    result = annotate(
        store,
        "容器 负责隔离。",
        domain="k8s",
        judge=judge,
        record_reviews=True,
        now=now,
    )

    assert "容器" not in result["undefined"]
    assert result["judge"]["reviews_written"] == 1
    card = next(item for item in load_terms(store) if item["term"] == "容器")
    assert card["pending_reviews"][0]["source"] == "jargon_judge"


def test_record_reviews_false_does_not_write(tmp_path):
    store, now = _seed(tmp_path)
    judge = LLMJargonJudge(
        rules=JargonRules(),
        client=FakeClient(_payload("review")),
    )

    annotate(
        store,
        "容器 负责隔离。",
        domain="k8s",
        judge=judge,
        record_reviews=False,
        now=now,
    )

    assert all(item["term"] != "容器" for item in load_terms(store))


def test_code_fragments_are_not_sent_to_judge(tmp_path):
    store, now = _seed(tmp_path)
    client = FakeClient(_payload("define"))
    judge = LLMJargonJudge(rules=JargonRules(), client=client)

    result = annotate(
        store,
        "调优参数是 -XX:MaxGCPauseMillis=200。",
        domain="jvm",
        judge=judge,
        record_reviews=False,
        now=now,
    )

    payload = " ".join(
        message["content"]
        for message in client.messages
        if message["role"] != "system"
    )
    assert "- 词：PauseMillis" not in payload
    assert "- 词：Max" not in payload
    assert "PauseMillis" not in result["undefined"]


def test_project_term_is_defined_but_not_persisted(tmp_path):
    store, now = _seed(tmp_path)
    payload = {
        "decisions": [
            {
                "term": "灰烬",
                "sense": "term",
                "scope": "project",
                "gap": "unknown",
                "action": "define",
                "confidence": 0.9,
                "reason": "项目内部命名",
            }
        ]
    }
    judge = LLMJargonJudge(rules=JargonRules(), client=FakeClient(payload))

    result = annotate(
        store,
        "灰烬 负责失败后的补偿。",
        domain="project",
        judge=judge,
        record_reviews=True,
        now=now,
    )

    assert "灰烬" in result["undefined"]
    assert result["local_terms"][0]["scope"] == "project"
    assert all(item["term"] != "灰烬" for item in load_terms(store))


def test_project_term_writes_local_glossary_not_reader(tmp_path):
    store, now = _seed(tmp_path)
    glossary_root = tmp_path / "glossary"
    payload = {
        "decisions": [
            {
                "term": "容器",
                "sense": "term",
                "scope": "project",
                "gap": "unknown",
                "action": "define",
                "confidence": 0.9,
                "definition": "项目内的隔离单元",
                "reason": "项目内部命名",
            }
        ]
    }
    judge = LLMJargonJudge(rules=JargonRules(), client=FakeClient(payload))

    annotate(
        store,
        "容器 负责隔离。",
        domain="demo",
        judge=judge,
        record_local_terms=True,
        project="demo",
        glossary_root=glossary_root,
        now=now,
    )

    local = load_local_terms(glossary_path("demo", glossary_root))
    assert local[0]["term"] == "容器"
    assert local[0]["scope"] == "project"
    assert local[0]["definition"] == "项目内的隔离单元"
    assert all(item["term"] != "容器" for item in load_terms(store))


def test_model_term_review_does_not_write_pending(tmp_path):
    store, now = _seed(tmp_path)
    payload = {
        "decisions": [
            {
                "term": "容器",
                "sense": "term",
                "scope": "model",
                "gap": "unsure",
                "action": "review",
                "confidence": 0.4,
                "reason": "模型自造词",
            }
        ]
    }
    judge = LLMJargonJudge(rules=JargonRules(), client=FakeClient(payload))

    result = annotate(
        store,
        "容器 是模型临时命名。",
        domain="general",
        judge=judge,
        record_reviews=True,
        now=now,
    )

    assert "容器" not in result["undefined"]
    assert result["local_terms"][0]["scope"] == "model"
    assert all(item["term"] != "容器" for item in load_terms(store))


def test_judge_failure_falls_back_to_define(tmp_path):
    store, now = _seed(tmp_path)
    judge = LLMJargonJudge(rules=JargonRules(), client=BoomClient())

    result = annotate(
        store,
        "容器 负责隔离。",
        domain="k8s",
        judge=judge,
        record_reviews=True,
        now=now,
    )

    assert "容器" in result["undefined"]
    assert "provider down" in result["judge"]["error"]


def test_jargon_config_is_loaded():
    rules = load_config().jargon

    assert rules.enabled is False
    assert rules.review_as == "skip"
    assert rules.max_candidates == 50
    assert rules.reader_profile == ""
