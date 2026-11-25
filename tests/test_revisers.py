"""LLMReviser tests for definition insertion and untrusted model outputs."""

from __future__ import annotations

import json

from haoliang.llm import ChatResult
from haoliang.reader_config import RevisionRules, load_config
from haoliang.revisers import LLMReviser, build_editor_prompt
from haoliang.revision import Edit, RevisionRequest


class FakeClient:
    def __init__(self, payload: dict | str):
        self.payload = (
            payload if isinstance(payload, str) else json.dumps(payload)
        )
        self.messages: list[dict[str, str]] = []
        self.kwargs: dict = {}

    def complete(self, messages, **kwargs) -> ChatResult:
        self.messages = messages
        self.kwargs = kwargs
        return ChatResult(text=self.payload, usage={"total_tokens": 10})


def _reviser(payload: dict | str, **rule_changes) -> tuple[LLMReviser, FakeClient]:
    rules = RevisionRules(**rule_changes)
    client = FakeClient(payload)
    return LLMReviser(rules=rules, client=client), client


def test_llm_reviser_inserts_definition():
    reviser, _ = _reviser(
        {
            "definitions": [
                {"term": "GC", "definition": "垃圾回收"},
            ],
            "notes": "给 GC 补了定义",
        }
    )

    result = reviser.revise(
        RevisionRequest(draft="GC 会在这里停顿。", undefined=["GC"])
    )

    assert result.text == "GC（垃圾回收） 会在这里停顿。"
    assert result.edits == (Edit(0, 2, "GC（垃圾回收）"),)
    assert result.usage == {"total_tokens": 10}
    assert "给 GC 补了定义" in result.notes


def test_llm_reviser_rejects_alias_only_definition():
    reviser, _ = _reviser(
        {
            "definitions": [
                {"term": "容器", "definition": "container"},
            ],
            "notes": "给容器补了英文名",
        }
    )

    result = reviser.revise(
        RevisionRequest(draft="容器 负责隔离。", undefined=["容器"])
    )

    assert result.text == "容器 负责隔离。"
    assert "invalid Chinese definition" in result.notes


def test_llm_reviser_inserts_at_first_prose_occurrence():
    reviser, _ = _reviser({"definitions": [{"term": "GC", "definition": "垃圾回收"}]})
    draft = "`GC` 是选项名。GC 可能暂停应用线程。GC 也可能并发执行。"

    result = reviser.revise(RevisionRequest(draft=draft, undefined=["GC"]))

    assert result.text == "`GC` 是选项名。GC（垃圾回收） 可能暂停应用线程。GC 也可能并发执行。"
    assert len(result.edits) == 1


def test_llm_reviser_rejects_definitions_outside_targets():
    reviser, _ = _reviser({"definitions": [{"term": "停顿", "definition": "暂停"}]})
    draft = "GC 会在这里停顿。"

    result = reviser.revise(RevisionRequest(draft=draft, undefined=["GC"]))

    assert result.text == draft
    assert not result.edits


def test_llm_reviser_rejects_definition_that_escapes_parentheses():
    reviser, _ = _reviser(
        {"definitions": [{"term": "GC", "definition": "垃圾回收）从不暂停应用线程。（"}]}
    )
    draft = "GC 可能暂停应用线程。"

    result = reviser.revise(RevisionRequest(draft=draft, undefined=["GC"]))

    assert result.text == draft
    assert not result.edits


def test_llm_reviser_uses_request_budget_when_explicit():
    reviser, client = _reviser({"definitions": []}, max_terms=1)

    reviser.revise(
        RevisionRequest(
            draft="draft",
            undefined=["GC", "K8s", "etcd"],
            max_terms=3,
        )
    )

    prompt = client.messages[1]["content"]
    assert "GC" in prompt
    assert "K8s" in prompt
    assert "etcd" in prompt


def test_llm_reviser_caps_terms_before_the_model_call():
    reviser, client = _reviser({"definitions": []}, max_terms=1)

    reviser.revise(
        RevisionRequest(draft="draft", undefined=["GC", "K8s", "etcd"])
    )

    prompt = client.messages[1]["content"]
    assert "GC" in prompt
    assert "K8s" not in prompt
    assert "etcd" not in prompt


def test_llm_reviser_ignores_legacy_payloads_that_change_claims():
    draft = "GC 可能暂停应用线程。"
    changed = "GC（垃圾回收）从不暂停应用线程。"
    for payload in (
        {"text": changed},
        {"edits": [{"find": draft, "replace": changed}]},
    ):
        reviser, _ = _reviser(payload)
        result = reviser.revise(RevisionRequest(draft=draft, undefined=["GC"]))

        assert result.text == draft
        assert not result.edits


def test_llm_reviser_handles_invalid_json():
    reviser, _ = _reviser("这不是 JSON")

    result = reviser.revise(
        RevisionRequest(draft="GC 会在这里停顿。", undefined=["GC"])
    )

    assert result.text == "GC 会在这里停顿。"
    assert "not valid JSON" in result.notes


def test_llm_reviser_enforces_growth_limit():
    reviser, _ = _reviser(
        {"definitions": [{"term": "GC", "definition": "垃圾回收，Garbage Collection"}]},
        max_growth_chars=1,
    )

    result = reviser.revise(
        RevisionRequest(draft="GC 会在这里停顿。", undefined=["GC"])
    )

    assert result.text == "GC 会在这里停顿。"
    assert "growth limit" in result.notes


def test_revision_rules_come_from_config():
    rules = load_config().revision

    assert rules.max_terms == 5
    assert rules.max_revisions_per_turn == 1
    assert rules.model == "deepseek-chat"


def test_editor_prompt_puts_the_draft_before_the_terms():
    prompt = build_editor_prompt("draft text", ["GC"])

    assert prompt.index("draft text") < prompt.index("GC")
