"""Pending review tests: unresolved judgments do not change knowledge."""

from __future__ import annotations

from datetime import datetime, timezone

from haoliang.cli import main
from haoliang.reader_config import load_config
from haoliang.term_store import load_terms
from haoliang.terms import (
    add_pending_review,
    clear_pending_reviews,
    new_term,
    term_knowledge,
    validate_term,
)


def _card():
    return new_term(
        "t_0001",
        "容器",
        "k8s",
        now=datetime(2026, 11, 19, tzinfo=timezone.utc),
        config=load_config(),
    )


def test_pending_review_does_not_touch_knowledge_or_decay():
    card = _card()
    before = (
        term_knowledge(card),
        card["stability_days"],
        card["last_evidence_at"],
    )

    add_pending_review(
        card,
        source="jargon_judge",
        domain="k8s",
        reason="sense=term, gap=unsure",
        confidence=0.52,
        time="2026-11-19T10:00:00+00:00",
    )

    assert term_knowledge(card) == before[0]
    assert card["stability_days"] == before[1]
    assert card["last_evidence_at"] == before[2]
    assert card["pending_reviews"][0]["domain"] == "k8s"
    assert card["pending_reviews"][0]["confidence"] == 0.52


def test_clear_pending_reviews_can_target_one_domain():
    card = _card()
    add_pending_review(
        card, source="judge", domain="k8s", reason="unsure", time="2026-11-19"
    )
    add_pending_review(
        card, source="judge", domain="java", reason="unsure", time="2026-11-19"
    )

    clear_pending_reviews(card, domain="k8s")

    assert [item["domain"] for item in card["pending_reviews"]] == ["java"]


def test_validate_accepts_pending_reviews():
    card = _card()
    add_pending_review(
        card, source="judge", domain="k8s", reason="unsure", time="2026-11-19"
    )

    assert validate_term(card) == []


def test_review_cli_writes_pending_review(tmp_path):
    store = tmp_path / "reader"

    code = main(
        [
            "review",
            "容器",
            "--domain",
            "k8s",
            "--reason",
            "sense=term, gap=unsure",
            "--store",
            str(store),
            "--json",
        ]
    )

    assert code == 0
    cards = load_terms(store)
    assert cards[0]["term"] == "容器"
    assert cards[0]["pending_reviews"][0]["source"] == "jargon_judge"
    assert cards[0]["evidence"] == []


def test_pending_list_and_resolve(tmp_path):
    store = tmp_path / "reader"
    assert (
        main(
            [
                "review",
                "容器",
                "--domain",
                "k8s",
                "--reason",
                "sense=term, gap=unsure",
                "--store",
                str(store),
            ]
        )
        == 0
    )

    assert main(["pending", "list", "--store", str(store)]) == 0
    assert (
        main(
            [
                "pending",
                "resolve",
                "t_0001",
                "--action",
                "known",
                "--store",
                str(store),
            ]
        )
        == 0
    )

    card = load_terms(store)[0]
    assert card["pending_reviews"] == []
    assert card["evidence"][0]["kind"] == "self_report"
    assert card["evidence"][0]["weight"] == 0.45
    assert card["evidence"][0]["source"] == "pending:user"


def test_pending_resolve_gold_writes_labeled_known(tmp_path):
    store = tmp_path / "reader"
    assert (
        main(
            [
                "review",
                "容器",
                "--domain",
                "k8s",
                "--reason",
                "sense=term, gap=unsure",
                "--store",
                str(store),
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "pending",
                "resolve",
                "t_0001",
                "--action",
                "known",
                "--gold",
                "--store",
                str(store),
            ]
        )
        == 0
    )

    card = load_terms(store)[0]
    assert card["evidence"][0]["kind"] == "labeled_known"
    assert card["evidence"][0]["weight"] == 0.9
    assert card["evidence"][0]["source"] == "gold"


def test_pending_resolve_unknown_defaults_to_self_report_unknown(tmp_path):
    store = tmp_path / "reader"
    assert (
        main(
            [
                "review",
                "容器",
                "--domain",
                "k8s",
                "--reason",
                "sense=term, gap=unsure",
                "--store",
                str(store),
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "pending",
                "resolve",
                "t_0001",
                "--action",
                "unknown",
                "--store",
                str(store),
            ]
        )
        == 0
    )

    card = load_terms(store)[0]
    assert card["evidence"][0]["kind"] == "self_report_unknown"
    assert card["evidence"][0]["weight"] == -0.3


def test_self_report_is_deduped_per_day(tmp_path):
    store = tmp_path / "reader"
    assert (
        main(
            [
                "review",
                "容器",
                "--domain",
                "k8s",
                "--reason",
                "sense=term, gap=unsure",
                "--store",
                str(store),
            ]
        )
        == 0
    )

    for attempt in range(2):
        if attempt:
            assert (
                main(
                    [
                        "review",
                        "容器",
                        "--domain",
                        "k8s",
                        "--reason",
                        "sense=term, gap=unsure",
                        "--store",
                        str(store),
                    ]
                )
                == 0
            )
        assert (
            main(
                [
                    "pending",
                    "resolve",
                    "t_0001",
                    "--action",
                    "known",
                    "--store",
                    str(store),
                ]
            )
            == 0
        )

    card = load_terms(store)[0]
    self_reports = [
        event for event in card["evidence"] if event["kind"] == "self_report"
    ]
    assert len(self_reports) == 1
    assert term_knowledge(card) == 0.45


def test_pending_resolve_archive_sets_status(tmp_path):
    store = tmp_path / "reader"
    assert (
        main(
            [
                "review",
                "容器",
                "--domain",
                "k8s",
                "--reason",
                "sense=term, gap=unsure",
                "--store",
                str(store),
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "pending",
                "resolve",
                "t_0001",
                "--action",
                "archive",
                "--store",
                str(store),
            ]
        )
        == 0
    )

    card = load_terms(store)[0]
    assert card["status"] == "archived"
    assert card["pending_reviews"] == []


def test_pending_resolve_targets_one_reason(tmp_path):
    store = tmp_path / "reader"
    for reason in ("sense=term,gap=unsure", "low confidence"):
        assert (
            main(
                [
                    "review",
                    "容器",
                    "--domain",
                    "k8s",
                    "--reason",
                    reason,
                    "--store",
                    str(store),
                ]
            )
            == 0
        )

    assert (
        main(
            [
                "pending",
                "resolve",
                "t_0001",
                "--action",
                "unknown",
                "--reason",
                "low confidence",
                "--store",
                str(store),
            ]
        )
        == 0
    )

    card = load_terms(store)[0]
    assert len(card["pending_reviews"]) == 1
    assert card["pending_reviews"][0]["reason"] == "sense=term,gap=unsure"


def test_draft_command_writes_trace(tmp_path):
    store = tmp_path / "reader"
    trace = tmp_path / "usage.jsonl"
    assert (
        main(
            [
                "propose",
                "GC",
                "--domain",
                "jvm",
                "--store",
                str(store),
            ]
        )
        == 0
    )

    code = main(
        [
            "draft",
            "GC 是什么",
            "--draft",
            "GC 会在这里停顿。",
            "--domain",
            "jvm",
            "--store",
            str(store),
            "--no-revise",
            "--trace",
            str(trace),
            "--json",
        ]
    )

    assert code == 0
    assert trace.exists()
    assert "GC" in trace.read_text(encoding="utf-8")
