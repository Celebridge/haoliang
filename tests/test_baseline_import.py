"""Baseline questionnaire import: parsing, level mapping, evidence safety."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from haoliang.reader_config import load_config
from haoliang.term_store import load_terms, save_terms
from haoliang.terms import can_promote, familiarity, new_term, record_evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from import_baseline import (
    _load_spec,
    _score_to_level,
    apply_entries,
    parse_json_spec,
    parse_markdown,
)

SPEC_PATH = ROOT / "docs" / "onboarding" / "baseline-java-2026-11-22.json"
SHEET_PATH = ROOT / "docs" / "onboarding" / "baseline-java-2026-11-22.md"
NOW = datetime(2026, 11, 23, tzinfo=timezone.utc)


def _entries():
    spec = _load_spec(SPEC_PATH)
    entries, warnings = parse_markdown(SHEET_PATH, spec["groups"])
    assert not warnings
    return spec, entries


def test_sheet_and_spec_agree_on_all_82_terms():
    _spec, entries = _entries()
    json_entries, _ = parse_json_spec(SPEC_PATH)

    assert len(entries) == 82
    assert len(json_entries) == 82
    assert {(domain, term) for domain, term, _ in entries} == {
        (domain, term) for domain, term, _ in json_entries
    }
    assert all("*" not in term for _, term, _ in entries)
    assert ("db", "最左前缀", "") in entries
    assert ("db", "联合索引", "") in entries


def test_unfilled_sheet_reports_blank_rows():
    _spec, entries = _entries()

    assert sum(1 for _, _, score in entries if score) == 0
    assert sum(1 for _, _, score in entries if not score) == 82


def test_shared_name_uses_its_questionnaire_section(tmp_path):
    sheet = tmp_path / "answers.md"
    sheet.write_text("## 1. 数据库\n节点 2\n## 2. K8s\n节点 0\n", encoding="utf-8")
    groups = [
        {"id": "db", "title": "数据库", "terms": [{"term": "节点"}]},
        {"id": "k8s", "title": "K8s", "terms": [{"term": "节点"}]},
    ]

    entries, warnings = parse_markdown(sheet, groups)

    assert entries == [("db", "节点", "2"), ("k8s", "节点", "0")]
    assert not warnings


def test_import_maps_scores_to_configured_levels(tmp_path):
    spec, entries = _entries()
    config = load_config()
    filled = [(domain, term, "2") for domain, term, _ in entries]

    report = apply_entries(
        tmp_path,
        filled,
        now=NOW,
        config=config,
        score_to_level=_score_to_level(spec),
    )

    assert report["created"] == 82
    cards = load_terms(tmp_path)
    assert len(cards) == 82
    for card in cards:
        view = familiarity(card, now=NOW, config=config)
        assert view["id"] == "usable"
        assert card["evidence"][0]["source"] == "baseline:questionnaire"
        assert card["evidence"][0]["kind"] == "self_report"


def test_score_three_reaches_explainable_without_gold_label(tmp_path):
    spec, entries = _entries()
    config = load_config()
    filled = [
        (domain, term, "3" if term == "synchronized" else "1")
        for domain, term, _ in entries
    ]

    apply_entries(
        tmp_path,
        filled,
        now=NOW,
        config=config,
        score_to_level=_score_to_level(spec),
    )

    by_term = {card["term"]: card for card in load_terms(tmp_path)}
    assert familiarity(by_term["synchronized"], now=NOW, config=config)["id"] == "explainable"
    assert by_term["synchronized"]["evidence"][0]["kind"] == "self_report"
    assert familiarity(by_term["HashMap"], now=NOW, config=config)["id"] == "heard"


def test_reimport_score_zero_clears_baseline_evidence(tmp_path):
    spec, entries = _entries()
    config = load_config()
    all_two = [(domain, term, "2") for domain, term, _ in entries]
    apply_entries(
        tmp_path,
        all_two,
        now=NOW,
        config=config,
        score_to_level=_score_to_level(spec),
    )

    changed = [
        (domain, term, "0" if term == "联合索引" else "2")
        for domain, term, _ in entries
    ]
    apply_entries(
        tmp_path,
        changed,
        now=NOW,
        config=config,
        score_to_level=_score_to_level(spec),
    )

    card = next(c for c in load_terms(tmp_path) if c["term"] == "联合索引")
    assert card["status"] == "proposal"
    assert card["evidence"] == []
    assert card["last_evidence_at"] is None


def test_baseline_survives_more_than_a_few_days(tmp_path):
    spec, entries = _entries()
    config = load_config()
    term = "synchronized"
    domain = next(d for d, t, _ in entries if t == term)
    apply_entries(
        tmp_path,
        [(domain, term, "2")],
        now=NOW,
        config=config,
        score_to_level=_score_to_level(spec),
    )

    card = next(c for c in load_terms(tmp_path) if c["term"] == term)
    week_later = datetime(2026, 11, 30, tzinfo=timezone.utc)
    assert familiarity(card, now=week_later, config=config)["id"] == "usable"


def test_import_never_overwrites_user_evidence(tmp_path):
    spec, entries = _entries()
    config = load_config()
    domain, term, _ = next(item for item in entries if item[1] == "联合索引")
    card = new_term("t_0001", term, domain, config=config)
    record_evidence(
        card,
        "labeled_known",
        weight=0.8,
        source="user:explicit_label",
        time=NOW.isoformat(),
        config=config,
    )
    save_terms(tmp_path, [card])

    filled = [(d, t, "3") for d, t, _ in entries]
    report = apply_entries(
        tmp_path,
        filled,
        now=NOW,
        config=config,
        score_to_level=_score_to_level(spec),
    )

    saved = next(c for c in load_terms(tmp_path) if c["term"] == term)
    assert saved["evidence"] == card["evidence"]
    assert any("non-baseline" in reason for _, reason in report["skipped"])


def test_baseline_self_report_never_satisfies_strong_gate():
    config = load_config()
    card = new_term("t_0001", "synchronized", "java", config=config)
    record_evidence(
        card,
        "self_report",
        weight=0.9,
        source="baseline:questionnaire",
        time="2026-11-20",
        config=config,
    )
    record_evidence(card, "used", time="2026-11-21", config=config)
    record_evidence(card, "used", time="2026-11-22", config=config)

    check = can_promote(card, now=NOW, config=config)

    assert check["ok"] is False
    assert check["counts"]["strong"] == 0


def test_duplicate_rows_with_different_scores_abort(tmp_path):
    spec, _ = _entries()
    config = load_config()
    entries = [("db", "联合索引", "2"), ("db", "联合索引", "3")]

    with pytest.raises(ValueError, match="duplicate row"):
        apply_entries(
            tmp_path,
            entries,
            now=NOW,
            config=config,
            score_to_level=_score_to_level(spec),
        )
