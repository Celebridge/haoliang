"""term 卡的层级、证据累积、升级与遗忘测试。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from haoliang.reader_config import (
    DecayRules,
    Level,
    ReaderConfig,
    load_config,
)
from haoliang.terms import (
    can_promote,
    confirm,
    effective_knowledge,
    familiarity,
    new_term,
    record_evidence,
    term_retention,
    validate_term,
)


def _card() -> dict:
    return new_term("t_0001", "safepoint", "jvm", now=datetime(2026, 11, 1, tzinfo=timezone.utc))


def _with_promotion(config: ReaderConfig, **changes) -> ReaderConfig:
    return ReaderConfig(
        levels=config.levels,
        evidence_weights=config.evidence_weights,
        promotion=replace(config.promotion, **changes),
        decay=config.decay,
    )


def test_levels_are_thresholds_not_indexes():
    config = load_config()

    assert config.level_of(0.0).id == "unknown"
    assert config.level_of(0.3).id == "heard"
    assert config.level_of(0.7).id == "usable"
    assert config.level_of(0.95).id == "explainable"


def test_custom_level_needs_no_code_change():
    config = load_config()
    extra = Level("can-debug", "能调试", 0.8, "reason")
    custom = ReaderConfig(
        levels=config.levels + (extra,),
        evidence_weights=config.evidence_weights,
        promotion=config.promotion,
        decay=config.decay,
    )
    custom = ReaderConfig(
        levels=tuple(sorted(custom.levels, key=lambda level: level.min_knowledge)),
        evidence_weights=custom.evidence_weights,
        promotion=custom.promotion,
        decay=custom.decay,
    )

    assert custom.level_of(0.85).id == "can-debug"
    assert custom.level_of(0.95).id == "explainable"


def test_self_report_alone_cannot_promote():
    config = load_config()
    card = _card()
    now = datetime(2026, 11, 2, tzinfo=timezone.utc)

    record_evidence(card, "self_report", time="2026-11-02", config=config)
    check = can_promote(card, now=now, config=config)

    assert check["ok"] is False
    assert check["counts"]["positive"] == 1
    assert check["counts"]["strong"] == 0


def test_three_positive_events_on_one_day_cannot_promote():
    config = load_config()
    card = _card()
    now = datetime(2026, 11, 17, tzinfo=timezone.utc)

    for hour in ("09:00:00", "12:00:00", "18:00:00"):
        record_evidence(card, "used", time=f"2026-11-17T{hour}", config=config)

    check = can_promote(card, now=now, config=config)

    assert check["counts"]["positive"] == 3
    assert check["counts"]["distinct_days"] == 1
    assert check["ok"] is False


def test_two_days_with_strong_evidence_can_promote():
    config = load_config()
    card = _card()
    now = datetime(2026, 11, 18, tzinfo=timezone.utc)

    record_evidence(card, "used", time="2026-11-16", config=config)
    record_evidence(card, "used", time="2026-11-17", config=config)
    record_evidence(card, "explained", time="2026-11-17", config=config)

    check = can_promote(card, now=now, config=config)
    assert check["ok"] is True
    assert check["counts"] == {
        "positive": 3,
        "distinct_days": 2,
        "strong": 1,
        "recent_negative": 0,
    }

    confirm(card, now=now, config=config)
    assert card["status"] == "canon"
    assert validate_term(card) == []


def test_decay_downgrades_effective_level():
    config = load_config()
    card = _card()
    record_evidence(card, "explained", time="2026-01-01", config=config)

    later = datetime(2026, 2, 1, tzinfo=timezone.utc)
    retention = term_retention(card, now=later, config=config)
    knowledge = effective_knowledge(card, now=later, config=config)
    view = familiarity(card, now=later, config=config)

    assert retention < 0.01
    assert knowledge < 0.01
    assert view["id"] == "unknown"


def test_promotion_thresholds_are_configurable():
    config = load_config()
    card = _card()
    now = datetime(2026, 11, 18, tzinfo=timezone.utc)

    record_evidence(card, "used", time="2026-11-16", config=config)
    record_evidence(card, "explained", time="2026-11-17", config=config)

    assert can_promote(card, now=now, config=config)["ok"] is False

    relaxed = _with_promotion(config, min_positive_evidence=2)
    assert can_promote(card, now=now, config=relaxed)["ok"] is True


def test_canon_without_evidence_is_invalid():
    card = _card()
    card["status"] = "canon"

    errors = validate_term(card)

    assert "canon term must have at least one evidence item" in errors


def test_strong_evidence_grows_stability_and_negative_shrinks_it():
    config = load_config()
    card = _card()

    record_evidence(card, "explained", time="2026-11-01", config=config)
    grown = card["stability_days"]
    assert grown > config.decay.initial_stability_days

    record_evidence(card, "asked", time="2026-11-02", config=config)
    assert card["stability_days"] < grown


def test_custom_decay_config_changes_retention():
    config = load_config()
    later = datetime(2026, 1, 10, tzinfo=timezone.utc)

    fast = ReaderConfig(
        levels=config.levels,
        evidence_weights=config.evidence_weights,
        promotion=config.promotion,
        decay=DecayRules(
            initial_stability_days=1.0,
            growth_factor=1.0,
            shrink_factor=0.5,
            retention_threshold=0.7,
        ),
    )
    slow = ReaderConfig(
        levels=config.levels,
        evidence_weights=config.evidence_weights,
        promotion=config.promotion,
        decay=DecayRules(
            initial_stability_days=30.0,
            growth_factor=1.0,
            shrink_factor=0.5,
            retention_threshold=0.7,
        ),
    )

    fast_card = new_term("t_0002", "x", "jvm", now=datetime(2026, 1, 1, tzinfo=timezone.utc), config=fast)
    fast_card["last_evidence_at"] = "2026-01-01"
    fast_card["evidence"] = [{"kind": "explained", "weight": 0.8, "time": "2026-01-01"}]

    slow_card = new_term("t_0003", "x", "jvm", now=datetime(2026, 1, 1, tzinfo=timezone.utc), config=slow)
    slow_card["last_evidence_at"] = "2026-01-01"
    slow_card["evidence"] = [{"kind": "explained", "weight": 0.8, "time": "2026-01-01"}]

    assert term_retention(fast_card, now=later, config=fast) < term_retention(
        slow_card, now=later, config=slow
    )


def test_new_term_rejects_empty_term_or_domain():
    with pytest.raises(ValueError):
        new_term("t_0001", "", "jvm")
    with pytest.raises(ValueError):
        new_term("t_0002", "GC", "")
