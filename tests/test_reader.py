"""reader_slice 的匹配、分桶、衰减与预算测试。"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from haoliang.cli import main
from haoliang.reader import domain_snapshot, reader_slice
from haoliang.reader_config import Level, ReaderConfig, load_config
from haoliang.term_store import save_terms
from haoliang.terms import new_term, record_evidence


def _seed(store) -> object:
    config = load_config()
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    cards = []

    safepoint = new_term("t_0001", "safepoint", "jvm", now=now, config=config)
    record_evidence(safepoint, "explained", time="2026-11-01", config=config)
    cards.append(safepoint)

    gc = new_term("t_0002", "gc", "jvm", now=now, config=config)
    record_evidence(gc, "asked", time="2026-11-01", config=config)
    cards.append(gc)

    etcd = new_term("t_0003", "etcd", "k8s", now=now, config=config)
    record_evidence(etcd, "asked", time="2026-11-01", config=config)
    cards.append(etcd)

    cri = new_term("t_0004", "CRI", "k8s", now=now, config=config)
    record_evidence(cri, "used", time="2026-11-01", config=config)
    cards.append(cri)

    save_terms(store, cards)
    return config


def test_slice_matches_term_and_marks_known(tmp_path):
    store = tmp_path / "reader"
    config = _seed(store)
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)

    result = reader_slice(store, "safepoint 是什么", now=now, config=config)

    assert [item["term"] for item in result["known"]] == ["safepoint"]
    assert result["known"][0]["action"] == "none"
    assert result["summary"]["known"] == 1


def test_slice_separates_unknown_and_advice(tmp_path):
    store = tmp_path / "reader"
    config = _seed(store)
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)

    result = reader_slice(store, "safepoint 和 etcd", now=now, config=config)

    assert [item["term"] for item in result["unknown"]] == ["etcd"]
    assert "先定义" in result["advice"]


def test_slice_domain_fallback_surfaces_unknown_terms(tmp_path):
    store = tmp_path / "reader"
    config = _seed(store)
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)

    result = reader_slice(store, "解释一下 JVM", domain="jvm", now=now, config=config)

    assert [item["term"] for item in result["unknown"]] == ["gc"]
    assert result["unknown"][0]["match"] == "domain"
    assert result["summary"]["domain_terms"] == 2
    # Only candidates that survived the index narrowing are scored.
    assert result["summary"]["candidate_terms"] == 2


def test_slice_decay_moves_known_to_unknown(tmp_path):
    store = tmp_path / "reader"
    config = load_config()
    card = new_term(
        "t_0001",
        "safepoint",
        "jvm",
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
        config=config,
    )
    record_evidence(card, "explained", time="2026-01-01", config=config)
    save_terms(store, [card])

    result = reader_slice(
        store,
        "safepoint",
        now=datetime(2026, 2, 1, tzinfo=timezone.utc),
        config=config,
    )

    assert result["known"] == []
    assert [item["term"] for item in result["unknown"]] == ["safepoint"]
    assert result["unknown"][0]["action"] == "define"


def test_slice_respects_max_terms(tmp_path):
    store = tmp_path / "reader"
    config = _seed(store)
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)

    result = reader_slice(
        store,
        "解释一下",
        domain="k8s",
        max_terms=1,
        now=now,
        config=config,
    )

    assert len(result["unknown"]) + len(result["uncertain"]) == 1


def test_slice_no_match_returns_empty(tmp_path):
    store = tmp_path / "reader"
    config = _seed(store)
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)

    result = reader_slice(store, "foobar", now=now, config=config)

    assert result["known"] == []
    assert result["unknown"] == []
    assert "没有匹配到词条" in result["advice"]


def test_cli_slice_json(tmp_path, capsys):
    store = tmp_path / "reader"
    _seed(store)
    capsys.readouterr()

    code = main(["slice", "safepoint", "--store", str(store), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    terms = payload["known"] + payload["uncertain"] + payload["unknown"]
    assert any(item["term"] == "safepoint" for item in terms)


def test_reader_slice_supports_custom_levels(tmp_path):
    store = tmp_path / "reader"
    config = load_config()
    custom = ReaderConfig(
        levels=tuple(
            sorted(
                config.levels + (Level("can-debug", "能调试", 0.8, "reason"),),
                key=lambda level: level.min_knowledge,
            )
        ),
        evidence_weights=config.evidence_weights,
        promotion=config.promotion,
        decay=config.decay,
    )
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    card = new_term("t_0001", "safepoint", "jvm", now=now, config=custom)
    record_evidence(card, "explained", time="2026-11-01", config=custom)
    save_terms(store, [card])

    result = reader_slice(store, "safepoint", now=now, config=custom)

    assert result["ranked"][0]["familiarity"] == "can-debug"
    assert result["ranked"][0]["action"] == "reason"
    assert [item["term"] for item in result["known"]] == ["safepoint"]


def test_tie_break_prefers_lower_knowledge(tmp_path):
    store = tmp_path / "reader"
    config = load_config()
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)

    unknown = new_term(
        "t_0001",
        "buffer-unknown",
        "os",
        aliases=["buffer"],
        now=now,
        config=config,
    )
    known = new_term(
        "t_0002",
        "buffer-known",
        "db",
        aliases=["buffer"],
        now=now,
        config=config,
    )
    record_evidence(known, "explained", time="2026-11-01", config=config)
    save_terms(store, [unknown, known])

    result = reader_slice(store, "buffer", now=now, config=config)

    assert [item["id"] for item in result["ranked"][:2]] == ["t_0001", "t_0002"]
    assert result["ranked"][0]["familiarity"] == "unknown"
    assert result["ranked"][1]["familiarity"] == "usable"


def test_sensitive_cards_are_not_recalled_by_default(tmp_path):
    store = tmp_path / "reader"
    config = load_config()
    now = datetime(2026, 11, 20, tzinfo=timezone.utc)
    card = new_term(
        "t_0001",
        "私密术语",
        "jvm",
        sensitivity="sensitive",
        now=now,
        config=config,
    )
    record_evidence(card, "labeled_known", time=now.isoformat(), config=config)
    save_terms(store, [card])

    hidden = reader_slice(store, "私密术语", now=now, config=config)
    visible = reader_slice(
        store,
        "私密术语",
        include_sensitive=True,
        now=now,
        config=config,
    )

    assert all(item["term"] != "私密术语" for item in hidden["known"])
    assert [item["term"] for item in visible["known"]] == ["私密术语"]


def test_domain_snapshot_groups_levels_and_limits(tmp_path):
    store = tmp_path / "reader"
    config = _seed(store)
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)

    jvm = domain_snapshot(store, "jvm", now=now, config=config)
    k8s = domain_snapshot(store, "k8s", max_terms=1, now=now, config=config)

    assert [item["term"] for item in jvm["known"]] == ["safepoint"]
    assert [item["term"] for item in jvm["unknown"]] == ["gc"]
    assert [item["term"] for item in k8s["uncertain"]] == ["CRI"]
    assert len(k8s["unknown"]) == 1
    assert k8s["counts"]["terms"] == 2
