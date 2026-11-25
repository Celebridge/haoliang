"""TermIndex: alias lookup and one-pass ledger matching."""

from __future__ import annotations

from datetime import datetime, timezone

from haoliang.annotate import annotate
from haoliang.term_store import TermIndex, load_terms, save_terms
from haoliang.terms import new_term, record_evidence


def _card(term_id: str, term: str, domain: str, aliases: list[str] | None = None):
    return new_term(term_id, term, domain, aliases=aliases)


def test_index_finds_alias_and_prefers_domain():
    cards = [
        _card("t_0001", "safepoint", "jvm", aliases=["安全点", "STW"]),
        _card("t_0002", "节点", "db", aliases=["node"]),
        _card("t_0003", "节点", "k8s", aliases=["node"]),
    ]
    index = TermIndex(cards)

    assert index.find("安全点", "jvm")["id"] == "t_0001"
    assert index.find("节点", "db")["id"] == "t_0002"
    assert index.find("节点", "k8s")["id"] == "t_0003"


def test_iter_matches_finds_overlapping_names():
    cards = [
        _card("t_0001", "GC", "jvm"),
        _card("t_0002", "Full GC", "jvm"),
    ]
    index = TermIndex(cards)

    keys = {key for _, _, key in index.iter_matches("Full GC 和 GC")}

    assert {"gc", "full gc"} <= keys


def test_annotate_accepts_reused_index(tmp_path):
    store = tmp_path / "reader"
    now = datetime(2026, 11, 19, tzinfo=timezone.utc)
    safepoint = _card("t_0001", "safepoint", "jvm", aliases=["安全点"])
    record_evidence(safepoint, "labeled_known", time=now.isoformat())
    save_terms(
        store,
        [
            safepoint,
            _card("t_0002", "GC", "jvm"),
        ],
    )
    index = TermIndex(load_terms(store))

    result = annotate(
        store,
        "GC 会在 safepoint 停顿。",
        domain="jvm",
        index=index,
        now=now,
    )

    assert "GC" in result["undefined"]
    assert "safepoint" not in result["undefined"]


def test_find_falls_back_to_another_domain_when_requested_domain_is_missing():
    card = _card("t_0001", "线程", "java")
    index = TermIndex([card])

    assert index.find("线程", "jvm") is card
