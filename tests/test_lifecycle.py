"""Term lifecycle: archived and merged cards stop being recalled."""

from __future__ import annotations

from datetime import datetime, timezone

from haoliang.annotate import annotate
from haoliang.concepts import derive_domain_profiles
from haoliang.reader_config import load_config
from haoliang.term_store import TermIndex, save_terms
from haoliang.terms import (
    archive_term,
    merge_terms,
    new_term,
    record_evidence,
    restore_term,
    term_knowledge,
)


def _card(term_id: str, term: str, domain: str = "jvm", aliases=None):
    return new_term(term_id, term, domain, aliases=aliases)


def test_archived_card_is_not_matched_or_flagged(tmp_path):
    store = tmp_path / "reader"
    now = datetime(2026, 11, 19, tzinfo=timezone.utc)
    card = _card("t_0001", "旧名", aliases=["old"])
    record_evidence(card, "labeled_known", time=now.isoformat())
    archive_term(card, reason="不再跟踪")
    save_terms(store, [card])

    index = TermIndex([card])
    assert index.find("旧名") is None
    assert index.is_archived("旧名") is card

    result = annotate(store, "旧名 会在这里出现。", domain="jvm", now=now)
    assert result["undefined"] == []
    assert result["candidates"][0]["status"] == "archived"


def test_merge_resolves_old_term_to_target():
    source = _card("t_0001", "旧名", aliases=["old"])
    target = _card("t_0002", "新名")
    terms = [source, target]

    merge_terms(terms, "t_0001", "t_0002", reason="改名")
    index = TermIndex(terms)

    assert source["status"] == "archived"
    assert source["merged_into"] == "t_0002"
    assert index.find("旧名")["id"] == "t_0002"
    assert index.find("old")["id"] == "t_0002"
    assert index.find("新名")["id"] == "t_0002"


def test_merged_alias_to_archived_target_is_not_recalled():
    source = _card("t_0001", "旧名")
    target = _card("t_0002", "新名")
    terms = [source, target]
    merge_terms(terms, "t_0001", "t_0002", reason="改名")
    archive_term(target, reason="不再跟踪")

    index = TermIndex(terms)

    assert index.find("旧名") is None
    assert index.find("新名") is None
    assert index.is_archived("旧名") is target
    assert index.is_archived("新名") is target


def test_merge_moves_source_evidence_to_target():
    source = _card("t_0001", "旧名")
    target = _card("t_0002", "新名")
    record_evidence(source, "explained", time="2026-11-19")
    terms = [source, target]

    merge_terms(terms, "t_0001", "t_0002", reason="改名")

    assert target["evidence"]
    assert target["last_evidence_at"] == source["last_evidence_at"]
    assert term_knowledge(target) > 0


def test_restore_clears_merge_marker():
    card = _card("t_0001", "旧名")
    archive_term(card, merged_into="t_0002", reason="改名")

    restore_term(card)

    assert card["status"] == "proposal"
    assert "merged_into" not in card
    assert "archived_at" not in card


def test_domain_profile_ignores_archived_cards():
    config = load_config()
    now = datetime(2026, 11, 19, tzinfo=timezone.utc)
    active = _card("t_0001", "active")
    archived = _card("t_0002", "archived")
    record_evidence(archived, "labeled_known", time=now.isoformat())
    archive_term(archived, reason="old")

    profiles = derive_domain_profiles(
        [active, archived],
        config=config,
        now=now,
        domain="jvm",
    )

    assert profiles[0]["terms"] == 1
