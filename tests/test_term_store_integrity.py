"""Term store governance invariants."""

from __future__ import annotations

import pytest

from haoliang.term_store import save_terms, store_integrity_errors
from haoliang.terms import new_term


def _card(term_id: str, term: str, domain: str, aliases: list[str] | None = None):
    return new_term(term_id, term, domain, aliases=aliases)


def test_duplicate_term_domain_is_rejected(tmp_path):
    cards = [
        _card("t_0001", "节点", "db"),
        _card("t_0002", "节点", "db"),
    ]

    assert any("duplicate term/domain" in item for item in store_integrity_errors(cards))
    with pytest.raises(ValueError):
        save_terms(tmp_path / "reader", cards)


def test_alias_collision_in_same_domain_is_rejected(tmp_path):
    cards = [
        _card("t_0001", "容器", "k8s", aliases=["container"]),
        _card("t_0002", "业务容器", "k8s", aliases=["container"]),
    ]

    assert any("alias collision" in item for item in store_integrity_errors(cards))
    with pytest.raises(ValueError):
        save_terms(tmp_path / "reader", cards)


def test_same_alias_in_different_domains_is_allowed(tmp_path):
    cards = [
        _card("t_0001", "节点", "db", aliases=["node"]),
        _card("t_0002", "节点", "k8s", aliases=["node"]),
    ]

    assert store_integrity_errors(cards) == []
    save_terms(tmp_path / "reader", cards)


def test_duplicate_id_is_rejected(tmp_path):
    cards = [
        _card("t_0001", "alpha", "db"),
        _card("t_0001", "beta", "db"),
    ]

    assert any("duplicate id" in item for item in store_integrity_errors(cards))
    with pytest.raises(ValueError):
        save_terms(tmp_path / "reader", cards)


def test_secondary_domain_duplicate_is_rejected(tmp_path):
    cards = [
        new_term(
            "t_0001",
            "节点",
            "k8s",
            domains=["k8s", "db"],
        ),
        _card("t_0002", "节点", "db"),
    ]

    assert any("duplicate term/domain" in item for item in store_integrity_errors(cards))
    with pytest.raises(ValueError):
        save_terms(tmp_path / "reader", cards)
