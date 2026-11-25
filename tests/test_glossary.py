"""Local glossary: project/model terms never enter the reader ledger."""

from __future__ import annotations

from haoliang.cli import main
from haoliang.glossary import (
    find_local_term,
    glossary_path,
    load_local_terms,
    new_local_term,
    save_local_terms,
    upsert_local_term,
    validate_local_term,
)
from haoliang.term_store import load_terms


def _term() -> dict:
    return new_local_term(
        "g_0001",
        "Aether",
        "project",
        "demo",
        definition="内部的失败补偿层",
        reason="项目内部命名",
        confidence=0.9,
    )


def test_new_local_term_is_valid():
    item = _term()

    assert validate_local_term(item) == []
    assert item["kind"] == "local-term"
    assert item["status"] == "active"


def test_upsert_updates_definition_and_keeps_first_seen():
    item = _term()
    terms = [item]
    updated = new_local_term(
        "g_0002",
        "Aether",
        "project",
        "demo",
        definition="更新后的定义",
        now=None,
    )

    upsert_local_term(terms, updated)

    assert len(terms) == 1
    assert terms[0]["definition"] == "更新后的定义"
    assert terms[0]["first_seen"] == item["first_seen"]


def test_upsert_keeps_definition_when_new_value_is_empty():
    item = _term()
    terms = [item]
    updated = new_local_term(
        "g_0002",
        "Aether",
        "project",
        "demo",
        definition="",
        reason="",
    )

    upsert_local_term(terms, updated)

    assert terms[0]["definition"] == "内部的失败补偿层"
    assert terms[0]["reason"] == "项目内部命名"
    assert terms[0]["confidence"] == 0.9


def test_glossary_cli_add_list_show(tmp_path):
    root = tmp_path / "glossary"

    assert (
        main(
            [
                "glossary",
                "add",
                "Aether",
                "--scope",
                "project",
                "--project",
                "demo",
                "--definition",
                "内部的失败补偿层",
                "--root",
                str(root),
            ]
        )
        == 0
    )

    items = load_local_terms(glossary_path("demo", root))
    assert items[0]["term"] == "Aether"
    assert main(["glossary", "list", "--project", "demo", "--root", str(root)]) == 0
    assert main(["glossary", "show", "g_0001", "--project", "demo", "--root", str(root)]) == 0


def test_observe_rejects_local_term(tmp_path):
    store = tmp_path / "reader"
    root = tmp_path / "glossary"
    save_local_terms(glossary_path("demo", root), [_term()])

    code = main(
        [
            "observe",
            "Aether",
            "--domain",
            "project",
            "--kind",
            "asked",
            "--store",
            str(store),
            "--glossary-root",
            str(root),
        ]
    )

    assert code == 1
    assert load_terms(store) == []


def test_observe_rejects_non_domain_scope(tmp_path):
    store = tmp_path / "reader"

    code = main(
        [
            "observe",
            "Aether",
            "--domain",
            "project",
            "--kind",
            "asked",
            "--scope",
            "project",
            "--store",
            str(store),
        ]
    )

    assert code == 1
    assert load_terms(store) == []


def test_find_local_term_by_project(tmp_path):
    root = tmp_path / "glossary"
    save_local_terms(glossary_path("demo", root), [_term()])

    assert find_local_term(load_local_terms(glossary_path("demo", root)), "Aether") is not None
