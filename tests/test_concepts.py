"""Concept cards and derived domain profiles."""

from __future__ import annotations

from datetime import datetime, timezone

from haoliang.cli import main
from haoliang.concepts import (
    concepts_for_term,
    derive_domain_profiles,
    load_concepts,
    new_concept,
    save_concepts,
    validate_concept,
    walk_concept,
)
from haoliang.reader_config import load_config
from haoliang.terms import new_term, record_evidence


def _concept() -> dict:
    return new_concept(
        "c_0001",
        "jvm-gc",
        "jvm",
        "JVM 垃圾回收",
        ["t_0001", "t_0002"],
        summary="GC、Full GC、新生代、老年代之间的关系",
    )


def test_new_concept_is_valid():
    card = _concept()

    assert validate_concept(card) == []
    assert card["kind"] == "concept"
    assert card["status"] == "proposal"


def test_concept_members_must_be_term_ids():
    card = _concept()
    card["members"] = ["GC"]

    errors = validate_concept(card)

    assert any("members[0]" in error for error in errors)


def test_load_save_concepts_roundtrip(tmp_path):
    path = tmp_path / "concepts.jsonl"
    save_concepts(path, [_concept()])

    loaded = load_concepts(path)

    assert loaded[0]["concept"] == "jvm-gc"
    assert loaded[0]["members"] == ["t_0001", "t_0002"]


def test_concepts_for_term():
    concept = _concept()

    assert concepts_for_term([concept], "t_0002") == [concept]
    assert concepts_for_term([concept], "t_9999") == []


def test_walk_concept_follows_prerequisites():
    base = new_concept(
        "c_0002",
        "jvm-base",
        "jvm",
        "JVM 基础",
        ["t_0002"],
    )
    concept = _concept()
    concept["prerequisites"] = ["c_0002"]

    walked = walk_concept([concept, base], "c_0001", max_hops=1)

    assert [item["id"] for item in walked] == ["c_0001", "c_0002"]
    assert [item["depth"] for item in walked] == [0, 1]


def test_concept_cli_add_list_show(tmp_path):
    store = tmp_path / "reader"

    code = main(
        [
            "concept",
            "add",
            "c_0001",
            "--concept",
            "jvm-gc",
            "--domain",
            "jvm",
            "--label",
            "JVM 垃圾回收",
            "--member",
            "t_0001",
            "--store",
            str(store),
        ]
    )

    assert code == 0
    concepts = load_concepts(store / "concepts.jsonl")
    assert concepts[0]["label"] == "JVM 垃圾回收"

    assert main(["concept", "list", "--store", str(store)]) == 0
    assert main(["concept", "show", "c_0001", "--store", str(store)]) == 0


def test_derive_domain_profiles_from_terms():
    config = load_config()
    now = datetime(2026, 11, 19, tzinfo=timezone.utc)
    java = new_term("t_0001", "Spring 事务", "java", now=now, config=config)
    record_evidence(java, "explained", time="2026-11-19", config=config)
    k8s = new_term("t_0002", "Pod", "k8s", now=now, config=config)
    record_evidence(k8s, "asked", time="2026-11-19", config=config)

    profiles = derive_domain_profiles([java, k8s], config=config, now=now)
    by_domain = {item["domain"]: item for item in profiles}

    assert by_domain["java"]["knowledge"] > by_domain["k8s"]["knowledge"]
    assert by_domain["java"]["terms"] == 1
    assert by_domain["java"]["source"] == "derived"
