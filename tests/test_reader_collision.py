"""跨领域 collision 测试：domain 是软先验，不是硬过滤。"""

from __future__ import annotations

from datetime import datetime, timezone

from haoliang.reader import reader_slice
from haoliang.reader_config import load_config
from haoliang.term_store import find_by_name, load_terms, save_terms
from haoliang.terms import new_term


def _seed(store) -> tuple[object, datetime]:
    config = load_config()
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    cards = [
        new_term(
            "t_0001",
            "k8s-scheduler",
            "k8s",
            aliases=["scheduler", "调度器", "kube-scheduler"],
            now=now,
            config=config,
        ),
        new_term(
            "t_0002",
            "java-scheduler",
            "java",
            aliases=["scheduler", "线程调度"],
            now=now,
            config=config,
        ),
        new_term(
            "t_0003",
            "db-buffer",
            "db",
            aliases=["buffer", "buffer pool", "缓冲池"],
            now=now,
            config=config,
        ),
        new_term(
            "t_0004",
            "os-buffer",
            "os",
            aliases=["buffer", "page cache", "页缓存"],
            now=now,
            config=config,
        ),
        new_term(
            "t_0005",
            "safepoint",
            "jvm",
            aliases=["safepoint", "安全点", "STW"],
            domains=["jvm", "k8s"],
            now=now,
            config=config,
        ),
        new_term(
            "t_0006",
            "jvm-gc",
            "jvm",
            aliases=["GC", "垃圾回收", "garbage collection"],
            now=now,
            config=config,
        ),
    ]
    save_terms(store, cards)
    return config, now


def test_soft_domain_prior_resolves_scheduler(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = reader_slice(
        store,
        "scheduler 怎么选节点",
        domain="k8s",
        domain_mode="soft",
        now=now,
        config=config,
    )

    assert result["ranked"][0]["id"] == "t_0001"
    assert result["ranked"][0]["domain_match"] is True
    assert result["ranked"][0]["matched_by"] == "scheduler"


def test_global_mode_leaves_scheduler_ambiguous(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = reader_slice(
        store,
        "scheduler 怎么选节点",
        domain="k8s",
        domain_mode="global",
        now=now,
        config=config,
    )

    top_ids = {item["id"] for item in result["ranked"][:2]}
    assert top_ids == {"t_0001", "t_0002"}
    assert all(item["domain_match"] is False for item in result["ranked"])


def test_hard_mode_drops_cross_domain_card(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = reader_slice(
        store,
        "微服务里的 safepoint 停顿",
        domain="microservices",
        domain_mode="hard",
        now=now,
        config=config,
    )

    assert result["ranked"] == []
    assert {"jvm", "k8s"} <= {entry["name"] for entry in result["available_domains"]}


def test_soft_mode_keeps_cross_domain_card(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = reader_slice(
        store,
        "微服务里的 safepoint 停顿",
        domain="microservices",
        domain_mode="soft",
        now=now,
        config=config,
    )

    assert result["ranked"][0]["id"] == "t_0005"
    assert result["ranked"][0]["domain_match"] is False


def test_buffer_collision_prefers_db(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = reader_slice(
        store,
        "buffer pool 怎么调",
        domain="db",
        domain_mode="soft",
        now=now,
        config=config,
    )

    assert result["ranked"][0]["id"] == "t_0003"


def test_alias_match_is_recorded(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = reader_slice(
        store,
        "垃圾回收",
        domain="jvm",
        now=now,
        config=config,
    )

    assert result["ranked"][0]["id"] == "t_0006"
    assert result["ranked"][0]["matched_by"] == "垃圾回收"


def test_find_by_name_matches_alias_and_falls_back_across_domain(tmp_path):
    store = tmp_path / "reader"
    _seed(store)

    card = find_by_name(load_terms(store), "GC", domain="k8s")

    assert card is not None
    assert card["id"] == "t_0006"
