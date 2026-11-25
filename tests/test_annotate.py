"""annotate(draft) tests: merge, ledger matching, undefined terms."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from haoliang.annotate import annotate, merge_candidates
from haoliang.cli import main
from haoliang.extractors import TermCandidate
from haoliang.reader_config import load_config
from haoliang.term_store import save_terms
from haoliang.terms import new_term, record_evidence


def _seed(store) -> tuple[object, datetime]:
    config = load_config()
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)

    safepoint = new_term(
        "t_0001",
        "safepoint",
        "jvm",
        aliases=["安全点"],
        now=now,
        config=config,
    )
    record_evidence(safepoint, "explained", time="2026-11-01", config=config)

    jvm = new_term("t_0002", "JVM", "jvm", now=now, config=config)
    record_evidence(jvm, "explained", time="2026-11-01", config=config)

    gc = new_term("t_0003", "GC", "jvm", now=now, config=config)

    spring = new_term(
        "t_0004",
        "Spring 事务",
        "java",
        aliases=["事务传播"],
        now=now,
        config=config,
    )
    record_evidence(spring, "explained", time="2026-11-01", config=config)

    btree = new_term(
        "t_0005",
        "B+ 树",
        "db",
        aliases=["B+Tree"],
        now=now,
        config=config,
    )

    save_terms(store, [safepoint, jvm, gc, spring, btree])
    return config, now


def test_merge_candidates_joins_hyphenated_tokens():
    text = "otel-collector 负责采集"
    candidates = [
        TermCandidate("otel", 0, 4, "unknown", "test"),
        TermCandidate("collector", 5, 14, "unknown", "test"),
    ]

    merged = merge_candidates(text, candidates)

    assert len(merged) == 1
    assert merged[0].text == "otel-collector"
    assert merged[0].start == 0
    assert merged[0].end == 14


def test_annotate_marks_known_unknown_and_not_in_ledger(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)
    draft = "safepoint 是 JVM 的安全点；GC 会在这里停顿；otel-collector 负责采集。"

    result = annotate(store, draft, now=now, config=config)

    by_text = {item["text"]: item for item in result["candidates"]}
    assert by_text["safepoint"]["status"] == "known"
    assert by_text["safepoint"]["defined"] is True
    assert by_text["JVM"]["status"] == "known"
    assert by_text["GC"]["status"] == "unknown"
    assert by_text["otel-collector"]["status"] == "not_in_ledger"

    assert "GC" in result["undefined"]
    assert "otel-collector" in result["undefined"]
    assert "safepoint" not in result["undefined"]


def test_annotate_merges_connector_tokens(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "REQUIRES_NEW 会挂起外层事务。",
        now=now,
        config=config,
    )
    texts = {item["text"] for item in result["candidates"]}

    assert "REQUIRES_NEW" in texts
    assert "_" not in texts


def test_annotate_merges_english_phrases(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "tree shaking 依赖 ES module 的静态结构。",
        now=now,
        config=config,
    )
    texts = {item["text"] for item in result["candidates"]}

    assert "tree shaking" in texts
    assert "ES module" in texts
    assert "tree" not in texts


def test_annotate_prefers_domain_specific_card(tmp_path):
    store = tmp_path / "reader"
    config = load_config()
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)

    db_node = new_term("t_0101", "节点", "db", now=now, config=config)
    record_evidence(db_node, "labeled_known", time="2026-11-01", config=config)
    k8s_node = new_term("t_0102", "节点", "k8s", now=now, config=config)
    save_terms(store, [db_node, k8s_node])

    db_result = annotate(store, "节点通过指针连接。", domain="db", now=now, config=config)
    k8s_result = annotate(
        store,
        "节点通过指针连接。",
        domain="k8s",
        now=now,
        config=config,
    )

    assert "节点" not in db_result["undefined"]
    assert "节点" in k8s_result["undefined"]


def test_annotate_parenthetical_alias_is_defined(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "STW（Stop-The-World）：JVM 暂停所有应用线程。"
        "IHOP（InitiatingHeapOccupancyPercent，初始堆占用阈值）时触发。",
        now=now,
        config=config,
    )
    by_text = {item["text"]: item for item in result["candidates"]}

    assert by_text["Stop-The-World"]["defined"] is True
    assert by_text["InitiatingHeapOccupancyPercent"]["defined"] is True
    assert "Stop-The-World" not in result["undefined"]
    assert "InitiatingHeapOccupancyPercent" not in result["undefined"]


def test_annotate_filters_cjk_boundary_fragments():
    from haoliang.annotate import _is_fragment

    for text in ("否走索引", "器会", "列拿", "回表指", "回表太"):
        assert _is_fragment(text) is True


def test_annotate_does_not_match_ascii_substring(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(store, "GCC 是编译器。", now=now, config=config)
    by_text = {item["text"]: item for item in result["candidates"]}

    assert by_text["GC"]["status"] == "fragment"
    assert "GC" not in result["undefined"]


def test_annotate_does_not_treat_question_is_as_definition(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(store, "GC 是否会停顿？", now=now, config=config)
    by_text = {item["text"]: item for item in result["candidates"]}

    assert by_text["GC"]["defined"] is False
    assert "GC" in result["undefined"]


def test_annotate_does_not_merge_two_ledger_terms(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(store, "JVM GC 停顿很长。", now=now, config=config)
    texts = {item["text"] for item in result["candidates"]}

    assert "JVM" in texts
    assert "GC" in texts
    assert "JVM GC" not in texts


def test_annotate_parenthetical_aside_is_not_a_definition(tmp_path):
    from haoliang.annotate import _tail_is_definition

    assert _tail_is_definition("（Eden 和 Survivor 的拷贝）很长") is False

    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "GC 停顿（Eden 和 Survivor 的拷贝）。",
        now=now,
        config=config,
    )
    by_text = {item["text"]: item for item in result["candidates"]}

    assert by_text["Eden"]["defined"] is False
    assert by_text["Survivor"]["defined"] is False


def test_annotate_chinese_alias_followed_by_definition(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "混合回收（Mixed GC）就是一次 Young GC 加上一部分 Old Region 的回收。",
        now=now,
        config=config,
    )
    by_text = {item["text"]: item for item in result["candidates"]}

    assert by_text["Mixed GC"]["defined"] is True
    assert "Mixed GC" not in result["undefined"]


def test_annotate_detects_compound_terms(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "联合索引的最左前缀原则，范围查询后面的列用不上索引。",
        domain="db",
        now=now,
        config=config,
    )
    texts = {item["text"] for item in result["candidates"]}

    assert "联合索引" in texts
    assert "最左前缀" in texts
    assert "范围查询" in texts


def test_compound_candidates_skip_function_words():
    from haoliang.annotate import compound_candidates

    assert compound_candidates("这个索引和那个索引") == []


def test_annotate_chinese_alias_after_latin_head(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "initial-mark（初始标记，搭一次 young GC 的便车）",
        now=now,
        config=config,
    )
    by_text = {item["text"]: item for item in result["candidates"]}

    assert by_text["初始标记"]["defined"] is True
    assert "初始标记" not in result["undefined"]


def test_annotate_ignores_markdown_code_span(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "`where a` 和 `and b` 都是 SQL 片段。",
        domain="db",
        now=now,
        config=config,
    )
    by_text = {item["text"]: item for item in result["candidates"]}

    assert by_text["where a"]["status"] == "fragment"
    assert by_text["and b"]["status"] == "fragment"
    assert "where a" not in result["undefined"]
    assert "and b" not in result["undefined"]


def test_annotate_filters_camel_case_fragments(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "G1MixedGCCountTarget 是参数。",
        now=now,
        config=config,
    )
    by_text = {item["text"]: item for item in result["candidates"]}

    assert by_text["G1Mixed"]["status"] == "fragment"
    assert by_text["CountTarget"]["status"] == "fragment"
    assert "G1Mixed" not in result["undefined"]
    assert "CountTarget" not in result["undefined"]


def test_annotate_does_not_merge_spaced_slash_list():
    from haoliang.extractors import TermCandidate

    text = "Eden / Survivor"
    candidates = [
        TermCandidate("Eden", 0, 4, "noun", "jieba"),
        TermCandidate("Survivor", 7, 15, "noun", "jieba"),
    ]

    merged = merge_candidates(text, candidates)

    assert [item.text for item in merged] == ["Eden", "Survivor"]


def test_annotate_term_defined_in_glossary_section(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "术语表：\nRegion：堆内存划分出的区域。\n正文：Region 会被回收。",
        now=now,
        config=config,
    )
    by_text = {item["text"]: item for item in result["candidates"]}

    assert by_text["Region"]["defined"] is True
    assert "Region" not in result["undefined"]


def test_annotate_role_phrase_is_not_a_definition(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "GC 作为 safepoint 的触发者，会让线程停顿。",
        now=now,
        config=config,
    )
    by_text = {item["text"]: item for item in result["candidates"]}

    assert by_text["GC"]["defined"] is False


def test_annotate_treats_appositive_as_defined(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "B+ 树这类索引结构适合范围查询。",
        now=now,
        config=config,
    )
    by_text = {item["text"]: item for item in result["candidates"]}

    assert by_text["B+ 树"]["defined"] is True
    assert "B+ 树" not in result["undefined"]


def test_annotate_prioritizes_ledger_unknown_over_not_in_ledger(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "容器 里跑着服务，GC 会在这里停顿。",
        now=now,
        config=config,
    )

    assert result["undefined"][0] == "GC"


def test_annotate_matches_multiword_ledger_term(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "Spring 事务的传播行为默认是 REQUIRED。",
        now=now,
        config=config,
    )
    by_text = {item["text"]: item for item in result["candidates"]}

    assert "Spring 事务" in by_text
    assert by_text["Spring 事务"]["status"] == "known"
    assert "Spring" not in by_text


def test_annotate_matches_term_tokenizer_splits(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "B+ 树的叶子节点通过双向链表连接。",
        now=now,
        config=config,
    )
    by_text = {item["text"]: item for item in result["candidates"]}

    assert "B+ 树" in by_text
    assert "B" not in by_text
    assert by_text["B+ 树"]["status"] == "unknown"


def test_annotate_filters_single_cjk_fragment(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(store, "观察 STW 时长。", now=now, config=config)
    texts = {item["text"] for item in result["candidates"]}

    assert "STW" in texts
    assert "时" not in texts


def test_annotate_cli_json(tmp_path, capsys):
    store = tmp_path / "reader"
    _seed(store)
    capsys.readouterr()

    code = main(
        [
            "annotate",
            "safepoint 是 JVM 的安全点；GC 会停顿。",
            "--store",
            str(store),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "GC" in payload["undefined"]


def test_find_leaks_flags_harness_words():
    from haoliang.annotate import find_leaks

    leaks = find_leaks(
        "账本里 Spring AOP 只是 heard 级证据，所以下面补一句定义。\n"
        "不属于正文，写到这里。"
    )

    assert leaks
    assert leaks[0]["line"] == 1
    assert "账本" in leaks[0]["hits"]
    assert "heard" in leaks[0]["hits"]
    assert "级证据" in leaks[0]["hits"]
    assert len(leaks) == 2


def test_find_leaks_ignores_clean_text():
    from haoliang.annotate import find_leaks

    assert find_leaks("代理在初始化末尾才生成，所以 @PostConstruct 里是原始对象。") == []


def test_annotate_reports_leaks(tmp_path):
    store = tmp_path / "reader"
    config, now = _seed(store)

    result = annotate(
        store,
        "账本里这个词只是 heard，所以补一句。",
        now=now,
        config=config,
    )

    assert result["leaks"]
    assert result["leaks"][0]["hits"]


def test_find_leaks_flags_conversational_meta():
    from haoliang.annotate import find_leaks

    leaks = find_leaks(
        "触发初始化的六个主动引用场景你背过，不展开。\n"
        "答这道题时提一句能显出边界感。"
    )

    assert len(leaks) == 2
    assert "你背过" in leaks[0]["hits"]
    assert "答这道题时" in leaks[1]["hits"]


def test_find_leaks_flags_placeholder_first_line():
    from haoliang.annotate import find_leaks

    assert find_leaks("正文\n代理在初始化末尾才生成。")[0]["hits"] == ["占位标签"]
    assert find_leaks("A.\n\n代理在初始化末尾才生成。")[0]["hits"] == ["占位标签"]
    assert find_leaks("代理在初始化末尾才生成。") == []
