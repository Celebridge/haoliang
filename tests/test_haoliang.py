"""Haoliang 的核心行为测试。

测试策略：
- 用 tests/fixtures 里的迷你问题库（8 题）跑完整流程；
- 不依赖网络/真实模型，embedding 用 HashEmbedder；
- 每个测试独立用 tmp_path，避免互相污染。

这些测试覆盖的是「流程是否正确」，不是「检索效果好不好」。
检索效果需要用真实 embedding + eval 查询集单独评估。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from haoliang.cards import build_cards
from haoliang.embed import HashEmbedder, embed_cards
from haoliang.graph import build_edges, write_edges
from haoliang.index import build_index
from haoliang.query import query_store, stats

FIXTURES = Path(__file__).parent / "fixtures"


def build_store(tmp_path: Path) -> Path:
    """在临时目录里构建一个完整的迷你 store。

    等价于 CLI 的 `haoliang build`，只是把参数缩小到测试规模。
    """
    store = tmp_path / "store"

    # 1) 解析问题 + 答案，生成卡片
    build_cards(FIXTURES / "source.md", FIXTURES / "answers.json", store)

    # 2) 读出卡片，算向量
    cards = [
        json.loads(line)
        for line in (store / "cards.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    embed_cards(cards, HashEmbedder(dim=128), store)

    # 3) 用向量构图，写到 edges.jsonl
    embeddings = np.load(store / "embeddings.npy")
    edges = build_edges(cards, embeddings, top_k=3, min_sim=0.0)
    write_edges(edges, store)

    # 4) 建 SQLite 索引
    build_index(store)

    return store


def test_build_cards(tmp_path: Path):
    """卡片字段、类型映射、敏感标记、跳过状态、manifest 统计。"""
    store = build_store(tmp_path)
    cards = [
        json.loads(line)
        for line in (store / "cards.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert len(cards) == 8

    by_id = {c["id"]: c for c in cards}
    assert by_id["A1"]["type"] == "fact"
    assert by_id["B2"]["type"] == "event"
    assert by_id["L1"]["sensitivity"] == "sensitive"
    assert by_id["W1"]["status"] == "skipped"
    assert by_id["B1"]["claims"][0]["text"].startswith("第一份工作")

    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["total"] == 8
    assert manifest["counts"]["skipped"] == 1


def test_graph_has_multiple_edge_types(tmp_path: Path):
    """图里至少要有语义边和线程边。"""
    store = build_store(tmp_path)
    edges = [
        json.loads(line)
        for line in (store / "edges.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    kinds = {e["type"] for e in edges}

    assert "semantic" in kinds
    assert "thread" in kinds


def test_graph_can_drop_semantic_and_keep_direction():
    """语义边可以从图里去掉；next 这种时间边必须有方向。"""
    cards = [
        {
            "id": "A",
            "thread": "t",
            "tags": [],
            "entities": [],
            "links": {"next": ["B"]},
        },
        {
            "id": "B",
            "thread": "t",
            "tags": [],
            "entities": [],
            "links": {},
        },
    ]
    embeddings = np.zeros((2, 4), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 0] = 1.0

    edges = build_edges(cards, embeddings, include_semantic=False)
    kinds = {e["type"] for e in edges}

    assert "semantic" not in kinds
    next_edge = next(e for e in edges if e["type"] == "next")
    assert next_edge["directed"] is True
    assert (next_edge["src"], next_edge["dst"]) == ("A", "B")


def test_edge_type_weights_normalize():
    """类型权重是唯一真相：人工边压回类型上限，LLM 边保留 confidence。"""
    from haoliang.graph import normalize_edge_weight

    assert normalize_edge_weight("related", 1.0) == 0.62
    assert normalize_edge_weight("part_of", 1.0) == 0.55
    assert normalize_edge_weight("explains", 0.7) == 0.7
    assert normalize_edge_weight("explains", 1.0) == 0.9
    assert normalize_edge_weight("next", 1.0) == 1.0
    # 没有登记的类型（如 semantic）保留原始权重
    assert normalize_edge_weight("semantic", 0.83) == 0.83


def test_query_finds_parent_card(tmp_path: Path):
    """问「父亲」时，C1 应该出现在结果里，并且 explain 能给出路径。"""
    store = build_store(tmp_path)
    result = query_store(store, "父亲是什么样的人", top_k=5, explain=True)
    ids = [c["id"] for c in result["cards"]]

    assert "C1" in ids
    assert any("path" in c for c in result["cards"])


def test_sensitive_and_skipped_filters(tmp_path: Path):
    """敏感卡和跳过卡默认不返回，显式开关才返回。"""
    store = build_store(tmp_path)

    public = query_store(store, "依恋风格", top_k=8)
    assert "L1" not in [c["id"] for c in public["cards"]]

    sensitive = query_store(store, "依恋风格", top_k=8, include_sensitive=True)
    assert "L1" in [c["id"] for c in sensitive["cards"]]

    skipped = query_store(store, "人生七章", top_k=8, include_skipped=True)
    assert "W1" in [c["id"] for c in skipped["cards"]]


def test_stats(tmp_path: Path):
    """统计信息应该和构建结果一致。"""
    store = build_store(tmp_path)
    info = stats(store)

    assert info["cards"] == 8
    assert info["edges"] > 0


def test_persona_compile(tmp_path: Path):
    """合成人设编译：只输出有答案的题。"""
    from haoliang.persona import compile_persona

    answers = tmp_path / "answers.json"
    answers.write_text(
        json.dumps(
            {"name": "样例读者", "city_work": "示例市，服务端开发"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    preset = tmp_path / "preset.json"
    preset.write_text(
        json.dumps(
            {
                "title": "合成示例",
                "questions": [
                    {"id": "name", "text": "如何称呼？", "kind": "cognition"},
                    {"id": "city_work", "text": "在哪工作？", "kind": "cognition"},
                    {"id": "unanswered", "text": "未回答的问题？", "kind": "cognition"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    text = compile_persona(preset, answers)

    assert "# 关于我" in text
    assert "样例读者" in text
    assert "示例市，服务端开发" in text
    assert "未回答的问题" not in text


def test_clean_summary_respects_limit():
    from haoliang.summarize import _clean_summary

    text = _clean_summary("桥" * 200, 80)
    assert len(text) <= 80
    assert text.endswith("…")


def test_logical_adjacency_skips_semantic_and_weak_edges():
    """deep 游走只走非语义边，并且要过滤低权重边。"""
    from haoliang.query import _logical_adjacency

    edges = [
        {"src": "A", "dst": "B", "type": "related", "weight": 1.0, "why": "因果"},
        {"src": "A", "dst": "C", "type": "semantic", "weight": 0.9},
        {"src": "A", "dst": "D", "type": "thread", "weight": 0.5},
        {"src": "B", "dst": "E", "type": "entity", "weight": 0.8},
    ]

    adj = _logical_adjacency(edges, ["A", "B", "C", "D", "E"], min_weight=0.6)

    assert [n for n, _, _, _ in adj["A"]] == ["B"]
    assert ("E", "entity", 0.8, "") in adj["B"]
    # 语义边和低权重边都不该出现
    assert all(kind != "semantic" for _, kind, _, _ in adj["A"])
    assert "D" not in [n for n, _, _, _ in adj["A"]]
