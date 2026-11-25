"""卡片异构图：语义边、实体边、标签边、线程边、显式关系边。

为什么需要图：
- 纯向量检索只能找到「字面/语义相近」的卡；
- 但人的记忆是连着的：同一件事、同一个人、同一条时间线、同一个主题。
- 图把这些关系显式建出来，检索时从种子卡扩散出去，能找到「语义不远但很相关」的卡。

边类型和默认权重（当前版本，graph.py 是唯一真相）：

| type | weight | 含义 |
|---|---:|---|
| next | 1.00 | 阶段之间的顺序，结构边 |
| causes | 0.95 | LLM 抽的因果，带 why |
| explains | 0.90 | LLM 抽的解释，带 why |
| contradicts | 0.90 | 矛盾，带 why |
| before | 0.80 | 章内时间推进 |
| entity | 0.55 | 共享实体（无 why，低于游走阈值） |
| related | 0.62 | 人写的未指定关系 |
| part_of | 0.55 | 卡属于哪个阶段（归属，不是证据） |
| thread | 0.40 | 同 thread 相邻 |
| tag | 0.30 | 共享标签 |
| semantic | 原始 cosine | 默认不进图，只做候选 |

一句话：只有带 `why` 的边，才配拿到高于游走阈值的权重。
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

EDGE_TYPE_WEIGHTS: dict[str, float] = {
    "next": 1.0,
    "causes": 0.95,
    "explains": 0.9,
    "contradicts": 0.9,
    "before": 0.8,
    "entity": 0.55,
    "related": 0.62,
    "part_of": 0.55,
    "thread": 0.4,
    "tag": 0.3,
}


def normalize_edge_weight(kind: str, weight: float | None = None) -> float:
    """把边权压回它所属类型的区间。

    LLM 边自带 confidence（0.6–1.0），用 min(confidence, 类型上限) 保留强弱；
    规则/人工边没有 confidence，直接用类型默认值。
    """
    base = EDGE_TYPE_WEIGHTS.get(kind)
    if base is None:
        return round(float(weight if weight is not None else 1.0), 4)
    if weight is None:
        return base
    return round(min(float(weight), base), 4)


def apply_type_weights(edges: list[dict]) -> list[dict]:
    """给已有的边重新套类型权重（不用重算 embedding）。"""
    out = []
    for edge in edges:
        item = dict(edge)
        item["weight"] = normalize_edge_weight(item["type"], item.get("weight"))
        out.append(item)
    return out


def _edge_key(a: str, b: str, kind: str) -> tuple[str, str, str]:
    """生成无向边的唯一 key：小的 id 放前面，避免 A→B 和 B→A 重复。"""
    return (a, b, kind) if a <= b else (b, a, kind)


def semantic_candidates(
    cards: list[dict],
    embeddings: np.ndarray,
    top_k: int = 20,
    min_sim: float = 0.0,
) -> dict[str, list[tuple[str, float]]]:
    """向量 kNN 候选，用于「生成候选」而不是「当图边」。

    语义相似应该直接由向量检索承担；把它写成边只会让图退化成
    第二个向量索引，还会让游走沿着语义漂移。
    """
    ids = [c["id"] for c in cards]
    if not embeddings.size or len(embeddings) != len(cards):
        return {cid: [] for cid in ids}

    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -1.0)

    out: dict[str, list[tuple[str, float]]] = {}
    for i, card in enumerate(cards):
        picked: list[tuple[str, float]] = []
        for j in np.argsort(sim[i])[::-1][:top_k]:
            score = float(sim[i, j])
            if score >= min_sim:
                picked.append((ids[int(j)], score))
        out[card["id"]] = picked
    return out


def build_edges(
    cards: list[dict],
    embeddings: np.ndarray,
    top_k: int = 8,
    min_sim: float = 0.15,
    include_semantic: bool = True,
) -> list[dict]:
    """从卡片和向量构建所有边，返回去重后的边列表。

    `include_semantic=False` 时不把向量 kNN 写成边——语义相似交给检索层，
    图只保留 embedding 表达不了的关系。
    """
    ids = [c["id"] for c in cards]
    idx = {cid: i for i, cid in enumerate(ids)}  # id → 矩阵行号
    edges: dict[tuple[str, str, str], dict] = {}

    def add(
        a: str,
        b: str,
        kind: str,
        weight: float,
        *,
        source: str = "rule",
        directed: bool = False,
        why: str = "",
    ) -> None:
        """加一条边；同一对节点同一种边只保留权重最高的。"""
        if a == b or a not in idx or b not in idx:
            return
        key = _edge_key(a, b, kind)
        weight = normalize_edge_weight(kind, weight)
        if key in edges and weight <= edges[key]["weight"]:
            return
        payload = {
            "src": key[0],
            "dst": key[1],
            "type": kind,
            "weight": weight,
            "source": source,
        }
        if directed:
            payload["directed"] = True
            # 有向边不能交换端点，否则方向就丢了
            payload["src"], payload["dst"] = a, b
        if why:
            payload["why"] = why
        edges[key] = payload

    # 1) 语义 kNN：默认关掉。语义相似由 query 的向量检索直接负责，
    #    写进图里只会让游走沿着“像不像”漂移。
    if include_semantic:
        for src, candidates in semantic_candidates(
            cards, embeddings, top_k=top_k, min_sim=min_sim
        ).items():
            for dst, score in candidates:
                add(src, dst, "semantic", score, source="embedding")

    # 2) 线程边：同一 thread 内按 id 排序，相邻卡片连起来。
    by_thread: dict[str, list[dict]] = defaultdict(list)
    for card in cards:
        by_thread[card["thread"]].append(card)

    for thread_cards in by_thread.values():
        thread_cards.sort(key=lambda c: c["id"])
        # itertools.pairwise 产生 (a,b), (b,c), ...，比 zip(xs, xs[1:]) 更清晰
        for a, b in itertools.pairwise(thread_cards):
            add(a["id"], b["id"], "thread", 0.5)

    # 3) 实体边：共享同一个实体（人、公司、地点等）的卡片两两相连。
    by_entity: dict[str, list[str]] = defaultdict(list)
    for card in cards:
        for entity in card.get("entities", []):
            by_entity[entity].append(card["id"])

    for members in by_entity.values():
        # 组合而不是排列：每对只加一次
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                add(a, b, "entity", 0.8)

    # 4) 标签边：共享非通用标签的卡片两两相连。
    # 跳过 section:/type:/thread:，否则同一章节的所有卡会连成一团。
    by_tag: dict[str, list[str]] = defaultdict(list)
    for card in cards:
        for tag in card.get("tags", []):
            if tag.startswith(("section:", "type:", "thread:")):
                continue
            by_tag[tag].append(card["id"])

    for members in by_tag.values():
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                add(a, b, "tag", 0.35)

    # 5) 显式关系：卡片 schema 里人工写的 related / contradicts / part_of / next。
    #    part_of 现在指向阶段卡（S1..），不再是 section 名。
    #    next 是时间轴上的“下一段”，必须有方向。
    for card in cards:
        links = card.get("links") or {}
        for rel in ("related", "contradicts"):
            for target in links.get(rel, []) or []:
                add(card["id"], str(target), rel, 1.0, source="human")
        for target in links.get("part_of", []) or []:
            add(card["id"], str(target), "part_of", 1.0, source="human", directed=True)
        for target in links.get("next", []) or []:
            add(card["id"], str(target), "next", 1.0, source="human", directed=True)

    # 排序后输出，保证每次 build 的 edges.jsonl 稳定可 diff
    return sorted(edges.values(), key=lambda e: (e["src"], e["dst"], e["type"]))


def write_edges(edges: list[dict], out_dir: str | Path) -> None:
    """把边写成 JSONL。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "edges.jsonl").open("w", encoding="utf-8") as f:
        for edge in edges:
            f.write(json.dumps(edge, ensure_ascii=False) + "\n")


def load_edges(store_dir: str | Path) -> list[dict]:
    """读取 edges.jsonl；文件不存在时返回空列表。"""
    path = Path(store_dir) / "edges.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def adjacency(edges: list[dict], ids: list[str]) -> np.ndarray:
    """把边列表转成 n×n 的邻接矩阵（无向图，权重取最大值）。

    矩阵元素 mat[i][j] 就是节点 i 到 j 的边权；没有边就是 0。
    """
    idx = {cid: i for i, cid in enumerate(ids)}
    n = len(ids)
    mat = np.zeros((n, n), dtype=np.float32)

    for edge in edges:
        a, b = edge.get("src"), edge.get("dst")
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        w = float(edge.get("weight", 0.0))
        # 同一对节点可能有多条不同类型的边，取最强的那条
        mat[i, j] = max(mat[i, j], w)
        mat[j, i] = max(mat[j, i], w)

    return mat


def personalized_pagerank(
    edges: list[dict],
    ids: list[str],
    seeds: dict[str, float],
    alpha: float = 0.85,
    iterations: int = 40,
) -> np.ndarray:
    """在无向加权图上跑 Personalized PageRank（PPR）。

    公式：
        r = alpha * P.T @ r + (1 - alpha) * s

    其中：
    - s 是种子分布（只在 seed 卡上有权重，归一化到和为 1）
    - P 是行归一化的转移矩阵：P[i][j] = edge(i,j) / sum_j edge(i,j)
    - alpha 是「继续沿着图走」的概率，1-alpha 是「跳回种子」的概率
    - 迭代 40 次通常已经收敛，对 492 个节点足够

    直观理解：从种子卡出发随机游走，经常被走到的卡，就是和种子关系近的卡。
    """
    n = len(ids)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    mat = adjacency(edges, ids)

    # 每一行除以该行的权重和，得到转移概率。
    # 孤立节点（行和为 0）用 1.0 兜底，避免 0/0。
    row = mat.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    transition = mat / row

    # 构造种子向量 s
    s = np.zeros(n, dtype=np.float32)
    idx = {cid: i for i, cid in enumerate(ids)}
    for cid, weight in seeds.items():
        if cid in idx:
            s[idx[cid]] = float(weight)

    total = float(s.sum())
    if total == 0:
        return np.zeros(n, dtype=np.float32)
    s /= total

    # 幂迭代
    r = s.copy()
    for _ in range(iterations):
        r = alpha * (transition.T @ r) + (1.0 - alpha) * s

    # 迭代后归一化，方便上层当分数用
    total = float(r.sum())
    return r / total if total else r


def shortest_path(
    edges: list[dict],
    starts: list[str],
    target: str,
    max_depth: int = 4,
) -> list[str] | None:
    """从任意种子节点到目标节点的最短路径（BFS，无向）。

    用于 `--explain`：告诉用户「这张卡是怎么被找到的」。
    例如：R19 → A18 → I22，说明检索从「怕被说不靠谱」走到了相关经历。
    """
    if target in starts:
        return [target]

    # 先建邻接表（节点 → 邻居列表），BFS 比邻接矩阵更省事
    adj: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        a, b = edge["src"], edge["dst"]
        adj[a].append(b)
        adj[b].append(a)

    # 队列里同时保存「当前节点 + 从起点到这里的路径」
    q: deque[tuple[str, list[str]]] = deque((s, [s]) for s in starts)
    seen = set(starts)

    while q:
        node, path = q.popleft()
        if len(path) > max_depth + 1:
            continue
        for nxt in adj[node]:
            if nxt == target:
                return path + [nxt]
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, path + [nxt]))

    return None
