"""检索：向量种子 → PPR 图扩散 → 重排 → top-k。

整体流程：
1. 加载卡片、向量、边；
2. 默认过滤 skipped / sensitive；
3. 用向量相似度找 top-k 种子卡；
4. 从种子卡跑 PPR，得到全图的相关性分数；
5. 综合「语义相似度 + 图分数 + 重要性」重排；
6. 可选：为每张结果找一条从种子到它的路径，方便解释。
"""

from __future__ import annotations

import heapq
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .embed import cosine, load_embeddings, make_embedder
from .graph import load_edges, personalized_pagerank, shortest_path


def load_cards(store_dir: str | Path) -> list[dict]:
    """从 store 目录读取 cards.jsonl。"""
    path = Path(store_dir) / "cards.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `haoliang build` first")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


_EMBEDDER_CACHE: dict[tuple, object] = {}


def _embedder_from_meta(meta: dict):
    """用 build 时记录的 provider/model 重建 embedder。

    query 必须和卡片用同一个 embedding 模型，否则向量不在同一个空间。
    base_url / dimensions / api_key_env 也要从 meta 里恢复；key 本身不落盘。

    同一个进程里缓存 embedder：BGE 模型只加载一次，OpenAI client 和
    单条 query 缓存也能复用。
    """
    provider = meta.get("provider", "hash")
    model = meta.get("model")
    dim = int(meta.get("dim", 256))
    key = (
        provider,
        model,
        dim,
        meta.get("base_url"),
        meta.get("api_key_env", "OPENAI_API_KEY"),
        meta.get("dimensions"),
        int(meta.get("batch_size", 10)),
        float(meta.get("request_interval", 0.0)),
    )
    if key not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[key] = make_embedder(
            provider,
            model=model,
            dim=dim,
            base_url=meta.get("base_url"),
            api_key_env=meta.get("api_key_env", "OPENAI_API_KEY"),
            dimensions=meta.get("dimensions"),
            batch_size=int(meta.get("batch_size", 10)),
            request_interval=float(meta.get("request_interval", 0.0)),
        )
    return _EMBEDDER_CACHE[key]


def query_store(
    store_dir: str | Path,
    query: str,
    top_k: int = 10,
    depth: int = 2,
    alpha: float = 0.85,
    include_sensitive: bool = False,
    include_skipped: bool = False,
    explain: bool = False,
    use_graph: bool = True,
    mode: str | None = None,
    max_hops: int = 2,
    min_edge_weight: float = 0.6,
    seed_pool: int = 8,
) -> dict:
    """执行一次检索，返回结构化结果。

    参数：
    - store_dir：store 目录（cards.jsonl / embeddings.npy / edges.jsonl）
    - query：用户问题
    - top_k：返回多少张卡，也决定种子数量（至少 5 个）
    - depth：--explain 时路径搜索的最大深度
    - alpha：PPR 的继续游走概率，越大越偏向图扩散
    - include_sensitive：是否允许敏感卡进入结果
    - include_skipped：是否允许跳过卡进入结果
    - explain：是否为结果附加最短路径
    - use_graph：是否使用 PPR 图扩散；False 时退化为纯向量排序（评测对照用）
    - mode：vector / graph / deep；显式传入时覆盖 use_graph
    - max_hops：deep 模式沿非语义边最多走几跳
    - min_edge_weight：deep 模式允许通过的最低边权
    - seed_pool：deep 模式从向量前几名里挑入口卡
    """
    if mode == "deep":
        return query_deep(
            store_dir,
            query,
            top_k=top_k,
            max_hops=max_hops,
            min_edge_weight=min_edge_weight,
            seed_pool=seed_pool,
            include_sensitive=include_sensitive,
            include_skipped=include_skipped,
        )
    if mode == "vector":
        use_graph = False
    elif mode == "graph":
        use_graph = True
    store_dir = Path(store_dir)
    cards = load_cards(store_dir)
    embeddings, meta = load_embeddings(store_dir)
    edges = load_edges(store_dir)

    if len(cards) != len(embeddings):
        raise ValueError(
            "cards.jsonl and embeddings.npy disagree on length; rebuild the store"
        )

    ids = [c["id"] for c in cards]

    # 1) 计算允许参与检索的卡片下标。
    # 默认排除 skipped 和 sensitive；它们是「存在但默认不召回」。
    allowed: list[int] = []
    for i, card in enumerate(cards):
        if not include_skipped and card.get("status") == "skipped":
            continue
        if not include_sensitive and card.get("sensitivity") == "sensitive":
            continue
        allowed.append(i)

    if not allowed:
        return {"query": query, "cards": [], "seeds": [], "note": "no allowed cards"}

    # 2) 用同一个 embedder 编码 query，算余弦相似度。
    embedder = _embedder_from_meta(meta)
    qvec = embedder.embed([query])[0]
    sims = cosine(qvec, embeddings)

    # 把不允许的卡片设成 -1，这样它们不会被选成种子。
    # np.full_like 创建和 sims 同形状的数组，先全部填 -1。
    masked = np.full_like(sims, -1.0)
    masked[allowed] = sims[allowed]

    # 3) 选种子：至少 5 个，最多 top_k 个。
    seed_n = min(max(top_k, 5), len(allowed))
    seed_idx = np.argsort(masked)[::-1][:seed_n]
    seeds = {
        ids[int(i)]: float(max(masked[int(i)], 0.0))
        for i in seed_idx
    }
    # 相似度为 0 的种子没有意义，去掉
    seeds = {k: v for k, v in seeds.items() if v > 0}

    # 4) 如果一个种子都没有，退化为「按重要性返回」。
    # 这通常发生在 query 和所有卡片都完全没有语义重叠时（比如 hash embedder）。
    if not seeds:
        ranked = sorted(
            (cards[i] for i in allowed),
            key=lambda c: (c.get("importance", 0), c.get("updated_at") or ""),
            reverse=True,
        )[:top_k]
        return {
            "query": query,
            "cards": [_card_result(c, 0.0, 0.0, None) for c in ranked],
            "seeds": [],
            "note": "no semantic seeds; returned high-importance cards",
        }

    # 5) 从种子出发跑 PPR；纯向量基线跳过图计算。
    if use_graph:
        ppr = personalized_pagerank(edges, ids, seeds, alpha=alpha)
    else:
        ppr = np.zeros(len(ids), dtype=np.float32)

    # 6) 重排：60% 语义 + 30% 图 + 10% 重要性。
    # 这个权重是 v1 的经验值，后续应该用评测集调。
    ppr_max = float(ppr.max()) or 1.0
    importance_max = max(
        (c.get("importance", 0) for c in cards), default=1
    ) or 1

    ranked: list[tuple[float, int]] = []
    for i in allowed:
        card = cards[i]
        semantic = float(max(sims[i], 0.0))
        graph_score = float(ppr[i] / ppr_max)
        imp = float(card.get("importance", 0)) / importance_max
        if use_graph:
            score = 0.60 * semantic + 0.30 * graph_score + 0.10 * imp
        else:
            # 纯向量基线：不掺图分数和重要性
            score = semantic
        ranked.append((score, i))

    ranked.sort(reverse=True)

    # 7) 组装结果；explain 时补一条从种子到目标的路径。
    results = []
    for score, i in ranked[:top_k]:
        card = cards[i]
        path = None
        if explain:
            path = shortest_path(edges, list(seeds), card["id"], max_depth=depth)
        results.append(_card_result(card, score, float(ppr[i]), path))

    return {
        "query": query,
        "embedder": {
            "provider": meta.get("provider"),
            "model": meta.get("model"),
        },
        "seeds": sorted(seeds.items(), key=lambda kv: kv[1], reverse=True)[:seed_n],
        "cards": results,
        "note": "",
    }


def _logical_adjacency(
    edges: list[dict],
    ids: list[str],
    *,
    min_weight: float,
    include_semantic: bool = False,
) -> dict[str, list[tuple[str, str, float, str]]]:
    """把边转成邻接表：id → [(邻居, 边类型, 权重, 理由)]。

    语义边默认不参与游走：它和向量种子是同一个信号，
    沿着 semantic 走等于把 top-k 再排一遍，不会产生新的链。
    """
    known = set(ids)
    adj: dict[str, list[tuple[str, str, float, str]]] = defaultdict(list)
    for edge in edges:
        kind = str(edge.get("type", ""))
        if kind == "semantic" and not include_semantic:
            continue
        weight = float(edge.get("weight", 0.0) or 0.0)
        if weight < min_weight:
            continue
        a, b = edge.get("src"), edge.get("dst")
        if a not in known or b not in known or a == b:
            continue
        why = str(edge.get("why") or "")
        adj[a].append((b, kind, weight, why))
        # 有向边（next / part_of / causes）不能反向走，否则时间轴和因果会倒流
        if not edge.get("directed"):
            adj[b].append((a, kind, weight, why))
    return adj


def query_deep(
    store_dir: str | Path,
    query: str,
    top_k: int = 6,
    max_hops: int = 2,
    min_edge_weight: float = 0.6,
    seed_pool: int = 8,
    include_sensitive: bool = False,
    include_skipped: bool = False,
) -> dict:
    """deep 模式：选入口卡，再沿非语义边做有界游走，返回一棵“链”。

    和旧版的区别：
    - 旧版只读入口卡的 links.related，本质是一次性取列表；
    - 新版沿 edges.jsonl 走，最多 max_hops 跳，边权低于 min_edge_weight 不走；
    - 入口卡从 top seed_pool 里挑，不再要求向量 top1 恰好是 link 卡；
    - 选卡时保证父节点已入选，所以结果一定连通，不会出现断链的叶子。

    `semantic` 边默认不参与游走：它和向量种子是同一个信号。
    """
    store_dir = Path(store_dir)
    cards = load_cards(store_dir)
    embeddings, meta = load_embeddings(store_dir)
    edges = load_edges(store_dir)
    ids = [c["id"] for c in cards]
    idx = {cid: i for i, cid in enumerate(ids)}

    allowed: list[int] = []
    for i, card in enumerate(cards):
        if not include_skipped and card.get("status") == "skipped":
            continue
        if not include_sensitive and card.get("sensitivity") == "sensitive":
            continue
        allowed.append(i)
    if not allowed:
        return {
            "query": query,
            "mode": "deep",
            "cards": [],
            "seeds": [],
            "note": "no allowed cards",
        }

    embedder = _embedder_from_meta(meta)
    qvec = embedder.embed([query])[0]
    sims = cosine(qvec, embeddings)
    masked = np.full_like(sims, -1.0)
    masked[allowed] = sims[allowed]

    adj = _logical_adjacency(edges, ids, min_weight=min_edge_weight)

    # 入口卡：在向量 top seed_pool 里挑“有非语义边可走”的卡；
    # link 卡是人工设计的入口，给一点加成。
    pool = np.argsort(masked)[::-1][: max(1, min(seed_pool, len(allowed)))]
    anchor: str | None = None
    anchor_score = float("-inf")
    for i in pool:
        cid = ids[int(i)]
        degree = len(adj.get(cid, []))
        if degree == 0:
            continue
        card = cards[idx[cid]]
        bonus = 0.15 if "layer:link" in card.get("tags", []) else 0.0
        score = float(masked[int(i)]) + bonus + 0.01 * min(degree, 10)
        if score > anchor_score:
            anchor, anchor_score = cid, score

    if anchor is None:
        result = query_store(
            store_dir,
            query,
            top_k=top_k,
            include_sensitive=include_sensitive,
            include_skipped=include_skipped,
            use_graph=True,
        )
        result["mode"] = "deep-fallback"
        result["note"] = "no seed has non-semantic edges; used PPR"
        return result

    # 有界最佳优先游走：只从已入选的节点往外扩，保证结果连通。
    chosen: dict[str, dict] = {
        anchor: {
            "hop": 0,
            "parent": None,
            "edge": None,
            "weight": 1.0,
            "why": "",
            "path": 1.0,
            "score": 1.0,
        }
    }
    counter = itertools.count()
    frontier: list[tuple[float, int, str, dict]] = []

    def push(
        node: str,
        path_score: float,
        hop: int,
        parent: str,
        kind: str | None,
        weight: float,
        why: str,
    ) -> None:
        relevance = max(float(sims[idx[node]]), 0.0)
        score = 0.7 * path_score + 0.3 * relevance
        heapq.heappush(
            frontier,
            (
                -score,
                next(counter),
                node,
                {
                    "hop": hop,
                    "parent": parent,
                    "edge": kind,
                    "weight": weight,
                    "why": why,
                    "path": path_score,
                    "score": score,
                },
            ),
        )

    for nbr, kind, weight, why in adj.get(anchor, []):
        push(nbr, weight, 1, anchor, kind, weight, why)

    while len(chosen) < max(1, top_k) and frontier:
        _, _, node, info = heapq.heappop(frontier)
        if node in chosen:
            continue
        chosen[node] = info
        if info["hop"] >= max_hops:
            continue
        for nbr, kind, weight, why in adj.get(node, []):
            if nbr in chosen:
                continue
            push(
                nbr,
                info["path"] * weight * 0.7,
                info["hop"] + 1,
                node,
                kind,
                weight,
                why,
            )

    def chain_to(node: str) -> list[str]:
        path = [node]
        cur = chosen[node]["parent"]
        while cur:
            path.append(cur)
            cur = chosen[cur]["parent"]
        return list(reversed(path))

    results: list[dict] = []
    for cid, info in chosen.items():
        card = cards[idx[cid]]
        item = _card_result(card, info["score"], 0.0, None)
        if info["parent"]:
            item["via"] = info["parent"]
            item["edge"] = info["edge"]
            item["edge_weight"] = round(float(info["weight"]), 3)
            item["hop"] = info["hop"]
            if info["why"]:
                item["why"] = info["why"]
        results.append(item)

    edge_list = [
        {
            "src": info["parent"],
            "dst": cid,
            "type": info["edge"],
            "weight": round(float(info["weight"]), 3),
            **({"why": info["why"]} if info["why"] else {}),
        }
        for cid, info in chosen.items()
        if info["parent"]
    ]

    return {
        "query": query,
        "mode": "deep",
        "anchor": anchor,
        "seeds": sorted(
            [
                (ids[int(i)], float(masked[int(i)]))
                for i in pool
                if masked[int(i)] > 0
            ],
            key=lambda kv: kv[1],
            reverse=True,
        ),
        "params": {
            "max_hops": max_hops,
            "min_edge_weight": min_edge_weight,
            "seed_pool": seed_pool,
            "include_semantic": False,
        },
        "edges": edge_list,
        "chains": {cid: chain_to(cid) for cid in chosen if cid != anchor},
        "cards": results,
        "note": "anchor + bounded non-semantic walk",
    }


def _card_result(
    card: dict,
    score: float,
    ppr: float,
    path: list[str] | None,
) -> dict:
    """把一张卡裁剪成检索结果，只保留对调用方有用的字段。"""
    out = {
        "id": card["id"],
        "score": round(float(score), 4),
        "ppr": round(float(ppr), 6),
        "title": card.get("title", ""),
        "summary": card.get("summary", ""),
        "section": card.get("section"),
        "thread": card.get("thread"),
        "type": card.get("type"),
        "sensitivity": card.get("sensitivity"),
    }
    if path:
        out["path"] = path
    return out


def stats(store_dir: str | Path) -> dict:
    """返回 store 的统计信息，用于 `haoliang stats`。"""
    store_dir = Path(store_dir)
    cards = load_cards(store_dir)
    manifest = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
    edges = load_edges(store_dir)
    return {
        "cards": len(cards),
        "edges": len(edges),
        "counts": manifest.get("counts", {}),
        "by_thread": manifest.get("by_thread", {}),
    }
