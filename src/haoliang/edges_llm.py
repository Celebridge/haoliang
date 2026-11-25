"""用 LLM 抽取卡片之间的非语义关系（因果、矛盾、同一件事、解释）。

为什么需要这一步：
- 语义 kNN 只能回答“这两张卡像不像”；
- 但“A 导致了 B”“A 和 B 矛盾”“A 是 B 的背景”是 embedding 编码不了的关系；
- 人生链条（anchor → bridge → leaf）靠的是这些有方向、有理由的边。

做法：
- 先用语义 kNN + 共享实体生成候选（每张卡 40 个左右，相似度只当排序不当门槛）；
- 再把「锚卡摘要 + 候选摘要」发给 LLM，让它只判断关系类型，不给它整张图；
- 只保留 confidence 达标、并且不是 none 的边；
- 每条边带 `why`，这是后面解释链条的依据。

不做全量两两比较：503 张卡的全量是 12.6 万对，既贵又大部分无关。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .embed import load_embeddings
from .query import load_cards

RELATIONS = {
    "evidence_for": "src 提供事实/来源，dst 的某条 claim 用它当依据",
    "causes": "src 导致 dst 发生，src 必须更早",
    "explains": "src 说明 dst 为什么是这样",
    "supersedes": "dst 是 src 后来的版本，时间更晚、态度或事实更新",
    "contradicts": "同一时间、同一主体、两个不能同时成立的判断",
    "part_of": "src 属于 dst（章节/阶段/事件组）",
}

# 有方向的关系：src → dst 表示时间/因果上的流向
DIRECTIONAL = {"evidence_for", "causes", "explains", "supersedes", "part_of"}

# 默认只要能证明的关系。复述、同主题、实例-通则都不建边。
# 这批参数是拿 173 条失败样本校准出来的，见 docs/edge-design.md。
DEFAULT_RELATIONS = ("evidence_for", "causes", "explains", "supersedes", "contradicts")


def build_candidates(
    cards: list[dict],
    embeddings: np.ndarray,
    *,
    top_n: int = 40,
    min_sim: float = 0.45,
    entity_slots: int = 8,
) -> dict[str, list[str]]:
    """给每张卡生成候选：语义近邻 + 共享实体。

    为什么不再只取 top-20：
    - 人工写的 68 条 related 边里，一半排在源卡相似度的第 21 名之后；
    - 中位相似度 0.58，比随机 top-20 邻居还低。

    所以相似度只用来排序，不用来当门槛；实体邻居必须留出位置，
    否则会被语义结果挤掉（之前 `picked[:top_n]` 把实体全截断了）。
    """
    ids = [c["id"] for c in cards]
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -1.0)

    by_entity: dict[str, list[str]] = defaultdict(list)
    for card in cards:
        for entity in card.get("entities", []) or []:
            by_entity[str(entity)].append(card["id"])

    semantic_slots = max(1, top_n - entity_slots)

    result: dict[str, list[str]] = {}
    for i, card in enumerate(cards):
        picked: list[str] = []
        seen: set[str] = set()

        for j in np.argsort(sim[i])[::-1][:semantic_slots]:
            cid = ids[int(j)]
            if cid == card["id"] or cid in seen:
                continue
            if float(sim[i, int(j)]) < min_sim:
                break
            seen.add(cid)
            picked.append(cid)

        # 实体邻居按「共同实体数」排序，而不是相似度：
        # 同一个人/公司/地点，即使字面不像，也应该被看到。
        overlap: dict[str, int] = defaultdict(int)
        for entity in card.get("entities", []) or []:
            for cid in by_entity.get(str(entity), []):
                if cid != card["id"]:
                    overlap[cid] += 1
        for cid, _ in sorted(overlap.items(), key=lambda kv: kv[1], reverse=True):
            if len(picked) >= top_n:
                break
            if cid not in seen:
                seen.add(cid)
                picked.append(cid)

        result[card["id"]] = picked
    return result


class EdgeExtractor:
    """一次判断一张锚卡与若干候选卡的关系。"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        max_out: int = 4,
        relations: tuple[str, ...] | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install haoliang[openai] to use edges-llm") from exc

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is not set")

        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.max_out = max_out
        self.relations = tuple(relations or DEFAULT_RELATIONS)
        unknown = [r for r in self.relations if r not in RELATIONS]
        if unknown:
            raise ValueError(f"unknown relations: {unknown}")

    def _system_prompt(self) -> str:
        rels = "\n".join(f"- {k}: {RELATIONS[k]}" for k in self.relations)
        return f"""你是 Haoliang 的关系判定器。给你一张锚卡 A（问题、摘要、claims）和若干候选卡 B。

只判断一件事：**B 是不是 A 的某条 claim 的依据、来源、前因或后续。**

第一步，把每个候选分到四类之一：
- source：B 提供了 A 某条 claim 赖以成立的事实、来源或前因
- same_fact：B 在讲同一件事（换说法、更粗或更细的写法、实例与通则）
- context：B 只是同一主题，支撑不了任何一条 claim
- none：无关

**只有 source 才输出。** same_fact / context / none 一律不要出现在结果里。

第二步，对 source 判定类型和方向。src / dst 必须从给定的 id 里选：
{rels}

硬规则（违反就丢掉，不要输出）：
1. **复述不算 source**：如果 B 只是把 A 的话展开、换说法、举了个例子，判 same_fact。
2. **方向用先后校验**：causes 的 src 必须早于 dst；分不清先后就不要输出 causes。
3. **前后变化不算矛盾**：同一件事在不同时间的两种态度，用 supersedes，不用 contradicts。
4. **汇总卡之间不互相解释**：带 layer:link 或 type:stage 的卡之间，只允许 part_of。
5. **explains 必须多出一个原因或机制**：如果两张卡可以互相改写（同义、详略、实例与通则），就是复述，判 same_fact。
6. **evidence 必须原样抄**：抄不出原文短语，就不要输出。

反例（这些都不该输出）：
- Z1(人生四阶段) → B8(转折点)：B8 只是把 Z1 里提到的一段单独展开，是 same_fact
- Z2(学Java→后端) → B4(决定学Java)：方向反了，应是 B4 → Z2
- R3(羡慕雷军) → R34(以前信努力现在不信)：时间变化，是 supersedes，不是 contradicts
- A12(哮喘/熬夜) → H1(具体作息)：同一事实的粗细写法，是 same_fact
- Z11(希望伴侣理解) → W5(18到28岁自述)：W5 提供背景，方向应是 W5 → Z11
- Z1 → Z2：两张都是汇总卡，只允许 part_of

**必须给证据原文**：evidence 必须是从 A 或 B 的摘要里**原样抄下来的短语**（<=20 字，不要改写、不要总结）。
抄不出来，说明你只是在复述或联想，不该输出这条边。

输出严格 JSON：
{{"edges":[{{"src":"...","dst":"...","type":"...","claim":"锚卡的那条 claim 原文","evidence":"从摘要里原样抄的短语","why":"<=30字，这条证据支撑了什么","confidence":0.0}}]}}

- src / dst 只能取给定的 id，不能自造
- 最多 {self.max_out} 条，取最强、最具体的
- confidence 低于 0.7 不要输出"""

    def extract(self, anchor: dict, candidates: list[dict]) -> list[dict]:
        payload = {
            "anchor": {
                "id": anchor["id"],
                "question": anchor.get("question", ""),
                "summary": anchor.get("summary") or anchor.get("text", "")[:200],
                "claims": [c.get("text", "") for c in anchor.get("claims", [])],
                "tags": anchor.get("tags", []),
                "time": anchor.get("time"),
            },
            "candidates": [
                {
                    "id": c["id"],
                    "question": c.get("question", ""),
                    "summary": c.get("summary") or c.get("text", "")[:200],
                    "tags": c.get("tags", []),
                    "time": c.get("time"),
                }
                for c in candidates
            ],
        }
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=1200,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        content = (resp.choices[0].message.content or "{}").strip()
        if content.startswith("```"):
            content = content.strip("`").removeprefix("json").strip()
        data = json.loads(content)
        raw = data.get("edges", []) or []

        allowed_ids = {anchor["id"]} | {c["id"] for c in candidates}
        by_id = {c["id"]: c for c in candidates}
        by_id[anchor["id"]] = anchor

        kept: list[dict] = []
        for item in raw:
            src = str(item.get("src", ""))
            dst = str(item.get("dst", ""))
            relation = str(item.get("type", ""))
            if src not in allowed_ids or dst not in allowed_ids or src == dst:
                continue
            if relation not in self.relations:
                continue
            if not self._check_rule4(src, dst, relation, by_id):
                continue
            if not self._check_time(src, dst, relation, by_id):
                continue
            if not self._check_evidence(src, dst, item.get("evidence", ""), by_id):
                continue
            item["src"], item["dst"], item["type"] = src, dst, relation
            kept.append(item)
            if len(kept) >= self.max_out:
                break
        return kept

    def _check_evidence(self, src, dst, evidence, by_id) -> bool:
        """证据必须是两张卡摘要里原样出现的短语。

        这是防复述的机械检查：模型如果只是在联想，就抄不出一段真实存在的原文。
        """
        text = str(evidence or "").strip()
        if len(text) < 2:
            return False
        pool = "".join(
            (by_id[src].get("summary") or "")
            + (by_id[dst].get("summary") or "")
            + (by_id[src].get("text") or "")[:400]
            + (by_id[dst].get("text") or "")[:400]
        )
        compact = "".join(pool.split())
        return "".join(text.split()) in compact

    def _check_rule4(self, src, dst, relation, by_id) -> bool:
        """汇总卡之间只允许 part_of。"""
        def is_summary(card: dict) -> bool:
            tags = card.get("tags") or []
            return "layer:link" in tags or "type:stage" in tags

        if relation == "part_of":
            return True
        return not (is_summary(by_id[src]) and is_summary(by_id[dst]))

    def _check_time(self, src, dst, relation, by_id) -> bool:
        """causes 的 src 不能晚于 dst；supersedes 的 dst 不能早于 src。"""
        def year(card: dict) -> int | None:
            value = (card.get("time") or {}).get("from")
            if not value:
                return None
            text = str(value)[:4]
            if not text.isdigit():
                return None
            y = int(text)
            return y if 1900 <= y <= 2030 else None

        a, b = year(by_id[src]), year(by_id[dst])
        if a is None or b is None:
            return True
        if relation == "causes":
            return a <= b
        if relation == "supersedes":
            return b >= a
        return True


def extract_edges(
    cards: list[dict],
    embeddings: np.ndarray,
    extractor: EdgeExtractor,
    *,
    include_sensitive: bool = False,
    limit: int = 0,
    min_confidence: float = 0.6,
    top_n: int = 20,
    ids: list[str] | None = None,
) -> tuple[list[dict], dict]:
    cards_by_id = {c["id"]: c for c in cards}
    candidates = build_candidates(cards, embeddings, top_n=top_n)

    wanted = set(ids or [])
    targets = [
        c
        for c in cards
        if c.get("status") != "skipped"
        and (not wanted or c["id"] in wanted)
        and (include_sensitive or c.get("sensitivity") != "sensitive")
    ]
    if limit:
        targets = targets[:limit]

    edges: list[dict] = []
    failures = 0
    for no, anchor in enumerate(targets, start=1):
        cand_cards = [
            cards_by_id[cid]
            for cid in candidates.get(anchor["id"], [])
            if cid in cards_by_id
            and (include_sensitive or cards_by_id[cid].get("sensitivity") != "sensitive")
        ]
        if not cand_cards:
            continue
        try:
            raw = extractor.extract(anchor, cand_cards)
        except Exception as exc:  # noqa: BLE001 - 单卡失败不应中断全量
            failures += 1
            print(f"[{no}/{len(targets)}] {anchor['id']} failed: {exc}")
            continue

        kept = 0
        for item in raw:
            src = str(item.get("src", ""))
            dst = str(item.get("dst", ""))
            relation = str(item.get("type", "none"))
            confidence = float(item.get("confidence", 0) or 0)
            if src not in cards_by_id or dst not in cards_by_id:
                continue
            if relation not in extractor.relations:
                continue
            if confidence < min_confidence:
                continue

            edges.append(
                {
                    "src": src,
                    "dst": dst,
                    "type": relation,
                    "weight": round(confidence, 3),
                    "source": "llm",
                    "directed": relation in DIRECTIONAL,
                    "claim": str(item.get("claim", ""))[:60],
                    "why": str(item.get("why", ""))[:60],
                }
            )
            kept += 1

        print(f"[{no}/{len(targets)}] {anchor['id']} → {kept} edges")
        time.sleep(0.2)

    # 去重：同一对、同一类型只保留最高置信度
    best: dict[tuple[str, str, str], dict] = {}
    for edge in edges:
        key = (edge["src"], edge["dst"], edge["type"])
        if key not in best or edge["weight"] > best[key]["weight"]:
            best[key] = edge

    stats = {
        "anchors": len(targets),
        "edges": len(best),
        "failures": failures,
        "relations": dict(
            sorted(
                {
                    rel: sum(1 for e in best.values() if e["type"] == rel)
                    for rel in extractor.relations
                }.items()
            )
        ),
    }
    return sorted(best.values(), key=lambda e: (e["src"], e["dst"], e["type"])), stats


def write_edge_file(edges: list[dict], out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for edge in edges:
            f.write(json.dumps(edge, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="haoliang-edges-llm")
    parser.add_argument("--store", default="var/store-summarized")
    parser.add_argument("--out", default="var/edges-llm.jsonl")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--max-out", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument(
        "--relations",
        default=",".join(DEFAULT_RELATIONS),
        help="comma separated relation types to keep",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="", help="comma separated card ids for a trial run")
    parser.add_argument("--include-sensitive", action="store_true")
    args = parser.parse_args(argv)

    cards = load_cards(args.store)
    embeddings, _meta = load_embeddings(args.store)
    extractor = EdgeExtractor(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_out=args.max_out,
        relations=tuple(x.strip() for x in args.relations.split(",") if x.strip()),
    )
    edges, stats = extract_edges(
        cards,
        embeddings,
        extractor,
        include_sensitive=args.include_sensitive,
        limit=args.limit,
        min_confidence=args.min_confidence,
        top_n=args.candidates,
        ids=[x.strip() for x in args.ids.split(",") if x.strip()],
    )
    write_edge_file(edges, args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
