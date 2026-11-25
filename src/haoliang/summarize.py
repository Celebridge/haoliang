"""用 LLM 给卡片生成信息密度更高的摘要。

背景：
- 人的原话对模型来说噪音很多；
- 直接把长回答塞进上下文，会挤掉更重要的证据卡；
- 摘要用于 embedding 和 query 结果，`text` 仍保留全文。

默认不把敏感卡发到云端；要处理敏感卡需显式 `--include-sensitive`。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .embed import embed_cards, make_embedder
from .graph import build_edges, write_edges
from .index import build_index
from .query import load_cards


def _clean_summary(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 1)].rstrip() + "…"
    return text


def _normalize_time(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    start = value.get("from")
    end = value.get("to")
    precision = value.get("precision") or "unknown"
    if start in (None, ""):
        return None
    return {"from": str(start), "to": str(end) if end else None, "precision": str(precision)}


class Summarizer:
    """一个很小的摘要 agent：批量输入卡片，输出 summary/claims/entities/time。"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        max_chars: int = 120,
        batch_size: int = 8,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install haoliang[openai] to use summarize") from exc

        import os

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is not set")

        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.max_chars = max_chars
        self.batch_size = max(1, batch_size)

    def _system_prompt(self) -> str:
        return f"""你是 Haoliang 的卡片压缩器。把个人档案回答压缩成信息密度高、可检索的摘要。
输出严格 JSON 对象：
{{"items":[{{"id":"...","summary":"<={self.max_chars}字","claims":["<=40字原子陈述"],"entities":["人物/公司/地点/项目"],"time":{{"from":"YYYY","to":"YYYY|null","precision":"year|month|day|unknown"}}|null}}]}}
规则：
- 只压缩，不添加原文没有的信息；
- 保留人物、时间、事件、因果、态度/评价；
- 去掉口语、重复、情绪铺垫、例子细节；
- 保留限定词（可能/不确定/一直/以前/现在）；
- summary 最多 {self.max_chars} 个中文字符；
- claims 最多 8 条，每条最多 40 个中文字符。"""

    def summarize_batch(self, cards: list[dict]) -> dict[str, dict]:
        payload = [
            {
                "id": card["id"],
                "question": card.get("question", ""),
                "text": card.get("text", "")[:4000],
            }
            for card in cards
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            max_tokens=4000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
        )
        content = resp.choices[0].message.content or "{}"
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            content = content.removeprefix("json").strip()
        data = json.loads(content)
        items = data.get("items", data if isinstance(data, list) else [])
        return {str(item.get("id")): item for item in items if item.get("id")}


def summarize_cards(
    cards: list[dict],
    summarizer: Summarizer,
    *,
    include_sensitive: bool = False,
    limit: int = 0,
) -> tuple[list[dict], dict]:
    """返回更新后的卡片副本和统计。"""
    targets = []
    skipped_sensitive = 0
    for card in cards:
        if card.get("status") == "skipped" or not card.get("text"):
            continue
        if card.get("sensitivity") == "sensitive" and not include_sensitive:
            skipped_sensitive += 1
            continue
        targets.append(card)

    if limit:
        targets = targets[:limit]

    updated = {card["id"]: dict(card) for card in targets}
    batches = [
        targets[i : i + summarizer.batch_size]
        for i in range(0, len(targets), summarizer.batch_size)
    ]

    for no, batch in enumerate(batches, start=1):
        print(f"batch {no}/{len(batches)} ({len(batch)} cards)")
        results = summarizer.summarize_batch(batch)
        for card in batch:
            item = results.get(card["id"])
            if not item:
                continue
            target = updated[card["id"]]
            summary = _clean_summary(item.get("summary", ""), summarizer.max_chars)
            if summary:
                target["summary"] = summary
            claims = [
                str(claim).strip()
                for claim in (item.get("claims") or [])
                if str(claim).strip()
            ][:8]
            if claims:
                target["claims"] = [
                    {
                        "id": f"{card['id']}.{i}",
                        "text": claim[:60],
                        "time": None,
                        "confidence": None,
                    }
                    for i, claim in enumerate(claims, start=1)
                ]
            entities = [
                str(x).strip() for x in (item.get("entities") or []) if str(x).strip()
            ]
            if entities:
                target["entities"] = sorted(set(target.get("entities", []) + entities))
            time_value = _normalize_time(item.get("time"))
            if time_value:
                target["time"] = time_value
        time.sleep(0.2)

    merged = [updated.get(card["id"], card) for card in cards]
    stats = {
        "summarized": len(updated),
        "skipped_sensitive": skipped_sensitive,
        "batches": len(batches),
    }
    return merged, stats


def rebuild_store(
    cards: list[dict],
    source_store: Path,
    out: Path,
    *,
    device: str | None = None,
) -> None:
    """用更新后的卡片重建 embeddings / edges / index。"""
    meta = json.loads((source_store / "embedding_meta.json").read_text(encoding="utf-8"))
    provider = meta.get("provider", "bge")
    model = meta.get("model")
    dim = int(meta.get("dim", 1024))
    embedder = make_embedder(
        provider,
        model=model,
        dim=dim,
        device=device,
        batch_size=16,
    )
    embed_cards(cards, embedder, out)
    import numpy as np

    matrix = np.load(out / "embeddings.npy")
    edges = build_edges(cards, matrix, top_k=8, min_sim=0.2)
    write_edges(edges, out)
    build_index(out)

    answered = [c for c in cards if c["status"] == "answered"]
    manifest = {
        "name": "Haoliang summarized",
        "version": 1,
        "counts": {
            "total": len(cards),
            "answered": len(answered),
            "skipped": sum(1 for c in cards if c["status"] == "skipped"),
            "sensitive": sum(1 for c in cards if c["sensitivity"] == "sensitive"),
            "chars": sum(len(c.get("summary", "")) for c in answered),
            "tokens_est": (sum(len(c.get("summary", "")) for c in answered) + 1) // 2,
        },
        "edges": len(edges),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("edges", len(edges))


def summarize_store(
    store: str | Path,
    out: str | Path,
    *,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    api_key_env: str = "DEEPSEEK_API_KEY",
    max_chars: int = 120,
    batch_size: int = 8,
    limit: int = 0,
    include_sensitive: bool = False,
    device: str | None = None,
    rebuild: bool = True,
) -> dict:
    store = Path(store)
    out = Path(out)
    cards = load_cards(store)

    summarizer = Summarizer(
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        max_chars=max_chars,
        batch_size=batch_size,
    )
    merged, stats = summarize_cards(
        cards,
        summarizer,
        include_sensitive=include_sensitive,
        limit=limit,
    )

    out.mkdir(parents=True, exist_ok=True)
    with (out / "cards.jsonl").open("w", encoding="utf-8") as f:
        for card in merged:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")

    if rebuild:
        rebuild_store(merged, store, out, device=device)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="haoliang-summarize")
    parser.add_argument("--store", default="var/store-links")
    parser.add_argument("--out", default="var/store-summarized")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-chars", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-sensitive", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--no-rebuild", action="store_true")
    args = parser.parse_args(argv)

    summarize_store(
        args.store,
        args.out,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_chars=args.max_chars,
        batch_size=args.batch_size,
        limit=args.limit,
        include_sensitive=args.include_sensitive,
        device=args.device,
        rebuild=not args.no_rebuild,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
