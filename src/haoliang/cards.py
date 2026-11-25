"""把 source.md + answers/current.json 组装成 Haoliang 卡片。

数据流：
    source.md（问题） + answers/current.json（答案）
        → build_card() 逐题组装
        → cards.jsonl（一行一张卡）
        → manifest.json（统计信息）

卡片是 Haoliang 的最小记忆单位：一个问题一张卡；长答案再拆成 `claims`。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .source import SourceItem, load_answers, load_export_meta, parse_source

# 章节 → 卡片类型。
# 这个映射是人工规则，不是模型推理；改这里就能改卡片语义。
SECTION_TYPES = {
    "A": "fact",
    "B": "event",
    "C": "event",
    "D": "fact",
    "E": "event",
    "F": "fact",
    "G": "fact",
    "H": "fact",
    "I": "trait",
    "J": "trait",
    "K": "value",
    "L": "relationship",
    "M": "relationship",
    "N": "preference",
    "O": "preference",
    "P": "preference",
    "Q": "goal",
    "R": "event",
    "S": "preference",
    "T": "instruction",
    "U": "fact",
    "V": "instruction",
    "W": "exercise",
}

# 章节 → 主题线程。线程用于把同一主题的卡片串成链。
SECTION_THREADS = {
    "A": "identity",
    "B": "timeline",
    "C": "family",
    "D": "identity",
    "E": "career",
    "F": "money",
    "G": "health",
    "H": "daily",
    "I": "personality",
    "J": "emotions",
    "K": "values",
    "L": "relationships",
    "M": "relationships",
    "N": "interests",
    "O": "daily",
    "P": "ai",
    "Q": "goals",
    "R": "timeline",
    "S": "preferences",
    "T": "ai",
    "U": "meta",
    "V": "ai",
    "W": "exercises",
}

# 敏感卡片：不删除，只在卡片上打 `sensitivity: sensitive`。
# 检索默认过滤，显式传 `--include-sensitive` 才返回。
SENSITIVE_IDS = {"D11", "R7", "R8", "H14", "G20", "J14"}
SENSITIVE_SECTIONS = {"L"}  # L 组整组（亲密关系）默认视为敏感

# 匹配 19xx / 20xx 年份，用于粗略抽取时间
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def _summary(text: str, limit: int = 180) -> str:
    """把多行文本压成一行摘要，超长截断。

    `" ".join(text.split())` 会把所有连续空白（换行、制表符、多个空格）
    统一成一个空格，相当于 Java 的 `text.replaceAll("\\s+", " ")`。
    """
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _claims(text: str, card_id: str) -> list[dict]:
    """把长答案拆成若干条 claim（可独立检索的最小陈述）。

    规则：
    - 先按行拆；
    - 单行超过 360 字，再按中文/英文句末标点拆；
    - 最多保留 12 条，避免一张卡无限膨胀。
    """
    if not text.strip():
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    claims: list[str] = []

    for line in lines:
        if len(line) > 360:
            # `(?<=[。！？!?；;])` 是「零宽后顾」：在句末标点之后切分，
            # 但标点本身保留在上一段里。
            parts = re.split(r"(?<=[。！？!?；;])", line)
            claims.extend(p.strip() for p in parts if p.strip())
        else:
            claims.append(line)

    # 一张卡最多 12 条 claim；完整原文仍在 card["text"] 里
    claims = claims[:12]

    # enumerate(..., start=1) 生成 1、2、3……，claim id 形如 A1.1、A1.2
    return [
        {"id": f"{card_id}.{i}", "text": claim, "time": None, "confidence": None}
        for i, claim in enumerate(claims, start=1)
    ]


def _extract_time(text: str) -> dict | None:
    """从文本里粗略抽取年份范围。

    这是「够用就好」的规则抽取，不是时间解析器：
    - 没年份 → None
    - 一个年份 → from=该年，to=None
    - 多个年份 → from=最早，to=最晚
    """
    years = sorted({int(y) for y in YEAR_RE.findall(text)})
    if not years:
        return None
    if len(years) == 1:
        return {"from": str(years[0]), "to": None, "precision": "year"}
    return {"from": str(years[0]), "to": str(years[-1]), "precision": "year"}


def _confidence(text: str, skipped: bool) -> str:
    """用关键词粗判答案的确定性。

    这不是严格的置信度模型，只是给检索/展示一个提示：
    - 跳过 → low
    - 出现「不知道/不确定/可能/应该…」→ medium
    - 其他 → high
    """
    if skipped:
        return "low"
    markers = ("不知道", "不确定", "可能", "应该", "想不起来", "记不清")
    if any(m in text for m in markers):
        return "medium"
    return "high"


def _entities(text: str, known: list[str] | None = None) -> list[str]:
    """从文本里匹配已知实体名。

    实体表由外部传入（`--entities`），这里只做字符串包含判断。
    没有实体表时返回空列表，避免瞎猜。
    """
    if not known:
        return []
    return [name for name in known if name and name in text]


def build_card(
    item: SourceItem,
    answer: dict,
    exported_at: str | None = None,
    known_entities: list[str] | None = None,
) -> dict:
    """把「一个问题 + 一个答案」组装成一张卡片。

    返回的是普通 dict，最终会序列化成 JSONL，所以字段名就是卡片 schema。
    """
    # `answer.get("text") or ""`：如果 text 是 None/空字符串，就落到 ""
    text = str(answer.get("text") or "").strip()
    skipped = bool(answer.get("skip")) or not text

    # 答案里的 ts 是毫秒时间戳；转成 ISO 8601 字符串，方便跨语言处理
    ts = answer.get("ts")
    updated_at = (
        datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
        if isinstance(ts, (int, float))
        else None
    )

    thread = SECTION_THREADS.get(item.section_id, "misc")
    card_type = SECTION_TYPES.get(item.section_id, "fact")

    # 标签同时保留 section/thread/type，方便调试和过滤
    tags = [f"section:{item.section_id}", f"thread:{thread}", f"type:{card_type}"]

    sensitivity = (
        "sensitive"
        if item.id in SENSITIVE_IDS or item.section_id in SENSITIVE_SECTIONS
        else "private"
    )

    summary = "（跳过）" if skipped else _summary(text)

    card = {
        "id": item.id,
        "section": item.section_id,
        "section_title": item.section_title,
        "type": card_type,
        "thread": thread,
        # 卡片标题直接用问题文本（截断到 90 字）
        "title": _summary(item.text, 90),
        "question": item.text,
        "summary": summary,
        "claims": [] if skipped else _claims(text, item.id),
        "text": text,
        "tags": tags,
        "entities": _entities(text, known_entities),
        "time": None if skipped else _extract_time(text),
        "confidence": _confidence(text, skipped),
        "sensitivity": sensitivity,
        "status": "skipped" if skipped else "answered",
        "importance": item.priority,
        # 显式关系先留空；后续可以人工补，或由导入器补
        "links": {
            "supersedes": None,
            "contradicts": [],
            "related": [],
            "part_of": [thread],
        },
        "source": {
            "file": "answers/current.json",
            "question_id": item.id,
            "exported_at": exported_at,
        },
        "updated_at": updated_at,
    }
    return card


def build_cards(
    source_md: str | Path,
    answers_json: str | Path,
    out_dir: str | Path,
    known_entities: list[str] | None = None,
) -> dict:
    """批量构建卡片，写出 cards.jsonl 和 manifest.json，返回 manifest。"""
    sections = parse_source(source_md)
    answers = load_answers(answers_json)
    export_meta = load_export_meta(answers_json)
    exported_at = export_meta.get("exportedAt")

    cards: list[dict] = []
    seen_ids: set[str] = set()
    for section in sections:
        for item in section.items:
            if item.id in seen_ids:
                raise ValueError(f"duplicate question id in source: {item.id}")
            seen_ids.add(item.id)
            # 没答的问题也保留成 skipped 卡片，保证 492 题不漏
            answer = answers.get(item.id, {"text": "", "skip": True})
            cards.append(build_card(item, answer, exported_at, known_entities))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # 类似 Java 的 Files.createDirectories

    # JSONL：一行一个 JSON 对象。比一个大 JSON 数组更适合流式读写。
    cards_path = out_dir / "cards.jsonl"
    with cards_path.open("w", encoding="utf-8") as f:
        for card in cards:
            # ensure_ascii=False：中文原样写入，不转成 \uXXXX
            f.write(json.dumps(card, ensure_ascii=False) + "\n")

    manifest = _manifest(cards, export_meta)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _manifest(cards: list[dict], export_meta: dict) -> dict:
    """生成统计清单：总数、已答、跳过、敏感、字数、按章节/类型/线程分布。"""
    by_section: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_thread: dict[str, int] = {}

    for card in cards:
        by_section[card["section"]] = by_section.get(card["section"], 0) + 1
        by_type[card["type"]] = by_type.get(card["type"], 0) + 1
        by_thread[card["thread"]] = by_thread.get(card["thread"], 0) + 1

    # 生成器表达式 + sum：等价于 Java 里遍历计数，但更紧凑
    answered = sum(1 for c in cards if c["status"] == "answered")
    skipped = sum(1 for c in cards if c["status"] == "skipped")
    chars = sum(len(c["text"]) for c in cards if c["status"] == "answered")

    return {
        "name": "Haoliang",
        "version": 1,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source": {
            "exportedAt": export_meta.get("exportedAt"),
            "meta": export_meta.get("meta", {}),
        },
        "counts": {
            "total": len(cards),
            "answered": answered,
            "skipped": skipped,
            "sensitive": sum(1 for c in cards if c["sensitivity"] == "sensitive"),
            "chars": chars,
            # 粗估 token：中文大约 2 字符 ≈ 1 token
            "tokens_est": (chars + 1) // 2,
        },
        # sorted(dict.items()) 按 key 排序，输出稳定，方便 diff
        "by_section": dict(sorted(by_section.items())),
        "by_type": dict(sorted(by_type.items())),
        "by_thread": dict(sorted(by_thread.items())),
    }
