"""解析 personal-profile 的问题库 source.md 和答案 answers/current.json。

这个模块只做结构解析，不做语义理解：
1. 把 source.md 里的「章节 → 问题」解析成 Python 对象；
2. 把答案 JSON 读成普通 dict，交给 cards.py 组装卡片。

给 Java 背景的读者：
- 这里的 dataclass 类似 Java 的 record/POJO，只装数据，不带复杂行为。
- `str | Path` 是 Python 3.10+ 的联合类型，表示「字符串或 Path 对象」。
"""

# 让类型注解延迟求值：可以在运行时安全使用 `list[str]`、`str | Path` 这类写法。
# 它不影响程序逻辑，只是让类型标注更灵活。
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# 章节标题，例如：`## A. 身份与基本信息 ★★★`
# group(1) 是章节字母，group(2) 是标题（可能带 ★）。
SECTION_RE = re.compile(r"^## ([A-Z]|[0-9]+)\. (.+)$")

# 问题行，兼容两种写法：
#   - A1. 你希望我怎么称呼你？
#   ### A1. 你希望我怎么称呼你？
ITEM_RE = re.compile(r"^(?:- |### )([A-Z][0-9]+)\. (.+)$")


@dataclass
class SourceItem:
    """一个问题。相当于 Java 里的简单 POJO。"""

    id: str
    text: str
    section_id: str
    section_title: str
    priority: int = 0


@dataclass
class SourceSection:
    """一个章节，包含若干问题。"""

    id: str
    title: str
    priority: int
    items: list[SourceItem]


def parse_source(path: str | Path) -> list[SourceSection]:
    """把 source.md 解析成章节列表。

    解析规则：
    - `## X. 标题` 开始一个新章节；
    - `- X1. 问题` 或 `### X1. 问题` 是章节里的问题；
    - `## 0. ...` 是说明性章节，跳过；
    - 标题里的 ★ 数量作为优先级（3 星 = 3）。
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    sections: list[SourceSection] = []
    current: SourceSection | None = None  # 当前正在处理的章节

    for line in lines:
        # 先判断是不是章节标题
        m = SECTION_RE.match(line)
        if m:
            sid, title = m.group(1), m.group(2).strip()
            if sid == "0":
                # `## 0. 三种填法` 这种说明章节不进入卡片
                current = None
                continue
            # 标题里的 ★ 数量就是优先级；展示时把 ★ 去掉
            priority = title.count("★")
            clean = title.replace("★", "").strip()
            current = SourceSection(id=sid, title=clean, priority=priority, items=[])
            sections.append(current)
            continue

        # 还没进入任何章节时，忽略文件头部的说明文字
        if current is None:
            continue

        # 再判断是不是问题行
        m = ITEM_RE.match(line)
        if m:
            current.items.append(
                SourceItem(
                    id=m.group(1),
                    text=m.group(2).strip(),
                    section_id=current.id,
                    section_title=current.title,
                    priority=current.priority,
                )
            )

    return sections


def load_answers(path: str | Path) -> dict:
    """读取 answers/current.json，返回其中的 `answers` 字典。

    导出的 JSON 结构大致是：
        {
          "app": "...",
          "exportedAt": "...",
          "answers": {"A1": {"text": "...", "skip": false, "ts": ...}, ...}
        }
    这里只关心 `answers` 部分。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    answers = data.get("answers")
    if not isinstance(answers, dict):
        # 用 TypeError 而不是 ValueError：这是「数据结构不对」，不是「值不合法」
        raise TypeError(f"{path}: missing 'answers' object")
    return answers


def load_export_meta(path: str | Path) -> dict:
    """读取导出文件里除 `answers` 之外的元信息（导出时间、版本等）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # dict comprehension：过滤掉 answers，只保留元信息
    return {k: v for k, v in data.items() if k != "answers"}
