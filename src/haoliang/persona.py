"""把小问题包（preset）编译成一段可粘贴的人设简报。

场景：大肥鱼 5 分钟开局。用户回答 12 个问题，生成一份 Markdown，
直接粘到 DSH / 大肥鱼的人设设置里。
"""

from __future__ import annotations

import json
from pathlib import Path

# 大肥鱼四种记忆 → 中文小标题。
# 这四个 key 和 dsh-feed-the-seek 的 cognition/experience/resonance/self 对齐。
GROUP_TITLES = {
    "cognition": "基本信息",
    "experience": "最近的经历",
    "resonance": "情绪上，你可以这样对我",
    "self": "怎么和我相处",
}


def load_answers(path: str | Path) -> dict:
    """读取答案文件。

    兼容两种格式：
    - 导出文件：{"answers": {...}}
    - 直接的对象：{"name": "...", "city_work": "..."}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("answers"), dict):
        return data["answers"]
    if isinstance(data, dict):
        return data
    raise TypeError("answers must be a JSON object")


def compile_persona(preset_path: str | Path, answers_path: str | Path) -> str:
    """把 preset + answers 编译成 Markdown 人设简报。"""
    preset = json.loads(Path(preset_path).read_text(encoding="utf-8"))
    answers = load_answers(answers_path)

    # 按 kind 分组：cognition / experience / resonance / self
    groups: dict[str, list[tuple[str, str]]] = {}

    for question in preset.get("questions", []):
        # `or ""` 处理 None；strip() 去掉首尾空白
        value = str(answers.get(question["id"], "") or "").strip()
        if not value:
            continue

        kind = question.get("kind", "cognition")
        groups.setdefault(kind, []).append((question["text"], value))

    lines = [
        "# 关于我",
        "",
        f"> 由 Haoliang 根据「{preset.get('title', preset.get('id', 'preset'))}」生成。",
        "",
    ]

    # 固定顺序输出，保证每次生成的 diff 稳定
    for kind in ("cognition", "experience", "resonance", "self"):
        items = groups.get(kind)
        if not items:
            continue

        lines.append(f"## {GROUP_TITLES.get(kind, kind)}")
        lines.append("")

        for question, value in items:
            # rstrip("？?")：去掉问号，让「问：答」读起来更顺
            lines.append(f"- **{question.rstrip('？?')}**：{value}")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
