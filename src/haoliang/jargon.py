"""LLM jargon judge for `not_in_ledger` candidates.

The ledger decides known/unknown. This adapter only judges terms that are not
in the ledger at all. It sees the draft, the sentence containing each term,
the domain, and a reader profile. Output is `define / skip / review`.

`review` is not a decision to explain: the caller maps it to the configured
`review_as` behavior and (optionally) records a pending review on the card.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .llm import ChatModel, LLMSettings, OpenAICompatibleClient, strip_code_fence
from .reader_config import JargonRules

JARGON_SYSTEM = """你是濠梁的术语判断器。对每个候选词做两步判断，不要改稿。

第一步：语境义（sense）
- term：在这个句子里是领域术语、专有名词、代码标识符
- common：普通名词、动词、形容词
- ambiguous：语境不足，可能是术语也可能是普通词

第二步：术语范围（scope）
- domain：领域共识术语，换一个项目/对话仍然成立
- project：当前项目、代码库、团队内部才成立的名词
- model：模型自造词、临时命名、当前对话里才有的概念
- common：普通词

第三步：读者缺口（gap）
- unknown：读者很可能不懂
- knows：读者很可能懂
- unsure：无法判断

决策：
- common → skip
- scope=project / model 且当前草稿需要解释 → define，但**不要**把它当读者知识
- domain + unknown → define
- domain + knows → skip
- ambiguous 或 unsure → review
- 不要因为“这个词重要”就 define；只判断读者是否需要解释。
- 如果候选只是代码标识符的碎片（例如 PauseMillis 来自 -XX:MaxGCPauseMillis），sense=ambiguous，action=review。

输出 JSON：
{"decisions":[{"term":"...","sense":"term|common|ambiguous","scope":"domain|project|model|common","gap":"unknown|knows|unsure","action":"define|skip|review","confidence":0.0,"definition":"","reason":"<=20字"}]}

对于 `scope=project/model` 且 `action=define`，在 `definition` 里写一句不超过 30 字的本地定义；其他情况 `definition` 可以为空。

例子：
1. 句子：在 K8s 里，sidecar 和业务容器共享网络命名空间。候选：容器。读者 K8s 弱项。
   → sense=term, scope=domain, gap=unknown, action=define
2. 句子：Java 的容器类包括 ArrayList 和 HashMap。候选：容器。读者 Java 强项。
   → sense=term, scope=domain, gap=knows, action=skip
3. 句子：这块逻辑我们叫「灰烬层」，负责失败后的补偿。候选：灰烬层。
   → sense=term, scope=project, gap=unknown, action=define
4. 句子：模型在这里造了一个词「语义锚点」，指把回答挂回证据。候选：语义锚点。
   → sense=term, scope=model, gap=unknown, action=define
5. 句子：我们先在 K8s 的 sidecar 里打开 debug 日志。候选：debug。读者懂常规后端。
   → sense=common, scope=common, gap=knows, action=skip"""


@dataclass(frozen=True)
class JargonDecision:
    term: str
    action: str
    confidence: float = 0.0
    sense: str = ""
    scope: str = "domain"
    gap: str = ""
    definition: str = ""
    reason: str = ""


@runtime_checkable
class JargonJudge(Protocol):
    def judge(
        self,
        draft: str,
        candidates: list[dict],
        *,
        domain: str | None = None,
        reader_profile: str = "",
        domain_profile: dict | None = None,
    ) -> list[JargonDecision]:
        ...


def sentence_for(text: str, start: int, end: int) -> str:
    """Return the sentence containing one candidate span."""
    boundaries = "。！？!?；;\n"
    left = max((text.rfind(ch, 0, start) for ch in boundaries), default=-1)
    right_candidates = [
        index for ch in boundaries if (index := text.find(ch, end)) >= 0
    ]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : right + 1].strip()


def build_prompt(
    draft: str,
    candidates: list[dict],
    *,
    domain: str | None,
    reader_profile: str,
    domain_profile: dict | None = None,
) -> str:
    terms = "\n".join(
        f"- 词：{item['term']}\n  句子：{item['sentence']}\n"
        f"  账本：{item.get('status', 'not_in_ledger')}"
        for item in candidates
    )
    profile_line = ""
    if domain_profile:
        profile_line = (
            f"领域先验：{domain_profile.get('domain')} "
            f"knowledge={domain_profile.get('knowledge')} "
            f"confidence={domain_profile.get('confidence')}\n\n"
        )
    return f"""读者画像：
{reader_profile}

{profile_line}草稿领域：{domain or "general"}

草稿全文：
{draft}

候选词：
{terms}

请逐个判断，返回 JSON。"""


def _load_json(raw: str) -> dict | None:
    text = strip_code_fence(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


class LLMJargonJudge:
    """One isolated call per draft for the not-in-ledger candidates."""

    def __init__(
        self,
        *,
        rules: JargonRules | None = None,
        client: ChatModel | None = None,
    ):
        self.rules = rules or JargonRules()
        self.client = client or OpenAICompatibleClient(
            LLMSettings(
                model=self.rules.model,
                base_url=self.rules.base_url,
                api_key_env=self.rules.api_key_env,
                temperature=self.rules.temperature,
                max_tokens=self.rules.max_tokens,
            )
        )
        self.last_usage: dict[str, int] = {}

    def judge(
        self,
        draft: str,
        candidates: list[dict],
        *,
        domain: str | None = None,
        reader_profile: str = "",
        domain_profile: dict | None = None,
    ) -> list[JargonDecision]:
        self.last_usage = {}
        selected = candidates[: max(0, self.rules.max_candidates)]
        if not selected:
            return []

        reply = self.client.complete(
            [
                {"role": "system", "content": JARGON_SYSTEM},
                {
                    "role": "user",
                    "content": build_prompt(
                        draft,
                        selected,
                        domain=domain,
                        reader_profile=reader_profile or self.rules.reader_profile,
                        domain_profile=domain_profile,
                    ),
                },
            ],
            temperature=self.rules.temperature,
            max_tokens=self.rules.max_tokens,
            response_format={"type": "json_object"},
        )
        self.last_usage = dict(reply.usage)
        data = _load_json(reply.text)
        if data is None:
            raise ValueError("judge output was not a valid JSON object")

        allowed = {str(item["term"]).casefold() for item in selected}
        decisions: dict[str, JargonDecision] = {}
        raw_decisions = data.get("decisions")
        if not isinstance(raw_decisions, list):
            raise TypeError("judge output is missing a decisions list")
        for item in raw_decisions:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            action = str(item.get("action", "")).strip().casefold()
            if term.casefold() not in allowed or action not in {
                "define",
                "skip",
                "review",
            }:
                continue
            try:
                confidence = float(item.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            confidence = max(0.0, min(1.0, confidence))
            sense = str(item.get("sense", "")).strip().casefold()
            scope = str(item.get("scope", "domain")).strip().casefold()
            if scope not in {"domain", "project", "model", "common"}:
                scope = "domain"
            gap = str(item.get("gap", "")).strip().casefold()
            definition = str(item.get("definition", "")).strip()
            reason = str(item.get("reason", "")).strip()

            if action == "define" and confidence < self.rules.confidence_threshold:
                action = "review"
                reason = (reason + "；低置信度") if reason else "低置信度"
            elif (
                action == "skip"
                and confidence < self.rules.confidence_threshold
                and scope == "domain"
                and sense == "term"
            ):
                action = "review"
                reason = (reason + "；低置信度 skip") if reason else "低置信度 skip"

            decisions[term.casefold()] = JargonDecision(
                term=term,
                action=action,
                confidence=confidence,
                sense=sense,
                scope=scope,
                gap=gap,
                definition=definition,
                reason=reason,
            )

        return [
            decisions[str(item["term"]).casefold()]
            for item in selected
            if str(item["term"]).casefold() in decisions
        ]
