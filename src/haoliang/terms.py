"""term 卡的证据累积、遗忘曲线、层级推导。

核心约定：

- 卡片只存 `evidence`、`stability_days`、`last_evidence_at`，不存层级序号。
- 掌握度 `knowledge` 是证据权重的和，范围 0–1。
- 有效掌握度 `effective_knowledge = knowledge * retention`，随时间衰减。
- 层级由 `ReaderConfig.level_of(effective_knowledge)` 推导，配置改了层级就变了。
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from .reader_config import ReaderConfig, load_config

TERM_ID_RE = re.compile(r"^t_[0-9]{4,6}$")
STATUSES = {"proposal", "canon", "archived"}
AUTHORS = {"player", "ai-proposed", "imported"}
SENSITIVITIES = {"private", "sensitive"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """Treat naive datetimes as UTC; convert aware ones to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _as_utc(parsed)


def _require_time(value: str | None) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    parsed = _parse_time(value)
    if parsed is None:
        raise ValueError(f"invalid time: {value!r}")
    return parsed


def _days_between(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds() / 86400.0)


def _latest_positive_time(card: dict) -> datetime | None:
    latest: datetime | None = None
    for event in card.get("evidence", []):
        if float(event.get("weight", 0.0)) <= 0:
            continue
        parsed = _parse_time(event.get("time"))
        if parsed is not None and (latest is None or parsed > latest):
            latest = parsed
    return latest


def new_term(
    term_id: str,
    term: str,
    domain: str,
    *,
    aliases: list[str] | None = None,
    domains: list[str] | None = None,
    author: str = "ai-proposed",
    sensitivity: str = "private",
    now: datetime | None = None,
    config: ReaderConfig | None = None,
) -> dict:
    """创建一张空的 term 卡（proposal，无证据）。

    `aliases` 是同一个概念的其它写法（中英、缩写、同义词）。
    `domains` 是这张卡出现的上下文；不传时等于 `[domain]`。
    """
    cfg = config or load_config()
    term = str(term).strip()
    domain = str(domain).strip()
    if not term:
        raise ValueError("term must not be empty")
    if not domain:
        raise ValueError("domain must not be empty")
    timestamp = _as_utc(now or _now()).isoformat()
    clean_aliases = [str(alias).strip() for alias in (aliases or []) if str(alias).strip()]
    clean_domains = [
        str(item).strip()
        for item in (domains or [domain])
        if str(item).strip()
    ]
    clean_domains = list(dict.fromkeys(clean_domains))
    if domain not in clean_domains:
        clean_domains.insert(0, domain)
    card = {
        "id": term_id,
        "kind": "term",
        "term": term,
        "domain": domain,
        "aliases": clean_aliases,
        "domains": clean_domains,
        "evidence": [],
        "stability_days": cfg.decay.initial_stability_days,
        "last_evidence_at": None,
        "pending_reviews": [],
        "status": "proposal",
        "author": author,
        "sensitivity": sensitivity,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    return card


def term_knowledge(card: dict, config: ReaderConfig | None = None) -> float:
    """证据权重的和，clamp 到 0–1。"""
    total = sum(float(event.get("weight", 0.0)) for event in card.get("evidence", []))
    return max(0.0, min(1.0, total))


def term_retention(
    card: dict,
    now: datetime | None = None,
    config: ReaderConfig | None = None,
) -> float:
    """艾宾浩斯式保留率：R = exp(-Δt / S)。

    衰减从最近一条**正证据**开始算。负证据会降低 knowledge，但不会重置
    遗忘时钟；否则用户说一句“不懂”反而会让旧知识重新变新。
    """
    cfg = config or load_config()
    last = _latest_positive_time(card)
    if last is None:
        return 1.0
    current = _as_utc(now or _now())
    stability = float(card.get("stability_days") or cfg.decay.initial_stability_days)
    stability = max(stability, cfg.decay.min_stability_days)
    return math.exp(-_days_between(last, current) / stability)


def effective_knowledge(
    card: dict,
    now: datetime | None = None,
    config: ReaderConfig | None = None,
) -> float:
    """当前真正能假设的掌握度：知识 × 保留率。"""
    return term_knowledge(card, config) * term_retention(card, now, config)


def familiarity(
    card: dict,
    now: datetime | None = None,
    config: ReaderConfig | None = None,
) -> dict:
    """把连续掌握度投影成配置里的层级。"""
    cfg = config or load_config()
    knowledge = effective_knowledge(card, now, cfg)
    level = cfg.level_of(knowledge)
    return {
        "id": level.id,
        "label": level.label,
        "min_knowledge": level.min_knowledge,
        "explain": level.explain,
        "knowledge": round(knowledge, 4),
        "retention": round(term_retention(card, now, cfg), 4),
        "status": card.get("status", "proposal"),
    }


def record_evidence(
    card: dict,
    kind: str,
    *,
    quote: str = "",
    source: str = "",
    time: str | None = None,
    weight: float | None = None,
    config: ReaderConfig | None = None,
) -> dict:
    """追加一条证据，并更新稳定性与最后证据时间。

    `kind` 的默认权重来自配置；传入 `weight` 可以覆盖，用于实验或自定义证据。
    """
    cfg = config or load_config()
    if weight is None:
        if kind not in cfg.evidence_weights:
            raise ValueError(f"unknown evidence kind: {kind}")
        weight = cfg.evidence_weights[kind]
    weight = float(weight)
    if not -1.0 <= weight <= 1.0:
        raise ValueError("evidence weight must be in [-1, 1]")

    event_time = _require_time(time) or _now()
    card.setdefault("evidence", []).append(
        {
            "kind": kind,
            "weight": weight,
            "time": event_time.isoformat(),
            "source": source,
            "quote": quote,
        }
    )

    last = _parse_time(card.get("last_evidence_at"))
    if last is None or event_time > last:
        card["last_evidence_at"] = event_time.isoformat()

    stability = float(card.get("stability_days") or cfg.decay.initial_stability_days)
    if weight >= cfg.promotion.strong_weight:
        stability *= cfg.decay.growth_factor
    elif weight < 0:
        stability *= cfg.decay.shrink_factor
    card["stability_days"] = round(
        max(stability, cfg.decay.min_stability_days), 2
    )
    card["updated_at"] = event_time.isoformat()
    return card


def add_pending_review(
    card: dict,
    *,
    source: str,
    domain: str,
    reason: str,
    confidence: float | None = None,
    time: str | None = None,
) -> dict:
    """记录一次“拿不准”的判断，不改变掌握度和衰减。

    review 不是证据：它只表示 judge 当前无法判断。等用户确认后，才写
    `labeled_known` / `asked`。
    """
    if not str(domain).strip():
        raise ValueError("pending review domain must not be empty")
    if not str(reason).strip():
        raise ValueError("pending review reason must not be empty")

    event_time = _require_time(time) or _now()
    item = {
        "time": event_time.isoformat(),
        "source": str(source),
        "domain": str(domain),
        "reason": str(reason),
    }
    if confidence is not None:
        value = float(confidence)
        if not 0.0 <= value <= 1.0:
            raise ValueError("pending review confidence must be in [0, 1]")
        item["confidence"] = round(value, 3)

    for existing in card.get("pending_reviews", []):
        if (
            existing.get("source") == item["source"]
            and existing.get("domain") == item["domain"]
            and existing.get("reason") == item["reason"]
        ):
            return card

    card.setdefault("pending_reviews", []).append(item)
    card["updated_at"] = event_time.isoformat()
    return card


def clear_pending_reviews(
    card: dict,
    *,
    domain: str | None = None,
    reason: str | None = None,
) -> dict:
    """清掉未决判断；可按 domain / reason 只清匹配项。"""
    def keep(item: dict) -> bool:
        if domain is not None and item.get("domain") != domain:
            return True
        return bool(reason is not None and item.get("reason") != reason)

    card["pending_reviews"] = [
        item for item in card.get("pending_reviews", []) if keep(item)
    ]
    return card


def archive_term(
    card: dict,
    *,
    merged_into: str | None = None,
    reason: str = "",
    now: datetime | None = None,
) -> dict:
    """Archive a term. `merged_into` points to the surviving card."""
    if merged_into is not None:
        if merged_into == card.get("id"):
            raise ValueError("a term cannot merge into itself")
        if not TERM_ID_RE.match(str(merged_into)):
            raise ValueError("merged_into must be a term id")
        card["merged_into"] = merged_into
    # Re-archiving a merged card must keep the link: the old name stays
    # resolvable to the surviving card. Only restore_term clears it.
    timestamp = _as_utc(now or _now()).isoformat()
    card["status"] = "archived"
    card["archived_at"] = timestamp
    card["archive_reason"] = reason
    card["pending_reviews"] = []
    card["updated_at"] = timestamp
    return card


def restore_term(card: dict, *, now: datetime | None = None) -> dict:
    """Bring an archived card back as a proposal."""
    card["status"] = "proposal"
    card.pop("merged_into", None)
    card.pop("archived_at", None)
    card.pop("archive_reason", None)
    card["updated_at"] = _as_utc(now or _now()).isoformat()
    return card


def validate_merge(terms: list[dict], source_id: str, target_id: str) -> None:
    """Reject self-merges, missing targets, archived targets and cycles."""
    if source_id == target_id:
        raise ValueError("source and target must differ")
    by_id = {str(card.get("id")): card for card in terms}
    source = by_id.get(source_id)
    target = by_id.get(target_id)
    if source is None:
        raise ValueError(f"source not found: {source_id}")
    if target is None:
        raise ValueError(f"target not found: {target_id}")
    if target.get("status") == "archived":
        raise ValueError("target is archived")
    if source.get("status") == "archived":
        raise ValueError("source is archived")

    seen = {target_id}
    current = target
    while current.get("merged_into"):
        nxt = str(current["merged_into"])
        if nxt in seen:
            raise ValueError("merge cycle detected")
        seen.add(nxt)
        current = by_id.get(nxt)
        if current is None:
            break
    if source_id in seen:
        raise ValueError("merge cycle detected")


def merge_terms(
    terms: list[dict],
    source_id: str,
    target_id: str,
    *,
    reason: str = "",
    now: datetime | None = None,
) -> tuple[dict, dict]:
    """Archive source and point it at target. Names stay resolvable via index.

    Evidence is **moved**, not copied. Copying would make `restore_term(source)`
    double-count the same events on two active cards.
    """
    validate_merge(terms, source_id, target_id)
    by_id = {str(card.get("id")): card for card in terms}
    source = by_id[source_id]
    target = by_id[target_id]
    # Moved evidence and pending text inherit the stricter visibility.
    if source.get("sensitivity") == "sensitive":
        target["sensitivity"] = "sensitive"
    source_evidence = source.get("evidence", [])
    if source_evidence:
        existing_keys = {
            (
                event.get("kind"),
                event.get("time"),
                event.get("source"),
                event.get("quote"),
            )
            for event in target.get("evidence", [])
        }
        for event in source_evidence:
            key = (
                event.get("kind"),
                event.get("time"),
                event.get("source"),
                event.get("quote"),
            )
            if key not in existing_keys:
                target.setdefault("evidence", []).append(dict(event))
                existing_keys.add(key)
        source["evidence"] = []
    source_last = _parse_time(source.get("last_evidence_at"))
    target_last = _parse_time(target.get("last_evidence_at"))
    if source_last is not None and (
        target_last is None or source_last > target_last
    ):
        target["last_evidence_at"] = source_last.isoformat()
    target["stability_days"] = max(
        float(target.get("stability_days") or 0.0),
        float(source.get("stability_days") or 0.0),
    )

    # Union domains so "old name in domain X" keeps resolving after the merge.
    target_domains: list[str] = []
    for domain in [
        str(target.get("domain", "")),
        *(target.get("domains") or []),
        *(source.get("domains") or []),
        str(source.get("domain", "")),
    ]:
        domain = domain.strip()
        if domain and domain not in target_domains:
            target_domains.append(domain)
    if target_domains:
        target["domains"] = target_domains
        target["domain"] = target_domains[0]

    # A human-confirmed source must not lose its status just because it was
    # merged into a proposal.
    if source.get("status") == "canon" and target.get("status") == "proposal":
        target["status"] = "canon"
        target["confirmed_at"] = source.get("confirmed_at") or _as_utc(
            now or _now()
        ).isoformat()

    # Pending reviews are questions about the old name; keep them on the
    # surviving card instead of dropping them with the archive.
    existing_pending = {
        (item.get("source"), item.get("domain"), item.get("reason"))
        for item in target.get("pending_reviews", [])
    }
    for item in source.get("pending_reviews", []):
        key = (item.get("source"), item.get("domain"), item.get("reason"))
        if key not in existing_pending:
            target.setdefault("pending_reviews", []).append(dict(item))
            existing_pending.add(key)

    target["updated_at"] = _as_utc(now or _now()).isoformat()
    archive_term(source, merged_into=target_id, reason=reason, now=now)
    return source, target


def can_promote(
    card: dict,
    now: datetime | None = None,
    config: ReaderConfig | None = None,
) -> dict:
    """检查 proposal → canon 的条件，返回可解释的结果。"""
    cfg = config or load_config()
    current = _as_utc(now or _now())

    if card.get("status") == "archived":
        return {
            "ok": False,
            "already": False,
            "reasons": ["archived"],
            "counts": {"positive": 0, "distinct_days": 0, "strong": 0, "recent_negative": 0},
        }

    if card.get("status") == "canon":
        return {
            "ok": True,
            "already": True,
            "reasons": [],
            "counts": {"positive": 0, "distinct_days": 0, "strong": 0, "recent_negative": 0},
        }

    evidence = card.get("evidence", [])
    positives = [event for event in evidence if float(event.get("weight", 0.0)) > 0]
    # A self-report ("我懂") may shape the interaction layer, but it is never
    # the strong evidence that promotes a card to canon.
    strong = [
        event
        for event in positives
        if float(event.get("weight", 0.0)) >= cfg.promotion.strong_weight
        and event.get("kind") != "self_report"
        and not str(event.get("source", "")).startswith("baseline:")
    ]

    days = set()
    for event in positives:
        parsed = _parse_time(event.get("time"))
        if parsed is not None:
            days.add(parsed.date())

    cooldown_start = current.timestamp() - cfg.promotion.negative_cooldown_days * 86400
    recent_negative = []
    for event in evidence:
        if float(event.get("weight", 0.0)) >= 0:
            continue
        parsed = _parse_time(event.get("time"))
        if parsed is not None and parsed.timestamp() >= cooldown_start:
            recent_negative.append(event)

    reasons: list[str] = []
    if len(positives) < cfg.promotion.min_positive_evidence:
        reasons.append(
            f"positive evidence {len(positives)} < {cfg.promotion.min_positive_evidence}"
        )
    if len(days) < cfg.promotion.min_distinct_days:
        reasons.append(
            f"distinct days {len(days)} < {cfg.promotion.min_distinct_days}"
        )
    if len(strong) < cfg.promotion.min_strong_evidence:
        reasons.append(
            f"strong evidence {len(strong)} < {cfg.promotion.min_strong_evidence}"
        )
    if recent_negative:
        reasons.append(
            f"negative evidence in last {cfg.promotion.negative_cooldown_days} days"
        )

    return {
        "ok": not reasons,
        "already": False,
        "reasons": reasons,
        "counts": {
            "positive": len(positives),
            "distinct_days": len(days),
            "strong": len(strong),
            "recent_negative": len(recent_negative),
        },
    }


def confirm(
    card: dict,
    now: datetime | None = None,
    config: ReaderConfig | None = None,
) -> dict:
    """人审通过：proposal → canon。条件不满足时抛错。"""
    cfg = config or load_config()
    check = can_promote(card, now, cfg)
    if not check["ok"]:
        raise ValueError("cannot promote: " + "; ".join(check["reasons"]))
    card["status"] = "canon"
    card["confirmed_at"] = _as_utc(now or _now()).isoformat()
    errors = validate_term(card)
    if errors:
        raise ValueError("invalid term after confirm: " + "; ".join(errors))
    return card


def validate_term(card: dict) -> list[str]:
    """轻量结构校验。

    这是运行时的第一道门；`schema/term.schema.json` 是外部契约。
    两者保持同一组规则，但这里不引入 jsonschema 依赖。
    """
    errors: list[str] = []
    for field in (
        "id",
        "kind",
        "term",
        "domain",
        "evidence",
        "status",
        "author",
        "sensitivity",
    ):
        if field not in card:
            errors.append(f"missing field: {field}")

    if card.get("kind") != "term":
        errors.append("kind must be 'term'")
    if not TERM_ID_RE.match(str(card.get("id", ""))):
        errors.append("id must match t_0000")
    if not str(card.get("term", "")).strip():
        errors.append("term must not be empty")
    elif str(card.get("term")) != str(card.get("term")).strip():
        errors.append("term must not have leading or trailing whitespace")
    if not str(card.get("domain", "")).strip():
        errors.append("domain must not be empty")
    elif str(card.get("domain")) != str(card.get("domain")).strip():
        errors.append("domain must not have leading or trailing whitespace")

    aliases = card.get("aliases")
    if aliases is not None:
        if not isinstance(aliases, list):
            errors.append("aliases must be a list")
        else:
            seen_aliases: set[str] = set()
            for index, alias in enumerate(aliases):
                if not isinstance(alias, str) or not alias.strip():
                    errors.append(f"aliases[{index}] must be a non-empty string")
                    continue
                if alias != alias.strip():
                    errors.append(
                        f"aliases[{index}] must not have leading or trailing whitespace"
                    )
                key = alias.casefold()
                if key in seen_aliases:
                    errors.append(f"duplicate alias: {alias}")
                seen_aliases.add(key)

    domains = card.get("domains")
    if domains is not None:
        if not isinstance(domains, list) or not domains:
            errors.append("domains must be a non-empty list")
        else:
            for index, item in enumerate(domains):
                if not isinstance(item, str) or not item.strip():
                    errors.append(f"domains[{index}] must be a non-empty string")
                elif item != item.strip():
                    errors.append(
                        f"domains[{index}] must not have leading or trailing whitespace"
                    )
            if card.get("domain") not in domains:
                errors.append("domain must be in domains")
    if card.get("status") not in STATUSES:
        errors.append(f"status must be one of {sorted(STATUSES)}")
    for field in ("merged_into", "superseded_by"):
        value = card.get(field)
        if value is not None:
            if not TERM_ID_RE.match(str(value)):
                errors.append(f"{field} must be a term id")
            elif str(value) == str(card.get("id")):
                errors.append(f"{field} cannot point to itself")
    if card.get("author") not in AUTHORS:
        errors.append(f"author must be one of {sorted(AUTHORS)}")
    if card.get("sensitivity") not in SENSITIVITIES:
        errors.append(f"sensitivity must be one of {sorted(SENSITIVITIES)}")

    evidence = card.get("evidence")
    if not isinstance(evidence, list):
        errors.append("evidence must be a list")
        evidence = []

    for index, event in enumerate(evidence):
        if not isinstance(event, dict):
            errors.append(f"evidence[{index}] must be an object")
            continue
        for field in ("kind", "weight", "time"):
            if field not in event:
                errors.append(f"evidence[{index}] missing {field}")
        weight = event.get("weight")
        if not isinstance(weight, (int, float)) or not -1.0 <= float(weight) <= 1.0:
            errors.append(f"evidence[{index}].weight must be in [-1, 1]")

    if card.get("status") == "canon" and not evidence:
        errors.append("canon term must have at least one evidence item")

    pending = card.get("pending_reviews")
    if pending is not None:
        if not isinstance(pending, list):
            errors.append("pending_reviews must be a list")
        else:
            for index, item in enumerate(pending):
                if not isinstance(item, dict):
                    errors.append(f"pending_reviews[{index}] must be an object")
                    continue
                for field in ("time", "source", "domain", "reason"):
                    if not str(item.get(field, "")).strip():
                        errors.append(f"pending_reviews[{index}] missing {field}")
                confidence = item.get("confidence")
                if confidence is not None and (
                    not isinstance(confidence, (int, float))
                    or not 0.0 <= float(confidence) <= 1.0
                ):
                    errors.append(
                        f"pending_reviews[{index}].confidence must be in [0, 1]"
                    )

    return errors
