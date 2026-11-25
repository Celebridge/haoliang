"""Resolve pending reviews without turning a casual click into a gold label.

`pending_resolve` is the only path that turns a conversation into evidence.
Defaults are deliberately conservative:

- `known`   → `self_report` (+0.45, one click is `heard`, not `usable`)
- `unknown` → `self_report_unknown` (-0.3)
- `--gold`  → `labeled_known` / `asked` (only for explicit human labeling)

A self-report is deduplicated per calendar day: a second click refreshes
retention, it does not stack another +0.45. Only a user may confirm a
self-report; an agent may list or archive, but not certify its own guess.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .reader_config import ReaderConfig, load_config
from .terms import archive_term, clear_pending_reviews, record_evidence

ACTIONS = {"known", "unknown", "archive"}


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _record_once_per_day(
    card: dict,
    kind: str,
    *,
    quote: str,
    source: str,
    now: datetime | None,
    config: ReaderConfig,
) -> dict:
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    for event in card.get("evidence", []):
        if event.get("kind") != kind:
            continue
        # Baseline self-reports and pending resolutions share the `self_report`
        # kind. Keep the baseline event separate, but still dedupe MCP and CLI
        # confirmations of the same day against each other.
        is_baseline = event.get("source") == "baseline:questionnaire"
        if is_baseline != (source == "baseline:questionnaire"):
            continue
        event_time = _parse_time(event.get("time"))
        if event_time is None or event_time.date() != moment.date():
            continue
        event["time"] = moment.isoformat()
        event["source"] = source
        if quote:
            event["quote"] = quote
        last = _parse_time(card.get("last_evidence_at"))
        if last is None or moment > last:
            card["last_evidence_at"] = moment.isoformat()
        card["updated_at"] = moment.isoformat()
        return event
    return record_evidence(
        card,
        kind,
        quote=quote,
        source=source,
        time=moment.isoformat(),
        config=config,
    )


def resolve_pending(
    card: dict,
    action: str,
    *,
    domain: str | None = None,
    reason: str | None = None,
    note: str = "",
    confirmed_by: str,
    gold: bool = False,
    source: str = "pending",
    now: datetime | None = None,
    config: ReaderConfig | None = None,
) -> dict:
    """Apply one pending decision and clear the matching pending item(s)."""
    cfg = config or load_config()
    if action not in ACTIONS:
        raise ValueError("action must be known, unknown or archive")

    if action in {"known", "unknown"} and confirmed_by != "user":
        raise ValueError(
            "only a user-confirmed resolution may write evidence; "
            "an agent may list or archive, not certify its own guess"
        )
    if action in {"known", "unknown"} and card.get("status") == "archived":
        raise ValueError("cannot resolve an archived card; restore it first")
    if action in {"known", "unknown"}:
        matches = [
            item
            for item in card.get("pending_reviews", [])
            if (domain is None or item.get("domain") == domain)
            and (reason is None or item.get("reason") == reason)
        ]
        if not matches:
            raise ValueError("no matching pending review to resolve")
        if len(matches) > 1:
            raise ValueError(
                "multiple pending reviews match; pass domain and/or reason "
                "to resolve exactly one"
            )

    if action == "known":
        kind = "labeled_known" if gold else "self_report"
        _record_once_per_day(
            card,
            kind,
            quote=note or "pending review resolved: known",
            source="gold" if gold else source,
            now=now,
            config=cfg,
        )
    elif action == "unknown":
        kind = "asked" if gold else "self_report_unknown"
        _record_once_per_day(
            card,
            kind,
            quote=note or "pending review resolved: unknown",
            source="gold" if gold else source,
            now=now,
            config=cfg,
        )
    else:
        archive_term(
            card,
            reason=note or "pending review archived",
            now=now,
        )

    clear_pending_reviews(card, domain=domain, reason=reason)
    return card
