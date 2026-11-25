"""Import a filled baseline questionnaire into a Haoliang reader store.

The Markdown answer sheet is the source of truth. The JSON file next to it
provides two things:

- `groups`: term -> domain mapping for the sheet;
- `scale`: score -> level mapping (0/1/2/3 -> configured level ids).

Weights are not hardcoded: a score maps to a level id, and the evidence weight
is the midpoint between its threshold and the next one, using 1 as the upper
bound for the top level.

Baseline self-reports are stored as `self_report` evidence with
`source="baseline:questionnaire"`. They affect familiarity, but `can_promote`
never counts a self-report as the strong evidence required for canon.

Usage:
    PYTHONPATH=src .venv/bin/python eval/import_baseline.py \
        docs/onboarding/baseline-java-2026-11-22.md \
        --store var/reader \
        --domain-map docs/onboarding/baseline-java-2026-11-22.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from haoliang.reader_config import load_config
from haoliang.term_store import (
    card_domains,
    edit_terms,
    next_id,
    term_names,
    upsert,
)
from haoliang.terms import new_term, record_evidence

BASELINE_SOURCE = "baseline:questionnaire"
SCORE_KEYS = {"0", "1", "2", "3"}

# `term（hint）   2` / `term  2` / `**term（hint）**   3` / `term   __`
LINE_RE = re.compile(r"^\s*(?P<term>.+?)\s+(?P<score>[0-3]|_+)\s*$")
GROUP_RE = re.compile(r"^##\s+(?P<title>.+)$")
HINT_RE = re.compile(r"[（(][^（()）]*[)）]\s*$")


def _clean_term(raw: str) -> str:
    """Remove Markdown emphasis, code ticks and the trailing hint."""
    term = raw.strip().strip("`").strip()
    term = term.strip("*").strip()
    hinted = HINT_RE.sub("", term).strip()
    return hinted or term


def _load_spec(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    groups = data.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"domain map has no groups: {path}")
    scale = data.get("scale")
    if not isinstance(scale, list) or not scale:
        raise ValueError(f"domain map has no scale: {path}")
    return data


def _score_to_level(spec: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in spec.get("scale", []):
        key = str(item.get("key", "")).strip()
        level = str(item.get("level", "")).strip()
        if key not in SCORE_KEYS:
            raise ValueError(f"scale key must be 0-3: {key!r}")
        if not level:
            raise ValueError(f"scale entry {key!r} has no level id")
        mapping[key] = level
    missing = SCORE_KEYS - set(mapping)
    if missing:
        raise ValueError(f"scale is missing scores: {sorted(missing)}")
    return mapping


def parse_markdown(path: Path, groups: list[dict]) -> tuple[list[tuple], list[str]]:
    """Return [(domain, term, score)] plus warnings.

    Blank rows (`__`) are returned with an empty score so the caller can report
    how many rows are filled and how many are still blank.
    """
    domains_by_term: dict[str, set[str]] = {}
    for group in groups:
        for item in group.get("terms", []):
            key = str(item["term"]).strip().casefold()
            domains_by_term.setdefault(key, set()).add(group["id"])

    entries: list[tuple[str, str, str]] = []
    warnings: list[str] = []
    current_domain: str | None = None

    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw.strip()
        if not line:
            continue
        group_match = GROUP_RE.match(line)
        if group_match:
            title = group_match.group("title")
            current_domain = None
            for group in groups:
                if str(group.get("title", "")) in title:
                    current_domain = group["id"]
                    break
            if current_domain is None and re.match(r"^##\s+\d+[.、]", line):
                warnings.append(
                    f"line {line_number}: unknown section {title!r}; "
                    "rows below it are skipped until a known section"
                )
            continue
        if line.startswith(("|", ">", "-")):
            continue
        body = line.strip("`").strip()
        match = LINE_RE.match(body)
        if not match:
            continue
        term = _clean_term(match.group("term"))
        if not term or term.startswith("```"):
            continue
        score = "" if set(match.group("score")) == {"_"} else match.group("score")
        domains = domains_by_term.get(term.casefold(), set())
        if current_domain in domains or not domains:
            domain = current_domain
        elif len(domains) == 1:
            domain = next(iter(domains))
        else:
            warnings.append(f"line {line_number}: ambiguous domain for term: {term}")
            continue
        if domain is None:
            warnings.append(f"line {line_number}: no domain for term: {term}")
            continue
        entries.append((domain, term, score))
    return entries, warnings


def parse_json_spec(path: Path) -> tuple[list[tuple], list[str]]:
    """Read terms/scores embedded in the JSON spec, if any."""
    data = _load_spec(path)
    entries: list[tuple] = []
    for group in data.get("groups", []):
        for item in group.get("terms", []):
            score = item.get("score", "")
            score = "" if score is None else str(score).strip()
            entries.append((group["id"], str(item["term"]).strip(), score))
    return entries, []


def _find_exact_active(
    terms: list[dict],
    term: str,
    domain: str,
) -> tuple[dict | None, str]:
    """Find an active card by exact name/alias and domain.

    Returns (card, reason). `reason` is non-empty when a card with that name
    exists but is archived or merged, so the import can skip it explicitly.
    """
    needle = term.casefold()
    for card in terms:
        if domain not in card_domains(card):
            continue
        if not any(needle == name.casefold() for name in term_names(card)):
            continue
        if card.get("merged_into"):
            return None, f"merged into {card['merged_into']}"
        if card.get("status") == "archived":
            return None, "archived"
        return card, ""
    return None, ""


def _has_non_baseline_evidence(card: dict) -> bool:
    return any(
        event.get("source") != BASELINE_SOURCE
        for event in card.get("evidence", [])
    )


def _drop_baseline_evidence(card: dict) -> None:
    card["evidence"] = [
        event
        for event in card.get("evidence", [])
        if event.get("source") != BASELINE_SOURCE
    ]
    card["last_evidence_at"] = None


def _level_weights(levels) -> dict[str, float]:
    """Place a self-report in the middle of its level, not on the threshold.

    Exactly `min_knowledge` would fall back to the previous level after the
    first hour of decay. The midpoint still comes from config, so changing a
    threshold changes the imported weight.
    """
    ordered = sorted(levels, key=lambda level: level.min_knowledge)
    weights: dict[str, float] = {}
    for index, level in enumerate(ordered):
        upper = (
            ordered[index + 1].min_knowledge
            if index + 1 < len(ordered)
            else 1.0
        )
        weights[level.id] = round((level.min_knowledge + upper) / 2, 4)
    return weights


def apply_entries(
    store: Path,
    entries: list[tuple[str, str, str]],
    *,
    now: datetime,
    config,
    score_to_level: dict[str, str],
) -> dict:
    with edit_terms(store) as terms:
        levels = {level.id: level for level in config.levels}
        weights = _level_weights(config.levels)
        report = {
            "created": 0,
            "updated": 0,
            "unchanged": 0,
            "blank": 0,
            "skipped": [],
        }

        seen: dict[tuple[str, str], str] = {}
        for domain, term, score in entries:
            key = (domain, term.casefold())
            if key in seen:
                if seen[key] != score:
                    raise ValueError(
                        f"duplicate row with different scores: {term} ({domain}) "
                        f"{seen[key]!r} vs {score!r}"
                    )
                report["unchanged"] += 1
                continue
            seen[key] = score

            if not score:
                report["blank"] += 1
                continue

            level_id = score_to_level.get(score)
            if level_id is None:
                report["skipped"].append((term, f"unknown score {score!r}"))
                continue
            level = levels.get(level_id)
            if level is None:
                report["skipped"].append((term, f"level {level_id!r} not in config"))
                continue
            # The lowest level means "no evidence", not a tiny positive weight.
            weight = (
                0.0
                if level.min_knowledge <= 0.0
                else float(weights[level_id])
            )

            card, reason = _find_exact_active(terms, term, domain)
            if card is not None and _has_non_baseline_evidence(card):
                report["skipped"].append((term, "has non-baseline evidence"))
                continue
            if card is None and reason:
                report["skipped"].append((term, reason))
                continue

            if card is None:
                card = new_term(
                    next_id(terms),
                    term,
                    domain,
                    author="imported",
                    now=now,
                    config=config,
                )
                upsert(terms, card)
                report["created"] += 1
            else:
                report["updated"] += 1

            _drop_baseline_evidence(card)
            card["stability_days"] = config.decay.initial_stability_days
            card["updated_at"] = now.isoformat()
            if weight <= 0:
                # Score 0: no evidence. Keep status untouched: an existing card
                # may be canon or proposal for reasons outside this questionnaire.
                continue

            record_evidence(
                card,
                "self_report",
                weight=weight,
                quote=f"基线问卷自评：{level_id}",
                source=BASELINE_SOURCE,
                time=now.isoformat(),
                config=config,
            )
            # A one-time calibration should not fade at the same speed as a single
            # conversational observation.
            card["stability_days"] = max(
                float(card.get("stability_days") or 0.0),
                config.decay.baseline_stability_days,
            )

    return report


def _resolve_spec(source: Path, domain_map: str | None) -> Path | None:
    if domain_map:
        path = Path(domain_map)
        if not path.exists():
            raise ValueError(f"domain map not found: {path}")
        return path
    if source.suffix.lower() == ".md":
        path = source.with_suffix(".json")
        if not path.exists():
            raise ValueError(
                f"markdown source needs a domain map next to it: {path}"
            )
        return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="filled markdown sheet or JSON spec")
    parser.add_argument("--store", default="var/reader")
    parser.add_argument("--domain-map", default=None, help="JSON spec for domains")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"error: source not found: {source}")
        return 1
    now = datetime.now(timezone.utc)
    config = load_config()

    try:
        spec_path = _resolve_spec(source, args.domain_map)
        spec = _load_spec(source if spec_path is None else spec_path)
        groups = spec.get("groups", [])
        score_to_level = _score_to_level(spec)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    try:
        if source.suffix.lower() == ".md":
            entries, warnings = parse_markdown(source, groups)
        else:
            entries, warnings = parse_json_spec(source)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    filled = [item for item in entries if item[2]]
    print(
        f"parsed {len(entries)} terms, {len(filled)} filled, "
        f"{len(entries) - len(filled)} blank"
    )
    for warning in warnings:
        print("warn:", warning)
    if not entries:
        print("error: no questionnaire rows found")
        return 1
    if not filled:
        print(
            "error: no filled scores; fill the markdown sheet "
            "(0/1/2/3) before importing"
        )
        return 1

    try:
        report = apply_entries(
            Path(args.store),
            entries,
            now=now,
            config=config,
            score_to_level=score_to_level,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    print(
        f"created={report['created']} updated={report['updated']} "
        f"unchanged={report['unchanged']} blank={report['blank']} "
        f"skipped={len(report['skipped'])}"
    )
    for term, reason in report["skipped"]:
        print(f"skip: {term} ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
