"""Concept cards and derived domain profiles.

Three layers:

- term card      atomic evidence: does the reader know this word?
- concept card   explanation unit: which terms belong together, what comes first
- domain profile prior for unseen terms: is this domain strong or weak?

Term cards stay the source of truth. Concept cards group them; domain profiles
are derived from them so they cannot drift away from the ledger.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from .reader_config import ReaderConfig, load_config
from .term_store import card_domains
from .terms import familiarity

CONCEPT_ID_RE = re.compile(r"^c_[0-9]{4,6}$")
STATUSES = {"proposal", "canon"}
AUTHORS = {"player", "ai-proposed", "imported"}
SENSITIVITIES = {"private", "sensitive"}
RELATIONS = {"confused_with", "part_of", "used_in", "learned_from"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_concept(
    concept_id: str,
    concept: str,
    domain: str,
    label: str,
    members: list[str],
    *,
    summary: str = "",
    prerequisites: list[str] | None = None,
    related: list[dict] | None = None,
    author: str = "ai-proposed",
    sensitivity: str = "private",
    now: datetime | None = None,
) -> dict:
    """Create a concept card. `members` are term ids."""
    timestamp = (now or _now()).isoformat()
    card = {
        "id": concept_id,
        "kind": "concept",
        "concept": concept,
        "domain": domain,
        "label": label,
        "summary": summary,
        "members": [str(item) for item in members if str(item).strip()],
        "prerequisites": [str(item) for item in (prerequisites or []) if str(item).strip()],
        "related": list(related or []),
        "status": "proposal",
        "author": author,
        "sensitivity": sensitivity,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    return card


def validate_concept(card: dict) -> list[str]:
    """Light structural validation, mirroring `schema/concept.schema.json`."""
    errors: list[str] = []
    for field in (
        "id",
        "kind",
        "concept",
        "domain",
        "label",
        "members",
        "status",
        "author",
        "sensitivity",
    ):
        if field not in card:
            errors.append(f"missing field: {field}")

    if card.get("kind") != "concept":
        errors.append("kind must be 'concept'")
    if not CONCEPT_ID_RE.match(str(card.get("id", ""))):
        errors.append("id must match c_0000")
    for field in ("concept", "domain", "label"):
        if not str(card.get(field, "")).strip():
            errors.append(f"{field} must not be empty")

    members = card.get("members")
    if not isinstance(members, list) or not members:
        errors.append("members must be a non-empty list")
    else:
        for index, member in enumerate(members):
            if not re.match(r"^t_[0-9]{4,6}$", str(member)):
                errors.append(f"members[{index}] must be a term id")

    prerequisites = card.get("prerequisites", [])
    if not isinstance(prerequisites, list):
        errors.append("prerequisites must be a list")
    else:
        for index, item in enumerate(prerequisites):
            if not CONCEPT_ID_RE.match(str(item)):
                errors.append(f"prerequisites[{index}] must be a concept id")

    related = card.get("related", [])
    if not isinstance(related, list):
        errors.append("related must be a list")
    else:
        for index, item in enumerate(related):
            if not isinstance(item, dict):
                errors.append(f"related[{index}] must be an object")
                continue
            if item.get("type") not in RELATIONS:
                errors.append(f"related[{index}].type is invalid")
            if not str(item.get("target", "")).strip():
                errors.append(f"related[{index}].target must not be empty")

    if card.get("status") not in STATUSES:
        errors.append(f"status must be one of {sorted(STATUSES)}")
    if card.get("author") not in AUTHORS:
        errors.append(f"author must be one of {sorted(AUTHORS)}")
    if card.get("sensitivity") not in SENSITIVITIES:
        errors.append(f"sensitivity must be one of {sorted(SENSITIVITIES)}")
    return errors


def load_concepts(path: str | Path) -> list[dict]:
    file = Path(path)
    if not file.exists():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def concepts_path(store: str | Path) -> Path:
    return Path(store) / "concepts.jsonl"


def walk_concept(
    concepts: list[dict],
    concept_id: str,
    *,
    max_hops: int = 2,
) -> list[dict]:
    """Walk prerequisites and non-semantic related edges from one concept."""
    by_id = {str(card.get("id")): card for card in concepts}
    if concept_id not in by_id:
        return []

    queue: deque[tuple[str, int]] = deque([(concept_id, 0)])
    seen: set[str] = set()
    ordered: list[dict] = []
    while queue:
        current, depth = queue.popleft()
        if current in seen or depth > max_hops:
            continue
        card = by_id.get(current)
        if card is None:
            continue
        seen.add(current)
        ordered.append(
            {
                "id": current,
                "label": card.get("label", ""),
                "concept": card.get("concept", ""),
                "depth": depth,
                "members": card.get("members", []),
            }
        )
        if depth == max_hops:
            continue
        for prerequisite in card.get("prerequisites", []):
            queue.append((str(prerequisite), depth + 1))
        for related in card.get("related", []):
            if related.get("type") in {"confused_with", "part_of", "used_in"}:
                queue.append((str(related.get("target")), depth + 1))
    return ordered


def save_concepts(path: str | Path, concepts: list[dict]) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    # Validate everything before touching the destination. Otherwise a bad
    # card in the middle would truncate the file and lose the rest.
    for card in concepts:
        errors = validate_concept(card)
        if errors:
            raise ValueError(
                f"invalid concept {card.get('id')}: " + "; ".join(errors)
            )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=file.parent,
        prefix=file.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        try:
            for card in concepts:
                handle.write(json.dumps(card, ensure_ascii=False) + "\n")
        except Exception:
            handle.close()
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, file)


def concepts_for_term(
    concepts: list[dict],
    term_id: str,
    *,
    include_sensitive: bool = False,
) -> list[dict]:
    """Return the concept cards that contain one term.

    Sensitive concept labels are not recalled by default.
    """
    return [
        card
        for card in concepts
        if term_id in card.get("members", [])
        and (include_sensitive or card.get("sensitivity") != "sensitive")
    ]


def derive_domain_profiles(
    terms: list[dict],
    *,
    config: ReaderConfig | None = None,
    now: datetime | None = None,
    domain: str | None = None,
    include_sensitive: bool = False,
) -> list[dict]:
    """Aggregate term familiarity into a per-domain prior.

    The profile is derived, not hand-maintained. A term with more evidence
    contributes more weight; domains with fewer terms get lower confidence.
    `domain` narrows the aggregation when only one prior is needed.
    """
    cfg = config or load_config()
    buckets: dict[str, dict] = {}

    for card in terms:
        if card.get("status") == "archived" or card.get("merged_into"):
            continue
        if not include_sensitive and card.get("sensitivity") == "sensitive":
            continue
        card_domains_list = card_domains(card)
        if domain is not None and domain not in card_domains_list:
            continue
        view = familiarity(card, now=now, config=cfg)
        weight = max(1.0, float(len(card.get("evidence", []))))
        for item_domain in card_domains_list:
            if domain is not None and item_domain != domain:
                continue
            bucket = buckets.setdefault(
                item_domain,
                {"terms": 0, "weight": 0.0, "knowledge": 0.0},
            )
            bucket["terms"] += 1
            bucket["weight"] += weight
            bucket["knowledge"] += float(view["knowledge"]) * weight

    profiles: list[dict] = []
    for item_domain, bucket in sorted(buckets.items()):
        knowledge = (
            bucket["knowledge"] / bucket["weight"] if bucket["weight"] else 0.0
        )
        profiles.append(
            {
                "domain": item_domain,
                "knowledge": round(knowledge, 4),
                "confidence": round(
                    min(
                        1.0,
                        bucket["terms"]
                        / max(cfg.retrieval.domain_confidence_terms, 1.0),
                    ),
                    3,
                ),
                "terms": bucket["terms"],
                "source": "derived",
                "updated_at": (now or _now()).isoformat(),
            }
        )
    return profiles
