"""Per-project local glossary for terms that are not reader knowledge.

Project-internal names and model coinages belong to a project/session, not to
the reader's long-term vocabulary. They live in `var/glossary/<project>.jsonl`
and never enter `terms.jsonl`.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

LOCAL_ID_RE = re.compile(r"^g_[0-9]{4,6}$")
SCOPES = {"project", "model"}
STATUSES = {"active", "archived"}
SENSITIVITIES = {"private", "sensitive"}
SAFE_PROJECT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def default_glossary_root() -> Path:
    return Path("var") / "glossary"


def glossary_path(project: str, root: str | Path | None = None) -> Path:
    safe = SAFE_PROJECT_RE.sub("_", project.strip()) or "general"
    return Path(root) / f"{safe}.jsonl" if root else default_glossary_root() / f"{safe}.jsonl"


def new_local_term(
    term_id: str,
    term: str,
    scope: str,
    project: str,
    *,
    definition: str = "",
    reason: str = "",
    confidence: float | None = None,
    source: str = "jargon_judge",
    first_seen: str | None = None,
    expires_at: str | None = None,
    sensitivity: str = "private",
    now: datetime | None = None,
) -> dict:
    timestamp = (now or _now()).isoformat()
    item = {
        "id": term_id,
        "kind": "local-term",
        "term": term,
        "scope": scope,
        "project": project,
        "definition": definition,
        "reason": reason,
        "source": source,
        "first_seen": first_seen or timestamp,
        "last_seen": timestamp,
        "expires_at": expires_at,
        "status": "active",
        "sensitivity": sensitivity,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    if confidence is not None:
        item["confidence"] = round(max(0.0, min(1.0, float(confidence))), 3)
    return item


def validate_local_term(item: dict) -> list[str]:
    errors: list[str] = []
    for field in (
        "id",
        "kind",
        "term",
        "scope",
        "project",
        "definition",
        "source",
        "status",
        "sensitivity",
    ):
        if field not in item:
            errors.append(f"missing field: {field}")
    if item.get("kind") != "local-term":
        errors.append("kind must be 'local-term'")
    if not LOCAL_ID_RE.match(str(item.get("id", ""))):
        errors.append("id must match g_0000")
    if item.get("scope") not in SCOPES:
        errors.append(f"scope must be one of {sorted(SCOPES)}")
    if not str(item.get("term", "")).strip():
        errors.append("term must not be empty")
    if not str(item.get("project", "")).strip():
        errors.append("project must not be empty")
    if item.get("status") not in STATUSES:
        errors.append(f"status must be one of {sorted(STATUSES)}")
    if item.get("sensitivity") not in SENSITIVITIES:
        errors.append(f"sensitivity must be one of {sorted(SENSITIVITIES)}")
    confidence = item.get("confidence")
    if confidence is not None and (
        not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        errors.append("confidence must be in [0, 1]")
    return errors


def load_local_terms(path: str | Path) -> list[dict]:
    file = Path(path)
    if not file.exists():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_local_terms(path: str | Path, terms: list[dict]) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    for item in terms:
        errors = validate_local_term(item)
        if errors:
            raise ValueError(
                f"invalid local term {item.get('id')}: " + "; ".join(errors)
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
            for item in terms:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception:
            handle.close()
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, file)


def next_local_id(terms: list[dict]) -> str:
    numbers = [
        int(str(item.get("id", ""))[2:])
        for item in terms
        if str(item.get("id", "")).startswith("g_")
        and str(item.get("id", ""))[2:].isdigit()
    ]
    return f"g_{max(numbers, default=0) + 1:04d}"


def upsert_local_term(terms: list[dict], item: dict) -> dict:
    key = (
        str(item["term"]).casefold(),
        str(item["project"]),
        str(item["scope"]),
    )
    for existing in terms:
        if (
            str(existing.get("term", "")).casefold(),
            str(existing.get("project", "")),
            str(existing.get("scope", "")),
        ) == key:
            existing.update(
                {
                    "definition": item.get("definition")
                    or existing.get("definition", ""),
                    "reason": item.get("reason") or existing.get("reason", ""),
                    "confidence": (
                        item.get("confidence")
                        if item.get("confidence") is not None
                        else existing.get("confidence")
                    ),
                    "source": item.get("source") or existing.get("source", ""),
                    "last_seen": item.get("last_seen", existing.get("last_seen")),
                    "updated_at": item.get("updated_at", existing.get("updated_at")),
                    "status": "active",
                }
            )
            return existing
    terms.append(item)
    return item


def find_local_term(
    terms: list[dict],
    term: str,
    *,
    project: str | None = None,
) -> dict | None:
    needle = term.casefold()
    for item in terms:
        if str(item.get("term", "")).casefold() != needle:
            continue
        if project is not None and item.get("project") != project:
            continue
        return item
    return None


def find_local_by_id(terms: list[dict], term_id: str) -> dict | None:
    return next((item for item in terms if item.get("id") == term_id), None)


def iter_local_terms(root: str | Path | None = None) -> list[dict]:
    directory = Path(root) if root else default_glossary_root()
    if not directory.exists():
        return []
    items: list[dict] = []
    for file in sorted(directory.glob("*.jsonl")):
        items.extend(load_local_terms(file))
    return items


def find_local_term_in_root(
    term: str,
    *,
    root: str | Path | None = None,
    project: str | None = None,
) -> dict | None:
    if project:
        return find_local_term(
            load_local_terms(glossary_path(project, root)),
            term,
            project=project,
        )
    return find_local_term(iter_local_terms(root), term)
