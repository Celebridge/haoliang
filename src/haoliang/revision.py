"""Revision: one isolated editor pass over a draft.

The writer produces a draft; `annotate` returns undefined terms; the
reviser edits only those spans. The main conversation should keep only the
final text, not the revision prompt/response.

This module defines the contract. An LLM adapter implements `Reviser`; tests
use a fake. `apply_edits()` is deterministic and independent of any model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    replacement: str


@dataclass(frozen=True)
class RevisionRequest:
    draft: str
    undefined: list[str]
    max_terms: int = 0


@dataclass(frozen=True)
class RevisionResult:
    text: str
    edits: tuple[Edit, ...] = ()
    notes: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class Reviser(Protocol):
    def revise(self, request: RevisionRequest) -> RevisionResult:
        ...


def apply_edits(text: str, edits: list[Edit] | tuple[Edit, ...]) -> str:
    """Apply non-overlapping edits, right-to-left.

    Right-to-left keeps earlier offsets stable while later spans are replaced.
    Overlapping edits are rejected: the caller should merge them first.
    """
    ordered = sorted(edits, key=lambda edit: edit.start, reverse=True)
    previous_start = len(text) + 1
    for edit in ordered:
        if edit.start < 0 or edit.end < edit.start or edit.end > len(text):
            raise ValueError(f"invalid edit span: {edit}")
        if edit.end > previous_start:
            raise ValueError("edits overlap")
        previous_start = edit.start

    result = text
    for edit in ordered:
        result = result[: edit.start] + edit.replacement + result[edit.end :]
    return result


def revise(
    annotate_result: dict,
    reviser: Reviser,
    *,
    max_terms: int = 5,
) -> RevisionResult:
    """Run one editor pass for the first `max_terms` undefined terms.

    No undefined terms means no editor call at all.
    """
    text = str(annotate_result.get("text", ""))
    undefined = [str(term) for term in annotate_result.get("undefined", [])]
    selected = undefined[: max(0, max_terms)]
    if not selected:
        return RevisionResult(text=text, edits=(), notes="nothing to revise")
    return reviser.revise(
        RevisionRequest(draft=text, undefined=selected, max_terms=max_terms)
    )
