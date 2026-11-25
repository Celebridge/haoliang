"""Revision contract tests: deterministic edits + one editor pass."""

from __future__ import annotations

import pytest

from haoliang.revision import (
    Edit,
    RevisionRequest,
    RevisionResult,
    apply_edits,
    revise,
)


def test_apply_edits_inserts_definition():
    text = "GC 会在这里停顿。"

    result = apply_edits(text, [Edit(0, 2, "GC（垃圾回收）")])

    assert result == "GC（垃圾回收） 会在这里停顿。"


def test_apply_edits_multiple_non_overlapping():
    text = "A 和 B"

    result = apply_edits(text, [Edit(0, 1, "A1"), Edit(4, 5, "B1")])

    assert result == "A1 和 B1"


def test_apply_edits_rejects_overlap():
    with pytest.raises(ValueError):
        apply_edits("abc", [Edit(0, 2, "x"), Edit(1, 3, "y")])


def test_apply_edits_rejects_out_of_bounds():
    with pytest.raises(ValueError):
        apply_edits("abc", [Edit(0, 4, "x")])


class FakeReviser:
    def revise(self, request: RevisionRequest) -> RevisionResult:
        return RevisionResult(
            text=request.draft + "!",
            edits=(),
            notes=",".join(request.undefined),
        )


def test_revise_caps_undefined_terms():
    result = revise(
        {"text": "draft", "undefined": ["A", "B", "C", "D"]},
        FakeReviser(),
        max_terms=2,
    )

    assert result.text == "draft!"
    assert result.notes == "A,B"


def test_revise_skips_when_no_undefined():
    class Boom:
        def revise(self, request: RevisionRequest) -> RevisionResult:
            raise AssertionError("should not be called")

    result = revise({"text": "draft", "undefined": []}, Boom())

    assert result.text == "draft"
    assert result.notes == "nothing to revise"
