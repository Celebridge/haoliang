"""Reader primitives shared by MCP and other harnesses.

The CLI is one-shot. MCP/harness is long-lived, so this layer owns a
`ReaderRuntime` cache and a write guard. It never writes `canon`; `confirm`
stays a human CLI action.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from .annotate import annotate, find_leaks, term_is_defined
from .glossary import (
    find_local_term_in_root,
    glossary_path,
    iter_local_terms,
    load_local_terms,
    new_local_term,
    next_local_id,
    save_local_terms,
    upsert_local_term,
)
from .jargon import JargonDecision, JargonJudge, LLMJargonJudge
from .pending import resolve_pending
from .reader import domain_snapshot, reader_slice
from .reader_config import ReaderConfig
from .revisers import LLMReviser
from .revision import Reviser, RevisionResult, revise
from .runtime import ReaderRuntime
from .term_store import (
    edit_terms,
    find_by_id,
    find_by_name_exact,
    next_id,
    upsert,
)
from .terms import (
    familiarity,
    new_term,
    record_evidence,
)

# `observe` is an observation API, not a user-confirmation API. Self-reports,
# gold labels and quiz results must go through the human paths so an agent
# cannot mint strong evidence for canon.
OBSERVE_KINDS = {
    "asked",
    "used",
    "no_question",
    "applied",
    "explained",
    "corrected",
}


_CacheKey = tuple[str, str | None, str | None]


@dataclass(frozen=True)
class _JudgeReplay:
    decisions: tuple[JargonDecision, ...]
    last_usage: dict
    review_terms: frozenset[str]

    def judge(self, draft: str, candidates: list[dict], **kwargs) -> list[JargonDecision]:
        names = {item["term"].casefold() for item in candidates}
        return [item for item in self.decisions if item.term.casefold() in names]


class ReaderService:
    def __init__(
        self,
        store: str | Path = "var/reader",
        *,
        allow_write: bool = False,
        glossary_root: str | Path = "var/glossary",
        config: ReaderConfig | None = None,
        config_path: str | Path | None = None,
        judge: JargonJudge | None = None,
        reviser: Reviser | None = None,
    ):
        self.store = Path(store)
        self.allow_write = allow_write
        self.glossary_root = Path(glossary_root)
        self.runtime = ReaderRuntime(
            store,
            config=config,
            config_path=config_path,
        )
        self._judge = judge
        self._reviser = reviser
        self._default_judge: LLMJargonJudge | None = None
        self._default_reviser: LLMReviser | None = None
        self._judge_cache: dict[_CacheKey, _JudgeReplay] = {}
        self._cache_generation = self.runtime.generation
        self._last_write_judge: tuple[_CacheKey, _JudgeReplay] | None = None

    ANNOTATION_CACHE_SIZE = 16

    def _require_write(self) -> None:
        if not self.allow_write:
            raise PermissionError("write tools are disabled; start with --allow-write")

    def _invalidate(self) -> None:
        self.runtime.invalidate()
        self._judge_cache.clear()
        self._last_write_judge = None

    def _sync_cache_generation(self) -> None:
        self.runtime.refresh()
        if self.runtime.generation != self._cache_generation:
            self._judge_cache.clear()
            self._last_write_judge = None
            self._cache_generation = self.runtime.generation

    def _jargon_judge(self) -> JargonJudge:
        if self._judge is not None:
            return self._judge
        rules = self.runtime.config.jargon
        if self._default_judge is None or self._default_judge.rules != rules:
            self._default_judge = LLMJargonJudge(rules=rules)
        return self._default_judge

    def _revision_engine(self) -> Reviser:
        if self._reviser is not None:
            return self._reviser
        rules = self.runtime.config.revision
        if self._default_reviser is None or self._default_reviser.rules != rules:
            self._default_reviser = LLMReviser(rules=rules)
        return self._default_reviser

    def _analyze(
        self,
        text: str,
        domain: str | None,
        project: str | None,
        judge: JargonJudge | None,
        *,
        record_reviews: bool = False,
        record_local_terms: bool = False,
    ) -> dict:
        return annotate(
            self.store,
            text,
            domain=domain,
            index=self.runtime.index,
            extractor=self.runtime.extractor,
            judge=judge,
            record_reviews=record_reviews,
            record_local_terms=record_local_terms,
            project=project,
            glossary_root=self.glossary_root,
            config=self.runtime.config,
            replay_review_terms=judge.review_terms if isinstance(judge, _JudgeReplay) else frozenset(),
        )

    def domain_profile(self, domain: str, *, max_terms: int = 12) -> dict:
        self.runtime.refresh()
        result = domain_snapshot(
            self.store,
            domain,
            max_terms=max_terms,
            config=self.runtime.config,
            index=self.runtime.index,
        )
        return {
            "domain": result["domain"],
            "profile": result["profile"],
            "known": result["known"],
            "uncertain": result["uncertain"],
            "unknown": result["unknown"],
            "counts": result["counts"],
            "advice": result["advice"],
        }

    def brief(
        self,
        query: str,
        *,
        domain: str | None = None,
        domain_mode: str = "soft",
        max_terms: int = 12,
    ) -> dict:
        self.runtime.refresh()
        result = reader_slice(
            self.store,
            query,
            domain=domain,
            domain_mode=domain_mode,
            max_terms=max_terms,
            include_domain_fallback=False,
            config=self.runtime.config,
            index=self.runtime.index,
        )
        fields = (
            "id",
            "term",
            "domain",
            "familiarity",
            "action",
            "knowledge",
            "retention",
            "status",
            "concepts",
            "match",
            "matched_by",
            "domain_match",
        )
        advice = result["advice"]
        if result["summary"]["candidate_terms"] == 0:
            advice = (
                "账本没有匹配到该话题的词条；不要据此推断读者水平，"
                "按一般读者处理或先问一句。"
            )
        return {
            "query": query,
            "domain": domain,
            "domain_mode": domain_mode,
            "domain_profile": result.get("domain_profile"),
            "coverage": result["summary"],
            "known": [
                {key: item.get(key) for key in fields} for item in result["known"]
            ],
            "uncertain": [
                {key: item.get(key) for key in fields}
                for item in result["uncertain"]
            ],
            "unknown": [
                {key: item.get(key) for key in fields} for item in result["unknown"]
            ],
            "advice": advice,
        }

    def annotate(
        self,
        text: str,
        *,
        domain: str | None = None,
        judge: bool = False,
        record_reviews: bool = False,
        record_local_terms: bool = False,
        project: str | None = None,
    ) -> dict:
        if record_reviews or record_local_terms:
            self._require_write()
        self._sync_cache_generation()

        key = (text, domain, project)
        write_mode = record_reviews or record_local_terms
        judge_engine = None
        if judge:
            judge_engine = None if write_mode else self._judge_cache.get(key)
            if judge_engine is None:
                judge_engine = self._jargon_judge()

        result = self._analyze(
            text,
            domain,
            project,
            judge_engine,
            record_reviews=record_reviews,
            record_local_terms=record_local_terms,
        )
        metadata = result["judge"]
        if metadata["reviews_written"] or metadata["local_terms_written"]:
            self._invalidate()
            self._sync_cache_generation()
        if judge and not metadata["error"] and not isinstance(judge_engine, _JudgeReplay):
            replay = _JudgeReplay(
                tuple(JargonDecision(**decision) for decision in metadata["decisions"]),
                deepcopy(metadata["usage"]),
                frozenset(term.casefold() for term in metadata["review_terms_written"]),
            )
            if write_mode:
                # Only the follow-up revision replays the write-time judgment.
                self._last_write_judge = (key, replay)
            else:
                if len(self._judge_cache) >= self.ANNOTATION_CACHE_SIZE:
                    self._judge_cache.pop(next(iter(self._judge_cache)))
                self._judge_cache[key] = replay
        return result

    def revise(
        self,
        text: str,
        *,
        domain: str | None = None,
        judge: bool = False,
        max_terms: int | None = None,
        record_reviews: bool = False,
        record_local_terms: bool = False,
        project: str | None = None,
    ) -> dict:
        if record_reviews or record_local_terms:
            self._require_write()
        if max_terms is not None and max_terms < 0:
            raise ValueError("max_terms must be >= 0")
        annotation = None
        if not record_reviews and not record_local_terms:
            self._sync_cache_generation()
            key = (text, domain, project)
            last_write = self._last_write_judge
            if last_write is not None and last_write[0] == key:
                self._last_write_judge = None
                annotation = self._analyze(text, domain, project, last_write[1])
            elif key in self._judge_cache:
                annotation = self._analyze(text, domain, project, self._judge_cache[key])
        if annotation is None:
            annotation = self.annotate(
                text,
                domain=domain,
                judge=judge,
                record_reviews=record_reviews,
                record_local_terms=record_local_terms,
                project=project,
            )
        limit = (
            self.runtime.config.revision.max_terms
            if max_terms is None
            else max_terms
        )
        if annotation["undefined"] and limit > 0:
            result = revise(
                annotation,
                self._revision_engine(),
                max_terms=limit,
            )
        else:
            result = RevisionResult(
                text=text,
                notes=(
                    "nothing to revise"
                    if not annotation["undefined"]
                    else "max_terms is 0"
                ),
            )
        after = [
            term
            for term in annotation["undefined"]
            if not term_is_defined(result.text, term)
        ]
        notes = result.notes
        if after:
            suffix = "未覆盖：" + "、".join(after)
            notes = f"{notes}；{suffix}" if notes else suffix
        return {
            "draft": text,
            "undefined": annotation["undefined"],
            "undefined_before": annotation["undefined"],
            "undefined_after": after,
            "review": annotation.get("review", []),
            # Report leaks from the edited text, not only from the draft.
            "leaks": find_leaks(result.text),
            "final": result.text,
            "edits": [
                {
                    "start": edit.start,
                    "end": edit.end,
                    "replacement": edit.replacement,
                }
                for edit in result.edits
            ],
            "notes": notes,
            "usage": result.usage,
            "max_terms": limit,
        }

    def observe(
        self,
        term: str,
        domain: str,
        kind: str,
        *,
        quote: str = "",
        source: str = "",
        scope: str = "domain",
    ) -> dict:
        self._require_write()
        if scope != "domain":
            raise ValueError("observe only records domain terms; use glossary")
        if kind not in OBSERVE_KINDS:
            raise ValueError(
                f"observe does not accept {kind!r}; use pending_resolve for "
                "user confirmation"
            )
        local = find_local_term_in_root(
            term,
            root=self.glossary_root,
        )
        if local is not None:
            raise ValueError(f"{term} is a local {local.get('scope')} term")
        self.runtime.refresh()
        with edit_terms(self.store) as terms:
            card = find_by_name_exact(terms, term, domain)
            if card is not None and (
                card.get("status") == "archived" or card.get("merged_into")
            ):
                raise ValueError(f"{term} is archived; restore it first")
            if card is None:
                card = new_term(
                    next_id(terms),
                    term,
                    domain,
                    config=self.runtime.config,
                )
                upsert(terms, card)
            record_evidence(
                card,
                kind,
                quote=quote,
                source=source,
                config=self.runtime.config,
            )
        self._invalidate()
        return self._term_summary(card)

    def propose(
        self,
        term: str,
        domain: str,
        *,
        aliases: list[str] | None = None,
    ) -> dict:
        self._require_write()
        local = find_local_term_in_root(term, root=self.glossary_root)
        if local is not None:
            raise ValueError(
                f"{term} is a local {local.get('scope')} term; "
                "use glossary_add, not a reader card"
            )
        self.runtime.refresh()
        with edit_terms(self.store) as terms:
            existing = find_by_name_exact(terms, term, domain)
            if existing is not None:
                if existing.get("status") == "archived" or existing.get("merged_into"):
                    raise ValueError(f"{term} is archived; restore it first")
                return self._term_summary(existing)
            card = new_term(
                next_id(terms),
                term,
                domain,
                aliases=aliases,
                author="ai-proposed",
                config=self.runtime.config,
            )
            upsert(terms, card)
        self._invalidate()
        return self._term_summary(card)

    def pending_list(self) -> list[dict]:
        self.runtime.refresh()
        return [
            {
                "id": card["id"],
                "term": card["term"],
                "domain": card["domain"],
                "pending_reviews": deepcopy(card.get("pending_reviews", [])),
            }
            for card in self.runtime.terms
            if card.get("pending_reviews")
            and card.get("status") != "archived"
            and card.get("sensitivity") != "sensitive"
        ]

    def pending_resolve(
        self,
        term_id: str,
        action: str,
        *,
        domain: str | None = None,
        reason: str | None = None,
        note: str = "",
        confirmed_by: str = "",
    ) -> dict:
        self._require_write()
        if action not in {"known", "unknown", "archive"}:
            raise ValueError("action must be known, unknown or archive")
        if action in {"known", "unknown"} and not confirmed_by:
            raise ValueError(
                "confirmed_by is required for known/unknown; use 'user' only "
                "after the user explicitly confirms the term"
            )
        self.runtime.refresh()
        with edit_terms(self.store) as terms:
            card = find_by_id(terms, term_id)
            if card is None:
                raise ValueError(f"term not found: {term_id}")
            resolve_pending(
                card,
                action,
                domain=domain,
                reason=reason,
                note=note,
                confirmed_by=confirmed_by,
                source="pending:mcp",
                config=self.runtime.config,
            )
        self._invalidate()
        return self._term_summary(card)

    def glossary_add(
        self,
        term: str,
        scope: str,
        project: str,
        *,
        definition: str = "",
        reason: str = "",
    ) -> dict:
        self._require_write()
        path = glossary_path(project, self.glossary_root)
        items = load_local_terms(path)
        card = new_local_term(
            next_local_id(items),
            term,
            scope,
            project,
            definition=definition,
            reason=reason,
            source="mcp",
        )
        saved = upsert_local_term(items, card)
        save_local_terms(path, items)
        return saved

    def glossary_list(self, project: str | None = None) -> list[dict]:
        if project:
            items = load_local_terms(glossary_path(project, self.glossary_root))
        else:
            items = iter_local_terms(self.glossary_root)
        return [
            item for item in items if item.get("sensitivity") != "sensitive"
        ]

    def _term_summary(self, card: dict) -> dict:
        view = familiarity(card, config=self.runtime.config)
        return {
            "id": card["id"],
            "term": card["term"],
            "domain": card["domain"],
            "status": card.get("status"),
            "familiarity": view,
            "evidence_count": len(card.get("evidence", [])),
            "pending_reviews": deepcopy(card.get("pending_reviews", [])),
        }
