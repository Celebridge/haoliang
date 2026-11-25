"""Long-lived reader state for MCP / harness use.

`TermIndex` and the jieba extractor are expensive to rebuild. A server should
load them once and refresh only when `terms.jsonl` or `concepts.jsonl` changes.
"""

from __future__ import annotations

from pathlib import Path

from .concepts import concepts_path, load_concepts
from .extractors import TermExtractor, make_extractor
from .reader_config import DEFAULT_CONFIG_PATH, ReaderConfig, load_config
from .term_store import TermIndex, load_terms, terms_path


def _mtime(path: Path) -> int:
    return path.stat().st_mtime_ns if path.exists() else 0


class ReaderRuntime:
    def __init__(
        self,
        store: str | Path,
        *,
        config: ReaderConfig | None = None,
        config_path: str | Path | None = None,
    ):
        self.store = Path(store)
        # An explicitly injected config is owned by the caller. Otherwise keep
        # the path so a long-lived MCP process can pick up config edits.
        self._config_path: Path | None = (
            None
            if config is not None
            else Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        )
        self.config = config or load_config(self._config_path)
        self._config_mtime = -1
        self._terms_mtime = -1
        self._concepts_mtime = -1
        self.terms: list[dict] = []
        self.concepts: list[dict] = []
        self.index = TermIndex([])
        self.extractor: TermExtractor = make_extractor("jieba")
        self.generation = 0

    def refresh(self) -> None:
        changed = False
        if self._config_path is not None:
            config_mtime = _mtime(self._config_path)
            if config_mtime != self._config_mtime:
                self.config = load_config(self._config_path)
                self._config_mtime = config_mtime
                changed = True
        terms_mtime = _mtime(terms_path(self.store))
        concepts_mtime = _mtime(concepts_path(self.store))
        if terms_mtime != self._terms_mtime:
            self.terms = load_terms(self.store)
            self.index = TermIndex(self.terms)
            self.extractor = make_extractor("jieba")
            self._terms_mtime = terms_mtime
            changed = True
        if concepts_mtime != self._concepts_mtime:
            self.concepts = load_concepts(concepts_path(self.store))
            self._concepts_mtime = concepts_mtime
            changed = True
        if changed:
            self.generation += 1

    def invalidate(self) -> None:
        self._terms_mtime = -1
        self._concepts_mtime = -1
