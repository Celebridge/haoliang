"""terms.jsonl 的读写与 id 分配。

这一层只做存储：

- 一个 store 目录下有一个 `terms.jsonl`；
- 一行一张 term 卡；
- 写入用临时文件 + `os.replace`，避免中途失败留下半个文件；
- 不判断 familiarity，不决定能不能升级——那是 `terms.py` 的事。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9_+#.-]+|[\u4e00-\u9fff]+", re.IGNORECASE)


def terms_path(store: str | Path) -> Path:
    return Path(store) / "terms.jsonl"


class _AhoCorasick:
    """Small pure-Python Aho-Corasick automaton.

    Good enough for a few thousand names; avoids an extra dependency. The
    automaton is built once per `TermIndex`, not per draft.
    """

    def __init__(self, words: list[str]):
        self.goto: list[dict[str, int]] = [{}]
        self.fail: list[int] = [0]
        self.out: list[list[str]] = [[]]
        for word in words:
            if not word:
                continue
            node = 0
            for char in word:
                nxt = self.goto[node].get(char)
                if nxt is None:
                    nxt = len(self.goto)
                    self.goto[node][char] = nxt
                    self.goto.append({})
                    self.fail.append(0)
                    self.out.append([])
                node = nxt
            self.out[node].append(word)

        queue: deque[int] = deque()
        for child in self.goto[0].values():
            queue.append(child)
        while queue:
            node = queue.popleft()
            for char, nxt in self.goto[node].items():
                queue.append(nxt)
                fallback = self.fail[node]
                while fallback and char not in self.goto[fallback]:
                    fallback = self.fail[fallback]
                self.fail[nxt] = self.goto[fallback].get(char, 0)
                self.out[nxt].extend(self.out[self.fail[nxt]])

    def iter(self, text: str):
        node = 0
        for index, char in enumerate(text):
            while node and char not in self.goto[node]:
                node = self.fail[node]
            node = self.goto[node].get(char, 0)
            for word in self.out[node]:
                yield index - len(word) + 1, index + 1, word


class TermIndex:
    """Lookup index over term cards.

    - `by_name`: term/alias → cards (active + merged; archived-only excluded)
    - `by_token` / `by_domain`: active cards only
    - `archived_by_name`: archived cards, used to avoid reviving them
    """

    def __init__(self, terms: list[dict]):
        self.terms = sorted(terms, key=lambda card: str(card.get("id", "")))
        self.by_id = {str(card.get("id", "")): card for card in self.terms}
        self.active_terms = [
            card
            for card in self.terms
            if card.get("status") != "archived" and not card.get("merged_into")
        ]
        self.lookup_terms = [
            card
            for card in self.terms
            if card.get("status") != "archived" or card.get("merged_into")
        ]
        self.by_name: dict[str, list[dict]] = defaultdict(list)
        self.by_token: dict[str, list[dict]] = defaultdict(list)
        self.by_domain: dict[str, list[dict]] = defaultdict(list)
        self.archived_by_name: dict[str, list[dict]] = defaultdict(list)
        for card in self.lookup_terms:
            self._index_names(card, self.by_name)
        self._names_by_id: dict[str, dict[str, str]] = defaultdict(dict)
        for card in self.lookup_terms:
            target = self._resolve(card)
            if target is None or target.get("status") == "archived":
                continue
            # Older ledgers may have merged a sensitive card into a visible
            # one. Do not expose the source name in the visible card's slice.
            if card.get("sensitivity") == "sensitive" and target.get("sensitivity") != "sensitive":
                continue
            names = self._names_by_id[target["id"]]
            for name in term_names(card):
                names.setdefault(name.casefold(), name)
        for card in self.active_terms:
            seen: set[str] = set()
            for key in self._names_by_id.get(card["id"], {}):
                for token in TOKEN_RE.findall(key):
                    if token in seen:
                        continue
                    seen.add(token)
                    self.by_token[token].append(card)
            for domain in card_domains(card):
                self.by_domain[domain].append(card)
        for card in self.terms:
            if card.get("status") == "archived" and not card.get("merged_into"):
                self._index_names(card, self.archived_by_name)
        self.automaton = _AhoCorasick(list(self.by_name))

    @staticmethod
    def _index_names(card: dict, target: dict[str, list[dict]]) -> None:
        seen: set[str] = set()
        for name in term_names(card):
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            target[key].append(card)

    def _resolve(self, card: dict) -> dict | None:
        seen: set[str] = set()
        current = card
        while current.get("merged_into"):
            target_id = str(current["merged_into"])
            if target_id in seen:
                return None
            seen.add(target_id)
            target = self.by_id.get(target_id)
            if target is None:
                return None
            current = target
        return current

    def names_for_card(self, card: dict) -> Iterable[str]:
        """Names resolving to a live card, including names inherited by merge."""
        return self._names_by_id.get(card["id"], {}).values()

    def find(
        self, term: str, domain: str | None = None, *, include_sensitive: bool = True
    ) -> dict | None:
        fallback = None
        for card in self.by_name.get(term.casefold(), []):
            resolved = self._resolve(card)
            if resolved is None or resolved.get("status") == "archived":
                continue
            if not include_sensitive and (
                card.get("sensitivity") == "sensitive"
                or resolved.get("sensitivity") == "sensitive"
            ):
                continue
            if not domain or domain in card_domains(resolved):
                return resolved
            if fallback is None:
                fallback = resolved
        return fallback

    def is_archived(self, term: str, domain: str | None = None) -> dict | None:
        for card in self.by_name.get(term.casefold(), []):
            if domain and domain not in card_domains(card):
                continue
            resolved = self._resolve(card)
            if resolved is not None and resolved.get("status") == "archived":
                return resolved
        matches = self.archived_by_name.get(term.casefold(), [])
        if not matches:
            return None
        if domain:
            for card in matches:
                if domain in card_domains(card):
                    return card
        return matches[0]

    def iter_matches(self, text: str):
        """Yield (start, end, name_key) for every term/alias in `text`."""
        yield from self.automaton.iter(text.casefold())

    def candidate_cards(
        self,
        query: str,
        *,
        domain: str | None = None,
        include_domain: bool = False,
    ) -> list[dict]:
        """Narrow cards for a query without scoring the whole ledger."""
        selected: dict[str, dict] = {}
        for _start, _end, key in self.iter_matches(query):
            for card in self.by_name.get(key, []):
                resolved = self._resolve(card)
                if resolved is None or resolved.get("status") == "archived":
                    continue
                selected[str(resolved.get("id", ""))] = resolved
        for token in TOKEN_RE.findall(query.casefold()):
            for card in self.by_token.get(token, []):
                selected[str(card.get("id", ""))] = card
        if include_domain and domain:
            for card in self.by_domain.get(domain, []):
                selected[str(card.get("id", ""))] = card
        return [selected[key] for key in sorted(selected)]


def load_terms(store: str | Path) -> list[dict]:
    path = terms_path(store)
    if not path.exists():
        return []
    terms = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Restore the visibility of older merges in memory. A later write carries
    # the inherited marker forward, including through a chain of merges.
    by_id = {card["id"]: card for card in terms}
    for card in terms:
        if card.get("sensitivity") != "sensitive":
            continue
        target = by_id.get(card.get("merged_into"))
        while target is not None and target.get("sensitivity") != "sensitive":
            target["sensitivity"] = "sensitive"
            target = by_id.get(target.get("merged_into"))
    return terms


@contextmanager
def _term_write_lock(path: Path) -> Iterator[None]:
    # Lock a stable sidecar: replacing terms.jsonl changes its inode.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield  # Closing the descriptor releases the lock, including on failure.


@contextmanager
def edit_terms(store: str | Path) -> Iterator[list[dict]]:
    """Read, mutate and commit one fresh ledger snapshot under a writer lock."""
    with _term_write_lock(terms_path(store)):
        original = load_terms(store)
        terms = deepcopy(original)
        yield terms
        if terms != original:
            save_terms(store, terms)


def save_terms(
    store: str | Path,
    terms: list[dict],
    *,
    validate: bool = True,
) -> None:
    """Atomically replace a complete snapshot. Use edit_terms for live updates."""
    path = terms_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    if validate:
        errors = store_integrity_errors(terms)
        if errors:
            raise ValueError("invalid term store: " + "; ".join(errors[:10]))
    ordered = sorted(terms, key=lambda term: str(term.get("id", "")))
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        try:
            for term in ordered:
                handle.write(json.dumps(term, ensure_ascii=False) + "\n")
        except Exception:
            handle.close()
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


def store_integrity_errors(terms: list[dict]) -> list[str]:
    """Check governance invariants that `validate_term` cannot see.

    - one card per `(term, domain)`;
    - one alias per `(domain, alias)`.

    Cross-domain aliases are allowed: `节点` can exist in `db` and `k8s`.
    """
    errors: list[str] = []
    seen_ids: dict[str, str] = {}
    seen_terms: dict[tuple[str, str], str] = {}
    for card in terms:
        term = str(card.get("term", "")).strip().casefold()
        domain = str(card.get("domain", "")).strip()
        card_id = str(card.get("id", ""))
        if card_id in seen_ids:
            errors.append(
                f"duplicate id: {card_id} used by "
                f"{seen_ids[card_id]} and {card.get('term')}"
            )
        else:
            seen_ids[card_id] = str(card.get("term", ""))
        for current_domain in card_domains(card) or [domain]:
            key = (term, current_domain)
            if key in seen_terms:
                errors.append(
                    f"duplicate term/domain: {card.get('term')} "
                    f"({current_domain}) {seen_terms[key]} vs {card_id}"
                )
            else:
                seen_terms[key] = card_id

    alias_owner: dict[tuple[str, str], str] = {}
    for card in terms:
        card_id = str(card.get("id", ""))
        names = [str(card.get("term", ""))]
        aliases = card.get("aliases") or []
        if isinstance(aliases, list):
            names.extend(str(alias) for alias in aliases)
        for current_domain in card_domains(card):
            for name in names:
                name = name.strip().casefold()
                if not name:
                    continue
                key = (current_domain, name)
                if key in alias_owner and alias_owner[key] != card_id:
                    errors.append(
                        f"alias collision in {current_domain}: {name} "
                        f"{alias_owner[key]} vs {card_id}"
                    )
                else:
                    alias_owner[key] = card_id
    return errors


def next_id(terms: list[dict]) -> str:
    numbers: list[int] = []
    for term in terms:
        raw = str(term.get("id", ""))
        if raw.startswith("t_") and raw[2:].isdigit():
            numbers.append(int(raw[2:]))
    return f"t_{max(numbers, default=0) + 1:04d}"


def find_by_id(terms: list[dict], term_id: str) -> dict | None:
    for term in terms:
        if term.get("id") == term_id:
            return term
    return None


def card_domains(card: dict) -> list[str]:
    """卡片出现的所有领域；没有 domains 时退回单个 domain。"""
    domains = card.get("domains")
    if isinstance(domains, list) and domains:
        return [str(item) for item in domains if str(item).strip()]
    domain = card.get("domain")
    return [str(domain)] if domain else []


def term_names(card: dict) -> list[str]:
    """term + aliases，统一用于匹配。"""
    names = [str(card.get("term", ""))]
    aliases = card.get("aliases") or []
    if isinstance(aliases, list):
        names.extend(str(alias) for alias in aliases)
    return [name for name in names if name.strip()]


def _resolve_merged(terms: list[dict], card: dict) -> dict:
    """Follow merged_into so writes land on the surviving card."""
    by_id = {str(item.get("id", "")): item for item in terms}
    seen: set[str] = set()
    current = card
    while current.get("merged_into"):
        target_id = str(current["merged_into"])
        if target_id in seen:
            break
        seen.add(target_id)
        target = by_id.get(target_id)
        if target is None:
            break
        current = target
    return current


def find_by_name(terms: list[dict], term: str, domain: str | None = None) -> dict | None:
    """按 term 或 alias 查找；domain 只是优先，不是硬条件。"""
    needle = term.casefold()
    matches = [
        card
        for card in terms
        if any(needle == name.casefold() for name in term_names(card))
    ]
    if not matches:
        return None
    if domain:
        for card in matches:
            if domain in card_domains(card):
                return _resolve_merged(terms, card)
    return _resolve_merged(terms, matches[0])


def find_by_name_exact(
    terms: list[dict],
    term: str,
    domain: str,
) -> dict | None:
    """按 term/alias + 指定 domain 精确查找，不回退到其它领域。

    A merged source resolves to its surviving card.
    """
    needle = term.casefold()
    for card in terms:
        if domain not in card_domains(card):
            continue
        if any(needle == name.casefold() for name in term_names(card)):
            return _resolve_merged(terms, card)
    return None


def upsert(terms: list[dict], card: dict) -> None:
    for index, existing in enumerate(terms):
        if existing.get("id") == card.get("id"):
            terms[index] = card
            return
    terms.append(card)
