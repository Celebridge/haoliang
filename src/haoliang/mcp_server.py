"""Haoliang MCP server: reader primitives for a long-lived harness.

Run:

    haoliang-mcp --store var/reader --allow-write

Read tools are enabled by default. Write tools require `--allow-write`.
`confirm` is intentionally not exposed: canon stays a human CLI action.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .reader_tools import ReaderService


def create_server(
    store_dir: str,
    *,
    allow_write: bool = False,
    glossary_root: str | None = None,
    config_path: str | None = None,
):
    """Create a FastMCP server backed by a persistent ReaderService."""
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install haoliang[mcp] to run the MCP server") from exc

    service = ReaderService(
        store_dir,
        allow_write=allow_write,
        glossary_root=glossary_root or "var/glossary",
        config_path=config_path,
    )
    server = FastMCP("haoliang")

    @server.tool(name="haoliang_brief")
    def haoliang_brief(
        query: str,
        domain: str | None = None,
        domain_mode: str = "soft",
        max_terms: int = 12,
    ) -> dict:
        """Read the reader slice: known / uncertain / unknown terms before writing."""
        return service.brief(
            query,
            domain=domain,
            domain_mode=domain_mode,
            max_terms=max_terms,
        )

    @server.tool(name="haoliang_domain_profile")
    def haoliang_domain_profile(domain: str, max_terms: int = 12) -> dict:
        """Read a compact domain-wide profile before writing.

        Use this for baseline depth; use haoliang_brief for the query-specific
        terms.
        """
        return service.domain_profile(domain, max_terms=max_terms)

    @server.tool(name="haoliang_annotate")
    def haoliang_annotate(
        text: str,
        domain: str | None = None,
        judge: bool = False,
        record_reviews: bool | None = None,
        record_local_terms: bool | None = None,
        project: str | None = None,
    ) -> dict:
        """Lint a draft; optionally judge and write pending/local terms.

        `leaks` in the reply lists harness words that leaked into the draft.
        """
        return service.annotate(
            text,
            domain=domain,
            judge=judge,
            record_reviews=(
                service.allow_write if record_reviews is None else record_reviews
            ),
            record_local_terms=(
                service.allow_write
                if record_local_terms is None
                else record_local_terms
            ),
            project=project,
        )

    @server.tool(name="haoliang_revise")
    def haoliang_revise(
        text: str,
        domain: str | None = None,
        judge: bool = False,
        max_terms: int | None = None,
        record_reviews: bool | None = None,
        record_local_terms: bool | None = None,
        project: str | None = None,
    ) -> dict:
        """Run one isolated editor pass over undefined terms."""
        # revise is documented as a read tool. Writes stay opt-in so one
        # editor call cannot create pending reviews as a side effect.
        return service.revise(
            text,
            domain=domain,
            judge=judge,
            max_terms=max_terms,
            record_reviews=(
                False if record_reviews is None else record_reviews
            ),
            record_local_terms=(
                False if record_local_terms is None else record_local_terms
            ),
            project=project,
        )

    @server.tool(name="haoliang_observe")
    def haoliang_observe(
        term: str,
        domain: str,
        kind: str,
        quote: str = "",
        source: str = "",
        scope: str = "domain",
    ) -> dict:
        """Record one domain-term observation. Requires --allow-write.

        Allowed kinds: asked / used / no_question / applied / explained /
        corrected. Self-reports and gold labels go through pending_resolve.
        """
        return service.observe(
            term,
            domain,
            kind,
            quote=quote,
            source=source,
            scope=scope,
        )

    @server.tool(name="haoliang_propose")
    def haoliang_propose(
        term: str,
        domain: str,
        aliases: list[str] | None = None,
    ) -> dict:
        """Create a proposal term card. Never writes canon."""
        return service.propose(term, domain, aliases=aliases)

    @server.tool(name="haoliang_pending_list")
    def haoliang_pending_list() -> list[dict]:
        """List cards with unresolved pending reviews."""
        return service.pending_list()

    @server.tool(name="haoliang_pending_resolve")
    def haoliang_pending_resolve(
        term_id: str,
        action: str,
        domain: str | None = None,
        reason: str | None = None,
        note: str = "",
        confirmed_by: str = "",
    ) -> dict:
        """Resolve a pending review: known / unknown / archive.

        `confirmed_by="user"` is required for known/unknown, not archive.
        """
        return service.pending_resolve(
            term_id,
            action,
            domain=domain,
            reason=reason,
            note=note,
            confirmed_by=confirmed_by,
        )

    @server.tool(name="haoliang_glossary_add")
    def haoliang_glossary_add(
        term: str,
        scope: str,
        project: str,
        definition: str = "",
        reason: str = "",
    ) -> dict:
        """Add a project/model term to the local glossary, not the reader ledger."""
        return service.glossary_add(
            term,
            scope,
            project,
            definition=definition,
            reason=reason,
        )

    @server.tool(name="haoliang_glossary_list")
    def haoliang_glossary_list(project: str | None = None) -> list[dict]:
        """List project/model local terms."""
        return service.glossary_list(project)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(prog="haoliang-mcp")
    parser.add_argument(
        "--store",
        default=os.environ.get("HAOLIANG_STORE", "var/reader"),
    )
    parser.add_argument(
        "--glossary-root",
        default=os.environ.get("HAOLIANG_GLOSSARY", "var/glossary"),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--allow-write", action="store_true")
    args = parser.parse_args()

    server = create_server(
        str(Path(args.store)),
        allow_write=args.allow_write,
        glossary_root=args.glossary_root,
        config_path=args.config,
    )
    server.run()


if __name__ == "__main__":
    main()
