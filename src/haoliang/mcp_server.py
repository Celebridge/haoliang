"""Haoliang MCP server: reader primitives for a long-lived harness.

Run:

    haoliang-mcp --store var/reader --allow-write

Read tools are enabled by default. Write tools require `--allow-write`.
`confirm` is intentionally not exposed: canon stays a human CLI action.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from time import perf_counter

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
    activity_path = Path(store_dir) / "mcp-activity.jsonl"
    log_failed = False

    def record_activity(event: str, **fields) -> None:
        nonlocal log_failed
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            "pid": os.getpid(),
            "write_enabled": allow_write,
            **fields,
        }
        line = (json.dumps(entry, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            activity_path.parent.mkdir(parents=True, exist_ok=True)
            # One append per event keeps records separate across MCP processes.
            fd = os.open(activity_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "wb", buffering=0) as stream:
                stream.write(line)
        except OSError as exc:
            if not log_failed:
                logging.getLogger(__name__).warning(
                    "Haoliang MCP activity log unavailable (%s)", type(exc).__name__,
                )
                log_failed = True

    def logged_tool(name: str):
        def register(handler):
            @wraps(handler)
            def logged(*args, **kwargs):
                start = perf_counter()
                fields = {"status": "error"}
                try:
                    result = handler(*args, **kwargs)
                except Exception as exc:
                    if isinstance(exc, PermissionError):
                        fields["status"] = "denied"
                    fields["error_type"] = type(exc).__name__
                    raise
                else:
                    fields["status"] = "ok"
                    if name == "haoliang_brief":
                        fields["matched_terms"] = result["coverage"]["terms"]
                    elif name == "haoliang_observe":
                        fields["evidence_written"] = 1
                    elif name == "haoliang_annotate":
                        judge = result["judge"]
                        fields["reviews_written"] = judge["reviews_written"]
                        fields["local_terms_written"] = judge["local_terms_written"]
                    return result
                finally:
                    record_activity(
                        "tool_call",
                        tool=name,
                        duration_ms=round((perf_counter() - start) * 1000, 3),
                        **fields,
                    )

            return server.tool(name=name)(logged)

        return register

    @logged_tool("haoliang_brief")
    def haoliang_brief(
        query: str,
        domain: str | None = None,
        domain_mode: str = "soft",
        max_terms: int = 12,
    ) -> dict:
        """Check the reader's term familiarity before answering this query."""
        return service.brief(
            query,
            domain=domain,
            domain_mode=domain_mode,
            max_terms=max_terms,
        )

    @logged_tool("haoliang_domain_profile")
    def haoliang_domain_profile(domain: str, max_terms: int = 12) -> dict:
        """Get domain-wide familiarity to set explanation depth."""
        return service.domain_profile(domain, max_terms=max_terms)

    @logged_tool("haoliang_annotate")
    def haoliang_annotate(
        text: str,
        domain: str | None = None,
        judge: bool = False,
        record_reviews: bool | None = None,
        record_local_terms: bool | None = None,
        project: str | None = None,
    ) -> dict:
        """Check a draft for undefined terms and internal jargon leaks.

        Write mode enables both record_* flags by default.
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

    @logged_tool("haoliang_revise")
    def haoliang_revise(
        text: str,
        domain: str | None = None,
        judge: bool = False,
        max_terms: int | None = None,
        record_reviews: bool | None = None,
        record_local_terms: bool | None = None,
        project: str | None = None,
    ) -> dict:
        """Define missing terms in one draft pass. Writes are opt-in."""
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

    @logged_tool("haoliang_observe")
    def haoliang_observe(
        term: str,
        domain: str,
        kind: str,
        quote: str = "",
        source: str = "",
        scope: str = "domain",
    ) -> dict:
        """Log observed term evidence with scope="domain".

        kind is asked, used, no_question, applied, explained, or corrected.
        """
        return service.observe(
            term,
            domain,
            kind,
            quote=quote,
            source=source,
            scope=scope,
        )

    @logged_tool("haoliang_propose")
    def haoliang_propose(
        term: str,
        domain: str,
        aliases: list[str] | None = None,
    ) -> dict:
        """Propose a domain term card without evidence."""
        return service.propose(term, domain, aliases=aliases)

    @logged_tool("haoliang_pending_list")
    def haoliang_pending_list() -> list[dict]:
        """List pending reviews with card IDs."""
        return service.pending_list()

    @logged_tool("haoliang_pending_resolve")
    def haoliang_pending_resolve(
        term_id: str,
        action: str,
        domain: str | None = None,
        reason: str | None = None,
        note: str = "",
        confirmed_by: str = "",
    ) -> dict:
        """Resolve a review or archive a card.

        For known or unknown, require explicit user confirmation and confirmed_by="user".
        """
        return service.pending_resolve(
            term_id,
            action,
            domain=domain,
            reason=reason,
            note=note,
            confirmed_by=confirmed_by,
        )

    @logged_tool("haoliang_glossary_add")
    def haoliang_glossary_add(
        term: str,
        scope: str,
        project: str,
        definition: str = "",
        reason: str = "",
    ) -> dict:
        """Add an internal glossary term with scope "project" or "model"."""
        return service.glossary_add(
            term,
            scope,
            project,
            definition=definition,
            reason=reason,
        )

    @logged_tool("haoliang_glossary_list")
    def haoliang_glossary_list(project: str | None = None) -> list[dict]:
        """List local glossary terms, optionally filtered by project."""
        return service.glossary_list(project)

    record_activity("server_start")
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
