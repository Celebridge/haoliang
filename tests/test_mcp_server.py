"""MCP server smoke test (skipped when mcp is not installed)."""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from haoliang.mcp_server import create_server


def test_create_server_registers_reader_tools(tmp_path):
    server = create_server(
        str(tmp_path / "reader"),
        allow_write=False,
        glossary_root=str(tmp_path / "glossary"),
    )

    assert server.name == "haoliang"
    tools = server._tool_manager._tools  # mcp 2.x internal registry
    assert "haoliang_brief" in tools
    assert "haoliang_domain_profile" in tools
    assert "haoliang_annotate" in tools
    assert "haoliang_pending_resolve" in tools
