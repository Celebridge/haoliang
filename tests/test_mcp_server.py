"""MCP server smoke test (skipped when mcp is not installed)."""

from __future__ import annotations

import json

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


def test_activity_records_calls_without_logging_reader_content(tmp_path):
    store = tmp_path / "reader"
    glossary = str(tmp_path / "glossary")
    writable = create_server(str(store), allow_write=True, glossary_root=glossary)
    tools = writable._tool_manager._tools
    sensitive = "private-test-text"

    tools["haoliang_observe"].fn(
        term=sensitive, domain="test", kind="used", quote=sensitive, source=sensitive,
    )
    tools["haoliang_brief"].fn(query=sensitive)
    tools["haoliang_annotate"].fn(text=sensitive, judge=False)
    with pytest.raises(ValueError):
        tools["haoliang_observe"].fn(term=sensitive, domain="test", kind=sensitive)

    read_only = create_server(str(store), glossary_root=glossary)
    with pytest.raises(PermissionError):
        read_only._tool_manager._tools["haoliang_propose"].fn(term=sensitive, domain="test")

    log = (store / "mcp-activity.jsonl").read_text(encoding="utf-8")
    entries = [json.loads(line) for line in log.splitlines()]
    starts = [entry for entry in entries if entry["event"] == "server_start"]
    assert [entry["write_enabled"] for entry in starts] == [True, False]
    calls = [entry for entry in entries if entry["event"] == "tool_call"]
    assert [(call["tool"], call["status"]) for call in calls] == [
        ("haoliang_observe", "ok"),
        ("haoliang_brief", "ok"),
        ("haoliang_annotate", "ok"),
        ("haoliang_observe", "error"),
        ("haoliang_propose", "denied"),
    ]
    assert calls[0]["evidence_written"] == 1
    assert calls[1]["matched_terms"] == 1
    assert calls[2]["reviews_written"] == calls[2]["local_terms_written"] == 0
    assert "evidence_written" not in calls[3]
    assert calls[3]["error_type"] == "ValueError"
    assert calls[4]["error_type"] == "PermissionError"
    assert all(call["duration_ms"] >= 0 for call in calls)
    assert sensitive not in log
    card = json.loads((store / "terms.jsonl").read_text(encoding="utf-8"))
    assert len(card["evidence"]) == 1


def test_unavailable_activity_log_does_not_block_tools(tmp_path, caplog):
    store = tmp_path / "reader"
    (store / "mcp-activity.jsonl").mkdir(parents=True)
    server = create_server(str(store), allow_write=True, glossary_root=str(tmp_path / "glossary"))
    result = server._tool_manager._tools["haoliang_observe"].fn(
        term="GC", domain="jvm", kind="used",
    )
    assert result["evidence_count"] == 1
    assert caplog.text.count("Haoliang MCP activity log unavailable") == 1
