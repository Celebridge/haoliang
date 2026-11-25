"""term CLI 的端到端测试：observe → propose → confirm → terms。"""

from __future__ import annotations

import json
from pathlib import Path

from haoliang.cli import main


def _read_terms(store: Path) -> list[dict]:
    path = store / "terms.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_cli_observe_creates_and_appends(tmp_path: Path):
    store = tmp_path / "reader"

    code = main(
        [
            "observe",
            "safepoint",
            "--domain",
            "jvm",
            "--kind",
            "used",
            "--time",
            "2026-11-16",
            "--store",
            str(store),
        ]
    )
    assert code == 0

    terms = _read_terms(store)
    assert len(terms) == 1
    assert terms[0]["term"] == "safepoint"
    assert terms[0]["evidence"][0]["kind"] == "used"

    code = main(
        [
            "observe",
            "safepoint",
            "--domain",
            "jvm",
            "--kind",
            "explained",
            "--time",
            "2026-11-17",
            "--store",
            str(store),
        ]
    )
    assert code == 0

    terms = _read_terms(store)
    assert len(terms) == 1
    assert len(terms[0]["evidence"]) == 2


def test_cli_propose_creates_proposal(tmp_path: Path):
    store = tmp_path / "reader"

    code = main(["propose", "safepoint", "--domain", "jvm", "--store", str(store)])

    assert code == 0
    terms = _read_terms(store)
    assert len(terms) == 1
    assert terms[0]["status"] == "proposal"
    assert terms[0]["evidence"] == []


def test_cli_confirm_rejects_insufficient_evidence(tmp_path: Path, capsys):
    store = tmp_path / "reader"
    main(
        [
            "observe",
            "safepoint",
            "--domain",
            "jvm",
            "--kind",
            "self_report",
            "--time",
            "2026-11-16",
            "--store",
            str(store),
        ]
    )

    code = main(["confirm", "t_0001", "--store", str(store)])

    assert code == 1
    assert "cannot promote" in capsys.readouterr().err


def test_cli_confirm_promotes_with_enough_evidence(tmp_path: Path):
    store = tmp_path / "reader"
    base = [
        "observe",
        "safepoint",
        "--domain",
        "jvm",
        "--store",
        str(store),
    ]

    assert main(base + ["--kind", "used", "--time", "2026-11-16"]) == 0
    assert main(base + ["--kind", "used", "--time", "2026-11-17"]) == 0
    assert main(base + ["--kind", "explained", "--time", "2026-11-17"]) == 0
    assert main(["confirm", "t_0001", "--store", str(store)]) == 0

    terms = _read_terms(store)
    assert terms[0]["status"] == "canon"


def test_cli_terms_lists_and_filters(tmp_path: Path, capsys):
    store = tmp_path / "reader"
    main(["observe", "safepoint", "--domain", "jvm", "--kind", "used", "--store", str(store)])
    main(["observe", "etcd", "--domain", "k8s", "--kind", "asked", "--store", str(store)])
    capsys.readouterr()  # 丢弃 setup 的输出

    code = main(["terms", "--domain", "jvm", "--json", "--store", str(store)])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["term"] for item in payload] == ["safepoint"]


def test_cli_show_returns_json(tmp_path: Path, capsys):
    store = tmp_path / "reader"
    main(["observe", "safepoint", "--domain", "jvm", "--kind", "used", "--store", str(store)])
    capsys.readouterr()  # 丢弃 setup 的输出

    code = main(["show", "t_0001", "--json", "--store", str(store)])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["term"] == "safepoint"
    assert payload["familiarity"]["id"] in {"heard", "usable", "explainable"}
