"""SQLite 索引：结构化字段 + FTS5 全文检索。

为什么还要 SQLite：
- 向量检索负责「语义相近」，但关键词/人名/项目名用 FTS5 更准；
- SQLite 单文件、无需服务，适合 local-first；
- 以后可以在同一张库里加向量表或缓存。

当前只建三类表：
- cards：卡片的结构化字段
- cards_fts：FTS5 虚拟表，用于关键词检索
- edges：图的边，方便 SQL 查询
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .query import load_cards


def build_index(store_dir: str | Path) -> Path:
    """重建 index.sqlite，并返回数据库路径。

    每次 build 都删掉旧库重建：492 张卡的数据量很小，
    全量重建比做增量迁移简单、可靠。
    """
    store_dir = Path(store_dir)
    cards = load_cards(store_dir)

    edges_path = store_dir / "edges.jsonl"
    edges = (
        [
            json.loads(line)
            for line in edges_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if edges_path.exists()
        else []
    )

    db_path = store_dir / "index.sqlite"
    if db_path.exists():
        db_path.unlink()  # 删除旧文件

    # sqlite3.connect 返回 Connection；with 语句在这里不一定自动关闭，
    # 所以下面用 try/finally 显式 close()。
    conn = sqlite3.connect(db_path)
    try:
        # executescript 可以一次执行多条 SQL
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE cards(
              id TEXT PRIMARY KEY,
              section TEXT,
              type TEXT,
              thread TEXT,
              title TEXT,
              summary TEXT,
              text TEXT,
              sensitivity TEXT,
              status TEXT,
              importance INTEGER,
              updated_at TEXT
            );

            -- FTS5 是 SQLite 的全文检索扩展；id 不参与分词索引
            CREATE VIRTUAL TABLE cards_fts USING fts5(
              id UNINDEXED, title, summary, text
            );

            CREATE TABLE edges(
              src TEXT, dst TEXT, type TEXT, weight REAL,
              PRIMARY KEY(src, dst, type)
            );
            """
        )

        # executemany + 参数化查询：一次插入多行，避免手写 SQL 拼接
        conn.executemany(
            "INSERT INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    c["id"],
                    c.get("section"),
                    c.get("type"),
                    c.get("thread"),
                    c.get("title"),
                    c.get("summary"),
                    c.get("text"),
                    c.get("sensitivity"),
                    c.get("status"),
                    c.get("importance", 0),
                    c.get("updated_at"),
                )
                for c in cards
            ],
        )

        conn.executemany(
            "INSERT INTO cards_fts(id, title, summary, text) VALUES (?,?,?,?)",
            [
                (
                    c["id"],
                    c.get("title", ""),
                    c.get("summary", ""),
                    c.get("text", ""),
                )
                for c in cards
            ],
        )

        conn.executemany(
            "INSERT OR REPLACE INTO edges(src, dst, type, weight) VALUES (?,?,?,?)",
            [
                (e["src"], e["dst"], e["type"], e["weight"])
                for e in edges
            ],
        )

        conn.commit()
    finally:
        conn.close()

    return db_path


def search_fts(store_dir: str | Path, query: str, limit: int = 20) -> list[dict]:
    """用 FTS5 做关键词检索。

    注意：FTS5 的 MATCH 语法不是普通 SQL 的 LIKE。
    - 单个词：`父亲`
    - 多个词：`父亲 OR 母亲`
    - 短语：`"第一份工作"`
    """
    db_path = Path(store_dir) / "index.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT c.id, c.section, c.thread, c.title, c.summary, c.sensitivity, c.status
            FROM cards_fts f
            JOIN cards c ON c.id = f.id
            WHERE cards_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
    finally:
        conn.close()

    # sqlite3 默认返回 tuple；这里手动映射成 dict，调用方更好用
    keys = ["id", "section", "thread", "title", "summary", "sensitivity", "status"]
    return [dict(zip(keys, row)) for row in rows]
