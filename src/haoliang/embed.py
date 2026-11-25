"""Embedding 后端，以及把卡片批量转成向量。

三种后端：
- HashEmbedder：不依赖网络/模型，只用字符 n-gram 哈希。适合离线测试，不是语义模型。
- OpenAIEmbedder：调用 OpenAI embeddings API（生产可用）。
- BGEEmbedder：本地 sentence-transformers + BGE-M3（生产可用，需额外依赖）。

向量统一做 L2 归一化，这样余弦相似度可以直接用点积计算。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Embedder 的「接口」。

    Python 没有 Java 的 interface 关键字，但 `Protocol` 可以做结构化类型：
    只要一个类有 name/model/dim 三个属性和 embed() 方法，就算 Embedder。
    这里用它来给三个实现类一个共同的类型标注。
    """

    name: str
    model: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """确定性的、无依赖的兜底实现。

    原理：把文本拆成 1/2/3 字符的 n-gram，对每个 n-gram 做哈希，
    落到固定维度的桶里，再按哈希位决定 +1/-1。最后 L2 归一化。

    它不是语义模型：只是让相同字符片段有相似向量，
    用来跑通流程和做单元测试。真实检索请用 openai 或 bge。
    """

    name = "hash"

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model = f"hash-char-ngram-{dim}"

    def _one(self, text: str) -> np.ndarray:
        """把一段文本转成一个 dim 维向量。"""
        vec = np.zeros(self.dim, dtype=np.float32)

        # 去掉所有空白（中文分词意义不大，先按字符处理）
        text = "".join(text.lower().split())
        chars = list(text)

        # 生成 1-gram、2-gram、3-gram
        ngrams: list[str] = []
        for n in (1, 2, 3):
            ngrams.extend(
                "".join(chars[i : i + n])
                for i in range(max(0, len(chars) - n + 1))
            )

        if not ngrams:
            return vec

        for gram in ngrams:
            # blake2b 比 md5/sha1 快，digest_size=8 足够做桶索引
            h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h, "big") % self.dim

            # 用哈希最低位做符号，减少不同 n-gram 挤在同一桶时的偏差
            sign = 1.0 if (h[0] & 1) else -1.0
            vec[idx] += sign

        norm = float(np.linalg.norm(vec))
        if norm:
            vec /= norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        """批量接口：返回 shape=(len(texts), dim) 的矩阵。"""
        if not texts:
            # 空输入也要返回正确的二维形状，避免上层报 index error
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._one(t) for t in texts])


class OpenAIEmbedder:
    """OpenAI / OpenAI-compatible embeddings 后端。

    需要安装可选依赖：`pip install "haoliang[openai]"`。
    默认读 `OPENAI_API_KEY`；也可以指定其他环境变量（例如火山方舟的 `ARK_API_KEY`），
    并通过 `base_url` 指向兼容接口（例如 `https://ark.cn-beijing.volces.com/api/v3`）。
    """

    name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        dim: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        dimensions: int | None = None,
        batch_size: int = 10,
        request_interval: float = 0.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 可选依赖
            raise RuntimeError(
                "Install haoliang[openai] to use OpenAI embeddings"
            ) from exc

        # 优先用显式传入的 key，否则从环境变量读；不要把 key 写进代码/日志
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(
                f"{api_key_env} is not set; export it or pass --api-key-env"
            )

        # base_url=None 时使用 OpenAI 默认地址；Ark 等兼容服务传入自己的 /v3 地址
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.dim = dim or 3072
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.dimensions = dimensions
        self.batch_size = max(1, int(batch_size))
        self.request_interval = max(0.0, float(request_interval))
        # 单条 query 的缓存：评测时 vector/graph 两种模式会重复编码同一个 query
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        # 单条文本（通常是 query）命中缓存时直接返回，避免重复 API 调用
        if len(texts) == 1:
            cached = self._cache.get(texts[0])
            if cached is not None:
                return cached.reshape(1, -1)

        # Ark 等兼容接口对单次请求的条数/总 token 有限制，分批发送。
        # 每批保留原始顺序，最后按批次拼接。
        all_vectors: list[list[float]] = []
        batches = list(range(0, len(texts), self.batch_size))

        for batch_no, start in enumerate(batches):
            batch = texts[start : start + self.batch_size]

            kwargs: dict = {"model": self.model, "input": batch}
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions

            # 429 是订阅套餐常见的限流；指数退避后重试同一批
            for attempt in range(5):
                try:
                    resp = self.client.embeddings.create(**kwargs)
                    break
                except Exception as exc:
                    message = str(exc).lower()
                    if "429" not in message and "rate" not in message:
                        raise
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)

            # API 返回的顺序不保证和输入一致；按 item.index 排回去
            vectors = [
                item.embedding
                for item in sorted(resp.data, key=lambda x: x.index)
            ]
            all_vectors.extend(vectors)

            # 主动放慢节奏，避免下一批立刻撞上限流
            if self.request_interval and batch_no < len(batches) - 1:
                time.sleep(self.request_interval)

        arr = np.asarray(all_vectors, dtype=np.float32)

        # 用实际返回维度更新 self.dim，避免不同模型维度写死
        self.dim = int(arr.shape[1])

        # L2 归一化：让余弦相似度 = 点积，后续计算更简单
        # axis=1 表示「每一行」；keepdims=True 保持 (n,1) 形状，便于广播
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # 避免除零
        arr = arr / norms

        if len(texts) == 1:
            self._cache[texts[0]] = arr[0].copy()

        return arr


class BGEEmbedder:
    """本地 BGE-M3 后端。

    需要安装可选依赖：`pip install "haoliang[local]"`。
    首次运行会从 Hugging Face 下载模型，之后走本地缓存。
    """

    name = "bge"

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 10,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - 可选依赖
            raise RuntimeError(
                "Install haoliang[local] to use BGE embeddings"
            ) from exc

        self.model = model
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self._model = SentenceTransformer(model, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        arr = self._model.encode(
            texts,
            normalize_embeddings=True,  # 内部做 L2 归一化
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        return np.asarray(arr, dtype=np.float32)


def make_embedder(
    name: str = "hash",
    model: str | None = None,
    dim: int = 256,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    dimensions: int | None = None,
    device: str | None = None,
    batch_size: int = 10,
    request_interval: float = 0.0,
) -> Embedder:
    """工厂函数：按名字创建 embedder。类似 Java 的简单 Factory。"""
    name = name.lower()

    if name == "hash":
        return HashEmbedder(dim=dim)
    if name == "openai":
        return OpenAIEmbedder(
            model=model or "text-embedding-3-large",
            base_url=base_url,
            api_key_env=api_key_env,
            dimensions=dimensions,
            batch_size=batch_size,
            request_interval=request_interval,
        )
    if name in {"bge", "bge-m3", "local"}:
        return BGEEmbedder(
            model=model or "BAAI/bge-m3",
            device=device,
            batch_size=batch_size,
        )

    raise ValueError(f"unknown embedder: {name}")


def embed_cards(
    cards: list[dict],
    embedder: Embedder,
    out_dir: str | Path,
) -> dict:
    """把每张卡转成向量，保存为 embeddings.npy，并写 embedding_meta.json。

    .npy 是 numpy 的二进制矩阵格式；meta 里记录 provider/model/dim/ids，
    检索时必须用同一个模型重新编码 query，否则向量空间对不上。
    """
    texts = [card_text(card) for card in cards]
    matrix = embedder.embed(texts)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", matrix)

    meta = {
        "provider": embedder.name,
        "model": embedder.model,
        # matrix.ndim == 2 表示正常的二维矩阵；空输入时用 embedder.dim 兜底
        "dim": int(matrix.shape[1]) if matrix.ndim == 2 else embedder.dim,
        "count": int(matrix.shape[0]),
        "ids": [card["id"] for card in cards],
    }

    # 记录 query 时重建 embedder 所需的非敏感配置（不要写 key 本身）
    for key in (
        "base_url",
        "api_key_env",
        "dimensions",
        "device",
        "batch_size",
        "request_interval",
    ):
        value = getattr(embedder, key, None)
        if value is not None:
            meta[key] = value
    (out_dir / "embedding_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta


def card_text(card: dict) -> str:
    """决定一张卡用哪些文本去算 embedding。

    目前用：问题 + 摘要 + 前 4 条 claim。
    不直接把整段原文全塞进去，是为了控制输入长度，也避免长答案稀释主题。
    """
    parts = [card.get("question", ""), card.get("summary", "")]
    parts.extend(c.get("text", "") for c in card.get("claims", [])[:4])
    return "\n".join(p for p in parts if p)


def load_embeddings(store_dir: str | Path) -> tuple[np.ndarray, dict]:
    """从 store 目录加载向量矩阵和元信息。"""
    store_dir = Path(store_dir)
    matrix = np.load(store_dir / "embeddings.npy")
    meta = json.loads((store_dir / "embedding_meta.json").read_text(encoding="utf-8"))
    return matrix.astype(np.float32), meta


def cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """计算 query 与矩阵中每一行的余弦相似度。

    `@` 是 numpy 的矩阵乘法运算符（等价于 np.matmul）。
    matrix 形状 (n, d)，q 形状 (d,)，结果是 (n,)。
    """
    q = query.astype(np.float32)
    norm = float(np.linalg.norm(q))
    if norm:
        q = q / norm

    m = matrix.astype(np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms) @ q
