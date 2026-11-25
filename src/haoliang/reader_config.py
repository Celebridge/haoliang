"""读者模型配置。

层级不是代码里的枚举，而是配置里的阈值：

- 卡片只存连续掌握度 `knowledge`（0–1），不存「第几级」。
- `Level.min_knowledge` 是该层级的门槛。
- 加层级、改名字、调阈值都只改 `config/reader.json`，不改代码。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Editable installs read the checkout file; wheels carry the same config.
_BUNDLED_CONFIG_PATH = Path(__file__).with_name("_default_reader.json")
DEFAULT_CONFIG_PATH = (
    _BUNDLED_CONFIG_PATH
    if _BUNDLED_CONFIG_PATH.is_file()
    else Path(__file__).resolve().parents[2] / "config" / "reader.json"
)


@dataclass(frozen=True)
class Level:
    """一个 familiarity 层级。

    `min_knowledge` 是进入该层级的最低掌握度。它不是序号：
    两个层级之间可以插入新层级，只要阈值仍然递增。
    """

    id: str
    label: str
    min_knowledge: float
    explain: str


@dataclass(frozen=True)
class PromotionRules:
    """proposal → canon 的升级条件。"""

    min_positive_evidence: int
    min_distinct_days: int
    min_strong_evidence: int
    strong_weight: float
    negative_cooldown_days: int


@dataclass(frozen=True)
class DecayRules:
    """遗忘曲线参数。"""

    initial_stability_days: float
    growth_factor: float
    shrink_factor: float
    retention_threshold: float
    min_stability_days: float = 0.5
    baseline_stability_days: float = 45.0


@dataclass(frozen=True)
class RetrievalRules:
    """检索排序参数。"""

    domain_boost: float = 0.10
    domain_confidence_terms: float = 5.0


@dataclass(frozen=True)
class RevisionRules:
    """revision 的预算与 editor 连接参数。

    预算属于策略，和层级阈值一样放在配置里：

    - `max_terms`：一次最多补几个词；
    - `max_revisions_per_turn`：harness 每轮最多跑几次 revision；
    - `max_growth_chars`：改稿最多比原稿长多少字符；
    """

    max_terms: int = 5
    max_revisions_per_turn: int = 1
    max_growth_chars: int = 240
    min_similarity: float = 0.6  # Legacy config field; definition insertion does not use it.
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.1
    max_tokens: int = 1600


@dataclass(frozen=True)
class JargonRules:
    """`not_in_ledger` 术语判断的模型配置和预算。"""

    enabled: bool = False
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 2400
    max_candidates: int = 50
    confidence_threshold: float = 0.6
    review_as: str = "skip"
    reader_profile: str = ""


@dataclass(frozen=True)
class ReaderConfig:
    levels: tuple[Level, ...]
    evidence_weights: dict[str, float]
    promotion: PromotionRules
    decay: DecayRules
    retrieval: RetrievalRules = field(default_factory=RetrievalRules)
    revision: RevisionRules = field(default_factory=RevisionRules)
    jargon: JargonRules = field(default_factory=JargonRules)

    def __post_init__(self) -> None:
        # A ReaderConfig may be built directly in tests or by an embedder.
        # Normalize and validate here so level_of() never depends on caller
        # ordering or on a non-empty tuple.
        object.__setattr__(
            self,
            "levels",
            tuple(sorted(self.levels, key=lambda level: level.min_knowledge)),
        )
        _validate_levels(self.levels)

    def level_of(self, knowledge: float) -> Level:
        """返回 knowledge 命中的最高层级。"""
        current = self.levels[0]
        for level in self.levels:
            if knowledge >= level.min_knowledge:
                current = level
            else:
                break
        return current


def load_config(path: str | Path | None = None) -> ReaderConfig:
    """从指定 JSON 读取配置，未指定时使用项目默认配置。"""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data = json.loads(config_path.read_text(encoding="utf-8"))

    levels = tuple(
        Level(
            id=str(item["id"]),
            label=str(item["label"]),
            min_knowledge=float(item["min_knowledge"]),
            explain=str(item["explain"]),
        )
        for item in data["levels"]
    )
    # 不依赖 JSON 里的顺序；只依赖 min_knowledge。
    levels = tuple(sorted(levels, key=lambda level: level.min_knowledge))
    _validate_levels(levels)

    evidence_weights = {
        str(kind): float(weight)
        for kind, weight in data["evidence_weights"].items()
    }

    promotion_data = data["promotion"]
    promotion = PromotionRules(
        min_positive_evidence=int(promotion_data["min_positive_evidence"]),
        min_distinct_days=int(promotion_data["min_distinct_days"]),
        min_strong_evidence=int(promotion_data["min_strong_evidence"]),
        strong_weight=float(promotion_data["strong_weight"]),
        negative_cooldown_days=int(promotion_data["negative_cooldown_days"]),
    )

    decay_data = data["decay"]
    decay = DecayRules(
        initial_stability_days=float(decay_data["initial_stability_days"]),
        growth_factor=float(decay_data["growth_factor"]),
        shrink_factor=float(decay_data["shrink_factor"]),
        retention_threshold=float(decay_data["retention_threshold"]),
        min_stability_days=float(decay_data.get("min_stability_days", 0.5)),
        baseline_stability_days=float(
            decay_data.get("baseline_stability_days", 45.0)
        ),
    )

    retrieval_data = data.get("retrieval", {})
    retrieval = RetrievalRules(
        domain_boost=float(retrieval_data.get("domain_boost", 0.10)),
        domain_confidence_terms=float(
            retrieval_data.get("domain_confidence_terms", 5.0)
        ),
    )

    revision_data = data.get("revision", {})
    revision = RevisionRules(
        max_terms=int(revision_data.get("max_terms", 5)),
        max_revisions_per_turn=int(
            revision_data.get("max_revisions_per_turn", 1)
        ),
        max_growth_chars=int(revision_data.get("max_growth_chars", 240)),
        min_similarity=float(revision_data.get("min_similarity", 0.6)),
        model=str(revision_data.get("model", "deepseek-chat")),
        base_url=str(revision_data.get("base_url", "https://api.deepseek.com")),
        api_key_env=str(revision_data.get("api_key_env", "DEEPSEEK_API_KEY")),
        temperature=float(revision_data.get("temperature", 0.1)),
        max_tokens=int(revision_data.get("max_tokens", 1600)),
    )

    jargon_data = data.get("jargon", {})
    review_as = str(jargon_data.get("review_as", "skip"))
    if review_as not in {"skip", "define"}:
        raise ValueError("jargon.review_as must be 'skip' or 'define'")
    jargon = JargonRules(
        enabled=bool(jargon_data.get("enabled", False)),
        model=str(jargon_data.get("model", "deepseek-chat")),
        base_url=str(jargon_data.get("base_url", "https://api.deepseek.com")),
        api_key_env=str(jargon_data.get("api_key_env", "DEEPSEEK_API_KEY")),
        temperature=float(jargon_data.get("temperature", 0.0)),
        max_tokens=int(jargon_data.get("max_tokens", 2400)),
        max_candidates=int(jargon_data.get("max_candidates", 50)),
        confidence_threshold=float(
            jargon_data.get("confidence_threshold", 0.6)
        ),
        review_as=review_as,
        reader_profile=str(jargon_data.get("reader_profile", "")),
    )

    return ReaderConfig(
        levels,
        evidence_weights,
        promotion,
        decay,
        retrieval,
        revision,
        jargon,
    )


def _validate_levels(levels: tuple[Level, ...]) -> None:
    if not levels:
        raise ValueError("config must define at least one level")
    if levels[0].min_knowledge != 0.0:
        raise ValueError("the lowest level must start at min_knowledge=0.0")

    seen: set[str] = set()
    previous = -1.0
    for level in levels:
        if not level.id:
            raise ValueError("level id must not be empty")
        if level.id in seen:
            raise ValueError(f"duplicate level id: {level.id}")
        if not 0.0 <= level.min_knowledge <= 1.0:
            raise ValueError(f"level {level.id}: min_knowledge must be in [0, 1]")
        if level.min_knowledge <= previous:
            raise ValueError("levels must have strictly increasing min_knowledge")
        if level.explain not in {"define", "define_brief", "none", "reason"}:
            raise ValueError(
                f"level {level.id}: explain must be define/define_brief/none/reason"
            )
        seen.add(level.id)
        previous = level.min_knowledge
