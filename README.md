# 濠梁 · Haoliang

濠梁是一个本地优先的读者模型。它用术语证据记录你对概念的掌握程度，在 AI 回答之前提供与问题有关的词条，在回答之后标出读者可能需要解释的术语。

- 每张术语卡保存证据、所属领域和衰减参数，熟悉度在读取时按配置计算。
- `slice` 返回本题相关知识。领域优先用于术语消歧，独立的跨领域词仍可匹配。
- `annotate` 检查术语在首次正文出现时是否给出解释。可选的 `revise` 只在原文插入术语释义。
- CLI 与 MCP 使用同一个本地 JSONL 账本。MCP 默认开启只读模式。

## 安装与体验

需要 Python 3.10 或更高版本。在源码目录执行：

```bash
uv venv
uv pip install -e ".[mcp]"
source .venv/bin/activate

haoliang propose "安全点" --domain jvm --store var/reader-demo
haoliang slice "JVM 为什么要等线程到安全点" --domain jvm --store var/reader-demo
haoliang annotate "JVM 会等线程到安全点。" --domain jvm --store var/reader-demo
```

`propose` 新建待积累证据的词条。可以用 [Java 后端示例问卷](docs/onboarding/README.md) 建立自己的起始画像。词条、配置和默认层级的格式分别见 `schema/term.schema.json` 与 `config/reader.json`。

## 连接 Codex

在 Codex 的 `config.toml` 加入下面的 MCP 配置，替换绝对路径后开启新对话。

```toml
[mcp_servers.haoliang]
command = "/path/to/haoliang/.venv/bin/haoliang-mcp"
args = ["--store", "/path/to/haoliang/var/reader-demo"]
enabled_tools = ["haoliang_brief", "haoliang_domain_profile"]
```

`haoliang_brief` 用于回答前的一次问题相关查询。`haoliang_domain_profile` 用于查看一个领域的整体画像。

## 数据与模型调用

本地账本默认放在 `var/`，Git 忽略该目录。切片和不带模型判定的术语检查在本地运行。打开模型判定或调用独立改稿时，草稿会发给所配置的模型端点。这两种能力需要安装 `haoliang[openai]` 并配置对应的服务。

## 开发验证

```bash
uv pip install -e ".[dev,mcp]"
python -m pytest -q
python -m ruff check src tests eval
```

测试使用合成数据和模拟模型响应。仓库采用 [MIT 许可证](LICENSE)。
