# 濠梁 · Haoliang

**让 AI 从你已经懂的地方讲起。**

AI 解释技术问题时总在堆术语，让人觉得它「不说人话」？濠梁记录你对技术术语的掌握程度，让助手在回答前知道哪些概念可以直接用，哪些需要先解释。它开源、本地优先，可以通过模型上下文协议（MCP）接入 Codex。

Haoliang is an open-source, local-first MCP server for personalized technical explanations. It gives AI assistants a knowledge profile across chats, so they can start from what you know and flag undefined jargon.

[看个例子](#濠梁能解决什么问题) · [试一下](#安装与体验) · [接入 Codex](#连接-codex)

## 濠梁能解决什么问题

假设你熟悉垃圾回收（GC），但没学过安全点。你问「为什么 GC 停顿要等线程到达 safepoint？」

- **直接回答示例**　“GC 暂停应用线程前，要等线程到达 safepoint。”
- **参考读者画像的回答示例**　“safepoint（安全点）是线程可以被安全暂停的位置。垃圾回收要暂停线程时，还在运行的线程得先走到这样的位置，所以会有一段等待。”

这个例子里，GC 可以当作已知前提，safepoint 需要先解释。濠梁也能在草稿写完后标出漏讲的术语，供助手按需补一句。掌握度保存在本地账本，开新对话也可以继续用。

## 工作方式

- 每张术语卡保存证据、所属领域和衰减参数，熟悉度在读取时按配置计算。
- 回答前，`slice` 只返回本题相关词条。领域用于优先匹配同名词的不同含义。
- `annotate` 检查术语在首次正文出现时是否给出解释。可选的 `revise` 只在原文插入术语释义。
- CLI 与 MCP 使用同一个本地 JSONL 账本。MCP 默认开启只读模式。

## 安装与体验

需要 Python 3.10 或更高版本和 `uv`。下面用一个独立的演示账本复现上面的 GC 例子，无需模型 API 密钥。

```bash
git clone https://github.com/Celebridge/haoliang.git
cd haoliang
uv venv
uv pip install -e ".[mcp]"
source .venv/bin/activate

haoliang observe GC --kind explained --domain jvm --store var/reader-demo
haoliang propose safepoint --domain jvm --store var/reader-demo
haoliang slice "为什么 GC 停顿要等线程到达 safepoint？" --domain jvm --store var/reader-demo
```

首次运行，最后一条命令会给出这条建议：

```text
先定义：safepoint；可以直接用：GC
```

演示把 GC 设为「会用」，给 safepoint 建一张还没积累证据的卡。想建立自己的画像，可以用 [Java 后端示例问卷](docs/onboarding/README.md) 起步。卡片格式与默认熟悉度层级见 `schema/term.schema.json` 和 `config/reader.json`。

## 连接 Codex

在 Codex 的 `~/.codex/config.toml` 加入下面的配置，把示例路径替换成克隆目录的绝对路径。重启 Codex，在新对话中输入 `/mcp`，确认 `haoliang` 已连接。 [Codex MCP 配置文档](https://developers.openai.com/codex/mcp)

```toml
[mcp_servers.haoliang]
command = "/path/to/haoliang/.venv/bin/haoliang-mcp"
args = ["--store", "/path/to/haoliang/var/reader-demo"]
enabled_tools = ["haoliang_brief", "haoliang_domain_profile"]
```

在新对话中用下面这句话试一次。看到 `haoliang_brief` 被调用，就可以检查它是否将 GC 和 safepoint 分别归入已知和需解释的词。

> 先用 haoliang_brief 看我对本题术语的掌握程度，再解释为什么 GC 停顿要等线程到达 safepoint。

`haoliang_domain_profile` 用于查看一个领域的整体画像。想让助手平时也先查画像，可以把上面的调用要求写入 Codex 的个人指令。

## 数据与模型调用

本地账本默认放在 `var/`，Git 忽略该目录。切片和不带模型判定的术语检查在本地运行。打开模型判定或调用独立改稿时，草稿会发给所配置的模型端点。这两种能力需要安装 `haoliang[openai]` 并配置对应的服务。

## 开发验证

```bash
uv pip install -e ".[dev,mcp]"
python -m pytest -q
python -m ruff check src tests eval
```

测试使用合成数据和模拟模型响应。仓库采用 [MIT 许可证](LICENSE)。
