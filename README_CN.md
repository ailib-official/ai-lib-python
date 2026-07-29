# ai-lib-python

**[AI-Protocol](https://github.com/ailib-official/ai-protocol) 协议运行时** — 异步 Python 参考实现（v**1.0.1**）。

[English](README.md)

`ai-lib-python` 是单包结构，在**模块层面**做执行层 / 策略层（E/P）分离。多数应用从根包导入：

```python
from ai_lib_python import AiClient, Message, StreamingEvent
```

## 工作原理

**默认聊天路径：** `AiClient` 加载 provider manifest → 按 manifest 算子构建 **`Pipeline`** → 经 **`HttpTransport`**（httpx）发 HTTP。流式帧归一为 **`StreamingEvent`**。

聊天路径是协议驱动的，但并非“零厂商代码”：仓库仍包含厂商解码器/映射、可选 **`ProviderDriver`**（高级 / 测试），以及 embeddings、STT、TTS、rerank 的独立 HTTP 客户端。

| 层 | 包 / 模块 | 职责 |
|----|-----------|------|
| 执行层 (E) | `client`、`protocol`、`pipeline`、`transport`、`types`、`structured`、可选能力模块 | 确定性执行、manifest 加载、HTTP |
| 策略层 (P) | `resilience`、`cache`、`routing`、`plugins`、`guardrails`、`batch`、`telemetry`、`tokens`、`registry` | 重试、限流、路由、遥测 — 在客户端旁按需接入 |
| 门面 | `ai_lib_python`（根包） | 稳定导入 + 示例 + 合规测试 |

已发布至 [PyPI](https://pypi.org/project/ai-lib-python/)：**`ai-lib-python` 1.0.1**。需要 Python **3.10+**。

> **说明：** Git `main` 可能包含尚未打进最近一次 PyPI 发版的协议 / 身份工作（例如 marketplace 别名解析）。请按目标 tag 锁定依赖版本；见 [CHANGELOG](CHANGELOG.md) 的 `Unreleased`。

## 快速开始

```bash
pip install ai-lib-python
export DEEPSEEK_API_KEY="your-key"
```

```python
import asyncio
from ai_lib_python import AiClient, Message

async def main() -> None:
    client = await AiClient.create("deepseek/deepseek-chat")

    response = await (
        client.chat()
        .messages([
            Message.system("You are a helpful assistant."),
            Message.user("Hello!"),
        ])
        .temperature(0.7)
        .max_tokens(500)
        .execute()
    )

    print(response.content)
    await client.close()

asyncio.run(main())
```

同一示例：`python examples/basic_chat.py`（需要 `OPENAI_API_KEY` 或改模型）。

流式构建器也支持在 `client.chat()` 上使用 `.system()` / `.user()` 简写。

### 流式

```python
import asyncio
from ai_lib_python import AiClient

async def main() -> None:
    client = await AiClient.create("deepseek/deepseek-chat")

    async for event in (
        client.chat()
        .user("Write a haiku about Python.")
        .stream()
    ):
        if event.is_content_delta:
            print(event.as_content_delta.content, end="", flush=True)
        elif event.is_stream_end:
            break

    await client.close()

asyncio.run(main())
```

同一示例：`python examples/streaming.py`。

### 调用统计

`ChatResponse` 不内嵌统计。请使用 `execute_with_stats()` 或 `stream_with_stats()`：

```python
response, stats = await client.chat().user("Hello!").execute_with_stats()
print(stats.latency_ms, stats.input_tokens, stats.output_tokens)
```

### 生产弹性（opt-in）

```python
client = await (
    AiClient.builder()
    .model("deepseek/deepseek-chat")
    .production_ready()  # ResilientConfig.production() 默认
    .build()
)
```

`production_ready()` 接入策略层 `resilience` 模块。仅调用 `AiClient.create()` **不会**启用。

## 公共 API（包根）

始终从 `ai_lib_python` 导出：

- **客户端：** `AiClient`、`AiClientBuilder`、`ChatResponse`、`CallStats`
- **类型：** `Message`、`MessageRole`、`MessageContent`、`ContentBlock`、`StreamingEvent`、`ToolCall`、`ToolDefinition`
- **错误：** `AiLibError`、`ProtocolError`、`TransportError`
- **特性探测：** `HAS_VISION`、`HAS_AUDIO`、`HAS_TELEMETRY`、`HAS_TOKENIZER`、`HAS_WATCHDOG`、`HAS_KEYRING`、`require_extra`
- **版本：** `__version__`（来自已安装发行版元数据）

子包（按需显式导入）：

- **执行层：** `ai_lib_python.pipeline`、`protocol`、`transport`、`structured`、`embeddings`、`stt`、`tts`、`rerank`、`multimodal`、`mcp`、`computer_use`
- **扩展类型：** `ai_lib_python.types` — `ExecutionResult`、`ExecutionMetadata`、`ExecutionUsage`，以及 text-tool / TTC（`StandardTextToolParser`、`ToolCallingPolicy`、`TextToolConfig` 等）
- **策略层：** `ai_lib_python.resilience`、`cache`、`routing`、`plugins`、`guardrails`、`batch`、`telemetry`、`tokens`、`registry`
- **高级：** `ai_lib_python.drivers` — `ProviderDriver`、`create_driver`（默认 `AiClient` 聊天路径不使用）

### extras 实际提供什么

| Extra | 获得什么 | 说明 |
|-------|----------|------|
| `vision` | 基于 Pillow 的图像块 | `HAS_VISION` |
| `audio` | 音频辅助（`soundfile`） | `HAS_AUDIO` |
| `embeddings` | `EmbeddingClient` | 协议化构建：`from_model` / `from_manifest`（无静默 OpenAI 主机默认） |
| `structured` | 结构化 / JSON 模式辅助 | 标记型 extra（代码本身可导入） |
| `stt` / `tts` / `reranking` | `SttClient`、`TtsClient`、`RerankerClient` | 独立服务客户端；rerank 支持 `from_model` / `from_manifest` |
| `batch` / `agentic` | 批处理 / agentic 标记 | 策略 / 能力标记 |
| `contact` | 策略层安装标记 | 路由、弹性、守卫、批处理、插件、遥测 — 物理拆包暂缓 |
| `telemetry` | OpenTelemetry sinks | `HAS_TELEMETRY`；反馈类型在 `telemetry` 子包 |
| `tokenizer` | Token 计数（tiktoken） | `HAS_TOKENIZER` |
| `full` | 全部能力 extras + `watchdog` + `keyring` | 含 `contact` |
| `dev` / `docs` / `jupyter` | 仅工具链 | pytest/mypy/ruff；mkdocs；ipywidgets |

```bash
pip install ai-lib-python[full]
```

多数 extras 是**标记**（依赖列表为空）：模块已在 wheel 中；安装 extra 用于显式能力契约，或在需要真实依赖时使用（`vision`、`audio`、`telemetry`、`tokenizer`、`full`）。

### 能力边界（如实说明）

| 区域 | 包内已有 | 不包含 |
|------|----------|--------|
| **MCP**（`mcp` 模块） | `McpToolBridge` 格式转换 | 接入 `AiClient` 的 MCP 服务端传输 |
| **Computer Use**（`computer_use`） | `ComputerAction`、`SafetyPolicy` 校验 | 截图 / 输入执行环境 |
| **热重载** | `AiClientBuilder.hot_reload()` 标志 + 内存缓存；`ProtocolLoader.clear_cache()` | 自动文件监视（需 `watchdog`；`HAS_WATCHDOG`）— 当前无自动重载 |
| **`ProviderDriver`** | 公开 `drivers` 模块 | 默认 `AiClient` 聊天路径 |
| **限流环境变量** | 经 `AiClientBuilder` / `resilience` 配置 | 运行时**不读取** `AI_LIB_RPS` / `AI_LIB_RPM` |

## 高级：`ProviderDriver`

`ai_lib_python.drivers` 提供 `ProviderDriver`、`create_driver` 以及 OpenAI / Anthropic / Gemini 驱动。**`AiClient` 聊天不走该路径**；它使用 manifest 构建的 `Pipeline`。驱动用于合规测试与自定义集成。

## 弹性

- **内置于 `AiClient`：** 可选 `max_inflight` 背压（builder 或 `AI_LIB_MAX_INFLIGHT`）。
- **策略层 opt-in：** `ai_lib_python.resilience`（重试、限流、熔断）— 使用 `production_ready()` 或显式 `ResilientConfig`。
- **`AiClient.create()` 默认不启用。**

## 协议 Manifest

解析顺序：

1. `AiClientBuilder.protocol_path(...)` / `ProtocolLoader(base_path=...)`
2. `AI_PROTOCOL_DIR` / `AI_PROTOCOL_PATH`（本地目录；远程加载支持 GitHub raw URL）
3. 开发路径：`ai-protocol/`、`../ai-protocol/`、…
4. 兜底：GitHub raw `ailib-official/ai-protocol`（`main`）

每个 base path：`dist/v2/providers/<id>.json` → `dist/v1/providers/<id>.json` → 源码树 `v2` / `v1` YAML/JSON 降级。

**身份 / 别名（在 `main` 上，见 Unreleased）：** `load_provider` 通过 `dist/provider-identity.json`（多 family 映射）解析 marketplace 别名，例如 `google` → `gemini`、`kimi` → `moonshot`。解析/校验错误不会被别名查找掩盖。

Manifest 缓存：仅内存。`hot_reload=True` 只保存标志，**不监视文件** — 变更后请调用 `ProtocolLoader.clear_cache()` 或重建客户端。

## API 密钥（BYOK 链）

1. builder / `AiClient.create(..., api_key=...)` 显式覆盖
2. Manifest 声明的环境变量（`endpoint.auth` / 顶层 `auth`）
3. `<PROVIDER_ID>_API_KEY`（CI/容器推荐）
4. 已安装 `keyring` 时的操作系统密钥环（`HAS_KEYRING`；含于 `[full]`）

## 环境变量

| 变量 | 用途 |
|------|------|
| `AI_PROTOCOL_DIR` / `AI_PROTOCOL_PATH` | 本地 manifest 目录或 GitHub raw 基址 |
| `AI_PROXY_URL` | 显式代理（需 `AI_HTTP_TRUST_ENV=1`） |
| `HTTP_PROXY` / `HTTPS_PROXY` | 标准代理变量（需 `AI_HTTP_TRUST_ENV=1`） |
| `NO_PROXY` / `AI_PROXY_NO_PROXY` | 不走代理的主机 |
| `AI_HTTP_TIMEOUT_SECS` | HTTP 超时 |
| `AI_LIB_MAX_INFLIGHT` | 并发背压（也可经 builder） |

跨运行时代理语义：[CROSS_RUNTIME.md](https://github.com/ailib-official/ai-protocol/blob/main/docs/CROSS_RUNTIME.md)。

## 标准错误码（V2）

| 码 | 名称 | 可重试 | 可回退 |
|----|------|--------|--------|
| E1001 | `invalid_request` | 否 | 否 |
| E1002 | `authentication` | 否 | 是 |
| E1003 | `permission_denied` | 否 | 否 |
| E1004 | `not_found` | 否 | 否 |
| E1005 | `request_too_large` | 否 | 否 |
| E2001 | `rate_limited` | 是 | 是 |
| E2002 | `quota_exhausted` | 否 | 是 |
| E3001 | `server_error` | 是 | 是 |
| E3002 | `overloaded` | 是 | 是 |
| E3003 | `timeout` | 是 | 是 |
| E4001 | `conflict` | 是 | 否 |
| E4002 | `cancelled` | 否 | 否 |
| E9999 | `unknown` | 否 | 否 |

## 测试

```bash
pip install -e ".[dev]"
pytest tests/unit/ -v
```

合规（跨运行时 YAML）：

```bash
# POSIX
COMPLIANCE_DIR=../ai-protocol/tests/compliance pytest tests/compliance/ -v

# Windows PowerShell
$env:COMPLIANCE_DIR = "D:\ai-protocol\tests\compliance"
pytest tests/compliance/ -v
```

Mock 集成（需要 [ai-protocol-mock](https://github.com/ailib-official/ai-protocol-mock)）：

```bash
MOCK_HTTP_URL=http://localhost:4010 pytest tests/integration/ -v
```

## 示例

| 示例 | 主题 |
|------|------|
| `basic_chat.py` | 快速开始、`execute_with_stats` |
| `streaming.py` | `is_content_delta`、`stream_with_stats` |
| `resilience.py` | 策略层 |
| `multimodal.py` | Vision extra |
| `tool_calling.py` | Tools |
| `multi_provider_production.py` | 路由 / 回退 |
| `guardrails_production.py` | Guardrails |
| `concurrent_production.py` | 并发 |
| `providers.py` | Provider 切换 |

## 相关

- [AI-Protocol](https://github.com/ailib-official/ai-protocol) — 规范与 manifests
- [ai-lib-rust](https://github.com/ailib-official/ai-lib-rust) — Rust 运行时
- [ai-lib-ts](https://github.com/ailib-official/ai-lib-ts) — TypeScript 运行时
- [ai-lib-go](https://github.com/ailib-official/ai-lib-go) — Go 运行时

## 许可证

双许可：[Apache-2.0](LICENSE-APACHE) 或 [MIT](LICENSE-MIT)。
