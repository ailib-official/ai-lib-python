# ai-lib-python

**Protocol runtime for [AI-Protocol](https://github.com/ailib-official/ai-protocol)** — async Python reference implementation (v**1.2.0**).

[中文文档](README_CN.md)

`ai-lib-python` is a single package with **execution / policy module separation** (E/P at the module level). Most apps import from the root:

```python
from ai_lib_python import AiClient, Message, StreamingEvent
```

## How it works

**Default chat path:** `AiClient` loads a provider manifest → builds a **`Pipeline`** from manifest operators → sends HTTP via **`HttpTransport`** (httpx). Streaming frames are normalized to **`StreamingEvent`**.

This is protocol-driven for chat, but not “zero provider code”: the repo also ships provider-specific decoders/mappers, optional **`ProviderDriver`** implementations (advanced / tests), and standalone HTTP clients for embeddings, STT, TTS, and rerank.

| Layer | Packages / modules | Responsibility |
|-------|-------------------|----------------|
| Execution (E) | `client`, `protocol`, `pipeline`, `transport`, `types`, `structured`, optional capability modules | Deterministic execution, manifest loading, HTTP |
| Policy (P) | `resilience`, `cache`, `routing`, `plugins`, `guardrails`, `batch`, `telemetry`, `tokens`, `registry` | Retry, rate limits, routing, telemetry — opt-in beside the client |
| Facade | `ai_lib_python` (root) | Stable imports + examples + compliance tests |

Published on [PyPI](https://pypi.org/project/ai-lib-python/): **`ai-lib-python` 1.2.0**. Python **3.10+**.

> **Note:** Git `main` may include protocol/identity work landed after the last PyPI cut (e.g. marketplace alias resolve). Match dependency versions to the tag you intend; see [CHANGELOG](CHANGELOG.md) `Unreleased`.

## Quick start

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

Same example: `python examples/basic_chat.py` (requires `OPENAI_API_KEY` or change the model).

The fluent builder also supports `.system()` / `.user()` shorthands on `client.chat()`.

### Streaming

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

Same example: `python examples/streaming.py`.

### Call statistics

`ChatResponse` does not embed stats. Use `execute_with_stats()` or `stream_with_stats()`:

```python
response, stats = await client.chat().user("Hello!").execute_with_stats()
print(stats.latency_ms, stats.input_tokens, stats.output_tokens)
```

### Production resilience (opt-in)

```python
client = await (
    AiClient.builder()
    .model("deepseek/deepseek-chat")
    .production_ready()  # ResilientConfig.production() defaults
    .build()
)
```

`production_ready()` wires the policy-layer `resilience` module. It is **not** enabled by `AiClient.create()` alone.

## Public API (package root)

Always exported from `ai_lib_python`:

- **Client:** `AiClient`, `AiClientBuilder`, `ChatResponse`, `CallStats`
- **Types:** `Message`, `MessageRole`, `MessageContent`, `ContentBlock`, `StreamingEvent`, `ToolCall`, `ToolDefinition`
- **Errors:** `AiLibError`, `ProtocolError`, `TransportError`
- **Feature probes:** `HAS_VISION`, `HAS_AUDIO`, `HAS_TELEMETRY`, `HAS_TOKENIZER`, `HAS_WATCHDOG`, `HAS_KEYRING`, `require_extra`
- **Version:** `__version__` (from installed distribution metadata)

Subpackages (import explicitly when needed):

- **Execution:** `ai_lib_python.pipeline`, `protocol`, `transport`, `structured`, `embeddings`, `stt`, `tts`, `rerank`, `multimodal`, `mcp`, `computer_use`
- **Types (extended):** `ai_lib_python.types` — `ExecutionResult`, `ExecutionMetadata`, `ExecutionUsage`, text-tool / TTC (`StandardTextToolParser`, `ToolCallingPolicy`, `TextToolConfig`, …)
- **Policy:** `ai_lib_python.resilience`, `cache`, `routing`, `plugins`, `guardrails`, `batch`, `telemetry`, `tokens`, `registry`
- **Advanced:** `ai_lib_python.drivers` — `ProviderDriver`, `create_driver` (not used by default `AiClient` chat path)

### What extras actually do

| Extra | What you get | Notes |
|-------|--------------|-------|
| `vision` | Pillow-backed image blocks | Marked via `HAS_VISION` |
| `audio` | Audio helpers (`soundfile`) | `HAS_AUDIO` |
| `embeddings` | `EmbeddingClient` | Protocolized builders: `from_model` / `from_manifest` (no silent OpenAI host default) |
| `structured` | Structured / JSON mode helpers | Marker extra (code always importable) |
| `stt` / `tts` / `reranking` | `SttClient`, `TtsClient`, `RerankerClient` | Standalone service clients; rerank supports `from_model` / `from_manifest` |
| `batch` / `agentic` | Batch / agentic markers | Policy / capability markers |
| `contact` | Policy-layer install marker | Routing, resilience, guardrails, batch, plugins, telemetry — physical split deferred |
| `telemetry` | OpenTelemetry sinks | `HAS_TELEMETRY`; feedback types in `telemetry` subpackage |
| `tokenizer` | Token counting (tiktoken) | `HAS_TOKENIZER` |
| `full` | All capability extras + `watchdog` + `keyring` | Includes `contact` |
| `dev` / `docs` / `jupyter` | Tooling only | pytest/mypy/ruff; mkdocs; ipywidgets |

```bash
pip install ai-lib-python[full]
```

Many extras are **markers** (empty dependency lists): the modules ship in the wheel; install the extra when you want an explicit capability contract or when real deps are needed (`vision`, `audio`, `telemetry`, `tokenizer`, `full`).

### Honest capability boundaries

| Area | In the package | Not included |
|------|----------------|--------------|
| **MCP** (`mcp` module) | `McpToolBridge` format conversion | MCP server transport wired into `AiClient` |
| **Computer Use** (`computer_use`) | `ComputerAction`, `SafetyPolicy` validation | Screenshot / input execution environment |
| **Hot reload** | `AiClientBuilder.hot_reload()` flag + in-memory cache; `ProtocolLoader.clear_cache()` | Automatic file watching (needs `watchdog`; `HAS_WATCHDOG`) — no auto-reload today |
| **`ProviderDriver`** | Public `drivers` module | Default `AiClient` chat path |
| **Rate limit env** | Configure via `AiClientBuilder` / `resilience` | `AI_LIB_RPS` / `AI_LIB_RPM` are **not** read by the runtime |

## Advanced: `ProviderDriver`

`ai_lib_python.drivers` exposes `ProviderDriver`, `create_driver`, and OpenAI / Anthropic / Gemini drivers. **`AiClient` does not use this path** for chat; it uses `Pipeline` from manifests. Drivers are for compliance tests and custom integrations.

## Resilience

- **Built into `AiClient`:** optional `max_inflight` backpressure via builder (also `AI_LIB_MAX_INFLIGHT`).
- **Opt-in policy layer:** `ai_lib_python.resilience` (retry, rate limiter, circuit breaker) — use `production_ready()` or explicit `ResilientConfig`.
- **Not auto-enabled** on `AiClient.create()`.

## Protocol manifests

Resolution order:

1. `AiClientBuilder.protocol_path(...)` / `ProtocolLoader(base_path=...)`
2. `AI_PROTOCOL_DIR` / `AI_PROTOCOL_PATH` (local dir; GitHub raw URL supported for remote load)
3. Dev paths: `ai-protocol/`, `../ai-protocol/`, …
4. Fallback: GitHub raw `ailib-official/ai-protocol` (`main`)

Per base path: `dist/v2/providers/<id>.json` → `dist/v1/providers/<id>.json` → source `v2` / `v1` YAML/JSON degrade.

**Identity / aliases (on `main`, see Unreleased):** `load_provider` resolves marketplace aliases via `dist/provider-identity.json` (multi-family map), e.g. `google` → `gemini`, `kimi` → `moonshot`. Parse/validation errors are not masked by alias lookup.

Manifest cache: in-memory. `hot_reload=True` stores a flag but does **not** watch files — call `ProtocolLoader.clear_cache()` or rebuild the client after manifest changes.

## API keys (BYOK chain)

1. Explicit override on builder / `AiClient.create(..., api_key=...)`
2. Manifest-declared env from `endpoint.auth` / top-level `auth`
3. `<PROVIDER_ID>_API_KEY` (recommended for CI/containers)
4. OS keyring when `keyring` is installed (`HAS_KEYRING`; included in `[full]`)

## Environment variables

| Variable | Purpose |
|----------|---------|
| `AI_PROTOCOL_DIR` / `AI_PROTOCOL_PATH` | Local manifest directory or GitHub raw base URL |
| `AI_PROXY_URL` | Explicit proxy when `AI_HTTP_TRUST_ENV=1` |
| `HTTP_PROXY` / `HTTPS_PROXY` | Standard proxy vars (with `AI_HTTP_TRUST_ENV=1`) |
| `NO_PROXY` / `AI_PROXY_NO_PROXY` | Hosts that bypass proxy |
| `AI_HTTP_TIMEOUT_SECS` | HTTP timeout |
| `AI_LIB_MAX_INFLIGHT` | Concurrency backpressure (also via builder) |

Cross-runtime proxy semantics: [CROSS_RUNTIME.md](https://github.com/ailib-official/ai-protocol/blob/main/docs/CROSS_RUNTIME.md).

## Standard error codes (V2)

| Code | Name | Retryable | Fallbackable |
|------|------|-----------|--------------|
| E1001 | `invalid_request` | No | No |
| E1002 | `authentication` | No | Yes |
| E1003 | `permission_denied` | No | No |
| E1004 | `not_found` | No | No |
| E1005 | `request_too_large` | No | No |
| E2001 | `rate_limited` | Yes | Yes |
| E2002 | `quota_exhausted` | No | Yes |
| E3001 | `server_error` | Yes | Yes |
| E3002 | `overloaded` | Yes | Yes |
| E3003 | `timeout` | Yes | Yes |
| E4001 | `conflict` | Yes | No |
| E4002 | `cancelled` | No | No |
| E9999 | `unknown` | No | No |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/unit/ -v
```

Compliance (cross-runtime YAML):

```bash
# POSIX
COMPLIANCE_DIR=../ai-protocol/tests/compliance pytest tests/compliance/ -v

# Windows PowerShell
$env:COMPLIANCE_DIR = "D:\ai-protocol\tests\compliance"
pytest tests/compliance/ -v
```

Mock server integration (requires [ai-protocol-mock](https://github.com/ailib-official/ai-protocol-mock)):

```bash
MOCK_HTTP_URL=http://localhost:4010 pytest tests/integration/ -v
```

## Examples

| Example | Topic |
|---------|-------|
| `basic_chat.py` | Quick start, `execute_with_stats` |
| `streaming.py` | `is_content_delta`, `stream_with_stats` |
| `resilience.py` | Policy layer |
| `multimodal.py` | Vision extra |
| `tool_calling.py` | Tools |
| `multi_provider_production.py` | Routing / fallbacks |
| `guardrails_production.py` | Guardrails |
| `concurrent_production.py` | Concurrency |
| `providers.py` | Provider switching |

## Related

- [AI-Protocol](https://github.com/ailib-official/ai-protocol) — specification & manifests
- [ai-lib-rust](https://github.com/ailib-official/ai-lib-rust) — Rust runtime
- [ai-lib-ts](https://github.com/ailib-official/ai-lib-ts) — TypeScript runtime
- [ai-lib-go](https://github.com/ailib-official/ai-lib-go) — Go runtime

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
