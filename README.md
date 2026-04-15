# ai-lib-python

**Official Python Runtime for AI-Protocol** — The canonical Pythonic implementation for unified AI model interaction

[![PyPI Version](https://img.shields.io/pypi/v/ai-lib-python.svg)](https://pypi.org/project/ai-lib-python/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green.svg)](LICENSE)

`ai-lib-python` is the Python runtime implementation for the [AI-Protocol](https://github.com/ailib-official/ai-protocol) specification, embodying the core design principle:

> **All logic is operators, all configuration is protocol**

## 🎯 Design Philosophy

Unlike traditional adapter libraries that hardcode provider-specific logic, `ai-lib-python` is a **protocol-driven runtime**:

- **Zero Hardcoding** — All behavior is driven by protocol manifests (YAML/JSON)
- **Operator Pipeline** — Decoder → Selector → Accumulator → FanOut → EventMapper
- **Hot Reload** — Protocol configurations can be updated at runtime without restart
- **Unified Interface** — Single API for all providers, no provider-specific code needed

## 🏗️ v0.8 Architecture: Logical Separation

Starting from v0.8.0, `ai-lib-python` adopts an **execution/policy logical separation** architecture. Unlike Rust, Python maintains a single-package structure (following Python ecosystem conventions), with clear module-level separation:

```
ai_lib_python/
├── ─────────────────────────────────────────────────────────
│   E Layer (Execution) — Deterministic execution, minimal deps
├───────────────────────────────────────────────────────────
│   client/          Unified client interface
│   protocol/        Protocol loading and validation
│   pipeline/        Operator pipeline
│   transport/       HTTP transport layer
│   drivers/         Provider drivers
│   types/           Type system (Message, Event, Tool)
│   structured/      Structured output
│   embeddings/      Embedding generation
│   mcp/             MCP tool bridging
│   computer_use/    Computer Use abstraction
│   multimodal/      Multimodal support
│   stt/ / tts/      Speech recognition/synthesis
│   rerank/          Re-ranking
│
├── ─────────────────────────────────────────────────────────
│   P Layer (Policy) — Policy decisions, may be stateful
├───────────────────────────────────────────────────────────
│   routing/         Model routing
│   cache/           Response caching
│   batch/           Batch processing
│   plugins/         Plugin system
│   resilience/      Resilience (retry/circuit-breaker/rate-limit)
│   telemetry/       Telemetry and observability
│   guardrails/      Input/output guardrails
│   tokens/          Token counting and cost estimation
│   registry/        Capability registry
```

### Benefits of E/P Separation

| Aspect | E Layer Modules | P Layer Modules |
|--------|-----------------|-----------------|
| **Responsibility** | Deterministic execution, protocol loading, type conversion | Policy decisions, caching, routing, telemetry |
| **Dependencies** | Minimal, stateless | May be stateful, depends on E layer |
| **Use Case** | Edge devices, serverless, microservices | Server-side, full applications |

### Installation Options

```bash
# Basic installation (E-layer core capabilities)
pip install ai-lib-python

# Full installation (all capabilities)
pip install ai-lib-python[full]
```

### Capability Extras

**Execution Layer Capabilities**:
- `[vision]` — Image processing (Pillow)
- `[audio]` — Audio processing (soundfile)
- `[embeddings]` — Embedding generation
- `[structured]` — Structured output / JSON mode
- `[stt]` — Speech-to-text
- `[tts]` — Text-to-speech
- `[reranking]` — Document re-ranking

**Policy Layer Capabilities**:
- `[batch]` — Batch processing
- `[telemetry]` — OpenTelemetry integration
- `[tokenizer]` — Token counting (tiktoken)

**Meta-extras**:
- `[full]` — Enable all capabilities
- `[dev]` — Development dependencies (pytest, mypy, ruff)
- `[docs]` — Documentation build (mkdocs)

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from ai_lib_python import AiClient, Message

async def main():
    # Protocol-driven: supports any provider defined in ai-protocol manifests
    client = await AiClient.create("anthropic/claude-3-5-sonnet")

    # Simple chat
    response = await (
        client.chat()
        .system("You are a helpful assistant.")
        .user("Hello!")
        .execute()
    )

    print(response.content)
    await client.close()

asyncio.run(main())
```

### Streaming Response

```python
import asyncio
from ai_lib_python import AiClient, Message
from ai_lib_python.types.events import StreamingEvent

async def main():
    client = await AiClient.create("openai/gpt-4o")

    # Streaming chat
    async for event in client.chat().user("Tell me a joke").stream():
        if isinstance(event, StreamingEvent.PartialContentDelta):
            print(event.content, end="", flush=True)
        elif isinstance(event, StreamingEvent.StreamEnd):
            print()  # newline

    await client.close()

asyncio.run(main())
```

### Production Configuration

```python
from ai_lib_python import AiClient

# Enable full production capabilities: retry, circuit breaker, rate limiting
client = await (
    AiClient.builder()
    .model("deepseek/deepseek-chat")
    .production_ready()  # One-click enable all resilience patterns
    .build()
)
```

### Multimodal

```python
from ai_lib_python import Message, MessageContent, ContentBlock

# Image + text
message = Message(
    role="user",
    content=MessageContent.blocks([
        ContentBlock.text("Describe this image"),
        ContentBlock.image_from_file("./photo.jpg"),
    ])
)

response = await client.chat().messages([message]).execute()
```

## 🔧 Configuration

### Protocol Manifest Search Path

The runtime searches for protocol configurations in the following order:

1. `AI_PROTOCOL_DIR` / `AI_PROTOCOL_PATH` environment variable
2. Common development paths: `ai-protocol/`, `../ai-protocol/`, `../../ai-protocol/`
3. Final fallback: GitHub raw `ailib-official/ai-protocol`

### API Keys

**Recommended** (production):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
```

**Optional** (local development): OS keyring (requires `keyring` package)

### Environment Variables

```bash
# Proxy
export AI_PROXY_URL="http://user:pass@host:port"

# Timeout
export AI_HTTP_TIMEOUT_SECS=30

# Concurrency limit
export AI_LIB_MAX_INFLIGHT=10

# Rate limiting
export AI_LIB_RPS=5  # or AI_LIB_RPM=300
```

### HTTP proxies (cross-runtime parity with ai-lib-rust)

| Variable | Purpose |
|----------|---------|
| `AI_PROXY_URL` | Explicit proxy URL for outbound requests when `AI_HTTP_TRUST_ENV=1` (Python also honors httpx `trust_env` only in that mode). |
| `HTTP_PROXY` / `HTTPS_PROXY` | Standard vars; in Rust they are merged with `AI_PROXY_URL` as candidate routes. In Python, enable via `AI_HTTP_TRUST_ENV=1` so local/mock traffic is not accidentally proxied. |
| `NO_PROXY` / `AI_PROXY_NO_PROXY` | Comma-separated hosts that must bypass the proxy (include mock hostnames, API hostnames that must be direct, and `127.0.0.1`). Rust documents `AI_PROXY_NO_PROXY`; set the same where your stack reads it. |

With a proxy: set `NO_PROXY` to include the mock server host (for example `NO_PROXY=192.168.2.13,localhost,127.0.0.1`).

Or in code: `AiClient.create("openai/gpt-4o", base_url="http://localhost:4010")`.

For shared semantics across runtimes, see [CROSS_RUNTIME.md](https://github.com/ailib-official/ai-protocol/blob/main/docs/CROSS_RUNTIME.md).

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v
```

### Compliance Tests (Cross-Runtime Consistency)

Install test dependencies first (`pytest` is not a default runtime dependency):

```bash
python -m pip install -e ".[dev]"
```

```bash
# Run compliance tests
python -m pytest tests/compliance/ -v

# Specify compliance directory (POSIX)
COMPLIANCE_DIR=../ai-protocol/tests/compliance python -m pytest tests/compliance/ -v

# Windows PowerShell
$env:COMPLIANCE_DIR = "D:\ai-protocol\tests\compliance"
python -m pytest tests/compliance/ -v

# Wave-5 execution-layer-only subset (skips resilience-heavy cases)
$env:COMPLIANCE_SUBSET = "e_only"   # PowerShell
# COMPLIANCE_SUBSET=e_only python -m pytest tests/compliance/ -v   # POSIX
python -m pytest tests/compliance/ -v
```

### Testing with Mock Server

```bash
# Start ai-protocol-mock
docker-compose up -d

# Run tests with mock
MOCK_HTTP_URL=http://localhost:4010 pytest tests/integration/ -v
```

## 📊 Observability

### Call Statistics

```python
response = await client.chat().user("Hello").execute()
print(f"request_id: {response.stats.request_id}")
print(f"latency_ms: {response.stats.latency_ms}")
print(f"tokens: {response.usage}")
```

### Telemetry Feedback (opt-in)

```python
from ai_lib_python.telemetry import FeedbackEvent, ChoiceSelectionFeedback

await client.report_feedback(
    FeedbackEvent.ChoiceSelection(
        request_id=response.stats.request_id,
        chosen_index=0,
    )
)
```

## 🔄 Error Codes (V2 Specification)

All provider errors are normalized to 13 standard error codes:

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

## 🤝 Community & Contributing

### Use Cases

- **Server Applications** — Use `pip install ai-lib-python[full]` for complete capabilities
- **Edge/Serverless** — Basic installation, add extras as needed
- **Microservices** — Combine with telemetry for distributed tracing

### Contributing Guidelines

1. Code must pass `ruff` checks and `mypy` type checking
2. New features must include tests; compliance tests must pass
3. Follow [Python type hints best practices](https://typing.readthedocs.io/)

### Code of Conduct

- Respect all contributors
- Welcome participants from all backgrounds
- Focus on technical discussions, avoid personal attacks
- Report issues via GitHub Issues

## 🔗 Related Projects

| Project | Description |
|---------|-------------|
| [AI-Protocol](https://github.com/ailib-official/ai-protocol) | Protocol specification (v1.5 / V2) |
| [ai-lib-rust](https://github.com/ailib-official/ai-lib-rust) | Rust runtime |
| [ai-lib-ts](https://github.com/ailib-official/ai-lib-ts) | TypeScript runtime |
| [ai-lib-go](https://github.com/ailib-official/ai-lib-go) | Go runtime |
| [ai-protocol-mock](https://github.com/ailib-official/ai-protocol-mock) | Mock server |

## 📄 License

This project is dual-licensed:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

You may choose either.

---

**ai-lib-python** — Where protocol meets Pythonic 🚀
