# ai-lib-python

**AI-Protocol 官方 Python 运行时** — 统一 AI 模型交互的规范 Python 实现

[![PyPI Version](https://img.shields.io/pypi/v/ai-lib-python.svg)](https://pypi.org/project/ai-lib-python/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green.svg)](LICENSE)

`ai-lib-python` 是 [AI-Protocol](https://github.com/ailib-official/ai-protocol) 规范的 Python 运行时实现，体现了核心设计原则：

> **一切逻辑皆算子，一切配置皆协议**

## 🎯 设计哲学

与传统硬编码 Provider 逻辑的适配器库不同，`ai-lib-python` 是一个**协议驱动的运行时**：

- **零硬编码** — 所有行为由协议 Manifest（YAML/JSON）驱动
- **算子流水线** — Decoder → Selector → Accumulator → FanOut → EventMapper
- **热重载** — 协议配置可在运行时更新，无需重启应用
- **统一接口** — 单一 API 适配所有 Provider，开发者无需关心底层差异

## 🏗️ v0.8 架构：逻辑分层

从 v0.8.0 开始，`ai-lib-python` 采用**执行层/策略层逻辑分离**的架构设计。与 Rust 不同，Python 版本保持单包结构（符合 Python 生态习惯），但在模块层面清晰划分职责：

```
ai_lib_python/
├── ─────────────────────────────────────────────────────────
│   E 层（执行层）— 确定性执行，最小依赖
├───────────────────────────────────────────────────────────
│   client/          统一客户端接口
│   protocol/        协议加载与验证
│   pipeline/        算子流水线
│   transport/       HTTP 传输层
│   drivers/         Provider 驱动
│   types/           类型系统（Message, Event, Tool）
│   structured/      结构化输出
│   embeddings/      嵌入向量生成
│   mcp/             MCP 工具桥接
│   computer_use/    Computer Use 抽象
│   multimodal/      多模态支持
│   stt/ / tts/      语音识别/合成
│   rerank/          重排序
│
├── ─────────────────────────────────────────────────────────
│   P 层（策略层）— 策略决策，可有状态
├───────────────────────────────────────────────────────────
│   routing/         模型路由
│   cache/           响应缓存
│   batch/           批处理执行
│   plugins/         插件系统
│   resilience/      弹性策略（重试/熔断/限流）
│   telemetry/       遥测与可观测性
│   guardrails/      输入/输出守卫
│   tokens/          Token 计数与成本估算
│   registry/        能力注册表
```

### E/P 分层的优势

| 维度 | E 层模块 | P 层模块 |
|------|----------|----------|
| **职责** | 确定性执行、协议加载、类型转换 | 策略决策、缓存、路由、遥测 |
| **依赖** | 最小化，无状态 | 可有状态，依赖 E 层 |
| **适用场景** | 边缘设备、Serverless、微服务 | 服务端、完整应用 |

### 安装选项

```bash
# 基础安装（包含 E 层核心能力）
pip install ai-lib-python

# 完整安装（包含所有能力）
pip install ai-lib-python[full]
```

### 能力 Extras

**执行层能力**：
- `[vision]` — 图像处理（Pillow）
- `[audio]` — 音频处理（soundfile）
- `[embeddings]` — 嵌入向量生成
- `[structured]` — 结构化输出/JSON 模式
- `[stt]` — 语音转文字
- `[tts]` — 文字转语音
- `[reranking]` — 文档重排序

**策略层能力**：
- `[batch]` — 批处理执行
- `[telemetry]` — OpenTelemetry 集成
- `[tokenizer]` — Token 计数（tiktoken）

**Meta-extras**：
- `[full]` — 启用所有能力
- `[dev]` — 开发依赖（pytest, mypy, ruff）
- `[docs]` — 文档构建（mkdocs）

## 🚀 快速开始

### 基本用法

```python
import asyncio
from ai_lib_python import AiClient, Message

async def main():
    # 协议驱动：支持 ai-protocol manifest 中定义的任何 provider
    client = await AiClient.create("anthropic/claude-3-5-sonnet")

    # 简单聊天
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

### 流式响应

```python
import asyncio
from ai_lib_python import AiClient, Message
from ai_lib_python.types.events import StreamingEvent

async def main():
    client = await AiClient.create("openai/gpt-4o")

    # 流式聊天
    async for event in client.chat().user("讲一个笑话").stream():
        if isinstance(event, StreamingEvent.PartialContentDelta):
            print(event.content, end="", flush=True)
        elif isinstance(event, StreamingEvent.StreamEnd):
            print()  # 换行

    await client.close()

asyncio.run(main())
```

### 生产环境配置

```python
from ai_lib_python import AiClient

# 启用完整生产能力：重试、熔断、限流
client = await (
    AiClient.builder()
    .model("deepseek/deepseek-chat")
    .production_ready()  # 一键启用所有弹性策略
    .build()
)
```

### 多模态

```python
from ai_lib_python import Message, MessageContent, ContentBlock

# 图像 + 文本
message = Message(
    role="user",
    content=MessageContent.blocks([
        ContentBlock.text("描述这张图片"),
        ContentBlock.image_from_file("./photo.jpg"),
    ])
)

response = await client.chat().messages([message]).execute()
```

## 🔧 配置

### 协议 Manifest 搜索路径

运行时按以下顺序查找协议配置：

1. `AI_PROTOCOL_DIR` / `AI_PROTOCOL_PATH` 环境变量
2. 常见开发路径：`ai-protocol/`、`../ai-protocol/`、`../../ai-protocol/`
3. 最终兜底：GitHub raw `ailib-official/ai-protocol`

### API 密钥

**推荐方式**（生产环境）：

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
```

**可选方式**（本地开发）：操作系统密钥环（需安装 `keyring` 包）

### 环境变量

```bash
# 代理
export AI_PROXY_URL="http://user:pass@host:port"

# 超时
export AI_HTTP_TIMEOUT_SECS=30

# 并发限制
export AI_LIB_MAX_INFLIGHT=10

# 速率限制
export AI_LIB_RPS=5  # 或 AI_LIB_RPM=300
```

## 🧪 测试

### 单元测试

```bash
# 运行所有测试
pytest tests/ -v

# 仅运行单元测试
pytest tests/unit/ -v
```

### 兼容性测试（跨运行时一致性）

```bash
# 运行兼容性测试
pytest tests/compliance/ -v

# 指定兼容性测试目录
COMPLIANCE_DIR=../ai-protocol/tests/compliance pytest tests/compliance/ -v
```

### 使用 Mock 服务器测试

```bash
# 启动 ai-protocol-mock
docker-compose up -d

# 使用 mock 运行测试
MOCK_HTTP_URL=http://localhost:4010 pytest tests/integration/ -v
```

## 📊 可观测性

### 调用统计

```python
response = await client.chat().user("Hello").execute()
print(f"request_id: {response.stats.request_id}")
print(f"latency_ms: {response.stats.latency_ms}")
print(f"tokens: {response.usage}")
```

### 遥测反馈（opt-in）

```python
from ai_lib_python.telemetry import FeedbackEvent, ChoiceSelectionFeedback

await client.report_feedback(
    FeedbackEvent.ChoiceSelection(
        request_id=response.stats.request_id,
        chosen_index=0,
    )
)
```

## 🔄 错误码（V2 规范）

所有 Provider 错误归一化为 13 个标准错误码：

| 错误码 | 名称 | 可重试 | 可回退 |
|--------|------|--------|--------|
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

## 🤝 社区与贡献

### 适用场景

- **服务端应用** — 使用 `pip install ai-lib-python[full]` 获得完整能力
- **边缘/Serverless** — 基础安装即可，按需添加 extras
- **微服务** — 结合 telemetry 实现分布式追踪

### 贡献指南

1. 代码需通过 `ruff` 检查和 `mypy` 类型检查
2. 新功能需包含测试，兼容性测试必须通过
3. 遵循 [Python 类型提示最佳实践](https://typing.readthedocs.io/)

### 行为准则

- 尊重所有贡献者
- 欢迎不同背景的参与者
- 专注于技术讨论，避免人身攻击
- 发现问题请通过 GitHub Issues 反馈

## 🔗 相关项目

| 项目 | 说明 |
|------|------|
| [AI-Protocol](https://github.com/ailib-official/ai-protocol) | 协议规范（v1.5 / V2） |
| [ai-lib-rust](https://github.com/ailib-official/ai-lib-rust) | Rust 运行时 |
| [ai-lib-ts](https://github.com/ailib-official/ai-lib-ts) | TypeScript 运行时 |
| [ai-lib-go](https://github.com/ailib-official/ai-lib-go) | Go 运行时 |
| [ai-protocol-mock](https://github.com/ailib-official/ai-protocol-mock) | Mock 服务器 |

## 📄 许可证

本项目采用双许可证：

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

您可任选其一。

---

**ai-lib-python** — 协议与 Pythonic 的完美结合 🚀
