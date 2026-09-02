# 协议转换

## 概述

协议转换模块实现不同 LLM API 格式之间的双向转换，使客户端可以使用统一的 OpenAI 格式访问各种 Provider。

## 目录结构

```
lime-rs/src/converter/
├── mod.rs                    # 模块入口
├── protocol_selector.rs      # 协议选择器
├── openai_to_cw.rs           # OpenAI → CodeWhisperer
├── cw_to_openai.rs           # CodeWhisperer → OpenAI
├── anthropic_to_openai.rs    # Anthropic → OpenAI
└── openai_to_antigravity.rs  # OpenAI → Antigravity
```

## 转换流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      客户端请求                                  │
│                   (OpenAI 格式)                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Protocol Selector                             │
│  根据目标 Provider 选择转换器                                    │
└─────────────────────────────────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ OpenAI → CW     │  │ OpenAI → Claude │  │ OpenAI → AG    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                     │                     │
          ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Provider API                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    响应转换 (反向)                               │
│                   CW/Claude/AG → OpenAI                          │
└─────────────────────────────────────────────────────────────────┘
```

## OpenAI → CodeWhisperer

### 请求转换

```rust
// OpenAI 格式
{
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "tools": [...],
    "stream": true
}

// CodeWhisperer 格式
{
    "conversationState": {
        "currentMessage": {
            "userInputMessage": {
                "content": "...",
                "userInputMessageContext": {...}
            }
        },
        "chatTriggerType": "MANUAL",
        "customizationArn": "..."
    }
}
```

### 工具转换

```rust
// OpenAI function tool
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {...}
    }
}

// CW tool format
{
    "name": "get_weather",
    "inputSchema": {...}
}
```

### 特殊工具支持

| 工具类型 | OpenAI 格式 | CW 格式 |
|----------|-------------|---------|
| web_search | `{"type": "web_search"}` | 内置支持 |
| web_search_20250305 | Claude Code 格式 | 转换为 CW 格式 |

## OpenAI → Antigravity

### 请求结构

```rust
// Antigravity 请求格式 (参考 CLIProxyAPI)
{
    "project": "lime",
    "request": {
        "contents": [...],
        "systemInstruction": {...},
        "generationConfig": {...},
        "tools": [...],
        "safetySettings": [...]
    },
    "model": "gemini-2.0-flash"
}
```

### 工具定义转换

```rust
// OpenAI 格式
{
    "type": "function",
    "function": {
        "name": "tool_name",
        "parameters": {...}
    }
}

// Antigravity 格式
{
    "functionDeclarations": [{
        "name": "tool_name",
        "parametersJsonSchema": {...}  // 注意字段名变化
    }]
}
```

### 安全设置

```rust
// 默认安全设置
const DEFAULT_SAFETY_SETTINGS: &[SafetySetting] = &[
    SafetySetting {
        category: "HARM_CATEGORY_HATE_SPEECH",
        threshold: "OFF",
    },
    SafetySetting {
        category: "HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold: "OFF",
    },
    // ...
];
```

## Anthropic → OpenAI

### 响应转换

```rust
// Anthropic 响应
{
    "content": [
        {"type": "text", "text": "..."},
        {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
    ],
    "stop_reason": "end_turn"
}

// OpenAI 响应
{
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "...",
            "tool_calls": [...]
        },
        "finish_reason": "stop"
    }]
}
```

## 流式响应处理

### SSE 格式转换

```rust
// OpenAI SSE
data: {"choices":[{"delta":{"content":"Hello"}}]}

// CW SSE
data: {"messageMetadata":{"..."},"assistantResponseEvent":{"content":"Hello"}}
```

### 转换器实现

```rust
pub struct StreamConverter {
    buffer: String,
    state: StreamState,
}

impl StreamConverter {
    pub fn process_chunk(&mut self, chunk: &str) -> Vec<String> {
        // 解析 SSE 事件
        // 转换格式
        // 返回 OpenAI 格式的 SSE 事件
    }
}
```

## 错误处理

### 错误映射

| Provider 错误 | OpenAI 错误码 |
|---------------|---------------|
| CW ThrottlingException | 429 |
| CW ValidationException | 400 |
| Claude rate_limit_error | 429 |
| Claude invalid_request_error | 400 |

### 错误转换

```rust
pub fn convert_error(provider_error: ProviderError) -> OpenAIError {
    match provider_error {
        ProviderError::RateLimit => OpenAIError {
            code: 429,
            message: "Rate limit exceeded",
            type_: "rate_limit_error",
        },
        // ...
    }
}
```

## 相关文档

- [providers.md](providers.md) - Provider 系统
- [server.md](server.md) - HTTP 服务器
