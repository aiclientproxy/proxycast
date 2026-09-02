# HTTP 服务器

## 概述

HTTP 服务器提供 OpenAI 和 Claude 兼容的 API 端点，支持流式响应。

## 目录结构

```
lime-rs/src/
├── server/
│   ├── mod.rs              # 服务器入口
│   ├── routes.rs           # 路由定义
│   ├── handlers.rs         # 请求处理器
│   └── middleware.rs       # 中间件
├── server_utils.rs         # 工具函数
└── streaming/              # 流式响应
    ├── mod.rs
    └── sse.rs
```

## API 端点

### OpenAI 兼容端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 聊天补全 |
| `/v1/models` | GET | 模型列表 |
| `/v1/embeddings` | POST | 文本嵌入 |

### Claude 兼容端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/messages` | POST | 消息 API |

### 管理端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/metrics` | GET | 指标统计 |

## 请求处理流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      HTTP 请求                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Middleware                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ 认证        │  │ 日志        │  │ 流量监控                │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Router                                      │
│  根据路径和模型选择处理器                                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Handler                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ 请求验证    │  │ 协议转换    │  │ Provider 调用           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      响应                                        │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │ JSON 响应   │  │ SSE 流式    │                               │
│  └─────────────┘  └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## 服务器配置

```rust
pub struct ServerConfig {
    pub host: String,           // 监听地址
    pub port: u16,              // 监听端口
    pub cors_enabled: bool,     // CORS 支持
    pub max_body_size: usize,   // 最大请求体
    pub timeout: Duration,      // 请求超时
}

// 默认配置
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            cors_enabled: true,
            max_body_size: 10 * 1024 * 1024,  // 10MB
            timeout: Duration::from_secs(300),
        }
    }
}
```

## 中间件

### 认证中间件

```rust
pub async fn auth_middleware(req: Request, next: Next) -> Response {
    // 检查 Authorization header
    let auth_header = req.headers().get("Authorization");
    
    match auth_header {
        Some(value) => {
            // 验证 Bearer token
            if validate_token(value) {
                next.run(req).await
            } else {
                Response::unauthorized()
            }
        }
        None => Response::unauthorized(),
    }
}
```

### 流量监控中间件

```rust
pub async fn flow_monitor_middleware(req: Request, next: Next) -> Response {
    let start = Instant::now();
    let request_id = generate_request_id();
    
    // 记录请求
    flow_monitor.record_request(&request_id, &req).await;
    
    let response = next.run(req).await;
    
    // 记录响应
    flow_monitor.record_response(&request_id, &response, start.elapsed()).await;
    
    response
}
```

## 流式响应

### SSE 实现

```rust
pub async fn stream_response(
    provider_stream: impl Stream<Item = Result<Bytes>>,
) -> impl IntoResponse {
    let stream = provider_stream.map(|chunk| {
        match chunk {
            Ok(data) => {
                // 转换为 OpenAI SSE 格式
                let converted = convert_to_openai_sse(&data);
                Ok::<_, Error>(Event::default().data(converted))
            }
            Err(e) => Err(e),
        }
    });
    
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
}
```

### 流式转换

```rust
// Provider 响应 → OpenAI SSE
pub fn convert_stream_chunk(chunk: &ProviderChunk) -> String {
    let delta = ChatCompletionChunk {
        id: chunk.id.clone(),
        choices: vec![Choice {
            delta: Delta {
                content: chunk.content.clone(),
                tool_calls: chunk.tool_calls.clone(),
            },
            finish_reason: chunk.finish_reason.clone(),
        }],
    };
    
    format!("data: {}\n\n", serde_json::to_string(&delta).unwrap())
}
```

## 错误处理

### 错误响应格式

```rust
#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

// 示例
{
    "error": {
        "message": "Rate limit exceeded",
        "type": "rate_limit_error",
        "code": "429"
    }
}
```

### 错误处理器

```rust
pub async fn error_handler(err: Error) -> Response {
    let (status, error_response) = match err {
        Error::Validation(msg) => (
            StatusCode::BAD_REQUEST,
            ErrorResponse::new("invalid_request_error", msg),
        ),
        Error::RateLimit => (
            StatusCode::TOO_MANY_REQUESTS,
            ErrorResponse::new("rate_limit_error", "Rate limit exceeded"),
        ),
        Error::Provider(e) => (
            StatusCode::BAD_GATEWAY,
            ErrorResponse::new("provider_error", e.to_string()),
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorResponse::new("internal_error", "Internal server error"),
        ),
    };
    
    (status, Json(error_response)).into_response()
}
```

## 相关文档

- [converter.md](converter.md) - 协议转换
- [providers.md](providers.md) - Provider 系统
