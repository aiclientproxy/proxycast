# ProxyCast 技术规格文档

## 项目概述

ProxyCast 是一个桌面端 AI API 代理工具，将各种大模型客户端 API（Gemini CLI、Qwen Code、Kiro Claude 等）统一转换为本地 OpenAI 兼容接口。

## 技术栈

基于 pubcast 项目技术栈：

| 类别 | 技术 |
|------|------|
| 框架 | Tauri 2.0 (Rust + Web) |
| 前端 | React 18 + TypeScript |
| 构建 | Vite 5 |
| UI | Tailwind CSS + Radix UI |
| 图标 | Lucide React |

### 核心依赖

```json
{
  "@tauri-apps/api": "^2.0.0",
  "@tauri-apps/plugin-shell": "^2.0.0",
  "@radix-ui/react-*": "UI 组件",
  "react": "^18.3.1",
  "tailwindcss": "^3.4.x"
}
```

## 支持的 Provider 渠道

参考 AIClient-2-API，需支持以下渠道：

| Provider | 协议 | 说明 |
|----------|------|------|
| `claude-kiro-oauth` | OpenAI/Claude | Kiro OAuth 访问 Claude Sonnet 4.5 |
| `gemini-cli-oauth` | OpenAI/Claude/Gemini | Gemini CLI OAuth |
| `openai-qwen-oauth` | OpenAI/Claude | 通义千问 OAuth |
| `openai-custom` | OpenAI | 自定义 OpenAI 兼容 API |
| `claude-custom` | Claude | 自定义 Claude API |
| `gemini-antigravity` | Gemini | Antigravity 协议 |
| `openaiResponses-custom` | OpenAI Responses | 结构化对话 |

## 核心功能模块

### 1. API 代理服务 (Rust 后端)

```
src-tauri/
├── src/
│   ├── main.rs           # 入口
│   ├── server.rs         # HTTP 服务器 (axum/actix-web)
│   ├── providers/        # Provider 实现
│   │   ├── mod.rs
│   │   ├── kiro.rs       # Kiro Claude
│   │   ├── gemini.rs     # Gemini CLI
│   │   ├── qwen.rs       # Qwen Code
│   │   ├── openai.rs     # OpenAI Custom
│   │   └── claude.rs     # Claude Custom
│   ├── converter.rs      # 协议转换 (OpenAI <-> Claude <-> Gemini)
│   ├── token.rs          # Token 管理/刷新
│   └── config.rs         # 配置管理
```

### 2. 前端 UI (React)

```
src/
├── App.tsx
├── components/
│   ├── Dashboard.tsx     # 仪表盘
│   ├── ProviderConfig.tsx # Provider 配置
│   ├── TokenManager.tsx  # Token 管理
│   ├── LogViewer.tsx     # 日志查看
│   └── ui/               # 通用 UI 组件
├── hooks/
│   └── useTauri.ts       # Tauri API hooks
└── lib/
    └── utils.ts
```

## API 端点设计

### 路由模式

```
http://localhost:8999/{provider}/v1/chat/completions
http://localhost:8999/{provider}/v1/messages
```

### 支持的端点

| 端点 | 协议 | 说明 |
|------|------|------|
| `/v1/chat/completions` | OpenAI | 聊天补全 |
| `/v1/messages` | Claude | Anthropic 消息 |
| `/v1/models` | OpenAI | 模型列表 |
| `/health` | - | 健康检查 |

## 配置文件结构

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8999,
    "apiKey": "your-api-key"
  },
  "providers": {
    "kiro": {
      "enabled": true,
      "credentialsPath": "~/.aws/sso/cache/kiro-auth-token.json",
      "region": "us-east-1"
    },
    "gemini": {
      "enabled": false,
      "credentialsPath": "~/.gemini/oauth_creds.json",
      "projectId": ""
    },
    "qwen": {
      "enabled": false,
      "credentialsPath": "~/.qwen/oauth_creds.json"
    },
    "openai": {
      "enabled": false,
      "apiKey": "",
      "baseUrl": "https://api.openai.com/v1"
    },
    "claude": {
      "enabled": false,
      "apiKey": "",
      "baseUrl": "https://api.anthropic.com"
    }
  },
  "defaultProvider": "kiro"
}
```

## Token 凭证路径

| 服务 | 默认路径 |
|------|----------|
| Kiro | `~/.aws/sso/cache/kiro-auth-token.json` |
| Gemini | `~/.gemini/oauth_creds.json` |
| Qwen | `~/.qwen/oauth_creds.json` |
| Antigravity | `~/.antigravity/oauth_creds.json` |

## 协议转换

支持三种协议互转：

```
OpenAI <---> Claude <---> Gemini
```

### 转换矩阵

| 输入协议 | 输出 Provider | 说明 |
|----------|---------------|------|
| OpenAI | kiro | OpenAI -> CodeWhisperer |
| OpenAI | gemini | OpenAI -> Gemini |
| Claude | kiro | Claude -> CodeWhisperer |
| Claude | gemini | Claude -> Gemini |
| Claude | openai | Claude -> OpenAI |

## UI 功能

1. **仪表盘** - 服务状态、请求统计
2. **Provider 管理** - 启用/禁用、配置凭证
3. **Token 管理** - 查看/刷新 OAuth Token
4. **日志查看** - 实时请求日志
5. **设置** - 端口、API Key 等

## 开发计划

### Phase 1: 基础框架
- [ ] Tauri 项目初始化
- [ ] 基础 UI 布局
- [ ] 配置管理

### Phase 2: Kiro Provider
- [ ] Kiro OAuth Token 读取
- [ ] CodeWhisperer API 调用
- [ ] OpenAI/Claude 协议支持

### Phase 3: 其他 Provider
- [ ] Gemini CLI OAuth
- [ ] Qwen OAuth
- [ ] OpenAI/Claude Custom

### Phase 4: 高级功能
- [ ] Provider Pool 管理
- [ ] 自动 Token 刷新
- [ ] 请求日志/统计
