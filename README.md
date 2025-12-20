<div align="center">

# ProxyCast 🚀

**把你的 AI 客户端额度用到任何地方**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-blue.svg)](https://tauri.app/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://react.dev/)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)

</div>

---

## 🤔 这个工具能帮你做什么？

**场景一：换个更好用的 IDE**
> 我有 Kiro 账号，每月免费用 Claude Sonnet 4.5，但 Kiro IDE 不太顺手。我想用 Claude Code 或 Cursor 来写代码，但又不想额外付费买 API。

**场景二：把额度分享给其他工具**
> Claude Code 这个月额度还剩很多，与其浪费不如转给 Cherry Studio 聊天用，或者给我的 AI Agent 项目提供 API 接口。

**场景三：统一管理多个 AI 账号**
> 我有 Kiro、Gemini CLI、通义千问好几个账号，想统一管理，哪个有额度就用哪个。

**ProxyCast 就是解决这些问题的工具** —— 它把你已有的 AI 客户端凭证转换成标准 OpenAI API，让任何支持 OpenAI 接口的工具都能用上你的免费额度。

---

## 💡 工作原理

```
你的 AI 客户端凭证          ProxyCast              任意 OpenAI 兼容工具
┌─────────────────┐      ┌─────────────┐      ┌─────────────────────┐
│ Kiro OAuth      │      │             │      │ Claude Code         │
│ Gemini OAuth    │ ───▶ │  本地 API   │ ───▶ │ Cherry Studio       │
│ 通义千问 OAuth   │      │  代理服务    │      │ Cursor / Cline      │
│ ...             │      │             │      │ 你的 AI Agent       │
└─────────────────┘      └─────────────┘      └─────────────────────┘
```

> **💡 与 AIClient-2-API 的区别**
> 
> ProxyCast 是 [AIClient-2-API](https://github.com/justlovemaki/AIClient-2-API) 的桌面版本，提供更友好的图形界面和一键操作体验，无需命令行配置。

---

## ✨ 核心特性

### 🎯 多 Provider 统一管理
- **Kiro Claude** - 通过 OAuth 免费使用 Claude Sonnet 4.5
- **Gemini CLI** - 通过 OAuth 突破 Gemini 免费限制
- **Gemini API Key** - 多账号负载均衡，支持模型排除
- **通义千问** - 通过 OAuth 使用 Qwen3 Coder Plus
- **OpenAI Codex** - 通过 OAuth 使用 GPT 模型
- **iFlow** - 支持 OAuth 和 Cookie 两种认证方式
- **Vertex AI** - Google Cloud AI 平台，支持模型别名
- **OpenAI 自定义** - 配置自定义 OpenAI 兼容 API
- **Claude 自定义** - 配置自定义 Claude API

### 🖥️ 友好的图形界面
- **Dashboard** - 服务状态监控、API 测试面板
- **Provider 管理** - 一键加载凭证、Token 刷新、默认 Provider 切换
- **设置页面** - 服务器配置、端口设置、API Key 管理
- **日志查看** - 实时日志记录、操作追踪

### 🔄 智能凭证管理
- 自动检测凭证文件变化（每 5 秒）
- 一键读取本地 OAuth 凭证
- Token 过期自动刷新
- 环境变量导出（.env 格式）
- **配额超限自动切换** - 自动切换到下一个可用凭证
- **预览模型回退** - 主模型配额用尽时尝试预览版本
- **Per-Key 代理** - 为每个凭证单独配置代理

### 🔐 安全与管理
- **HTTPS 部署** - 当前版本不内置 TLS，请使用反向代理进行 HTTPS 终止
- **远程管理 API** - 通过 API 远程管理配置和凭证
- **访问控制** - 支持 localhost 限制和密钥认证

### 🔌 Amp CLI 集成
- 支持 `/api/provider/{provider}/v1/*` 路由模式
- 模型映射 - 将不可用模型映射到可用替代
- 管理端点代理 - 代理认证和账户功能

### 🌐 完整 API 兼容
- `/v1/chat/completions` - OpenAI Chat API
- `/v1/models` - 模型列表
- `/v1/messages` - Anthropic Messages API
- `/v1/messages/count_tokens` - Token 计数
- `/api/provider/{provider}/v1/*` - Amp CLI 路由
- `/v0/management/*` - 远程管理 API

---

## 📸 界面截图

### Dashboard - 服务控制与 API 测试
![Dashboard](docs/images/420984ac-8287-44c6-b209-bbffd59dc0eb.png)

### 凭证池 - 多凭证管理
![Provider Pool](docs/images/44dfe29b-ef1e-4fab-82cc-8fbb3d5ae673.png)

### API Server - 路由与日志
![API Server](docs/images/54223543-bfb1-4a96-a9d8-a1b9b2f9c154.png)

### 设置页面 - 服务器配置
![Settings](docs/images/83270842-209e-4a36-98e3-607642273806.png)

### AI Clients - 客户端配置
![AI Clients](docs/images/c9d36815-a29a-437e-9f26-872ed0949921.png)

### MCP 服务器管理
![MCP](docs/images/fc54f689-b092-4180-a531-a41e310c42cb.png)

### Prompts 管理
![Prompts](docs/images/ff6d0f51-a703-4642-a4ab-2b9512b332af.png)

---

## 🚀 快速开始

### 下载安装

从 [Releases](https://github.com/aiclientproxy/proxycast/releases) 页面下载对应平台的安装包：

- **macOS (Apple Silicon)**: `ProxyCast_x.x.x_aarch64.dmg`
- **Windows (x64)**: `ProxyCast_x.x.x_x64-setup.exe`

### 凭证文件位置

| Provider | 默认路径 | 说明 |
|----------|---------|------|
| **Kiro** | `~/.aws/sso/cache/kiro-auth-token.json` | Kiro OAuth Token |
| **Gemini** | `~/.gemini/oauth_creds.json` | Gemini CLI OAuth |
| **Qwen** | `~/.qwen/oauth_creds.json` | 通义千问 OAuth |

> **提示**: `~` 表示用户主目录（macOS: `/Users/用户名`，Windows: `C:\Users\用户名`）

### 使用步骤

1. **启动应用** - 打开 ProxyCast
2. **加载凭证** - 进入 Provider 管理页面，点击"一键读取凭证"
3. **启动服务** - 在 Dashboard 点击"启动服务器"
4. **配置客户端** - 在 Cherry-Studio、Cline 等工具中配置：
   ```
   API Base URL: http://localhost:8999/v1
   API Key: 启动时自动生成的密钥（可在设置页查看/修改）
   ```

---

## 🔧 API 使用示例

### OpenAI Chat Completions

```bash
curl http://localhost:8999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "claude-sonnet-4-5-20250514",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
```

### Anthropic Messages API

```bash
curl http://localhost:8999/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-5-20250514",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

---

## 🛠️ 开发构建

### 环境要求

- Node.js >= 20.0.0
- Rust >= 1.70
- pnpm 或 npm

### 本地开发

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run tauri dev
```

### 构建发布

```bash
# 构建生产版本
npm run tauri build
```

---

## 📄 开源协议

本项目采用 [GNU General Public License v3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0) 协议开源。

## 🙏 致谢

- [AIClient-2-API](https://github.com/justlovemaki/AIClient-2-API) - 核心逻辑参考
- [Tauri](https://tauri.app/) - 跨平台桌面框架
- [shadcn/ui](https://ui.shadcn.com/) - UI 组件库

---

## ⚠️ 免责声明

### 使用风险提示
本项目（ProxyCast）仅供学习和研究使用。用户在使用本项目时需自行承担所有风险。作者不对因使用本项目而导致的任何直接、间接或后果性损失负责。

### 第三方服务责任声明
本项目是一个 API 代理工具，不提供任何 AI 模型服务。所有 AI 模型服务均由各自的第三方提供商（如 Google、Anthropic、阿里云等）提供。用户在通过本项目访问这些服务时，应遵守各第三方服务的使用条款和政策。

### 数据隐私声明
本项目在本地运行，不收集或上传任何用户数据。但用户在使用本项目时应保护好自己的 API 密钥和其他敏感信息。
