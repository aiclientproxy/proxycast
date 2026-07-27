## Lime v1.114.0

### 新功能

- 新增 Gemini GenerateContent current transport，覆盖 SSE、文本与图片输入、工具调用、usage/finish 事件，以及 `thoughtSignature` 的持久化工具历史。
- 将 Ollama 执行统一到无认证的 OpenAI Responses transport，保留 `/api/tags` 作为独立模型发现入口。
- 支持官方 OpenAI Responses Hosted Web Search 与 Image Generation，在 Provider 执行链中投影完整工具生命周期并保留后续回合历史。
- 新增 App Server v2 `model/rerouted` transient 通知，在可信官方 Responses 返回实际模型变化时提供标准投影。

### 修复

- 修复按 API Key 写入的模型发现缓存无法被 catalog 正确读取的问题，统一 credential-scoped cache 与 Provider readiness。
- 修复 Provider 配置变更后旧模型、凭证、effort 或 service tier 继续参与 Turn 的问题，Turn 启动前会按最新 catalog 协调有效选择。
- 修复仅凭 Provider/模型名称或无能力快照的 `models[]` 条目获得执行权限的问题，`inferred_hint` 现在稳定 fail closed。
- 修复模型选择变化在前台、后台 continuation、queued resume、workflow retry 和 mailbox 路径上的设置投影不一致问题。

### 优化与重构

- 将 Provider typed `models[]`、能力 provenance、catalog refresh、路由 preflight 与 durable Thread selection 收敛到同一 current 控制链。
- 统一 Agent、媒体生成、设置页和模型选择器的 Provider/model 解析，减少前端默认值与后端可执行能力漂移。
- 删除 `OllamaChat`/NDJSON 执行协议与 `custom_models/customModels` 双轨，不保留名称猜测或旧 selection store 回退。
- 将 reroute 与设置变化通过现有 transient RuntimeEvent/v2 projector 发布，避免写入 EventLog 或在 cold resume 中重复投影。

### 测试与质量

- 补充 Gemini、Ollama Responses、Hosted Web Search、模型 reroute、能力 admission 与 catalog reconciliation 的 Rust 回归。
- 新增公开 JSON-RPC 模型选择刷新测试，并扩展 Provider 配置、模型选择、图片/视频/语音入口和 GUI fixture 覆盖。
- 同步 App Server JSON schema 与 TypeScript generated protocol types，强化 current bridge、真实 Electron Gate B 和无 mock fallback 证据。

### 文档

- 更新模型 transport、能力信任边界、catalog/Turn selection 数据流、数据库字段与 Codex 对齐执行记录。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与锁文件版本统一提升到 `1.114.0`。

**完整变更**: `v1.113.0` -> `v1.114.0`
