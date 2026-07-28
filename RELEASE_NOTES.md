## Lime v1.115.0

### 新功能

- 新增 Vertex Gemini current transport，按 project/location 构建专用 endpoint，使用 Bearer access token，并复用 Gemini canonical lowering、SSE 与工具历史语义。
- 新增 Azure OpenAI Responses current route，支持资源根地址、typed `api-version` 和 `api-key` 认证，同时对不受支持的 Chat Completions、WebSocket 与 hosted tools fail closed。
- 打通 Fal 与 xAI 视频生成的公开媒体任务链；xAI request id、轮询状态和终态会持久化，App Server 重启后可继续轮询而不会重复创建任务。
- 新增 `model/list/updated` typed 通知，Provider、凭证或目录变化后，已挂载的前端模型视图会自动失效并刷新。
- Hosted Image Generation 结果和会话 sidecar 图片现在可在 Agent 时间线中直接预览，并为不可用媒体提供稳定占位状态。

### 修复

- 修复模型切换时旧 reasoning effort、service tier 或 collaboration model 残留的问题；Thread 设置会先通过最新目录 preflight，再原子持久化。
- 修复 Provider 正常结束但只返回 reasoning 或空内容时过早结束的问题，新增独立、有限的空响应重采样预算，并保留工具快照与 Provider token budget 约束。
- 修复 HTTP/WebSocket 忽略服务端 `x-should-retry`、`Retry-After` 与 quota reset 提示的问题，避免不可重试请求被放大。
- 修复不同 credential、protocol、endpoint 或 API version 共享 circuit health 的问题，并提供不泄漏凭证和 endpoint 的 exact-route 健康快照。
- 修复模型目录刷新后专用媒体模型继续作为 Agent chat 选择、以及无能力 provenance 的模型获得执行权限的问题。

### 优化与重构

- 将 `model/list.capabilitySnapshot` 设为公开能力读取的唯一事实源，删除无产品消费者的全局 `modelProvider/capabilities/read` 协议、schema、client 与测试正向面。
- 将视频 lowering、网络执行和 Provider 状态统一收敛到 `model-provider`，媒体 runtime 只负责进度与 durable artifact，App Server 只负责 route、凭证和 worker 编排。
- 将 Provider 目录选择、默认值与刷新协调拆分到单一 owner，并统一 foreground、background、queued、retry 与 mailbox 的模型设置投影。
- 收敛 Agent 图片、流式媒体引用和 canonical Item 内容投影，减少历史恢复与实时渲染之间的重复转换。

### 测试与质量

- 补充 Azure Responses、Vertex Gemini、xAI/Fal 视频、Provider health/retry、credential reroute、空响应重采样和模型目录刷新 Rust 回归。
- 新增公开 JSON-RPC 视频任务与模型目录更新覆盖，同步 App Server JSON schema、TypeScript generated protocol types 和 Electron 直连通知测试。
- 扩展 Agent 图片附件、Hosted Image、流式媒体引用、模型注册表自动刷新及五语言不可用状态的前端回归。
- 新增 Codex Item/Event 渲染覆盖清单与治理守卫，为后续 ConversationProjection 单一读模型迁移建立可验证基线。

### 文档

- 更新模型 catalog/admission、Provider health/retry、Vertex Gemini、视频媒体任务与 Agent empty-response 的 current 架构事实源。
- 新增 Codex 渲染对齐 v2 的 Item/Event 投影矩阵、实施计划和完成定义，明确不改变多模型、多模态 current owner。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与锁文件版本统一提升到 `1.115.0`。

**完整变更**: `v1.114.0` -> `v1.115.0`
