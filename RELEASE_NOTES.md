## Lime v1.116.0

### 新功能

- Agent 对话改为直接按 canonical Turn / Item 渲染，实时事件、冷启动读取和会话恢复共享同一 ConversationProjection；Hook、Sleep、Review、MCP 和动态工具状态可在统一时间线中呈现。
- 新增统一 Pending Interaction 层，集中承接审批、用户补充输入、MCP elicitation 和动态工具交互，并在恢复后重建仍待处理的请求。
- 媒体读取迁移到 Lime-owned v2 `media/read`，使用 canonical `threadId` 读取 bounded sidecar，并在图片可用、不可用和异常状态下提供稳定预览结果。
- 官方 Agnes API Hub 现在可识别 `agnes-2.0-flash` 的 canonical 视觉、工具、流式与推理能力；非官方或非 HTTPS endpoint 不会获得同等能力授权。

### 修复

- 修复长会话恢复时重复合成消息、实时与历史内容漂移、推理/工具正文重复以及待处理交互丢失的问题。
- 修复超长历史一次挂载全部 Item 和 assistant 长正文导致的打开卡顿；默认只渲染 bounded Turn window，旧记录和完整正文按需展开。
- 修复 sidecar 媒体仍依赖 v0 session identity、读取失败被静默吞掉或异常 payload 进入预览的问题；协议漂移和未投影通知现在会产生可见诊断。
- 修复多 Agent 工具结果只携带活动事件而缺少 typed 状态事实的问题，子 Agent 的完成、失败和等待状态可进入 canonical 投影。

### 优化与重构

- 删除 canonical Item 到旧 Message 的二次合成、三个流式内容同步 hook、重复 approval/user-input API 和旧 MCP elicitation dialog owner，Renderer 只保留一条 current 投影链。
- 将 assistant message phase、MCP tool result/error、动态工具多模态输出和 Agent control state 收紧为 typed protocol，并同步 Rust、JSON schema 与 TypeScript client。
- 将历史窗口、长正文 preview、媒体异常态与 notification drift 设为 fail-visible 边界，不再通过通用 extension 或生产 mock fallback 猜测展示。
- 模型目录 taxonomy 现在可从 canonical catalog 补齐 task family、modality 和 runtime feature，同时保持 endpoint provenance 的严格校验。

### 测试与质量

- 扩展 direct TurnTimeline、历史 preview、恢复 replay、Pending Interaction、MCP elicitation、媒体异常态和协议漂移的前端与协议回归。
- 新增长列表 Electron fixture、canonical thread seed/oracle、Agent runtime screenshot 与 tool execution current contract，覆盖真实 preload/IPC、App Server JSON-RPC、read model 和 GUI 状态。
- 更新 Refactor v2 Item/Event projection inventory、render coverage fixture、legacy surface guard 和验证证据，守住单一 current owner 与零生产 mock fallback。

### 文档

- 更新 Renderer ConversationProjection、direct TurnTimeline、thread-scoped media read、异常态展示和长列表性能的架构事实源与 Refactor v2 执行记录。
- 新增 LimeCore / AsterRouter / Codex App Server 的云端多模型平台目标架构，明确商业控制面、模型数据面和桌面 runtime 的唯一 owner。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与锁文件版本统一提升到 `1.116.0`。

**完整变更**: `v1.115.0` -> `v1.116.0`
