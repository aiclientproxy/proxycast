## Lime v1.130.0

### 新功能

- 为 Responses provider 增加按 Turn 隔离的 WebSocket 会话，支持严格前缀校验、`previous_response_id` 增量请求、prompt cache key 与完整请求回退。
- 将异步 Hook 结果接入 session-owned 生命周期：活动回合使用 steer，空闲回合进入 mailbox，并投影 Started/Completed、warning 与取消状态。
- 将音频转写接入 App Server transcription worker，消费 `openai_audio_transcription` route，生成 workspace transcript artifact 并回写 provider diagnostics 与任务状态。

### 修复

- 修复标准 `AGENTS.md` / `AGENTS.override.md` 的全局与工作区发现优先级，同时保留 `.lime` 规则作为受控 fallback，避免运行时遗漏项目指令。
- 修复 MCP 工具并发判断，只有显式 `read_only_hint=true` 且 server opt-in 的工具才进入并发路径，缺失或 false 继续串行。
- 修复 session mailbox、turn lifecycle、external backend 事件和 ProjectThread-first 边界中的恢复与状态投影问题。

### 优化与重构

- 在 model-provider current client 内收敛请求 lowering、响应增量状态、WebSocket 连接复用和媒体输入处理，保持 provider 网络边界单一。
- 将 Hook discovery、runtime 与 lifecycle 拆分为可观察的 tool-runtime owner，清理重复兼容路径与已删除的 evidence export 测试入口。
- 更新多模态执行合同、媒体任务索引、Harness 架构图、治理 catalog、Claw fixture 与五语言资源，使转写和 runtime facts 使用同一 current 主链。

### 测试与质量

- 补充 Responses 增量请求、模型/工具/指令变化回退、MCP 只读并发、标准 AGENTS 发现、异步 Hook mailbox 和 session-loop 的 Rust 回归。
- 更新媒体任务、runtime facts、ProjectThread-first、modality、治理与 current fixture 契约；最终门禁结果以本发布执行计划为准。

### 文档

- 更新架构、Playwright E2E、模型能力、Skills、Workflow、Warp 和 Agent Runtime 路线图文档，明确转写 worker 与 provider route 的 current owner。
- 新增 `v1.130.0` 发布执行计划并维护执行计划导航；补充 Lime Agent Harness current 架构图源文件与渲染资产。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.130.0`。
- Windows/macOS packaged parity、签名、公证和正式发布资产证据不在本地候选中冒充完成，待 CI/平台流程提供证据。

**完整变更**: `v1.129.0` -> `v1.130.0`
