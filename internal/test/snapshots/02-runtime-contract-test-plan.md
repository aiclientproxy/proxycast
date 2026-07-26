# Runtime / Contract 快照测试方案

Codex 的 Core、App Server suite、streaming 和 CLI 快照不能全部转为前端视觉测试。它们应先
固化为 Lime current owner 的结构化 contract、integration 或 projection test，再由 GUI 场景
消费结果。

## P0：Thread/Turn/Item 与恢复

| ID | 来源主题 | Lime owner | 断言 |
| --- | --- | --- | --- |
| `runtime-compaction` | manual/pre-turn/mid-turn/remote/resume-fork compaction | `agent-runtime` + App Server + ThreadStore | replacement history、window lineage、incoming message 和 rollback 语义稳定 |
| `runtime-pending-input` | queued mail、reasoning 后 user input、不抢占 | `agent-runtime` / queue | pending input 顺序、turn identity、继续执行状态一致 |
| `runtime-model-layout` | model switch、cwd、personality、subagent visible layout | `model-provider` + session config | 每 Turn 固化 route/model/reasoning；switch 原子提交；read model 与 UI 同源 |
| `runtime-context-budget` | additional context、token budget、context window | `agent-runtime` | context 预算、截断和错误是 typed 状态，不是字符串推断 |
| `runtime-history-fork` | fork startup context、rollback、resume | App Server + ThreadStore | parent/child lineage、Item 顺序、冷启动恢复一致 |

## P0：工具、MCP 和 reverse request

| ID | 来源主题 | Lime owner | 断言 |
| --- | --- | --- | --- |
| `runtime-mcp-exposure` | deferred tool expose/recover/resume | `tool-runtime` + MCP owner | 同一 snapshot 的 tool exposure 不漂移；更新不重复 |
| `runtime-mcp-elicitation` | schema/form/session persist | `lime-mcp` + typed responder | request scope、field schema、response action 和 outer request id 匹配 |
| `runtime-tool-output` | command output、stderr、tool terminal | `tool-runtime` + Item projection | output delta、terminal、interrupted 和 error 都有 canonical Item |
| `runtime-approval` | guardian/network/permissions/approval | App Server reverse request | outer JSON-RPC id 与 domain request id 不混用；settle 一次 |
| `runtime-multi-agent` | spawn/wait/followup/interrupt/list | `agent-runtime` AgentControl | sender/receiver thread lineage、mailbox、worker terminal 和 restart 稳定 |

## P1：Provider、环境和错误

| ID | 来源主题 | Lime owner | 断言 |
| --- | --- | --- | --- |
| `runtime-provider-lowering` | remote request diff、prompt cache、service tier | `model-provider` | structured request/lowering 可审计；未知 protocol fail closed |
| `runtime-environment-context` | world state、permissions、plugins、realtime | runtime context owner | 环境/权限/插件说明只从结构化 facts 注入，不从自然语言猜测 |
| `runtime-safety-errors` | sandbox, policy, oversized input, expired signature | `tool-runtime` / App Server | typed failure、retryability 和 GUI error surface 一致 |
| `runtime-diagnostics` | doctor、hooks、MCP startup、connectivity | App Server diagnostics | 诊断进入 evidence/read model，不泄露 secrets |

## 多模型 contract oracle

多模型 contract 以 `grok-build@6e386420825bd44ae648c63e7c8cba12fcec9401` 为语义事实源，
不使用 Codex TUI picker 文案裁决协议：

| Case | grok-build oracle | Lime contract 断言 |
| --- | --- | --- |
| `model-catalog-precedence` | `xai-grok-models/src/lib.rs`、`agent/models.rs` | catalog/default 来源有确定优先级；真实 catalog 不可用时状态可区分；未知模型 fail closed |
| `model-effort-remap` | `pager/src/acp/model_state.rs`、`reasoning_efforts_menu_renders_and_remaps_on_wire.rs` | 选项 id/label/value 分离；请求只发送目标模型声明的 canonical value |
| `model-switch-atomic` | `agent/handlers/model_switch.rs`、`session/acp_session_impl/model_switch.rs` | model、reasoning、context window、compaction/capability 与持久化状态同批收敛；失败不留下半切换状态 |
| `model-switch-compatibility` | `agent/handlers/model_switch.rs` | 若目标 model 需要不兼容 runtime profile，已有 turn 时 typed reject；零 turn 仅在 owner 能完成重建时切换 |
| `provider-readiness` | `agent/models.rs` | allowlist、认证、capability 与 readiness 由结构化 catalog/facts 决定，不按名称放行 |
| `provider-circuit-breaker` | `xai-circuit-breaker/src/breaker.rs`、`retry_policy.rs` | retry budget、open/half-open/closed 转换可审计；用户锁定模型不被静默 fallback 改写 |

上述 contract 必须在 Lime current App Server/model-provider owner 实现；前端 mock catalog 只能作为
component fixture，不能作为 provider readiness 或 model switch 成功证据。

## Deferred / 不进入 Lime 前端 current

- Codex CLI doctor 的终端报告布局。
- TUI terminal title、Zellij raw terminal、光标和 ANSI 背景细节。
- ChatGPT 专属 prompt、登录、usage/credits 文案，除非 Lime 已有对应产品 owner。
- 仅用于 upstream Rust prompt/request golden 的完整模型请求文本；Lime 使用结构化 request
  capture 和 marker，不保存 secret 或完整用户 prompt。

## 验证顺序

```text
Rust/domain related
  -> App Server public JSON-RPC integration
  -> current runtime fixture
  -> Renderer projection/component
  -> Gate A
  -> Gate B Electron
```

不得用 `npm run test:e2e` 单独声称 Electron Gate B；不得用 GUI mock fallback 覆盖 contract
缺口。相关入口以 `internal/aiprompts/quality-workflow.md` 为准。
