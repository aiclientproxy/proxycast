# 验证与证据矩阵

> status: historical verification matrix
> owner: quality-workflow
> last_verified: 2026-07-30
>
> 本矩阵记录旧广义 Refactor v2 切片的验证等级和 dated evidence；当前渲染对齐执行事实以 `internal/refactor/v2/IMPLEMENTATION-PLAN.md`、`internal/aiprompts/quality-workflow.md` 与当前代码为准。

## 风险到门禁

| 变更                            | 最小验证                                                     | 产品证据                                                  |
| ------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| protocol/method/schema/scope    | `npm run test:contracts` + Rust protocol tests               | schema fixture diff、typed client round trip              |
| agent-runtime queue/turn/tool   | Rust related + `npm run smoke:agent-runtime-current-fixture` | accepted/queued/completed/failed/interrupted/resume       |
| Item/materialization/read model | projection unit + thread-store tests                         | create/update/remove/rollback/stale/pagination/repair     |
| provider/capability/lowering    | provider tests + contract tests                              | each protocol request/event/media/usage/error             |
| MCP/skills/multi-agent          | domain tests + current smoke                                 | snapshot/approval/edge/mailbox/recovery                   |
| Renderer projection             | Vitest + ESLint/typecheck                                    | Gate A screenshot/state assertions                        |
| Electron/Workspace/bridge       | `npm run verify:gui-smoke`                                   | Gate B real Electron/preload/IPC/App Server/read model/UI |
| i18n                            | `npm run i18n:check:json`                                    | five locale keys and visible states                       |
| legacy deletion                 | `npm run governance:legacy-report`                           | zero production references + negative guard               |
| script/package boundary         | `npm run governance:scripts`                                 | no unregistered root script/package                       |

## Evidence record format

每个切片新增不可变记录：

```yaml
slice: S1
owner: agent-runtime
workspace_commit: <lime commit or working-tree digest>
codex_commit: 5c19155cbd93bfa099016e7487259f61669823ff
opencode_commit: 9976269ab1accfc9f9dc98a4a688c516934de422
commands:
  - npm run test:contracts
result: pass|fail|blocked
scope: <paths>
positive_cases: <fixture ids>
negative_cases: <guard ids>
gate_a: <evidence path or n/a>
gate_b: <evidence path or n/a>
deleted_surfaces: <paths>
remaining_blocker: <none or exact reason>
```

证据失败必须保留原始输出摘要和复现命令；不能覆盖为“后续已通过”的历史记录。当前状态由最新汇总表引用记录。

## Gate A/B 规则

- Gate A 只证明 browser/renderer projection，不能证明 Electron。
- Gate B 必须真实经过 Electron、preload/IPC、`app_server_handle_json_lines`、App Server JSON-RPC、runtime/read model 和可见 UI。
- 生产路径不允许 mock fallback；测试 fixture 的 mock 必须显式标注且仍经过 current bridge。

## 2026-07-15 已验证切片

- [S6u Team memory / selection retirement](./2026-07-15-s6u-team-memory-selection-retirement.md)：Renderer/Rust/App Server 旧 Team memory shadow、selected-team metadata 和 recent selection 已删除。
- [S6v Workspace agentTeam settings retirement](./2026-07-15-s6v-workspace-agent-team-settings-retirement.md)：Workspace `agentTeam` DTO、serde 正向测试和 GUI fixture 已删除。
- [S6n2 Synthetic Team projection retirement](./2026-07-15-s6n2-synthetic-team-projection-retirement.md)：raw subagent replay、runtime Team snapshot、Team control synthetic event 已删除。
- [S4am spawn_agent V2 contract gap](./2026-07-15-s4am-spawn-agent-v2-contract-gap.md)：`fork_turns` lifecycle/lineage/返回错误补偿已完成复核：Rust/Renderer 参数边界一致，child history 只重建 completed final-answer canonical Item，source mapping、EventLog/sidecar 清理、稳定重试与 cold restart 均有回归。hard-crash pending-spawn recovery 仍是明确 blocker；`agent_type/model/reasoning_effort/service_tier` 继续 fail closed，拆给各自 current owner。
- [S4an dead subagent tool exposure](./2026-07-15-s4an-dead-subagent-tool-exposure.md)：零生产 caller 的旧 subagent fixed whitelist API、常量和正向测试已删除；current provider step snapshot 与 AgentControl gateway 不变。
- [S4ao spawn canonical path output](./2026-07-15-s4ao-spawn-canonical-path-output.md)：spawn result 的 `task_name` 对齐 Codex canonical full path；恒空 `nickname` 删除，durable `message_id` 保留。

本批次验证：`npm run typecheck`、`npm run test:contracts`、`npm run governance:legacy-report`、`npm run verify:gui-smoke`、projection/UI package tests、Rust workspace/evidence 定向测试均通过。

## 完成度口径

完成度以执行计划中的 slice 退出条件和证据记录为准；未满足退出条件的切片不得标记完成。当前批次已满足 S6u、S6v、S6n2 的最小验证门禁，后续切片继续沿用同一格式记录。

## 2026-07-30 V2-04 Host Capabilities

- [V2-04 Host Capabilities 与 Product-Scope Reverse Request](./2026-07-30-v2-04-host-capabilities-gate-b.md)：`currentTime/read`、`item/permissions/requestApproval`、`item/tool/call` 的 typed producer、exact waiter、Electron Host binding 与 DynamicTool canonical payload 已完成；DynamicTool 真实 Electron Gate B 通过，currentTime/permission 由 Rust/Electron 定向回归覆盖。收尾复核已通过全量 Vitest 112/112 批、`npm run test:contracts`、`npm run test:rust:changed`（因 lockfile 扩大为 workspace unit tests）、`npm run smoke:agent-runtime-current-fixture` 和 `git diff --check`；为复用 Vitest 续跑状态，`verify:local` 采用 app-version/i18n/lint/typecheck 子项加全量 Vitest 的等价组合，而未从首批重跑。
