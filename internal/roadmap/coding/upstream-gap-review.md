# 上游 Coding 能力差距复核

> 状态：active
> 日期：2026-06-15
> 范围：复核 `/Users/coso/Documents/dev/rust/codex` 中与 coding 主线相关的执行、沙箱、审批、输出和工具编排能力，映射到 Lime current 事实源。

## 结论

Lime 的 coding 骨架已经不是缺“能不能改文件”的问题；P1-P4 current 主链、Workbench projection、policy metadata、output refs、patch/file/command/test facts、Windows sandbox setup/readiness、双 mode restricted-token runner、P2-A 执行端有界输出捕获，以及 P2-B no-sandbox shell live process 都已落地。Windows runner schema v3 `7/7` 与复验 run 已形成平台 evidence。继续对比后，真正高价值遗漏集中在两条骨干和一个策略体验增强：

1. **统一进程生命周期 owner**：上游有可 write / interrupt / terminate / stream / poll 的 unified exec process；Lime 已有 process owner、本地 runner、App Server `executionProcess/*` 控制面与 no-sandbox shell live process，但 command/test 默认执行尚未切到 sandbox-aware control owner。
2. **Windows sandbox 完整性**：Lime 已有 backend plan/readiness contract，以及 elevated sandbox-account、unelevated current-user restricted-token runner、短生命周期 ACL lease、TokenDefaultDacl、Job Object、ConPTY、显式 handle allowlist、Firewall/WFP 和有界 pipe reader；Windows runner schema v3 `7/7` 与复验 run 已完成当前平台 evidence。后续仅在 token/ACL/network 改动时回归。
3. **审批缓存与重试体验**：Lime 已有 `action.required` 和多来源策略，但缺 session-scope approval key、sandbox denied 后安全升级重试和规则草案沉淀。

这些不是 UI polish，也不是旧路清理；它们直接决定 coding turn 在长命令、大输出、Windows 和审批重跑场景下是否可持续。

## 对比证据

| 上游参考能力          | 上游证据                                                                                                                                                                                 | Lime current 状态                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 缺口判断                                                                                                                                           |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 统一进程对象          | `core/src/unified_exec/process.rs` 暴露 write / terminate / interrupt / output receiver / state；`tools/runtimes/unified_exec.rs` 把审批、sandbox、network 与 process manager 接在一起。 | `tool-runtime::execution_process` 已提供 process snapshot、stdout/stderr delta、有界 retained output、stdin write、interrupt、terminate、status 与本地 process runner；App Server current 已提供 `executionProcess/start`、`writeStdin`、`interrupt`、`terminate`、`status`、`drainOutput` 控制面；no-sandbox shell path 已在 Lime preflight 与 Agent registry permission/safety preflight 之后走 live process；`executionProcess/start` 已收紧为受控 current 入口，workspace sandbox backend required / enforced 时 fail-closed，不再允许 `cwd` 覆盖 policy 判定后的实际工作目录。 | P2-B no-sandbox 路径和 App Server 受控启动已落地；缺口收缩为“command/test 默认执行切到 sandbox-aware process runner/control owner，并接 UI 控制”。 |
| Head/tail 输出缓冲    | `core/src/unified_exec/head_tail_buffer.rs` 在进程读取阶段限制保留字节，保留头尾并记录 omitted bytes。                                                                                   | `tool-runtime::execution_process::BoundedProcessOutput` 限制 live process retained output并记录 omitted bytes；Windows restricted-token blocking pipe reader 已复用同一 owner。                                                                                                                                                                                                                                                                                                                                                                                                        | current foundation 已覆盖 Windows 有界读取；仍需 Windows 真机大输出证据。                                                                          |
| 审批缓存与重试        | `core/src/tools/sandboxing.rs` 有 approval cache、sandbox override、denied-read preservation；`tools/orchestrator.rs` 有 approval -> sandbox -> attempt -> denied retry。                | Lime 有 `ToolExecutionPolicyService` 多来源规则、`action.required`、审批后续跑测试和 sandbox blocked metadata。                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 缺“同一命令 approval key 复用 / sandbox denied 后升级重试 / proposed rule amendment”这一条统一执行语义。                                           |
| 持久 capability SID   | `windows-sandbox-rs/src/cap.rs` 按 workspace / writable root 持久化 SID，并用 canonical path key 去重。                                                                                  | current foundation 使用每次运行创建的 capability SID 与短生命周期 ACL lease，尚未持久化。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 在真机验证临时 lease 正确后，再裁决是否需要持久 SID owner。                                                                                         |
| TokenDefaultDacl      | `windows-sandbox-rs/src/token.rs` 设置 token default DACL，避免受限 token 创建管道 / IPC 对象失败。                                                                                      | current foundation 已设置 restricted token default DACL，但仅有非 Windows 测试和 SDK 源码核对。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 需要 Windows 真机 PowerShell pipeline 与子进程 IPC 证据。                                                                                           |
| 扩展启动信息          | `windows-sandbox-rs/src/process.rs` 使用 `STARTUPINFOEXW`、handle allowlist、`lpDesktop` 和 private desktop 选项。                                                                       | current foundation 已使用 `STARTUPINFOEXW`、Job Object attribute 与显式 handle allowlist；private desktop/ConPTY 尚未实现。                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 先验证当前启动边界，再单独裁决 private desktop 与 ConPTY。                                                                                          |
| Read deny / 网络强制  | 上游 Windows sandbox 有 deny-read resolver / state、WFP setup、network proxy / approval cancellation。                                                                                   | current foundation 已覆盖 workspace/explicit write root 和 `.git/.codex/.agents` write deny；read deny 与 WFP/network deny 未 enforce。                                                                                                                                                                                                                                                                                                                                                                                                                                              | 文件写边界需先取得 Windows 平台证据，再分别补 read deny 与 network deny。                                                                           |
| 远程 exec server / FS | 上游 `exec-server` 有 remote process / file system / relay。                                                                                                                             | Lime 外部 harness 只能作为 compat adapter，主事实源是 App Server / RuntimeCore。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 非当前主线 blocker；只有需要远程 workspace coding 时才进入 P6 current adapter。                                                                    |

## 高价值补齐顺序

### P2-A：执行端输出流控前移

目标：在 executor / tool outcome 进入 RuntimeCore 前就限制内存增长。

状态：`partial / live-process retained output only`。

落点：

- `lime-rs/crates/tool-runtime/src/execution_process.rs`
- `lime-rs/crates/app-server/src/runtime/output_refs.rs`

已落动作：

- live process retained output 已使用有界 buffer，并输出 `outputBytes / outputOmittedBytes / outputTruncated`。
- Windows restricted-token blocking pipe reader 已接入同一有界输出语义；仍缺 Windows 真机大输出和 omitted-bytes 证据。
- RuntimeCore 继续负责 snapshot；executor 不直接写 App Server sidecar，避免跨层依赖。

收益：解决大输出在进入 output ref 之前压爆内存的问题，直接服务 command/test coding 主线。

### P2-B：统一 live process lifecycle

目标：让 command/test execution 从“批处理终态”升级为“可观察、可中断、可续写”的进程对象。

状态：`in_progress / no-sandbox live process done`。

落点：

- `lime-rs/crates/agent/src/agent_tools/execution/process.rs`
- `lime-rs/crates/app-server/src/runtime_backend/coding_events/command.rs`
- `packages/agent-runtime-projection/src/coding.ts`

动作：

- 已定义 current process owner：start snapshot、stdout/stderr output delta、bounded retained output、stdin write、interrupt、terminate、status、本地 process runner。
- 已把现有 shell batch bridge 接到 process metadata：`processId / executionProcessStatus / outputBytes / outputOmittedBytes / outputTruncated` 透传到 `tool.output.delta` / `command.output` metadata。
- 已通过 App Server current JSON-RPC 暴露 `executionProcess/start|writeStdin|interrupt|terminate|status|drainOutput`，并同步 protocol schema、processor、client 与 contract guard。
- no-sandbox shell path 已接入 live process：先过 `ToolExecutionDecision`，再复用 Agent `ToolRegistry::check_tool_permissions`，且需要 workspace sandbox backend 的命令继续走 Agent sandbox executor。
- 下一刀把 command/test 默认执行切到 sandbox-aware `LocalExecutionProcessHandle` / execution process control owner，并让 Workbench UI 的停止、输入和状态刷新复用同一 current API。

收益：长任务、用户中断、实时日志、测试服务器、交互式 shell 才能成为产品能力，而不是一次性命令结果。

### P2-C：Windows restricted token 完整性补齐

目标：维护已由 Windows runner evidence 证明的 current restricted-token backend，并确保 `elevated/unelevated` setup mode 与实际执行路径一致。

落点：

- `lime-rs/crates/tool-runtime/src/execution_process/windows.rs`
- `lime-rs/crates/tool-runtime/src/execution_process/windows_acl.rs`
- `lime-rs/crates/tool-runtime/src/execution_process/windows_attr.rs`
- `lime-rs/crates/tool-runtime/src/execution_process.rs`

动作：

- 已建立专用 Windows runner、restricted token、TokenDefaultDacl、短生命周期 ACL lease、`STARTUPINFOEXW`、Job Object、handle allowlist 和有界 stdout/stderr owner；未取得平台证据时继续 fail closed。
- 先在 Windows/MSVC toolchain 完成编译，再用真机覆盖 workspace write、外部路径拒绝、metadata write deny、ACL rollback、large output 与 timeout descendant kill。
- 平台证据完整后才允许把 backend plan 提升为真实 `Ready/enforced=true`，并继续验证 PowerShell pipeline 与子进程 IPC。
- 后续再裁决持久 capability SID、private desktop/ConPTY、read deny 与网络 deny；这些能力分别取证，不和基础 readiness 提升混在一起。

收益：降低 Windows PowerShell / 子进程 / 工具链执行失败概率，为 read-only / workspace-write 提供更完整边界。

### P2-D：审批缓存、重试与规则草案

目标：把当前 `action.required` 从“可确认”推进到“确认后同类命令可复用、sandbox denied 可安全升级重试”。

落点：

- `lime-rs/crates/agent/src/agent_tools/execution/decision.rs`
- `lime-rs/crates/agent/src/agent_tools/execution/service.rs`
- `src/components/settings-v2/system/execution-policy/**`

动作：

- 为 shell command 建稳定 approval key：canonical command + cwd + sandbox policy + requested permissions。
- 支持 session-scope approval cache，不写入全局配置。
- 当用户选择持久化规则时，只生成 settings 草案，仍通过 current 配置写链保存。

收益：减少重复审批，同时避免把一次性确认误写成永久放行。

## 暂不优先

| 能力                                  | 原因                                                                                                               |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 远程 exec server 全量迁移             | Lime 主线是 App Server current 本地/桌面 Workbench；远程 workspace 需要单独 P6 adapter，不应阻塞当前 coding 骨架。 |
| 终端/TUI 视觉复制                     | Lime 是 GUI 桌面产品，终端 UI 只能借鉴信息层级，不能成为 current surface。                                         |
| 外部工具市场/插件协议                 | Lime 已有 Tool inventory / MCP owner；除非进入多租户工具分发，不应扩大 coding 主线范围。                           |
| 全量 Windows WFP / deny-read 一次完成 | 风险和验证成本高，应在 P2-C 基础稳定后分两刀落地。                                                                 |

## current / compat / deprecated / dead 分类

| 类型       | 路径 / 能力                                                                                                    | 说明                                                                |
| ---------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| current    | App Server JSON-RPC + RuntimeCore + ExecutionBackend + AgentUI projection + Coding Workbench                   | 后续 coding 能力只向这里收敛。                                      |
| current    | `patch-apply` crate、runtime output refs、file checkpoint refs、policy service、workspace sandbox backend plan | 已经是 Lime current owner，继续小步补强。                           |
| compat     | 外部 CLI / harness adapter                                                                                     | 只能输出 RuntimeEvent / ReadModel adapter，不允许成为生产必需主链。 |
| deprecated | 旧 thread item 推断 coding 状态、旧 `code_orchestrated` 入口                                                   | 只允许历史 hydrate / compat 归一，不允许新增状态逻辑。              |
| dead       | `lime-rs/src/**`、旧 Tauri command wrapper、生产 mock fallback                                                 | 不得恢复。                                                          |

## 下一刀建议

**P2-C Windows restricted-token runner 基线已完成。** readiness、setup mode、elevated sandbox-account、unelevated current-user restricted-token runner、ACL/Job/ConPTY、Firewall/WFP、warning lifecycle 和 readiness read-back 已收敛到 `tool-runtime` current owner；Windows runner schema v3 `7/7`、quality run 与 readiness 复验 run 是平台 evidence，不能用 macOS 结果替代。下一刀回到 P2-B 默认 command/test sandbox-aware process owner，不得用 no-sandbox process 绕过 sandbox。
