# 上游 Coding 能力差距复核

> 状态：active
> 日期：2026-06-15
> 范围：复核 `/Users/coso/Documents/dev/rust/codex` 中与 coding 主线相关的执行、沙箱、审批、输出和工具编排能力，映射到 Lime current 事实源。

## 结论

Lime 的 coding 骨架已经不是缺“能不能改文件”的问题；P1-P4 current 主链、Workbench projection、policy metadata、output refs、patch/file/command/test facts、Windows sandbox readiness 控制面、P2-A 执行端有界输出捕获，以及 P2-B no-sandbox shell live process 都已落地。Windows restricted-token runner 仍未实现，继续对比后，真正高价值遗漏集中在两条骨干和一个策略体验增强：

1. **统一进程生命周期 owner**：上游有可 write / interrupt / terminate / stream / poll 的 unified exec process；Lime 已有 process owner、本地 runner、App Server `executionProcess/*` 控制面与 no-sandbox shell live process，但 command/test 默认执行尚未切到 sandbox-aware control owner。
2. **Windows sandbox 完整性**：Lime 已有 backend plan/readiness contract，但 restricted-token command runner 仍为 `Planned/enforced=false`，还没有可声称的 ACL、Job Object、pipe reader 或 Windows 平台 enforce evidence；实现 runner 后再补持久 capability SID、TokenDefaultDacl、扩展启动 handle allowlist/private desktop、read deny 和网络强制。
3. **审批缓存与重试体验**：Lime 已有 `action.required` 和多来源策略，但缺 session-scope approval key、sandbox denied 后安全升级重试和规则草案沉淀。

这些不是 UI polish，也不是旧路清理；它们直接决定 coding turn 在长命令、大输出、Windows 和审批重跑场景下是否可持续。

## 对比证据

| 上游参考能力          | 上游证据                                                                                                                                                                                 | Lime current 状态                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 缺口判断                                                                                                                                           |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 统一进程对象          | `core/src/unified_exec/process.rs` 暴露 write / terminate / interrupt / output receiver / state；`tools/runtimes/unified_exec.rs` 把审批、sandbox、network 与 process manager 接在一起。 | `tool-runtime::execution_process` 已提供 process snapshot、stdout/stderr delta、有界 retained output、stdin write、interrupt、terminate、status 与本地 process runner；App Server current 已提供 `executionProcess/start`、`writeStdin`、`interrupt`、`terminate`、`status`、`drainOutput` 控制面；no-sandbox shell path 已在 Lime preflight 与 Agent registry permission/safety preflight 之后走 live process；`executionProcess/start` 已收紧为受控 current 入口，workspace sandbox backend required / enforced 时 fail-closed，不再允许 `cwd` 覆盖 policy 判定后的实际工作目录。 | P2-B no-sandbox 路径和 App Server 受控启动已落地；缺口收缩为“command/test 默认执行切到 sandbox-aware process runner/control owner，并接 UI 控制”。 |
| Head/tail 输出缓冲    | `core/src/unified_exec/head_tail_buffer.rs` 在进程读取阶段限制保留字节，保留头尾并记录 omitted bytes。                                                                                   | `tool-runtime::execution_process::BoundedProcessOutput` 限制 live process retained output 并记录 omitted bytes；当前没有通用 sandbox output buffer，也没有 Windows restricted-token pipe reader。                                                                                                                                                                                                                                                                                                                                                                                   | P2-A 只完成 live process retained output；sandbox runner 输出流控必须在各 backend 执行边界复用同一语义。                                           |
| 审批缓存与重试        | `core/src/tools/sandboxing.rs` 有 approval cache、sandbox override、denied-read preservation；`tools/orchestrator.rs` 有 approval -> sandbox -> attempt -> denied retry。                | Lime 有 `ToolExecutionPolicyService` 多来源规则、`action.required`、审批后续跑测试和 sandbox blocked metadata。                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 缺“同一命令 approval key 复用 / sandbox denied 后升级重试 / proposed rule amendment”这一条统一执行语义。                                           |
| 持久 capability SID   | `windows-sandbox-rs/src/cap.rs` 按 workspace / writable root 持久化 SID，并用 canonical path key 去重。                                                                                  | Windows runner 未实现；当前只有 backend plan/readiness contract。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | runner 基线建立后才有 capability SID 与 ACL owner。                                                                                                |
| TokenDefaultDacl      | `windows-sandbox-rs/src/token.rs` 设置 token default DACL，避免受限 token 创建管道 / IPC 对象失败。                                                                                      | Windows runner 未实现；当前没有 restricted token 或 token DACL。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | runner 基线必须包含可供 PowerShell pipeline 与子进程 IPC 使用的 default DACL。                                                                     |
| 扩展启动信息          | `windows-sandbox-rs/src/process.rs` 使用 `STARTUPINFOEXW`、handle allowlist、`lpDesktop` 和 private desktop 选项。                                                                       | Windows runner 未实现；当前没有 Windows sandbox process launch owner。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | runner 需要明确 handle allowlist 与 desktop，不能依赖宽泛继承。                                                                                    |
| Read deny / 网络强制  | 上游 Windows sandbox 有 deny-read resolver / state、WFP setup、network proxy / approval cancellation。                                                                                   | Lime policy 能输出 network risk metadata，但 Windows runner 尚未实现，read/network deny 均未 enforce。                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 先完成 restricted-token 文件写边界，再分别补 read deny 与 network deny 的 Windows 平台证据。                                                       |
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
- sandbox backend 仍需在真实执行边界接入同一有界输出语义；不存在 Windows restricted-token pipe reader。
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

目标：从 `Planned/enforced=false` 的 fail-closed backend plan 实现到“Windows coding 日常可信”。

落点：

- `lime-rs/crates/tool-runtime/src/sandbox/windows.rs`
- `lime-rs/crates/tool-runtime/src/execution_process.rs`

动作：

- 先建立 Windows runner 执行边界，restricted-token backend 不再经过 argv wrapper lowering；未 enforce 时继续 fail closed。
- 创建 restricted token，为 workspace / writable root 建立可回滚 ACL，并将 backend plan 提升为真实 `Ready/enforced=true`。
- 持久化 workspace / writable root capability SID，按 canonical path key 管理；同时设置 TokenDefaultDacl。
- 使用 `STARTUPINFOEXW` + handle allowlist，并显式设置 desktop；stdout/stderr 必须复用有界输出 owner。
- 后续再补 read deny 与网络 deny；这两项需要单独 Windows 实机验证，不和本轮基础补丁混在一起。

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

最值得继续做 **P2-C Windows restricted-token runner 基线**。readiness 已进入 Desktop current 控制面，但 runner 仍为 `Planned/enforced=false`；下一步必须先在 `tool-runtime` 建立真实执行边界，再用 Windows 真机证明 workspace write、外部路径拒绝、ACL 恢复、timeout 终止进程树和大输出不死锁。P2-B 默认 command/test sandbox-aware process owner 随后继续，不得用 no-sandbox process 绕过该 blocker。
