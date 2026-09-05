# Code Mode Codex 架构对齐计划

状态：complete（stdio/gRPC 主链、typed content、UUID 边界、session lease、execution
reservation 和双 transport 重连 binding 已接通；本机 V8 相关 crate 验证已通过，真实
host 断线后的 pending cleanup 与 generation 重绑已由 process integration 覆盖）

## 目标

将 Lime 的 Code Mode 从 `tool-runtime`/`agent-runtime` 内嵌实现收敛到与
`/Users/coso/Documents/dev/rust/codex/codex-rs` 对齐的四层 crate 边界：

```text
code-mode-protocol -> code-mode-runtime -> code-mode-host
                   \-> code-mode (session facade)
```

Agent Runtime 只保留会话编排和工具 handler；协议、V8 执行、进程 host/client
不再新增在 Agent Runtime 内。

## 窄写集

- `lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`
- `lime-rs/crates/code-mode-protocol/**`
- `lime-rs/crates/code-mode-runtime/**`
- `lime-rs/crates/code-mode-host/**`
- `lime-rs/crates/code-mode/**`
- `lime-rs/crates/tool-runtime/src/code_mode*`（仅公开边界与兼容导出）
- `lime-rs/crates/agent-runtime/src/code_mode*`、相关 import（迁移调用）
- `lime-rs/crates/app-server/src/processor/dispatch/v2_ingress.rs`、
  `lime-rs/crates/app-server/src/processor/tests/command_exec.rs`（command/exec 权限边界回归）
- 本计划与 `internal/aiprompts/architecture.md`

不修改当前 CLI/TUI、Electron、发布脚本及其并行写集。

## 阶段

1. 建立四个独立 crate 和 workspace 依赖，暴露与 Codex 对齐的 protocol、runtime、host、facade 模块。
2. 将现有 stdio 协议、V8 runtime、进程 session provider 接入新 crate，保持真实执行链。
3. 把 Agent Runtime 的生产 factory 和 provider-turn 调用迁到 `code-mode` facade。
4. 增补 crate 级定向测试、治理扫描和锁文件；确认旧路径仅为显式兼容导出。
5. 用户已确认删除旧实现；`tool-runtime/src/code_mode/**` 旧 process/V8 文件、host binary 已物理删除，空目录已清理。
6. 按 Codex 目录补入协议的 `description/response/runtime/session/json_schema_types/grpc` owner，以及 host/facade 的 gRPC transport；当前实现已覆盖真实 OpenSession、Subscribe、Execute、Wait、Terminate、delegate completion 和 CloseSession 链。
7. `code-mode-protocol/src/lib.rs` 已从约 790 行收敛为约 100 行模块索引；host wire 定义已迁入 `host/message.rs`、`payload.rs`、`types.rs` owner，`host/mod.rs` 只做边界导出。
8. `code-mode/src/grpc_session.rs` 已迁为与 Codex 同名目录，拆出 callbacks、completion、conversion、deadline、generation、operations、reconnect、state、transport；gRPC stream 断开会退休 binding，下一次操作由单飞 coordinator 重建 session。
9. Runtime response 已从 `output: String` 迁为 canonical typed `content_items`，text/image/audio 经 V8 runtime、stdio、gRPC host/client 保真传递；Agent Runtime 仅在最终 tool result 边界投影文本。
10. gRPC host/client 已统一使用 UUID identity，并补齐 identifier/tool-filter/schema/tool-kind
    边界校验、typed content/duration conversion、精确订阅路由、close-pending 清理和断线时
    pending callback 取消；notification 也通过 typed ack/cancel 完成；OpenSession stream 丢弃和
    CloseSession 都会回收 session lease，execution ID 先 reservation/去重再 runtime admission；
    reconnect 在新 generation 发布前退休旧 binding。
11. process-owned facade 已按 gRPC reconnect owner 收敛：共享 `ProcessHost` 复用存活的
    stdio sidecar，`ReconnectableSession` 以单飞 opening coordinator 管理
    `SessionBinding(connection, session_id, generation)`；连接死亡后先 best-effort 关闭旧
    binding，再创建新 host/session。首代继续暴露纯 cell ID，后续代际使用 `gN:<remote>`，
    execute initial response、wait/terminate outcome 和 delegate callback 均走同一映射。
12. process connection 目录补齐 Codex 的 `reader` owner；driver session registry 记录
    active cells，initial/wait/terminate 返回值执行 identity 校验，连接失败和 session close
    会向 delegate 发出一次性 `cell_closed`，旧代 cell 不会泄漏到新 binding。
13. process `session/open` 仅在非默认资源限制时发送 `cellExecutionLimits`，与 Codex
    command lowering 保持一致；默认会话继续由 host/runtime 使用默认限制。
14. process connection driver 已按 Codex owner 真实拆分：`state` 只组合连接生命周期，
    `commands` 负责请求登记与 caller cancellation，`responses` 负责 host response 路由，
    `request_tracker` 管理 pending promises，`session_registry` 管理 session/active cells，
    `delegate_runtime` 管理 callback cancellation，`cleanup` 统一 cell close 通知；这些模块
    不再只是目录占位或重复协议类型。
15. `driver.rs` 已成为真实 event driver：`reader` 只解码并投递 `DriverEvent`，driver 串行
    调用 response owner、收敛失败和取消；ProcessConnection 不再由 reader 直接执行状态迁移。
16. driver response 边界对 delegate 和 `cell/closed` 均校验 active cell；stale/duplicate
    cell 事件会 fail closed，并由专门回归测试锁定。
17. `delegate_runtime` 记录 callback 所属 active cell；cell close/session close/connection
    failure 会撤销对应 callback cancellation token，避免 stale delegate future 继续回写。

## 退出条件

- workspace manifest 中存在 `code-mode-protocol`、`code-mode-runtime`、`code-mode-host`、`code-mode`。
- `code-mode-host` 提供 stdio host binary，`code-mode` 提供 process-owned session provider。
- V8 的 exec/wait/store/load/nested dispatch/取消/终止行为由新 runtime crate 的真实实现承接。
- Agent Runtime 不再直接依赖 process protocol 私有模块。
- 受影响 Rust crate 定向检查通过，或记录明确的 `rusty_v8` 环境阻塞。

## 当前分类

- 新四 crate：`current`。
- `tool-runtime::code_mode`：迁移期间 `compat`，只允许委托到 current owner。
- 旧物理实现：`dead / deleted / forbidden-to-restore`；只允许在负向治理 evidence 中出现。
- `code-mode-protocol` 的 gRPC schema/build 及 `code-mode-host`/`code-mode` 的 gRPC transport：`current`。
- `code-mode` 的 process host/session reconnect、generation projection：`current`。

## 验证

```bash
cargo check --manifest-path "lime-rs/Cargo.toml" -p code-mode-protocol -p code-mode-runtime -p code-mode-host -p code-mode
cargo test --manifest-path "lime-rs/Cargo.toml" -p code-mode-protocol -p code-mode-runtime -p code-mode-host -p code-mode
npm run governance:legacy-report
npm run test:rust:related -- lime-rs/crates/code-mode-protocol lime-rs/crates/code-mode-runtime lime-rs/crates/code-mode-host lime-rs/crates/code-mode
```

## 最终证据与环境说明

- gRPC host/client 的 in-flight execute/wait、resource-limit admission、close-pending、
  OpenSession lease 回收、重复 execution ID、精确路由和 typed conversion 已由 host service/
  robustness 测试及真实 TCP facade 测试覆盖；tool/notification ack-cancel 语义也已接通；
  generation parser 已覆盖旧 cell 拒绝。
- process host integration 已覆盖真实 `code-mode-host` stdio 子进程断线：pending wait fail-closed、
  active cell 单次 `cell_closed`、下一次操作自动重建 host/session 并递增 generation、旧 cell
  ID 拒绝和新 generation execute 成功。
- 在本机可用的 V8 archive/binding 注入下，`agent-runtime` 全量 library 211 tests、
  `code-mode` 24 unit + 3 gRPC integration、`code-mode-host` 23、
  `code-mode-protocol` 5、`code-mode-runtime` 10 全部通过。
- `app-server` 的 `command/exec` 权限边界现由 v2 ingress 在 typed lowering 前拒绝客户端
  `grantedPermissions`，handler 与 JSON-RPC 集成回归共 3 项通过并稳定返回 `-32602`。
- `npm run governance:legacy-report`、`npm run test:contracts`、`cargo fmt -- --check` 与
  `git diff --check` 通过；默认 v8.150.4.0 Darwin archive URL 仍返回 HTTP 404，验证继续使用
  成对注入的本机 archive/binding。

## 最近验证（2026-09-05）

- `cargo test --manifest-path "lime-rs/Cargo.toml" -p code-mode-host --no-fail-fast`
  通过：23 tests，包含 close-pending 与 terminate 的 tool/notification typed cancellation。
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p code-mode --test grpc`
  通过：3 tests（真实 TCP gRPC nested tool/notification、started-cell 回收、terminate
  callback 取消）。
- 之前已通过五 crate 定向测试：`code-mode` 4 unit + 3 integration、`code-mode-host` 15
  unit、`code-mode-protocol` 5、`code-mode-runtime` 3；本轮 host 生命周期/去重改动后 host
  测试增至 23。
- `npm run governance:legacy-report`：边界违规 0，分类漂移候选 0；`git diff --check` 与
  `cargo fmt` 通过。
- process facade generation regression：首代/重连代际 cell ID、initial/wait projection 和
  delegate callback 映射测试通过；四个 Code Mode crate 组合测试通过（24 code-mode unit、
  3 gRPC integration、23 host、5 protocol、10 runtime）。
- process driver lifecycle regression：active-cell registry、断线 cleanup、重复关闭防护和
  initial response identity 校验测试通过；`remote_session/connection/reader.rs` 已进入 current
  编译图。
- process driver owner split：`state`/`commands`/`responses`/`request_tracker`/`session_registry`/
  `delegate_runtime`/`cleanup` 已进入真实编译图；driver lifecycle 测试迁移到
  `remote_session::connection::driver::tests` 并保持通过。
- process delegate lifecycle regression：active-cell delegate callback 在 `cell/closed` 时被
  撤销；stale delegate/cell event fail-closed 测试通过。
- process host reconnect integration：真实 stdio host 断线后 pending request、active-cell cleanup、
  generation 重绑和 stale cell rejection 测试通过。
- upstream consumer regression：`tool-runtime` library 361 tests、`agent-runtime` library 211
  tests 全部通过；`provider_turn::code_mode` 已公开转发 current `CodeModeToolKind`，消除迁移
  后 typed tool 测试的模块路径断裂。
- upstream related validation：`agent-runtime` 全量 211 tests 在收紧 current import、保留测试路径
  re-export 后通过；App Server `command/exec` 权限边界回归已在 v2 ingress 收口并通过。
- App Server command/exec boundary regression：v2 ingress、handler 与 JSON-RPC integration
  三项 `grantedPermissions` rejection tests 全部通过，unknown field 不再被 typed lowering
  静默丢弃。
