# Agent 未完成会话恢复计划

> 状态：implemented / Gate B verified
> 更新：2026-08-21
> 主链：Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item -> GUI
> 参考：`/Users/coso/Documents/dev/rust/codex`

## 1. 目标

桌面端必须用真实执行 owner 判断 Turn 是否仍在运行，不能仅凭持久历史中的
`Active / InProgress`、时间戳或 Renderer 本地状态恢复“运行中”。

本计划覆盖：

- Renderer reload 后继续观察仍由当前 App Server 持有的 live Turn。
- Electron/App Server restart 后，把没有 live owner 的孤儿 `InProgress` Turn 只在读模型中投影为 `Interrupted`。
- 首页、侧栏、会话详情和输入框消费同一 Thread/Turn 投影。
- 真实 Electron Gate B 同时验证 preload/IPC、`app_server_handle_json_lines`、App Server、read model 和 GUI。

不覆盖 live Provider 质量、网络断线重连、媒体任务和依赖 Docker/Pier 的 DeepSWE separate verifier。

## 2. 唯一事实源

current 读取链固定为：

```text
Renderer typed gateway
  -> Electron preload / IPC
  -> app_server_handle_json_lines
  -> thread/list | thread/read | thread/turns/list | thread/resume
  -> RuntimeCore + ProjectionStore + session_loops
  -> Thread / Turn / Item projection
  -> home / sidebar / detail / inputbar
```

执行与历史分别有明确 owner：

- live execution owner：`session_loops.active_turn_id`，只存在于当前 App Server 进程。
- durable history owner：canonical Thread/Turn/Item 与 ProjectionStore；它保留实际写入的 `Active / InProgress` 历史，不因读取而改写。
- response/read projection：结合 durable history 与 live owner 生成当前可展示状态。
- Renderer：只消费投影；`isSending / activeStream` 只能表达当前页面的瞬时交互。

禁止恢复旧 `agentSession/*` production method、30 分钟时间戳启发式、Renderer mock fallback、App Server mock backend 或第二套 Electron 业务后端。

## 3. 状态语义

| 场景 | live owner | Thread 读状态 | 最新 Turn 读状态 | GUI 行为 |
| --- | --- | --- | --- | --- |
| 正常运行 | 同一 `active_turn_id` | `active` | `inProgress` | 显示运行态与停止按钮 |
| Renderer reload | owner 仍在 | `active` | `inProgress` | `thread/resume` 续接同一 Turn 的订阅 |
| Electron/App Server restart | owner 不存在 | `idle` 或 `notLoaded` | `interrupted` | 不显示运行态，不发送伪取消 |
| 正常终态 | owner 不存在 | `idle` 或 `notLoaded` | `completed/failed/interrupted` | 输入框恢复可用 |

固定规则：

1. `thread/read`、`thread/list`、`thread/turns/list` 都以同一个 live owner 判断运行态。
2. 无 live owner 的 `InProgress` 只在 response/read model 中映射为 `Interrupted`。
3. 冷读不写入 `turn.completed`、`turn.canceled` 或其它 synthetic terminal event。
4. `SessionLoadContext.stored` 保留 durable/operational 原始状态，供 AgentControl、mailbox、workflow 和显式恢复使用；只归一化返回给读取者的 snapshot。
5. 未加载 Thread 投影为 `notLoaded`；已加载但没有 active Turn 投影为 `idle`。
6. terminal Turn 即使残留历史 `active_turn_id` 字段，也不能重新判成 running。

## 4. 产品行为

### 4.1 Renderer reload

Renderer reload 不终止 App Server sidecar。产品可以在首页保留轻量恢复卡，侧栏和详情继续显示同一 running Turn；用户打开详情后仍可停止该 Turn。`thread/resume` 只恢复当前连接的事件订阅，不创建新 Turn，也不修改 durable history。

### 4.2 Electron/App Server restart

进程重启后旧 session loop 已不存在，因此不能声称后台继续执行：

- 首页不展示“正在继续”的 running 恢复卡。
- 侧栏、详情和输入框全部进入非 running 状态。
- 原 Turn 历史仍可见，但状态为 `interrupted`。
- 不自动调用 `thread/resume` 冒充续跑。
- 不显示停止按钮，不发送 `turn/interrupt`。
- 不向已退出 external backend 发送 cancel，也不生成 `turn.canceled` 事件。

用户后续可在同一 Thread 显式发送新输入；那是新 Turn，不是恢复旧执行。

### 4.3 多未完成会话

同一 App Server 内多个 live owner 可以同时在侧栏显示 running。restart 后必须逐个重新核对 owner；所有没有 live owner 的旧 `InProgress` Turn 都投影为 interrupted，不能只修正当前打开的会话。

## 5. 实现切片

### P0：读模型 owner 收敛

- `runtime/status.rs` 删除基于更新时间的 stale running 启发式。
- `thread_read.rs` 的 read/list/turns-list 使用 `session_loops.active_turn_id`。
- `read_model.rs`、`load_context.rs`、`session_lifecycle.rs` 和 `projection_store.rs` 保持 durable 与 response snapshot 边界一致。
- inline Rust 回归覆盖 cold restart、live owner、terminal fail-closed 和 durable no-mutation。

### P1：GUI 投影统一

- 首页、侧栏、详情和输入框只消费 current Thread/Turn read model。
- reload 保留 live running/reconnect；restart 清除 running presentation。
- 多会话状态按各自 canonical session/thread/turn identity 隔离。

### P2：Gate 合同清理

- `reopen-running-turn-cdp-gate` 直接使用 `thread/start` 返回的 canonical UUID。
- direct `turn/start` 使用 `{ threadId, input: UserInput[], ... }`，Turn id 取响应 `turn.id`。
- direct cancel 只使用 `turn/interrupt { threadId, turnId }`。
- 删除固定 session/thread id、legacy runtime options/event name、queue/skip-resume 测试参数和旧 `agentSession` 正向合同。
- GUI trace observer 按 `threadId + typed input[]` 识别 current `turn/start`。

### P3：真实 Electron 证据

reload 模式必须证明：

- Electron/preload/IPC 与 current JSON-RPC method 命中。
- 同一 Turn 在 reload 前后保持 running。
- `thread/resume` 命中同一 Thread。
- GUI 停止后 backend、read model、sidebar 和 inputbar 同步收口。

restart 模式必须证明：

- 同一历史 Turn 在 read model 中为 interrupted。
- Thread 为 idle/notLoaded，GUI 不再显示 running。
- 没有 `thread/resume`、`turn/interrupt` 或 synthetic canceled event。
- 多会话模式下所有孤儿 Turn 同时转为 inactive。

## 6. 验收命令

```bash
npm run test:rust:related -- \
  lime-rs/crates/app-server/src/runtime/thread_read.rs \
  lime-rs/crates/app-server/src/runtime/status.rs \
  lime-rs/crates/app-server/src/runtime/read_model.rs \
  lime-rs/crates/app-server/src/runtime/load_context.rs \
  lime-rs/crates/app-server/src/runtime/session_lifecycle.rs \
  lime-rs/crates/app-server/src/runtime/projection_store.rs

npx vitest run \
  scripts/agent-runtime/reopen-running-turn-cdp-gate.test.mjs \
  scripts/agent-runtime/claw-chat-current-fixture-smoke.test.mjs

npm run smoke:agent-runtime-current-fixture
npm run test:contracts
npm run verify:gui-smoke

LIME_ELECTRON_FIXTURE_BUILD_READY=1 \
npm run smoke:reopen-running-turn-cdp-gate -- \
  --reopen-mode reload \
  --presentation-mode background \
  --multi-running-sessions \
  --timeout-ms 240000

LIME_ELECTRON_FIXTURE_BUILD_READY=1 \
npm run smoke:reopen-running-turn-cdp-gate -- \
  --reopen-mode restart \
  --presentation-mode background \
  --multi-running-sessions \
  --timeout-ms 240000
```

Gate B evidence 只保存 method、transport、status、canonical identity 和必要 marker，不保存 token、Provider secret、完整 prompt 或用户私密内容。

## 7. 当前证据

2026-08-21 已完成：

- App Server related：`1664 passed / 0 failed`。
- Gate/GUI trace observer Vitest：`2 files / 91 tests`。
- 旧固定 identity、legacy turn payload、30 分钟启发式与 synthetic restart cancel 路径已删除。
- 定向前端回归：5 个文件、109 个测试通过，覆盖停止 ThreadItem、partial + `(已停止)`、runtime `turn_canceled`、timeline merge 与 canonical MessageList DOM 合同。
- `npm run typecheck`：通过；修复 readonly projection、可空消息和 canonical `agent_message` 收窄错误。
- `npm run test:contracts`：通过，App Server client 299 项及命令、Harness、modality、scripts、release、docs 守卫全部通过。
- `npm run smoke:agent-runtime-current-fixture`：通过；current fixture 的 unit/history、stream、Electron fixture guard 与 GUI 场景全部通过，`liveProviderUsed=false`。
- `npm run verify:gui-smoke`：通过；真实 Electron Host、preload/IPC、App Server sidecar、Claw workbench reload 和 Memory settings smoke 通过。
- Reload Gate B：`.lime/cdp-evidence/reopen-running-turn-owner-based-reload-multi-fixed-v38-summary.json`，`completedGateB=true`、`failedAssertions=[]`、`consoleErrors=[]`；证明同一 Turn reload 恢复、停止收口、partial 保留和多会话取消隔离。
- Restart Gate B：`.lime/cdp-evidence/reopen-running-turn-owner-based-restart-multi-fixed-v39-summary.json`，`completedGateB=true`、`failedAssertions=[]`、`consoleErrors=[]`；证明孤儿 Turn 投影为 `interrupted`、Thread/UI inactive、多会话孤儿同时收口，且无 `thread/resume`、`turn/interrupt` 或 synthetic canceled event。
- Gate B 仍是 controlled fixture 证据：不证明 live Provider、DeepSWE 能力、DeepSWE score、Docker/Pier verifier 或平台打包证据。

## 8. 分类与边界

- `current`：live owner + canonical Thread/Turn/Item + response-only orphan normalization。
- `test-only`：受控 external backend，用于可重复 Electron Gate B；不是生产 fallback，也不证明 live Provider。
- `dead / deleted / forbidden-to-restore`：旧 `agentSession/*` 正向方法、固定 identity、时间戳 stale-running 推断、cold restart 自动续跑、synthetic cancel/terminal event。
- `skipped / user decision`：Docker、Colima、Lima 和 Pier separate verifier；不得安装或启动，也不得把 controlled evidence 冒充 DeepSWE score。

下一刀回到 Codex 对齐主链的剩余能力矩阵：优先检查真实 provider/runtime readiness 与工具生命周期是否有 current owner；本计划的 reload/restart 恢复闭环已完成，不再新增第二套恢复路径。
