# Codex Runtime 缺口对齐实施计划

> 状态：实施完成；本计划 owner 门禁及 GUI smoke 已通过，全仓 release evidence 受并行媒体 fixture 与本机磁盘余量阻塞
> 更新时间：2026-08-16
> 目标：只补齐 Codex 中 Lime 已确认缺失且属于 Lime 产品范围的 runtime 语义，不引入远端 exec-server、账号、marketplace 或第二套业务后端。

## 1. 范围与 owner

本次只处理四项：

1. Responses WebSocket 的 Turn-scoped 增量请求复用、`previous_response_id`、严格前缀校验、完整请求回退和 Responses prompt cache key。
2. MCP 工具级 `annotations.read_only_hint` 并发判定。
3. Codex 标准 `AGENTS.md` / `AGENTS.override.md` 项目规则发现兼容。
4. Async Hook 的 active/idle 结果归属、additional context 注入和生命周期隔离。

owner 固定为：

```text
Responses wire/session  -> lime-rs/crates/model-provider
Turn/session orchestration -> lime-rs/crates/agent-runtime / current provider adapter
MCP tool definition/execution -> lime-rs/crates/tool-runtime
AGENTS discovery/prompt assembly -> lime-rs/crates/agent + scheduler
Hook execution lifecycle -> lime-rs/crates/tool-runtime + agent runtime session owner
```

不做：

- Codex 远端 `exec-server`、`environment/*`、remote control、账号/计费和 marketplace。
- 用 mock 替代生产 bridge、provider 或 App Server。
- 恢复已退役 runtime、compat crate 或第二业务后端。
- 将 provider-specific remote compaction 替换 Lime 当前 durable provider-neutral compaction。

## 2. Agent Verification Contract

```text
改动名称：Codex Runtime 缺口对齐
执行计划文件：internal/exec-plans/codex-runtime-gap-alignment-plan.md
负责人：root
预算标签：budget:normal
风险等级：P0/P1
影响模块：model-provider、agent-runtime、tool-runtime、agent prompt/scheduler
不做范围：GUI 视觉、Electron IPC 新命令、远端 exec-server、live Provider
```

### Current 主链

```text
thread/turn start
  -> App Server runtime
  -> agent-runtime provider_turn / session actor
  -> model-provider current client / Responses wire
  -> tool-runtime MCP/Hook owner
  -> Thread/Turn/Item projection
```

本次不新增 App Server JSON-RPC 方法，不改变 Electron Desktop Host 边界；WebSocket 请求复用只在 provider 网络 owner 内生效。

### Happy Path

```text
同一 Turn 的第二次 Responses 请求历史是第一次请求的严格前缀
  -> 发送 previous_response_id + 新增 input
  -> 服务端返回 response.completed
  -> 保存 response id / output baseline

MCP 工具声明 read_only_hint=true
  -> step snapshot 将其标记为可并发
  -> provider_turn 与其它可并发工具并行执行

工作目录存在标准 AGENTS.md
  -> 从项目根到 cwd 按顺序加载
  -> 同层 AGENTS.override.md 优先
  -> 有界合并到 runtime prompt

Async Hook 在 active Turn 完成
  -> 不阻塞当前 sampling
  -> 结果在下一次 active sampling 注入

Async Hook 在 idle 完成
  -> 缓存在 session actor
  -> 下一次用户 Turn 开始前注入
```

### Evidence Layers

| Layer | 是否需要 | 证据 |
| --- | --- | --- |
| deterministic-smoke | 是 | Rust unit/integration tests、provider request capture |
| runtime-transcript | 是 | provider turn、MCP、Hook lifecycle 事件断言 |
| gui-trace | 否 | 本次不改 GUI owner |
| release-artifact | 否 | 本次不改版本、Forge 或依赖 |

### 必跑命令

```bash
npm run test:rust:related -- lime-rs/crates/model-provider lime-rs/crates/tool-runtime lime-rs/crates/agent-runtime lime-rs/crates/agent
npm run verify:local
```

按实际影响补充：

```bash
npm run test:contracts
npm run smoke:agent-runtime-current-fixture
```

未跑命令必须在完成记录中说明原因。由于本次不改 GUI、Electron IPC 或 App Server schema，`verify:gui-smoke` 不是最低门槛；若实际改动穿透这些边界，立即扩大验证范围。

## 3. 阶段与退出条件

### 阶段 A：Responses WebSocket 增量请求

- [x] 增加可测试的请求属性比较和 response item 前缀比较。
- [x] 为 Responses 请求引入 Turn-scoped session 状态，禁止跨 Thread/Turn 共享历史。
- [x] 记录 `response_id`、服务端新增 output items 和 `prompt_cache_key`。
- [x] 前缀不匹配、模型/工具/指令/路由/压缩变化时发送完整请求。
- [x] 连接失败或 HTTP fallback 使用完整逻辑请求，不把压缩 delta 当作 replay truth。
- [x] 添加连续请求、前缀失败、属性变化、空 delta、模型切换和压缩边界测试。

退出条件：provider capture 能证明增量请求和完整回退均符合预期，且不同 Turn 不共享 continuation state。

### 阶段 B：MCP 工具级并发

- [x] 在 `McpStepSnapshot` 捕获 route 时合并 server opt-in 与 `read_only_hint`。
- [x] 保持 hint 缺失/false 默认串行。
- [x] 补工具级 snapshot 单测和 provider_turn 执行测试。

退出条件：只读 hint 工具可与安全工具并发，未声明工具仍被串行护栏保护。

### 阶段 C：标准 AGENTS 发现

- [x] 兼容标准全局 `AGENTS.md` 与项目层 `AGENTS.md` / `AGENTS.override.md`。
- [x] 保留项目根到 cwd 的顺序、项目根边界、去重和总预算。
- [x] 明确 `.lime/AGENTS.md` 是否作为 Lime 专属 fallback；不得无记录改变优先级。
- [x] 补 global/project/nested/override/budget/symlink 回归。

退出条件：标准 Codex 项目规则可被 Lime runtime 读取，并且现有 `.lime` 规则行为没有未记录的回归。

### 阶段 D：Async Hooks

- [x] 保留 Hook snapshot 不可变和 trust/fail-closed 边界。
- [x] 将 async handler 从同步裁决路径分离，结果通过 session-owned channel 回传。
- [x] active Turn 结果注入下一次 sampling，idle 结果在下一 Turn 前消费。
- [x] session close、turn cancel、重复结果和 warning/context 生命周期有测试。

退出条件：async hook 不阻塞当前 sampling，结果不会注入错误 Turn，也不会在 session 关闭后产生副作用。

## 4. 进度记录

- 2026-08-16：完成 Codex/Lime 静态对比，确认四项范围；已有上下文 fragment、compaction、model switch、tool cancellation、MCP immutable snapshot、Multi-Agent 和 App Server transport 不重复建设。
- 2026-08-16：建立本计划，开始阶段 A。
- 2026-08-16：完成阶段 A：Responses WebSocket Turn-scoped session、严格前缀增量、`previous_response_id`、完整回退和 Responses prompt cache key；补充真实 WebSocket capture 与 lowering 回归。
- 2026-08-16：完成阶段 B：MCP 工具级 `annotations.read_only_hint=true` 并发；缺失/false 继续串行；补充 snapshot 与 provider-turn 回归。
- 2026-08-16：完成阶段 C：标准全局/项目 `AGENTS.md`、`AGENTS.override.md` 发现，并保留 `.lime` fallback、根边界、顺序、去重、预算和 symlink 语义；补充标准路径与 fallback 回归。
- 2026-08-16：完成阶段 D：Async Hook discovery、后台 command、生命周期 Started/Completed、active steer、idle mailbox、warning/context 投影和 cancellation 隔离；补充 discovery、runtime、session-loop 回归。
- 2026-08-16：修复 `lime-skills` 对编码本地 `skill://%2F.../SKILL.md` 的 URI 判定，避免把本地路径误当远程 authority；标准 `skill://server/...` 行为保持不变，3 个 URI 选择回归通过。
- 2026-08-16：清理五个 locale 中无源码引用的 7 个旧 Harness / Registered Skills key，`i18n:unused` 恢复为 `unused=0`，保留同文件其它既有文案改动。
- 2026-08-16：同步当前 Codex 参考源漂移：Codex 已从 `ModelInfo` 移除 `supports_parallel_tool_calls` 并在 Prompt 固定为 `true`；治理测试同时接受历史字段转发和当前固定值形态，不改变 Lime runtime 策略。
- 2026-08-16：完成不间断 `npm run verify:local` 复验：i18n、lint、typecheck、120 个 Vitest 批次和 contracts 全部通过；Rust changed selector 对并行工作树中的 `lime-rs/resources/default-skills/transcription_generate/SKILL.md` 按 fail-closed 退出，随后以 workspace unit/integration 扩大覆盖。
- 2026-08-16：workspace Rust unit 全部通过；workspace integration 在执行到 `app-server/tests/media_task_jsonrpc.rs` 时仅 `image_task_complete_rejects_wrong_task_type` 因独立媒体夹具缺少 `media_model_ref` 失败。本计划四个 owner 的定向、反向依赖和 current fixture 回归仍全部通过，未越界修改媒体链。
- 2026-08-16：重新执行真实 Electron GUI smoke、workspace rustfmt 与 diff 检查，均通过；旧 `audio.rs` 格式阻塞已消除。

## 5. 完成记录

```text
主线目标是否完成：是；四项 Codex runtime 缺口均已在 Lime current owner 内实现并有回归覆盖。
实现完成度：100%（阶段 A/B/C/D 全部完成）。
已跑验证：
  - `cargo test --manifest-path "lime-rs/Cargo.toml" -p model-provider --no-fail-fast`：258 passed。
  - `cargo test --manifest-path "lime-rs/Cargo.toml" -p tool-runtime --no-fail-fast`：356 passed。
  - `cargo test --manifest-path "lime-rs/Cargo.toml" -p agent-runtime --no-fail-fast`：210 passed。
  - `cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent prompt::runtime_agents --no-fail-fast`：14 passed。
  - `cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-core app_paths --no-fail-fast`：18 passed。
  - `cargo check --manifest-path "lime-rs/Cargo.toml" -p lime-agent`：通过。
  - `npm run test:rust:related -- lime-rs/crates/model-provider lime-rs/crates/tool-runtime lime-rs/crates/agent-runtime lime-rs/crates/agent`：通过；覆盖 owner 及反向依赖 `agent-runtime`、`app-server`、`lime-agent`、`lime-cli`、`lime-embedding`、`lime-mcp`、`lime-media-runtime`、`lime-processor`、`lime-scheduler`、`lime-server`、`lime-services`、`lime-skills`、`model-provider`、`tool-runtime`。
  - `npm run test:contracts`：通过（app-server-client 299 checks，其他 contracts/governance/docs 检查通过）。
  - `npm run smoke:agent-runtime-current-fixture`：通过；真实 Electron、preload、App Server sidecar、GUI fixture 全链路通过。
  - `npm run verify:local`：最新一次不间断执行中，i18n、lint、typecheck、完整 120 个 Vitest 批次和下游 contracts 全部通过；随后 Rust changed selector 因无法把并行工作树中的 `lime-rs/resources/default-skills/transcription_generate/SKILL.md` 映射到 workspace crate 而按 fail-closed 退出。
  - `npm run test:rust:unit -- --workspace`：通过；workspace 所有 Rust lib unit tests 无失败。
  - `npm run test:rust:integration -- --workspace`：扩大验证执行到 App Server integration；本计划相关测试均通过，仅独立媒体夹具 `image_task_complete_rejects_wrong_task_type` 因 `media_model_ref_missing` 失败。
  - `npm run verify:gui-smoke`：两次通过；最新证据 `standalone-shell-01-20260816152010-71415` result=`pass`，真实 Electron renderer、preload、App Server sidecar、工作台重载和 memory settings smoke 全链路通过。
  - `cargo test -p lime-skills agent_selection::tests::selects_skill_from`：3 passed。
  - `npm run i18n:unused -- --check`：通过，`unused=0`。
  - `npm test -- --run src/lib/governance/codexModelToolCallPolicyOrigin.test.ts src/lib/governance/codexModelExecutionPolicyOrigin.test.ts`：5 passed。
  - `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check`：通过。
  - `git diff --check`：通过；计划文件无尾随空白。
未跑或未通过验证及原因：
  - 聚合 `npm run verify:local` 未取得零退出码：smart Rust selector 无法映射上述 default skill 资源路径；workspace unit 已完整通过，workspace integration 的唯一失败属于并行媒体任务链。
  - 精确复跑 `image_task_complete_rejects_wrong_task_type` 未进入测试体：App Server integration 二进制在仅剩约 1.5 GiB 磁盘空间时链接失败；不删除用户数据或共享构建缓存来规避该环境限制。
环境依赖：Rust V8 测试需先运行 `node "scripts/lib/rusty-v8-artifacts.mjs"`，将其输出的 `RUSTY_V8_ARCHIVE` 与 `RUSTY_V8_SRC_BINDING_PATH` 作为 Cargo 命令级环境变量；本次已使用临时 artifact 完成验证。
残余风险：Async Hook 结果依赖 `pending_input` 注入；无该上下文的直接 provider 调用路径只保证执行与生命周期投影，不会追加 provider 上下文。失败/阻断/中止结果当前通过 `AgentEvent::Warning` 投影，尚无专用 warning protocol。
是否达到 Lime 本次实现门槛：是；current owner、定向测试、related 反向依赖、contracts、完整 Vitest 和真实 Electron GUI smoke 均通过。
是否可进入全仓 release evidence：否；并行媒体 fixture 仍有 `media_model_ref_missing`，且本机磁盘余量不足以完成其精确重链接。两项均不属于本计划 runtime 写集。
治理分类：`current` 为 model-provider、tool-runtime、agent-runtime 与标准 AGENTS prompt owner；`compat` 仅保留 `.lime/AGENTS.md` fallback 且不承接新语义；`deprecated` 无新增；`dead`/已删除 runtime 未恢复，无第二套后端或生产 mock fallback。
下一刀：媒体任务责任开发者修复或补齐 `media_model_ref` fixture，并在释放安全磁盘空间后重跑 workspace integration / `verify:local:full`；本计划不再有 runtime 实现任务。
```

## 6. 架构确认

```text
架构影响：非重大。保持 provider 网络、tool-runtime、agent-runtime 和 prompt owner 不变；不新增 crate、协议方法或跨层事实源。
架构图已更新：不适用；本次只在既有 owner 内补齐行为和测试。
责任开发者确认：root，2026-08-16
确认内容：已核对目录归属、数据流、依赖方向、协议边界和验证门禁。
```
