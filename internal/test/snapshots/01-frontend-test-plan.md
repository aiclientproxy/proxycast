# Lime 前端测试场景矩阵

本文件把 Codex TUI/GUI 相关快照合并为 Lime 可维护的前端测试场景。每个场景均需在实现时
补 source snapshot 路径引用；完整逐文件映射在 `03-source-to-scenario-map.md`。

## P0：Agent 主路径

| ID | 场景 | 主要断言 | 测试层 | 证据 |
| --- | --- | --- | --- | --- |
| `turn-stream-complete` | delta -> final -> completed | assistant 文本不重复、不丢尾；同一 `threadId/turnId/itemId`；运行指示器退出 | projection + component | Gate B |
| `turn-stream-repair` | 丢失 delta 时由 final/read model 修复 | 不依赖固定 timer；终态可见且只出现一次 | integration | Gate B |
| `turn-interrupt-resume` | Esc/停止/取消后恢复输入 | running item 结束为 aborted/canceled；composer 可用；后续 turn 可提交 | component + integration | Gate B |
| `turn-budget-limit` | token/goal budget 到限 | 显示明确终态和下一步；不伪造成功 | projection + component | Gate A/B |
| `tool-running-terminal` | command/tool start、delta、success/error/interrupt | 工具卡状态可观察、输出不乱序、终态单一 | projection + component | Gate B |
| `tool-unified-exec-wait` | unified exec wait/empty/non-empty/unknown end | 等待态、探索态、最终消息不互相覆盖 | projection | Gate B |
| `approval-four-states` | pending/approved/denied/timed-out | 请求摘要、按钮、终态和后续 assistant 状态一致 | component + contract | Gate B |
| `approval-parallel-aggregate` | 并行 guardian/approval | 聚合状态不早结束；任一拒绝/超时按协议收敛 | integration | Gate B |
| `history-replay-isomorphic` | 刷新、resume、冷启动后恢复 | user/assistant/reasoning/tool 顺序和身份稳定；active MCP 不变 completed | projection + component | Gate B |
| `history-fork-lineage` | fork/resume/side conversation | parent/child thread、显示名、来源 turn 清晰且不串线 | integration + component | Gate B |
| `history-compaction-replacement` | 手动、自动、mid-turn compaction | replacement/window lineage 只注入一次；旧摘要不重复 | runtime contract | Gate B |

## P0：输入和布局

| ID | 场景 | 主要断言 | 测试层 | 证据 |
| --- | --- | --- | --- | --- |
| `composer-submit` | 普通输入提交 | 用户消息先显示；提交后进入 running；输入框不会清空过早 | component | Gate A/B |
| `composer-queue-steer` | active turn steer；TUI 本地 queue 延后 | active turn 只提交 `turn/steer(expectedTurnId)`，不创建第二个 Turn；Lime 不暴露 public queued-turn 写平面 | unit + contract | Gate B |
| `composer-mention-slash` | slash、skill、file、plugin mention popup | 目标 mention 精确；空格/尾缀/歧义不误匹配 | unit + component | Gate A |
| `composer-parent-owned` | parent-owned thread 拒绝输入/设置快捷键 | 用户看到明确不可操作状态，不向错误 thread 发请求 | component + contract | Gate B |
| `layout-resize-reflow` | 1280x820 -> 880x720 等 viewport | 消息锚点、输入栏、右侧 workspace、item 顺序稳定 | projection + DOM geometry | Gate A |
| `layout-narrow-overflow` | 长文案/长路径/窄弹窗 | 不重叠、不横向溢出、不截断关键动作 | component + screenshot | Gate A |
| `layout-markdown-rich` | heading/table/code/link/CJK/混合宽度 | 使用统一 rich renderer；file link 保留 path/line | projection + component | Gate A |

`composer-queue-steer` 需要按 owner 拆开理解：Codex App Server current protocol 只提供
`turn/start`、`turn/steer` 与 `turn/interrupt`；Codex TUI 的 `message_queue` 和
`pending_input_preview` 是 `ChatWidget::input_queue` 持有的本地 UI 调度，不是 public App Server
read/write model。Lime Renderer 不复制第二套 scheduler，因此这部分 snapshot 记录为 `defer / TUI-only`；
current 回归只证明 canonical `thread/read` 选择 active turn、`turn/steer` 保持同一 Turn identity，
以及 public queue method、`queueIfBusy`、PendingTurn GUI 不回流。

## P1：设置、状态和工具面板

| ID | 场景 | 主要断言 | 测试层 | 证据 |
| --- | --- | --- | --- | --- |
| `model-picker` | model/reasoning/effort selection | catalog 驱动可见项；model-specific effort id 映射 canonical wire value；切换失败不提交 UI 假状态 | unit + component + contract | Gate A/B |
| `permissions-picker` | read/workspace/full access/profile | 选择结果回到当前 thread/session；不混用 workspace 默认值 | component + contract | Gate B |
| `status-surface-matrix` | footer/header/title/model/reasoning/goal/rate-limit | 一个 presentation owner；metadata 与 runtime status 同源 | projection + component | Gate A |
| `status-running-completed` | working/paused/blocked/complete/usage-limited | status、footer、composer 和完成耗时一致 | projection + component | Gate A/B |
| `mcp-inventory-elicitation` | startup/loading/inventory/elicitation | server/tool/resource 数量准确；敏感值掩码；表单响应 typed | component + contract | Gate B |
| `hooks-lifecycle` | session/pre/post tool、running/completed/blocked | hook 状态不闪烁、不重复，诊断可读 | projection + component | Gate B |
| `multi-agent-roster` | spawn/wait/followup/interrupt/list | transcript、roster、delegation edge、worker notification 同 identity | projection + component | Gate B |
| `plugin-marketplace` | loading/search/install/remove/detail/error | marketplace/filter/installed state、详情和错误可恢复 | component + integration | Gate A/B |
| `apps-and-capabilities` | Apps/skills/capability popup | loading/empty/error/installed 状态明确，不走旧 catalog alias | component + contract | Gate A/B |
| `plan-goal` | plan mode、goal active/paused/blocked/budget | plan/goal 不覆盖消息和输入；恢复后状态准确 | projection + component | Gate A/B |
| `feedback-update-usage` | feedback consent、update、usage/rate limit | 二次确认、无输出、登录/限额错误有稳定文案 | component | Gate A |

## P1：媒体、产物和错误

| ID | 场景 | 主要断言 | 测试层 | 证据 |
| --- | --- | --- | --- | --- |
| `media-image-attachment` | local/remote/foreign image | 图片引用、占位、历史恢复和不支持媒体行为清晰 | projection + component | Gate A/B |
| `media-image-generation` | begin/success/failure | working status 恢复；失败不会伪造 artifact | integration + component | Gate B |
| `artifact-diff-preview` | apply patch、diff、artifact、inline visualization | path、line、hunk、preview action 和错误状态可追溯 | component + integration | Gate A/B |
| `error-config-provider` | config/provider/network/startup error | fail closed；错误归属明确；不回退 mock | contract + component | Gate B |
| `error-safety-sandbox` | safety/access/windows sandbox | deny/enable/fallback 提示和 retry 路径稳定 | component + contract | Gate B |
| `error-mcp-hook` | MCP/hook invalid/timeout/diagnostics | 失败进入历史或诊断 surface，不吞掉 terminal 状态 | integration + component | Gate B |

## P2：产品壳和边界

| ID | 场景 | 主要断言 | 测试层 | 证据 |
| --- | --- | --- | --- | --- |
| `onboarding-setup` | onboarding、cwd、resume、profile | 当前 Lime onboarding owner 的字段、错误和恢复路径稳定 | component + Gate A |
| `settings-memory-personality` | memories、personality、developer/setup | 只验证 Lime 现役设置；Codex ChatGPT 专属项不进入 current | component |
| `search-resume-picker` | thread search/resume picker | 搜索、排序、分页、窄宽和加载错误可用 | component + Gate A |
| `platform-windows` | Windows path/wrap/sandbox/platform snapshot | 合并到同一业务场景的 platform matrix；不重复实现 | Gate A/B |
| `cli-doctor` | doctor/report/terminal title 等 CLI/TUI 专属 | 记录 deferred/contract；不改 Lime GUI owner | deferred |

## 实现约束

- 复杂筛选、分组、状态机、runtime 参数投影先抽到 selector/View Model，再写 unit test。
- component test 只保留真实 DOM 接线和少量关键分支，不把所有快照重写成 regex。
- screenshot 只覆盖稳定几何状态；固定 viewport/locale/font，禁用动画，mask 时间、随机 ID、光标。
- Playwright 使用 locator 和 web-first assertion；状态等待依赖业务事件，不使用固定 sleep。
- 真实 Electron 场景必须检查 `window.__LIME_ELECTRON__`、`electronAPI.invoke`、
  `app_server_handle_json_lines` 和当前 JSON-RPC method。

## 多模型裁决

Codex 的 `model_selection_popup`、`model_reasoning_selection_popup` 和 list selection 快照继续作为
弹窗、搜索、滚动、窄宽与键盘交互来源。多模型业务断言不从这些 TUI 文案推导，而按
`/Users/coso/Documents/dev/rust/grok-build` 的控制平面设计覆盖：

1. 模型列表来自 current catalog；未知、隐藏、未就绪或被 allowlist 排除的模型 fail closed。
2. reasoning effort 选项由目标模型元数据提供，允许显示 id/label 与实际 wire value 不同；选择
   `deep -> xhigh` 这类映射时，UI 显示值、session 状态和请求 wire value 必须分别断言。
3. 不支持 reasoning effort 的模型不展示可提交选项，也不把旧模型 effort 静默带入请求。
4. model switch 只在 App Server/session owner 成功后提交 UI current 状态；失败保留原 model、
   reasoning、history 和 provider route，并展示 typed error。
5. provider readiness、capability、retry/circuit breaker 均从结构化 facts 读取，不按 provider 名称
   或模型前缀推断，不用静默 fallback 掩盖切换失败。

主要 oracle：

- `crates/codegen/xai-grok-pager/src/acp/model_state.rs`
- `crates/codegen/xai-grok-shell/src/agent/handlers/model_switch.rs`
- `crates/codegen/xai-grok-shell/src/session/acp_session_impl/model_switch.rs`
- `crates/codegen/xai-grok-pager/tests/pty_e2e/reasoning_efforts_menu_renders_and_remaps_on_wire.rs`
