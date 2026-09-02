# DevLime 对齐 Codex / Claude Code 差距清单

> 状态：待执行  
> 创建时间：2026-06-03  
> 适用范围：Lime 对齐 Codex 与 Claude Code 的 current 主线、GUI 工具渲染、runtime / host surface / extension 系统。  
> 参考优先级：不确定时优先参考 `/Users/coso/Documents/dev/rust/codex`，其次参考 `/Users/coso/Documents/dev/js/claudecode`。若两者冲突，以 Codex current 行为为准；若与 Lime GUI current 架构冲突，必须写明取舍。

## 使用方式

本文不是公众号文章素材，而是后续修复认领清单。其他进程可以复制本文件 path 后按单项认领：

`internal/exec-plans/devlime-codex-claude-alignment-gap-list.md`

认领前必须：

1. 记录 `git status --short --untracked-files=all`。
2. 声明本轮认领项和窄写集。
3. 先读相关事实源，不要按截图或印象 hard code。
4. 如果涉及 GUI，必须做 Playwright / GUI 验证；不能只跑单测。
5. 修复完成后把验证结果回写到本清单或对应批次文档。

## 分类口径

- `current gap`：Lime 已决定要进入 current 主链，但能力还缺或不稳定。
- `product choice`：上游有，但 Lime 是否需要还没定，不能默认当缺陷。
- `compat cleanup`：旧实现、旧文档或旧入口仍可能误导 AI / 开发者。
- `test gap`：实现可能已部分存在，但缺少足够回归或 GUI 证据。
- `blocked`：缺宿主能力、凭证、环境或产品决策，不能靠局部代码硬补。

## G1. 工具调用进入模型请求的全链路证据不足

- 分类：`current gap` / `test gap`
- 现象：日志里出现过 `messages=1, tools=0`，说明某些入口没有把工具传给 provider request；另一些场景工具已调用但 UI 显示错序。
- 参考：
  - Lime：`lime-rs/crates/agent-rust/crates/agent/src/tools/mod.rs`
  - Lime：`lime-rs/crates/agent-rust/crates/agent/src/tools/registry.rs`
  - Lime：`src/components/agent/chat/hooks/agentStreamCompletionController.ts`
  - Codex：优先查 tool registry、MCP tool call、app-server / protocol 相关实现
- 期望：
  - 对每类入口记录 provider、model、tools count、tool names、policy source。
  - 多模型 provider 下不 hard code OpenAI 专属逻辑。
  - 搜索 / browser / MCP / shell / file / task board 工具都能进入统一证据链。
- 建议验证：
  - 按 `internal/exec-plans/agent-tools-test-batches/` 七个批次执行。
  - Web/Search 批次必须跑实时流式和历史恢复两遍。

## G2. Agent Chat 工具渲染仍未完成全工具族覆盖

- 分类：`current gap` / `test gap`
- 现象：上一轮已修 `web_search` 代表错序，但真实工具面包含 native、gated、Agent/Team、Task Board、MCP resource、deferred MCP 等几十个工具。
- 参考：
  - `internal/exec-plans/agent-tools-test-batches/README.md`
  - `internal/exec-plans/agent-tools-test-batches/batch-01-file-search-tools.md`
  - `internal/exec-plans/agent-tools-test-batches/batch-07-task-board-tools.md`
- 期望：
  - 每个工具族都验证 `正文片段 -> 工具过程 -> 正文片段`。
  - 连续工具折叠后，展开仍保留工具名、参数摘要、结果摘要、失败状态。
  - raw JSON / stdout / source result 不污染最终正文。
- 建议验证：
  - 每批次独立运行单元测试 + Playwright。
  - 回写每批次 DOM index、截图路径、控制台状态。

## G3. Web/Search 来源引用与 Codex app 视觉细节未完全对齐

- 分类：`current gap` / `test gap`
- 现象：字体、引用来源位置、host/url 字号、hover 前隐藏按钮、空白间距、历史会话展示仍多次被指出与参考截图不一致。
- 参考：
  - `src/components/agent/chat/utils/searchResultPreview.ts`
  - `src/components/agent/chat/components/messageListInlineProcess.ts`
  - `src/components/agent/chat/components/StreamingRenderer.tsx`
  - `internal/exec-plans/agent-tools-test-batches/cross-agent-screenshot-alignment-prompt.md`
- 期望：
  - 同一搜索场景在实时流式和历史恢复中视觉一致。
  - 未悬停时多余操作按钮隐藏，悬停后显示且不挤压布局。
  - 来源 title / host / URL 字体和位置稳定。
- 建议验证：
  - 使用 cross-agent prompt 同时交给 Codex 和 Claude Code 采样。
  - 固定 viewport：`1440x1000`、`390x844`。
  - 记录 computed style 表。

## G4. 文件修改卡与撤销交互仍需截图级复刻

- 分类：`current gap`
- 现象：用户要求文件修改卡可点击展开、可撤销，并细节复刻 Codex app；历史会话中出现过多余元素和重复卡片。
- 参考：
  - `src/components/agent/chat/components/FileChangesSummaryCard*`
  - `src/components/agent/chat/utils/fileChangesUndo*`
  - `internal/exec-plans/agent-tools-test-batches/batch-01-file-search-tools.md`
- 期望：
  - 文件修改进入单一 summary card。
  - 展开态、折叠态、撤销可用/禁用态都清晰。
  - 不重复显示 artifact / trailing timeline。
- 建议验证：
  - 使用安全 fixture，不在用户真实项目上随意撤销。
  - 覆盖有 checkpoint / 无 checkpoint 两种路径。

## G5. Markdown 输出协议需要重新设计并固定事实源

- 分类：`current gap`
- 现象：多次局部修补后仍有 Markdown 渲染不好、工具输出与正文穿插、来源引用位置漂移的问题。
- 参考：
  - Codex：app / output renderer / protocol 相关实现
  - Claude Code：`src/query.ts`、`src/QueryEngine.ts`、message rendering / hooks 输出顺序
  - Lime：`internal/roadmap/markdown/` 既有方案
- 期望：
  - 从 system prompt 约定、runtime event schema、frontend renderer 三层统一。
  - 不靠前端猜测把 raw tool output 修成 Markdown。
  - 历史恢复和实时流式使用同一投影模型。
- 建议下一刀：
  - 先审阅 `internal/roadmap/markdown/`，确认是否已有最新方案。
  - 若方案过时，重写为“事件协议优先”的路线图。

## G6. Tool policy 仍需证明不依赖 OpenAI / 单模型特例

- 分类：`current gap` / `test gap`
- 现象：用户指出 `runtime_policy_test_request` 不能 hard code OpenAI，因为 Lime 是多模型。
- 参考：
  - provider routing / runtime policy 相关测试
  - `lime-rs/crates/agent-rust/crates/agent/src/providers/*`
  - `lime-rs/crates/agent-rust/crates/agent/src/agents/*`
- 期望：
  - 工具策略由 capability / provider contract / runtime tool surface 决定。
  - 测试覆盖 OpenAI 以外 provider 的缺省、降级、工具暴露场景。
  - 搜索和思考是否启用由模型判断，不再靠 UI 设置或硬规则代替。

## G7. `code_orchestrated` 策略意义需要最终收口

- 分类：`compat cleanup`
- 现象：用户质疑 `code_orchestrated` 策略是否还有意义；如果没有意义应清理。
- 参考：
  - `internal/exec-plans/upstream-runtime-alignment-plan.md`
  - `lime-rs/src/commands/agent_cmd/runtime_turn.rs`
  - runtime strategy 相关测试
- 期望：
  - 明确 `code_orchestrated` 是 current、compat 还是 dead。
  - 如果只剩旧策略名，删除或降级为 compat，不继续扩展。
  - 不能为了保留旧测试而 hard code 新断言。

## G8. hooks / skills extension 链路仍明显落后 Claude Code

- 分类：`current gap`
- 现状：
  - `SkillTool` prompt/workflow execution 已推进到 current。
  - `SkillExecutionMode::Agent` 仍未实现。
  - frontmatter hooks、project hooks、plugin hooks、hot reload、prompt/mcp/agent hook executor 仍未完整。
- 参考：
  - Claude Code：`src/skills/loadSkillsDir.ts`
  - Claude Code：`src/utils/hooks.ts`
  - Claude Code：`src/utils/hooks/execPromptHook.ts`
  - Claude Code：`src/utils/hooks/execAgentHook.ts`
  - Lime：`lime-rs/crates/agent-rust/crates/agent/src/skills/*`
  - Lime：`lime-rs/crates/agent-rust/crates/agent/src/hooks/*`
- 期望：
  - skill frontmatter 可以声明 hooks。
  - project / plugin hooks 有稳定 bootstrap。
  - prompt / mcp / agent hooks 不再是占位成功。
- 下一刀：
  - 先做 project hooks 加载入口和 hook event matrix，不直接跳到 Agent-mode skill。

## G9. Remote control / mobile push host surface 未进入 current

- 分类：`blocked` / `product choice`
- 现状：
  - `classifierPermissionsEnabled`、`voiceEnabled` 已有 current 落点。
  - `remoteControlAtStartup` 无真实 remote-control 宿主面。
  - `taskCompleteNotifEnabled / inputNeededNotifEnabled / agentPushNotifEnabled` 无 mobile push control plane。
- 参考：
  - `internal/exec-plans/upstream-runtime-alignment-progress.md`
  - `lime-rs/crates/agent-rust/crates/agent/src/tools/config_tool.rs`
  - `lime-rs/src/commands/gateway_channel_cmd.rs`
- 期望：
  - 先做产品决策：Lime 是否要实现 remote control / mobile push。
  - 未决策前不要把 OS auto-launch 或更新提醒硬映射成这些 setting。

## G10. `bridge:` remote peer messaging 暂未进入 current

- 分类：`current gap` / `product choice`
- 现状：
  - Team peers 与 synthetic `uds:<session-id>` local peer messaging 已是 current。
  - `bridge:` remote peer identity / ingress 仍缺宿主底座。
- 参考：
  - Claude Code：peer address / SendMessage / bridge 相关实现
  - Lime：`lime-rs/crates/agent-rust/crates/agent/src/tools/agent_control.rs`
  - Lime：`lime-rs/crates/agent-rust/crates/agent/src/tools/team_tools.rs`
- 期望：
  - 不把 browser connector / ChromeBridge 直接冒充 `bridge:` peer。
  - 若要做 remote peer messaging，先设计 remote session identity + ingress。

## G11. Memory / assistant 长期运行时还缺产品层闭环

- 分类：`current gap` / `product choice`
- 现状：
  - Lime 已有 Memory / Compaction 主链。
  - Claude Code 的长期 assistant、daily log、dream/consolidation、team memory 方向仍未完全产品化到 Lime。
- 参考：
  - Claude Code：`src/memdir/*`
  - Claude Code：assistant / KAIROS / TEAMMEM 相关代码
  - Lime：`internal/aiprompts/memory-compaction.md`
- 期望：
  - 决定 Lime 是否需要长期 assistant 模式。
  - 如果需要，先定义 GUI 状态、用户可见记忆边界、隐私和清理机制。

## G12. Coordinator / multi-agent 需要从“工具存在”升级到产品闭环

- 分类：`current gap`
- 现状：
  - Lime 已有 Agent / Team / Task 相关工具和 taxonomy。
  - 但多 Agent 的拆解、并行、合并、冲突处理、证据归并还需要继续验证。
- 参考：
  - Claude Code：`COORDINATOR_MODE`
  - Lime：`internal/aiprompts/task-agent-taxonomy.md`
  - Lime：`internal/aiprompts/task-agent-taxonomy.md` 与 `internal/aiprompts/query-loop.md`
- 期望：
  - 子代理过程在 GUI 中不重复、不吞最终答复。
  - 多代理结果有证据、归因和状态收敛。
  - Team / Agent 工具批次覆盖历史恢复。

## G13. Codex 式 schema / snapshot / protocol guard 还不够完整

- 分类：`current gap` / `test gap`
- 现状：
  - Lime 有 contracts、governance、GUI smoke。
  - 但对 Codex 那种 app-server schema fixture、UI snapshot、协议文档自动回写的体系还没有完全等价。
- 参考：
  - Codex：`justfile`
  - Codex：`codex-rs/app-server/README.md`
  - Codex：`app-server-protocol` schema fixtures
  - Lime：`npm run test:contracts`
  - Lime：`npm run governance:legacy-report`
- 期望：
  - 协议变更必须有机器可验证 fixture。
  - GUI 输出变化有截图或 snapshot 证据。
  - 文档新鲜度检查覆盖更多 current 主链文档。

## G14. 旧 UI 设置“是否搜索 / 是否思考”需要确认彻底下线

- 分类：`compat cleanup`
- 现状：用户已明确希望去掉这类选择，让模型自行分析。
- 期望：
  - UI 不再暴露“是否搜索 / 是否思考”的人工选择。
  - runtime 不再通过旧设置强行覆盖模型判断。
  - 如需策略，只通过 tool availability、permission、system prompt 与 evidence 约束。
- 建议验证：
  - 搜索设置页、Agent 输入框、runtime policy、历史配置迁移都要盘点。

## G15. 旧逻辑清理需要继续守 current 边界

- 分类：`compat cleanup`
- 现状：用户多次指出“旧的逻辑要清理掉”“如果没意义就删除”。
- 期望：
  - 对每条旧实现标记 `current / compat / deprecated / dead`。
  - dead surface 要加治理守卫，防止 AI 重新引用。
  - 清理动作必须说明如何直接帮助当前主线交付。
- 建议命令：
  - `npm run governance:legacy-report`
  - `npm run test:contracts`

## 当前最高优先级建议

1. 先跑 `agent-tools-test-batches` 七个批次，补齐工具渲染与 GUI 证据。
2. 用 `cross-agent-screenshot-alignment-prompt.md` 让 Codex / Claude Code 两边产出同口径截图和 style 表，集中修 Web/Search、文件卡、Markdown 三条最用户可见的差距。
3. 再推进 `hooks / skills` 和 `remote / peer messaging` 这类 runtime current gap。
4. `remoteControlAtStartup / mobile push / KAIROS / Ultraplan` 先按 product choice 处理，不要在没有宿主决策前硬补。

## 交付记录模板

```md
## Gap 修复记录

- 日期：
- 认领项：
- 认领写集：
- 参考来源：
- 变更摘要：
- 测试命令：
- GUI / Playwright 证据：
- 当前分类变化：
- 剩余缺口：
- 下一刀：
```
