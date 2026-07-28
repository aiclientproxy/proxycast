# `internal/aiprompts` 索引

本目录存放 Lime 仓库的模块级说明、工程流程和治理文档。
根 `AGENTS.md` 只保留仓库级规则；超过“规则/入口”粒度的说明统一放到这里。

## 使用原则

1. **先按场景找入口** - 不确定从哪里开始时，优先读本页
2. **先读对应文档再改代码** - 尤其是命令边界、GUI 主路径、迁移收口、Provider 与凭证
3. **GUI 改动优先看质量链路** - Lime 是 GUI 桌面产品，先看 `quality-workflow.md` 与 `playwright-e2e.md`
4. **新旧并存问题先看治理文档** - 避免在 compat / deprecated 路径上继续长新表面
5. **Agent 重构按领域选择事实源** - Thread / Turn / Item、App Server、runtime、tool lifecycle、MCP、Skills、Multi-Agent、history hydrate、projection、测试护栏和命名按 `/Users/coso/Documents/dev/rust/codex` 收敛；多模型 catalog/default/switch、provider capability/readiness、retry/circuit breaker 与多模态 sampling 按 `/Users/coso/Documents/dev/rust/grok-build` 收敛；OpenCode 仅辅助 provider wire、canonical content 与媒体 lowering；已退役 runtime 没有迁移路线，能力缺口只能在 current owner 重建
6. **新增命名不要加品牌前缀** - 新程序、目录、crate/package、Electron IPC channel、App Server 方法、API 网关、类型、模块和脚本默认使用简洁领域名，不要加 `Lime` / `lime_` / `lime-`，也不要把已退役 runtime 名称或 `agent_runtime_*` 带入 current API；只有对外品牌、历史兼容或生态固定命名才例外，并在计划里说明
7. **新增 Agent 逻辑默认走 App Server** - 新 AI Agent、runtime、host integration、跨 App 复用能力先落到 `app-server` crates、JSON-RPC 协议、client 与 RuntimeCore；Electron 只作为 Desktop Host bridge，负责 IPC 和桌面壳能力，不是第二套后端或业务 adapter；旧 `agent_runtime_*` 只允许作为 retired guard / 历史 evidence，已退役 runtime 不得恢复为受控迁移残留
8. `lime-rs/src/**` 已删除，不得恢复 - 业务逻辑、领域服务、runtime 分支、API adapter、数据访问或跨 App 复用能力一律进入 `lime-rs/crates/**` 下的 App Server / RuntimeCore / services / core / agent / 协议 client crates，桌面壳能力进 Electron Desktop Host
9. `lime-rs/src/commands/**` 已删除，不得恢复旧 wrapper - 该目录不再承接新的业务逻辑、API adapter、runtime 分支、领域服务实现、compat wrapper 或退场 stub；新增后端能力进 App Server crates / RuntimeCore / services，桌面壳能力进 Electron Desktop Host
10. 超过 `1000` 行的非生成代码先拆分 - 接近 `800` 行进入预警；触碰前优先按领域、职责、数据边界或协议边界拆小，并复用项目已有模块化模式；无法本轮拆分时，必须在执行计划登记 blocker、风险和退出条件，不继续追加新业务逻辑
11. `src/lib/dev-bridge/**` 按职责治理 - `safeInvoke`、HTTP client、`app_server_handle_json_lines`、bridge availability / event listener capability 是 current renderer bridge；旧命令 policy / mock fallback 是迁移期 `compat / deprecated`；清命令时先收缩 policy、mock、旧 smoke 和 contract guard，不要把整目录删除当作默认动作；跨命令组长期残留必须回挂 `../exec-plans/tech-debt-tracker.md` 的 `CCD-012`

## 按场景导航

### 架构与治理

- `overview.md` - 项目架构总览与模块分层
- `agent-protocol-standards-map.md` - Agent 公共标准、Lime 开发协议、友链与未来拆分候选地图
- `agent-ui-runtime-standard.md` - Agent UI / Runtime 四包职责、宿主接入、主聊天 projection 迁移分类与守卫要求
- Agent UI / Runtime 当前宿主链路固定为 `AgentRuntimeClient -> projectAgentUiState -> AgentUiProjectionView`；任何第二宿主复用都必须回到 `agent-ui-runtime-standard.md`。
- `query-loop.md` - 运行时 Query Loop current 主链、提交入口与执行边界
- `prompt-foundation.md` - 基础 Prompt current 主链、system prompt 组装顺序与 current/compat 分类
- `task-agent-taxonomy.md` - Task / Agent / Coordinator current taxonomy、current/compat 分类与协调边界
- `remote-runtime.md` - Remote runtime current 主链、远程入口 current/compat 分类与控制面归属
- `memory-compaction.md` - Memory / Compaction current 主链、来源链/持久记忆/压缩边界与 current/compat 分类
- `persistence-map.md` - Runtime 文件快照持久化主链、artifact sidecar/version/checkpoint 边界
- `state-history-telemetry.md` - State / History / Telemetry current 主链、session/thread/request/evidence/history 边界与 current/compat 分类
- `governance.md` - 新旧并存治理、迁移收口、禁止回流
- `parallel-agent-collaboration.md` - 多 Agent / 多终端并行开发时的写集声明、避让和冲突处理协议
- `harness-engine-governance.md` - Harness Engine 事实源、evidence pack、replay / analysis / review 治理规范
- `quality-workflow.md` - 本地校验、GUI smoke、契约检查、CI 门禁
- `../tests/agent-ops-qc.md` - Agent 运营级测试体系、qcloop 场景、Evidence Pack 与发布证据门禁
- `../tests/agent-qc-p0-scenarios.md` - Agent QC P0 场景执行手册、GUI/runtime 证据要求与失败沉淀规则
- `../tests/lime-agent-qc-rollout-plan.md` - Lime 样本产品的 Agent 运营级测试落地计划
- `command-runtime.md` - `@` / `/` / 轻卡 / viewer / 功能方案包实施手册
- `skill-standard.md` - 统一技能标准、skill / adapter / runtime binding 边界
- `../tech/plugin/README.md` - Lime Plugin Package v1、Agent Skills 目录契约、worker / workbench / hooks / CLI / connector 打包标准
- `site-adapter-standard.md` - 站点适配器标准、来源导入边界、运行时收敛规则
- `web-browser-scene-skill.md` - 网页 / 浏览器场景技能设计提案，说明如何把外部 web skill 思路收敛到 Lime 主线
- `project-heatmap.md` - 仓库热力图与治理候选分析
- `limecore-collaboration-entry.md` - 跨仓库联动入口
- `../tech/harness/README.md` - Lime Harness Engineering 总入口与实施蓝图

### GUI 与前端

- `design-language.md` - 全局 UI 视觉语言
- `components.md` - React 组件约定
- `hooks.md` - 自定义 Hooks
- `lib.md` - 前端工具库与运行时封装
- `workspace.md` - Workspace 边界与工作区模型
- `playwright-e2e.md` - GUI 续测、Playwright MCP、交互验证
- `performance-profiling.md` - 性能分析与剖析方法

### 后端与运行时

- `commands.md` - Electron Desktop Host bridge、App Server JSON-RPC、legacy desktop facade 与前端网关协议同步点
- `services.md` - Rust 服务层
- `server.md` - HTTP Server 与接口边界
- `mcp.md` - MCP 服务器与工具管理

### Provider 与数据

- `providers.md` - Provider 接入与认证方式
- `credential-pool.md` - 凭证池退役边界、启动清理与守卫
- `converter.md` - 协议转换与兼容层
- `database.md` - 数据库层与持久化

## 常见入口建议

- **判断某个 Agent 能力该留在 Lime 内部还是拆成公共标准**：先读 `agent-protocol-standards-map.md`
- **改 Agent UI / Runtime 标准包、projection selector、runtime client 或共享 UI primitive**：先读 `agent-ui-runtime-standard.md`
- **改 UI / 页面结构**：先读 `design-language.md`，再看 `quality-workflow.md`
- **改 system prompt / subagent prompt / plan prompt / prompt_context / augmentation 顺序**：先读 `prompt-foundation.md`，再回看 `query-loop.md`
- **改 turn 提交 / prompt 组包 / queue / compaction / evidence 主链**：先读 `query-loop.md`
- **改 subagent / automation / execution tracker / scheduler taxonomy**：先读 `task-agent-taxonomy.md`
- **讨论 `/goal`、ThreadGoal 或跨 turn 目标续跑**：先读 `architecture.md`、`task-agent-taxonomy.md` 与 `query-loop.md`，再读 `../research/codex-goal/README.md`
- **讨论 Coding Agent、Skill Forge 或能力生成 draft**：先读 `query-loop.md` 与 `skill-standard.md`，再读 `../research/pi-mono-coding-agent/README.md` 与 `../roadmap/skill-forge/coding-agent-layer.md`
- **改 channels / browser connector / DevBridge remote runtime**：先读 `remote-runtime.md`
- **改记忆来源链 / working memory / durable memory / Team Memory / compaction**：先读 `memory-compaction.md`
- **改 FileArtifact / artifact sidecar / versions / file checkpoint / evidence 中的文件快照**：先读 `persistence-map.md`
- **改 session detail / thread read / requestTelemetry / evidence / history-record**：先读 `state-history-telemetry.md`
- **改 Electron IPC / App Server / Bridge / mock / legacy desktop facade**：先读 `commands.md`，再看 `quality-workflow.md`
- **碰到 `lime-rs/src/**`**：先判断是否只是 bootstrap / runner 接线、compat facade 委托或撤注册机械修复；凡是业务逻辑、领域服务、runtime 分支、API adapter、数据访问或跨 App 复用能力，默认迁往 `lime-rs/crates/\*\*`或 Electron Desktop Host。先读`commands.md`、`governance.md`与`quality-workflow.md`
- **碰到 `lime-rs/src/commands/**`**：默认只做旧 Tauri wrapper 删除清理、撤注册后的机械编译修复或 blocker 登记；不要在该目录新增实现、compat wrapper 或退场 stub。先读 `commands.md`、`governance.md`、`../exec-plans/tauri-wrapper-quick-cleanup-queue.md`和`../exec-plans/tauri-wrapper-command-inventory.md`
- **碰到超过 `1000` 行的非生成代码文件**：优先按领域、职责、数据边界或协议边界拆分，不继续追加新业务逻辑；接近 `800` 行时也要优先拆新模块。无法本轮拆分时，把原因、风险、退出条件和下一次拆分入口登记到执行计划
- **碰到 `src/lib/dev-bridge/**`**：先读 `commands.md`与`governance.md`；保留 current renderer bridge 传输，按命令组清 `commandPolicy`、`mockPriorityCommands`、旧 smoke、负向测试和 contract guard，把删不动的旧 policy 登记到对应执行计划；跨命令组长期 residual 必须同步回挂 `../exec-plans/tech-debt-tracker.md`的`CCD-012`
- **改 `@` / `/` / 轻卡 / viewer / ServiceSkill 场景**：先读 `command-runtime.md`
- **改 Claw 技能 / Service Skill / 统一 Skills 标准**：先读 `skill-standard.md`
- **改插件包 / Plugin 包结构 / `plugin.json` / `app.runtime.yaml` / `skills/**/SKILL.md`**：先读 `../tech/plugin/README.md`，再看 `skill-standard.md`与`governance.md`
- **改站点适配器 / 导入外部 adapter**：先读 `site-adapter-standard.md`，再看 `web-browser-scene-skill.md` 与 `quality-workflow.md`
- **做网页登录态访问 / 网页导出 / Markdown 落盘场景**：先读 `web-browser-scene-skill.md`
- **改 Workspace / GUI 壳 / 主路径**：先读 `workspace.md`、`quality-workflow.md`、`playwright-e2e.md`
- **做迁移 / 收口 / 去兼容层**：先读 `governance.md`
- **发现已退役 runtime 路径或能力缺口**：先读 `governance.md`；其实现已删除且禁止恢复，按 `/Users/coso/Documents/dev/rust/codex` 在 current owner 重建缺口，不存在独立迁移路线图
- **多个 Agent / 终端同跑一个任务**：先读 `parallel-agent-collaboration.md`，再声明本轮写集
- **改 handoff / evidence pack / replay / review / HarnessStatusPanel**：先读 `state-history-telemetry.md`，再看 `harness-engine-governance.md` 与 `governance.md`
- **改 Provider / 凭证加载 / Token 刷新**：先读 `providers.md`、`credential-pool.md`
- **做跨仓库联动**：先读 `limecore-collaboration-entry.md`

## 对应 Codex Skills

- **治理收口**：`.codex/skills/lime-governance/`
- **GUI 设计语言**：`.codex/skills/lime-design-language/`
- **工程质量 / 交付判断**：`.codex/skills/lime-quality-workflow/`
- **命令边界 / 契约同步**：`.codex/skills/lime-command-boundary/`
- **GUI 续测 / Playwright MCP**：`.codex/skills/lime-playwright-e2e/`
- **热力图 / 治理优先级**：`.codex/skills/lime-project-heatmap/`
- **项目技能提炼**：`.codex/skills/project-skill-factory/`

## 维护规则

1. 新增长期文档后，要同步更新本索引
2. 根 `AGENTS.md` 不再堆叠长流程，统一链接到这里
3. 如果某段说明已经变成长期流程或模块说明，应从根规则迁到本目录
4. 如果某条工作流已经高频复用到值得做成 skill，同步检查 `.codex/skills/README.md`
