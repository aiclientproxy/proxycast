# Codex Desktop 选择性 Goose 参考执行计划

> status: `in-progress / P3-external-platform-evidence; P4-local-complete`
> owner: Desktop Host + App Server + agent-ui；具体实现仍归既有 current owner
> started: 2026-09-03
> product target: Codex Desktop
> public semantic reference: `/Users/coso/Documents/dev/rust/codex`、Codex App Server 文档
> open-source desktop reference: Goose `1.49.0` (`794b04a0b1f4c58378ef3738dade297c13690b77`, Apache-2.0)
> Lime product chain: `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`

## 1. 主目标

在不复制 Codex Desktop 私有实现、也不把 Goose 变成第二产品方向的前提下，补齐 Lime 面向 Codex Desktop 的桌面可靠性、恢复、诊断和交互体验。

本计划的固定口径是：

> Codex Desktop 是唯一产品目标和语义事实源；Codex CLI/App Server 是公开协议与 runtime 事实源；Goose 仅作为 Apache-2.0 开源桌面实现参考，只吸收经 Lime current owner、Codex 语义和 Gate B 验证认可的局部机制。

Goose 的代码优先作为行为和工程取舍的参考，不默认复制代码。若确需移植代码，必须保留 Apache-2.0 与 NOTICE 义务，并在变更说明中记录来源、改动范围和许可证审查结果。

## 2. 范围与非目标

### 2.1 纳入范围

- sidecar 启动、readiness、失败诊断、stderr 有界尾部和跨平台清理。
- system sleep/resume 后的 App Server 连接恢复、超时、退避和旧连接隔离。
- 脱敏诊断 bundle，覆盖 Desktop Host、sidecar、App Server 和 Thread/Turn identity 摘要。
- Codex Desktop 风格的 Thread history、resume、fork、工具生命周期、approval、MCP App 和子 Agent activity 展示。
- 将 Goose Recipe 的参数表单、结构化输出、retry/history 等 UI 经验映射到 Lime 已有 Skills、Plugins、Automation 和 Scheduled Tasks。
- Windows packaged 与 macOS Gate B 的真实交付证据补齐。

### 2.2 明确不做

- 不把 Goose ACP 接入 Lime，不新增 ACP adapter 或 ACP 事实源。
- 不引入 Goose 的 `Session/Message` 作为 Thread/Turn/Item 的替代模型。
- 不恢复旧 agent loop、旧 runtime、旧 Electron 业务后端或生产 mock fallback。
- 不建立 Goose Recipe runtime、Recipe storage、第二套 scheduler 或第二套 Skills/Plugins catalog。
- 不把 Goose 的 Autonomous 默认、模型权限判断或工具授权逻辑迁入 Lime。
- 不声称 Goose 的实现、静态 `app.asar` 或浏览器投影证明了 Codex Desktop 的私有行为。
- 不为 Codex Desktop 私有且 Lime 未声明的能力（例如设备密钥、ScreenCaptureKit 私有桥）猜测 API。

## 3. 事实源、证据等级与决策规则

### 3.1 事实源优先级

1. Codex Desktop 可观察行为：已安装应用的明确运行证据；静态 bundle 只能作为设计参考。
2. Codex CLI/App Server：公开协议、Thread/Turn/Item、审批、工具生命周期、恢复和 sandbox 语义。
3. Lime 仓库：current owner、协议、read model、GUI 和现有 Gate A/B 证据。
4. Goose：仅证明一种成熟开源桌面实现如何处理工程问题，不能证明 Codex Desktop 需求。

### 3.2 证据等级

| 等级                     | 可以证明                                                              | 不能证明                              |
| ------------------------ | --------------------------------------------------------------------- | ------------------------------------- |
| `codex-desktop-observed` | Codex Desktop 真实可观察交互或安装包行为                              | 未运行路径、私有实现源码              |
| `codex-rust`             | Codex 开源 runtime、App Server、协议和测试语义                        | 私有 Desktop GUI/native 行为          |
| `goose-static`           | Goose 源码中存在某种实现机制                                          | Codex Desktop 采用该机制、Lime 已完成 |
| `lime-local`             | Lime owner、协议、单测、fixture 和本地实现                            | packaged 平台能力                     |
| `gate-a`                 | Renderer/浏览器投影和交互                                             | Electron IPC、sidecar、系统 API       |
| `gate-b`                 | 真实 Electron、preload/IPC、App Server、RuntimeCore/read model 和 GUI | 未显式覆盖的平台权限、live provider   |
| `platform-packaged`      | 实际安装包、sidecar、签名和平台生命周期                               | 未运行或未覆盖的平台                  |

任何候选能力必须同时记录：目标依据、Goose 参考位置、Lime current owner、采用决策、证据等级和不确定性。Goose 证据不得升级为 Codex Desktop 事实。

## 4. 三方能力决策矩阵（P0 产物）

P0 先建立并冻结以下矩阵；未进入矩阵的 Goose 能力不得进入实现。

| 能力                       | Goose 可参考机制                                                                | Lime current owner                            | 决策                                    | 优先级 |
| -------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------- | --------------------------------------- | ------ |
| sidecar 启动与健康检查     | binary resolution、loopback readiness、fingerprint、stderr tail、跨平台 cleanup | `app-server-daemon`、Electron Desktop Host    | `采用机制，保持 App Server 主链`        | P1     |
| 启动失败诊断               | startup stage journal、有限历史文件、URL/token 脱敏                             | Desktop Host、现有 diagnostics owner          | `采用旁路诊断，不新增业务事实源`        | P1     |
| sleep/resume 重连          | generation、initialize timeout、指数退避、recovery listener                     | `app-server-client`、Desktop Host lifecycle   | `采用语义，按 JSON-RPC 连接实现`        | P1     |
| Thread history/resume/fork | session list、恢复、分叉、工具展开                                              | `thread-store`、App Server、`agent-ui`        | `采用交互，严格消费 Thread/Turn/Item`   | P2     |
| tool/approval lifecycle    | 工具调用展开/折叠、内联审批                                                     | `tool-runtime`、App Server、`agent-ui`        | `部分采用，安全决策不下沉到 GUI`        | P2     |
| MCP App inline surface     | MCP App 内联渲染                                                                | MCP current owner、Right Surface、`agent-ui`  | `部分采用，先验证 owner 与 Gate B`      | P2     |
| 子 Agent activity          | 子 Agent 会话入口、活动状态                                                     | `agent-runtime`、App Server、`agent-ui`       | `采用现有 Multi-Agent projection`       | P2     |
| Recipe 参数表单            | Recipe 参数 schema/form                                                         | Skills、Plugins、Automation UI                | `仅采用 UI 表达，不引入 Recipe runtime` | P2     |
| JSON Schema 输出           | 结构化输出约束                                                                  | RuntimeCore、artifact/structured-output owner | `采用已有 schema owner`                 | P2     |
| retry/run-now/history      | workflow 控制与历史                                                             | Scheduled Tasks/Automation owner              | `映射已有能力，不建第二 scheduler`      | P2     |
| ACP                        | ACP transport/session                                                           | 无；App Server JSON-RPC 是 current            | `明确排除`                              | P0     |
| Session/Message 模型       | Goose transcript 模型                                                           | Thread/Turn/Item                              | `明确排除`                              | P0     |
| Autonomous 默认            | Goose 默认执行策略                                                              | approval、sandbox、tool-runtime               | `明确排除`                              | P0     |

### 4.1 Goose 参考索引

P0 矩阵和后续实现评估优先从以下已固定版本读取；其它 Goose 文件只有在矩阵新增条目后才能进入写集：

| 参考位置                                                 | 只提取的机制                                                                                               | 不提取                                                    |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `ui/desktop/src/gooseServe.ts`                           | binary resolution、loopback server、readiness polling、TLS fingerprint、stderr tail、Windows/macOS cleanup | ACP 作为 Lime 协议、Goose server 作为业务后端             |
| `ui/desktop/src/startupDiagnostics.ts`                   | 启动阶段 journal、有限历史保留、URL/token 脱敏、失败上下文                                                 | Goose diagnostics schema 作为 Lime 业务 read model        |
| `ui/desktop/src/acp/acpConnection.ts`                    | resume 重连、connection generation、指数退避、initialize timeout、recovery listener                        | ACP transport、ACP session identity                       |
| `ui/desktop/src/` 中的 session/tool/MCP/workflow surface | 会话列表、工具调用展开/折叠、审批内联、MCP App、Recipe 表单和 retry/history 的交互取舍                     | `Session/Message`、Recipe runtime/storage、第二 scheduler |

外部语义证据：

- Codex 开源边界：<https://learn.chatgpt.com/docs/open-source>
- Codex App Server：<https://learn.chatgpt.com/docs/app-server>
- Codex App / Thread 体验：<https://learn.chatgpt.com/docs/app>
- Codex worktree 与 subagent 语义：<https://learn.chatgpt.com/docs/environments/git-worktrees>、<https://learn.chatgpt.com/docs/agent-configuration/subagents>

### 4.2 扩展参考候选池（2026-09-03）

候选按“对 Codex Desktop 的帮助”而不是 GitHub star 排序。星标和活跃度只作发现信号，不能替代代码、许可证和 Gate B 审查。

| 项目                                                                    | 形态 / 许可证                                                                                   | 最值得参考                                                                                                                         | 与 Lime 的关系                                                                                                    | 建议级别                   |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------- |
| [Apache Maka](https://github.com/apache/maka)                           | Electron + React；Apache-2.0；ASF Incubating                                                    | 单一 Runtime Host、append-only Runtime Event Log、Session/Turn/AgentRun、RecoveryResolver、sandbox/permission、真实 Electron smoke | 架构和恢复语义与 Lime 最接近；但当前是 nightly，macOS 仅 arm64，Windows 为 unsigned preview，不是生产依赖         | **A：架构首选**            |
| [Zed](https://github.com/zed-industries/zed)                            | 原生 Rust/GPUI 桌面；主要 GPL-3.0-or-later                                                      | Agent Panel、ThreadStore、工具权限、Worktree、长历史和原生桌面交互                                                                 | 产品体验和 Rust 桌面工程很强；许可证不适合直接复制代码，且 Zed 的 ACP/Agent 模型不能替代 Lime App Server          | **A：GUI/桌面首选**        |
| [Claude Code Haha](https://github.com/NanmiCoder/cc-haha)               | Electron + React；MIT                                                                           | 多会话、Worktree、diff review、MCP/SubAgent 管理、权限 UI、Workflow、桌面打包                                                      | 功能形态最接近“Codex Desktop + Agent 工作台”；需逐项审查其 runtime、IPC、session store 是否形成双轨               | **A-：产品形态参考**       |
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm)            | 跨平台 Desktop/本地优先；MIT                                                                    | Agent workspace、MCP、scheduled tasks、Agent Builder、Open Computer 方向、跨平台发布                                               | 适合借鉴工作区/应用集成和本地数据体验；RAG/多用户产品面较重，不能作为 Thread/Turn/Item 语义来源                   | **B：工作区/运维参考**     |
| [Cherry Studio](https://github.com/CherryHQ/cherry-studio)              | Electron；AGPL-3.0                                                                              | 多 provider、模型目录、MCP 管理、助手/话题 UI、跨平台发布                                                                          | 适合 provider/catalog/MCP 管理面；AGPL 和 Enterprise 分层需先做许可证审查，Agent runtime 不是 Lime 事实源         | **B：模型/MCP UI 参考**    |
| [LobsterAI](https://github.com/netease-youdao/LobsterAI)                | Electron；MIT                                                                                   | Cowork session、artifact、权限提示、MCP、scheduled tasks、IM/远程入口、诊断与修复经验                                              | UI/桌面运维很有价值；其明确以 OpenClaw 作为 runtime/gateway，不能把 OpenClaw 结构或 IPC 业务逻辑引入 Lime         | **B：UX/运维参考**         |
| [Open Cowork](https://github.com/OpenCoworkAI/open-cowork)              | Electron；MIT                                                                                   | WSL2/Lima VM sandbox、Skills、MCP connector、permission dialog、headless/JSONL 演进路线                                            | 适合 Windows/macOS sandbox UX；项目 roadmap 仍承认 single-session/single-turn 等缺口，不能作为完成的 runtime 模板 | **B-：安全 UX 参考**       |
| [LobeHub](https://github.com/lobehub/lobehub)                           | Web + Electron Desktop；许可证需审查（README 为 Community License，`package.json` 有 MIT 标识） | Agent as unit of work、Agent Groups、Skills、Schedule、Workspace、Memory、desktop bridge                                           | 多 Agent/control-plane 参考价值高，但产品面比 Codex Desktop 更宽；先做许可证和 runtime/desktop owner 审查         | **B：多 Agent 控制面参考** |
| [OpenHands Agent Canvas](https://github.com/All-Hands-AI/OpenHands)     | Web control center；MIT                                                                         | 多 Agent backend、Agent Server、Automation、schedule/webhook、backend switching                                                    | 不是 native desktop；适合 workflow/control-plane 和远程 backend 抽象，不进入 Desktop Host 参考池                  | **C：控制面参考**          |
| [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter) | Rust CLI/TUI；Apache-2.0                                                                        | Codex exec protocol 兼容、MCP、Skills、sandbox/approval、provider/harness portability                                              | 不是桌面端，且 Lime 已以 Codex-rs 为 runtime 事实源；只做协议/便携性旁证                                          | **C：协议旁证**            |

以下项目暂不进入首轮参考：

- [Void](https://github.com/voideditor/void)：仓库已 archived，不作为活跃实现依据。
- [Continue](https://github.com/continuedev/continue)：README 已声明不再积极维护，且主要是 IDE 插件/CLI。
- Jan、Chatbox：桌面壳、模型配置和基础会话体验成熟，但 Agent runtime、恢复和工具权限深度不足，放在一般客户端 UX 备查。

推荐研究顺序：`Maka（架构/恢复） -> Zed（桌面 Agent UX） -> cc-haha（Electron 产品形态） -> LobeHub/AnythingLLM/Cherry Studio（多 Agent、工作区、模型与 MCP 管理） -> LobsterAI/Open Cowork（安全与运维案例）`。OpenHands 和 Open Interpreter 只在 workflow 或协议专项需要时研究。

## 5. 分阶段执行

### P0：三方能力矩阵与边界冻结

目标：把“Codex Desktop 目标”“Codex 公开语义”“Goose 实现参考”“Lime 当前状态”分栏记录，防止参考项目反向定义产品。

写集：本计划、必要的 `internal/research/` 证据索引、`internal/exec-plans/README.md` 导航；不改业务代码。

退出条件：

- 每个候选能力有事实源、证据等级、owner、决策和不确定性。
- 采用项、部分采用项、暂不采用项和明确排除项均有理由。
- 许可证边界、禁止依赖和禁止新事实源已写入计划。

### P1：桌面启动、sidecar 与恢复可靠性

目标：把 Goose 中成熟的启动诊断和恢复机制映射到 Lime 现有 Desktop Host/App Server owner。

建议写集：

- `lime-rs/crates/app-server-daemon/**`：readiness、启动阶段、stderr 有界尾部、退出清理和诊断字段。
- `electron/**` Desktop Host sidecar/lifecycle owner：system sleep/resume、跨平台 child cleanup、失败上下文转发。
- `src/lib/api/**` `app-server-client` 相关 current gateway：connection generation、initialize timeout、backoff 和 recovery 状态投影。
- 对应 Rust/TypeScript tests、Electron fixture 和 evidence summary。

约束：不新增 ACP；不让 Renderer 直连 sidecar；不让诊断字段承载业务状态；不把 URL、token、credential、prompt 原文或敏感绝对路径写入诊断。

退出条件：

- sidecar 启动失败可定位到阶段、退出码和有界 stderr 尾部，并能安全清理子进程树。
- resume 后旧连接不能回写新连接；初始化、退避和最终失败状态均可测试。
- macOS 与 Windows 至少各有一个真实 packaged 或 Gate B 证据；未覆盖平台必须标记未验证。

### P1：诊断 bundle 与可观测性

目标：提供可导出、可脱敏、可回放摘要的旁路诊断，不形成第二事实源。

建议写集：现有 diagnostics builder/exporter、Desktop Host 诊断事件、App Server readiness 适配、相关 i18n 文案和单测。

bundle 最小字段：platform/arch、应用与 sidecar 版本、resource manifest 摘要、readiness 阶段、IPC/App Server 连接状态、Thread/Turn identity 摘要、最近错误分类、evidence id。

退出条件：

- token、credential、完整 prompt、完整 provider payload、完整 stderr 和用户敏感路径均被排除或脱敏。
- 诊断导出失败不影响 Thread/Turn 主链，历史保留数量和大小有上限。
- 同一 bundle 能由本地测试稳定生成并被版本化 schema 校验。

### P2：Codex Desktop 风格的会话与工具体验

目标：改善 Codex Desktop 对齐体验，但所有状态仍来自 canonical Thread/Turn/Item。

建议写集：`agent-ui` Thread 页面、history/resume/fork、tool lifecycle、approval、MCP App 和 Subagent activity 的现有组件与 projections；必要时补 App Server typed read model，不新增 GUI 私有 store。

交互优先级：

1. Thread history、resume、fork、archive/delete 的单一入口。
2. Turn 状态、工具调用、approval、sandbox 结果的可展开/折叠展示。
3. 子 Agent activity 与 child Thread 入口，保留 parent/child identity。
4. MCP App inline surface，明确 pending/loading/error/unsupported 状态。

退出条件：

- GUI 不从 Goose Session/Message 或本地派生 transcript 读取状态。
- 工具、approval、子 Agent 和 MCP App 的显示状态可从 Thread/Turn/Item 或 typed event 重建。
- Gate A 证明交互，Gate B 证明真实 Electron/preload/IPC/App Server/runtime/read model 链。

### P2：Workflow 交互的选择性映射

目标：借鉴 Goose Recipe 的成熟表单和历史交互，提升 Lime 已有 Skills/Plugins/Automation 的可用性。

建议写集：现有 Skills/Plugins/Automation/Scheduled Tasks UI、JSON Schema 表单适配、retry/run-now/history projection；不新增 recipe 命名、目录、runtime 或 catalog。

退出条件：

- 输入 schema、执行身份、权限和 artifact 仍由 Lime current owner 提供。
- retry/run-now/history 复用现有 scheduler/automation contract。
- 没有出现第二套 workflow storage、scheduler、execution history 或生产 mock fallback。

### P3：安全与跨平台交付

目标：将 Goose 的权限 UX 经验限制在展示层，并继续完成 Codex Desktop 平台 parity 主线。

建议写集：`tool-runtime` permission/approval UI contract、`app-server-daemon` packaged readiness、Forge/Squirrel、macOS native helper 与现有 Windows restricted execution owner。

约束：sandbox、approval、capability、credential、network 和 process cleanup 继续由 Codex/Lime current owner 决定；Goose Autonomous 和模型权限逻辑不作为安全依据。

退出条件：

- Windows packaged/Squirrel、macOS 权限/bookmark 的 Gate B 结果回写既有 `codex-desktop-platform-parity-plan.md`。
- UI 不能把“已安装/已启用”伪造成“模型可调用/权限已授予”。
- 失败默认 fail closed，缺少 capability 或 readiness 时不得猜测成功。

### P4：治理、负向守卫与验收

目标：防止参考 Goose 后形成第二 runtime、第二 catalog 或第二协议。

建议写集：治理扫描、protocol/catalog negative guard、依赖清单、许可证说明、执行计划状态和 evidence index。

负向守卫至少覆盖：

- Goose ACP、Session/Message、Recipe runtime/storage、Autonomous default 不出现在 current 主链。
- 不恢复已退役 runtime、旧 Electron 业务 IPC、生产 mock fallback。
- 新增能力仍沿 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item -> GUI`。

当前机器入口：`npm run governance:desktop-reference-boundary`。版本化证据索引为
`internal/exec-plans/codex-desktop-selective-goose-reference-evidence.json`；索引与守卫共同固定
Codex 目标依据、Goose 参考 commit/Apache-2.0 边界、Lime current owner、验证层级和未关闭平台项。

## 6. Current owner 与写集边界

| 领域                  | 唯一 current owner                           | Goose 参考只允许落点                      |
| --------------------- | -------------------------------------------- | ----------------------------------------- |
| sidecar 生命周期      | `app-server-daemon` + Electron Desktop Host  | 启动阶段、readiness、cleanup、诊断机制    |
| JSON-RPC 连接恢复     | `app-server-client`、App Server transport    | generation、timeout、退避、重连监听       |
| Agent 回合与历史      | RuntimeCore、`agent-runtime`、`thread-store` | session UX 只能转成 Thread/Turn/Item 交互 |
| 工具与权限            | `tool-runtime`、App Server                   | 工具状态和 approval 展示，不复制安全判定  |
| MCP / Apps / Skills   | 既有 MCP、Plugin、Skills current catalog     | inline UI 与表单表达，不建第二 catalog    |
| Workflow / Automation | 既有 Skills、Plugins、Scheduled Tasks        | 参数表单、retry/history 的 UX 取舍        |
| 诊断                  | 现有 diagnostics owner                       | startup journal、脱敏、有限保留           |

禁止写入的平级 owner：Goose ACP adapter、Goose session store、Goose recipe engine、Electron renderer business backend、旧 agent runtime。

## 7. Agent Verification Contract

### 7.1 基本信息

```text
改动名称：Codex Desktop 选择性 Goose 参考
执行计划文件：internal/exec-plans/codex-desktop-selective-goose-reference-plan.md
负责人：Desktop Host / App Server / agent-ui current owner
预算标签：budget:normal
风险等级：P1
影响模块：sidecar lifecycle、App Server transport、diagnostics、Thread/Turn/Item GUI、Skills/Automation UX
不做范围：ACP、Goose Session/Message、Recipe runtime、第二 scheduler、生产 mock、Codex 私有 native API 猜测
```

### 7.2 Current 主链

```text
前端入口：现有 Thread / Settings Developer / Skills / Automation 页面
前端网关：现有 typed App Server client 与 diagnostics gateway
Electron Desktop Host bridge：sidecar lifecycle、sleep/resume、诊断导出
App Server method：只复用或扩展已有 JSON-RPC method；新增 method 必须同步 protocol/schema/client/fixture
RuntimeCore / service owner：RuntimeCore、agent-runtime、tool-runtime、app-server-daemon
read model：Thread/Turn/Item、readiness/connection diagnostic summary
runtime event：App Server typed notification、sidecar lifecycle event
Evidence Pack 字段：platform、sidecar/app version、readiness、connection generation、thread/turn identity、redaction result、evidence id
GUI surface：Thread workspace、tool/approval timeline、Subagent/MCP App、Developer diagnostics
```

### 7.3 Happy Path

```text
用户输入 / Agent 输入：打开 Lime、启动 sidecar、进入 Thread、执行含工具或子 Agent 的 Turn；系统睡眠后恢复
预期 runtime events：readiness -> initialize -> thread/turn/item events；resume 触发新 connection generation
预期 tool calls：仅来自当前 Turn 冻结的 tool snapshot 和 tool-runtime policy
预期 approval / sandbox：沿现有 approval、sandbox、capability contract；缺失时 fail closed
预期 artifact：Thread/Turn/Item projection、脱敏 diagnostic bundle、GUI evidence summary
预期 evidence：deterministic fixture、targeted GUI trace、真实 Electron Gate B（按阶段）
预期 GUI 状态：启动状态可解释、恢复后状态连续、工具/审批/子 Agent 状态可重建
失败时应停在哪一层：readiness、initialize、connection recovery 或 policy boundary；不得伪造成功 Turn
```

### 7.4 Evidence Layers

| Layer               | 本计划要求 | 计划路径                                                                  | 不足时处理                     |
| ------------------- | ---------- | ------------------------------------------------------------------------- | ------------------------------ |
| deterministic-smoke | 是         | `npm run test:contracts`、Rust related tests、runtime fixture             | 阻断对应阶段                   |
| gui-trace           | P2/P3 是   | `npm run verify:gui-smoke`、Playwright/Electron evidence                  | 只能标 Gate A，不能宣称 Gate B |
| runtime-transcript  | P1/P2 是   | `npm run smoke:agent-runtime-current-fixture` 与 Thread/Turn/Item summary | 补 identity/lifecycle 断言     |
| release-artifact    | P3 是      | Windows/macOS packaged evidence、Forge/Squirrel 产物                      | 标记平台未验证，不升级完成度   |

### 7.5 必跑命令

```bash
# C0
npm run test:contracts
npm run governance:legacy-report
npm run governance:desktop-reference-boundary

# C1
npm run test:rust:related -- lime-rs/crates/app-server-daemon lime-rs/crates/app-server lime-rs/crates/app-server-client
npm run smoke:agent-runtime-current-fixture

# C2（触及 GUI / bridge / Desktop Host 时）
npm run verify:gui-smoke

# C3 / platform evidence（仅对应阶段需要）
npm run verify:local
npm run verify:app-version
```

`verify:app-version` 仅在版本、Forge、workspace manifest 或 packaged metadata 改动时运行。若新增或治理脚本，追加 `npm run governance:scripts`。

### 7.6 Agent QC 场景映射

```text
P0：sidecar readiness failure、connection generation stale-event rejection、diagnostic redaction
P1：agent runtime current fixture 的 cold start/resume/tool lifecycle
P2：Thread resume/fork、approval display、Subagent activity、MCP App pending/error、workflow retry/history
```

Supervisor 只判断用户可见状态连续性、恢复后的 identity 一致性和诊断脱敏结果；schema、bridge、mock、owner 和 evidence scope 由 contract/守卫/人工验收判断。

## 8. 完成标准、分类与回写

### 8.1 计划完成标准

- P0 矩阵冻结，所有采用项有 owner、写集、证据等级和退出条件。
- P1 启动/恢复/诊断至少有 deterministic 与真实 Electron 证据。
- P2 交互全部消费 canonical Thread/Turn/Item，未形成 Goose 平行模型。
- P3 平台证据已回写 Codex Desktop 平台 parity 主计划。
- P4 负向守卫通过，仓库没有 Goose ACP、Recipe runtime 或旧 runtime 回流。

### 8.2 分类口径

- `current`：Lime App Server JSON-RPC、RuntimeCore、Thread/Turn/Item、`app-server-daemon`、Desktop Host、typed client、现有 Skills/Plugins/Automation owner。
- `compat`：默认不新增；只有真实外部协议或数据迁移约束时才允许，并写明退出时间和删除入口。
- `deprecated`：只允许迁出，不新增业务逻辑。
- `dead`：Goose ACP/session/recipe 平行实现、旧 agent loop、旧 Electron 业务后端、生产 mock fallback；禁止恢复。

### 8.3 失败回写

| 失败类型               | 回写资产                                                      | 关闭条件                                             |
| ---------------------- | ------------------------------------------------------------- | ---------------------------------------------------- |
| readiness/cleanup 失败 | `app-server-daemon` 单测、Electron fixture、platform evidence | 进程树、stderr tail、退出状态可断言                  |
| stale connection 回写  | client/unit、App Server notification fixture                  | generation 和 identity 断言通过                      |
| 诊断泄露               | redaction test、schema guard、负向扫描                        | token/prompt/path 等敏感项零命中                     |
| GUI 状态漂移           | Thread/Turn/Item projection test、Playwright/Electron trace   | Gate B 证明真实 current bridge                       |
| Goose surface 回流     | governance negative guard                                     | ACP/session/recipe/旧 runtime 仅出现在 guard/history |

### 8.4 本轮实施记录（2026-09-03）

- 采用 Goose ACP 连接恢复中的两个局部语义：连接代际和系统恢复后的新连接；落点是现有 `ElectronAppServerHost` 与 `AppServerSidecarLifecycle.restart()`，没有引入 ACP transport 或 ACP session identity。
- `ElectronAppServerHost` 为每次连接安装单调代际守卫，旧连接的 notification、server request handler 在 lifecycle 或 generation 不匹配时直接忽略，避免旧连接回写新连接的 recent notification 或动态工具状态。
- Electron Desktop Host 监听 `powerMonitor.resume`，并以单个恢复 Promise 合并并发恢复事件；恢复只重建 stdio/App Server 连接，Thread/Turn/Item 继续由 App Server durable owner 提供。
- 当 sidecar 已进入既有自动重启退避时，系统恢复路径等待原有 restart waiter，不重复调用 `restart()`，避免同一 lifecycle 并行拉起两个 sidecar。
- 现有启动 handshake 已具备 initialize timeout 和最多 20 行 stderr 尾部，本轮未重复实现 Goose `startupDiagnostics` 的第二套 journal；诊断 bundle 已按既有 diagnostics owner 补齐 Host 阶段、退出上下文、脱敏与上限。
- `packages/app-server-client` 的 sidecar owner 现在在运行期也只保留 20 行 stderr tail，避免长时间运行无限累积原始日志；错误对象和 Host 诊断继续在边界处二次脱敏。
- 新增 `app_server_host_diagnostics` 只读 Desktop Host 命令，纳入现有 crash diagnostic bundle；它只报告启动阶段、连接代际、sidecar 运行态和最多 20 行脱敏 stderr，不进入 App Server protocol、Thread/Turn/Item 或业务 read model。
- Host 诊断在 renderer gateway 侧执行 schema fail-closed，拒绝未知阶段、负连接代际、超长或超过 20 行 stderr；凭证和常见用户绝对路径在 Electron Host 侧先脱敏。
- 负向确认：未增加 ACP、Goose `Session/Message`、Recipe runtime/storage、第二 scheduler、旧 runtime 或生产 mock fallback。
- P2 current owner 审计通过：Thread history/resume/fork/archive、tool/approval lifecycle、Subagent/MCP App/Skills、Workflow retry/history 均已有 canonical Thread/Turn/Item 或现有 scheduler owner；定向测试 145/145，未新增平行 store、Session/Message 模型或 Recipe runtime。
- Agent Runtime current fixture 通过，覆盖真实 Electron 的历史恢复、停止后继续、审批三态、active steer、Plan hydrate、Skills、MCP structured content、artifact/workbench、typed retry 和 Claw 主链；结果明确 `liveProviderUsed=false`，因此只作为 fixture/Gate B 证据，不升级为 live provider 或平台发布证据。
- 验证：`npx vitest run electron/appServerHost.test.ts`（31/31）；`npx vitest run src/lib/api/desktopHostDiagnostics.test.ts src/lib/crashDiagnostic.test.ts src/components/layout/CrashRecoveryPanel.test.tsx src/components/settings-v2/system/developer/index.test.tsx src/components/settings-v2/system/experimental/index.test.tsx electron/ipcChannels.test.ts electron/preload.test.ts`（69/69）；`npm --prefix packages/app-server-client run build`、`npx vitest run packages/app-server-client/tests/client.test.mjs`（97/97）；`npm run typecheck:electron`；`npm run test:contracts`。

### 8.5 P3 候选身份与发布 provenance 增量（2026-09-03）

- 参考选择再次校准：Codex release 会将 `GITHUB_SHA` peel 到实际 commit；Goose release 使用 GitHub/Sigstore SLSA build provenance。Lime 只采用“发布资产可追溯到不可变源码”和“证据 fail closed”机制，不采用 Goose 的 Session、Recipe、ACP、bundle runtime 或更新语义。
- 新增统一 release candidate identity 校验，完整 SHA 必须是 40 位 Git commit，run ID 必须是有界安全标识。prepare job 要求输入 ref 与 workflow commit 一致，后续 release job 一律 checkout 不可变 `github.sha`，避免排队期间 `source_ref` 漂移后生成错误 provenance；独立 Windows test workflow则绑定其明确选择并实际 checkout 的测试 commit。
- Windows Squirrel、Code Mode、native host 三份 summary 现在同时记录 `candidateSha`；packaged validator 要求 SHA、`candidateRunId`、version、installed executable、resources root 和资源 manifest 全部来自同一候选，否则生成结构化失败证据。
- macOS packaged Gate B 现在记录 platform、arch、version、SHA、run ID；release runner 额外要求顶层 app 与嵌套 native helper 都通过严格 codesign、使用同一 Developer ID Team，且 Gatekeeper assessment 和 stapler validation 通过。
- GitHub Release 资产在上传前使用固定版本的 `actions/attest-build-provenance` 生成 SLSA provenance；workflow guard 固定权限、action digest、subject path、候选身份采集和 macOS release-trust 参数，防止后续静默退化。
- 新 GitHub Release 在 source identity 校验后只创建为 draft；平台矩阵和 packaged evidence 未全部通过时不对外发布，最终 publish job 在资产 attestation 和上传成功后解除 draft。
- 为遵守仓库文件规模约束，macOS release trust、release candidate workflow guard 和 release matrix guard 已拆入独立 `scripts/electron/lib/` owner；主 macOS Gate 为 778 行，release guard 为 955 行，没有继续向超限单文件堆叠。
- 定向证据/guard 回归 `83/83`、`npm run test:contracts`（App Server client 299 checks，command/scripts/release/docs 治理通过）、`npm run typecheck:electron`、`npm run verify:app-version` 和 `git diff --check` 通过。
- 当前只完成证据合同和 release workflow 的 deterministic 门禁；没有真实触发 release runner，因此 Windows Squirrel 安装/升级/卸载、macOS Developer ID/notarization 实际结果和系统 TCC 权限撤销恢复仍是 `OPEN_REF`。

### 8.6 P4 回流守卫与 bookmark 冷启动证据（2026-09-03）

- 新增 `governance:desktop-reference-boundary` 并接入 `test:contracts`：扫描生产源码、JavaScript/Cargo manifest 与版本化 evidence index，拒绝 Goose/ACP 平行 transport、Session/Message owner、Recipe runtime/storage、Autonomous default、第二 runtime/catalog 的明确路径、symbol、protocol method 和依赖回流。
- 守卫没有粗暴禁用通用 `session` / `recipe` 数据字段；只拦截能够形成平行 owner 的组合命名。清退了无消费者的 `DiscordThreadBindingsConfig.spawn_acp_sessions` Rust/TypeScript 字段，不保留 compat。
- evidence index 固定 Goose `1.49.0` / commit `794b04a0b1f4c58378ef3738dade297c13690b77` / Apache-2.0，当前采用项均为机制参考，`codeCopied=false`、`dependencyAdded=false`；任何条目必须同时给出 Codex 目标依据、Lime owner、实现路径、验证命令和证据等级，不能只凭 `goose-static` 入场。
- macOS Gate B 改为通过 `SystemUtilityHost` stable ID owner，在同一隔离 userData 上连续启动三次真实 Electron：首次持久化；第二次冷启动 resolve/start 后在 active lease 上 revoke；第三次确认 `bookmark_unavailable`，再对同一 ID regrant/start/stop。真实本地 packaged `1.138.0` 基线证据位于 `.lime/qc/gui-evidence/macos-native-host-gate-b-bookmark-recovery/summary.json`，结果 `passed`，App Server trace 为 `electron-ipc`，console/page/invoke unexpected error 均为 0。
- 真实运行同时暴露 `macos-release-trust` 抽取后 `verifyCodeSignature` 未导出的回归，已恢复单一 helper 导出并补回归测试。contextBridge 会丢弃 Error 自定义 code，因此 Gate 只对精确的稳定 `bookmark_unavailable` code 或领域消息二选一归类，不接受任意失败。
- P4 本地验收已闭环；上述 bookmark 证据只关闭应用管理的 revoke/regrant，不代表系统 TCC 已撤销，也不代表 Developer ID/notarized/stapled release 候选。后两项继续留在 P3 external platform evidence。

## 9. 架构影响与下一刀

本文件本轮新增 Desktop Host 只读诊断 IPC 边界，已按重大架构变更更新 `internal/aiprompts/architecture.md` 并完成责任开发者确认；App Server JSON-RPC、RuntimeCore、Thread/Turn/Item 主链未改变。

P4 本地治理已完成，下一刀回到 **P3 external platform evidence**：在同一 release workflow run 上取得 Windows `windows-2022` 和 macOS release runner 产物，核对结构化 summary 与 SLSA attestation，再在临时或 self-hosted runner 补系统 TCC 撤销恢复。当前环境不会重置用户机器的 TCC，也不能伪造未运行平台证据；未覆盖项保持 `OPEN_REF`。

## 10. 2026-09-03 最终本地验收与未关闭证据

- 两个并行只读门禁已结束：`npm run test:contracts` 通过协议生成漂移、App Server client、命令契约、Harness、modality、脚本治理、release workflow、desktop reference boundary、清理报告和文档边界检查；`npm run governance:legacy-report` 摘要为扫描 2062 个源码文件、1353 个测试文件、零引用候选、零分类漂移、零边界违规。
- 修复 `lime-rs/crates/agent/Cargo.toml` 与 `lime-rs/crates/mcp/Cargo.toml` 的内部库 doctest 配置，使 workspace doctest 不再把内部模块当作独立外部 crate 编译；该修复不改变 runtime、协议或业务 owner。
- 最新 `npm run verify:local` 已通过：119/119 Vitest、contracts、治理、Rust workspace lib/tests 与 doctest、真实 Electron `verify:gui-smoke` 全部通过；另有 `npm run typecheck:electron`、`npm run verify:app-version`、`npm run governance:scripts` 和 `git diff --check` 通过。
- 当前已完成的实现仍只属于 Lime current chain：Desktop Host sidecar readiness/cleanup、connection generation 与 resume recovery、20 行脱敏 stderr tail、只读 Host diagnostics、macOS native helper/bookmark/window/display/media/Apple Events 边界，以及 release candidate identity/provenance guards。
- 选择性参考裁决没有变化：Goose 只贡献 sidecar readiness、bounded diagnostics、generation/recovery 与退避等工程机制；ACP transport、`Session/Message`、Recipe runtime/storage/catalog、Autonomous 默认、第二 scheduler/catalog/backend、renderer mock fallback 和 Codex 私有 native API 猜测均排除。
- 仍未关闭且不得本地伪造的证据：Windows `windows-2022` packaged/Squirrel 安装升级卸载及安装后 Gate B；Windows packaged sidecar/runner/native host/Code Mode 资源证据；macOS Developer ID/notarized/stapled release runner；系统 TCC 权限真实撤销/恢复；Codex Desktop 实时 accessibility/screenshot 对照；Chronicle/PIP/Computer Use 等私有能力的产品范围与系统级证据。

计划状态保持 `in-progress / P3-external-platform-evidence; P4-local-complete`，不因本地门禁通过而标记总目标完成。

## 11. 2026-09-03 Windows 生命周期断言与本地门禁复验

- Windows Squirrel RC smoke 现在显式执行已安装候选的卸载路径：调用候选安装目录自己的 `Update.exe --uninstall`，等待 `Update.exe`、候选 `app-*`、`Lime.exe` 和快捷方式消失；成功 summary 必须包含安装、升级、卸载三类断言。
- `windows-packaged-evidence.mjs` 对 packaged summary 强制检查卸载字段，避免只伪造 `result=pass` 而遗漏安装后生命周期；失败路径执行有界进程停止和临时目录清理。
- 定向回归 `npx vitest run scripts/electron/windows-squirrel-rc-smoke.test.mjs scripts/electron/windows-packaged-evidence.test.mjs` 为 `24/24`，`npm run test:contracts`、`cargo fmt --manifest-path lime-rs/Cargo.toml --all -- --check` 和 `git diff --check` 通过。
- 最新 `npm run verify:local` 在 TUI model picker 修复后完整通过：119/119 Vitest、contracts、治理、Rust workspace lib/tests/doctest 和真实 Electron GUI smoke；Rust workspace 中包含 TUI 45/45、`app-server-client` 41/41、`lime-cli` 26/26。
- TUI model picker 保持 current owner：`/model` 空命令通过 App Server `model/list` 获取非隐藏 catalog，弹层选择后只调用 `thread/settings/update`；没有引入 Goose Session/Message、ACP 或第二 catalog。该实现属于 CLI/TUI surface，不改变 Desktop runtime 主链。
- 证据边界不变：本地 deterministic/GUI 结果不能替代 Windows `windows-2022` packaged 安装升级卸载、macOS release trust、系统 TCC 撤销恢复或 Codex Desktop 实时 accessibility/screenshot 证据；这些继续标记 `OPEN_REF`。
