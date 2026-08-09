# Plugin v3 验证合同

状态：`in-progress / Windows-platform-gap`

独立逐项对照见 [Codex parity 与平台矩阵](./05-codex-parity-matrix.md)。该矩阵区分
行为已实现、测试证据缺口和 Windows 平台缺口；未补齐前不得标记 v3 complete。

## 标准包矩阵

必须使用固定 fixture 覆盖：

- 最小标准包：根 `plugin.json`，只有 `$schema`、`name`。
- Skills 包：多个直接子目录；嵌套孙目录不得被发现。
- MCP stdio：`./` command、`args/env/cwd` placeholder、PLUGIN_DATA 持久目录。
- MCP Streamable HTTP：`headers`、URL userinfo/fragment、HTTP/HTTPS 和 loopback 规则。
- Codex extension：显式 adapter 可读；portable loader 不读取 `.codex-plugin`。
- Codex Apps：extension 只声明包内相对配置路径；独立 Apps JSON 投影 connector id；旧
  inline object fail closed，非法 Apps 配置隔离。
- 非法 manifest、非法 mcp.json、单个非法 server、symlink/越界路径、超预算 package。

## 最低命令

```bash
npm run governance:legacy-report
npm run test:contracts
npm run test:rust:related -- lime-rs/crates/app-server lime-rs/crates/mcp lime-rs/crates/skills
npm run smoke:plugin-package-electron-gate-b
npm run smoke:agent-runtime-current-fixture
npm run verify:gui-smoke
```

Rust 定向测试至少覆盖 `app-server`、`mcp`、`skills`、`runtime-core`、`tool-runtime`；
协议变更必须同步 schema、generated client、catalog 和 fixture。

## 当前证据

| 验证 | 状态 | 证据/缺口 |
|---|---|---|
| App Server package/store tests | `passed` | 标准安装、digest、optional version、Skills、Codex extension、MCP、enabled/uninstall |
| Codex Apps adapter / exact JSON-RPC | `passed` | 2026-08-09：adapter unit 1/1、`apps_jsonrpc` 1/1、Gate B runner guard 1/1；标准根 manifest + extension Apps path + 独立 Apps JSON |
| Codex Apps Electron Gate B | `passed on macOS` | 2026-08-09：真实 Electron/preload/IPC、required methods 7/7、install notification、pending -> disabled fresh read；errors/mock/legacy 全为 0 |
| MCP parser tests | `passed` | 2026-08-09：`lime-mcp` 160 tests 全通过；Agent Plugins parser macOS 8/8，覆盖 placeholder、opaque args/env、reserved env、cwd/path、HTTP headers/URL、explicit null、failure isolation、symlink 越界 |
| DevBridge/catalog 定向 tests | `passed` | 2026-08-08：4 files，270 tests |
| legacy report | `passed` | 2026-08-09：零引用候选 0、分类漂移候选 0、边界违规 0 |
| scripts governance | `passed` | 2026-08-09：冻结基线通过，retired/untracked root 与一级目录均为 0 |
| Plugin catalog/mention 定向 tests | `passed` | 2026-08-08：6 files，36 tests；canonical mention 为 `plugin://<name>@<marketplace>` |
| Plugin Gate B runner 定向 tests | `passed` | 2026-08-09：2 files，18 tests；包含 MCP App 累计 resource/HTML 等待器回归；Plugin 场景不依赖无关 dynamic tool |
| Plugin Lab i18n cleanup | `passed` | 2026-08-08：五语言删除 780 条旧 Lab/sidebar 资源；JSON 结构、i18n coverage 100%、治理目录与负向测试通过 |
| `test:contracts` | `passed` | 2026-08-09：protocol 915 类型无漂移（923 definitions，跳过 8 个 envelope/meta）；App Server client 301 checks；command/harness/modality/release/docs 全通过 |
| Rust related tests | `passed` | 2026-08-09：`lime-mcp` 全量 160/160；app-server、mcp、skills related gate 全通过，GoalContinuation 回归已修复 |
| Agent fixture | `passed` | 2026-08-09：current fixture 全场景通过，包含 Content Factory Article Editor 聚合闭环；canonical session identity、`artifact-article-1`、workspace patch evidence、reload/cold restore 和 Gate B 一致性均通过；证据 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-content-factory-article-workspace-regression-summary.json` |
| `verify:gui-smoke` | `passed` | 2026-08-09：Electron renderer/host build、真实 App Server 初始化、reload、memory settings smoke 通过；最新 evidence `standalone-shell-01-20260809090458-56606` |
| 标准包 Plugin Gate B | `passed on macOS` | 2026-08-09：根 `plugin.json`/`mcp.json`、合法 Skill、canonical mention、真实 install/turn/MCP Item/Right Surface/cold restore/卸载历史闭环；resource/HTML=4/4、mock 0、console error 0 |
| Windows matrix | `platform-gap / evidence-missing` | `build-windows-test` 已独立运行 `lime-mcp agent_plugin_config` 并上传 `lime-windows-agent-plugin-path-contract-tests`，同时运行 packaged Plugin Gate B；junction/reparse point、env 大小写、root/data、packaged Gate B 尚无真实 Windows runner artifact |
| Codex parity matrix | `in-progress / Windows gap` | macOS/Unix parser edge-case 独立测试已通过；Windows env/path/junction/root-data/install 证据仍缺失 |
| `verify:local` | `passed on macOS` | 2026-08-09 从头通过版本/i18n/lint、前端 120 批、contracts、Bridge、Rust workspace 全量与 GUI smoke；Windows platform-gap 仍未消除 |
| Plugin v3 现役文档回流守卫 | `passed` | `npm run docs:boundary` 检查 Writing、插件发布文档和 aiprompts 导航，不允许旧技术标准或旧 manifest 合同回流 |

## Gate B

真实 Electron 证据必须证明：

1. 根标准包从目录/marketplace 进入 `plugin/list`、`plugin/read`、`plugin/install`。
2. preload/IPC 只转发 `app_server_handle_json_lines`。
3. App Server 生成 installed/enabled activation snapshot。
4. Skills/MCP 进入真实新 thread/turn；MCP tool 使用标准 `mcp.json`。
5. canonical Item、Right Surface 和 Renderer reload/cold restore 使用同一 identity。
6. 卸载后历史可读但不会重跑旧 worker。
7. worker、legacy command、renderer mock fallback 命中数为零。

2026-08-09 的 macOS arm64 证据位于
`.lime/qc/gui-evidence/plugin-package-electron-gate-b/plugin-package-electron-gate-b-summary.json`，
其中 `standardManifestSeen`、`standardMcpConfigSeen`、`pluginSkillProjected`、
`pluginSkillContextSeen`、`rendererConfirmedSubmitted`、`mcpLedgerAccepted`、
`coldRestoreCompleted`、`historyReadableAfterUninstall` 均为 `true`，
`pluginMentionPath` 为 `plugin://mcp-elicitation-plugin@local`，
`productionMockFallbackHitCount` 为 `0`，`mcpAppResourceReadCount` 与
`mcpAppHtmlLoadCount` 均为 `4`，`mcpAppToolCallCount` 为 `1`。

## 跨平台

macOS 与 Windows 必须分别验证：

- plugin root/data root 解析；
- command/cwd containment；
- 环境变量大小写；
- symlink/junction/reparse point 行为；
- 安装中断、恢复、卸载和目录清理。

本机 macOS 不能提供 Windows 证据：`x86_64-pc-windows-msvc` target 的 `ring` C 依赖在
交叉构建时缺少 `assert.h`，且没有 MSVC/Windows SDK、linker 或 runtime。Windows parser、
文件系统语义与 Electron Gate B 必须在 Windows runner/真机完成。

## 完成判定

只有同时满足以下条件才能标记 v3 complete；当前仍不得标记完成：

- 标准包 contract 与 Codex parity matrix 全通过。
- 旧 package、旧 worker、旧 manager、旧 protocol 和旧脚本无生产正向引用。
- `governance:legacy-report` 通过，历史文本只存在于显式 evidence/negative guard。
- `npm run verify:local`、协议合同、Rust 定向测试和 Gate B 均通过（macOS）；Windows
  junction/reparse/env/root-data/packaged Gate B 证据仍是交付前 blocker。
- 未验证项、平台缺口和剩余风险写回执行计划，不用“兼容”掩盖。
