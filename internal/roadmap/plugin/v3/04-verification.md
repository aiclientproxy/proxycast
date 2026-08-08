# Plugin v3 验证合同

状态：`in-progress`

独立逐项对照见 [Codex parity 与平台矩阵](./05-codex-parity-matrix.md)。该矩阵区分
行为已实现、测试证据缺口和 Windows 平台缺口；未补齐前不得标记 v3 complete。

## 标准包矩阵

必须使用固定 fixture 覆盖：

- 最小标准包：根 `plugin.json`，只有 `$schema`、`name`。
- Skills 包：多个直接子目录；嵌套孙目录不得被发现。
- MCP stdio：`./` command、`args/env/cwd` placeholder、PLUGIN_DATA 持久目录。
- MCP Streamable HTTP：`headers`、URL userinfo/fragment、HTTP/HTTPS 和 loopback 规则。
- Codex extension：显式 adapter 可读；portable loader 不读取 `.codex-plugin`。
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
| App Server package/store tests | `passed previously` | 标准安装、digest、optional version、Skills、Codex extension、MCP、enabled/uninstall |
| MCP parser tests | `passed previously` | placeholder、reserved env、cwd、HTTP、failure isolation |
| DevBridge/catalog 定向 tests | `passed` | 2026-08-08：4 files，270 tests |
| legacy report | `passed` | 2026-08-08：边界违规 0、分类漂移候选 0 |
| Plugin catalog/mention 定向 tests | `passed` | 2026-08-08：6 files，36 tests；canonical mention 为 `plugin://<name>@<marketplace>` |
| Plugin Gate B runner 定向 tests | `passed` | 2026-08-08：2 files，6 tests；Plugin 场景不依赖无关 dynamic tool |
| `test:contracts` | `pending rerun` | 本轮 catalog/policy/guard 变更后必须重跑 |
| Rust related tests | `pending rerun` | 本轮 core/processor dead surface 删除后必须重跑 |
| Agent fixture / GUI smoke | `pending rerun` | 证明未影响 current Agent/GUI 主链 |
| 标准包 Plugin Gate B | `passed on macOS` | 2026-08-08：根 `plugin.json`/`mcp.json`、合法 Skill、canonical mention、真实 install/turn/MCP Item/Right Surface/cold restore/卸载历史闭环；mock 0、console error 0 |
| Windows matrix | `missing` | junction/reparse point、env 大小写、root/data 行为未验证 |
| Codex parity matrix | `in-progress` | `05-codex-parity-matrix.md` 已建立；parser edge-case 仍需补独立测试 |

## Gate B

真实 Electron 证据必须证明：

1. 根标准包从目录/marketplace 进入 `plugin/list`、`plugin/read`、`plugin/install`。
2. preload/IPC 只转发 `app_server_handle_json_lines`。
3. App Server 生成 installed/enabled activation snapshot。
4. Skills/MCP 进入真实新 thread/turn；MCP tool 使用标准 `mcp.json`。
5. canonical Item、Right Surface 和 Renderer reload/cold restore 使用同一 identity。
6. 卸载后历史可读但不会重跑旧 worker。
7. worker、legacy command、renderer mock fallback 命中数为零。

2026-08-08 的 macOS arm64 证据位于
`.lime/qc/gui-evidence/plugin-package-electron-gate-b/plugin-package-electron-gate-b-summary.json`，
其中 `standardManifestSeen`、`standardMcpConfigSeen`、`pluginSkillProjected`、
`pluginSkillContextSeen`、`rendererConfirmedSubmitted`、`mcpLedgerAccepted`、
`coldRestoreCompleted`、`historyReadableAfterUninstall` 均为 `true`，
`pluginMentionPath` 为 `plugin://mcp-elicitation-plugin@local`，
`productionMockFallbackHitCount` 为 `0`。

## 跨平台

macOS 与 Windows 必须分别验证：

- plugin root/data root 解析；
- command/cwd containment；
- 环境变量大小写；
- symlink/junction/reparse point 行为；
- 安装中断、恢复、卸载和目录清理。

## 完成判定

只有同时满足以下条件才能标记 v3 complete；当前不得标记完成：

- 标准包 contract 与 Codex parity matrix 全通过。
- 旧 package、旧 worker、旧 manager、旧 protocol 和旧脚本无生产正向引用。
- `governance:legacy-report` 通过，历史文本只存在于显式 evidence/negative guard。
- `npm run verify:local`、协议合同、Rust 定向测试和 Gate B 均通过。
- 未验证项、平台缺口和剩余风险写回执行计划，不用“兼容”掩盖。
