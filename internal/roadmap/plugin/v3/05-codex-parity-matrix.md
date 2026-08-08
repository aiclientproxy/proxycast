# Codex Parity 与平台矩阵

状态：`in-progress / independently auditable`

更新时间：2026-08-08

## 判定口径

本矩阵只比较可观察行为和测试证据，不复制 Codex 私有存储、TUI 或内部类型。Agent
Plugins v1.0.0 portable contract 优先；Codex 的 `.codex-plugin/plugin.json` 只作为
显式 extension adapter，不作为 portable loader 输入。

状态含义：

- `verified`：Lime 有 current owner 和正向/负向测试，且已有对应真实证据。
- `implemented-test-gap`：实现已覆盖该语义，但独立测试尚未达到 Codex 对照粒度。
- `platform-gap`：需要 Windows 真机或 Windows 文件系统语义，当前 macOS 无法证明。
- `intentional-adapter`：Codex 私有扩展边界，Lime 只保留适配，不复制 legacy loader。

## 行为矩阵

| 能力 | Codex 基线 | Lime current owner / 证据 | 状态 |
|---|---|---|---|
| 根 `plugin.json`、标准 schema、`name`、可选 `version` | `codex-rs/core/tests/suite/plugins.rs:168-196`；Codex manifest/resource model | `lime-rs/crates/app-server/src/local_data_source/plugin_catalog.rs:1014-1182`；`plugin_catalog/tests.rs:127-160,205-238` | `verified` |
| 未知 manifest 字段报告并忽略，非法类型 fail closed | Codex manifest/provider 解析与 schema tests | `plugin_catalog.rs:1014-1062`；`plugin_catalog/tests.rs:161-218` | `verified` |
| `skills/` 只发现直接子目录，嵌套目录不进入 catalog | `core/tests/suite/plugins.rs:168-198,446-504` | `plugin_catalog.rs` direct-child discovery；`plugin_catalog/tests.rs:141-158`；Agent runtime fixture | `verified` |
| Codex extension metadata precedence | Codex `.codex-plugin` fixture 与 extension merge tests | `plugin_catalog/tests.rs:161-202`；`plugin_catalog.rs:1111-1160` | `intentional-adapter` |
| 根 `mcp.json`、标准 MCP schema、manifest 不内联 MCP | `core/tests/suite/plugins.rs:662-752`；Codex `agent_plugin_config.rs:38-108` | `plugin_catalog.rs:488-506,856-870`；`plugin_catalog/tests.rs:300-378` | `verified` |
| stdio bare/`./` command、`PLUGIN_ROOT`/`PLUGIN_DATA` 注入与 placeholder lowering | Codex `agent_plugin_config.rs:150-245`；`core/tests/suite/plugins.rs:662-752` | `lime-rs/crates/mcp/src/agent_plugin_config.rs:152-250`；测试 `505-538`；Plugin Gate B MCP tool | `verified` |
| cwd/command containment、`..`、portable separators | Codex `agent_plugin_config.rs:349-428`；`plugin_config_tests.rs:129-206,358-504` | `agent_plugin_config.rs:394-414`；`plugin_catalog/tests.rs:380-412` | `implemented-test-gap` |
| Streamable HTTP URL、loopback HTTP、userinfo/fragment、headers | Codex `agent_plugin_config.rs:265-332`；`plugin_config_tests.rs:129-206,432-458` | `agent_plugin_config.rs:253-346`；parser sibling isolation test `541-553` | `implemented-test-gap` |
| explicit `null`、unknown server shape、SSE fail closed | Codex `plugin_config_tests.rs:208-326,432-458` | `agent_plugin_config.rs:123-149,287-291` | `implemented-test-gap` |
| 单个非法 server 不影响健康 sibling | Codex `plugin_config_tests.rs:208-232` | `agent_plugin_config.rs:109-120`；`plugin_catalog/tests.rs:380-412` | `verified` |
| symlink/越界路径 fail closed | Codex `agent_plugin_config.rs:409-482`；`plugin_config_tests.rs:358-432` | `agent_plugin_config.rs:383-454`；`plugin_catalog/tests.rs:224-238` | `implemented-test-gap` |
| install digest、同版本幂等/冲突、optional version | Codex plugin provider/load outcome and package fixtures | `plugin_catalog/tests.rs:34-125` | `verified` |
| enabled state、activation snapshot、MCP/Skill turn context | Codex `load_outcome.rs:38-205`；`core/tests/suite/plugins.rs:446-752` | `plugin_catalog.rs:392-486`；`plugin_catalog/tests.rs:34-86`；Agent fixture | `verified` |
| reload/cold restore、卸载后历史可读且不重跑旧 worker | Codex turn/plugin integration semantics | `.lime/qc/gui-evidence/plugin-package-electron-gate-b/plugin-package-electron-gate-b-summary.json` | `verified`（macOS arm64） |

## Windows 矩阵

以下项目不能用 macOS 结果替代，当前统一标记为 `platform-gap`：

| 项目 | Codex 规则/实现 | Lime 当前实现 | 缺失证据 |
|---|---|---|---|
| 环境变量大小写与重复键 | `agent_plugin_config.rs:177-199,257-263` | `agent_plugin_config.rs:177-197,294-300` | Windows 真实进程与 duplicate case-insensitive fixture |
| drive-relative / UNC / extended path | `agent_plugin_config.rs:158-175,484-494` | `agent_plugin_config.rs:160-175,456-466` | Windows `C:relative`、UNC、`\\?\` 矩阵 |
| junction/reparse point containment | Codex path resolver `agent_plugin_config.rs:431-482` | Lime path resolver `416-454` | Windows junction/reparse 创建、越界与清理 |
| root/data 持久目录 | Codex `PLUGIN_DATA` runtime fixture `plugins.rs:707-735` | Lime MCP lowering `103-107,224-230` | Windows AppData/AgentRoot 真实路径与冷恢复 |
| install 中断、恢复、卸载清理 | Codex plugin integration fixtures | Lime store tests + macOS Gate B | Windows packaged Electron Gate B |

## 结论与退出条件

- 当前不能把 Plugin v3 标为 `complete`：仍有 `implemented-test-gap` 和 `platform-gap`。
- 不新增 compat wrapper，不恢复旧 package/worker/SDK；测试缺口只能在 current owner
  补齐，Windows 缺口只能通过 Windows 真实矩阵消除。
- V3-2 的下一刀是把 `implemented-test-gap` 行补成可独立运行的 parser tests；V3-6
  的下一刀是运行 Windows junction/reparse/env/root-data/install matrix，再重跑最终
  contracts、Agent fixture、GUI smoke 与 `verify:local`。
