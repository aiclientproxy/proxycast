# Plugin v3 清理账本

状态：`active / major-deletions-complete`

## 清理原则

1. 先建立标准 current owner 和负向守卫，再迁移调用者，最后物理删除。
2. 不做长期双读、双写、自动转换、旧 worker fallback 或空壳 facade。
3. 删除前必须证明构建图、生产入口、正向 fixture 和脚本引用清零。
4. 历史文档可以留作 evidence，但不得被导航、测试或实现当作 current source。

## 迁移后保留

| Surface | 处理 |
|---|---|
| App Server plugin domain | 继续作为唯一 catalog/install/activation owner |
| `lime-mcp`、Skills、RuntimeCore、tool-runtime | 继续作为领域 owner；增加标准 lowering，不承载 package discovery |
| typed Renderer gateway、Claw mention、Right Surface | 继续消费 projection，删除本地 registry 合并 |
| Codex extension adapter | 仅保留明确外部生态需要的 adapter；禁止新增 Lime 私有 manifest 字段 |

## 已删除

| Surface | 已删除内容 | 回流门槛 |
|---|---|---|
| Lime 私有 manifest/package projection | `lime.plugin.package.v1`、`plugin_packages/**`、旧 protocol/schema/client | 只能出现在 retired guard 或历史 evidence |
| 私有 MCP 选择 | `.mcp.json`、manifest 内联/自定义 `mcpServers` | portable loader 只读根 `mcp.json` |
| 旧 renderer/install/runtime | manifest registry、SDK、shell、readiness、publish/package UI | Renderer 只能消费 typed `plugin/*` projection |
| 旧 Electron/App Server worker | `pluginRuntimeTaskHost`、`pluginTaskWorker`、`plugin_worker_*` | Desktop Host 只能转发 App Server JSONL |
| 旧 Lime plugin manager | `lime-core/src/plugin/**` 与 processor Plugin hook | 不得恢复 workspace/build 依赖 |
| 旧存储和错误面 | `installed_plugins` DAO/schema、`StepError::Plugin`、`ProcessError::PluginError` | installed state 只归 App Server v3 store |
| 旧命令 policy | `plugin_runtime_*`、`pluginUiRuntime/*`、`pluginInstalled/*` 特殊 timeout/no-mock/current catalog | current catalog 只登记七个 `plugin/*` 方法 |
| MCP 私有 runtime fixture | `--allow-plugin-runtime-fixture`、`plugin_runtime_capabilities`、`plugin_mcp_targets` 正向 smoke、transport summary 与五语孤立文案 | Rust inventory 负向测试与 smoke source guard 禁止私有投影回流 |
| 孤立 connector smoke | 无 npm/CI/文档入口且仍调用 `plugin_runtime_get_task` 的 `connector-outbox-smoke.mjs` | `scripts/plugin/` 只保留 connector production delivery current 检查 |
| Plugin v2 active 执行计划 | `plugin-v2-current-plan.md` 与 exec-plan current 导航 | v3 执行计划是唯一 active Plugin 计划；v2 roadmap 仅保留历史快照 |

## 剩余 residual

| 分类 | Surface | 退出条件 |
|---|---|---|
| `verification-gap` | Windows junction/reparse point、环境变量大小写与数据根行为 | Windows 真实环境通过验证矩阵 |
| `verification-gap` | Codex parity 独立矩阵与最终分层门禁 | parity 逐项可审计，contracts、治理、Rust related、Agent fixture、GUI smoke、`verify:local` 全通过 |

## 已完成门槛：标准包 Gate B

macOS arm64 独立 Electron Gate B 已证明根 `plugin.json`、根 `mcp.json`、合法 Skill、
canonical Plugin mention、真实 Agent turn、canonical MCP Tool Item、elicitation、Right
Surface、reload/cold restore 与卸载后历史读取；worker、legacy command、生产 mock fallback
和 console error 命中均为零。证据位于
`.lime/qc/gui-evidence/plugin-package-electron-gate-b/plugin-package-electron-gate-b-summary.json`。

## 已完成门槛：旧 server manager

`lime-core::plugin::PluginManager`、processor Plugin hook 与孤立 error variant 已退出构建图并
物理删除。`lime-server`/`lime-processor` 不再拥有 Plugin 业务事实源；任何恢复都属于回流。

## 数据处置

仓库无外部用户和历史兼容负担，因此不实现长期数据迁移器。旧 installed/setup/cache：

- v3 不读取、不转换、不继承；
- 新数据库不再创建 `installed_plugins`；既有孤立表不进入 runtime 或迁移输入；
- Thread 历史保留可读性，但不恢复旧 worker 或旧能力执行。

## 回流守卫

生产代码、协议 catalog、脚本、current 文档禁止出现以下语义：

```text
lime.plugin.package.v1
contributions.runtime
contributions.workbench
app.runtime.yaml
pluginUiRuntimeStart / pluginUiRuntimeStatus / pluginUiRuntimeStop
pluginLocalPackage/inspect / pluginLocalPackage/export / pluginPackage/fetchCloud
pluginInstalled/* / pluginShell/* / pluginUiRuntime/* / plugin_runtime_*
```

`.codex-plugin/plugin.json` 只能出现在 Codex extension adapter、历史 evidence 或负向测试；
标准 loader 必须只认根 `plugin.json`。
