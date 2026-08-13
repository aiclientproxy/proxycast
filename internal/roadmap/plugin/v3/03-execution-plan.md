# Plugin v3 实施计划

状态：`in-progress / final-platform-gate`

## 写集纪律

每阶段开始前在 `internal/exec-plans/` 更新执行计划，声明精确写集、脏热区避让、
current/deprecated/dead 变化、架构影响、验证命令和阻塞。协议、App Server plugin
domain、MCP/Skills、Renderer gateway、清理守卫不能在同一阶段由不同变更重写同一事实源。

## 阶段

| 阶段 | 目标 | 主要写集 | 退出条件 |
|---|---|---|---|
| V3-0 | `complete` | v3 文档、governance scanner、现役文档回流守卫、执行计划 | 标准合同、删除分类、架构确认和旧技术文档删除已落盘 |
| V3-1 | `complete` | `plugin_catalog.rs`、MCP adapter、fixtures | 根 `plugin.json` + direct-child `skills/` + 根 `mcp.json` 已由 current loader 读取 |
| V3-2 | `complete on macOS/Unix` | MCP lowering、PLUGIN_ROOT/DATA、path/security tests、activation snapshot | current owner 与 macOS/Unix parity tests 已通过；Windows 语义仍待真实 runner |
| V3-3 | `complete` | protocol、typed client、plugin gateway、Claw、Right Surface | typed `plugin/*` 与标准包真实 Electron install/turn/tool/restore Gate B 已完成 |
| V3-4 | `complete` | package API、发布脚本、SDK/renderer/worker | 旧 protocol/client/bridge/scripts/fixtures 正向引用已清零并删除 |
| V3-5 | `complete` | processor/core plugin manager 与孤立 storage/error | 旧 manager/build hook/DAO/schema/error 已物理删除 |
| V3-6 | `in-progress` | parity evidence、文档、guards、全量验证 | Windows runner evidence 与 baseline lint 收口后才能 complete |

## 目录 UI 收敛（2026-08-13）

- `complete`：侧栏的 `Skills`、`专家` 平级入口已移除，`插件` 成为唯一目录入口。
- `complete`：`AppPageContent` 为既有 `plugins`、`skills`、`experts` 页面增加共享
  `插件 / Skills / 专家` Tab 壳，三套业务页面与路由参数仍由原 owner 承接。
- `complete`：侧栏 `插件` 在三个子页面均保持激活；Skills/专家 Tab 间切换保留当前项目
  作用域，插件 Tab 不制造额外参数。
- `complete`：共享 Tab 条显式越过 Electron 顶部拖拽层并声明 `no-drag`，避免真实桌面中
  可见按钮被透明拖拽区域吞掉点击；其余顶部空白仍保留窗口拖拽能力。
- 架构影响：非重大。未改变协议、Electron/App Server 边界、RuntimeCore 或 catalog owner，
  因此无需修改全局架构图。
- 验证退出条件：导航定义、稳定 DOM、Tab 切换、深链参数透传、五语言、lint/typecheck、
  GUI smoke 均有结果；环境阻塞必须显式记录。

### 目录 UI 验证记录

- `2026-08-13`：定向 Vitest `4 files / 38 tests` 通过，覆盖共享 Tab 的 Electron
  `no-drag` 边界、页面切换、侧栏激活态与导航定义；定向 ESLint、Prettier 和
  `git diff --check` 通过。
- `2026-08-13 / Gate A`：系统 Chrome browser mirror 中，三个 Tab 的中心点
  `elementFromPoint` 均命中对应 `BUTTON`；计算样式为 `z-index: 1001`、
  `-webkit-app-region: no-drag`。`插件 -> Skills -> 专家 -> 插件` 实际切换通过，
  console/page error 为 `0`。该证据只证明 Renderer 投影和命中测试。
- `2026-08-13 / Gate B`：使用隔离 `ELECTRON_E2E_USER_DATA_DIR` 与独立 CDP 端口连接
  真实 Electron `http://127.0.0.1:1420/?nativeStartup=1`，确认
  `window.__LIME_ELECTRON__ === true` 且 preload invoke 可用；三个 Tab 坐标均命中对应
  `BUTTON`，三次实际点击切换成功，console/page error 为 `0`。
- `2026-08-13 / GUI smoke`：修复后重新构建 Renderer/Desktop Host 并运行
  `npm run verify:gui-smoke`，结果为 `pass`；renderer、App Server、Claw shell reload 与
  memory settings 准备态均通过。
- 全量前端 TypeScript 校验仍受当前工作树既有测试类型漂移阻塞，本切片文件未出现在错误
  列表中；不得据此声称全量 TypeScript 已通过。Electron host typecheck 已随 GUI smoke 通过。

## 不可跳过的删除顺序

```text
标准 parser + negative guard
  -> 迁移 App Server/GUI/fixture
  -> 清零旧 protocol 和 scripts
  -> 清零发布与 server 构建入口
  -> 物理删除旧 package/worker/manager
  -> 更新架构和路线图事实源
  -> final legacy report + Gate B
```

任何阶段失败都只能修复 current owner；不得恢复旧实现作为 fallback。

## 架构确认点

`internal/aiprompts/architecture.md` 已更新 Plugin owner、标准 package boundary、Codex
extension boundary、MCP 安全边界和旧实现删除状态；责任开发者确认已于 2026-08-08 记录。

## 下一执行刀

1. 固化并逐项执行 [Codex parity matrix](./05-codex-parity-matrix.md)：manifest、Skills、MCP、
   installed/enabled、reload/cold restore；macOS/Unix parser edge-case 已补独立测试并标绿，
   Windows 语义仍不得用 macOS 证据替代。
2. 补 Windows junction/reparse point、环境变量大小写与 root/data 行为证据；现役文档回流守卫已纳入 `npm run docs:boundary`。
3. contracts、治理、Rust related、Agent fixture 和 GUI smoke 已完成并通过；`lime-mcp`
   全量为 160/160，Rust related 通过。旧 `apps_jsonrpc` fixture 已迁到标准根 manifest +
   Codex Apps 配置路径 adapter；Apps JSON-RPC、adapter unit、runner guard 与迁移后的真实
   macOS Apps Gate B 均通过。2026-08-09 已从头完成 `verify:local`：版本/i18n/lint、前端
   120 批、contracts、Rust workspace 全量、Bridge 与 GUI smoke 均通过。
   Windows 交叉检查另受缺少 `assert.h`、MSVC/Windows SDK、linker 和 runtime 阻塞，不能
   作为 Windows parity 证据。

macOS 标准包 Gate B 已于 2026-08-09 通过，证据位于
`.lime/qc/gui-evidence/plugin-package-electron-gate-b/plugin-package-electron-gate-b-summary.json`。

2026-08-09 重跑后仍通过；Gate B 的 MCP App 等待器已修复累计计数竞态，定向守卫为
2 files / 18 tests。真实证据包含首次恢复、renderer reload、cold restart、卸载后历史读取，
resource/HTML 累计为 4/4，且未命中 legacy facade、production mock fallback 或 console error。

Codex Apps extension fixture 已迁到标准根 manifest + `apps: "./apps.json"` adapter，并于
2026-08-09 通过 Apps 专用真实 Electron Gate B。证据路径为
`.lime/qc/project-gates/standalone-apps-catalog-20260809T054741397Z-147740/apps-catalog-gate-b/apps-catalog-gate-b-summary.json`；
七个 required method 全部命中，pending -> disabled fresh read 完成，错误与 mock/legacy 命中均为 0。

Content Factory Article Workspace 聚合 Agent fixture 已于 2026-08-09 重跑通过，证据位于
`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-content-factory-article-workspace-regression-summary.json`；
场景创建的 canonical session identity 已贯穿 Article Editor、read model、reload/cold restore
与 Gate B，`artifact-article-1`、workspace patch evidence 和 `source: workspace_patch` 均已投影。
