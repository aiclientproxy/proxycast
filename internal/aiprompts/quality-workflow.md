# 工程质量工作流

状态：current / Refactor v2

本页定义 Lime 的最低交付门禁。测试策略与作者规则见 [../test/testing-strategy-2026.md](../test/testing-strategy-2026.md)，第二期覆盖计划见 [../roadmap/benchmark/README.md](../roadmap/benchmark/README.md)。

## 唯一受测产品链

```text
Electron Desktop Host
  -> App Server JSON-RPC
  -> RuntimeCore
  -> Thread / Turn / Item projection
  -> GUI
```

测试必须说明覆盖了链路中的哪一段。旧命令、旧 runtime、生产 mock fallback、外部数据集 runner 或历史 release report 通过，均不能证明 current 产品链可交付。

## 选择最低门禁

先跑最贴风险的定向测试，再按跨层影响扩大。全量检查不能替代真实 GUI 或跨层证据；跨层证据还必须说明是否实际经过目标进程边界。

| 改动 | 最低验证 |
| --- | --- |
| 纯 TypeScript selector/projection/parser | `npm run test:related -- <paths...>`；必要时 lint/typecheck |
| React 组件/hook | related unit/component；用户可见变更补五语言与稳定 DOM 断言 |
| Rust crate | `npm run test:rust:related -- <paths...>`；跨 crate 行为追加 integration layer |
| App Server method/protocol/schema/generated client | `npm run test:contracts` + 公共 JSON-RPC 集成测试 |
| Agent loop/queue/turn/item/read model | Rust related/integration + `npm run smoke:agent-runtime-current-fixture` |
| tool/approval/sandbox/context | owner 集成测试 + current runtime fixture；命令边界追加 contracts |
| MCP/Skills/Multi-Agent | owner 集成测试 + `npm run smoke:mcp-current` 或对应 current fixture |
| Provider/lowering/multimodal | provider request capture + Rust related/integration + modality contracts |
| GUI 壳/Workspace/主路径 | related tests + `npm run verify:gui-smoke` + 风险匹配的 Gate A |
| Electron/preload/IPC/read model/恢复 | `npm run test:contracts` + 对应真实 Electron Gate B fixture |
| 版本/Forge/workspace manifest | `npm run verify:app-version`；release 边界追加 Forge guard |
| 脚本目录 | `npm run governance:scripts` |
| legacy/dead surface 删除 | `npm run governance:legacy-report` + 相关负向回流守卫 |

默认本地入口是 `npm run verify:local`。需要完整本地门禁时使用 `npm run verify:local:full`。前端全量测试中断后使用 `npm run test:resume`，不要从第一批无差别重跑。

## 证据等级

| 等级 | 证明内容 | 不能证明 |
| --- | --- | --- |
| Unit | 纯转换、状态转换、selector/projection 的确定性 | 跨模块接线、进程边界 |
| Domain integration | current Rust/TS owner 与可控依赖的协作 | App Server 公共协议、Electron |
| App Server integration | public JSON-RPC、notification、read model、恢复 | Renderer/Electron 可见状态 |
| Current fixture | RuntimeCore/provider fixture/tool/event terminal 主链 | live provider 或真实桌面壳，除非 fixture 显式启动 Electron |
| Gate A | browser/Renderer projection、DOM、交互、可见状态 | Electron main、preload、IPC |
| Gate B | Electron、preload/IPC、`app_server_handle_json_lines`、App Server、runtime/read model、GUI | live provider，除非场景显式使用并记录 |
| Live/eval | 指定 provider/model/config 下的能力与稳定性 | 其他 provider、地区或平台的普遍正确性 |
| Platform/packaged | 实际 macOS/Windows 和打包产物 | 未运行平台 |

`npm run test:e2e` 是 Vitest 的 e2e 分层入口，不等于 Electron Gate B。Gate A 不能替代 Gate B。

## 测试作者合同

对齐 Codex 的测试标准：

1. Agent 逻辑变更优先写 integration test；单元测试只守住真正独立的代数、转换和状态转移。
2. App Server 测试必须从 public JSON-RPC 进入，不直接调用私有 handler 伪造跨层成功。
3. Provider fixture 必须捕获结构化请求；优先断言完整 request、event、Thread/Turn/Item 或 read model 对象。
4. 等待业务事件或 terminal predicate，不用固定 sleep、长 grace timer 或测试侧合成 `completed`。
5. 测试不修改进程全局环境；从 builder 注入路径、配置、时钟、provider 和 capability。
6. 每个测试使用隔离 app data、workspace、数据库和端口；不得依赖真实用户目录或上一个测试留下的状态。
7. 不测试静态定义值，不为已删除逻辑保留正向行为测试。旧名字只允许出现在负向 guard 或不可变 evidence。
8. 测试 helper 只承接环境与观察能力，不暴露第二套 production API；仅一个测试使用的 helper 留在测试文件内。
9. 测试失败信息必须包含 scenario/identity/expected/actual，不能只报 timeout。
10. 修复线上或 Gate 缺陷时，在最接近根因的 owner 层补回归；必要时再补一条跨层产品证据。

## 前端分层

- `*.unit.test.ts`：纯 View Model、projection、selector、parser、formatter、state machine、request builder。
- `*.component.test.tsx`：React 渲染、DOM 事件、hook 生命周期和少量关键接线。
- `*.contract.test.ts`：protocol、typed gateway、Desktop Host、App Server client、catalog 边界。
- `*.integration.test.ts`：文件系统、进程、本地 server、数据库或多模块流程。
- `*.e2e.test.ts`：Vitest 内的显式产品级流程；仍需按证据等级说明是否触达 Electron。
- `*.live.test.ts`：真实网络/provider，默认跳过并要求显式授权。

复杂组件中的筛选、分组、状态机和 runtime 参数投影必须先抽到纯 owner。component test 不重复铺满业务分支；`npm run test:layers:stats` 的 migration candidates 作为治理输入，不是失败门禁。

## Rust 分层

Rust 先 related、后 integration、再按风险扩大：

```bash
npm run test:rust:related -- <paths...>
npm run test:rust:integration:related -- <paths...>
npm run test:rust:unit -- -p <crate> <filter>
npm run test:rust:integration -- -p <crate> --test <target>
```

Agent/runtime 核心行为需要跨 owner 时，不把所有 case 堆进实现文件的 inline unit tests；使用专用测试模块或 integration target。`cargo nextest` 只在工具链、archive 和 CI shard 稳定后作为执行加速器，不改变本地 related-first 规则。

Windows restricted execution 改动必须把 `windows-sandbox-runner.exe` 与 `app-server.exe`、`code-mode-host.exe`、`windows-sandbox-setup.exe` 成组构建。Electron Windows resources 和 `app-server.release.json` 必须包含 runner 及其 `windowsSandboxRunnerSha256`，`verify-package-resources` 对缺失或 digest 漂移 fail closed。源码/静态门禁不替代 `windows-restricted-execution-evidence-v3` 七项真实 Windows/MSVC 矩阵；矩阵通过前 readiness 保持 `Planned/enforced=false`。

## Gate A 与 Gate B

Gate A 使用可控数据验证 Renderer projection、DOM、交互、错误文案和五语言资源。它可以使用显式 fixture，但证据必须标明 `test-only`。

Gate B 必须同时证明：

1. 真实 Electron Desktop Host 和 preload/contextBridge 存在。
2. IPC 命中 `app_server_handle_json_lines`。
3. 命中场景声明的 current App Server method。
4. runtime/read model/GUI 使用同一 thread/turn/item identity。
5. production mock fallback 命中为零。
6. 场景以真实 terminal 或明确 pending 状态结束，不靠固定等待猜测。

需要人工点击、截图或复用会话时继续阅读 [playwright-e2e.md](playwright-e2e.md)。

## 生产与测试边界

- 生产 Renderer、Electron、App Server 和 GUI smoke 不得回退 `defaultMocks`、`mockPriorityCommands`、`invokeMockOnly`、renderer mock 或 App Server mock backend。
- 测试可以显式使用 `src/lib/desktop-host/` fixture、unavailable backend 或受控 external backend，但不得把它们作为可交付的生产链证据，也不能冒充 live provider 证据。
- 用户可见文案覆盖 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR`；protocol enum/schema/evidence facts 不本地化。
- live test 必须显式授权、隔离凭证、记录模型与配置，并禁止保存 secret、完整真实用户内容或敏感本地路径。

## 旧测试处理

| 分类 | 处理 |
| --- | --- |
| current | 受测 owner 与 v2 一致，保留并赋稳定场景 ID |
| rewrite | 风险仍有效但入口/fixture/断言过时，迁移后删除原测试 |
| merge | 重复 boundary/source-string guard 合并到 owner 级测试或治理扫描 |
| dead | 测静态值、旧命令正向行为、已删除 runtime 或脱离构建图，直接删除 |
| live-only | 移出默认门禁，显式授权运行 |

不要为了维持旧报告或旧命令可运行而恢复 wrapper、catalog、fixture 或 mock。

## 架构确认与汇报

重大架构变更必须更新 [architecture.md](architecture.md)，并由责任开发者在执行计划和 PR 描述完成架构确认。仅新增或重写测试通常不构成架构变更；如果为了可测性改变 public boundary、owner 或依赖方向，则按重大变更处理。

交付汇报必须说明：风险类型、实际命令、证据等级、未执行原因、GUI/Gate B 状态、remaining blocker。路线图任务还要说明本轮证明了哪条主链，而不只写“测试通过”。

## 常用入口

```bash
npm run test:related -- <paths...>
npm run test:rust:related -- <paths...>
npm run test:contracts
npm run smoke:agent-runtime-current-fixture
npm run verify:gui-smoke
npm run governance:legacy-report
npm run verify:local
```
