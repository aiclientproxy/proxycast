# 旧 Plugin 文档提炼矩阵

状态：`historical-audit-complete`

更新时间：2026-08-04

## 目的

旧 Plugin 文档包含已经验证过的产品直觉、交互方案和问题记录，也包含与 Codex/current 主链冲突的历史实现假设。本文件把两者拆开，避免“保留文档”被误解为“保留旧事实源”。

原则只有一条：历史文档可以被阅读和提炼，v2 合同才可以驱动实现。

## 结论总览

| 旧内容                                                   | 处理                                    | v2 落点                                                                                                                                              |
| -------------------------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Plugin 是分发、授权和能力组织的根对象                    | 吸收                                    | [01-product-contract.md](./01-product-contract.md)                                                                                                   |
| Claw 是对话、运行、审批和事实链主工作区                  | 吸收                                    | [01-product-contract.md](./01-product-contract.md)、[03-architecture-and-command-contracts.md](./03-architecture-and-command-contracts.md)           |
| Right Surface 是唯一物理右栏                             | 吸收并收紧                              | [04-app-center-and-claw-surfaces.md](./04-app-center-and-claw-surfaces.md)                                                                           |
| 显式 `@plugin` 激活                                      | 吸收并改用稳定 plugin identity          | [01-product-contract.md](./01-product-contract.md)、[04-app-center-and-claw-surfaces.md](./04-app-center-and-claw-surfaces.md)                       |
| 历史恢复 selected object、tabs、产物和插件上下文         | 吸收并改为 thread/read model projection | [03-architecture-and-command-contracts.md](./03-architecture-and-command-contracts.md)、[07-verification-contract.md](./07-verification-contract.md) |
| 安装、启用、授权、运行权限分离                           | 吸收                                    | [01-product-contract.md](./01-product-contract.md)、[02-package-marketplace-installation.md](./02-package-marketplace-installation.md)               |
| WebContentsView、Host policy、surface action 回流        | 吸收                                    | [03-architecture-and-command-contracts.md](./03-architecture-and-command-contracts.md)、Browser 路线图                                               |
| 插件自带 runtime/workbench、任意 renderer contract       | 淘汰                                    | 不进入 v2；统一到 Skills/MCP/Hooks/Right Surface                                                                                                     |
| LimeCore marketplace 作为 Lime consumer runtime 的强依赖 | 拆分                                    | consumer 先支持 bundled/repo/personal/configured/remote；发布平台另行治理                                                                            |
| 内容工厂、独立应用壳、发布后台同属 Plugin 根主线         | 拆分                                    | 内容工厂是插件样例，发布平台是独立范围，不阻塞 P0 consumer                                                                                           |
| 旧 `plugin.json`、`lime.plugin.package.v1`               | 淘汰                                    | `.codex-plugin/plugin.json`                                                                                                                          |
| 旧 worker 和 renderer mock                               | 淘汰                                    | RuntimeCore、MCP、Skills 与 current read model                                                                                                       |

## 按文档提炼

### `prd.md`：保留产品问题和用户路径

保留内容：

- 安装、授权、启用、显式激活、产物查看、历史恢复和卸载的完整用户路径。
- 用户需要知道来源、状态、权限、失败原因和恢复动作。
- 插件中心不能只是卡片列表，Claw 才是调用和继续工作的主场。
- 复杂结果应进入 Right Surface，不能把用户带到第二个任务系统。

修正内容：

- 旧 PRD 把发布、注册、租户 enablement 和消费 runtime 交织在一起；v2 把它们分成 consumer protocol、local install store 和独立 publishing scope。
- 旧 PRD 允许 workspace app/runtime 作为插件贡献；v2 只允许 Skills、MCP servers/apps、Hooks、assets 和受管 UI resource。

### `architecture.md`：保留边界图，替换事实源

保留内容：

- Host、App Server、Claw、Runtime、Right Surface、artifact/evidence 的分层思路。
- Surface 不直接调用 provider、filesystem、secret 或旧 desktop facade。
- action 必须通过统一 intent 回流 runtime，并产生可追踪 evidence。
- 历史恢复优先恢复对象和 surface 状态，不重跑危险 action。

替换内容：

- 旧图将 LimeCore marketplace 和本地 registry 合并视为主要事实源；v2 改为 App Server plugin domain 统一 projection。
- 旧图允许插件 workbench shell 成为能力边界；v2 规定 Right Surface 是宿主边界，插件提供 resource/descriptor，不拥有独立业务后端。

### `interface-contracts.md`：保留行为合同，废弃类型合同

保留内容：

- activation intent 的来源区分：user、route、restore、runtime。
- surface action 的确认、权限、错误和 evidence 回流要求。
- 历史恢复中的 primary/selected object 优先级。
- 稳定错误必须面向用户可恢复，而不是暴露 raw JSON。

废弃内容：

- 旧 `PluginManifest`、`activeWorkspaceAppId`、`historyRestore` 和 renderer/workbench contribution 类型不能直接变成 v2 schema。
- v2 只从 Codex manifest、App Server protocol 和现有 Right Surface contract 重新生成类型。

### `technical-baseline.md`：保留 Host 安全底线

可直接吸收的底线：

- Electron Host 管理 view、bounds、权限、错误边界和生命周期。
- 插件不能自行携带第二套 tab/history/permission/runtime 逻辑。
- `<webview>`、BrowserView、raw Tauri command、生产 mock 不作为新主路径。
- App Center 的安装/升级/卸载动作必须经过 current readiness 和 policy。

需要改写的地方：

- 旧 baseline 把插件独立 UI runtime 当成可选承载；v2 仅保留 MCP/App UI resource 和 Right Surface host。
- `articleDraft`、`imageGenerationSet` 等业务 artifact 不属于 Plugin 通用 manifest；它们应由对应领域 projection 定义。

### `prototype.md` / `prototype.html`：保留信息架构，不复制视觉实现

保留内容：

- App Center 首页、详情、能力列表、安装状态、composer 激活 strip、Right Surface 和历史恢复的布局关系。
- 详情页需要把 Overview、Skills、能力/权限、来源和状态分层展示。
- 用户需要在当前 Claw 上下文看到插件已激活，而不是被跳转到独立应用页。

不直接复用：

- 原型中的 `Renderer` tab、独立 workbench shell 和旧命令名称。
- 旧视觉尺寸、文案、品牌图标；实现必须遵守 Lime 当前主题和五语言合同。

### `history-product-workspace.md`：提炼恢复规则

v2 采用以下有效规则：

1. 先恢复 thread/read model，再恢复 plugin identity 和能力快照。
2. 再恢复 primary/selected object、surface tab 和布局状态。
3. 恢复失败显示可理解原因，不暴露内部 JSON 或绝对路径。
4. 不恢复未完成危险 action、不复活过期权限、不自动重跑外部副作用。
5. 历史恢复不得抢占用户当前正在使用的 thread 或 Right Surface tab。

旧的“工作台应用”命名和业务产物类型不作为 v2 通用协议字段。

### `user-operations-guide.md`：提炼运营诊断

保留内容：

- 安装、启用、禁用、卸载、授权和 blocked 状态要分别排查。
- 诊断需要关联 source、version/digest、policy、auth、runtime readiness 和 evidence。
- 卸载不应误删共享 connector 凭证；历史记录应保持可读。
- GUI 验收要覆盖安装态刷新、Right Surface 产物和历史入口。

替换内容：

- “客户端上报到 LimeCore 是唯一安装态事实”的旧说法改为 App Server installed projection；外部审计只消费受控摘要。
- 旧远端运行和发布授权不进入 v2 P0。

### `implementation-plan.md`：保留顺序，冻结进度记录

可复用顺序：

```text
contract
  -> marketplace/manifest
  -> explicit activation
  -> Right Surface
  -> history restore
  -> evidence
```

需要舍弃的路线：

- 先做大量 renderer registry 和 plugin worker，再补 current runtime。
- 将内容工厂 dogfood、LimeCore 发布、租户灰度和客户端消费混成一个完成度。
- 用“骨架完成”替代真实 Electron/App Server/RuntimeCore 证据。

旧进度日志保留为历史审计，不再更新；新的进度只写入 `internal/exec-plans/` 和 v2 验收证据。

### `e2e-evidence.md` 与 JSON：保留证据结构，不能当 Gate B

可复用内容：

- 证据摘要、版本、scenario、安装态、激活态、Right Surface 和失败项应有固定字段。
- 证据需要同时记录客户端状态和服务端审计，避免只截 UI。
- 未覆盖项必须明确列出，不把 partial evidence 写成通过。

不能直接复用的原因：

- 旧证据围绕发布中心、LimeCore catalog 和 content-factory fixture，不能证明 v2 的真实 Electron/preload/App Server JSON-RPC/RuntimeCore 链。
- v2 Gate B 以 [07-verification-contract.md](./07-verification-contract.md) 为唯一验收合同，旧证据只能作为历史背景。

### `deverlop/*`：独立范围输入

发布中心文档仍有价值，尤其是签名、target 预检、审计、灰度、回滚和证据要求。但它们不应继续放大 Plugin consumer runtime 的范围。

处理方式：

- 当前保留原文作为历史范围输入。
- v2 只引用其安全和发布边界，不依赖其 API、数据库或 LimeCore endpoint。
- 未来如重新建设发布平台，应创建独立 roadmap/change，并重新确认 owner、协议和权限。

## 采纳后的不变量

以下不变量同时来自旧文档的有效经验和 v2 的 current 纠偏：

1. App Center 是发现、详情和生命周期入口，Claw 是调用和继续工作入口。
2. App Server 是 Plugin state/readiness 的事实源；Renderer 不扫描目录、不合并 registry。
3. Plugin 不拥有 agent loop、thread store、独立 worker 或第二右栏。
4. 所有高风险 action 经过统一权限、确认、事件和 evidence。
5. 历史恢复恢复状态和引用，不重放旧副作用。
6. 历史文档可以提供证据和上下文，但不覆盖 v2 protocol、current owner 和 Gate B 结果。

## 审计完成条件

- 根 README 能导航到历史参考和 v2 current。
- v2 文档能明确指出旧文档的吸收、替换和暂缓内容。
- 历史文档没有被标记为 current 实现输入。
- 新实现计划只引用 v2 文档、current owner 和新的 Gate B evidence。
- 旧文档保持可读，除非发现敏感数据、断链或明确错误需要单独修订。
