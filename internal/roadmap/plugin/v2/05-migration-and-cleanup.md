# Plugin v2 迁移与清理账本

状态：`historical-docs-frozen / implementation-migration-pending`

更新时间：2026-08-04

## 目标

本文件定义 Plugin v2 从旧包标准、旧前端 registry 和插件专用 worker 迁入 current 主链的唯一清理顺序。它是删除账本，不是兼容层设计。

迁移完成后的唯一产品链为：

```text
.codex-plugin/plugin.json / marketplace.json
  -> App Server plugin domain
  -> Skills / MCP / Hooks
  -> RuntimeCore / Tool Runtime
  -> Thread / Turn / Item projection
  -> App Center / Claw / Right Surface
```

## 清理原则

1. 先建立 current owner 和回归，再迁移调用，最后删除旧 owner。
2. 不增加 `v1 | v2` 双读、manifest 自动转换或 renderer fallback。
3. 旧代码只有在调用者清零、fixture 已迁移、回流守卫生效后才能删除。
4. 发布、审核和商业化平台不是 Plugin consumer runtime 的组成部分；需要时进入独立路线图。
5. 删除动作必须保留可审计证据，并遵循仓库危险操作确认要求。
6. 重大架构落地时，同一变更集必须更新 `internal/aiprompts/architecture.md` 和对应执行计划。

## 分类定义

| 分类                   | 允许行为                       | 禁止行为                                      |
| ---------------------- | ------------------------------ | --------------------------------------------- |
| `current`              | 扩展、修复、补测试             | 绕过 owner 建第二套实现                       |
| `deprecated`           | 迁出调用、补观测、为删除做准备 | 新增能力、新增调用者、作为 fallback           |
| `dead`                 | 删除、加禁止回流守卫           | 恢复、包装、继续维护                          |
| `historical reference` | 查阅、引用、向 v2 提炼有效决策 | 直接作为 current 实现或验收依据、独立继续演进 |
| `historical evidence`  | 查阅和复核历史结果             | 冒充当前 Gate B 证据、覆盖 current 结果       |

## Current：保留并对齐

| 能力             | Current owner                                                      | v2 动作                                                                 |
| ---------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| Plugin protocol  | `lime-rs/crates/app-server-protocol`、App Server request processor | 增加 Codex-compatible marketplace/plugin 方法与 projection              |
| Plugin domain    | `lime-rs/crates/app-server` 内聚后的 plugin owner                  | 拆分 discovery、manifest、install store、activation，避免继续堆入单文件 |
| Skills           | 现有 Skills discovery/loader owner                                 | 按 plugin identity 和 source authority 装配                             |
| MCP servers/apps | 现有 MCP owner                                                     | 复用 server lifecycle、auth、tool/resource 和 App UI 协议               |
| Hooks            | RuntimeCore / tool runtime 对应 lifecycle owner                    | 只接受 manifest 声明并进入统一权限与事件链                              |
| Agent execution  | `agent-runtime`、RuntimeCore、`tool-runtime`                       | Plugin 仅贡献能力，不拥有 agent loop                                    |
| Read model       | App Server、`thread-store` 与 projection package                   | 记录 plugin identity、tool item、surface descriptor 和恢复状态          |
| Desktop bridge   | Electron preload/IPC 到 `app_server_handle_json_lines`             | 只转发 current JSON-RPC，不承接 catalog 和业务状态                      |
| App Center       | `src/features/plugin/ui/**` 的 current 页面 owner                  | 改为消费 App Server projection，不再合并 registry                       |
| Claw             | composer、streaming、thread/read model owner                       | 增加结构化 `@plugin` mention 与调用状态                                 |
| Right Surface    | `RightSurfaceRegistry` 及 surface owner                            | 承载 MCP/App UI、Browser、结构化结果和文件预览                          |

## Deprecated：只迁出

以下区域在 v2 迁移期间可以被读取和测试，但不得新增产品能力：

| 旧事实或实现                                             | 当前问题                                                  | 迁出目标                                                  |
| -------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| 根目录 `plugin.json` 与 `lime.plugin.package.v1`         | 与 Codex 包标准分叉                                       | `.codex-plugin/plugin.json`                               |
| `contributions.runtime` / `contributions.workbench`      | 把 agent runtime 和 UI host 私有化                        | Skills、MCP servers/apps、Hooks、Right Surface descriptor |
| `app.runtime.yaml` 主入口                                | 引入插件专用 worker 生命周期                              | RuntimeCore/MCP/Skill current activation                  |
| `plugin_packages/plugin_manifest.rs` 旧 projection       | 解析、兼容、UI/runtime 推断混在一起且超过文件治理阈值     | 小职责 manifest parser 与 protocol projection             |
| `marketplaceRegistryLoader.ts`                           | Renderer 合并 marketplace、installed、manifest、readiness | App Server `plugin/list` 与 `plugin/installed`            |
| `src/features/plugin/manifest/**` 旧解析链               | Renderer 直接理解安装包标准                               | Renderer 只使用 protocol-generated types                  |
| `src/features/plugin/install/**` 中本地事实拼装          | installed/setup/cache 多源并存                            | App Server installed store 与原子安装事务                 |
| `src/features/plugin/runtime/**` 私有 runtime/projection | 与 RuntimeCore、MCP、read model 重叠                      | current runtime 与统一 item projection                    |
| `pluginUiRuntimeStart/Status/Stop` 命令族                | 第二套 UI runtime 生命周期                                | MCP/App UI resource 与 Right Surface lifecycle            |
| `src/features/plugin/publish/**`                         | 发布后台混入插件消费主线                                  | 独立发布平台路线图或删除                                  |
| `src/features/plugin/packaging/**`                       | 自定义桌面应用打包与 Plugin 包混为一体                    | 只保留 Codex-compatible 包校验所需能力                    |

`deprecated` 不等于立即整目录删除。每个子域必须先完成调用图、测试与数据所有权核对，再按最小写集迁出。

## Dead：迁移后删除

满足对应退出条件后，下列路径或语义应删除：

| Dead surface                      | 候选路径                                                                          | 删除前置条件                                                       |
| --------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Electron plugin worker            | `electron/pluginRuntimeTaskHost.ts`、`electron/pluginTaskWorker.ts`               | 所有执行已进入 App Server/RuntimeCore；IPC 调用清零                |
| App Server plugin worker          | `lime-rs/crates/app-server/src/runtime/plugin_worker_*`、`plugin_task_runtime.rs` | runtime tests 与 read model fixture 已迁移到 current tool/MCP item |
| 独立 Plugin UI host               | `src/features/plugin/runtime/uiExtensionHost.ts` 及私有 bridge                    | MCP/App UI 可在 Right Surface 完成同等用户流程                     |
| Renderer capability mock fallback | `src/features/plugin/sdk/MockCapabilityHost.ts` 等生产可达 mock                   | mock 仅存在于测试夹具且构建守卫证明生产不可达                      |
| 旧 package fixtures               | `src/features/plugin/testing/fixtures/**/plugin.json`、`app.runtime.yaml`         | fixture 已换成 `.codex-plugin/plugin.json` 与 MCP/Skill/Hook 样例  |
| 旧技术标准                        | `internal/tech/plugin/lime-plugin-package-v1.md`                                  | v2 contract 已成为唯一文档入口，入站引用清零                       |
| 旧 smoke scripts                  | `smoke:plugin-ui-runtime-*`、`smoke:plugin-runtime-*` 对应脚本                    | Gate B current plugin fixture 覆盖安装到 Right Surface 闭环        |

不要用空壳 facade 保留旧命令名。调用者迁完后直接删除命令、类型、fixture 和脚本入口。

## 冻结保留的历史文档

以下文档保留在仓库中，但冻结为 `historical reference` 或 `historical evidence`：

| 分类         | 文件                                                                                         | 保留价值                           |
| ------------ | -------------------------------------------------------------------------------------------- | ---------------------------------- |
| 历史设计参考 | `prd.md`、`architecture.md`、`interface-contracts.md`、`technical-baseline.md`               | 产品需求、分层、接口和旧实现约束   |
| 历史体验参考 | `prototype.md`、`prototype.html`、`user-operations-guide.md`、`history-product-workspace.md` | 页面结构、用户流程、恢复和运维体验 |
| 历史跟踪     | `implementation-plan.md`                                                                     | 已实施切片、问题背景和验证记录     |
| 历史证据     | `e2e-evidence.md`、`evidence/plugin-productization-e2e-summary.json`                         | 旧链路的 E2E 结果与局限            |
| 独立范围输入 | `deverlop/plugin-publish-center-prd.md`、`deverlop/plugin-publish-limecore-server-plan.md`   | 发布平台需求，供未来独立路线图提炼 |

只有 `internal/roadmap/plugin/README.md` 与 `internal/roadmap/plugin/v2/**` 可以继续演进 Plugin current 路线。历史文档若发现仍有效的决策，应在 v2 对应文档中重述并记录当前依据，而不是直接恢复其事实源地位。

## 入站引用迁移

已知需要同步的 current 入站引用：

| 来源                                        | 动作                                                                              |
| ------------------------------------------- | --------------------------------------------------------------------------------- |
| `internal/roadmap/browser/README.md`        | 将旧 PRD、architecture、contracts、baseline 链接改到 v2 产品、架构和 surface 合同 |
| `internal/roadmap/agentworkbench/README.md` | 保持指向根 `plugin/README.md`，由根索引导航 v2                                    |
| `internal/roadmap/zuanjia/README.md`        | 保持指向根 `plugin/README.md`，不复制 Plugin 规则                                 |
| `internal/tech/plugin/README.md`            | v2 实施时改为 deprecated 声明，最终删除旧标准入口                                 |

历史执行日志中出现旧路径不需要机械重写；如果文档仍被 `docs:boundary` 当作 current 输入，则应由边界规则显式排除历史记录，而不是伪造过去的证据。

## 数据与状态迁移

本仓库没有外部用户和兼容负担，因此不做长期格式迁移器。开发态本地数据按以下策略处理：

1. v2 installed store 使用新的 schema/version 和独立目录。
2. 首次启动发现旧 installed/setup/cache 时，只显示“一次性清理旧插件数据”诊断，不自动转换为已安装 v2 插件。
3. bundled 插件由当前 marketplace 重新解析与安装，不继承旧 readiness。
4. connector 凭证由既有 auth owner 管理；删除插件只解除引用，不擅自删除共享凭证。
5. 删除旧缓存前列出 namespace、预计体积和恢复边界，并走用户确认。
6. thread 历史保留旧 tool/item 文本投影，但不恢复已删除 worker；恢复时明确显示历史能力不可重跑。

## 回流守卫

V2-0 和 V2-6 至少建立以下静态守卫：

```text
forbidden manifest: schemaVersion = lime.plugin.package.v1
forbidden fields: contributions.runtime, contributions.workbench
forbidden entry: app.runtime.yaml
forbidden production imports: MockCapabilityHost, mockCapabilityProfile
forbidden commands: pluginUiRuntimeStart, pluginUiRuntimeStatus, pluginUiRuntimeStop
forbidden renderer behavior: fs/path scan of plugin package or marketplace
```

守卫应扫描生产代码、package scripts、协议 catalog 和 current 文档；测试 fixture 例外必须精确到文件，不能按目录放行。

## 单切替换顺序

```text
冻结旧标准并加“禁止新增”守卫
  -> 建立 v2 parser/protocol/store
  -> 迁移 Skills/MCP/Hooks activation
  -> 迁移 App Center 与 @plugin
  -> 迁移 Right Surface 与历史投影
  -> 迁移 Gate B fixture
  -> 清零旧调用和旧命令
  -> 删除 worker、旧 manifest、旧脚本并冻结历史文档
  -> 加“禁止恢复”守卫
```

任一步出现主链缺口都应修复 current owner，不允许临时回落到旧 worker 或 renderer registry。

## 删除完成条件

- `rg` 无生产代码引用旧 manifest、旧 worker 命令和旧 registry loader。
- package scripts 不再暴露旧 runtime smoke。
- protocol/client/catalog/fixture 中不存在旧命令。
- App Center 与 Claw 只消费 App Server v2 projection。
- Gate B current fixture 覆盖安装、调用、Right Surface、恢复、卸载。
- `governance:legacy-report` 将旧能力标记为 `dead/deleted/forbidden-to-restore`。
- 根 Plugin README 以 v2 为 current 导航，并将旧文档明确分区为历史参考。
