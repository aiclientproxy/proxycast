# Plugin v2 实施计划

状态：`active / P1-P5 core implemented`

更新时间：2026-08-05

## 主目标

以最短可验证路径建立 Codex-compatible Plugin consumer runtime，并在同一主链内完成 App Center、Claw `@plugin` 和 Right Surface。实施期间不维持 v1/v2 产品双轨。

## 写集纪律

每个阶段开始前，在 `internal/exec-plans/` 建立或更新执行计划，声明：

- 本阶段目标和退出条件
- 精确写集与脏热区避让
- current/deprecated/dead 变化
- 架构图确认
- 定向测试与 Gate B 证据路径
- 阻塞和未验证原因

并行开发时，protocol、App Server plugin domain、RuntimeCore、App Center、Claw/Right Surface 五个写集不得由不同变更同时重写同一事实。

## 阶段总览

| 阶段 | 主结果                        | 前置        | 完成信号                                              |
| ---- | ----------------------------- | ----------- | ----------------------------------------------------- |
| V2-0 | 冻结标准与回流守卫            | v2 文档确认 | 旧标准禁止新增，执行计划落盘                          |
| V2-1 | Manifest/marketplace/protocol | V2-0        | App Server 可列出和读取 bundled/repo/personal 插件    |
| V2-2 | 原子安装与 installed store    | V2-1        | install/update/uninstall 可恢复且无半安装态           |
| V2-3 | RuntimeCore activation        | V2-2        | Skills/MCP/Hooks 在新 thread 可追踪装配               |
| V2-4 | App Center                    | V2-1、V2-2  | 页面只消费 App Server projection                      |
| V2-5 | Claw 与 Right Surface         | V2-3        | `@plugin` 到 tool item/UI surface 闭环                |
| V2-6 | 旧实现删除与文档冻结          | V2-4、V2-5  | 旧 worker、manifest、registry、命令清零，历史文档冻结 |
| V2-7 | 跨平台与 release gate         | V2-6        | macOS/Windows 与 Gate B 全通过                        |

## 当前实施快照

| 阶段 | 状态    | 2026-08-05 事实                                                                                   |
| ---- | ------- | ------------------------------------------------------------------------------------------------- |
| V2-0 | DONE    | v2 文档、唯一 manifest 与 current owner 已冻结                                                    |
| V2-1 | DONE    | typed catalog/read/install/installed/enabled protocol 已接入                                      |
| V2-2 | PARTIAL | staging、digest、幂等与原子替换已接入；崩溃恢复/残留清理仍待补                                    |
| V2-3 | PARTIAL | Plugin Skills 与 MCP 已装配；独立 Apps/Hooks 仍待补                                               |
| V2-4 | PARTIAL | App Center 已只读 typed catalog；真实安装点击与窄窗证据仍待补                                     |
| V2-5 | PARTIAL | `plugin://`、MCP App Right Surface 与 reload 恢复已接入；picker 点击、Browser/file surface 仍待补 |
| V2-6 | PENDING | 未物理删除旧实现；删除前必须再次取得用户确认                                                      |
| V2-7 | PARTIAL | macOS core Gate B 已通过；Windows 与完整 P0/release matrix 未通过                                 |

当前完成度为 `75%`。当前主目标已从“建立 parser/protocol”转为“补齐真实用户点击、卸载历史、cold restore 与跨平台证据”。

## V2-0：冻结旧标准

### 写集

- `internal/roadmap/plugin/v2/**`
- `internal/roadmap/plugin/README.md`
- `internal/exec-plans/<plugin-v2-plan>.md`
- 既有 governance 检查的最小扩展点

### 工作项

1. 确认 `.codex-plugin/plugin.json` 是唯一 manifest。
2. 确认 App Server 是 catalog、installed、enabled、policy、auth readiness 唯一事实源。
3. 把旧 `lime.plugin.package.v1` 和 worker 命令列为 deprecated。
4. 增加“禁止新增旧字段/新调用者”的窄范围守卫。
5. 记录架构图确认，标出 current owner 与待删除 owner。

### 退出条件

- v2 系列文档通过格式和链接检查。
- 旧文档不再被根 README 作为 current 导航。
- 新代码无法继续引入旧 manifest 字段或生产 mock。

## V2-1：Manifest、Marketplace 与 Protocol

### 建议 owner

```text
lime-rs/crates/app-server/src/plugins/
  manifest.rs
  marketplace.rs
  resolver.rs
  projection.rs

lime-rs/crates/app-server-protocol/
  plugin v2 request/response types
```

最终目录以现有 crate 边界为准；不要新建与既有 owner 重叠的 crate。

### 工作项

1. 解析 `.codex-plugin/plugin.json`，校验 identity、version、relative paths 和 capability declarations。
2. 解析 repo/personal marketplace，并支持 bundled/configured/remote source descriptor。
3. 保留 source authority，禁止把远端路径当本地路径。
4. 实现 `marketplace/add|remove|upgrade` 与 `plugin/list|installed|read` current protocol。
5. 返回完整 `PluginSummary` / `PluginDetail`，包含 installed、enabled、policy、auth、availability 和版本字段。
6. 生成或共享 protocol types，Renderer 不手写镜像类型。

### 退出条件

- bundled、repo、personal 三类来源可被同一 list API 表达。
- 非法 manifest、路径越界、重复 ID、冲突版本返回稳定错误码。
- Renderer 没有参与目录扫描或 manifest 解析。
- `npm run test:contracts` 通过。

## V2-2：安装 Store 与事务

### 工作项

1. 定义 installed record、source lock、content digest、enabled state 和 schema version。
2. 下载或复制到 staging，完成大小、路径、digest、manifest 和 policy 校验后原子切换。
3. 实现 `plugin/install`、`plugin/uninstall` 和 update flow。
4. 安装与 connector auth 分离；`ON_INSTALL` 只触发授权流程，不把授权写成安装成功的必要伪状态。
5. 卸载前关闭能力装配，清理包数据，保留共享凭证和 thread 历史。
6. 崩溃恢复时回滚 staging 或完成已提交事务，不留悬空状态。

### 退出条件

- 安装中断、磁盘不足、digest 不一致和重复请求均有确定结果。
- update 失败保留上一可用版本。
- uninstall 后 package、index、activation projection 无残留。
- macOS 与 Windows 路径测试通过。

## V2-3：Skills、MCP 与 Hooks 激活

### 工作项

1. 将 installed+enabled 插件解析为 inert capability descriptors。
2. 新 thread 创建或明确 reload 时生成 activation snapshot。
3. Skills 进入现有 skill discovery；MCP servers/apps 进入现有 MCP lifecycle；Hooks 进入统一事件与权限 owner。
4. 为每个 capability 携带 plugin ID、marketplace、version、source authority 和 digest。
5. tool call、hook invocation、auth challenge 和失败事件进入 Thread/Turn/Item read model。
6. enable/disable 不篡改运行中的 turn；新状态从下一个规定边界生效。

### 退出条件

- 不显式 mention 时，Skill/MCP 仍可按描述被正常发现。
- 显式 mention 时只收窄插件上下文，不绕过权限。
- 插件 MCP 启停与普通 MCP 共用 lifecycle 和诊断。
- 运行路径不进入 plugin worker。

## V2-4：App Center 重做

### 写集

- `src/features/plugin/ui/**`
- App Server plugin gateway/client
- 五语言资源与定向测试

### 工作项

1. 页面改为 `All / Installed / source marketplace` 信息架构。
2. 增加搜索、来源筛选、详情、安装、启停、更新、卸载和 auth 入口。
3. 详情披露 Skills、Hooks、Apps、MCP servers、source、auth、privacy、terms。
4. 保持 Lime 当前主题、密度、颜色、圆角和 icon 语言，只复刻 Codex 的产品结构与状态语义。
5. 通过按钮或下拉收纳低频动作；不恢复发布后台和独立 runtime page。
6. 删除 renderer registry 合并、manifest parsing 和 mock fallback。

### 退出条件

- App Center 只调用 current plugin gateway。
- 加载、空、错误、安装中、管理员禁用、待授权、可更新状态完整。
- `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR` 文案齐全。
- 桌面与窄窗口无溢出、重叠或布局变形。

## V2-5：Claw Mention 与 Right Surface

### 工作项

1. `@` picker 接入 installed projection，并把可安装建议与可调用插件明确区分。
2. composer 写入结构化 `plugin://<plugin-id>` mention，不依赖显示名反解析。
3. turn request、RuntimeCore trace、tool item 和历史恢复贯穿同一 plugin identity。
4. MCP/App UI resource 注册到 Right Surface；Browser intent 复用 browser tab；文件和结构化结果复用既有 surface。
5. 后台结果进入 pending badge，不抢用户当前右侧 tab。
6. surface action 回流 App Server current action/turn contract，不直连 provider、文件系统或插件 worker。

### 退出条件

- `@plugin` 可从 composer 触发真实 MCP/Skill 调用。
- Right Surface 显示与当前 turn/tool item 对应的真实结果。
- 关闭、重开和恢复 thread 后 identity 与 surface state 一致。
- 插件 UI 不启动任意 iframe、本地 server 或私有 worker bridge。

## V2-6：旧路删除

### 工作项

1. 生成旧 manifest、命令、import、script 和 fixture 调用图。
2. 迁移最后调用者和测试。
3. 删除 Electron/App Server plugin worker、plugin UI runtime、renderer registry 和旧包 parser。
4. 删除旧 smoke scripts、package scripts 和 fixture。
5. 将 v1 技术标准与旧路线图冻结为历史参考，current 导航只进入 v2。
6. 将“禁止新增”守卫升级为“禁止恢复”守卫。

### 退出条件

- [05-migration-and-cleanup.md](./05-migration-and-cleanup.md) 的 dead 清单全部满足前置条件。
- `rg`、protocol catalog、package scripts 和 build graph 均无旧入口。
- `governance:legacy-report` 无 current/compat 误判。
- current Gate B fixture 在删除后仍通过。

删除属于高风险操作，执行前必须单独列出精确文件并取得明确确认。

## V2-7：跨平台与 Release Gate

### 工作项

1. 在 macOS 验证 bundled、repo、本地安装、auth、调用、UI、更新和卸载。
2. 在 Windows 验证路径、权限、进程、归档和原子替换语义。
3. 验证应用升级后的 bundled marketplace upgrade 与 user-installed 保留策略。
4. 完成恶意包、损坏包、管理员禁用和网络中断测试。
5. 运行完整 Gate B 并记录证据摘要。

### 退出条件

- [07-verification-contract.md](./07-verification-contract.md) 的 P0 场景全部通过。
- `npm run verify:local` 通过。
- manifest/workspace/Electron 版本变化时 `npm run verify:app-version` 通过。
- release evidence 可追踪 plugin/thread/turn/item/surface identity。

## 测试扩展策略

| 风险               | 最小验证                               | 扩展条件                                      |
| ------------------ | -------------------------------------- | --------------------------------------------- |
| 文档/守卫          | 格式、链接、`git diff --check`         | 影响 docs boundary 时跑 contracts             |
| Parser/protocol    | Rust 单元、schema fixture、contracts   | 影响多 crate 时跑 related tests               |
| Install store      | 单元、崩溃恢复、路径安全               | 涉及 OS 行为时双平台                          |
| Runtime activation | current fixture、MCP/Skill integration | 影响 Thread/Turn/Item 时扩大 read model tests |
| GUI                | component tests、GUI smoke             | 影响 bridge/Right Surface 时必须 Gate B       |
| 清理               | 旧引用扫描、contracts、current fixture | 删除共享 owner 时跑 verify:local              |

## 不允许的实施捷径

- 先在前端写一份 v2 marketplace 假数据等待后端补齐。
- 用 feature flag 长期保留 v1/v2 两套 production path。
- 把 Codex manifest 转换成 `lime.plugin.package.v1` 后继续走旧 worker。
- 以“兼容 Desktop”为由在 Electron main process 重建业务状态。
- 以 UI demo 代替真实 App Server/RuntimeCore 调用证据。
- 将发布中心、审核后台或商业化能力塞回 P0 consumer runtime。

## 完成度口径

阶段完成度只按退出条件计算，不按代码量或页面可见程度计算。任何阶段存在 production mock、legacy command hit、身份断链或未记录的跨 owner fallback，完成度不得超过 `90%`，也不得进入 release evidence。
