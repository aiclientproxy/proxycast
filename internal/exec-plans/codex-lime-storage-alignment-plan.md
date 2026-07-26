# Codex / Lime 存储对齐执行计划

状态：`in_progress`

日期：2026-07-19

范围：`internal/refactor/v1` 的 Agent runtime、App Server、Thread/Turn/Item、Electron 数据根、用户指令、Skills/Plugins/MCP、日志和可重建缓存。

负责人：待责任开发者确认

相关事实源：

- [存储对齐事实与方案](../refactor/data/01-storage-alignment-plan.md)
- [Codex `~/.codex` 与 Lime 语义分组矩阵](../refactor/data/02-codex-lime-storage-matrix.md)
- [Codex `~/.codex` 与 Lime 一一对照账本](../refactor/data/03-one-to-one-storage-alignment-plan.md)
- [Lime 全局架构](../aiprompts/architecture.md)
- [Lime 治理规则](../aiprompts/governance.md)

## 1. 目标和边界

本计划只对照 Codex 的 `~/.codex`。不对照 VS Code、VS Code 插件或其他 Electron profile，也不把 Codex 的产品专属目录强行复制到 Lime。

目标是让 Lime 的存储具有一个可解释、可按 session/date 清理的事实源：

> canonical rollout 文件保存 Thread/Turn/Item 的 durable history；小型 SQLite 保存 metadata、索引、目标状态和可重建 read model；日志、trace、telemetry、sidecar、cache 和 host profile 均不得成为第二份 transcript。

迁移白名单固定为：**模型控制面语义状态 + 已下载模型文件**。模型控制面只允许迁移 `api_key_providers`、`api_keys`（仅加密 ciphertext）、`provider_ui_state`、`user_model_preferences` 和 `settings.providers.active_tab`；旧 Lime session、event、projection、Product DB 其他表、sidecar、cache、日志、配置和 retired runtime 数据均不做 import/merge/backfill。模型控制面 source 只读、按 SQLite 事务读取 WAL，不复制整库；已下载模型仍按 staging/checksum/atomic publish 迁移。current 主链验证后其余旧路径按精确路径清理。本计划不直接授权删除真实用户目录，物理清理仍必须在 maintenance 阶段获得明确确认。

Agent runtime/storage 以 `/Users/coso/Documents/dev/rust/codex` 为准；多模型 catalog、switch、capability、readiness、retry/circuit breaker 参考 `/Users/coso/Documents/dev/rust/grok-build`，Provider/Model 分层、credential 隔离、capability/lowering 和有限 transient retry 参考 `/Users/coso/Documents/dev/js/opencode`。参考实现只提供语义，不复制其目录或数据库；多模型最终 owner 仍是 `App Server + model-provider`，不改变 `App Server -> ThreadStore` 的存储边界。

不做范围：

- 不实现 Codex 未有的 Lime 独有产品能力的目录 parity；`models/`、`projects/`、Chromium cache、Agent Apps 和 OAuth 仍由各自产品 owner 管理。
- 不把 `~/.lime` 变成数据库、session 或日志根；它只承载用户主动管理的 instruction、skills 和少量配置。
- 不保留新的平行 runtime、旧 projection 镜像或“先双写再说”的长期 compat 路径。

## 2. 目标路径契约

### 2.1 平台路径

| 语义                               | macOS                                          | Windows                                                                           | 允许内容                                                                       |
| ---------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `InstallRoot`                      | `.app` bundle                                  | `%LOCALAPPDATA%\\lime`                                                            | 程序、updater 和 Squirrel 产物；禁止业务数据写入                               |
| `AppDataRoot`，唯一机器级 Agent 根 | `~/Library/Application Support/lime`           | `%LOCALAPPDATA%\\LimeCloud\\lime`                                                 | App Server、SQLite、rollout、sidecar、模型、项目索引和机器级 cache             |
| `AgentRoot`                        | `<AppDataRoot>/app-server`                     | `<AppDataRoot>\\app-server`                                                       | 唯一 ThreadStore/runtime owner                                                 |
| `HostUserData`                     | 与 `AppDataRoot` 物理同根、按 owner 分区       | `%APPDATA%\\lime`                                                                 | 小型 Electron profile/config 和窗口状态；Windows 不放 Agent DB、模型或 runtime |
| `HostSessionData`                  | 与现有 host profile 一起治理                   | `<AppDataRoot>\\host-session`                                                     | Chromium cookies/storage/network state/cache；不能整体标为可重建               |
| `UserHome`                         | `~/.lime`                                      | `%USERPROFILE%\\.lime`                                                            | `AGENTS.md`、用户 skills、少量用户配置；可备份、可迁移、可手工编辑             |
| `Workspace`                        | `<cwd>/.lime`                                  | `<cwd>\\.lime`                                                                    | 工作区 instruction 和项目级 metadata                                           |
| OS cache/log（可选）               | `~/Library/Caches/lime`、`~/Library/Logs/lime` | `%LOCALAPPDATA%\\LimeCloud\\lime\\Cache`、`%LOCALAPPDATA%\\LimeCloud\\lime\\Logs` | 仅可重建 cache 和有界诊断日志                                                  |

启动契约必须显式解析并记录：

```text
InstallRoot  = macOS .app bundle
               Windows %LOCALAPPDATA%\\lime
HostUserData = Electron profile/config root
HostSessionData = Electron Session profile/storage/cache root
AppDataRoot  = macOS ~/Library/Application Support/lime
               Windows %LOCALAPPDATA%\\LimeCloud\\lime
AgentRoot    = AppDataRoot/app-server
UserHome     = $HOME/.lime
Workspace     = cwd/.lime
```

Windows 固定把 Electron profile 留在 `%APPDATA%\\lime`，Host 显式把 `%LOCALAPPDATA%\\LimeCloud\\lime\\app-server` 作为 App Server `--data-dir`。`sessionData` 必须单独验证，避免 Chromium profile/cache 全部进入 roaming。便携/E2E 模式可显式替换受控根，但不得使用 Squirrel 安装根 `%LOCALAPPDATA%\\lime`。

`LIME_AGENT_RUNTIME_ROOT` 只允许测试、便携模式或明确的运维迁移使用；存在该变量时必须替换 `AgentRoot`，不得和默认 root 双写。

### 2.2 AgentRoot 目标拓扑

```text
<AgentRoot>/
  sessions/YYYY/MM/DD/rollout-*.jsonl
  archived_sessions/rollout-*.jsonl
  sqlite/state.sqlite
  sqlite/thread_history.sqlite
  sqlite/goals.sqlite
  sqlite/memories.sqlite
  observability/log/lime.log
  observability/events/
  observability/traces/
  observability/telemetry.sqlite
  artifacts/sessions/<session-id>/
  memories/
  cache/
```

Lime 的 `projects/`、`models/`、插件包和 MCP OAuth 可留在 `AppDataRoot` 的产品旁路，但必须有独立 owner、retention 和清理单位，不得伪装成 ThreadStore。

### 2.3 分类规则

| 分类          | 本计划处理方式                                                   |
| ------------- | ---------------------------------------------------------------- |
| `current`     | 唯一可继续演进的 owner；新写入只能进入这里                       |
| `compat`      | 仅允许模型控制面/模型文件迁移 adapter；source 只读、按白名单校验 |
| `deprecated`  | 立即冻结新写入，只允许盘点和清理                                 |
| `dead`        | 无生产入口或已替代；负向扫描通过后删除并加回流守卫               |
| `rebuildable` | 可由 canonical 或 metadata 重建，不是用户 durable truth          |

目标分类：`AgentRoot` 下 canonical rollout、ThreadStore metadata、goal state 和可重建 history 为 `current`；模型控制面/模型文件迁移器是唯一 `compat`；根级 `runtime/**`、旧 `sessions/`、旧 `session_repository` 和旧 projection 为 `deprecated -> dead`；空 `app.db`、无 consumer 的旧 projection、`aster/` 等为 `dead`。

## 3. 逐项对齐执行清单

下表按语义引用 [02 矩阵](../refactor/data/02-codex-lime-storage-matrix.md)，逐路径完成状态以 [03 一一账本](../refactor/data/03-one-to-one-storage-alignment-plan.md) 为准；未列为 `current` 的目录不得通过“有相似路径”标记完成。

|   # | 对齐动作                                                                              | 阶段  | 完成断言                                                    |
| --: | ------------------------------------------------------------------------------------- | ----- | ----------------------------------------------------------- |
|   1 | 新 Thread 直接写 `sessions/YYYY/MM/DD/rollout-*.jsonl`；旧 event/session JSONL 不迁移 | P0/P1 | state DB 能唯一定位 rollout，cold/live/replay identity 一致 |
|   2 | 增加 `archived_sessions/`，archive 只做原子移动和 metadata patch                      | P1    | archive/unarchive 不复制 transcript                         |
|   3 | 从混合 `lime.db` 收口 Thread metadata、path、archive、fork、watermark 的 state owner  | P0/P1 | metadata 可由 rollout backfill，含 canonical path           |
|   4 | 将逐 event 的 `projected_items` 改为 coalesced、可丢弃的 history read model           | P1    | 删除 projection 后可从 rollout 全量 rebuild                 |
|   5 | 明确 `session_index` 仅为 index/fallback，不能成为第二 history owner                  | P0/P1 | title/index round-trip 不保存 transcript                    |
|   6 | 对 Lime 是否需要 `history.jsonl` 做 scope decision；若需要则独立限额                  | P0    | 不与 Thread rollout 或 request log 混用                     |
|   7 | 为 goals 建立独立 state owner，禁止 projection GC 误删                                | P1    | goal API 删除与 history 删除隔离                            |
|   8 | 将 memory pipeline workset 与用户 memory artifact 分离                                | P1/P2 | pipeline DB 可重建，用户文件独立保留                        |
|   9 | 根 `memory/`、`harness/memories/` 不迁移，`app-server/memories/` 只承接新数据         | P2    | 旧 owner 无读写入口                                         |
|  10 | 选定一个 typed config owner，旧 YAML/JSON 不导入                                      | P0/P2 | fresh config round-trip 通过且不双写                        |
|  11 | 将全局 instruction 固定为 `UserHome/AGENTS.md`                                        | P0/P2 | AppData 清理不删除 instruction                              |
|  12 | 区分 user home 与 workspace `.lime` instruction precedence                            | P0/P1 | user/workspace discovery 有跨平台测试                       |
|  13 | 将 auth/credential 从 rollout、日志和普通 DB payload 中隔离                           | P1    | revoke/rotation、support bundle 脱敏通过                    |
|  14 | permission/rules 归 `tool-runtime` owner，不进入 ThreadStore                          | P1    | policy 变化不触发 history migration                         |
|  15 | prompt template/skills package 归 skills owner，不写入 history DB                     | P1/P2 | package 级清理可执行                                        |
|  16 | 删除非模型 migration marker；模型 manifest 不承载业务 payload                         | P0/P2 | 仅模型迁移可生成 marker                                     |
|  17 | version/update/install 元数据归 Electron host owner                                   | P0    | 不进入 Agent history                                        |
|  18 | 分层收口 user/system/workspace skills，每层一个 owner                                 | P1/P2 | package 目录可独立清理                                      |
|  19 | 收口 plugin installed package、manifest 和 runtime cache                              | P1/P2 | package/version 清理不扫 Thread DB                          |
|  20 | 合并 plugin staging/cache 为单一临时根并设 TTL                                        | P3    | 失败任务可 dry-run GC                                       |
|  21 | 将 MCP catalog cache 与 snapshot metadata 分离                                        | P1/P3 | server/generation/TTL 清理可验证                            |
|  22 | 将 OAuth lock 与 credential 分离，stale lock 可回收                                   | P1/P3 | lock 不阻塞 thread delete                                   |
|  23 | vendor/package import 只属于产品安装 staging，不是旧 AppData 迁移                     | P2    | staging 完成后按 TTL 清理                                   |
|  24 | 明确 connectors/computer-use owner，不为 parity 伪造 ThreadStore 目录                 | P0/P1 | connector/package 级清理有 owner                            |
|  25 | 将日志、trace、telemetry 收口到 bounded observability owner                           | P1/P3 | 按 thread/date/TTL/bytes 清理，不影响 rollout               |
|  26 | 根 `logs/`、`request_logs/` 只保留一个诊断 text owner                                 | P0/P1 | 默认有界，不能作为 history 读取源                           |
|  27 | 如需 shell snapshot，单独放在 `app-server/shell_snapshots`                            | P1/P3 | 不写入 canonical item，按 session/TTL 清理                  |
|  28 | process manager 状态归 `tool-runtime`，生命周期结束即回收                             | P1/P3 | 重启可重建，不进入 history DB                               |
|  29 | generated image/tool output 只存 artifact ref，blob 按 session/project 目录管理       | P1/P3 | DB 无大 blob，删除能定位目录                                |
|  30 | 分离 host cache、runtime cache、download staging，统一目录级 GC                       | P3    | cache 可重建且有 size/TTL 预算                              |
|  31 | 仅模型迁移 backup/staging 带 manifest/TTL，校验后回收                                 | P2/P3 | staging 不永久留在 current root                             |
|  32 | 旧 `CODEX_SQLITE_HOME` 类 root、根 DB 和空 projection 不迁移，直接进入清理清单        | P2    | 无生产 reader/writer 且有回流 guard                         |
|  33 | Codex feature-specific 目录无 Lime owner 时明确排除                                   | P0    | 不新增无语义 parity 目录                                    |

## 4. 分阶段实施

### P0：路径契约、盘点和旧写入冻结

写集：`lime-rs/crates/core/src/app_paths.rs`、`lime-rs/crates/core/src/database_path_migration.rs`、`lime-rs/crates/core/src/migration_manifest.rs`、`lime-rs/crates/app-server/src/runtime/storage_roots.rs`、`LocalAppDataSource` 的领域路径注入、`lime-services` 的显式资源路径 consumer、Electron App Server launch、storage report/guard 测试和本计划关联文档。

动作：

1. 先建立数据安全基线：Electron、standalone、测试和遗漏参数入口默认全部为 `retain`；启动流程不得触发 `clear-rows/drop-tables/delete-file`、`VACUUM`、source checkpoint 或 target 覆盖，破坏性动作移到独立 maintenance 命令。
2. 建立单一不可变 roots contract；resolver 只解析路径，不创建目录、不复制数据、不写 marker。所有生产 writer 通过构造注入获得 `HostUserData`、`HostSessionData`、`AppDataRoot`、`AgentRoot`、`UserHome` 和 `Workspace` 中所需的最小 root。
3. 增加只读 `storage report`：path、owner、治理状态、持久性、对齐决策、证据等级、文件数、字节数、最大单文件、SQLite 主库/WAL/SHM、活跃 writer、page/freelist、canonical/projected 行数；不得输出 prompt、credential 或原始 payload。
4. 将旧 `runtime/**`、根 `sessions/`、旧 `session_repository` 标成 `deprecated -> dead` cleanup candidate；不得进入启动读取或迁移。Windows Squirrel `InstallRoot` 只能由模型迁移器 exact allowlist 只读扫描。新增负向扫描，阻止第二 runtime root、新的 `projection_1.sqlite`、durable temp fallback 和 UserHome 非 allowlist writer。
5. 为 canonical、observability、projection、sidecar、cache、download 分别登记 retention/size 预算；canonical rollout 默认不自动删除。

退出条件：同一启动配置只有一个 `AgentRoot`；所有入口默认 `retain`；source inventory 前后 fingerprint 不变；macOS/Windows/E2E 路径契约测试通过；旧 root 无新 writer；HostUserData 无机器资产；治理扫描通过。

### P1：Codex-style canonical ThreadStore 和 SQLite 边界

写集：`lime-rs/crates/thread-store/**`、App Server ThreadStore/ProjectionStore/EventLog/repair、协议 schema/fixture、相关 Renderer read model。

动作：

1. canonical append 写入 `sessions/YYYY/MM/DD/rollout-*.jsonl`，记录 source、cwd、provider/model、fork lineage 和 provenance。
2. `ThreadStore::append_items` 只追加 canonical item；metadata patch 独立事务；archive 使用原子 move。
3. `thread_history.sqlite` 只保存 coalesced Thread/Turn/Item snapshot、watermark 和 index；不得把每个 delta/lifecycle event 永久化为 history item。
4. EventLog 只作为有界诊断/repair 输入；GUI、`thread/read`、replay 和 export 只消费 canonical/read model。
5. 统一 Item identity、sequence、tail repair、unknown line、compaction/rollback/fork lineage，并覆盖 cold/live/replay/restart。

退出条件：projection 删除后可从 rollout 重建；一个 session 的删除只需一个 canonical 文件、一个 artifact root 和一组 metadata rows；不存在第二 transcript owner。

### P2：仅模型迁移和旧 owner 收口

模型控制面迁移顺序固定为：`exact-source inventory -> model-control signal selection -> read-only SQLite transaction -> whitelist copy -> marker`。source 候选只来自同一 Lime 产品的旧 `lime.db`、`app.db`、`proxycast.db` 路径；按 API Key/custom provider/model preference 信号选择唯一 source，空 system catalog 不得抢先写 marker。目标是 `AgentRoot/lime.db`，不复制 Product DB 其他表、不读取明文 credential、不清理 source。

已下载模型迁移顺序固定为：`dry-run -> model manifest -> staging copy -> checksum/required-file verify -> atomic publish -> source cleanup pending`。目标是 `<AppDataRoot>/models/<domain>/<model-id>`；manifest 只记录 model id/version、相对文件、bytes、checksum 和状态，不记录 session/credential。

其他旧数据统一执行：`dry-run -> exact-path cleanup manifest -> current owner/no-reader verify -> 明确确认 -> cleanup`。禁止创建 session/event/DB/config/memory importer，禁止复制旧 projection/Product DB，禁止从旧 root backfill current Thread。

退出条件：模型迁移具有 macOS/Windows 中断恢复和 checksum 证据；旧 session/DB/event/runtime 路径无生产 reader/writer；每个旧路径只有 `cleanup_pending / cleaned / retained_by_explicit_owner` 结果；回流守卫通过。

### P2.1：多模型控制面（Grok + OpenCode reference）

这不是旧数据迁移。参考 `grok-build` 的 `ModelsManager`、`models_cache.json`、model switch 和 `xai-circuit-breaker`，以及 OpenCode 的 ProviderRecord/ModelInfo/Credential/lowering/retry 分层，在 Lime current owner 中收口：

1. App Server 拥有 catalog generation、current/default selection、selectable predicate 和 refresh single-flight；Renderer/Electron 只投影。
2. Provider catalog cache 从 Product DB `settings/provider_models_fetch_cache:*` 迁出为 `<AgentRoot>/cache/models/<scope-hash>.json`，包含 schema/app version、auth kind、provider/tenant、origin/protocol、ETag、fetched/expires 和 capability taxonomy version。旧 DB cache 不迁移，直接停止 reader/writer 后清理。
3. capability snapshot 同时供 `model/list`、picker、route resolver、provider lowering 和 runtime preflight 消费；未知能力、credential/readiness 未满足、allowlist 零匹配时 fail closed。
4. model switch 使用独立 current Thread 方法，校验 selectable、active-turn policy 和 agent/harness compatibility 后原子 patch Thread default model；`agentSession/update` 不再承担 model lifecycle。
5. OpenCode 的 Provider 与 Model 两层投影合并为 Lime 的统一 `Provider/Model/Capability/Readiness/Policy` snapshot；credentials 继续只由 App Server SQLite `api_keys` 管理，Renderer 不保存 catalog 真相。
6. Provider request/API 与 Model lowering/capability 分离；OpenCode 的 transient retry 只允许在未产生用户可见 item 时有限执行，retry/circuit breaker 运行态只在内存和有界 telemetry，不进入 ThreadStore。
7. 手动添加继续使用现有 `modelProvider/create/update`、`modelProviderKey/*`、`modelProvider/fetchModels` 边界；不从 Content Studio、Kiro 或其他产品目录自动导入。

退出条件：删除 catalog cache 可从远端/bundled source 重建；Product DB 不再保存 catalog JSON；catalog refresh、model switch、route request 使用同一 generation；无重复 Turn/Item；macOS/Windows 路径和原子发布证据通过。逐项映射见 [03 一一账本 GM1-GM10](../refactor/data/03-one-to-one-storage-alignment-plan.md)。

### P3：按 session/date 的清理和 GC

提供统一的 `storage report` 和 `gc --dry-run`，清理主键固定为：

```text
thread_id/session_id
  -> canonical rollout path
  -> owned artifact directory
  -> metadata/index rows
  -> derived projection/telemetry/cache
```

清理顺序：

1. session delete：确认没有 active writer，flush/close rollout，删除 canonical 文件和同 session artifact，最后在一个事务内删 metadata/index；任一步失败不得返回成功。
2. archive/unarchive：只移动 active/archive 文件并 patch path/status，不复制 transcript。
3. date GC：日期只作 selector；先输出日期 manifest（Thread、相对 rollout path、文件数、字节、active/archive/pin/writer 状态、checksum、artifact refs），确认后逐 Thread 重新校验并移动到同盘 quarantine，禁止以 `rm -rf YYYY/MM/DD` 为主算法。
4. derived GC：projection、telemetry、trace、sidecar、MCP/plugin/cache、失败下载按 TTL/orphan/size 清理；canonical rollout、archived rollout、用户 projects 默认不自动删除。
5. SQLite 维护：只在停写、备份和重建校验后 checkpoint/VACUUM；记录 freelist、WAL、projection/canonical ratio 和 orphan count。

退出条件：单 session 和单日期清理均可 dry-run、可审计、可回滚；删除一个 projection 文件不影响 canonical read；goal、memory、auth 不被 history GC 误删。

### P4：验证、灰度和收尾

先定向验证，再 Gate A，再 Gate B；任何失败必须回写本计划的进度和 blocker，不得用旧 mock 或旧 runtime 目录替代证据。

## 5. 验证门禁

### 5.1 数据和治理门禁

```bash
git diff --check
npm run governance:legacy-report
npm run test:contracts
npm run test:rust:related -- lime-rs/crates/app-server lime-rs/crates/thread-store lime-rs/crates/agent-protocol
```

必须有定向回归：

- platform path resolution、single-root/no-dual-write；
- canonical append、metadata patch、identity/sequence/FK 原子边界；
- JSONL tail repair、unknown line、重复 sequence、projection lag 和全量 rebuild；
- archive/unarchive/delete 的 rollout、metadata、history、sidecar 一致性；
- restart/resume/fork/compaction 不创建第二 root，不丢 provider/model/route；
- storage report 和 manifest 不泄露 prompt、token、credential 或原始 payload；
- old root、old DB、`projected_items` 高频写入和 dead path 的负向 guard。

### 5.2 GUI / Electron Gate A

```bash
npm run verify:gui-smoke
npm run smoke:agent-runtime-current-fixture
npm run bridge:health -- --timeout-ms 120000
```

证明 renderer 的 `thread/list`、`thread/read`、分页 history、archive/delete 和 reload 只走 current App Server/ThreadStore read model；不从 `runtime/events`、mock projection 或文本拼装第二份 history。

### 5.3 真实 Electron Gate B

必须证明完整链路：

```text
Electron Desktop Host
  -> preload/IPC
  -> app_server_handle_json_lines
  -> App Server JSON-RPC
  -> RuntimeCore
  -> ThreadStore / canonical rollout
  -> read model / GUI
```

至少覆盖：新建 session、跨日 rollout、restart/resume、archive/unarchive、单 session 删除 dry-run、按日期 GC dry-run、projection rebuild，以及真实 GUI reload 后 history identity 不变。Gate B 不接受 mock backend、旧 root 或直接读取本地 SQLite 的测试捷径。

### 5.4 跨平台门禁

macOS 必须验证 `Application Support/lime` 与 `~/.lime` 的职责分离。Windows runner 必须验证 `%LOCALAPPDATA%\\LimeCloud\\lime`、Squirrel 安装包根 `%LOCALAPPDATA%\\lime` 不被业务写入、`%APPDATA%\\lime` profile、`%USERPROFILE%\\.lime`、`sessionData` 和路径锁/WAL/原子 rename；不能用 macOS 结果伪造 Windows 证据。

## 6. 回滚和数据安全

- P0/P1 只允许新增 schema、current writer、report 和 guard，并冻结旁路 writer；切换前保留旧 source 只读，不做删除。
- P2 每次迁移必须有版本化 manifest、同盘 staging/备份目录、输入 checksum、输出 checksum、计数对账和可校验状态；source 只读打开，导入失败不覆盖 target/canonical 文件。
- P3 的 `gc --dry-run` 是默认入口；真实删除、旧目录删除、数据库清理和 VACUUM 都属于危险操作，必须单独确认。
- 不允许 `DELETE FROM projected_items`、无条件递归删除日期目录或同时写两个 root 作为“修复”。

## 7. 进度和架构确认

| 阶段                                 | 状态          | 责任人 | 证据/阻塞                                                                                                                                                                                                                                                                                                                    |
| ------------------------------------ | ------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0 路径契约与冻结                    | `in_progress` | 待定   | Electron/App Server 双根注入、Windows HostSessionData 路径、叶子 root bypass 清零、旧 Product DB 整库迁移 owner 与 `migration_v6` 非模型 project/path 迁移删除已完成；Windows runner、cleanup manifest 和架构确认仍阻塞                                                                                                                    |
| P1 canonical ThreadStore/SQLite 边界 | `in_progress` | 待定   | dated rollout、archive move、v2 API、跨日/restart/rebuild 与三库物理 owner 已实现；macOS Gate B 已通过。`last_sequence` history owner、deprecated projection reader 退役、旧 NULL row exact cleanup guard 与 Windows 证据仍阻塞                                                                                              |
| P2 仅模型迁移与旧 owner 清理         | `in_progress` | 待定   | Content Studio 跨产品导入已完整撤回：清理基线为 93 个系统 Provider、0 个自定义 Provider、0 条 API Key，错误 marker 已删除；后续用户手动添加不计入迁移。Kiro 不进入 importer；Content Studio/sibling product 路径必须拒绝。已下载模型 manifest/staging/checksum/atomic publish 与 Windows 证据仍待完成，其他旧数据只做 exact-path cleanup manifest |
| P3 session/date GC                   | `pending`     | 待定   | 待 dry-run、保留策略和 artifact owner                                                                                                                                                                                                                                                                                        |
| P4 Gate A/B 收尾                     | `pending`     | 待定   | 历史 Product DB migration Gate B 仅作 superseded safety evidence；session files root、E2E/custom root 隔离及 canonical archive/unarchive/restart/read-model Gate B 已通过。projection rebuild、session/date GC dry-run Gate B 与 Windows evidence 仍未完成                                                                   |

### 2026-07-19 P0 实现记录

- Electron 新增统一 `appDataPaths` resolver：macOS 保持 `Application Support/lime`；Windows 使用 `%LOCALAPPDATA%\\LimeCloud\\lime`，避开 Squirrel 安装包根 `%LOCALAPPDATA%\\lime`。`ELECTRON_E2E_USER_DATA_DIR` 在 E2E 模式下优先级高于 ambient `LIME_AGENT_RUNTIME_ROOT`，缺失时 fail closed。
- Electron 在构造 Host consumer 前固定 lowercase `HostUserData`，App Server `--data-dir` 与子进程 `LIME_AGENT_RUNTIME_ROOT` 使用同一个最终 `AgentRoot`；Voice model 和 browser connector 已改为显式 `AppDataRoot` writer。
- Rust `preferred_agent_root()` 现在只负责解析 effective AgentRoot；新增不读取 `LIME_AGENT_RUNTIME_ROOT` 的 `platform_default_agent_root()`，custom/portable/E2E 根不会借 override 逃逸到真实 AppData。旧 local/roaming `lime/app-server` 和 Windows Squirrel 根不得作为非模型迁移源；只允许进入精确 cleanup inventory，模型迁移另走 models allowlist。
- 正常启动链已移除 Product DB destructive cleanup：Electron、App Server client、App Server CLI 和 smoke env 不再接受或生成 `clear-rows/drop-tables/delete-file`；旧 flag 会被 App Server 作为 unknown argument 拒绝，迁移 source 固定保留。clear/drop/delete 实现只保留为 core 内部 test-only evidence，真实 maintenance 必须等待版本化 manifest。
- 正常数据库启动不再执行凭证池破坏性清理：`init_database_at_path()` 仍运行 current schema migration，但不再删除 legacy credential。旧表和目录分类为 `deprecated/cleanup source`，不迁移，只能经独立 cleanup manifest、停写验证和明确确认处理。
- **Superseded evidence**：旧 Product DB 的 immutable backup、fingerprint、staging/no-clobber cutover 与 `storage-migration.v1 / database-path-v1` 曾验证非破坏迁移安全；用户已明确非模型数据不迁移，这套状态机不再是 active owner，后续应删除生产入口与回流引用。可复用的 checksum/staging 原则只进入模型迁移器。
- 新增 `storageRootBoundary` 负向守卫：Rust root bypass 基线已从 19 个文件缩到 4 个，HostUserData 只允许 host config/profile，Electron machine asset 只能进入已登记的 `models`/`connectors` owner；Plugin package/runtime 只能使用 `PluginDataSource` 注入的 `<AgentRoot>/plugins`，正常启动链不得恢复 Product DB/credential destructive cleanup 或 `.migration_completed`。任何新旁路 writer、平行顶层目录、启动期凭证删除或时间戳 marker 回流都会失败。
- `sessionFile/*` current 链已收敛到 `AgentRoot/artifacts/sessions/<session-id>`；旧 session 文件不迁移，只进入 cleanup inventory。本轮没有读取、复制或删除旧用户数据。
- `LogStore` current writer 已收敛到 `AgentRoot/observability/log/lime.log`；旧 log 不迁移，只进入 cleanup inventory。Codex-style structured log SQLite 的 thread/process 分区与 bytes/TTL 预算仍归 O1。
- Voice model 全链已收敛：Electron `VoiceModelHost` 是 `AppDataRoot/models/voice/<model-id>` 的唯一目录 writer；Renderer `voiceModels.ts` 在 `voiceModel/testTranscribeFile` 前读取 install state，并通过必填协议字段 `install_dir` 交给 App Server；App Server 与 `lime-services` 共用显式绝对路径校验并校验 required files。旧 default credential 回读与 `best_effort_data_dir()` 平台 fallback 已删除，缺字段、空值或相对路径均 fail closed。
- Connect registry current 路径已收敛到 `AgentRoot/cache/connect/registry.json`；旧 `AppDataRoot/connect` 不迁移、不双读，只进入 TTL/bytes cleanup inventory。
- `agent/src/durable_memory_fs.rs` 与 `agent/src/tool_io_offload.rs` 经全仓 consumer scan 仅剩 crate 导出和模块自测，没有产品调用者，分类从“active writer”纠正为 `dead candidate`。生产构建中的 `AppDataRoot/harness/{memories,tool-io}` 默认写入已删除；只有显式 `LIME_DURABLE_MEMORY_DIR` / `LIME_TOOL_IO_OFFLOAD_DIR` 配置才允许运行，否则 fail closed。测试继续使用隔离 temp fixture。它们不得为了目录 parity 迁入 `AgentRoot`；物理删除仍需明确确认，退出时同步移除 crate 导出和 guard 例外。
- `core::PluginLoader` 没有产品 `load/load_all` 调用，真实 Plugin 安装与运行由 App Server `PluginDataSource` current 主链承接。旧 loader 的 `best_effort_runtime_subdir("plugins")` 已删除；默认 `PluginManager` 固定 disabled 且不持有路径，只有 `PluginManager::new(explicit_path, config)` 可启用。分类为 `dead / frozen`，不得恢复自动平台发现。
- `infra::RequestLogger` 没有产品构造者，current request telemetry 由 `TelemetryStore` 承接。旧 `resolve_request_logs_dir()` 默认 writer、`Default` 和 `with_defaults` 已删除；构造必须传入非空 `log_dir`，空路径 fail closed。raw request JSONL 分类为 `dead candidate / frozen`，不得作为 Thread history 或 TelemetryStore 的第二事实源。
- runtime AGENTS 全局层已从 `best_effort_user_memory_path()` 改为严格 `resolve_user_memory_path()`；平台根解析失败时仅记录告警并跳过全局层，不再回退 temp。Workspace `.lime/AGENTS.md` 继续独立发现和加载，分类为 `current / fail closed`。
- Plugin package、installed state、seeded runtime cache 和 worker runtime 已统一由 `LocalAppDataSource` 注入 `<AgentRoot>/plugins`；`local_folder` 只消费 state 中的显式 `sourceUri`，`cloud_release` 缺少 PluginDataSource root 时 fail closed。`preferred_data_dir()` 平台猜根和 RuntimeCore 第二套 package root 已移除，分类为 `current / injected owner`。
- 本记录不代表 state/read-model 拆分、HostSessionData、已下载模型迁移或清理已完成；非模型 migration state machine 已从 active scope 移除。P1/P2 为 `in_progress`，P3 为 `pending`，未执行删除、清理、checkpoint 或 VACUUM。
- **2026-07-19 启动与模型源诊断**：Electron 日志中的 `spawn .../lime-rs/target/debug/app-server ENOENT` 发生在 App Server 进程边界之前；当前 `npm run electron:dev` 会先构建 sidecar，旧/手动 Electron 实例可能在 dev parent 退出后继续报该路径错误。current `app-server` 已重新构建并通过 macOS RPATH 准备，`--help` 与真实 `initialize` JSON-RPC 均成功；`smoke:app-server-stdio` 仍被共享工作树 fixture 的旧 `sessionId`/新 `threadId` 协议漂移阻塞，不归本存储写集。只读盘点确认 `~/.lime` 没有模型数据库，当前 Lime `app-server/lime.db` 为 93 个系统 Provider、0 个 API Key；已知有 8 个自定义 Provider、9 个 API Key 的源是独立 `content-studio/app-server/lime.db`，不属于同一 Lime 产品边界，未自动纳入候选。跨应用仅模型控制面导入需单独明确确认，确认前不写真实数据库。
- **2026-07-20 跨产品误导入撤回**：此前显式传入 `content-studio/app-server/lime.db` 导入的 8 个 Content Studio 自定义 Provider、9 条关联 API Key 和 `migration.model_control.v1` marker 已按用户确认完整撤回。清理前备份为 `/tmp/lime-model-control-target-before-cleanup-20260720T082900.db`，权限 `0600`、SHA-256 `1cac22b0857dd75b9a1ebb920187a8b7d3faf6c2a39570f83b67b2bc228c0425`、`integrity_check=ok`；清理基线恢复为 93 个系统 Provider、0 个自定义 Provider、0 条 API Key，`agent_sessions=0`。系统 Provider、session、MCP、UI state、模型偏好和 `model_route_generation` 未删除；Content Studio source 未修改。后续用户手动添加属于 current 用户状态，不得再解释为迁移结果。
- **2026-07-20 startup recovery 事实源收口**：真实 Electron 首次启动仍因 `recover_agent_control_spawns` 在 transport 前扫描 1047 个 deprecated raw EventLog 并对完整文件做 SHA-256 而超过 30 秒；macOS `sample` 明确定位到 `EventLogWriter::list_queued_session_ids -> scan_event_log_path`。生产恢复已删除 raw EventLog queued discovery 及其 helper；queued projection 现在必须 join current `canonical_threads`，production 还要求非空 `rollout_path`，因此旧 `runtime/projection_1.sqlite` 的 2 条 stale queued row 不再触发恢复。真实 AgentRoot 首 JSONL 响应为 `3198ms`；Electron 日志出现 `app-server ready protocol=appserver.v0 version=1.107.0`，仅保留 1 个 sidecar，Bridge health `57ms`。清理完成时真实 `app_server_handle_json_lines -> modelProvider/list` 返回 `93/0/0`；之后用户手动添加的数据不属于 migration evidence。
- 本轮定向回归：Electron storage filter 31/31；`lime-core app_paths` 34/34、`lime-core` 全量 unit 688/688、core plugin 98/98、`lime-agent` 274/274（unit）、durable memory root 4/4、tool I/O offload 6/6、infra telemetry 49/49、session files 11/11、logger 10/10、manifest 2/2、cleanup 5/5、startup migration 7/7、API key migration 2/2、App Server 参数 16/16、session/log root 注入 2/2、support bundle 3/3、log helper 1/1、WebSocket test-only store 1/1、Connect registry cache 注入 1/1、RelayRegistry cache round-trip 1/1、Plugin resolver 9/9、Plugin worker turn 7/7；storage-root guard 11/11；`lime-services` default、`local-whisper`、`local-sensevoice` 显式模型路径测试均为 2/2；voice install-dir Rust unit 1/1、公共 JSON-RPC 2/2、Renderer gateway 11/11、App Server client 63/63、protocol codegen 731 类型零漂移。Rustfmt、scoped Prettier、`npm run docs:boundary`、`npm run governance:legacy-report`（边界违规 0）、`npm run governance:scripts` 和 scoped `git diff --check` 通过。
- `npm run smoke:session-files-electron-fixture` Gate B 通过：`ok=true`，真实 Electron/preload/IPC、`app_server_handle_json_lines` 和全部 `sessionFile/*` current 方法均命中；`resolvePath` 返回 `<E2ERoot>/electron-user-data/app-server/artifacts/sessions/<session-id>/files/...`，`storageContract=AgentRoot/artifacts/sessions`，旧 `session_files_*` 命令为 0，console error 为 0。证据：`.lime/qc/gui-evidence/session-files-electron-fixture/session-files-electron-fixture-summary.json`。
- `npm run smoke:settings-provider-migration-electron-fixture` Gate B 通过：`ok=true`，manifest 为 `storage-migration.v1 / database-path-v1 / completed / copied`，source/target SHA-256 长度均为 64，旧 Product DB 保留且 `oldProductDbUserSchemaObjectCount=84`，重启后 Provider 继续可见；只读目录场景 `permissionSourceUnchanged=true`、manifest/target 均未生成，console/page/invoke error 与 renderer crash 均为 0。证据：`.lime/qc/project-gates/standalone-shell-02-20260719T045436132Z-617795/shell-02-provider-migration/settings-provider-migration-fixture-summary.json`。
- `npm run smoke:settings-developer-electron-fixture -- --timeout-ms 120000` Gate B-F 在重建 `app-server` 后通过：`result=pass`、24/24 断言通过；真实 Electron/preload/IPC、`app_server_handle_json_lines`、全部 current diagnostics 方法、`get_config` 均命中，`electron-ipc`、legacy/mock/console/page/invoke error 均为 0。脱敏 clipboard sink 仅记录 payload shape 布尔，其中 `currentLogPathAtAgentRoot=true`（规范化后以 `/app-server/observability/log/lime.log` 结尾），不保存绝对路径、日志、配置、Provider/MCP 数据或凭证。证据：`.lime/qc/project-gates/standalone-settings-developer-20260719T063651110Z-33309/settings-developer-current-diagnostics/settings-developer-fixture-summary.json`。
- `npm run verify:gui-smoke` 首次在 `app-server-initialize` 失败：E2E Host 把临时 AgentRoot 同时传入 `--data-dir` 与 `LIME_AGENT_RUNTIME_ROOT`，旧判定又通过读取该 override 的 `preferred_agent_root()` 把临时根误当平台默认根，进而扫描真实 macOS Product DB 并因活动 WAL fail closed。修复后同一 Gate B-F 通过，21/21 断言成立、App Server IPC 命中 33 次、legacy/mock/console/page/invoke error 均为 0；未删除、checkpoint 或修改真实 Product DB/WAL。失败证据：`.lime/qc/project-gates/standalone-shell-01-20260719073842-4780/shell-01-electron-smoke/summary.json`；通过证据：`.lime/qc/project-gates/standalone-shell-01-20260719075756-97852/shell-01-electron-smoke/summary.json`。
- `npm run test:contracts` 完整通过：protocol type generation 无漂移，App Server client 292 项、command、harness、modality、scripts、Electron release、cleanup report 与 docs boundary 契约均通过。
- **2026-07-20 清理后复验**：`rustfmt --check`、EventLog test-only scanner 19/19、`npm run governance:legacy-report`、`npm run test:contracts` 和 `npm run verify:gui-smoke` 均通过；运行态只保留 1 个 current sidecar。清理时目标 SQLite `integrity_check=ok`、`93/0/0`、session 0、错误 marker 0，真实 `modelProvider/list` 返回同样 `93/0/0`。此前的 `101/8/9` 仅是已撤回的 Content Studio 误导入历史，不得作为完成证据。pending recovery 与扩大 `test:rust:related` 的当前复跑仍被并行 v2 协议热区 `processor/dispatch/v2_ingress.rs:38` 未覆盖 `ThreadGoalSet/Get/Clear` 阻塞。
- **2026-07-20 Provider 保存与测试修复**：模型控制面所有 current JSON-RPC method 已进入 30 秒 App Server IPC 档，`modelProvider/testConnection`、`modelProvider/testChat`、`modelProvider/fetchModels` 进入 60 秒 provider-network 档，不再被 Renderer 的 5 秒 truth timeout 提前终止。添加页在配置已保存但测试失败时保留表单、展示可读恢复提示，不再静默进入详情；按钮改为高对比主操作并显示保存/测试进度。桥接策略与错误格式化 43/43、添加页 19/19、Provider 设置页 22/22、scoped ESLint 与五语言 JSON/键完整性通过；Gate A 浏览器镜像确认启用按钮白字、48px、控制台 0 error。隔离 Gate B fixture 命中 46 次 Electron IPC，覆盖 create/update/key/fetch/testConnection/testChat，legacy/mock 为 0；fixture 最终被既有非流式 Chat fixture 的 `stream ended before terminal event` 阻塞，因此本轮只声明保存/测试 current 主链已命中，不声明 CRUD fixture 全绿。
- 本次 voice model services 切片首次复跑 contracts 时，App Server client contract 曾因并行 Agent Runtime projection 迁移短暂中止；本切片未补回已删除旧文件。并行 owner 同步 current projection imports 与 contract allowlist 后，最终 `npm run test:contracts` 已完整通过：protocol 731 类型零漂移、App Server client 292 项以及 command/harness/modality/scripts/release/cleanup/docs 均通过。
- 完整 Electron 共享套件未在本切片重跑；历史结果为 103/108，5 个失败均在并行 `turn/start -> thread/read` canonical identity 测试。本切片以更贴近改动的 session files Gate B 作为真实 GUI/Bridge 证据，没有把历史共享套件结果伪装为当前通过。

### 2026-07-26 P0 根注入与整库迁移退役记录

- Electron 在 host 初始化期一次性解析 `HostSessionData/AppDataRoot/AgentRoot`；Windows `sessionData` 从 `%APPDATA%\lime` 移到 `%LOCALAPPDATA%\LimeCloud\lime\host-session`，E2E 使用隔离根。sidecar 环境通过 `APP_SERVER_APP_DATA_DIR` 与既有 AgentRoot override 同时下发双根。
- App Server `StorageRoots` 与 `LocalAppDataSource` 同时接收显式 AppDataRoot/AgentRoot；Product DB 只落 `AgentRoot/lime.db`。diagnostics、support bundle、Soul style-pack 与 MCP OAuth 均只消费注入根，storage guard 的叶子 root bypass 基线已降为零。style-pack 同根安装幂等、异根拒绝；MCP OAuth 未注入 root 时 fail closed。
- `core::app_paths` 的 AppDataRoot/Product DB path resolver 已改为只解析、不建目录。旧 Product DB 整库复制、`storage-migration.v1 / database-path-v1` manifest 与 startup cleanup 三个 owner 已从编译图删除，legacy catalog 将其分类为 `dead / deleted / forbidden-to-restore`；模型控制面 exact allowlist 迁移保持唯一 DB `compat`。
- 2026-07-27 最终复验：Electron path + sidecar 环境 `39/39`、Electron typecheck、`lime-core config::yaml` `17/17`、`lime-core app_paths` `17/17`、`lime-core --lib` `670/670`、App Server style-pack 注入/异根拒绝 `2/2`、storage/legacy guard `238/238`、`cargo check -p app-server`、scoped `git diff --check` 与 `npm run governance:legacy-report`（边界违规 `0`）通过；此前 App Server storage roots `2/2`、diagnostics/support bundle `14/14`、MCP OAuth store `6/6`、未注入 fail-closed `1/1` 继续有效。`npm run test:contracts` 完整通过（App Server client `284 checks`）。`npm run verify:gui-smoke -- --timeout-ms 180000` 通过真实 Electron/preload/IPC/App Server sidecar，App Server ready、Claw shell reload 与 Memory 设置页均成功，evidence `result=pass`。Windows 真机 runner 仍待完成。
- `database/migration_v6.rs`、启动注册、`app_paths` managed project 复制 helper 及正向迁移测试已删除；legacy catalog 与 storage guard 禁止旧文件、迁移标记和复制符号回流。该 surface 分类为 `dead / deleted / forbidden-to-restore`，不以 no-op 包装保留。Windows 真机 runner、cleanup manifest、P1 其余项和责任开发者架构确认仍为 `pending`，不得标记 P0 完成。
- App Server 配置读取边界会把已落盘的 `routing.default_provider` 一次性归一化到 current 顶层 `default_provider`，顶层已存在时以顶层为准，其他未知 routing 字段继续 fail closed；这避免旧 current YAML 令 Windows sidecar 在本地数据源初始化前退出。`core/src/config/yaml.rs` 当前超过 1000 行，本轮只加入该必要边界修复；退出条件是下一次扩展配置解析前先把 YAML value normalization 与对应测试拆到独立子模块，使 owner 回落到 800 行附近。

### 2026-07-19 P1 dated rollout 实现记录

- 新增 App Server 内部 `RolloutStore`，production `ProjectionStore::initialize_with_agent_root` 只从显式 `AgentRoot` 派生 `sessions/YYYY/MM/DD/rollout-YYYY-MM-DDTHH-MM-SS-<thread-id>.jsonl`。日期取 Thread 创建时本地日期，metadata 首行保存 UTC 时间、session/thread identity、相对 path 与初始 Thread；后续行保存 typed `ThreadHistoryChangeSet`，不写 raw provider/lifecycle `AgentEvent`。
- `canonical_threads` 增加 nullable、唯一的 `rollout_path`。新 Thread 在 SQLite transaction 中先完成 row/identity 约束，再同步创建并 `sync_data` metadata 文件，最后 commit metadata row；history apply 先 normalize 并完成未提交 projection 约束，再同步 append rollout，最后 commit SQLite。文件成功而 DB commit 失败时，同 `(thread_id, sequence, fingerprint)` 重试不重复 append；同 sequence 不同 fingerprint fail closed。
- production constructor 遇到旧 `rollout_path IS NULL` 行拒绝继续 history append，不猜目录、不静默生成残缺 transcript；旧行不迁移，所属旧库进入 exact-path cleanup manifest。rollout relative path 只允许 `sessions` 下 normalized component，thread id 只允许安全文件字符，并拒绝现存 symlink 前缀。
- ThreadStore `archive_thread/unarchive_thread` 已在同一 AgentRoot 内把 active rollout 原子 move 到 `archived_sessions/<filename>` 并更新 state `rollout_path`。公共 owner 已切到 v2 `thread/archive` / `thread/unarchive`，并发出 `thread/archived` / `thread/unarchived`；Renderer typed client、session facade、Sidebar 与归档页均已接管。`agentSession/archiveMany` 和 `AgentSessionUpdateParams.archived` 已删除并补负向守卫。
- production constructor 在 `canonical_threads` 为空时扫描 active/archived rollout，以单个 SQLite transaction 重建 Thread/Turn/Item、history apply fingerprint、archive flag 与 current relative path；若 `projected_sessions/turns/items/watermarks` 也全部为空，同一事务重建 AgentSession overview、turn 与有界 item/message summary。已有 legacy projected row 时 no-clobber，不清表、不隐式迁移。history record 同时保存 request fingerprint 与 canonical `content_digest`，payload 篡改 fail closed；用户消息全文仍只在 rollout，SQLite 只保留按 512-byte 阈值截断的单份 summary。
- raw EventLog 仍保留为 bounded diagnostics 与旧 projected read-model repair 输入，但启用 RolloutStore 的 production instance 已停止用它替换 canonical ThreadStore。独立 `runtime/projection_1.sqlite` 仍是 `deprecated / frozen-for-removal` 过渡 projection。
- 定向验证：canonical ThreadStore 29/29、projection rebuild 1/1、storage roots 1/1、production main constructor 1/1、storage-root guard 12/12、v2 protocol 12/12、TypeScript App Server client 71/71、`thread_v2_jsonrpc` 7/7、session-history fixture guard 5/5 均通过。`node scripts/check-app-server-client-contract.mjs` 为 299 checks，完整 `npm run test:contracts`、`npm run governance:legacy-report`（边界违规 0）与 `npm run governance:scripts` 通过。
- `npm run smoke:agent-session-history-electron-fixture -- --timeout-ms 240000` macOS Gate B 通过：真实 Electron/preload/IPC 命中 `app_server_handle_json_lines` 与 v2 `thread/start/archive/unarchive/read/list/turns/list/resume`；归档后 `sessions` 为 0、`archived_sessions` 为 1，反归档后恢复到 `sessions/YYYY/MM/DD` 且 archive 为 0；sidecar 重启后读回同一 archived Thread，`thread/archived` 与 `thread/unarchived` notification 均命中。canonical read model 恢复 3 个 Turn、9 个 Item，DOM 顺序稳定，console/page error 均为 0。证据：`.lime/qc/gui-evidence/agent-session-history-electron-fixture/agent-session-history-electron-fixture-summary.json`。
- Gate B oracle 通过 current `app_server_drain_events(includeRecent=true)` 读取 Electron Host recent notification buffer，避免 Renderer 后台 event drain 抢先消费后产生假阴性；archive/unarchive notification 现在均为硬断言。该观察入口只属于 `test-only`，不成为第二条业务通道。
- 本切片未实现 session/date dry-run GC、Windows runner、模型迁移或真实用户数据清理；P1 保持 `in_progress`，责任开发者架构确认仍为 `pending`。

### 2026-07-19 P1 SQLite 物理 owner 拆分记录

- production storage contract 新增 `<AgentRoot>/sqlite/state.sqlite` 与 `<AgentRoot>/sqlite/thread_history.sqlite`；`runtime/projection_1.sqlite` 保留为可丢弃的 deprecated projection。App Server main constructor 必须使用 projection/state/history/AgentRoot 四路径构造器，单文件 constructor 仅供隔离测试。
- 表归属固定为：state 只含 `canonical_threads`、`canonical_thread_spawn_edges`；history 只含 `canonical_turns`、`canonical_items`、`canonical_history_applies`；projection 只含 `projected_sessions`、`projected_turns`、`projected_items`、`projection_watermarks`。新增物理 inventory 回归，任一表落错库即失败。
- 所有组合事务以 state DB 为 main，先 attach history、再 attach projection。这样 deprecated projection 中即使存在旧 `canonical_*` 壳，也不能覆盖 current state 的名称解析。history 内部恢复 item -> turn FK；state/history 跨库不伪造 FK，delete 显式按 item、turn、apply、metadata 顺序执行。
- constructor 现在区分两种 rebuild：state/history 全空时从 dated rollout 重建 canonical；canonical 已存在但 projection 四表全空时独立重建 projection。两种路径都不读取、复制或迁移旧 session/event/DB/config/log/cache/memory 数据。
- Electron thread-read fixture 已改为向 state/history owner 写入隔离测试数据，通过 SQLite `ATTACH` 完成一个 fixture transaction；fixture 明确禁止 `projection_1.sqlite` canonical 写入，静态守卫 5/5 通过。本切片未修改或删除真实用户数据库。
- 暂留债务：`canonical_threads.last_sequence` 仍作为 attached transaction 的 commit watermark 留在 state，退出条件是迁入 history projection state 并补 rollback/restart 回归；`projection_1.sqlite` 仍有读取消费者，只有 GUI/read model 完全转向 ThreadStore 后才能标成 `dead` 并进入 exact-path cleanup manifest。
- 当前共享工作树的 App Server 全量 unit 为 1283 passed / 26 failed；失败集中在并行 provider/model、旧 `sessionId` 契约、permission preflight 与 plugin generation 热区，不属于本切片写集。SQLite 定向测试、contracts、治理扫描和 Electron Gate B 以本节后续验证结果为准，不得用该共享全量结果替代。

这是重大架构变更：durable history 将从高频 event/projection 双轨迁到“canonical rollout 文件 + 小型 state/read-model SQLite”。进入 release evidence 前必须由责任开发者确认 [architecture.md](../aiprompts/architecture.md) 的 Agent 主链、`App Server -> ThreadStore -> GUI` 存储边界，并在本计划填写确认人和日期。

架构文档已增加 `6.2.1 Storage alignment target`，明确当前 SQLite/EventLog 三写仍是 `deprecated / frozen-for-migration` 过渡实现；责任开发者确认仍为 `pending`，不得据此关闭 P1。

## 8. 完成定义

本计划完成必须同时满足：

1. 每个 current Thread/Session 只有一个 canonical rollout path；AppDataRoot 下只有一个 AgentRoot，`~/.lime` 不写 DB/history/log。
2. Thread/Turn/Item durable history 可从 JSONL 冷启动、重建和恢复；SQLite 中没有等价完整 transcript 或逐 delta 永久副本。
3. archive 是移动，delete/date GC 按文件和 artifact 目录执行，并输出可审计 manifest；derived projection/cache 可独立删除和重建。
4. `current / compat / deprecated / dead` 分类、旧路径回流守卫、治理扫描、contracts、Rust 定向测试和 Gate A/B 均通过。
5. macOS 证据完整，Windows 路径和原子文件操作有真实 runner 证据；未验证平台不得标记完成。
