# Lime v1 存储对齐方案

状态：`in_progress`

日期：2026-07-19

范围：`internal/refactor/v1` 的 Agent runtime、App Server、Thread/Turn/Item、Electron userData、项目资料、模型下载和可重建缓存。

本文件是存储域的事实盘点和执行方案。迁移白名单是模型控制面语义状态与已下载模型；旧 Lime 会话、事件、projection、Product DB 其他表、缓存、日志和旁路文件不做内容迁移。模型控制面只迁移 provider/key ciphertext/UI state/model preference/active tab，按 source signal 选取同一产品旧库并通过 WAL 只读事务读取，不复制整库。本文不授权直接删除真实用户目录，物理清理仍须重新盘点并经过显式 maintenance 边界。

Codex `~/.codex` 与 Lime 的语义分组见 [02-codex-lime-storage-matrix.md](02-codex-lime-storage-matrix.md)，实际顶层项与 Lime 两个根的一一执行账本见 [03-one-to-one-storage-alignment-plan.md](03-one-to-one-storage-alignment-plan.md)。

## 1. Codex 对照范围

本方案只以实际 `~/.codex` 作为 Codex storage inventory baseline：rollout/session、state/log SQLite、goals、memory、skills/plugins、MCP cache 和 shell snapshots。VS Code、扩展目录与其他 profile 均不在一一对照范围内；Codex 源码只能辅助验证观察项的 writer/reader 语义，不能把本机不存在的路径冒充实际基线。

## 2. 当前本机盘点

### 2.1 Lime 顶层占用

本节是 `captured_at=2026-07-19T10:00:28+08:00` 的只读快照；活跃 Electron/Codex 进程会改变大小、文件数和 WAL，迁移前必须重新生成带时间戳的 inventory。`~/Library/Application Support/lime` 当时约 7.5 GB；`du`、Finder 和运行时文件的统计口径可能存在差异。主要目录如下：

| 路径          |   约占用 | 当前用途判断                                                   | 分类                                                |
| ------------- | -------: | -------------------------------------------------------------- | --------------------------------------------------- |
| `app-server/` |  4.17 GB | 当前 App Server product DB、runtime event、projection、sidecar | `current`，但内部有重复持久化                       |
| `runtime/`    |  1.52 GB | 另一套 event JSONL、旧 projection、sidecar                     | `deprecated`，禁止继续写入                          |
| `models/`     |   796 MB | voice/whisper 下载模型及未完成下载                             | `current` 产品资产；下载临时文件需治理              |
| `Partitions/` |   468 MB | embedded browser 的 cookies/storage/profile/cache 混合数据     | `current` host state，不能整体标为可重建            |
| `Code Cache/` | 345.8 MB | Electron renderer code cache                                   | `current` host cache，可重建                        |
| `harness/`    | 262.6 MB | harness 工具 IO/memory                                         | `compat` 或 `deprecated`，由 live consumer 扫描决定 |
| `aster/`      | 244.8 MB | 退役 runtime 遗留目录                                          | `dead` 候选，先完成负向扫描                         |
| `projects/`   |  73.6 MB | Lime 产品项目资料                                              | `current`，不属于 Codex ThreadStore                 |
| `sessions/`   |  12.7 MB | 旧 session repository/旁路文件                                 | `dead` 候选，不迁移，切换后按精确路径清理           |

其他小目录（`mcp`、`plugins`、`agent-apps`、`memory`、`request_logs`、`logs` 等）必须按下文的持久性分类统一归属，不能继续由任意模块直接拼路径。

### 2.2 Lime 当前 Agent runtime 的双轨证据

为避免把 Electron `userData` 与系统的 `Application Support` 父目录混淆，本文用 `<electronUserData>` 表示 `app.getPath("userData")`，本机值为 `~/Library/Application Support/lime`。Electron 在 [appServerHost.ts](../../../electron/appServerHost.ts) 中把 App Server data dir 固定为 `<electronUserData>/app-server`（约 1080-1082 行）。App Server 启动时又在 [main.rs](../../../lime-rs/crates/app-server/src/main.rs) 同时挂载 product DB、EventLog、TraceEvent、ProjectionStore 和 TelemetryStore（约 158-183 行）。[storage_roots.rs](../../../lime-rs/crates/app-server/src/runtime/storage_roots.rs) 将它们派生为：

```text
<electronUserData>/app-server/lime.db
<electronUserData>/app-server/runtime/events/
<electronUserData>/app-server/runtime/traces/
<electronUserData>/app-server/runtime/sidecar/
<electronUserData>/app-server/runtime/projection_1.sqlite
<electronUserData>/app-server/runtime/telemetry_1.sqlite
```

但 [app_paths.rs](../../../lime-rs/crates/core/src/app_paths.rs) 仍把 `logs`、`request_logs`、`projects`、`sessions`、`agent` 等解析到 `<electronUserData>/runtime/<name>`（约 103-130、298-316 行）。这产生了两个物理 runtime root：

| root                                          |                               文件数/占用 | 关键证据                         |
| --------------------------------------------- | ----------------------------------------: | -------------------------------- |
| `lime/runtime/events`                         |                   740 个 JSONL，约 698 MB | 与 App Server event 文件名无重叠 |
| `lime/runtime/projection_1.sqlite`            | 约 760 MB；`projected_items` 约 68,508 行 | 旧 projection 仍存在             |
| `lime/app-server/runtime/events`              |                1,047 个 JSONL，约 1.06 GB | 当前 App Server writer           |
| `lime/app-server/runtime/projection_1.sqlite` |  约 2.54 GB；`projected_items` 442,131 行 | 当前 projection                  |
| `lime/app-server/runtime/sidecar`             |                                  约 92 MB | session/output sidecar           |

当前 projection 的 `canonical_threads` 约 137 行、旧 projection 仅 1 行；两库的 session/thread 集合没有重叠。这证明根级 `runtime` 是独立旧 owner，不是当前库的镜像；按当前决策它不再作为迁移源，只用于停写识别和精确清理。

两个 event 目录没有同名 JSONL，说明不是同一份数据的镜像，而是两套独立历史写入路径。仅合并目录名不能解决问题，必须先确定 canonical owner。

### 2.3 Projection 膨胀证据

当前 projection DB 的 SQLite `dbstat` 结果：

| 文件/表                                                 |   约占用 | 说明                           |
| ------------------------------------------------------- | -------: | ------------------------------ |
| `app-server/runtime/projection_1.sqlite`                |  2.54 GB | page count 620,120，freelist 0 |
| 其中 `projected_items`                                  |  2.28 GB | 442,131 行                     |
| 其中 `canonical_items`                                  |  78.9 MB | 7,976 行                       |
| `lime/runtime/projection_1.sqlite` 的 `projected_items` | 716.1 MB | 68,508 行                      |

[projection_schema.rs](../../../lime-rs/crates/app-server/src/runtime/projection_schema.rs) 把每个 `AgentEvent` 都落到 `projected_items`；[projection_store.rs](../../../lime-rs/crates/app-server/src/runtime/projection_store.rs) 的 `insert_projected_item`（约 1604-1628 行）写入 `event_id`、sequence、event type 和 `payload_summary_json`。这会把高频 delta/lifecycle event 长期保留为每行记录，而 canonical ThreadStore 只需要 coalesce 后的 Item snapshot。

另外，`app-server/lime.db` 文件约 386 MB，但 page count 96,598 中有 87,348 个 freelist page，理论上约 358 MB 是可回收空间；`app-server/projection.sqlite` 是空文件，根目录 `lime/app.db` 也为空或近空，均说明迁移/旧入口没有彻底退场。任何 `VACUUM` 或删除动作都必须在停写、备份和校验之后执行，本方案不直接执行。

### 2.4 Codex 本机结构基线

本机 `~/.codex` 的只读盘点会被活跃进程持续改变。下表只记录职责和量级，不是稳定容量基线；任何数值报告必须携带 `captured_at`、平台、文件数、主文件/`-wal`/`-shm` 和活跃 writer：

| 路径                                                    |                                               约占用/数量 | 语义                                                                   |
| ------------------------------------------------------- | --------------------------------------------------------: | ---------------------------------------------------------------------- |
| `sessions/`                                             |            活跃变化；同时存在日期 JSONL 与一个根级旧 JSON | active canonical rollout；按 `YYYY/MM/DD` 分区，旧 JSON 单列清理       |
| `archived_sessions/`                                    |                                   本次观察到 rollout 文件 | archive 是移动/状态边界，不是复制一份 transcript                       |
| `logs_2.sqlite*`                                        |              主库和 WAL 均可能达到 GB 级，且有活跃 writer | 运行日志/观测旁路；必须按 DB maintenance 处理，不能按目录直接删除      |
| `state_5.sqlite*`                                       | 活跃 writer；包含 thread/index/job/tool/config 等语义状态 | 不只是小索引，允许字段和 payload 上限必须逐表确认                      |
| `history.jsonl`、`history.json`、`session_index.jsonl`  |                                          三个不同 surface | input history、未知 owner JSON、session index 分开治理，不替代 rollout |
| `skills/`、`plugins/`、`mcp-cache/`、`shell_snapshots/` |                                                依功能变化 | 独立 owner；缓存和资产不混入 thread history                            |

实际目录证明 rollout 与多个 SQLite surface 分工，但不能只凭文件名断言每个库都可重建。源码可作为 writer/reader 的补充证据，例如 canonical rollout 先写 JSONL、history projection 再 materialize；这类源码结论必须在一一账本中标成 `writer-confirmed`，与 `observed` 分开。`~/.codex/sqlite/codex-dev.db` 在本次盘点时仍被桌面进程打开，不能把整个 `sqlite/` 判成旧根。Lime v1 的目标仍是 canonical rollout + 有边界的 state/read model，但当前 event/projection/canonical SQLite 三写尚未收口。

## 3. 目标事实源与分类

### 3.1 唯一事实源声明

> Lime Agent runtime 的 durable truth 是 App Server current owner 下的 Codex-style canonical rollout（按 Thread/Session 分文件），Thread metadata、graph、job/tool state 等按领域进入独立 SQLite 状态；Thread history/read model 只有在全量 rebuild 回归通过后才可标为可重建 projection。EventLog、trace、telemetry、sidecar 和 GUI cache 都不得成为第二 transcript owner。

### 3.2 目标物理拓扑

保留 Electron 的 host profile root，不把所有内容搬到用户主目录；在平台解析出的 `AppDataRoot` 下让 `app-server` 成为唯一 Agent runtime root：

```text
<AppDataRoot>/                           Lime durable product + Agent root
  app-server/                            Agent runtime current owner
    sessions/YYYY/MM/DD/rollout-*.jsonl  canonical active rollout
    archived_sessions/rollout-*.jsonl    canonical archived rollout
    sqlite/state.sqlite                  Thread metadata/graph/job/tool/catalog state
    sqlite/thread_history.sqlite         rebuildable Thread/Turn/Item read model
    sqlite/goals.sqlite                   user-visible goal state, not cache
    sqlite/memories.sqlite                memory pipeline/semantic state; rebuild 需实证
    observability/log/lime.log            bounded text diagnostics, not transcript
    observability/events/                bounded diagnostic events, not transcript
    observability/traces/                bounded traces/evidence
    observability/telemetry.sqlite       bounded metrics/usage
    artifacts/sessions/<id>/             sidecar/output/checkpoint blobs
    memories/                            memory owner
    cache/                               runtime catalog/MCP/plugin caches
  projects/                              Lime product workspace materials
  models/                                Lime product model assets/downloads
  plugins/                              Lime plugin packages (if still current)
  mcp/                                   OAuth/connection state (credential policy applies)
<HostUserData>/                          Electron profile/config root
<HostSessionData>/                       Chromium cookies/storage/network/cache root
```

这里的目录名是建议，不是立即迁移目标。macOS 上 `<AppDataRoot>` 与 `<HostUserData>` 可以物理同根；Windows 固定按 3.3 节拆分 local durable data 与 roaming host profile。实现时优先复用已有 `app-server` root，禁止同时引入第三个 runtime root。Codex 的 `sessions/YYYY/MM/DD`、`archived_sessions`、metadata 与 projection 分工必须对齐；Lime 的 `projects`、`models`、`artifacts` 等产品特有数据不应伪装成 Codex session。

### 3.3 全局应用数据与用户 Home 的边界

结论是：**按数据职责分开，不能把同一份 Agent runtime 分成两套写入。**

| 数据职责                           | macOS                                          | Windows                                                                           | 允许内容                                                                          |
| ---------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 安装根                             | `.app` bundle                                  | `%LOCALAPPDATA%\\lime`                                                            | 程序/updater；业务数据禁止写入                                                    |
| 应用/机器数据（唯一 current root） | `~/Library/Application Support/lime`           | `%LOCALAPPDATA%\\LimeCloud\\lime`                                                 | App Server、SQLite、sessions、模型、sidecar、项目索引、机器级缓存                 |
| Electron profile/config            | 同上，按 owner 分区                            | `%APPDATA%\\lime` 仅放小型 roaming 配置                                           | Preferences、窗口状态；不能承载 Agent DB/history/模型                             |
| Electron sessionData               | 与现有 host profile 一起治理                   | `%LOCALAPPDATA%\\LimeCloud\\lime\\host-session`                                   | cookies、storage、network state 与 cache；不能整体标成可重建                      |
| 用户主动管理的 home                | `~/.lime`                                      | `%USERPROFILE%\\.lime`                                                            | `AGENTS.md`、用户 skills、少量用户配置；可备份/迁移，不放 DB、session、日志、模型 |
| 工作区配置                         | `<project>/.lime`                              | `<project>\\.lime`                                                                | `AGENTS.md`、`AGENTS.override.md` 与项目级 metadata；按 Codex precedence 选择     |
| OS cache/log（可选）               | `~/Library/Caches/lime`、`~/Library/Logs/lime` | `%LOCALAPPDATA%\\LimeCloud\\lime\\Cache`、`%LOCALAPPDATA%\\LimeCloud\\lime\\Logs` | 仅可重建 cache、诊断日志；不作为 Thread/Turn/Item 事实源                          |

这样分开的原因不是“目录更整齐”，而是生命周期和安全边界不同：

1. Application Support/LocalAppData 是应用管理的机器状态，适合锁文件、SQLite WAL、模型和大缓存；应用升级或卸载可以按产品策略处理它们。
2. `~/.lime` 是用户拥有的跨应用配置，应该能独立备份、迁移和手工编辑；不能因为清理应用缓存而丢失用户 instructions/skills。
3. Windows 的 `%APPDATA%` 是 roaming data，只适合很小的可漫游配置；session、SQLite、模型和凭证不得放进 roaming root。Lime 当前 `dirs::data_local_dir()` 选择 `%LOCALAPPDATA%` 是正确方向，但 Electron 的 `app.getPath("userData")` 默认可能指向 `%APPDATA%`，不能直接把它当作 Agent data root。
4. 两个根都可以存在，但 `app-server`、`runtime/events`、`projection`、`sidecar`、`models` 等每个数据类别只能有一个 current owner。禁止通过“同时写 `Application Support/lime` 和 `~/.lime`”解决迁移问题。

当前代码的实际语义是：`preferred_data_dir()` 使用 macOS `Application Support/lime`、Windows `%LOCALAPPDATA%/LimeCloud/lime`；`preferred_agent_root()` 在其下固定追加 `app-server`；`user_home_dir()` 使用 `$HOME/.lime`，且 `resolve_user_memory_path()` 和 `resolve_home_skills_dir()` 把它作为首选用户 home；`resolve_workspace_runtime_agents_path()` 则使用 `<cwd>/.lime/AGENTS.md`。因此 `.lime` 不能继续被统称为 compat：它是 **user-home current（只限用户文件）**，旧路径不再提供迁移入口。Squirrel 安装包根 `%LOCALAPPDATA%/lime` 不属于 AppDataRoot，不能写入 Agent 数据。

Lime 的推荐解析契约是：

```text
InstallRoot  = macOS .app bundle
               Windows %LOCALAPPDATA%\\lime
HostUserData = Electron profile/config root # macOS 可与 AppDataRoot 合并
HostSessionData = Electron Session cookies/storage/network/cache root
AppDataRoot  = macOS ~/Library/Application Support/lime
               Windows %LOCALAPPDATA%\\LimeCloud\\lime
AgentRoot    = AppDataRoot/app-server       # 唯一 Agent runtime
UserHome     = $HOME/.lime                  # 仅用户文件
Workspace    = cwd/.lime                    # 仅工作区文件
```

macOS 上 `HostUserData` 与 `AppDataRoot` 可以物理同根但必须按 owner 分区。Windows 固定保留 Electron profile 在 `%APPDATA%\\lime`，由 Host 显式把 `%LOCALAPPDATA%\\LimeCloud\\lime\\app-server` 作为 `--data-dir` 传给 App Server，并单独验证 `sessionData`。便携/E2E 模式可显式设置受控根，但不得使用 Squirrel 安装根 `%LOCALAPPDATA%\\lime`。

不能让 `app.getPath("userData")` 在 Windows 上隐式决定大体积 Agent data 的位置。`LIME_AGENT_RUNTIME_ROOT` 只能用于测试、便携模式或显式运维迁移；设置它时必须替换 `AgentRoot`，不能再与默认 root 双写。

### 3.4 current / compat / deprecated / dead

| surface                                                                     | 目标分类                               | 处理原则                                                          |
| --------------------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------- |
| `app-server` 下的 canonical rollout + ThreadStore                           | `current`                              | 唯一可新增能力的 owner                                            |
| metadata/graph/mailbox SQLite、rebuildable history                          | `current`                              | 可多文件，但只能由 ThreadStore owner 管理                         |
| 旧 Lime rollout/event/DB import adapter                                     | `dead / forbidden-to-restore`          | 不迁移旧历史；不得进入启动链或 current catalog                    |
| `lime/runtime/**`、root `sessions/`、旧 `session_repository`                | `deprecated -> dead`                   | 立即停止新写入；新根验证后按 manifest 精确清理                    |
| `lime/app-server/projection.sqlite`、root `app.db`、无 consumer 的 `aster/` | `dead`                                 | 先做负向扫描和备份校验，再删除并加回流守卫                        |
| `harness/`                                                                  | `compat` 或 `deprecated`               | 只有真实当前测试/诊断 consumer 才可保留；不进入生产 Agent history |
| `models/`                                                                   | `current` 产品旁路                     | 唯一允许迁移的旧数据；checksum 校验后原子发布                     |
| `projects/`、插件包、MCP OAuth                                              | `current` 产品旁路                     | 不做旧数据迁移；新 owner 生效后旧副本按产品策略清理               |
| `Cache`、`Code Cache`、GPU/Dawn cache                                       | `current` host cache                   | 由 Electron/Chromium 管理，可重建，禁止被 runtime 读取为事实      |
| `Partitions/`、Cookies、Local/Session Storage、Trust Tokens                 | `current` host semantic/security state | 由 `HostSessionData` owner 管理，不能整体标成 cache 或无条件删除  |

### 3.5 数据库写入边界：学 Codex 的“文件事实源 + 小索引库”

“很多写入数据库的方式不对”需要精确拆成两类：关系型 metadata 写入本身没有问题；把高频事件、transcript、完整 payload、sidecar 内容和可重建列表都塞进 SQLite，才是当前 Lime 难清理的根因。

| 数据                                                                                | Codex/Lime 目标事实源                 | 是否允许进入 SQLite                                     |
| ----------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------- |
| 用户/助手 canonical rollout、compaction、rollback marker、unknown line              | `sessions/YYYY/MM/DD/rollout-*.jsonl` | 禁止存一份等价完整副本                                  |
| Thread metadata、rollout path、archive/status、fork lineage、provider/model default | `state.sqlite`/产品 metadata DB       | 允许；必须可由 rollout backfill                         |
| Thread/Turn/Item page/read model                                                    | `thread_history.sqlite`               | 允许；只能是可丢弃、可重建 projection                   |
| mailbox、agent graph、goal 状态                                                     | 对应小型 state DB                     | 允许；这是语义状态，不是 cache                          |
| 高频 delta、trace、诊断 event、usage telemetry                                      | 有界 JSONL/专用 telemetry DB          | 允许短期写入；必须有 TTL/size 上限，不得成为 transcript |
| 图片、文件快照、tool output、插件包、模型文件                                       | `artifacts/`、`models/`、插件目录     | 禁止把 blob 内容写进主 DB                               |

当前 [projection_schema.rs](../../../lime-rs/crates/app-server/src/runtime/projection_schema.rs) 的 `projected_items.payload_summary_json` 和 [projection_store.rs](../../../lime-rs/crates/app-server/src/runtime/projection_store.rs) 的逐 `AgentEvent` 插入违反了最后一条 history 边界：即使字段名叫 summary，它仍会为每个 delta/lifecycle event 保留一行，最终形成 2.28 GB 的派生表。P1 必须把它改成 canonical item coalesce；旧 `projected_items` 不迁移，current rebuild 验证后直接进入清理清单。

### 3.6 目录级清理协议

清理的主键应是 `thread_id/session_id -> canonical rollout path -> owned artifact directory`，而不是“在多个数据库里猜哪些行属于这个 session”。目标操作顺序：

```text
resolve thread_id from current state index
  -> resolve one canonical rollout path
  -> stop/flush live writer
  -> delete or rename that session directory/file
  -> delete owned artifacts/sidecars by the same session id
  -> delete metadata/index rows in one transaction
  -> mark/rebuild derived history and telemetry
```

具体规则：

1. `archive` 只把 rollout 文件从 `sessions/YYYY/MM/DD/` 原子移动到 `archived_sessions/`，并 patch metadata；不复制 transcript，不扫描全库。
2. `delete` 先确认 canonical path 和 artifact ownership，再删除该 session 的目录/文件，最后删除 state/index 行；任一步失败都不能 ack 删除完成。
3. `projection`、`thread_history.sqlite`、telemetry、cache 都视为 derived；删除其文件或对应分区后可从 canonical rollout/backfill 重建，不得阻塞用户删除。
4. 每个 session artifact 目录只能由该 session owner 写入，文件名必须包含稳定 session/thread identity；禁止跨 session 共享无 provenance 的 blob 目录。
5. `gc --dry-run` 只扫描四类目录：active sessions、archived sessions、artifacts、cache；输出 orphan path、bytes、last modified 和 reason。确认后按目录/文件删除，不执行无条件全库 SQL 清空。
6. 数据库保留 `rollout_path`、`artifact_root`、`projection_watermark` 等定位信息；删除后由 manifest/foreign-key 检查阻止“文件已删、索引仍指向旧路径”回流。

这不是“完全不用数据库”：数据库负责快速定位和状态一致性，文件系统负责大体积、按 session 可选择清理的 canonical 内容。两者之间必须只有一个 path/identity contract。

### 3.7 Codex-style 日期分区文件契约

以实际 `~/.codex/sessions` 的日期化 rollout 形态为准，Lime 的 canonical rollout 目标布局固定为：

```text
<AgentRoot>/sessions/YYYY/MM/DD/rollout-YYYY-MM-DDThh-mm-ss-<uuid>.jsonl
<AgentRoot>/archived_sessions/rollout-YYYY-MM-DDThh-mm-ss-<uuid>.jsonl
```

规则如下：

1. 一个 Thread/Session 对应一个 rollout JSONL 文件；文件名中的时间和 UUID 用于人类定位和冲突避免，真正 identity 仍以首条 session metadata 与 state DB 的 `thread_id/session_id` 为准。
2. JSONL 是 append-only canonical source；每行保存可恢复的 rollout item/event，启动时只修复尾部损坏，不把同一内容再写入 `projected_items`。
3. `YYYY/MM/DD` 按 session 创建时间分区，便于备份、按日期列举和批量清理；日期目录不是删除授权，删除前必须确认 active writer、归档状态和用户保留策略。
4. archive 是文件移动到 `archived_sessions/` 加 metadata patch，不复制一份历史；unarchive 是反向移动并更新 path。
5. state DB 只保存相对 rollout path、时间、title、status、provider/model、archive/fork lineage 和 projection watermark；不能保存完整 transcript/blob 作为第二事实源。
6. 单 session 清理按 state DB 定位一个文件；按日期清理先生成该日期的 session manifest，展示数量/字节/active 状态，确认后删除文件和对应 artifact，再删除索引行。
7. 文件大小、压缩和 TTL 是独立策略：Lime 可另行制定并验证压缩方案，但不得用“日期目录”掩盖单文件失控，也不得默认删除用户历史。

## 4. 分阶段执行计划

### P0：盘点、冻结旧写入、建立预算

写集：`lime-rs/crates/app-server/src/runtime/storage_roots.rs`、`lime-rs/crates/core/src/app_paths.rs`、Electron sidecar launch、相关测试和 `internal/refactor/data/` 文档。

1. 让所有 runtime writer 从一个 `StorageRoots`/`ThreadStorePaths` 实例获取路径；`app_paths::resolve_runtime_subdir("sessions")` 等旧解析不得用于生产读写或启动迁移。
2. 增加只读 storage report：路径、owner、持久性、文件数、字节数、最大单文件、SQLite page/freelist、canonical/projected 行数，并标记 `current / compat / stale / rebuildable`。报告不能读取或输出 prompt、credential、payload 内容。
3. 在启动和 CI 增加回流守卫：生产代码不得出现第二 `runtime` root、`projection_1.sqlite` 新建路径、旧 `session_repository` 写入口；测试夹具必须显式声明临时 data root。
4. 为每类数据登记初始预算（可配置而非硬编码）：canonical rollout 不自动删除；diagnostic events/traces 按天数和字节双上限；derived projection 以 canonical bytes 的倍数和绝对上限双约束；sidecar/cache/download 有 orphan/TTL 清理。具体数值由 Lime 压测和产品保留策略决定，不从目录外观推断 Codex 策略。
5. 为每个数据库写入点登记 `owner / source-of-truth / rebuildable / cleanup policy`；没有这四项的表不得新增字段或 payload 写入。

退出条件：同一启动配置解析出唯一 current App Server root；新测试能证明旧 root 只读；storage report 可重复生成并能选择 current root（版本/mtime/backfill 完成）；`npm run governance:legacy-report` 和相关 Rust tests 通过。

### P1：Codex-style ThreadStore 落地

写集：`lime-rs/crates/thread-store`、App Server `ProjectionStore`/EventLog/repair、schema/fixtures。

1. canonical 写入按 `sessions/YYYY/MM/DD/rollout-*.jsonl` 分区，并保存 Thread/Session metadata、source、cwd、provider/model、fork lineage；archive 用移动到 `archived_sessions`，不复制 transcript。
2. `ThreadStore::append_items` 只追加 canonical item；metadata 走独立 patch；rollout byte offset、ordinal、malformed tail、unknown item 和 repair provenance 进入 projection state。
3. `thread_history.sqlite` 只保存 coalesce 后的 Thread/Turn/Item read model，能从 rollout 全量重建；不能把每一个 delta event 当作永久 Item。
4. `EventLog` 降级为有界诊断/repair 输入。它可以记录 event order、repair、late event 和 provenance，但不得被 `thread/read` 或 GUI 当作 transcript。
5. canonical read、live notification、replay/evidence 使用相同 Item identity；renderer 不从 event 或文本合成第二份 history。
6. 先建立新路径和 identity contract，不迁移旧表内容：每个新 Thread 必须能从 state DB 定位唯一 rollout 文件和 artifact root；`projected_items` 不再作为清理所需的唯一索引。

退出条件：Codex `ThreadHistoryBuilder` 的 coalesce/rollback/fork/compaction lineage 在 Lime cold/live/replay/restart 中等价；projection 损坏时可从 canonical rollout 重建；不存在第二 transcript DB；删除一个 session 只需定位并处理其 canonical/artifact 目录和一组 metadata rows，不需要跨旧库猜测。

### P2：仅模型迁移与旧 owner 清理

旧 Agent 数据不做 import、merge、backfill 或 rollback。会话、event、projection、Product DB、sidecar、cache、日志、旧配置和 retired runtime 在 current writer、冷启动和 GUI 主链验证后，进入 `dry-run -> exact-path manifest -> 明确确认 -> cleanup`；禁止复制旧库或恢复 compat reader。

模型控制面与模型文件是两条独立迁移边界：模型控制面只读选取同一产品旧库中的 provider/key ciphertext/UI state/model preference/active tab；模型文件从已登记旧模型根只读枚举 `model id/version/required files/bytes/checksum`，复制到同盘 staging，校验完整性后原子 rename 到 `<AppDataRoot>/models`。失败保留 source，不发布半模型；成功后 source 进入独立清理 manifest。两者均不得携带 session、其他 DB 表、明文 credential、日志或缓存。

退出条件：生产代码不存在旧 session/DB/event importer；旧 root 无 writer/reader；模型迁移有 macOS/Windows checksum 与中断恢复测试；其余旧路径均有精确 cleanup row 和回流守卫。

### P3：清理、保留和容量控制

1. canonical rollout、archived rollout、用户项目文件默认不自动删除；删除/归档通过 App Server API，先删 canonical 关联再删 derived/sidecar。
2. `projected_items`、诊断 event、trace、telemetry、sidecar、MCP/catalog cache、失败下载统一实现 TTL/orphan 清理和显式 `dry-run` 报告；诊断日志至少达到 Codex 的分区/TTL 基线，projection 必须支持从 canonical 全量重建后再清理旧页/WAL。
3. voice/whisper 下载和唯一允许的模型迁移采用 staging + checksum + 原子 rename；启动时清理超时 `.download`/`extract`，禁止同一模型留下多个失败副本。
4. Electron Chromium cache 只由 Electron session/Chromium 生命周期清理；App Server 不读取 `Partitions`、`Code Cache` 或 `Local Storage` 作为业务事实。
5. 增加删除/归档后的空间回收观测：SQLite freelist、WAL、单 session bytes、projection/canonical ratio、sidecar orphan count。
6. 提供统一的 `storage report` 和 `gc --dry-run` 结果，清理命令必须按 owner 目录执行；禁止把 `DELETE FROM projected_items` 当作完整 session 清理，也禁止只删数据库不删 canonical/artifact 文件。

## 5. 验证门禁

### 数据契约

```bash
npm run test:contracts
npm run test:rust:related -- lime-rs/crates/app-server lime-rs/crates/thread-store lime-rs/crates/agent-protocol
npm run governance:legacy-report
```

必须补的定向回归：

- canonical append 与 metadata patch 的事务边界；
- JSONL 尾部损坏、重复 sequence、unknown line、projection lag 和全量 rebuild；
- `thread/read`、live notification、replay/export 的 Item ID/terminal status 一致；
- archive/unarchive/delete 的 rollout、metadata、history、sidecar 一致；
- restart/resume/fork/compaction 不创建第二 root、不丢 provider/model/route；
- storage report 不泄露 prompt、token、credential 或原始 payload。

### GUI / Electron

Agent runtime 或 bridge 路径改变后运行：

```bash
npm run verify:gui-smoke
npm run smoke:agent-runtime-current-fixture
npm run bridge:health -- --timeout-ms 120000
```

Gate B 必须证明真实 `Electron -> preload/IPC -> app_server_handle_json_lines -> App Server -> RuntimeCore -> ThreadStore -> GUI`，不能用 mock projection 或旧 `runtime/` 目录代替。

## 6. 目前结论与下一刀

当前差距不是 Codex 的目录层级少几个文件，而是 Lime 仍有两个 runtime root、多个 transcript/投影表示和没有执行的清理边界。最优先的一刀是 **P0-存储边界 + P1 canonical ThreadStore**：先冻结 `lime/runtime` 写入，把 App Server `app-server` 定为唯一 Agent root，再把高频 `projected_items` 改成可重建、可限额的 read model。完成这两步前，不应继续增加新的 history、event、session 或 projection 表。

本方案完成定义：

1. 生产路径只有一个 App Server runtime root 和一个 ThreadStore canonical owner；
2. Lime 可以解释每个字节属于 durable、derived、cache、asset 或 legacy；
3. canonical history 可重建，derived projection 可清理，删除/归档有可验证的空间回收；
4. `current / compat / deprecated / dead` 分类和回流守卫与 `internal/refactor/v1` 保持同步。
