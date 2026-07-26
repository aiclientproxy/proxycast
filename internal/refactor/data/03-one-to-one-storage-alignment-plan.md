# `~/.codex` -> Lime 存储一一对齐计划

状态：`in_progress`

快照：`captured_at=2026-07-19T10:00:28+08:00`，`platform=macOS`，`codex_home=~/.codex`

执行状态与跨模块进度以 [Codex / Lime 存储对齐执行计划](../../exec-plans/codex-lime-storage-alignment-plan.md) 为准；本文件是逐路径对照账本。目录正在被 Codex 和 Lime 活跃写入，大小、文件数和 `mtime` 只能属于某次带时间戳的 inventory，不能成为稳定规范。

## 1. 边界与硬结论

1. Codex 对照面只取实际 `~/.codex`。VS Code、扩展目录、其他 Codex/Electron profile 不进入一一对照清单。
2. 对照的是职责、事实源和清理边界，不是机械复制目录名。Codex 有而 Lime 无 owner 的功能必须 `exclude`，不能为目录 parity 造空实现。
3. canonical Thread/Turn/Item 采用一个 Thread 一个 rollout 文件，路径为 `sessions/YYYY/MM/DD/rollout-*.jsonl`；SQLite 只承载 metadata、语义状态和经验证可重建的 read model。
4. `-wal`、`-shm` 是主 SQLite 的伴生文件，不是独立数据库。旧数据库不迁移；清理必须先停写并把主库及 sidecar 作为一个 exact-path unit，禁止逐文件删除。
5. `~/.codex/sqlite/codex-dev.db` 在本次盘点时被桌面进程打开，且包含 automation、inbox、local thread catalog 表，不能把整个 `sqlite/` 判为旧库或直接按目录清理。
6. 迁移白名单是模型控制面语义状态与已下载模型。旧 session/event/projection/DB 其他表/cache/log/config/memory/plugin state 不做 import/merge/backfill；模型控制面只复制 provider/key ciphertext/UI state/model preference/active tab，物理清理执行 `dry-run -> exact-path manifest -> current owner/no-reader verify -> 明确确认 -> cleanup`。
7. 多模型 catalog、switch、capability、readiness 和 circuit breaker 参考 `/Users/coso/Documents/dev/rust/grok-build`；Provider/Model 分层、credential 隔离、capability/lowering 和 transient retry 参考 `/Users/coso/Documents/dev/js/opencode`。只吸收语义，不复制目录或数据库；存储和 Agent runtime owner 仍服从 Codex，模型控制面唯一 owner 是 `App Server + model-provider`。

## 2. 分类轴

每个 surface 必须分别登记以下字段，禁止继续把 `current candidate / wrong owner / rebuildable` 混在一个“分类”列里：

| 维度     | 允许值                                                                |
| -------- | --------------------------------------------------------------------- |
| 治理状态 | `current / compat / deprecated / dead`                                |
| 持久性   | `durable / semantic-state / derived / cache / ephemeral / credential` |
| 对齐决策 | `align / adapt / exclude`                                             |
| 实现差距 | `implemented / partial / missing`                                     |
| 证据等级 | `observed / schema-confirmed / writer-confirmed / inferred / unknown` |

机器可读 `StorageManifest` 还必须包含：`capturedAt`、平台、实际路径、类型、大小、文件数、`mtime`、主文件/伴生文件关系、活跃 writer、schema 摘要、逻辑 owner、事实源、重建来源及验证状态、敏感级别/权限、retention、cleanup selector/mechanism、停写要求、rollback、责任人、优先级和退出证据。

## 3. 固定根契约

| Root              | macOS                                | Windows                           | 允许内容                                                                 |
| ----------------- | ------------------------------------ | --------------------------------- | ------------------------------------------------------------------------ |
| `InstallRoot`     | `.app` bundle                        | `%LOCALAPPDATA%\\lime`            | 程序、updater、Squirrel 产物；业务数据禁止写入                           |
| `HostUserData`    | `~/Library/Application Support/lime` | `%APPDATA%\\lime`                 | 小型 Electron host 配置、窗口状态；Windows 不放 Agent DB、模型和 runtime |
| `HostSessionData` | 与现有 host profile 一起治理         | `<AppDataRoot>\\host-session`     | Chromium cookies、storage、network state 和 cache；不能整体标成可重建    |
| `AppDataRoot`     | `~/Library/Application Support/lime` | `%LOCALAPPDATA%\\LimeCloud\\lime` | 应用管理的机器数据、机器资产和产品旁路                                   |
| `AgentRoot`       | `<AppDataRoot>/app-server`           | `<AppDataRoot>\\app-server`       | 唯一 Agent runtime、ThreadStore、Agent observability owner               |
| `UserHome`        | `~/.lime`                            | `%USERPROFILE%\\.lime`            | `AGENTS.md`、用户 skills/rules/prompts 和可手工管理的文本配置            |
| `Workspace`       | `<project>/.lime`                    | `<project>\\.lime`                | 工作区 instruction 和项目级 metadata                                     |
| `E2ERoot`         | `<E2E userData>/app-server`          | 同左                              | 测试唯一根；ambient override 不得逃逸到真实目录                          |

macOS 允许 `HostUserData` 与 `AppDataRoot` 物理同根，但 owner 必须按子目录和 API 分开。Windows 固定 roaming host profile + local durable data；文档和代码统一使用小写 `%APPDATA%\\lime`。`LIME_AGENT_RUNTIME_ROOT` 只允许显式便携/测试入口，E2E 时必须被隔离策略约束，不能覆盖 E2E 根。

## 4. Codex -> Lime 顶层逐项账本

本表覆盖快照中 `~/.codex` 的 56 个顶层项。`任务` 指向第 7 节，不等于已经完成。

| ID  | `~/.codex` 实际项                 | 已观察职责/持久性                                                                                             | Lime 当前对应面                                                                     | 目标决策、治理状态与任务                                                                                   |
| --- | --------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| C01 | `.DS_Store`                       | OS debris / ephemeral                                                                                         | 两个 Lime 根均存在                                                                  | `exclude / dead`，不导入；X1                                                                               |
| C02 | `.agents/`                        | `skills` symlink alias / package discovery                                                                    | `$HOME/.agents/skills`、`~/.lime/skills`                                            | 外部 provider 只读发现，Lime-owned package 归 `UserHome/skills`；`adapt / current`；K1                     |
| C03 | `.app-server-state-reconciled-v1` | 一次性 marker / semantic-state                                                                                | 非模型 marker 不进入 current；旧 `.migration_completed`                             | 不迁移；`dead / forbidden-to-restore`，只进 cleanup manifest；X1                                           |
| C04 | `.codex-global-state.json.bak`    | host state backup                                                                                             | `Preferences`、`config*.backup`                                                     | Host owner，绑定版本和 TTL；`adapt / current`；H1                                                          |
| C05 | `.codex-global-state.json`        | host/global state                                                                                             | `Preferences`、`config.json`                                                        | 与 runtime config 拆开，归 `HostUserData`；`adapt / current`；H1                                           |
| C06 | `.personality_migration`          | migration marker                                                                                              | 非模型 marker 不进入 current；旧 `.migration_completed`                             | 不迁移；旧 marker 为 `dead`，按 exact path 清理；X1                                                        |
| C07 | `.tmp/`                           | plugin/marketplace staging / cache                                                                            | `agent-apps/staging`、`plugins/setup`                                               | 单一 staging root + TTL；`adapt / current-rebuildable`；K1/G1                                              |
| C08 | `AGENTS.md`                       | 用户全局 instruction / durable                                                                                | 代码目标 `~/.lime/AGENTS.md`，本机当前缺失                                          | `UserHome/AGENTS.md` 唯一 owner；`align / current`；U1                                                     |
| C09 | `ambient-suggestions/`            | feature output/cache                                                                                          | 无等价 owner                                                                        | 不造目录；`exclude / dead-unimplemented`；X1                                                               |
| C10 | `archived_sessions/`              | archived canonical rollout / durable                                                                          | ThreadStore 已原子 move；legacy overview 仍仅 DB flag                               | `<AgentRoot>/archived_sessions`，archive 为原子 move；`align / partial`；T1                                |
| C11 | `auth.json`                       | credential；本次仅检查 metadata，权限观察为 `0644`，需收紧                                                    | product DB api keys/provider、OAuth files                                           | 平台 credential store 或明确加密 owner；不进 history/log；权限/rotation/revoke 回归；`adapt / current`；C1 |
| C12 | `auth_back.json`                  | credential recovery backup                                                                                    | `machine_id_backups`、配置备份                                                      | 禁止无期限明文备份；加密、权限、TTL；`adapt / deprecated`；C1/M1                                           |
| C13 | `backups/`                        | recovery root，本次为空                                                                                       | 散落 `*.bak*`                                                                       | 非模型 backup 不迁移；模型 staging/backup 归 M1，其他 exact-path cleanup；M1/G1                            |
| C14 | `chrome-native-hosts-v2.json`     | connector/native-host config                                                                                  | `connectors`、browser connector config                                              | connector owner，版本化 current schema；`adapt / current`；K1                                              |
| C15 | `chrome-native-hosts.json`        | 旧/并行 connector config，owner 待证实                                                                        | `connectors`                                                                        | 不迁移；consumer scan 后冻结旧 reader 并 exact-path cleanup；`deprecated`；K1/X1                           |
| C16 | `computer-use/`                   | external integration config/assets                                                                            | `connectors`、embedded browser                                                      | `<AppDataRoot>/connectors/computer-use`；`adapt / product-current`；K1                                     |
| C17 | `config.json`                     | owner 未从目录本身证实                                                                                        | Lime `config.json`、`config.yaml`、DB settings                                      | 先拆 host state 与 runtime config，禁止按文件名合并；`adapt / partial`；C1/H1                              |
| C18 | `config.toml.before-*.bak`        | config backup                                                                                                 | `config.yaml.backup`                                                                | 不迁移；current config owner 只接新写入，backup 按 TTL 清理；`deprecated`；C1/X1                           |
| C19 | `config.toml`                     | 用户 runtime config / durable                                                                                 | `config.yaml`、`config.json`、DB settings                                           | 若 Lime 需要则新建 typed owner；旧格式不导入；`align / missing`；C1                                        |
| C20 | `generated_images/`               | generated artifact / durable output                                                                           | `projects`、runtime sidecar/media blobs                                             | project 或 `<AgentRoot>/artifacts/generated-images`，DB 只存 ref；`adapt / partial`；A1                    |
| C21 | `goals_1.sqlite-shm`              | `goals_1.sqlite` 共享内存 sidecar                                                                             | 无独立 GoalStore                                                                    | 与主库共生命周期，禁止单独清理；`align / missing`；S1                                                      |
| C22 | `goals_1.sqlite-wal`              | `goals_1.sqlite` WAL sidecar                                                                                  | 无独立 GoalStore                                                                    | 不迁移；停写后与主库作为一个 exact-path cleanup unit；`align / missing`；S1/G1                             |
| C23 | `goals_1.sqlite`                  | goal semantic state，活跃 writer                                                                              | objective/automation 表与文件，语义未对齐                                           | `<AgentRoot>/sqlite/goals.sqlite` 或明确 product owner；`adapt / partial`；S1                              |
| C24 | `history.jsonl`                   | 用户输入历史，当前约 12 MiB / durable convenience                                                             | `request_logs`、`logs` 语义不同                                                     | P0 决定产品是否需要；需要则 `UserHome/history.jsonl` 且有界，不参与 thread GC；D1                          |
| C25 | `history.json`                    | 与 `history.jsonl` 不同，owner 未证实                                                                         | 无精确对应                                                                          | 不猜语义；consumer 证实前 `exclude / compat-review`；D1/X1                                                 |
| C26 | `installation_id`                 | machine/install identity / semantic-state                                                                     | machine id/config/backups                                                           | `<AppDataRoot>/installation_id` 或 secure host owner；`adapt / current`；H1/C1                             |
| C27 | `instructions.md`                 | legacy/global instruction，本次为空                                                                           | 旧 `instructions.md`/`AGENT.md` discovery                                           | 不迁移；`UserHome/AGENTS.md` 只接新写入，旧入口退役；`deprecated -> dead`；U1/X1                           |
| C28 | `logs_2.sqlite-shm`               | log DB sidecar，活跃 writer                                                                                   | telemetry DB/log dirs                                                               | 与主库共生命周期；`adapt / partial`；O1                                                                    |
| C29 | `logs_2.sqlite-wal`               | log DB WAL，盘点时体量很大且活跃                                                                              | telemetry DB/log dirs                                                               | 停写/checkpoint 规则必须显式，禁止按目录删除；`adapt / partial`；O1/G1                                     |
| C30 | `logs_2.sqlite`                   | structured diagnostics，活跃 writer                                                                           | telemetry DB、traces、logs/request_logs                                             | `<AgentRoot>/observability/logs.sqlite`，按时间/bytes 分区；`adapt / partial`；O1                          |
| C31 | `log/`                            | text/process logs                                                                                             | `AgentRoot/observability/log/lime.log`；旧 `AppDataRoot/logs`                       | text log 已 `current`；旧 root 为 `deprecated/read-only source`；structured log 仍 `partial`；O1/G1        |
| C32 | `mcp-cache/`                      | 实际含 npm `_cacache/_npx/_logs`，属于启动依赖 cache                                                          | `mcp`、runtime/plugin caches                                                        | `<AppDataRoot>/cache/mcp`，按 server/generation/TTL；`adapt / partial`；K1/G1                              |
| C33 | `mcp-oauth-locks/`                | OAuth coordination lock / ephemeral                                                                           | `mcp/oauth`                                                                         | token 与 lock 分离，stale lock TTL；`adapt / partial`；K1/G1                                               |
| C34 | `memories_1.sqlite-shm`           | memory DB sidecar，活跃 writer                                                                                | 无统一 pipeline DB                                                                  | 与主库共生命周期；`adapt / missing`；S1                                                                    |
| C35 | `memories_1.sqlite-wal`           | memory DB WAL sidecar                                                                                         | 无统一 pipeline DB                                                                  | 与主库共生命周期；`adapt / missing`；S1                                                                    |
| C36 | `memories_1.sqlite`               | 含 raw memory/rollout summary，尚未证明可重建                                                                 | 多个 memory/memories 目录                                                           | 不迁移；冻结旧 reader 后整库清理，current memory owner只接新数据；`deprecated`；S1/X1                      |
| C37 | `memories/`                       | consolidated artifacts，本次目录为空                                                                          | `app-server/memories`、根 `memories`、`memory`、`~/.lime/memory`                    | 不迁移；`<AgentRoot>/memories` 只接新数据；旧目录 exact-path cleanup；S1/X1                                |
| C38 | `node_repl/`                      | tool runtime state/cache                                                                                      | `harness/tool-io`、tool offload                                                     | `<AgentRoot>/tmp/node-repl` 或 tool owner，重启可回收；`adapt / missing`；K1/G1                            |
| C39 | `pets/`                           | feature-specific，本次为空                                                                                    | 无 owner                                                                            | 不造目录；`exclude / dead-unimplemented`；X1                                                               |
| C40 | `plugins/`                        | installed runtime bundle + cache，内部需拆 owner                                                              | `plugins`、`agent-apps` 多根                                                        | `<AppDataRoot>/plugins` 包 owner，runtime cache 分离；`adapt / partial`；K1                                |
| C41 | `process_manager/`                | process coordination state                                                                                    | runtime sidecar/process state                                                       | `<AgentRoot>/process-manager`，短生命周期且可恢复；`adapt / partial`；K1/G1                                |
| C42 | `prompts/`                        | user prompt templates / durable                                                                               | skills/content materials 多处                                                       | `UserHome/prompts` 或 skill package owner；`align / partial`；U1/K1                                        |
| C43 | `rules/`                          | permission/tool policy / durable                                                                              | approval/config modules                                                             | `UserHome/rules`，归 `tool-runtime`；`align / partial`；U1/K1                                              |
| C44 | `session_index.jsonl`             | title/name index，不是 transcript                                                                             | title/list 混在 product/projection DB                                               | 可选 `<AgentRoot>/index/session_index.jsonl` fallback，不能成为第二 history；`adapt / missing`；T1         |
| C45 | `sessions/`                       | active rollout；包含日期 JSONL 和一个旧根 JSON                                                                | 新 Thread dated rollout 已 current；旧 event/session 不迁移                         | `<AgentRoot>/sessions/YYYY/MM/DD/rollout-*.jsonl` 唯一 canonical；`align / partial`；T1/X1                 |
| C46 | `shell_snapshots/`                | shell environment snapshots                                                                                   | harness/tool-io、sidecar                                                            | `<AgentRoot>/shell_snapshots`，按 session/TTL；`align / missing`；K1/G1                                    |
| C47 | `skills/`                         | installed user/system skills packages                                                                         | AppDataRoot skills、`~/.lime/skills`、provider roots                                | user/system/workspace 分层，每层一个 owner；`align / partial`；U1/K1                                       |
| C48 | `sqlite/`                         | 混合目录；`codex-dev.db` 活跃，其他 DB 副本需逐项审计                                                         | 根 DB、旧 runtime DB、retired DB                                                    | 禁止整目录判旧或清理；子项逐库登记；`adapt / partial`；S1/M1                                               |
| C49 | `state_5.sqlite-shm`              | state DB sidecar，活跃 writer                                                                                 | product/projection DB sidecars                                                      | 与主库共生命周期；`align / partial`；S1                                                                    |
| C50 | `state_5.sqlite-wal`              | state DB WAL sidecar，活跃 writer                                                                             | product/projection DB sidecars                                                      | 停写后随主库处理；`align / partial`；S1                                                                    |
| C51 | `state_5.sqlite`                  | thread/index/job/tool/config semantic state，含 preview、job JSON、dynamic tools、spawn edges 等，活跃 writer | `sqlite/state.sqlite` 已接管 Thread metadata/spawn edges；其他 product state 仍分散 | 保持 state 只含 metadata/graph；`last_sequence` 迁到 history projection state；`adapt / partial`；S1/T1    |
| C52 | `tmp/`                            | command wrapper symlink/temp，与 `.tmp` 不同                                                                  | temp/tool IO 多处                                                                   | OS/temp owner，退出时回收；`adapt / ephemeral`；G1                                                         |
| C53 | `update-check.json`               | update state                                                                                                  | Electron update/config                                                              | `HostUserData` update owner；`adapt / current`；H1                                                         |
| C54 | `vendor_imports/`                 | vendor/skill import cache                                                                                     | plugin/skill staging                                                                | 不作为旧 AppData 迁移源；旧 cache 按 TTL 清理；`deprecated`；K1/G1/X1                                      |
| C55 | `version.json`                    | version/update metadata                                                                                       | Electron/app version metadata                                                       | `HostUserData` update owner；`adapt / current`；H1                                                         |
| C56 | `visualizations/`                 | generated artifact，按年分区                                                                                  | projects、sidecar/artifacts                                                         | project 或 `<AgentRoot>/artifacts/visualizations/YYYY`；`adapt / partial`；A1                              |

### 4.1 必须拆开的二级项

| 实际项                                         | 不能合并的原因                           | 计划动作                                                               |
| ---------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------- |
| `sessions/YYYY/MM/DD/rollout-*.jsonl`          | current 日期化格式                       | T1 canonical contract                                                  |
| `sessions/rollout-*.json`                      | 根级旧格式                               | 不迁移；不得与 JSONL 拼接，进入 exact-path cleanup manifest            |
| `.agents/skills`                               | symlink alias                            | K1 只读 provider discovery，不复制 package                             |
| `.tmp/plugins/`                                | plugin checkout/staging                  | K1 staging job owner                                                   |
| `.tmp/bundled-marketplaces/`                   | bundled marketplace materialization      | K1 cache/版本 owner                                                    |
| `.tmp/marketplaces/`                           | user marketplace materialization         | K1 cache/版本 owner                                                    |
| `.tmp/plugins.sync.lock`、`.tmp/plugins.sha`   | coordination/hash marker                 | K1/G1 stale lock 与 generation 清理                                    |
| `plugins/.plugin-appserver/`                   | 二进制 runtime bundle                    | K1 package/version 原子安装与清理                                      |
| `plugins/cache/`                               | 可重建 cache                             | K1/G1 TTL/size GC                                                      |
| `mcp-cache/<server>/_cacache/_npx/_logs`       | 启动依赖与 npm cache/log                 | K1/G1 按 server 和 generation 清理                                     |
| `sqlite/codex-dev.db`                          | 活跃 desktop automation/inbox/catalog DB | S1 product-specific current；Lime 无同义能力时 `exclude`，禁止按目录删 |
| `sqlite/{state,logs,goals,memories}_*.sqlite*` | 与根级同名库并存，活跃性不同             | M1 逐库 fingerprint/writer/schema 对账                                 |
| `tmp/arg0`、`tmp/path`                         | command wrapper symlink/temp             | G1 进程生命周期回收                                                    |
| `generated_images/<uuid>/`                     | artifact job 单位                        | A1 按 job/session/project 管理                                         |
| `visualizations/YYYY/`                         | artifact 日期分区                        | A1 日期是 selector，不是无条件删除授权                                 |

## 5. Lime AppDataRoot 反向逐项账本

本表覆盖 `~/Library/Application Support/lime` 在 `captured_at=2026-07-19T10:00:28+08:00` 观察到的 58 个稳定/可追溯顶层项。Electron 测试在 `10:19:44+08:00` 另生成了 `DIPS-wal`、`SharedStorage-wal`、`SingletonCookie`、`SingletonLock`、`SingletonSocket` 五个 volatile host 项，已在 L59-L63 单列；这类项按模式审计，不能作为 Agent 数据。Windows 上 host profile 项应落 `HostUserData`/`HostSessionData`，机器资产和 Agent 数据应落 LocalAppData。

| ID  | Lime 实际项                                    | 目标 owner / 处理                                                  | 分类与任务                            |
| --- | ---------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------- |
| L01 | `.DS_Store`                                    | 不迁移                                                             | `dead`；X1                            |
| L02 | `.com.limecloud.lime.dev.ERcAJc`               | 未命名 JSON temp，先 consumer/fingerprint                          | `deprecated-review`；M1/X1            |
| L03 | `.com.limecloud.lime.dev.JncgCl`               | 同上，禁止仅按前缀批删                                             | `deprecated-review`；M1/X1            |
| L04 | `.com.limecloud.lime.dev.oLC4uN`               | 同上，禁止仅按前缀批删                                             | `deprecated-review`；M1/X1            |
| L05 | `.migration_completed`                         | 非模型 migration marker 与整库 manifest owner 均已退役            | `dead / forbidden-to-restore`；X1     |
| L06 | `Cache/`                                       | `HostSessionData` cache                                            | `current/rebuildable`；H1/G1          |
| L07 | `Code Cache/`                                  | `HostSessionData` code cache                                       | `current/rebuildable`；H1/G1          |
| L08 | `Cookies`                                      | `HostSessionData` credential-adjacent web state                    | `current/semantic-state`；H1          |
| L09 | `Cookies-journal`                              | 随 `Cookies` 主库                                                  | `current/sidecar`；H1                 |
| L10 | `DIPS`                                         | `HostSessionData` privacy state                                    | `current/semantic-state`；H1          |
| L11 | `DawnGraphiteCache/`                           | `HostSessionData` graphics cache                                   | `current/rebuildable`；H1/G1          |
| L12 | `DawnWebGPUCache/`                             | `HostSessionData` graphics cache                                   | `current/rebuildable`；H1/G1          |
| L13 | `DevToolsActivePort`                           | 仅进程生命周期                                                     | `ephemeral`；H1/G1                    |
| L14 | `GPUCache/`                                    | `HostSessionData` graphics cache                                   | `current/rebuildable`；H1/G1          |
| L15 | `Local State`                                  | host profile state                                                 | `current/semantic-state`；H1          |
| L16 | `Local Storage/`                               | host/webview semantic state，不得整体当 cache                      | `current/semantic-state`；H1          |
| L17 | `Network Persistent State`                     | host network state                                                 | `current/semantic-state`；H1          |
| L18 | `Partitions/`                                  | cookies/storage/profile/cache 混合，按 partition owner             | `current/mixed`；H1/G1                |
| L19 | `Preferences`                                  | small host config                                                  | `current/semantic-state`；H1          |
| L20 | `Session Storage/`                             | web session state                                                  | `current/semantic-state`；H1          |
| L21 | `Shared Dictionary/`                           | network/cache owner                                                | `current/rebuildable`；H1/G1          |
| L22 | `SharedStorage`                                | web storage state                                                  | `current/semantic-state`；H1          |
| L23 | `TransportSecurity`                            | host network security state                                        | `current/semantic-state`；H1          |
| L24 | `Trust Tokens`                                 | security state                                                     | `current/credential-adjacent`；H1/C1  |
| L25 | `Trust Tokens-journal`                         | 随 `Trust Tokens` 主库                                             | `current/sidecar`；H1/C1              |
| L26 | `agent-apps/`                                  | `<AppDataRoot>/agent-apps` package owner，state/cache/staging 拆开 | `current/partial`；K1                 |
| L27 | `app-server/`                                  | 唯一 `AgentRoot`，内部迁到 rollout/sqlite/observability/artifacts  | `current/partial`；R1/T1/S1/O1        |
| L28 | `app.db`                                       | 本次 0 byte，consumer guard 后禁止重建                             | `dead-candidate`；X1                  |
| L29 | `aster/`                                       | 退役 runtime 数据不迁移；无 reader 后按 exact path 清理            | `deprecated -> dead/forbidden`；X1    |
| L30 | `blob_storage/`                                | Chromium/web storage owner                                         | `current/semantic-state`；H1          |
| L31 | `channels/`                                    | product channel owner                                              | `current/product-specific`；A1        |
| L32 | `config.json`                                  | 与 host state/runtime config 拆 owner                              | `current/partial`；C1/H1              |
| L33 | `config.yaml`                                  | 不迁移；停止旧 reader/writer，current typed config 只接新写入      | `deprecated -> dead`；C1/X1           |
| L34 | `config.yaml.backup`                           | 旧 config backup，不迁移                                           | `deprecated -> dead`；C1/G1/X1        |
| L35 | `connect/`                                     | 旧 Connect registry cache；不迁移，current cache 只接新写入        | `deprecated/cleanup source`；G1/X1    |
| L36 | `connectors/`                                  | `<AppDataRoot>/connectors` 唯一 package/config owner               | `current/partial`；K1                 |
| L37 | `harness/`                                     | 只允许测试/evidence；生产 writer 必须迁出                          | `deprecated/test-only`；M1/X1         |
| L38 | `lime.db`                                      | 根级旧 Product DB；不迁移，停写后与 sidecar 精确清理               | `deprecated/cleanup source`；X1       |
| L39 | `lime.db-shm`                                  | 随根级旧 Product DB                                                | `deprecated/sidecar`；M1              |
| L40 | `lime.db-wal`                                  | 随根级旧 Product DB                                                | `deprecated/sidecar`；M1              |
| L41 | `lime.db.before-agent-session-reset-*.bak-shm` | 不完整备份 sidecar，先验证主文件/consumer                          | `deprecated-review`；M1/G1            |
| L42 | `lime.db.before-agent-session-reset-*.bak-wal` | 同上                                                               | `deprecated-review`；M1/G1            |
| L43 | `logs/`                                        | `<AgentRoot>/observability/log` bounded owner                      | `deprecated -> current target`；O1/M1 |
| L44 | `machine_id_backups/`                          | installation/credential owner，敏感且有 TTL                        | `compat/recovery`；C1/M1/G1           |
| L45 | `mcp/`                                         | `<AppDataRoot>/mcp`，credential/lock/cache 分开                    | `current/partial`；K1/C1              |
| L46 | `memories/`                                    | 不迁移；current memory owner 只接新写入，旧目录进入清理清单        | `deprecated -> dead`；S1/X1           |
| L47 | `memory/`                                      | session/agent memory 多目录旧 owner                                | `deprecated`；S1/M1                   |
| L48 | `models/`                                      | `<AppDataRoot>/models` machine asset；Windows 禁止 roaming         | `current`；A1/R1                      |
| L49 | `plugins/`                                     | `<AppDataRoot>/plugins` 唯一 installed-package owner               | `current/partial`；K1                 |
| L50 | `projects/`                                    | Lime product data，保持独立 owner                                  | `current/product-specific`；A1        |
| L51 | `request_logs/`                                | observability，不得作为 input/thread history                       | `deprecated -> current target`；O1/M1 |
| L52 | `runtime/`                                     | 第二 runtime root，不迁移                                          | `deprecated -> dead`；T1/G1/X1        |
| L53 | `sessions/`                                    | 旧 session repository不迁移；current rollout/artifact 只接新数据   | `deprecated -> dead`；T1/X1           |
| L54 | `site-adapters/`                               | connector/site integration owner                                   | `current/product-specific`；K1        |
| L55 | `skills/`                                      | managed/system skills；与 UserHome user skills 分层                | `current/partial`；K1/U1              |
| L56 | `startup/`                                     | startup state/marker，owner 与 TTL 待登记                          | `current-review`；H1/M1               |
| L57 | `webview_profiles/`                            | `HostSessionData` partition/profile owner                          | `current/semantic-state`；H1          |
| L58 | `workspaces/`                                  | product workspace index，不替代 `<project>/.lime`                  | `current/product-specific`；A1        |
| L59 | `DIPS-wal`                                     | Chromium DIPS sidecar，随 host session state                       | `current/volatile-sidecar`；H1/G1     |
| L60 | `SharedStorage-wal`                            | Chromium SharedStorage sidecar，随主状态处理                       | `current/volatile-sidecar`；H1/G1     |
| L61 | `SingletonCookie`                              | Electron 单实例协调文件                                            | `ephemeral`；H1/G1                    |
| L62 | `SingletonLock`                                | Electron 单实例锁，进程结束回收                                    | `ephemeral`；H1/G1                    |
| L63 | `SingletonSocket`                              | Electron 单实例 socket marker                                      | `ephemeral`；H1/G1                    |

## 6. Lime UserHome 反向逐项账本

`~/.lime` 目标只保留用户主动管理的文本配置和 packages。当前 12 个顶层项中只有 `skills/` 直接符合目标职责；其余生成数据必须迁回明确的 AppData/Product/Agent owner。

| ID  | `~/.lime` 实际项       | 目标 owner / 处理                                      | 分类与任务                         |
| --- | ---------------------- | ------------------------------------------------------ | ---------------------------------- |
| U01 | `.DS_Store`            | 不迁移                                                 | `dead`；X1                         |
| U02 | `agent-app-studio/`    | `<AppDataRoot>/agent-apps/studio` 或产品 project owner | `deprecated in UserHome`；A1/M1    |
| U03 | `aster/`               | 退役 runtime 不迁移；无 reader 后删除                  | `deprecated -> dead/forbidden`；X1 |
| U04 | `content-factory-app/` | `<AppDataRoot>/projects` 下产品 owner                  | `deprecated in UserHome`；A1/M1    |
| U05 | `downvideo/`           | 明确 download/project owner，cache 与成果分开          | `deprecated in UserHome`；A1/M1    |
| U06 | `materials/`           | `<AppDataRoot>/projects/materials` 或 workspace owner  | `deprecated in UserHome`；A1/M1    |
| U07 | `memory/`              | 不迁移；`<AgentRoot>/memories` 只接新写入              | `deprecated in UserHome`；S1/X1    |
| U08 | `skills/`              | `UserHome/skills` 唯一 Lime-owned user skills root     | `current`；U1/K1                   |
| U09 | `task-logs/`           | `<AgentRoot>/observability/tasks`，有界                | `deprecated in UserHome`；O1/M1/G1 |
| U10 | `tasks/`               | product job owner，语义状态与 artifacts 分开           | `deprecated in UserHome`；S1/A1/M1 |
| U11 | `terminal_blocks/`     | `<AgentRoot>/artifacts/terminal-blocks`，DB 只存 ref   | `deprecated in UserHome`；A1/M1    |
| U12 | `url_parse_tasks/`     | product task/artifact owner                            | `deprecated in UserHome`；S1/A1/M1 |

目标 `UserHome` 还应允许当前未出现的 `AGENTS.md`、`config.toml`、`rules/`、`prompts/`；这些是目标合同，不得伪装成本机已经对齐。

## 7. 执行任务与硬退出条件

| 任务 | 阶段    | 动作                                                                                                                            | 硬退出条件                                                                              |
| ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| R1   | P0.0    | 固定 `InstallRoot/HostUserData/HostSessionData/AppDataRoot/AgentRoot/UserHome/Workspace/E2ERoot`，resolver 只解析不创建、不迁移 | 同一启动只有一个 AgentRoot；Windows Agent 数据不在 roaming、UserHome、temp、InstallRoot |
| D1   | P0.0    | 逐项决定 `history.jsonl/history.json` 是否属于 Lime 产品                                                                        | 不把 request log 冒充 input history；不需要则明确 `exclude`                             |
| H1   | P0.1    | Electron host profile、sessionData、模型和 connector 分根                                                                       | Windows `%APPDATA%\\lime` 无模型、Agent DB/runtime；profile 项不被整体标 cache          |
| C1   | P0.1    | credential、installation identity、typed config 收口                                                                            | secrets 不进 rollout/log/普通 DB payload；权限/rotation/revoke/backup TTL 有回归        |
| U1   | P0.1    | UserHome 收口为用户文件；停止所有生成型 writer                                                                                  | `~/.lime` 无 DB、session、log、task、artifact、模型新写入                               |
| K1   | P0.1/P1 | skills/plugins/MCP/connectors/process/tool temp 分 owner                                                                        | 每个 package/cache/lock 有唯一 root 和清理单位；生产不读 harness                        |
| T1   | P1      | 日期化 canonical rollout、archive move、state rollout path                                                                      | 每 Thread 唯一 rollout；跨日 resume 不移动；projection 删除可 rebuild                   |
| S1   | P1      | state/goals/memories/read-model SQLite 分语义 owner                                                                             | 每个表声明事实源、重建证据、payload 上限和 GC 隔离；WAL/SHM 随主库                      |
| O1   | P1      | logs/traces/telemetry/request logs 合并为 bounded observability                                                                 | 按 thread/date/bytes/TTL dry-run；不被 thread/read 当 transcript                        |
| A1   | P1      | generated images、visualizations、tool output、models、projects 建 artifact/product owner                                       | DB 不存大 blob；按 session/project/job 精确定位目录                                     |
| M1   | P2      | 仅模型迁移器：只读 inventory、model manifest、staging copy、checksum/required-file verify、atomic publish                       | 非模型 source 拒绝；target 冲突 fail closed；macOS/Windows 中断恢复通过                 |
| G1   | P3      | session/date/derived/cache GC 与 quarantine                                                                                     | 默认 dry-run；active/pinned/changed 跳过；不递归删除整个日期目录                        |
| X1   | P3      | 删除 dead surface 并补回流守卫                                                                                                  | consumer scan、迁移/备份/确认完成；禁止再次创建旧 root/path                             |

### 7.1 P0 当前阻塞

以下任一未完成，P0 不得标记完成：

1. **已完成切片**：Electron、App Server client、App Server CLI 和 active smoke 已移除 Product DB destructive cleanup 参数/env；旧 flag fail closed，启动迁移固定保留 source。正常数据库启动也不再清空 `provider_pool_credentials` 或递归删除 `AppDataRoot/credentials`，零消费者 cleanup API 已移除；clear/drop/delete 只能等待独立 maintenance manifest、备份和确认。
2. **已被新决策取代（superseded）**：旧 Product DB 的 immutable backup/staging/cutover 曾用于非破坏迁移验证，但非模型数据现在明确不迁移。不得继续补旧库发布恢复；后续只保留 no-clobber/停写验证并转为 exact-path cleanup manifest。
3. **已被新决策取代（superseded）**：`storage-migration.v1 / database-path-v1` 只保留为历史 evidence，不再是 active migration owner。active manifest 仅有模型迁移 manifest 和非模型 cleanup manifest，禁止复用旧 DB 状态机迁 session/config/memory/Product DB。
4. **已完成切片**：Windows legacy candidate `%LOCALAPPDATA%\\lime` 已从递归迁移根剔除，只允许精确 DB/app-server 候选；InstallRoot 零写入回归通过。
5. **已完成切片**：E2E 缺少 `ELECTRON_E2E_USER_DATA_DIR` 时 fail closed，且 E2E root 优先于 ambient `LIME_AGENT_RUNTIME_ROOT`。
6. **已完成切片**：`VoiceModelHost` 使用 `AppDataRoot/models/voice`，`SystemUtilityHost` 使用唯一 connector owner 下的 `AppDataRoot/connectors/browser`；HostUserData machine-asset allowance 已从守卫移除。
7. **已完成切片**：Electron 显式解析 `HostSessionData/AppDataRoot/AgentRoot`；Windows `sessionData` 固定为 `<AppDataRoot>/host-session`，E2E 使用隔离根。App Server 通过 `APP_SERVER_APP_DATA_DIR` 接收 AppDataRoot，`StorageRoots` 与 `LocalAppDataSource` 同时接收 AppDataRoot/AgentRoot；diagnostics、support bundle、Soul style-pack 与 MCP OAuth 均只消费注入根，叶子 root bypass 基线为零。style-pack 同根安装幂等、异根 fail closed；MCP OAuth 未注入时 fail closed。`preferred_data_dir` 与 Product DB path resolver 只解析不创建目录。
8. **进行中**：旧 Product DB 整库复制、版本化 manifest、启动 cleanup owner，以及复制旧 managed project 并改写 workspace/session path 的 `migration_v6` 均已删除并补 dead-surface guard。cleanup manifest 语义、Windows 真机 runner 与责任开发者架构确认仍未完成。

### 7.1.1 2026-07-26 至 2026-07-27 根注入增量证据

- Electron path 与 sidecar 环境测试 `39/39`、Electron typecheck、`lime-core config::yaml` `17/17`、`lime-core app_paths` `17/17`、`lime-core --lib` `670/670`、App Server style-pack 注入/异根拒绝 `2/2`、storage/legacy guard `238/238` 已通过；此前 App Server storage roots `2/2`、diagnostics/support bundle `14/14`、MCP OAuth store `6/6` 及未注入 fail-closed `1/1` 继续有效。
- `cargo check -p app-server`、`npm run test:contracts`（App Server client `284 checks`）、scoped `git diff --check` 与 `npm run governance:legacy-report` 通过，legacy report 边界违规为 `0`。`npm run verify:gui-smoke -- --timeout-ms 180000` 通过真实 Electron/preload/IPC/App Server sidecar，App Server ready、Claw shell reload 与 Memory 设置页均成功，evidence `result=pass`。本证据不替代 Windows 真机 runner。
- 当前分类：`AppDataRoot -> AgentRoot -> StorageRoots/LocalAppDataSource` 为 `current`；模型控制面/模型文件迁移为唯一允许继续演进的 `compat`；旧整库迁移、manifest、cleanup 与 `migration_v6` 非模型 project/path 迁移均为 `dead / deleted / forbidden-to-restore`。

### 7.2 T1 dated rollout 当前切片

| 子项                                     | 当前状态                          | 证据 / 剩余差距                                                                                               |
| ---------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `sessions/YYYY/MM/DD/rollout-*.jsonl`    | `current / implemented for new`   | production 显式 AgentRoot；创建时本地日期；metadata UTC；typed change set；29/29 定向测试                     |
| `canonical_threads.rollout_path`         | `current metadata / implemented`  | relative path 唯一；restart 沿用；旧 NULL row 不迁移，旧库进入 cleanup manifest                               |
| raw `runtime/events/session_*.jsonl`     | `deprecated -> dead / frozen`     | 不迁移、不回填 canonical；生产 queued recovery reader已删除，diagnostics 切换后进入 exact-path cleanup        |
| SQLite `canonical_turns/items/applies`   | `current / rebuildable history`   | 已物理拆到 `sqlite/thread_history.sqlite`；同库 item -> turn FK；可从 rollout rebuild                         |
| SQLite `canonical_threads/spawn_edges`   | `current / metadata state`        | 已物理拆到 `sqlite/state.sqlite`；只保存 metadata/graph；跨库删除显式编排                                     |
| `runtime/projection_1.sqlite`            | `deprecated / rebuildable`        | queued row 必须 join current canonical Thread + production rollout；stale legacy row 不恢复；待 consumer 退役 |
| `archived_sessions/` + archive/unarchive | `current / v2 + ThreadStore`      | canonical 原子 move + path patch + crash/retry；`thread/archive` / `thread/unarchive` 已接入；Gate B 待补     |
| rollout rebuild / cold replay            | `current / canonical + projected` | empty canonical/projected 同事务重建；legacy projection no-clobber；真实 GUI Gate B 待补                      |
| session/date dry-run GC                  | `missing`                         | 属于 G1；必须从 state path 枚举，不得按日期目录递归删除                                                       |

本切片已把新写入事实源、ThreadStore archive、canonical 与 AgentSession projected cold rebuild 接到 dated rollout，并由 Codex v2 `thread/archive` / `thread/unarchive` 与 Renderer consumer 接管；state/history/projection 三库物理 owner 已拆分并补 inventory 守卫。T1 仍不能标记完成：下一刀是把 `last_sequence` 迁入 history projection state、退役 deprecated projection consumer，并实现 session/date GC dry-run 与 Windows Gate B。

### 7.3.1 2026-07-20 误导入撤回与启动证据

- Content Studio 跨产品导入已完整撤回：删除 8 个错误自定义 Provider、9 条关联 API Key 和错误 migration marker；清理基线恢复为 93 个系统 Provider、0 个自定义 Provider、0 条 API Key，target session 保持 0。之后用户手动添加的 Provider/Key 是 current 用户状态，不属于迁移。
- 清理前 SQLite 备份位于 `/tmp/lime-model-control-target-before-cleanup-20260720T082900.db`，权限 `0600`、SHA-256 `1cac22b0857dd75b9a1ebb920187a8b7d3faf6c2a39570f83b67b2bc228c0425`、`integrity_check=ok`。Content Studio source 未修改；系统 Provider、session、MCP、UI state、模型偏好和 `model_route_generation` 未删除。Kiro 明确排除，当前没有需要自动迁移的原 Lime Provider/Key 数据。
- raw EventLog queued recovery 已从 production 删除；真实 1047 个旧 event 文件不再阻塞启动。旧 projection queued row 必须由 current canonical Thread/rollout 锚定；真实首 JSONL 响应 `3198ms`，Electron/Bridge/Provider list 链路通过。
- 清理后复验：目标 SQLite `integrity_check=ok`、清理时 `modelProvider/list=93/0/0`、session 0、错误 marker 0，运行态只有 1 个 current sidecar。此前 `101/8/9` 只是已撤回的错误导入历史，不得作为完成证据。EventLog 回归 19/19、治理报告零违规、contracts 与 `verify:gui-smoke` 通过；pending recovery 的当前复跑仍被共享 v2 协议热区 `processor/dispatch/v2_ingress.rs:38` 缺少 `ThreadGoalSet/Get/Clear` 分支阻塞，不属于本存储写集。
- M1 仍未完成：已下载模型的 manifest/staging/checksum/atomic publish、Windows 中断恢复与备份 TTL/cleanup maintenance 仍是退出条件。

### 7.3 已确认的旁路 writer

| Writer                                            | 当前写入                                                                                        | 目标 owner / 阶段                                                                 |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `agent/src/tool_io_offload.rs`                    | 无产品 caller；生产默认 writer 已删除，只有显式 `LIME_TOOL_IO_OFFLOAD_DIR` 才运行               | `dead candidate / frozen`；不得迁目录续命，确认后删除模块/导出                    |
| `agent/src/durable_memory_fs.rs`                  | 仅 crate 导出、无产品 caller；生产默认 writer 已删除，只有显式 `LIME_DURABLE_MEMORY_DIR` 才运行 | `dead candidate / frozen`；不得迁目录续命，确认后删除模块/导出                    |
| `core/src/plugin/loader.rs`                       | 无产品 `load/load_all` 调用；默认 Manager 已 disabled 且无路径                                  | `dead / frozen`；current owner 是 App Server `PluginDataSource`                   |
| `core/src/session_files/storage.rs`               | `AgentRoot/artifacts/sessions/<id>`；旧 `AppDataRoot/sessions` 冻结                             | `current`；旧 source 不迁移，P2/P3 按 exact path 清理                             |
| `core/src/logger.rs`                              | `AgentRoot/observability/log/lime.log`；旧 `AppDataRoot/logs` 冻结                              | `current` text log；旧 source 不迁移，P3 按 date/bytes 清理                       |
| `infra/src/telemetry/logger.rs`                   | 无产品构造者；raw JSONL 默认 writer 已删除，构造必须显式传入 `log_dir`                          | `dead candidate / frozen`；current owner 是 `TelemetryStore`                      |
| `app-server/src/local_data_source/connect.rs`     | `AgentRoot/cache/connect/registry.json`；旧 `AppDataRoot/connect` 冻结                          | `current/rebuildable cache`；旧 source 不迁移，P3 TTL/bytes GC                    |
| `mcp/src/oauth_store.rs`                          | `AppDataRoot/mcp/oauth/*.json`                                                                  | OS secure credential owner；lock/token 分离；P1                                   |
| `services/src/material_service.rs`                | `UserHome/materials`                                                                            | AppDataRoot project/material owner；旧数据不迁移，P0 freeze 后清理                |
| `app-server/src/plugin_packages/paths.rs`         | `<AgentRoot>/plugins`，由 `LocalAppDataSource` 派生并注入                                       | `current/injected owner`；P0 路径收口完成，P1 package/version retention           |
| `app-server/src/runtime/plugin_task_runtime.rs`   | cloud package 从 `<AgentRoot>/plugins/packages/<hash>` 解析；local folder 使用显式 `sourceUri`  | `current/fail closed`；P0 root 收口完成，P1 signature/worker cache GC             |
| `app-server/src/runtime/soul/style_pack_paths.rs` | 自行发现 `AppDataRoot/soul/style-packs`                                                         | 注入 product asset root；P1                                                       |
| `electron/voiceModelHost.ts`                      | `AppDataRoot/models/voice`                                                                      | `current`；P0 路径切片完成，P1 retention/下载 staging                             |
| `electron/systemUtilityHost.ts`                   | `AppDataRoot/connectors/browser`                                                                | `current`；P0 路径切片完成，P1 connector package/清理                             |
| Renderer `localStorage`                           | session/workspace/provider/topic/plugin run/diagnostics metadata                                | 只保留 theme/layout/ephemeral UI；domain truth 迁 App Server state/read model；P1 |

`services/src/backup_service.rs` 和 `context_memory_service.rs` 当前无生产消费者，分别写 `UserHome/backups`、`UserHome/memory`；它们是 `dead candidate`，不得为了目录对齐迁入 current 主链。`SystemUtilityHost` 已停止从 HostUserData 派生 connector 路径，后续 connector package/安装清理仍归 AppDataRoot owner。

语音模型路径当前分类为 `current / single owner`：Electron `VoiceModelHost` 是 `AppDataRoot/models/voice/<model-id>` 的唯一 writer；Renderer 在调用 `voiceModel/testTranscribeFile` 前读取 install state，协议以必填 `install_dir` 把精确路径传给 App Server；App Server 与 `lime-services` 只校验并消费显式绝对路径。`default_voice_model_install_dir()`、default credential 路径回读和所有平台 root fallback 已删除并由 storage guard 阻止回流。

E2E/custom root 当前分类为 `current / isolated override`：`preferred_agent_root()` 可以返回显式 override，但“是否平台默认根”只允许由不读取 override 的 `platform_default_agent_root()` 判定。只有真实平台默认 AgentRoot 能扩展全局 legacy source；custom/portable/E2E 根只扫描同一受控 root 的 parent，禁止触达真实 AppData。纯函数回归覆盖 override isolation，真实 Electron Gate B-F 已证明临时 AgentRoot 启动不会再因真实 Product DB 的 WAL 失败。

P0 负向守卫必须阻止：Agent durable writer 直接调用 `preferred_data_dir()`、`resolve_sessions_dir()`、`resolve_logs_dir()`、`resolve_request_logs_dir()` 或 `best_effort_*()`；`~/.lime` allowlist 之外的生产写入；HostUserData 下的模型/connector/Agent 数据；同一 projection DB 同时承载 `canonical_*` 与 `projected_*`；archive 只改 flag；delete 缺 owned artifact/trace/telemetry 枚举；renderer `localStorage` 保存 durable domain truth。

## 8. 模型迁移与日期清理协议

### 8.1 多模型控制面对齐（Grok + OpenCode）

多模型控制面与已下载模型文件迁移是两个 owner：控制面归 App Server + `model-provider`，只保存用户选择和小型语义状态；模型文件迁移归 `<AppDataRoot>/models`；远端 catalog snapshot 是可重建 cache，不进入 Product DB 或 ThreadStore。Grok 提供 catalog manager/switch/refresh/circuit breaker 参考，OpenCode 提供 Provider/Model/Credential/capability/lowering/retry 分层参考。

| ID   | `grok-build` 事实                                                        | Lime 当前对应面                                                                  | 差距与对齐动作                                                                                                           | 存储/清理合同                                                                        |
| ---- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| GM1  | `ModelsManager` 是 catalog、当前选择和刷新唯一 owner                     | `ModelRegistryService`、`model/list`、Renderer provider model list 多层状态      | 收口一个 App Server model control owner；Renderer/Electron 只投影，不各自维护可选模型 truth                              | catalog 内存态可重建；用户默认/当前 Thread model 是小型 semantic state               |
| GM2  | 远端 catalog + bundled defaults + config overrides                       | Lime 有 Provider API catalog、canonical registry 和大量本地推断                  | 明确 precedence，并让 server metadata 优先；本地推断只能标 `inferred`，未知 capability fail closed                       | source/version/origin 写入 cache metadata，不写 Thread history                       |
| GM3  | `models_cache.json`：版本、auth method、origin、ETag、5 分钟 TTL、原子写 | Lime 将 `provider_models_fetch_cache:*` JSON 放 Product DB `settings`，TTL 10 天 | 迁出 Product DB，改为 `<AgentRoot>/cache/models/<scope-hash>.json`；key 必含 provider/tenant/auth kind/origin/protocol   | 整个 cache 文件/目录可按 TTL/bytes 删除；不迁移旧 DB cache，不需要逐行维护主库       |
| GM4  | ETag 变化触发刷新；失败保留现有 catalog；并发 refresh 合并               | Lime 有 fetch cache/generation，但无统一 ETag refresh owner                      | 增加 refresh generation/ETag 和 single-flight；失败不得清空仍有效 catalog                                                | cache 更新用 temp + fsync + atomic rename；损坏/版本不符直接丢弃                     |
| GM5  | `allowed/hidden/disabled` 与 auth visibility；零匹配阻断 prompt          | Lime 有 provider enabled models/readiness，但选择规则散落                        | 建统一 selectable predicate；credential/entitlement/provider readiness/capability 任一未知均不可进入可选列表             | 规则属于 typed config/managed policy；catalog cache 不反向覆盖用户 policy            |
| GM6  | model capability metadata 驱动 reasoning、image、context window          | Lime `ModelInfo` 已有 capability/modalities/context/reasoning 字段               | 禁止 UI/route 以 model 名称猜能力；同一 capability snapshot 供 picker、route lowering 和 runtime preflight 消费          | capability snapshot 随 catalog cache，可重建；Thread 只记录当次解析后的 model ref    |
| GM7  | switch 校验 model 是否可选，并检查 agent/harness compatibility           | Lime 主要经 `agentSession/update` 改 model metadata，缺独立 switch 生命周期      | 新增 current `thread/model/set` 或等价短领域方法；切换先校验 catalog/readiness/capability/active-turn policy，再原子更新 | 只持久化 Thread 默认 model ref 和必要 generation；不得复制整份 catalog 到 session DB |
| GM8  | catalog refresh 后当前模型失效则可解释地重选 default                     | Lime 缺少统一 current/default reselection contract                               | 定义 `current valid -> preserve`、`missing/forbidden -> reselect`、`no selectable -> fail closed`，并通知 GUI            | 重选是 metadata patch + notification；不修改既有 rollout 历史                        |
| GM9  | 独立 `RetryPolicy` 分类 retry/auth-refresh/terminal                      | Lime request retry 有 5xx/backoff，但错误分类仍散落                              | 将 status/provider failure 分类收口 `model-provider`；同一请求只在明确 retryable 且未产生用户可见 item 时重试            | retry state 默认内存/有界 telemetry，不进 ThreadStore                                |
| GM10 | sliding-window CircuitBreaker：closed/open/half-open、min samples、probe | Lime 已有 `current_client/health.rs` 内存熔断器，但不是共享公共边界              | 提升为 `model-provider` current 公共组件，按 credential/provider route key 隔离；暴露 `retry_after` 和有界观测           | breaker 运行态 ephemeral；只写聚合 telemetry，应用重启不需要迁移                     |
| OM1  | OpenCode `ProviderRecord + Map<ModelID, ModelInfo>`                      | Lime `ProviderInfo`、`ModelInfo` 和 Renderer registry 仍有多层投影               | App Server 产出统一 `Provider/Model/Capability/Readiness/Policy` snapshot；Renderer 只读                                | snapshot 可重建；SQLite 只保存 Provider 配置和用户选择                               |
| OM2  | `Credential` 独立于 catalog                                              | Lime App Server SQLite `api_keys` 已是独立 owner                                 | 保持 `modelProviderKey/*` 唯一写边界；catalog cache、rollout、日志不得复制 credential                                    | ciphertext 只在 credential owner；support bundle/trace 脱敏                          |
| OM3  | Provider API/request 与 Model API/request 可合并 lowering                | Lime provider type、host、model capability 与 route lowering 分散                | `model-provider` 先解析 Provider base，再按 Model capability/variant lowering；禁止 Renderer 以模型名猜协议             | lowering 结果属于当次 runtime request，不把完整 request 写入 ThreadStore             |
| OM4  | Provider 可用性统一由 disabled、credential/integration、policy 决定      | Lime enabled、key readiness、route policy 分散                                   | selectable predicate 一次计算 availability/reason/recovery action，供 picker、switch、route 共用                        | readiness 默认内存/小型 semantic state，不进入 rollout                               |
| OM5  | Model 自带 capability、variant、cost、limit、disabled                    | Lime 已有部分 capability/modalities/context/reasoning 字段                       | 补齐统一 schema 与 generation；未知 capability fail closed，不从名称推断成 durable truth                               | capability 随 catalog cache，可按 scope/TTL 整体清理                                 |
| OM6  | retry 只对明确 transient error 有限执行                                  | Lime retry/error 分类散落                                                        | 仅在未产生用户可见 Item 时按 provider error taxonomy 有限重试；auth/quota/policy/invalid request 直接 terminal            | retry/circuit 状态 ephemeral，只写有界聚合 telemetry                                 |

多模型完成门禁：`model/list`、Thread model switch、route resolver 与 provider request 使用同一 catalog generation/capability snapshot；catalog cache 删除后可重建；旧 Product DB `provider_models_fetch_cache:*` 无 reader/writer；无可选模型、能力未知、credential 未就绪时 fail closed；retry 与 breaker 不制造重复 Turn/Item。

### 8.2 已下载模型迁移 Manifest 最小字段

```text
migrationId + schemaVersion + state
source model root + model id/version + relative files
target model path + required-file set
per-file bytes/checksum + aggregate digest
startedAt/verifiedAt/publishedAt
cleanupAuthorizedAt
```

状态机固定为：

```text
detected -> planned -> staged -> verified -> published
         -> source_cleanup_pending -> completed
```

### 8.3 日期化 rollout 与 GC

1. 日期取 Thread 创建时的本地日期；metadata 保存 UTC 时间。跨日续聊追加原文件，不移动日期目录。
2. 日期只是 selector。执行前从 state index 解析 `threadId -> rollout path -> artifact refs`，不能使用文件 `mtime` 或 `rm -rf YYYY/MM/DD` 作为主算法。
3. archived 默认不参与 active date GC，必须显式 `includeArchived`；canonical history 默认不自动 TTL。
4. dry-run manifest 记录相对路径、createdAt、active/archive、writer/pin 状态、bytes、checksum、artifact refs 和 hold reason。
5. 执行时重新校验 manifest digest、fingerprint 和 path containment；macOS/Windows 不跟随越出 AgentRoot 的 symlink，Windows 额外拒绝 junction/reparse escape。
6. 先锁/flush writer，写 deletion intent，再移动到同盘 quarantine，最后事务更新 metadata。失败必须可重试，空日期目录只在内容处理完后回收。
7. 自动 GC 只作用于 derived projection、logs/traces/telemetry、cache 和失败下载；goal、memory、credential、projects 与 canonical rollout 不得被连带删除。

## 9. 验证与完成定义

当前基线（2026-07-19）：Electron storage filter 31/31、`lime-core app_paths` 34/34、`lime-core` 全量 unit 688/688、core plugin 98/98、`lime-agent` 274/274（unit）；durable memory root 4/4、tool I/O offload 6/6、infra telemetry 49/49、session files 11/11、logger 10/10、manifest 2/2、cleanup 5/5、startup migration 7/7、API key migration 2/2、App Server 参数 16/16、session/log root 注入 2/2、support bundle 3/3、log helper 1/1、Connect registry cache 注入 1/1、RelayRegistry cache round-trip 1/1、Plugin resolver 9/9、Plugin worker turn 7/7、storage-root guard 12/12、dated canonical rollout 29/29、projection rebuild status 1/1、`lime-services` default/`local-whisper`/`local-sensevoice` 显式模型路径测试各 2/2、voice install-dir Rust unit 1/1、公共 JSON-RPC 2/2、Renderer gateway 11/11、App Server client 63/63、protocol codegen 731 类型零漂移；文档/代码 scoped Prettier、scoped `git diff --check`、Rustfmt、`npm run docs:boundary`、`npm run governance:legacy-report`（边界违规 0）与 `npm run governance:scripts` 通过。正常启动已证明 deprecated credential 行保留，且不再含旧凭证目录递归删除入口；Product DB migration 真实 Electron Gate B 已证明 copied manifest、source retained、restart readback 和只读目录 fail closed；session files Gate B 已证明 `AgentRoot/artifacts/sessions`、完整 current JSON-RPC 方法链、零旧命令和零 console error；Developer diagnostics Gate B-F 已证明 text log diagnostics 经真实 Electron current App Server 链路命中；E2E/custom root Gate B-F 已证明临时 AgentRoot 不再扩展真实 AppData legacy source，21/21 断言、33 次 App Server IPC、零 legacy/mock/console/page/invoke error。text log writer/read/clear/support bundle 的 Rust 注入链与 Electron diagnostics 证据已通过；canonical rollout 已覆盖 local unit 的日期路径、restart、跨日、archive/unarchive、overview bridge、canonical + AgentSession projection cold rebuild、legacy projection no-clobber、content tamper 与 commit retry，但 Codex v2 archive/unarchive/date dry-run Gate B、structured log SQLite 分区预算、HostSessionData、其余 4 个旁路与 Windows runner 仍未完成。

当前扩大门禁中，`npm run test:contracts` 与 `npm run verify:gui-smoke` 已于 2026-07-20 复跑通过；contracts 为 protocol 740 类型零漂移、App Server client 299 checks，GUI smoke 当前证据为 `.lime/qc/project-gates/standalone-shell-01-20260719213407-25882/shell-01-electron-smoke/summary.json`。贴存储写集的 EventLog 19/19、模型迁移器 3/3、治理报告和 SQLite/真实 JSON-RPC 行级对账均通过；pending recovery 与扩大 `test:rust:related` 当前复跑被共享 v2 协议热区 `processor/dispatch/v2_ingress.rs:38` 缺少 `ThreadGoalSet/Get/Clear` 分支阻塞，因此不能把共享 Rust 全量状态标记为全绿。历史失败 smoke `.lime/qc/project-gates/standalone-shell-01-20260719073842-4780/shell-01-electron-smoke/summary.json` 只保留为根因证据，未恢复旧 `agentSession/update` 命令，也未绕过 WAL 安全边界。

| 断言                   | 最小证据                                                                                                                     |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| inventory 完整         | 56 个 Codex 顶层项、58 个 Lime 稳定 AppData 项、5 个 volatile host 项、12 个 UserHome 项均有 ledger row；新增项使 guard 失败 |
| 非模型零迁移           | session/event/DB/cache/log importer 为零；旧 source 只出现在 cleanup inventory/guard                                         |
| 模型 source 真正只读   | inventory/copy 前后模型 source fingerprint 相同；无 source marker 写入                                                       |
| 单一根                 | macOS/Windows/override/E2E 纯函数测试 + 进程写入快照                                                                         |
| InstallRoot 零业务写入 | Windows Squirrel 安装前后快照/ProcMon，仅 updater/app 产物变化                                                               |
| E2E 不逃逸             | 同时设置 E2E root 和 ambient override，所有 writer 仍位于 E2ERoot                                                            |
| canonical 唯一         | state `rollout_path` 唯一；cold/live/replay/restart identity 一致                                                            |
| projection 可丢弃      | 删除临时 fixture projection 后可从 rollout 全量 rebuild                                                                      |
| 日期 GC 不误删         | active、pinned、archived、跨日 resume、symlink/junction fixture                                                              |
| DB/WAL 清理完整        | 停写后主库/WAL/SHM exact-path manifest；无 import/checkpoint/VACUUM；确认后作为一个 unit 清理                                |
| 无旁路 writer          | 负向扫描 durable writer 中的 `dirs::*`、`preferred_data_dir`、`best_effort_*`、直接 `userData` 拼接                          |
| 真实产品闭环           | Gate B 新建、续聊、重启、archive/unarchive、rebuild、date dry-run 后 GUI history identity 不变                               |

完成定义：所有账本项都有唯一 owner 或明确 `exclude`；`current/compat/deprecated/dead` 与实际 writer 一致；旧 source 无新读写；只有模型迁移器存在；其余旧数据有精确清理证据；macOS 与 Windows 真机证据齐全。未完成 Windows runner、模型迁移恢复演练或 Gate B 时，计划只能保持 `in_progress`。
