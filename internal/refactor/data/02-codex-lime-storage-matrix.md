# Codex `~/.codex` 与 Lime 存储语义分组矩阵

状态：`in_progress`

日期：2026-07-19

对照范围只包含实际 Codex `~/.codex`。本文件按语义分组，不是逐路径 inventory；56 个 Codex 顶层项、58 个 Lime AppData 项和 12 个 Lime UserHome 项的精确账本见 [03-one-to-one-storage-alignment-plan.md](03-one-to-one-storage-alignment-plan.md)，发生冲突时以实际账本为准。

迁移白名单是模型控制面语义状态与 `models/`。旧 Lime 会话、event、projection、Product DB 其他表、日志、cache、memory、config、plugin/skill 状态和 retired runtime 均不做内容迁移；它们只保留到 current owner 验证和 exact-path cleanup。模型控制面只读迁移 provider/key ciphertext/UI state/model preference/active tab，不复制整库。多模型控制面另以 `/Users/coso/Documents/dev/rust/grok-build` 为 primary reference，不改变本矩阵的 Codex storage 基线。

## 1. 分类口径

本表“当前分类”是便于阅读的历史简写。执行时必须把治理状态、持久性、对齐决策、实现差距和证据等级分列，不能把 `wrong owner`、`missing` 或 `rebuildable` 当作治理状态。

| 分类          | 含义                                               |
| ------------- | -------------------------------------------------- |
| `current`     | Lime 后续唯一可继续演进的 owner                    |
| `compat`      | 仅模型控制面/模型文件迁移 adapter；source 只读、按白名单校验    |
| `deprecated`  | 停止新写入，只允许盘点和清理                       |
| `dead`        | 无生产入口或已被替代，删除并补回流守卫             |
| `missing`     | Lime 尚未实现；不能用“有相似目录”冒充完成          |
| `rebuildable` | 可从 canonical 文件/其他事实源重建，不是用户事实源 |

本矩阵的关键原则：Codex rollout 文件是对话事实源；SQLite 只保存 metadata、索引、目标状态或可重建 projection；Lime 不得把同一份 transcript 再写入多个数据库。

## 2. Durable Thread/Turn/Item

|   # | Codex `~/.codex`                                               | Codex 语义                                                                                      | Lime 当前对应物                                                                                                                  | 当前分类                                      | Lime 目标                                                                  | 清理单位/验收                                                   |
| --: | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------- |
|   1 | `sessions/YYYY/MM/DD/rollout-YYYY-MM-DDThh-mm-ss-<uuid>.jsonl` | 每个 Thread/Session 一个 append-only canonical rollout 文件；日期只做分区                       | 新 Thread 已写 `<AgentRoot>/sessions/YYYY/MM/DD/rollout-*.jsonl`；旧 event/session JSONL 不迁移                                  | `current / implemented`                       | 保持 dated rollout 为唯一 durable truth                                    | 按 rollout 文件定位；日期只作 manifest selector，不递归盲删     |
|   2 | `archived_sessions/rollout-*.jsonl`                            | archive 是 rollout 文件移动，不复制 transcript                                                  | `<AgentRoot>/archived_sessions/` 已由 v2 archive/unarchive 原子 move 接管                                                        | `current / implemented`                       | 保持文件 move + state path patch                                           | active/archive 两处唯一定位，macOS Gate B 已通过                |
|   3 | `state_5.sqlite`                                               | `threads`/spawn edges 保存 rollout path、archive、preview、cwd、provider、recency 等 metadata   | `<AgentRoot>/sqlite/state.sqlite` 只含 `canonical_threads` 与 `canonical_thread_spawn_edges`；旧 DB 不迁移                       | `current / partial alignment`                 | 后续将表名/列继续压到 Codex metadata 语义；`last_sequence` 迁出 state      | DB 行含 canonical path；文件动作成功后再处理 metadata           |
|   4 | `thread_history_1.sqlite`                                      | Codex 源码独立保存 `thread_turns`、`thread_items` 与 projection state；可从 rollout 重建        | `<AgentRoot>/sqlite/thread_history.sqlite` 独立保存 canonical turns/items/applies；`runtime/projection_1.sqlite` 仍是 deprecated | `current history` + `deprecated projection`   | history 只存 coalesced snapshot/watermark；退役旧 projected read model     | 物理 inventory 防落错库；projection 为空可从 canonical 独立重建 |
|   5 | `session_index.jsonl`                                          | append-only title/name 索引；DB current 后只能 fallback                                         | Lime 无独立同语义 index；标题和列表混在 projection/product DB                                                                    | `missing`                                     | 可选 `app-server/index/session_index.jsonl`，仅作为 rebuild/fallback       | 不导入旧 index；不能再成为第二 history owner                    |
|   6 | `history.jsonl`                                                | 用户输入便利历史，不是 Thread rollout；本机已达到 MiB 级                                        | Lime `request_logs/`、`logs/` 是请求/诊断旁路，没有稳定 input-history owner                                                      | `missing`                                     | 若产品需要，新增 `user/history.jsonl` 并制定 Lime 自己的限额；否则明确排除 | 单独按 bytes/条数清理，不参与 thread 删除                       |
|   7 | `goals_1.sqlite`                                               | Goal 状态是用户可见语义状态，不是可随意清理的 cache                                             | Lime `objective`/automation 相关表和文件，未证明与 Codex GoalStore 等价                                                          | `missing`/`wrong owner`                       | `app-server/sqlite/goals.sqlite` 或明确 product owner                      | goal 删除需走 API；不能随 projection GC 删除                    |
|   8 | `memories_1.sqlite`                                            | 本机含 raw memory/rollout summary，尚无“可重建”实证                                             | `app-server/memories/`、根 `memory/`、`harness/memories/` 多处并存                                                               | `deprecated`/`wrong owner`                    | current memory owner 只接新数据；旧 memory DB/目录不迁移                   | 停写和无 reader 后按 DB + sidecar / 目录精确清理                |
|   9 | `memories/`                                                    | consolidated memory artifacts（`MEMORY.md`、summary、rollout summaries）是用户可见 memory owner | Lime `app-server/memories`、根 `memory`、`harness/memories`                                                                      | `current` candidate + `deprecated` duplicates | 只保留 `app-server/memories/`；根/`harness` 不迁移，验证无 reader 后清理   | 按 memory artifact 文件清理；不能连带删除 current memory        |

## 3. 配置、身份与用户指令

|   # | Codex `~/.codex`                                                                                       | Codex 语义                                          | Lime 当前对应物                                                 | 当前分类                                     | Lime 目标                                                            | 清理单位/验收                                                    |
| --: | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------- |
|  10 | `config.toml`                                                                                          | 用户配置事实源，profiles/SQLite home/feature policy | `config.yaml`、`config.json`、product DB settings               | `current` 但重复格式                         | 选定一个 typed config owner，只接新写入；旧格式不导入                | fresh config round-trip；不再双写 YAML/JSON                      |
|  11 | `instructions.md`                                                                                      | 用户全局 instruction                                | `~/.lime/AGENTS.md`、旧 `AGENT.md`/`instructions.md`            | `deprecated`                                 | `~/.lime/AGENTS.md` 只接新写入；旧入口停止发现                       | 文件级 exact-path cleanup；应用清理不删除 current 文件           |
|  12 | `AGENTS.md`                                                                                            | 用户全局/项目 instruction discovery                 | 用户 home 与 workspace `.lime` 的 `AGENTS*.md`                  | `current` candidate                          | 同层 override 替代 shared；不写入 session DB                         | 按文件/项目清理；source precedence 有测试                        |
|  13 | `auth.json`、`auth_back.json`                                                                          | 本地 auth credential 与恢复备份                     | Lime product DB `api_keys/providers`、配置文件、OAuth 文件      | `current` security owner 未收口              | 平台 credential store 或明确加密 credential owner；不进 rollout/日志 | credential revoke/rotation；support bundle 不泄露；backup 有 TTL |
|  14 | `rules/default.rules`                                                                                  | shell/permission policy                             | Lime permission/approval/config 模块；无一一对应文件            | `partial`                                    | 归 `tool-runtime`/permission owner；不放 ThreadStore                 | policy 文件变更不触发 history migration                          |
|  15 | `prompts/`                                                                                             | 用户 prompt 模板                                    | Lime skills/content factory/project materials 多处              | `partial`/product-specific                   | 归 skills/prompt owner；不混入 rollout/state DB                      | 按模板文件/skill package 清理                                    |
|  16 | `config.json`、`.codex-global-state.json`、`.personality_migration`、`.app-server-state-reconciled-v1` | 全局状态/一次性 marker                              | Lime `config.json`、`config.yaml`、旧 migration manifest/marker | `current` metadata + `dead` timestamp marker | current typed config 只接新写入；非模型 marker 不进入 current        | cleanup manifest 不承载业务 payload；旧 marker 禁止恢复          |
|  17 | `version.json`、`update-check.json`、`installation_id`                                                 | 安装/更新元数据                                     | Lime version/update/config files                                | `current` host metadata                      | 归 Electron host/update owner                                        | 可重建或按安装实例清理，不进 Agent history                       |

## 4. Skills、Plugins、MCP 与外部集成

|   # | Codex `~/.codex`                             | Codex 语义                                              | Lime 当前对应物                                                                                 | 当前分类                     | Lime 目标                                                         | 清理单位/验收                               |
| --: | -------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------- | ----------------------------------------------------------------- | ------------------------------------------- |
|  18 | `skills/`、`.agents/skills`                  | user/system skills package 与 catalog                   | `~/Library/Application Support/lime/skills`、`~/.lime/skills`、`.agents/skills`、provider roots | `current` 但多 root          | user skills、system skills、workspace skills 分层；每层一个 owner | 按 package 目录清理；catalog 可重建         |
|  19 | `plugins/`                                   | installed plugin packages/cache/manifest                | Lime `plugins/installed`、`plugins/setup`、`agent-apps/installed/packages`                      | `current` candidate 多 owner | 收敛到 `app-server/plugins` 或明确 host package owner             | 按 package/version 目录清理；安装状态可重建 |
|  20 | `.tmp/plugins`、plugin cache                 | plugin sync/staging 临时数据                            | Lime `agent-apps/staging`、`plugins/setup`、临时下载目录                                        | `rebuildable`                | 单一 staging/cache root；失败任务 TTL                             | 按 staging job/TTL 清理，不进 product DB    |
|  21 | `mcp-cache/`                                 | 本机实际主要是 MCP 启动依赖的 npm `_cacache/_npx/_logs` | Lime `mcp/`、model/provider DB、runtime snapshot                                                | `partial`/wrong owner        | `app-server/cache/mcp`；配置、credential、lock、依赖 cache 分离   | 按 server/generation/TTL 清理               |
|  22 | `mcp-oauth-locks/`                           | OAuth lock/coordination，不是业务历史                   | Lime `mcp/oauth` 与 OAuth state                                                                 | `current` candidate          | lock 与 token/credential 分开；lock 可删                          | stale lock TTL；不得阻塞 thread delete      |
|  23 | `vendor_imports/`                            | skills/vendor import cache                              | Lime plugin/skill package import 目录，未形成单一 owner                                         | `deprecated`                 | 不作为旧 AppData 迁移源；current package owner 只接新安装         | 旧 cache 按 TTL/exact path 清理             |
|  24 | `chrome-native-hosts*.json`、`computer-use/` | Codex 外部连接/Computer Use 配置                        | Lime connectors/browser/embedded browser                                                        | `product-specific`           | 明确 connector owner；不为 parity 强行并入 ThreadStore            | connector/package 级清理                    |

## 5. 运行日志、进程与缓存

|   # | Codex `~/.codex`                                                 | Codex 语义                                                                                             | Lime 当前对应物                                                                                       | 当前分类                                | Lime 目标                                                                                                      | 清理单位/验收                                                                |
| --: | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
|  25 | `logs_2.sqlite*`                                                 | 活跃 structured log DB；主库/WAL 可能达到 GB 级                                                        | `app-server/runtime/telemetry_1.sqlite`、`runtime/traces`、`runtime/events`、根 `logs`/`request_logs` | `current` 但无统一 retention            | `app-server/observability/` 下专用 log/trace/telemetry owner                                                   | Lime 自定 thread/date/TTL/bytes 预算；停写/checkpoint 与 DB sidecar 一致处理 |
|  26 | `log/`                                                           | optional text logs                                                                                     | `AgentRoot/observability/log/lime.log`；旧 `AppDataRoot/logs`                                         | `current` text + `deprecated` source    | text diagnostics 只用注入的 LogStore 路径；structured logs 单独对齐                                            | 按文件/日期/bytes 清理，不作为 transcript                                    |
|  27 | `shell_snapshots/`                                               | shell environment snapshot，重启/复现辅助                                                              | Lime `harness/tool-io`、sidecar/runtime 输出，未证明等价                                              | `missing`/product-specific              | 若需要，`app-server/shell_snapshots` 单独 owner；不得写入 canonical item                                       | 按 session/TTL 清理                                                          |
|  28 | `process_manager/`                                               | 进程管理协调状态                                                                                       | Lime runtime sidecar/session process 旁路，未形成 Codex 等价 owner                                    | `missing`                               | 明确 `tool-runtime` process owner；短生命周期状态不进 history DB                                               | process/session 结束即清理，重启可恢复/重建                                  |
|  29 | `generated_images/`                                              | 用户生成图片输出                                                                                       | Lime `projects`、`AgentRoot/artifacts/sessions`、media task blobs；旧 `AppDataRoot/sessions` 不迁移   | `current` partial + `deprecated` source | session 文件固定 `AgentRoot/artifacts/sessions/<session-id>`；其他生成物归 project/artifact owner，DB 只存 ref | 按 artifact/session/project 目录清理；禁止回写旧 root                        |
|  30 | `tmp/`、`.tmp/`（本机无顶层 `cache/`）                           | command temp 与 plugin/marketplace staging 是不同生命周期                                              | Lime `Cache`、`Code Cache`、`GPUCache`、download staging、agent-app staging                           | `rebuildable`                           | host cache、runtime cache、download staging 分根                                                               | 目录级 TTL/size GC；`tmp` 与 `.tmp` 不合并 owner                             |
|  31 | `backups/`                                                       | 临时恢复备份                                                                                           | Lime `lime.db.*.bak`、bootstrap backup、旧 migration backups                                          | `deprecated`，模型 staging 除外         | 非模型 backup 不迁移；模型 staging/backup 独立 manifest/TTL                                                    | 非模型按 exact path，模型按 migration id 清理                                |
|  32 | `sqlite/`                                                        | 混合目录；`codex-dev.db` 本次被桌面进程打开并含 automation/inbox/local catalog，其他同名 DB 活跃性不同 | Lime 根 `runtime/`、根 `lime.db`、空 `projection.sqlite`、旧 `aster` DB                               | `mixed`，禁止整目录分类                 | 子库逐项登记 writer/schema/fingerprint；Lime 无同义能力时 `exclude`，不能机械迁入                              | 禁止 root cleanup；逐库停写、对账和 exact-path 决策                          |
|  33 | `ambient-suggestions/`、`node_repl/`、`pets/`、`visualizations/` | Codex feature-specific cache/output                                                                    | Lime 无等价 current owner                                                                             | `missing`/scope decision                | 不为目录 parity 伪造实现；需要时登记独立产品 owner                                                             | 无 owner 则不进入 current catalog                                            |

## 6. Lime 独有目录，不强行映射 Codex

| Lime 路径                                                   | 约占用/语义                              | 处理                                                                 |
| ----------------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------- |
| `models/`                                                   | voice/whisper 本地模型和下载 staging     | `current` 产品资产；模型文件迁移白名单，checksum 校验后原子发布      |
| `projects/`                                                 | Lime 内容工厂/工作区资料                 | `current` 产品数据；按 project/artifact 清理，不伪装成 Codex rollout |
| `Cache`、`Code Cache`、`GPUCache`                           | Chromium/embedded browser cache          | `current` host cache；可重建，不被 Agent runtime 读取                |
| `Partitions/`、Cookies、Local/Session Storage、Trust Tokens | Chromium profile/semantic/security state | `current` host state；不能整体标为 cache 或无条件删除                |
| `agent-apps/`、`plugins/`                                   | Lime App Center/package runtime          | 收敛 package owner；状态/缓存/安装包分开                             |
| `mcp/`、`connectors/`                                       | Lime OAuth/connectors                    | 保留产品能力，但 credential、lock、cache 分离                        |
| `harness/`                                                  | 测试证据、tool IO、memory                | 只允许测试/诊断 consumer；生产 runtime 不读取                        |
| `aster/`                                                    | 退役 runtime DB/session/event            | `dead/deleted/forbidden-to-restore`；不迁移，按 exact path 回收      |
| `runtime/`                                                  | 根级旧 event/projection/sidecar          | `deprecated -> dead`；不迁移，冻结写入后按 manifest 清理             |

## 7. 对齐顺序与硬退出条件

1. 先建立 `StorageManifest`，逐项记录 path、owner、source-of-truth、rebuildable、retention、cleanup unit；未知项标为 `missing`，不能默认为 current。
2. 先实现 Codex-style rollout path/index contract；旧 history/projection 不迁移，禁止复制旧 `projection_1.sqlite`。
3. `ThreadStore` 只写 canonical rollout + metadata patch；`projected_items` 改为 coalesced rebuildable view；高频 delta 不再永久逐行入主 history DB。
4. `archive/delete/gc` 必须按 rollout/artifact 目录和 path identity 操作，DB 只负责索引/状态；清理结果要记录 manifest 和 bytes。
5. 旧 Lime `runtime/`、旧根 DB、`aster`、重复 skills/plugins root 在 current owner 验证后停止读取并删除回流入口；只有模型文件允许迁移。

### 验收矩阵

| 断言                                                     | 验证                                      |
| -------------------------------------------------------- | ----------------------------------------- |
| 每个 current Thread 只有一个 canonical rollout path      | state DB -> path 唯一性测试               |
| rollout 文件删除后可精确删除 metadata/artifact，不扫全库 | session delete dry-run + integration test |
| projection 删除后可重建，且不改变 canonical history      | rebuild/cold read/replay round-trip       |
| archive 是移动不是复制                                   | active/archive path transition test       |
| root runtime/旧 SQLite 无新写入                          | path scan + governance negative guard     |
| logs/traces/cache 有 TTL/size 预算                       | storage report + GC dry-run fixture       |
| goal/memory/auth 等语义数据不会被 GC 误删                | owner-specific delete tests               |
