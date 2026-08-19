# 第二期测试实施计划

> status: active / EVAL-01 Agnes diagnostic evidence collected; gpt-5.5 attribution pass; scoring and full platform RC blocked
> owner: quality-workflow
> started: 2026-07-15
> target: Refactor v2 current chain

## 1. 实施原则

第二期按“先可定位的确定性测试，再产品闭环，再非确定性评估”推进。每个切片必须同时交付测试、稳定 fixture/helper、机器可读结果和删除项；不先建新的总 runner，也不恢复旧 Benchmark 包装层。

测试实现的价值按“是否暴露并归属真实 Lime 缺陷”衡量，不按新增测试文件或 case 数量衡量。DeepSWE task 只有在 workspace、current chain 和 verifier 都有效时才计分；无效 trial 只作为基础设施或产品诊断 evidence。

测试失败必须指向一个明确 owner：protocol、agent-runtime、tool-runtime、model-provider、App Server/read model、Electron host 或 Renderer projection。跨层总脚本只能聚合已有证据，不能成为业务真相。

## 2. 基线冻结

T0 开始时记录：

```yaml
codex_commit: 5c19155cbd93bfa099016e7487259f61669823ff
refactor_v2_status: completed implementation baseline
lime_head: <git rev-parse HEAD>
working_tree_digest: <tracked + untracked product paths>
os_arch: <platform/version/arch>
node_npm: <versions>
rust_toolchain: <versions>
```

测试体系重建期间允许修改测试和 current 实现，但每个 evidence 必须绑定自己的候选摘要；旧 evidence 不自动升级为新候选证据。

## 3. 实施切片

| Slice                     | 优先级 | Owner                                         | 目标                                                                                 | 退出条件                                                                                         |
| ------------------------- | ------ | --------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| T0 inventory/reset        | P0     | quality-workflow                              | 盘点 v2 后真实测试覆盖，删除旧 Benchmark surface                                     | current 导航零旧命令；每个现存 smoke/manifest 有分类                                             |
| T1 integration harness    | P0     | app-server-test-client + runtime test support | 建立可控 response server、公共 client、事件等待、临时数据目录和整对象断言 helper     | 至少 3 条跨 crate 场景复用；无固定 sleep                                                         |
| T2 protocol/client        | P0     | app-server-protocol + app-server-client       | method/schema/generated types/request-response/notification 同步                     | create/read/list/turn/control round trip；未知字段和版本失败可定位                               |
| T3 turn lifecycle         | P0     | agent-runtime + app-server                    | accepted/queued/started/completed/failed/interrupted/cancel/resume                   | 正常、并发、取消、重入和 terminal 单一性全覆盖                                                   |
| T4 item/read model        | P0     | app-server + thread-store + projection        | message/reasoning/tool/content part 的 create/update/order/repair/pagination         | 重启恢复前后整投影相等；stale/out-of-order fail closed                                           |
| T5 tool/policy/context    | P0     | tool-runtime + agent-runtime                  | 工具执行、审批、sandbox、截断、context/compaction/cache                              | allow/deny/cancel/timeout/oversize/compaction 均有集成证据                                       |
| T6 MCP/Skills/Multi-Agent | P0     | app-server + domain owners                    | MCP lifecycle、Skills metadata、parent-child edge、mailbox、control tools            | restart 后 identity/edge/mailbox 可恢复；故障隔离成立                                            |
| T7 provider/multimodal    | P0     | model-provider                                | capability、lowering、stream、usage、error、message parts/media                      | 每种支持协议有 request capture；unsupported fail closed                                          |
| T8 GUI Gate A/B           | P0     | Renderer + Electron host                      | 关键状态在 DOM 可见，真实产品链 identity 一致                                        | P0 场景 Gate A/B 证据齐全；生产 mock 命中为零                                                    |
| T9 DeepSWE coding         | P1     | evaluation + agent-runtime                    | DeepSWE Core 跑 Smoke 10 / Release 20；Desktop Smoke 5 走真实 Electron + Pier 双门禁 | schema/Pier 固定；Core 三轮 baseline；Desktop 5 题 Gate B/verifier 同 run 通过；基础设施失败为零 |
| T10 live/eval             | P1     | evaluation                                    | live provider 与非确定性产品任务质量                                                 | 样本、模型、配置、grader、成本、pass@k/pass^k 可复核                                             |
| T11 platform/release      | P1     | release + quality                             | macOS/Windows、packaged app、并发、长稳与证据汇总                                    | RC required 场景全绿；无 retry 掩盖；平台差异已归属                                              |

## 4. 每个切片的交付合同

每个切片必须提交以下内容：

1. 对应 [scenario-matrix.md](./scenario-matrix.md) 的稳定场景 ID。
2. 受测公共边界和唯一 owner。
3. 正常、失败、取消/恢复中与风险匹配的 case。
4. 结构化 fixture；禁止在生产路径新增 mock fallback。
5. 定向运行命令和预计时长。
6. 机器可读结果，至少包含 scenario、candidate、proof level、status、duration、artifact paths 和 failure owner。
7. 被替代测试/fixture 的删除清单与回流守卫。

## 5. T0 详细任务

### T0.1 生成 inventory

按 owner 统计 Rust/TS 测试、fixture、smoke、Gate A/B 和 live 测试，至少记录：

- 受测 public boundary；
- 是否触达 current product chain；
- 是否只断言静态值或 mock 调用；
- 是否依赖固定 timeout、共享全局状态或顺序；
- 是否仍使用 retired type/command/fixture；
- 是否与其他测试重复证明同一事实。

### T0.2 现有测试分类

| 分类      | 处理                                                                       |
| --------- | -------------------------------------------------------------------------- |
| current   | 场景语义与 v2 owner 一致；加入稳定 ID 并保留                               |
| rewrite   | 有效风险仍存在，但 fixture/入口/断言基于旧 owner；迁移后删除原测试         |
| merge     | 多个边界 guard 重复证明同一事实；合并到 owner 级 contract/integration test |
| dead      | 测试静态值、已删除实现、旧命令正向行为或脱离构建图；直接删除               |
| live-only | 依赖真实账号/网络/模型；移出默认门禁并显式授权                             |

### T0.3 第一批优先审计

- 文件名或内容包含 `legacy`、`compat`、`agent_runtime_*`、旧 Team/roster/raw subagent 的正向测试。
- 只检查源码字符串的 `*BoundaryGuard.test.ts`，判断应归治理扫描还是行为测试。
- React 大型挂载测试里可下沉到 selector/projection/state machine 的分支。
- 使用固定 sleep、超大 timeout、共享 `.lime` 状态或已有用户数据的 smoke。
- 只验证 script 能生成报告、但不验证 current runtime 的 runner test。

## 6. T1 Harness 目标形状

不要复制 Codex crate 名称，但复制其测试思想：

```text
TestRuntimeBuilder
  -> isolated app/data/workspace dirs
  -> deterministic provider response server
  -> RuntimeCore/App Server public boundary
  -> captured structured requests
  -> event wait/predicate
  -> Thread/Turn/Item/read-model snapshot
```

公共 helper 只解决环境、fixture、事件收集和结构化断言，不暴露生产私有状态。单个测试需要的特殊 helper 保持在测试文件内，避免形成第二套 runtime API。

## 7. 运行与 CI 策略

| 时机              | 策略                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------ |
| 本地开发          | `test:related` / `test:rust:related`，先小后大                                       |
| PR                | diff selector 选择 L0-L3；GUI/bridge 风险追加 L4-L6                                  |
| Nightly           | 全量 deterministic 场景、故障注入、flaky/retry=0 审计                                |
| Release candidate | 冻结候选，运行 required 场景与 macOS/Windows packaged matrix                         |
| Live eval         | 独立凭证、预算和数据保留策略；不得被默认 PR 触发                                     |
| Desktop coding    | DeepSWE Desktop Smoke 5；真实 Electron、live provider、隔离 workspace、Pier verifier |

Rust CI 在测试量和平台矩阵稳定后可引入 `cargo nextest` archive/shard；它是执行加速器，不改变测试 owner、场景定义或本地 related-first 规则。

DeepSWE 分为两条互不冒充的 lane。Core lane 通过 App Server current public chain 在固定 DeepSWE agent environment 生成 patch，并由 separate verifier 评分；Desktop lane 复用五语言 Smoke 5，从真实 Electron GUI 输入原始 instruction，记录 preload/IPC/App Server/runtime/read model/GUI/patch identity，再把同一 patch 交给 Pier。`DesktopCodingPass` 只有在 Gate B 与 verifier 同时通过时成立。Nightly 运行 Core Smoke 10，冻结 RC 运行 Core Release 20 与 Desktop Smoke 5 的三 trial；外部补充集只进入显式 live/eval lane。

Windows RC 的 current 入口是 `scripts/electron/windows-squirrel-rc-smoke.mjs`，由手工 Windows package workflow 和 release Windows matrix 共用。单版本模式运行精确 Forge Squirrel Setup、验证 install root/shortcut，并让 installed `Lime.exe` 复用 `SHELL-01`；N-1 模式必须从低于候选的最近稳定 GitHub Release 安装旧版，经真实 preload/IPC/current updater 从隔离 `RELEASES + full.nupkg` feed 下载并进入 restarting，观察候选 `app-<version>/Lime.exe` 落盘后再跑候选 `SHELL-01`。summary 只有 N-1 version/install、feed request、downloaded terminal、install request 与 candidate path 六项全真才可写 `passed`；单版本安装不得冒充 N-1 receipt，任一模式都不得冒充 `SOAK-01`。

macOS 本地 packaged current 入口是 `forge.config.mjs -> electron-forge package -> scripts/electron/verify-package-resources.mjs -> packaged SHELL-01`。本地无 Developer ID 时必须生成完整 ad-hoc sealed-resource signature，所有 per-file option 均关闭 hardened runtime 并禁用 timestamp；正式 signing 路径仍必须启用 hardened runtime、runtime signature flag 和后续 notarization。`codesign --verify --deep --strict`、Helper/sidecar flags 与 packaged Gate B 缺一不可。该本地闭环不能冒充 Developer ID、notarization、DMG 安装或 N-1 update receipt。

2026-08-18 正式 Developer ID arm64 package 已越过本机签名环境 blocker：System identity 的错误 `TrustAsRoot` 覆盖已移除，Forge 使用 `Developer ID Application: Digital Easy Inc. (93RF7VQ26C)` 完成 package。主 App、四个 Helper、`app-server` 与 `code-mode-host` 均通过 `codesign --verify --deep --strict` 和资源 verifier，Authority、TeamIdentifier、runtime flags 一致；签名后 sidecar 由 verifier 按 `macos-signed-sidecar` 合同接受。独立 DMG maker 与 `npm run electron:make:zip-local-feed` 均整体成功；最终 DMG 为 `330311401` bytes、SHA-256 `18fcccac3175a383a755a13f6e502197b07f54db5f1e884d593242b54bab8481` 且 `hdiutil verify` 通过，ZIP 为 `331037460` bytes、SHA-256 `659ba14d9ed9a97225e81eb52c7e6fe88af774557ee75fb5943dbf448bbceeba`。组合 maker 访问线上 `RELEASES.json` 曾因 TLS 断连失败，随后 current local-feed 入口已完成 ZIP；本地 manifest 只作验证，不得发布。当前未配置 `APPLE_ID`、app-specific password 与 `APPLE_TEAM_ID`，App 的 `spctl` 结果为 `Unnotarized Developer ID`，DMG 为 `no usable signature`；notarization/staple、DMG 安装/升级、Developer ID packaged Gate B 和 Windows receipt 仍缺失，所以 T11/第二期完成度保持不变。退出条件是补齐这些平台 receipt，而不是重复 package 或签名构建。

SOAK-01 先用 current controlled provider 做短校准，证明 repeated turns、cancel/continue terminal、read model 与 Electron restart oracle 能发现实现漂移；再在冻结候选上复跑。稳定入口为 `npm run smoke:agent-runtime-soak-current-fixture`。完整 receipt 必须在同一 Electron/App Server 生命周期记录逐轮 Thread/Turn/Item 数量、唯一 terminal、Electron/App Server PID 与 RSS 趋势，并至少执行两次 cold restart；独立启动多次且每次清空 app data 只能证明 cleanup 基线，不能证明长生命周期无泄漏。2026-07-17 修后 `10 rounds x 2 cold restarts` receipt 已满足本地实现合同：每轮唯一 completed turn、Thread/Turn/Item identity 在两次重启后稳定，RSS 在预算内，旧/最终进程树全部退出。fixture 只清 idle connection，active provider request 必须由 current `model-provider` 在 terminal 前释放；冻结 RC 仍须重跑，但不再扩建平行 SOAK runner。

## 8. 退役账本

2026-07-15 已删除：

- `scripts/agent-qc/benchmark*.mjs` 旧 runner 与 runner tests；
- `internal/test/benchmark-release.manifest.json`；
- `internal/test/agent-qc-benchmark.manifest.json`；
- `agent-qc:benchmark*` / `agent-qc:benchmark-release*` npm 入口；
- 旧 dataset selection、version test plan、progress；
- 不对应真实代码 owner 的三份伪代码 test-case 指南。
- 仍把 `agent-qc:benchmark:plan/compare` 和旧 manifest 当作 current 的专用 flag differential research 页；关联导航、进度记录和方案稿已收敛到第二期 Benchmark 事实源。

禁止恢复同名 wrapper。未来若需要聚合器，必须消费第二期机器可读场景结果，且放入已有测试领域目录，不在 `scripts/` 根或 `agent-qc` 下重建平行事实源。

DeepSWE 数据集不属于退役 surface。新的选题与执行合同见 [deepswe-coding-slice.md](./deepswe-coding-slice.md)；旧 runner 删除是为了防止 dry-run 结果冒充真实 Lime coding score。

本地 `.lime/benchmark/runs` 中 45 个旧 runner 产物已于 2026-07-15 删除；固定 source cache 保留给 current adapter。随后 9 个已被回归覆盖的 bring-up run 和两条诊断 run 内的仓库内 clone 也已删除，只保留当前有效 JSON/patch evidence。T9 的 source preflight、adapter v6、provider step/token/usage、真实 request tool catalog、runtime step/token cap、generation diagnostics、wall-timeout terminal cleanup 和 TS/Go/Rust 诊断 true run 已完成；Agnes thinking on/off 及 Smoke 10 短题对照仍为 0-byte patch。DSW-06 最小写入探针已通过，确认 `apply_patch` schema 合同和 patch lifecycle 正常，并补上 provider step exhaustion 误分类守卫。DSW-07 已完成 schema `1.3` source 与 `datacurve-pier==0.3.1` 固定，Core Release 20 `205/205`、Desktop Smoke 5 `53/53` preflight 通过；Pier separate verifier 仍因本机无容器运行时阻塞。

## 9. 当前下一刀

T7、PRV-05/06、CTX-01/02、App Server transport 与 LIV-03 已关闭。T9 的历史 run、patch 和 verifier 产物已清理，所以仍没有 DeepSWE 能力 baseline，更没有桌面 coding 分数。DSW-06 最小写入探针只证明 current `apply_patch` 写链，不替代完整题或 verifier。DSW-07 已将固定源成组迁移到 schema `1.3` commit `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`，并固定 `datacurve-pier==0.3.1`；Core Release 20 `205/205` 与 Desktop Smoke 5 `53/53` preflight 通过。DSW-08 的 batch planner/aggregator 已实现并保持 fail-closed；Smoke 10 plan `105/105` ready，空的 Release 20 run 根目录聚合仍为 `blocked`。下一刀补可用容器 runtime，取得首个 `reward.json`，随后完成 Core Smoke 10 三轮聚合和 Desktop Smoke 5。T11 的 Developer ID package/DMG/ZIP 已有本机证据，但 notarization、安装/升级、Developer ID packaged Gate B 与 Windows N-1 receipt 仍缺失。EVAL-01 保持 `current / diagnostic`；不得用确定性 runner、开发态 Electron 或无 verifier patch 冒充 release score。

测试体系基线为 `100%`；按 T0-T11 退出条件计算，第二期实现整体完成度为 `80%`。

## 10. EVAL-01 首次真实 Gate B 归因（2026-07-17）

- 场景入口：`npm run smoke:claw-chat-ready-streaming -- --provider-preference custom-637ea2d5-e430-43de-86de-39c5f1735438 --model-preference agnes-2.0-flash --timeout-ms 240000 --prefix phase2-eval01-agnes-stable`。
- 真实链路：Chromium GUI -> DevBridge `electron-host` -> Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore/provider -> Thread/Turn/Item/read model；不是 fixture、不是 renderer mock、不是 App Server mock backend。
- 首次失败归因：恢复 turn 已在 App Server/read model 完成，但 sidebar 在发送热路径延迟刷新期间没有新 session identity，标题回退把重复的旧“E2E 中断测试：请输出 80 行。”会话当成目标；同时 helper 原先没有稳定 DOM identity。该失败属于 current GUI/session navigation owner 与测试观察层的真实缺口，已修复。
- current 修复：`useAppSidebarSessions` 在 `reason=created` 时立即加入带 id 的“未命名对话”占位项；`AppSidebarConversationRow` 投影 `data-session-id`；smoke 只按 id 定位并在短窗口重试，不再按标题猜会话。sidebar 相关 5 项定向回归通过。
- 修复后观察：sidebar identity 已关闭。无 watcher 的 3030 Host 下，Agnes 两次复核均完成长流、中断、同 session recovery，但 WebSearch/WebFetch 无 tool event，120 秒后由 public cancel 收敛；固定 gpt-5.5 对照在同一 Host 完整 `pass`，无 mock/阻塞 console error。EVAL-01 仍为 `current / diagnostic`，不能写 Agnes pass@k 或 score。
- 观察器修复：首次空任务 turn 的 provider/model 可能位于 `metadata.harness.model_request_policy` 而非顶层 runtimeRequest；smoke 已按 policy 读取并记录来源，避免把合法 session-default routing 判成失败。
- 退出条件：只有 Agnes 在隔离 Host 中同一 session 同时满足 WebSearch/WebFetch tool started/result、terminal event、read-after-event、GUI 正文和无 blocking error，才冻结 EVAL-01 baseline。gpt-5.5 仅作 Lime/provider 归因对照，不替代 Agnes baseline。
