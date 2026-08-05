# Refactor v2 测试体系第二期执行计划

> status: active / macOS local packaged closed; Windows/full L8 and DeepSWE scoring pending
> owner: quality-workflow
> created: 2026-07-15
> last_updated: 2026-07-17
> source: `internal/research/refactor/v2/**`
> architecture_impact: confirmed in `internal/aiprompts/architecture.md` 6.1/6.2/6.3; developer/PR diagram confirmation still required

## 1. 主目标

以 Refactor v2 完成后的唯一产品链为测试对象，删除重构前的 Benchmark runner、外部数据集门禁和失真测试指南，建立可执行的第二期测试计划：

```text
Electron Desktop Host
  -> App Server JSON-RPC
  -> RuntimeCore
  -> Thread / Turn / Item projection
  -> GUI
```

第二期不继承旧 Benchmark 的通过状态。旧报告只能作为历史 evidence，不能作为当前候选门禁。

## 2. 本轮写集

- `internal/roadmap/benchmark/**`
- `internal/test/**`
- `internal/aiprompts/quality-workflow.md`
- `scripts/agent-qc/benchmark*.mjs`
- `package.json` 中 `agent-qc:benchmark*` 命令
- `internal/exec-plans/refactor-v2-test-phase-2-plan.md`
- `internal/exec-plans/README.md`
- `.gitignore` 中本计划的精确跟踪例外
- `scripts/harness/deepswe-*.mjs`
- `electron/main.ts`、`electron/windowsSquirrelStartup*`
- `scripts/electron/{smoke,windows-squirrel-rc-smoke,release-workflow-guard}*`
- `forge.config.mjs`、`scripts/electron/{forge-config,verify-package-resources,current-entrypoints}.test.mjs`
- `.github/workflows/{build-windows-test,release}.yml`
- `src/components/agent/chat/components/ToolCallDisplay{,.test}.tsx`
- `lime-rs/crates/{core,skills,model-provider,agent-runtime,agent,tool-runtime,app-server}/**` 中由真实 DeepSWE trajectory 直接定位的 owner 修复

## 3. 避让集

- conversation import、Plugin、App Server protocol/client 与 GUI 并行热区
- `internal/research/refactor/v2/13-evidence/verification.md` 的并行修改
- `internal/exec-plans/project-gate-a-b-acceptance-plan.md` 及其证据目录
- `package.json` 中非 Benchmark 的当前工作树改动

## 4. 执行阶段

| 阶段                    | 状态        | 退出条件                                                                                                                                                         |
| ----------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R0 旧基线退役           | completed   | 旧 Benchmark runner、manifest、npm 入口和重复文档退出 current 导航                                                                                               |
| R1 策略重写             | completed   | 质量规则、测试分层、测试作者规则和证据等级对齐 v2/Codex                                                                                                          |
| R2 第二期路线图         | completed   | 场景矩阵、实施切片、owner、门禁、退出条件和删除账本完整                                                                                                          |
| R3 引用收口             | completed   | current 文档无悬空链接，历史 evidence 不再被导航为 current                                                                                                       |
| R4 验证                 | completed   | JSON、文档守卫、脚本治理、合同与定向测试通过；跨写集阻塞已精确记录                                                                                               |
| R5 DeepSWE 缺陷发现     | completed   | Agnes 主测与 gpt-5.5 对照在仓库外隔离 workspace 完成 terminal；确定性 Lime 缺陷已回归；Pier evidence 齐全或 blocker 明确                                         |
| R6 MCP 故障闭环         | completed   | MCP-02 reverse JSON-RPC Gate B 与 MCP-03 单 server 故障隔离 fresh evidence 通过；暴露的 Desktop Host JSON-RPC error 转发缺陷在 current owner 修复                |
| R7 GUI/ELN 首批闭环     | completed   | GUI-01/02、ELN-01/03 fresh Gate B/packaged evidence 通过；Gate B probe identity 误配与 App Server smoke 用户 data root 污染已关闭                                |
| R8 PRV/transport 闭环   | completed   | PRV-06 capability/WebSocket/session fallback、App Server 默认栈 dispatcher、stdio 非 turn 并发及完整 Electron current fixture 通过                               |
| R9 Windows RC 自动化    | in_progress | Forge Squirrel lifecycle、N-1 -> candidate current updater、installed SHELL-01 与 fail-closed summary 已接入 Windows CI；真实 Windows receipt 待完成             |
| R10 SOAK-01 缺陷发现    | completed   | 同生命周期 10x2 receipt 已关闭 Host admission、历史 timeline 与 SSE terminal HTTP body 延迟释放缺陷；冻结 RC 只复跑合同，不再扩建 runner                         |
| R11 macOS local package | completed   | fresh Forge package 通过 deep/strict codesign，Helper/sidecar 为纯 ad-hoc，packaged SHELL-01 Gate B 通过；正式 Developer ID/notarization/DMG 仍属完整 RC blocker |

## 5. 删除裁决

- `dead`：重构前 `benchmark-release-v1`、旧 Terminal-Bench/DeepSWE true-run、旧 Managed Objective differential manifest 及配套 runner/test。
- `dead`：旧 dataset selection、版本测试计划和按日期追加的 progress 日志。
- `deprecated -> rewrite`：旧测试总览、单元/集成/E2E/Agent evaluation 指南。
- `current`：Rust related tests、App Server public JSON-RPC integration、current runtime fixture、Gate A、真实 Electron Gate B、live provider 显式 lane。
- `current / adapter v5`：DeepSWE v2 Smoke 10 / Release 20、仓库外 task workspace、current-chain adapter、逐步 provider usage、真实 request tool catalog、runtime step/token cap、单次 run generation controls、wall time 兜底、partial patch 和分层 failure evidence；诊断 true run 已完成，Pier verifier 阻塞。

## 6. 完成定义

1. 新测试策略以 current 产品链为唯一事实源。
2. Agent runtime 逻辑变更默认要求公共边界集成测试，不以组件 mock 或脚本拼装替代。
3. 场景覆盖正常、失败、取消、排队、恢复、分页、过期事件、工具审批和跨进程可见状态。
4. 所有门禁写明能证明和不能证明的内容。
5. 旧 Benchmark 命令、manifest 和 current 导航引用归零。
6. 本轮只完成测试体系基线重建；第二期测试实现按路线图切片继续推进，不伪报覆盖完成。

## 7. 验证记录

### 7.1 已通过

| 命令/检查                                                                                                                                                                       | 结果                                                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `npm run test:contracts`                                                                                                                                                        | 通过                                                                                                                                                      |
| `npm run governance:scripts`                                                                                                                                                    | 通过                                                                                                                                                      |
| `npm run docs:boundary`                                                                                                                                                         | 通过                                                                                                                                                      |
| `npm run governance:legacy-report`                                                                                                                                              | 通过；零引用候选 0、分类漂移 0、边界违规 0。此前并行写入期间的 `history_builder.rs` 瞬时违规已消失，历史过程仍保留在 7.2。                                     |
| `npx vitest run scripts/harness/deepswe-coding-slice.test.mjs`                                                                                                                  | 3/3 通过；source、Smoke 10、Release 20 合同有效                                                                                                           |
| `npx vitest run scripts/harness/deepswe-adapter.test.mjs scripts/harness/deepswe-coding-slice.test.mjs`                                                                         | 25/25 通过；精确 base、仓库外 workspace、runtime step/token/generation owner、wall-timeout 终态取消、累计 usage、failure owner 与 verifier blocker 有回归 |
| `cargo test --manifest-path lime-rs/Cargo.toml -p agent-runtime`                                                                                                                | 118/118 通过；`max_turns=2` 在第三次 sampling 前停止，provider request/step 均严格为 2                                                                    |
| `cargo test --manifest-path lime-rs/Cargo.toml -p app-server session_config_projects_bounded_deepswe_provider_step_budget`                                                      | 通过；DeepSWE step budget 投影到 current `AgentSessionConfig.max_turns`，且不能扩大默认上限                                                               |
| `cargo build --manifest-path lime-rs/Cargo.toml -p app-server`                                                                                                                  | 通过；Agnes stdio true run 使用新构建 current App Server                                                                                                  |
| `cargo test -p agent-runtime provider_turn`                                                                                                                                     | 12/12 通过；cwd 注入及 commentary/final phase 通过                                                                                                        |
| `cargo test -p model-provider current_client`                                                                                                                                   | 29/29 通过；空 tool placeholder、SSE idle timeout 和 error chain 通过                                                                                     |
| App Server coding projection 定向测试                                                                                                                                           | 4/4 通过；commentary/final、取消和失败 item terminal 通过                                                                                                 |
| App Server current tool inventory / confirmation 定向测试                                                                                                                       | 3/3 通过；删除 `Bash` 正向 fixture 残留，改为 `exec_command` + `cmd`，并验证 approval resume 后投影为 canonical `Command` item                            |
| App Server session/projection blocker 定向测试                                                                                                                                  | 4/4 通过；WebSearch canonical item 合同、failed-delete 原子性、empty-prefix crash-tail 修复和 orphan child restart cleanup 已闭环                         |
| Gate B `inputbar-pending-steer-rich-restore`                                                                                                                                    | 通过；真实 Electron/IPC/App Server/read model/GUI identity 一致，rich text/image/path/skill 恢复断言全通过，console/page error 为 0                       |
| `npm run harness:deepswe:preflight`                                                                                                                                             | 通过；Release 20 共 61 项 source/schema/verifier/image 检查通过                                                                                           |
| `npx vitest run scripts/harness/deepswe-adapter.test.mjs`                                                                                                                       | 22/22 通过；runtime token terminal 可由 evidence 反推，wall timeout 会取消并等待真实 terminal；step/token/generation metadata 同步下发                    |
| `cargo test --manifest-path lime-rs/Cargo.toml -p agent-runtime provider_token_budget -- --nocapture`                                                                           | 1/1 通过；预算恰好耗尽后 provider request=1、工具执行=0、request/step trace 仅 attempt 1                                                                  |
| `cargo test --manifest-path lime-rs/Cargo.toml -p app-server session_config_projects_bounded_deepswe_provider_step_budget -- --nocapture`                                       | 1/1 通过；只投影有界正整数 step/token budget，越界 step 与零 token fail closed                                                                            |
| Agnes Rust run `20260716T120650Z-fd-deterministic-multi-key-sorting`                                                                                                            | 2 request / 2 step，15,065/12,000 tokens；attempt 2 的 4 个工具调用零执行，无 attempt 3，runtime 自主 cancel                                              |
| `cargo test --manifest-path lime-rs/Cargo.toml -p model-provider current_client -- --nocapture`                                                                                 | 33/33 通过；公开 `CurrentProvider.stream` capture 证明 401/429 单次 terminal、501/505 后第 3 次成功、最终 503 分类正确                                    |
| `cargo test --manifest-path lime-rs/Cargo.toml -p agent-runtime provider_failure_trace_preserves_auth_rate_limit_and_server_categories -- --nocapture`                          | 1/1 通过；auth/rate_limit/server 与 retryable/non-retryable rejection 进入 current provider trace                                                         |
| `npm run smoke:agent-runtime-multimodal-capture`                                                                                                                                | 通过；provider wire 收到 data URL、tools=0、max_tokens=128、enable_thinking=false，canonical read/evidence 只保留 sidecar reference 且无 base64           |
| Agnes LIV-03 `smoke:agent-runtime-multimodal-capture -- --allow-live-provider ...`                                                                                              | 通过；`agnes-2.0-flash` 在 25.5 秒内识别 apple/red，真实 App Server turn 为 completed，read/evidence 无 inline base64                                     |
| `cargo test --manifest-path lime-rs/Cargo.toml -p app-server input_media -- --nocapture`                                                                                        | 2/2 通过；inline 图片落 sidecar、provider-only hydrate、无 sidecar/非法 payload fail closed                                                               |
| `npm run test:rust:related -- <DeepSWE owner paths>`                                                                                                                            | 通过；agent-protocol 30/30、agent-runtime 118/118、app-server 1159/1159，全部 scoped 反向依赖 crate 为 0 failed                                           |
| `cargo test --manifest-path lime-rs/Cargo.toml -p app-server provider_history -- --nocapture`                                                                                   | 11/11 通过；batch parts、同 item completed suffix、commentary/final turn-wide full snapshot 完整且不重复                                                  |
| `npm run test:rust:related -- lime-rs/crates/app-server/src/runtime/provider_history.rs`                                                                                        | 1169/1169 通过；App Server current provider history 反向边界无回归                                                                                        |
| `npx vitest run scripts/electron/current-entrypoints.test.mjs scripts/lib/electron-fixture-build.test.mjs`                                                                      | 23/23 通过；renderer smoke build 保留共享 `dist`，build freshness/lock 合同有效                                                                           |
| approval resume + renderer concurrent build                                                                                                                                     | 通过；构建期间完整 approval/respond/second no-prompt Gate B 闭环通过，未再出现 `ERR_FILE_NOT_FOUND`                                                       |
| `npm run smoke:agent-runtime-current-fixture`                                                                                                                                   | 通过；history/terminal、approval、cancel/continue、queue/hydrate、Coding Workbench、MCP、Skills 与真实 Electron current fixture 全绿                      |
| `cargo test -p app-server agent_mailbox_delivery`                                                                                                                               | 9/9 通过；多 Result identity、Result failed terminal、partial delta retry 和 terminal-only ack 回归通过                                                   |
| `cargo test -p app-server agent_terminal_activity`                                                                                                                              | 15/15 通过；child result、wait/restart/recovery 和 parent mailbox activity 全绿                                                                           |
| `cargo test -p app-server agent_control`                                                                                                                                        | 24/24 通过；Codex AgentControl spawn/list/send/follow-up/interrupt/wait/restart 合同全绿                                                                  |
| `cargo test --manifest-path lime-rs/Cargo.toml -p app-server --lib agent_control::tests`                                                                                        | 26/26 通过；AGT-04 双 child 同时运行、mailbox 路由、Result 一次聚合、混合 failed/completed 终态和 list 状态隔离全绿                                       |
| `cargo test --manifest-path lime-rs/Cargo.toml -p app-server --lib canonical_message_lifecycle`                                                                                 | 10/10 通过；canonical UserMessage 在 synthesized terminal snapshot 后保留原始 task content，AgentMessage/reasoning fail-closed 未回归                     |
| `cargo test --manifest-path lime-rs/Cargo.toml -p app-server --lib commits_large_codex_command_history_within_linear_time_budget`                                               | 通过；1200 commands、fidelity 零丢项，current 增量实现独立耗时 3.51s，低于 30s owner budget                                                               |
| `npm run test:rust:related -- <event-store/materializer/performance owner paths>`                                                                                               | 1157/1157 通过；sequence/tool lifecycle fail-closed、canonical notification/read model 与 import 性能反向依赖全绿                                         |
| `npm run test:rust:related -- lime-rs/crates/app-server/src/runtime/agent_mailbox_delivery.rs lime-rs/crates/app-server/src/runtime/event_store/canonical_message_lifecycle.rs` | 1152/1152 通过；反向依赖未回归                                                                                                                            |
| `npm run smoke:agent-control-cold-restart-gate-b -- --cold-restart`                                                                                                             | 通过；真实 Electron 冷重启后 6 个工具、terminal mailbox Result、child identity、visible DOM 和截图一致                                                    |
| `node --test scripts/electron/current-docs-guard.test.mjs`                                                                                                                      | 12/12 通过                                                                                                                                                |
| Electron current rules guard                                                                                                                                                    | 10/10 通过                                                                                                                                                |
| `npm run test:bridge`                                                                                                                                                           | 37/37 通过                                                                                                                                                |
| `npx vitest run electron/appServerHost.test.ts scripts/mcp/current-smoke.test.mjs`                                                                                              | 34/34 通过；Desktop Host 业务错误保持 JSON-RPC response，MCP failure isolation guard 有回归                                                               |
| `npm run smoke:mcp-current -- --allow-write-fixture`                                                                                                                            | 通过；MCP-03 fresh current evidence 覆盖失败 start、stopped 状态及健康 server 的 status/tool/resource 连续可用                                            |
| `npm run smoke:mcp-elicitation-gate-b`                                                                                                                                          | 通过；MCP-02 真实 Electron reverse JSON-RPC、Renderer form、MCP accept ledger 与 provider continuation 全绿                                               |
| Claw `complete` fresh Gate B                                                                                                                                                    | 通过；GUI-01/ELN-01 Electron/preload/IPC/App Server/runtime/read model/DOM identity 一致，legacy/mock/page error 为 0                                     |
| Claw `cancel-then-continue` fresh Gate B                                                                                                                                        | 通过；GUI-02 cancel、canceled read model、输入恢复与同 session 后续 turn complete 全绿                                                                    |
| Claw `terminal-failed-after-answer` fresh Gate B                                                                                                                                | 通过；ELN-03 backend failure 保留 partial answer、read model failed、GUI 输入恢复，legacy/mock/page error 为 0                                            |
| 四条 App Server stdio/external/packaged smoke                                                                                                                                   | 通过；每条使用隔离临时 `dataDir`，unavailable fail-closed、success、packaged lifecycle 与 packaged backend crash 全绿                                     |
| Claw Gate B contract + smoke guards                                                                                                                                             | 66/66 通过；已知 turnId 只接受精确 evidence 匹配，后续 event-read probe 不得冒充产品 turn                                                                 |
| `npm run test:resume` 的批次 36-109                                                                                                                                             | 全部通过；未重复运行已完成批次                                                                                                                            |
| `git diff --check`                                                                                                                                                              | 通过                                                                                                                                                      |
| 旧 Benchmark 入口扫描                                                                                                                                                           | 仅命中本计划的删除账本和 DeepSWE 迁移说明，无 current 命令/manifest/runner 命中                                                                           |

`npm run verify:local` 已通过版本一致性、i18n、lint 和 typecheck；前端 smart suite 首轮受并行工作区失败中断，随后从批次 36 续跑到 109 并全部通过。

### 7.2 当前阻塞

- 原 3 条 App Server session/projection 失败已关闭：`WebSearch` 旧断言改为 canonical `web_search` item；failed-delete 测试在 durable pending-spawn 依赖恢复后验证内存原子性；empty-prefix crash tail 在无 terminal 有效前缀时只截断安全尾部，不放宽 projection watermark fail-closed。
- Gate B `inputbar-pending-steer-rich-restore` 已独立通过，先前 `identityConsistent=false` 未复现。该旧 evidence 不能继续作为 current blocker。
- live provider preflight 发现真实用户数据中存在 open child + identity + pending mailbox、但 session event/projection history 缺失的 orphan。此前 App Server 启动会以 `SessionNotFound` 退出；current recovery 现删除 unusable child 的 mailbox、identity、session data 和 graph edge，定向回归通过。
- Agnes DSW-02 run `20260716T020910Z-go-genai-streamed-function-args` 在 1,200,000ms 内产生 1,763 个 event、15 个 coding tool item、0 tool failure，但没有任何文件写入，最终为 `budget` + empty patch。它证明 current cwd、command/Read、provider stream 和隔离 workspace 可用，同时再次暴露 Agnes 长 reasoning/只读探索吞吐不足。
- provider usage evidence 缺陷已关闭：current runtime 现在每个 sampling step 投影 `provider.step`，multi-step usage 在 lime-agent 累计，并由 App Server 写入 trajectory/turn terminal；timeout/cancel 不再丢失已消耗 usage。
- adapter v3 Agnes run `20260716T033001Z-go-genai-streamed-function-args` 在 16-provider-step 预算内产生 2,350 个 App Server event、33 个 trajectory item 和 16 个 tool call；16/16 step usage 完整，累计 input 268,495、output 3,829、budget 272,324 tokens。第 16 步经 current cancel 进入 `canceled`，failure owner 为 `budget`，patch 为 0 bytes。
- 该 run 唯一工具失败来自 Agnes 在命令正文中少写临时 cwd 片段；工具实际 `cwd` 正确，随后 15 次工具调用成功。这进一步把无 patch 归到 Agnes coding 吞吐，而不是 Lime cwd、sandbox、transport 或 App Server。
- provider request tool catalog 缺口已关闭：`provider.request.started` 直接携带每次 sampling 的稳定去重工具名，不再用 turn 前未初始化的 inventory 近似。Agnes 同题 3 次旧 request 和 2 次 v4 request 都完整下发 27 个工具，`apply_patch` 每次存在。
- provider step 越界已关闭：旧 2-step run `20260716T081020Z-go-genai-streamed-function-args` 在 adapter 轮询取消前启动 attempt 3；v4 把上限投影到 current reply loop，fresh run `20260716T083349Z-go-genai-streamed-function-args` 只有 attempt 1、2，`budgetCancellation=null`，不存在第 3 次 sampling。
- provider token 越界已关闭：旧 Rust run `20260716T113222Z-fd-deterministic-multi-key-sorting` 在 7 个 completed step 累计 158,754/150,000 tokens 后仍启动 attempt 8；fresh run `20260716T120650Z-fd-deterministic-multi-key-sorting` 在 attempt 2 累计 15,065/12,000 后停止，attempt 2 的 4 个工具调用未执行且不存在 attempt 3。
- MCP-03 首次 current smoke 已关闭：App Server 正确生成 `mcpServer/start` JSON-RPC error，但 Electron `handleJsonLines` 把 `AppServerRequestError` 抛成 IPC/DevBridge error，renderer 原 request 收不到 error line。current Host 现只在 JSONL 转发边界恢复 error response 和原 request id；fresh evidence 证明健康 server 不受失败 server 影响。
- MCP-02 fresh Gate B 已关闭：runtime stdio connection 的 elicitation capability、Renderer form submit、MCP accept ledger、provider final text、真实 Electron/preload/App Server 均通过；management connection 未错误广告 capability，legacy MCP 命中为零。
- `smoke:agent-runtime-current-fixture` 与 `verify:gui-smoke` 本轮均已完整通过；前者覆盖真实 Electron Gate B 的 cancel/continue、approval、queue、MCP、Skills、media contentParts 和 Coding Workbench，后者证明默认 GUI 启动、preload/IPC、App Server sidecar 和 shell 健康。
- Pier receipt 虽记录 `datacurve-pier 0.3.0`，但 editable source `/tmp/lime-pier-source-20260715` 已删除，wrapper 当前报 `ModuleNotFoundError: No module named 'pier'`；本机也没有 Docker/Podman/nerdctl/Colima。package 与 container runtime 是 verifier 双重 blocker，当前仍无有效 DeepSWE 分数。
- EVAL-01 仍为 `current / diagnostic`：无 watcher 的 3030 Electron Host 下，Agnes 两次复核都完成长流、中断、同 session recovery，但 WebSearch/WebFetch 回合没有 tool event，120 秒后由 smoke 发送真实 cancel；同 Host 的固定 gpt-5.5 对照完整 `pass`。该对照支持 provider/model 稳定性归因，不足以冻结 Agnes baseline 或计入 pass@k。
- `npm run governance:legacy-report` 当前仍因并行写集已有的 `rust-runtime-file-checkpoint-inline-content-leak` 阻塞；本轮未修改该 Codex import/history 热区，不能把该失败归因到 EVAL-01 或 smoke 观察器。

## 8. 完成度与下一刀

- 本轮“测试体系基线重建”完成度：`100%`。R0-R4 均已完成，旧基线已退出，v2/Codex 场景矩阵和 DeepSWE Coding 切片已落库。
- “第二期测试实现”整体完成度仍为 `80%`。确定性 T0-T8 已形成主要 current 证据，CTX-02、PRV-06、App Server transport 与 LIV-03 已关闭；T9 仍是缺陷诊断而非有效评分。T11 已关闭 macOS 本地 packaged 三项缺陷与 SOAK-01 本地 controlled-provider 实现合同，Windows Squirrel 自动化也已实现，但正式 macOS RC、真实 Windows receipt、N-1 update 与冻结候选复跑未闭环，因此不增加完成度。
- DeepSWE 当前状态：`diagnostic_true_runs_blocked`。adapter v5 缺陷发现链已完成；已有 Agnes TS/Go/Rust、thinking on/off 主测和 gpt-5.5 对照 trajectory，逐步 usage、真实 request tool catalog、runtime step/token cap 与显式 generation controls 证据完整。Agnes 能稳定消费 current coding tools，但在固定预算内无 patch；本机 Pier editable 安装失效且无容器 verifier，尚无有效 Lime App Server DeepSWE 分数。
- current App Server blocker、queue/restore Gate B 和 LIV-03 已关闭；第二期仍未达到完成口径，因为 DSW-02 未产生 non-empty candidate，DSW-03 Smoke 10 calibration 尚未开始，真实 Windows/L8 receipt 也未生成。
- AGT-04 owner 场景与 Codex import 性能阻塞均已关闭；App Server related gate 当前为 1157/1157。性能修复没有提高 30s 阈值，而是把独立 1200-command commit 从 106.7s 降到 3.51s。
- 下一刀：触发并消费首份 Windows RC receipt，在有正式凭证的冻结候选上补 macOS Developer ID/notarization/DMG receipt，并复跑已关闭的 SOAK-01 合同。DeepSWE scoring 只在 non-empty candidate、可用 Pier package 与容器 verifier 三者齐备后恢复。

## 9. DeepSWE 继续实施记录

2026-07-15 用户要求继续清理 `dead/deleted` 并完成 DeepSWE Coding：

- 已物理删除 `.lime/benchmark/runs` 的 45 个旧运行目录（约 27 MB）；保留固定 source cache。
- 已新增 `harness:deepswe:preflight` 与 `harness:deepswe:run`，不恢复 `agent-qc:benchmark*`。
- adapter 使用 `workspace/ensure -> agentSession/start/update/turn/start/read -> evidence/export`，输出 run context、trajectory、Thread/Turn/Item、tool lifecycle 和 patch。
- Pier 只接收 candidate patch；临时 replay task 不包含 DeepSWE reference solution，必须返回 `reward.json`、`ctrf.json` 和 `test-stdout.txt`。
- `DSW-00` 已完成：Release 20 共 61 项 source/schema/verifier/image 检查通过。
- `DSW-01` 已执行 Agnes 主测与 gpt-5.5 对照。真实 trajectory 已定位并修复 cwd 传播、模型可见 cwd、skill root 隔离、空 tool placeholder、Bash 重定向误判、伪 final、失败 item 未终态、SSE 总 timeout、错误链丢失、workspace 依赖污染和大 patch `ENOBUFS`。
- gpt-5.5 对照固化 6 文件、189 增/29 删、1,235,341 字节 patch；原 trial 因宿主 `node_modules` 污染不计分，但 patch/evidence 保留用于诊断。
- adapter v2 Agnes run `20260715T212218Z-happy-dom-abort-pending-body-reads` 记录 3,691 条事件，活动 SSE 约 78 分钟且未再被 600 秒总时限截断；最终 5,400,000ms 预算耗尽、empty patch，归 `budget`。
- 本地 receipt 记录的 Pier `0.3.0` 是 editable 安装，源目录 `/tmp/lime-pier-source-20260715` 已删除，wrapper 已不可导入 `pier`；再加上无 Docker、Podman、nerdctl 或 Colima，adapter 会把 package/container 两项都写成 verifier blocker，不生成伪造分数。
- 已删除 9 个过时 bring-up run（约 155 MB）和两条保留诊断 run 中的仓库内 clone；只保留 JSON/patch evidence 与固定 source cache。

## 10. 2026-07-16 current 主链缺陷记录

- 扩展 Rust related 不是为了扩大 case 数量，而是验证 DeepSWE 修复是否破坏反向依赖；它直接发现并关闭 3 个 App Server session/projection blocker。
- `Bash` 退役迁移残留已从 App Server 正向 fixture 清除，current 测试只消费 `exec_command` / canonical `Command`。
- Gate B queue/restore 已在相同 external controlled fixture 下通过；先前失败 evidence 不再代表 current 状态。
- AGT-03 cold-restart Gate B 首次 fresh run 暴露真实 Lime 缺陷：Result mailbox 只写 `message.delta(status=completed)`，多个结果在同一 turn 触发 canonical Item identity conflict，且 partial delta 会被错误 ack。现已在 `app-server` owner 修复为 `message.delta(status=in_progress) -> message.completed(terminal)`，canonical ack 只接受 terminal Item；9 条 owner 回归、1152 条 related Rust、真实 Electron cold-restart Gate B 全部通过。
- 同一修复暴露旧 Gate B runner 的截断断言：`wait_agent` 返回多个 terminal activity 后，500 字符 preview 中找不到 `timed_out`。runner 现用完整 current read-model tool output 做场景断言，evidence 仍保留短 preview；该改动属于测试方案修正，不是生产 mock 或放宽门禁。
- startup orphan failure 与 live 模型无关；它来自 durable agent graph 和缺失 session history 的恢复边界，已在 App Server owner 层补回归。

## 11. 2026-07-16 Agnes DSW-02 记录

- 固定 provider `custom-637ea2d5-e430-43de-86de-39c5f1735438`、model `agnes-2.0-flash`；Agnes provider 只有该文本模型，两个 `agnes-image-*` 不进入 coding eval。
- Go task 在系统临时 workspace 和 SQLite vacuum snapshot 中执行；模型先误搜 TypeScript，随后正确读取 `types.go`、`models.go`、`live.go`，说明工具/cwd/stream current 链有效。
- 20 分钟内没有 `Write`、patch 或 git diff；失败归 `budget`，不归 `tool-runtime`、`transport` 或 `app-server`。
- adapter timeout 原先会丢失 partial `currentChain`；现已在 Error 上携带 provider/model/session/turn/timestamp/evidenceCapture，并由 CLI catch 持久化，当前 adapter 18/18 回归通过。
- 已删除本地旧 BrowserGym、Tau-Bench、Terminal-Bench、WebArena source cache 及旧 fixture/release evidence，约 373 MB；DeepSWE source、Pier 和 v2 diagnostic runs 保留为 current。

## 12. 2026-07-16 adapter v3 与 Agnes usage 记录

- `agent-runtime` 新增每步 `ProviderStep`，`lime-agent` 累加所有 sampling step usage，App Server 投影 current `provider.step` 并把累计 usage 带到 `turn.completed`。
- DeepSWE adapter v3 新增 `provider-steps.json`、provider step budget、token budget 和 current `agentSession/turn/cancel`；默认 32 steps / 500,000 tokens，token 公式为 `max(0, input_tokens - cached_input_tokens) + output_tokens`。
- Agnes v3 true run 以 16 steps / 500,000 tokens 诊断预算运行，16/16 usage 完整，最终由 step budget 取消且无 candidate。该结果已经能区分模型行为与 Lime 基础设施，不再继续无差别刷 Go/Rust 题。
- 两条全套负载 flaky 已关闭：terminal activity 测试改为 `timeout_ms=0` 的状态驱动恢复；confirmation 测试移除进程全局环境修改并在双 worker runtime 中完成初始化。related 反向依赖全套通过且未复现。
- 聚合 Agent Runtime fixture 发现 renderer smoke build 回流为 `LIME_VITE_EMPTY_OUT_DIR=1`：build lock 只串行构建动作，不能保护并行 Gate B fixture 对共享 `dist/index.html` 的消费，导致 approval reload 命中 `ERR_FILE_NOT_FOUND`。已恢复 `LIME_VITE_EMPTY_OUT_DIR=0` 并反转 current entrypoint guard；旧实现上的回归断言先失败，修复后并发 build + approval Gate B 与完整聚合 smoke 均通过。

## 13. 2026-07-16 AGT-03 current 缺陷闭环

- 失败证据：`.lime/qc/phase2-agt03-agent-control-cold-restart-20260716.json`。原始报错为 `AgentMessage event message.delta targets Item ... while ... is active`；这不是 Agnes/provider 行为，而是 Lime canonical lifecycle 与 mailbox Result 终态不一致。
- 根因：旧 `mailbox_message_runtime_event` 对 `AgentMailboxMessageKind::Result` 只生成 delta，并把 presentation status 写成 `completed`；旧 `canonical_mailbox_item_exists` 只按 Item ID 判断存在，未检查 `ItemStatus::is_terminal()`。因此两个不同 Result 在一个 turn 中违反 Codex 的 start/complete 顺序，崩溃重放也可能提前 ack。
- current 修复：Result 生成 `message.delta(in_progress)` 和同 itemId 的 `message.completed`；`canonical_mailbox_item_is_terminal` 只允许 terminal Item ack；重放时可由 mailbox messageId 去重 delta 并补齐缺失 completion；Failed Result 投影为 `ItemStatus::Failed`。canonical lifecycle 的 fail-closed identity 规则保持不变。
- 测试方案修正：Gate B runner 保留短 preview 作为 evidence 展示，但场景断言消费完整 read-model tool output，避免多 mailbox Result 让旧截断字符串断言误报。
- fresh 证据：`.lime/qc/phase2-agt03-agent-control-cold-restart-20260716-fixed-v2.json`，真实 Electron/preload/IPC/App Server/runtime/read model/GUI 冷重启链通过，6/6 AgentControl tool completed，visible DOM 和截图通过；`liveProviderUsed=false`。

## 14. 2026-07-16 AGT-04 并发 child 缺陷闭环

- 失败证据来自双 child owner 回归：session/thread、agent path 和 mailbox item identity 均正确，但 child canonical `UserMessage.content` 为 `""`，只有 identity metadata 的 `last_task_message` 保留原文。旧测试只断言 item 存在，没有验证模型实际收到的 canonical task 内容。
- 根因：canonical lifecycle 为 `message.created` 合成 `item.started -> item.completed`；terminal snapshot 没有重复内容，而 `thread_item_projection::merge_payload` 缺少 `UserMessage` 合并规则，导致最后一个空 terminal snapshot 覆盖先前完整 payload。
- current 修复：UserMessage snapshot 只在新 content 非空时覆盖，并保留已有 `client_id`；不新增兼容 payload、不放宽 canonical lifecycle 校验。
- AGT-04 现验证两个 child 同时处于 running、session/thread/mailbox 路由不串、两个 terminal Result 一次聚合且二次 wait 不重复、并发 QueueOnly message 各入目标 mailbox、failed child 与 completed sibling 状态独立。
- 定向验证：AgentControl 26/26、canonical message lifecycle 10/10；`smoke:agent-runtime-current-fixture` 在重新构建 renderer/Electron/App Server 产物后通过，`liveProviderUsed=false`。后续 current owner 关闭 import 性能退化，App Server related 全量更新为 1157/1157。

## 15. 2026-07-16 Codex import 性能缺陷闭环

- 失败证据：1200-command import 在全量/独立运行分别耗时 174.4s/106.7s，超过固定 30s owner budget；fidelity 内容正确，因此属于 event processing 复杂度退化，不是 parser、模型或数据源失败。
- 根因：旧 `append_runtime_events_to_stored_session` 为每个候选事件重新扫描并 clone 全部 validation context；canonical notification 同样为每个事件重建累计 history 并全量 materialize。长历史因此形成重复 O(n²) 工作。
- current 修复：sequence verifier 和 tool lifecycle validator 改为一次初始化、逐事件 validate-and-observe；canonical notification 使用与 Codex change accumulator 同语义的 `IncrementalMaterializer`，每个事件只返回本次改变的 Turn/Item snapshot。fail-closed 合同和最终 read model 保持不变。
- fresh evidence：独立 1200-command 门禁 3.51s；`npm run test:rust:related -- <event-store/materializer/performance owner paths>` 为 1157/1157；此前完整 `smoke:agent-runtime-current-fixture` 在上述增量 notification 改动落盘后通过，`liveProviderUsed=false`。

## 16. 2026-07-16 DeepSWE tool catalog 与 runtime step cap 闭环

- 旧证据缺口：冷启动 `agentSession/toolInventory/read` 返回 `agentInitialized=false` 和 0 个工具，不能证明模型请求实际拿到什么。current `provider.request.started` 现从 sampling 冻结的 `RuntimeToolStepSnapshot` 记录稳定、去重的 `tool_names`；adapter 汇总每步和全 run tool catalog。
- Agnes run `20260716T081020Z-go-genai-streamed-function-args` 证明三次 request 都有相同 27 个工具，`Read`、`Grep`、`exec_command`、`apply_patch` 每次存在；无 patch 因而不是 Lime 隐藏写工具。该 run 同时暴露 2-step 预算仍启动 attempt 3，因为 adapter 只能在 evidence 轮询后取消。
- current 修复：DeepSWE 把 `runtimeRequest.metadata.harness.provider_budget.max_provider_steps` 投影到 `AgentSessionConfig.max_turns`，由 reply loop 在捕获工具 snapshot 和启动 provider stream 前检查。adapter v5 外部轮询只取消 token budget，避免 step budget 双 owner 竞态。
- fresh evidence：`20260716T083349Z-go-genai-streamed-function-args` 只有两个 request、两个 completed step，`budgetCancellation=null`，runtime max-turn 文案进入 canonical Thread/Turn/Item；累计 budget tokens 19,905，两个 request 均有完整 27 工具和 `apply_patch`。Agnes 仍无 patch，failure owner 为 `model`，属于模型在 2-step 诊断预算内只读探索。

## 17. 2026-07-16 MCP-02/MCP-03 current 缺陷闭环

- MCP-03 首次真实运行在健康 stdio server 旁启动一个必然失败的 server。App Server 已生成正确 JSON-RPC error，但 `ElectronAppServerHost.handleJsonLines` 把 `AppServerRequestError` 继续抛到 IPC/DevBridge，renderer 原 request 无法观察业务 error line。
- current 修复保留高阶 `request()` 的异常合同；只有 JSONL 转发边界把 `AppServerRequestError.messages/response` 恢复为 JSON-RPC response，并还原 renderer 原 request id。transport timeout、stale restart 和 sidecar lifecycle 错误仍按原语义抛出。
- MCP-03 fresh evidence：`.lime/qc/gui-evidence/mcp-current/phase2-mcp03-failure-isolation-20260716-fixed-summary.json`。`appServerHandleJsonLinesSeen=true`，失败 start 可观察且 server 为 stopped；健康 server 仍 running，tool 仍可 list/call，resource 仍可 read，legacy MCP command 为 0。
- MCP-02 fresh Gate B：`.lime/qc/gui-evidence/mcp-elicitation-gate-b/mcp-elicitation-gate-b-summary.json`，`checkedAt=2026-07-16T08:56:37.214Z`。真实 Electron/preload/`app_server_handle_json_lines`、runtime elicitation capability、Renderer form、MCP accept ledger、provider 第二次请求及 final text 全部通过，console error 和 missing method 均为 0。
- 回归与合同：Electron Host + MCP guard 34/34；`npm run test:contracts` 通过。contract guard 同步改为守住 current 增量 `EventValidationContext -> AgentEventSequenceValidator/ToolLifecycleValidator -> validate_and_observe` owner，不再要求已退役的逐事件全历史校验调用字符串。

## 18. 2026-07-16 GUI-01/02 与 ELN-01/03 缺陷闭环

- GUI-01/ELN-01 首次 fresh complete Gate B 业务 turn 已完成，但 identity collector 把后续 `event-read-probe` 的 `agentSession/turn/start` response 当成产品 turn，误报 `turnConsistent=false`。根因是 `findByTurnId` 精确匹配失败后无条件退回最后一个 entry。
- current 修复：已知 primary turnId 时，App Server/backend evidence 只接受精确匹配；不存在精确 request-log response 时保持 null，由 Electron trace、runtime ledger 与 read model 提供独立同 turn 证据。回归先在旧逻辑下稳定失败，修复后 Gate B contract + smoke guards 66/66。
- GUI-01/ELN-01 fresh evidence：`.lime/qc/gui-evidence/claw-chat-current-fixture/phase2-gui01-eln01-20260716-fixed-summary.json`。Renderer/trace/runtime/read model session/turn/item identity 一致，Electron/preload/IPC/current JSON-RPC、terminal DOM 与 screenshot 通过，legacy/mock/page error 为 0。
- GUI-02 fresh evidence：`.lime/qc/gui-evidence/claw-chat-current-fixture/phase2-gui02-20260716-summary.json`。GUI stop 命中 current cancel，read model 为 canceled、输入框恢复；同 session 发送“继续输出”后第二 turn 完成，legacy/mock/page error 为 0。
- ELN-03 fresh Electron evidence：`.lime/qc/gui-evidence/claw-chat-current-fixture/phase2-eln03-backend-failure-20260716-summary.json`。backend 在 partial answer 后失败，GUI 保留正文且不泄露底层 failure text，read model 为 failed，输入框恢复，identity 与 current bridge 全绿。
- packaged unavailable 首次运行报真实用户 data root 下固定 session 的 `SequenceRegression`，证明旧 App Server smoke 未隔离持久化状态。隔离后继续暴露 stdio 把先到的 notification 误当 response，以及 packaged failure 漏复制 runtime dylib；现分别按 request id 等待 response、复用 Electron runtime library copy owner。stdio、external、packaged lifecycle、packaged backend failure 四条 smoke 全部注入本轮临时 `dataDir`，fresh success/failure/unavailable 均通过；contract guard 扩为 291 项并逐入口防回流。未删除或修改真实用户目录数据。

## 19. 2026-07-16 multimodal 与 runtime token budget 闭环

- 多模态旧缺口是 inline base64 进入 canonical persistence，历史恢复无法在 provider 边界安全重放。current App Server 现先写 sidecar，EventLog/read model 只保存 reference；provider 执行前校验 sha256/大小并瞬时 hydrate。Chat/Responses/Anthropic lowering、历史图片恢复和 provider capture 均通过。
- Codex 语义已固化：当前回合新图片遇到 text-only 模型在网络前拒绝；历史图片降为明确占位文本，不阻塞后续文本回合。2026-07-16 的 Agnes registry/cache 曾把 `agnes-2.0-flash` 错标为 text-only；该事实已由 2026-07-17 LIV-03 修复和 live evidence 取代。
- `governance:legacy-report` 首轮发现 `input_media.rs` 重复写死 `sidecarRef`；实现改为复用 `output_refs::SIDECAR_REF_FIELD`，未扩大白名单。复跑为零引用候选 0、分类漂移 0、边界违规 0。
- DeepSWE Rust 旧 run 在 token 超限后仍启动下一 sampling。current `agent-runtime` 现在拥有 provider step/token budget；带工具调用的 step 达标后，工具零执行、下一 request 零启动。fresh Agnes run 的 adapter `requestedAt=null`，证明终止来自 runtime 而非轮询抢先 cancel。
- 验证：受影响 Rust related 13 个 crate 全部退出 0；adapter 20/20；App Server/agent-runtime 定向回归全绿；multimodal provider capture、contracts、scripts/legacy governance、完整 Agent Runtime Electron fixture 与 `verify:gui-smoke` 通过。
- 架构图文字已同步到 `internal/aiprompts/architecture.md` 6.2/6.3；进入 release evidence 前仍需责任开发者在 PR 描述确认该架构变更。

## 20. 2026-07-16 PRV-05 provider failure 语义闭环

- 旧测试全部绿色，但固化了错误重试集合：请求层自动重试 408/409/425/429，只列举 500/502/503/504，既会把 rate limit 放大成五次请求，也不符合 Codex 的全部 5xx 规则。
- Codex current `ModelProviderInfo -> ApiRetryConfig` 默认 `request_max_retries=4`、`retry_429=false`、`retry_5xx=true`、`retry_transport=true`。Lime 保持相同的初始请求加 4 次上限，并把自动重试决策与错误 `retryable` 分类拆开。
- current `model-provider` 现在只为全部 5xx 和 transport 执行请求层重试；401/403/429 直接返回 terminal。429 仍分类为 `RateLimit` 且 `retryable=true`，表示失败性质，不再被误用为本请求继续发送的许可。
- 测试改从公开 `CurrentProvider.stream(CurrentProviderRequest)` 进入 localhost capture：401 和 429 均只收到 1 个 HTTP request；501/505 后第 3 次成功；最终 503 保留 `ProviderInternal + retryable`。agent-runtime trace 的 auth/rate_limit/server 分类回归通过。
- related 验证覆盖 agent-runtime、App Server、agent、CLI、MCP、media-runtime、processor、scheduler、server、services、skills、model-provider、tool-runtime 共 13 个 crate，全部 0 failed；其中 agent-runtime 120/120、App Server 1166/1166、model-provider 128/128。

## 21. 2026-07-16 CTX-01 provider history 闭环与 PRV-06 审计

- CTX-01 旧实现只从 `message.delta/message.delta_batch/message.batch` 的标量 `text/delta/content` 读取正文，batch array 会静默丢失；`message.completed` 即使携带 full-text snapshot 也完全忽略。GUI/read model 的 full-text 去重测试因此不能证明下一轮 provider 实际收到完整历史。
- current `provider_history` 现在递归解析 `deltas/messages/items/parts/content`，并记录每个 assistant item 的累计正文。completed snapshot 先与整个 assistant turn 比较，再回退同 item 前缀，只补缺失后缀；覆盖单 item `delta(prefix) -> completed(full)` 以及 commentary/final 多 item的 turn-wide snapshot。
- owner 回归 11/11、App Server related 1169/1169。完整 `smoke:agent-runtime-current-fixture` 重新构建 App Server sidecar 后通过，覆盖真实 Electron Gate B 的 cancel/continue、pending steer queue、Plan/history hydrate、Skills、MCP、media、approval 和 Coding Workbench，`liveProviderUsed=false`。
- PRV-06 审计确认 Lime current provider 没有 WebSocket transport、provider capability 字段或 session-scoped fallback state。Codex 的实现会 capability-gate Responses WebSocket，在重试耗尽后把当前 session sticky 切到 HTTPS；这属于未实现产品能力，不新增 benchmark-only transport，也不把 HTTP SSE 单路径测试冒充 fallback 证据。

## 22. 2026-07-16 CTX-02 compaction/truncation 闭环

- 旧绿测只断言 `context.compaction.completed` 早于 `turn.started`，以及下一轮 runtime metadata 存在 summary context；测试 backend 没有覆盖 `start_turn_with_provider_history`，因此实际发送完整旧历史也会通过。
- 新回归从公开 `RuntimeCore::start_turn` 连续执行 6 个 turn，并在第 7 个 turn 捕获 `ExecutionBackend::start_turn_with_provider_history`。旧实现首先失败于摘要不含 `fact-from-turn-1`：`build_summary` 只复制最近 4 个 tail turn；继续检查可见 provider history 时，旧前缀仍完整存在。
- 对照 Codex `ContextManager::replace_history` / `replace_compacted_history` 后，Lime current owner 保持 durable EventLog、Thread/Turn/Item read model 全量，只重写 model-visible transcript。`session_context_compaction.v2` summary 只接续 `tailStartTurnId` 之前的 turn；provider history 从最新有效 tail event 重建，本轮 input 仍独立提交；无效 boundary fail open，不静默丢历史。
- 同一场景发现历史 `ToolOutput.outputRef` 会优先读取完整 sidecar，绕过 append 边界的 preview。current provider history 现只消费 canonical preview；异常未 offload 的 inline output 再按 10,000-byte provider 上限截断，full sidecar 只供显式 artifact/evidence/read owner。
- 失败证据：修复前 `auto_compaction_replaces_provider_prefix_with_summary_and_bounded_tail` 在 `summary.contains("fact-from-turn-1")` 处稳定失败。修复后 provider history 不含 turn 1/2，保留 turn 3-6，summary 含 turn 1/2 且不重复 turn 3，durable events 仍可读 turn 1。
- 验证：provider history 12/12、auto compaction 4/4、session compact 2/2；App Server unit/related 1171/1171；`npm run test:contracts` 的 App Server contract 291 项及命令/modality/scripts/docs 守卫全绿；完整 `npm run smoke:agent-runtime-current-fixture` 通过真实 Electron/preload/IPC/App Server/runtime/read model 的 cancel/continue、queue、approval、Plan/history、Skills、MCP、media 与 Coding Workbench，`liveProviderUsed=false`。
- 分类：App Server compaction/provider history 为 `current / closed`；旧“summary metadata 存在即算压缩成功”和“provider history 自动 rehydrate full output sidecar”语义为 `dead / deleted / forbidden-to-restore`。架构文字已同步；进入 release evidence 前仍需责任开发者在 PR 描述确认本次 provider-history rewrite 边界。

## 23. 2026-07-16 PRV-06 Responses WebSocket 与 session fallback

- 首次失败 capture 从公开 `CurrentProviderClient::stream` 进入 localhost fixture：`supports_websockets=true` 时旧实现实际仍是 `POST /v1/responses`，失败信息为 `capability=true must use a WebSocket upgrade, actual=POST /v1/responses HTTP/1.1`。这证明缺口属于 production transport，不是 benchmark case 数量不足。
- current capability owner 是 `ProviderRuntimeSpec.supports_websockets`；OpenAI/OpenAI Responses/Codex 显式声明支持，Gateway/NewApi/其他 provider 默认 false。direct runtime request 可以显式提交 `supportsWebsockets`，未声明即 HTTP；`model-provider` 不按名称猜测。
- `AgentRuntimeState` 从全局单 provider 改为 session map。同一 session 且 route/config 相同复用同一个 client，不同 session 隔离；route 变化替换当前 session client，`ExecutionBackend::close_session` 清理 transport state。
- `model-provider` 现在向 `ws(s)://.../v1/responses` 发真实 Upgrade，携带 `OpenAI-Beta: responses_websockets=2026-02-06`，发送去掉 HTTP-only `stream/background` 的 `response.create`。SSE 与 WebSocket 共享一个 Responses reducer；同一连接严格串行承载多个 sampling request，不 multiplex。
- 426 或 Upgrade 重试耗尽会立即/最终 replay HTTP，并把 session sticky 切到 HTTP。Upgrade 成功后若在首个 canonical event 前断线或命中 connection-limit error，也只 replay 一次完整 HTTP request；已经发出 text/tool event 后禁止 replay。取消或未完整消费的流会淘汰连接，避免下一 Turn 读取旧 frame。
- 证据：capability=false 为 `POST`；capability=true 成功链为单次 `GET` 加两个串行 `response.create`；426 跨两次 client request 为 `GET -> POST -> POST`；重试耗尽为 5 次 `GET` 后 1 次 `POST`；Upgrade 后首事件前 close 为 `GET -> POST`。RuntimeBackend 连续两个真实 turn 同样得到 `GET 426 -> POST -> POST`，证明 provider configuration 确实复用 session client。
- 已通过：`model-provider` 135/135、`lime-agent` 270/270、core provider spec 定向、App Server direct capability 与两 Turn fallback 定向、App Server 1175/1175、App Server client contract 291 项及完整 Electron current fixture。短问候 Gate B 的 `firstTextDeltaToFirstTextPaintMs=25`，预算未放宽。
- 文件退出条件：`current_client.rs` 在本轮前已因大型 inline tests 超过 1,400 行，本轮虽把 transport stream 放入 `current_client/websocket.rs`，主文件仍超过 1,000 行。进入 PRV-06 release evidence 前必须把 inline tests 按 lowering/stream/transport/websocket 拆到子模块；在此之前不得继续向主文件堆业务逻辑。
- 分类：Responses WebSocket transport、provider capability 和 session client cache 为 `current / closed`；旧“Responses 协议天然等于只走 HTTP SSE”和“全局 provider 足以表达 session fallback”的语义为 `dead / deleted / forbidden-to-restore`。架构文字已同步；进入 release evidence 前仍需责任开发者在 PR 描述确认 provider client lifetime 与 replay 边界。

## 24. 2026-07-17 PRV-06 扩大门禁与 App Server transport 闭环

- TS boundary 首次失败证明 `supportsWebsockets=true` 在 `runtimeRequest.providerConfig` normalization 中被丢弃。current `app-server-client` 现同时接受 camelCase/snake_case 布尔值并拒绝非法字符串，generated schema 与 Rust protocol 同步；client 66/66、normalization 2/2、protocol 49 + schema fixture 通过。
- MCP public JSON-RPC 精确回归在默认 2 MiB 栈稳定 stack overflow，`RUST_MIN_STACK=8MiB` 才通过，且第一条 `initialize` 已复现，排除 MCP server 本身。根因是 725 行 async dispatcher 把 290 个 handler future 内联进同一 future；current dispatcher 先将分支装箱，再统一 await，默认栈回归与 App Server 全量通过。
- 真实 Electron Gate B 首次 ledger 为空并在 30 秒超时。根因是 stdio loop 只 task 化 turn/media，任一非 turn 长请求会阻塞所有后续 Desktop Host IPC。current transport 保持 `initialize` 内联，初始化后 task 化全部 request，notification/response 保持内联，serialization scope 继续负责资源顺序；长 external turn 期间并发 `agentSession/list` 回归通过。
- App Server 全量扩大首次留下两个环境污染项。PTY 测试继承真实用户 zsh rc，只收到命令回显；旧 marker 断言还会把输入回显误当执行结果。current fixture 现注入确定性 shell，并要求 marker 同时出现在回显与命令输出，2/2 精确回归通过。plain directory status 则被无 deadline `git rev-parse` 卡死；对齐 Codex 后先做 `.git` ancestor preflight，仓库内 Git 改为 async process + `kill_on_drop` + 5 秒 timeout，合成 timeout 回归 50ms 通过。
- fresh 证据：App Server `1175 passed / 0 failed`；完整 `npm run smoke:agent-runtime-current-fixture` 通过 history、terminal、首页热路径、Coding Workbench、图片、cancel/continue、approval、queue、Plan、Skills、MCP、media、Expert 与 Content Factory，`liveProviderUsed=false`。短问候 provider wait 90ms、renderer apply 1ms、first text delta 到 paint 25ms；此前高负载下的 379ms 未复现，预算保持不变。
- contract guard 不再把旧 `.await` 源码形状当协议事实；method→handler 比较语义化忽略 `.await`、`.boxed()` 和同步 `ready(...)` 包装，MCP guard 同步。完整 `npm run test:contracts` 已通过，包含 App Server client contract 291 项及命令、modality、scripts、release、cleanup 与 docs 边界。
- 分类：dispatcher、stdio transport、Project Shell/Git process owner 与 PRV-06 均为 `current / closed`；旧单体 future、初始化后串行 stdio request、真实用户 rc 测试依赖和无界 blocking Git 为 `dead / deleted / forbidden-to-restore`。DeepSWE/Agnes 状态未变化：没有 non-empty candidate，且 Pier package/container runtime 均 blocked，不生成分数。

## 25. 2026-07-17 LIV-03 Agnes 多模态与 DeepSWE timeout 收口

- LIV-03 首次失败不是单一模型能力问题：`agnes-2.0-flash` 被 registry 标为 text-only，自定义 Provider UUID 绕过能力推断，旧 model cache 又持久化 `vision=false`。current model list 现稳定返回 `vision=true`、`inputModalities=["text","image"]`，旧错误 cache 不再污染路由。
- 普通 ReAct harness 曾向图片理解回合暴露完整 coding tool catalog，Agnes 因而幻觉调用不存在图片路径。current harness metadata 现在投影 `tool_surface=direct_answer` 与 `max_provider_steps=1`；确定性 provider capture 证明 request `tools=0`。
- Agnes 官方请求需要 `max_tokens` 与 `chat_template_kwargs.enable_thinking`。Lime 原先虽有 canonical generation/provider options，current client 和 provider lowering 会丢弃。current owner 现贯通 `CurrentProviderRequest -> CanonicalRequest -> Chat/Responses/Anthropic lowering`；capture 精确得到 `max_tokens=128`、`enable_thinking=false`。
- deterministic capture 同时证明图片只在 provider wire 瞬时 hydrate，Thread/Turn/Item read model 与 evidence 只保留 sidecar reference、没有 inline base64，turn 为 completed。随后 Agnes live 在 25.5 秒内识别 probe 中的 apple/red 并以真实 completed terminal 收敛，LIV-03 关闭。
- DeepSWE adapter 的 wall timeout 原先直接抛错并留下 running turn。adapter v4 现在先调用 public `agentSession/turn/cancel`，最多等待 10 秒读取真实 terminal，再写 terminal/partial evidence；timeout 仍归 failure，不会把 canceled 冒充成功。adapter 回归 21/21。
- 分类：generation lowering、Agnes capability/cache、direct-answer tool surface、multimodal provider capture 和 timeout cancel/wait 都是 `current / closed`；旧 text-only Agnes cache、图片理解 ReAct 工具面、generation options 丢弃和 timeout 后遗留 running turn 为 `dead / deleted / forbidden-to-restore`。DeepSWE scoring 仍 blocked：Agnes coding 无 non-empty candidate，Pier package/container runtime 均不可用。

## 26. 2026-07-17 approval compact timeline Gate B oracle 收口

- generation/direct-answer 改动后的完整 `smoke:agent-runtime-current-fixture` 首次在 approval resume 场景超时；backend `action/respond`、assistant final、completed terminal 和输入框恢复都已成立，但 `approvalRecordShape.recordCount=0`。
- 根因不是 production 审批记录丢失。当前工作树的新产品合同会把刚完成的长过程也折叠为 compact timeline，截图明确显示 `2 项 / 查看待处理项`；旧 Gate B 仍等待 `timeline-approval-record` 自动可见，因此把可展开的 current 记录误报成缺失。
- current Gate B oracle 现在先确认 terminal 文案、输入恢复和零 stop button；若要求审批记录且同 turn 存在 compact timeline，则点击 `message-list-historical-timeline-preview:leading`，展开后再断言唯一只读记录。completed/canceled 两条等待链共用同一判断，不新增 scenario 特判。
- fresh approval resume Gate B 得到唯一 `权限记录·本会话允许`，无原 prompt、请求/范围/来源详情泄漏；第二 turn 命中 session cache 且不重复弹审批。Electron/preload/IPC/App Server identity 一致，legacy/mock/page error 为零。
- 验证：completion wait + fixture guard `66/66`；定向 approval resume Gate B 通过；完整 `npm run smoke:agent-runtime-current-fixture` 重新通过首页、Workbench、图片、cancel/continue、四类 approval、queue/restore、Plan、Skills、MCP、media、Expert 和 Content Factory，`liveProviderUsed=false`。
- 分类：compact terminal timeline 与可展开审批审计记录为 `current / closed`；“terminal approval record 必须默认展开才算存在”的旧 oracle 为 `dead / deleted / forbidden-to-restore`。本轮没有为测试回退 production UI。

## 27. 2026-07-17 Agnes DeepSWE generation 对照准备

- adapter v5 新增 `--max-output-tokens` 和 tri-state `--enable-thinking true|false`，仅显式设置时投影到 `runtimeRequest.metadata.harness.generation`；未指定时不下发，不修改 production/provider 默认。
- `run-context.json` 记录本轮 generation controls，确保 Agnes 与归因对照可复核。adapter current-chain 回归 `22/22`，精确断言显式 `max_output_tokens`、`enable_thinking=false` 已进入 public turn request，未指定时则不存在 `generation` 字段。
- Pier 环境重新审计后确认是双重 blocker：本地 wrapper 依赖已删除的 editable source，且本机没有容器 runtime。旧“Pier 0.3.0 可用、仅缺容器”的 current 文档事实已删除。
- live run `20260717T031735Z-happy-dom-abort-pending-body-reads` 使用 Agnes、8 step、80,000 token budget、`maxOutputTokens=4096`、`enableThinking=false`。8/8 step 的 reasoning chars 为 0；累计 input 94,747、cached 6,144、output 443、budget 89,046，执行 5 次命令和 2 次文件读取，8 次 request 每次都有 `apply_patch`，仍未调用写工具，patch 为 0 bytes。
- 第 8 step 返回继续读取工具后，runtime 在工具执行前自主终止，turn 为真实 canceled、`budgetCancellation.requestedAt=null`，证明 generation 对照没有破坏 current budget owner。Agnes 关闭 thinking 后仍未形成 candidate，按退出条件停止重复 trial。
- 对照 Codex 后不把全工具面直接判为缺陷：Codex 也按 feature/runtime gate 暴露 image/multi-agent 等工具；Lime `compact_tools` 会同时切换轻量通用提示词，没有证据证明它能改善 coding。下一刀回到 Windows RC/L8，不增加 benchmark-only production 分支。

## 28. 2026-07-17 Windows Squirrel RC 自动化与 production 缺陷

- 审计确认 `build-windows-test.yml` 与 release Windows matrix 原先只做 Forge make、resource verify、stage/upload，没有安装 Setup、启动 installed app 或 current-chain round trip，不能证明 `PLT-02`。
- 对照当前 `electron-winstaller` 发现 production lifecycle 把 shortcut 参数拼成 `--createShortcut=Lime.exe` / `--removeShortcut=Lime.exe`，且等待 detached `Update.exe` close 才退出；官方合同要求两个独立参数并在 Squirrel 短 deadline 内退出。current owner 已提取纯 startup plan，改为 `flag, exeName` 两参数并在 1 秒内退出；旧等号参数和 wait-for-close 行为归 `dead / deleted / forbidden-to-restore`。
- 新 `windows-squirrel-rc-smoke.mjs` 运行精确版本 Setup、等待 `app-<version>/Lime.exe` 与 `Update.exe`、验证 install root 读写和 shortcut，再用 `LIME_ELECTRON_SMOKE_EXECUTABLE` 直启 packaged app；packaged branch 清空 `APP_SERVER_BIN`，禁止借源码 sidecar 冒充 packaged 证据。外层 L8 summary 合并内层 `SHELL-01`，并明确 N-1 update/soak 为 `not-exercised`。
- 手工 Windows package workflow 和正式 release Windows matrix 共用该 runner，失败时始终上传 `.lime/qc/windows-squirrel-rc/**`；release workflow guard 防止退回只构建/上传。Squirrel/Forge/release 定向 Vitest 51/51、Electron host build、`test:contracts`、版本一致性与本机真实 `verify:gui-smoke` 均通过。
- 当前机器为 macOS，尚未产生真实 Windows receipt，因此 `PLT-02` 保持 `in_progress / evidence pending`，第二期完成度仍为 `80%`。下一刀是运行 Windows workflow 并按首个失败 owner 继续修；之后补 N-1 updater，并在冻结候选复跑 `SOAK-01`。

## 29. 2026-07-17 SOAK-01 短校准与 cold-restart DOM identity 修复

- 先连续执行 3 轮真实 Electron `cancel-then-continue`。每轮 53 项断言全绿，read model 从 2-item canceled 收敛到 4-item completed，GUI 恰好两个 turn group，console/page error 为零；三轮结束后本轮临时 Electron/App Server 均退出。该结果只证明独立 lifecycle cleanup，不冒充同进程长稳。
- AgentControl cold restart 首跑 runtime/tool evidence 已通过，Electron PID 也真实替换，4 条 canonical SubAgent activity 和 child thread identity 均恢复；但 6 条 `tool-call-row` 的 `data-tool-call-id/name/status` 全为 `null`，唯一性与可见性断言失败。
- 根因是 production DOM projection 双轨：`InlineToolProcessStep` 投影 canonical tool identity，`ToolCallDisplay` 对相同 row 完全缺失。current owner 现让 grouped/ungrouped 两个分支都投影同一组 `id/name/status`，并扩展现有 owner 测试；没有解析可见文案或放宽 Gate B。
- 修后 owner 定向 45/45、ESLint、renderer typecheck 通过。相同 cold restart fresh evidence 为 `pass`：PID `41348 -> 43939`，6/6 AgentControl row 唯一 completed/visible，4 条 SubAgent activity、child thread、`agentSession/read`、`thread/list` 与最终正文稳定，invoke/console error 为零。
- 分类：`ToolCallDisplay` canonical DOM projection 为 `current / closed`；同 testid 不同 identity 语义的旧双轨为 `dead / deleted / forbidden-to-restore`。本节记录产生时 `SOAK-01` 仍是 `current / partial`；同生命周期 RSS/Thread/Turn/Item 与两次 restart 后续已由第 31 节关闭。

## 30. 2026-07-17 PLT-01 macOS 本地 packaged 签名与 Gate B 闭环

- 首个 fresh Forge package 的外层 `.app` 未执行签名，`codesign --verify --deep --strict` 以 `code has no resources but signature indicates they must be present` 失败。current Forge owner 现在在无正式凭证时使用 `identity="-"`、`identityValidation=false`，resource verifier 对主 `.app` 强制 deep/strict 验证并 fail closed。
- 第二个真实冷启动继续暴露 ad-hoc 与 hardened runtime 不兼容：Helper、Electron Framework、GPU/network service 与 packaged App Server 都是 `adhoc,runtime` 且无 Team ID，系统以 `mapping process and mapped file (non-platform) have different Team IDs` 拒绝映射。current local signing 对所有 per-file option 设置 `hardenedRuntime=false`、`timestamp="none"`；正式 Developer ID 路径仍保持 `hardenedRuntime=true` 与 `signatureFlags=["runtime"]`。
- fresh package 耗时 `764.74s`，`codesign --verify --deep --strict --verbose=2` 与 package resource verifier 均通过；`Lime Helper.app` 和 `Contents/Resources/app-server/darwin-arm64/app-server` 都是 `flags=0x2(adhoc)`，没有 `runtime`。当前 `@electron/osx-sign 1.3.3` 对 framework symlink 的重复遍历仍是本地 package 性能债，但本轮未在无确认下升级核心发布依赖。
- packaged `SHELL-01` 首跑又发现 launcher 把 `LIME_ELECTRON_SMOKE_EXECUTABLE` 重写到 `.lime/electron-dev-host/Lime.app`，导致开发 Electron 打印 CLI usage 且没有产品 evidence。packaged 分支现显式设置 `LIME_ELECTRON_BRAND_DEV_APP=0`，守卫 15/15 通过。
- 修后 fresh packaged Gate B-F 为 21/21：`file:///index.html?nativeStartup` 首次加载、reload 与设置/记忆页均 ready；preload invoke、`app_server_handle_json_lines` 和 packaged App Server `1.106.0` 可用。trace 有 40 次 successful `electron-ipc` App Server 命中，包含 `agentSession/list`、`modelProvider/list`、`workspace/default/ensure`、`workspace/ensureReady`；console/page/invoke/trace/crash/mock/legacy 均为 0，`2880x1840` 截图人工确认无空白、重叠。
- 分类：Forge macOS local signing、strict resource verifier 与 packaged smoke launcher 为 `current / closed`；未签外层 `.app`、无 Team ID 的 ad-hoc hardened runtime、packaged executable 回退 dev branding 为 `dead / deleted / forbidden-to-restore`。本地 packaged 子场景已关闭；正式 Developer ID signing、notarization、DMG install/update、N-1 updater 仍未执行，因此 `PLT-01` 完整 RC 与第二期整体完成度继续保持 `80%`。DeepSWE 状态不变：Agnes 无 non-empty candidate，Pier package/container 双重 blocker，不能生成有效分数。

## 31. 2026-07-17 SOAK-01 同生命周期闭环

- 同一 Electron/App Server 生命周期的 AgentControl `3 rounds x 2 cold restarts` 首次运行发现 Host 把合法 admission 错套独立 2 秒 identity read timeout。Lime App Server 会在 admission 前执行 workspace/provider preflight；Codex `turn/start` 同样让 admission 服从请求 deadline。current Host 已删除 2 秒窗口，让 250ms quick ack 与 canonical `agentSession/read` 共用 `turn/start` deadline；Host 定向 27/27。
- 长历史恢复又证明 compact preview 展开后仍永久隐藏 canonical tool/subagent rows。根因是 timeline 一直传入 `showOperationalDetails=false`，不是 read model 丢数据。current UI 只在 compact preview 阶段延迟挂载，materialize 后恢复 operational details；managed oracle 接受 preview 消失或真实 timeline 新增，再展开 process block，不再只数 preview。UI history/timeline 定向 27/27。
- 10 轮 timing 扩大在第 7、10 轮稳定出现 `181-182s` 尾延迟；public JSON-RPC/read model 已 terminal，但 `app-server <-> fixture` TCP 仍为 active。根因位于 `model-provider/current_client/stream.rs`：OpenAI `[DONE]`、Responses terminal batch 和 Anthropic `message_stop` 先 yield terminal，悬停 generator 继续持有 `reqwest::Response`；上层收到 Finish 后不再 poll，HTTP body 延迟释放。current 三条 stream 均在 terminal event 前 drop frames。共用 keep-alive owner fixture 逐条证明 peer close 不依赖下一次 poll，3/3 通过；Rust related 13 个 owner/反向依赖全绿，App Server `1181/1181`。fixture 只关闭 idle connection，禁止用 `closeAllConnections()` 强杀 active request 掩盖产品泄漏。
- 修后最终 receipt 为 `.lime/qc/soak-01/20260717-local/agent-control-soak-10x2-sse-fixed.json`。10 轮 duration 为 `4.450-8.398s`，每轮唯一 completed turn、item identity 唯一；两次重启后 10 个 session 的 Thread/Turn/Item 对象完全一致。总 RSS `466,624 -> 274,352 KiB`，App Server RSS `71,920 -> 67,248 KiB`；Electron PID `12587 -> 28108 -> 29467`，每棵旧进程树与最终进程树全部退出。Gate B 恢复 6 条 completed AgentControl row、4 条 canonical SubAgent activity，invoke/console error 为零。
- `SOAK-01` 的本地 controlled-provider 实现合同为 `current / closed`；稳定入口为 `npm run smoke:agent-runtime-soak-current-fixture`，冻结 RC 只复跑同一合同，不新增 runner。SSE terminal HTTP body 延迟释放为 `current / closed`；独立 2 秒 identity timeout、展开后永久隐藏 operational details、preview-count-only oracle 和“只看工具 completed、不观察资源/恢复趋势”的假绿口径为 `dead / deleted / forbidden-to-restore`。fixture idle cleanup 为 `test-only / closed`，不得冒充产品修复。
- `electron/appServerHost.ts` 当前 1269 行，超过仓库 1000 行边界。本轮只修 shared deadline，不在 SOAK 热区顺手重构；退出条件是在下一次 Host 业务扩展前，把 turn admission/proxy lifecycle 按既有 private owner 拆出并保持 `handleJsonLines`、timeout/cancel/stale-restart 合同与 27 条回归不变。
- 第二期实现完成度继续保持 `80%`。完整 DeepSWE 仍为 `diagnostic_true_runs_blocked`：Agnes TS/Go/Rust 题无 non-empty candidate，Pier editable package 失效且本机无容器 runtime；DSW-06 最小写入探针已通过，证明 Lime current `apply_patch` 写链可用。剩余交付是 EVAL-01、正式 macOS RC、真实 Windows receipt、N-1 updater 与冻结候选复跑，不再无差别重复 Agnes 0-byte trial。

## 32. 2026-07-17 Windows N-1 updater 缺陷与实证入口

- Electron 官方合同明确 available update 会自动下载，重复调用 `autoUpdater.checkForUpdates()` 会下载两次。真实入口审计证明 About 与 Sidebar 都会自动检查，用户随后启动安装时，`ElectronUpdateHost` 在 `downloading` 阶段仍再次检查；并发安装请求还会安排两次 `quitAndInstall()`。owner 回归先稳定复现 `checkForUpdates=2`、`quitAndInstall=2`，current 状态机现让 checking 复用在途 promise、downloading 等待既有 `update-downloaded`、completed 直接安装、installing/restarting 幂等返回。owner `8/8` 与 Electron typecheck 通过。
- `windows-squirrel-rc-smoke.mjs` 新增真实 N-1 模式，CDP/feed 驱动拆到 `scripts/electron/lib/windows-squirrel-n-minus-one.mjs`，避免 926 行单文件继续堆叠。runner 精确选择低于候选的最近稳定 tag，安装 GitHub Release N-1 Setup，通过仅监听 `127.0.0.1` 的候选 `RELEASES + full.nupkg` feed 驱动 N-1 preload/IPC/current updater，等待 completed、restarting、旧进程退出与 `app-<candidate>/Lime.exe` 落盘，再用候选 packaged executable 运行 `SHELL-01`。summary 只有六项 N-1 观测全真才写 `passed`。
- 手工 Windows workflow 还残留已不存在的 `package-lock.json + npm ci`，会在进入平台测试前失败；该 dead 安装路径已直接替换为仓库 current `pnpm-lock.yaml + pnpm install --frozen-lockfile`。手工 workflow 与 release Windows matrix 都会下载 N-1 Setup 并传入隔离候选 feed；release governance 阻止回退为单版本 smoke。
- 本地确定性验证为 updater owner、Windows runner 与 release guard 合计 `42/42`，`typecheck:electron`、`governance:electron-release-workflow`、`governance:scripts` 与 diff check 全绿。当前机器仍是 macOS，且工作树未提交/推送，不能触发包含本变更的 Windows workflow；因此 `PLT-02` 仍为 `in_progress / evidence pending`，第二期完成度保持 `80%`。下一刀是对冻结候选运行 Windows workflow，并以首个真实失败 owner 继续修复。

## 33. 2026-07-17 DSW-06 Agnes apply_patch 写链归因

- 修复前基线 `.lime/benchmark/v2/probes/20260717T091520Z-agnes-apply-patch-probe-before` 使用隔离临时 git workspace、Agnes `agnes-2.0-flash`、4 provider steps / 40,000 budget tokens。每次 request 的真实 tool catalog 都包含 `apply_patch`；第一次调用因模型发送没有 ` ` / `-` / `+` 行前缀的 update hunk 被 `tool-runtime` 正确拒绝，第二次调用 `- before` / `+ after` 被工具正确执行，但最终文件为 ` after\n`，patch 359 bytes。turn 产生真实 `patch.failed` 后 `patch.applied`，最后一步仍是读取，runtime 因 step cap 生成 max-turn completed 文案；旧 adapter 会把它当成 completed candidate。
- 归因结论：Lime patch parser/执行器没有把合法 patch 改坏；缺陷在 provider tool definition 的 schema 只有 marker 说明，没有 Codex 对齐的可执行行前缀示例，模型把 diff 内容空格当成文件内容。旧未接入 current runtime 的 shell prompt 资产同时是 dead，已删除并移除 agent export，避免恢复错误调用形态。
- current 修复：`tool-runtime::apply_patch_tool_definition` 明确 patch 字段、hunk 行前缀和 `-before`/`+after` 精确样例，并补 owner 回归；DeepSWE adapter 增加 `providerStepExhaustion`，最后一个 provider step 为 `tool_call` 时 fail closed，保留自然 `stop` 作为成功终态。
- 修复后 `.lime/benchmark/v2/probes/20260717T104119Z-agnes-apply-patch-probe-after` 经新构建 App Server stdio current chain，4 provider steps 内产生 `patch.applied`、精确 `after\n`、358-byte patch，catalog 每次含 `apply_patch`，budget reasons 为空，summary `passed=true`。这证明 Lime current 写链已闭环；完整 DeepSWE 题仍因 Agnes 预算内只读探索与 Pier 双重环境 blocker 无有效分数。
- 验证：`cargo test --manifest-path lime-rs/Cargo.toml -p tool-runtime runtime_apply_patch`（4/4）；`cargo test --manifest-path lime-rs/Cargo.toml -p lime-agent prompt_assets`（5/5）；`vitest run scripts/harness/deepswe-adapter.test.mjs scripts/harness/deepswe-coding-slice.test.mjs`（26/26）；App Server 重建；修复后真实 Agnes DSW-06 通过。未执行真实 Pier 分数验证，原因是本地 editable package 指向已删除 source 且无容器 runtime。

## 34. 2026-07-17 DeepSWE 当前链复测收口

- Agnes `20260717T111006Z-happy-dom-abort-pending-body-reads` 和 gpt-5.5 `20260717T111350Z-happy-dom-abort-pending-body-reads` 都使用仓库外系统临时 workspace、stdio App Server、`maxOutputTokens=4096`、`enableThinking=false`、8 step / 80,000 token 诊断预算。Agnes 8/8 request catalog 都含 `apply_patch`，仅做读取探索并以 `provider_steps` 终止；gpt-5.5 6 step 后累计 97,164 tokens，以 `token_budget` 终止。两者 patch 都为 0 字节。
- 为排除题目复杂度影响，Agnes 又运行 Smoke 10 短题 `superjson-error-stack-serialization`（`20260717T112458Z-superjson-error-stack-serialization`）。5 个 provider step、81,950 budget tokens 后由 runtime 取消；5/5 catalog 含 `apply_patch`，实际 lifecycle 只有 Glob/Grep/file artifact，没有写调用，patch 为 0 字节。
- 三次 evidence 的 Thread/Turn/Item、provider-steps、tool-lifecycle、trajectory、App Server evidence 与 failure classification 均可复核；未出现新的 Lime owner 失败。唯一模型侧异常是 Agnes 对不存在的 `async-task-manager/index.ts` 读取失败后自行通过目录探索恢复。
- 归因：当前 `apply_patch` 写链、工具目录投影、预算 enforcement、terminal/cancel 和 evidence capture 继续闭环；Agnes 在当前路由/预算内只读探索仍是模型吞吐问题。不能将这些 0-byte run 标为 Lime defect，也不能在 Pier editable source 失效和本机无容器 runtime 时生成分数。
- 当前阶段不再无差别重复同类 Agnes 题；DSW-02 保持 diagnostic-only，待 Agnes 路由或 verifier 环境变化后再恢复 scoring。第二期整体完成度维持 `80%`，下一刀仍是 Windows 真实 receipt、正式 macOS RC、EVAL-01 和冻结候选 SOAK 复跑。
- 验证：`npm run harness:deepswe:preflight` 20/20 task 通过；focused adapter budget/generation/verifier 回归 5/5；完整 `deepswe-adapter.test.mjs` + `deepswe-coding-slice.test.mjs` 在 `--testTimeout 120000 --hookTimeout 120000` 下 26/26。默认 20 秒 timeout 在本机隔离 git case 上会误报，未改变测试断言。
- dead 回流扫描又发现 `internal/research/agent/lime-agent-verification-plan/07-flag-differential-harness.md` 仍把已删除的 `agent-qc:benchmark:plan/compare` 和旧 manifest 作为现有入口。该专用 research 页已物理删除，Agent research 导航、90 天路线图、进度记录和方案稿已改为引用 `internal/roadmap/benchmark/**` current 事实源；DeepSWE adapter 的 retired-path guard 同时禁止文件、npm script、旧 manifest 引用和导航链接回流，focused 回归 1/1。

## 35. 2026-07-17 EVAL-01 Agnes Gate B 归因与 sidebar identity 修复

- 首次真实 `claw-chat-ready-streaming` 使用 Agnes `custom-637ea2d5-e430-43de-86de-39c5f1735438 / agnes-2.0-flash`。长流、中断、App Server `agentSession/turn/cancel`、同 session recovery turn completed 与 runtime/read model persistence 均成立；初次失败发生在 GUI 重开目标会话，证据为 `session-entry-missing`，但 App Server 已确认 recovery 文本持久化。
- 盘点四层后确认：`agentSession/list` 在发送热路径被延迟 30 秒，sidebar 没有为 fresh `created` session 保留 identity；仓库已有多个相同标题会话，smoke 的标题 fallback 错点旧 session。该问题是 current Renderer session navigation/read-model projection 缺口，不是 Agnes 生成问题。
- current owner 修复：`useAppSidebarSessions` 收到 `reason=created` 立即插入轻量占位项并保留后续真实 list merge；`AppSidebarConversationRow` 投影 `data-session-id`；smoke helper 删除标题 fallback，只按稳定 id 定位。新增/更新 sidebar 回归 5 项通过，`node --check scripts/claw-chat-ready-streaming-smoke.mjs` 通过。
- 修复后 Agnes `phase2-eval01-agnes-rerun` 证明 recovery 已进入正确 session，但当时仍因旧列表/标题 fallback把 live turn 发到上一 run 的 session；`phase2-eval01-agnes-final` 与 `phase2-eval01-agnes-stable` 均已证明 live turn 使用本次新 session，并在同一 session 产生非空 WebSearch/WebFetch tool call/result。两次都在 WebFetch 后 120 秒未出现 `turn.completed`；stable run 同时记录共享 `run-dev` watcher 的 `app-server host is stopping` / `ERR_CONNECTION_REFUSED`，不能计为 score。
- 归因：sidebar identity 缺口已关闭；Agnes live tool loop 当前为 `diagnostic / provider-or-host pending`，没有新的 Lime runtime/tool owner failure。EVAL-01 保持 `in_progress / evidence pending`，不伪造 pass@k、pass^k 或 baseline。
- 验证：`npx vitest run src/components/app-sidebar/AppSidebarConversationRow.test.tsx`（2/2）；sidebar 事件/创建占位/热路径相关定向（3/3）；`node --check scripts/claw-chat-ready-streaming-smoke.mjs`；Agnes 真实 Gate B 证据目录 `.lime/qc/gui-evidence/claw-chat-ready-streaming/phase2-eval01-agnes-{rerun,final,stable}-*`。当时全文件 sidebar run 暴露的本地导入 toast 异步失败，后续已在第 37 节按 current job fixture 合同修复并由 Sidebar/API 合计 64/64 回归关闭。
- 下一刀：冻结并提交 Windows 真实 receipt、正式 macOS RC 与 DeepSWE verifier 环境；EVAL-01 只在 Agnes 产生稳定工具/terminal evidence 后冻结 baseline，不再把 gpt-5.5 对照或模型超时伪报为 Agnes 通过。

## 36. 2026-07-17 EVAL-01 隔离 Host 与 gpt-5.5 对照收口

- 隔离命令使用 `LIME_ELECTRON_APP_SERVER_WATCH=0`、固定 3030 DevBridge、真实 `APP_SERVER_BACKEND_MODE=runtime` 和现有 Agnes provider 配置；renderer、Electron Desktop Host、App Server JSON-RPC、RuntimeCore、read model、GUI 均为 current 链，没有 fixture/mock fallback。
- Agnes `phase2-eval01-agnes-isolated-3030` 先证明长流首增量、stop、cancel、同 session recovery、WebSearch/WebFetch tool call/result、terminal、read-after-event 和 GUI recovery 全部成立；原 `verdict=fail` 只因首次空任务的 provider/model 位于 `metadata.harness.model_request_policy` 而非顶层 runtimeRequest，属于 smoke 观察器误判，已由 `runtimeRequestProviderModel` 按有效 policy 读取并记录来源。
- 观察器修复后的 Agnes `phase2-eval01-agnes-isolated-final` 与 `phase2-eval01-agnes-isolated-recheck` 均在无 watcher Host 下复现：长流、中断和 recovery 通过；WebSearch/WebFetch 没有 tool event，turn 120 秒后由 public `agentSession/turn/cancel` 收敛为 canceled。两次无 blocking console error、无 runtime mock fallback，不能归因 Lime owner，也不能冻结 Agnes baseline。
- 固定对照 `phase2-eval01-gpt55-isolated` 使用 `custom-1ae93b42-e57f-4a83-ac6e-3f5275a7b376 / gpt-5.5`，同一 3030 Host 下 `verdict=pass`：同一 session 的长流/cancel/recovery、WebSearch/WebFetch read-model output、`turn.completed`、read-after-event、GUI 正文和 provider/model routing 全部满足，console blocking error 与 runtime mock fallback 为 0。该结果将 Lime current 链与 Agnes live tool-loop 行为区分开，但不替代 Agnes baseline。
- 本轮 smoke 观察器验证：`node --check scripts/claw-chat-ready-streaming-smoke.mjs`、`npx prettier --check scripts/claw-chat-ready-streaming-smoke.mjs` 通过。证据目录为 `.lime/qc/gui-evidence/claw-chat-ready-streaming/phase2-eval01-agnes-isolated-{3030,final,recheck}-*` 与 `phase2-eval01-gpt55-isolated-*`。

## 37. 2026-07-17 本地历史导入异步 job fixture 收口

- Sidebar 全文件回归唯一失败用例为“项目范围导入本地历史对话确认后应带 `confirmed=true` 并打开导入会话”。盘点确认生产 current owner 已按 `thread/commit -> job/read -> completed.result` 编排，失败来自共享 `AppSidebar.testFixtures.tsx` 仍返回已退役的同步 `ConversationImportThreadCommitResponse`，且测试模块没有提供 `readConversationImportJob` / `waitForConversationImportJob`。
- fixture 已直接迁移到 current protocol：commit mock 返回 `ConversationImportThreadCommitStartResponse.job`，job 为 completed 并携带 canonical `result`；wait mock 只接受 completed job 并返回其 result。没有修改生产导入逻辑，也没有删掉 toast、放宽断言或加入固定 sleep。
- 定向验证：`npx vitest run src/components/AppSidebar.conversations.test.tsx src/lib/api/conversationImport.test.ts`（64/64），其中组件回归明确断言 `completed job -> canonical result -> toast/navigation` 顺序；导入 Electron guard 15/15、progress/ViewModel/row 回归 17/17；`npx prettier --check src/components/AppSidebar.testFixtures.tsx`；`git diff --check`。API 层仍保留后台 job 缺失时 fail-closed、job/read 终态和 abort 语义回归。
- 分类：异步导入 job、`job/read` 和 canonical result 为 `current / closed`；共享 fixture 的同步 commit response 与缺失 job observer 为 `dead / deleted / forbidden-to-restore`，不得恢复旧 wrapper 或用测试侧 sleep 模拟完成。
- 这次收口移除了一个误报的本地 blocker，但不改变第二期整体完成度（仍为 `80%`）：Windows 真实 receipt、正式 macOS Developer ID/notarization/DMG RC、DeepSWE verifier 环境和 Agnes EVAL-01 稳定 baseline 仍未完成。下一刀回到 Windows workflow / macOS RC 或 verifier 环境，不再重复该导入失败。

## 38. 2026-07-17 legacy report 并行写入瞬时违规复核

- `npm run governance:legacy-report` 复跑稳定通过：扫描 2394 个源码文件与 1470 个测试文件，零引用候选 `0`、分类漂移候选 `0`、边界违规 `0`。直接调用 `buildLegacySurfaceReport()` 对 `rust-runtime-file-checkpoint-inline-content-leak` 也得到 `violations=[]`。
- 先前同一命令曾在并行改写 Codex history import 文件期间报告 `history_builder.rs`；当前文件不再命中 `previousContent/beforeContent/oldContent` 规则，未添加白名单或放宽 guard。该结果证明治理守卫本身有效，属于并行写入竞态的历史 evidence，不是当前 blocker。
- 分类：治理目录扫描与 file-checkpoint sidecar 约束为 `current / closed`；历史瞬时违规不转成 `compat` 或 `deprecated` 例外。当前第二期剩余 blocker 仍是 Windows 真实 receipt、正式 macOS RC、DeepSWE verifier 和 Agnes EVAL-01 稳定 baseline。

## 39. 2026-07-17 Renderer projection 运行态与 artifact 语义收口

- projection 定向回归首次暴露 9 条真实失败，均来自 current Renderer projection，而不是测试夹具：未提供 `activeCurrentTurnId` 时，正在发送或恢复中的 turn 被误判为历史细节，导致 thinking/tool/streaming overlay 丢失；失败 turn 的友好 fallback 又被历史过滤覆盖回孤立标点；compact timeline 过滤把相对路径 `file_changes_batch` 当成 canonical `file_artifact` 并删除；patch timeline 把结构化 `file_changes_batch` 排在最终正文之后。
- current owner 修复：只有明确存在且不匹配的 active turn，或非发送态的 terminal turn，才隐藏历史运行细节；失败 fallback 在历史过滤前保留；timeline artifact 不再用 batch path 做过宽删除，保留过程汇总和 canonical 结果卡；timeline-owned patch content 先投影 `file_changes_batch`，再投影文本。
- 这组修复没有放宽断言或改成静态快照：`messageListItemProjection` 四个 owner 测试文件共 `42/42` 通过，覆盖运行中 web/tool、streaming final、reasoning 合并与替换、历史恢复、失败 fallback、apply_patch diff、相对/绝对 artifact path；renderer `tsc --noEmit` 和 owner ESLint 通过。
- 分类：active/historical turn projection、runtime failure fallback、file artifact/process distinction 与 patch ordering 为 `current / closed`；“无 active id 即一律隐藏细节”、按原始字符串误删 canonical artifact、正文优先于结构化 diff 的旧投影均为 `dead / deleted / forbidden-to-restore`。本轮仍不改变第二期整体完成度（`80%`）；Windows 真实 receipt、正式 macOS Developer ID/notarization/DMG RC、DeepSWE verifier 与 Agnes EVAL-01 稳定 baseline 仍是剩余 blocker。
- 验证命令：`npx vitest run src/components/agent/chat/components/messageListItemProjection.unit.test.ts src/components/agent/chat/components/messageListItemProjection.contentParts.unit.test.ts src/components/agent/chat/components/messageListItemProjection.timeline.unit.test.ts src/components/agent/chat/components/messageListItemProjection.artifacts.unit.test.ts`；`npx eslint --max-warnings 0 src/components/agent/chat/components/messageListItemProjection.ts`；`npx tsc --noEmit --project tsconfig.renderer.json`。

## 40. 2026-07-17 DeepSWE adapter TOML 环境 blocker 收口

- `verify:local` 的 smart Vitest 第 11 批首次暴露 4 个 DeepSWE adapter 失败。共同根因是 `readTaskToml` 通过 Python 3.11 `tomllib` 读取 task metadata；当前 macOS Python 没有该标准库，导致 Release 20 preflight、未授权 fail-closed 和 verifier-only product failure 保留测试在进入各自断言前全部失败。该问题是 adapter 环境依赖，不是 task source、Lime runtime、Agnes 或 Pier verifier 失败。
- current adapter 改用仓库已锁定的 Node `smol-toml` 结构化解析器，移除 Python subprocess 和版本假设；未引入 ad-hoc 行解析，也未改变 DeepSWE task schema、workspace 隔离、live authorization 或 verifier ownership。
- 修复后验证：`npx vitest run scripts/harness/deepswe-adapter.test.mjs scripts/harness/deepswe-coding-slice.test.mjs` 为 `26/26`；`npm run harness:deepswe:preflight` 为 Release 20 `61/61`；adapter ESLint、`npm run governance:scripts`、`npm run test:contracts` 全部通过。
- 分类：Node TOML metadata parser 和 adapter fail-closed/verifier evidence 为 `current / closed`；Python `tomllib` 运行时假设为 `dead / deleted / forbidden-to-restore`。完整 DeepSWE 仍是 `diagnostic_true_runs_blocked`：Agnes 选题固定预算内无 non-empty candidate，Pier editable package 已失效且本机无 Docker/Podman/nerdctl/Colima，不能生成 `reward.json`、`ctrf.json` 或 score；第二期整体完成度保持 `80%`。

## 41. 2026-08-04 Benchmark 新一轮 Gate B 验收

- 目标与写集：复跑 `internal/roadmap/benchmark` 的 current fixture、恢复、cancel/continue、approval/resume、unknown Item、AgentControl cold-restart、`home-hotpath` 和 `SOAK-01`，只新增 `.gstack/benchmark-reports/2026-08-04-benchmark.{md,json}` 与本轮失败证据；未修改业务源码。
- 真实 Electron 性能观测：通过 `http://127.0.0.1:9223` 确认 Lime/Electron 42.3.3、`window.__LIME_ELECTRON__=true` 和 `electronAPI.invoke`；reload 的 TTFB/FCP/DOM Interactive/Full Load 为 `49/412/316/1424ms`，250 个请求，transfer `505873` bytes，reload console/page error 均为 0。
- Gate B 通过：history/cache `31/31`、streaming completion `32/32`、fixture guard `99/99`；session `thread/start/read/list/resume`、cancel/continue、approval/resume、unknown Item 均为 current Electron/App Server/read model/GUI 闭环；AgentControl cold-restart 39 条 assertion 全绿，证据分别见 `.lime/qc/gui-evidence/benchmark-20260804/` 和对应 current fixture 目录。
- `home-hotpath` 保持失败：250ms budget 下 `homeInputToSendDispatchMs` 两次为 `287ms`、`274ms`，7 月成功基线为 `100ms`、`114ms`；provider wait `90ms`、renderer 首次 delta apply `2ms`，turn/read model/GUI/Electron IPC 成功，invoke/page/console error 为 0。归因仍是 Renderer 首发 dispatch 性能回归，不是 provider、IPC、Runtime 或 terminal failure。
- `SOAK-01` 未通过：10 轮 / 2 次 cold restart 在首轮失败，唯一失败断言为 `waitAgentReturnedTerminalResult`；其他工具、turn terminal 和 `thread/read` 完成。单轮 AgentControl cold-restart 后续通过，不能替代完整 soak receipt。
- 最后一次 cold-restart recheck 在 `launch-electron` 失败：`canvasImageInsertHistory-BQ3-T3zp.js` 不提供导出 `c`，bridge 未就绪；该证据归类为构建产物/启动环境不一致，需清理并一致构建后复测，当前不直接归因 runtime owner。
- Gate B 结论：`partial evidence / failed acceptance`。核心链路已证明，但 `home-hotpath`、`SOAK-01` 和 cold-restart recheck 仍是 release evidence blocker；下一刀为 dispatch 性能定位、构建产物一致性复测和 SOAK-01 10x2 完整重跑。

## 42. 2026-08-04 Benchmark Gate B 修复后复验

- 目标与写集：在第 41 节失败现场上继续复验 `home-hotpath`、AgentControl cold-restart 和 `SOAK-01`；写集为 SOAK 观察器、观察器单测、已有 Renderer dispatch/构建一致性修复及 `.gstack/benchmark-reports/2026-08-04-benchmark.{md,json}`，未修改 Rust RuntimeCore 生产逻辑，未放宽投影 repair 的 fail-closed 断言。
- SOAK 观察器根因：v2 `thread/list` 的 `Thread` 主键字段是 `id`，不是 `threadId`；旧观察器因此把真实 canonical thread 误判为未列出。请求同时从旧 `includeArchived` 对齐到 v2 `archived: false`，断言现在要求 `id + sessionId` 精确匹配，并由 `tool-execution-soak-evidence.test.mjs` 的 10 条单测覆盖。
- `home-hotpath` 修复后真实 Electron Gate B 通过：`homeInputToSendDispatchMs=37ms`（预算 250ms），pending preview paint `63ms`，first text paint `543ms`，provider wait `90ms`，renderer 首次 delta apply `1ms`，first delta 到 paint `23ms`；Electron/preload/IPC/App Server/read model/GUI 闭环成功，console/page error 为 0。证据：`.lime/qc/gui-evidence/benchmark-20260804/benchmark-gateb-home-hotpath-after-observer-fix-20260804-summary.json`。
- 一致 renderer 重建后 AgentControl cold-restart 独立复验通过，旧 `canvasImageInsertHistory` export mismatch 未复现；证据：`.lime/qc/gui-evidence/benchmark-20260804/benchmark-gateb-agent-control-cold-restart-after-renderer-rebuild-20260804.json`。
- `SOAK-01` 10 轮 / 2 次 cold restart 通过：`allRoundsPassed`、`roundSessionsIsolated`、`readModelsStableAcrossColdRestarts`、`everyPreviousProcessTreeExited`、`finalProcessTreeExited` 全为 `true`；总 RSS `561,312 -> 442,768 KiB`（delta `-118,544 KiB`），App Server RSS delta `+8,976 KiB`，10 轮 duration `2,820-5,371ms`。证据：`.lime/qc/soak-01/agent-control-soak-final.json`。
- 验证：SOAK evidence 单测 `10/10`、单轮 cold-restart、10x2 SOAK、home-hotpath 及一致构建后的 cold-restart 全部通过；未执行 commit/push/reset/分支操作。
- 分类与结论：v2 `thread/list` 字段归一化和 SOAK 观察器为 `current / closed`；旧 `threadId` 观察字段与 `includeArchived` 请求为 `dead / deleted / forbidden-to-restore`；历史失败 JSON/截图保留为诊断 evidence。当前本轮 Gate B 为 `passed`，但本地 controlled fixture 不能替代 live provider、Windows RC 或正式签名/notarization 证据。
