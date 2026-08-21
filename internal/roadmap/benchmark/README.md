# Refactor v2 测试与 Benchmark 第二期

> status: active / DeepSWE scoring, desktop coding Gate B, and platform receipts pending
> owner: quality-workflow + runtime domain owners
> last_verified: 2026-08-21
> refactor_source: `internal/research/refactor/v2/**`
> codex_commit: `5c19155cbd93bfa099016e7487259f61669823ff`

## 1. 结论

重构前的 Benchmark 体系已经退役。旧体系把 release checklist、外部数据集下载、脚本 dry-run 和一次性报告拼成版本门禁，但它没有以 Refactor v2 的唯一产品链为测试对象，不能证明 current runtime 正确：

```text
Electron Desktop Host
  -> App Server JSON-RPC
  -> RuntimeCore
  -> Thread / Turn / Item projection
  -> GUI
```

第二期从零建立测试基线，不继承旧 runner 的通过状态。旧 `benchmark-release-v1`、旧 DeepSWE/Terminal-Bench true-run、Managed Objective differential 和历史 progress 只存在于 Git history。DeepSWE 本身继续使用，但改为 [v2 Coding 切片](./deepswe-coding-slice.md) 和新的 App Server current adapter。

## 2. 第二期目标

1. 用确定性集成测试守住 Agent loop、公共 JSON-RPC、状态机、持久化和恢复。
2. 用 current fixture 守住 provider/tool/runtime 的真实事件链，不靠生产 mock fallback。
3. 用 Gate A 验证 Renderer projection，用 Gate B 验证真实 Electron 产品链。
4. 把 live provider、随机性评估、外部 benchmark 与确定性 release gate 分开。
5. 每个测试都有明确 owner、风险、证据等级和失败归属；不再用“跑了很多脚本”代替覆盖证明。
6. 对 coding agent 同时给出独立 verifier 的代码正确性证据与真实 Electron Gate B 的桌面产品证据；任何单边通过都不能写成桌面 coding 已交付。

## 3. 文档索引

| 文档                                                                                                             | 作用                                                    |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| [phase-2-test-plan.md](./phase-2-test-plan.md)                                                                   | 第二期实施切片、顺序、owner、退出条件和退役账本         |
| [scenario-matrix.md](./scenario-matrix.md)                                                                       | v2/Codex 对齐的稳定场景 ID、证据等级与优先级            |
| [deepswe-coding-slice.md](./deepswe-coding-slice.md)                                                             | DeepSWE Smoke 10 / Release 20 选题、adapter、指标与门禁 |
| [../../aiprompts/quality-workflow.md](../../aiprompts/quality-workflow.md)                                       | 仓库级最低门禁与交付规则                                |
| [../../test/testing-strategy-2026.md](../../test/testing-strategy-2026.md)                                       | 测试分层、作者合同、CI lane 与 evidence 规则            |
| [../../research/refactor/v2/13-evidence/verification.md](../../research/refactor/v2/13-evidence/verification.md) | v2 历史切片的验证合同和不可变 evidence                  |
| [../../exec-plans/refactor-v2-test-phase-2-plan.md](../../exec-plans/refactor-v2-test-phase-2-plan.md)           | 本轮重建执行记录                                        |

## 4. 测试层级

| 层级               | 主要对象                                                  | 证据                                    | 是否阻断发布            |
| ------------------ | --------------------------------------------------------- | --------------------------------------- | ----------------------- |
| L0 静态与治理      | schema、generated client、依赖、旧路回流                  | lint/typecheck/contracts/governance     | 是                      |
| L1 纯单元          | parser、selector、projection、lowering、状态转换          | 整对象断言、表驱动 case                 | 是，按受影响范围        |
| L2 领域集成        | agent-runtime、tool-runtime、model-provider、thread-store | 真实 domain owner + 可控依赖            | 是                      |
| L3 App Server 集成 | public JSON-RPC、notification、read model、恢复           | client round trip + 结构化事件          | 是                      |
| L4 current fixture | RuntimeCore、provider fixture、工具、MCP、multi-agent     | current event stream + terminal state   | 是，Agent 主路径        |
| L5 Gate A          | Renderer projection、DOM、交互、五语言                    | browser fixture + 可见状态              | GUI 变更时是            |
| L6 Gate B          | Electron/preload/IPC/App Server/runtime/read model/GUI    | 真实 Electron + 同一 identity 链        | 主路径/桌面边界变更时是 |
| L7 live/eval       | live provider、非确定性任务、外部 benchmark               | transcript、grader、pass@k/pass^k、成本 | 显式 release lane 才是  |
| L8 platform/soak   | macOS/Windows、packaged app、并发、长稳                   | 平台矩阵、资源与恢复报告                | release candidate 是    |

L0-L6 主要验证确定性正确性；L7 验证模型能力和稳定性；L8 验证交付环境。三者不能互相替代。

## 5. Codex 对齐规则

- Agent 逻辑变更优先写集成测试，而不是只补内部函数单测。
- App Server 测试必须经过公共 JSON-RPC API；不直接调用 handler 私有实现伪造成功。
- Provider 使用可控 response server/fixture，保存并断言结构化 request、tool output 和 terminal event。
- 等待业务事件，不用固定 sleep 或 grace timer 合成完成态。
- 优先比较完整对象和完整投影；字段逐个断言只用于解释关键差异。
- 不测试静态常量，不为已经删除的实现继续写正向或行为测试。
- GUI 可见变化必须有稳定 DOM/快照证据；Lime 的最终产品证据仍是 Electron Gate B，而不是 Codex TUI snapshot。
- 测试资源和首方 binary 必须通过仓库统一 helper 定位，不能依赖当前工作目录偶然成立。

## 6. Release Lane

| Lane              | 触发                            | 必须包含                                                                 |
| ----------------- | ------------------------------- | ------------------------------------------------------------------------ |
| PR related        | 每次改动                        | L0 + 受影响 L1/L2；命令边界加 contracts                                  |
| Runtime current   | Agent/runtime 改动              | L2 + L3 + `smoke:agent-runtime-current-fixture`                          |
| GUI current       | GUI/Workspace 改动              | related tests + Gate A + `verify:gui-smoke`                              |
| Desktop current   | Electron/bridge/read model 改动 | contracts + 对应 Gate B fixture                                          |
| Nightly           | 每夜或定期                      | 扩展 L2-L6、恢复/故障注入、flaky/retry 统计                              |
| Release candidate | 版本候选                        | L0-L6 全部 required 场景 + L8 平台矩阵                                   |
| Live quality      | 显式授权                        | L7 provider/eval 与 DeepSWE coding slice；记录模型、配置、成本与样本版本 |
| Desktop coding    | coding/runtime/Workspace RC     | DeepSWE Desktop Smoke 5；真实 Electron Gate B 与 Pier verifier 双通过    |

外部公开 benchmark 只进入 `Live quality`、`Desktop coding` 或研究 lane。没有 current adapter、固定环境和可复核 grader 前，不得进入 release required。DeepSWE Core 负责长程代码正确性，Desktop Smoke 负责 Electron 产品链；SWE-bench-Live MultiLang/Windows、SWE-bench Multimodal 与 Terminal-Bench 2.1 只能补覆盖，不替代 DeepSWE。OSWorld/WindowsAgentArena 仅在产品声明 computer-use 能力时启用，不能把通用桌面操作分数冒充代码 verifier 分数。

## 7. 当前状态

- `current`：仓库现有 related tests、Rust layer runner、App Server contracts、current runtime fixture、Gate A/B smoke。
- `current`：DeepSWE adapter v6 使用仓库外系统临时 workspace，经 App Server current chain 投影逐步 provider usage、每次真实 sampling 的 tool catalog 与单次 run generation controls；`agent-runtime` 在工具执行和下一次 sampling 前执行 step/token budget，adapter token polling 只作 timeout race fallback；wall timeout 会先取消并等待真实 terminal。gpt-5.5 只在区分 Lime 与模型问题时作固定对照。
- `current / closed`：DSW-07 已将 source cache 固定到 `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`，113 题统一为 schema `1.3`；agent/verifier 均为 `no-network`，`verifier.collect` 统一生成 `/logs/artifacts/model.patch`，`LICENSE`/`PROVENANCE.md` 进入 preflight。Core Release 20 为 `205/205`、Desktop Smoke 5 为 `53/53` 检查通过；隔离 Python 3.12 工具目录的 `datacurve-pier==0.3.1` 版本检查通过。
- `current / incomplete`：DSW-08 已实现 `scripts/harness/deepswe-benchmark.mjs` 的 Smoke 10/Release 20 计划、live batch 编排和三 trial 聚合；`pass@1` 按标准 `c/n` 估计，`pass@3`/`pass^3`、wall time、budget token、失败 owner 与 identity 计数均只从当前 source/schema/adapter 的 verifier-complete trial 计算。Smoke 10 plan 为 `105/105` ready；历史 adapter run 和 batch summary 已清理，Release 20 aggregate 当前 `observedRunCount=0`、`scoreEligible=false`、所有 score 为 `null`。当前仍无 live provider、Pier artifacts 或 score，第二期保持 `80%`。
- `current / incomplete`：本地不再保留 DeepSWE 旧 run、patch 或 verifier 产物；历史诊断证据已按确认删除。当前结果只能证明 adapter/preflight 合同，不能证明 DeepSWE 能力基线或桌面 coding。
- `planned / required`：DeepSWE Desktop Smoke 5 从 TS、Go、Python、Rust、JavaScript 各取一题，通过真实 Electron GUI 输入原始 instruction，观察 preload/IPC/App Server/runtime/read model、审批/取消/恢复、文件 diff 与 terminal，再把同一 patch 交给 Pier separate verifier。受控五题 Gate B 已完成，但 `DesktopCodingPass` 仍只在 live Gate B 与 verifier 同时通过时为真。
- `current / controlled-product-path`：Desktop Smoke 5 manifest、统一 trial evidence/verdict 合同和受控 runner 已落库：`internal/test/deepswe-desktop-smoke-v1.json`、`scripts/harness/deepswe-desktop-contract.mjs`、`scripts/harness/deepswe-desktop-controlled-smoke.mjs`。2026-08-19 fresh suite `.lime/benchmark/v2/desktop/controlled-artifact-final/20260819T022022Z/summary.json` 已让五题全部通过真实 Electron bring-up、原始 instruction、外部隔离 workspace、Read/Glob/Grep/apply_patch/exec_command、native test、Thread/Turn/Item、GUI terminal/diff、artifact 正文预览、session reopen、approval resume、cancel no-ghost-write 与零 mock/invoke/console/page error；五题均命中 `artifact/read`，且不再出现 `App Server artifact 内容不可用`。该 lane 明确不生成 DeepSWE score，`verifier=not_run` 时 `desktopCodingPass=false`。
- `current / fail-closed`：`deepswe-desktop-benchmark.mjs` 只接受 `deepswe-desktop-trial-v1`，并要求 task/base commit/instruction/workspace、同一 trial identity、非空 patch、工具阶段、terminal/read model、GUI identity、mock/error、Pier `reward.json`/`ctrf.json`/`test-stdout.txt` 与 patch SHA 全部一致；controlled provider 或缺失 Pier/container 只能得到 `product_path_only`/`incomplete`，不能拼接成桌面 coding pass。2026-08-21 聚合仍明确为 `product_path_only`，5/5 controlled Gate B 通过但 `liveTrialCount=0`。
- `planned / external coverage`：SWE-bench-Live MultiLang/Windows 补滚动污染与 Windows/PowerShell，SWE-bench Multimodal 补视觉 issue，Terminal-Bench 2.1 补 shell/环境诊断。外部 source、任务集、镜像和 grader 必须固定版本，基础设施失败使批次无效而不是计模型失败。
- `deprecated`：现有 Agent QC/Harness manifest 只作为第二期场景迁移输入，未逐项映射到 [scenario-matrix.md](./scenario-matrix.md) 前不算新门禁。
- `dead`：旧 Benchmark runner、旧 release manifest、旧外部数据集 wrapper、旧 progress 文档和专用 flag differential research 页，已删除；DeepSWE 数据集不属于 dead。
- `current / closed`：原 3 条 session/projection Rust blocker 和 Gate B queue/restore identity 已有定向回归与真实 Electron 通过证据；App Server orphan child restart 也已补 fail-closed cleanup。
- `current / closed`：provider 每步 usage 丢失、multi-step usage 只保留最后一步和 timeout/cancel usage 丢失已在 current owner 修复；adapter v3 的 Agnes Go run 已验证 16/16 step usage、累计 272,324 budget tokens 和 current cancel 终态。
- `current / closed`：DeepSWE 真实 request tool catalog 与 runtime step cap 已闭环。历史 adapter v4 Agnes run `20260716T083349Z-go-genai-streamed-function-args` 严格产生 2 request / 2 completed step、`budgetCancellation=null`，两次均有完整 27 工具和 `apply_patch`；在当前 adapter v5 规则下，最后一步仍为 `tool_call` 的结果会标记为 `provider_steps` exhaustion，不再把 max-turn 文案冒充自然完成。
- `current / closed`：EVAL-01 Agnes Gate B 首次真实回合暴露 sidebar 在发送热路径 30 秒延迟窗口内不显示新建 session，重复标题会把恢复/后续回合误导到旧 session。current owner 现对 `created` 事件立即插入带真实 id 的轻量占位项，conversation button 投影 `data-session-id`，完整列表刷新仍按原延迟策略执行；sidebar owner 定向用例与实际 Agnes `claw-chat-ready-streaming` 重跑已证明长输出、中断、同 session 恢复和身份重开闭环。
- `current / diagnostic`：sidebar identity 缺口已由 current 占位 session + `data-session-id` + id-only smoke 定位并关闭。无 watcher 的 3030 Host 下，Agnes 两次复核都完成长流、中断、同 session recovery，但 WebSearch/WebFetch 回合无 tool event，120 秒后真实 cancel；固定 gpt-5.5 对照在同一 Host 完整 `pass`。该证据支持 provider/model 稳定性归因，不冻结 Agnes baseline、不计入 pass@k；证据前缀：`.lime/qc/gui-evidence/claw-chat-ready-streaming/phase2-eval01-agnes-isolated-{3030,final,recheck}-*`、`phase2-eval01-gpt55-isolated-*`。
- `current / closed`：DeepSWE runtime token budget 已闭环。旧 Rust run `20260716T113222Z-fd-deterministic-multi-key-sorting` 在累计 158,754 tokens 后仍启动 attempt 8；fresh run `20260716T120650Z-fd-deterministic-multi-key-sorting` 在 attempt 2 累计 15,065/12,000 tokens 后由 runtime 终止，attempt 2 返回的 4 个工具调用均未执行，也没有 attempt 3，`budgetCancellation.requestedAt=null`。
- `current / closed`：DeepSWE adapter 曾把 task TOML 读取绑定到 Python 3.11 `tomllib`，在当前 macOS Python 上先于业务断言失败，造成 preflight、live authorization 和 verifier-only 证据的假 blocker。current adapter 已改用锁定的 Node `smol-toml` 结构化解析；2026-08-19 fresh adapter/coding slice `24/24`、Release 20 preflight `205/205`、Desktop Smoke 5 preflight `53/53` 和 scripts governance 通过。Python 解析器假设已删除，不恢复；容器 runtime 缺失仍阻止 verifier 产生 score。
- `current / closed`：adapter v5 generation 对照已完成。Agnes run `20260717T031735Z-happy-dom-abort-pending-body-reads` 显式 `maxOutputTokens=4096`、`enableThinking=false`；8/8 step reasoning chars 为 0，累计 output 仅 443 tokens，执行 5 次命令和 2 次文件读取，8 次 request 均有 `apply_patch`，最终仍为 0-byte patch。runtime 在完成当前 sampling 后累计 89,046/80,000 budget tokens 并自主 canceled，第 8 step 的工具零执行、无 attempt 9。
- `current / closed`：DSW-06 最小 Agnes 写入探针先红后绿。修复前 `apply_patch` 第一次因模型缺少 hunk 行前缀失败，第二次虽成功但把 `- before/+ after` 的空格写入文件，turn 仍为 completed；修复 `tool-runtime` 的结构化 schema 示例和行前缀合同后，真实 stdio current chain `20260717T104119Z-agnes-apply-patch-probe-after` 在 4 step 内产生 `patch.applied`、精确 `after\n` 和 358-byte patch。未恢复旧 shell prompt 资产，该 dead 文件已物理删除。
- `current / closed`：2026-08-18 CodeMode CodeCell trace 对齐已接入唯一 App Server `TraceEventWriter + code_cell reducer/replay` owner。真实 Electron Gate B evidence `.lime/qc/gui-evidence/code-mode-electron-gate-b/code-mode-electron-gate-b-summary.json` 证明 `diagnostics/trace/read` 经 preload/IPC/App Server 读取完整 lifecycle，canonical `item_*` source/output identity、summary-only redaction、GUI terminal 和 zero mock/error 全部通过。对应确定性门禁：`tool-runtime` 356/356（SBX-01/02 与 apply_patch owner）、`agent-runtime` DSW-05 1/1、App Server 1686/1686、DeepSWE adapter 相关合同通过、Release 20 preflight 205/205；Rust fresh 复跑统一经过仓库 V8 artifact runner。
- `current / closed`：2026-08-19 approval resume/decline/cancel 的真实 Electron fixture 已统一到 `item/commandExecution/requestApproval` reverse response。request id 与 thread/turn 精确匹配，`serverRequest/resolved` 早于 runtime terminal，backend continuation 与 GUI/read model 终态完整，Renderer `agentSession/action/respond` 命中为零；定向回归 `86/86`，三类 Gate B 均通过。
- `current / diagnostic`：2026-08-19 fresh `verify:gui-smoke` 与 CodeMode Electron Gate B 通过，后者保持 21 项 assertion 全真、mock/invoke/console/page error 为零。聚合 `smoke:agent-runtime-current-fixture` 已通过 approval 全矩阵、Coding Workbench、Plan、Skills、MCP 与 media 场景，随后在并行 Expert Panel 改动处因 catalog candidate identity 从 `skill:capability-report` 漂移为 `skill:user:capability-report` 而失败；该失败不归 approval/DeepSWE owner，也不能写成聚合门禁全绿。
- `current / diagnostic`：本轮未运行 live provider，不生成 DeepSWE score；`datacurve-pier==0.3.1` 已在隔离工具目录可执行，但本机无 Docker/Podman/nerdctl/Colima，verifier 仍为环境 blocker。Developer ID package/DMG/ZIP 已有本机产物证据，但 notarization、DMG 安装/升级与 Windows packaged/N-1 仍等待平台 receipt。
- `current / diagnostic`：最新 Agnes/gpt-5.5 DeepSWE 复测均通过 Lime current App Server stdio 链完成证据闭环，但没有形成 candidate。Agnes `20260717T111006Z-happy-dom-abort-pending-body-reads` 为 8 step / 80,000 budget、8/8 catalog 含 `apply_patch`、0-byte patch、`provider_steps` 终止；gpt-5.5 `20260717T111350Z-happy-dom-abort-pending-body-reads` 为 6 step、97,164 budget tokens 后 `token_budget` 终止、0-byte patch；Agnes `20260717T112458Z-superjson-error-stack-serialization` 为 5 step、81,950 budget tokens 后 runtime 取消、0-byte patch。三次都没有 Lime owner 失败，不能计入 DeepSWE score。
- `current / closed`：PRV-04/ITM-05 deterministic 链已证明 inline 图片先落 sidecar、provider wire 才瞬时 hydrate、read model/evidence 零 base64；当前回合 text-only fail-before-network、历史图片降为占位文本。LIV-03 进一步关闭 Agnes capability/custom-provider/cache、direct-answer 工具面和 generation lowering 缺陷：capture 为 `tools=0`、`max_tokens=128`、`enable_thinking=false`，Agnes live 在 25.5 秒内识别 apple/red 并以 completed terminal 收敛。
- `current / closed`：Agent Runtime 聚合 Gate B 发现并关闭 renderer smoke build 清空共享 `dist` 的并发回归；approval 场景在 renderer 并发构建期间和完整聚合 current fixture 中均已通过。
- `current / closed`：新 compact terminal timeline 让 approval 只读记录默认位于“查看待处理项”之下，旧 Gate B 因等待记录自动可见而误报超时。current oracle 现先确认 terminal/input 恢复，再展开同 turn timeline 验证唯一脱敏记录；resume/decline/cancel/full-access 与完整 Agent Runtime Electron fixture fresh 通过，production UI 未回退。
- `current / closed`：AGT-03 multi-agent cold-restart 首次 fresh Gate B 暴露 mailbox Result lifecycle/terminal-ack 缺陷；current App Server owner 已补 `delta(in_progress) -> message.completed`、terminal-only ack、partial replay 补 completion 和 Failed Result 状态回归。9 条 mailbox、15 条 terminal activity、24 条 agent-control、1152 条 app-server related Rust 及真实 Electron cold-restart Gate B 全绿。
- `current / closed`：AGT-03 Gate B runner 已从 500 字符 tool preview 断言改为完整 current read-model tool output；preview 只用于 evidence 展示，避免多 Result terminal activity 造成测试误报。
- `current / closed`：AGT-04 双 child 并发回归发现 canonical UserMessage 在 synthesized `item.completed` 后被空 payload 覆盖。current projection owner 已补 UserMessage snapshot merge，两个 child 可同时运行、按 session/thread/mailbox 隔离、一次聚合 Result，且 failed sibling 不污染 completed sibling；AgentControl 26/26、canonical lifecycle 10/10、完整 Agent Runtime current fixture 通过。
- `current / closed`：1200-command Codex import 性能门禁曾在全量/独立运行耗时 174.4s/106.7s。根因是逐事件重复复制 validation context 和 canonical notification 全历史 materialize；current owner 改为增量 sequence/tool lifecycle validator 与 incremental notification materializer 后独立耗时 3.51s，App Server related 1157/1157。
- `current / closed`：MCP-02 fresh Gate B 已证明 server-originated elicitation 经真实 Electron/preload、`app_server_handle_json_lines`、App Server reverse JSON-RPC 到 Renderer form，再回到 MCP accept ledger 并让 provider 完成第二次请求；runtime stdio connection 精确广告 `elicitation` capability，management connection 保持 absent。
- `current / closed`：MCP-03 首次真实运行暴露 Desktop Host 把 App Server JSON-RPC 业务错误抛成 IPC error。`handleJsonLines` 现只在 JSONL 转发边界恢复 `AppServerRequestError` response 与 renderer 原 request id，高阶 `request()` 和 transport timeout/stale restart 语义不变。fresh evidence 证明失败 server 为 stopped，健康 server 的 status、tool list/call 与 resource read 连续可用，legacy MCP 命中为零。
- `current / closed`：GUI-01/ELN-01 fresh complete Gate B 首次误报 identity，因为 collector 在找不到产品 turn response 时回退到后续 `event-read-probe`。current evidence matcher 现对已知 turnId 只接受精确匹配；fresh evidence 中 Renderer/trace/runtime/read model identity 一致，Electron/preload/IPC/current JSON-RPC 全绿，legacy/mock/page error 为零。
- `current / closed`：GUI-02 fresh `cancel-then-continue` Gate B 证明 stop 命中 `agentSession/turn/cancel`，read model 进入 canceled、输入框恢复，同 session 后续 turn 完成；ELN-03 backend-failure Gate B 证明 partial answer 保留、失败详情只进 read model、GUI 输入恢复，零 legacy/mock/error。
- `current / closed`：ELN-03 packaged unavailable fail-closed 首次被真实用户 data root 的旧固定 session 污染并报 `SequenceRegression`。四条 current App Server stdio/external/packaged smoke 现全部注入临时 `dataDir`，success/failure/unavailable fresh 运行全绿；contract guard 防止恢复默认用户目录。
- `current / closed`：PRV-05 对照 Codex 后发现旧绿测固化了错误 retry 集合：Lime 请求层重试 429/408/409/425 且漏掉部分 5xx。current owner 已收敛为初始请求加 4 次有界重试、全部 5xx 与 transport 可重试、429 不在请求层自动重试；401、429 和最终 5xx 分别保留 auth/rate_limit/server terminal 分类。公开 `CurrentProvider.stream` localhost capture、agent-runtime trace 和 13 个 related crate 全绿。
- `current / closed`：CTX-01 发现 provider history 只读取标量 `message.delta`，`message.delta_batch` 数组静默丢失，`message.completed` full-text 也完全忽略。current owner 现解析 batch parts，并先按整个 assistant turn、再按 item 前缀补 completed snapshot 后缀；同 item 和 commentary/final turn-wide snapshot 都不重复。owner 11/11、App Server 1169/1169 和完整 Agent Runtime Electron fixture 通过。
- `current / closed`：CTX-02 的旧测试只断言 compaction event 和 summary metadata 存在，未观察 backend 实际收到的 provider history。新 public `RuntimeCore::start_turn -> ExecutionBackend::start_turn_with_provider_history` capture 首先证明旧摘要只复制最近 tail、且下一轮仍发送全部旧 turn。current owner 现让 `session_context_compaction.v2` 摘要只接续 tail 之前的前缀，durable EventLog/read model 保持全量，provider history 从最新 `tailStartTurnId` 重建；带 `outputRef` 的历史工具结果只使用 canonical preview，异常超长 inline output 继续按 10,000-byte 上限截断，不再读取 full sidecar。owner 定向、App Server 1171/1171、contracts 291 项和完整 Agent Runtime Electron fixture 均通过，`liveProviderUsed=false`。
- `current / closed`：PRV-06 已进入正式 owner。`ProviderRuntimeSpec`/direct request 显式投影 capability，`AgentRuntimeState` 按 session 复用并清理 provider client，`model-provider` 发送真实 Upgrade 与 `response.create`、串行复用连接，并让 SSE/WS 共用 reducer；426、重试耗尽和首个可见 event 前断线会安全 replay HTTP，fallback 跨 Turn sticky。首次 capture 稳定证明旧实现即使 capability=true 仍只发 `POST /v1/responses`。model-provider 135/135、agent 270/270、App Server 1175/1175、contracts 291 项和完整 Electron current fixture 已通过；短问候 `firstTextDeltaToFirstTextPaintMs=25`，预算未放宽。
- `current / closed`：扩大门禁直接发现并关闭三项 Lime App Server 缺陷：TS client 丢失 `supportsWebsockets` capability；大型 async dispatcher 在默认 2 MiB 栈溢出；stdio 把非 turn request 串行化而被长 turn head-of-line 阻塞。dispatcher 现装箱 handler future，初始化后 request task 化，contract guard 按 method→handler 语义归一化。
- `current / closed`：全量门禁中的宿主环境噪声被下沉为确定性 owner 修复。PTY 测试不再继承用户 zsh rc；plain directory Git status 零子进程返回，仓库内 Git 使用异步 5 秒 deadline/kill-on-drop。App Server 全量因此无需过滤达到 1175/1175。
- `current / closed`：`PLT-01` 本地 macOS packaged 首跑连续暴露三项真实缺陷：Forge 未给外层 `.app` 生成 sealed-resource signature；ad-hoc 签名错误启用 hardened runtime，因无 Team ID 导致 Helper、renderer、network service 与 packaged App Server 冷启动崩溃；packaged `SHELL-01` 又被 launcher 重写到 dev-branded Electron。current Forge owner 现让本地 package 使用纯 ad-hoc、`hardenedRuntime=false`、`timestamp=none`，正式 Developer ID 路径继续使用 runtime/signing；resource verifier 强制 `codesign --verify --deep --strict`，packaged smoke 禁止 dev branding。fresh package 的 Helper/sidecar 均为 `flags=adhoc`，Gate B-F 21/21、40 次 current App Server IPC 命中，console/page/invoke/trace/crash/mock/legacy 均为零，截图无空白或重叠。
- `current / platform evidence partial`：2026-08-18 已移除 System identity 的错误 `TrustAsRoot` 覆盖，Forge 使用 `Developer ID Application: Digital Easy Inc. (93RF7VQ26C)` 完成 arm64 package；主 App、四个 Helper、`app-server` 与 `code-mode-host` 均通过严格签名/资源校验，Authority、TeamIdentifier 和 hardened runtime flag 一致。独立 DMG maker 与仓库 current local-feed ZIP 入口整体退出码均为 `0`；最终 DMG 为 `330311401` bytes、SHA-256 `18fcccac3175a383a755a13f6e502197b07f54db5f1e884d593242b54bab8481` 且 `hdiutil verify` 通过，ZIP 为 `331037460` bytes、SHA-256 `659ba14d9ed9a97225e81eb52c7e6fe88af774557ee75fb5943dbf448bbceeba`。组合 maker 的线上 `RELEASES.json` 请求曾因 TLS 断连失败，但不否定随后由正式 local-feed 入口完成的 ZIP 产物；本地 `RELEASES.json` 仅作验证，不得发布。当前没有 `APPLE_ID`、app-specific password 或 `APPLE_TEAM_ID`，`spctl` 对 App 明确返回 `Unnotarized Developer ID`、对 DMG 返回 `no usable signature`；未执行 notarization、DMG 安装/升级 receipt 或 Developer ID packaged Gate B，因此仍不关闭完整 `PLT-01`。
- `current / implemented`：Windows Squirrel lifecycle 已按 Forge 当前 `electron-winstaller` 合同把 shortcut flag 与 `Lime.exe` 拆成独立参数，并在 1 秒内退出 host，不再等待 detached `Update.exe`。N-1 current runner 会选择低于候选的最近稳定 Release，安装旧版 Setup，经真实 preload/IPC updater 请求隔离 `RELEASES + full.nupkg` feed，等待 downloaded/restarting、旧进程退出与候选 app path，再直启候选 executable 复用 `SHELL-01`。重复 `checkForUpdates` 和并发 `quitAndInstall` 竞态已在 `ElectronUpdateHost` 修复；手工 Windows workflow 的 dead `package-lock.json + npm ci` 已替换为 current pnpm 安装链。
- `current / closed`：SOAK-01 短校准的 AgentControl cold restart 首跑暴露 `ToolCallDisplay` 与 `InlineToolProcessStep` 对同一 `tool-call-row` 投影不一致：前者丢失 canonical `id/name/status`，导致 Electron 重启前后 6 条 row 都无法按 identity 证明唯一性。current UI owner 已统一三个 DOM 属性；复跑在 PID `41348 -> 43939` 后恢复 6/6 completed tool row、4 条 canonical SubAgent activity、同一 child thread 和零 console/invoke error。
- `current / closed`：SOAK-01 同生命周期校准又发现 Electron Host 把合法 turn admission 错套独立 2 秒 identity read timeout；Lime App Server 的 preflight 本就可能在 admission 前执行，Codex 也让 `turn/start` admission 服从请求 deadline。current Host 现让 250ms quick ack 与 canonical read 共用 `turn/start` deadline，旧 2 秒窗口已删除，Host 回归 27/27。
- `current / closed`：Renderer projection 定向回归曾暴露 active turn 缺省时误隐藏 reasoning/tool/streaming、失败 fallback 被历史过滤覆盖、相对 artifact path 误删 canonical `file_artifact` 和 patch diff 排序错误。current owner 已按 turn terminal 状态、failure fallback 优先级、过程汇总/结果快照分离和结构化 diff 顺序修复；projection 四个 owner 测试文件 `42/42`，renderer typecheck/ESLint 通过。
- `current / closed`：长历史 SOAK 首次证明 compact preview 展开后仍永久隐藏 canonical tool/subagent rows。current timeline 只在 preview 阶段延迟挂载，materialize 后恢复 operational details；managed oracle 同时从“preview 数量必须下降”改为接受真实 timeline 挂载，再展开 process block。UI 定向 27/27，旧永久隐藏语义和 preview-count-only oracle 均为 `dead / deleted / forbidden-to-restore`。
- `current / closed`：SOAK 扩到 10 轮后，第 7、10 轮在 runtime 已 terminal、child 已打印 pass 后仍各等待约 182 秒；现场 TCP 证明 fixture 与 App Server 的 provider connection 仍为 active。根因是 `model-provider` 的 OpenAI `[DONE]`、Responses terminal batch 和 Anthropic `message_stop` 在先 yield terminal 后仍由悬停 generator 持有 `reqwest::Response`，上层收到 Finish 后停止 poll，HTTP body 延迟释放。三条 current stream 现都在 terminal event 前释放 frames；owner keep-alive 回归逐条证明消费者无需再次 poll 即可观察 peer close。fixture 只清理 idle connection，不使用 `closeAllConnections()` 掩盖 active request。
- `current / closed`：修后 controlled-provider `10 rounds x 2 cold restarts` receipt `agent-control-soak-10x2-sse-fixed.json` 全绿。round duration 为 `4.450-8.398s`，总 RSS `466,624 -> 274,352 KiB`，App Server RSS `71,920 -> 67,248 KiB`；10 个 session 每轮唯一 completed turn、item identity 唯一，重启后全部 Thread/Turn/Item 对象稳定。Electron PID `12587 -> 28108 -> 29467`，每棵旧进程树和最终进程树均退出；GUI 恢复 6/6 completed tool row、4 条 canonical SubAgent activity，invoke/console error 为零。正式冻结 RC 仍须复跑同一合同，但不再扩建 runner。
- `blocked / evidence pending`：当前机器不是 Windows，无法在本轮生成真实 `PLT-02` Platform/packaged receipt。必须由包含当前 runner 的 `Build Windows Test Package` 或 release Windows matrix 上传 `.lime/qc/windows-squirrel-rc/**`，且 `remainingClaims.nMinusOneUpdate=passed`、候选 `SHELL-01` 全绿后，才能把 N-1 install/update/path/permissions/current chain 标为 closed。
- `blocked`：完整 DeepSWE TS/Go/Rust 与 thinking on/off 题目仍在固定预算内只读探索且无 candidate；最小 DSW-06 已证明 Lime current `apply_patch` 写链可用。Pier `0.3.1` CLI 已固定并可执行，但本机无 Docker/Podman/nerdctl/Colima，separate verifier 仍不能启动，DeepSWE 评分仍不能产生。
- `next`：为已签名的 macOS arm64 候选补 Apple notarization/staple、DMG 安装/升级和 Developer ID packaged Gate B receipt；并行执行包含 N-1 模式的真实 Windows Squirrel RC workflow，在冻结候选上复跑已关闭的 SOAK-01 合同。任一首个产品失败都回到 Lime current owner 修。DeepSWE 只在 Agnes 能产生 candidate 且 verifier runtime 可用后恢复 scoring。

## 8. 完成口径

第二期只有在以下条件全部满足后才能标记完成：

1. P0 场景全部有稳定自动化 owner，且失败可定位到单一边界。
2. Agent runtime 关键逻辑均有 L2/L3 集成证据。
3. GUI 主路径均有 Gate A；关键桌面链均有 Gate B。
4. restart、resume、cancel、queue、stale/out-of-order、tool approval、MCP、multi-agent 和 multimodal 至少各有一条失败恢复场景。
5. macOS 与 Windows 的 release candidate 证据齐全。
6. current 导航中不存在旧 Benchmark 命令、manifest 或伪代码测试指南。

当前测试体系基线完成度为 `100%`；按 T0-T11 交付合同计算，第二期测试实现整体完成度仍为 `80%`。SOAK-01 的本地 controlled-provider 实现合同已关闭；macOS 本地 Forge package、严格 ad-hoc 签名和 packaged Gate B 也已关闭。正式 Developer ID package、DMG maker、current local-feed ZIP 和签名/完整性校验已经完成，但 notarization、DMG 安装/升级、Developer ID packaged Gate B 与 Windows receipt 尚未完成，因此不关闭完整 `PLT-01/02`，也不增加完成度。剩余主要是 DeepSWE Smoke/Release 有效评分、Agnes EVAL-01 稳定 tool/terminal baseline、PLT-01/02 完整 RC 与冻结候选复跑，而不是继续增加同义单测。
