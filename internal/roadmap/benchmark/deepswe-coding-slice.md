# DeepSWE Coding 测试切片 v2

> status: DSW-07 schema 1.3 + Pier 0.3.1 contract passed / 0 verified score / Desktop Smoke 5 controlled Gate B and recovery passed / combined verdict pending live verifier
> owner: evaluation + agent-runtime
> source_commit: `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`
> source_schema: `1.3`
> pier: `datacurve-pier==0.3.1`
> source_tasks: 113
> upstream_audit: `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9` / schema `1.3` / Pier `0.3.1`
> manifest: `internal/test/deepswe-coding-slice-v2.json`

## 1. 定位

DeepSWE Core 用于评估 Lime 的长程 coding 能力，不单独验证协议和 GUI 的确定性正确性。它属于 L7 coding eval，并与 L6 Desktop Smoke 配对：

- 不能替代 `test:contracts`、Rust/App Server integration、current runtime fixture 或 Gate A/B。
- 必须通过 Lime App Server JSON-RPC current 链运行，不能用 Codex CLI、mini-swe-agent 或旧 Agent runtime 结果冒充 Lime 分数。
- verifier 在独立环境中运行；reference solution 和 verifier tests 对 Agent 不可见。
- 只有 DeepSWE/Pier verifier 与真实 Electron Gate B 同时通过，才能声明桌面 coding 成功；Core、Desktop 任一单边通过都必须保留自己的证据边界。

本阶段的首要产出不是 pass@k，而是从真实 coding trajectory 中定位 Lime owner 缺陷。每个失败先区分 `Lime product`、`adapter`、`model`、`task environment` 和 `verifier`；只有 App Server current chain、任务隔离和 separate verifier 都有效时，结果才进入能力分数。

## 2. 为什么重新选题

旧 `deepswe-fixed-ten` 只有 dry-run runner，没有真实 Lime Agent execution，且固定十题未写清语言分布、能力维度和回归阈值。v2 切片重新固定 source commit，并按 Lime/Codex coding 风险选题：

- 流式协议与增量输出；
- 取消、超时、资源回收；
- 持久化、缓存、回放与损坏恢复；
- 并发、依赖图、multi-agent/tool routing；
- parser、typed API、错误合同；
- 确定性排序、冲突处理和结构化重写；
- TypeScript、Go、Python、Rust、JavaScript 五种语言。

## 3. 两级切片

### Smoke 10

| Task                                       | Lang   | 重点                                         |
| ------------------------------------------ | ------ | -------------------------------------------- |
| `happy-dom-abort-pending-body-reads`       | TS     | bugfix、取消、资源回收                       |
| `ofetch-per-origin-circuit-breaker`        | TS     | 网络状态机、并发、重试                       |
| `go-genai-streamed-function-args`          | Go     | provider stream、function args、SDK contract |
| `ytt-jsonpath-query-api`                   | Go     | parser、公共 API、结构化错误                 |
| `httpx-multipart-response-parsing`         | Python | sync/async stream parser、畸形输入           |
| `ipython-session-bundle-replay`            | Python | replay、持久化、redaction                    |
| `boa-hierarchical-evaluation-cancellation` | Rust   | 层级取消、runtime                            |
| `fd-deterministic-multi-key-sorting`       | Rust   | CLI、文件系统、确定性                        |
| `csstree-shorthand-expansion-compression`  | JS     | 双向转换、复杂语法                           |
| `yjs-map-conflict-detection`               | JS     | 冲突处理、分布式状态                         |

用途：adapter bring-up、模型/runtime 候选快速比较、版本 RC 的 coding smoke。默认每题 1 trial；正式模型 bake-off 每题 3 trials。

### Release 20

在 Smoke 10 基础上增加：

| Task                                         | Lang   | 重点                                |
| -------------------------------------------- | ------ | ----------------------------------- |
| `superjson-error-stack-serialization`        | TS     | 序列化、错误、redaction             |
| `awilix-async-container-initialization`      | TS     | async lifecycle、依赖图、失败恢复   |
| `claude-code-by-agents-recursive-delegation` | TS     | multi-agent、递归委派、tool routing |
| `vitest-duration-sharding`                   | TS     | test infra、sharding、determinism   |
| `prometheus-typed-label-sorting`             | Go     | bugfix、typed sorting               |
| `pebble-durability-wait-apis`                | Go     | 持久化、并发、wait API              |
| `python-statemachine-state-data-scoping`     | Python | 状态机、history、scope              |
| `bandit-incremental-cache-control`           | Python | cache、CLI、损坏恢复                |
| `gql-incremental-graphql-delivery`           | Python | 增量协议、streaming                 |
| `oxvg-structural-selector-preservation`      | Rust   | 精确结构重写、回归修复              |

用途：大版本 RC、默认 coding 模型或 tool policy 变更、RuntimeCore/coding tools 重大变更。

### Desktop Smoke 5

Desktop lane 不复制另一套题目，固定复用 Smoke 10 中五种语言各一题：

| Task                                 | Lang   | 桌面主风险                                    |
| ------------------------------------ | ------ | --------------------------------------------- |
| `happy-dom-abort-pending-body-reads` | TS     | 长回合、取消、terminal 后零幽灵写入           |
| `go-genai-streamed-function-args`    | Go     | streaming/tool loop、命令输出与 diff 投影     |
| `httpx-multipart-response-parsing`   | Python | 多文件读取、同步/异步实现与测试结果可见性     |
| `fd-deterministic-multi-key-sorting` | Rust   | CLI、文件系统、审批/sandbox 与确定性 patch    |
| `yjs-map-conflict-detection`         | JS     | 冲突处理、历史恢复、artifact/workbench 一致性 |

每题从真实 Electron 打开仓库外隔离 workspace，经 GUI 输入原始 DeepSWE instruction；必须证明 preload/contextBridge、Electron IPC、`app_server_handle_json_lines`、App Server、RuntimeCore、Thread/Turn/Item、GUI terminal 和 patch identity 一致，production mock、invoke error、console/page error 为零。完成后同一 patch 进入 Pier separate verifier。Desktop lane 默认每题 1 trial；冻结 RC 每题 3 trials。

当前实现入口：

```bash
npm run harness:deepswe:desktop:preflight
npm run harness:deepswe:desktop:controlled
npm run harness:deepswe:desktop:aggregate -- .lime/benchmark/v2/desktop/controlled/<run-id>
```

`deepswe-desktop-trial-v1` 是单 trial 的唯一事实源，必须同时绑定原始 instruction SHA、仓库 base commit、workspace、session/thread/turn、provider tool lifecycle、projected lifecycle、test stdout、GUI/read model、patch SHA 与 verifier 状态。受控 runner 的 `trialKind=controlled_product_smoke` 只证明真实桌面产品链；只有 `trialKind=live_deepswe` 且 Gate B、Pier artifacts、同一 patch SHA 都通过，才允许 `DesktopCodingPass=true`。

2026-08-19 的完整受控复验已覆盖五题：`happy-dom-abort-pending-body-reads`、`go-genai-streamed-function-args`、`httpx-multipart-response-parsing`、`fd-deterministic-multi-key-sorting`、`yjs-map-conflict-detection` 均通过 Electron Gate B。runner 对 TypeScript fixture 使用无依赖 Node test 执行 `.ts` 源码逻辑，并在临时 Electron 环境显式复用已安装的 Rust stable toolchain；这些只解决受控环境可复现性，不扩大能力声明。

## 4. Lime Adapter 合同

新 adapter 必须：

1. 在任务声明的 base commit 启动 workspace；clone 必须位于 Lime 仓库外的系统临时目录，且 Node 等工具不得向上解析到 Lime 的依赖。
2. 启动 Lime App Server/RuntimeCore current owner，并通过 public JSON-RPC 创建 thread/turn。
3. 把 DeepSWE instruction 作为用户输入；不得注入 task-specific 解法或 reference patch。
4. 只开放 Lime current coding tools、审批和 sandbox policy；记录最终有效 tool catalog。
5. 等待真实 terminal event，不靠固定 sleep 合成完成；wall timeout 必须先请求 current turn cancel 并等待 terminal，再固化失败 evidence。
6. 每个 sampling step 投影 `provider.step`，导出逐步与累计 usage；provider step/token budget 由 runtime 在工具执行和下一次 sampling 前执行，adapter evidence 轮询只作 timeout race fallback，wall time 只做总兜底。
7. 导出 Thread/Turn/Item、tool lifecycle、trajectory、`provider-steps.json`、patch 和运行上下文。
8. 由 DeepSWE/Pier separate verifier 应用 patch 并生成 `reward.json`、`ctrf.json` 和 stdout。
9. 将失败归类为 `agent-runtime`、`model`、`tool-runtime`、`app-server`、`transport`、`harness`、`environment`、`verifier` 或 `budget`。
10. terminal 或失败后先固化 partial/candidate patch，再清理临时 clone；run evidence 留在 `.lime/benchmark/v2/runs`。

生产 GUI 不是 DeepSWE Core runner 的必经入口，但 Core adapter 必须使用与 GUI 相同的 App Server/runtime 公共链，不能建立 benchmark-only runtime。任何“桌面端 coding”声明必须额外运行 Desktop Smoke 5，不能用 Core 的 `stdio` 或 dev-bridge 结果替代 Electron Gate B。

当前 adapter 已落在：

- `scripts/harness/deepswe-adapter.mjs`：CLI、live gate、单题执行和 verifier-only 恢复；
- `scripts/harness/deepswe-adapter-core.mjs`：source/task preflight、仓库外隔离 workspace、App Server current chain、证据、patch 与 Pier 交接；
- `scripts/harness/deepswe-adapter.test.mjs`：current method、隔离 git、reference solution 隔离、verifier 证据和旧 runner 不回流守卫。

正式入口：

```bash
npm run harness:deepswe:preflight
npm run harness:deepswe:run -- --task happy-dom-abort-pending-body-reads --allow-live-provider
npm run harness:deepswe:run -- --verifier-only --run-dir <existing-run-dir>
```

Pier 本地工具必须使用隔离 Python 3.12 环境安装固定版本：

```bash
uv venv --python 3.12 --allow-existing .lime/benchmark/tools
uv pip install --python .lime/benchmark/tools/bin/python datacurve-pier==0.3.1
.lime/benchmark/tools/bin/pier --version
```

最后一条必须输出 `0.3.1`；它只证明 Pier CLI 可加载，不代表 verifier 已运行。`runtimePrerequisites` 会另外检查容器 CLI，缺少 Docker/Podman/nerdctl/Colima 时保持 blocked。

adapter 不把 reference solution 复制到 Agent workspace。Lime current chain 结束后只把 `patch.diff` 放入临时 Pier replay task，由 Pier 在 separate verifier environment 应用并判分。Verifier preflight 在 candidate patch 固化后执行，这样缺少容器运行时时仍保留 Lime 缺陷证据；但该 trial 不得产生或冒充 DeepSWE 分数。`--verifier-only` 会保留既有 product failure，并单独记录 verifier blocker。

adapter v6 默认最多运行 32 个 provider step、消耗 500,000 budget tokens，并每 30 秒捕获一次 current evidence。step/token 预算通过 `runtimeRequest.metadata.harness.provider_budget` 投影到 `AgentSessionConfig`，token 计算为 `max(0, input_tokens - cached_input_tokens) + output_tokens`；current reply loop 在工具执行和下一次 sampling 前停止。adapter 只为 timeout race 保留 token evidence polling，不再成为预算执行 owner。诊断时可以显式收紧预算，并用 `--max-output-tokens`、`--enable-thinking true|false` 覆盖单次 run 的 generation controls；实际值和 enforcement owner 必须写入 run context。wall time 只作为最后的环境保护，触发后调用 `agentSession/turn/cancel` 并最多等待 10 秒真实 terminal，不能留下 running turn。

## 5. 指标

必报：

- pass@1；bake-off 时增加 pass@3 与 pass^3；
- 每题 wall time、model latency、token/费用；
- provider step 数、每步 usage、累计 usage 和预算终止原因；
- tool calls、tool failures、approval/sandbox failures；
- patch size、changed files、test result；
- no-op、build failure、timeout、verifier failure；
- 按语言和 focus 聚合的通过率。
- Desktop lane 的 Gate B assertion、mock/invoke/console/page error、GUI terminal、diff/artifact 可见性和 restart/cancel 后写入完整性。
- `DesktopCodingPass = DeepSWEVerifierPass && ElectronGateBPass`；基础设施不完整时为 invalid，不降格为 pass，也不混入模型失败。

只比较相同 source commit、task slice、模型、provider、tool policy、预算和 adapter version 的运行。任一维度变化必须重建 baseline。

## 6. 门禁阶段

### Calibration

adapter 首次完成后运行 Smoke 10 三个独立批次。此阶段只做信息性报告，不设置分数阈值；环境或 verifier failure 必须为零。

### Candidate Gate

有稳定 baseline 后：

- Smoke 10：pass@1 相对稳定基线最多下降 1 题；同一任务连续两次从 pass 变 fail 时阻断。
- Release 20：总体 pass@1 回退不超过 10 个百分点；任一语言不得从非零直接降为零。
- 环境、adapter、App Server 或 verifier failure 不计 agent 失败，但任何非零基础设施失败都使本次结果无效，不能用剩余题目计算 release verdict。

阈值可在三轮 calibration 后收紧，不能为了让候选通过而临时放宽。

### Desktop Candidate Gate

- Desktop Smoke 5 每题先完成 1 次 bring-up，再完成 3 trials；基础设施、transport、GUI observer 与 verifier failure 必须为零。
- 同一 trial 必须关联一个 task/base commit、workspace、session/thread/turn、patch SHA-256、Gate B summary 与 Pier result，禁止拼接不同 run 的证据。
- 任一题 verifier 通过但 Gate B 失败，归桌面产品失败；Gate B 通过但 verifier 失败，归 coding/model/tool 结果失败；缺 Pier、容器、凭证或平台时整批为 invalid。
- macOS 与 Windows packaged claim 仍需各自 L8 receipt；开发态 Electron 通过不能替代 packaged parity。

## 7. 版本与合规

- 本地 source cache 不进入 Git；版本化 manifest 只记录 source commit、任务 ID 和策略。
- 旧固定 commit `3cda408...` 未包含仓库根 license；DSW-07 已将 source 固定到 `435ee89...`，并完成 Apache-2.0 `LICENSE`、`PROVENANCE.md`、schema `1.3`、`network_mode`、`[[verifier.collect]]`、Pier 与 verifier evidence 的成组迁移。当前不分发任务内容或镜像，评分仍需 live candidate 与 separate verifier。
- 不保存 API key、Authorization、真实用户数据或 reference solution 到 evidence。
- 公开 benchmark prompt 不进入 Lime system prompt、skill 或 task-specific routing，防止过拟合和数据泄漏。

## 8. 缺陷发现闭环

每个诊断 trial 按以下顺序处理：

1. 先验证 workspace、cwd、skill root、provider/model 和 tool catalog 是否真实隔离。
2. 读取 trajectory 和 Thread/Turn/Item，判断模型实际做了什么，不用 terminal 文案替代行为证据。
3. Lime 边界缺陷在对应 owner 修复，并补最小回归；模型任务失误只记录，不通过改 prompt 或放宽 sandbox 掩盖。
4. 用同一任务和 Agnes 重跑；需要区分模型与 Lime 时才使用固定 gpt-5.5 对照。
5. candidate patch 必须进入 Pier separate verifier；Docker/Pier 不可用时只允许写 blocker，不能手工判 pass。
6. 只有能稳定复现的 Lime 根因才下沉到 L2-L6 门禁，DeepSWE task 本身不复制成大量仓库测试用例。

## 9. 2026-07-15 诊断事实

- Agnes run `20260715T180858Z-happy-dom-abort-pending-body-reads` 暴露 cwd 未传工具、模型不可见 cwd、workspace skill 混入、Bash `2>/dev/null` 误判、空 tool placeholder、伪 `final_answer`、失败 item 未终态和 provider stream 错误信息过粗；该 trial 的 workspace 位于 Lime 仓库内，因此不计能力分数。
- gpt-5.5 对照 `20260715T184451Z-happy-dom-abort-pending-body-reads` 生成 6 文件、189 增/29 删、1,235,341 字节 candidate patch，同时暴露 Node 向上解析 Lime `node_modules` 和 patch capture `ENOBUFS`；该 trial 同样不计能力分数。
- 两个模型都在活动 SSE 约 600 秒后出现 `error decoding response body`。根因是 Lime reqwest client 对整个 response 设置 600 秒总 timeout；Codex 使用逐事件 idle timeout。Lime 已移除总 timeout，改为 5 分钟逐 chunk idle timeout，并保留 error source chain。
- 仓库外 Agnes run `20260715T204006Z-happy-dom-abort-pending-body-reads` 进一步证明全量 clone 会暴露未来 `origin/master`；模型切换分支后生成 1,219,383 字节伪 patch。adapter v2 现只 shallow-fetch 精确 base commit、移除 remote/ref，并拒绝 base 后的非候选 committer。
- adapter v2 Agnes run `20260715T212218Z-happy-dom-abort-pending-body-reads` 使用精确 base、无 remote、仓库外 workspace，记录 3,691 条 App Server 事件。单次活动 SSE 越过旧 600 秒总时限并持续约 78 分钟，证明 idle-timeout 修复有效；Agnes 在 5,400,000ms 总预算内始终没有写文件，trial 以 `budget` + empty patch 结束。
- 同一题的 gpt-5.5 对照能够形成 6 文件 candidate，因而 Agnes 结果不再指向 Lime coding tools/cwd/sandbox。Go/Rust 诊断题暂停，直到 Agnes 路由能在固定预算内产生 terminal candidate；否则继续跑题只是在重复模型吞吐失败。
- 本地 Pier wrapper 的 receipt 虽记录 `datacurve-pier 0.3.0`，但它是指向已删除 `/tmp/lime-pier-source-20260715` 的 editable 安装；当前执行直接报 `ModuleNotFoundError: No module named 'pier'`。同时本机没有 Docker/Podman/nerdctl/Colima 等容器运行时，因此 Pier package 与 container runtime 都是 verifier blocker；`reward.json`、`ctrf.json`、`test-stdout.txt` 尚未产生。
- DSW-02 Agnes run `20260716T020910Z-go-genai-streamed-function-args` 固定 `agnes-2.0-flash`，20 分钟内产生 1,763 个 event、15 个 coding tool item、0 tool failure；cwd、command、Read 和 provider stream 均有效，但工作树始终为空，最终归 `budget`。这把同一只读探索模式从 TS 题扩展复现到 Go 题。
- 本次 timeout 暴露 adapter evidence 丢失：partial trajectory 已存在，但 CLI result 的 `currentChain` 为 null。adapter 现把 provider/model/session/turn/timestamps/evidenceCapture 附到 failure 并持久化，回归 16/16 通过。
- adapter v3 run `20260716T033001Z-go-genai-streamed-function-args` 首次用结构化 provider budget 重跑 Agnes：16/16 个 step 均有 usage，累计 input 268,495、output 3,829、budget 272,324 tokens，共 16 个 tool call、2,350 个 App Server event 和 33 个 trajectory item。第 16 步触发 current `agentSession/turn/cancel`，turn 终态为 `canceled`，failure owner 为 `budget`，patch 仍为 0 bytes。
- 该 v3 run 的 16 个工具中 15 个成功；唯一失败是 Agnes 在命令正文中把正确临时 cwd 少写一段后显式 `cd`，随后通过 current cwd 恢复，不是 Lime 丢失 cwd。逐步 usage、工具终态和取消链完整，因而本轮无 patch 应归 Agnes coding 吞吐，而不是 `tool-runtime`、`transport` 或 `app-server`。
- adapter v3 诊断 run `20260716T081020Z-go-genai-streamed-function-args` 首次从真实 sampling snapshot 固化 tool catalog：3 次 request 都下发相同 27 个工具，`Read`、`Grep`、`exec_command` 和 `apply_patch` 每次都存在。因此 Agnes 无 patch 不是 Lime 隐藏写工具；但预算为 2 completed steps 时仍启动 attempt 3，暴露 adapter 事后轮询取消无法约束下一次 sampling 的 Lime 预算越界。
- adapter v4 将 step 上限下沉到 current reply loop。Agnes 同题重跑 `20260716T083349Z-go-genai-streamed-function-args` 只产生 attempt 1、2 两个 `provider.request.started` 和两个 completed `provider.step`，`budgetCancellation=null`，随后由 runtime 输出 max-turn terminal message；累计 budget tokens 19,905，两个 request 的 27 工具目录均完整且都有 `apply_patch`。该历史证据的最后一步仍为 `tool_call`，当前 adapter v5 会归 `provider_steps` exhaustion；不能继续写成自然完成且纯属模型失败。
- Rust 诊断 run `20260716T113222Z-fd-deterministic-multi-key-sorting` 使用 8 steps / 150,000 tokens：7 个 completed step 累计 158,754 budget tokens、13 个成功工具 item、0-byte patch，但 evidence 已出现 attempt 8 的 `provider.request.started`。这证明 adapter 的 token polling 晚于下一次 sampling，是 Lime runtime budget owner 缺口。
- current 修复把正整数 `token_budget` 从受控 harness metadata 投影到 `AgentSessionConfig`，按非缓存 input + output 累计；带工具调用的 step 达标后，在任何工具执行和下一次 provider request 前返回 canceled execution。owner 回归直接断言 request=1、tool execution=0、ProviderStep/request trace 仅 attempt 1。
- fresh Rust run `20260716T120650Z-fd-deterministic-multi-key-sorting` 使用 8 steps / 12,000 tokens：attempt 1/2 累计 15,065，request/step 序列严格为 `1,2`；attempt 2 返回 4 个 tool call，但 read model 只存在 attempt 1 已完成的 `Glob`，没有 attempt 3。`budgetCancellation.requestedAt=null` 证明 runtime 自主终止。Agnes 仍为 0-byte patch，继续归模型只读探索，不归 Lime budget/tool exposure。
- adapter v5 为同题对照增加显式 `--max-output-tokens` 与 tri-state `--enable-thinking true|false`；参数只投影到当前 run 的 `runtimeRequest.metadata.harness.generation` 并进入 `run-context.json`，未指定时不下发，也不改变生产默认。本次 Agnes candidate 诊断固定使用 `happy-dom-abort-pending-body-reads`、`enable_thinking=false` 和有限 output/step/token budget，检验关闭长 reasoning 后能否从只读探索转为写 patch。
- adapter v5 Agnes 对照 `20260717T031735Z-happy-dom-abort-pending-body-reads` 已完成：`enable_thinking=false` 后 8/8 step 的 `reasoningChars=0`，但模型仍只执行 5 次目录/定位命令和 2 次文件读取；第 8 step 返回的继续读取工具因 runtime budget 达标而未执行。累计 input 94,747、cached input 6,144、output 443、budget 89,046，8 次 request 都有完整 27 工具和 `apply_patch`，最终 `patch.diff=0`。runtime 自主生成 canceled terminal，`budgetCancellation.requestedAt=null`；关闭 thinking 没有把 Agnes 从只读探索转成 candidate。
- 本轮没有据此缩减 production 工具面。Codex 同样按 feature/runtime gate 动态暴露 coding、image 和 multi-agent 工具；Lime 现有 `compact_tools` 同时切换轻量通用提示词，不能在没有对照证据时冒充 coding 修复。Agnes coding trial 至此按退出条件停止，等待模型路由变化后再恢复。
- DSW-06 最小 Agnes 写入探针先红后绿。修复前 `20260717T091520Z-agnes-apply-patch-probe-before` 中 `apply_patch` 第一次因缺少 hunk 行前缀失败，第二次使用 `- before/+ after` 成功却把前导空格写入 `probe.txt`，turn 仍错误显示 completed；根因是 Lime `apply_patch` JSON schema 只写了 marker，没有给模型可执行的行前缀示例。
- current `tool-runtime` 现把 `apply_patch` 的 patch 字段、`*** Begin Patch`/`*** End Patch`、`-before`/`+after` 精确示例和“不要无意增加空格”写进 provider tool definition；无运行时消费者的旧 shell prompt 资产已物理删除。adapter v5 新增 `providerStepExhaustion`，最后一步仍为 `tool_call` 时 fail closed，不把 runtime max-turn 文案当 candidate。
- 修复后 `20260717T104119Z-agnes-apply-patch-probe-after` 经 stdio current chain 产生 4 个 provider step、`patch.applied`、精确 `after\n`、358-byte git patch，provider catalog 每次含 `apply_patch`，`provider_steps` 原因为空；DSW-06 通过。该证据只证明 Lime 写链和合同正常，不扩展为完整 DeepSWE score。
- 2026-07-16 的 registry/cache 曾把 `agnes-2.0-flash` 标记为 `vision=false`、text-only/chat；这是随后 LIV-03 定位并关闭的 Lime capability/cache 缺陷，不影响本节 coding trial 的历史结论。

## 12. 2026-07-17 当前链 DeepSWE 归因复测

- Agnes `20260717T111006Z-happy-dom-abort-pending-body-reads` 在仓库外隔离 workspace 经 stdio current chain 运行 8 step / 80,000 token，8/8 request 的真实 tool catalog 都包含 `apply_patch`；实际只完成目录/读取探索，patch 为 0 字节，最后一步仍是 `tool_call`，runtime 以 `provider_steps` 终止。唯一失败是读取不存在的 `async-task-manager/index.ts`，随后通过 `ls` 找到真实 `AsyncTaskManager.ts`，不属于 Lime 写链缺陷。
- gpt-5.5 固定对照 `20260717T111350Z-happy-dom-abort-pending-body-reads` 同配置运行 6 step，在 97,164 budget tokens 后以 `token_budget` 终止；patch 为 0 字节，尚未形成 candidate。该结果只说明本次预算不足以完成题目，不替代历史 candidate，也不计入模型能力分数。
- Agnes `20260717T112458Z-superjson-error-stack-serialization` 换用 Smoke 10 的短题，在 5 step / 80,000 token budget 后由 runtime 取消；5/5 request catalog 含 `apply_patch`，9 个 lifecycle item 中只有 Glob/Grep 与 file artifact，没有实际写调用，patch 仍为 0 字节。
- 三次复测的 current Thread/Turn/Item、provider usage、tool lifecycle 和 terminal evidence 均完整；没有发现新的 `agent-runtime`、`tool-runtime`、`app-server` 或 `transport` owner 缺陷。结论仍是 Agnes 在当前路由/预算下只读探索，不能将无 patch 归咎 Lime。
- 这三次 run 保留为 diagnostic evidence：`20260717T111006Z-happy-dom-abort-pending-body-reads`、`20260717T111350Z-happy-dom-abort-pending-body-reads`、`20260717T112458Z-superjson-error-stack-serialization`。当时的 Pier editable package 已失效；当前已切换到 `datacurve-pier==0.3.1`，但本机仍无容器 runtime，因此仍不生成 `reward.json`、`ctrf.json` 或 DeepSWE score。

## 16. 2026-08-19 DSW-08 批量计划与聚合器

- `scripts/harness/deepswe-benchmark.mjs` 提供互斥的 `--plan`、`--run`、`--aggregate` 模式；真实运行必须显式传 `--allow-live-provider`，并把 `--transport`、provider/model、budget 和 generation controls 原样传给 adapter。
- 当前 identity 固定为 source commit `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`、task schema `1.3`、adapter `deepswe-current-chain-adapter-v6`。聚合器只从当前 identity 的 trial 选择样本；旧 identity 只记入 `invalidIdentityCount`，不占用当前 trial 槽位。
- `npm run harness:deepswe:batch:plan` 已验证 Smoke 10 的 `105/105` preflight checks；旧 adapter run 和 batch summary 已清理，`npm run harness:deepswe:batch:aggregate` 对空的 Release 20 run 根目录保持 `blocked`，`observedRunCount=0`，所有 score/时延/预算均为 `null`。
- 退出条件仍是 Smoke 10/Release 20 产生同一 identity、非空 patch、current chain completed 和 Pier 三件 artifacts 的可复核三 trial；缺容器、缺 verifier evidence 或未授权 live provider 均保持 fail-closed。

## 10. 实施顺序

1. `DSW-00`：已完成；source commit、20 个 task path、task schema 和 verifier metadata 共 61 项检查通过。
2. `DSW-01`：adapter v6、仓库外隔离、runtime step/token cap、显式 generation 诊断参数、partial evidence、真实 tool catalog、逐步 usage 和 DSW-06 写入探针已完成；等待完整 DeepSWE 题产生非空 candidate 且容器可用的 Pier verifier 后关闭评分链。
3. `DSW-02`：TS/Go/Rust 及 Smoke 10 短题、thinking on/off 对照已证明 Agnes 能稳定使用 current coding tools，但会在固定预算内只读探索、无 patch；暂停继续刷同类题，直到 Agnes 路由能产生 non-empty candidate。
4. `DSW-03`：完成 Smoke 10 三轮 calibration 并冻结 baseline。
5. `DSW-04`：运行 Release 20，建立语言/focus 分层结果。
6. `DSW-05`：把真实失败中可确定复现的 runtime/tool 缺陷回写为 L2-L6 内部回归场景。
7. `DSW-07`：固定审计后的 DeepSWE schema `1.3` commit 与 Pier `0.3.1`，迁移 `network_mode`、`verifier.collect`、license/provenance 和 preflight。
8. `DSW-08`：实现 Smoke 10/Release 20 批量编排与三 trial 聚合，输出 pass@1/pass@3/pass^3、成本、时延和 infra validity。
9. `DSW-09`、`DSW-10`：controlled Desktop Smoke 5 的真实 Electron Gate B、取消/恢复已完成；`DSW-11` 的 Gate B + Pier combined verdict 仍等待 live candidate、同一 patch SHA 与 separate verifier artifacts。

## 11. 2026-07-17 wall-timeout terminal cleanup

- adapter 过去只有 token evidence 达预算时调用 `agentSession/turn/cancel`；总 wall timeout 会直接写 partial evidence 并抛错，真实 App Server turn 可能继续 running。
- current adapter 在 wall/turn-start transport timeout 后调用 public cancel，继续读取同一 session/turn，只有观察到 `completed/failed/interrupted/canceled/cancelled/aborted` 才标记 terminal evidence；10 秒内未收敛则保留 partial 和取消错误。
- timeout 仍按 `budget` failure 分类，`currentChain.status=timeout`，并单独记录 `terminalStatus` 与 `timeoutCancellation`；不能因为取消成功就把 trial 当成 completed candidate。
- `npx vitest run scripts/harness/deepswe-adapter.test.mjs` 为 21/21，新增回归精确断言 `cancelStatus=canceled`、terminal evidence 和无成功冒充。

## 13. 2026-08-18 完整性审计与外部补充集

- 原 `.lime/benchmark/v2/runs` 的 16 个历史 adapter result、patch、partial evidence 和 batch summary 已按确认删除；当前目录为空，不能再作为 DeepSWE baseline 或 score 输入。新的 run 必须从 current source/schema/adapter identity 重新开始。
- 当前机器找不到 Pier、Docker、Podman、nerdctl 或 Colima，不能执行 separate verifier。该环境缺口不阻止 adapter/preflight 回归，但阻止任何真实 DeepSWE score 和 `DesktopCodingPass`。
- DeepSWE 继续作为主要长程 coding benchmark。SWE-bench-Live MultiLang/Windows 用于滚动、多语言和 Windows/PowerShell 补充；SWE-bench Multimodal 用于视觉软件 issue；Terminal-Bench 2.1 用于 shell、依赖和环境诊断。它们各自保留 source/grader/version，不合并成一个失去归因能力的总分。
- OSWorld V2 与 WindowsAgentArena V2 测量 computer-use。只有 Lime 产品明确支持通过视觉/鼠标键盘操作其它桌面应用时，才选择其中 VS Code 场景；当前不作为 coding release gate。

## 14. 2026-08-19 Desktop Smoke 5 全题受控复验

- fresh suite：`.lime/benchmark/v2/desktop/controlled-final/20260819T000134Z/summary.json`，五题 `controlledTrialCount=5`、`controlledGateBComplete=true`，每题真实 Electron/preload/IPC/App Server/RuntimeCore/read model/GUI、native test、terminal、diff、session reopen 与零 mock/invoke/console/page error 均通过。
- 受控失败修复：Node 23.4 不接受 `--experimental-strip-types`，TypeScript fixture 改为 Node 内置 test 读取并执行 `.ts` 函数；Electron 临时 `HOME` 使 rustup 找不到 stable，runner 仅在该受控环境注入现有 `RUSTUP_HOME/CARGO_HOME`，Rust fixture 使用 `cargo +stable`。
- 结论仍为 `status=product_path_only`、`desktopCodingPass=false`：受控 provider 没有 live DeepSWE sampling，`verifier=not_run` 且没有 Pier `reward.json`、`ctrf.json`、`test-stdout.txt`，因此不产生分数，也不关闭 DSW-09/10/11。

## 15. 2026-08-19 Artifact 正文与 Recovery 完整复验

- App Server production runtime 的 artifact content owner 改为 `WorkspaceArtifactContentProvider`：inline content 优先；无 inline 时只从 artifact metadata 声明的 `cwd`、`workingDir`、`working_dir` 或 `environments[].cwd` 解析 workspace，canonical 后读取 workspace 内相对/绝对路径。它拒绝 `..` 越界、workspace 外绝对路径、非 UTF-8 和超过 1 MiB 的文件，也不回退进程 cwd。
- Desktop Gate B 新增 `artifactContentAvailable`：真实点击消息 artifact card，等待 `canvas-workbench-code-preview`，断言修改后的独有正文 marker 可见、错误文案 `App Server artifact 内容不可用` 不可见、Electron IPC trace 命中 `artifact/read`，并单独保存 `*.artifact-preview.png`。
- fresh suite：`.lime/benchmark/v2/desktop/controlled-artifact-final/20260819T022022Z/summary.json`。TS、Go、Python、Rust、JavaScript 五题均 `gateBPass=true`、`artifactContentAvailable=true`；`controlledGateBComplete=true`、`recoveryCoverageComplete=true`，每题 session reopen、approval resume、cancel no-ghost-write 通过，native test 通过，mock/invoke/console/page error 均为零。
- 前一次 Yjs attempt 6 无首事件的现场保留为一次性 transport/进程时序诊断：连接诊断单题证明第 6 次 SSE response 正常 `finish/close`，随后无重试的完整五题 suite 也通过。未找到可重复的 fixture index、tool defer 或 stream close 缺陷，因此不增加掩盖失败的自动重试，也不放宽 terminal fail-closed。
- 证据边界不变：完整 suite 仍为 `status=product_path_only`、`liveTrialCount=0`、`desktopCodingPass=false`。受控 provider、artifact 正文与 recovery 只能证明真实产品链，不能替代 live DeepSWE sampling、Pier verifier、同一 patch SHA 或三轮能力统计。

## 17. 2026-08-21 Desktop approval convergence 与合同复验

- App Server `thread/read` 在 loaded runtime 的 active owner 已退出、canonical Turn 尚未完成 terminal projection 的窗口，过去可能短暂返回 `interrupted`。current `thread_read` 保留仍为 `inProgress` 的 persisted Turn，直到 terminal projection 收敛；冷启动 orphan 和被新 active Turn 取代的旧 Turn 仍保持 interrupted 语义。回归 `loaded_runtime_preserves_turn_until_terminal_projection_converges` 为 `1/1`。
- `waitForSpecificTerminalThread` 增加 `acceptableStatuses` 参数：approval resume 只接受 `completed`，cancel 保持既有取消终态集合。单题 approval 证据明确为 `terminalStatus=completed`、tool `completed`、marker 存在和 `doneInReadModel=true`。
- 本轮合同验证：Desktop Smoke 5 preflight `53/53`；Release 20 preflight `205/205`；Smoke 10 batch plan `105/105`；DeepSWE adapter 与 Desktop contract `36/36`；受控 smoke、desktop benchmark、coding slice、batch benchmark 和 provider fixture 回归共 `41/41`。
- 最新 controlled suite `.lime/benchmark/v2/desktop/controlled/20260821T011259Z/summary.json` 为 5/5 Gate B、恢复覆盖完整、零 mock/invoke/console/page error；聚合器按设计拒绝提升为 `DesktopCodingPass`，失败项仅为 `liveTrialPerTask` 与 `allLiveDesktopCodingPass`。
- 当前边界：未调用 live provider，未安装依赖，未运行 Docker/Podman/nerdctl/Colima，未生成 `reward.json`、`ctrf.json` 或 `test-stdout.txt`。因此 DSW-11、DeepSWE score、pass@k 与 DesktopCodingPass 继续保持 blocked/incomplete，不使用 controlled fixture 冒充能力评分。
- Release 20 空批次聚合复验：`npm run harness:deepswe:batch:aggregate` 以预期退出码 `2` 返回 `status=blocked`，`observedRunCount=0`、`infraValid=false`、20 题均 `missing_trials`，`passAt1/passAt3/passPower3/wallTimeMs/budgetTokens` 全为 `null`；没有把缺失样本折算成 0 分。
