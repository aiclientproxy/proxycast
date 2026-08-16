# 多模型多模态统一运行时实现计划

> 状态：in progress / audio TTS execution current
> 更新时间：2026-08-16
> Owner：Runtime / Model Registry / App Server / Media Task 主链
> 关联 PRD：`internal/roadmap/models/prd.md`

## 目标

把 Lime 的多模型、多模态能力统一收敛到 App Server JSON-RPC / RuntimeCore current 主链。Agent loop 与 Thread/Turn/Item 服从 Codex；model catalog、default/switch、capability 与媒体产品语义以 `grok-build` 为主参考；OpenCode 只补 provider wire、canonical content 与媒体 lowering。首期先建立 typed model/provider contract，再逐步落 RouteResolver、canonical LLM request/event、协议 mapper 和媒体任务复用链路。

## 事实源分类

- `current`：`model/list`、`modelProvider/*`、`modelProviderKey/*` App Server JSON-RPC methods，`ModelRegistryService`，`lime-rs/crates/core/src/models/model_registry.rs`，RuntimeCore / App Server protocol，以及由统一 catalog admission 驱动的图片/视频 media task worker。
- `compat`：服务端内部从现有 service / DAO JSON projection 读取并投影成 protocol DTO 的适配函数；它只允许停留在 App Server local data source 边界，不出现在 App Server JSON-RPC 合同里。
- `deprecated`：Agent `CredentialBridge` 中 provider/model 字符串映射。后续只能委托 `ResolvedModelRoute`，不能继续长 provider-specific 规则。
- `dead`：旧 Tauri Provider facade、`lime-rs/src/**`、`lime-rs/resources/models` 本地 catalog、旧 `get_api_key_providers` / `get_system_provider_catalog` 命令族。

## 实施阶段

### Phase 0：计划与版本化

- [x] PRD 落到 `internal/roadmap/models/prd.md`。
- [x] 执行计划落到 `internal/exec-plans/models-runtime-unification-plan.md`。
- [x] `.gitignore` 放行 models roadmap 和本计划，保证它们成为 versioned artifact。

### Phase 1：Typed Model / Provider 合同

- [x] 在 `app-server-protocol` 增加 `ModelInfo`、`ModelCapabilitiesInfo`、`ProviderInfo`、`ProviderKeyInfo`。
- [x] `model/list`、`modelProvider/list`、`modelProvider/catalog/list`、`modelProvider/read`、`modelProvider/create/update`、`modelProvider/fetchModels`、`modelProviderKey/create/update` 改为单一 typed response。
- [x] `modelProvider/create`、`modelProvider/update` 改为 typed params，不再接收 `provider: Value` / `patch: Value`。
- [x] 服务端在 `local_data_source/model_projection.rs` 把现有 service / DAO JSON projection 收敛成 protocol DTO；JSON 只停留在内部适配边界。
- [x] 前端 API 网关从 App Server camelCase DTO 投影到现有 UI snake_case view model，不再把 raw App Server 对象透传给 UI。
- [x] 同步 JSON schema、TS client 生成物、手写 protocol 类型和 API 网关定向测试。
- [x] 完成 `model/list/updated` typed 全局失效事件：Provider/Key/目录 mutation 经 App Server event bridge 广播，Renderer 模型注册表失效缓存并刷新已挂载视图。

### Phase 2：RouteResolver 合同

- [x] 定义 `ModelTaskRequest` / `ResolvedModelRoute` / `RouteFailure`。
- [x] 聊天 runtime 写入 typed `modelTaskRequest` / `resolvedRoute` / `routeFailure` evidence，并让 read model 投影到 `thread_read.model_routing`。
- [x] 把 `ModelTaskRequest` 构建抽到 App Server 内部共享边界，聊天 runtime 和媒体 task artifact 不再各自拼同一套字段。
- [x] 聊天 runtime 已基于 `ModelTaskRequest.requirements` 与注册表 `CapabilitySnapshot` 产出 `capability_gap`，并在 capability gap 时通过 `RouteFailure(category=capability_gap)` 阻断执行。
- [x] 新增通用 `model_route_assembly`，把 provider / endpoint / auth / protocol / route defaults / capability snapshot / `RouteFailure` 组装从聊天模块抽出，聊天内部 `model_route_resolver` 只负责 ready candidate、registry metadata、provider record 和 evidence 编排。
- [x] 媒体 task 创建链路在 provider record、enabled key、registry declared model 都明确时，复用 `model_route_assembly` 写入真实 `resolved_route` / `resolvedRoute`，并保留 capability gap fail-closed 语义。
- [ ] 把 RouteResolver 从 App Server 内部模块提升为可复用 RuntimeCore resolver，并覆盖音频 / 转写 / embedding 等非 chat task 的 worker 执行；当前 `ResolvedModelRoute` 组装、route evidence bundle、decision/fallback/not_possible evidence 生成已在 RuntimeCore，App Server 只保留 provider record / readiness / registry metadata 适配；图片、Fal 视频、xAI 视频和 OpenAI-compatible 音频 worker 已消费 `resolved_route` 的 endpoint/protocol 与 durable `credentialRef`，通过 App Server media task worker 精确取凭证后进入 `model-provider` 网络边界；转写与 embedding 仍需完成 task worker / current caller 收口。
- [x] 媒体 task 创建链路在 registry 明确声明能力快照时写入 `route_failure` / `model_route_assessment`，并在 `capability_gap` 时标记任务为 blocked。
- [x] 图片/视频 worker 消费 task payload 中的 capability/route failure 并 fail closed，不再请求 provider；只有 catalog 显式声明 `image_generation` / `video_generation` 和对应输出模态时才生成 ready execution binding。
- [x] 拆出 `media_task_payload.rs`，让媒体 payload / `ModelTaskRequest` 构建离开超过 `1000` 行的 `media_task.rs`。
- [x] 图片 worker 消费 `resolved_route` 的 provider / model / protocol / endpoint 安全投影，并消费通用 `route_failure` fail closed；旧 route-only task artifact 按 current execution binding 幂等补齐。
- [x] App Server 为图片和视频任务写入不含 secret 的 `model_route_execution` / `modelRouteExecution` 执行绑定，worker 按 durable `credentialRef` 精确取凭证，再由 media-runtime 执行 resolved Provider route。
- [ ] 补齐剩余 provider protocol coverage；OpenAI Images API、Responses image-generation、Fal video-generation 与 xAI video-generation lowering/transport 已进入 `model-provider`。xAI current worker 已覆盖 start、durable request id、poll、done/failed/expired/timeout/cancelled，并可在重启后从 request id 恢复 poll；普通 OpenAI 视频声明继续 fail closed。

### Phase 3：Canonical LLM Runtime

- [x] 定义 canonical `Request` / `Message` / `ContentPart` / `LlmEvent`。
- [x] 增加 OpenAI Chat、OpenAI Responses、Anthropic Messages、Gemini、Ollama、OpenAI Images、Responses image-generation、Fal video-generation 与 xAI video-generation 的 protocol mapper 边界。
- [x] Agent adapter 降级为 compat backend adapter，只做合同转换和事件投影。

### Phase 4：媒体任务复用

- [x] 图片任务通过 RouteResolver 选择模型和协议，并由 App Server worker + media-runtime 消费 `resolved_route` / `model_route_execution` 的 direct-provider 执行绑定。
- [x] Fal 与 xAI 视频任务通过 public `mediaTaskArtifact/video/create`、统一 catalog admission、exact credential route、App Server worker、`model-provider` transport 和 media-runtime durable artifact 进入 terminal；xAI 任务持久化 request id/status，恢复时只 poll 不重复 start。
- [x] 音频任务通过 RouteResolver 选择 `openai_audio_speech`，由 App Server audio worker 消费 durable credential、写入 workspace 相对音频路径并完成 artifact；转写 worker 仍未接入 RuntimeCore mapper。
- [x] 图片 / 视频 media task artifact 持久化 `resolved_route`、`llm_events`、`provider_diagnostics`，同时保留 snake/camel 双字段便于 current UI / evidence 消费。
- [ ] 转写 media task artifact 持久化 executable `resolved_route` / `llm_events` / `provider_diagnostics`；音频已具备真实 TTS worker，云端 embedding 已经委托 `model-provider`，退出条件是 transcription task worker 与 embedding current caller 完成 route 消费。
- [ ] 前端设置页、Agent Chat、媒体工作台统一消费 typed capabilities。

## 本轮当前刀

本轮继续固定参考边界：`grok-build` 是多模型控制面的 primary reference；`opencode` 只补 provider wire、canonical media content 和 lowering。两者都必须翻译到 Lime current owner，不直接成为 runtime、catalog 或 GUI owner。

本轮新增的候选解析 current slice：RuntimeCore 持有 typed `CandidateModelSet`，App Server 适配 configured Provider 声明模型与缓存目录；能力/任务族/输入输出模态过滤先于 OEM 偏好、连续性和成本排序，显式选择保持首位但仍由后续 capability gate fail closed。候选集合及选择依据写入 routing decision evidence。

本轮快速完成 Phase 4 的视频媒体任务产品执行骨架，并保持图片 current 闭环：

1. 在 `runtime-core::llm_protocol::canonical` 保持 `Request` / `Message` / `ContentPart` / `LlmEvent` 为模型协议唯一输入边界。
2. 把 OpenAI Chat、OpenAI Responses、Anthropic Messages、Gemini、Ollama、OpenAI Images、Responses image-generation、Fal/xAI video-generation 的协议映射集中到 `model-provider`。
3. 图片和视频 worker 请求体构建及 Provider HTTP 复用 `model-provider`，不在 `media-runtime` 维护第二套 Images API / Responses image-generation / video wire。
4. 图片和视频 worker 在 running / succeeded / failed 状态写入 canonical LLM event 投影，并把 provider/model/protocol/transport/credential 诊断持久化到 task payload。
5. 只做纯协议映射，不在 mapper 里承接数据库、认证、transport 或 GUI 逻辑。
6. App Server 保留 provider record、credential 和 readiness facade，media-runtime 只执行 App Server 已解析的 endpoint/protocol，并使用按 durable ref 精确取得的凭证。
7. 音频只允许 `openai_audio_speech` 明确协议并由 dedicated worker 执行；本地 Whisper/SenseVoice、百度/讯飞保留各自 owner，转写 task worker 后续接入；视频只允许 Fal/xAI 明确协议，普通 OpenAI 视频声明继续 fail closed。
8. 通过定向 Rust 测试和全文扫描确认协议层与媒体 evidence 已经收口。

这样做的主线收益是：模型输入/输出、工具调用、图片/文件 parts、provider wire body 和图片/视频任务 evidence 已经有统一协议边界；专用媒体模型不会混入 chat picker，后续新增协议只需要补 catalog capability、mapper 和 dedicated worker，不需要在 App Server、worker、前端三处重复补特判。

## 验证计划

- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-protocol`
- `cargo check --manifest-path "lime-rs/Cargo.toml" -p app-server`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server model_route_assembly -- --nocapture`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::model_route_resolver -- --nocapture`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::model_route_contract -- --nocapture`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::model_routing -- --nocapture`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server model_task_contract -- --nocapture`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server media_task_payload -- --nocapture`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime::tests::read_model::read_session_projects_model_routing_into_thread_read -- --nocapture`
- `cargo check --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime model_route -- --nocapture`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime --test model_route_execution -- --nocapture`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime capability_gap -- --nocapture`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server --test media_task_jsonrpc xai_video_task_start_and_poll_run_from_public_jsonrpc`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server --test media_task_jsonrpc video_task_create_executes_current_worker_from_public_jsonrpc`
- `npm run check:protocol-types`
- `npm run test:contracts`

本轮不触碰用户可见 GUI，默认不跑 `npm run verify:gui-smoke`。如果后续 Phase 4 改媒体工作台或模型设置页，再补 GUI smoke 和受影响 UI 回归。

## 风险与退出条件

- `local_data_source/model_projection.rs` 仍从现有 service / DAO JSON projection 读取，字段缺失时使用空值；退出条件是 `ModelRegistryService` / Provider DAO 直接返回 protocol DTO 或专用 projection。
- `customModels` 等可空数组依赖 schema 生成器正确保留数组 item 类型；生成器已补 `type: ["array", "null"]` 支持，后续必须由 `npm run check:protocol-types` 守住。
- 本轮没有修改 Provider 数据库存储 schema，也没有改旧数据格式；旧 Provider / API Key 数据由投影层自动映射到 typed response，route evidence 只进入事件 / artifact payload，因此无需新增数据迁移脚本。后续若 Phase 2/4 引入新的持久化 route / model binding 字段，必须补启动期迁移或 scripts 迁移入口。
- 当前能力门禁只在注册表明确返回能力快照时阻断；direct provider config、registry 缺失或模型没有声明任务族 / 模态 / 能力时不按未知能力失败。退出条件是 RouteResolver 具备显式的 `unknown / inferred / declared` 能力置信度，并能把用户可行动错误返回给 GUI。
- 媒体任务写入 `model_task_request` 与 capability assessment，并只在最终 provider/model、credential-scoped metadata 与 capability snapshot 明确时为图片/视频写入 ready `resolved_route` 与 `model_route_execution`。worker 以 `media_task_worker + resolved_route_credential_ref` 校验执行绑定，按同一 durable ref 精确取 key；route-only task artifact 会执行期按同一 current 合同补齐顶层 binding。音频仍没有产品 worker；视频 current 骨架覆盖 Fal 同步响应和 xAI 异步 start/poll/terminal/recovery，live provider 证据与 GUI/history projection 仍是退出条件。
- `runtime_contract` 仍是 App Server protocol 和 TS client 中的 JSON/unknown 字段，因为前端工作台、Skill launch、ImageTaskViewer 和历史 task payload 都在消费它；本轮只把 App Server 默认生成和列表投影收敛到单一模块，不做全协议 typed 化。退出条件是同步 protocol schema、TS generated/client、前端 view model、mock/fixture 后，把 `runtimeContract?: unknown` 升级为 typed display/runtime metadata DTO。
- `runtime_backend/model_routing.rs` 仍超过 `1000` 行，本轮没有继续往里追加业务逻辑，只删除了迁移后未使用的 `service_model_slot()` helper。退出条件是后续把 profile slot 解析、readiness、payload projection 分拆成独立小模块。

## 进度日志

### 2026-07-28

- 视频 transport 从 `media-runtime` 收回唯一网络 owner `model-provider`：Fal 同步 POST；xAI `POST /videos/generations -> request_id -> GET /videos/{id}`，覆盖 done/failed/expired/timeout/cancelled，并严格使用 resolved auth header/prefix。
- `media-runtime` 只持久化 `provider_task.protocol/request_id/status` 与 terminal artifact；App Server media scheduler 仅恢复 stale、已有 request id 的 xAI running task，恢复只 poll、不重复 POST。普通 OpenAI 视频声明在路由和 lowering 两侧 fail closed。
- 公共 JSON-RPC xAI start/poll 与 Fal 回归均通过；`model-provider` lowering/transport、media-runtime start/poll/resume 和 App Server scheduler 定向测试已通过。最终 `npm run test:contracts` 通过 `808` generated protocol types、`284` client checks 及 modality/scripts/Electron release/docs guards；workspace rustfmt、`git diff --check` 与 legacy governance 通过，治理摘要为零引用候选 `0`、边界违规 `0`、既有 deprecated 分类漂移 `1`。分类：上述为 `current`；media-runtime 直接发视频 HTTP、把所有视频诊断写成 Fal、普通 OpenAI 视频 fallback 为 `dead / deleted / forbidden-to-restore`；无新增 `compat/deprecated`。

### 2026-07-22

- 图片媒体执行合同已从旧本机 broker 描述改为 current App Server media task worker：binding key 复用 `mediaTaskArtifact/image/create`，endpoint 只取 `resolved_route`，凭证只按 `resolved_route_credential_ref` 精确解析，telemetry transport 标记为 `provider_http`。App Server 不再为没有产品 worker 消费者的视频 task 写 ready execution binding。
- 删除固定本机图片/视频 path、route header 和 runner-config endpoint evidence；route/evidence 不写 Provider base URL 或明文 secret。
- Renderer Fal / Gemini / OpenAI-compatible / standard Provider 直连 executors、本机图片 HTTP executor、batch runner、历史状态、错误解析、正向测试与旧 live test 已按 `dead / deleted / forbidden-to-restore` 清理；current 工作台创建、轮询和取消均走 `mediaTaskArtifact/*` App Server JSON-RPC。`useImageGen` 已收缩为 Provider、模型和尺寸选择 owner，不再暴露 `generateImage/cancelGeneration`。
- 图片工作台“保存到素材库”已从 `hookImageId -> useImageGen image-gen-history` 迁到 `ImageWorkbenchOutput.url -> projectMaterial/importFromUrl` current 主链，保存中状态归 `useWorkspaceImageWorkbenchActionRuntime`。`useImageGen.saveImagesToResource/backfillImagesToResource` 及其历史记录类型和测试已物理删除，图片生成产品能力保持在 current media task 主链。
- Renderer 图片执行面收口验证：`useImageGen.preference`、图片 catalog 与 legacy catalog 共 243/243 通过；工作台 Provider/画布边界、素材导入与 action runtime 共 45/45 通过；`npm run governance:legacy-report` 为 0 分类漂移、0 边界违规。`image-command` Gate B controlled fixture 为 `ok=true`、`backendMode=runtime`，真实 Electron/preload/IPC/App Server 主链命中，任务产物 `succeeded`、可见预览刷新后恢复，legacy command、mock fallback、页面和控制台错误均为 0。仓库级 `npm run typecheck` 被并行协议热区 `packages/agent-runtime-client/src/eventVerifier.ts` 对新增 reasoning notification 的未穷尽类型收窄阻断，本轮图片定向测试与 renderer production build 均通过，未跨写集修复该错误。`npm run test:related -- <图片路径>` 还暴露智能入口把仓库 `electron/` 目录当文件读取的 `EISDIR` 发现器问题，因此改用明确测试文件取得上述结果。
- 图片 Gate B Electron fixture 已转绿：`claw-chat-current-fixture-image-command-scoped-credential-v2-summary.json` 为 `ok=true`。受控 Electron 场景证明文本模型发现与 presentation 请求均携带 credential，纯文生图的 `ModelTaskRequest.requirements.inputModalities=[text]`，图片 Provider 收到一次带凭证的 `/v1/images/generations`，`lime-image-api-worker` 将同一 task 推进 `succeeded`，read model / GUI 进入 terminal，刷新后任务卡与预览图恢复，且 production mock fallback、console/page/invoke error 命中均为 0。本证据等级为 Gate B controlled fixture，不扩张为 live Provider 证明。
- 验证：App Server route binding 1/1、image payload 1/1、media-runtime route unit 8/8、route integration 9/9、media-runtime lib 52/52、治理边界扫描通过。
- 本次补证：`media_task_payload::tests` 6/6；`image-command` Gate B controlled fixture 通过。`npm run test:rust:related -- lime-rs/crates/app-server/src/media_task_payload.rs` 为 1463/1469，余下 6 项均位于并行修改的 canonical projection / mailbox 热区，不属于图片 payload 或 worker 主链，本车道未跨写集处理。

### 2026-06-18

> 以下为历史执行 evidence；其中 `local_lime_service` / 本机 broker 与视频 worker 的描述已被 2026-07-22 direct-provider 图片合同取代，不代表 current 产品路径。

- Phase 3 canonical LLM protocol 第一刀：新增 `runtime-core/src/llm_protocol/`，定义 `LlmRequest`、`LlmMessage`、`LlmInputPart`、`LlmOutputPart`、`LlmEvent`、`ProviderWireRequest`，显式支持 text / image / audio / file / tool call / tool result。
- 协议 mapper 已拆成 `mapper/mod.rs`、`common.rs`、`openai_responses.rs`、`openai_chat.rs`、`anthropic_messages.rs`、`gemini.rs`、`ollama_chat.rs`，避免继续形成单文件巨型 mapper。
- OpenAI Responses mapper 修正工具声明为顶层 `type=function`，并保持 tool call 写入顶层 `function_call`、tool result 写入 `function_call_output`；OpenAI Chat、Anthropic、Gemini、Ollama 分别保持各自 provider-native body shape。
- `OpenaiImages`、`Fal`、`BedrockConverse`、`Unknown` 当前不进入 canonical LLM mapper，直接返回 `UnsupportedProtocol`，避免把媒体协议伪装成聊天协议。
- Agent adapter 已收口到 route protocol：`RuntimeProviderConfig` 通过 `ResolvedModelRoute.protocol` 选择 `ChatCompletions` / `Responses`，continuation capability 也只看 route protocol，不再按 provider/model 字符串猜测。
- `OPENAI_FORCE_RESPONSES_API` 只保留为 Agent OpenAI provider 内部 compat env flag，不再作为 runtime 路由事实源；相关 env-mutating 测试已改为串行执行，避免并发污染。
- Phase 2/3 交界收口：`runtime-core::route_evidence_payload` 成为 `modelTaskRequest` / `resolvedRoute` / `routeFailure` evidence bundle 的唯一组装事实源；App Server `runtime_backend/model_route_contract.rs` 删除重复实现，`model_route_resolver.rs` 只调用 RuntimeCore API。
- Phase 2/3 交界继续收口：RuntimeCore 新增 `route_resolution_evidence_payloads`，统一生成 chat resolver 的 decision / fallback / not_possible evidence payload；App Server `runtime_backend/model_route_resolver.rs` 不再自己拼 fallback / not_possible route evidence，只保留 DB、Provider readiness、registry metadata 与 direct provider config 适配。
- Phase 3/4 交界继续收口：RuntimeCore 新增 OpenAI Images API 与 Responses image-generation body builder，`media-runtime` 图片 worker 的 Images API / Responses image-generation 请求体构建改为复用 RuntimeCore `LlmRequest` mapper；worker 内部删除重复的 Responses image-generation tool/model 归一化 helper，避免协议 body shape 在 RuntimeCore 与 media worker 双写。
- Phase 3/4 交界继续收口：RuntimeCore 新增 Fal video-generation body builder，视频 worker 的本地 broker 请求体改为复用 RuntimeCore `LlmRequest` + 受控 metadata mapper；`media-runtime/src/video_worker.rs` 不再手写 Fal body shape，只保留 task payload 读取、local broker endpoint / Authorization header、route preflight、响应解析和状态编排。
- 安全边界保持不变：图片 worker 仍只调用本地服务 broker endpoint，继续由 `model_route_execution` 校验 `local_lime_service`、`secretMaterialStatus=not_embedded` 与 route provider/model/protocol 一致；RuntimeCore mapper 不处理 API key、endpoint 改写、SSE 解析或 fallback。
- 视频安全边界保持不变：视频 worker 仍只调用本地服务 broker endpoint，继续由 `model_route_execution` 校验 `local_lime_service`、`secretMaterialStatus=not_embedded` 与 route provider/model/protocol 一致；RuntimeCore 的 Fal mapper 只构建 body，不处理 API key、endpoint 改写或响应解析。
- Phase 4 图片 / 视频媒体任务 evidence 收口：新增 RuntimeCore canonical `LlmEvent -> runtime event` 纯映射，`media-runtime` 在 image/video running、success、failed patch 中写入 `llm_events` / `llmEvents` 与 `provider_diagnostics` / `providerDiagnostics`；route-aware task 会记录 provider/model/protocol，旧 payload 没有 route 时不伪造 provider/model。
- Phase 4 route 集成回归补强：`model_route_execution` image/video 集成测试同时断言最终 task payload 的 `turn.completed`、provider/model/protocol、`local_lime_service` transport 和 `not_embedded` credential，确保媒体复用不是只停留在请求体覆盖。
- Phase 4 保守边界：本轮接通 OpenAI-compatible TTS worker，并把云端 embedding/ OpenAI Whisper wire 收回 `model-provider`；本地 Whisper/SenseVoice、百度和讯飞仍由各自领域 owner 保留。转写 task artifact worker 与统一 route projection 仍是下一刀。
- 清理外部参考项目名残留：代码注释改成中性“多协议模型运行时”表述；Provider 图标 surface 和本地 provider 识别里的旧 provider id 按 `dead` 清掉；删除已无引用的旧 provider 图标资源文件。
- 本轮没有数据库 schema、Provider 存储或 task artifact 结构变更；旧数据无需迁移脚本。后续如果把 `TaskExecutionContract` typed 化或把 route/event 写入持久化 schema，再补启动期迁移或 scripts 迁移入口。
- 验证结果：`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-protocol -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p runtime-core model_route -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-agent-target" cargo test --manifest-path "lime-rs/crates/agent-rust/Cargo.toml" -p agent-core test_uses_responses_api -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::model_route_resolver -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-runtime-core-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p runtime-core llm_protocol -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-runtime-core-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p runtime-core -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-media-runtime-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime execute_image_generation_task_should_support_responses_image_generation_executor -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-media-runtime-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime execute_image_generation_task_should_fallback_to_images_api_when_responses_route_missing -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-media-runtime-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime --test model_route_execution -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-media-runtime-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-media-runtime-target" cargo check --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime`、`CARGO_TARGET_DIR="/tmp/lime-media-video-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime execute_video_generation_task_should_advance_task_file_to_succeeded -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-media-video-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime --test model_route_execution -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-media-video-check-target" cargo check --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime`、`CARGO_TARGET_DIR="/tmp/lime-runtime-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p runtime-core model_route -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-app-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::model_route_resolver -- --nocapture`、`rustfmt --edition 2021 --check ...`、`npx prettier --check "internal/exec-plans/models-runtime-unification-plan.md" "internal/roadmap/models/prd.md" "packages/app-server-client/src/generated/protocol-types.ts"`、`git diff --check -- ...` 均通过；外部参考项目名的内容扫描和文件名扫描均无结果。`/tmp/lime-model-route-target` 曾因并发 Cargo 编译产生 dep-info / fingerprint 写入错误，已改用独立 target 目录重跑相关测试并通过。

### 2026-06-17

- 新增本计划，绑定 `internal/roadmap/models/prd.md`。
- 确认模型 Provider current 主链为 App Server JSON-RPC `model*` / `modelProvider*`。
- 用户明确要求“重构、清理，不留历史包袱”后，Phase 1 改为单一 typed JSON-RPC contract，删除 `provider: Value` / `patch: Value` 入参和 `models/providers/provider/key: Value` 出参。
- 新增 App Server protocol DTO：`ModelInfo`、`ModelCapabilitiesInfo`、`ProviderInfo`、`ProviderKeyInfo`。
- 新增 `local_data_source/model_projection.rs`，把 service / DAO 现有 JSON projection 约束在 App Server 内部边界。
- 前端 `modelRegistry.ts` / `apiKeyProvider.ts` 改为消费 typed App Server DTO，再投影到现有 UI view model。
- Electron Host 保留 legacy 数组返回面，但从 typed App Server `ModelProviderListResponse` / `ModelListResponse` 读取时显式兜底为 `[]`，避免 schema 默认数组在 TS 中表现为 optional 后破坏桌面壳类型检查。
- 同步 schema fixtures、`packages/app-server-client` 生成物和定向测试。
- 未修改 Provider 数据库存储 schema；旧数据通过投影层自动迁移到 typed contract，本轮无需新增迁移脚本。
- 验证结果：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-protocol`、`cargo check --manifest-path "lime-rs/Cargo.toml" -p app-server`、`npm run check:protocol-types`、`npm run test:contracts`、`npm run typecheck:electron`、`npx vitest run "electron/hostCommands.test.ts"`、`npx vitest run "src/lib/api/modelRegistry.test.ts" "src/lib/api/apiKeyProvider.test.ts"`、`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all --check`、`npx prettier --check ...` 均通过。
- `npm run typecheck` 仍复现 TypeScript 编译器内部 `Debug Failure. No error for last overload signature`，没有落到业务文件诊断；后续若要继续收窄，需要单独做 TS 编译器级别排查，不应把它误判成本轮模型协议回归。
- Phase 2 第一刀：新增 App Server protocol DTO：`ModelRefSource`、`ModelRef`、`ModelTaskKind`、`ModelTaskSource`、`CapabilityRequirement`、`ModelTaskRequest`、`ProtocolKind`、`EndpointKind`、`EndpointInfo`、`AuthKind`、`AuthMaterialRef`、`TransportKind`、`FramingKind`、`RouteDefaults`、`CapabilitySnapshot`、`RoutingDecision`、`RouteFailureCategory`、`RouteFailure`、`ResolvedModelRoute`。
- 同步 schema fixtures 和 `packages/app-server-client/src/generated/protocol-types.ts`；`scripts/generate-protocol-types.mjs` 生成后格式化生成物，避免 protocol check 与 Prettier 打架。
- 新增 `runtime_backend/model_route_contract.rs`，让聊天 runtime 的 `routing.decision.made`、`routing.fallback.applied`、`routing.not_possible` 事件写入 typed `modelTaskRequest` / `resolvedRoute` / `routeFailure` evidence。
- 新增 `model_task_contract.rs`，把 `ModelTaskRequest` 构建规则从聊天 backend 与媒体任务里抽到 App Server 内部共享边界；媒体 image/video/audio artifact payload 写入 `model_task_request` / `modelTaskRequest`。
- `runtime/read_model.rs` 的 `thread_read.model_routing` 保留 typed route evidence，便于 GUI、evidence export 和后续调试复用。
- 聊天 runtime 已根据 `ModelTaskRequest.requirements` 与注册表 `CapabilitySnapshot` 做保守能力门禁：注册表明确返回能力时，缺 `vision` / `image` 等能力会产出 `RouteFailure(category=capability_gap)` 并在执行前 fail closed；未知能力快照不阻断。
- 为恢复 App Server 编译，修复当前工作树中 conversation import `codex` path helper 可见性与 `ImportedRuntimeEventTurn` trait 实现签名不匹配；这是验证阻塞修复，不属于模型运行时主线。
- 未修改数据库 schema；没有旧数据需要迁移，本轮无需新增迁移脚本。
- 验证结果：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all --check`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::model_route_contract -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server media_task::tests -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime::tests::read_model::read_session_projects_model_routing_into_thread_read -- --nocapture`、`cargo check --manifest-path "lime-rs/Cargo.toml" -p app-server`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-protocol`、`npm run check:protocol-types`、`npm run test:contracts` 均通过。
- Phase 2 媒体阻断收口：`LocalAppDataSource` 在 image / video / audio task 创建前读取 `ModelRegistryService`，只有当 registry 中明确匹配到 provider/model 且声明了能力快照时才做 capability assessment；未知或未声明能力不阻断。
- `media_task` payload 在 capability gap 时写入 `failure_code=failureCode=capability_gap`、`route_failure/routeFailure`、`model_route_assessment/modelRouteAssessment`，并继续保留 `model_task_request/modelTaskRequest`；仍不写 `resolved_route`，避免伪造 endpoint/auth/protocol。
- `lime-media-runtime` 的 image / video worker 已在执行前消费 `failure_code=capability_gap` 并 fail closed，不再请求 provider；audio 由 App Server media worker 消费 `resolved_route`，写入 workspace 相对音频路径并通过 complete API 形成结果与事件投影。
- 为避免继续扩张巨型文件，新增 `media_task_payload.rs` 承接媒体 payload、runtime contract 和 `ModelTaskRequest` 构建；`media_task.rs` 降到 `731` 行，只保留 task artifact 读写、列表、完成态和取消编排。
- 未修改数据库 schema；route assessment 仍是 task payload JSON evidence，旧任务缺字段时按 accepted / 正常执行路径处理，本轮无需新增迁移脚本。
- 新增回归：`model_task_contract::tests::media_route_assessment_reports_capability_gap`、`media_task_payload::tests::image_payload_marks_route_failure_when_assessment_blocks`、`execute_image_generation_task_should_fail_closed_on_capability_gap`、`execute_video_generation_task_should_fail_closed_on_capability_gap`。
- 验证结果：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all --check`、`cargo check --manifest-path "lime-rs/Cargo.toml" -p app-server`、`cargo check --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server model_task_contract -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server media_task_payload -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::model_route_contract -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime::tests::read_model::read_session_projects_model_routing_into_thread_read -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime capability_gap -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-protocol`、`npm run check:protocol-types`、`npm run test:contracts` 均通过。
- Phase 2 RouteResolver 抽象：新增 `runtime_backend/model_route_resolver.rs`，把聊天路由里的 ready candidate 选择、provider readiness、registry metadata、provider record、`ModelTaskRequest`、`ResolvedModelRoute`、`routing.decision.made` / `routing.fallback.applied` / `routing.not_possible` evidence payload 统一到内部 resolver 输出。
- `runtime_backend.rs` 现在只负责 turn 生命周期编排和事件顺序，消费 `ChatModelRouteResolution` 后继续交给现有 Agent provider wire；未新增 App Server JSON-RPC method，未修改 Electron bridge，未改变 provider 执行协议。
- 删除 `model_routing::ModelRoutingDecision::service_model_slot()` 迁移后未使用 helper，避免在超过 `1000` 行的 routing 文件继续留下 dead surface。
- 新增 resolver 回归覆盖：direct provider config 不被 registry unknown 阻断、禁用 provider 返回 `RouteFailure(category=provider_disabled)`、声明能力不匹配的 image chat 返回 `capability_gap`、Anthropic provider 映射到 `anthropic_messages` / provider base URL / API key ref、unready coding slot fallback 到 ready base slot 并保留 attempts evidence。
- 未修改数据库 schema；聊天 route evidence 仍落 runtime event payload，旧 session / task 数据无需迁移，本轮无需新增迁移脚本。
- 验证结果：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all --check`、`cargo check --manifest-path "lime-rs/Cargo.toml" -p app-server`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::model_route_resolver -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::model_route_contract -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::model_routing -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime::tests::read_model::read_session_projects_model_routing_into_thread_read -- --nocapture` 均通过。
- Phase 2 通用 route assembly：新增 `model_route_assembly.rs`，把 `ResolvedModelRoute` 中的 protocol / endpoint / auth / transport / framing / defaults / capability snapshot / failure 构造从 `runtime_backend/model_route_contract.rs` 抽离；`model_route_contract.rs` 降为聊天 `ModelTaskRequest` 与 evidence payload adapter。
- 新增非 chat 回归：`model_route_assembly::tests::image_task_route_preserves_task_model_ref_and_reports_capability_gap`，验证 image task route 可以保留 `ModelRefSource::Task`、`image_generation_model` routing slot，并在模型只有 chat/text 能力时输出 `capability_gap=task_family:image_generation`。
- 媒体 artifact 仍未写入真实 `resolved_route`；本轮只完成可复用 route assembly，不改变 media worker provider wire，也不伪造 endpoint/auth/protocol。
- 未修改数据库 schema；route assembly 是运行时内存合同抽象，旧数据无需迁移，本轮无需新增迁移脚本。
- 验证结果：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all --check`、`cargo check --manifest-path "lime-rs/Cargo.toml" -p app-server`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server model_route_assembly -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::model_route_resolver -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::model_route_contract -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime::tests::read_model::read_session_projects_model_routing_into_thread_read -- --nocapture`、`npm run test:contracts` 均通过。
- Phase 2 媒体 route 持久化：`LocalAppDataSource` 创建 image / video / audio task 时把 `ApiKeyProviderService`、`ModelRegistryService` 和 `DbConnection` 一并传入 media route assessment；只有 provider record 存在、provider 启用且有 enabled key、registry 匹配到 declared model capability snapshot 时，才通过 `model_route_assembly` 生成真实 `ResolvedModelRoute` 并写入 payload 的 `resolved_route` / `resolvedRoute` / `model_route_assessment.resolvedRoute`。
- 未知 provider、registry 未声明能力或无法匹配 provider/model 时仍保持旧路径：只写 `model_task_request`，不伪造 route；capability gap 时继续写 `failure_code=capability_gap` 与 `route_failure`，worker fail closed。
- 新增回归：`media_task_payload::tests::image_payload_contains_resolved_route_when_assessment_resolves_route` 覆盖 media artifact payload 写入 typed `resolved_route`，并确认 `ModelRefSource::Task` 与 `image_generation_model` routing slot 保留。
- 未修改数据库 schema；`resolved_route` 是 task artifact payload JSON evidence，旧 task 缺字段时 worker 仍按原 runner config 执行，本轮无需新增迁移脚本。
- 验证结果：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all --check`、`cargo check --manifest-path "lime-rs/Cargo.toml" -p app-server`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server media_task_payload -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server model_route_assembly -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::model_route_resolver -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime::tests::read_model::read_session_projects_model_routing_into_thread_read -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime capability_gap -- --nocapture`、`npm run test:contracts` 均通过。
- Phase 2 媒体 worker route consumption：新增 `lime-media-runtime/src/model_route.rs`，在 media runtime 内部解析 task artifact 中的 `resolved_route` / `model_route_assessment.resolvedRoute` / `route_failure` JSON 投影；为避免层级反向依赖，当前不让 media runtime 直接依赖 App Server protocol crate。
- 图片 worker 与视频 worker 在执行前优先消费 `resolved_route.modelRef.providerId/modelId` 覆盖旧 payload provider/model；图片 worker 根据 route protocol 保守映射现有 executor mode（`openai_responses/codex_responses -> responses_image_generation`，`openai_images/fal -> images_api`）。旧 task 缺字段时继续使用原 payload / runner config。
- worker 现在消费通用 `route_failure` fail closed，不再只识别 `failure_code=capability_gap`；仍保留 capability gap 回归，避免已阻断的媒体任务误请求 Provider。
- 明确安全边界：本轮不使用 `resolved_route.endpoint.baseUrl` 直连 Provider，因为当前 worker 只有本地 Lime 服务 API key；直接覆盖 endpoint 会把本地服务 key 发往第三方 Provider。Provider credential 仍由本地 Lime 服务 / App Server current owner 解析。完整 provider protocol mapper / credential resolver 留到下一刀。
- 新增 integration 回归：`lime-media-runtime/tests/model_route_execution.rs` 覆盖 image/video worker 使用 `resolved_route` 的 provider/model 进入 `X-Provider-Id` 和请求 body；新增 `model_route` 单元测试覆盖 top-level / nested route projection 与 capability gap failure 解析。
- Phase 2 route protocol 修正：`model_route_assembly` 的协议推断不再只按 provider 类型决定；image task 在 registry/runtime feature 声明 `images_api` 时输出 `ProtocolKind::OpenaiImages`（Fal 输出 `Fal`），声明 `responses_api` 时输出 `OpenaiResponses`，让 `ResolvedModelRoute.protocol` 更贴近媒体 worker 的执行语义。
- 未修改数据库 schema；`resolved_route` 仍是 task artifact payload JSON evidence，旧 task 缺字段时按原 runner config 执行，本轮无需新增迁移脚本。
- 验证结果：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all --check`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server model_route_assembly -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server media_task_payload -- --nocapture`、`cargo check --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime model_route -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime --test model_route_execution -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime capability_gap -- --nocapture` 均通过。
- Phase 2 执行绑定清理：新增 `model_route_execution.rs`，App Server 在 route accepted 的媒体 task payload 中写入不含 secret 的 `model_route_execution` / `modelRouteExecution`，声明 executor 为 `local_lime_service`、endpoint 来自 runner config、credential owner 为本地 Lime 服务 / `api_key_provider_store`；`resolved_route.endpoint.baseUrl` 仍只作为 evidence，不被 worker 用作请求地址。
- `lime-media-runtime` 对 `modelRouteExecution` 做 fail-closed 校验：只接受本地 Lime 服务 broker、对应媒体任务的 binding key、`runner_config` endpoint source、`secretMaterialStatus=not_embedded`，并校验 execution route 的 provider/model/protocol 与 `resolvedRoute` 一致；发现 direct provider、内嵌 secret 或 binding 不匹配时以 `unsupported_route_execution` 失败，不请求 provider。
- 旧 task 数据处理：没有数据库 schema 改动；执行期自动迁移 route-only task artifact，worker 会补齐顶层 `model_route_execution` / `modelRouteExecution` 后继续执行，不覆盖既有 `model_route_assessment` evidence，因此无需外部迁移脚本。
- 新增回归：`model_route_execution::tests::image_route_execution_binding_delegates_credentials_to_local_service`、`model_route::tests::local_execution_patch_migrates_route_only_image_payload`、`model_route::tests::route_execution_validation_rejects_embedded_secret`、`image_worker_fails_closed_on_embedded_route_secret`、`video_worker_fails_closed_on_non_local_route_execution`，并扩展 image/video integration 断言旧 route-only task 会自动补齐 execution binding。
- 验证结果：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all --check`、`cargo check --manifest-path "lime-rs/Cargo.toml" -p app-server`、`cargo check --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server model_route_execution -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server media_task_payload -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime model_route -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime --test model_route_execution -- --nocapture` 均通过。
- Phase 2 route preflight 清理：`media-runtime/src/model_route.rs` 现在集中负责 image/video 的 route failure、protocol 支持、execution binding 校验和 route-only payload 迁移；`lib.rs` / `video_worker.rs` 只保留 patch / fail 编排与最终请求输入组装，避免继续在两个 worker 中复制兼容判断。
- route-only 迁移边界收紧：执行期补齐 `modelRouteExecution` 时复用 `resolvedRoute.auth` 的 `credentialRef` / `headerName` / `headerPrefix` 元数据，不再写死 Authorization/Bearer；已有 App Server current `model_route_assessment.routeExecution` 时不会生成顶层迁移字段，避免把 current evidence 反向覆盖成 compat 形态。
- 测试清理：能力缺口 worker 回归从 `media-runtime/src/lib.rs` 和 `video_worker.rs` 巨型实现文件移到 `tests/model_route_execution.rs`，实现文件不再继续承载新增 route 合同测试。
- 验证结果：`rustfmt --edition 2021 --check "lime-rs/crates/media-runtime/src/model_route.rs" "lime-rs/crates/media-runtime/src/lib.rs" "lime-rs/crates/media-runtime/src/video_worker.rs" "lime-rs/crates/media-runtime/tests/model_route_execution.rs" "lime-rs/crates/app-server/src/model_route_execution.rs"`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo check --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime model_route -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime --test model_route_execution -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-media-runtime capability_gap -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server model_route_execution -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server media_task_payload -- --nocapture` 均通过。使用独立 target 目录是为了避开当前工作区已有长时间 cargo 进程持有共享 target 锁。
- Phase 2 audio route 收口：`create_audio_media_task_artifact` 与 image/video 共用 route assessment，`create_audio_payload` 持久化 `resolved_route` / `model_route_execution`；audio worker 只接受 `openai_audio_speech`，从 durable `credentialRef` 精确取凭证后调用 `model-provider`，失败/重试/过期恢复均回写当前 task artifact。
- 新增回归：`media_task_payload` 的 audio typed request 保持 `voice_generation_model` routing slot；route assessment 成功时写入 executable `resolved_route` 和 `mediaTaskArtifact/audio/create` binding，缺 route assessment 时仍不伪造执行字段。
- 验证结果：`npx prettier --check "internal/exec-plans/models-runtime-unification-plan.md" "internal/roadmap/models/prd.md"`、`rustfmt --edition 2021 "lime-rs/crates/app-server/src/local_data_source/media_tasks.rs" "lime-rs/crates/app-server/src/local_data_source/impls/media.rs" "lime-rs/crates/app-server/src/media_task.rs" "lime-rs/crates/app-server/src/media_task_payload.rs"`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo check --manifest-path "lime-rs/Cargo.toml" -p app-server`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server media_task_payload -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server model_route_execution -- --nocapture` 均通过。
- Phase 2 runtime contract 生成 / 投影侧清理：新增 `media_runtime_contract.rs`，把 image / video / audio 默认 `runtime_contract` 的 contract key、routing slot、required capability、execution profile、executor adapter、executor binding 和 policy refs 收敛为单一 typed builder；`media_task_payload.rs` 不再保留三份手写 JSON，文件从 `591` 行降到 `512` 行；`media_task.rs` 的列表索引不再直接读取 `runtime_contract` 内部 JSON path，改为消费 `MediaRuntimeContractProjection`，文件降到 `665` 行。
- 明确分类：`model_task_request` / `resolved_route` / `model_route_execution` 是可执行路由 `current` 合同；`runtime_contract` 当前只保留为 GUI / Skill 元数据 `compat` JSON 字段，不参与 worker route credential / endpoint 决策。
- 新增回归：`media_runtime_contract::tests::default_image_contract_preserves_existing_json_shape`、`media_runtime_contract::tests::override_contract_is_preserved`、`media_runtime_contract::tests::projection_reads_top_level_and_contract_metadata`、`media_runtime_contract::tests::projection_falls_back_to_camel_runtime_contract_key`，确保默认 contract JSON shape 不破坏前端现有消费，调用方传入的历史/custom `runtime_contract` 不被覆盖，且列表索引 projection 同时支持 snake/camel 历史字段。
- 验证结果：`rustfmt --edition 2021 "lime-rs/crates/app-server/src/media_runtime_contract.rs" "lime-rs/crates/app-server/src/lib.rs" "lime-rs/crates/app-server/src/media_task_payload.rs" "lime-rs/crates/app-server/src/media_task.rs"`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server media_runtime_contract -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server media_task_payload -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo check --manifest-path "lime-rs/Cargo.toml" -p app-server`、`git diff --check -- "lime-rs/crates/app-server/src/media_runtime_contract.rs" "lime-rs/crates/app-server/src/media_task_payload.rs" "lime-rs/crates/app-server/src/media_task.rs" "lime-rs/crates/app-server/src/lib.rs"` 均通过。
- Phase 2 前端 route fallback 清理：图片工作台草稿任务重试不再从 `runtimeContract.providerId/model` 回填 provider/model 到新任务参数；`runtimeContract` 在前端继续作为 ImageTaskViewer / 预览文案元数据，不能再反向成为新任务路由事实源。原 task artifact 重试仍可复用 payload 中的显式 `provider_id/model`，因为那是原任务的输入事实，而不是展示 contract。
- 新增回归：`useWorkspaceImageWorkbenchActionRuntime.taskActions.test.tsx` 增加“草稿重试不应从 runtimeContract 展示元数据恢复 provider/model 路由”，覆盖未显式选择 provider/model 时重试请求应交由 App Server RouteResolver 重新解析。
- 验证结果：`npx prettier --check "src/components/agent/chat/workspace/useWorkspaceImageWorkbenchActionRuntime.ts" "src/components/agent/chat/workspace/useWorkspaceImageWorkbenchActionRuntime.taskActions.test.tsx"`、`npx vitest run "src/components/agent/chat/workspace/useWorkspaceImageWorkbenchActionRuntime.taskActions.test.tsx"` 均通过。
- Phase 2 `@配音` route 假面清理：`resolveServiceModelSendOverrides` 不再把 `service_scene_run.preferred_provider_id/preferred_model_id` 提升成聊天发送 `providerOverride/modelOverride`。TTS 偏好仍保留在 `harness.service_scene_launch.service_scene_run`，供未来 voice runtime 消费；外层 Agent turn 继续由当前聊天模型/RouteResolver 决策，避免把 TTS 模型误当聊天模型。
- `voice_generation` 合同恢复 executable：App Server 默认 voice `runtime_contract` 与 route execution binding 均指向 `mediaTaskArtifact/audio/create`；`lime-audio-provider-worker` 消费 `openai_audio_speech` route，写入音频文件、result、`llm_events` 和 `provider_diagnostics`。
- 新增 / 更新回归：`media_runtime_contract::tests::default_voice_contract_is_executable`、audio worker workspace path guard、`useWorkspaceSendActions.test.tsx` 的 `@配音` provider/model override 断言。
- 验证结果：`rustfmt --edition 2021 "lime-rs/crates/app-server/src/media_runtime_contract.rs"`、`CARGO_TARGET_DIR="/tmp/lime-model-route-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server media_runtime_contract -- --nocapture`、`npx prettier --write "src/components/agent/chat/workspace/commands/serviceModelHelpers.ts" "src/lib/governance/modalityRuntimeContracts.ts" "src/lib/governance/modalityRuntimeContracts.test.ts"`、`npx vitest run "src/components/agent/chat/workspace/useWorkspaceSendActions.test.tsx"`、`npx vitest run "src/lib/governance/modalityRuntimeContracts.test.ts"` 均通过。
- 2026-06-17 23:30 CST：Phase 2 `voice_generation` 历史包袱继续清理。`modalityExecutionProfiles.json` 将 `voice_generation_profile` 与 `service_skill:voice_runtime` 从 `current` 降为 `compat`，移除 `executor_invoked` 证据，只保留 GUI / Skill 元数据与 audio_task 草稿索引；`modalityArtifactGraph.json` 也移除 `audio_task` / `audio_output` 的 `executor_invoked`，避免 artifact graph 继续暗示已有可执行音频 worker。
- 新增治理守卫：`scripts/check-modality-runtime-contracts.mjs` 对 `route_execution_status=metadata_only` 的 contract 强制要求 `route_execution_exit_condition`，并禁止 contract / execution profile / executor adapter / artifact graph 声明 `executor_invoked` 或把 profile / adapter 标为 `current`。`modalityExecutionProfiles.ts` 透传 lifecycle，`modalityExecutionProfiles.test.ts` 固定 `voice_generation` 在 audio worker 接入前只能作为 metadata-only compat。
- Active 文档清理：`internal/roadmap/warp/execution-profile.md`、`implementation-plan.md`、`README.md`、`internal/aiprompts/commands.md`、`quality-workflow.md` 不再宣称配音媒体 worker / 本地 ServiceSkill 已真实执行音频；`internal/aiprompts/prompt-foundation.md` 顺手收掉已删除 `lime-rs/src/**` 作为 Prompt current owner 的旧路回流，改指向 App Server request context 与 `lime-rs/crates/agent`。
- 旧数据处理：本轮只调整 registry / 文档 / 守卫，不改数据库 schema 和 task artifact payload；既有 audio task 仍按 metadata-only 读取，因此无需迁移脚本。
- 验证结果：`npm run governance:modality-contracts`、`npx vitest run "src/lib/governance/modalityExecutionProfiles.test.ts" "src/lib/governance/modalityRuntimeContracts.test.ts"`、`git diff --check -- "scripts/check-modality-runtime-contracts.mjs" "src/lib/governance/modalityExecutionProfiles.ts" "src/lib/governance/modalityExecutionProfiles.test.ts" "src/lib/governance/modalityExecutionProfiles.json" "src/lib/governance/modalityRuntimeContracts.json" "src/lib/governance/modalityArtifactGraph.json"` 均通过。
- 2026-06-17 23:45 CST：Phase 2 `voice_generation` 假 executor surface 继续清理。App Server 默认 voice `runtime_contract` 不再序列化 `executor_adapter` / `executor_binding`，前端 `modalityRuntimeContracts.json` 将 `voice_generation` contract lifecycle 降为 `compat` 并移除 executor binding；`resolveModalityExecutionProfileBinding` 在 metadata-only contract 下不再从 profile 自动回填 adapter。`src/lib/desktop-host/mediaTaskMocks.ts` 和音频任务索引测试也不再输出 `service_skill:voice_runtime` executor adapter key。
- 旧数据自动迁移边界：`MediaRuntimeContractProjection` 对 `voice_generation` / `route_execution_status=metadata_only` payload 自动压制历史 executor adapter / binding 字段；旧 JSON artifact 无需重写，但 `list_media_task_artifacts.modality_runtime_contracts` 不再聚合这些假 executor 维度。
- 守卫补强：`scripts/check-modality-runtime-contracts.mjs` 允许 metadata-only contract 不声明 executor binding，同时会从 profile 下挂 adapter 继续检查 metadata-only adapter 不能被误升为 `current` 或声明 `executor_invoked`。Warp contract schema / execution profile / implementation plan 文档同步改为“可执行 current contract 才必须声明 executor_binding”，防止旧规则逼出假执行器。
- 2026-08-16：audio worker 补齐输出目录/写入/artifact 完成失败的 `failed` 回写，并在取消竞态下只清理 worker 自己生成的默认音频路径；`TaskArtifactPatch` 对嵌套 object 改为递归合并，避免完成状态覆盖 `audio_path` / `mime_type` 等已持久化字段。`TranscriptionGenerate` 与 `Embedding` 暂不写入 executable `model_route_execution`，直到各自的 task worker / current caller 真正消费 `resolved_route`；相关负向回流测试已补齐。`lime-media-runtime` 56 项、`app-server-protocol` 120 项、`npm run test:contracts` 与模态守卫通过；app-server 全量检查仍被本机 `rusty_v8 v150.4.0` arm64 预编译包 404 阻断。
