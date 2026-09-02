# Lime Soul 个性化实施计划

> 状态：current rollout plan
> 更新时间：2026-07-06
> 目标：用小步方式把全局 Soul 能力接入 Lime 个性化主链，同时避免和 Memory 形成重复事实源。

## 1. 实施原则

1. 先文档边界，再实现。
2. 先交互人格，再正式创作声线。
3. 先 current config，再导入 / 导出。
4. 先短 prompt section，再做摘要和自动建议。
5. 每一刀都必须继续收敛到 `memory_profile_prompt_service` / `runtime_turn` / `Generation Brief`。
6. `SOUL.md` 只能是导入 / 导出格式，不进入运行时读取主链。
7. Expert Plaza 只做当前专家会话的局部 persona / skill / workflow 联动，不写回 Global Soul。
8. Style Profile 必须走 `Style Pack Registry` / `Style Resolver` / prompt context，不让 UI 或 i18n 自己 hard-code 四套句库。

## 2. 当前实施快照

截至 2026-07-06，本路线图按“先全局交互人格、再创作声线”的顺序推进：

1. 已完成 Phase 0 文档边界，`internal/roadmap/soul/` 是当前规划事实源。
2. 已完成 Phase 1 全局 Soul 设置，配置落在 `memory.soul`，复用 current app config 读写，不新增 Tauri 命令。
3. 已完成 Phase 2 普通聊天注入，`memory_profile_prompt_service` 统一构建 `【全局交互人格】` section，关闭或空配置不注入。
4. 已推进 Phase 2.5 Style Profile：四个首发风格已按 built-in Style Pack seed 进入 registry 口径，并开始同步到前端 resolver 与 App Server `memory_soul_prompt_context`；后续重点是工具 lifecycle facts、UI descriptor 和真实 GUI evidence。
5. 已完成 Phase 3 `SOUL.md` 导入 / 导出，导入前预览，导入后写 current config，不保留文件路径依赖。
6. 已接入 Phase 4 current 主链，`memory.soul.artifact_voice` 仍是正式内容声线的保存配置事实源，但普通设置页不再暴露声线 ID / evidence 表单；hook 只读取保存配置，发送层按“本轮开关 + 无显式 generation_brief”把保存声线作为 fallback 归一化到 `artifact.generation_brief`；正式 Artifact metadata 默认补 `generation_brief` 声线边界，`voice_source=none`，不继承 Global Soul / Expert Persona；显式 `generation_brief` 可独立保留并覆盖保存声线，但不会单独触发 Artifact 生成合同。
7. 已推进 Phase 5 专家联动，`expertRuntimeBinding` 输出 personality boundary metadata，`runtime_turn` 将 `harness.expert` 识别为专家会话上下文。

本轮清理：

1. `normalizeSoulArtifactVoiceConfig` 会在配置层清理缺失 / 未知 `voice_source` 下的孤儿 `creator_voice_id` / `brand_voice_id`，避免保存配置本身变成隐式声线事实源。
2. `artifactGenerationBriefMetadata` 是前端工作区发送层收敛显式 voice metadata、清理 camelCase alias、深合并 `artifact` metadata 的 current helper。
3. `artifact_generation_brief_boundary_service` 是后端 Generation Brief 声线边界默认值、alias normalize 和显式 voice metadata 保留的单一事实源。
4. `artifact_request_metadata_service` 只负责 artifact metadata 归一化并调用该边界服务。
5. `artifact_prompt_service` 只负责把已经归一化的边界投影成 prompt section，不再维护另一份默认值逻辑。
6. 清理了只带 `generation_brief` 时被 request metadata 删除、反向误触发 Stage / Schema prompt、workspace base / sendOptions 浅合并丢 artifact 字段，以及缺失 / 未知 `voice_source` 仍携带 voice ID 的边界风险。
7. 清理了保存 Soul 声线提前 merge 到 `workspaceRequestMetadataBase` 的旧做法；保存声线不再作为基础 metadata 常驻，只在发送瞬间按本轮开关和显式覆盖规则进入请求，并写入 `diagnostics.soul_artifact_voice` 解释来源与 guard 结果。
8. 清理了设置页不合理表单：`MemorySettings` 从 `日常记忆 / AI 个性 / 创作声线 / 高级` 收口为 `日常记忆 / AI 个性 / 高级`；AI 个性只保留当前状态、初始模板和重置，不再让普通用户手填 Soul 字段；SOUL.md 内容查看提前到高级页，导入仅作为迁移草稿入口。

当前不要宣称完成：

1. Phase 4 自动 voice evidence 投影、创作流程内声线选择、诊断层详情展示和完整 GUI / E2E 证据。
2. Phase 2.5 全对话输出面：不能只用欢迎语、profile id 或少量 runtime status 证明完成；还缺工具前后、正文段落、失败恢复、结尾建议的真实 transcript / GUI 证据。
3. Phase 6 品牌 / 项目声线包。
4. Phase 7 session-scoped personality overlay。

## 3. Phase 0：口径和路线图落盘

目标：

1. 固定 Soul 是 Memory 个性化子路线。
2. 固定 OpenClaw / Hermes 的参考边界。
3. 固定不新增 `soul_*` 事实源。
4. 固定 `SOUL.md` 只做 import / export。
5. 固定 artifact 默认不受 Product Soul 污染。

主产物：

1. `internal/roadmap/soul/README.md`
2. `internal/roadmap/soul/prd.md`
3. `internal/roadmap/soul/architecture.md`
4. `internal/roadmap/soul/diagrams.md`
5. `internal/roadmap/soul/rollout-plan.md`
6. `internal/roadmap/soul/acceptance.md`

验收：

- 文档明确回链 `internal/roadmap/memory/README.md`。
- 文档使用 `current / compat / deprecated / dead` 分类语言。
- 文档明确不新增平行事实源。

## 4. Phase 1：全局 Soul 设置

目标：

1. 在设置页提供 AI 个性初始模板、当前状态和高级迁移入口。
2. 支持选择模板、保存、重置；不让普通用户直接手填 Soul prompt 字段。
3. 配置写入 current app config 或 memory profile 投影。
4. 不触及正式 artifact。
5. 不触碰 Expert Catalog 作为全局事实源。

建议改动：

1. 扩展设置页 IA，将入口放在用户资料 / 记忆偏好附近。
2. 定义 Soul config 类型和默认值。
3. 复用现有配置读写网关，不新增裸 `invoke`。
4. 用户可见文案覆盖 Lime 五语言。
5. 组件测试覆盖保存、关闭、重置和空态。

不做：

1. 不导入文件。
2. 不影响 Generation Brief。
3. 不新增数据库表。
4. 不把专家 `personaRef` 复制到 Soul config。

验证：

```bash
npm exec vitest run "src/components/settings-v2"
npm run test:contracts
```

如果设置页主路径改动明显，补：

```bash
npm run verify:gui-smoke
```

## 5. Phase 2：普通聊天交互注入

目标：

1. Global Soul 影响普通聊天语气。
2. prompt section 统一由 `memory_profile_prompt_service` 构建。
3. section 使用稳定 marker 防重复。
4. 关闭 Soul 后不再注入。
5. 专家会话可以继承 Global Soul 的交互节奏，但 Expert Persona 仍由 expert runtime metadata 定义。

建议改动：

1. 在 `memory_profile_prompt_service` 增加 `build_global_soul_prompt`。
2. 保持与 memory profile、memory sources 同一 merge 边界。
3. 更新 runtime turn 测试，断言普通聊天包含 Soul marker。
4. 更新关闭状态测试，断言 marker 缺失。
5. 保持系统、安全、用户当前指令优先级说明。

不做：

1. 不把完整 Markdown 文件直接塞进 prompt。
2. 不让 Soul section 包含路径、密钥或诊断信息。
3. 不改变工具执行策略。
4. 不让 Expert Persona 反向覆盖 Global Soul。

验证：

```bash
cargo test --manifest-path "lime-rs/Cargo.toml" memory_profile_prompt_service
npm run test:contracts
```

## 5.5 Phase 2.5：Style Pack Registry 与全输出面口吻

目标：

1. 四个首发风格注册为 built-in Style Pack seed，而不是一个共享 default pack 或组件里的四个分支。
2. `memory_soul_prompt_context` 携带 profile id、pack id、voice primitives、surface contracts、anti-repetition rules、few-shot anchors 和 risk fallback。
3. i18n 只保留 neutral UI copy；禁止 `agentChat.soulInteraction.<tone>.*` profile 句库。
4. 工具调用前、中、后、正文小标题、段落转折、失败恢复和结尾建议都按 L0-L4 分层验收。

建议改动：

1. `src/lib/soul/style-profiles/builtInProfiles.ts` 表达四个 built-in Style Pack seed。
2. `lime-rs/crates/app-server/src/runtime/soul/style_profile.rs` 读取同一组 `src/lib/soul/style-profiles/packs/*.json` built-in manifest，不再维护后端四套静态 profile 对象。
3. `memory.soul.style_profile_id` 配置层保存 registry profile id 字符串，不再用四个 built-in 枚举或旧 alias 映射限制风格包。
4. `prompt_context.rs` 输出 surface contract、anti-repetition、few-shot anchor 和 risk fallback。
5. 工具 lifecycle read model 增加 `phase / status / toolName / facts / riskLevel / styleLevel / profileId / packId`。
6. Agent Chat 工具卡片、折叠条和 timeline 只消费 descriptor，不从 profile id 拼人格化文案。

不做：

1. 不训练 LoRA / QLoRA 作为首版依赖。
2. 不新增 `personalstyle` Runtime、表或 prompt composer。
3. 不把 few-shot anchor 作为固定可渲染句子。
4. 不让正式 artifact 默认继承 Product Soul。

验证：

```bash
npm exec vitest run src/lib/soul
cargo test --manifest-path "lime-rs/Cargo.toml" soul_prompt_context
node scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs --scenario soul-style --timeout-ms 180000 --prefix soul-style-smoke
```

## 6. Phase 3：SOUL.md 导入 / 导出

目标：

1. 支持从 OpenClaw / Hermes 风格 `SOUL.md` 导入。
2. 导入前展示预览和风险提示。
3. 导入后写入 current config，不依赖原始文件。
4. 支持导出当前 Soul 配置为 Markdown。

建议改动：

1. 增加 import adapter，输入为 Markdown 文本或用户选择文件内容。
2. adapter 输出 Soul draft、风险提示、项目规则提示。
3. 导入确认后调用 current 配置写入。
4. 导出只读取 current config。
5. 加测试覆盖空文件、项目规则、普通 persona、避免事项。

不做：

1. 不新增文件 watcher。
2. 不记录导入文件路径作为运行时依赖。
3. 不直接把导入文本作为 system prompt。

验证：

```bash
npm exec vitest run "src/components/settings-v2" "src/lib"
npm run test:contracts
```

如新增 Tauri 文件读取命令，补 Rust 定向测试。

## 7. Phase 4：Creator / Brand Voice 进入 Generation Brief

目标：

1. 正式创作可显式使用个人 / 品牌声线。
2. Product Soul 默认不影响 artifact。
3. `personality_boundary_guard` 决定 voice 字段是否进入本轮。
4. 影响解释能关联用户配置或 evidence。

建议改动：

1. 在生成入口增加“使用我的创作声线”策略。
2. 从 `unified_memory_*`、knowledge pack 或显式配置读取 voice evidence。
3. 在 Generation Brief 编译阶段加入 voice section。
4. 诊断层显示 voice 来源和 guard 结果。
5. 测试 artifact 默认不包含 Product Soul。
6. 测试 Expert Persona 默认不直接进入正式 artifact。

当前已落地的最小切片：

1. 已在前端工作区发送层新增 `artifactGenerationBriefMetadata` helper，将显式 voice metadata 归一化为 `artifact.generation_brief`。
2. 已清理 `generationBrief` / `generation_brief` 别名并保持 sendOptions 覆盖 workspace base 的优先级。
3. 已下线普通设置页里的 `正式内容声线` 表单；正式内容声线应在创作流程中选择或由后续 evidence 流程形成，不再要求用户在设置页填写 voice ID / evidence refs。
4. 已在 `useSoulArtifactVoiceGenerationBrief` 从保存配置读取 `memory.soul.artifact_voice` 并投影为保存声线候选；发送层不再提前合并到 workspace base，而是在本轮开关开启、且没有显式 `generation_brief` 时才应用。
5. 已在配置 normalize、前端 metadata helper 和 Rust 归一化边界清理互斥或孤儿的 `creator_voice_id` / `brand_voice_id`，防止保存声线和本轮显式声线混合，也防止缺失 / 未知 `voice_source` 隐式带入正式产物声线。
6. 已在 artifact request metadata 归一化阶段补齐 `generation_brief` 边界。
7. 已在 Artifact prompt composition 中增加 `【Generation Brief 声线边界】` section。
8. 默认 `voice_source=none`，`inherits_global_soul=false`，`inherits_expert_persona=false`。
9. 显式 `generation_brief` 只作为声线 guard / 诊断上下文独立保留，不作为 Artifact 交付合同开关。
10. 已在 Workspace 输入区增加“创作声线”本轮开关；关闭后本轮不注入保存声线，但保留 `diagnostics.soul_artifact_voice.status=disabled_for_turn` 解释拦截原因。
11. 该切片已经具备显式配置、本轮开关、发送层 fallback 和诊断投影，但还不等于完整 Creator / Brand Voice 产品能力；自动 evidence 投影、诊断详情 UI 和完整 E2E 仍需继续。

不做：

1. 不把 Companion Soul 注入 artifact。
2. 不把所有聊天历史当 voice evidence。
3. 不让用户无感知地改变正式输出声线。
4. 不把专家目录当成品牌声线 evidence，除非用户显式确认。

验证：

```bash
npm exec vitest run "src/components/agent/chat" "src/features/knowledge"
cargo test --manifest-path "lime-rs/Cargo.toml" runtime_turn
```

GUI 主链补：

```bash
npm run verify:gui-smoke
```

## 8. Phase 5：Expert Persona 联动收口

目标：

1. 专家会话能清楚展示当前专家 persona、memory template、skill 和 workflow。
2. 专家会话可继承 Global Soul 的沟通节奏。
3. 专家人格不写回 Global Soul。
4. 专家结果要进入正式创作声线时，必须经用户确认和 Generation Brief guard。

建议改动：

1. 保持 `expertRuntimeBinding` 是 Expert Persona 的 runtime metadata 投影边界。
2. 在专家启动或诊断层标明 Global Soul 只影响交互方式。
3. 为专家会话补测试，断言 expert metadata 与 Soul config 分离。
4. 后续如支持“将专家输出沉淀为声线”，必须写入 Memory / Knowledge evidence，而不是 Soul config。

不做：

1. 不把 Expert Catalog 作为 Global Soul 事实源。
2. 不把专家 `personaRef` 转成 `SOUL.md` 自动导入。
3. 不让专家会话默认污染普通聊天或正式 artifact。

验证：

```bash
npm exec vitest run "src/features/experts" "src/components/experts"
npm run test:contracts
```

## 9. Phase 6：品牌 / 项目声线包

目标：

1. 支持将 Creator / Brand Voice 作为知识库或灵感库对象管理。
2. 支持多品牌、多项目选择。
3. 声线包进入 Generation Brief 前必须经过用户选择或任务匹配。
4. 支持查看影响解释。

建议改动：

1. 复用 Knowledge / `unified_memory_*`，不新增 `voice_*` 主链。
2. 声线包作为 knowledge pack / memory identity projection。
3. 生成入口可以选择目标声线。
4. 保存满意结果时可建议更新声线，但必须用户确认。

不做：

1. 不自动从所有历史结果训练声线。
2. 不默认跨项目共享品牌声线。
3. 不绕过删除 / 禁用 / 导出边界。

验证：

```bash
npm exec vitest run "src/features/knowledge" "src/components/memory"
npm run test:contracts
```

## 10. Phase 7：临时 personality overlay

目标：

1. 支持当前 session 临时切换模式。
2. 切换不清空历史。
3. overlay 不默认沉淀为长期 Soul。
4. 用户可选择将 overlay 保存为长期配置。

建议改动：

1. 在 session state 增加 scoped overlay。
2. 提交 turn 时通过 runtime metadata 或 config snapshot 传递短 overlay。
3. runtime 添加 pivot marker，帮助模型从下一轮切换风格。
4. 清除 overlay 后回到 Global Soul。
5. 明确 overlay 不影响 artifact，除非进入 Generation Brief。

不做：

1. 不把 overlay 自动写入 `memory.profile`。
2. 不把 overlay 写进历史 memory。
3. 不允许 overlay 覆盖安全 / 工具约束。

验证：

```bash
npm exec vitest run "src/components/agent/chat"
cargo test --manifest-path "lime-rs/Cargo.toml" runtime_turn
```

## 11. 迁移与兼容

1. 现有 `memory.profile` 不迁移。
2. 现有用户资料不迁移。
3. 现有 project memory 的 character `personality` 保持 compat，不作为 Soul 主链。
4. OpenClaw / Hermes 用户通过导入 `SOUL.md` 迁移。
5. 导入后的内容由 Lime current 配置接管。
6. Expert Catalog / expert agent instance 保持专家系统主链，不迁移为 Soul。

## 12. 回滚策略

1. Phase 1 可通过关闭配置入口隐藏。
2. Phase 2 可通过 Soul enabled=false 停止 prompt 注入。
3. Phase 3 可禁用导入 / 导出入口，不影响已有配置。
4. Phase 4 可关闭创作声线进入 Generation Brief。
5. Phase 7 可清除 session overlay。
6. Expert 联动可关闭诊断和继承提示，不删除专家目录。

回滚不应删除用户已有配置，除非用户显式重置。

## 13. 质量门槛

涉及文案：

```text
zh-CN / zh-TW / en-US / ja-JP / ko-KR
```

涉及命令：

```bash
npm run test:contracts
```

涉及 GUI 主路径：

```bash
npm run verify:local
npm run verify:gui-smoke
```

涉及 Rust runtime：

```bash
cargo test --manifest-path "lime-rs/Cargo.toml"
```

## 14. 当前验证记录

截至 2026-06-03：

1. Phase 4 前端定向测试已通过：`npm exec vitest run "src/lib/soul/soulConfig.unit.test.ts" "src/hooks/useSoulArtifactVoiceGenerationBrief.test.tsx" "src/components/settings-v2/general/memory/index.test.tsx" "src/components/agent/chat/utils/artifactGenerationBriefMetadata.unit.test.ts" "src/components/agent/chat/workspace/workspaceSendHelpers.test.ts" "src/components/agent/chat/workspace/useWorkspaceInputbarSceneRuntime.test.tsx"`，当前结果为 6 files / 50 tests passed。
2. 测试覆盖：Soul config normalization、SOUL.md import / export、设置页默认隐藏高级 / 声线表单、AI 个性模板写入 `memory.soul`、SOUL.md 文件内容查看、保存配置到 Generation Brief 的候选投影、发送时 saved fallback、本轮关闭不注入 artifact 但保留诊断、显式 `generation_brief` 覆盖保存声线且不继承保存 evidence、Generation Brief alias normalize、互斥 voice ID 清理、缺失 / 未知 `voice_source` 不保留孤儿 voice ID、Workspace 输入区本轮开关展示和切换。
3. 本轮设置页五语言 key 已对齐；`npm run i18n:unused -- --check` 仍因既有 `agent` / `agentMessageList` 未引用 key 失败，但 `settings` namespace 已清到 `unused=0`。
4. Playwright 轻量 GUI 检查已覆盖本地 `http://127.0.0.1:1420/` 设置页：记忆页默认只展示 `日常记忆 / AI 个性 / 高级` 三个 tab；`AI 个性` tab 只展示当前状态、三个初始模板、重置和边界说明，无 input / textarea；`高级` tab 将 `SOUL.md 文件内容` 放在导入区之前，空配置显示“当前还没有可查看的 SOUL.md 内容”。
5. `git diff --check` 已针对本轮设置页、五语言和 Soul 路线图写集通过。
6. Rust `artifact_generation_brief_boundary_service` 定向测试已通过：`cargo test --manifest-path "lime-rs/Cargo.toml" artifact_generation_brief_boundary_service -- --test-threads=1`，结果为 6 tests passed。
7. `npm run test:contracts` 已通过。
8. `npm run bridge:health -- --timeout-ms 120000` 已通过，DevBridge health status 为 `ok`。
9. `npm run verify:gui-smoke -- --reuse-running` 本轮已通过 `workspace-ready`、`browser-runtime`、`site-adapters`、`agent-service-skill-entry`、`agent-runtime-tool-surface`、`agent-runtime-tool-surface-page`、`code-runtime-fixture`、`agent-runtime-approval-sandbox`、`at-command-registry` 与 `plugins` 等 GUI / runtime 主路径 smoke；其中 page smoke 已真实打开 `http://127.0.0.1:1420/`，完成输入、发送、打开 code session、查看 runtime summary、审批与文件 checkpoint 回滚等步骤。
10. 完整 `verify:gui-smoke` 尚未闭合：本轮失败点为后续 `smoke:knowledge-gui` 的 `page.reload({ waitUntil: "domcontentloaded" })` 超时；失败后脚本已清理 smoke Chrome profile。定向复跑 `npm run smoke:knowledge-gui -- --app-url http://127.0.0.1:1420/ --health-url http://127.0.0.1:3030/health --invoke-url http://127.0.0.1:3030/invoke --timeout-ms 600000 --interval-ms 1000 --i18n-patch-metrics-output /Users/coso/Documents/dev/ai/aiclientproxy/lime/.lime/i18n/patch-metrics.json` 已越过 reload，并推进到 Agent 结果回流，但失败在点击“保存到项目资料”控件；脚本枚举显示该按钮存在且未禁用。该失败当前归类为 GUI smoke 的 knowledge 页面交互命中 / 脚本稳定性缺口，不是 Soul 声线 metadata 边界失败，但在复跑通过前不能宣称完整 GUI smoke 已通过。
11. 真实 live Provider / 图片生成提交仍按仓库默认门禁跳过；`at-command-registry` 只验证 `@配图` 选择和 prompt 填充，未提交真实模型或图片 Provider。
