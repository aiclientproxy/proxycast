# Lime v1.130.0 发布执行计划

状态：release-candidate-ready-waiting-git-confirmation
日期：2026-08-17
目标版本：`1.130.0`
目标 tag：`v1.130.0`

## 主目标

发布当前 `main` 工作树中的 Codex runtime gap alignment、转写 worker current owner、provider 增量请求、Hook 生命周期、MCP 工具并发、标准 AGENTS 发现、ProjectThread-first 与 Harness/治理收敛改动；完成版本事实源、双语单页 release notes、质量门禁、release commit、tag、`main`/tag 推送和远端复核。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划及执行计划导航。
- `candidate changes`：版本元数据写入前工作树中全部 81 个受跟踪改动和 4 个未跟踪文件，覆盖 RuntimeCore/App Server/model-provider、tool-runtime、agent-runtime、媒体/Skills、脚本、测试、架构、路线图、治理、多语言和 Harness 图资产；治理门禁修复额外更新 2 个受跟踪事实源。计入发布元数据后，提交前候选共 90 个受跟踪文件和 5 个新增文件。
- `excluded changes`：无。用户请求为完整发布，未跟踪文件与本轮 runtime/Harness 主题直接相关；不覆盖或拆分并发写集。

## 退出条件

- 根应用、CLI npm 包、Rust workspace 与 Cargo.lock 统一为 `1.130.0`，双语 release notes 只保留 v1.130.0，目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck` 必须执行；按风险执行 `npm run test:contracts`、受影响 Rust/前端定向测试、`npm run smoke:agent-runtime-current-fixture`、`npm run verify:gui-smoke`，未执行或失败项必须原样记录。
- staged 内容与上述候选范围一致；获得危险操作确认后创建 `Release v1.130.0` commit、`v1.130.0` tag，推送 `main` 和 tag，并复核本地/远端状态。
- 收尾记录 current/compat/deprecated/dead 分类、完成度、GUI/Gate B 状态和未覆盖的平台/打包证据，不把环境限制误报为通过。

## 当前验证记录

- v1.129.0 为当前 HEAD 与 `origin/main`，`v1.130.0` 本地和远端 tag 均不存在。
- 版本事实源和双语 release notes 已更新；候选包含版本元数据之外的全部现有工作树改动。
- commit、tag、push 尚未执行，等待危险操作确认；staged 摘要和 tag 状态将在确认请求前再次复核。

## 门禁结果

- 通过：`npm run verify:app-version`。
- 通过：`npm run typecheck`。
- 通过：`npm run test:contracts`；在治理修复后独立复跑第二次仍通过，覆盖 protocol types、App Server client contract、command/harness/modality/scripts/Electron release workflow/docs boundary 等合同。
- 通过：`npm run test:rust:related -- lime-rs/crates/agent-runtime lime-rs/crates/agent lime-rs/crates/app-server lime-rs/crates/model-provider lime-rs/crates/tool-runtime lime-rs/crates/core lime-rs/crates/skills`。首次仅 `tool-runtime` 的 3 个测试因缺少 `code-mode-host` 前置产物失败；使用 `scripts/lib/rusty-v8-artifacts.mjs` 提供的 V8 环境预构建该产物后完整重跑通过，其中 `agent-runtime` 210、`app-server` 1678、`tool-runtime` 356 个测试通过。
- 通过：受影响前端/治理定向 Vitest，10 个文件共 441 个测试；Claw scenario registry 6 个测试；治理修复后的 `legacySurfaceCatalog.test.ts` 224 个测试。
- 通过：`npm run smoke:agent-runtime-current-fixture`，覆盖真实 Electron/App Server current fixture，`liveProviderUsed=false`。
- 通过：`npm run verify:gui-smoke`；App Server 报告版本 `1.130.0`，Gate B evidence 为 `standalone-shell-01-20260816225825-49240`。
- 通过：`npm run governance:legacy-report`。首次发现 `.lime/AGENTS.md` fallback 与治理规则冲突，已将其收敛为标准 AGENTS owner 下的受控 `compat`，更新 `internal/aiprompts/governance.md` 与 `src/lib/governance/legacySurfaceCatalog.json` 后重跑通过：扫描 2115 个源码文件，分类漂移 0、边界违规 0。
- 已知非产品失败：两次 `npm run test:related -- ...` 均触发 Vite runner 的 `EISDIR ... /lime/electron` 缺陷；随后使用等价 `npx vitest run <files>` 验证实际测试全部通过。该 runner 缺陷不冒充产品测试失败，也不在本次候选中扩写修复范围。
- 提交前仍需复跑 `git diff --check`、`npm run verify:app-version`，并复核 staged 内容和本地/远端 tag。

本地不执行 Windows/macOS packaged parity、签名、公证、正式 release asset 和 CI 门禁；这些证据不在本候选中宣称完成。

## 风险与限制

- 当前候选横跨 Rust runtime、provider、App Server、脚本、治理和五语言资源；门禁必须以实际命令结果为准。
- 本机不能替代 Windows/macOS packaged parity、签名、公证和 CI 发布资产证据；任何失败或未执行项都保留在最终记录中。

## 架构确认

架构影响：重大。候选更新了 `internal/aiprompts/architecture.md`，明确转写 worker、provider route 和 current runtime 链路；同时纳入 `internal/tech/lime-agent-harness-architecture.*` 图源与渲染资产。责任开发者：root，确认日期：2026-08-17。

## 收尾记录

- `current`：Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI 主链，以及 model-provider、tool-runtime Hook/MCP、transcription worker 和标准 AGENTS owner；本次能力继续在这些 owner 演进。
- `compat`：`.lime/AGENTS.md` / `.lime/AGENTS.override.md` 仅作为标准 AGENTS 发现之后的受控 fallback，由 current owner 委托，不形成第二套实现。
- `deprecated`：本候选未新增或保留独立演进的 deprecated 能力。
- `dead / deleted`：移除已脱离构建图的 `runtime/tests/evidence_exports.rs` 测试入口，并由治理 catalog/测试阻止旧 evidence-export 专用入口回流。
- 当前完成度：95%。版本、release notes、候选范围、架构确认与本地质量门禁已完成；剩余 5% 是 staged 复核、用户危险操作确认、release commit、tag、`main`/tag 推送和远端复核。
