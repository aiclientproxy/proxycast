# Lime v1.138.0 发布执行计划

状态：`seventh-follow-up-ready / clean CI timezone verification pending`
日期：2026-09-01
目标版本：`1.138.0`
目标 tag：`v1.138.0`

## 主目标

发布 `v1.137.0` 之后当前工作树中的 Composer、prompt history、模型路由、Desktop capability/readiness、macOS native host、跨平台资源 manifest、Windows packaged Gate B、Agent GUI 与治理改动，完成版本事实源、双语单页 release notes、质量门禁及 release commit/tag/push。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md` 与本计划。
- `candidate changes`：当前工作树中的全部已跟踪产品/文档/测试/schema/生成物改动，以及新增的 Desktop Host/native helper、platform/resource contract、prompt history、Composer Controller、thread queue typed action和对应计划/测试文件。
- `excluded changes`：根目录 `history-DndxIXaj-D1E13r2C.js`、`history-Y-6ViEXW.js`、`local-conversation-page-B5LUHmAw.js`、`use-chatgpt-composer-controller-CrYCVUro.js`。这些文件是 Codex Desktop 压缩 bundle 只读参考，不是 Lime 源码或发布产物。
- `package.json` 边界：工作树曾被外部 Codex Desktop `openai-codex-electron@26.825.41651` manifest 替换；发布准备恢复 Lime `HEAD` manifest 的脚本、依赖和仓库身份，仅把版本提升到 `1.138.0`，不把外部工程 metadata 纳入候选。

## 退出条件

- 根应用、CLI npm 包、Rust workspace 与 Cargo.lock 统一为 `1.138.0`；双语 release notes 只保留 v1.138.0；本地和远端目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck` 必须通过；执行 `npm run test:contracts`、`npm run verify:gui-smoke`、Agent current fixture、受影响 Rust/前端定向测试和 `git diff --check`。
- staged 内容包含全部 candidate changes 和 release metadata，明确排除 4 个 Codex bundle 参考文件；完成 `Release v1.138.0` commit、`v1.138.0` tag，推送 `main` 和 tag，并复核远端 Release/Windows workflow。

## 架构确认

本轮重大架构变更已在 `internal/aiprompts/architecture.md` 记录并确认：Composer Controller、App Server PromptHistoryStore、Desktop capability/readiness、macOS native host 与 desktop resource manifest 均有唯一 current owner；Electron 继续只承接 Desktop Host/IPC/sidecar 生命周期，Agent 产品链保持 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`。责任开发者确认：root，2026-09-01。

## 验证记录

### 发布后 follow-up（2026-09-01）

- 已完成首轮发布：commit `f79830804`、tag `v1.138.0` 已推送，GitHub Release 已公开，但当前 Release 尚无安装资产。
- Release workflow `33513208903` 失败于 Windows `Prepare sherpa-onnx runtime`：7-Zip 对 `.tar.bz2` 只解出中间 `.tar`，命令返回成功但预期 DLL 不存在；Quality workflow `33513188500` 同因 Windows sherpa 步骤失败，Frontend Full 另因 component migration candidates `38 > 37` 失败。
- 本地 follow-up 已让 sherpa 解压在“命令成功但库缺失”时继续执行 tar fallback，并新增回归测试；预算事实源同步为当前已存在的 `38`（脚本默认值、Makefile、Quality workflow 与治理文档）。
- 首轮 tag 已存在且绑定 `f79830804`，不能静默覆盖；完成修复验证后仍需明确执行远端 `v1.138.0` tag 重建，再重新运行 Release workflow，或改发补丁版本。
- Follow-up 验证：sherpa/预算脚本回归 `19/19`，Windows native host、entrypoint 与 Squirrel 合同 `39/39`，`npm run lint`、`npm run typecheck`、`cargo fmt --all -- --check`、Rust related（含 `lime-agent`、App Server、MCP、services、model-provider、tool-runtime 及反向依赖）均通过。
- 真实 Electron 证据：`npm run smoke:agent-runtime-current-fixture` 全部场景通过（`liveProviderUsed=false`）；`npm run verify:gui-smoke` 串行复跑通过，run `standalone-shell-01-20260901143936-28201`，`appserver.v0` version `1.138.0`，summary `result=pass`。并行构建产生的首次资源哈希错配未计入证据。
- 首次 follow-up commit `ba9dbdaae` 已推送并重建 `v1.138.0` tag。Quality run `33521136350` 中 Bridge、GUI Smoke、Integrity、Rust Full 均通过；Release run `33521150327` 中 macOS arm64/x64 均通过，但发布资产因 Windows job 失败未进入聚合上传。
- 第二轮失败根因已由任务日志确认：Windows native host 在 `get_CurrentNativeWindowHandle` 传入 `HWND*`，而 Windows SDK 接口要求 `int*`；Windows Quality 的 PowerShell 命中系统 `tar` 并对 `.tar.bz2` 超时；Frontend Full 的 `electron/systemUtilityHost.test.ts` 在 Linux runner 硬编码了 macOS Accessibility 状态。
- 第二轮修复：native window handle 改为 SDK 契约的 `int` 后再安全提升为 `uintptr_t`；Windows 7-Zip 解出中间 `.tar` 后用同一 7-Zip 解第二层，继续保留 tar 兜底；环境预览测试按真实宿主平台断言 Accessibility。定向回归 `24/24` 通过，仍待 commit/tag 更新后由 Windows runner 证明 MSVC build、Squirrel、CodeMode Gate B 与 native host Gate B。
- 第二轮 follow-up commit `1ac3364de` 已证明 sherpa 双层解压通过、GUI Smoke 通过，但 Release run `33569593531` 的 Windows MSVC 编译仍在 `get_CurrentNativeWindowHandle` 失败。Windows SDK 事实源显示 `typedef void *UIA_HWND`，接口参数为 `UIA_HWND *`；此前 `HWND` 和 `int` 均不匹配。第三轮修复改为直接使用 `UIA_HWND`，并让 `cl` 继承 CI 标准输出，后续失败不再退化为不可读 Buffer 字节数组。
- Quality run `33569587544` 的 Frontend Full 在第 32/119 批出现 5 个 DeepSWE 失败。根因不是 `PATH` 并发污染：默认测试直接依赖被 `.gitignore` 排除的 `.lime/benchmark/sources/deep-swe` 本地 cache，干净 CI 中 source cwd 不存在，Node 启动 `git` 因此报 `spawnSync git ENOENT`，其余断言随 task metadata 缺失级联失败。默认测试现改用临时自包含 source fixture，production preflight 仍默认执行真实 `git rev-parse` 并校验固定 commit；无 live 授权路径也在读取外部 task cache 前 fail closed。
- 第三轮本地验证：Windows/sherpa/Electron 定向回归 `31/31`；DeepSWE adapter/benchmark `29/29`；对应 smart runner 完整并行批次 `108/108`；真实本地 DeepSWE Release 20 preflight `205/205`；受影响 ESLint、`npm run typecheck`、`npm run test:contracts`、`npm run verify:app-version`、脚本治理和 `git diff --check` 全部通过。仍待 follow-up commit/tag 更新后由 `windows-2022` 证明 MSVC、Squirrel、安装态 CodeMode/native host Gate B，并由干净 Linux runner 复验 Frontend Full。
- 第四轮 Desktop Smoke 夹具修复：首次定向复跑仍有 `12` 个桌面契约级联失败，原因是临时 source fixture 的根目录未遵循 `loadTaskDefinition` 的既有 `sourceRoot/tasks/<id>` 契约；桌面 preflight/evidence 读取路径统一补齐 `tasks`，production 默认仍指向真实 `.lime/benchmark/sources/deep-swe`。修复后 Desktop contract、Desktop benchmark、DeepSWE adapter 与 benchmark 定向测试 `46/46` 通过，受影响 ESLint 与 `git diff --check` 通过；仍待提交后由干净 CI 复验 Frontend Full。
- Quality run `33574821717` 已证明 Desktop Smoke cache 修复通过原失败批次，但 Frontend Full 在第 `37/119` 批发现下一处干净工作区依赖：`packages/app-server-client/tests/apps.test.mjs` 保留对发布产物 `../dist/index.js` 的契约验证，而根测试入口未先构建该包。根 `pretest` 现统一执行 App Server client build，`test:frontend:all`、`test:resume`、`test:related` 与 `test:changed` 均委托唯一 `npm test` 入口；不把发布产物测试降级为源码直连。package 测试 `131/131`、根入口定向批次和参数透传已通过，仍待提交后由干净 CI 完成全部 Frontend Full。
- Quality run `33575987530` 已跨过 App Server client 发布产物批次并运行至第 `91/119` 批，随后发现 `automationDraft.test.ts` 把宿主默认时区硬编码为 `Asia/Shanghai`；产品实现按 `Intl.DateTimeFormat().resolvedOptions().timeZone` 读取用户宿主时区，Linux CI 因此正确返回 `UTC`。测试现按宿主 IANA 时区断言，显式 automation profile 的 `Asia/Shanghai` 断言保持不变；`TZ=UTC` 精确回归 `4/4`、受影响 ESLint 与 `git diff --check` 通过，仍待干净 CI 完成剩余批次。
- Quality run `33577323728` 已跨过第 `91/119` 批，并在第 `112/119` 批发现同类断言：Workspace Service Skill 创建计划任务的测试把默认时区硬编码为 `Asia/Shanghai`，而产品仍正确使用宿主 IANA 时区。该断言现与产品契约一致，显式时区行为未改；`TZ=UTC npm test -- --resume` 已从第 `112/119` 批续跑并完成至 `119/119`，仍待提交后由干净 Linux runner 完成最终复验。

已通过：

```text
npm run verify:app-version
npm run typecheck
npm run test:contracts
npm run smoke:agent-runtime-current-fixture
npm run verify:gui-smoke
npm test -- --resume                           # 119/119 批次
TZ=UTC npm test -- --resume                    # 第 112-119 批通过
npx vitest run <受影响前端/Electron/脚本精确入口>  # 36/36
npx eslint <受影响前端路径> --max-warnings 0
git diff --check
```

- `test:contracts`：App Server client 299 checks，modality、命令、脚本、release workflow 与文档边界均通过。
- 完整 Vitest：首次运行前 56 批通过；修复 `electron/systemUtilityHost.ts` 已明确使用但未登记的 macOS security-scoped bookmark 存储根基线后，从第 57 批续跑完成全部 119/119 批次；定向治理测试 15/15 通过，未放宽扫描规则。
- Agent current fixture：完整覆盖首页长短输入、停止后继续、审批、Plan、Skills/MCP、媒体、代码工作台和文章工作台；`liveProviderUsed=false`。
- 首页 Gate B：`claw-chat-current-fixture-home-hotpath-prewarm-fix-summary.json` 与聚合回归均为 `ok=true`；预热 session 被正式发送复用，输入后、提交切换和完成后三个稳定窗口均通过。
- 最终 GUI smoke：`.lime/qc/project-gates/standalone-shell-01-20260901112400-63590/shell-01-electron-smoke/summary.json` 为 `result=pass`，App Server `protocol=appserver.v0 version=1.138.0`；此前完整 GUI shell Gate B-F `.lime/qc/project-gates/standalone-shell-01-20260901104344-8413/shell-01-electron-smoke/summary.json` 为 21/21，真实 Electron/preload/IPC/App Server JSON-RPC 命中，错误、legacy command 与 mock fallback 均为 0。
- `npm run test:related -- ...` 的收集器曾把 `electron/` 目录作为文件读取并报 `EISDIR`；直接 Vitest 精确入口覆盖相同受影响测试并全部通过，不属于测试断言失败。
- 额外执行的完整 Rust workspace 测试未完成：裸 `npm run test:rust` 下载 denoland `rusty_v8 v150.4.0` Apple Silicon 预编译包时遇到上游 URL 404；改用仓库 `resolveRustyV8CargoEnv` 后完成核心 crate 编译，但在链接 `app-server` 测试产物时因本机磁盘仅余约 1.4 GiB 报 `No space left on device`。该项不是默认发布硬门禁，未将环境失败表述为测试通过。
- Windows packaged Squirrel、安装后 Agent/CodeMode Gate B 和发布资产尚未执行；它们必须在 commit/tag 推送后由 `windows-2022` release workflow 生成平台证据。

## 收尾记录

- `current`：Composer Controller、prompt history、App Server typed client、Desktop capability/readiness、macOS native host、desktop resource manifest、Windows packaged Gate B、canonical model/reasoning/queue projection。
- `compat`：无新增 compat wrapper。
- `deprecated`：无新增 deprecated owner。
- `dead / deleted`：外部 Codex manifest 和 4 个压缩 bundle 不进入 release；不恢复旧 runtime、生产 mock fallback 或第二持久化 owner。
- 当前完成度：`90%`；首轮 commit/tag/push 与 GitHub Release 已完成，follow-up 修复和质量预算已落地，仍待 follow-up commit、远端同名 tag 重建确认、Release workflow 资产和 Windows packaged Gate B。
