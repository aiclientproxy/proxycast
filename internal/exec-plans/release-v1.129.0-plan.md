# Lime v1.129.0 发布执行计划

状态：release-candidate-gates-partial
日期：2026-08-16
目标版本：`1.129.0`
目标 tag：`v1.129.0`

## 主目标

发布当前工作树中的多模型路由与多模态能力候选，覆盖候选模型集/OEM 路由策略、语音合成/转写/嵌入 provider lowering、媒体任务 worker、App Server 协议与 runtime 事实投影、证据导出旧面清理、GUI/Skills 面收敛及脚本、治理和多语言回归；完成版本事实源、双语单页 release notes、质量门禁、release commit、tag、main/tag 推送与远端复核。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划及执行计划导航。
- `candidate changes`：当前工作树全部已跟踪和未跟踪改动，包括 RuntimeCore/App Server/model-provider、协议 schema/客户端、媒体与嵌入、GUI、Skills、脚本、测试、架构、治理和多语言文件。
- `excluded changes`：无。用户请求为完整发布，当前工作树改动全部纳入本候选；不覆盖或拆分并发写集。

## 退出条件

- 四个版本事实源与双语单页 release notes 统一为 `1.129.0`，目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、受影响 Rust/前端定向测试、`npm run smoke:agent-runtime-current-fixture` 与 `npm run verify:gui-smoke` 按风险执行；无法执行的门禁必须记录原因。
- staged 内容与上述候选范围一致，完成危险操作确认后创建 `Release v1.129.0` commit、`v1.129.0` tag，推送 `main` 和 tag，并复核本地/远端状态。
- 计划记录当前/兼容/废弃/已删除分类、完成度和剩余证据缺口；不将未执行的平台/打包证据误报为已完成。

## 当前验证记录

- 版本更新前：根应用、CLI npm 包与 Rust workspace 为 `1.128.0`；目标 tag 不存在。
- 版本事实源、release notes 与本计划已更新；当前候选为 259 个已跟踪文件差异及 10 个未跟踪文件。

## 门禁结果

- `npm run verify:app-version`：通过，根应用、CLI npm 包、Rust workspace 与 Cargo.lock 均为 `1.129.0`。
- `npm run typecheck`：通过。
- `npm run test:contracts`：通过；协议生成 983 类型，App Server client 299 checks 通过。
- `npm run governance:legacy-report`：通过；扫描 2115 个源码文件，分类漂移 0、边界违规 0、零引用候选 0。
- `npm run verify:gui-smoke`：通过；真实 Electron/App Server `1.129.0`，Gate B evidence `standalone-shell-01-20260816014540-32363`。
- `npm run test:rust:related -- <paths...>`：编译通过，1664 passed、9 failed；失败集中在 external backend 事件顺序、Agent terminal recovery、queue restore、mock/unavailable backend 与 model route resolver，未作为已通过门禁记录。
- `npm run smoke:agent-runtime-current-fixture`：前置历史/流式/审批/Plan/图片/Coding 场景通过；Skills Runtime workspace 手动启用阶段失败，断言 `readModelSkillGateObserved`，摘要见 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-skills-runtime-regression-summary.json`。

本地未执行 Windows/macOS packaged parity、签名、公证、正式 release asset 和 CI 门禁；这些不在本地候选中宣称完成。发布提交将保留上述失败证据和限制，不掩盖为全绿。

## 风险与限制

- 当前工作树包含大量跨层删除与重构，Rust related、contracts、typecheck 和 GUI smoke 需以实际结果为准。
- 本机无法替代 Windows/macOS 全平台 packaged parity、签名、公证和 CI 发布资产证据；未执行或失败项不得宣称为全绿。

## 架构确认

架构影响：重大。候选已更新 `internal/aiprompts/architecture.md`，涉及 RuntimeCore 模型候选路由、provider 多模态 lowering、App Server media worker 与 GUI runtime facts owner。责任开发者：root，确认日期：2026-08-16。
