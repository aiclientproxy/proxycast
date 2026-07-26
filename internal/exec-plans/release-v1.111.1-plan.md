# Lime v1.111.1 发布执行计划

状态：preparing-release-candidate
日期：2026-07-25
目标版本：`1.111.1`
目标 tag：`v1.111.1`

## 主目标

将 `v1.111.0` 已发布后的修复提交 `114455c39` 作为补丁版 release candidate，完成版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送与远端复核。已有 `v1.111.0` tag 不重打、不覆盖。

## 当前阶段与下一刀

- 当前阶段：已确认 `v1.111.1` 尚无本地或远端 tag，正在更新 release metadata。
- 下一刀：完成版本一致性与双语 notes 后执行稳定发布门禁。

## Release Candidate

- 基线：`v1.111.0`；当前 `main` 与 `origin/main` 同指 `114455c39`。
- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、双语 release notes、本计划、旧 v1.111.0 计划状态和执行计划索引。
- `candidate changes`：`v1.111.0..114455c39` 的修复提交，包括 release workflow、Docs workflow、App Server runtime owner 拆分、current Agent Runtime 边界守卫和验证修复。
- `excluded changes`：`scripts/agent-runtime/claw-chat-current-fixture-approval-read-model.mjs` 为并发进程持有的未提交改动；`tsconfig.node.tsbuildinfo` 为 typecheck 自动生成物。两者不进入本 release candidate，也不删除或覆盖。

## 写集与退出条件

- 只写版本事实源、双语 release notes、本计划、v1.111.0 计划收尾状态和执行计划索引；并发 fixture 与生成 build info 只读避让。
- 所有版本事实源同步为 `1.111.1`，release notes 只保留当前版本单页。
- `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、`npm run governance:electron-release-workflow`、`npm run governance:scripts` 和 `git diff --check` 通过。
- Agent runtime / App Server 风险追加 `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server`、`npm run smoke:agent-runtime-current-fixture`、`npm run verify:gui-smoke`。
- `npm run verify:local` 若再次被既有 `index.projectRestore.test.tsx` 异步测试漂移阻断，保留完整失败证据，不将其写成通过；候选相关门禁必须独立通过。
- 获得 v1.111.1 Git 写操作确认后，显式暂存候选，执行 `git commit -m "Release v1.111.1"`、`git tag v1.111.1`、`git push origin main`、`git push origin v1.111.1`，并复核本地/远端 commit 与 tag。

## 验证记录

待补充。所有命令、证据等级、环境限制和远端 workflow 状态在每个阶段完成后回写本文件。

## 分类

- `current`：版本事实源、Forge 读取根版本、current App Server/Agent Runtime 修复和真实 Electron Gate B 证据。
- `compat`：无新增 compat owner；现有兼容边界不扩展。
- `deprecated`：旧入口只保留迁出约束，不因补丁版恢复。
- `dead / deleted / forbidden-to-restore`：已删除 queue controller 和 Renderer goal state 旧 metadata 只出现在负向边界守卫或历史 evidence。
