---
name: lime-release-workflow
description: 准备并执行 Lime 发版；覆盖版本号、发布说明、产物清理、验证门禁、commit、tag 和推送。
---

# Lime 发版流程

## 核心原则

- 先读 `AGENTS.md`、`internal/aiprompts/quality-workflow.md` 和当前 `git status --short`，再修改版本或发版文件。
- 发版属于高风险流程。删除文件、`git commit`、`git tag`、`git push`、覆盖已有 tag、终止用户进程前，必须按仓库危险操作格式向用户确认。
- 如果工作区已有未提交改动，先把 release candidate 范围判清楚；版本号和 release notes 只是发布 metadata，不能默认代表本次发布全部内容。
- 只有用户明确要求“只更新版本 / 只写 release note / 不提交其它改动”时，才允许只修改和提交发版事实源；否则发版提交必须纳入本次 release candidate 的全部产品 / 文档 / 测试改动，或先停止并让用户确认排除清单。
- 版本号事实源以 `package.json`、`forge.config.mjs`、App Server manifest 与 `lime-rs/Cargo.toml` 的 workspace version 为准，完成后必须跑 `npm run verify:app-version`。
- Electron 发布 / 签名 / 公证 / updater metadata 的 current 打包事实源是 `forge.config.mjs`、`electron-forge package`、`electron-forge make` 与 Forge 官方 maker；旧 builder 配置 / CLI、自定义 Windows installer maker 与旧 YAML / blockmap updater metadata 按 `dead` 处理，不得写回 release workflow、docs、quality guard 或 i18n evidence。运行时更新以 `electron/updateHost.ts` + Electron 内置 `autoUpdater` 为 current；Windows installer 必须走 Forge Squirrel。
- 发版前默认必须跑通 `npm run typecheck`。这是发布硬门禁，必须在收尾和任何 commit / tag / push 确认前明确执行并记录结果；不能用 `npm run typecheck:electron`、`npm run lint`、局部单测、Rust 测试或 `verify:app-version` 替代。若用户明确要求“马上递交”或“立即递交”，可以把 `typecheck` 降级为非阻断项，直接进入发布收口，但最终汇报必须明确注明 `typecheck` 未执行或未通过。
- 默认发版门禁不要再自动执行裸 `npm run lint`、裸 `npm test` 或全量 `cargo test --manifest-path "lime-rs/Cargo.toml"`。这些命令在当前仓库经常耗时长、续跑差或被大写集噪音阻断；只有用户明确要求“完整矩阵 / 全量门禁 / CI 同款验证”，或当前改动风险确实需要时才执行，并优先使用下方的定向 / 续跑入口。
- Lime 是 GUI 桌面产品；即使静态检查和单测通过，涉及发布也要尽量跑 `npm run verify:gui-smoke`，跑不了要说明环境限制。
- 用户要求“发版 / 发布 / release”时，默认目标不是只准备版本文件，而是完成一次端到端发布：整理 release candidate、更新版本与 release notes、跑门禁、创建 release commit、创建 tag、推送 main 和 tag。`git commit` / `git tag` / `git push` 仍必须按危险操作格式请求一次明确确认；拿到确认后必须继续执行到底，并做 tag / 远端状态复核，不能把“是否提交 / 是否打 tag / 是否推送”留给用户自己处理。

## 入口判断

1. 明确目标版本。
   - 用户给出 `vX.Y.Z` 或 `X.Y.Z` 时，规范化为 `X.Y.Z` 与 tag `vX.Y.Z`。
   - 用户未给版本时，读取 `node scripts/app-version.mjs`，默认建议 patch bump；不能擅自发版。
2. 检查现状。
   - `git status --short`
   - `git log --oneline --decorate --max-count=20`
   - `git tag --list "vX.Y.Z"`
   - `rg -n '旧版本|目标版本'` 覆盖 `package.json`、存在的 npm/pnpm lockfile、`packages/cli/package.json`、`forge.config.mjs`、App Server manifest、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES*.md`
3. 如果发现上一轮发版命令仍在跑，只停止自己启动且已被用户中断的进程；不杀掉明显属于用户的 dev server / test / Electron 进程。

## Release Candidate 范围确认

发版前必须先回答“这次发布到底包含哪些改动”。不要把 dirty worktree 当成噪音，也不要默认只提交版本文件。

1. 用 `git status --short`、`git diff --stat`、`git diff --name-only` 和 `git ls-files --others --exclude-standard` 盘点当前未提交 / 未跟踪文件。
2. 如果用户说“发版 vX.Y.Z”“递交发布”“发布推送”，且工作区有未提交产品改动，默认这些改动属于本次 release candidate。
3. 将改动分成：
   - `release metadata`：版本号、release notes、manifest、lockfile 版本行。
   - `candidate changes`：本次要发布的代码、文档、测试、schema、生成物、脚本。
   - `excluded changes`：明确不进入本次发布的本地实验、临时文件、个人环境改动。
4. 如果无法从上下文确定哪些文件应排除，必须停下来向用户确认；不能自行假设“只提交发版文件”。
5. 如果需要提交全部当前工作树，`git add` 应覆盖 tracked 和 untracked candidate files，例如 `git add -A` 或显式文件清单；提交前用 `git diff --cached --stat` 复核 staged 内容。
6. 如果本地已经创建或推送 tag 后才发现漏发 candidate changes，不要继续普通提交并假装同一版本完成；必须说明现状，并在“重打 tag”或“发补丁版”之间让用户选择。已推送 tag 的删除 / 重建 / force push 必须单独危险确认。

## 更新版本事实源

必须同步这些事实源；仓库不存在或不包含显式版本的文件不要强行创建或改写：

- `package.json`
- 存在的 npm/pnpm lockfile
- `packages/cli/package.json`
- `forge.config.mjs`（仅当存在显式版本；当前通常从 `package.json` 读取）
- App Server manifest / release manifest（仅当存在显式发布版本）
- `lime-rs/Cargo.toml`
- `lime-rs/Cargo.lock`

更新方式优先使用结构化解析；锁文件和 Cargo lock 的 workspace 版本可做机械替换，但替换前后要用 `rg` 复核旧版本残留。

改完立即执行：

```bash
npm run verify:app-version
```

## 更新 Release Notes

事实源是根目录：

- `RELEASE_NOTES.md`：中文 primary
- `RELEASE_NOTES.en.md`：英文 companion
- Release Notes 采用“当前版本单页”策略：每次发布都必须替换整份 `RELEASE_NOTES*.md`，只保留目标版本内容；不得把旧版本历史继续追加或保留在文件里。历史版本只从 git history / tag / GitHub Release 查。

生成内容时同时参考：

- `git log <last-tag>..HEAD --pretty=format:"%s (%h)" --no-merges`
- release candidate 的 `git diff --stat`、`git diff --name-only`、关键文件 diff、未跟踪文件清单
- 用户明确要求删除或替换的旧 artifact

整文件格式保持：

```markdown
## Lime vX.Y.Z

### 新功能

### 修复

### 优化与重构

### 测试与质量

### 文档

### 其他

**完整变更**: `vA.B.C` -> `vX.Y.Z`
```

英文 companion 使用对应英文标题，保留 “Simplified Chinese release notes are the primary version” 说明。不要把内部 commit message 逐条照搬成 release note；要合并成面向发布的可读条目。写完后用 `sed -n '1,120p' RELEASE_NOTES*.md` 或等价方式确认文件内没有旧版本标题。

## 删除旧内容

- “删除旧的”必须先确认对象：旧 release note 内容、旧 artifact 文件、旧 tag、旧 release asset 是不同风险级别。
- 替换 `RELEASE_NOTES*.md` 并移除旧版本 release note 历史是发版默认行为，不需要额外确认；删除文件、删除本地/远端 tag、移除 release asset 必须确认。
- 删除前用 `git status --short` 与 `git diff -- <path>` 确认目标不是用户未说明的新改动。

## 验证矩阵

默认发版验证按“稳定必要门禁优先，重型矩阵按需扩展”执行。版本改动后必须先跑：

```bash
npm run verify:app-version
npm run typecheck
```

然后按改动风险补充：

```bash
npm run test:contracts      # 协议 / Bridge / command catalog / mock / governance 变更
npm run verify:gui-smoke    # GUI 壳、Workspace、主路径或正式发布尽量执行；跑不了要说明环境限制
```

不要把下面这些命令作为默认发版阻断项：

- `npm run lint`
- 裸 `npm test`
- 全量 `cargo test --manifest-path "lime-rs/Cargo.toml"`
- 全量 `cargo clippy --manifest-path "lime-rs/Cargo.toml"`

只有用户明确要求“完整矩阵 / 全量门禁 / CI 同款验证”，或当前改动直接需要它们时才执行。需要验证时优先使用更稳定的定向入口：

```bash
npx eslint "<changed-path>" --max-warnings 0
npm run test:related -- <files>
npm run test:changed -- <ref>
npm run test:resume
npm run test:rust:changed
npm run test:rust:related -- <paths...>
npm run test:rust:unit -- -p <crate> <filter>
cargo test --manifest-path "lime-rs/Cargo.toml" -p <crate> <filter>
```

前端全量 Vitest 通过 `scripts/run-vitest-smart.mjs` 分批执行并写入 `.lime/test/vitest-smart-last-run.json`。如果 `npm test` 已经失败或被用户中断，继续发版验证时默认先执行 `npm run test:resume`，也可以用 `npm test -- --from-batch <N>`、`npm test -- --only-batch <N>` 精确补批次。只有测试收集规则、批次大小、依赖图或目标分支已经改变，才从头执行裸 `npm test`，并在汇报中说明原因。

如果只做 release note 或文档更新，可按 `internal/aiprompts/quality-workflow.md` 降级；正式发版默认仍保留 `npm run typecheck`，但用户明确要求“马上递交”或“立即递交”时可跳过该项，最终汇报必须说明门禁降级理由。

## Commit / Tag / Push

完整发版默认必须包含 commit / tag / push。不要在验证通过后只汇报“可以提交了”就结束；应主动给出危险操作确认请求。只有用户明确说“只准备、不提交 / 不打 tag / 不推送”时，才允许停在准备态，并在最终汇报中标明不是完整发布。

拿到 git 写操作确认后，必须把 `git add`、`git commit`、`git tag`、`git push origin main`、`git push origin <tag>` 与远端 tag 复核连续执行到底；不能在 commit 后停下让用户自己打 tag，也不能在 tag 后停下让用户自己 push。若中间失败，先修复或明确阻塞，不要把剩余步骤转交给用户。

执行任何 git 写操作前，先汇总：

- 目标版本与 tag
- 将纳入提交的文件列表，必须区分 `release metadata` 与 `candidate changes`
- `git diff --cached --stat` 的 staged 摘要
- 已通过 / 未通过 / 未执行的验证
- `npm run typecheck` 默认必须已通过；若用户明确要求“马上递交”或“立即递交”，可在未执行或未通过时继续请求 commit / tag / push 确认，但必须在汇报中明确说明这一降级
- 是否存在未提交或未跟踪但未纳入发布的改动；如果存在，必须列出排除原因或等待用户确认

然后按危险操作格式请求确认。确认后再执行：

```bash
git add <release-candidate-files>
git commit -m "Release vX.Y.Z"
git tag vX.Y.Z
git push origin main
git push origin vX.Y.Z
```

确认后不要只执行其中一部分。不要依赖 `git push origin main --follow-tags` 作为唯一 tag 推送方式：轻量 tag 不会被 `--follow-tags` 推送，容易造成 main 已发布但远端 release tag 缺失。若 main 已推送但 `git ls-remote --tags origin "refs/tags/vX.Y.Z"` 为空，必须继续执行 `git push origin vX.Y.Z` 并再次复核。若 `git add`、`git commit`、`git tag` 或任一 `git push` 失败，必须修复或明确说明阻塞点；成功后立即复核：

```bash
git status --short
git log --oneline --decorate --max-count=3
git tag --list "vX.Y.Z"
git ls-remote --tags origin "refs/tags/vX.Y.Z"
```

如果 tag 已存在或已推送，不要覆盖；先说明本地与远端 tag 状态，并单独确认删除或重建策略。重打已推送 tag 时，必须明确说明会改写远端发布引用，并优先建议补丁版，除非用户明确要求保留同一版本号。

## 收尾输出

最终汇报必须包含：

- 本轮完成度百分比
- 版本号、tag、release note 文件
- 实际修改的发版事实源
- 验证命令结果与任何环境限制
- 是否已经 commit/tag/push
- 剩余缺口和下一刀
