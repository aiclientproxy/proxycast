# Codex TUI Snapshot Inventory

状态：已建立，迁移进行中  
上游：`/Users/coso/Documents/dev/rust/codex/codex-rs/tui`  
基线 commit：`cac96cd7b1756ab42e8925d938817a2ac10ebb6e`  
snapshot 数量：`802`  
排序后相对路径 SHA-256：`da5d7b14f30cccefa3132e2c33b4aaf1e0460ce9a342fc571f857ce15a21ef03`

逐项路径、内容 SHA-256、分类与规则 owner 记录在
[`tui-codex-snapshot-inventory.json`](./tui-codex-snapshot-inventory.json)。

## 分类口径

规则按以下顺序执行：

1. Codex account、onboarding、migration、update、marketplace 等产品专属能力归 `dead`。
2. 不依赖 Codex runtime/state 的纯终端算法归 `direct`。
3. 跨 App Server、runtime 或持久化状态 owner 的能力归 `contract`。
4. 不属于当前 Codex Desktop parity 优先级的终端能力归 `defer`。
5. 其余交互场景归 `merge`，只迁移行为并接入 Lime current owner。

`hook_blocked_failed_feedback_history` 表示 Hook 执行反馈，不是 Codex 产品反馈，明确归
`merge`。

| 分类 | 数量 | 当前处理方式 |
| --- | ---: | --- |
| `direct` | 48 | 在 `tui` 内适配纯渲染/终端算法 |
| `merge` | 579 | 合入现有 App、composer、entry、view 与 picker |
| `contract` | 80 | 先对齐 App Server canonical contract，再迁 UI |
| `defer` | 25 | 保留账本，不进入当前 P3 写集 |
| `dead` | 70 | 禁止迁入 current TUI |

## 模块裁决

- `direct`：`diff_render`、`markdown_render`、`insert_history`、`render`、`terminal_hyperlinks`
- `contract`：`app`、`app_backtrack`、`cwd_prompt`、`multi_agents`、`resume_picker`、`unarchive_prompt`
- `defer`：`custom_terminal`、`git_action_directives`、`inline_visualization`、`keymap_setup`、`startup_hooks_review`
- `merge`：除逐项 `dead` 命中外的其余模块
- `dead`：只存在于 Codex 产品面的 account、onboarding、migration、update、marketplace、rate-limit 等场景

## 刷新与守卫

显式指定 Codex checkout 后刷新：

```bash
CODEX_TUI_REFERENCE="/path/to/codex/codex-rs/tui" npm run inventory:tui-codex
```

CI 只校验已提交账本，不依赖仓库外 checkout：

```bash
npx vitest run scripts/app-server/tui-snapshot-inventory.test.mjs
```

刷新后必须人工复核新增路径；不能仅因文件名匹配就把跨 runtime/state 行为标记为
`direct`。
