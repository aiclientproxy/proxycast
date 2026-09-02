# Agent Workbench 并行工作流

> 状态：active
> 目的：允许多个进程同时推进 `lime-agent-workbench`，减少互相覆盖。

## 并行前置规则

每个进程开始前必须先执行：

```bash
git status --short
```

然后声明本轮只认领一个窄写集。不要多个进程同时编辑同一文件；导航文件和版本文件由最后一个集成进程统一修改。

当前实现阶段的任务认领以 [task-board.md](task-board.md) 为准；本文件保留文档站阶段的并行拆分和通用协作规则。

## Workstream A：Subagents 标准

| 项 | 内容 |
| --- | --- |
| 主目标 | 补齐 `/subagents` 顶层标准页。 |
| 推荐写集 | `docs/subagents.md`、必要时 `docs/concepts/agents.md`。 |
| 暂不触碰 | `docs/.vitepress/config.mts`，除非当前进程负责集成导航。 |
| 依赖 | Runtime 事件、TaskSnapshot、ExecutionGraph、Subagents 概念。 |
| 验收 | 页面明确 runtime facts、UI surfaces、control plane、降级状态和禁止方向。 |

建议章节：

1. Subagents 在 Lime 中是什么。
2. Runtime 事实与 scope ids。
3. ExecutionGraph / Subagents 投影。
4. Handoff / review / worker notification。
5. 与 Plugin / Content Studio 的采用边界。
6. Conformance fixture。
7. current / compat / deprecated / dead。

## Workstream B：SDK 包边界

| 项 | 内容 |
| --- | --- |
| 主目标 | 把 TypeScript SDK 文档从包名规划补成实现边界。 |
| 推荐写集 | `docs/sdk/typescript/**`。 |
| 暂不触碰 | `packages/**` 生产代码，除非另行认领实现任务。 |
| 依赖 | AI SDK `UIMessage.parts` 参考、AG-UI event stream 参考、Lime current packages。 |
| 验收 | 四包职责、依赖方向、API 草案、fixture replay、版本策略完整。 |

建议新增或补强页面：

- `docs/sdk/typescript/react-surfaces.md`
- `docs/sdk/typescript/conformance.md`
- `docs/sdk/typescript/package-boundaries.md`

## Workstream C：Runtime 配合

| 项 | 内容 |
| --- | --- |
| 主目标 | 把 App Server / RuntimeCore / ExecutionBackend 与 AgentUI 的配合写成实现合同。 |
| 推荐写集 | `docs/contracts/app-server-host.md`、`docs/contracts/runtime-read-model.md`、`docs/quickstart/runtime-provider.md`。 |
| 暂不触碰 | Lime 主仓 `src/lib/dev-bridge/**`、`lime-rs/**`。 |
| 依赖 | `internal/roadmap/agentruntime/README.md`、`internal/roadmap/plugin/v3/README.md`。 |
| 验收 | Runtime provider 不依赖 React；UI 不拥有 facts；read model / event stream repair / action response 有明确 owner。 |

## Workstream D：Content Studio / Product App 接入

| 项 | 内容 |
| --- | --- |
| 主目标 | 把 Workbench 标准落到 Content Studio 和未来 Plugins 的接入路径。 |
| 推荐写集 | `docs/tutorials/content-studio.md`、`docs/profiles/content-studio.md`、`docs/quickstart/product-app.md`。 |
| 暂不触碰 | `content-studio` 生产代码。 |
| 依赖 | Host Provider Runtime PRD、Plugin runtime roadmap。 |
| 验收 | 产品应用只提交业务 context 和 callbacks，不传 Provider key、不读 App Server DB、不自建 process component。 |

## Workstream E：导航 / 构建 / 发布集成

| 项 | 内容 |
| --- | --- |
| 主目标 | 统一接入 nav/sidebar、构建验证和发布准备。 |
| 推荐写集 | `docs/.vitepress/config.mts`、`README.md`、`docs/development/updates.md`、`package.json`、`package-lock.json`。 |
| 前置 | A-D 至少一个内容页已完成。 |
| 验收 | `npm run docs:build` 通过；版本和更新记录一致。 |

注意：修改版本号、commit、tag、push 属于高风险操作，必须由用户明确要求后执行。

## 推荐并行顺序

```text
A Subagents 标准
B SDK 包边界
C Runtime 配合
D Product App 接入
  -> E 导航 / 构建 / 发布集成
```

如果只开两个进程：

| 进程 | 认领 |
| --- | --- |
| 进程 1 | A + C |
| 进程 2 | B + D |
| 集成进程 | E |

## 冲突处理

| 情况 | 处理 |
| --- | --- |
| 发现目标文件已有未理解改动 | 停止写入该文件，切只读审阅。 |
| 两个进程都需要改导航 | 先不接导航，完成内容页后交给 E。 |
| 发现文档标准和 Lime current 实现冲突 | 先写 conflict note 到对应页面，不直接改生产代码。 |
| 发现需要生产实现才能闭环 | 回写 `iteration-plan.md` 的 P1/P2，不在文档进程里夹写代码。 |

## 每个进程收尾格式

每个进程完成后，在 `iteration-plan.md` 追加一行进度日志：

```markdown
| 2026-06-10 | 完成 Workstream A：Subagents 标准页。 | `docs/subagents.md`、`npm run docs:build`。 |
```

如果没有跑构建，要明确写原因和下一步。
