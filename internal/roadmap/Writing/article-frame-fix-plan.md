# 修复：聊天产物卡收口为"完整文章产物"

> 状态：`legacy historical reference`（2026-09-02）；本文记录的修复已被 Writing v2 与 current Article artifact contract 承接，不再作为独立实施计划。

更新时间：2026-06-30
状态：历史修复已收口
主线：Writing 闭环 P4「通用 ArtifactFrame 与右侧 Article Editor」回归修复

右侧承载标准统一见 `../rightsurface/README.md`；本计划只聚焦文章产物框与 Article Editor 子面，不重复 dock / tab 规则。

## 1. 用户期望（事实源）

产物卡 = **一篇完整文章（最终产物本身）**：

- 不是过程内容，不是步骤汇总；
- 生成过程中在产物卡内**流式输出**正文；
- 完成后可**点击展开**，通过**右侧栏打开 Article Editor 编辑**。

## 2. 已修复现象

`@写文章` 发送后不再出现独立过程态承载卡；过程态留在对话流里，最终文章直接进入 `articleArtifacts` 文章框。

## 3. 根因定位

| 现象 | 代码位置 | 说明 |
| --- | --- | --- |
| 对话流过程态误入产物框 | `src/components/agent/chat/components/ArticleArtifactFrame.tsx:32-79`（`processSteps`）、`:164-183`（渲染）、`:82-103`（`facts`）、`:217-232`（facts chips 渲染） | 已删除，过程态不再作为独立卡片的一部分。 |
| "本轮执行已完成"兜底文案 | `src/components/agent/chat/hooks/agentStreamCompletionController.ts:16-21` + `agent.json:666` | 保持待查，若仍出现则单独修。 |
| 一输入即完成态 | App Server runtime streaming projection（初始快照 `status:"streaming"`）+ 前端框头完成态汇总 | 已转成文章框流式状态，保留给最终产物展示。 |

点击展开右侧编辑器接线（`onArtifactClick`）已贯穿 `MessageList`，**无需改动**。

## 4. 方案

### Stage 1：产物卡收敛为"纯文章产物框"
**Goal**：`ArticleArtifactFrame` 只渲染 [标题栏 + 流式/完成状态 + 完整正文容器 + 打开编辑器入口]。
**已完成**：
- 删除独立过程态承载组件与投影分支。
- 文章框保留标题、summary、流式 loading 态、正文 `MarkdownRenderer`、`onArtifactClick` 打开右侧编辑器。
- `articleArtifactProjection.ts` 只保留文章框模型，不再派生对话流过程态投影。
**Success**：产物卡内只见文章正文与标题。
**Tests**：`MessageArtifactCards.test.tsx` 已补充流式文章回归。

### Stage 2：消除"一输入即完成"的空回复兜底观感
**Goal**：写文章 runtime turn 流式期间仍显示文章框的 streaming 态，不回落成普通聊天长文。
**后续**：若还出现"本轮执行已完成"兜底，再单独定位 completion controller。

### Stage 3：文档与回归
**Goal**：同步路线图、补 GUI 冒烟。
**已做**：
- 路线图、产品需求、架构、workflow、sequence、prototype 已改成“过程态留在对话流”。
- 五语言文案不再引用旧汇总态 key。
**Success**：路线图与实现一致，GUI 冒烟待跑。

## 5. 非目标

- 不改右侧 Article Editor 布局与编辑/持久化/历史恢复链路。
- 不改 `onArtifactClick` → 右侧栏打开编辑器的既有接线。
- 不改 workspace_patch / articleDraft 后端数据结构（仅调整前端卡片消费与兜底文案）。
