# Lime Soul 个性化 PRD

> 状态：current PRD
> 更新时间：2026-07-06
> 关联路线图：[../memory/README.md](../memory/README.md)
> 产品口径：普通用户看到 `AI 个性 / 声线`；底层工程继续使用 `Personality Layer / memory profile / Generation Brief`。

## 1. 背景

OpenClaw 和 Hermes 都已经证明，长期稳定的人格 / 声线会显著改变 AI 产品的体感：

1. OpenClaw 用 workspace `SOUL.md` 让工程 Agent 拥有明确 voice，同时和 `AGENTS.md`、`MEMORY.md` 分工。
2. Hermes 用全局 `SOUL.md` 作为 primary identity，让每次会话都有稳定人格；同时用 `/personality` 做临时模式切换。
3. Lime 当前 memory 路线图已经定义了 `Personality Layer`，但还没有把它产品化成普通用户可管理的入口。
4. Lime 已有 `memory.profile`、用户资料、灵感库、知识库和 runtime prompt composition，不能再新增一个平行 `SOUL.md` 系统。

本 PRD 的核心判断：

**Lime 需要全局 Soul 能力，但它必须是 Memory 个性化主链的一个产品化投影，而不是新的文件协议或新的记忆系统。**

## 2. 用户与场景

### 2.1 普通用户

用户特征：

1. 希望 Lime 的回答更像自己想要的助手。
2. 不想维护 Markdown 文件或理解 prompt 工程。
3. 希望能用几个清晰选项控制语气、直接程度、解释深度。

核心场景：

1. 设置 Lime 默认更直接、更简洁或更有耐心。
2. 让 Lime 少说客套话，多给实际建议。
3. 关闭或重置个性化语气。
4. 确认正式内容没有被助手人格污染。
5. 切换贱兮兮执行官、温柔陪伴型助理、拽酷行动派、冷静专业型搭档等 Style Profile，并在工具调用前后和正文细节里感受到差异。

### 2.2 进阶创作者 / 品牌用户

用户特征：

1. 关心正式内容是否符合个人 IP 或品牌声线。
2. 能理解“交互人格”和“创作声线”的差异。
3. 需要导入已有 `SOUL.md` 或导出备份。

核心场景：

1. 从 OpenClaw / Hermes 导入 `SOUL.md`。
2. 把全局语气整理成正式创作声线。
3. 在一次任务里临时切换成“老师 / 严格评审 / 研究伙伴”等模式。
4. 查看本次正式创作为什么使用某个声线。

### 2.3 开发者 / 内测诊断用户

用户特征：

1. 需要排查人格 prompt 是否重复注入。
2. 需要验证 artifact 是否被 Product Personality 污染。
3. 需要对照 Memory、Soul、Generation Brief 的边界。

核心场景：

1. 查看某一轮是否注入 Global Soul。
2. 查看 `Generation Brief` 中是否包含 creator / brand voice。
3. 验证 `SOUL.md` 导入只是写入 current 配置。
4. 验证禁用 Soul 后下一轮不再注入。
5. 验证 Style Pack Registry 中的 profile 没有退化成 UI / i18n 硬编码句库。

## 3. 产品目标

### 3.1 P0 目标

1. 固定 Soul 与 Memory 的产品边界。
2. 提供全局 AI 个性 / 声线设置的目标形态。
3. 默认只影响聊天交互语气、解释方式和追问方式。
4. 明确不新增 `soul_*` 数据库主链。
5. 明确 `SOUL.md` 只做导入 / 导出格式。

### 3.2 P1 目标

1. 用户能编辑、保存、重置全局 Soul。
2. 用户能关闭全局 Soul。
3. 普通聊天能吸收 Global Soul。
4. 正式创作默认不受 Product Soul 影响。
5. 诊断能说明当前 turn 是否使用了 Soul section。
6. Style Profile 能覆盖工具调用前、中、后、正文转折、失败恢复和结尾建议，不只改变欢迎语。

### 3.2.1 P1.5 目标：可切换 Style Profile

1. 四个首发风格作为 built-in Style Pack seed 注册到 `Style Pack Registry`。
2. 每个 Style Pack 提供 voice primitives、surface contracts、anti-repetition rules、few-shot anchors 和 risk fallback。
3. i18n 只提供 neutral UI copy；不得新增 `agentChat.soulInteraction.<tone>.*` profile 句库。
4. 工具事实、搜索来源、图片结果、任务状态继续以 Agent Runtime / App Server read model 为事实源。
5. 高风险、权限、删除、生产 API、医疗、法律、财务等场景强制降级到冷静专业口吻。

### 3.3 P2 目标

1. 支持导入 / 导出 `SOUL.md`。
2. 导入前展示解析预览和风险提示。
3. 导入后写入 Lime current 配置，而不是运行时读取原文件。
4. 导出时生成兼容 OpenClaw / Hermes 心智的 Markdown。

### 3.4 P3 目标

1. 支持 Creator / Brand Voice 进入 `Generation Brief`。
2. 用户能选择某次正式创作是否使用个人 / 品牌声线。
3. 每个进入 `Generation Brief` 的 voice 字段都有用户显式配置或 evidence。
4. 支持临时 personality overlay，但默认不写入长期 Soul。

## 4. 非目标

本 PRD 不做：

1. 不新增 `soul_*` 数据库、DAO、Repository 或 Tauri 命令族作为主链。
2. 不默认读取项目根 `SOUL.md`。
3. 不默认读取 `~/.lime/SOUL.md` 并覆盖设置页。
4. 不复制 OpenClaw 的 workspace bootstrap 文件体验作为普通用户主入口。
5. 不复制 Hermes 的全局文件事实源。
6. 不让 Product Personality 默认进入文章、脚本、海报文案、PPT 等正式 artifact。
7. 不让 Companion Soul 影响正式内容。
8. 不把所有聊天历史自动抽取成声线。
9. 不在多个组件里各自拼装 Soul prompt。
10. 不把四种 Style Profile 做成四套本地 i18n 句库。
11. 不让工具卡片、timeline、runtime status 根据 profile id 拼最终展示句子。
12. 不把 few-shot anchor 当固定开场白或固定承接句复读。

## 5. 前台信息架构

### 5.1 设置页入口

普通用户看到：

```text
AI 个性 / 声线
  - 当前风格
  - 回答直接程度
  - 解释深度
  - 追问倾向
  - 避免事项
  - 预览
  - 保存 / 重置 / 关闭
```

进阶用户展开：

```text
高级
  - 导入 SOUL.md
  - 导出 SOUL.md
  - 本次创作声线使用策略
  - 临时模式
```

开发者 / 诊断入口：

```text
诊断
  - 当前 turn 是否注入 Global Soul
  - Generation Brief voice 字段
  - prompt section marker
  - evidence / 来源
```

### 5.2 用户可见对象

普通用户可见字段：

1. 名称。
2. 一句话描述。
3. 语气标签。
4. 回答风格。
5. 避免事项。
6. 影响范围：聊天交互 / 正式创作。
7. 当前状态：开启 / 关闭 / 仅本次。

普通用户不可见字段：

1. prompt section marker。
2. memory profile 字段名。
3. `runtime_turn` 阶段名。
4. provider prompt cache 细节。
5. raw `SOUL.md` 注入片段。

## 6. 功能需求

### 6.1 全局 Soul 设置

P1：

1. 用户可以配置全局交互人格。
2. 字段聚焦在语气、直接程度、解释方式、追问方式和避免事项。
3. 保存后下一轮普通聊天生效。
4. 关闭后下一轮普通聊天不再注入 Soul section。
5. 重置回 Lime 默认产品语气。

验收：

- 空配置不注入 Soul section。
- 重复保存不产生重复 prompt section。
- 当前用户指令优先于全局 Soul。

### 6.2 SOUL.md 导入 / 导出

P2：

1. 用户选择或粘贴 `SOUL.md` 内容。
2. Lime 解析为可预览的 Soul draft。
3. 用户确认后写入 current 配置。
4. 导入不保留文件路径依赖。
5. 导出从 current 配置生成 Markdown。

验收：

- 导入前必须预览。
- 空文件不会覆盖现有配置。
- 导入含明显项目规则时提示用户迁移到项目 / 知识库规则。
- 导出内容不包含密钥、路径或运行时诊断。

### 6.3 普通聊天注入

P1：

1. 在 runtime prompt composition 中追加 Global Soul section。
2. section 有稳定 marker，避免重复注入。
3. 注入仅影响交互语气、解释风格和追问方式。
4. 安全、系统、开发者、用户当前指令优先。

验收：

- 普通聊天能体现 Soul 风格。
- 同一工具生命周期里的工具前说明、工具中状态、工具后承接和最终正文都能体现正确 profile。
- 正文小标题、段落转折、风险提示和结尾建议不能退回默认助手口吻。
- 工具执行、安全确认、错误处理不被 Soul 弱化。
- 关闭 Soul 后同一类 turn 不再包含 marker。
- 本地 UI copy 只使用 neutral i18n key + descriptor metadata，不出现 profile 句库。

### 6.4 正式创作声线

P3：

1. 正式 artifact 默认不直接注入 Global Soul。
2. 如果用户选择使用个人 / 品牌声线，必须进入 `Generation Brief`。
3. `personality_boundary_guard` 决定哪些 voice 字段可进入本轮。
4. 结果页或诊断层能解释声线来源。

验收：

- Product Soul 不会默认污染正式内容。
- Creator / Brand Voice 使用必须可解释。
- Companion Soul 不进入 artifact。

### 6.5 临时 personality overlay

P3：

1. 用户可以在当前 session 临时切换模式。
2. overlay 不默认写入长期配置。
3. 切换后不清空历史，但需要给模型明确 pivot。
4. 清除 overlay 后回到 Global Soul。

验收：

- overlay 只影响当前 session 或当前任务。
- 用户确认沉淀前不改变长期 Soul。
- 旧历史可读，但后续回复风格能切换。

## 7. 优先级

| 优先级 | 能力                      | 原因                               |
| ------ | ------------------------- | ---------------------------------- |
| P0     | 文档和边界固定            | 防止和 Memory 分叉                 |
| P1     | 全局 Soul 设置 + 聊天注入 | 立即改善产品体感                   |
| P1.5   | Style Pack Registry       | 让风格不止欢迎语，且不靠硬编码句库 |
| P2     | SOUL.md 导入 / 导出       | 承接 OpenClaw / Hermes 迁移        |
| P3     | Generation Brief 声线     | 让正式创作更像用户                 |
| P4     | 临时 overlay / 品牌包     | 进阶能力，风险更高                 |

## 8. 风险

1. **和 Memory 重复**
   - 规避：Soul 只作为 Personality Layer 子路线，不新增事实源。
2. **正式内容被助手人格污染**
   - 规避：artifact 默认不注入 Product Soul，必须通过 Generation Brief。
3. **用户把项目规则写进 SOUL.md**
   - 规避：导入预览提示迁移到项目规则或知识库。
4. **prompt 变长影响成本**
   - 规避：P1 只注入短 section，长文本后续摘要。
5. **多入口配置不一致**
   - 规避：设置页、导入、runtime 都写同一 current 配置。
6. **风格变成千篇一律模板**
   - 规避：Style Pack 只提供规则、surface contract 和 few-shot anchor；UI / i18n 禁止 profile 句库。
7. **用户只在欢迎语感受到差异**
   - 规避：验收覆盖工具调用前后、正文段落、失败恢复和结尾建议。

## 9. 成功指标

1. 普通用户能在设置页理解并修改 Lime 的默认交流方式。
2. 开启 Soul 后聊天体感明显更贴近配置。
3. 关闭 Soul 后运行时不再注入对应 section。
4. SOUL.md 导入不造成新的文件依赖。
5. Memory、Soul、Generation Brief 的工程边界可测试、可解释。
6. 同一真实任务切换两种 Style Profile 后，工具 lifecycle 和正文细节有可见差异，但事实、来源、数字和错误原因保持一致。
