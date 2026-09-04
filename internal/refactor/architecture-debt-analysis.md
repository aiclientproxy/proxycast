# 架构债证据底座

> 状态：evidence（2026-06-11 实测采集，2026-06-17 复核标注治理进展）
> 采集方式：依赖图扫描 + import grep + 文件大纲抽查 + 本轮 `project_git` 新增方法的真实写集
> 用途：支撑 `v2/IMPLEMENTATION-PLAN.md` 的优先级排序；任何条目的优先级争议先回到本文件核对证据
> 复核约定：原始证据（2026-06-11 快照）保留不改；治理后状态用 **【2026-06-17 复核】** 行内标注，便于对照"病灶 → 已修"

所有路径相对仓库根。行数为采集日快照，复查时允许漂移。

---

## 轴 A · 协议链路靠人肉同步（最高杠杆）

**机制**：Rust 侧 `app-server-protocol` 已用 schemars 生成 JSON Schema（`schema_export.rs` 的 `generate_json_schema_bundle()`），但 TS 侧 `packages/app-server-client/src/protocol.ts`（约 3600 行）是**手写**的，无生成标记、无 codegen 脚本。协议四侧同步（AGENTS.md 硬规则 3）完全靠人肉对齐。

**实测写集**：本轮新增 `project_git` 4 个方法（`read_project_git_status` 等），git status 显示真实触点：

```
lime-rs/crates/app-server-protocol/src/protocol/v0.rs            # 模块挂载
lime-rs/crates/app-server-protocol/src/protocol/v0/project_git.rs # 新增类型
lime-rs/crates/app-server-protocol/src/protocol/v0/catalog.rs     # 目录注册
lime-rs/crates/app-server-protocol/src/protocol/v0/method_names.rs # 方法名常量
lime-rs/crates/app-server-protocol/src/schema_export/registry.rs  # schema 导出注册
lime-rs/crates/app-server/src/runtime.rs                          # RuntimeCore 方法（runtime.rs:4856-4917）
lime-rs/crates/app-server/src/processor.rs                        # handler 分发
lime-rs/crates/services/src/lib.rs + project_git_service.rs       # 业务实现
packages/app-server-client/src/protocol.ts + index.ts             # TS 手抄协议
src/lib/api/projectGit.ts                                         # 前端网关
```

**结论**：1 个能力 ≈ 10+ 文件触点，其中 protocol crate 4 处注册 + TS 2 处手抄是纯机械劳动。这是 runtime.rs / processor.rs / protocol.ts 三个最大文件持续膨胀的直接机制，也是协议漂移（Rust/TS 不一致）的常驻风险源。

> **【2026-06-17 复核】** TS 侧人肉同步已消除：`protocol.ts` 顶部转为 `// @generated types re-export`，类型来自 `generated/protocol-types.ts`（3641 行，自定义 JSON Schema→TS 转换器生成），生成/漂移检查脚本 `generate:protocol-types` / `check:protocol-types` 已就位。剩余仅 Rust 侧 4 处注册的宏收敛（R-10 二期，可选）。

---

## 轴 B · App Server 双中心 Facade

**机制**：所有 JSON-RPC 方法汇聚到两个单一 impl 块：

- `lime-rs/crates/app-server/src/processor.rs`（5041 行）：单一 `impl RequestProcessor`，238 个 `async fn handle_*`，职责是请求路由分发。
- `lime-rs/crates/app-server/src/runtime.rs`（8105 行）：单一 `impl RuntimeCore`，521 个 async fn，职责横跨会话、模型、工具调度、事件分发、artifact、git……

它们不是经典 God object（业务实现已大多下沉 `services` / `local_data_source`），而是**强制中心化的接线层**：任何新方法都必须在这两个 impl 块各加一段，文件只能单调变大。`runtime/tests.rs` 4428 行同理。

**佐证**：`local_data_source.rs` 周边已经长出子模块结构（`local_data_source/plugins/`、`automation/`、`knowledge/` 等），说明仓库已有"按 domain 拆模块"的成熟先例，唯独 processor/runtime 的方法注册没有跟上。

**结论**：拆 8000 行为 4 个 2000 行文件不解决问题；要改的是**注册模式**——按 domain 把 handler 和 RuntimeCore 方法下放到 `runtime/<domain>.rs` + 注册表/宏接线，让新增方法默认不触碰中心文件。

> **【2026-06-17 复核】** 已落地：`runtime.rs` **8105 → 588 行**（仅结构体 + 接线），实现下放到 `runtime/` 52 个 domain 子模块；`processor.rs` 单文件 → `processor/` 目录 24 个 domain 模块（`mod.rs` 2444 行做 dispatch）；`runtime/tests.rs` 4428 行已下放到 `runtime/tests/*.rs` 各 domain。新增方法的标准写集不再撑大中心文件。剩余：`processor/mod.rs` 仍偏大、agent `agent.rs`（R-21）未动。

---

## 轴 C · 前端分层倒置 + 状态无分层

### C-1 依赖方向违例（lib → components/features，实测 ≥6 处）

```typescript
// src/lib/workspace/workbenchCanvas.ts
export { CanvasFactory } from "@/components/workspace/canvas/CanvasFactory";
export { NotionEditor } from "@/components/workspace/document/editor/NotionEditor";
// src/lib/imageGeneration.ts
import { IMAGE_GEN_MODELS, type ImageGenModel } from "@/components/image-gen/types";
// src/lib/workspace/workbenchWorkflow.ts
export { useWorkflow } from "@/components/workspace/hooks/useWorkflow";
// src/lib/navigation/sidebarNav.ts
import { resolvePluginHostFlags } from "@/features/plugin/featureFlag";
// src/lib/api/agentApps.ts
import { buildInstalledAppPreview } from "@/features/plugin/install/installedAppPreview";
```

且 `features/plugin/ui/` 反向 import `components/agent/chat/` 的 UI 组件（如 `ThinkingBlock`）；反方向（components → features）为零。当前没有任何 ESLint / 结构测试约束 import 方向，违例只会增加。

### C-2 状态无分层（巨型文件的真实成因）

- `src/components/agent/chat/AgentChatWorkspace.tsx`（7029 行）：100+ import、52 个 useState、约 170 个 hook 调用、1000+ 行 JSX；职责横跨路由导航、数据同步、runtime 集成、媒体任务、canvas、artifact、skill 触发至少 8 个正交关注点。
- `src/components/agent/chat/workspace/useWorkspaceSendActions.ts`（5117 行）：**单个** hook 函数约 4000 行，内嵌 20+ 个 workbench 命令解析器、30+ 个 Parsed 类型。
- 状态管理三轨并存且无约定：`contexts/` 4 个文件、`stores/` 1 个 Zustand store、少量 Jotai atom，其余全部组件内 useState。

**结论**：这两个文件大的根因 60% 是缺分层（状态机/解析器/编排逻辑没有按 `AGENTS.md` 硬规则 9 抽 View Model / projection），40% 是缺拆文件。仓库已有正确模式可复用：`packages/agent-runtime-projection`（被 import 30 次）和 `components/agent/chat/projection/`。

---

## 轴 D · 后端调用网关叠床架屋

前端到后端至少四套封装并存：

| 层 | 规模 | 角色 |
|---|---|---|
| `src/lib/api/` | 208 文件，被 import 约 940 次 | 主网关（current） |
| `src/lib/dev-bridge/` | 16 文件 | renderer bridge（current）+ 迁移期 commandPolicy/mock（compat，归 CCD-012） |
| `src/lib/desktop-host/` | 73 文件 | mock / 测试夹具（current test 事实源） |
| `packages/app-server-client` | 4 文件 | App Server JSON-RPC client（current，被 lib/api 包装） |

同一能力多路径实例：媒体任务预览 runtime（`useWorkspaceImageTaskPreviewRuntime.ts` 等）直接 import `@/lib/dev-bridge`，绕过 `lib/api`；runtime 事件既可走 `agentRuntimeEvents.ts` 也可走 `dev-bridge/safeListen`。

**结论**：调用路径不唯一 → 后端改动影响面不可控、mock 边界难守。目标拓扑应是 `components/features → lib/api → app-server-client`，dev-bridge 只作为 lib/api 内部传输细节，不直接暴露给业务代码。本轴与 `tech-debt-tracker.md` CCD-012 是同一治理面，不另起炉灶。

---

## 轴 E · core/services 垃圾抽屉化 + 重复定义

- `lime-core` 被 11+ crate 依赖，内容横跨 config（`config/types.rs` 4966 行）、models、database、plugin——基础 crate 不最小化，等于全 workspace 耦合中心。与 Codex 仓库显式写出的 "resist adding code to codex-core" 是同一病灶（见 `codex-engineering-patterns.md` § 2）。
- `lime-rs/crates/services/` 32 个 service 平铺，`lib.rs` 注释里按依赖分了四类但目录结构不体现；`model_registry_service.rs` 4689 行内含多协议（OpenAI/Anthropic/Gemini/Ollama）抓取 + 缓存 + 校验。
- 重复定义实例：模型注册类型在 `lime_core::models::model_registry`（20+ 类型）与 `services/model_registry_service.rs`（`ProviderModelsCachePayload` 等缓存类型）两处维护；config 在 `core/config/types.rs` 与 `runtime.rs` 的 executor context builder 两套。

**agent-rust 定位（附带结论）**：fork 自 `agentcloud/agent-rust` v0.27.2，已无 .git/submodule/上游 remote，**完全自有化**，不在外层 workspace members（`exclude` + path 依赖引入）。其 `agents/agent.rs` 8206 行（Agent 执行主循环）、`scheduler/types.rs` 5617 行（类型堆叠）适用与轴 B 相同的模块化策略，无上游同步顾虑。

---

## 轴 F · 体量债存量（症状清单）

233 个文件超 1000 行（前端 107、Rust 126）。Top 文件清单与棘轮护栏规格见 `file-size-ratchet-guard-spec.md`。本轴只做两件事：止血（棘轮）+ 作为 A/B/C 轴落地后的验收指标，不再单独作为重构主线。

---

## packages/ 消费关系（佐证，供轴 D/收缩决策用）

| Package | 在 src/ 被 import 次数 | 备注 |
|---|---|---|
| agent-runtime-projection | 30 | current，正确模式样板 |
| agent-ui-contracts | 28 | current 类型契约 |
| app-server-client | 20 | current |
| agent-runtime-client | 6 | 仅 `features/plugin/runtime` 用 |
| agent-runtime-ui | 1 | 几乎未用 |
| plugin-runtime | **0** | 零引用，候选下线 |
| cli | 0（外发 CLI） | 保留 |
