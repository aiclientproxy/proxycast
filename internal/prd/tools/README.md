# Lime 工具治理总览

更新时间：2026-06-13

## 1. 背景

当前 Lime 的工具体系已经不只是单一的 Agent native tools，还同时包含：

- Agent 默认内置工具
- Lime 注入工具
- Workbench 专属工具
- Browser Assist 兼容工具面
- Lime MCP runtime tools
- Agent ExtensionManager 注入后的 prefixed tools

这套能力本身已经接近 Tool Calling 2.0，但过去存在两个核心问题：

1. **工具事实源分裂**
   - MCP schema metadata、Agent runtime registry、Lime 注入 extension、provider 转换层分别做了解析
   - 同一个字段（如 `deferred_loading` / `allowed_callers` / `input_examples`）在多处重复解释

2. **权限平面混杂**
   - “工具是否应该进入上下文”
   - “工具是否允许某个 caller 调用”
   - “工具调用后是否需要 sandbox / approval”
   - “参数是否受限”
     过去没有被严格分层，导致工具越多，上下文与权限越容易错乱

本次治理的目标，是把 Lime 的工具系统收敛到一条清晰主链路，参考 Codex 的思路：

- **小而稳定的常驻工具面**
- **按需搜索 / 延迟加载的动态工具**
- **工具发现与权限执行分离**
- **MCP 作为独立体系接入，但统一进入 Agent runtime**

---

## 2. 本次结论

### 2.1 不是“没有 Tool Search”，而是“已经有一半，但事实源没收口”

Lime 实际已经具备这些能力：

- `search_tools`
- `list_tools_for_context`
- `ToolSearch` bridge tool
- `deferred_loading`
- `allowed_callers`
- `input_examples`
- MCP -> Agent extension 注入

真正的问题不是缺能力，而是：

- metadata 解析分散
- native tool 目录不完整
- MCP / extension / provider 多处重复解释
- runtime 缺少一份可审计的“工具库存快照”

### 2.2 现役事实源

本轮治理后，建议把事实源固定为：

- **工具元数据事实源**：`lime-rs/crates/core/src/tool_calling.rs`
- **native 工具目录事实源**：`lime-rs/crates/agent/src/agent_tools/catalog.rs`
- **执行权限事实源**：`lime-rs/crates/agent/src/agent_tools/execution.rs`
- **MCP runtime 工具事实源**：`lime-rs/crates/mcp/src/manager.rs`
- **Agent 注入工具面事实源**：`lime-rs/crates/agent` + App Server `RuntimeBackend`
- **工具库存 / 审计快照事实源**：`lime-rs/crates/agent/src/agent_tools/inventory.rs`
- **工具库存读取入口**：App Server JSON-RPC `agentSession/toolInventory/read`

### 2.3 当前 / 兼容 / 待清理分类

| 分类           | 路径 / 对象                                                       | 说明                                                                                                                |
| -------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| current        | `lime_core::tool_calling`                                         | 统一 metadata 读取与打分                                                                                            |
| current        | `lime-rs/crates/agent/src/agent_tools/catalog.rs`                 | 完整 native tool 目录与默认授权子集                                                                                 |
| current        | `lime-rs/crates/agent/src/agent_tools/execution.rs`               | 统一 execution 层的 warning / sandbox / 参数限制事实源                                                              |
| current        | `lime-rs/crates/mcp/src/manager.rs`                               | MCP tools runtime registry                                                                                          |
| current        | `lime-rs/crates/app-server/src/runtime_backend/tool_inventory.rs` | App Server `agentSession/toolInventory/read` 到 runtime inventory builder 的桥接                                    |
| current        | `lime-rs/crates/agent/src/agent_tools/inventory.rs`               | runtime 工具库存快照                                                                                                |
| compat         | `workspace_allowed_tool_names(...)`                               | 当前保留为旧调用入口别名，实际委托默认授权目录                                                                      |
| dead           | `agent_runtime_get_tool_inventory`                                | 旧 Desktop / Tauri facade 已退出现役入口；不得回到 Electron Host、DevBridge truth、desktop-host mock 或前端 gateway |
| dead / deleted | `lime-rs/crates/agent/src/tool_permissions.rs`                    | 已由 current execution owner 取代并物理删除；forbidden-to-restore                                                   |
| dead / deleted | `lime-rs/crates/agent/src/shell_security.rs`                      | 已由 current execution owner 取代并物理删除；forbidden-to-restore                                                   |

> 旧权限实现及其正向测试夹具已删除；后续只能扩展 current execution owner，不得恢复平行权限系统。

---

## 3. 本次已落地实现

### 3.1 元数据收口

已统一到 `lime_core::tool_calling`：

- `extract_tool_surface_metadata`
- `tool_visible_in_context`
- `tool_matches_caller`
- `score_tool_match`
- `normalize_tool_caller`

以下模块都已切换到共享逻辑：

- `lime-rs/crates/mcp/src/manager.rs`
- `lime-rs/crates/agent/src/agent_tools/inventory.rs`
- `lime-rs/crates/providers/src/providers/openai_custom.rs`
- `lime-rs/crates/providers/src/providers/claude_custom.rs`

### 3.2 native 工具目录补全

`lime-rs/crates/agent/src/agent_tools/catalog.rs` 已升级为完整目录，覆盖：

- Agent built-ins
- Lime 注入工具
- Workbench 工具面
- Browser Assist 兼容前缀

并明确了：

- `ToolSourceKind`
- `ToolPermissionPlane`
- `ToolLifecycle`
- `workspace_default_allow`

### 3.3 runtime 库存快照

新增：

- `lime-rs/crates/agent/src/agent_tools/inventory.rs`
- App Server JSON-RPC `agentSession/toolInventory/read`
- `src/lib/api/agentRuntime/inventoryClient.ts` 对应 helper

这条命令可以一次返回：

- 当前 surface 的 catalog tools
- 默认允许工具集合
- runtime registry tools
- extension surfaces
- searchable / loaded extension tools
- MCP servers 与 MCP tools
- 映射缺口与可见性统计

### 3.4 执行权限事实源

新增：

- `lime-rs/crates/agent/src/agent_tools/execution.rs`

负责：

- `Bash` / `Task*` 的 warning gate 语义
- workspace 参数限制模板
- sandbox profile 归类
- execution permission 模板生成
- inventory execution profile 暴露
- 默认策略 + persisted policy + runtime override 合并

其中策略覆盖入口已经收口为：

- **持久化覆盖**：`lime-rs/crates/core/src/config/types.rs` -> `NativeAgentConfig.tool_execution`
- **运行时覆盖**：`request.metadata.harness.executionPolicy` / `execution_policy`
- **有效策略解析**：`lime-rs/crates/agent/src/agent_tools/execution.rs::resolve_tool_execution_policy`

结果：

- 旧命令层不再手工拼整段 `ToolPermission` 模板
- execution 层事实源从命令层 if/else 收回 `agent_tools` 边界
- inventory 现在可直接审计 `execution_warning_policy` / `execution_restriction_profile` / `execution_sandbox_profile`
- `agentSession/toolInventory/read` 可通过 `metadata` 观察 runtime override 后的 effective profile
- inventory 同时暴露每个 execution 字段的来源：
  - `execution_warning_policy_source`
  - `execution_restriction_profile_source`
  - `execution_sandbox_profile_source`

### 3.5 前端契约同步

已同步更新：

- `src/lib/governance/agentCommandCatalog.json`
- `src/lib/tauri-mock/core.ts`
- `src/lib/dev-bridge/mockPriorityCommands.ts`
- `src/lib/api/agent.test.ts`
- `src/lib/api/agentRuntime.ts`

### 3.6 轻量测试通道

为避免 `lime` 主包在本地因 Tauri 链接过大而降低回归效率，已补一条对齐 Codex 思路的轻量测试通道：

- `lime-rs/crates/agent/src/lib.rs`
- `lime-rs/crates/agent/src/agent_tools/mod.rs`

这条通道直接复用 `lime-agent` crate 的：

- `lime-rs/crates/agent/src/agent_tools/catalog.rs`
- `lime-rs/crates/agent/src/agent_tools/execution.rs`
- `lime-rs/crates/agent/src/agent_tools/inventory.rs`

用于承接纯逻辑单测，而不是复制第二份实现。

结论：

- **runtime 事实源没有新增**
- **测试入口新增了一条更轻的执行面**
- `ToolSearch` 与 inventory 的 extension 状态判定也已继续收口到共享 helper，避免主包再次长出重复逻辑

### 3.7 旧权限表面删除

本轮继续做了一刀减法：

- `lime-rs/crates/agent/src/tool_permissions.rs`
- `lime-rs/crates/agent/src/shell_security.rs`

上述文件及 `lime-rs/crates/agent/tests/legacy_permission_surfaces.rs` 已物理删除。当前边界为：

- 不再通过 `lime-agent` crate 根对外 `pub mod`
- 不再通过 `lime-agent` crate 根对外 `pub use`
- 不再进入 `lime-agent` 的 `lib.rs` 编译图
- 不再进入正常 `cargo check` / 运行时编译图
- 不再通过测试夹具恢复或正向验证旧权限语义

同时新增了两层守卫：

- `scripts/report-legacy-surfaces.mjs`：防止旧权限模块重新公开、重新挂回 `lib.rs` 编译图，或被上层重新依赖
- `src/lib/governance/legacyToolPermissionGuard.test.ts`：防止旧文件、正向测试夹具、模块出口和上层类型引用恢复

---

## 4. 当前确认的静态工具面

### 4.1 Core surface

- **Agent built-ins 与 current tool surface**
  `Read` / `Write` / `Edit` / `Glob` / `Grep` / `Bash` / `LSP` / `Skill` / `TaskCreate` / `TaskList` / `TaskGet` / `TaskUpdate` / `TaskOutput` / `TaskStop` / `NotebookEdit` / `EnterPlanMode` / `ExitPlanMode` / `EnterWorktree` / `ExitWorktree` / `WebFetch` / `WebSearch` / `AskUserQuestion` / `SendUserMessage`

- **Lime injected current tools**  
  `ToolSearch` / `ListMcpResourcesTool` / `ReadMcpResourceTool` / `Agent`

- **Team runtime current surface**  
  `Agent` / `SendMessage` / `TeamCreate` / `TeamDelete` / `ListPeers`

- **Compat tool name**
  当前不再保留 `SubAgentTask`；team runtime 已收敛到 `Agent / SendMessage / TeamCreate / TeamDelete / ListPeers`

- **说明**  
  Core surface 现已按 current surface 收敛；精确数量与分类以 `lime-rs/crates/agent/src/agent_tools/catalog.rs` 为准。

### 4.2 Workbench surface

在 Core 之上额外增加 8 个：

- `social_generate_cover_image`
- `lime_create_video_generation_task`
- `lime_create_broadcast_generation_task`
- `lime_create_cover_generation_task`
- `lime_create_modal_resource_search_task`
- `lime_create_image_generation_task`
- `lime_create_url_parse_task`
- `lime_create_typesetting_task`

- **Workbench surface catalog total**：34 个

### 4.3 Browser Assist surface

目录里只保留一个前缀入口：

- `mcp__lime-browser__*`

但它实际映射到 Agent browser runtime 的一组 prefixed tools。
参考 Agent 的 `chrome_mcp/tools.rs`，当前浏览器工具定义为 **17 个**。

- **Browser Assist surface catalog total**：27 个
- **Workbench + Browser Assist 全量 surface**：35 个

---

## 5. 为什么这套方案比现状合理

### 5.1 更像 Codex，而不是“把所有 schema 全塞 prompt”

Codex 的思路是：

- 常驻工具面尽量小
- 动态工具按 thread 存储
- 通过 `defer_loading` 控制是否默认进入上下文
- 权限配置与工具发现分离

Lime 现在的目标状态也应该是：

- native 常驻面稳定
- MCP / long-tail tools 搜索后按需进入
- `allowed_callers` 只管调用者可见性
- sandbox / approval 只管执行权限

### 5.2 MCP 单独成体系，但不单独造第二套 agent 认知

MCP 在 Lime 里仍然是独立运行时：

- server 启停
- tool cache
- prompt/resource
- runtime list/search/call

但一旦进入 Agent，会通过 Agent `ExtensionManager` 统一挂接。
这样模型只面对一个工具宇宙，不需要理解两套完全不同的上下文协议。

### 5.3 权限终于能分层

建议永久保留三层概念：

1. **目录层**
   - 这个工具是否存在
   - 属于哪个 surface / source / lifecycle

2. **上下文层**
   - 这个工具是否默认进入上下文
   - 是否 deferred
   - caller 是否匹配

3. **执行层**
   - 参数限制
   - sandbox
   - approval
   - workspace allowlist

这三层不再混写，后续就不会随着工具数增加而指数级混乱。

---

## 6. 文档索引

- `internal/prd/tools/architecture.md`：架构、时序、流程图、Codex 对照
- `internal/prd/tools/inventory.md`：工具盘点、分类、库存命令说明
- `internal/prd/tools/development-plan.md`：开发计划、验收标准、下一刀
