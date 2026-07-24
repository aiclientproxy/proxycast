# Lime 工具库存与分类

## 1. 静态 catalog 盘点

## 1.1 Core surface

### Agent built-ins 与 current surface

- `Read`
- `Write`
- `Edit`
- `Glob`
- `Grep`
- `Bash`
- `LSP`
- `Skill`
- `TaskCreate`
- `TaskList`
- `TaskGet`
- `TaskUpdate`
- `TaskOutput`
- `TaskStop`
- `NotebookEdit`
- `EnterPlanMode`
- `ExitPlanMode`
- `EnterWorktree`
- `ExitWorktree`
- `WebFetch`
- `WebSearch`
- `AskUserQuestion`
- `SendUserMessage`

### Lime injected current tools

- `ToolSearch`
- `ListMcpResourcesTool`
- `ReadMcpResourceTool`
- `Agent`

### Team runtime current surface

- `Agent`
- `SendMessage`
- `TeamCreate`
- `TeamDelete`
- `ListPeers`

### Compat tool name

- 当前不再保留 `SubAgentTask`

### Core 总数

- 以 `lime-rs/crates/agent/src/agent_tools/catalog.rs` 当前目录为准，不再手写旧 surface 固定数值

---

## 1.2 Workbench surface 增量（8）

- `social_generate_cover_image`
- `lime_create_video_generation_task`
- `lime_create_broadcast_generation_task`
- `lime_create_cover_generation_task`
- `lime_create_modal_resource_search_task`
- `lime_create_image_generation_task`
- `lime_create_url_parse_task`
- `lime_create_typesetting_task`

### Workbench 总数

- 以 `lime-rs/crates/agent/src/agent_tools/catalog.rs` 当前目录为准

---

## 1.3 Browser Assist

目录层只保留一个兼容前缀：

- `mcp__lime-browser__*`

它不是一个单独真实工具，而是一组 browser runtime tools 的聚合入口。  
参考 Agent `chrome_mcp/tools.rs`，当前浏览器工具定义为 **17 个**。

### Browser Assist 总数

- 以 `lime-rs/crates/agent/src/agent_tools/catalog.rs` 当前目录为准

### Workbench + Browser Assist 总数

- 以 `lime-rs/crates/agent/src/agent_tools/catalog.rs` 当前目录为准

---

## 2. 默认授权子集

Core surface 当前默认 allow 的工具集已经收敛到 current surface，重点包括：

- `Skill`
- `TaskOutput`
- `EnterPlanMode`
- `ExitPlanMode`
- `WebSearch`
- `AskUserQuestion`
- `ToolSearch`
- `Agent`
- `SendMessage`
- `TeamCreate`
- `TeamDelete`
- `ListPeers`

结论：

- 默认 allowlist 继续优先保留常驻、小而稳的工具面
- `SubAgentTask` 已从应用层删除，不再作为 current 默认协作工具面
- `read` / `write` / `edit` / `bash` / `WebFetch` 这类需要参数约束或更强执行控制的工具仍不应默认放开

这符合“常驻工具面小而稳”的原则。

---

## 3. 当前库存读取入口

## 3.1 后端入口

- App Server JSON-RPC `agentSession/toolInventory/read`

实现路径：

- `lime-rs/crates/app-server-protocol/src/protocol/v0/agent_session.rs`
- `lime-rs/crates/app-server/src/runtime_backend/tool_inventory.rs`
- `lime-rs/crates/agent/src/agent_tools/inventory.rs`

旧 `agent_runtime_get_tool_inventory` 已从 Electron Host、DevBridge truth、desktop-host mock 和前端 gateway 退场，只允许保留在 retired guard、负向测试和历史 evidence 中。

## 3.2 前端 helper

- `src/lib/api/agentRuntime/inventoryClient.ts`
- `getAgentRuntimeToolInventory(...)`

### 调用示例

```ts
import { getAgentRuntimeToolInventory } from "@/lib/api/agentRuntime/inventoryClient";

const snapshot = await getAgentRuntimeToolInventory({
  caller: "assistant",
  workbench: true,
  browserAssist: true,
  metadata: {
    harness: {
      executionPolicy: {
        toolOverrides: {
          bash: {
            warningPolicy: "none",
          },
        },
      },
    },
  },
});
```

## 3.3 返回内容

库存快照会同时返回：

- 请求 caller / surface
- 当前 agent 是否初始化
- warnings
- MCP servers
- 默认 allow 工具列表
- catalog tools
- runtime registry tools
- extension surfaces
- extension tools
- mcp tools
- catalog / registry 对应的 effective execution profile（warning / restriction / sandbox）
  - 默认策略：`execution.rs`
  - 持久化覆盖：`NativeAgentConfig.tool_execution`
  - 运行时覆盖：`request.metadata.harness.executionPolicy`
- catalog / registry 对应的 provenance 字段：
  - `execution_warning_policy_source`
  - `execution_restriction_profile_source`
  - `execution_sandbox_profile_source`
- counts

---

## 4. 这份库存解决了什么问题

过去你只能分别从这些地方猜测工具面：

- catalog
- registry
- mcp manager
- extension manager
- `ToolSearch` 输出

现在一条命令就能同时回答这些问题：

1. **静态目录里一共有多少工具？**
2. **当前 surface 下哪些是默认允许的？**
3. **Agent runtime registry 里实际注册了哪些工具？**
4. **哪些 runtime tools 没被 catalog 覆盖？**
5. **当前有哪些 extension surfaces？**
6. **哪些 extension tools 处于 deferred / loaded / visible？**
7. **MCP 真实运行了哪些 servers 和 tools？**

---

## 5. 分类建议

## 5.1 current

- `lime-rs/crates/core/src/tool_calling.rs`
- `lime-rs/crates/agent/src/agent_tools/catalog.rs`
- `lime-rs/crates/agent/src/agent_tools/execution.rs`
- `lime-rs/crates/agent/src/agent_tools/inventory.rs`
- `lime-rs/crates/mcp/src/manager.rs`
- `lime-rs/crates/app-server/src/runtime_backend/tool_inventory.rs`

## 5.2 compat

- `workspace_allowed_tool_names(...)`

## 5.3 deprecated

当前目录层没有新增 deprecated 工具；建议不要提前扩充 deprecated 层。

## 5.4 dead / deleted

- `lime-rs/crates/agent/src/tool_permissions.rs`：已物理删除，forbidden-to-restore
- `lime-rs/crates/agent/src/shell_security.rs`：已物理删除，forbidden-to-restore
- `lime-rs/crates/agent/tests/legacy_permission_surfaces.rs`：旧语义正向测试夹具已物理删除，forbidden-to-restore

---

## 6. 建议的库存使用方式

### 6.1 PR / 回归检查

每次工具相关改动，至少回答：

- registry tools 是否出现 catalog 未覆盖项
- extension surface 是否混入不该存在的 caller
- MCP tools 是否无意中全部默认进入上下文

### 6.2 调试上下文爆炸

先看：

- `default_allowed_tools`
- `registry_visible_total`
- `extension_tool_visible_total`
- `mcp_tool_visible_total`

如果这些数字异常上升，说明“默认进入上下文”的面在膨胀。

### 6.3 调试权限错乱

先分清问题属于哪层：

- catalog 层：目录不全 / 生命周期错
- context 层：`deferred_loading` / `allowed_callers` 错
- execution 层：sandbox / approval / 参数限制错

此时优先看：

- `execution_warning_policy`
- `execution_restriction_profile`
- `execution_sandbox_profile`
- `execution_warning_policy_source`
- `execution_restriction_profile_source`
- `execution_sandbox_profile_source`
- `request.metadata` 是否传入了 runtime override

---

## 7. 结论

库存命令不是为了“多一个调试页面”，而是为了把工具系统从“猜”变成“看得见”。

只要 inventory 这层一直存在，后续无论工具从 20 个涨到 200 个，都还能保持：

- 工具目录可审计
- 上下文暴露可审计
- MCP 注入可审计
- 权限平面可审计
