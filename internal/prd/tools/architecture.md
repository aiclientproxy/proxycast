# Lime 工具治理架构

## 1. 设计目标

本次工具治理的设计目标只有四个：

1. **只保留一套 metadata 解释器**
2. **只保留一份 native tool catalog**
3. **MCP 独立运行，但统一注入 Agent runtime**
4. **把“工具发现”和“工具权限执行”彻底分开**

---

## 2. 核心事实源

### 2.1 统一元数据事实源

路径：`lime-rs/crates/core/src/tool_calling.rs`

负责：

- `deferred_loading`
- `always_visible`
- `allowed_callers`
- `tags`
- `input_examples`
- tool search 打分
- caller 归一化

任何地方如果还在自己读 `x-lime` / `x_lime`，都应视为治理退化。

### 2.2 native 目录事实源

路径：`lime-rs/src/agent_tools/catalog.rs`

负责：

- 工具目录完整性
- source / lifecycle / capability / permission_plane
- 默认 allowlist 子集
- Workbench / Browser Assist surface 裁剪
- MCP extension surface 聚合

### 2.3 执行权限事实源

路径：`lime-rs/src/agent_tools/execution.rs`

负责：

- execution 层 warning gate 收口
- workspace 参数限制模板收口
- sandbox profile 归类
- permission 模板生成
- inventory execution profile 暴露
- 默认策略、persisted policy、runtime session override 的优先级合并

### 2.4 MCP runtime 事实源

路径：`lime-rs/crates/mcp/src/manager.rs`

负责：

- MCP server lifecycle
- tool cache
- `list_tools`
- `list_tools_for_context`
- `search_tools`
- runtime metadata 继承与自动 defer 策略

### 2.5 Agent 注入事实源

路径：`lime-rs/src/commands/agent_cmd.rs`

负责：

- `tool_search` bridge tool
- MCP -> Agent extension 注入
- workspace tool allowlist
- runtime tool inventory 命令

### 2.6 轻量测试载体

路径：

- `lime-rs/crates/agent/src/lib.rs`
- `lime-rs/crates/agent/src/agent_tools/mod.rs`

负责：

- 把 `lime-rs/src/agent_tools/catalog.rs`
- 把 `lime-rs/src/agent_tools/execution.rs`
- 把 `lime-rs/src/agent_tools/inventory.rs`

以模块方式复用到 `lime-agent` crate 内，供纯逻辑单测执行。

注意：

- 它不是新的 runtime 事实源
- 它只是测试载体，避免 `lime` 主包为 Tauri/App wiring 做超大链接
- 运行时事实源仍然是 app crate 下的 `catalog.rs` / `execution.rs` / `inventory.rs`

---

## 3. 总体架构图

```mermaid
graph TD
    UI[前端 / agentRuntime.ts] --> CMD[agent_cmd.rs]

    CMD --> CAT[agent_tools/catalog.rs]
    CMD --> EXEC[agent_tools/execution.rs]
    CMD --> INV[agent_tools/inventory.rs]
    CMD --> META[lime_core::tool_calling.rs]
    CMD --> MCP[lime_mcp::manager.rs]
    CMD --> AGENT[Agent Agent]
    TEST[lime-agent test carrier] -. 复用纯逻辑 .-> CAT
    TEST -. 复用纯逻辑 .-> EXEC
    TEST -. 复用纯逻辑 .-> INV

    MCP --> MCPTOOLS[MCP Runtime Tools]
    MCP --> PROMPTS[MCP Prompts / Resources]

    AGENT --> REG[ToolRegistry]
    AGENT --> EXT[ExtensionManager]

    MCPTOOLS --> EXT
    META --> MCP
    META --> CMD
    META --> PROVIDERS[Provider Tool Conversion]

    CAT --> EXEC
    CAT --> INV
    EXEC --> INV
    REG --> INV
    EXT --> INV
    MCP --> INV
```

---

## 4. 初始化时序图

下面是 Lime 启动 Agent 并把 MCP 工具面注入 Agent 的主链路。

```mermaid
sequenceDiagram
    participant UI as 前端
    participant CMD as agent_init
    participant STATE as AgentState
    participant MCP as McpClientManager
    participant META as lime_core::tool_calling
    participant EXT as Agent ExtensionManager

    UI->>CMD: agent_init()
    CMD->>STATE: init_agent_with_db()
    CMD->>MCP: ensure_lime_mcp_servers_running()
    CMD->>MCP: list_tools()
    MCP->>META: extract_tool_surface_metadata()
    MCP-->>CMD: McpToolDefinition[]
    CMD->>CMD: build_mcp_extension_surface()
    CMD->>EXT: add_client(extension, bridge_client)
    CMD->>CMD: ensure_tool_search_tool_registered()
    CMD-->>UI: initialized + tool surface ready
```

---

## 5. 工具检索 / 按需加载流程图

这个流程对应 Anthropic Tool Search / Codex defer loading 的同类思路。

```mermaid
flowchart TD
    A[用户任务] --> B{现有默认上下文能否完成?}
    B -- 是 --> C[直接使用默认可见工具]
    B -- 否 --> D[调用 tool_search]
    D --> E[查询 native registry + extension searchable tools]
    E --> F{目标工具是否 deferred?}
    F -- 否 --> G[直接执行]
    F -- 是 --> H[通过 ExtensionManager 加载目标工具]
    H --> I[把工具加入可执行工具面]
    I --> G
    G --> J[执行时再走 sandbox / approval / 参数限制]
```

关键点：

- `tool_search` 负责“找工具”
- `allowed_callers` 负责“谁能看见 / 调用”
- `deferred_loading` 负责“是否默认进上下文”
- sandbox / approval 负责“执行时能不能做”

---

## 6. 权限平面拆分

## 6.1 目录层

由 `catalog.rs` 定义。

回答的问题：

- 这个工具属于哪个产品 surface
- 是 current 还是 compat
- 是 Agent builtin、Lime injected 还是 Browser compatibility

## 6.2 上下文层

由 `tool_calling.rs` + `mcp manager` + `tool_search` 定义。

回答的问题：

- 默认可见还是 deferred
- caller 是否匹配
- 是否需要 tool_search 后再进入

## 6.3 执行层

由 `agent_tools/execution.rs` + Agent `ToolPermissionManager` + workspace sandbox / runtime approval 定义。

回答的问题：

- 参数是否受限
- 是否要求审批
- 是否要进入 sandbox
- 当前 workspace 是否允许

> 结论：**不要再让目录层和执行层共享同一套“权限”语义。**

---

## 7. 与 Codex 的对照

## 6.4 执行策略解析时序图

下面是一次工具执行前，effective execution policy 的解析链路。

```mermaid
sequenceDiagram
    participant UI as 前端 / request.metadata
    participant CMD as agent_cmd.rs
    participant CFG as NativeAgentConfig.tool_execution
    participant EXEC as agent_tools/execution.rs
    participant PERM as ToolPermissionManager

    UI->>CMD: submit turn / get inventory(metadata)
    CMD->>CFG: 读取 persisted policy
    CMD->>EXEC: resolve_tool_execution_policy(tool, config, metadata)
    EXEC->>EXEC: 合并默认策略
    EXEC->>EXEC: 叠加 persisted override
    EXEC->>EXEC: 叠加 runtime override
    EXEC-->>CMD: effective policy
    CMD->>PERM: build_workspace_execution_permissions(...)
    CMD-->>UI: inventory / runtime 权限结果
```

结论：

- `catalog.rs` 仍只定义目录层事实
- `execution.rs` 独占执行层合并逻辑
- `agent_cmd.rs` 只负责 orchestration，不再手工散写 permission 模板
- provenance 也在 `execution.rs` 统一生成，inventory 只消费结果，不再自行推断来源

---

## 7. 与 Codex 的对照

## 7.1 Codex 怎么做

参考：

- `codex-rs/app-server-protocol/src/protocol/v2.rs`
- `codex-rs/state/src/runtime/threads.rs`

Codex 的关键点：

1. **小型常驻工具配置**
   - `ToolsV2` 只保留少量稳定入口，例如 `web_search`、`view_image`

2. **动态工具独立建模**
   - `DynamicToolSpec` 单独描述动态工具
   - 包含 `defer_loading`

3. **线程级动态工具持久化**
   - `thread_dynamic_tools`
   - 动态工具跟着 thread，而不是全局把所有 schema 塞进 prompt

4. **权限与工具发现分离**
   - `AppToolsConfig` 管具体 app tool 的 enable / approval mode
   - sandbox / approval 是另一套系统

## 7.2 Lime 应该学什么

Lime 不需要一比一复制 Codex，但要学到这三个原则：

1. **常驻工具面要小**
2. **长尾工具靠搜索和 deferred loading**
3. **权限执行不要和工具目录绑死**

## 7.3 Lime 当前对应关系

| Codex                           | Lime 对应实现                                                        |
| ------------------------------- | -------------------------------------------------------------------- |
| `DynamicToolSpec.defer_loading` | `x-lime.deferred_loading` + `McpToolDefinition.deferred_loading`     |
| `thread_dynamic_tools`          | Agent ExtensionManager searchable tools + MCP runtime cache          |
| 小常驻工具面                    | `workspace_default_allowed_tool_names(...)`                          |
| app tool config / approval 分离 | `catalog.rs` + `execution.rs` + workspace sandbox / Agent permission |
| persisted permissions profile   | `NativeAgentConfig.tool_execution`                                   |
| thread/request runtime override | `request.metadata.harness.executionPolicy`                           |

---

## 8. MCP 在架构里的位置

MCP 不是 native tools 的附属物，也不是另起一套 Agent。

它在 Lime 中应被视为：

- **独立的 runtime tool fabric**
- 但通过 **统一的 extension 注入边界** 进入 Agent

这意味着：

- MCP server 启停、缓存、resource/prompt 仍然独立
- Agent 只看到统一后的 prefixed tool surface
- 工具搜索可以同时搜 native 与 extension
- 库存命令可以同时做 catalog / runtime / MCP 三视角盘点

---

## 9. dead / deleted 说明

以下路径已退出主链路并物理删除：

- `lime-rs/crates/agent/src/tool_permissions.rs`
- `lime-rs/crates/agent/src/shell_security.rs`
- `lime-rs/crates/agent/tests/legacy_permission_surfaces.rs`

状态：

- `dead / deleted / forbidden-to-restore`
- current 权限执行只归 `agent_tools/execution.rs` 与 `tool-runtime` 对应 owner
- 不得恢复旧文件、测试夹具、crate 出口或兼容包装
