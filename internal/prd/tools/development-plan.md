# Lime 工具治理开发计划

## 1. 目标

把 Lime 的工具系统从“多源并存、权限混写、上下文不可审计”收敛为：

- 单一 metadata 事实源
- 单一 native catalog
- MCP runtime 独立但统一注入
- inventory 可审计
- 权限平面分层

---

## 2. 本轮已完成

## Phase A：事实源收口（已完成）

- [x] 把 tool metadata 解析统一到 `lime_core::tool_calling`
- [x] MCP manager 改为复用共享 metadata
- [x] `tool_search` bridge tool 改为复用共享 metadata
- [x] provider 转换层改为复用共享 metadata

### 验收标准

- 不再存在多处独立解析 `deferred_loading` / `allowed_callers` / `input_examples`
- tool search 与 MCP list/search 的语义一致

---

## Phase B：native catalog 完整化（已完成）

- [x] 建立完整 `ToolCatalogEntry`
- [x] 引入 `ToolSourceKind`
- [x] 引入 `ToolPermissionPlane`
- [x] 引入 `ToolLifecycle`
- [x] 补全 core / workbench / browser assist tools
- [x] 形成默认 allowlist 子集

### 验收标准

- 能回答“当前到底有哪些 native tools”
- 能区分 current / compat
- 能区分 session allowlist / parameter restricted / caller filtered

---

## Phase C：runtime inventory（已完成）

- [x] 新增 `agent_tools/inventory.rs`
- [x] 新增 App Server `agentSession/toolInventory/read`
- [x] 接入 `agentRuntime/inventoryClient.ts`
- [x] 接入 App Server protocol / client / contract guard / API test
- [x] 清退旧 `agent_runtime_get_tool_inventory` Electron facade 与 desktop-host mock

### 验收标准

- 一个 App Server current method 能同时输出 catalog / registry / extension / MCP 四视角
- 能看到 visible / deferred / caller_allowed 状态
- 能发现 registry 未被 catalog 覆盖的漂移项

---

## Phase D：文档与治理说明（已完成）

- [x] 输出 `internal/prd/tools/README.md`
- [x] 输出 `internal/prd/tools/architecture.md`
- [x] 输出 `internal/prd/tools/inventory.md`
- [x] 输出 `internal/prd/tools/development-plan.md`

### 验收标准

- 有架构图
- 有时序图
- 有流程图
- 有 current / compat / dead 分类
- 有 Codex / Agent / Lime 对照

---

## Phase E：测试矩阵补强（本轮完成）

- [x] 补齐 `catalog.rs` 的 surface / lifecycle / default allowlist 边界测试
- [x] 补齐 `inventory.rs` 的 caller / extension source / deferred 状态测试
- [x] 补齐 `mcp manager` 的冲突保序与默认 defer 阈值测试
- [x] 补齐 `tool_search` 的 extension 前缀匹配与状态判定测试
- [x] 补齐 provider 层对 `x-lime` / `x_lime` alias 与去重行为测试
- [x] 补齐前端 `agentRuntime.ts` inventory helper 默认参数测试
- [x] 新增 `lime-agent` 轻量测试载体，复用 `catalog.rs` / `inventory.rs` 纯逻辑模块，绕开主包 Tauri 大链接
- [x] 删除 `tool_permissions.rs` / `shell_security.rs` 及其正向 integration test 夹具，并补 forbidden-to-restore 治理守卫

## Phase F：执行权限事实源收口（本轮完成）

- [x] 新增 `lime-rs/crates/agent/src/agent_tools/execution.rs`
- [x] 把 workspace execution permission 模板从 `agent_cmd.rs` 收回 `agent_tools` 边界
- [x] 统一 `bash` / `Task` warning gate 语义
- [x] 把 execution profile 暴露到 inventory / `agentRuntime.ts`
- [x] 补齐 `execution.rs` 与 inventory 的定向测试
- [x] 新增 `NativeAgentConfig.tool_execution`，承接 persisted policy 覆盖
- [x] 让 `request.metadata.harness.executionPolicy` 承接 runtime session override
- [x] 让 `agentSession/toolInventory/read` 支持 `metadata`，返回 runtime override 后的 effective profile
- [x] 给 inventory 增加 provenance 字段，逐项标记 `default` / `persisted` / `runtime`
- [x] 补齐配置层 default / alias / roundtrip 测试

### 验收标准

- 对齐 Codex 风格的四类断言：
  1. **默认值是否稳定**
  2. **legacy / alias 字段是否兼容**
  3. **显式配置是否覆盖默认策略**
  4. **持久语义是否在多层之间保持一致**
- 新增工具治理逻辑至少要被以下矩阵之一覆盖：
  - catalog 边界
  - runtime inventory 快照
  - MCP runtime 过滤 / 搜索 / defer
  - provider metadata 透传
  - 前端命令 helper

### 本轮验证策略

- **小包优先**：优先跑 `lime-providers`、`lime-mcp` 与前端 `agent.test.ts`
- **轻量逻辑优先**：`catalog.rs` / `inventory.rs` 的纯逻辑单测优先从 `lime-agent` 执行，测试名使用模块前缀：
  - `agent_tools::catalog::tests::...`
  - `agent_tools::inventory::tests::...`
- **主包降级为编译检查**：`lime` 主包测试会触发超大链接；在当前环境磁盘仅余约 `4.4Gi` 时，优先做 `cargo check` / 定向编译验证
- **边界说明**：若后续 CI 或本地磁盘空间恢复，应追加一次 `lime` 主包的完整定向测试，把 `catalog.rs` / `inventory.rs` / `agent_cmd.rs` 新增测试全部实跑

### 推荐命令

```bash
# 轻量 Rust 纯逻辑测试
cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent \
  agent_tools::catalog::tests::test_tool_catalog_entries_for_surface_counts_and_lifecycle_boundaries -- --exact

cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent \
  agent_tools::inventory::tests::test_build_tool_inventory_marks_extension_sources_and_statuses -- --exact

# MCP / Provider 定向测试
cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-mcp empty_query_prioritizes_always_visible_then_name -- --nocapture
cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-providers supports_x_lime_alias -- --nocapture

# 前端契约与治理守卫
npm test -- "src/lib/api/agent.test.ts"
npm test -- "src/lib/governance/legacyToolPermissionGuard.test.ts"
npm run test:contracts
npm run governance:legacy-report
```

---

## 3. 下一刀建议

## 3.1 优先级 P1：删除旧权限系统并封住回流（已完成）

本轮已完成：

- `tool_permissions.rs` / `shell_security.rs` 已按 `dead / deleted / forbidden-to-restore` 物理删除
- `tests/legacy_permission_surfaces.rs` 正向测试夹具已同步删除
- 已从 `lime-agent` crate 根和正常编译图移除旧公开表面
- 已在治理脚本与 Vitest 护栏中禁止文件、模块出口和类型引用回流

---

## 3.2 优先级 P1：把“策略覆盖层”接到 execution 事实源

本轮已完成：

- `workspace allowlist` / `parameter restriction` / `warning gate` / `sandbox profile` 已有统一 execution 事实源
- `agent_cmd.rs` 已从手工 permission 模板拼装降级为 orchestration
- inventory 已能审计 execution profile
- `NativeAgentConfig.tool_execution` 已承接 persisted policy
- `request.metadata.harness.executionPolicy` 已承接 runtime session override
- inventory 已可查看 runtime override 后的 effective execution profile
- inventory 已可逐项查看 execution provenance，而不是只看最终值

剩余下一步建议：

- 如果后续要做 UI 可视化，再把 effective policy 与来源标识直接展示在工具调试页
- 保持 `catalog.rs` 只描述目录层，`execution.rs` 只描述执行层，避免覆盖逻辑再次散回命令层

---

## 3.3 优先级 P2：继续做减法，而不是再加抽象

不要再新增：

- 第二套 tool metadata parser
- 第二套 native tool list
- 第二套 browser tool 目录
- 第二套 MCP 注入路径

后续所有新工具都应满足：

1. 在 `catalog.rs` 有记录
2. metadata 走 `lime_core::tool_calling`
3. runtime inventory 能看见

---

## 4. 风险与对策

| 风险                        | 说明                                                | 对策                                       |
| --------------------------- | --------------------------------------------------- | ------------------------------------------ |
| catalog 与 runtime 再次漂移 | 新增工具时只改注册不改目录                          | 以 inventory + command catalog 作为守卫    |
| MCP caller 语义再次分裂     | MCP tool schema / extension allowed_caller 各自解释 | 一律先过 `tool_calling.rs`                 |
| 权限语义再次混写            | catalog、allowlist、sandbox、approval 又掺在一起    | 强制按目录层 / 上下文层 / 执行层汇报与设计 |
| 旧权限系统回流              | 新代码恢复或重新引用 `tool_permissions.rs`          | 按 forbidden-to-restore 执行仓库级扫描守卫 |

---

## 5. 最终建议

这次治理完成后，后续迭代请遵守三条硬规则：

1. **新增工具先入 catalog**
2. **新增 metadata 字段先入 `tool_calling.rs`**
3. **新增运行时注入能力必须能被 inventory 看见**

只要守住这三条，Lime 的工具数继续增长，也不会再回到“上下文失控 + 权限混乱”的状态。
