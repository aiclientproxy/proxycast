# 治理规则

## 唯一事实源

同一种能力在同一时期只能有一个继续演进的 owner。先写清事实源，再动代码：

- `current`：唯一产品路径，可以新增能力。
- `compat`：只做边界适配和委托，不得新增业务逻辑。
- `deprecated`：只允许迁出和删除。
- `dead`：无入口或已被替代，删除并补回流守卫。

Agent 产品主链固定为：

`Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`

其中 provider request/lowering 归 `model-provider`，工具定义、权限和执行归 `tool-runtime`，回合编排归 `agent-runtime` 与 App Server，持久化/read model 归 App Server、`thread-store` 与 repository。Electron 只负责 desktop host，不成为第二套 runtime。

运行时项目指令的 current 入口是标准 `CODEX_HOME/AGENTS.md` 与项目层 `AGENTS.md` / `AGENTS.override.md`；`lime-rs/crates/agent/src/prompt/runtime_agents.rs` 可继续读取 `.lime/AGENTS.md` / `.lime/AGENTS.override.md` 作为只委托旧文件位置的 `compat` fallback。该 fallback 不得扩散到其它 owner，也不得承接新语义。

`lime-providers`（原 workspace crate `crates/providers`）已完成消费者迁移并物理删除，
当前分类为 `dead / deleted / forbidden-to-restore`。它只能出现在历史 evidence 与负向
回流守卫；不得恢复 workspace 成员、依赖、catalog、fixture 或兼容包装。未知
model/capability/credential readiness 必须继续 fail closed，不能由 provider 名称、
legacy alias 或默认值放行。

Agent loop、状态机、Thread/Turn/Item、工具、MCP、Skills、Multi-Agent、历史恢复和 GUI 护栏按 `/Users/coso/Documents/dev/rust/codex` 收敛。多模型控制平面的 model catalog、model switch、provider capability、provider readiness 和 retry/circuit breaker 按 `/Users/coso/Documents/dev/rust/grok-build` 收敛；provider wire 的 endpoint、canonical content、媒体 lowering 和多协议 stream 可参考 `/Users/coso/Documents/dev/js/opencode`。grok-build/OpenCode 只提供 provider/model 机制参考，不成为 Lime runtime 或 GUI owner。

## 直接替换

当没有外部用户、持久化数据或发布协议约束时，旧路径直接替换并删除：

- 旧命令、旧 gateway、旧组件、旧 Hook、旧 storage key 和旧 fixture 命名。
- 生产 mock fallback、重复 read model、平行状态机和无引用目录。
- `protocol/v0`、`agentSession/*` production method、重复 provider crate 和声明但未实现的 transport variant。
- 与 current owner 冲突的文档、路线图、skill 和执行计划 active checklist。

不要为了降低 diff 新增新的 `compat` 层。历史计划、Git history 和 evidence 可以保留为历史记录，但不得被当前规则、导航或测试当成 owner。

## 修改前盘点

对涉及 runtime、命令、会话或工具的改动，盘点四层：

1. 入口层：页面、组件、Hook、前端 API。
2. 服务层：Electron bridge、App Server JSON-RPC、RuntimeCore、领域 service。
3. 存储层：schema、DAO、repository、缓存和迁移。
4. 旁路层：telemetry、evidence、replay、搜索、导出和测试 fixture。

若无法指出唯一 current owner，不扩展功能；先完成收敛。

## 守卫与验证

- 旧路回流优先由结构测试、catalog、contract test 和目录扫描阻止。
- 命令和 bridge 变更执行 `npm run test:contracts`。
- 旧路径清理执行 `npm run governance:legacy-report` 和相关定向测试。
- GUI 主路径变更执行 `npm run verify:gui-smoke`；真实产品闭环需要 Gate B Electron 证据。
- 生产不允许 mock fallback。fixture 可以使用受控 external backend，但必须清楚标注。

## 文档维护

文档只记录当前 owner、输入输出边界、验证入口和未完成 blocker。发现文档引用已删除目录、旧命令、旧 runtime 类型或不存在的文件时，直接更新或删除；不要把历史实现平移为“参考架构”。
