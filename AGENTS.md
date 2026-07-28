# Lime 源码仓库指南

本仓库没有外部用户和历史兼容负担。发现重复实现、临时路径或已脱离构建图的代码，优先删除或直接替换，不保留双轨。

## 事实源

1. 代码仓库是唯一记录系统。影响实现的决策、计划和验证结果必须落在仓库内。
2. 根文件只保留仓库级约束和导航；领域规则放在 `internal/aiprompts/`，执行进度放在 `internal/exec-plans/`。
3. 每种能力只能有一个继续演进的 owner。`current` 可扩展；`compat` 只能委托；`deprecated` 只能迁出；`dead` 应删除并补回流守卫。
4. Agent runtime 的唯一产品链是 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`。Electron 不承接第二套业务后端。
5. Agent loop、状态机、Thread/Turn/Item、工具生命周期、MCP、Skills、Multi-Agent、历史恢复和 GUI 护栏对齐 `/Users/coso/Documents/dev/rust/codex`。多模型控制面的 catalog、默认选择、model switch、provider capability/readiness、retry/circuit breaker 与多模态 sampling 对齐 `/Users/coso/Documents/dev/rust/grok-build`；provider wire、canonical content 与媒体 lowering 可辅助参考 `/Users/coso/Documents/dev/js/opencode`。
6. Rust 后端只能落在 `lime-rs/crates/**` 的既有领域 owner。
7. Provider 网络边界归 `model-provider`；工具定义、权限和执行归 `tool-runtime`；会话/回合编排归 `agent-runtime` 与 App Server；投影和持久化归 App Server、`thread-store` 与对应 repository。
8. 新命名使用短的领域词。禁止把产品品牌、已退役 runtime 名称或冗长实现词带进新的 crate、命令、API、类型、模块和脚本。
9. 已退役 runtime 的 vendor、workspace crate、迁移目录与专用 skill 均为 `dead / deleted / forbidden-to-restore`。能力缺口只能参考 Codex 在 current owner 重建；不得恢复依赖、catalog、文档入口、fallback 或 compat 包装。

## 工程约束

1. 全程使用中文；代码注释遵循所在文件既有语言。
2. 先读后写，保持改动集中；不主动提交、推送、重置或创建分支。
3. 工作树存在未知改动时，遵循 `internal/aiprompts/parallel-agent-collaboration.md`：声明窄写集、避让脏热区，不覆盖他人改动。
4. 重大架构变更必须在同一变更集中更新 `internal/aiprompts/architecture.md`，并由责任开发者在执行计划和 PR 描述填写架构图确认。未确认不得标记完成或进入 release evidence。
5. 非生成文件接近 `800` 行时优先拆分；超过 `1000` 行不得继续堆叠业务逻辑，除非执行计划记录退出条件。
6. 用户数据、日志、缓存和凭证必须走平台 API 或统一封装；新增行为默认同时考虑 macOS 与 Windows。
7. `scripts/` 根目录与一级领域目录受冻结基线保护。新增脚本优先进入既有 `scripts/<domain>/`、`scripts/lib/` 或所属 package。
8. 生产路径不得回退 mock。mock 只允许测试夹具明确使用；renderer、Electron、App Server、GUI smoke 和业务 E2E 都必须走真实 current bridge。
9. 用户可见文案必须覆盖 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR`，并补稳定回归。
10. 配置、协议和依赖改动必须同步 schema、消费者、文档、锁文件与测试。

## 协议与质量

1. 改 Electron/App Server 命令时，同步 Desktop Host/preload、JSON-RPC protocol/client、前端 gateway、catalog 与测试 fixture，并执行 `npm run test:contracts`。
2. 普通改动优先运行受影响的定向测试；Rust 变更先用 `npm run test:rust:related -- <paths...>` 或对应 crate 测试，再按风险扩大。
3. GUI、Bridge、Workspace 或 Agent 主路径改动至少运行 `npm run verify:gui-smoke`；Agent chat 当前 fixture 改动先运行 `npm run smoke:agent-runtime-current-fixture`。
4. 真实交互证据分级：浏览器投影是 Gate A；Gate B 必须证明真实 Electron、preload/IPC、`app_server_handle_json_lines`、App Server JSON-RPC、runtime/read model 和用户可见状态。
5. 默认本地门禁为 `npm run verify:local`。全量前端测试优先用 `npm run test:resume` 续跑，避免无差别重跑。
6. 更新版本、Forge、Electron 或 workspace manifest 时执行 `npm run verify:app-version`。

## 执行方式

1. 长任务必须更新 `internal/exec-plans/`，记录目标、写集、退出条件、验证和阻塞。
2. 路线图任务先说明主目标、当前阶段和下一刀；治理删除必须直接服务主链交付。
3. 用户明确无兼容需求时，直接迁移调用并删除旧入口，不新增包装层。
4. 收尾报告当前/兼容/废弃/已删除分类、验证结果、未验证原因和完成度百分比。

## 导航

- 文档入口：`docs/README.md`
- 工程入口：`internal/aiprompts/README.md`
- 全局架构：`internal/aiprompts/architecture.md`
- 架构与边界：`internal/aiprompts/overview.md`、`internal/aiprompts/commands.md`
- 治理：`internal/aiprompts/governance.md`
- 质量与 GUI：`internal/aiprompts/quality-workflow.md`、`internal/aiprompts/playwright-e2e.md`
- 执行计划：`internal/exec-plans/README.md`

## 高频命令

```bash
npm run verify:local
npm run test:contracts
npm run verify:gui-smoke
npm run smoke:agent-runtime-current-fixture
npm run smoke:claw-chat-current-fixture
npm run governance:legacy-report
npm run governance:scripts
npm run electron:dev
npm run bridge:health -- --timeout-ms 120000
npm run test:rust:related -- <paths...>
cargo test --manifest-path "lime-rs/Cargo.toml"
```
