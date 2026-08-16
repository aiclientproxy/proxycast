# 项目概览

状态：current

本页是全局架构图的简短入口，不定义第二套架构。目录归属、依赖方向、协议边界、Thread / Turn / Item 语义与重大变更确认规则以 [architecture.md](architecture.md) 为唯一裁决源。

## 产品链

```text
React Renderer
  -> Electron Desktop Host
  -> App Server JSON-RPC
  -> RuntimeCore / agent-runtime
  -> model-provider + tool-runtime
  -> Thread/Turn/Item + ProjectionStore
  -> Renderer projection / Runtime Facts
```

- Renderer 负责产品交互、局部显示状态和 i18n；不保存运行时真相，也不拼 provider 请求。
- Electron 负责窗口、preload、IPC 白名单、系统能力、sidecar 生命周期和更新；不成为业务后端。
- App Server 是跨应用业务协议入口，负责 JSON-RPC、初始化、handler、canonical read model 和领域接线。
- `agent-runtime` 负责回合生命周期与状态机；`model-provider` 负责多模型 capability、canonical content 和 provider lowering；`tool-runtime` 负责工具权限、调度、MCP 与结果归一。
- `thread-store` 与 ProjectionStore 承担可恢复的 Thread / Turn / Item 读取事实；UI 缓存和 stream buffer 不得反向成为真相。
- Provider 网络只有 `model-provider` 一个 current owner；已删除的 `lime-providers` 属于 `dead / forbidden-to-restore`，只能出现在历史 evidence 或负向守卫。

## 参考与边界

- Agent runtime、App Server、Thread / Turn / Item、工具生命周期、MCP、Skills、Multi-Agent、history hydrate、projection 与测试护栏对齐本地 Codex：`/Users/coso/Documents/dev/rust/codex`。
- 多模型控制平面的 model catalog、model switch、capability、provider readiness 与 retry/circuit breaker 以本地 grok-build：`/Users/coso/Documents/dev/rust/grok-build` 为主参考；provider wire 的多协议 endpoint、canonical content、媒体和 lowering 选择性参考本地 OpenCode：`/Users/coso/Documents/dev/js/opencode`。
- 运行时 owner 服从 Codex；model control 服从 grok-build；provider wire 由 Lime `model-provider` 统一承接并吸收 OpenCode 的协议边界；这些参考都不能替代 Lime 的桌面产品、i18n 或交付边界。

## 继续阅读

- 全局目录与 crate/package 准入：[architecture.md](architecture.md)
- Electron / App Server / renderer 命令契约：[commands.md](commands.md)
- provider 与多模型边界：[providers.md](providers.md)
- 当前治理与退场规则：[governance.md](governance.md)
- 质量门禁与 Gate A / Gate B：[quality-workflow.md](quality-workflow.md)
- Workspace 领域边界：[workspace.md](workspace.md)
