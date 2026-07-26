# Codex Snapshot 对齐方案

状态：663 条映射方案完整；P0 owner 回归与多模型审计持续收尾，精确 Gate 尚未完成
范围：`/Users/coso/Documents/dev/rust/codex/codex-rs` 的全部 `.snap`
对应主线：`internal/refactor/v1`

## 结论

Codex snapshot 是 Lime 的场景来源和状态机 oracle，不是可以直接复制的 TUI golden file。
Lime 是 Electron GUI 产品，最终测试必须按 `Electron Desktop Host -> App Server JSON-RPC ->
RuntimeCore -> Thread/Turn/Item projection -> GUI` 分层记录证据。

本轮扫描得到 `663` 个源快照、`658` 个去平台后缀的逻辑场景和 `5` 个平台变体。完整逐文件
映射见 [03-source-to-scenario-map.md](03-source-to-scenario-map.md)，按 owner 分块的原始
路径见 [inventory/00-summary.md](inventory/00-summary.md)。

实现覆盖、精确 Gate 状态和 owner blocker 见
[04-implementation-status.md](04-implementation-status.md)。整体 GUI smoke 通过只证明 current
主链可用，不自动把每个故障注入场景标记为 Gate B 已覆盖。

## 快照如何迁移

| Codex 证据                    | Lime 方案落点                                | 说明                                                               |
| ----------------------------- | -------------------------------------------- | ------------------------------------------------------------------ |
| TUI 文本整屏                  | structured projection / read model assertion | 验证身份、顺序、状态和终态，不保留终端空格                         |
| 弹窗/列表/输入交互            | React component + DOM interaction            | 使用 role、accessible name、稳定 `data-*` 状态                     |
| 窄宽、换行、滚动、面板锚点    | Gate A screenshot / DOM geometry             | 固定 viewport、字体和 locale；遮蔽动态时间与光标                   |
| 流式/工具/审批/恢复           | Gate B Electron fixture                      | 必须证明 preload、IPC、App Server、read model 和 GUI 同一 identity |
| CLI/远端 prompt/request shape | runtime/App Server contract test             | 不把 CLI 输出伪装成前端视觉证据                                    |

## 测试层级

- `unit`：纯 selector、projection、formatter、状态机和稳定快照摘要。
- `component`：React DOM、hook 生命周期、键盘/鼠标事件和少量接线。
- `contract`：JSON-RPC、typed gateway、Thread/Turn/Item、reverse server request。
- `integration`：RuntimeCore、ThreadStore、MCP、工具执行、历史恢复。
- `gate-a`：Renderer/DOM/布局/交互。只能证明投影和浏览器页面可用。
- `gate-b`：真实 Electron Desktop Host 到 GUI 的 current 主链。不能由 Gate A 替代。

## 状态与命名

方案中的 scenario id 使用短领域名：

`turn-*`、`tool-*`、`approval-*`、`history-*`、`composer-*`、`command-*`、`model-*`、
`status-*`、`mcp-*`、`multi-agent-*`、`markdown-*`、`media-*`、`plugin-*`、`settings-*`、
`error-*`、`layout-*`、`runtime-*`、`cli-*`。

同一业务状态的 Codex 平台变体合并为一个逻辑场景，平台差异进入 `platform` 矩阵；不要为
Windows 换行或 macOS 终端颜色复制一套 Lime 业务测试。

## 必须遵守的边界

1. 生产 Renderer、Electron、App Server 和 GUI smoke 不得回退 mock。
2. 测试等待业务事件或 terminal predicate，不用固定 sleep 合成完成态。
3. 用户可见断言覆盖 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR`；protocol/evidence 字段不本地化。
4. 旧 runtime、旧命令和 TUI 专属能力只可进入 `defer` 或负向 guard，不恢复为 current owner。
5. 截图只保存必要 UI 证据，不保存 token、API key、完整 prompt 或敏感本地路径。

## 推荐执行顺序

1. 先跑 [02-runtime-contract-test-plan.md](02-runtime-contract-test-plan.md) 的 P0 contract。
2. 再跑 [01-frontend-test-plan.md](01-frontend-test-plan.md) 的 P0 component/projection。
3. 对 `layout-*`、`approval-*`、`history-*`、`tool-*` 补 Gate A/Gate B。
4. 最后处理 plugin、settings、feedback、update、CLI 和平台差异的 P1/P2 项。

## 参考事实源

- [Lime v1 重构 README](/Users/coso/Documents/dev/ai/aiclientproxy/lime/internal/refactor/v1/README.md)
- [Lime 质量工作流](/Users/coso/Documents/dev/ai/aiclientproxy/lime/internal/aiprompts/quality-workflow.md)
- [Lime Playwright 续测手册](/Users/coso/Documents/dev/ai/aiclientproxy/lime/internal/aiprompts/playwright-e2e.md)
- [Codex snapshot 原始目录](/Users/coso/Documents/dev/rust/codex/codex-rs)
- [grok-build 多模型事实源](/Users/coso/Documents/dev/rust/grok-build)

Thread/Turn/Item、流式、工具、审批、历史恢复、GUI 状态以
`codex@4c43465133428898aa84f0bfc02c306ed65fb66a` 为 oracle。model catalog、model
switch、reasoning effort 菜单与 wire value、provider capability/readiness、retry/circuit
breaker 以 `grok-build@6e386420825bd44ae648c63e7c8cba12fcec9401` 为 oracle。Codex 的 model
picker 快照只提供交互和布局场景，不裁决 Lime 的多模型控制平面语义。
