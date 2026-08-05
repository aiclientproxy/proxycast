# Plugin v2 路线图

状态：`active / macOS core Gate B passed`

更新时间：2026-08-05

## 主目标

Plugin v2 把 Lime 的插件能力收敛为一条与 Codex 对齐、由 App Server 持有事实、同时适配 Lime App Center 与 Claw 右侧工作区的产品链：

```text
Marketplace / local package
  -> App Server plugin domain
  -> installed plugin projection
  -> Skills / MCP / Hooks activation
  -> RuntimeCore turn context
  -> App Center / Claw composer / Right Surface
```

本轮不是给旧应用中心换皮，也不是继续扩展 `lime.plugin.package.v1`。目标是直接替换旧包标准、旧前端 registry 合并和旧插件 worker UI 协议，建立唯一继续演进的 Plugin owner。

## 事实源声明

1. 包结构、marketplace、插件 ID、Skills/MCP/Hooks 组成与安装语义对齐 Codex。
2. Lime 的生产命令链固定为 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> GUI`。
3. App Server 是插件目录、安装状态、启用状态、授权状态和运行时装配的唯一事实源。
4. Renderer 只消费 projection；不得扫描插件目录、解析 manifest、合并多份 registry 或启动第二套业务后端。
5. Claw 右侧只承载 MCP/App UI、浏览器或声明式结果投影；插件执行仍归 RuntimeCore、MCP 与工具运行时。

## 文档导航

| 文档                                                                                   | 职责                                                   |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| [00-research-findings.md](./00-research-findings.md)                                   | 官方文档、Codex 源码和本机 Desktop 安装包研究结论      |
| [01-product-contract.md](./01-product-contract.md)                                     | 产品对象、用户流程、状态模型与范围边界                 |
| [02-package-marketplace-installation.md](./02-package-marketplace-installation.md)     | 包结构、manifest、marketplace、安装与存储合同          |
| [03-architecture-and-command-contracts.md](./03-architecture-and-command-contracts.md) | owner、JSON-RPC、运行时装配与事件投影                  |
| [04-app-center-and-claw-surfaces.md](./04-app-center-and-claw-surfaces.md)             | App Center、详情、composer mention 与 Claw 右侧交互    |
| [05-migration-and-cleanup.md](./05-migration-and-cleanup.md)                           | current/deprecated/dead 分类、旧实现退役与历史文档治理 |
| [06-implementation-plan.md](./06-implementation-plan.md)                               | 分阶段写集、退出条件和主链顺序                         |
| [07-verification-contract.md](./07-verification-contract.md)                           | 单元、协议、集成、Gate A/B 与跨平台验收                |
| [08-legacy-synthesis.md](./08-legacy-synthesis.md)                                     | 旧文档价值提炼、冲突决策与历史参考边界                 |

## 非目标

- 不复刻 Codex/ChatGPT 的配色、圆角、字体或图标主题。
- 不把 Plugin、Skill、MCP、Agent App 继续做成相互独立的四套安装系统。
- 不在 Plugin v2 首阶段建设公共商业化发布平台、结算或审核后台。
- 不保留旧 manifest 双读双写、旧 worker fallback 或 renderer mock 生产降级。
- 不让 Electron main process 成为插件业务后端。

## 当前阶段与下一刀

当前阶段是 `V2-5 Claw 与 Right Surface / V2-7 release gate`。

已完成的 current 主链包括：唯一 `.codex-plugin/plugin.json` manifest、App Server typed catalog/install/enable store、Plugin Skill/MCP activation、结构化 `plugin://` mention、MCP tool/elicitation、MCP App Right Surface 与 Renderer reload 恢复。2026-08-05 fresh Gate B 证据位于 `.lime/qc/gui-evidence/plugin-v2-current-electron-fixture/`。

下一刀是把 Gate B 从受控 runtime 安装动作推进到真实 App Center 安装点击、Claw `@` picker 点击和卸载后历史可读，补齐 GB-01/07/13；随后补跨进程 cold restore、Windows 安装到卸载证据和剩余 Apps/Hooks/Browser/file surface。旧实现的物理删除仍须先列出精确目标并取得用户确认，旧版高价值文档继续保留为历史资料。

当前路线图完成度：`75%`。核心 macOS runtime/Right Surface 闭环已通过，但尚未达到跨平台 release gate。

## 完成定义

Plugin v2 只有同时满足以下条件才可标记完成：

- `.codex-plugin/plugin.json` 成为唯一插件 manifest。
- bundled、repo、personal、configured、workspace/remote marketplace 可由同一 list projection 表达。
- 安装、启用、授权、可用性和管理员策略是不同字段，不再压成一个布尔值。
- 插件 Skills、MCP servers/apps 与 Hooks 在新 thread/turn 中可追踪地装配。
- App Center 和 Claw 使用同一 plugin identity 与 installed projection。
- MCP/App UI 能进入 Claw Right Surface，且不存在第二套插件业务后端。
- 旧包标准、旧前端 registry 与旧 plugin worker UI 路径已删除并有回流守卫；旧路线图冻结为历史参考，不再形成并行事实源。
- macOS 与 Windows 至少各有一条安装、激活、运行和卸载证据。
- Gate B 证明真实 Electron、preload/IPC、App Server JSON-RPC、runtime/read model 与用户可见状态闭环。

## 参考优先级

1. `/Users/coso/Documents/dev/rust/codex`
2. OpenAI 官方 Plugins / Skills / MCP 文档
3. 本机 `/Applications/ChatGPT.app` 的真实安装包结构
4. Lime current App Server、MCP、Skills、Right Surface owner
5. `internal/roadmap/plugin/` v1 文件仅作为冻结的历史参考，不再是实现依据
