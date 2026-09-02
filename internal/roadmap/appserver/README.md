# App Server 路线图

> 状态：current planning source
> 更新时间：2026-06-08
> Owner：Lime Runtime / App Server / 独立 App 集成

## 1. 定位

本目录定义 App Server 的产品、架构、协议和迁移路线。

App Server 是 Lime AI Agent Runtime 的服务化边界。它不是把现有 `agent_cmd` 原样包一层，而是先把可复用公共部分抽成 host-agnostic `RuntimeCore`，再把 Agent、未来更多执行后端、桌面壳和独立 App 都接到这条公共服务层上。

本路线图参考 `/Users/coso/Documents/dev/rust/codex` 时，只参考其中 `codex-rs` 的 App Server / protocol / client / daemon / Rust runtime 分层：独立 protocol crate、server processor、client facade、transport/lifecycle、`initialize -> initialized` 门禁、`<resource>/<method>` RPC 命名、bounded event / backpressure 和 daemon 生命周期。该仓库是 Codex CLI 版本代码，不包含 Codex App 前端实现；不得把它当作 Codex App UI、Electron shell、托盘、Dock、updater 或桌面产品交互的参考来源。

目标形态：

```text
Lime Desktop
Content Studio
更多独立 App
  -> App Server Client
  -> App Server
  -> Lime RuntimeCore
  -> ExecutionBackend Adapter
  -> Tool / Skill / Workspace / Artifact / Evidence / Policy
```

## 2. 事实源声明

从本路线图起，跨 App 的 Agent 执行能力只允许向下面的事实源收敛：

```text
App Server + JSON-RPC Protocol + RuntimeCore + ExecutionBackend Adapters
```

分类：

| 分类                | 对象                                                                            | 说明                                                                                                                           |
| ------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `current`           | `internal/roadmap/appserver/*`                                                  | App Server 路线图、PRD、架构和协议规划。                                                                                       |
| `current target`    | `RuntimeCore`                                                                   | 跨 App、跨执行后端复用的公共 runtime service 层。                                                                              |
| `current target`    | `ExecutionBackend`                                                              | Agent 和未来更多执行后端的适配接口。                                                                                           |
| `current reference` | `lime-rs/crates/agent`                                                          | 当前最接近 runtime core 的 crate，后续继续拆公共模型与服务。                                                                   |
| `current bridge`    | `src/lib/dev-bridge/safeInvoke.ts`、HTTP client、`app_server_handle_json_lines` | Lime Desktop renderer 进入 Electron Desktop Host / App Server JSON-RPC 的 current 传输、可用性探测和事件监听边界。             |
| `current client`    | Lime Desktop / content-studio / 更多独立 App 的 app-server client               | 独立 App 只通过协议消费 runtime，不直接 import Lime 内部实现。                                                                 |
| `dead / deleted`    | `lime-rs/src/commands/**`                                                       | 已物理删除；只允许负向回流守卫或不可变历史 evidence 引用，不得恢复 stub / compat wrapper。                                   |
| `compat cleanup`    | `src/lib/dev-bridge/commandPolicy.ts` 中的旧命令 truth / no-mock fallback       | 只作为迁移期 fail-closed / 阻塞记录存在；旧命令迁到 current 后必须从 production truth、mock policy、旧 smoke 和 catalog 撤出。 |
| `compat`            | legacy desktop facade                                                           | 迁移期继续服务 Lime Desktop，但只能委托 App Server / Electron Desktop Host current 主链，不继续拥有独立 runtime 事实。         |
| `deprecated`        | 壳层内直接拼接 runtime 业务逻辑                                                 | 只允许迁移和下线，不允许新增能力。                                                                                             |
| `dead`              | 新独立 App 自建完整 Agent runtime                                               | 不再作为 Lime 生态扩展方向。                                                                                                   |

`lime-rs/src/commands/**` 的历史实现只能作为迁移参考，不能再作为新业务逻辑、API adapter、runtime 分支或领域服务的落点。新增 Rust 后端能力必须进入 App Server crates / RuntimeCore / services 等 current 事实源；窗口、托盘、Dock、updater、shell、deep link 等桌面壳能力进入 Electron Desktop Host。若某个能力仍只能通过 `commands/` wrapper 才能跑通，应把它判为 current 主链缺口，而不是继续补旧 wrapper。

`src/lib/dev-bridge/**` 不等同于旧 Rust DevBridge，也不是命令迁移完成后的整体删除目标。App Server current 化时，renderer 仍需要 `safeInvoke` / HTTP client / `app_server_handle_json_lines` 作为传输和诊断边界；后续治理只收缩旧命令 policy、mock fallback、旧 smoke 和 retired guard。目录级退场必须另行证明已有替代 renderer bridge 事实源覆盖这些 current 能力。

后续每个 App Server command group 收口时，都必须把 `src/lib/dev-bridge` 作为治理检查项：保留 current renderer bridge；清出已迁旧命令在 `commandPolicy.ts`、`mockPriorityCommands.ts`、desktop-host mock、旧 smoke 和 catalog 中的 production truth / mock fallback；删不动且跨命令组长期存在的 residual 必须回挂 `internal/exec-plans/tech-debt-tracker.md` 的 `CCD-012` 并写清退出条件，不能只留在聊天、handoff、旧 smoke 备注或单次校验输出里。

## 3. 文档索引

| 文档                                                                                           | 作用                                                                                                                                                                                                         |
| ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [prd.md](./prd.md)                                                                             | 产品目标、用户故事、非目标、验收口径。                                                                                                                                                                       |
| [architecture.md](./architecture.md)                                                           | App Server 总体架构、分层、事实源、服务抽象和部署形态。                                                                                                                                                      |
| [protocol.md](./protocol.md)                                                                   | JSON-RPC 协议、方法命名、对象模型、事件和错误规范。                                                                                                                                                          |
| [service-extraction.md](./service-extraction.md)                                               | 从现有 Lime runtime 抽服务的边界、迁移映射和退场条件。                                                                                                                                                       |
| [consumer-integration.md](./consumer-integration.md)                                           | content-studio 和未来独立 App 如何消费 App Server、TS client、sidecar binary 与 release manifest。                                                                                                           |
| [frontend-electron-migration.md](./frontend-electron-migration.md)                             | Electron Desktop Host current 契约、renderer / bridge / sidecar 边界与验收口径。                                                                                                                             |
| [release-updater.md](./release-updater.md)                                                     | Electron release、签名、公证、updater feed、包资源和下个版本平稳迁移口径。                                                                                                                                   |
| [desktop-platform-product-app-boundary-prd.md](./desktop-platform-product-app-boundary-prd.md) | App Server / Desktop Platform / Product App 的上层边界决策，说明 Desktop Platform 为什么不多余、App Server 不做桌面 Host、Product App 不复制平台能力，并包含架构图、时序图、流程图和 App Server 侧开发任务。 |
| [provider-store-data-root-prd.md](./provider-store-data-root-prd.md)                           | Provider store、API Key 单事实源、App Server 显式数据根、空库初始化和 Desktop Platform / Product App 集成计划。                                                                                              |
| [sequences.md](./sequences.md)                                                                 | 初始化、会话、工具审批、事件流、独立 App 集成等时序图。                                                                                                                                                      |
| [flowcharts.md](./flowcharts.md)                                                               | 从用户输入到 runtime 执行、服务抽取、迁移替换和多 App 复用流程图。                                                                                                                                           |
| [implementation-plan.md](./implementation-plan.md)                                             | 分阶段实施计划、退出条件、验证和治理守卫。                                                                                                                                                                   |

## 4. 主目标

1. 把 Lime 的 AI Agent Rust runtime 抽成可复用本地 app-server。
2. 为 Lime Desktop 和独立 App 提供统一 JSON-RPC 服务边界。
3. 让 content-studio 成为第一批复用方，但不把方案绑定到单一 App。
4. 逐步把 legacy desktop command glue 迁出或下线；确需临时兼容时只能在 App Server / Electron Desktop Host current 边界内投影，不能在 `lime-rs/src/commands/**` 保留 thin facade。
5. 统一 session / thread / turn / task / tool / action / artifact / evidence facts。
6. 让未来 App 只接入 App Server Client 和 Capability SDK，不复制 runtime 逻辑。
7. 让 Agent 作为第一个 `ExecutionBackend` 接入，为后续更多执行后端留出清晰接口。

## 5. 非目标

本路线图不做：

1. 不重写 Lime Desktop 壳层。
2. 不把 App Server 设计成云端多租户服务。
3. 不让独立 App 直接调用 Lime 内部 Rust 模块。
4. 不新增第二套 tool runtime、skill runtime、workspace runtime。
5. 不在 App 侧用 UI-only state 模拟 Agent 执行成功。
6. 不恢复已经物理删除的 `lime-rs/src/commands/**`、stub、compat wrapper 或 thin facade。
7. 不把 `Agent`、`runtime_turn.rs` 或 legacy desktop command DTO 直接定义成公共协议。

## 6. 当前执行顺序

```text
P0 路线图与公共边界冻结
-> P1 codex-rs app-server-style crate 家族骨架
-> P2 RuntimeCore / ExecutionBackend 接口
-> P3 RuntimeBackend adapter
-> P4 App Server 接入 RuntimeCore
-> P5 Lime Desktop Host bridge / compat facade
-> P6 content-studio app-server client
-> P7 Tool / Action / Artifact / Evidence 事件闭环
-> P8 多独立 App 复用和能力发现
-> P9 retired desktop runtime glue 退场审计
```

## 7. 下一刀

下一刀固定为：

**先建立 codex-rs app-server-style crate 家族和公共 runtime 边界，不先把 Agent 直接塞进协议。**

原因：

1. crate 边界先稳定，后续公共代码才不会继续散在壳层。
2. `RuntimeCore / ExecutionBackend` 先定义清楚，Agent 才能作为 adapter 接入，而不是绑死公共层。
3. 协议只暴露 session / turn / event / action / artifact / evidence 等稳定事实，不暴露具体 backend。
4. 完整 tool / artifact / evidence 迁移应跟随 runtime core facts，而不是先做壳层包装。

## 8. 跨项目阅读顺序

涉及 `lime-desktop-platform`、`zhongcao`、App Server 和 Provider/key 时，开发者应按以下顺序读：

1. [desktop-platform-product-app-boundary-prd.md](./desktop-platform-product-app-boundary-prd.md)：先确认四层边界和选定路线。
2. [provider-store-data-root-prd.md](./provider-store-data-root-prd.md)：再确认 Provider/key、App Server DB 和数据根落点。
3. 对应项目的 `internal/roadmap/*`：最后看 Desktop Platform 或 Product App 需要做的具体任务。
