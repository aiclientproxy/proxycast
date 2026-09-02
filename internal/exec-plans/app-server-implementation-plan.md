# App Server 主链执行计划

> 状态：current / 持续收敛；更新时间：2026-09-02；路线图：`internal/roadmap/appserver/README.md`；架构事实源：`internal/aiprompts/architecture.md`、`internal/aiprompts/commands.md`

## 1. 目标

Desktop 业务能力统一通过以下主链运行：

```text
Renderer typed gateway
  -> Electron Desktop Host JSONL transport
  -> App Server JSON-RPC
  -> RuntimeCore / domain owner
  -> Thread / Turn / Item read model
  -> Renderer projection
```

Electron 只拥有桌面宿主能力和 sidecar 生命周期，不承接第二套业务后端。

## 2. Current Owner

| 边界                                      | Owner                                           |
| ----------------------------------------- | ----------------------------------------------- |
| JSON-RPC method、DTO、schema              | `app-server-protocol`                           |
| request routing、notification、read model | `app-server`                                    |
| typed client/transport                    | `app-server-client`、`app-server-transport`     |
| daemon 与进程生命周期                     | `app-server-daemon`、Electron Desktop Host      |
| Agent 回合与 canonical projection         | `runtime-core`、`agent-runtime`、`thread-store` |
| provider request/lowering                 | `model-provider`                                |
| tool/permission/execution                 | `tool-runtime`                                  |

## 3. 收敛规则

1. 新业务 method 必须进入 App Server protocol 和对应领域 owner。
2. Electron IPC 只用于窗口、文件选择、系统权限、通知、native view、更新和 sidecar。
3. Renderer 不直连 provider、数据库、Rust 私有 service 或生产 mock。
4. 旧命令没有外部兼容负担时，迁移调用后直接删除，不新增 wrapper。
5. App Server 错误必须结构化并 fail closed；协议或 capability 不明确时不得猜测成功。
6. Thread/Turn/Item identity 在 request、runtime event、read model 和 GUI 中保持一致。
7. schema、typed client、catalog、fixture、文档和测试必须在同一变更集同步。

## 4. 当前工作项

- 持续把遗留 Desktop 业务 facade 迁入 App Server current method。
- 收敛 daemon/sidecar readiness、operation lock、shutdown 和恢复行为。
- 让所有 GUI 主路径消费 canonical read model，删除 Renderer 派生真相。
- 补齐 hosted connector、跨平台通知与 packaged Windows 的真实 readiness 证据。
- 删除已脱离构建图的命令、mock、脚本、文档和 catalog alias，并用治理守卫防回流。

## 5. 退出条件

单个迁移项完成必须满足：

1. current method 已在 protocol、handler、domain owner 和 typed client 落地。
2. 所有生产消费者已迁移，旧入口已物理删除。
3. notification/read model 与 canonical identity 有稳定断言。
4. 生产 mock fallback 命中为零。
5. contract、领域测试和风险对应的 GUI/Gate B 证据通过。

## 6. 验证

```bash
npm run test:contracts
npm run test:rust:related -- lime-rs/crates/app-server lime-rs/crates/app-server-protocol
npm run smoke:agent-runtime-current-fixture
npm run verify:gui-smoke
npm run governance:legacy-report
```

完整本地门禁使用 `npm run verify:local`。若改动触及版本、Forge 或 workspace manifest，追加 `npm run verify:app-version`。

## 7. 分类

- `current`：App Server JSON-RPC、RuntimeCore、领域 crates、typed client、canonical projection。
- `compat`：只有真实外部协议或数据迁移约束时允许，且必须记录退出条件。
- `deprecated`：仅允许迁出，不新增逻辑。
- `dead`：已删除 Desktop 业务命令、旧应用编排 runtime、生产 mock fallback 和重复 read model；禁止恢复。
