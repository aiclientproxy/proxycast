# Codex 工程模式借鉴（五轴对照版）

> 状态：参考材料（v2，2026-06-11 基于 codex 仓库深扫重写）
> 对比对象：/Users/coso/Documents/dev/rust/codex（codex-rs workspace，125 crates，~97 万行 Rust）
> 用途：为 `v2/IMPLEMENTATION-PLAN.md` 提供可复用的参考实现；为 `directory-architecture-blueprint.md` 提供目标拓扑佐证
> 阅读方式：按 Lime 的轴 A-F 逐轴对照，每轴给出 codex 的做法、证据路径、Lime 的借鉴决策

---

## 轴 A · 协议 Rust→TS 代码生成（codex 已完整解决，直接抄）

### codex 的做法

**单一宏定义整个协议面**。`codex-rs/app-server-protocol/src/protocol/common.rs` 用 `client_request_definitions!` 宏一次性定义 method 名、params、response、序列化作用域：

```rust
client_request_definitions! {
    ThreadStart => "thread/start" {
        params: v2::ThreadStartParams,
        serialization: None,
        response: v2::ThreadStartResponse,
    },
    // 支持 #[experimental("reason")] 标记实验性方法
}
```

宏展开自动派生：`ClientRequest` enum、响应 enum、`method()` / `id()`、序列化逻辑——**新增方法零样板**。

**TS 类型自动生成**。类型结构体同时挂 `JsonSchema`（schemars）+ `TS`（ts-rs v11）derive，`#[ts(export_to = "v2/")]` 指定输出目录：

- 生成命令：`just write-app-server-schema` → `cargo run -p codex-app-server-protocol --bin write_schema_fixtures`
- 生成物：`app-server-protocol/schema/typescript/*.ts`（每类型一文件，头部 `// GENERATED CODE! DO NOT MODIFY BY HAND!`）+ `schema/json/`（JSON Schema）
- 生成时自动过 Prettier（`export.rs`）
- 生成物提交进 repo，TS SDK（`sdk/typescript/`）直接消费

**codex 新增一个 JSON-RPC 方法的写集**：v2 类型文件 1 处 + 宏条目 1 行 + domain processor 实现 + 中心 match 1 个 arm + 跑生成命令。对比 Lime 当前 ~10 个手写触点（含 TS 手抄 2 处），这就是轴 A 的目标态。

### Lime 借鉴决策（R-10 的实现路线修正）

1. **优先评估 ts-rs 路线**：Lime 的 `app-server-protocol` 已有 schemars；ts-rs 与 schemars 可共存于同一结构体。比起"从 JSON Schema bundle 再生成 TS"的两跳方案，ts-rs 直接 derive 一跳到位、类型保真度更高（codex 已验证 v11 可用）。R-10 执行时先做 spike 对比两条路线。
2. **生成物进 repo + 命令固化**：照抄 codex 的 `schema/typescript/` 模式，Lime 落点 `packages/app-server-client/src/generated/`；生成命令进 npm scripts（Lime 无 just）。
3. **codex 的缺口 Lime 要补上**：codex **没有 CI 防漂移检查**（只靠 AGENTS.md 文字提醒"改了就跑生成"）。Lime 的 R-10 必须加 `生成后 git diff --exit-code` 进 CI——这是我们比 codex 多走的半步。
4. **宏定义协议面**（`client_request_definitions!` 的等价物）作为 R-10 的二期：先把 TS 生成打通（消除最大人肉量），再评估用宏收敛 Rust 侧 4 处注册（`v0.rs`/`catalog.rs`/`method_names.rs`/`schema_export/registry.rs`）为 1 处。

---

## 轴 B · JSON-RPC 方法分发组织（codex 半解决，抄长处、避短处）

### codex 的做法

**实现层按 domain 模块化**：`codex-rs/app-server/src/request_processors/` 下约 35 个 domain processor（`thread_processor.rs`、`config_processor.rs`、`git_processor.rs`、`mcp_processor.rs`…），每个是独立 struct、构造时注入依赖（ThreadManager、ConfigManager 等）。

**路由层保持单中心但极薄**：`message_processor.rs` 1431 行，持有各 processor 字段，`handle_initialized_client_request()` 单一 match，每个 arm 只有 3-4 行委托：

```rust
ClientRequest::ThreadStart { params, .. } => self
    .thread_processor.start(params).await
    .map(|response| Some(response.into())),
```

**没有** trait registry / 宏动态注册——显式 match 换来编译期完备性检查和 IDE 可导航性。

### 与 Lime 现状的关键差异

| | codex | Lime 现状 |
|---|---|---|
| 中心路由文件 | message_processor.rs **1431 行**（纯接线） | processor.rs **5041 行** + runtime.rs **8105 行**（接线+实现混在 impl 块里） |
| 方法实现位置 | 35 个 domain processor | 全部堆在两个中心 impl 块 |
| 最大 processor | thread_processor.rs 4268 行（仍超线，codex 自己的债） | — |

**结论性证据**：codex 用"中心 match 1 行接线 + domain processor 承载实现"把路由文件压在 1.4K 行——证明 Lime 的 R-20（按 domain 拆 `runtime/<domain>.rs` + 中心文件收缩为接线）是已被验证的形态，且**不需要发明 trait registry**，显式薄 match 就够。

### Lime 借鉴决策

1. R-20 的目标形态直接对标：`processor.rs` 收缩到 ~1500 行以内的纯 match 接线；实现全部下放 domain 模块。
2. codex 的教训也要吸收：`thread_processor.rs` 自己长到 4268 行——domain processor 本身也要受 Lime 棘轮（R-60）约束，domain 内继续按子职责拆。
3. codex 用 `inventory` crate 做实验性 API 的编译期自注册（`experimental_api.rs`）——Lime 暂不需要，登记为 R-20 后期可选项（若 domain 数量失控再启用）。

---

## 轴 C/F · 大文件与分层治理（codex 只有文档约束，Lime 要做得更硬）

### codex 的规则（AGENTS.md 原文要点）

- Rust 模块目标 **500 LoC**（不含测试）；超过 ~800 LoC 后新功能必须开新模块。
- **高频文件显式点名**：`tui/src/app.rs`、`bottom_pane/chat_composer.rs`、`chatwidget.rs` 等被写进规则，明确"禁止往这些文件加独立方法，保持 orchestration 职责"。
- 拆分时**测试和文档随代码迁移**。

### codex 的执行现实（实测）

- 超 1000 行文件约 **40 个**；最大 `chat_composer.rs` **11188 行**——规则点名保护的文件恰恰是最大的。
- `clippy.toml` / justfile / CI **均无文件体量机械约束**，全靠 code review 人肉执行。

**结论**：codex 证明了"只写规则不上守卫，热点文件照样失控"。这正是 Lime R-60 棘轮（baseline + CI 红线）的价值依据——Lime 233 个超线文件比 codex 的 40 个严重 6 倍，更不能只靠文档。

### Lime 借鉴决策

1. ✅ 抄"高频文件显式点名"：棘轮 baseline 的 `comment` 字段标注 `high-touch, split first`（`AgentChatWorkspace.tsx`、`runtime.rs`、`processor.rs`、`useWorkspaceSendActions.ts`）。
2. ✅ 抄"测试随代码迁移"（已在 R-20/R-32 执行清单）。
3. ✅ 超越 codex：R-60 机械棘轮 + R-30 import 方向 lint，codex 两样都没有。

---

## 轴 E · core 抗膨胀与 crate 粒度（codex 的结构性手段值得抄）

### codex 的做法

- **显式规则**（AGENTS.md）："resist adding code to codex-core!"——新功能先问有没有现成 crate，否则开新 crate，不进 core。
- **结构性拆分**：125 个 crate，其中关键模式是**协议/契约小 crate**：
  - `codex-protocol`（纯协议类型，不依赖业务）
  - `codex-app-server-protocol`（JSON-RPC 定义 + TS 生成）
  - `codex-tools`（ToolExecutor trait 等工具抽象）
  - `ext/extension-api`（扩展 API，解耦扩展与 core）
  - `codex-utils-*` 系列微型工具 crate
- **命名规范**：全部 `codex-` 前缀。

### Lime 借鉴决策

1. ✅ R-50 的 AGENTS.md 抗膨胀条款照抄精神："新增 Rust 逻辑禁止默认落 `lime-core` / `services` 平铺层"。
2. ✅ "契约小 crate"模式 Lime 已有雏形（`app-server-protocol`），方向正确；`lime-core` 里的 models/config 类型未来可参照 `codex-protocol` 拆纯类型 crate（进中长期 T2，见蓝图）。
3. ⏸️ crate 前缀：Lime 不发布 crates.io，暂不强制；若未来开放插件 SDK 再统一（蓝图 T3）。
4. ⚠️ 度量参照：codex 125 crate / 97 万行 ≈ 平均每 crate 8K 行；Lime 30 crate 承载约 40 万行，单 crate 明显过重——T2 拆 crate 的方向有据可依，但不急于近期。

---

## 顶层拓扑 · 一个后端多客户端壳（Lime 未来结构的直接参照）

codex 的顶层结构：

```
codex/
├── codex-rs/        # Rust workspace（后端 + TUI，125 crates）
│   ├── app-server/  app-server-protocol/  app-server-client/
│   ├── app-server-transport/  app-server-daemon/
│   ├── core/  protocol/  tools/  ext/*
│   └── tui/ cli/
├── sdk/typescript/  sdk/python/   # 多语言 SDK，消费生成的协议类型
├── codex-cli/                     # npm 分发壳
├── docs/  scripts/
```

所有客户端（CLI / TUI / IDE 扩展 / SDK）面对**同一个 app-server JSON-RPC 协议**，协议类型从 Rust 单向生成。这与 Lime 的 current 主线（Electron 只是 Desktop Host bridge，能力收敛 App Server）完全同构——Lime 的 `electron/` 对应 codex 的一个"客户端壳"，`src/`（React）是壳里的 renderer。未来若加第二个壳（Web/CLI），协议层不需要任何返工。这是蓝图 § 北极星的依据。

---

## 总结：抄什么、避什么

**直接抄的三个机制**：
1. ts-rs + derive 的协议 TS 生成链（→ R-10）：`app-server-protocol/src/protocol/common.rs` + `export.rs` + `schema/typescript/`。
2. domain processor + 薄中心 match（→ R-20）：`request_processors/` 35 模块 + 1431 行 message_processor。
3. 协议/契约小 crate 化 + resist-adding-to-core（→ R-50 / T2）。

**codex 没解决、Lime 要补的两件事**：
1. 协议生成无 CI 防漂移 → Lime R-10 加 `git diff --exit-code`。
2. 文件体量规则无机械守卫（chat_composer 11K 行）→ Lime R-60 棘轮。
