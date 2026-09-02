# Lime 未来代码结构指引（北极星蓝图）

> 状态：proposed（v3，2026-06-11；应用户要求以**目标态**为主体重写，恢复中长期 T1/T2/T3 规划）
> 上位规则：`AGENTS.md`、`internal/refactor/README.md`
> 拓扑参照：codex 的"一个 App Server + 多客户端壳"已验证形态（`codex-engineering-patterns.md` § 顶层拓扑）
> 定位：本文件描述**代码未来应该长什么样、新代码应该落在哪**；现状证据只在 `architecture-debt-analysis.md`，执行排期一律走 `v2/IMPLEMENTATION-PLAN.md`

---

## § 0 北极星：整体拓扑

Lime 的终态是一个**协议驱动的单后端、多客户端壳**产品：

```
                    ┌──────────────────────────────────┐
                    │   app-server (JSON-RPC, Rust)     │  ← 唯一后端事实源
                    │   协议类型单向生成所有客户端绑定     │
                    └──────────────────────────────────┘
                          ↑                ↑
              ┌───────────┴───┐    ┌──────┴────────────┐
              │ Electron Host │    │  未来壳（CLI/Web/   │
              │ (Desktop 壳)  │    │  插件 SDK，T3）     │
              └───────┬───────┘    └───────────────────┘
                      │ IPC bridge（薄）
              ┌───────┴───────┐
              │ src/ renderer │  ← React，按领域分层
              └───────────────┘
```

三条全局公理（一切结构决策从这里推导）：

1. **协议是生成物，不是手稿**——Rust 定义一次，TS 绑定生成；任何客户端壳不得手抄协议。
2. **中心文件只接线，不实现**——路由/dispatch/注册文件的体量上限是"每方法 1-3 行"；实现必须住在 domain 模块里。
3. **依赖方向单向且被 lint 锁死**——壳 → 领域 → 平台 → 协议/契约，禁止回流。

---

## § 1 前端（src/）未来结构

### 1.1 目标结构（T1 终态）

```
src/
├── domains/                  # 领域全栈：一个领域的 UI + 状态 + 编排住在一起
│   ├── agent/
│   │   ├── ui/               # React 组件（只渲染和事件接线）
│   │   ├── viewmodel/        # 状态机 / projection / selector（纯逻辑，*.unit.test.ts）
│   │   ├── commands/         # 输入命令解析器（纯函数）
│   │   └── index.ts          # 领域公开 API，外部只许 import 这里
│   ├── workspace/            # design / image / video / document 画布工作区
│   ├── knowledge/
│   ├── skills/
│   ├── settings/
│   └── voice/
├── platform/                 # 跨领域基础设施（不含任何业务语义）
│   ├── api/                  # 后端调用唯一网关（现 lib/api 收敛后）
│   │   └── generated 协议类型来自 @limecloud/app-server-client
│   ├── desktop-host/         # Electron bridge 封装（dev-bridge 收敛为其内部传输细节）
│   ├── i18n/  auth/  navigation/
├── shared/                   # 无业务语义的通用件
│   ├── ui/                   # 通用 UI kit（现 components/ui）
│   ├── hooks/  utils/  types/
├── pages/                    # 窗口/页面入口（只做组装）
├── App.tsx / main.tsx
```

### 1.2 分层与依赖铁律（R-30 lint 落地的就是这张图）

```
pages → domains/*/ui → domains/*/viewmodel → platform/api → @limecloud/app-server-client
  │          │                  │                │
  └──────────┴──────────────────┴────────────────┴──→ shared/*（任何层可用）
```

1. `platform/**`、`shared/**` 禁止 import `domains/**`、`pages/**`。
2. 领域之间禁止互相 import 内部文件，只许走对方 `index.ts` 公开 API。
3. 业务代码禁止 import `desktop-host` 传输细节（dev-bridge 等），后端调用唯一入口 `platform/api`。
4. `ui/` 内禁止业务状态机——状态、解析、格式化一律住 `viewmodel/`（即 AGENTS.md 硬规则 9 的物理化）。

### 1.3 新前端代码落点决策表

| 我要加… | 落点 |
|---|---|
| 一个领域的新界面/交互 | `domains/<x>/ui/` + 状态进 `viewmodel/` |
| 跨领域复用的纯 UI 组件 | `shared/ui/` |
| 一个新后端能力的前端调用 | `platform/api/<domain>.ts`（类型来自生成物） |
| 输入命令/文本解析逻辑 | `domains/<x>/commands/`（纯函数 + 单测） |
| 新窗口/页面 | `pages/`，内部只组装 domains |
| **任何情况下都不许** | 往 7000 行组件里加 useState；在 lib 里 import 组件；手写协议类型 |

### 1.4 现名→目标名映射（迁移时查表）

| 现在 | 未来 |
|---|---|
| `components/agent/chat/`（UI 部分） | `domains/agent/ui/` |
| `components/agent/chat/`（状态/hooks/解析） | `domains/agent/viewmodel/` + `commands/` |
| `features/plugin/runtime/` | `domains/agent/`（与上合并）或独立 `domains/plugin/` |
| `components/workspace/` + `components/image-gen/` 等 | `domains/workspace/` |
| `lib/api/` | `platform/api/` |
| `lib/dev-bridge/` | `platform/desktop-host/` 内部传输细节 |
| `components/ui/` | `shared/ui/` |
| `hooks/` `contexts/` `stores/` | 业务的下沉对应 domain `viewmodel/`，通用的进 `shared/hooks/` |

---

## § 2 Rust（lime-rs/）未来结构

### 2.1 目标结构（T1 终态：crate 边界不动，crate 内成形）

```
lime-rs/crates/
├── app-server-protocol/          # 协议唯一事实源
│   ├── src/protocol/v0/<domain>.rs   # 每个 domain 一个协议模块
│   ├── src/export/                   # schemars + ts-rs 导出（R-10）
│   └── （二期）宏定义协议面，4 处注册收敛为 1 处
├── app-server/
│   ├── src/processor.rs          # ≤1500 行：纯 match 接线（codex message_processor 形态）
│   ├── src/processor/<domain>.rs # domain handler
│   ├── src/runtime.rs            # 收缩：RuntimeCore 结构体 + 接线
│   ├── src/runtime/<domain>.rs   # domain 实现 + 自己的 #[cfg(test)]
│   └── src/local_data_source/    # 既有先例，保持
├── services/                     # 按 lib.rs 已有四类落成物理分组（T2）
├── core/                         # 只减不增（R-50 抗膨胀），T2 拆纯类型 crate
└── agent-rust/                   # 自有化 fork，独立子 workspace，内部同样 domain 子模块化
```

**核心不变量**：`protocol/v0/<domain>.rs` ↔ `processor/<domain>.rs` ↔ `runtime/<domain>.rs` ↔ 生成的 TS `<domain>` 类型，**四层共用同一套 domain 切分**，加一个能力时四层各落一个同名位置。

### 2.2 新 Rust 代码落点决策表

| 我要加… | 落点 |
|---|---|
| 新 JSON-RPC 方法 | `protocol/v0/<domain>.rs` 类型 + `runtime/<domain>.rs` 实现 + processor 1 行接线 + 跑 TS 生成 |
| 新业务服务 | 独立 `services/<domain>_service.rs`（T2 后进对应分组目录）；先自问"为什么不是独立模块" |
| 新共享类型 | 优先 `app-server-protocol`（若属协议）；**不默认进 lime-core** |
| 桌面壳能力（窗口/托盘/updater） | `electron/`（Desktop Host），不进 Rust |
| **任何情况下都不许** | 往 runtime.rs/processor.rs 的中心 impl 块直接加方法体；在 core 里堆新模型类型 |

---

## § 3 packages/ 未来结构

```
packages/
├── app-server-client/        # 协议生成物 + 薄客户端（R-10 后核心为 generated）
├── agent-ui-contracts/       # UI 类型契约（current）
├── agent-runtime-projection/ # 投影/selector 样板（current）
├── lime-cli-npm/             # 外发 CLI
└── (T3) plugin-sdk/          # 插件生态打开后新增
```

收缩动作：`plugin-runtime`（零引用）下线；`agent-runtime-ui`（1 处引用）并回 src；`agent-runtime-client` 观察后决定并入 `app-server-client` 或保留。

---

## § 4 中长期规划（T1 → T2 → T3）

### T1（1-3 个月）：机制成形——对应执行计划轴 A/B/C/D/F

- 协议 TS 生成上线（R-10），processor/runtime domain 化开跑（R-20），前端 import 方向 lint + 状态分层样板成立（R-30/31/32），网关收敛（R-40），棘轮上线（R-60）。
- **T1 不做目录大搬家**：`domains/` 物理目录可以等——R-30 的 lint 规则按现有路径先把方向锁死，R-32 抽出的 viewmodel/commands 直接按 § 1.1 形态新建，新代码先长成目标形状。
- **进度（2026-06-17）**：R-10/R-20/R-30/R-40/R-60 已落地，§ 6 前五项达标或接近；T1 剩余主项为 R-32（`AgentChatWorkspace` 拆分）与 R-21（agent `agent.rs`）。
- 退出标志：§ 6 验收指标前四项达标。

### T2（3-9 个月）：结构归位——存量迁移到目标结构

- **前端**：`components/agent` + `features/plugin` 归并为 `domains/agent/`，随后 workspace、knowledge、settings 逐领域迁移（§ 1.4 映射表），每领域一轮、commit 可回滚；`lib/` 拆解为 `platform/` + `shared/`。
- **Rust**：`services/` 按四类落物理分组；`lime-core` 拆"纯类型 crate"（参照 codex-protocol 模式），core 收缩为最小共享层；协议宏（`client_request_definitions!` 等价物）把 protocol crate 4 处注册收敛为 1 处。
- **packages**：完成 § 3 收缩。
- **internal/**：`test`/`testing`/`tests` 三目录归并、`bussniss` typo 修正、16 个一级目录收到 ~10 个。
- **scripts/**：存量按 `governance/`、`smoke/` 等 domain 归类，根目录收缩到 README + 极少数公开入口。
- 退出标志：`src/` 一级目录 = `domains/platform/shared/pages` 四件套；新贡献者按落点决策表不会放错位置。

### T3（9 个月+，按产品路线触发，不自动开工）

- **多壳扩展**：若出现第二客户端形态（Web / CLI / 远程），顶层向 codex 拓扑靠拢：`apps/desktop/`（现 src+electron）、`apps/<new>/`，协议生成物天然复用，零协议返工。
- **插件生态**：开放第三方插件时新增 `packages/plugin-sdk/`（runtime-api / ui-components / types）；届时 Rust crate 统一 `lime-` 前缀并评估发布。
- **触发条件**：写进路线图的产品决策，不由重构主线自行启动。

---

## § 5 防回退机制（结构靠守卫维持，不靠自觉）

| 公理/边界 | 守卫 |
|---|---|
| 协议不手抄 | 生成后 `git diff --exit-code` 进 CI（R-10） |
| 依赖方向单向 | ESLint import 边界 + 违例 baseline 只减不增（R-30） |
| 中心文件只接线 | 体量棘轮冻结 runtime.rs/processor.rs（R-60），行数只许降 |
| 领域只暴露 index.ts | lint 禁 deep import（T2 迁移时启用） |
| core 不膨胀 | AGENTS.md 抗膨胀条款 + review 推回（R-50） |

### § 6 验收指标

> 「当前」列 2026-06-18 复核刷新；括注为 2026-06-11 起点值，便于看偿还幅度。

| 指标 | 当前（2026-06-18） | T1 | T2 |
|---|---|---|---|
| 新增 1 个 JSON-RPC 方法的手写触点 | ~6（TS 类型已生成，起点 ~10） | ≤5 | ≤4（协议宏后） |
| TS 协议手写行数 | ≈0（已转 @generated re-export，起点 ~3600） | ≈0 | ≈0 |
| lib→components/features 依赖违例 | 0 新增（baseline 冻结 47 处，起点 6+ 无守卫） | 0 新增 | 0（含存量） |
| 业务直连 dev-bridge | baseline 冻结 11 处，0 新增（起点多处无守卫） | 0 | 0 |
| runtime.rs + processor.rs 合计 | ~3000 行接线（588 + 2444，起点 ~13000） | 单调下降 | ≤3000 行（纯接线）✅ 已达成 |
| `src/` 一级目录 | 11 个（未动） | 不变 | 4 个（domains/platform/shared/pages） |
| 超 800 行冻结文件 | 前端 152 + Rust 161（起点 322；`governance:file-size` 已恢复绿） | 只减不增 | <60（与 codex 同量级） |
