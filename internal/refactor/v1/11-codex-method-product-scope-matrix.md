# V1-01 Codex Method 产品范围矩阵

> status: `inventory-current / implementation-open`
> owner: `app-server-protocol` + 对应 current domain owner
> fixture: `internal/refactor/v1/fixtures/codex-method-product-scope.v0.1.json`
> upstream: `/Users/coso/Documents/dev/rust/codex@4c43465133428898aa84f0bfc02c306ed65fb66a`

## 目标

把 Codex App Server 注册表的全部 method 变成可审计产品裁决，禁止用“已有同类模块”或
legacy 同义命令冒充协议对齐。矩阵覆盖 `clientRequest`、`serverRequest`、
`serverNotification` 与 `clientNotification` 四个方向；每个方向化 method 只能属于：

- `implemented`：Lime generated manifest 存在同方向、同名 current contract，并有 owner/evidence。
- `planned`：能力属于 Lime 产品范围，但 exact method、shape、lifecycle 或证据尚未完成。
- `product-scope-excluded`：Codex/ChatGPT 专属、test-only、internal 或 deprecated surface，禁止复制或恢复兼容层。

## 当前盘点

| 状态                     | 数量 | 裁决                                                                                                                                                                                                                                         |
| ------------------------ | ---: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `implemented`            |   70 | 连接握手、核心 Thread/Turn/Item、thread subscription/lifecycle/content search/raw item injection/background terminals/elicitation/Guardian continuation、typed approval/MCP server request 与 model control plane（含 `model/verification`） |
| `planned`                |  110 | process/fs、config、`model/rerouted`、Hooks、Skills/Plugins/Apps、review 与 Windows sandbox                                                                                                                                                  |
| `product-scope-excluded` |   34 | Codex account/commerce、attestation、remote control、test-only、internal raw response 和 deprecated surface                                                                                                                                  |
| 合计                     |  214 | `130` client request、`11` server request、`72` server notification、`1` client notification                                                                                                                                                 |

`implemented` 只说明 method boundary 已存在并接入 current owner，不代表字段、恢复、GUI 或 Gate B 已全面 parity。
字段和 lifecycle 缺口继续由 Item inventory、gap register 与对应执行切片管理。

## 产品裁决

1. Codex/ChatGPT account、billing、attestation 和 remote-control client administration 不进入 Lime；不建 compat。
2. `applyPatchApproval`、`execCommandApproval`、`thread/rollback`、`item/fileChange/outputDelta`、`thread/compacted` 为 deprecated，禁止恢复。
3. `rawResponse/*` 绕过 canonical Thread/Turn/Item，保持 internal/excluded。
4. `executionProcess/*`、`fileSystem/*`、旧 skill/plugin method 即使功能相似，也不能算 Codex method parity。
5. Realtime、review、process、Windows sandbox 属于产品范围，当前标 `planned`，后续只能在既有 owner 补齐。

## 守卫

`src/lib/governance/codexMethodProductScopeBoundary.test.ts` 固定以下事实：

- 214 个方向化 identity 无遗漏、无重复，状态和方向计数稳定。
- planned 必须写 gap，excluded 必须写 rationale，所有组必须写 owner/evidence/priority。
- `implemented` 必须能在 Lime generated manifest 找到同方向、同名 contract。

上游 Codex revision 变化时，先重跑注册表审计并更新矩阵；不得只改 hash 或计数让守卫通过。

## 下一刀

`thread/inject_items` 已对齐 exact method/shape、Codex current `ResponseItem` validation union、active Turn
session actor delivery、durable provider-only history、Responses exact lowering 与非 Responses fail-closed。
Guardian reviewer producer/lifecycle 和 elicitation provider active-time pause consumer 仍是 runtime lifecycle
blocker，但不影响这两个 method boundary 的 implemented 分类。多模型控制平面的
`model/verification` 已接入可信 Responses metadata producer、Turn 级去重、exact v2 notification、schema 与
generated client；普通 provider fallback 继续只产生 `routing.fallback.applied`。下一刀只补可信
requested/server model mismatch 的 `model/rerouted` producer，再处理 provider adapter/hosted tool 闭环、
Skills/Plugins/Apps watcher/readiness 与 Hook lifecycle。每完成一个
method，必须同步 exact protocol、handler、typed client、fixture/evidence，再将其移入 `implemented`。
Codex 已明确将 `thread/rollback` 标记为即将删除，Lime 不新增该公开方法。当前产品范围完成度为
`70 / 180 = 38.9%`。分母较上一 revision 增加 `1`，来自 Codex HEAD 新增的
`externalAgentConfig/import/recordHistory`；该 P4 config-import method 已进入 `planned`，没有冒充 Lime
现有配置导入能力或恢复 compat。
