# Lime 测试体系

> status: current / Refactor v2
> owner: quality-workflow
> last_verified: 2026-07-17

## 1. 事实源

Lime 当前是 `Electron Desktop Host + App Server JSON-RPC` 桌面产品；RuntimeCore、Thread/Turn/Item projection 和 GUI 继续沿这条链收敛。

Lime 的唯一产品链是：

```text
Electron Desktop Host
  -> App Server JSON-RPC
  -> RuntimeCore
  -> Thread / Turn / Item projection
  -> GUI
```

仓库级最低门禁以 [../aiprompts/quality-workflow.md](../aiprompts/quality-workflow.md) 为准；完整策略以 [testing-strategy-2026.md](testing-strategy-2026.md) 为准；第二期覆盖建设以 [../roadmap/benchmark/README.md](../roadmap/benchmark/README.md) 为准。

## 2. 文档导航

| 文档                                                                               | 作用                                                            |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [testing-strategy-2026.md](testing-strategy-2026.md)                               | 分层、测试作者合同、CI/release lane、旧测试迁移规则             |
| [unit-tests.md](unit-tests.md)                                                     | 纯单元和组件测试边界                                            |
| [integration-tests.md](integration-tests.md)                                       | Rust domain 与 App Server public JSON-RPC 集成测试              |
| [e2e-tests.md](e2e-tests.md)                                                       | Gate A、Gate B、Vitest e2e 和 Electron smoke 的关系             |
| [agent-evaluation.md](agent-evaluation.md)                                         | live provider、grader、pass@k/pass^k 和非确定性评估             |
| [harness-evals.md](harness-evals.md)                                               | 现有 replay/harness 资产；第二期迁移输入，不是默认 release gate |
| [../aiprompts/playwright-e2e.md](../aiprompts/playwright-e2e.md)                   | 真实 GUI 续测与 Playwright 操作细则                             |
| [../roadmap/benchmark/scenario-matrix.md](../roadmap/benchmark/scenario-matrix.md) | 第二期稳定场景 ID 与证据等级                                    |

机器可读资产：

- `agent-qc-scenarios.manifest.json`、`agent-qc-gui-flows.manifest.json`、`agent-qc-evidence.schema.json`：现有 Agent QC 输入，第二期 T0 逐项审计前不自动算新门禁。
- `harness-evals.manifest.json`、`harness-fixtures/**`：replay/eval 输入，只能证明 manifest 声明的场景。
- `deepswe-coding-slice-v2.json`：DeepSWE Smoke 10 / Release 20 的版本化选题与 adapter v6 执行合同；当前已有真实 request tool catalog、runtime step/token cap、generation controls、批量 plan/aggregate 和诊断 true run。Agnes 正式题预算内仍无 candidate；Pier `0.3.1` CLI 已固定但本机无容器运行时，Verifier 仍阻塞。

旧 `benchmark-release` 和 `agent-qc-benchmark` manifest 已删除，禁止恢复同名入口。

## 3. 常用命令

### 快速反馈

```bash
npm run test:related -- <paths...>
npm run test:changed -- <ref>
npm run test:unit -- <paths...>
npm run test:component -- <paths...>
npm run test:contract -- <paths...>
npm run test:integration -- <paths...>
npm run test:resume
```

`npm run test:e2e` 只运行 Vitest e2e layer，不等于真实 Electron Gate B。

### Rust

```bash
npm run test:rust:related -- <paths...>
npm run test:rust:integration:related -- <paths...>
npm run test:rust:unit -- -p <crate> <filter>
npm run test:rust:integration -- -p <crate> --test <target>
```

### Contracts 与主路径

```bash
npm run test:contracts
npm run smoke:electron
npm run smoke:agent-runtime-current-fixture
npm run smoke:mcp-current
npm run verify:gui-smoke
npm run governance:legacy-report
```

### 聚合门禁

```bash
npm run verify:local
npm run verify:local:full
```

聚合命令不能替代受影响场景的公共边界集成测试或 Gate B。

## 4. 分层

| 层              | 适合验证                                                     | 不适合验证                  |
| --------------- | ------------------------------------------------------------ | --------------------------- |
| unit            | parser、selector、projection、lowering、state transition     | React/进程/网络/文件系统    |
| component       | DOM 渲染、事件、hook 生命周期、关键接线                      | Agent 状态机和跨进程主链    |
| contract        | schema、typed client、command/catalog/preload 边界           | runtime 行为正确性          |
| integration     | domain owner、App Server public API、store、provider fixture | Electron 可见状态           |
| current fixture | RuntimeCore/tool/provider/event/read model                   | live provider，除非显式使用 |
| Gate A          | Renderer projection 与交互                                   | Electron/preload/IPC        |
| Gate B          | 真实 Electron 产品链                                         | 未运行的 provider/平台      |
| live/eval       | 模型能力和非确定性稳定性                                     | 确定性协议正确性            |

## 5. Current / Deprecated / Dead

### current

- related-first 的 TS/Rust layer runners。
- `packages/app-server-client` 的 protocol/client contract 和 public JSON-RPC integration。
- `src/lib/desktop-host/` 的显式 host fixture；生产路径不得使用其 mock 作为 fallback。
- `smoke:agent-runtime-current-fixture`、`smoke:mcp-current` 等 current fixture。
- Gate A Renderer 证据与真实 Electron Gate B fixture。
- `verify:local`、`test:contracts`、`verify:gui-smoke`、governance guards。

### deprecated

- 未映射到第二期稳定场景 ID 的旧 Agent QC/Harness manifest。
- 仍基于旧 owner、旧命名或过大 React mount 的测试；只能迁移或合并。
- 只靠 source-string 搜索证明行为的 boundary tests；优先收进治理扫描或 owner contract test。

### dead

- 旧 Benchmark runner、release manifest 和旧 Terminal-Bench/DeepSWE wrapper；DeepSWE v2 选题仍是 current evaluation 输入。
- 测试静态定义、已删除 runtime/command 正向行为和脱离构建图的 fixture。
- 生产 mock fallback、renderer fallback 或旧 wrapper 通过的“主路径”测试。

## 6. 新增测试前

1. 在 [../roadmap/benchmark/scenario-matrix.md](../roadmap/benchmark/scenario-matrix.md) 找到或新增场景 ID。
2. 确定唯一 owner 和最低证据层，不从 UI 层重复测试后端所有分支。
3. 优先复用已有 builder、fixture、临时目录和事件等待 helper。
4. 正常与失败路径使用同一 public boundary。
5. 断言完整 request/event/read model；避免只验证 mock 被调用。
6. 写明定向命令、运行时间、live/secret/platform 约束。

## 7. Evidence 最小字段

```yaml
scenario: TRN-01
candidate: <commit-or-working-tree-digest>
proof_level: integration|fixture|gate-a|gate-b|live|platform
command: <stable repository command>
status: pass|fail|blocked
duration_ms: <number>
artifacts: <paths>
failure_owner: <domain-or-none>
mock_mode: none|explicit-test-fixture
```

失败 evidence 不覆盖；修复后新增记录。live evidence 不保存 secret、完整用户正文或敏感本地路径。
