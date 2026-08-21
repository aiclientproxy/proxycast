# `internal/tests` 测试文档

本目录放面向维护者和测试 Agent 阅读的测试方法论、运行手册与运营门禁说明。

## 当前入口

- `agent-ops-qc.md`：Agent 运营级测试体系，定义 qcloop、Evidence Pack、测试分层和发布门禁。
- `agent-qc-p0-scenarios.md`：P0 场景执行手册，说明核心场景如何选择、生成 qcloop payload、按 GUI flow 运行、验收和沉淀证据。
- `lime-agent-qc-rollout-plan.md`：Lime 作为样本产品的 Agent 运营级测试落地计划，说明从 partial evidence 到 P0 release gate 的分阶段路径。
- `lime-agent-autonomous-testing-plan.md`：Lime 作为实际 Agent 产品的自主化测试体系示例计划，说明风险地图、Agent 分工、测试组合和分阶段建设路径。
- `lime-agent-autonomous-test-execution-matrix.md`：Lime 自主测试执行矩阵，把 owner gate、P0 场景、证据层和失败回写落成 Agent 可执行步骤。
- `lime-agent-qc-current-blockers.md`：Lime 当前 P0 qcloop 阻断清单，记录真实 fail evidence、root cause 和关闭条件。
- `lime-agent-qc-qcloop-operations.md`：qcloop 批次只读监控、worker preflight、sidecar evidence、stale item 和重跑策略运维手册。
- `lime-agent-qc-stale-owner-intervention.md`：stale GUI qcloop owner 的只读取证、owner 确认和最小处置协议。
- `lime-agent-qc-evidence-contract.md`：qcloop worker / verifier 的结构化 evidence 输出契约，定义 `QCLOOP_WORKER_RESULT` 与 `QCLOOP_EVIDENCE_SUMMARY_JSON`。
- `ai-agent-testing-guide.md`：AI Agent 测试基础理论与评分器说明。

## 与 `internal/test` 的分工

- `internal/tests/`：人读文档、运行手册、运营测试策略。
- `internal/test/`：测试入口索引、机器可读 manifest/schema、harness eval 资产和测试用例模板。

如果文档会被 qcloop、CI 或脚本直接读取，优先放在 `internal/test/`；如果文档主要指导人和 Agent 如何执行测试，优先放在 `internal/tests/`。
