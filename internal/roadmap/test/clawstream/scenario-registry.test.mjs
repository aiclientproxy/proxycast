import fs from "node:fs";
import { describe, expect, it } from "vitest";

const registry = JSON.parse(
  fs.readFileSync(
    "internal/roadmap/test/clawstream/scenario-registry.json",
    "utf8",
  ),
);
const ledgerContent = fs.readFileSync(
  "internal/roadmap/test/clawstream/scenario-ledger.md",
  "utf8",
);

const validStatuses = new Set([
  "missing",
  "guard-needed",
  "partial",
  "partial+guard",
  "covered-electron",
]);
const validPriorities = new Set(["P0", "P1", "P2"]);
const validLayers = new Set(Object.keys(registry.layerDefinitions ?? {}));

function parseLedgerRows(content) {
  const rows = [];
  for (const line of content.split("\n")) {
    if (!line.startsWith("| `")) {
      continue;
    }
    const cells = line
      .split("|")
      .slice(1, -1)
      .map((cell) => cell.trim());
    if (cells.length !== 6) {
      continue;
    }
    const [
      scenarioId,
      codexSource,
      eventItem,
      projectionOracle,
      cleanupTarget,
      status,
    ] = cells;
    const id = scenarioId.match(/^`([^`]+)`$/)?.[1];
    const normalizedStatus = status.match(/^`([^`]+)`$/)?.[1];
    if (!id || !normalizedStatus) {
      continue;
    }
    rows.push({
      id,
      codexSource,
      eventItem,
      projectionOracle,
      cleanupTarget,
      status: normalizedStatus,
    });
  }
  return rows;
}

function countBy(entries, key) {
  return entries.reduce((counts, entry) => {
    const value = entry[key];
    counts[value] = (counts[value] ?? 0) + 1;
    return counts;
  }, {});
}

function expectNonPlaceholderCell(row, key) {
  const value = row[key];
  expect(typeof value, `${row.id}.${key}`).toBe("string");
  expect(value.trim(), `${row.id}.${key}`).toBe(value);
  expect(value.length, `${row.id}.${key}`).toBeGreaterThan(8);
  expect(value, `${row.id}.${key}`).not.toMatch(/\b(?:TBD|TODO|待补|待定)\b/i);
}

describe("clawstream scenario registry", () => {
  it("与 scenario-ledger 保持一行一场景同步", () => {
    const ledgerRows = parseLedgerRows(ledgerContent);
    const registryIds = registry.scenarios.map((scenario) => scenario.id);
    const ledgerIds = ledgerRows.map((row) => row.id);

    expect(new Set(registryIds).size).toBe(registryIds.length);
    expect(registryIds).toEqual(ledgerIds);

    const registryStatusById = new Map(
      registry.scenarios.map((scenario) => [scenario.id, scenario.status]),
    );
    for (const row of ledgerRows) {
      expect(registryStatusById.get(row.id)).toBe(row.status);
    }
  });

  it("scenario-ledger 每行都具备标准化验收骨架", () => {
    const ledgerRows = parseLedgerRows(ledgerContent);
    expect(ledgerRows.length).toBe(registry.scenarios.length);

    for (const row of ledgerRows) {
      expect(validStatuses.has(row.status)).toBe(true);
      expectNonPlaceholderCell(row, "codexSource");
      expectNonPlaceholderCell(row, "eventItem");
      expectNonPlaceholderCell(row, "projectionOracle");
      expectNonPlaceholderCell(row, "cleanupTarget");
    }
  });

  it("每个场景都有可执行的三层护栏骨架", () => {
    expect(registry.schemaVersion).toBe(1);
    expect(registry.status).toBe("current");
    expect(registry.owner).toBe(
      "internal/roadmap/test/clawstream/scenario-ledger.md",
    );
    expect(registry.scenarios.length).toBe(59);

    for (const scenario of registry.scenarios) {
      expect(scenario.id).toMatch(/^[a-z0-9]+(?:-[a-z0-9]+)*$/);
      expect(validPriorities.has(scenario.priority)).toBe(true);
      expect(typeof scenario.family).toBe("string");
      expect(scenario.family.trim().length).toBeGreaterThan(0);
      expect(validStatuses.has(scenario.status)).toBe(true);
      expect(Array.isArray(scenario.requiredLayers)).toBe(true);
      expect(scenario.requiredLayers.length).toBeGreaterThanOrEqual(2);
      expect(typeof scenario.nextDetail).toBe("string");
      expect(scenario.nextDetail.trim().length).toBeGreaterThan(0);

      for (const layer of scenario.requiredLayers) {
        expect(validLayers.has(layer)).toBe(true);
      }

      if (scenario.status === "covered-electron") {
        expect(scenario.requiredLayers).toContain("gui-electron");
      }
      if (scenario.priority === "P0") {
        expect(scenario.requiredLayers).toContain("projection-oracle");
      }
    }
  });

  it("registry 固定骨架字段和治理分类", () => {
    expect(registry.skeletonDefinition).toMatchObject({
      mode: "finish-skeleton-before-detail",
      ledgerColumns: [
        "scenarioId",
        "codexSource",
        "standardEventItem",
        "projectionGuiOracle",
        "cleanupTarget",
        "status",
      ],
      requiredScenarioFields: [
        "id",
        "priority",
        "family",
        "status",
        "requiredLayers",
        "nextDetail",
      ],
      requiredBatchFields: [
        "id",
        "priority",
        "family",
        "evidenceGate",
        "detailOrder",
        "verificationCommands",
      ],
    });

    expect(Object.keys(registry.governanceClassification)).toEqual([
      "current",
      "compat",
      "deprecated",
      "dead",
    ]);
    for (const [classification, entries] of Object.entries(
      registry.governanceClassification,
    )) {
      expect(entries.length, classification).toBeGreaterThan(0);
      for (const entry of entries) {
        expect(typeof entry).toBe("string");
        expect(entry.trim()).toBe(entry);
        expect(entry.length).toBeGreaterThan(0);
      }
    }
  });

  it("先完成全量骨架，再逐个补细节", () => {
    const priorityCounts = countBy(registry.scenarios, "priority");
    const statusCounts = countBy(registry.scenarios, "status");

    expect(priorityCounts).toMatchObject({ P0: 11, P1: 35, P2: 13 });
    expect(statusCounts["missing"] ?? 0).toBe(0);
    expect(statusCounts["covered-electron"]).toBe(23);
    expect(statusCounts["partial"] ?? 0).toBe(0);
    expect(statusCounts["partial+guard"]).toBe(36);
    expect(statusCounts["guard-needed"] ?? 0).toBe(0);
  });

  it("executionBatches 覆盖每个场景且不允许场景游离", () => {
    const scenarioById = new Map(
      registry.scenarios.map((scenario) => [scenario.id, scenario]),
    );
    const seenScenarioIds = [];
    const seenDetailIds = [];

    expect(Array.isArray(registry.executionBatches)).toBe(true);
    expect(registry.executionBatches.length).toBe(8);
    expect(registry.executionBatches.map((batch) => batch.order)).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8,
    ]);

    for (const batch of registry.executionBatches) {
      expect(batch.id).toMatch(/^skeleton-[a-z0-9]+(?:-[a-z0-9]+)*$/);
      expect(validPriorities.has(batch.priority)).toBe(true);
      expect(typeof batch.family).toBe("string");
      expect(batch.family.trim().length).toBeGreaterThan(0);
      expect(batch.mode).toBe("finish-skeleton-before-detail");
      expect(typeof batch.skeletonExit).toBe("string");
      expect(batch.skeletonExit.trim().length).toBeGreaterThan(0);
      expect(batch.evidenceGate).toMatch(/^[a-z0-9]+(?:-[a-z0-9]+)*$/);
      expect(Array.isArray(batch.scenarioIds)).toBe(true);
      expect(batch.scenarioIds.length).toBeGreaterThan(0);
      expect(Array.isArray(batch.detailOrder)).toBe(true);
      expect(batch.detailOrder.length).toBe(batch.scenarioIds.length);
      expect(new Set(batch.detailOrder).size).toBe(batch.detailOrder.length);
      expect(new Set(batch.detailOrder)).toEqual(new Set(batch.scenarioIds));
      expect(Array.isArray(batch.verificationCommands)).toBe(true);
      expect(batch.verificationCommands.length).toBeGreaterThan(0);

      for (const scenarioId of batch.scenarioIds) {
        const scenario = scenarioById.get(scenarioId);
        expect(scenario, `${scenarioId} must exist in scenarios`).toBeTruthy();
        expect(scenario.priority).toBe(batch.priority);
        expect(scenario.family).toBe(batch.family);
        seenScenarioIds.push(scenarioId);
      }

      for (const scenarioId of batch.detailOrder) {
        const scenario = scenarioById.get(scenarioId);
        expect(scenario, `${scenarioId} must exist in scenarios`).toBeTruthy();
        seenDetailIds.push(scenarioId);
      }

      for (const command of batch.verificationCommands) {
        expect(typeof command).toBe("string");
        expect(command.trim()).toBe(command);
        expect(command.length).toBeGreaterThan(0);
        expect(command).not.toContain("agent_runtime_");
      }
    }

    const multiAgentBatch = registry.executionBatches.find(
      (batch) => batch.id === "skeleton-p1-multi-agent-review",
    );
    expect(multiAgentBatch?.verificationCommands).toEqual(
      expect.arrayContaining([
        expect.stringContaining(
          '"lime-rs/crates/app-server/src/runtime/tests/agent_control.rs"',
        ),
        expect.stringContaining(
          '"lime-rs/crates/app-server/src/runtime/canonical_thread_store/agent_graph_tests.rs"',
        ),
        expect.stringContaining('"lime-rs/crates/thread-store/src/agent_graph.rs"'),
      ]),
    );
    expect(
      multiAgentBatch?.verificationCommands.join("\n"),
    ).not.toContain("runtime/tests/evidence_exports/team_facts.rs");
    for (const currentOwnerPath of [
      "lime-rs/crates/app-server/src/runtime/tests/agent_control.rs",
      "lime-rs/crates/app-server/src/runtime/canonical_thread_store/agent_graph_tests.rs",
      "lime-rs/crates/thread-store/src/agent_graph.rs",
    ]) {
      expect(fs.existsSync(currentOwnerPath), currentOwnerPath).toBe(true);
    }

    expect(new Set(seenScenarioIds).size).toBe(seenScenarioIds.length);
    expect(seenScenarioIds).toEqual(
      registry.scenarios.map((scenario) => scenario.id),
    );
    expect(new Set(seenDetailIds).size).toBe(seenDetailIds.length);
    expect(seenDetailIds).toEqual(
      expect.arrayContaining(registry.scenarios.map((scenario) => scenario.id)),
    );
  });
});
