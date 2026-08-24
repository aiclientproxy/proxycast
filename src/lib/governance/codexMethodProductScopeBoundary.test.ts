import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

const REPO_ROOT = process.cwd();
const MATRIX_PATH =
  "internal/refactor/v1/fixtures/codex-method-product-scope.v0.1.json";
const MANIFEST_PATH =
  "lime-rs/crates/app-server-protocol/schema/json/manifest.json";

type Direction =
  | "clientRequest"
  | "serverRequest"
  | "serverNotification"
  | "clientNotification";
type Status = "implemented" | "product-scope-excluded" | "planned";

interface MethodGroup {
  id: string;
  direction: Direction;
  productArea: string;
  status: Status;
  priority: "P0" | "P1" | "P2" | "P3" | "P4" | "none";
  owner: string;
  counterpartRule: "same-name" | "none";
  evidence: string[];
  gap?: string;
  rationale?: string;
  methods: string[];
}

interface Matrix {
  allowedStatuses: Status[];
  inventory: {
    total: number;
    byDirection: Record<Direction, number>;
    byStatus: Record<Status, number>;
  };
  groups: MethodGroup[];
}

interface Manifest {
  methods: Array<{
    kind: "request" | "serverRequest" | "notification";
    method: string;
  }>;
}

function readJson<T>(relativePath: string): T {
  return JSON.parse(readFileSync(join(REPO_ROOT, relativePath), "utf8")) as T;
}

function flatten(matrix: Matrix) {
  return matrix.groups.flatMap((group) =>
    group.methods.map((method) => ({ ...group, method })),
  );
}

function manifestKind(direction: Direction) {
  if (direction === "clientRequest") return "request";
  if (direction === "serverRequest") return "serverRequest";
  return "notification";
}

describe("Codex method product scope boundary", () => {
  it("221 个上游方法必须且只能落入一个三态分类", () => {
    const matrix = readJson<Matrix>(MATRIX_PATH);
    const entries = flatten(matrix);
    const identities = entries.map(
      ({ direction, method }) => `${direction}:${method}`,
    );

    expect(matrix.allowedStatuses).toEqual([
      "implemented",
      "product-scope-excluded",
      "planned",
    ]);
    expect(entries).toHaveLength(matrix.inventory.total);
    expect(new Set(identities).size).toBe(identities.length);

    for (const [direction, expected] of Object.entries(
      matrix.inventory.byDirection,
    ) as Array<[Direction, number]>) {
      expect(
        entries.filter((entry) => entry.direction === direction),
      ).toHaveLength(expected);
    }
    for (const [status, expected] of Object.entries(
      matrix.inventory.byStatus,
    ) as Array<[Status, number]>) {
      expect(entries.filter((entry) => entry.status === status)).toHaveLength(
        expected,
      );
    }

    expect(
      createHash("sha256")
        .update(identities.toSorted().join("\n"))
        .digest("hex"),
    ).toBe("a91f5bdaedc5382cd957b82b44824922c7594aa16b33293916b6f48af4d8f899");
  });

  it("每组必须声明 owner、证据、优先级以及 planned/excluded 原因", () => {
    const matrix = readJson<Matrix>(MATRIX_PATH);
    const groupIds = matrix.groups.map(({ id }) => id);

    expect(new Set(groupIds).size).toBe(groupIds.length);

    for (const group of matrix.groups) {
      expect(group.id.length).toBeGreaterThan(0);
      expect(group.productArea.length).toBeGreaterThan(0);
      expect(matrix.allowedStatuses).toContain(group.status);
      expect(group.owner.length).toBeGreaterThan(0);
      expect(group.evidence.length).toBeGreaterThan(0);
      expect(group.methods.length).toBeGreaterThan(0);
      for (const evidence of group.evidence) {
        expect(
          existsSync(join(REPO_ROOT, evidence)),
          `${group.id} 引用了不存在的 evidence: ${evidence}`,
        ).toBe(true);
      }

      if (group.status === "implemented") {
        expect(group.counterpartRule).toBe("same-name");
        expect(group.priority).not.toBe("none");
      } else if (group.status === "planned") {
        expect(group.counterpartRule).toBe("none");
        expect(group.gap?.length ?? 0).toBeGreaterThan(0);
        expect(group.priority).not.toBe("none");
      } else {
        expect(group.counterpartRule).toBe("none");
        expect(group.rationale?.length ?? 0).toBeGreaterThan(0);
        expect(group.priority).toBe("none");
      }
    }
  });

  it("implemented 只能引用 Lime manifest 中同方向的真实契约", () => {
    const matrix = readJson<Matrix>(MATRIX_PATH);
    const manifest = readJson<Manifest>(MANIFEST_PATH);
    const contracts = new Set(
      manifest.methods.map(({ kind, method }) => `${kind}:${method}`),
    );

    for (const entry of flatten(matrix).filter(
      ({ status }) => status === "implemented",
    )) {
      expect(
        contracts.has(`${manifestKind(entry.direction)}:${entry.method}`),
        `${entry.direction}:${entry.method} 缺少 Lime generated manifest 契约`,
      ).toBe(true);
    }
  });

  it("已删除的 compaction 旧方法不得回到 Lime manifest", () => {
    const manifest = readJson<Manifest>(MANIFEST_PATH);
    const methods = manifest.methods.map(({ method }) => method);

    expect(methods).toContain("thread/compact/start");
    expect(methods).not.toContain("agentSession/compact");
  });

  it("Codex realtime surface must stay outside the Desktop manifest", () => {
    const matrix = readJson<Matrix>(MATRIX_PATH);
    const manifest = readJson<Manifest>(MANIFEST_PATH);
    const realtimeEntries = flatten(matrix).filter(({ method }) =>
      method.startsWith("thread/realtime/"),
    );
    const contracts = new Set(
      manifest.methods.map(({ kind, method }) => `${kind}:${method}`),
    );

    expect(realtimeEntries).toHaveLength(14);
    for (const entry of realtimeEntries) {
      expect(entry.status).toBe("product-scope-excluded");
      expect(
        contracts.has(`${manifestKind(entry.direction)}:${entry.method}`),
      ).toBe(false);
    }
  });

  it("Codex process-local diagnostics must stay outside the Desktop manifest", () => {
    const matrix = readJson<Matrix>(MATRIX_PATH);
    const manifest = readJson<Manifest>(MANIFEST_PATH);
    const entry = flatten(matrix).find(
      ({ direction, method }) =>
        direction === "clientRequest" && method === "server/diagnostics",
    );

    expect(entry?.status).toBe("product-scope-excluded");
    expect(
      manifest.methods.some(
        ({ kind, method }) =>
          kind === "request" && method === "server/diagnostics",
      ),
    ).toBe(false);
  });

  it("only the one-shot fuzzy file search belongs to the Desktop manifest", () => {
    const matrix = readJson<Matrix>(MATRIX_PATH);
    const manifest = readJson<Manifest>(MANIFEST_PATH);
    const entries = flatten(matrix).filter(({ method }) =>
      method.startsWith("fuzzyFileSearch"),
    );
    const contracts = new Set(
      manifest.methods.map(({ kind, method }) => `${kind}:${method}`),
    );
    const oneShot = entries.find(
      ({ direction, method }) =>
        direction === "clientRequest" && method === "fuzzyFileSearch",
    );
    const sessionEntries = entries.filter(
      ({ method }) => method !== "fuzzyFileSearch",
    );

    expect(oneShot?.status).toBe("implemented");
    expect(contracts.has("request:fuzzyFileSearch")).toBe(true);
    expect(sessionEntries).toHaveLength(5);
    for (const entry of sessionEntries) {
      expect(entry.status).toBe("product-scope-excluded");
      expect(
        contracts.has(`${manifestKind(entry.direction)}:${entry.method}`),
      ).toBe(false);
    }
  });

  it("excluded migration, marketplace and share surfaces cannot enter Desktop", () => {
    const matrix = readJson<Matrix>(MATRIX_PATH);
    const manifest = readJson<Manifest>(MANIFEST_PATH);
    const excludedMethods = new Set([
      "externalAgentConfig/detect",
      "externalAgentConfig/import",
      "externalAgentConfig/import/readHistories",
      "externalAgentConfig/import/recordHistory",
      "marketplace/add",
      "marketplace/remove",
      "marketplace/upgrade",
      "plugin/share/checkout",
      "plugin/share/delete",
      "plugin/share/list",
      "plugin/share/save",
      "plugin/share/updateTargets",
      "externalAgentConfig/import/completed",
      "externalAgentConfig/import/progress",
    ]);
    const entries = flatten(matrix).filter(({ method }) =>
      excludedMethods.has(method),
    );

    expect(entries).toHaveLength(excludedMethods.size);
    for (const entry of entries) {
      expect(entry.status).toBe("product-scope-excluded");
      expect(
        manifest.methods.some(
          ({ kind, method }) =>
            kind === manifestKind(entry.direction) && method === entry.method,
        ),
      ).toBe(false);
    }
  });

  it("current environment and provider capability surfaces have same-name manifest contracts", () => {
    const matrix = readJson<Matrix>(MATRIX_PATH);
    const manifest = readJson<Manifest>(MANIFEST_PATH);
    const currentMethods = [
      "environment/add",
      "environment/info",
      "environment/status",
      "modelProvider/capabilities/read",
      "thread/environment/connected",
      "thread/environment/disconnected",
    ];
    const contracts = new Set(
      manifest.methods.map(({ kind, method }) => `${kind}:${method}`),
    );
    for (const method of currentMethods) {
      const entry = flatten(matrix).find(
        ({ method: candidate }) => candidate === method,
      );
      expect(entry?.status, `${method} must be current in the matrix`).toBe(
        "implemented",
      );
      expect(
        contracts.has(`${manifestKind(entry!.direction)}:${method}`),
        `${method} must be in the generated manifest`,
      ).toBe(true);
    }
  });
});
