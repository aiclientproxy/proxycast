import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

const REPO_ROOT = process.cwd();
const COVERAGE_PATH =
  "internal/refactor/v2/fixtures/render-projection-coverage.v0.1.json";

type Direction =
  | "clientRequest"
  | "serverRequest"
  | "serverNotification"
  | "clientNotification";

interface CoverageEntry {
  method: string;
  outlets: string[];
}

interface CoverageFixture {
  upstream: {
    revision: string;
  };
  factSources: {
    itemInventory: string;
    methodScope: string;
  };
  expected: {
    items: number;
    notifications: number;
    serverRequests: number;
  };
  items: Array<{
    type: string;
    outlet: string;
  }>;
  notifications: CoverageEntry[];
  serverRequests: Array<{
    method: string;
    owner: string;
  }>;
  limeExtensions: Array<{
    method: string;
    owner: string;
    outlets: string[];
  }>;
}

interface ItemInventoryFixture {
  items: Array<{
    type: string;
    location?: string;
  }>;
}

interface MethodScopeFixture {
  upstream: {
    revision: string;
  };
  groups: Array<{
    direction: Direction;
    methods: string[];
  }>;
}

function readJson<T>(relativePath: string): T {
  return JSON.parse(readFileSync(join(REPO_ROOT, relativePath), "utf8")) as T;
}

function unique(values: string[]): boolean {
  return new Set(values).size === values.length;
}

function methodsByDirection(
  fixture: MethodScopeFixture,
  direction: Direction,
): string[] {
  return fixture.groups
    .filter((group) => group.direction === direction)
    .flatMap((group) => group.methods)
    .toSorted();
}

describe("Codex render projection coverage boundary", () => {
  it("18 / 72 / 11 coverage entries must be complete and unique", () => {
    const coverage = readJson<CoverageFixture>(COVERAGE_PATH);
    const itemTypes = coverage.items.map((entry) => entry.type);
    const notifications = coverage.notifications.map((entry) => entry.method);
    const serverRequests = coverage.serverRequests.map((entry) => entry.method);

    expect(itemTypes).toHaveLength(coverage.expected.items);
    expect(notifications).toHaveLength(coverage.expected.notifications);
    expect(serverRequests).toHaveLength(coverage.expected.serverRequests);
    expect(unique(itemTypes)).toBe(true);
    expect(unique(notifications)).toBe(true);
    expect(unique(serverRequests)).toBe(true);
    expect(serverRequests).toContain("currentTime/read");

    for (const entry of coverage.items) {
      expect(entry.outlet).toBe("timeline");
    }
    for (const entry of coverage.notifications) {
      expect(entry.outlets.length).toBeGreaterThan(0);
      expect(unique(entry.outlets)).toBe(true);
    }
    for (const entry of coverage.serverRequests) {
      expect(entry.owner.length).toBeGreaterThan(0);
    }
  });

  it("coverage must reuse the v1 item and method facts without omissions", () => {
    const coverage = readJson<CoverageFixture>(COVERAGE_PATH);
    const itemInventory = readJson<ItemInventoryFixture>(
      coverage.factSources.itemInventory,
    );
    const methodScope = readJson<MethodScopeFixture>(
      coverage.factSources.methodScope,
    );
    const canonicalItemTypes = itemInventory.items
      .filter((entry) => entry.location === undefined)
      .map((entry) => entry.type)
      .toSorted();

    expect(
      existsSync(join(REPO_ROOT, coverage.factSources.itemInventory)),
    ).toBe(true);
    expect(existsSync(join(REPO_ROOT, coverage.factSources.methodScope))).toBe(
      true,
    );
    expect(coverage.upstream.revision).toBe(methodScope.upstream.revision);
    expect(coverage.items.map((entry) => entry.type).toSorted()).toEqual(
      canonicalItemTypes,
    );
    expect(
      coverage.notifications.map((entry) => entry.method).toSorted(),
    ).toEqual(methodsByDirection(methodScope, "serverNotification"));
    expect(
      coverage.serverRequests.map((entry) => entry.method).toSorted(),
    ).toEqual(methodsByDirection(methodScope, "serverRequest"));
  });

  it("Lime-owned model surfaces stay outside the Codex method identity set", () => {
    const coverage = readJson<CoverageFixture>(COVERAGE_PATH);
    const upstreamMethods = new Set([
      ...coverage.notifications.map((entry) => entry.method),
      ...coverage.serverRequests.map((entry) => entry.method),
    ]);
    const extensionMethods = coverage.limeExtensions.map(
      (entry) => entry.method,
    );

    expect(unique(extensionMethods)).toBe(true);
    expect(extensionMethods).toContain("model/list/updated");
    for (const entry of coverage.limeExtensions) {
      expect(upstreamMethods.has(entry.method)).toBe(false);
      expect(entry.owner).toContain("model-provider");
      expect(entry.outlets.length).toBeGreaterThan(0);
    }
  });
});
