import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";
import { listCodexV2NotificationMethods } from "@/lib/api/agentRuntime/appServerNotificationDrift";
import { isAppServerV2NotificationMethod } from "@/lib/api/agentRuntime/appServerV2Notification";

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
  schemaExcluded?: boolean;
}

interface CoverageFixture {
  upstream: {
    repository: string;
    revision: string;
    itemSchemaSource: string;
    itemSchemaSha256: string;
    notificationSchemaSource: string;
    notificationSchemaSha256: string;
    requestSchemaSource: string;
    requestSchemaSha256: string;
    schemaSnapshotSha256: string;
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
    schemaExcluded?: boolean;
  }>;
  limeExtensions: Array<{
    method: string;
    owner: string;
    outlets: string[];
  }>;
}

interface ItemInventoryFixture {
  upstream: {
    revision: string;
    schemaSource: string;
    schemaSha256: string;
  };
  items: Array<{
    type: string;
    location?: string;
    codexFields: string[];
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

interface JsonSchemaNode {
  $ref?: string;
  definitions?: Record<string, JsonSchemaNode>;
  enum?: unknown[];
  oneOf?: JsonSchemaNode[];
  properties?: Record<string, JsonSchemaNode>;
}

function readJson<T>(relativePath: string): T {
  return JSON.parse(readFileSync(join(REPO_ROOT, relativePath), "utf8")) as T;
}

function unique(values: string[]): boolean {
  return new Set(values).size === values.length;
}

function sha256(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function readPinnedUpstreamFile(
  fixture: CoverageFixture,
  relativePath: string,
): Buffer | null {
  if (!existsSync(fixture.upstream.repository)) {
    return null;
  }
  return execFileSync(
    "git",
    [
      "-C",
      fixture.upstream.repository,
      "show",
      `${fixture.upstream.revision}:${relativePath}`,
    ],
    { stdio: ["ignore", "pipe", "pipe"] },
  );
}

function readSchema(value: Buffer): JsonSchemaNode {
  return JSON.parse(value.toString("utf8")) as JsonSchemaNode;
}

function resolveDefinition(
  schema: JsonSchemaNode,
  reference: string,
): JsonSchemaNode {
  const prefix = "#/definitions/";
  if (!reference.startsWith(prefix)) {
    throw new TypeError(`Unsupported schema reference: ${reference}`);
  }
  const definition = schema.definitions?.[reference.slice(prefix.length)];
  if (!definition) {
    throw new TypeError(`Missing schema definition: ${reference}`);
  }
  return definition;
}

function threadItemSchema(schemaBundle: JsonSchemaNode): JsonSchemaNode {
  const schema = schemaBundle.definitions?.ThreadItem;
  if (!schema) {
    throw new TypeError("Missing upstream ThreadItem schema");
  }
  return schema;
}

function schemaVariantShape(
  schemaBundle: JsonSchemaNode,
  variant: JsonSchemaNode,
): { type: string; fields: string[] } {
  const resolved = variant.$ref
    ? resolveDefinition(schemaBundle, variant.$ref)
    : variant;
  const type = resolved.properties?.type?.enum?.[0];
  if (typeof type !== "string") {
    throw new TypeError("ThreadItem schema variant is missing type enum");
  }
  return {
    type,
    fields: Object.keys(resolved.properties ?? {})
      .filter((field) => field !== "type")
      .toSorted(),
  };
}

function schemaMethods(schema: JsonSchemaNode): string[] {
  return (schema.oneOf ?? [])
    .map((variant) => variant.properties?.method?.enum?.[0])
    .filter((method): method is string => typeof method === "string")
    .toSorted();
}

function schemaSnapshot(
  coverage: CoverageFixture,
  inventory: ItemInventoryFixture,
): Buffer {
  return Buffer.from(
    JSON.stringify({
      revision: coverage.upstream.revision,
      itemSchemaSha256: coverage.upstream.itemSchemaSha256,
      notificationSchemaSha256: coverage.upstream.notificationSchemaSha256,
      requestSchemaSha256: coverage.upstream.requestSchemaSha256,
      items: inventory.items
        .filter((item) => item.location === undefined)
        .map((item) => ({
          type: item.type,
          fields: item.codexFields.toSorted(),
        }))
        .toSorted((left, right) => left.type.localeCompare(right.type)),
      notifications: coverage.notifications
        .map((entry) => ({
          method: entry.method,
          schemaExcluded: entry.schemaExcluded === true,
        }))
        .toSorted((left, right) => left.method.localeCompare(right.method)),
      requests: coverage.serverRequests
        .map((entry) => ({
          method: entry.method,
          schemaExcluded: entry.schemaExcluded === true,
        }))
        .toSorted((left, right) => left.method.localeCompare(right.method)),
    }),
  );
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
  it("product-scope-excluded notifications remain diagnostics-only and outside the current projector", () => {
    const coverage = readJson<CoverageFixture>(COVERAGE_PATH);
    const excludedMethods = [
      "process/outputDelta",
      "process/exited",
    ];

    for (const method of excludedMethods) {
      const entry = coverage.notifications.find(
        (candidate) => candidate.method === method,
      );
      expect(entry).toBeDefined();
      expect(entry?.outlets).toEqual(["diagnostics"]);
      expect(isAppServerV2NotificationMethod(method)).toBe(false);
    }
  });

  it("19 / 72 / 11 coverage entries must be complete and unique", () => {
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
    expect(listCodexV2NotificationMethods()).toEqual(notifications.toSorted());

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
    expect(coverage.upstream.revision).toBe(itemInventory.upstream.revision);
    expect(coverage.upstream.revision).toBe(methodScope.upstream.revision);
    expect(coverage.upstream.itemSchemaSource).toBe(
      itemInventory.upstream.schemaSource,
    );
    expect(coverage.upstream.itemSchemaSha256).toBe(
      itemInventory.upstream.schemaSha256,
    );
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

  it("pinned upstream schema must match item fields and method inventories", () => {
    const coverage = readJson<CoverageFixture>(COVERAGE_PATH);
    const itemInventory = readJson<ItemInventoryFixture>(
      coverage.factSources.itemInventory,
    );
    const itemSchemaBytes = readPinnedUpstreamFile(
      coverage,
      coverage.upstream.itemSchemaSource,
    );
    const notificationSchemaBytes = readPinnedUpstreamFile(
      coverage,
      coverage.upstream.notificationSchemaSource,
    );
    const requestSchemaBytes = readPinnedUpstreamFile(
      coverage,
      coverage.upstream.requestSchemaSource,
    );

    expect(coverage.upstream.itemSchemaSha256).toMatch(/^[a-f0-9]{64}$/u);
    expect(coverage.upstream.notificationSchemaSha256).toMatch(
      /^[a-f0-9]{64}$/u,
    );
    expect(coverage.upstream.requestSchemaSha256).toMatch(/^[a-f0-9]{64}$/u);
    expect(sha256(schemaSnapshot(coverage, itemInventory))).toBe(
      coverage.upstream.schemaSnapshotSha256,
    );
    if (!itemSchemaBytes || !notificationSchemaBytes || !requestSchemaBytes) {
      return;
    }

    expect(sha256(itemSchemaBytes)).toBe(coverage.upstream.itemSchemaSha256);
    expect(sha256(notificationSchemaBytes)).toBe(
      coverage.upstream.notificationSchemaSha256,
    );
    expect(sha256(requestSchemaBytes)).toBe(
      coverage.upstream.requestSchemaSha256,
    );

    const upstreamItems = (
      threadItemSchema(readSchema(itemSchemaBytes)).oneOf ?? []
    )
      .map((variant) =>
        schemaVariantShape(readSchema(itemSchemaBytes), variant),
      )
      .toSorted((left, right) => left.type.localeCompare(right.type));
    const inventoriedItems = itemInventory.items
      .filter(
        (item) => item.location === undefined && item.codexFields.length > 0,
      )
      .map((item) => ({
        type: item.type,
        fields: item.codexFields.toSorted(),
      }))
      .toSorted((left, right) => left.type.localeCompare(right.type));
    expect(inventoriedItems).toEqual(upstreamItems);
    expect(schemaMethods(readSchema(notificationSchemaBytes))).toEqual(
      coverage.notifications
        .filter((entry) => entry.schemaExcluded !== true)
        .map((entry) => entry.method)
        .toSorted(),
    );
    expect(schemaMethods(readSchema(requestSchemaBytes))).toEqual(
      coverage.serverRequests
        .filter((entry) => entry.schemaExcluded !== true)
        .map((entry) => entry.method)
        .toSorted(),
    );
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
