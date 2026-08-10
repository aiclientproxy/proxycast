import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import * as ts from "typescript";
import { describe, expect, it } from "vitest";

const REPO_ROOT = process.cwd();
const INVENTORY_PATH = "internal/refactor/v1/fixtures/item-inventory.v0.1.json";
const EVENT_VERIFIER_PATH =
  "packages/agent-runtime-client/src/eventVerifier.ts";
const THREAD_ITEM_SCHEMA_PATH =
  "lime-rs/crates/app-server-protocol/schema/json/v2/ThreadItem.json";
const SERVER_NOTIFICATION_SCHEMA_PATH =
  "lime-rs/crates/app-server-protocol/schema/json/v2/ServerNotification.json";
const SERVER_REQUEST_SCHEMA_PATH =
  "lime-rs/crates/app-server-protocol/schema/json/v2/ServerRequest.json";

interface ItemInventoryEntry {
  type: string;
  location?: string;
  limeFields: string[];
  statusful: boolean;
  codexStreamMethods: string[];
  codexDeprecatedMethods?: string[];
  limeStreamMethods: string[];
  shape: "aligned" | "gap";
  fieldNotes?: string[];
}

interface ItemAdjacentMethod {
  method: string;
  kind: "notification" | "serverRequest";
  codex: "current" | "experimental" | "deprecated" | "internal";
  lime: "current" | "gap" | "excluded";
}

interface ItemInventoryFixture {
  lime: {
    lifecycle: string[];
  };
  itemAdjacentMethods: ItemAdjacentMethod[];
  items: ItemInventoryEntry[];
}

interface JsonSchemaNode {
  $defs?: Record<string, JsonSchemaNode>;
  $ref?: string;
  oneOf?: JsonSchemaNode[];
  properties?: Record<string, JsonSchemaNode>;
  const?: string;
}

function readInventory(): ItemInventoryFixture {
  return JSON.parse(
    readFileSync(join(REPO_ROOT, INVENTORY_PATH), "utf8"),
  ) as ItemInventoryFixture;
}

function canonicalItemTypesFromClient(): string[] {
  const source = readFileSync(join(REPO_ROOT, EVENT_VERIFIER_PATH), "utf8");
  const sourceFile = ts.createSourceFile(
    EVENT_VERIFIER_PATH,
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TS,
  );

  for (const statement of sourceFile.statements) {
    if (!ts.isVariableStatement(statement)) {
      continue;
    }
    for (const declaration of statement.declarationList.declarations) {
      if (
        !ts.isIdentifier(declaration.name) ||
        declaration.name.text !== "CANONICAL_ITEM_TYPES" ||
        !declaration.initializer
      ) {
        continue;
      }
      const initializer = ts.isAsExpression(declaration.initializer)
        ? declaration.initializer.expression
        : declaration.initializer;
      if (!ts.isArrayLiteralExpression(initializer)) {
        throw new TypeError(
          "CANONICAL_ITEM_TYPES must remain an array literal",
        );
      }
      return initializer.elements.map((element) => {
        if (!ts.isStringLiteral(element)) {
          throw new TypeError(
            "CANONICAL_ITEM_TYPES entries must remain string literals",
          );
        }
        return element.text;
      });
    }
  }

  throw new TypeError("Missing CANONICAL_ITEM_TYPES runtime owner");
}

function readThreadItemSchema(): JsonSchemaNode {
  return JSON.parse(
    readFileSync(join(REPO_ROOT, THREAD_ITEM_SCHEMA_PATH), "utf8"),
  ) as JsonSchemaNode;
}

function canonicalItemTypesFromRustSchema(schema: JsonSchemaNode): string[] {
  return (schema.oneOf ?? []).map((variant) => variantType(variant));
}

function variantType(variant: JsonSchemaNode): string {
  const type = variant.properties?.type?.const;
  if (!type) {
    throw new TypeError("ThreadItem schema variants must have type.const");
  }
  return type;
}

function variantFields(
  schema: JsonSchemaNode,
  variant: JsonSchemaNode,
): string[] {
  const directFields = Object.keys(variant.properties ?? {});
  const referencedFields = variant.$ref
    ? Object.keys(resolveLocalDefinition(schema, variant.$ref).properties ?? {})
    : [];
  return [...new Set([...referencedFields, ...directFields])]
    .filter((field) => field !== "type")
    .sort();
}

function resolveLocalDefinition(
  schema: JsonSchemaNode,
  reference: string,
): JsonSchemaNode {
  const prefix = "#/$defs/";
  if (!reference.startsWith(prefix)) {
    throw new TypeError(
      `Unsupported ThreadItem schema reference: ${reference}`,
    );
  }
  const definition = schema.$defs?.[reference.slice(prefix.length)];
  if (!definition) {
    throw new TypeError(`Missing ThreadItem schema definition: ${reference}`);
  }
  return definition;
}

function schemaConstStrings(relativePath: string): Set<string> {
  const schema = JSON.parse(
    readFileSync(join(REPO_ROOT, relativePath), "utf8"),
  ) as unknown;
  const constants = new Set<string>();

  function visit(value: unknown): void {
    if (Array.isArray(value)) {
      value.forEach(visit);
      return;
    }
    if (!value || typeof value !== "object") {
      return;
    }
    for (const [key, child] of Object.entries(value)) {
      if (key === "const" && typeof child === "string") {
        constants.add(child);
      } else {
        visit(child);
      }
    }
  }

  visit(schema);
  return constants;
}

describe("Codex v2 Item inventory boundary", () => {
  it("inventory 顶层 variant 必须与 runtime client canonical 列表一致", () => {
    const inventory = readInventory();
    const rustSchema = readThreadItemSchema();
    const topLevelTypes = inventory.items
      .filter((item) => item.location === undefined)
      .map((item) => item.type);

    expect(topLevelTypes).toEqual(canonicalItemTypesFromClient());
    expect(topLevelTypes).toEqual(canonicalItemTypesFromRustSchema(rustSchema));
    expect(new Set(topLevelTypes).size).toBe(topLevelTypes.length);

    const inventoryByType = new Map(
      inventory.items
        .filter((item) => item.location === undefined)
        .map((item) => [item.type, item]),
    );
    for (const variant of rustSchema.oneOf ?? []) {
      const type = variantType(variant);
      expect(inventoryByType.get(type)?.limeFields.toSorted()).toEqual(
        variantFields(rustSchema, variant),
      );
    }
  });

  it("MemoryCitation 只能作为 AgentMessage 嵌套 payload", () => {
    const inventory = readInventory();
    const memoryCitation = inventory.items.find(
      (item) => item.type === "memoryCitation",
    );

    expect(memoryCitation?.location).toBe("nested:agentMessage.memoryCitation");
    expect(canonicalItemTypesFromClient()).not.toContain("memoryCitation");
  });

  it("inventory 必须显式记录 lifecycle、stream 和字段收敛状态", () => {
    const inventory = readInventory();

    expect(inventory.lime.lifecycle).toEqual([
      "item/started",
      "item/completed",
    ]);
    for (const item of inventory.items) {
      expect(typeof item.statusful).toBe("boolean");
      expect(Array.isArray(item.codexStreamMethods)).toBe(true);
      expect(Array.isArray(item.limeStreamMethods)).toBe(true);
      expect(["aligned", "gap"]).toContain(item.shape);
      if (item.shape === "gap") {
        expect(item.fieldNotes?.length ?? 0).toBeGreaterThan(0);
      }
      expect(
        item.limeStreamMethods.every((method) =>
          item.codexStreamMethods.includes(method),
        ),
      ).toBe(true);
    }
  });

  it("item 邻接方法分类必须与 Lime 生成 Schema 一致", () => {
    const inventory = readInventory();
    const notificationMethods = schemaConstStrings(
      SERVER_NOTIFICATION_SCHEMA_PATH,
    );
    const serverRequestMethods = schemaConstStrings(SERVER_REQUEST_SCHEMA_PATH);
    const classifiedMethods = new Set<string>();

    for (const entry of inventory.itemAdjacentMethods) {
      expect(classifiedMethods.has(entry.method)).toBe(false);
      classifiedMethods.add(entry.method);

      const limeMethods =
        entry.kind === "notification"
          ? notificationMethods
          : serverRequestMethods;
      expect(limeMethods.has(entry.method), entry.method).toBe(
        entry.lime === "current",
      );
      if (entry.codex === "deprecated" || entry.codex === "internal") {
        expect(entry.lime).toBe("excluded");
      }
    }

    for (const item of inventory.items) {
      for (const method of item.codexStreamMethods) {
        expect(classifiedMethods.has(method)).toBe(true);
      }
      for (const method of item.codexDeprecatedMethods ?? []) {
        expect(classifiedMethods.has(method)).toBe(true);
      }
    }
  });
});
