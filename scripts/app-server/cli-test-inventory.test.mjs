import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const inventory = JSON.parse(
  readFileSync(
    path.resolve(
      process.cwd(),
      "internal/exec-plans/cli-codex-test-inventory.json",
    ),
    "utf8",
  ),
);

describe("Codex CLI test inventory", () => {
  it("records every upstream Rust test with stable source evidence", () => {
    expect(inventory.schemaVersion).toBe(1);
    expect(inventory.sourceCommit).toBe(
      "cac96cd7b1756ab42e8925d938817a2ac10ebb6e",
    );
    expect(inventory.sourceFileCount).toBe(53);
    expect(inventory.testCount).toBe(433);
    expect(inventory.entries).toHaveLength(inventory.testCount);
    expect(inventory.sourcePathSetSha256).toBe(
      "f9bac0cacb2726be1fa2c76dd00aab1ef2a6430ec308d552e33493a19783ade9",
    );
    expect(inventory.sourceTreeSha256).toBe(
      "d126a443a2f74c9fcfd626edc2f5f9d8e26410d6067d87d0b4affc53d81ce2b1",
    );

    const identities = inventory.entries.map(
      (entry) => `${entry.path}::${entry.testName}`,
    );
    expect(new Set(identities).size).toBe(inventory.testCount);
    for (const entry of inventory.entries) {
      expect(entry.path).not.toMatch(/^\//u);
      expect(entry.testName).toMatch(/^[A-Za-z][A-Za-z0-9_]*$/u);
      expect(entry.sourceLine).toBeGreaterThan(0);
      expect(entry.sourceFileSha256).toMatch(/^[a-f0-9]{64}$/u);
      expect([
        "direct",
        "contract",
        "product-specific",
        "cloud-deferred",
        "missing",
      ]).toContain(entry.classification);
      expect([
        "covered",
        "partial",
        "pending",
        "deferred",
        "excluded",
      ]).toContain(entry.status);
      expect(inventory.rules.some((rule) => rule.id === entry.rule)).toBe(true);
      expect(entry.rationale.length).toBeGreaterThan(20);
    }
  });

  it("keeps incomplete alignment explicit instead of claiming surface parity", () => {
    const actualCounts = Object.fromEntries(
      [
        "direct",
        "contract",
        "product-specific",
        "cloud-deferred",
        "missing",
      ].map((classification) => [
        classification,
        inventory.entries.filter(
          (entry) => entry.classification === classification,
        ).length,
      ]),
    );
    expect(actualCounts).toEqual(inventory.counts);
    expect(inventory.statusCounts.partial).toBeGreaterThan(0);
    expect(inventory.statusCounts.deferred).toBeGreaterThan(0);
    expect(inventory.statusCounts.excluded).toBeGreaterThan(0);
  });
});
