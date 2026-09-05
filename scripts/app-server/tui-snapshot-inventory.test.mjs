import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const inventory = JSON.parse(
  readFileSync(
    path.resolve(
      process.cwd(),
      "internal/exec-plans/tui-codex-snapshot-inventory.json",
    ),
    "utf8",
  ),
);

describe("Codex TUI snapshot inventory", () => {
  it("classifies every recorded snapshot with a stable relative path and hash", () => {
    expect(inventory.schemaVersion).toBe(1);
    expect(inventory.sourceCommit).toBe(
      "cac96cd7b1756ab42e8925d938817a2ac10ebb6e",
    );
    expect(inventory.sourcePathSetSha256).toBe(
      "da5d7b14f30cccefa3132e2c33b4aaf1e0460ce9a342fc571f857ce15a21ef03",
    );
    expect(inventory.snapshotCount).toBe(inventory.entries.length);
    expect(inventory.snapshotCount).toBe(802);
    expect(new Set(inventory.entries.map((entry) => entry.path)).size).toBe(
      inventory.snapshotCount,
    );
    for (const entry of inventory.entries) {
      expect(entry.path).not.toMatch(/^\//u);
      expect(entry.module).toMatch(/^[a-z][a-z0-9_]*$/u);
      expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/u);
      expect(["direct", "merge", "contract", "defer", "dead"]).toContain(
        entry.classification,
      );
      expect(inventory.rules.some((rule) => rule.id === entry.rule)).toBe(true);
    }
  });

  it("keeps the reviewed migration baseline stable", () => {
    const actualCounts = Object.fromEntries(
      ["direct", "merge", "contract", "defer", "dead"].map(
        (classification) => [
          classification,
          inventory.entries.filter(
            (entry) => entry.classification === classification,
          ).length,
        ],
      ),
    );
    expect(actualCounts).toEqual(inventory.counts);
    expect(actualCounts).toEqual({
      direct: 48,
      merge: 579,
      contract: 80,
      defer: 25,
      dead: 70,
    });
    expect(
      inventory.entries.find((entry) =>
        entry.path.includes("hook_blocked_failed_feedback_history"),
      )?.classification,
    ).toBe("merge");
  });
});
