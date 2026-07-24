import { existsSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

describe("agentStreamQueueController", () => {
  it("不得恢复 Renderer queued-turn 状态 owner", () => {
    expect(
      existsSync(
        join(
          process.cwd(),
          "src/components/agent/chat/hooks/agentStreamQueueController.ts",
        ),
      ),
    ).toBe(false);
  });
});
