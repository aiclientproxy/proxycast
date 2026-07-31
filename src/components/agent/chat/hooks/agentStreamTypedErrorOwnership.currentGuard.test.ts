import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

function readHookSource(fileName: string): string {
  return readFileSync(
    join(process.cwd(), "src/components/agent/chat/hooks", fileName),
    "utf8",
  );
}

describe("typed error current owner boundary", () => {
  it("raw runtime error aliases cannot synthesize a second terminal event", () => {
    const binding = readHookSource("agentStreamTurnEventBinding.ts");

    expect(binding).not.toContain('"runtime_error"');
    expect(binding).not.toContain('"runtime.error"');
    expect(binding).not.toContain("readRuntimeErrorMessage");
  });

  it("handler recognizes typed error only through protocol_method", () => {
    const controller = readHookSource("agentStreamTypedErrorController.ts");

    expect(controller).toContain('params.event.protocol_method !== "error"');
    expect(controller).toContain(
      'typeof params.event.will_retry !== "boolean"',
    );
    expect(controller).not.toContain("retryable");
  });
});
