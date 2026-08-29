import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { readAppServerRuntimeEvents } from "./app-server-stdio-transport.mjs";

const tempRoots = [];

afterEach(() => {
  for (const root of tempRoots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

function createEventsRoot(lines) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "deepswe-events-"));
  tempRoots.push(root);
  const eventsDir = path.join(root, "runtime/events/sessions");
  fs.mkdirSync(eventsDir, { recursive: true });
  fs.writeFileSync(
    path.join(eventsDir, "session.jsonl"),
    `${lines.join("\n")}\n`,
  );
  return root;
}

describe("DeepSWE App Server runtime evidence", () => {
  it("keeps only provider evidence for the canonical turn", async () => {
    const matching = {
      sequence: 2,
      sessionId: "session-1",
      threadId: "thread-1",
      turnId: "turn-1",
      type: "provider.step",
      payload: { attempt: 1, completed: true },
    };
    const root = createEventsRoot([
      JSON.stringify({ ...matching, sequence: 1, type: "reasoning.delta" }),
      JSON.stringify(matching),
      JSON.stringify({ ...matching, sequence: 3, turnId: "turn-2" }),
      JSON.stringify({
        ...matching,
        sequence: 4,
        type: "provider.request.started",
      }),
      JSON.stringify({ ...matching, sequence: 5, type: "provider.usage" }),
    ]);

    const events = await readAppServerRuntimeEvents(root, {
      sessionId: "session-1",
      threadId: "thread-1",
      turnId: "turn-1",
    });

    expect(events.map((event) => [event.sequence, event.type])).toEqual([
      [2, "provider.step"],
      [4, "provider.request.started"],
      [5, "provider.usage"],
    ]);
  });

  it("fails closed when the canonical event log is malformed", async () => {
    const root = createEventsRoot(["{not-json"]);

    await expect(
      readAppServerRuntimeEvents(root, {
        sessionId: "session-1",
        threadId: "thread-1",
        turnId: "turn-1",
      }),
    ).rejects.toThrow("invalid App Server runtime event JSON");
  });
});
