import assert from "node:assert/strict";
import { test } from "vitest";
import { AppServerClient } from "../dist/index.js";

test("typed client exposes the two session control requests", () => {
  const client = new AppServerClient({ initialRequestId: 40 });
  assert.deepEqual(
    client.updateThreadSettings({
      threadId: "thread-1",
      model: "model-b",
      modelProvider: "provider-b",
      serviceTier: null,
    }),
    {
      id: 40,
      method: "thread/settings/update",
      params: {
        threadId: "thread-1",
        model: "model-b",
        modelProvider: "provider-b",
        serviceTier: null,
      },
    },
  );
  assert.deepEqual(
    client.setThreadMemoryMode({
      threadId: "thread-1",
      mode: "disabled",
    }),
    {
      id: 41,
      method: "thread/memoryMode/set",
      params: {
        threadId: "thread-1",
        mode: "disabled",
      },
    },
  );
  assert.deepEqual(
    client.runThreadShellCommand({
      threadId: "thread-1",
      command: "printf shell",
    }),
    {
      id: 42,
      method: "thread/shellCommand",
      params: {
        threadId: "thread-1",
        command: "printf shell",
      },
    },
  );
});
