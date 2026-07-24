import assert from "node:assert/strict";
import { test } from "vitest";
import {
  AppServerClient,
  AppServerConnection,
  METHOD_ARTIFACT_WRITE,
} from "../dist/index.js";
import * as clientExports from "../dist/index.js";

const params = {
  threadId: "thread_1",
  turnId: "turn_2",
  artifact: {
    artifactRef: "artifact_doc_3",
    artifactDocumentId: "doc_3",
    path: "drafts/article.json",
    kind: "artifact_document",
    content: "{}",
  },
};

test("builds typed artifact/write requests", () => {
  const client = new AppServerClient();
  const request = client.writeArtifact(params);

  assert.equal(request.method, METHOD_ARTIFACT_WRITE);
  assert.deepEqual(request.params, params);
});

test("connection sends and decodes typed artifact/write responses", async () => {
  const sent = [];
  const connection = new AppServerConnection(
    {
      send(message) {
        sent.push(message);
      },
      async nextMessage() {
        return {
          id: 1,
          result: {
            threadId: "thread_1",
            turnId: "turn_2",
            artifactRef: "artifact_doc_3",
            eventId: "evt_1",
            sequence: 4,
            persistedAt: "2026-07-24T00:00:00.000Z",
            sidecar: {
              relativePath: "sessions/thread_1/artifact_doc_3.json",
              bytes: 2,
              sha256: "sha256:artifact",
              contentStatus: "available",
            },
          },
        };
      },
    },
    new AppServerClient(),
  );

  const result = await connection.writeArtifact(params);

  assert.equal(sent[0].method, METHOD_ARTIFACT_WRITE);
  assert.deepEqual(result.result, {
    threadId: "thread_1",
    turnId: "turn_2",
    artifactRef: "artifact_doc_3",
    eventId: "evt_1",
    sequence: 4,
    persistedAt: "2026-07-24T00:00:00.000Z",
    sidecar: {
      relativePath: "sessions/thread_1/artifact_doc_3.json",
      bytes: 2,
      sha256: "sha256:artifact",
      contentStatus: "available",
    },
  });
});

test("does not expose the retired generic runtime event append API", () => {
  assert.equal(
    "METHOD_AGENT_SESSION_RUNTIME_EVENTS_APPEND" in clientExports,
    false,
  );
  assert.equal(
    "appendAgentSessionRuntimeEvents" in new AppServerClient(),
    false,
  );
});
