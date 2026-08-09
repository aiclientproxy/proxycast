import { describe, expect, it } from "vitest";
import type { AppServerAgentEvent } from "@/lib/api/appServer";
import {
  readCanonicalAgentThreadTurn,
  readPatchItemFromPayload,
} from "./appServerEventTimelineReaders";

const event: AppServerAgentEvent = {
  eventId: "event-patch",
  sequence: 3,
  sessionId: "session-1",
  threadId: "thread-1",
  turnId: "turn-1",
  type: "patch.applied",
  timestamp: "2026-07-21T00:00:01.000Z",
  payload: {},
};

describe("readPatchItemFromPayload", () => {
  it("把 raw patch batch 投影为 Codex tagged changes", () => {
    expect(
      readPatchItemFromPayload(
        {
          patchId: "patch-1",
          changes: [
            { path: "add.ts", kind: "add", diff: "+added" },
            { path: "delete.ts", kind: "delete", diff: "-deleted" },
            { path: "update.ts", kind: "update", diff: "-old\n+new" },
            {
              path: "source.ts",
              kind: "update",
              movePath: "destination.ts",
              diff: "-source\n+destination",
            },
          ],
        },
        event,
        "completed",
      ),
    ).toMatchObject({
      id: "patch-1",
      type: "patch",
      status: "completed",
      paths: ["add.ts", "delete.ts", "update.ts", "source.ts"],
      changes: [
        { path: "add.ts", kind: { type: "add" }, diff: "+added" },
        { path: "delete.ts", kind: { type: "delete" }, diff: "-deleted" },
        {
          path: "update.ts",
          kind: { type: "update" },
          diff: "-old\n+new",
        },
        {
          path: "source.ts",
          kind: { type: "update", move_path: "destination.ts" },
          diff: "-source\n+destination",
        },
      ],
    });
  });
});

describe("readCanonicalAgentThreadTurn", () => {
  it("保留 canonical interrupted 状态而不是降为 canceled", () => {
    expect(
      readCanonicalAgentThreadTurn(
        {
          id: "turn-1",
          status: "interrupted",
          startedAt: "2026-07-21T00:00:00.000Z",
          completedAt: "2026-07-21T00:00:01.000Z",
        },
        event,
        "completed",
      ),
    ).toMatchObject({
      id: "turn-1",
      status: "interrupted",
    });
  });

  it("保留 opaque moderation metadata 供冷恢复投影使用", () => {
    expect(
      readCanonicalAgentThreadTurn(
        {
          id: "turn-1",
          status: "inProgress",
          startedAt: "2026-07-21T00:00:00.000Z",
          moderationMetadata: ["inline", 2],
        },
        event,
        "running",
      ),
    ).toMatchObject({
      moderation_metadata: ["inline", 2],
    });
  });
});
