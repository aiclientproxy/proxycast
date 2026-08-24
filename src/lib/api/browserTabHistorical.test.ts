import { describe, expect, it } from "vitest";
import {
  createBrowserTabHistoricalProjection,
  readBrowserTabHistoricalProjection,
} from "./browserTab";

describe("Browser historical projection", () => {
  it("只保留可见历史事实并固定为只读", () => {
    const projection = createBrowserTabHistoricalProjection({
      mode: "replay",
      browserSessionId: "browser-session-1",
      tabId: "browser-tab-1",
      threadId: "thread-1",
      url: "https://example.com/history",
      title: "History",
      pageRevision: 7,
      mark: "deliverable",
      origin: "user",
      snapshotId: "snapshot-7",
      replayedAt: "2026-08-23T00:00:00.000Z",
    });

    expect(projection).toEqual({
      browserSessionId: "browser-session-1",
      tabId: "browser-tab-1",
      threadId: "thread-1",
      url: "https://example.com/history",
      title: "History",
      pageRevision: 7,
      mark: "deliverable",
      origin: "user",
      selected: true,
      snapshotId: "snapshot-7",
      replayedAt: "2026-08-23T00:00:00.000Z",
      readOnly: true,
    });
  });

  it("拒绝缺失 historical/replay mode 或非法版本", () => {
    expect(
      readBrowserTabHistoricalProjection({
        browserSessionId: "browser-session-1",
        tabId: "browser-tab-1",
        threadId: "thread-1",
        url: "https://example.com",
      }),
    ).toBeNull();
    expect(
      createBrowserTabHistoricalProjection({
        mode: "historical",
        browserSessionId: "browser-session-1",
        tabId: "browser-tab-1",
        threadId: "thread-1",
        url: "https://example.com",
        pageRevision: -1,
      }),
    ).toBeNull();
  });

  it("不生成任何可恢复的运行态凭证或本地路径", () => {
    const projection = readBrowserTabHistoricalProjection({
      mode: "historical",
      browserSessionId: "browser-session-1",
      tabId: "browser-tab-1",
      threadId: "thread-1",
      url: "https://example.com",
      title: "History",
      pageRevision: 2,
      activeTurnId: "must-not-be-copied",
      approvalToken: "must-not-be-copied",
      artifactPath: "/Users/coso/secret.bin",
      pendingMutation: { action: "click" },
    });

    expect(projection).toMatchObject({
      readOnly: true,
      snapshotId: null,
      replayedAt: null,
    });
    expect(JSON.stringify(projection)).not.toContain("must-not-be-copied");
    expect(JSON.stringify(projection)).not.toContain("/Users/");
    expect(projection).not.toHaveProperty("activeTurnId");
    expect(projection).not.toHaveProperty("approvalToken");
    expect(projection).not.toHaveProperty("pendingMutation");
  });
});
