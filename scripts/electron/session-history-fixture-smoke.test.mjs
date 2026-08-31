import fs from "node:fs";
import { describe, expect, it } from "vitest";

function read(path) {
  return fs.readFileSync(path, "utf8");
}

const paths = {
  smoke: "scripts/electron/session-history-fixture-smoke.mjs",
  threadFixture:
    "scripts/electron/lib/session-history-thread-read-isomorphic-fixture.mjs",
  seedHelper: "scripts/electron/lib/session-history-canonical-thread-seed.mjs",
  threadOracle:
    "scripts/electron/lib/session-history-thread-read-isomorphic-oracle.mjs",
  longListFixture: "scripts/electron/lib/session-history-long-list-fixture.mjs",
  longListOracle: "scripts/electron/lib/session-history-long-list-oracle.mjs",
  archiveOracle:
    "scripts/electron/lib/session-history-thread-archive-oracle.mjs",
};

describe("agent session history Electron fixture smoke guard", () => {
  it("keeps the Gate on real Electron Desktop Host and App Server JSON-RPC", () => {
    const smoke = read(paths.smoke);

    expect(smoke).toContain("import { _electron as electron }");
    expect(smoke).toContain("electron.launch({");
    expect(smoke).toContain('"--use-mock-keychain"');
    expect(smoke).toContain("ELECTRON_E2E_USER_DATA_DIR");
    expect(smoke).toContain('LIME_ELECTRON_E2E: "1"');
    expect(smoke).toContain('LIME_ELECTRON_DEV_HTTP_BRIDGE: "0"');
    expect(smoke).toContain("window.__LIME_ELECTRON__ === true");
    expect(smoke).toContain('typeof window.electronAPI?.invoke === "function"');
    expect(smoke).toContain("window.electronAPI.supportsCommand");
    expect(smoke).toContain("app_server_handle_json_lines");
    expect(smoke).toContain('page.on("pageerror"');
    expect(smoke).toContain("pageErrors.length === 0");
    expect(smoke).toContain('APP_SERVER_BACKEND_MODE: "unavailable"');
  });

  it("runs canonical history and v2 archive lifecycle without legacy writes", () => {
    const smoke = read(paths.smoke);

    expect(smoke).toContain('"thread/start"');
    expect(smoke).toContain('"thread/archive"');
    expect(smoke).toContain('"thread/unarchive"');
    expect(smoke).toContain('"thread/read"');
    expect(smoke).toContain('"thread/list"');
    expect(smoke).toContain('"thread/turns/list"');
    expect(smoke).toContain('"thread/resume"');
    expect(smoke).toContain("runThreadArchivePhase");
    expect(smoke).toContain("runThreadUnarchivePhase");
    expect(smoke).toContain("findRolloutPaths");
    expect(smoke).toContain("archivedRolloutPaths");
    expect(smoke).toContain("restoredRolloutPaths");
    expect(smoke).toContain("seedThreadReadPageIsomorphicCanonicalThread");
    expect(smoke).toContain("runThreadReadPageIsomorphicReadPhase");
    expect(smoke).toContain("runThreadReadPageIsomorphicDomOracle");
    expect(smoke).toContain("databaseBootstrapRestart");
    expect(smoke).not.toContain('call("agentSession/update"');
    expect(smoke).not.toContain('call("agentSession/archiveMany"');
    expect(smoke).not.toContain("runSessionHistoryFixture");
    expect(smoke).not.toContain("runPersistedSessionArchivePhase");
    expect(smoke).not.toContain("runSidebarGuiArchivePhase");
    expect(smoke).not.toContain("runSettingsGuiRestorePhase");
    expect(smoke).not.toContain("archive-readback");
    expect(smoke).not.toContain("unarchive-readback");
  });

  it("locks archive/unarchive to v2 notifications and restart readback", () => {
    const oracle = read(paths.archiveOracle);

    expect(oracle).toContain('call("thread/start"');
    expect(oracle).toContain('call("thread/archive"');
    expect(oracle).toContain('call("thread/unarchive"');
    expect(oracle).toContain('"app_server_drain_events"');
    expect(oracle).toContain("includeRecent: true");
    expect(oracle).toContain('message?.method === "thread/archived"');
    expect(oracle).toContain('message?.method === "thread/unarchived"');
    expect(oracle).toContain(
      'assert(notificationSeen, "未观察到 thread/archived notification")',
    );
    expect(oracle).toContain(
      'assert(notificationSeen, "未观察到 thread/unarchived notification")',
    );
    expect(oracle).toContain("archivedRestartReadback: true");
    expect(oracle).toContain('"agentSession/update"');
    expect(oracle).toContain('"agentSession/archiveMany"');
    expect(oracle).not.toContain('call("agentSession/update"');
    expect(oracle).not.toContain('call("agentSession/archiveMany"');
  });

  it("keeps thread read/list/turns/resume on the v2 Thread/Turn/Item contract", () => {
    const fixture = [read(paths.threadFixture), read(paths.seedHelper)].join(
      "\n",
    );
    const oracle = read(paths.threadOracle);

    expect(fixture).toContain("THREAD_READ_PAGE_ISOMORPHIC");
    expect(fixture).toContain("canonical_threads");
    expect(fixture).toContain("thread_history.canonical_turns");
    expect(fixture).toContain("thread_history.canonical_items");
    expect(fixture).toContain('"sqlite"');
    expect(fixture).toContain('"state.sqlite"');
    expect(fixture).toContain('"thread_history.sqlite"');
    expect(fixture).not.toContain("projection_1.sqlite");
    expect(fixture).toContain('type: "userMessage"');
    expect(fixture).toContain("userInputs");
    expect(fixture).toContain("content: turn.userInputs");
    expect(fixture).toContain('type: "image"');
    expect(fixture).toContain('uri: "data:image/png;');
    expect(fixture).toContain('type: "reasoning"');
    expect(fixture).toContain('type: "agentMessage"');
    expect(fixture).toContain('phase: "final_answer"');
    expect(fixture).not.toContain('phase: "final"');
    expect(fixture).not.toContain("content: turn.userText");
    expect(fixture).not.toContain("projected_turns");
    expect(fixture).not.toContain("projected_items");
    expect(oracle).toContain('"thread/read"');
    expect(oracle).toContain('"thread/list"');
    expect(oracle).toContain('"thread/turns/list"');
    expect(oracle).toContain('"thread/items/list"');
    expect(oracle).toContain('"thread/resume"');
    expect(oracle).toContain("threadId: fixture.threadId");
    expect(oracle).toContain("includeTurns: true");
    expect(oracle).toContain("cursor: newestPage.nextCursor");
    expect(oracle).toContain("cursor: middlePage.nextCursor");
    expect(oracle).toContain("excludeTurns: true");
    expect(oracle).toContain("initialTurnsPage");
    expect(oracle).toContain("result?.resume?.initialTurnsPage?.data");
    expect(oracle).toContain("itemsPage");
    expect(oracle).toContain("assertOrderedUserInputContent");
    expect(oracle).toContain("imageAttachmentCount");
    expect(oracle).toContain("message-image-attachment-");
    expect(oracle).toContain('message?.method === "thread/started"');
    expect(oracle).not.toContain("sessionId: fixture.sessionId");
    expect(oracle).not.toContain("historyLimit:");
    expect(oracle).not.toContain("historyOffset:");
    expect(oracle).not.toContain("result?.resume?.resumed");
  });

  it("proves bounded direct Turn rendering for a deterministic long history", () => {
    const smoke = read(paths.smoke);
    const fixture = read(paths.longListFixture);
    const oracle = read(paths.longListOracle);

    expect(smoke).toContain("seedThreadReadLongListCanonicalThread");
    expect(smoke).toContain("runThreadReadLongListDomOracle");
    expect(smoke).toContain("assertThreadReadLongListDomOracle");
    expect(smoke).toContain("runThreadReadLongListColdRestartOracle");
    expect(smoke).toContain("assertThreadReadLongListColdRestartOracle");
    expect(smoke).toContain("threadReadLongListSummary");
    expect(smoke).toContain("threadReadLongListColdRestartSummary");
    expect(smoke).toContain("longListScreenshot");
    expect(smoke).toContain("longListColdRestartScreenshot");
    expect(smoke).toContain("clearBuffersAfterReady: false");
    expect(fixture).toContain("THREAD_READ_LONG_LIST_TURN_COUNT = 240");
    expect(fixture).toContain("THREAD_READ_LONG_LIST_ITEM_COUNT");
    expect(fixture).toContain("LONG_HISTORY_TERMINAL_MARKER");
    expect(fixture).toContain("seedCanonicalHistoryThread");
    expect(oracle).toContain("MAX_INITIAL_DOM_TURN_GROUPS");
    expect(oracle).toContain("MAX_INITIAL_DOM_TEXT_LENGTH");
    expect(oracle).toContain("window.__LIME_AGENTUI_PERF__?.summary");
    expect(oracle).toContain("clickToFirstMessageListPaintMs");
    expect(oracle).toContain("messageListComputeMaxMs");
    expect(oracle).toContain('"thread/read"');
    expect(oracle).toContain('"thread/turns/list"');
    expect(oracle).toContain('"thread/items/list"');
    expect(oracle).toContain('"thread/resume"');
    expect(oracle).toContain("COLD_RESTART_REQUIRED_METHODS");
    expect(oracle).toContain('navigation: "sidebar"');
    expect(oracle).toContain("processRestart: true");
    expect(oracle).toContain('"canonical_turn"');
    expect(oracle).toContain('"message_group"');
    expect(oracle).not.toContain("agent_runtime_");
  });

  it("does not use legacy runtime or production mock fallback", () => {
    const combined = Object.values(paths)
      .map((path) => read(path))
      .join("\n");

    expect(combined).not.toContain("agent_runtime_");
    expect(combined).not.toContain("mockPriorityCommands");
    expect(combined).not.toContain("defaultMocks");
    expect(combined).not.toContain("invokeMockOnly");
    expect(combined).not.toContain('APP_SERVER_BACKEND_MODE: "external"');
    expect(combined).not.toContain("APP_SERVER_BACKEND_COMMAND");
    expect(combined).not.toContain("--allow-live-provider");
  });
});
