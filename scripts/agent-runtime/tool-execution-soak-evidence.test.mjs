import { describe, expect, it } from "vitest";

import {
  buildSoakSummary,
  childArgsForRound,
  containsCanonicalThread,
  parsePosixProcessRows,
  parseWindowsProcessRows,
  resolveSoakRoundThreadIdentity,
  resolveSoakConfig,
  roundEvidencePath,
  summarizeProcessTree,
} from "./tool-execution-soak-evidence.mjs";

describe("tool execution SOAK evidence", () => {
  it("requires repeated rounds to include two cold restarts", () => {
    expect(() =>
      resolveSoakConfig([
        "--soak-rounds",
        "3",
        "--cold-restart",
        "--cold-restarts",
        "1",
      ]),
    ).toThrow(/至少需要两次 cold restart/);
    expect(
      resolveSoakConfig([
        "--soak-rounds",
        "3",
        "--cold-restart",
        "--cold-restarts",
        "2",
      ]),
    ).toEqual({ enabled: true, rounds: 3, coldRestarts: 2 });
  });

  it("keeps managed-only args away from the child smoke", () => {
    expect(
      childArgsForRound(
        [
          "--batch",
          "agent-control-tools",
          "--cold-restart",
          "--cold-restarts",
          "2",
          "--soak-rounds",
          "3",
          "--output",
          "old.json",
        ],
        "round.json",
      ),
    ).toEqual([
      "--batch",
      "agent-control-tools",
      "--output",
      "round.json",
    ]);
  });

  it("uses the requested output for the final round", () => {
    expect(roundEvidencePath("/tmp/soak.json", 0, 3)).toBe(
      "/tmp/soak-round-01.json",
    );
    expect(roundEvidencePath("/tmp/soak.json", 2, 3)).toBe(
      "/tmp/soak.json",
    );
  });

  it("reads SOAK threads by the canonical thread id", () => {
    expect(
      resolveSoakRoundThreadIdentity({
        runtime: {
          sessionId: "session-1",
          threadId: "thread-1",
        },
      }),
    ).toEqual({ sessionId: "session-1", threadId: "thread-1" });
    expect(() =>
      resolveSoakRoundThreadIdentity({
        runtime: { sessionId: "session-1" },
      }),
    ).toThrow(/canonical sessionId\/threadId/);
  });

  it("matches v2 thread/list entries by id and sessionId", () => {
    expect(
      containsCanonicalThread(
        [
          { id: "thread-1", sessionId: "session-1" },
          { id: "thread-2", sessionId: "session-2" },
        ],
        { sessionId: "session-1", threadId: "thread-1" },
      ),
    ).toBe(true);
    expect(
      containsCanonicalThread(
        [{ id: "thread-1", sessionId: "session-other" }],
        { sessionId: "session-1", threadId: "thread-1" },
      ),
    ).toBe(false);
  });

  it("parses POSIX and Windows process inventories", () => {
    expect(
      parsePosixProcessRows(
        "  10 1 2048 Electron .\n  11 10 1024 /tmp/app-server --stdio\n",
      ),
    ).toEqual([
      { pid: 10, ppid: 1, rssKb: 2048, command: "Electron ." },
      {
        pid: 11,
        ppid: 10,
        rssKb: 1024,
        command: "/tmp/app-server --stdio",
      },
    ]);
    expect(
      parseWindowsProcessRows(
        JSON.stringify({
          ProcessId: 10,
          ParentProcessId: 1,
          WorkingSetSize: 1048576,
          CommandLine: "Lime.exe",
        }),
      ),
    ).toEqual([
      { pid: 10, ppid: 1, rssKb: 1024, command: "Lime.exe" },
    ]);
  });

  it("summarizes only the managed Electron process tree", () => {
    const snapshot = summarizeProcessTree(
      [
        { pid: 10, ppid: 1, rssKb: 2000, command: "Electron ." },
        {
          pid: 11,
          ppid: 10,
          rssKb: 1000,
          command: "/tmp/app-server --stdio",
        },
        { pid: 12, ppid: 10, rssKb: 500, command: "Electron Helper" },
        { pid: 99, ppid: 1, rssKb: 9000, command: "unrelated" },
      ],
      10,
      "round-1",
    );
    expect(snapshot.processCount).toBe(3);
    expect(snapshot.totalRssKb).toBe(3500);
    expect(snapshot.appServerPids).toEqual([11]);
    expect(snapshot.appServerRssKb).toBe(1000);
  });

  it("fails the aggregate when a restart leaves a process behind", () => {
    const process = {
      processCount: 3,
      totalRssKb: 3500,
      appServerPids: [11],
      appServerRssKb: 1000,
    };
    const round = (index) => ({
      sessionId: `session-${index}`,
      durationMs: 1_000,
      process,
      assertions: { passed: true },
    });
    const summary = buildSoakSummary({
      finalShutdown: { exited: true },
      processSnapshots: [],
      restoredRounds: [round(1), round(2), round(3)],
      restarts: [
        {
          electronProcessReplaced: true,
          previousProcessTreeExit: { exited: true },
        },
        {
          electronProcessReplaced: true,
          previousProcessTreeExit: { exited: false },
        },
      ],
      rounds: [round(1), round(2), round(3)],
    });
    expect(summary.assertions.everyPreviousProcessTreeExited).toBe(false);
    expect(summary.assertions.finalProcessTreeExited).toBe(true);
  });

  it("detects canonical identity drift after cold restarts", () => {
    const process = {
      processCount: 3,
      totalRssKb: 3500,
      appServerPids: [11],
      appServerRssKb: 1000,
    };
    const round = {
      sessionId: "session-1",
      durationMs: 1_000,
      threadId: "thread-1",
      turnIds: ["turn-1"],
      itemIds: ["item-1"],
      turnStatusCounts: { completed: 1 },
      itemStatusCounts: { completed: 1 },
      itemKindCounts: { agent_message: 1 },
      process,
      assertions: { passed: true },
    };
    const summary = buildSoakSummary({
      finalShutdown: { exited: true },
      processSnapshots: [],
      restoredRounds: [{ ...round, itemIds: ["item-drift"] }],
      restarts: [
        {
          electronProcessReplaced: true,
          previousProcessTreeExit: { exited: true },
        },
        {
          electronProcessReplaced: true,
          previousProcessTreeExit: { exited: true },
        },
      ],
      rounds: [round],
    });
    expect(summary.assertions.readModelsStableAcrossColdRestarts).toBe(false);
  });

  it("rejects a round that stalls past the calibration budget", () => {
    const process = {
      processCount: 3,
      totalRssKb: 3500,
      appServerPids: [11],
      appServerRssKb: 1000,
    };
    const round = {
      sessionId: "session-1",
      threadId: "thread-1",
      durationMs: 180_000,
      turnIds: ["turn-1"],
      itemIds: ["item-1"],
      turnStatusCounts: { completed: 1 },
      itemStatusCounts: { completed: 1 },
      itemKindCounts: { agent_message: 1 },
      process,
      assertions: { passed: true },
    };
    const restart = {
      electronProcessReplaced: true,
      previousProcessTreeExit: { exited: true },
    };
    const summary = buildSoakSummary({
      finalShutdown: { exited: true },
      processSnapshots: [],
      restoredRounds: [round],
      restarts: [restart, restart],
      rounds: [round],
    });
    expect(summary.assertions.roundDurationWithinCalibrationBudget).toBe(
      false,
    );
  });
});
