import { describe, expect, it } from "vitest";

import {
  buildOutputDetailText,
  buildOutputPreviewText,
  hasCodeReviewSurface,
  normalizeReviewSummary,
  outputMentionsFile,
  rankFileChangesForOutput,
  resolveConfirmedFileChangeCount,
  resolveFocusDescriptionKey,
  resolveLatestReviewBoundary,
  resolvePrimaryActionKey,
  resolveReviewFocusTone,
  resolveReviewStatusPresentation,
  resolveReviewableFileChanges,
  selectLatestCheckpoint,
} from "./CodeReviewSummaryPanelViewModel";
import type { HarnessSessionState } from "../utils/harnessState";

type HarnessOutputSignal = HarnessSessionState["outputSignals"][number];

function createOutputSignal(
  partial: Partial<HarnessOutputSignal>,
): HarnessOutputSignal {
  return {
    id: "signal-test",
    toolCallId: "tool-test",
    toolName: "bash",
    title: "",
    summary: "",
    ...partial,
  };
}

describe("CodeReviewSummaryPanelViewModel", () => {
  it("应聚合可审阅文件并把失败输出相关文件排前面", () => {
    const harnessState = {
      activeFileWrites: [
        {
          path: "src/App.tsx",
          displayName: "App.tsx",
        },
      ],
      recentFileEvents: [
        {
          path: "src/ImageCard.tsx",
          displayName: "ImageCard.tsx",
          action: "edit",
          kind: "code",
        },
        {
          path: "src/notes.log",
          displayName: "notes.log",
          action: "persist",
          kind: "log",
        },
      ],
    } as never;
    const files = resolveReviewableFileChanges(harnessState);
    const ordered = rankFileChangesForOutput(files, createOutputSignal({
      title: "ImageCard.tsx failed",
      summary: "vitest failed",
      preview: "",
      content: "",
    }));

    expect(files.map((item) => item.displayName)).toEqual([
      "App.tsx",
      "ImageCard.tsx",
    ]);
    expect(ordered.map((item) => item.displayName)).toEqual([
      "ImageCard.tsx",
      "App.tsx",
    ]);
    expect(
      outputMentionsFile(
        createOutputSignal({
          title: "ImageCard.tsx failed",
          summary: "vitest failed",
          preview: "",
          content: "",
        }),
        ordered[0] || null,
      ),
    ).toBe(true);
  });

  it("应裁剪输出预览和摘要", () => {
    const detail = buildOutputDetailText(createOutputSignal({
      title: "测试失败",
      summary: "vitest failed",
      preview: "1 test failed",
    }));
    const preview = buildOutputPreviewText(createOutputSignal({
      title: "测试输出",
      summary: "line-1\nline-2\nline-3\nline-4\nline-5",
      preview: "",
      content: "",
    }));

    expect(detail).toBe("测试失败 · vitest failed");
    expect(preview).toContain("line-4");
    expect(preview).not.toContain("line-5");
  });

  it("应收敛 review 状态和焦点展示决策", () => {
    const summary = normalizeReviewSummary({
      total: 2,
      pending: 0,
      applied: 1,
      rejected: 1,
    });

    expect(
      resolveReviewStatusPresentation({
        failedOutputCount: 0,
        fileChangeCount: 2,
        outputSignalCount: 1,
        checkpointCount: 0,
        reviewSummary: summary,
      }),
    ).toMatchObject({
      key: "snapshots",
      labelKey: "agentChat.harness.codeReview.status.rollbackReviewed",
    });
    expect(
      resolveReviewFocusTone({
        failedOutputCount: 0,
        fileChangeCount: 2,
        outputSignalCount: 1,
        checkpointCount: 0,
        reviewSummary: summary,
      }),
    ).toBe("review");
    expect(
      resolveFocusDescriptionKey({
        failedOutputCount: 1,
        fileChangeCount: 1,
        outputSignalCount: 0,
        checkpointCount: 0,
        reviewSummary: summary,
      }),
    ).toBe("agentChat.harness.codeReview.focus.failedWithFiles");
  });

  it("应收敛主按钮、审阅 surface 和快照计数", () => {
    expect(
      resolvePrimaryActionKey({
        failedOutputCount: 1,
        fileChangeCount: 2,
        outputSignalCount: 1,
        checkpointCount: 0,
        reviewSummary: null,
      }),
    ).toBe("agentChat.harness.codeReview.action.viewFailedOutput");
    expect(
      hasCodeReviewSurface({
        failedOutputCount: 0,
        fileChangeCount: 0,
        outputSignalCount: 0,
        checkpointCount: 1,
        reviewSummary: null,
      }),
    ).toBe(true);
    expect(
      resolveConfirmedFileChangeCount({
        total: 1,
        pending: 0,
        applied: 1,
        rejected: 0,
      }),
    ).toBe(1);
    expect(selectLatestCheckpoint({ latest_checkpoint: null } as never)).toBe(
      null,
    );
  });

  it("应按 sequence 读取最新 review boundary", () => {
    const base = {
      id: "review-item",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 1,
      status: "completed" as const,
      started_at: "2026-06-06T00:00:00.000Z",
      updated_at: "2026-06-06T00:00:00.000Z",
      type: "review_boundary" as const,
      boundary: "entered" as const,
      review: "开始复核",
    };

    expect(
      resolveLatestReviewBoundary([
        base,
        {
          ...base,
          id: "review-item-2",
          sequence: 2,
          boundary: "exited",
          review: "发现 1 个问题",
        },
      ]),
    ).toEqual({
      status: "completed",
      turnId: "turn-1",
      review: "发现 1 个问题",
    });
    expect(resolveLatestReviewBoundary([])).toBeNull();
  });

});
