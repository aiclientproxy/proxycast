import { describe, expect, it } from "vitest";
import {
  buildCuratedTaskTemplateCopy,
  findCuratedTaskTemplateById,
  type CuratedTaskInputValues,
  type CuratedTaskTemplateItem,
} from "@/components/agent/chat/utils/curatedTaskTemplates";
import type { CuratedTaskRecommendationSignal } from "@/components/agent/chat/utils/curatedTaskRecommendationSignals";
import type { CuratedTaskReferenceEntry } from "@/components/agent/chat/utils/curatedTaskReferenceSelection";
import {
  buildActiveReviewBaselineModel,
  buildCuratedTaskLauncherReadiness,
  buildLauncherOutcomeSummary,
  buildLauncherStarterContract,
  planReferenceEntrySelection,
  resolvePrimarySuggestedTask,
  resolveSelectedReferenceEntries,
  selectLatestReviewTaskSignal,
} from "./CuratedTaskLauncherDialogViewModel";

const copy = {
  readinessReady: "可以开始",
  readinessMissing: (count: number) => `还差 ${count} 项`,
  outcomeWithFollowUp: ({
    followUp,
    outputHint,
  }: {
    followUp: string;
    outputHint: string;
  }) => `${outputHint}，后续 ${followUp}`,
  outcomeDefault: ({ outputHint }: { outputHint: string }) =>
    `将生成 ${outputHint}`,
  contractRequiredEmpty: "无需补充",
  carryFieldSeparator: " / ",
  carryReview: (fields: string) => `复盘沿用：${fields}`,
  carryDefault: (fields: string) => `沿用：${fields}`,
};

function task(overrides: Partial<CuratedTaskTemplateItem> = {}) {
  const resolved = findCuratedTaskTemplateById(
    "account-project-review",
    buildCuratedTaskTemplateCopy((key) => key),
  );
  if (!resolved) {
    throw new Error("missing account project review task fixture");
  }
  return {
    ...resolved,
    ...overrides,
  };
}

function referenceEntry(
  overrides: Partial<CuratedTaskReferenceEntry> = {},
): CuratedTaskReferenceEntry {
  return {
    id: "memory:content-pack:run:1",
    sourceKind: "memory",
    title: "AI 内容周报",
    summary: "当前已有一轮项目结果，可直接作为复盘基线。",
    category: "experience",
    categoryLabel: "成果",
    tags: ["复盘"],
    taskPrefillByTaskId: {
      "account-project-review": {
        project_goal: "AI 内容周报",
        existing_results:
          "这轮运行已产出项目结果 当前卡点：复核阻塞 当前判断：先补复核与修复 更适合去向：结果对齐",
      },
    },
    ...overrides,
  };
}

function reviewSignal(
  overrides: Partial<CuratedTaskRecommendationSignal> = {},
): CuratedTaskRecommendationSignal {
  return {
    createdAt: 200,
    source: "review_feedback",
    category: "experience",
    title: "最近判断已更新：短视频编排 · 补证据",
    summary: "这轮结果还缺证据。",
    tags: ["证据不足"],
    preferredTaskIds: ["account-project-review"],
    projectId: "project-1",
    sessionId: "session-1",
    ...overrides,
  };
}

describe("CuratedTaskLauncherDialogViewModel", () => {
  it("应计算必填项 readiness 和启动禁用态", () => {
    const currentTask = task();
    const missing = buildCuratedTaskLauncherReadiness({
      task: currentTask,
      inputValues: {
        project_goal: "AI 内容周报",
      },
      copy,
    });
    const ready = buildCuratedTaskLauncherReadiness({
      task: currentTask,
      inputValues: {
        project_goal: "AI 内容周报",
        existing_results: "已有结果包和复核结论。",
      },
      copy,
    });

    expect(missing).toMatchObject({
      isLaunchDisabled: true,
      requiredFieldCount: 2,
      filledRequiredFieldCount: 1,
      remainingRequiredFieldCount: 1,
      launcherReadinessLabel: "还差 1 项",
    });
    expect(ready).toMatchObject({
      isLaunchDisabled: false,
      filledRequiredFieldCount: 2,
      remainingRequiredFieldCount: 0,
      launcherReadinessLabel: "可以开始",
    });
  });

  it("应解析已选 reference 并规划勾选上限", () => {
    const entries = [
      referenceEntry({ id: "entry-1" }),
      referenceEntry({ id: "entry-2" }),
    ];

    expect(
      resolveSelectedReferenceEntries({
        referenceEntries: entries,
        selectedReferenceEntryIds: ["entry-2", "missing"],
      }),
    ).toEqual({
      selectedReferenceEntries: [entries[1]],
      missingSelectedReferenceCount: 1,
    });
    expect(
      planReferenceEntrySelection({
        currentIds: ["entry-1", "entry-2", "entry-3"],
        entryId: "entry-4",
        maxSelectionCount: 3,
      }),
    ).toEqual(["entry-1", "entry-2", "entry-3"]);
    expect(
      planReferenceEntrySelection({
        currentIds: ["entry-1", "entry-2"],
        entryId: "entry-2",
      }),
    ).toEqual(["entry-1"]);
    expect(
      planReferenceEntrySelection({
        currentIds: ["entry-1"],
        entryId: "entry-2",
      }),
    ).toEqual(["entry-1", "entry-2"]);
  });

  it("应构造 outcome 和 starter contract", () => {
    const currentTask = task({
      outputHint: "复盘结论",
      followUpActions: ["整理行动项"],
    });
    const presentationCopy = {
      formatFactItems: (visibleItems: string[], totalCount: number) =>
        `${visibleItems.join("、")} / ${totalCount}`,
    };

    expect(
      buildLauncherOutcomeSummary({
        task: currentTask,
        copy,
      }),
    ).toBe("复盘结论，后续 整理行动项");
    expect(
      buildLauncherStarterContract({
        task: currentTask,
        presentationCopy,
        copy,
      }),
    ).toMatchObject({
      requiredSummary: expect.stringContaining(
        "curatedTask.templates.account-project-review.fields.project_goal.label",
      ),
      outputSummary: expect.stringContaining(
        "curatedTask.templates.account-project-review.outputContract.0",
      ),
      followUpSummary: expect.stringContaining("整理行动项"),
    });
  });

  it("应选择最新 review signal 并解析推荐模板", () => {
    const latest = reviewSignal({
      title: "latest",
      createdAt: 300,
      preferredTaskIds: ["account-project-review"],
    });
    const selected = selectLatestReviewTaskSignal([
      reviewSignal({
        title: "other",
        source: "memory_reference",
        createdAt: 500,
      }),
      reviewSignal({ title: "old", createdAt: 100 }),
      latest,
    ]);
    const currentTask = findCuratedTaskTemplateById(
      "daily-trend-briefing",
      buildCuratedTaskTemplateCopy((key) => key),
    );

    const result = resolvePrimarySuggestedTask({
      currentTask,
      latestReviewTaskSignal: selected,
      curatedTaskTemplateCopy: buildCuratedTaskTemplateCopy((key) => key),
    });

    expect(selected?.title).toBe("latest");
    expect(result.reviewFeedbackProjection?.matchedCurrentTask).toBe(false);
    expect(result.primarySuggestedTask?.id).toBe("account-project-review");
  });

  it("应构造基线摘要和 carry hint", () => {
    const currentTask = task();
    const inputValues: CuratedTaskInputValues = {
      project_goal: "AI 内容周报",
      existing_results: "已有结果包",
    };

    const result = buildActiveReviewBaselineModel({
      task: currentTask,
      selectedReferenceEntries: [],
      seededReferenceEntries: [referenceEntry()],
      inputValues,
      copy,
    });

    expect(result.activeReviewBaselineSnapshot?.sourceTitle).toContain(
      "AI 内容周报",
    );
    expect(result.activeReviewBaselineHighlights.join("\n")).toContain(
      "当前判断",
    );
    expect(result.activeReviewBaselineCarryHint).toBe(
      "复盘沿用：curatedTask.templates.account-project-review.fields.project_goal.label / curatedTask.templates.account-project-review.fields.existing_results.label",
    );
  });
});
