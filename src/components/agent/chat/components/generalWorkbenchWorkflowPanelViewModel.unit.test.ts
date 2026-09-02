import { describe, expect, it } from "vitest";
import type { SidebarActivityLog } from "../hooks/useThemeContextWorkspace";
import type { ReviewFeedbackProjection } from "../utils/reviewFeedbackProjection";
import {
  buildActivitySectionSummary,
  buildActivityStepSummary,
  buildActivitySummary,
  buildGeneralWorkbenchActivityLogProjection,
  buildGeneralWorkbenchActivitySectionProjection,
  buildGeneralWorkbenchBranchSectionProjection,
  buildGeneralWorkbenchCreationTaskGroupProjection,
  buildGeneralWorkbenchCreationTaskSectionProjection,
  buildGeneralWorkbenchFollowUpProjection,
  buildGeneralWorkbenchRunDetailProjection,
  buildGeneralWorkbenchWorkflowCurrentProjection,
  buildCreationTaskSectionSummary,
  buildCuratedTaskFollowUpActionItems,
  buildCuratedTaskFollowUpActionPayload,
  buildCuratedTaskFollowUpHintText,
  buildGeneralWorkbenchWorkflowPanelViewModel,
  buildReviewFeedbackFollowUpActionPayload,
  buildRunDetailSummaryText,
  buildWorkflowResultHandoffText,
  calculateWorkflowProgressPercent,
  countCompletedWorkflowSteps,
  formatActivitySourceLabel,
  formatActivityStatusLabel,
  formatCreationTaskCountLabel,
  formatGateLabel,
  formatRunIdShort,
  formatRunStatusLabel,
  getBranchMetaText,
  getBranchStatusText,
  getCreationTaskTitle,
  getPrimaryActivityLog,
  listVisibleCuratedTaskFollowUpActions,
  selectLatestReviewFeedbackSignal,
  type GeneralWorkbenchWorkflowPanelTranslate,
} from "./generalWorkbenchWorkflowPanelViewModel";
import type { GeneralWorkbenchRunMetadataSummary } from "./generalWorkbenchWorkflowData";

function activityLog(
  overrides: Partial<SidebarActivityLog>,
): SidebarActivityLog {
  return {
    id: "log-1",
    name: "write_file",
    status: "completed",
    timeLabel: "10:30",
    ...overrides,
  };
}

const t = ((key: string, values?: Record<string, unknown>) => {
  const templates: Record<string, string> = {
    "generalWorkbench.workflow.followUp.banner.current":
      "已按“{{action}}”把这轮结果带回输入区，可继续改写后发送。",
    "generalWorkbench.workflow.followUp.banner.withCurrentTask":
      "已按“{{action}}”接着推进「{{title}}」，可继续改写后发送。",
    "generalWorkbench.workflow.followUp.banner.withTarget":
      "已切到“{{title}}”这条下一步，并带着这轮结果继续进入生成。",
    "generalWorkbench.workflow.followUp.hint.summaryOnly":
      "这轮结果的常见下一步是：{{summary}}。",
    "generalWorkbench.workflow.followUp.hint.taskOnly":
      "当前结果来自「{{title}}」，可继续围绕这轮产出补充下一稿、拆成更多版本，或回到首页继续下一轮。",
    "generalWorkbench.workflow.followUp.hint.withTaskAndSummary":
      "按「{{title}}」这条结果模板的常见闭环，建议先做：{{summary}}。",
    "generalWorkbench.workflow.followUp.prompt.current":
      "请基于当前结果继续：{{action}}",
    "generalWorkbench.workflow.followUp.prompt.withTask":
      "请基于「{{title}}」这轮结果继续：{{action}}",
    "generalWorkbench.workflow.handoff.defaultContinuing":
      "主稿、任务文件和运行产物会继续收进下方“产出记录 / 执行经过”；需要继续改写时，可从{{branchTitle}}或首页“继续上次做法”接着跑。",
    "generalWorkbench.workflow.handoff.defaultInitial":
      "主稿、任务文件和运行产物会收进下方“产出记录 / 执行经过”；需要继续改写时，可从{{branchTitle}}或首页“继续上次做法”接着跑。",
    "generalWorkbench.workflow.handoff.withDestination":
      "{{destination}} 需要继续改写时，可从{{branchTitle}}或首页“继续上次做法”接着跑。",
    "generalWorkbench.workflow.outputs.summary.countLabel": "{{count}} 条记录",
    "generalWorkbench.workflow.outputs.copyPath": "复制路径",
    "generalWorkbench.workflow.outputs.empty": "暂无产出记录",
    "generalWorkbench.workflow.outputs.summary.emptyMeta":
      "后续生成的任务文件与结果索引会按类型留在这里。",
    "generalWorkbench.workflow.outputs.summary.emptyTitle":
      "最近还没有新的产出记录",
    "generalWorkbench.workflow.outputs.summary.latestTimeFallback": "最近",
    "generalWorkbench.workflow.outputs.summary.latestTitle":
      "最近一次：{{label}}",
    "generalWorkbench.workflow.outputs.summary.meta":
      "{{time}} · 共 {{count}} 条产出记录，按 {{groupCount}} 类归档。",
    "generalWorkbench.workflow.outputs.summary.untitledTask": "未命名任务",
    "generalWorkbench.workflow.reviewFeedback.followUpBanner":
      "已切到“{{title}}”这条下一步，并带着这轮结果继续进入生成。",
    "generalWorkbench.workflow.activity.gate.publishConfirm": "发布闸门",
    "generalWorkbench.workflow.activity.gate.topicSelect": "选题闸门",
    "generalWorkbench.workflow.activity.gate.writeMode": "写作闸门",
    "generalWorkbench.workflow.activity.source.skill": "技能",
    "generalWorkbench.workflow.activity.source.tool": "工具",
    "generalWorkbench.workflow.activity.source.workflow": "Workflow",
    "generalWorkbench.workflow.activity.status.failed": "失败",
    "generalWorkbench.workflow.activity.status.recorded": "已记录",
    "generalWorkbench.workflow.activity.status.running": "处理中",
    "generalWorkbench.workflow.activity.summary.activeRun": "当前查看 {{run}}",
    "generalWorkbench.workflow.activity.summary.artifactBadge":
      "{{count}} 个产物",
    "generalWorkbench.workflow.activity.summary.artifactCount":
      "{{count}} 个产物",
    "generalWorkbench.workflow.activity.summary.artifactPath": "产物 {{path}}",
    "generalWorkbench.workflow.activity.summary.emptyMeta":
      "运行过程会记录在这里。",
    "generalWorkbench.workflow.activity.summary.emptyTitle": "暂无执行经过",
    "generalWorkbench.workflow.activity.summary.latestTimeFallback": "最近",
    "generalWorkbench.workflow.activity.summary.latestTitle":
      "最近一组：{{name}}",
    "generalWorkbench.workflow.activity.summary.nameFallback": "未命名步骤",
    "generalWorkbench.workflow.activity.summary.stepCount": "{{count}} 个步骤",
    "generalWorkbench.workflow.activity.empty": "暂无执行经过",
    "generalWorkbench.workflow.activity.openArtifact": "打开",
    "generalWorkbench.workflow.activity.openArtifactAria":
      "打开产物路径-{{path}}",
    "generalWorkbench.workflow.activity.revealArtifact": "定位",
    "generalWorkbench.workflow.activity.revealArtifactAria":
      "定位产物路径-{{path}}",
    "generalWorkbench.workflow.activity.viewRun": "查看运行 {{run}}",
    "generalWorkbench.workflow.runDetail.loading": "运行详情加载中",
    "generalWorkbench.workflow.runDetail.title": "当前查看运行",
    "generalWorkbench.workflow.runDetail.status.canceled": "已取消",
    "generalWorkbench.workflow.runDetail.status.error": "失败",
    "generalWorkbench.workflow.runDetail.status.queued": "排队中",
    "generalWorkbench.workflow.runDetail.status.running": "处理中",
    "generalWorkbench.workflow.runDetail.status.success": "已完成",
    "generalWorkbench.workflow.runDetail.status.timeout": "超时",
    "generalWorkbench.workflow.runDetail.artifactCount": "{{count}} 个产物",
    "generalWorkbench.workflow.runDetail.copyArtifact": "复制",
    "generalWorkbench.workflow.runDetail.copyArtifactAria":
      "复制产物路径-{{path}}",
    "generalWorkbench.workflow.runDetail.copyId": "复制运行ID",
    "generalWorkbench.workflow.runDetail.copyIdAria": "复制运行ID",
    "generalWorkbench.workflow.runDetail.copyRaw": "复制原始记录",
    "generalWorkbench.workflow.runDetail.copyRawAria": "复制原始记录",
    "generalWorkbench.workflow.runDetail.openArtifact": "打开",
    "generalWorkbench.workflow.runDetail.openArtifactAria":
      "打开产物路径-{{path}}",
    "generalWorkbench.workflow.runDetail.revealArtifact": "定位",
    "generalWorkbench.workflow.runDetail.revealArtifactAria":
      "定位产物路径-{{path}}",
    "generalWorkbench.workflow.runDetail.workflowFailure": "失败原因",
    "generalWorkbench.workflow.runDetail.workflowRetry": "重试关联",
    "generalWorkbench.workflow.runDetail.workflowWaitingAction": "等待动作",
    "generalWorkbench.workflow.runDetail.waitingAction.askUser": "人工确认",
    "generalWorkbench.workflow.runDetail.waitingAction.elicitation": "补充信息",
    "generalWorkbench.workflow.runDetail.waitingAction.toolConfirmation":
      "工具确认",
    "generalWorkbench.workflow.runDetail.summary.artifactCount":
      "{{count}} 个产物",
    "generalWorkbench.workflow.runDetail.summary.artifactPath": "产物 {{path}}",
    "generalWorkbench.workflow.runDetail.summary.curatedTask":
      "结果模板 {{title}}",
    "generalWorkbench.workflow.runDetail.summary.empty": "暂无运行摘要",
    "generalWorkbench.workflow.runDetail.summary.workflow":
      "工作流 {{workflow}}",
    "generalWorkbench.workflow.branch.create.draft": "新建草稿",
    "generalWorkbench.workflow.branch.create.version": "留一版",
    "generalWorkbench.workflow.branch.currentFocus": "当前焦点",
    "generalWorkbench.workflow.branch.deleteAria": "删除分支",
    "generalWorkbench.workflow.branch.empty.draft": "暂无可继续草稿",
    "generalWorkbench.workflow.branch.empty.version": "暂无可继续版本",
    "generalWorkbench.workflow.branch.focusFirstHint":
      "切到当前焦点后再继续处理这一条记录",
    "generalWorkbench.workflow.branch.meta.candidate.draft": "候选草稿",
    "generalWorkbench.workflow.branch.meta.candidate.version": "候选版本",
    "generalWorkbench.workflow.branch.meta.current.draft":
      "当前焦点落在这份草稿",
    "generalWorkbench.workflow.branch.meta.current.version": "当前焦点落在这版",
    "generalWorkbench.workflow.branch.meta.inProgress.draft": "正在推进的草稿",
    "generalWorkbench.workflow.branch.meta.inProgress.version":
      "正在推进的版本",
    "generalWorkbench.workflow.branch.meta.merged.draft": "已收进主稿",
    "generalWorkbench.workflow.branch.meta.merged.version": "已设为主稿",
    "generalWorkbench.workflow.branch.meta.pending": "待继续",
    "generalWorkbench.workflow.branch.primaryAction.draft": "收进主稿",
    "generalWorkbench.workflow.branch.primaryAction.version": "设为主稿",
    "generalWorkbench.workflow.branch.secondaryAction": "稍后继续",
    "generalWorkbench.workflow.branch.sectionTitle.draft": "可继续草稿",
    "generalWorkbench.workflow.branch.sectionTitle.version": "可继续版本",
    "generalWorkbench.workflow.branch.status.candidate": "候选",
    "generalWorkbench.workflow.branch.status.inProgress": "进行中",
    "generalWorkbench.workflow.branch.status.merged": "已合并",
    "generalWorkbench.workflow.branch.status.pending": "待处理",
    "generalWorkbench.workflow.branch.summary.empty.draft":
      "还没有可继续草稿。",
    "generalWorkbench.workflow.branch.summary.empty.version":
      "还没有可继续版本。",
    "generalWorkbench.workflow.branch.summary.multiple.draft":
      "当前焦点落在「{{title}}」，还有 {{count}} 份相关草稿。",
    "generalWorkbench.workflow.branch.summary.multiple.version":
      "当前焦点落在「{{title}}」，还有 {{count}} 个可继续版本。",
    "generalWorkbench.workflow.branch.summary.single.draft":
      "当前焦点落在「{{title}}」。",
    "generalWorkbench.workflow.branch.summary.single.version":
      "当前焦点落在「{{title}}」。",
    "generalWorkbench.workflow.completed.allDoneHint":
      "当前流程已完成，可回看下方记录",
    "generalWorkbench.workflow.completed.collapsedWithRemaining":
      "已完成项已收起，优先聚焦当前与后续任务",
    "generalWorkbench.workflow.completed.count": "已完成 {{count}} 项",
    "generalWorkbench.workflow.current.allCompleted": "当前流程已全部完成",
    "generalWorkbench.workflow.current.completedTitle": "当前流程已完成",
    "generalWorkbench.workflow.current.emptyTitle": "等待创建第一条任务",
    "generalWorkbench.workflow.current.remaining": "剩余 {{count}} 项待处理",
    "generalWorkbench.workflow.queue.hiddenCount":
      "已展示 {{visible}} 项，另有 {{hidden}} 项",
    "generalWorkbench.workflow.queue.item": "后续 {{index}}",
    "generalWorkbench.workflow.queue.pendingCount": "{{count}} 项待处理",
    "reviewFeedback.suggestion.matchedCurrentTask":
      "这轮判断仍建议围绕当前这一步继续推进，可直接沿当前结果往下做。",
    "reviewFeedback.suggestion.matchedWithTitle":
      "这轮判断仍建议围绕「{{title}}」继续推进，可直接沿当前结果往下做。",
    "reviewFeedback.suggestion.suggestedTasks":
      "这轮判断更建议优先回到「{{titles}}」；需要切换时，可从首页“继续上次做法”接着跑。",
    "reviewFeedback.suggestion.taskTitleSeparator": "」或「",
  };
  return (templates[key] ?? key).replace(
    /\{\{\s*([^}]+?)\s*\}\}/g,
    (_, name: string) => String(values?.[name.trim()] ?? ""),
  );
}) as GeneralWorkbenchWorkflowPanelTranslate;

function curatedTask(
  overrides: Partial<
    NonNullable<GeneralWorkbenchRunMetadataSummary["curatedTask"]>
  > = {},
): NonNullable<GeneralWorkbenchRunMetadataSummary["curatedTask"]> {
  return {
    taskId: "daily-trend-briefing",
    taskTitle: "每日趋势摘要",
    resultDestination: "趋势摘要会先写回当前内容。",
    followUpActions: ["继续展开其中一个选题", "生成首条内容主稿"],
    ...overrides,
  };
}

describe("generalWorkbenchWorkflowPanelViewModel", () => {
  it("应计算完成步骤数和进度百分比", () => {
    const workflowSteps = [
      { id: "done", title: "完成提纲", status: "completed" as const },
      { id: "active", title: "撰写主稿", status: "active" as const },
      { id: "skipped", title: "跳过封面", status: "skipped" as const },
    ];

    expect(countCompletedWorkflowSteps(workflowSteps)).toBe(1);
    expect(
      calculateWorkflowProgressPercent({
        completedSteps: 1,
        totalSteps: 4,
      }),
    ).toBe(25);
    expect(
      calculateWorkflowProgressPercent({
        completedSteps: 0,
        totalSteps: 0,
      }),
    ).toBe(0);
  });

  it("应构造 workflow panel 的纯派生状态", () => {
    const viewModel = buildGeneralWorkbenchWorkflowPanelViewModel({
      workflowSteps: [
        { id: "done", title: "完成提纲", status: "completed" },
        { id: "active", title: "撰写主稿", status: "active" },
      ],
      activityLogs: [
        activityLog({
          id: "log-1",
          runId: "run-1",
          status: "running",
          artifactPaths: [" content-posts/demo.md "],
        }),
        activityLog({
          id: "log-2",
          runId: "run-1",
          status: "completed",
          artifactPaths: ["content-posts/demo-cover.png"],
        }),
      ],
      creationTaskEvents: [
        {
          taskId: "video-1",
          taskType: "video_generate",
          path: "tasks/video-1.json",
          createdAt: 1_740_000_000,
          timeLabel: "10:40",
        },
      ],
      activeRunMetadata: JSON.stringify({
        workflow: "social",
        stages: ["topic_select", "write_mode"],
        artifact_paths: ["content-posts/demo.md"],
      }),
    });

    expect(viewModel.completedSteps).toBe(1);
    expect(viewModel.progressPercent).toBe(50);
    expect(viewModel.groupedActivityLogs).toHaveLength(1);
    expect(viewModel.groupedActivityLogs[0]).toMatchObject({
      key: "run:run-1",
      status: "running",
      artifactPaths: ["content-posts/demo.md", "content-posts/demo-cover.png"],
    });
    expect(viewModel.groupedCreationTaskEvents[0]).toMatchObject({
      key: "video_generate",
      label: "视频生成",
      latestTimeLabel: "10:40",
    });
    expect(viewModel.activeRunStagesLabel).toBe("选题闸门 → 写作闸门");
    expect(viewModel.runMetadataSummary).toMatchObject({
      workflow: "social",
      stages: ["topic_select", "write_mode"],
      artifactPaths: ["content-posts/demo.md"],
    });
    expect(viewModel.runMetadataText).toContain('"workflow": "social"');
  });

  it("应构造当前工作流摘要、队列截断和完成提示", () => {
    const projection = buildGeneralWorkbenchWorkflowCurrentProjection({
      workflowSteps: [
        { id: "done", title: "完成提纲", status: "completed" },
        { id: "active", title: "撰写主稿", status: "active" },
        { id: "error", title: "修复封面", status: "error" },
        { id: "pending-1", title: "生成摘要", status: "pending" },
        { id: "pending-2", title: "准备发布", status: "pending" },
      ],
      completedSteps: 1,
      progressPercent: 125,
      t,
    });

    expect(projection.currentWorkflowStep?.id).toBe("active");
    expect(projection.currentStepTitle).toBe("撰写主稿");
    expect(projection.currentStepIconStatus).toBe("active");
    expect(projection.currentStepStatus).toBe("active");
    expect(projection.currentStepStatusLabel).toBe("进行中");
    expect(projection.remainingSteps).toBe(4);
    expect(projection.visibleQueueSteps.map((step) => step.id)).toEqual([
      "error",
      "pending-1",
    ]);
    expect(projection.queueItems).toEqual([
      {
        id: "error",
        title: "修复封面",
        status: "error",
        indexLabel: "后续 1",
        statusLabel: "异常",
      },
      {
        id: "pending-1",
        title: "生成摘要",
        status: "pending",
        indexLabel: "后续 2",
        statusLabel: "待处理",
      },
    ]);
    expect(projection.hiddenQueueCount).toBe(1);
    expect(projection.workflowSummaryText).toBe(
      "正在推进，后续还有 3 项待处理",
    );
    expect(projection.workflowProgressLabel).toBe("已完成 1/5");
    expect(projection.remainingText).toBe("剩余 4 项待处理");
    expect(projection.progressBarPercent).toBe(100);
    expect(projection.progressPercentLabel).toBe("100%");
    expect(projection.queueHeaderText).toBe("已展示 2 项，另有 1 项");
    expect(projection.completedCountText).toBe("已完成 1 项");
    expect(projection.completedHintText).toBe(
      "已完成项已收起，优先聚焦当前与后续任务",
    );
  });

  it("当前工作流为空或已完成时应返回稳定 fallback", () => {
    const emptyProjection = buildGeneralWorkbenchWorkflowCurrentProjection({
      workflowSteps: [],
      completedSteps: 0,
      progressPercent: -20,
      t,
    });

    expect(emptyProjection.currentWorkflowStep).toBeNull();
    expect(emptyProjection.currentStepTitle).toBe("当前流程已完成");
    expect(emptyProjection.currentStepIconStatus).toBe("active");
    expect(emptyProjection.currentStepStatus).toBe("completed");
    expect(emptyProjection.currentStepStatusLabel).toBe("已完成");
    expect(emptyProjection.workflowSummaryText).toBe("等待创建第一条任务");
    expect(emptyProjection.workflowProgressLabel).toBe("等待开始");
    expect(emptyProjection.remainingText).toBe("当前流程已全部完成");
    expect(emptyProjection.progressBarPercent).toBe(0);
    expect(emptyProjection.queueHeaderText).toBeNull();
    expect(emptyProjection.completedCountText).toBeNull();
    expect(emptyProjection.completedHintText).toBeNull();

    const completedProjection = buildGeneralWorkbenchWorkflowCurrentProjection({
      workflowSteps: [
        { id: "done", title: "完成主稿", status: "completed" },
        { id: "skipped", title: "跳过封面", status: "skipped" },
      ],
      completedSteps: 1,
      progressPercent: 50,
      t,
    });

    expect(completedProjection.currentWorkflowStep).toBeNull();
    expect(completedProjection.workflowSummaryText).toBe("当前流程已完成");
    expect(completedProjection.visibleQueueSteps).toEqual([]);
    expect(completedProjection.queueItems).toEqual([]);
    expect(completedProjection.completedWorkflowSteps).toBe(1);
    expect(completedProjection.completedCountText).toBe("已完成 1 项");
    expect(completedProjection.completedHintText).toBe(
      "当前流程已完成，可回看下方记录",
    );
  });

  it("应构造产出记录摘要、handoff 文案和任务标题", () => {
    const groups = [
      {
        key: "typesetting",
        taskType: "typesetting",
        label: "排版优化",
        latestTimeLabel: "10:22",
        tasks: [],
      },
    ];

    expect(
      buildCreationTaskSectionSummary({
        groups,
        totalCount: 3,
        t,
      }),
    ).toEqual({
      title: "最近一次：排版优化",
      meta: "10:22 · 共 3 条产出记录，按 1 类归档。",
    });
    expect(formatCreationTaskCountLabel(2, t)).toBe("2 条记录");
    expect(getCreationTaskTitle(" .lime/tasks/image_generate/a.json ", t)).toBe(
      "a.json",
    );
    expect(getCreationTaskTitle("   ", t)).toBe("未命名任务");
    expect(
      buildGeneralWorkbenchCreationTaskGroupProjection({
        group: {
          key: "typesetting",
          taskType: "typesetting",
          label: "排版优化",
          latestTimeLabel: "10:22",
          tasks: [
            {
              taskId: "task-1",
              taskType: "typesetting",
              path: ".lime/tasks/typesetting/task-1.json",
              absolutePath: "/workspace/.lime/tasks/typesetting/task-1.json",
              createdAt: 1,
              timeLabel: "10:22",
            },
            {
              taskId: "task-2",
              taskType: "image_generate",
              path: "tasks/task-2.json",
              createdAt: 2,
              timeLabel: "10:24",
            },
          ],
        },
        t,
      }),
    ).toMatchObject({
      key: "typesetting",
      label: "排版优化",
      countLabel: "2 条记录",
      latestTimeLabel: "10:22",
      tasks: [
        {
          key: "task-1-.lime/tasks/typesetting/task-1.json",
          title: "task-1.json",
          copyTarget: "/workspace/.lime/tasks/typesetting/task-1.json",
          copyAriaLabel:
            "generalWorkbench.workflow.outputs.copyAbsolutePathAria",
        },
        {
          key: "task-2-tasks/task-2.json",
          title: "task-2.json",
          copyTarget: "tasks/task-2.json",
          copyAriaLabel: "generalWorkbench.workflow.outputs.copyPathAria",
        },
      ],
    });
    expect(
      buildGeneralWorkbenchCreationTaskSectionProjection({
        groups: [],
        t,
      }),
    ).toEqual({
      emptyText: "暂无产出记录",
      copyLabel: "复制路径",
      groups: [],
    });
    expect(
      buildGeneralWorkbenchCreationTaskSectionProjection({
        groups: [
          {
            key: "typesetting",
            taskType: "typesetting",
            label: "排版优化",
            latestTimeLabel: "10:22",
            tasks: [
              {
                taskId: "task-1",
                taskType: "typesetting",
                path: ".lime/tasks/typesetting/task-1.json",
                absolutePath: "/workspace/.lime/tasks/typesetting/task-1.json",
                createdAt: 1,
                timeLabel: "10:22",
              },
            ],
          },
        ],
        t,
      }),
    ).toMatchObject({
      emptyText: "暂无产出记录",
      copyLabel: "复制路径",
      groups: [
        {
          key: "typesetting",
          tasks: [
            {
              title: "task-1.json",
              copyTarget: "/workspace/.lime/tasks/typesetting/task-1.json",
            },
          ],
        },
      ],
    });
    expect(
      buildWorkflowResultHandoffText({
        branchSectionTitle: "可继续版本",
        hasRecordedOutputs: false,
        t,
      }),
    ).toContain("会收进下方");
    expect(
      buildWorkflowResultHandoffText({
        branchSectionTitle: "可继续版本",
        hasRecordedOutputs: true,
        resultDestination: "趋势摘要会先写回当前内容。",
        t,
      }),
    ).toBe(
      "趋势摘要会先写回当前内容。 需要继续改写时，可从可继续版本或首页“继续上次做法”接着跑。",
    );
  });

  it("应构造分支区排序、文案和当前分支摘要", () => {
    const projection = buildGeneralWorkbenchBranchSectionProjection({
      branchItems: [
        {
          id: "merged",
          title: "已合并版本",
          status: "merged",
          isCurrent: false,
        },
        {
          id: "current",
          title: "当前版本",
          status: "in_progress",
          isCurrent: true,
        },
        {
          id: "pending",
          title: "待处理版本",
          status: "pending",
          isCurrent: false,
        },
      ],
      isVersionMode: true,
      t,
    });

    expect(projection).toMatchObject({
      sectionTitle: "可继续版本",
      createLabel: "留一版",
      primaryActionLabel: "设为主稿",
      secondaryActionLabel: "稍后继续",
      currentBranchItem: {
        id: "current",
      },
      secondaryBranchCount: 2,
      summaryText: "当前焦点落在「当前版本」，还有 2 个可继续版本。",
      emptyText: "暂无可继续版本",
    });
    expect(projection.sortedBranchItems.map((item) => item.id)).toEqual([
      "current",
      "pending",
      "merged",
    ]);
    expect(projection.itemProjections).toMatchObject([
      {
        id: "current",
        title: "当前版本",
        status: "in_progress",
        isCurrent: true,
        statusLabel: "当前焦点",
        metaText: "当前焦点落在这版",
        deleteAriaLabel: null,
        hintText: null,
        actionItems: [
          {
            kind: "primary",
            status: "merged",
            label: "设为主稿",
          },
          {
            kind: "secondary",
            status: "pending",
            label: "稍后继续",
          },
        ],
      },
      {
        id: "pending",
        statusLabel: "待处理",
        metaText: "待继续",
        deleteAriaLabel: null,
        hintText: "切到当前焦点后再继续处理这一条记录",
        actionItems: [],
      },
      {
        id: "merged",
        statusLabel: "已合并",
        metaText: "已设为主稿",
        deleteAriaLabel: null,
        hintText: "切到当前焦点后再继续处理这一条记录",
        actionItems: [],
      },
    ]);
    expect(getBranchStatusText("candidate", t)).toBe("候选");
    expect(
      getBranchMetaText({
        item: projection.sortedBranchItems[0],
        isVersionMode: true,
        t,
      }),
    ).toBe("当前焦点落在这版");

    const draftProjection = buildGeneralWorkbenchBranchSectionProjection({
      branchItems: [
        {
          id: "draft",
          title: "候选草稿",
          status: "candidate",
          isCurrent: false,
        },
      ],
      isVersionMode: false,
      t,
    });
    expect(draftProjection.itemProjections[0]).toMatchObject({
      id: "draft",
      statusLabel: "候选",
      metaText: "候选草稿",
      deleteAriaLabel: "删除分支",
      hintText: "切到当前焦点后再继续处理这一条记录",
      actionItems: [],
    });
  });

  it("应构造当前模板内的建议下一步文案和 payload", () => {
    const currentTask = curatedTask({
      followUpActions: [
        "继续展开其中一个选题",
        "继续展开其中一个选题",
        "生成首条内容主稿",
      ],
      referenceMemoryIds: ["memory-1"],
      referenceEntries: [
        {
          id: "memory-1",
          sourceKind: "memory",
          title: "品牌风格样本",
          summary: "保留轻盈但专业的表达。",
          category: "context",
          categoryLabel: "参考",
          tags: ["品牌", "语气"],
        },
      ],
      launchInputValues: {
        theme_target: "AI 内容创作",
      },
    });

    expect(listVisibleCuratedTaskFollowUpActions(currentTask)).toEqual([
      "继续展开其中一个选题",
      "生成首条内容主稿",
    ]);
    expect(
      buildCuratedTaskFollowUpActionItems({ curatedTask: null, t }),
    ).toEqual([]);
    expect(
      buildCuratedTaskFollowUpActionItems({
        curatedTask: currentTask,
        t,
      }),
    ).toMatchObject([
      {
        action: "继续展开其中一个选题",
        ariaLabel: "generalWorkbench.workflow.followUp.applyAria",
        payload: {
          prompt: "请基于「每日趋势摘要」这轮结果继续：继续展开其中一个选题",
          capabilityRoute: {
            kind: "curated_task",
            taskId: "daily-trend-briefing",
          },
        },
      },
      {
        action: "生成首条内容主稿",
        ariaLabel: "generalWorkbench.workflow.followUp.applyAria",
        payload: {
          prompt: "请基于「每日趋势摘要」这轮结果继续：生成首条内容主稿",
          capabilityRoute: {
            kind: "curated_task",
            taskId: "daily-trend-briefing",
          },
        },
      },
    ]);
    expect(buildCuratedTaskFollowUpHintText(currentTask, t)).toContain(
      "按「每日趋势摘要」这条结果模板",
    );

    expect(
      buildCuratedTaskFollowUpActionPayload({
        action: "继续展开其中一个选题",
        curatedTask: currentTask,
        t,
      }),
    ).toMatchObject({
      prompt: "请基于「每日趋势摘要」这轮结果继续：继续展开其中一个选题",
      bannerMessage:
        "已按“继续展开其中一个选题”接着推进「每日趋势摘要」，可继续改写后发送。",
      capabilityRoute: {
        kind: "curated_task",
        taskId: "daily-trend-briefing",
        taskTitle: "每日趋势摘要",
        launchInputValues: {
          theme_target: "AI 内容创作",
        },
        referenceMemoryIds: ["memory-1"],
      },
    });
  });

  it("复盘建议下一步应切到目标模板并合并 reference prefill", () => {
    const payload = buildCuratedTaskFollowUpActionPayload({
      action: "生成下一轮内容方案",
      curatedTask: curatedTask({
        taskId: "account-project-review",
        taskTitle: "复盘这个账号/项目",
        launchInputValues: {
          target_audience: "关注 AI 内容的品牌运营",
        },
        referenceEntries: [
          {
            id: "memory-review-1",
            sourceKind: "memory",
            title: "本周账号复盘线索",
            summary: "封面信息过密，转化动作不够聚焦。",
            category: "experience",
            categoryLabel: "成果",
            tags: ["复盘", "增长"],
            taskPrefillByTaskId: {
              "social-post-starter": {
                subject_or_product:
                  "基于本周账号复盘，整理下一轮内容方向与重点动作。",
              },
            },
          },
        ],
      }),
      t,
    });

    expect(payload).toMatchObject({
      bannerMessage:
        "已切到“内容主稿生成”这条下一步，并带着这轮结果继续进入生成。",
      capabilityRoute: {
        kind: "curated_task",
        taskId: "social-post-starter",
        taskTitle: "内容主稿生成",
        launchInputValues: {
          subject_or_product:
            "基于本周账号复盘，整理下一轮内容方向与重点动作。",
          target_audience: "关注 AI 内容的品牌运营",
        },
      },
    });
    expect(payload?.prompt).toContain(
      "请承接这轮判断结论，直接生成下一轮最值得执行的内容方案。",
    );
    expect(payload?.prompt).toContain(
      "主题或产品信息：基于本周账号复盘，整理下一轮内容方向与重点动作。",
    );
  });

  it("review feedback payload 应优先沿 结果基线切到下游任务", () => {
    const projection: ReviewFeedbackProjection = {
      signal: {
        source: "review_feedback",
        category: "experience",
        title: "AI 内容周报 · 转成主稿",
        summary: "这轮判断已经清楚。",
        tags: ["复盘"],
        preferredTaskIds: ["social-post-starter"],
        createdAt: 1,
      },
      matchedCurrentTask: false,
      suggestedTasks: [
        {
          taskId: "social-post-starter",
          title: "内容主稿生成",
        },
      ],
      suggestedTaskTitles: ["内容主稿生成"],
      suggestionText: "这轮判断更建议优先回到「内容主稿生成」。",
    };

    const payload = buildReviewFeedbackFollowUpActionPayload({
      projection,
      curatedTask: curatedTask({
        taskId: "account-project-review",
        taskTitle: "复盘这个账号/项目",
        referenceEntries: [
          {
            id: "memory:ai-weekly:run:1",
            sourceKind: "memory",
            title: "AI 内容周报",
            summary: "当前已有一轮结果，可直接进入下游主稿。",
            category: "experience",
            categoryLabel: "成果",
            tags: ["复盘", "周报"],
            taskPrefillByTaskId: {
              "account-project-review": {
                project_goal: "AI 内容周报",
                existing_results:
                  "当前判断：适合继续放量 当前卡点：封面信息过密 经营动作：保留品牌联名方向 更适合去向：内容主稿生成",
              },
            },
          },
        ],
      }),
      t,
    });

    expect(payload).toMatchObject({
      bannerMessage:
        "已切到“内容主稿生成”这条下一步，并带着当前结果继续进入生成。",
      capabilityRoute: {
        kind: "curated_task",
        taskId: "social-post-starter",
        taskTitle: "内容主稿生成",
        referenceEntries: [
          expect.objectContaining({
            sourceKind: "memory",
            title: "AI 内容周报",
          }),
        ],
      },
    });
    expect(payload?.prompt).toContain("当前结果基线：AI 内容周报");
  });

  it("应选择最新 review feedback 并构造建议下一步投影", () => {
    const latestSignal = selectLatestReviewFeedbackSignal([
      {
        source: "memory_reference",
        category: "experience",
        title: "灵感记录",
        summary: "不是 review feedback。",
        tags: [],
        preferredTaskIds: ["viral-content-breakdown"],
        createdAt: 3,
      },
      {
        source: "review_feedback",
        category: "experience",
        title: "较早复盘",
        summary: "旧判断。",
        tags: ["复盘"],
        preferredTaskIds: ["account-project-review"],
        createdAt: 1,
      },
      {
        source: "review_feedback",
        category: "experience",
        title: "最新复盘",
        summary: "建议回到内容主稿。",
        tags: ["增长"],
        preferredTaskIds: ["social-post-starter"],
        createdAt: 5,
      },
    ]);

    expect(latestSignal?.title).toBe("最新复盘");

    const projection = buildGeneralWorkbenchFollowUpProjection({
      latestReviewSignal: latestSignal,
      runMetadataSummary: {
        workflow: "social_content_pipeline_v1",
        executionId: "exec-1",
        versionId: "ver-1",
        stages: ["write_mode"],
        artifactPaths: ["content-posts/demo.md"],
        curatedTask: curatedTask({
          taskId: "account-project-review",
          taskTitle: "复盘这个账号/项目",
          referenceEntries: [
            {
              id: "memory:ai-weekly:run:1",
              sourceKind: "memory",
              title: "AI 内容周报",
              summary: "当前已有一轮结果，可直接进入下游主稿。",
              category: "experience",
              categoryLabel: "成果",
              tags: ["复盘"],
              taskPrefillByTaskId: {
                "account-project-review": {
                  project_goal: "AI 内容周报",
                  existing_results:
                    "当前判断：适合继续放量 当前卡点：封面信息过密 经营动作：保留品牌联名方向 更适合去向：内容主稿生成",
                },
              },
            },
          ],
        }),
      },
      t,
    });

    expect(projection.shouldShowFollowUpHint).toBe(true);
    expect(projection.reviewFeedbackProjection).toMatchObject({
      matchedCurrentTask: false,
      suggestedTaskTitles: ["内容主稿生成"],
      suggestionText:
        "这轮判断更建议优先回到「内容主稿生成」；需要切换时，可从首页“继续上次做法”接着跑。",
    });
    expect(projection.reviewFeedbackFollowUpActionPayload).toMatchObject({
      bannerMessage:
        "已切到“内容主稿生成”这条下一步，并带着当前结果继续进入生成。",
      capabilityRoute: {
        kind: "curated_task",
        taskId: "social-post-starter",
        taskTitle: "内容主稿生成",
      },
    });
    expect(projection.resultBaselineSnapshot).toMatchObject({
      sourceTitle: "AI 内容周报",
      statusLabel: "适合继续放量",
      failureSignalLabel: "封面信息过密",
      operatingAction: "保留品牌联名方向",
      destinationsLabel: "内容主稿生成",
    });
    expect(projection.resultBaselineHighlights.length).toBeGreaterThan(0);
    const downstreamProjection = buildGeneralWorkbenchFollowUpProjection({
      latestReviewSignal: null,
      runMetadataSummary: {
        workflow: "social_content_pipeline_v1",
        executionId: "exec-2",
        versionId: "ver-2",
        stages: ["write_mode"],
        artifactPaths: ["content-posts/downstream.md"],
        curatedTask: curatedTask({
          taskId: "daily-trend-briefing",
          taskTitle: "每日趋势摘要",
          referenceEntries: [
            {
              id: "memory:ai-weekly:run:2",
              sourceKind: "memory",
              title: "AI 内容周报",
              summary: "当前已有一轮结果，可直接进入下游摘要。",
              category: "experience",
              categoryLabel: "成果",
              tags: ["复盘"],
              taskPrefillByTaskId: {
                "account-project-review": {
                  project_goal: "AI 内容周报",
                  existing_results:
                    "当前判断：适合继续放量 经营动作：保留品牌联名方向 更适合去向：内容主稿生成 / 渠道改写",
                },
              },
            },
          ],
        }),
      },
      t,
    });
    expect(downstreamProjection.resultBaselineSnapshot).toMatchObject({
      sourceTitle: "AI 内容周报",
      statusLabel: "适合继续放量",
      operatingAction: "保留品牌联名方向",
      destinationsLabel: "内容主稿生成 / 渠道改写",
    });
    expect(downstreamProjection.shouldShowFollowUpHint).toBe(true);
    expect(projection.curatedTaskFollowUpHintText).toContain(
      "按「复盘这个账号/项目」这条结果模板",
    );
  });

  it("应构造活动日志摘要、标签和当前运行提示", () => {
    const groups = buildGeneralWorkbenchWorkflowPanelViewModel({
      workflowSteps: [],
      activityLogs: [
        activityLog({
          id: "log-run-1",
          name: "research_topic",
          status: "completed",
          timeLabel: "10:20",
          runId: "run-abcdef123456",
          gateKey: "topic_select",
          source: "skill",
          artifactPaths: ["content-posts/research.md"],
        }),
        activityLog({
          id: "log-run-2",
          name: "write_file",
          status: "completed",
          timeLabel: "10:21",
          runId: "run-abcdef123456",
          gateKey: "write_mode",
          source: "tool",
        }),
      ],
      creationTaskEvents: [],
      activeRunMetadata: null,
    }).groupedActivityLogs;

    expect(formatGateLabel(t, "topic_select")).toBe("选题闸门");
    expect(formatGateLabel(t, "idle")).toBeNull();
    expect(formatRunIdShort("run-abcdef123456")).toBe("run-abcd…");
    expect(formatRunStatusLabel(t, "running")).toBe("处理中");
    expect(formatActivityStatusLabel(t, "completed")).toBe("已记录");
    expect(formatActivitySourceLabel(t, "skill")).toBe("技能");
    expect(formatActivitySourceLabel(t, "custom")).toBe("custom");
    expect(getPrimaryActivityLog(groups[0])?.name).toBe("research_topic");
    expect(
      buildActivityStepSummary({
        ...groups[0].logs[0],
        inputSummary: "读取选题",
        outputSummary: "生成研究记录",
      }),
    ).toBe("读取选题 → 生成研究记录");
    expect(
      buildActivitySummary(groups[0], formatGateLabel(t, groups[0].gateKey), t),
    ).toBe("选题闸门 · 产物 content-posts/research.md · 2 个步骤");
    expect(
      buildGeneralWorkbenchActivityLogProjection({
        group: groups[0],
        t,
      }),
    ).toMatchObject({
      key: "run:run-abcdef123456",
      status: "completed",
      statusLabel: "已记录",
      title: "research_topic",
      timeLabel: "10:20",
      sourceLabel: "技能",
      gateLabel: "选题闸门",
      stepCountLabel: "2 个步骤",
      artifactCountLabel: "1 个产物",
      summary: "选题闸门 · 产物 content-posts/research.md · 2 个步骤",
      runId: "run-abcdef123456",
      runLabel: "run-abcd…",
      artifactPaths: [],
      runAction: {
        runId: "run-abcdef123456",
        label: "查看运行 run-abcd…",
      },
      artifactActions: [],
      sessionId: null,
      steps: [
        {
          id: "log-run-1",
          name: "research_topic",
          timeLabel: "10:20",
          summary: null,
        },
        {
          id: "log-run-2",
          name: "write_file",
          timeLabel: "10:21",
          summary: null,
        },
      ],
    });
    expect(
      buildGeneralWorkbenchActivityLogProjection({
        group: {
          key: "artifact-only",
          status: "completed",
          source: "tool",
          timeLabel: "10:40",
          sessionId: "session-1",
          artifactPaths: ["content-posts/research.md"],
          logs: [
            activityLog({
              id: "log-artifact",
              name: "write_file",
              timeLabel: "10:40",
              artifactPaths: ["content-posts/research.md"],
            }),
          ],
        },
        t,
      }),
    ).toMatchObject({
      runId: null,
      runAction: null,
      artifactPaths: ["content-posts/research.md"],
      artifactActions: [
        {
          path: "content-posts/research.md",
          sessionId: "session-1",
          actions: [
            {
              kind: "reveal",
              label: "定位",
              ariaLabel: "定位产物路径-content-posts/research.md",
              targetPath: "content-posts/research.md",
            },
            {
              kind: "open",
              label: "打开",
              ariaLabel: "打开产物路径-content-posts/research.md",
              targetPath: "content-posts/research.md",
            },
          ],
        },
      ],
    });
    expect(
      buildGeneralWorkbenchActivitySectionProjection({
        groups: [],
        t,
      }),
    ).toEqual({
      emptyText: "暂无执行经过",
      loadingText: "运行详情加载中",
      runDetailTitle: "当前查看运行",
      logs: [],
    });
    expect(
      buildGeneralWorkbenchActivitySectionProjection({
        groups,
        t,
      }),
    ).toMatchObject({
      emptyText: "暂无执行经过",
      loadingText: "运行详情加载中",
      runDetailTitle: "当前查看运行",
      logs: [
        {
          key: "run:run-abcdef123456",
          runAction: {
            runId: "run-abcdef123456",
            label: "查看运行 run-abcd…",
          },
          artifactActions: [],
        },
      ],
    });
    expect(
      buildActivitySectionSummary({
        groups,
        activeRunDetail: { id: "run-abcdef123456" },
        t,
      }),
    ).toEqual({
      title: "最近一组：research_topic",
      meta: "10:20 · 已记录 · 技能 · 选题闸门 · 2 个步骤 · 1 个产物 · 当前查看 run-abcd…",
    });
  });

  it("应构造运行详情 summary fallback 和组合摘要", () => {
    expect(
      buildRunDetailSummaryText({
        runMetadataSummary: {
          workflow: null,
          executionId: null,
          versionId: null,
          stages: [],
          artifactPaths: [],
          curatedTask: null,
        },
        t,
      }),
    ).toBe("暂无运行摘要");

    expect(
      buildRunDetailSummaryText({
        runMetadataSummary: {
          workflow: "social_content_pipeline_v1",
          executionId: "exec-1",
          versionId: "ver-1",
          stages: ["write_mode"],
          artifactPaths: [
            "content-posts/demo.md",
            "content-posts/demo.publish-pack.json",
          ],
          curatedTask: curatedTask({
            taskTitle: "每日趋势摘要",
          }),
        },
        activeRunStagesLabel: "写作闸门",
        t,
      }),
    ).toBe(
      "结果模板 每日趋势摘要 · 写作闸门 · 工作流 social_content_pipeline_v1 · 2 个产物",
    );
    const runDetailProjection = buildGeneralWorkbenchRunDetailProjection({
      activeRunDetail: {
        id: "run-123456789",
        source: "skill",
        status: "success",
      },
      runMetadataSummary: {
        workflow: "social_content_pipeline_v1",
        executionId: "exec-1",
        versionId: "ver-1",
        stages: ["write_mode"],
        artifactPaths: ["content-posts/demo.md"],
        curatedTask: curatedTask({
          taskTitle: "每日趋势摘要",
        }),
      },
      runMetadataText: '{"workflow":"social_content_pipeline_v1"}',
      activeRunStagesLabel: "写作闸门",
      t,
    });

    expect(runDetailProjection).toEqual({
      id: "run-123456789",
      status: "success",
      statusLabel: "已完成",
      sourceLabel: "技能",
      badges: [
        "技能",
        "social_content_pipeline_v1",
        "每日趋势摘要",
        "1 个产物",
      ],
      summary:
        "结果模板 每日趋势摘要 · 写作闸门 · 工作流 social_content_pipeline_v1 · 产物 content-posts/demo.md",
      detailRows: [],
      actions: [
        {
          kind: "copy_id",
          label: "复制运行ID",
          ariaLabel: "复制运行ID",
          copyTarget: "run-123456789",
        },
        {
          kind: "copy_raw",
          label: "复制原始记录",
          ariaLabel: "复制原始记录",
          copyTarget: '{"workflow":"social_content_pipeline_v1"}',
        },
      ],
      artifactPaths: ["content-posts/demo.md"],
      artifacts: [
        {
          path: "content-posts/demo.md",
          actions: [
            {
              kind: "copy",
              label: "复制",
              ariaLabel: "复制产物路径-content-posts/demo.md",
              targetPath: "content-posts/demo.md",
            },
            {
              kind: "reveal",
              label: "定位",
              ariaLabel: "定位产物路径-content-posts/demo.md",
              targetPath: "content-posts/demo.md",
            },
            {
              kind: "open",
              label: "打开",
              ariaLabel: "打开产物路径-content-posts/demo.md",
              targetPath: "content-posts/demo.md",
            },
          ],
        },
      ],
    });

    const workflowRunDetailProjection =
      buildGeneralWorkbenchRunDetailProjection({
        activeRunDetail: {
          id: "workflow-run-1",
          source: "automation",
          status: "error",
        },
        runMetadataSummary: {
          workflow: "content_article_workflow",
          executionId: null,
          versionId: null,
          stages: [],
          artifactPaths: [],
          curatedTask: null,
        },
        runMetadataText: JSON.stringify({
          source: "workflow/read",
          workflow_read_model: {
            workflowRunId: "workflow-run-1",
            failure: {
              reasonCode: "worker_output",
            },
            retry: {
              sourceTurnId: "turn-source",
              rescheduledTurnId: "turn-retry",
              reasonCode: "manual_retry",
            },
            actions: [
              {
                actionType: "ask_user",
                requestId: "request-1",
                stepId: "approval",
              },
            ],
            steps: [
              {
                id: "draft",
                title: "起草正文",
                status: "failed",
                failure: {
                  message: "正文为空",
                },
              },
            ],
          },
        }),
        t,
      });

    expect(workflowRunDetailProjection.detailRows).toEqual([
      {
        key: "workflow-failure",
        label: "失败原因",
        value: "worker_output",
      },
      {
        key: "workflow-retry",
        label: "重试关联",
        value: "turn-source -> turn-retry · manual_retry",
      },
      {
        key: "workflow-waiting-action",
        label: "等待动作",
        value: "人工确认 / request-1 / approval",
      },
    ]);
  });

  it("应优先用 agentActionType 展示 workflow waiting action 类型", () => {
    const workflowRunDetailProjection =
      buildGeneralWorkbenchRunDetailProjection({
        activeRunDetail: {
          id: "workflow-run-1",
          source: "automation",
          status: "running",
        },
        runMetadataSummary: {
          workflow: "content_article_workflow",
          executionId: null,
          versionId: null,
          stages: [],
          artifactPaths: [],
          curatedTask: null,
        },
        runMetadataText: JSON.stringify({
          source: "workflow/read",
          workflow_read_model: {
            workflowRunId: "workflow-run-1",
            actions: [
              {
                actionType: "respond",
                agentActionType: "tool_confirmation",
                requestId: "request-tool-1",
                stepId: "approve-tool",
              },
            ],
            steps: [
              {
                id: "collect-input",
                title: "补充信息",
                status: "waiting_permission",
                requestId: "request-elicitation-1",
                agentActionType: "elicitation",
              },
            ],
          },
        }),
        t,
      });

    expect(workflowRunDetailProjection.detailRows).toEqual([
      {
        key: "workflow-waiting-action",
        label: "等待动作",
        value: "工具确认 / request-tool-1 / approve-tool",
      },
    ]);
  });
});
