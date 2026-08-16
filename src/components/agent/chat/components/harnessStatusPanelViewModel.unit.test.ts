import { beforeEach, describe, expect, it } from "vitest";
import type {
  AgentRuntimeToolInventoryCatalogEntry,
  AgentRuntimeToolInventoryNativeEntry,
  AgentRuntimeToolInventoryRuntimeEntry,
} from "@/lib/api/agentRuntime/toolInventoryTypes";
import type {
  HarnessActiveFileWrite,
  HarnessFileEvent,
  HarnessOutputSignal,
  HarnessSessionState,
} from "../utils/harnessState";
import type { ActionRequired } from "../types";

import {
  buildFileChangeReviewEntries,
  buildFileChangeReviewDiffSummary,
  buildFileFilterOptions,
  buildFilteredFileEvents,
  buildFilteredOutputSignals,
  buildFileReviewSummaryTextParts,
  buildOutputFilterOptions,
  buildOutputSignalDiffSummary,
  buildOutputStatusDescriptors,
  buildRuntimeFactSummary,
  buildRuntimeSummaryText,
  buildRuntimeTaskPresentation,
  buildRuntimeToolCapabilityGaps,
  buildToolInventorySourceStats,
  collectCatalogExecutionSources,
  collectNativeExecutionSources,
  countFileChangeStatuses,
  countCatalogToolsByInventoryFilter,
  describeApproval,
  describeAction,
  describeKind,
  findFirstUrl,
  formatExecutionRestrictionProfileLabel,
  formatExecutionSandboxProfileLabel,
  formatExecutionSourceLabel,
  formatExecutionWarningPolicyLabel,
  formatExtensionSourceKindLabel,
  formatHarnessTime,
  formatRuntimePhaseLabel,
  formatRuntimeProgressLabel,
  formatRuntimeToolAvailabilitySourceLabel,
  formatRuntimeToolSourceKindLabel,
  formatToolLifecycleLabel,
  formatToolPermissionPlaneLabel,
  formatToolSourceKindLabel,
  formatWriteSourceLabel,
  getActiveWriteDescription,
  getFileName,
  getOutputSignalPaths,
  getSignalPath,
  groupHarnessFileEvents,
  groupHarnessOutputSignals,
  isLikelyFilePath,
  isNoisyRuntimeOutputText,
  isSearchOutputSignal,
  joinDisplayParts,
  matchesCatalogToolInventoryFilter,
  matchesOutputFilter,
  normalizeUrlCandidate,
  pickCommandFromArguments,
  pickPathFromArguments,
  resolveApprovalRiskLabelKey,
  resolveApprovalActionLabelKey,
  resolveApprovalRiskKind,
  resolveDiffReviewStatusLabelKey,
  resolveExecutionSourceVariant,
  resolveFileChangeStatusLabelKey,
  resolveFileReviewActionLabelKey,
  resolveFileReviewKindLabelKey,
  resolveFileReviewPhaseLabelKey,
  resolveOutputCardPresentation,
  resolveOutputPathLabelKey,
  resolveFriendlyToolLabel,
  resolveRuntimeStatusLabel,
  resolveRuntimeStepStatus,
  sortRuntimeToolsByVisibility,
  splitTextIntoSegments,
  summarizeFileActions,
} from "./harnessStatusPanelViewModel";

function buildFileEvent(
  overrides: Partial<HarnessFileEvent> = {},
): HarnessFileEvent {
  return {
    id: "event-1",
    toolCallId: "tool-1",
    path: "/tmp/workspace/src/app.ts",
    displayName: "app.ts",
    kind: "code",
    action: "write",
    sourceToolName: "Write",
    timestamp: new Date("2026-05-26T11:00:00.000Z"),
    clickable: true,
    ...overrides,
  };
}

function buildOutputSignal(
  overrides: Partial<HarnessOutputSignal> = {},
): HarnessOutputSignal {
  return {
    id: "signal-1",
    toolCallId: "tool-1",
    toolName: "bash",
    title: "工具输出",
    summary: "已完成",
    ...overrides,
  };
}

function buildRuntimeStatus(
  overrides: Partial<NonNullable<HarnessSessionState["runtimeStatus"]>> = {},
): NonNullable<HarnessSessionState["runtimeStatus"]> {
  return {
    phase: "routing",
    title: "执行任务",
    detail: "",
    ...overrides,
  };
}

function buildCatalogTool(
  overrides: Partial<AgentRuntimeToolInventoryCatalogEntry> = {},
): AgentRuntimeToolInventoryCatalogEntry {
  return {
    name: "ReadFile",
    profiles: ["core"],
    capabilities: ["workspace_io"],
    lifecycle: "current",
    source: "agent_builtin",
    permission_plane: "session_allowlist",
    workspace_default_allow: true,
    execution_warning_policy: "none",
    execution_warning_policy_source: "default",
    execution_restriction_profile: "none",
    execution_restriction_profile_source: "default",
    execution_sandbox_profile: "none",
    execution_sandbox_profile_source: "default",
    ...overrides,
  };
}

function buildNativeTool(
  overrides: Partial<AgentRuntimeToolInventoryNativeEntry> = {},
): AgentRuntimeToolInventoryNativeEntry {
  return {
    name: "ReadFile",
    description: "读取文件",
    deferred_loading: false,
    always_visible: false,
    allowed_callers: [],
    tags: [],
    input_examples_count: 0,
    has_output_schema: false,
    caller_allowed: true,
    visible_in_context: true,
    ...overrides,
  };
}

function buildRuntimeTool(
  overrides: Partial<AgentRuntimeToolInventoryRuntimeEntry> = {},
): AgentRuntimeToolInventoryRuntimeEntry {
  return {
    name: "ReadFile",
    description: "读取文件",
    source_kind: "current_surface",
    deferred_loading: false,
    always_visible: false,
    allowed_callers: [],
    tags: [],
    input_examples_count: 0,
    has_output_schema: false,
    caller_allowed: true,
    visible_in_context: true,
    ...overrides,
  };
}

describe("harnessStatusPanelViewModel", () => {
  beforeEach(() => {
    document.documentElement.lang = "zh-CN";
    document.documentElement.dir = "ltr";
  });

  it("应解析工具友好标签", () => {
    expect(resolveFriendlyToolLabel()).toBeNull();
    expect(resolveFriendlyToolLabel("   ")).toBeNull();
    expect(resolveFriendlyToolLabel("TurnSummary")).toBe("当前任务摘要");
    expect(resolveFriendlyToolLabel("ReadFile")).toBe("文件读取");
  });

  it("应解析 Harness 文件名、动作和类型展示", () => {
    expect(getFileName("C:\\workspace\\src\\app.ts")).toBe("app.ts");
    expect(getFileName("/tmp/workspace/")).toBe("/tmp/workspace/");
    expect(describeAction("write")).toBe("写入");
    expect(describeAction("persist")).toBe("落盘");
    expect(describeKind("code")).toBe("代码");
    expect(describeKind("offload")).toBe("转存");

    expect(
      summarizeFileActions([
        buildFileEvent({ action: "write" }),
        buildFileEvent({ id: "event-2", action: "write" }),
        buildFileEvent({ id: "event-3", action: "edit" }),
      ]),
    ).toBe("写入 2 · 编辑 1");

    const fileEvents = [
      buildFileEvent({
        id: "event-old",
        path: "/tmp/workspace/src/app.ts",
        action: "write",
        timestamp: new Date("2026-05-26T11:00:00.000Z"),
      }),
      buildFileEvent({
        id: "event-new",
        path: "/tmp/workspace/src/app.ts",
        displayName: "renamed-app.ts",
        action: "edit",
        timestamp: new Date("2026-05-26T11:03:00.000Z"),
      }),
      buildFileEvent({
        id: "event-log",
        path: "/tmp/workspace/debug.log",
        displayName: "debug.log",
        kind: "log",
        action: "persist",
        timestamp: new Date("2026-05-26T11:02:00.000Z"),
      }),
    ];

    expect(
      buildFileFilterOptions(fileEvents).map((option) => option.value),
    ).toEqual(["all", "code", "log"]);
    expect(buildFilteredFileEvents(fileEvents, "code")).toHaveLength(2);
    expect(groupHarnessFileEvents(fileEvents)).toMatchObject([
      {
        key: "/tmp/workspace/src/app.ts",
        displayName: "renamed-app.ts",
        count: 2,
        actionSummary: "写入 1 · 编辑 1",
      },
      {
        key: "/tmp/workspace/debug.log",
        displayName: "debug.log",
        count: 1,
        actionSummary: "落盘 1",
      },
    ]);
  });

  it("应构建文件变更审阅条目并按最新状态汇总", () => {
    const activeWrite: HarnessActiveFileWrite = {
      id: "write-1",
      path: "/tmp/workspace/src/app.ts",
      displayName: "app.ts",
      phase: "streaming",
      status: "streaming",
      updatedAt: new Date("2026-05-26T11:03:00.000Z"),
      preview: "streaming preview",
      latestChunk: "streaming chunk",
      content: "streaming content",
    };
    const entries = buildFileChangeReviewEntries({
      activeFileWrites: [
        activeWrite,
        {
          ...activeWrite,
          id: "write-blank",
          path: "   ",
        },
      ],
      recentFileEvents: [
        buildFileEvent({
          id: "event-old",
          path: "/tmp/workspace/src/app.ts",
          displayName: "old-app.ts",
          action: "write",
          timestamp: new Date("2026-05-26T11:01:00.000Z"),
          preview: "old preview",
        }),
        buildFileEvent({
          id: "event-new",
          path: "/tmp/workspace/src/app.ts",
          displayName: "new-app.ts",
          action: "edit",
          timestamp: new Date("2026-05-26T11:04:00.000Z"),
          preview: "new preview",
        }),
        buildFileEvent({
          id: "event-log-persist",
          path: "/tmp/workspace/log.txt",
          displayName: "log.txt",
          kind: "log",
          action: "persist",
        }),
        buildFileEvent({
          id: "event-doc",
          path: "/tmp/workspace/docs/readme.md",
          displayName: "readme.md",
          kind: "code",
          action: "persist",
          timestamp: new Date("2026-05-26T11:02:00.000Z"),
        }),
        buildFileEvent({
          id: "event-read",
          path: "/tmp/workspace/src/read.ts",
          displayName: "read.ts",
          action: "read",
        }),
      ],
      decisions: {
        "/tmp/workspace/src/app.ts": "applied",
        "/tmp/workspace/docs/readme.md": "rejected",
      },
    });

    expect(entries).toHaveLength(2);
    expect(entries[0]).toMatchObject({
      key: "/tmp/workspace/src/app.ts",
      displayName: "new-app.ts",
      latestAction: "edit",
      count: 3,
      status: "applied",
      preview: "new preview",
    });
    expect(entries[0].actionSummaryItems).toEqual([
      { type: "phase", phase: "streaming", count: 1 },
      { type: "action", action: "write", count: 1 },
      { type: "action", action: "edit", count: 1 },
    ]);
    expect(
      buildFileReviewSummaryTextParts(entries[0].actionSummaryItems),
    ).toEqual([
      {
        labelKey: "agentChat.harness.fileReview.phaseCount",
        valueLabelKey: "agentChat.harness.fileReview.phase.streaming",
        count: 1,
      },
      {
        labelKey: "agentChat.harness.fileReview.actionCount",
        valueLabelKey: "agentChat.harness.fileReview.action.write",
        count: 1,
      },
      {
        labelKey: "agentChat.harness.fileReview.actionCount",
        valueLabelKey: "agentChat.harness.fileReview.action.edit",
        count: 1,
      },
    ]);
    expect(resolveFileReviewActionLabelKey("offload")).toBe(
      "agentChat.harness.fileReview.action.offload",
    );
    expect(resolveFileReviewKindLabelKey("artifact")).toBe(
      "agentChat.harness.fileReview.kind.artifact",
    );
    expect(resolveFileReviewPhaseLabelKey("failed")).toBe(
      "agentChat.harness.fileReview.phase.failed",
    );
    expect(resolveFileChangeStatusLabelKey("rejected")).toBe(
      "agentChat.harness.fileReview.status.rejected",
    );
    expect(resolveOutputPathLabelKey("artifact")).toBe(
      "agentChat.harness.outputs.paths.artifact",
    );
    expect(resolveApprovalRiskLabelKey("command")).toBe(
      "agentChat.harness.approvals.risk.command",
    );
    expect(entries[1]).toMatchObject({
      key: "/tmp/workspace/docs/readme.md",
      status: "rejected",
    });
    expect(countFileChangeStatuses(entries)).toEqual({
      pending: 0,
      applied: 1,
      rejected: 1,
    });
  });

  it("应构建文件审阅时间、写入描述和 diff 摘要", () => {
    const updatedAt = new Date("2026-05-26T11:03:00.000Z");
    const updatedAtLabel = updatedAt.toLocaleTimeString("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
    });
    const diffText = [
      "diff --git a/src/app.ts b/src/app.ts",
      "--- a/src/app.ts",
      "+++ b/src/app.ts",
      "@@ -1 +1 @@",
      "-old",
      "+new",
    ].join("\n");
    const activeWrite: HarnessActiveFileWrite = {
      id: "write-diff",
      path: "/tmp/workspace/src/app.ts",
      displayName: "app.ts",
      phase: "streaming",
      status: "streaming",
      source: "tool_start",
      updatedAt,
      content: diffText,
    };
    const [entry] = buildFileChangeReviewEntries({
      activeFileWrites: [activeWrite],
      recentFileEvents: [],
      decisions: {},
    });

    expect(entry).toBeDefined();
    expect(formatHarnessTime()).toBe("刚刚");
    expect(formatHarnessTime(updatedAt)).toBe(updatedAtLabel);
    expect(getActiveWriteDescription(activeWrite)).toBe(
      `正在写入 · 工具启动 · ${updatedAtLabel}`,
    );
    expect(resolveDiffReviewStatusLabelKey("added")).toBe(
      "agentChat.harness.diff.status.added",
    );
    expect(resolveDiffReviewStatusLabelKey()).toBe(
      "agentChat.harness.diff.status.unknown",
    );
    expect(buildFileChangeReviewDiffSummary(entry!)).toMatchObject({
      additions: 1,
      deletions: 1,
      hunks: 1,
      files: [{ path: "src/app.ts", status: "modified" }],
    });
    expect(
      buildOutputSignalDiffSummary(
        buildOutputSignal({
          content: diffText,
          outputFile: "/tmp/workspace/out.diff",
        }),
      ),
    ).toMatchObject({
      additions: 1,
      deletions: 1,
      hunks: 1,
      files: [{ path: "src/app.ts", status: "modified" }],
    });
  });

  it("应按输出信号特征匹配 Harness 输出过滤器", () => {
    expect(
      getSignalPath(buildOutputSignal({ outputFile: "/tmp/out.txt" })),
    ).toBe("/tmp/out.txt");
    expect(
      getSignalPath(buildOutputSignal({ offloadFile: "/tmp/offload.txt" })),
    ).toBe("/tmp/offload.txt");
    expect(
      getSignalPath(buildOutputSignal({ artifactPath: "artifact://result" })),
    ).toBe("artifact://result");

    expect(
      matchesOutputFilter(
        buildOutputSignal({ outputFile: "/tmp/out.txt" }),
        "path",
      ),
    ).toBe(true);
    expect(
      matchesOutputFilter(buildOutputSignal({ offloaded: true }), "offload"),
    ).toBe(true);
    expect(
      matchesOutputFilter(buildOutputSignal({ truncated: true }), "truncated"),
    ).toBe(true);
    expect(
      matchesOutputFilter(
        buildOutputSignal({ preview: "摘要预览" }),
        "summary",
      ),
    ).toBe(true);
    expect(
      matchesOutputFilter(
        buildOutputSignal({
          preview: "有路径时不算纯摘要",
          outputFile: "/tmp/out.txt",
        }),
        "summary",
      ),
    ).toBe(false);

    const outputSignals = [
      buildOutputSignal({
        id: "search-1",
        toolName: "WebSearch",
        title: "联网检索摘要",
      }),
      buildOutputSignal({
        id: "search-2",
        toolName: "web_search",
        title: "联网检索摘要",
      }),
      buildOutputSignal({
        id: "offload",
        offloadFile: "/tmp/offload.txt",
        offloaded: true,
      }),
    ];

    expect(
      buildOutputFilterOptions(outputSignals).map((option) => option.value),
    ).toEqual(["all", "path", "offload"]);
    expect(buildFilteredOutputSignals(outputSignals, "offload")).toHaveLength(
      1,
    );
    expect(groupHarnessOutputSignals(outputSignals)).toMatchObject([
      {
        type: "search_batch",
        signals: [{ id: "search-1" }, { id: "search-2" }],
      },
      { type: "single", signal: { id: "offload" } },
    ]);
  });

  it("应构建 Harness 输出状态、路径和卡片展示模型", () => {
    expect(
      buildOutputStatusDescriptors(
        buildOutputSignal({
          exitCode: 1,
          truncated: true,
          offloaded: true,
          sandboxed: false,
          stdoutLength: 12,
          stderrLength: 3,
          offloadOriginalChars: 1200,
          offloadOriginalTokens: 320,
        }),
      ),
    ).toEqual([
      {
        key: "exit-code",
        labelKey: "agentChat.harness.outputs.status.exitFailed",
        values: { code: 1 },
        variant: "destructive",
      },
      {
        key: "truncated",
        labelKey: "agentChat.harness.outputs.status.truncated",
        variant: "outline",
      },
      {
        key: "offloaded",
        labelKey: "agentChat.harness.outputs.status.offloaded",
        variant: "outline",
      },
      {
        key: "sandboxed",
        labelKey: "agentChat.harness.outputs.status.unsandboxed",
        variant: "outline",
      },
      {
        key: "stdout",
        labelKey: "agentChat.harness.outputs.status.stdout",
        values: { count: 12 },
        variant: "outline",
      },
      {
        key: "stderr",
        labelKey: "agentChat.harness.outputs.status.stderr",
        values: { count: 3 },
        variant: "destructive",
      },
      {
        key: "original-chars",
        labelKey: "agentChat.harness.outputs.status.originalChars",
        values: { count: 1200 },
        variant: "outline",
      },
      {
        key: "original-tokens",
        labelKey: "agentChat.harness.outputs.status.originalTokens",
        values: { count: 320 },
        variant: "outline",
      },
    ]);

    expect(
      buildOutputStatusDescriptors(buildOutputSignal({ exitCode: 0 })),
    ).toEqual([
      {
        key: "exit-code",
        labelKey: "agentChat.harness.outputs.status.exitSuccess",
        values: { code: 0 },
        variant: "secondary",
      },
    ]);

    expect(
      getOutputSignalPaths(
        buildOutputSignal({
          outputFile: "/tmp/out.txt",
          offloadFile: "/tmp/offload.txt",
          artifactPath: "artifact://result",
        }),
      ),
    ).toEqual([
      { key: "output", path: "/tmp/out.txt" },
      { key: "offload", path: "/tmp/offload.txt" },
      { key: "artifact", path: "artifact://result" },
    ]);

    expect(isNoisyRuntimeOutputText("JSON-RPC -32603 troubleshooting")).toBe(
      true,
    );
    expect(isNoisyRuntimeOutputText("普通业务错误")).toBe(false);

    expect(
      resolveOutputCardPresentation(
        buildOutputSignal({
          exitCode: 1,
          summary: "JSON-RPC -32603 troubleshooting",
          preview: "raw preview",
        }),
        { rawDetailsCollapsedHint: "已收起原始输出" },
      ),
    ).toMatchObject({
      preview: undefined,
      collapsedHint: "已收起原始输出",
      rawDetailsCollapsed: true,
      tone: "failed",
    });

    expect(
      resolveOutputCardPresentation(
        buildOutputSignal({
          exitCode: 1,
          summary: "脚本执行失败",
          preview: "stderr preview",
        }),
        { rawDetailsCollapsedHint: "已收起原始输出" },
      ),
    ).toEqual({
      summary: "脚本执行失败",
      preview: "stderr preview",
      collapsedHint: null,
      rawDetailsCollapsed: false,
      tone: "failed",
    });

    expect(
      resolveOutputCardPresentation(
        buildOutputSignal({
          exitCode: 0,
          summary: "完成",
          preview: "preview",
        }),
        { rawDetailsCollapsedHint: "已收起原始输出" },
      ),
    ).toEqual({
      summary: "完成",
      preview: "preview",
      collapsedHint: null,
      rawDetailsCollapsed: false,
      tone: "default",
    });
  });

  it("应构建 Harness runtime 任务展示模型", () => {
    expect(formatRuntimePhaseLabel(null)).toBe("空闲");

    const runningStatus = buildRuntimeStatus({
      title: "   ",
      detail: " 正在处理用户请求 ",
      checkpoints: ["  分析输入 ", "", "写入结果"],
    });
    expect(resolveRuntimeStepStatus(runningStatus)).toBe("active");
    expect(resolveRuntimeStatusLabel(runningStatus)).toBe("进行中");
    expect(buildRuntimeSummaryText(runningStatus)).toBe("正在处理用户请求");
    expect(
      formatRuntimeProgressLabel(runningStatus, ["分析输入", "写入结果"]),
    ).toBe("已记录 2 个任务节点");
    expect(buildRuntimeTaskPresentation(runningStatus)).toEqual({
      title: "正在整理当前任务",
      summaryText: "正在处理用户请求",
      phaseLabel: "处理中",
      statusLabel: "进行中",
      progressLabel: "已记录 2 个任务节点",
      stepStatus: "active",
      checkpoints: ["分析输入", "写入结果"],
    });

    const failedStatus = buildRuntimeStatus({
      phase: "failed",
      title: "失败任务",
      checkpoints: [],
    });
    expect(resolveRuntimeStepStatus(failedStatus)).toBe("error");
    expect(resolveRuntimeStatusLabel(failedStatus)).toBe("异常");
    expect(formatRuntimePhaseLabel(failedStatus)).toBe("需要处理");
    expect(formatRuntimeProgressLabel(failedStatus, [])).toBe(
      "等待处理异常后重试",
    );

    const cancelledStatus = buildRuntimeStatus({
      phase: "cancelled",
      checkpoints: [],
    });
    expect(resolveRuntimeStepStatus(cancelledStatus)).toBe("skipped");
    expect(resolveRuntimeStatusLabel(cancelledStatus)).toBe("已取消");
    expect(buildRuntimeSummaryText(cancelledStatus)).toBe(
      "当前流程已取消，可重新发起新的任务继续。",
    );
    expect(formatRuntimeProgressLabel(cancelledStatus, [])).toBe(
      "当前流程已取消",
    );

    expect(buildRuntimeTaskPresentation(null)).toBeNull();
    expect(
      buildRuntimeTaskPresentation(
        buildRuntimeStatus({
          metadata: { visibility: "diagnostics" },
        }),
      ),
    ).toBeNull();

    expect(
      buildRuntimeFactSummary({
        thread_id: "thread-1",
        decision_reason: "优先使用低延迟模型",
        fallback_chain: ["openai:gpt-5.4", "openai:gpt-5.4-mini"],
        oem_policy: {
          locked: true,
          selectedModel: "gpt-5.4-mini",
          tenantId: "tenant-1",
        },
      }),
    ).toEqual({
      threadId: "thread-1",
      status: null,
      turnCount: 0,
      itemCount: 0,
      pendingRequestCount: 0,
      artifactCount: 0,
      evidenceRefCount: 0,
      decisionReason: "优先使用低延迟模型",
      fallbackChain: ["openai:gpt-5.4", "openai:gpt-5.4-mini"],
      oemPolicy: {
        locked: true,
        selectedModel: "gpt-5.4-mini",
        tenantId: "tenant-1",
      },
    });
    expect(
      buildRuntimeFactSummary({
        thread_id: "thread-2",
        runtime_summary: {
          decisionReason: "运行时摘要回退",
          fallbackChain: ["local:model"],
        },
      }),
    ).toMatchObject({
      decisionReason: "运行时摘要回退",
      fallbackChain: ["local:model"],
    });
    expect(buildRuntimeFactSummary({ thread_id: "thread-empty" })).toEqual({
      threadId: "thread-empty",
      status: null,
      turnCount: 0,
      itemCount: 0,
      pendingRequestCount: 0,
      artifactCount: 0,
      evidenceRefCount: 0,
      decisionReason: null,
      fallbackChain: [],
      oemPolicy: null,
    });
  });

  it("应格式化工具库存和执行策略展示标签", () => {
    expect(formatWriteSourceLabel("tool_start")).toBe("工具启动");
    expect(formatWriteSourceLabel("artifact_snapshot")).toBe("快照同步");
    expect(formatWriteSourceLabel("custom_source")).toBe("custom_source");
    expect(formatWriteSourceLabel()).toBe("处理中");

    expect(formatExecutionSourceLabel("runtime")).toBe("运行时覆盖");
    expect(resolveExecutionSourceVariant("runtime")).toBe("default");
    expect(formatExecutionSourceLabel("persisted")).toBe("持久化覆盖");
    expect(resolveExecutionSourceVariant("persisted")).toBe("secondary");
    expect(formatExecutionSourceLabel("default")).toBe("默认策略");
    expect(resolveExecutionSourceVariant("default")).toBe("outline");

    expect(formatExecutionWarningPolicyLabel("shell_command_risk")).toBe(
      "命令风险告警",
    );
    expect(
      formatExecutionRestrictionProfileLabel(
        "workspace_absolute_path_required",
      ),
    ).toBe("必须提供绝对工作区路径");
    expect(formatExecutionSandboxProfileLabel("workspace_command")).toBe(
      "工作区命令沙箱",
    );
    expect(formatToolLifecycleLabel("deprecated")).toBe("待清理");
    expect(formatToolPermissionPlaneLabel("caller_filtered")).toBe(
      "调用方过滤",
    );
    expect(formatToolSourceKindLabel("lime_injected")).toBe("Lime 注入");
    expect(formatExtensionSourceKindLabel("mcp_bridge")).toBe("MCP Bridge");
    expect(formatRuntimeToolSourceKindLabel("current_surface")).toBe(
      "当前工具面",
    );
    expect(formatRuntimeToolAvailabilitySourceLabel("native_tools")).toBe(
      "native_tools",
    );
    expect(formatRuntimeToolAvailabilitySourceLabel("none")).toBe("未就绪");
  });

  it("应构建工具库存筛选、统计和排序模型", () => {
    const runtimeTool = buildCatalogTool({
      name: "RuntimeTool",
      execution_warning_policy_source: "runtime",
      execution_restriction_profile_source: "persisted",
      execution_sandbox_profile_source: "default",
    });
    const defaultTool = buildCatalogTool({ name: "DefaultTool" });
    const persistedTool = buildCatalogTool({
      name: "PersistedTool",
      execution_warning_policy_source: "persisted",
    });
    const catalogTools = [runtimeTool, defaultTool, persistedTool];

    expect(collectCatalogExecutionSources(runtimeTool)).toEqual([
      "runtime",
      "persisted",
      "default",
    ]);
    expect(matchesCatalogToolInventoryFilter(runtimeTool, "runtime")).toBe(
      true,
    );
    expect(matchesCatalogToolInventoryFilter(defaultTool, "default")).toBe(
      true,
    );
    expect(matchesCatalogToolInventoryFilter(persistedTool, "default")).toBe(
      false,
    );
    expect(countCatalogToolsByInventoryFilter(catalogTools, "all")).toBe(3);
    expect(countCatalogToolsByInventoryFilter(catalogTools, "runtime")).toBe(1);
    expect(countCatalogToolsByInventoryFilter(catalogTools, "persisted")).toBe(
      2,
    );
    expect(countCatalogToolsByInventoryFilter(catalogTools, "default")).toBe(1);
    expect(buildToolInventorySourceStats(catalogTools)).toEqual({
      default: 6,
      persisted: 2,
      runtime: 1,
    });

    expect(
      collectNativeExecutionSources(
        buildNativeTool({
          catalog_execution_warning_policy_source: "runtime",
          catalog_execution_sandbox_profile_source: "persisted",
        }),
      ),
    ).toEqual(["runtime", "persisted"]);

    expect(
      sortRuntimeToolsByVisibility([
        buildRuntimeTool({ name: "zeta", visible_in_context: false }),
        buildRuntimeTool({ name: "beta", visible_in_context: true }),
        buildRuntimeTool({ name: "alpha", visible_in_context: true }),
      ]).map((entry) => entry.name),
    ).toEqual(["alpha", "beta", "zeta"]);

    expect(
      buildRuntimeToolCapabilityGaps(false, {
        source: "none",
        known: true,
        agentInitialized: true,
        availableToolCount: 0,
        webSearch: false,
        subagentCore: false,
        subagentTeamTools: false,
        subagentRuntime: false,
        planRuntime: false,
        missingSubagentCoreTools: ["Agent", "SendMessage"],
        missingSubagentTeamTools: ["TeamCreate"],
        missingPlanTools: ["update_plan"],
      }),
    ).toEqual([]);
    expect(
      buildRuntimeToolCapabilityGaps(true, {
        source: "runtime_tools",
        known: true,
        agentInitialized: true,
        availableToolCount: 1,
        webSearch: false,
        subagentCore: false,
        subagentTeamTools: true,
        subagentRuntime: false,
        planRuntime: false,
        missingSubagentCoreTools: ["SendMessage"],
        missingSubagentTeamTools: [],
        missingPlanTools: ["update_plan"],
      }),
    ).toEqual([
      { key: "web_search", title: "WebSearch", missing: ["WebSearch"] },
      {
        key: "subagent_core",
        title: "子任务核心 tools",
        missing: ["SendMessage"],
      },
      {
        key: "plan_runtime",
        title: "Plan current tool",
        missing: ["update_plan"],
      },
    ]);
  });

  it("应拼接非空展示片段", () => {
    expect(joinDisplayParts([" 主状态 ", "", undefined, " 次状态 "])).toBe(
      "主状态 · 次状态",
    );
    expect(joinDisplayParts(["", null, undefined])).toBeUndefined();
  });

  it("应归一化 URL 并切分文本链接片段", () => {
    expect(normalizeUrlCandidate("https://a.test/path),")).toEqual({
      url: "https://a.test/path",
      trailing: "),",
    });
    expect(splitTextIntoSegments("看 https://a.test/x, 再继续")).toEqual([
      { type: "text", value: "看 " },
      { type: "url", value: "https://a.test/x" },
      { type: "text", value: "," },
      { type: "text", value: " 再继续" },
    ]);
    expect(findFirstUrl(undefined, "x https://a.test/y.")).toBe(
      "https://a.test/y",
    );

    expect(isLikelyFilePath("/tmp/a.md")).toBe(true);
    expect(isLikelyFilePath("C:\\work\\a.ts")).toBe(true);
    expect(isLikelyFilePath("src/app.ts")).toBe(true);
    expect(isLikelyFilePath("https://x.test/a.ts")).toBe(false);
  });

  it("应识别搜索输出信号", () => {
    expect(
      isSearchOutputSignal(buildOutputSignal({ toolName: "web_search" })),
    ).toBe(true);
    expect(
      isSearchOutputSignal(buildOutputSignal({ toolName: "Web Search" })),
    ).toBe(true);
    expect(
      isSearchOutputSignal(
        buildOutputSignal({ toolName: "bash", title: "联网检索摘要" }),
      ),
    ).toBe(true);
    expect(isSearchOutputSignal(buildOutputSignal({ toolName: "bash" }))).toBe(
      false,
    );
  });

  it("应解析人工确认风险、动作标签和描述", () => {
    const commandApproval: ActionRequired = {
      requestId: "approval-command",
      actionType: "tool_confirmation",
      toolName: "Bash",
      arguments: { cmd: "npm test" },
    };
    expect(pickCommandFromArguments(commandApproval.arguments)).toBe(
      "npm test",
    );
    expect(resolveApprovalRiskKind(commandApproval)).toBe("command");
    expect(resolveApprovalActionLabelKey(commandApproval)).toBe(
      "agentChat.harness.approvals.action.tool",
    );
    expect(describeApproval(commandApproval)).toContain("npm test");

    const fileApproval: ActionRequired = {
      requestId: "approval-file",
      actionType: "tool_confirmation",
      toolName: "WriteFile",
      arguments: {
        payload: { absolute_path: " /tmp/new.md " },
      },
    };
    expect(pickPathFromArguments(fileApproval.arguments)).toBe("/tmp/new.md");
    expect(resolveApprovalRiskKind(fileApproval)).toBe("file_change");
    expect(describeApproval(fileApproval)).toContain("/tmp/new.md");

    expect(
      resolveApprovalRiskKind({
        requestId: "approval-ask",
        actionType: "ask_user",
      }),
    ).toBe("input");
    expect(
      resolveApprovalActionLabelKey({
        requestId: "approval-elicitation",
        actionType: "elicitation",
      }),
    ).toBe("agentChat.harness.approvals.action.elicitation");
  });
});
