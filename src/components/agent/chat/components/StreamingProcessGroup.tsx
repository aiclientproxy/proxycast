import React, { useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown } from "lucide-react";
import type { AgentToolCallState as ToolCallState } from "@/lib/api/agentProtocol";
import { cn } from "@/lib/utils";
import { resolveThinkingDisplayParts } from "./thinkingBlockDisplay";
import { StreamingWebSearchProcessTimeline } from "./StreamingWebSearchProcessTimeline";
import {
  summarizeStreamingToolBatch,
  type ToolBatchSummaryDescriptor,
  type ToolBatchSummarySectionKind,
} from "../utils/toolBatchGrouping";
import type { SearchResultPreviewItem } from "../utils/searchResultPreview";
import {
  buildToolGroupHeadline,
  getToolDisplayInfo,
} from "../utils/toolDisplayInfo";
import { resolveToolProcessNarrative } from "../utils/toolProcessSummary";
import { resolveTerminalInteractionSummaries } from "../utils/toolProcessSummaryMetadata";
import type { StreamingProcessEntry } from "./StreamingProcessGroupModel";
import { resolveWorkspaceSkillRuntimeEnableResultDisplay } from "../utils/toolResultEnvelopeDisplay";

interface StreamingProcessSummaryCopy {
  completedThinking: () => string;
  failedSteps: (count: number) => string;
  runningSteps: (count: number) => string;
  completedSteps: (count: number) => string;
  runningThinking: () => string;
  structuredThinking: () => string;
  thinking: () => string;
  toolCalls: (count: number) => string;
  processMessages: (count: number) => string;
  thinkingNotes: (count: number) => string;
  separator: () => string;
}

function joinSummaryParts(
  parts: Array<string | null | undefined>,
  separator: string,
): string {
  return parts.filter(Boolean).join(separator);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function resolveToolCallMetadata(
  toolCall: ToolCallState,
): Record<string, unknown> | null {
  const merged = {
    ...(asRecord(toolCall.metadata) || {}),
    ...(asRecord(toolCall.result?.metadata) || {}),
  };
  return Object.keys(merged).length > 0 ? merged : null;
}

function resolveProcessThinkingDisplay(
  entry: Extract<StreamingProcessEntry, { kind: "thinking" }>,
  copy: StreamingProcessSummaryCopy,
) {
  return resolveThinkingDisplayParts(entry.text, entry.isActive === true, {
    labels: {
      completed: copy.completedThinking(),
      running: copy.runningThinking(),
      structuredFallback: copy.structuredThinking(),
    },
  });
}

function resolveToolCallRawResultText(toolCall: ToolCallState): string {
  return String(toolCall.result?.error || toolCall.result?.output || "");
}

function buildStreamingProcessSummary(
  entries: StreamingProcessEntry[],
  copy: StreamingProcessSummaryCopy,
): {
  summaryText: string;
  descriptor: ToolBatchSummaryDescriptor | null;
  metaText: string | null;
} {
  const toolEntries = entries.filter(
    (entry): entry is Extract<StreamingProcessEntry, { kind: "tool" }> =>
      entry.kind === "tool",
  );
  const thinkingCount = entries.filter(
    (entry) => entry.kind === "thinking",
  ).length;
  const summarizedBatchDescriptor =
    toolEntries.length > 0
      ? summarizeStreamingToolBatch(toolEntries.map((entry) => entry.toolCall))
      : null;
  const batchDescriptor = summarizedBatchDescriptor;
  if (batchDescriptor) {
    return {
      summaryText: batchDescriptor.title,
      descriptor: batchDescriptor,
      metaText: null,
    };
  }

  const toolCount = toolEntries.length;
  if (toolCount > 0) {
    const toolCalls = toolEntries.map((entry) => entry.toolCall);
    const families = new Set(
      toolCalls.map(
        (toolCall) => getToolDisplayInfo(toolCall.name, toolCall.status).family,
      ),
    );
    if (families.size === 1) {
      return {
        summaryText: buildToolGroupHeadline(toolCalls),
        descriptor: null,
        metaText: null,
      };
    }

    const failed = toolCalls.some((toolCall) => toolCall.status === "failed");
    const running = toolCalls.some((toolCall) => toolCall.status === "running");
    return {
      summaryText: failed
        ? copy.failedSteps(toolCount)
        : running
          ? copy.runningSteps(toolCount)
          : copy.completedSteps(toolCount),
      descriptor: null,
      metaText: null,
    };
  }

  const messageCount = entries.length - toolCount;
  const primarySummary = (() => {
    for (const entry of entries) {
      if (entry.kind === "thinking") {
        const thinkingDisplay = resolveProcessThinkingDisplay(entry, copy);
        if (thinkingDisplay.preview) {
          return thinkingDisplay.preview;
        }
        continue;
      }

      if (entry.kind === "tool") {
        const narrative = resolveToolProcessNarrative(entry.toolCall);
        if (narrative.preSummary || narrative.summary) {
          return narrative.preSummary || narrative.summary;
        }
        continue;
      }

      const prompt = entry.actionRequired.prompt?.trim();
      if (prompt) {
        return prompt.length <= 72
          ? prompt
          : `${prompt.slice(0, 71).trimEnd()}…`;
      }
    }

    return null;
  })();

  if (!primarySummary && toolCount === 0 && thinkingCount > 0) {
    const firstThinking = entries.find((entry) => entry.kind === "thinking");
    const thinkingDisplay =
      firstThinking?.kind === "thinking"
        ? resolveProcessThinkingDisplay(firstThinking, copy)
        : null;
    return {
      summaryText: thinkingDisplay?.statusLabel || copy.thinking(),
      descriptor: null,
      metaText: thinkingCount > 1 ? copy.thinkingNotes(thinkingCount) : null,
    };
  }

  if (!primarySummary) {
    const summaryParts: string[] = [];
    if (thinkingCount > 0) {
      summaryParts.push(copy.thinking());
    }
    if (toolCount > 0) {
      summaryParts.push(copy.toolCalls(toolCount));
    }
    if (messageCount > thinkingCount) {
      summaryParts.push(copy.processMessages(messageCount));
    }
    return {
      summaryText: summaryParts.join(copy.separator()),
      descriptor: null,
      metaText: null,
    };
  }

  if (toolCount === 0) {
    if (thinkingCount > 0) {
      const firstThinking = entries.find((entry) => entry.kind === "thinking");
      const thinkingDisplay =
        firstThinking?.kind === "thinking"
          ? resolveProcessThinkingDisplay(firstThinking, copy)
          : null;
      return {
        summaryText: thinkingDisplay?.statusLabel || copy.completedThinking(),
        descriptor: null,
        metaText: thinkingCount > 1 ? copy.thinkingNotes(thinkingCount) : null,
      };
    }

    return {
      summaryText: primarySummary,
      descriptor: null,
      metaText: thinkingCount > 1 ? copy.thinkingNotes(thinkingCount) : null,
    };
  }

  return {
    summaryText: primarySummary,
    descriptor: null,
    metaText:
      entries.length > 1
        ? joinSummaryParts(
            [
              thinkingCount > 0 ? copy.thinkingNotes(thinkingCount) : null,
              copy.toolCalls(toolCount),
            ],
            copy.separator(),
          )
        : null,
  };
}

export const GroupedProcessShell: React.FC<{
  groupMarker: string;
  children: React.ReactNode;
}> = ({ groupMarker, children }) => (
  <div className="flex items-start gap-2 py-1.5">
    <span className="pt-0.5 text-xs text-slate-300">{groupMarker}</span>
    <div className="min-w-0 flex-1">{children}</div>
  </div>
);

export const StreamingProcessGroup: React.FC<{
  entries: StreamingProcessEntry[];
  defaultExpanded?: boolean;
  onOpenUrlPreview?: (item: SearchResultPreviewItem) => void;
  renderEntry: (
    entry: StreamingProcessEntry,
    grouped: boolean,
    groupMarker: string,
    entries: StreamingProcessEntry[],
  ) => React.ReactNode;
}> = ({ entries, defaultExpanded = false, onOpenUrlPreview, renderEntry }) => {
  const { t } = useTranslation("agent");
  const resolveWebSearchSectionTitle = (kind: ToolBatchSummarySectionKind) =>
    kind === "web_fetch_pages"
      ? t("agentChat.processGroup.webSearch.section.webFetchPages", {
          defaultValue: "Read pages",
        })
      : t("agentChat.processGroup.webSearch.section.webSearchSources", {
          defaultValue: "Search sources",
        });
  const [expanded, setExpanded] = useState(defaultExpanded);
  const previousDefaultExpandedRef = useRef(defaultExpanded);
  const separator = t("agentChat.processGroup.separator", ", ");
  const { summaryText, descriptor, metaText } = useMemo(
    () =>
      buildStreamingProcessSummary(entries, {
        completedThinking: () =>
          t("agentChat.thinkingBlock.status.completed", {
            defaultValue: "已完成思考",
          }),
        runningThinking: () =>
          t("agentChat.thinkingBlock.status.running", {
            defaultValue: "思考中",
          }),
        structuredThinking: () =>
          t("agentChat.thinkingBlock.preview.structured", {
            defaultValue: "在整理结构化内容",
          }),
        failedSteps: (count: number) =>
          t("agentChat.processGroup.failedSteps", {
            count,
            defaultValue: "{{count}} steps failed",
          }),
        runningSteps: (count: number) =>
          t("agentChat.processGroup.runningSteps", {
            count,
            defaultValue: "{{count}} steps running",
          }),
        completedSteps: (count: number) =>
          t("agentChat.processGroup.completedSteps", {
            count,
            defaultValue: "{{count}} steps completed",
          }),
        thinking: () => t("agentChat.processGroup.thinking", "Reasoning"),
        toolCalls: (count: number) =>
          t("agentChat.processGroup.toolCalls", {
            count,
            defaultValue: "{{count}} tool calls",
          }),
        processMessages: (count: number) =>
          t("agentChat.processGroup.processMessages", {
            count,
            defaultValue: "{{count}} process messages",
          }),
        thinkingNotes: (count: number) =>
          t("agentChat.processGroup.thinkingNotes", {
            count,
            defaultValue: "{{count}} reasoning notes",
          }),
        separator: () => separator,
      }),
    [entries, separator, t],
  );
  const runtimeEnableMetaText = useMemo(() => {
    const summaries: string[] = [];
    for (const entry of entries) {
      if (entry.kind !== "tool") {
        continue;
      }
      const summary = resolveWorkspaceSkillRuntimeEnableResultDisplay({
        toolName: entry.toolCall.name,
        rawResultText: resolveToolCallRawResultText(entry.toolCall),
        metadata: resolveToolCallMetadata(entry.toolCall),
        translate: (key, defaultValue, options) =>
          String(t(key, { defaultValue, ...options })),
      });
      if (summary && !summaries.includes(summary)) {
        summaries.push(summary);
      }
    }
    return summaries.length > 0 ? summaries.join(separator) : null;
  }, [entries, separator, t]);
  const terminalInteractionMetaText = useMemo(() => {
    const summaries = entries.flatMap((entry) =>
      entry.kind === "tool"
        ? resolveTerminalInteractionSummaries(
            resolveToolCallMetadata(entry.toolCall),
          )
        : [],
    );
    return summaries.length > 0
      ? String(
          t("agentChat.processGroup.terminalInteractions", {
            defaultValue: "Input sent: {{value}}",
            value: summaries.join(separator),
          }),
        )
      : null;
  }, [entries, separator, t]);
  const combinedMetaText = joinSummaryParts(
    [metaText, runtimeEnableMetaText, terminalInteractionMetaText],
    separator,
  );
  const nonToolEntries = entries.filter((entry) => entry.kind !== "tool");
  const hasNonToolEntries = nonToolEntries.length > 0;
  const processKind = descriptor?.kind || "mixed";
  const isWebSearchProcess = descriptor?.kind === "web_search";

  React.useEffect(() => {
    if (previousDefaultExpandedRef.current !== defaultExpanded) {
      previousDefaultExpandedRef.current = defaultExpanded;
      setExpanded(defaultExpanded);
    }
  }, [defaultExpanded]);

  return (
    <div
      className="py-0.5"
      data-testid="streaming-process-group"
      data-process-kind={processKind}
      data-process-running={descriptor?.hasRunning ? "yes" : "no"}
      data-visual-tone={isWebSearchProcess ? "codex-activity" : "neutral"}
    >
      <button
        type="button"
        className={cn(
          "group flex w-full items-start gap-2 rounded-lg py-1.5 text-left transition-colors hover:bg-slate-50/70",
          isWebSearchProcess && "pr-1",
        )}
        onClick={() => setExpanded((current) => !current)}
        aria-expanded={expanded}
      >
        {isWebSearchProcess ? (
          <span
            className={cn(
              "mt-2 h-2 w-2 shrink-0 rounded-full",
              descriptor?.hasRunning
                ? "animate-pulse bg-sky-500 shadow-[0_0_0_4px_rgba(14,165,233,0.12)]"
                : "bg-slate-300 shadow-[0_0_0_4px_rgba(148,163,184,0.12)]",
            )}
            data-testid="streaming-process-status-bullet"
          />
        ) : (
          <ChevronDown
            className={cn(
              "mt-1 h-3.5 w-3.5 shrink-0 text-slate-400 transition-transform duration-200",
              expanded && "rotate-180",
            )}
          />
        )}
        <span
          className={cn(
            "min-w-0 flex-1 text-[13px] leading-6",
            isWebSearchProcess
              ? "font-medium text-slate-700"
              : "font-normal text-slate-600",
          )}
        >
          <span className="block break-words">{summaryText}</span>
          {combinedMetaText ? (
            <span className="mt-0.5 block text-xs font-normal leading-5 text-slate-500">
              {combinedMetaText}
            </span>
          ) : null}
          {!hasNonToolEntries &&
          descriptor?.kind !== "web_search" &&
          descriptor?.supportingLines?.length ? (
            <span className="mt-0.5 block space-y-0.5">
              {descriptor.supportingLines.slice(0, 5).map((line) => (
                <span
                  key={line}
                  className="block text-xs font-normal leading-5 text-slate-500"
                >
                  {line}
                </span>
              ))}
            </span>
          ) : null}
        </span>
        {isWebSearchProcess ? (
          <ChevronDown
            className={cn(
              "mt-1 h-3.5 w-3.5 shrink-0 text-slate-400 transition-transform duration-200",
              expanded && "rotate-180",
            )}
          />
        ) : null}
      </button>
      {expanded ? (
        <div className="ml-2">
          {descriptor?.kind === "web_search" &&
          descriptor.supportingLines.length > 0 ? (
            <GroupedProcessShell groupMarker="└">
              <StreamingWebSearchProcessTimeline
                entries={entries}
                descriptor={descriptor}
                hasNonToolEntries={hasNonToolEntries}
                onOpenUrlPreview={onOpenUrlPreview}
                resolveSectionTitle={resolveWebSearchSectionTitle}
                renderEntry={renderEntry}
              />
            </GroupedProcessShell>
          ) : (
            entries.map((entry, index) => (
              <React.Fragment key={entry.id}>
                {renderEntry(entry, true, index === 0 ? "└" : "·", entries)}
              </React.Fragment>
            ))
          )}
        </div>
      ) : null}
    </div>
  );
};
