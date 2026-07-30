import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, vi } from "vitest";

import { AgentThreadTimeline } from "./AgentThreadTimeline";
import type {
  ActionRequired,
  AgentThreadItem,
  AgentThreadTurn,
} from "../types";
import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import { changeLimeLocale } from "@/i18n/createI18n";
import type { ArtifactTimelineOpenTarget } from "../utils/artifactTimelineNavigation";

export const parseAIResponseMock = vi.fn();

function resolveMockToolText(toolCall: {
  name: string;
  arguments?: string;
  status?: string;
}) {
  let parsedArguments: Record<string, unknown> | null = null;
  if (toolCall.arguments) {
    try {
      parsedArguments = JSON.parse(toolCall.arguments) as Record<
        string,
        unknown
      >;
    } catch {
      parsedArguments = null;
    }
  }

  if (
    toolCall.name === "browser_navigate" &&
    typeof parsedArguments?.url === "string"
  ) {
    return `打开 ${parsedArguments.url}`;
  }

  if (
    (toolCall.name === "web_search" || toolCall.name === "search_query") &&
    typeof parsedArguments?.query === "string"
  ) {
    return `搜索 ${parsedArguments.query}`;
  }

  if (
    toolCall.name === "exec_command" &&
    typeof parsedArguments?.command === "string"
  ) {
    return `执行 ${parsedArguments.command}`;
  }

  if (
    toolCall.name === "lime_site_run" &&
    typeof parsedArguments?.adapter_name === "string"
  ) {
    return `执行 ${parsedArguments.adapter_name}`;
  }

  return toolCall.name;
}

export const mockToolCallItem = vi.fn(
  ({
    toolCall,
    onOpenSavedSiteContent,
    grouped,
    groupMarker,
  }: {
    toolCall: { name: string; arguments?: string; status?: string };
    onOpenSavedSiteContent?: (target: {
      projectId: string;
      contentId: string;
      title?: string;
    }) => void;
    grouped?: boolean;
    groupMarker?: string;
  }) => (
    <div
      data-testid="tool-call-item"
      data-has-open-saved-site-content={onOpenSavedSiteContent ? "yes" : "no"}
      data-grouped={grouped ? "yes" : "no"}
      data-group-marker={groupMarker || ""}
    >
      {resolveMockToolText(toolCall)}
      {toolCall.status === "running" ? " 进行中" : ""}
    </div>
  ),
);

vi.mock("@/components/workspace/a2ui/parser", () => ({
  parseAIResponse: (...args: unknown[]) => parseAIResponseMock(...args),
}));

vi.mock("@/components/workspace/a2ui/taskCardPresets", () => ({
  CHAT_A2UI_TASK_CARD_PRESET: {},
  TIMELINE_A2UI_TASK_CARD_PRESET: {},
}));

vi.mock("./MarkdownRenderer", () => ({
  MarkdownRenderer: ({ content }: { content: string }) => (
    <div data-testid="markdown-renderer">{content}</div>
  ),
}));

vi.mock("./A2UITaskCard", () => ({
  A2UITaskCard: () => <div data-testid="timeline-a2ui-card" />,
  A2UITaskLoadingCard: () => <div data-testid="timeline-a2ui-loading-card" />,
}));

vi.mock("./ToolCallDisplay", () => ({
  ToolCallItem: (props: {
    toolCall: { name: string; arguments?: string; status?: string };
    onOpenSavedSiteContent?: (target: {
      projectId: string;
      contentId: string;
      title?: string;
    }) => void;
    grouped?: boolean;
    groupMarker?: string;
  }) => mockToolCallItem(props),
}));

vi.mock("./DecisionPanel", () => ({
  DecisionPanel: ({
    request,
  }: {
    request: {
      prompt?: string;
      questions?: Array<{
        header?: string;
        question?: string;
        options?: Array<{ label: string }>;
      }>;
    };
  }) => (
    <div data-testid="decision-panel">
      {request.prompt || "decision"}
      {request.questions?.map((question) => (
        <div key={question.header || question.question}>
          {question.header}
          {question.options?.map((option) => (
            <span key={option.label}>{option.label}</span>
          ))}
        </div>
      ))}
    </div>
  ),
}));

vi.mock("./AgentPlanBlock", () => ({
  AgentPlanBlock: ({ content }: { content: string }) => (
    <div data-testid="agent-plan-block">{content}</div>
  ),
}));

interface MountedHarness {
  container: HTMLDivElement;
  root: Root;
}

const mountedRoots: MountedHarness[] = [];

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  await changeLimeLocale("zh-CN");
  HTMLElement.prototype.scrollIntoView = vi.fn();
  parseAIResponseMock.mockImplementation((content: string) => ({
    parts: content.trim() ? [{ type: "text", content: content.trim() }] : [],
    hasA2UI: false,
    hasWriteFile: false,
    hasPending: false,
  }));
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) break;
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.clearAllMocks();
});

export function at(second: number): string {
  return `2026-03-15T09:10:${String(second).padStart(2, "0")}Z`;
}

export function createTurn(
  overrides?: Partial<AgentThreadTurn>,
): AgentThreadTurn {
  return {
    id: "turn-1",
    thread_id: "thread-1",
    prompt_text: "请检查并发布文章",
    status: "completed",
    started_at: at(0),
    completed_at: at(9),
    created_at: at(0),
    updated_at: at(9),
    ...overrides,
  };
}

export function createBaseItem(
  id: string,
  sequence: number,
): Pick<
  AgentThreadItem,
  | "id"
  | "thread_id"
  | "turn_id"
  | "sequence"
  | "status"
  | "started_at"
  | "completed_at"
  | "updated_at"
> {
  const timestamp = at(sequence);
  return {
    id,
    thread_id: "thread-1",
    turn_id: "turn-1",
    sequence,
    status: "completed",
    started_at: timestamp,
    completed_at: timestamp,
    updated_at: timestamp,
  };
}

export function renderTimeline(
  items: AgentThreadItem[],
  props?: {
    isCurrentTurn?: boolean;
    turn?: Partial<AgentThreadTurn>;
    threadRead?: AgentRuntimeThreadReadModel | null;
    actionRequests?: ActionRequired[];
    onOpenSavedSiteContent?: (target: {
      projectId: string;
      contentId: string;
      title?: string;
    }) => void;
    onOpenSubagentSession?: (sessionId: string) => void;
    onOpenArtifactFromTimeline?: (target: ArtifactTimelineOpenTarget) => void;
    sourceMessageId?: string;
    onSaveFileArtifactAsKnowledge?: (source: {
      messageId: string;
      content: string;
      sourceName?: string;
      description?: string | null;
    }) => void;
    focusedItemId?: string | null;
    focusRequestKey?: number;
    deferCompletedSingleDetails?: boolean;
    collapseInactiveDetails?: boolean;
    showOperationalDetails?: boolean;
    showInlineStatusHint?: boolean;
  },
): HTMLDivElement {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(
      <AgentThreadTimeline
        turn={createTurn(props?.turn)}
        items={items}
        threadRead={props?.threadRead}
        actionRequests={props?.actionRequests}
        isCurrentTurn={props?.isCurrentTurn}
        onOpenArtifactFromTimeline={props?.onOpenArtifactFromTimeline}
        sourceMessageId={props?.sourceMessageId}
        onSaveFileArtifactAsKnowledge={props?.onSaveFileArtifactAsKnowledge}
        onOpenSavedSiteContent={props?.onOpenSavedSiteContent}
        onOpenSubagentSession={props?.onOpenSubagentSession}
        focusedItemId={props?.focusedItemId}
        focusRequestKey={props?.focusRequestKey}
        deferCompletedSingleDetails={props?.deferCompletedSingleDetails}
        collapseInactiveDetails={props?.collapseInactiveDetails}
        showOperationalDetails={props?.showOperationalDetails}
        showInlineStatusHint={props?.showInlineStatusHint}
      />,
    );
  });

  mountedRoots.push({ container, root });
  return container;
}

export function createFileArtifactItem(
  overrides: Partial<Extract<AgentThreadItem, { type: "file_artifact" }>> = {},
): Extract<AgentThreadItem, { type: "file_artifact" }> {
  return {
    ...createBaseItem("artifact-1", 1),
    type: "file_artifact",
    path: "exports/x-article-export/google/index.md",
    source: "artifact_snapshot",
    content: JSON.stringify({
      schemaVersion: "artifact_document.v1",
      artifactId: "artifact-document:demo",
      kind: "analysis",
      title: "季度复盘",
      status: "ready",
      language: "zh-CN",
      blocks: [
        { id: "hero-1", type: "hero_summary", summary: "摘要" },
        { id: "body-1", type: "rich_text", markdown: "正文" },
      ],
      sources: [],
      metadata: {},
    }),
    metadata: {
      artifact_id: "artifact-document:demo",
      artifact_block_id: ["hero-1", "body-1"],
    },
    ...overrides,
  };
}

export function createStructuredA2UIParseResult() {
  return {
    parts: [
      { type: "text" as const, content: "请先确认以下选项：" },
      {
        type: "a2ui" as const,
        content: {
          id: "form-1",
          root: "root",
          components: [],
          submitAction: {
            label: "提交",
            action: { name: "submit" },
          },
        },
      },
    ],
    hasA2UI: true,
    hasWriteFile: false,
    hasPending: false,
  };
}
