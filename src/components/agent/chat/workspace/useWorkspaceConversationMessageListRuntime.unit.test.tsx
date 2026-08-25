import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useWorkspaceConversationMessageListRuntime } from "./useWorkspaceConversationMessageListRuntime";

type HookProps = Parameters<
  typeof useWorkspaceConversationMessageListRuntime
>[0];

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

function createDefaultProps(): HookProps {
  const noop = vi.fn();

  return {
    actions: {
      onA2UISubmit: noop,
      onArtifactClick: noop,
      onCodeBlockClick: noop,
      onDeleteMessage: noop,
      onEditMessage: noop,
      onFileClick: noop,
      onInterruptCurrentTurn: noop,
      onLoadFullHistory: noop,
      onOpenArtifactFromTimeline: noop,
      onOpenMessagePreview: noop,
      onOpenSavedSiteContent: noop,
      onOpenSubagentSession: noop,
      onOpenUrlPreview: noop,
      onPermissionResponse: noop,
      onReplayPendingRequest: noop,
      onSaveMessageAsKnowledge: noop,
      onSaveMessageAsSkill: noop,
      onWriteFile: noop,
    },
    collapseCodeBlocks: true,
    emptyStateVariant: "task-center",
    focus: {
      focusedTimelineItemId: "item-1",
      timelineFocusRequestKey: 7,
    },
    input: {
      quoteInput: "引用输入",
      onQuoteInputChange: noop,
    },
    pendingPromotedA2UIActionRequest: {
      requestId: "a2ui-1",
    },
    projection: {
      currentTurnId: "turn-1",
      executionRuntime: {
        provider_selector: "openai",
        model_name: "gpt-5",
      } as never,
      isSending: true,
      messages: [{ id: "message-1", content: "hello" }] as never,
      pendingActions: [{ requestId: "pending-1" }] as never,
      sessionHistoryWindow: {
        hasFullHistory: false,
      } as never,
      submittedActionsInFlight: [{ requestId: "submitted-1" }] as never,
      threadItems: [{ id: "thread-item-1" }] as never,
      threadRead: { session_id: "session-1" } as never,
      todoItems: [{ content: "todo" }] as never,
      turns: [{ id: "turn-1" }] as never,
    },
    provider: {
      accessMode: "default" as never,
      model: "gpt-5",
      providerType: "openai" as never,
      reasoningEffort: "medium",
    },
    refreshSessionReadModel: noop,
    sceneSessionId: "session-1",
    shouldCollapseCodeBlock: noop,
  };
}

function renderHook(props?: Partial<HookProps>) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const mergedProps = {
    ...createDefaultProps(),
    ...props,
  };
  let latestValue: ReturnType<
    typeof useWorkspaceConversationMessageListRuntime
  >;

  function Probe(currentProps: HookProps) {
    latestValue = useWorkspaceConversationMessageListRuntime(currentProps);
    return null;
  }

  act(() => {
    root.render(<Probe {...mergedProps} />);
  });
  mountedRoots.push({ root, container });

  return {
    getValue: () => latestValue,
    props: mergedProps,
  };
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) {
      break;
    }
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.clearAllMocks();
});

describe("useWorkspaceConversationMessageListRuntime", () => {
  it("应集中组装 conversation message list current runtime", () => {
    const { getValue, props } = renderHook();
    const runtime = getValue();

    expect(runtime.emptyStateVariant).toBe("task-center");
    expect(runtime.quoteInput).toBe("引用输入");
    expect(runtime.providerType).toBe("openai");
    expect(runtime.model).toBe("gpt-5");
    expect(runtime.reasoningEffort).toBe("medium");
    expect(runtime.messages).toBe(props.projection.messages);
    expect(runtime.turns).toBe(props.projection.turns);
    expect(runtime.threadItems).toBe(props.projection.threadItems);
    expect(runtime.currentTurnId).toBe("turn-1");
    expect(runtime.pendingActions).toBe(props.projection.pendingActions);
    expect(runtime).not.toHaveProperty("queuedTurns");
    expect(runtime.focusedTimelineItemId).toBe("item-1");
    expect(runtime.timelineFocusRequestKey).toBe(7);
    expect(runtime.onA2UISubmit).toBe(props.actions.onA2UISubmit);
    expect(runtime.onPermissionResponse).toBe(
      props.actions.onPermissionResponse,
    );
    expect(runtime).not.toHaveProperty("onPromoteQueuedTurn");
  });

  it("刷新 read model 时应绑定当前 scene session", () => {
    const refreshSessionReadModel = vi.fn();
    const { getValue } = renderHook({
      refreshSessionReadModel,
      sceneSessionId: "scene-session-2",
    });

    getValue().onRefreshSessionReadModel?.({ timelineMode: "replace" });

    expect(refreshSessionReadModel).toHaveBeenCalledWith("scene-session-2", {
      timelineMode: "replace",
    });
  });
});
