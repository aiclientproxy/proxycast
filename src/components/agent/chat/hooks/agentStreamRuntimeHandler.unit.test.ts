import i18n from "i18next";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  parseAgentEvent,
  type AgentEvent,
  type AgentThreadItem,
} from "@/lib/api/agentProtocol";
import {
  MAX_PROJECTION_OUTPUT_BYTES,
  PROJECTION_OUTPUT_TRUNCATION_MARKER,
} from "@/lib/api/agentRuntime/conversationProjection";
import type { Message } from "../types";
import {
  clearAgentUiProjectionEvents,
  conversationProjectionStore,
  selectAgentUiProjectionEvents,
} from "../projection/conversationProjectionStore";
import { handleTurnStreamEvent } from "./agentStreamRuntimeHandler";
import { AGENT_STREAM_TEXT_DELTA_RENDER_FLUSH_MS } from "./agentStreamTimerController";
import {
  clearAllAgentStreamTextOverlays,
  getAgentStreamTextOverlay,
} from "./agentStreamTextOverlayStore";

const { mockToast } = vi.hoisted(() => ({
  mockToast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  },
}));

vi.mock("sonner", () => ({
  toast: mockToast,
}));

describe("agentStreamRuntimeHandler", () => {
  beforeEach(async () => {
    document.documentElement.lang = "zh-CN";
    if (i18n.isInitialized) {
      await i18n.changeLanguage("zh-CN");
    }
  });

  afterEach(() => {
    vi.useRealTimers();
    document.documentElement.lang = "";
    mockToast.success.mockReset();
    mockToast.error.mockReset();
    mockToast.info.mockReset();
    mockToast.warning.mockReset();
    clearAgentUiProjectionEvents();
    clearAllAgentStreamTextOverlays();
  });

  async function flushProjectionQueue() {
    await Promise.resolve();
  }

  it("应在 reducer 边界异步记录标准 Agent UI projection envelope", async () => {
    clearAgentUiProjectionEvents();
    const requestState = {
      accumulatedContent: "",
      currentTurnId: "pending-turn-local",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };

    handleTurnStreamEvent({
      data: {
        type: "runtime_status",
        status: {
          phase: "routing",
          title: "选择模型",
          detail: "正在选择可用模型",
        },
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: vi.fn() as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(
      selectAgentUiProjectionEvents(conversationProjectionStore.getSnapshot()),
    ).toEqual([]);
    await flushProjectionQueue();
    const projectionEvents = selectAgentUiProjectionEvents(
      conversationProjectionStore.getSnapshot(),
    );
    expect(projectionEvents).toEqual([
      expect.objectContaining({
        type: "run.status",
        sourceType: "runtime_status",
        sequence: 1,
        sessionId: "session-1",
        runId: "agent-runtime-test",
        messageId: "assistant-1",
        owner: "runtime",
        scope: "run",
        phase: "routing",
        surface: "runtime_status",
      }),
      expect.objectContaining({
        type: "metric.changed",
        sourceType: "performance_metric",
        sessionId: "session-1",
        owner: "diagnostics",
        scope: "session",
        surface: "diagnostics",
        payload: expect.objectContaining({
          metricPhase: "agentStream.firstRuntimeStatus",
          source: "agent-stream",
          metrics: expect.objectContaining({
            eventName: "agent-runtime-test",
            phase: "routing",
            title: "选择模型",
          }),
        }),
      }),
    ]);
    expect(
      (requestState as { agentUiEventSequence?: number }).agentUiEventSequence,
    ).toBe(1);
  });

  it("provider_trace 早于首字时应补上轻量等待态，并允许后续 runtime_status 覆盖", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-09T10:00:00.300Z"));
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-05-09T10:00:00.000Z"),
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: Date.now() - 300,
      firstEventReceivedAt: Date.now() - 250,
      firstRuntimeStatusAt: null,
      requestFinished: false,
    } as Parameters<typeof handleTurnStreamEvent>[0]["requestState"];
    let activated = false;
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: () => {
          activated = true;
        },
        isStreamActivated: () => activated,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-provider-trace",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages,
      setPendingActions: vi.fn() as never,
      getThreadItems: () => threadItems,
      setThreadItems,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    } satisfies Omit<Parameters<typeof handleTurnStreamEvent>[0], "data">;

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        tool_id: "tool-image-workflow-stable",
        tool_name: "lime_create_image_generation_task",
        arguments: JSON.stringify({
          prompt: "从花城汇看广州塔的春天照片",
        }),
      } as AgentEvent,
    });

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "provider_trace",
        stage: "request_started",
        provider: "openai",
        model: "gpt-5",
        attempt: 1,
        elapsed_ms: 0,
      } as AgentEvent,
    });

    expect(activated).toBe(true);
    expect(requestState.firstRuntimeStatusAt).toBe(Date.now());
    expect(messages[0]?.runtimeStatus).toMatchObject({
      phase: "routing",
      title: "正在启动处理流程",
    });

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "runtime_status",
        status: {
          phase: "context",
          title: "正在整理上下文",
          detail: "已收到真实运行时状态",
        },
      } as AgentEvent,
    });

    expect(messages[0]?.runtimeStatus).toMatchObject({
      phase: "context",
      title: "正在整理上下文",
      detail: "已收到真实运行时状态",
    });
  });

  it("ImageCommandWorkflow 创建图片任务后应立即挂上图片任务预览", () => {
    let messages: Message[] = [
      {
        id: "assistant-image-workflow",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-02T10:00:00.000Z"),
        contentParts: [],
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    } as Parameters<typeof handleTurnStreamEvent>[0]["requestState"];

    handleTurnStreamEvent({
      data: {
        type: "image_task_created",
        task_id: "task-image-workflow-1",
        task_type: "image_generate",
        task_family: "image",
        status: "pending_submit",
        normalized_status: "pending",
        artifact_path: ".lime/tasks/image_generate/task-image-workflow-1.json",
        response: {
          task_id: "task-image-workflow-1",
          task_type: "image_generate",
          task_family: "image",
          status: "pending_submit",
          normalized_status: "pending",
          path: ".lime/tasks/image_generate/task-image-workflow-1.json",
          artifact_path:
            ".lime/tasks/image_generate/task-image-workflow-1.json",
          record: {
            payload: {
              prompt: "画一张广州夏天的图",
              model: "fal-ai/nano-banana-pro",
              session_id: "session-1",
              turn_id: "turn-image-workflow",
              presentation: {
                assistant_intro: "好啊，我来画广州夏天的光感。",
              },
            },
          },
        },
        payload: {
          prompt: "画一张广州夏天的图",
          model: "fal-ai/nano-banana-pro",
          session_id: "session-1",
          turn_id: "turn-image-workflow",
          presentation: {
            assistant_intro: "好啊，我来画广州夏天的光感。",
          },
        },
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-image-workflow",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image-workflow",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "画一张广州夏天的图",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(messages[0]?.imageWorkbenchPreview).toMatchObject({
      taskId: "task-image-workflow-1",
      prompt: "画一张广州夏天的图",
      status: "running",
      taskFilePath: ".lime/tasks/image_generate/task-image-workflow-1.json",
      artifactPath: ".lime/tasks/image_generate/task-image-workflow-1.json",
      modelName: "fal-ai/nano-banana-pro",
    });
    expect(messages[0]?.content).toBe("好啊，我来画广州夏天的光感。");
    expect(messages[0]?.content).not.toContain("先获取下工具参数");
    expect(messages[0]?.content).not.toContain("马上生成");
    expect(messages[0]?.taskPreview).toBeUndefined();
    expect(requestState.hasMeaningfulCompletionSignal).toBe(true);
    expect(requestState.currentTurnId).toBe("turn-image-workflow");
    expect(messages[0]?.runtimeTurnId).toBe("turn-image-workflow");
  });

  it("ImageCommandWorkflow 创建图片任务后应允许对应 turn_completed 收起发送态", () => {
    let messages: Message[] = [
      {
        id: "assistant-image-workflow-terminal",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-02T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
      {
        id: "image-workbench:task-image-workflow-terminal:assistant",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-02T10:00:00.000Z"),
        imageWorkbenchPreview: {
          taskId: "task-image-workflow-terminal",
          prompt: "画一张广州夏天的图",
          mode: "generate",
          status: "running",
        },
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      currentTurnId: "pending-turn-local",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    } as Parameters<typeof handleTurnStreamEvent>[0]["requestState"];
    const setIsSending = vi.fn();
    const disposeListener = vi.fn();
    const setCurrentTurnId = vi.fn();
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener,
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-image-workflow-terminal",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image-workflow-terminal",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "@配图 画一张广州夏天的图",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: setCurrentTurnId as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: setIsSending as never,
    } satisfies Omit<Parameters<typeof handleTurnStreamEvent>[0], "data">;

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "image_task_created",
        task_id: "task-image-workflow-terminal",
        task_type: "image_generate",
        task_family: "image_generation",
        status: "pending_submit",
        normalized_status: "pending",
        artifact_path:
          ".lime/tasks/image_generate/task-image-workflow-terminal.json",
        payload: {
          prompt: "画一张广州夏天的图",
          session_id: "session-1",
          turn_id: "turn-image-workflow-terminal",
        },
      } as AgentEvent,
    });

    requestState.currentTurnId = null;
    requestState.activeTextSegmentTurnId = "turn-image-workflow-terminal";
    messages = messages.map((message) => ({
      ...message,
      runtimeTurnId: undefined,
    }));
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "token_usage_updated",
        usage: {
          input_tokens: 31_000,
          output_tokens: 0,
          cached_input_tokens: 0,
        },
      } as AgentEvent,
    });

    expect(messages[0]).toMatchObject({
      runtimeTurnId: "turn-image-workflow-terminal",
      usage: {
        input_tokens: 31_000,
        output_tokens: 0,
      },
    });

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-image-workflow-terminal",
          thread_id: "thread-image-workflow-terminal",
          prompt_text: "@配图 画一张广州夏天的图",
          status: "completed",
          started_at: "2026-07-02T10:00:00.000Z",
          completed_at: "2026-07-02T10:00:01.000Z",
          created_at: "2026-07-02T10:00:00.000Z",
          updated_at: "2026-07-02T10:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(requestState.currentTurnId).toBe("turn-image-workflow-terminal");
    expect(setCurrentTurnId).toHaveBeenCalledWith(
      "turn-image-workflow-terminal",
    );
    expect(messages[0]?.runtimeTurnId).toBe("turn-image-workflow-terminal");
    expect(messages[0]).toMatchObject({
      runtimeTurnId: "turn-image-workflow-terminal",
      usage: {
        input_tokens: 31_000,
        output_tokens: 0,
      },
    });
    expect(setIsSending).toHaveBeenCalledWith(false);
    expect(disposeListener).toHaveBeenCalled();
  });

  it("ImageCommandWorkflow 创建事件不应在前端改写已有模型文案", () => {
    let messages: Message[] = [
      {
        id: "assistant-image-polluted-created",
        role: "assistant",
        content:
          "好啊，先来Generate深圳夏day午后的城市照片，阳光明亮，真实摄影Style。",
        timestamp: new Date("2026-07-02T10:00:00.000Z"),
        isThinking: true,
        imageWorkbenchPreview: {
          taskId: "task-image-polluted-created",
          prompt: "用 Agnes 生成一张深圳夏天午后的城市照片，真实摄影风格",
          status: "running",
          caption: "搞定，深圳夏day午后的城市照片，真实摄影Style 已经做好了。",
        },
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const requestState = {
      accumulatedContent:
        "好啊，先来Generate深圳夏day午后的城市照片，阳光明亮，真实摄影Style。",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    } as Parameters<typeof handleTurnStreamEvent>[0]["requestState"];

    handleTurnStreamEvent({
      data: {
        type: "image_task_created",
        task_id: "task-image-polluted-created",
        task_type: "image_generate",
        task_family: "image_generation",
        status: "pending_submit",
        normalized_status: "pending",
        artifact_path:
          ".lime/tasks/image_generate/task-image-polluted-created.json",
        response: {
          task_id: "task-image-polluted-created",
          task_type: "image_generate",
          task_family: "image_generation",
          status: "pending_submit",
          normalized_status: "pending",
          artifact_path:
            ".lime/tasks/image_generate/task-image-polluted-created.json",
          record: {
            payload: {
              prompt: "用 Agnes 生成一张深圳夏天午后的城市照片，真实摄影风格",
              model: "agnes-image-2.1-flash",
              session_id: "session-1",
              turn_id: "turn-image-polluted-created",
            },
          },
        },
        payload: {
          prompt: "用 Agnes 生成一张深圳夏天午后的城市照片，真实摄影风格",
          model: "agnes-image-2.1-flash",
          session_id: "session-1",
          turn_id: "turn-image-polluted-created",
        },
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-image-polluted-created",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image-polluted-created",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content:
        "@配图 用 Agnes Generate一张深圳夏day午后的城市照片，真实摄影Style",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(messages[0]?.content).toContain("Generate深圳夏day午后");
    expect(messages[0]?.content).toContain("真实摄影Style");
    expect(messages[0]?.imageWorkbenchPreview?.caption).toContain("深圳夏day");
    expect(messages[0]?.imageWorkbenchPreview?.caption).toContain(
      "真实摄影Style",
    );
  });

  it("ImageCommandWorkflow 创建事件找不到 assistant shell 时应补回稳定图片轻卡", () => {
    let messages: Message[] = [];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    } as Parameters<typeof handleTurnStreamEvent>[0]["requestState"];

    handleTurnStreamEvent({
      data: {
        type: "image_task_created",
        task_id: "task-image-workflow-upsert",
        task_type: "image_generate",
        task_family: "image_generation",
        status: "pending_submit",
        normalized_status: "pending",
        artifact_path:
          ".lime/tasks/image_generate/task-image-workflow-upsert.json",
        payload: {
          prompt: "广州塔春天照片",
          session_id: "session-1",
          turn_id: "turn-image-workflow-upsert",
        },
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-image-workflow-upsert",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image-workflow-upsert",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "广州塔春天照片",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(messages).toHaveLength(1);
    expect(messages[0]).toMatchObject({
      id: "assistant-image-workflow-upsert",
      role: "assistant",
      isThinking: true,
      imageWorkbenchPreview: {
        taskId: "task-image-workflow-upsert",
        prompt: "广州塔春天照片",
        status: "running",
      },
    });
    expect(messages[0]?.content).toBe("");
    expect(messages[0]?.contentParts).toBeUndefined();
    expect(messages[0]?.content).not.toContain("已发起");
    expect(requestState.hasMeaningfulCompletionSignal).toBe(true);
  });

  it("ImageCommandWorkflow presentation 先于创建事件到达时应合入同一图片轻卡", () => {
    let messages: Message[] = [];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    } as Parameters<typeof handleTurnStreamEvent>[0]["requestState"];
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-image-presentation-before-created",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image-presentation-before-created",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "@配图 画一张深圳夏天的图",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    } satisfies Omit<Parameters<typeof handleTurnStreamEvent>[0], "data">;

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "image_task_presentation_generated",
        status: "generated",
        workflow_run_id: "workflow-image-race",
        session_id: "session-1",
        thread_id: "thread-1",
        turn_id: "turn-image-race",
        presentation: {
          assistant_intro: "好啊，我来做一张深圳夏天的明亮城市照片。",
          completion_caption:
            "完成了，深圳夏天的阳光、绿意和城市通透感已经出来。",
        },
      } as AgentEvent,
    });

    expect(requestState.pendingImageTaskPresentation).toMatchObject({
      assistantIntro: "好啊，我来做一张深圳夏天的明亮城市照片。",
      completionCaption: "完成了，深圳夏天的阳光、绿意和城市通透感已经出来。",
      workflowRunId: "workflow-image-race",
      turnId: "turn-image-race",
    });

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "image_task_created",
        task_id: "task-image-race",
        task_type: "image_generate",
        task_family: "image_generation",
        status: "pending_submit",
        normalized_status: "pending",
        artifact_path: ".lime/tasks/image_generate/task-image-race.json",
        response: {
          task_id: "task-image-race",
          task_type: "image_generate",
          task_family: "image_generation",
          status: "pending_submit",
          normalized_status: "pending",
          artifact_path: ".lime/tasks/image_generate/task-image-race.json",
          record: {
            payload: {
              prompt: "画一张深圳夏天的图",
              model: "agnes-image-2.1-flash",
              session_id: "session-1",
              turn_id: "turn-image-race",
              workflow_run_id: "workflow-image-race",
            },
          },
        },
        payload: {
          prompt: "画一张深圳夏天的图",
          model: "agnes-image-2.1-flash",
          session_id: "session-1",
          turn_id: "turn-image-race",
          workflow_run_id: "workflow-image-race",
        },
      } as AgentEvent,
    });

    expect(messages).toHaveLength(1);
    expect(messages[0]?.content).toBe(
      "好啊，我来做一张深圳夏天的明亮城市照片。",
    );
    expect(messages[0]?.contentParts).toEqual([
      {
        type: "text",
        text: "好啊，我来做一张深圳夏天的明亮城市照片。",
      },
    ]);
    expect(messages[0]?.imageWorkbenchPreview).toMatchObject({
      taskId: "task-image-race",
      prompt: "画一张深圳夏天的图",
      status: "running",
      modelName: "agnes-image-2.1-flash",
      caption: "完成了，深圳夏天的阳光、绿意和城市通透感已经出来。",
    });
    expect(requestState.pendingImageTaskPresentation).toBeNull();
  });

  it("ImageCommandWorkflow presentation 事件应替换状态模板并缓存完成描述", () => {
    let messages: Message[] = [
      {
        id: "assistant-image-presentation",
        role: "assistant",
        content: "已发起 广州塔春天照片 的图片生成",
        timestamp: new Date("2026-07-02T10:00:00.000Z"),
        isThinking: true,
        imageWorkbenchPreview: {
          taskId: "task-image-presentation",
          prompt: "从花城汇看广州塔的春天照片",
          status: "running",
          modelName: "fal-ai/nano-banana-pro",
        },
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    } as Parameters<typeof handleTurnStreamEvent>[0]["requestState"];

    handleTurnStreamEvent({
      data: {
        type: "image_task_presentation_generated",
        status: "generated",
        workflow_run_id: "workflow-image-presentation",
        session_id: "session-1",
        thread_id: "thread-1",
        turn_id: "turn-image-presentation",
        presentation: {
          assistant_intro: "好啊，我来按花城汇视角做一张广州塔春天照片。",
          completion_caption: "完成了，从花城汇望向广州塔的春日画面已经生成。",
        },
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-image-presentation",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image-presentation",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "@Nanobanana Pro 生成一张广州塔春天照片",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(messages[0]?.content).toBe(
      "好啊，我来按花城汇视角做一张广州塔春天照片。",
    );
    expect(messages[0]?.contentParts).toEqual([
      {
        type: "text",
        text: "好啊，我来按花城汇视角做一张广州塔春天照片。",
      },
    ]);
    expect(messages[0]?.content).not.toContain("已发起");
    expect(messages[0]?.imageWorkbenchPreview?.caption).toBe(
      "完成了，从花城汇望向广州塔的春日画面已经生成。",
    );
    expect(messages[0]?.imageWorkbenchPreview?.status).toBe("running");
  });

  it("ImageCommandWorkflow presentation 事件不应覆盖已有自然寒暄和思考", () => {
    let messages: Message[] = [
      {
        id: "assistant-image-presentation-natural",
        role: "assistant",
        content: "我先按花城汇视角构图，保留春花、广场和广州塔。",
        timestamp: new Date("2026-07-02T10:00:00.000Z"),
        isThinking: true,
        thinkingContent: "先判断视角、季节和画面主体。",
        contentParts: [
          { type: "thinking", text: "先判断视角、季节和画面主体。" },
          {
            type: "text",
            text: "我先按花城汇视角构图，保留春花、广场和广州塔。",
          },
        ],
        imageWorkbenchPreview: {
          taskId: "task-image-presentation-natural",
          prompt: "从花城汇看广州塔的春天照片",
          status: "running",
          modelName: "fal-ai/nano-banana-pro",
        },
      },
    ];
    const originalContentParts = messages[0]?.contentParts;
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const requestState = {
      accumulatedContent: "我先按花城汇视角构图，保留春花、广场和广州塔。",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    } as Parameters<typeof handleTurnStreamEvent>[0]["requestState"];

    handleTurnStreamEvent({
      data: {
        type: "image_task_presentation_generated",
        status: "generated",
        workflow_run_id: "workflow-image-presentation-natural",
        session_id: "session-1",
        thread_id: "thread-1",
        turn_id: "turn-image-presentation-natural",
        presentation: {
          assistant_intro: "好啊，我马上生成这张广州塔春天照片。",
          completion_caption: "完成了，花城汇望向广州塔的春日画面已经生成。",
        },
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-image-presentation-natural",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image-presentation-natural",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "@Nanobanana Pro 生成一张广州塔春天照片",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(messages[0]?.content).toBe(
      "我先按花城汇视角构图，保留春花、广场和广州塔。",
    );
    expect(messages[0]?.contentParts).toEqual(originalContentParts);
    expect(messages[0]?.thinkingContent).toBe("先判断视角、季节和画面主体。");
    expect(messages[0]?.imageWorkbenchPreview?.caption).toBe(
      "完成了，花城汇望向广州塔的春日画面已经生成。",
    );
    expect(JSON.stringify(messages[0])).not.toContain("马上生成");
  });

  it("ImageCommandWorkflow presentation 事件不应在前端语义改写已有模型文案", () => {
    let messages: Message[] = [
      {
        id: "assistant-image-polluted-presentation",
        role: "assistant",
        content:
          "好啊，先来Generate 深圳夏day午后的城市照片，阳光明亮，真实摄影Style。",
        timestamp: new Date("2026-07-02T10:00:00.000Z"),
        isThinking: true,
        imageWorkbenchPreview: {
          taskId: "task-image-polluted-presentation",
          prompt:
            "Generate 深圳夏day午后的城市照片，阳光明亮，街边绿树和高楼，真实摄影Style",
          status: "running",
          modelName: "agnes-image-2.1-flash",
        },
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const requestState = {
      accumulatedContent:
        "好啊，先来Generate 深圳夏day午后的城市照片，阳光明亮，真实摄影Style。",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    } as Parameters<typeof handleTurnStreamEvent>[0]["requestState"];

    handleTurnStreamEvent({
      data: {
        type: "image_task_presentation_generated",
        status: "generated",
        workflow_run_id: "workflow-image-polluted-presentation",
        session_id: "session-1",
        thread_id: "thread-1",
        turn_id: "turn-image-polluted-presentation",
        presentation: {
          assistant_intro:
            "好啊，先来Generate 深圳夏day午后的城市照片，阳光明亮，真实摄影Style。",
          completion_caption:
            "搞定，深圳夏day午后的城市照片，阳光明亮，真实摄影Style 已经做好了。",
        },
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-image-polluted-presentation",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image-polluted-presentation",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content:
        "@配图 用 Agnes Generate一张深圳夏day午后的城市照片，阳光明亮，真实摄影Style",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(messages[0]?.content).toContain("Generate 深圳夏day午后");
    expect(messages[0]?.content).toContain("真实摄影Style");
    expect(messages[0]?.imageWorkbenchPreview?.caption).toContain("搞定");
    expect(messages[0]?.imageWorkbenchPreview?.caption).toContain("深圳夏day");
    expect(messages[0]?.imageWorkbenchPreview?.caption).toContain(
      "真实摄影Style",
    );
  });

  it("ImageCommandWorkflow 空 turn_completed 后应保留寒暄和运行中图片轻卡", () => {
    let messages: Message[] = [
      {
        id: "assistant-image-workflow-stable",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-02T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    } as Parameters<typeof handleTurnStreamEvent>[0]["requestState"];
    const setIsSending = vi.fn();
    const disposeListener = vi.fn();
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener,
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-image-workflow-stable",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image-workflow-stable",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "@Nanobanana Pro 生成一张广州塔春天照片",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: setIsSending as never,
    } satisfies Omit<Parameters<typeof handleTurnStreamEvent>[0], "data">;

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: "好啊，我来按花城汇视角做一张广州塔春天照片。",
        phase: "final_answer",
      } as AgentEvent,
    });

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_end",
        tool_id: "tool-image-workflow-stable",
        result: {
          success: true,
          output: JSON.stringify({
            success: true,
            task_id: "task-image-workflow-stable",
            task_type: "image_generate",
            task_family: "image_generation",
            status: "pending_submit",
            normalized_status: "pending",
            artifact_path:
              ".lime/tasks/image_generate/task-image-workflow-stable.json",
            record: {
              payload: {
                prompt: "从花城汇看广州塔的春天照片",
                session_id: "session-1",
                turn_id: "turn-image-workflow-stable",
              },
            },
          }),
          metadata: {
            task_id: "task-image-workflow-stable",
            task_type: "image_generate",
            task_family: "image_generation",
            status: "pending_submit",
            normalized_status: "pending",
            artifact_path:
              ".lime/tasks/image_generate/task-image-workflow-stable.json",
          },
        },
      } as AgentEvent,
    });

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "image_task_created",
        task_id: "task-image-workflow-stable",
        task_type: "image_generate",
        task_family: "image_generation",
        status: "pending_submit",
        normalized_status: "pending",
        artifact_path:
          ".lime/tasks/image_generate/task-image-workflow-stable.json",
        response: {
          task_id: "task-image-workflow-stable",
          task_type: "image_generate",
          task_family: "image_generation",
          status: "pending_submit",
          normalized_status: "pending",
          artifact_path:
            ".lime/tasks/image_generate/task-image-workflow-stable.json",
          record: {
            payload: {
              prompt: "从花城汇看广州塔的春天照片",
              session_id: "session-1",
              turn_id: "turn-image-workflow-stable",
            },
          },
        },
        payload: {
          prompt: "从花城汇看广州塔的春天照片",
          session_id: "session-1",
          turn_id: "turn-image-workflow-stable",
        },
      } as AgentEvent,
    });

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-image-workflow-stable",
          thread_id: "thread-image-workflow-stable",
          prompt_text: "@Nanobanana Pro 生成一张广州塔春天照片",
          status: "completed",
          started_at: "2026-07-02T10:00:00.000Z",
          completed_at: "2026-07-02T10:00:01.000Z",
          created_at: "2026-07-02T10:00:00.000Z",
          updated_at: "2026-07-02T10:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(messages).toHaveLength(1);
    expect(messages[0]?.content).toBe(
      "好啊，我来按花城汇视角做一张广州塔春天照片。",
    );
    expect(messages[0]?.content).not.toContain("先获取下工具参数");
    expect(messages[0]?.content).not.toContain("马上生成");
    expect(messages[0]?.content).not.toContain("本轮执行已完成");
    expect(messages[0]?.content).not.toContain('"success":true');
    expect(JSON.stringify(messages[0]?.contentParts || [])).not.toContain(
      "task_id",
    );
    expect(messages[0]?.toolCalls).toBeUndefined();
    expect(messages[0]?.imageWorkbenchPreview).toMatchObject({
      taskId: "task-image-workflow-stable",
      prompt: "从花城汇看广州塔的春天照片",
      status: "running",
      phase: "queued",
    });
    expect(messages[0]?.taskPreview).toBeUndefined();
    expect(messages[0]?.isThinking).toBe(false);
    expect(setIsSending).toHaveBeenCalledWith(false);
    expect(disposeListener).toHaveBeenCalledTimes(1);
  });

  it("图片任务 tool_end 仅命中宽松 task 形态时也不回退普通工具过程", () => {
    let messages: Message[] = [
      {
        id: "assistant-image-workflow-loose-task",
        role: "assistant",
        content: "好啊，我来画一张深圳夏天的城市画面。",
        timestamp: new Date("2026-07-02T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const requestState = {
      accumulatedContent: "好啊，我来画一张深圳夏天的城市画面。",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    } as Parameters<typeof handleTurnStreamEvent>[0]["requestState"];
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-image-workflow-loose-task",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image-workflow-loose-task",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "@Nanobanana Pro 生成深圳夏天图片",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    } satisfies Omit<Parameters<typeof handleTurnStreamEvent>[0], "data">;

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        tool_id: "tool-image-loose-task",
        tool_name: "lime_create_image_generation_task",
        arguments: JSON.stringify({ prompt: "深圳夏天" }),
      } as AgentEvent,
    });
    expect(messages[0]?.toolCalls).toHaveLength(1);

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_end",
        tool_id: "tool-image-loose-task",
        result: {
          success: true,
          output: "",
          metadata: {
            id: "task-image-loose-task",
            task_family: "image_generation",
            status: "pending_submit",
          },
        },
      } as AgentEvent,
    });

    expect(messages[0]?.content).toBe("好啊，我来画一张深圳夏天的城市画面。");
    expect(messages[0]?.toolCalls).toBeUndefined();
    expect(messages[0]?.contentParts).toEqual([
      { type: "text", text: "好啊，我来画一张深圳夏天的城市画面。" },
    ]);
    expect(messages[0]?.taskPreview).toBeUndefined();
    expect(messages[0]?.imageWorkbenchPreview).toBeUndefined();
  });

  it("ImageCommandWorkflow 补参态应以 runtime_status 投影且不转成 action_required", () => {
    let messages: Message[] = [
      {
        id: "assistant-image-workflow-status",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-02T10:00:00.000Z"),
        contentParts: [],
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
      hasMeaningfulCompletionSignal: false,
    } as Parameters<typeof handleTurnStreamEvent>[0]["requestState"];

    handleTurnStreamEvent({
      data: {
        type: "runtime_status",
        status: {
          phase: "routing",
          title: "图片生成需要补充信息",
          detail: "缺少: project_root_path",
          metadata: {
            source: "image_command_workflow",
            agentui: {
              workflow_key: "image_command_workflow",
              status_kind: "image_task_parameters_required",
              missing: ["project_root_path"],
              missing_parameters: ["project_root_path"],
              image_task: {
                prompt: "画一张广州夏天的图",
              },
            },
          },
        },
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-image-workflow-status",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image-workflow-status",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(messages[0]?.runtimeStatus).toMatchObject({
      phase: "routing",
      title: "图片生成需要补充信息",
      detail: "缺少: project_root_path",
      metadata: {
        source: "image_command_workflow",
        agentui: {
          workflow_key: "image_command_workflow",
          status_kind: "image_task_parameters_required",
          missing: ["project_root_path"],
        },
      },
    });
    expect(requestState.hasMeaningfulCompletionSignal).toBe(true);
    expect(messages[0]?.actionRequests ?? []).toEqual([]);
  });

  it("图片任务 tool_end 不应把内部 task JSON 写入通用 artifact", () => {
    let messages: Message[] = [
      {
        id: "assistant-image-tool-result",
        role: "assistant",
        content: "画一张广州夏天的图",
        timestamp: new Date("2026-07-02T10:05:00.000Z"),
        contentParts: [],
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const onWriteFile = vi.fn();
    const toolNameByToolId = new Map<string, string>();
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    } as Parameters<typeof handleTurnStreamEvent>[0]["requestState"];
    const callbacks = {
      activateStream: vi.fn(),
      isStreamActivated: () => true,
      clearOptimisticItem: () => {},
      clearOptimisticTurn: () => {},
      disposeListener: () => {},
      clearActiveStreamIfMatch: () => true,
      appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
        parts,
    };
    const baseOptions = {
      requestState,
      callbacks,
      eventName: "agent-runtime-image-tool-result",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image-tool-result",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "画一张广州夏天的图",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId,
      onWriteFile,
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };
    const taskSnapshot = {
      success: true,
      task_id: "task-image-tool-result-1",
      task_type: "image_generate",
      task_family: "image",
      status: "pending_submit",
      normalized_status: "pending",
      artifact_path: ".lime/tasks/image_generate/task-image-tool-result-1.json",
      path: ".lime/tasks/image_generate/task-image-tool-result-1.json",
      record: {
        payload: {
          prompt: "画一张广州夏天的图",
          session_id: "session-1",
        },
      },
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        tool_id: "tool-image-task-1",
        tool_name: "mediaTaskArtifact/image/create",
        arguments: JSON.stringify({
          prompt: "画一张广州夏天的图",
        }),
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_end",
        tool_id: "tool-image-task-1",
        result: {
          success: true,
          output: JSON.stringify(taskSnapshot),
          metadata: taskSnapshot,
        },
      } as AgentEvent,
    });

    expect(onWriteFile).not.toHaveBeenCalled();
    expect(messages[0]?.artifacts ?? []).toEqual([]);
    expect(messages[0]?.imageWorkbenchPreview).toMatchObject({
      taskId: "task-image-tool-result-1",
      prompt: "画一张广州夏天的图",
      status: "running",
      taskFilePath: ".lime/tasks/image_generate/task-image-tool-result-1.json",
      artifactPath: ".lime/tasks/image_generate/task-image-tool-result-1.json",
    });
    expect(messages[0]?.taskPreview).toBeUndefined();
  });

  it("应把工具进度和输出增量异步写入 projection，并同步更新运行中工具卡", async () => {
    clearAgentUiProjectionEvents();
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-05-09T10:00:00.000Z"),
        toolCalls: [
          {
            id: "tool-1",
            name: "mcp__runner__execute",
            arguments: "{}",
            status: "running",
            startTime: new Date("2026-05-09T10:00:00.000Z"),
          },
        ],
        contentParts: [
          {
            type: "tool_use",
            toolCall: {
              id: "tool-1",
              name: "mcp__runner__execute",
              arguments: "{}",
              status: "running",
              startTime: new Date("2026-05-09T10:00:00.000Z"),
            },
          },
        ],
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_input_delta",
        tool_id: "tool-1",
        tool_name: "mcp__runner__execute",
        delta: '{"command"',
        accumulated_arguments: '{"command"',
        provider: "openai_compatible",
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_progress",
        tool_id: "tool-1",
        progress: {
          message: "正在处理第 2 项",
          progress: 2,
          total: 4,
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_output_delta",
        tool_id: "tool-1",
        delta: "partial output",
        output_kind: "log",
      } as AgentEvent,
    });

    expect(messages[0]?.toolCalls?.[0]).toMatchObject({
      arguments: '{"command"',
      progress: {
        message: "正在处理第 2 项",
        progress: 2,
        total: 4,
      },
      result: {
        success: true,
        output: "partial output",
        metadata: {
          streaming: true,
          output_kind: "log",
        },
      },
      logs: [
        '正在生成工具输入：{"command"',
        "正在处理第 2 项",
        "partial output",
      ],
    });
    expect(messages[0]?.contentParts?.[0]).toMatchObject({
      type: "tool_use",
      toolCall: {
        result: {
          output: "partial output",
        },
      },
    });

    expect(
      selectAgentUiProjectionEvents(conversationProjectionStore.getSnapshot()),
    ).toEqual([]);
    await flushProjectionQueue();
    expect(
      selectAgentUiProjectionEvents(conversationProjectionStore.getSnapshot()),
    ).toEqual([
      expect.objectContaining({
        type: "tool.args.delta",
        sourceType: "tool_input_delta",
        sequence: 1,
        toolCallId: "tool-1",
      }),
      expect.objectContaining({
        type: "tool.progress",
        sourceType: "tool_progress",
        sequence: 2,
        toolCallId: "tool-1",
      }),
      expect.objectContaining({
        type: "tool.output.delta",
        sourceType: "tool_output_delta",
        sequence: 3,
        toolCallId: "tool-1",
      }),
    ]);
  });

  it("legacy 工具投影出的 thread item 不应阻止 tool_end 更新 message 层", async () => {
    let messages: Message[] = [
      {
        id: "assistant-legacy-tool",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-21T10:00:00.000Z"),
        runtimeTurnId: "turn-legacy-tool",
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      currentTurnId: "turn-legacy-tool",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-legacy-tool-event-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-legacy-tool",
      activeSessionId: "session-legacy-tool",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      getThreadItems: () => threadItems,
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        tool_id: "tool-legacy-1",
        tool_name: "web_search",
        turn_id: "turn-legacy-tool",
        arguments: JSON.stringify({ query: "Codex skills" }),
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_end",
        tool_id: "tool-legacy-1",
        turn_id: "turn-legacy-tool",
        result: {
          success: true,
          output: "搜索完成",
          metadata: {
            sourceLinks: [
              {
                url: "https://example.com/codex-skills",
                title: "Codex Skills",
              },
            ],
          },
        },
      } as AgentEvent,
    });

    expect(threadItems[0]).toMatchObject({
      id: "tool-legacy-1",
      type: "tool_call",
      status: "completed",
      metadata: expect.objectContaining({
        runtime_event_source: "legacy_tool_event",
      }),
    });
    expect(messages[0]?.toolCalls?.[0]).toMatchObject({
      id: "tool-legacy-1",
      status: "completed",
      result: {
        success: true,
        output: "搜索完成",
      },
    });
    expect(messages[0]?.contentParts?.[0]).toMatchObject({
      type: "tool_use",
      toolCall: {
        id: "tool-legacy-1",
        status: "completed",
        result: {
          success: true,
          output: "搜索完成",
        },
      },
    });
  });

  it("已有 item lifecycle 时 legacy 工具增量不应新建 message.toolCalls", async () => {
    clearAgentUiProjectionEvents();
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-19T10:00:00.000Z"),
        runtimeTurnId: "turn-1",
      },
    ];
    let threadItems: AgentThreadItem[] = [
      {
        id: "tool-1",
        thread_id: "thread-1",
        turn_id: "turn-1",
        sequence: 2,
        type: "tool_call",
        status: "in_progress",
        tool_name: "mcp__runner__execute",
        arguments: { command: "npm test" },
        metadata: {
          source: "item_lifecycle",
        },
        started_at: "2026-06-19T10:00:00.000Z",
        updated_at: "2026-06-19T10:00:00.000Z",
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      currentTurnId: "turn-1",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-item-first-delta-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>([
        ["tool-1", "mcp__runner__execute"],
      ]),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      getThreadItems: () => threadItems,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_input_delta",
        tool_id: "tool-1",
        tool_name: "mcp__runner__execute",
        turn_id: "turn-1",
        delta: '{"command":"npm test"}',
        accumulated_arguments: '{"command":"npm test"}',
        provider: "openai_compatible",
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_progress",
        tool_id: "tool-1",
        turn_id: "turn-1",
        progress: {
          message: "正在执行测试",
          progress: 1,
          total: 2,
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_output_delta",
        tool_id: "tool-1",
        turn_id: "turn-1",
        delta: "partial output",
        output_kind: "log",
      } as AgentEvent,
    });

    expect(messages[0]?.toolCalls).toBeUndefined();
    expect(messages[0]?.contentParts).toBeUndefined();
    expect(threadItems[0]).toMatchObject({
      id: "tool-1",
      type: "tool_call",
      status: "in_progress",
      arguments: { command: "npm test" },
      output: "partial output",
      metadata: expect.objectContaining({
        source: "item_lifecycle",
        output_kind: "log",
        streaming: true,
        progress: {
          message: "正在执行测试",
          progress: 1,
          total: 2,
        },
      }),
    });
    expect(
      selectAgentUiProjectionEvents(conversationProjectionStore.getSnapshot()),
    ).toEqual([]);
    await flushProjectionQueue();
    expect(
      selectAgentUiProjectionEvents(conversationProjectionStore.getSnapshot()),
    ).toEqual([
      expect.objectContaining({
        type: "tool.args.delta",
        sourceType: "tool_input_delta",
        toolCallId: "tool-1",
      }),
      expect.objectContaining({
        type: "tool.progress",
        sourceType: "tool_progress",
        toolCallId: "tool-1",
      }),
      expect.objectContaining({
        type: "tool.output.delta",
        sourceType: "tool_output_delta",
        toolCallId: "tool-1",
      }),
    ]);
  });

  it("已有 item lifecycle 时 App Server tool.failed 不应再改 message 层工具卡", async () => {
    clearAgentUiProjectionEvents();
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-05-09T10:00:00.000Z"),
        toolCalls: [
          {
            id: "tool-failed-1",
            name: "Bash",
            arguments: "{}",
            status: "running",
            startTime: new Date("2026-05-09T10:00:00.000Z"),
          },
        ],
        contentParts: [
          {
            type: "tool_use",
            toolCall: {
              id: "tool-failed-1",
              name: "Bash",
              arguments: "{}",
              status: "running",
              startTime: new Date("2026-05-09T10:00:00.000Z"),
            },
          },
        ],
      },
    ];
    let threadItems: AgentThreadItem[] = [
      {
        id: "tool-failed-1",
        thread_id: "session-1",
        turn_id: "turn-1",
        type: "tool_call",
        sequence: 1,
        status: "in_progress",
        started_at: "2026-05-09T10:00:00.000Z",
        updated_at: "2026-05-09T10:00:00.000Z",
        tool_name: "Bash",
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );
    const parsed = parseAgentEvent({
      type: "tool.failed",
      toolCallId: "tool-failed-1",
      status: "failed",
      error: "exit code 101",
      output: "test failed",
      metadata: {
        failureCategory: "test_failed",
      },
    });

    expect(parsed).toBeTruthy();
    handleTurnStreamEvent({
      data: parsed as AgentEvent,
      requestState: {
        accumulatedContent: "",
        currentTurnId: "turn-1",
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>([["tool-failed-1", "Bash"]]),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      getThreadItems: () => threadItems,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(messages[0]?.toolCalls?.[0]).toMatchObject({
      id: "tool-failed-1",
      status: "running",
    });
    expect(messages[0]?.contentParts?.[0]).toMatchObject({
      type: "tool_use",
      toolCall: {
        id: "tool-failed-1",
        status: "running",
      },
    });
    expect(threadItems[0]).toMatchObject({
      id: "tool-failed-1",
      type: "tool_call",
      status: "failed",
      output: "test failed",
      error: "exit code 101",
    });
    expect(
      selectAgentUiProjectionEvents(conversationProjectionStore.getSnapshot()),
    ).toEqual([]);
    await flushProjectionQueue();
    expect(
      selectAgentUiProjectionEvents(conversationProjectionStore.getSnapshot()),
    ).toEqual([
      expect.objectContaining({
        type: "tool.failed",
        sourceType: "tool_end",
        toolCallId: "tool-failed-1",
      }),
    ]);
  });

  it("item_completed 只更新 canonical ThreadItem，不再改写 legacy Message 工具卡", () => {
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-18T08:00:00.000Z"),
        runtimeTurnId: "turn-1",
        toolCalls: [
          {
            id: "tool-search-1",
            name: "web_search",
            arguments: JSON.stringify({ query: "学习机评测" }),
            status: "running",
            startTime: new Date("2026-06-18T08:00:00.000Z"),
          },
        ],
        contentParts: [
          {
            type: "tool_use",
            toolCall: {
              id: "tool-search-1",
              name: "web_search",
              arguments: JSON.stringify({ query: "学习机评测" }),
              status: "running",
              startTime: new Date("2026-06-18T08:00:00.000Z"),
            },
          },
        ],
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );

    handleTurnStreamEvent({
      data: {
        type: "item_completed",
        protocol_method: "item/completed",
        protocol_revision: "fixture-revision",
        item: {
          id: "tool-search-1",
          thread_id: "session-1",
          turn_id: "turn-1",
          sequence: 4,
          status: "completed",
          started_at: "2026-06-18T08:00:00.000Z",
          updated_at: "2026-06-18T08:00:02.000Z",
          completed_at: "2026-06-18T08:00:02.000Z",
          type: "tool_call",
          tool_name: "web_search",
          arguments: { query: "学习机评测" },
          output: "权威评测摘要",
          success: true,
          metadata: {
            source: "item_lifecycle",
          },
        },
      } as AgentEvent,
      requestState: {
        accumulatedContent: "",
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-item-tool-sync-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      getThreadItems: () => threadItems,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(threadItems[0]).toMatchObject({
      id: "tool-search-1",
      type: "tool_call",
      status: "completed",
      output: "权威评测摘要",
    });
    expect(messages[0]?.toolCalls?.[0]).toMatchObject({
      id: "tool-search-1",
      status: "running",
    });
    expect(messages[0]?.toolCalls?.[0]?.result).toBeUndefined();
    expect(messages[0]?.contentParts?.[0]).toMatchObject({
      type: "tool_use",
      toolCall: {
        id: "tool-search-1",
        status: "running",
      },
    });
  });

  it("direct command output 只由 ConversationProjection 写回并保持有界 canonical 类型", () => {
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-29T01:00:00.000Z"),
        runtimeTurnId: "turn-1",
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const requestState = {
      accumulatedContent: "",
      currentTurnId: "turn-1",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );
    const dispatch = (data: AgentEvent) =>
      handleTurnStreamEvent({
        data,
        requestState,
        callbacks: {
          activateStream: () => {},
          isStreamActivated: () => true,
          clearOptimisticItem: () => {},
          clearOptimisticTurn: () => {},
          disposeListener: () => {},
          clearActiveStreamIfMatch: () => true,
          appendThinkingToParts: (
            parts: NonNullable<Message["contentParts"]>,
          ) => parts,
        },
        eventName: "agent-runtime-direct-command-projection-test",
        pendingTurnKey: "pending-turn",
        pendingItemKey: "pending-item",
        assistantMsgId: "assistant-1",
        activeSessionId: "session-1",
        resolvedWorkspaceId: "workspace-1",
        effectiveExecutionStrategy: "react",
        content: "",
        runtime: {} as never,
        warnedKeysRef: { current: new Set<string>() },
        actionLoggedKeys: new Set<string>(),
        toolLogIdByToolId: new Map<string, string>(),
        toolStartedAtByToolId: new Map<string, number>(),
        toolNameByToolId: new Map<string, string>(),
        setMessages: setMessages as never,
        setPendingActions: vi.fn() as never,
        getThreadItems: () => threadItems,
        setThreadItems: setThreadItems as never,
        setThreadTurns: vi.fn() as never,
        setCurrentTurnId: vi.fn() as never,
        setExecutionRuntime: vi.fn() as never,
        setIsSending: vi.fn() as never,
      });

    dispatch({
      type: "tool_output_delta",
      thread_id: "thread-1",
      turn_id: "turn-1",
      tool_id: "command-1",
      source_item_id: "command-1",
      protocol_method: "item/commandExecution/outputDelta",
      protocol_revision: "fixture-revision",
      delta: "buffered-before-start\n",
    } as AgentEvent);

    expect(threadItems).toEqual([]);
    expect(messages[0]?.toolCalls).toBeUndefined();
    expect(messages[0]?.contentParts).toBeUndefined();

    const commandItem: AgentThreadItem = {
      id: "command-1",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 3,
      status: "in_progress",
      started_at: "2026-07-29T01:00:00.000Z",
      updated_at: "2026-07-29T01:00:00.000Z",
      type: "command_execution",
      command: "generate-output",
      cwd: "/workspace",
    };
    dispatch({
      type: "item_started",
      protocol_method: "item/started",
      protocol_revision: "fixture-revision",
      item: commandItem,
    } as AgentEvent);

    expect(threadItems).toHaveLength(1);
    expect(threadItems[0]).toMatchObject({
      id: "command-1",
      type: "command_execution",
      aggregated_output: "buffered-before-start\n",
    });

    const oversizedDelta = `discarded-head-${"d".repeat(
      MAX_PROJECTION_OUTPUT_BYTES + 128,
    )}-kept-tail`;
    dispatch({
      type: "tool_output_delta",
      thread_id: "thread-1",
      turn_id: "turn-1",
      tool_id: "command-1",
      source_item_id: "command-1",
      protocol_method: "item/commandExecution/outputDelta",
      protocol_revision: "fixture-revision",
      delta: oversizedDelta,
    } as AgentEvent);

    expect(threadItems).toHaveLength(1);
    const streamedCommand = threadItems[0];
    expect(streamedCommand?.type).toBe("command_execution");
    if (streamedCommand?.type !== "command_execution") {
      throw new Error("expected command_execution after direct output delta");
    }
    expect(
      new TextEncoder().encode(streamedCommand.aggregated_output).byteLength,
    ).toBe(MAX_PROJECTION_OUTPUT_BYTES);
    expect(streamedCommand.aggregated_output).toContain(
      PROJECTION_OUTPUT_TRUNCATION_MARKER,
    );
    expect(streamedCommand.aggregated_output).toMatch(/-kept-tail$/u);
    expect(messages[0]?.toolCalls).toBeUndefined();
    expect(messages[0]?.contentParts).toBeUndefined();

    dispatch({
      type: "item_completed",
      protocol_method: "item/completed",
      protocol_revision: "fixture-revision",
      item: {
        ...commandItem,
        status: "completed",
        completed_at: "2026-07-29T01:00:01.000Z",
        updated_at: "2026-07-29T01:00:01.000Z",
        aggregated_output: "authoritative completed snapshot\n",
      },
    } as AgentEvent);

    expect(threadItems).toEqual([
      expect.objectContaining({
        id: "command-1",
        type: "command_execution",
        status: "completed",
        aggregated_output: "authoritative completed snapshot\n",
      }),
    ]);
    expect(messages[0]?.toolCalls).toBeUndefined();
    expect(messages[0]?.contentParts).toBeUndefined();
  });

  it("收到 turn_completed 时应把 usage 写回 assistant 消息", () => {
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "图片已经生成完成",
        timestamp: new Date("2026-04-07T10:00:00.000Z"),
        isThinking: true,
      },
    ];

    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );

    handleTurnStreamEvent({
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-test",
          thread_id: "thread-test",
          prompt_text: "test",
          status: "completed",
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
        usage: {
          input_tokens: 12_000,
          output_tokens: 19_000,
          cached_input_tokens: 8_000,
          cache_creation_input_tokens: 1_200,
        },
      } as AgentEvent,
      requestState: {
        accumulatedContent: "图片已经生成完成",
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "图片已经生成完成",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(setMessages).toHaveBeenCalled();
    expect(messages[0]).toMatchObject({
      isThinking: false,
      usage: {
        input_tokens: 12_000,
        output_tokens: 19_000,
        cached_input_tokens: 8_000,
        cache_creation_input_tokens: 1_200,
      },
    });
  });

  it("收到 turn_completed 时应保留已累积正文而不是用终态标记覆盖", () => {
    let messages: Message[] = [
      {
        id: "assistant-turn-completed",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-07T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent:
        "我先给出计划，不会直接改代码：\n<proposed_plan>\n- 确认计划模式请求进入 App Server\n- 输出 proposed_plan\n</proposed_plan>",
      currentTurnId: "turn-news",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setIsSending = vi.fn();
    const disposeListener = vi.fn();
    const removeQueuedTurnsFromProjection = vi.fn();
    const setThreadTurns = vi.fn(
      (value: unknown[] | ((prev: unknown[]) => unknown[])) => {
        if (typeof value === "function") {
          value([]);
        }
      },
    );
    const onComplete = vi.fn();

    handleTurnStreamEvent({
      data: {
        type: "turn_completed",
        text: "CLAW_NEWS_FIXTURE_DONE",
        usage: {
          input_tokens: 120,
          output_tokens: 24,
        },
        turn: {
          id: "turn-news",
          thread_id: "thread-news",
          prompt_text: "整理今天的国际新闻",
          status: "completed",
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => false,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener,
        clearActiveStreamIfMatch: () => false,
        removeQueuedTurnsFromProjection,
        appendThinkingToParts: (parts) => parts,
      },
      observer: {
        onComplete,
      },
      eventName: "agent-runtime-turn-completed",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-turn-completed",
      activeSessionId: "session-news",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "整理今天的国际新闻",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: setThreadTurns as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: setIsSending as never,
    });

    expect(onComplete).toHaveBeenCalledWith(
      "我先给出计划，不会直接改代码：\n<proposed_plan>\n- 确认计划模式请求进入 App Server\n- 输出 proposed_plan\n</proposed_plan>",
    );
    expect(messages[0]).toMatchObject({
      content:
        "我先给出计划，不会直接改代码：\n<proposed_plan>\n- 确认计划模式请求进入 App Server\n- 输出 proposed_plan\n</proposed_plan>",
      isThinking: false,
      usage: {
        input_tokens: 120,
        output_tokens: 24,
      },
    });
    expect(requestState.accumulatedContent).toBe(
      "我先给出计划，不会直接改代码：\n<proposed_plan>\n- 确认计划模式请求进入 App Server\n- 输出 proposed_plan\n</proposed_plan>",
    );
    expect(removeQueuedTurnsFromProjection).not.toHaveBeenCalled();
    expect(setIsSending).toHaveBeenCalledWith(false);
    expect(disposeListener).toHaveBeenCalledTimes(1);
  });

  it("收到 turn_failed 但已有有效正文时应按完成态收口且不显示失败条", () => {
    const partialAnswer = "已完成的部分回答：先确认时序，再输出结论。";
    const failureMessage = "provider stream closed before final answer";
    let messages: Message[] = [
      {
        id: "assistant-turn-failed",
        role: "assistant",
        content: partialAnswer,
        timestamp: new Date("2026-06-07T10:00:00.000Z"),
        isThinking: true,
        contentParts: [
          { type: "text", text: partialAnswer },
          {
            type: "tool_use",
            toolCall: {
              id: "tool-sequence-check",
              name: "sequence_check",
              arguments: "{}",
              status: "completed",
              result: { success: true, output: "ok" },
            },
          },
        ],
      },
    ];
    const requestState = {
      accumulatedContent: partialAnswer,
      currentTurnId: "turn-failed",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setIsSending = vi.fn();
    const disposeListener = vi.fn();
    const removeQueuedTurnsFromProjection = vi.fn();
    const onError = vi.fn();

    handleTurnStreamEvent({
      data: {
        type: "turn_failed",
        turn: {
          id: "turn-failed",
          thread_id: "thread-failed",
          prompt_text: "验证 terminal failed contract",
          status: "failed",
          error_message: failureMessage,
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener,
        clearActiveStreamIfMatch: () => true,
        removeQueuedTurnsFromProjection,
        appendThinkingToParts: (parts) => parts,
      },
      observer: {
        onError,
      },
      eventName: "agent-runtime-turn-failed",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-turn-failed",
      activeSessionId: "session-failed",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "验证 terminal failed contract",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: setIsSending as never,
    });

    expect(
      messages[0]?.content.match(new RegExp(partialAnswer, "g")) ?? [],
    ).toHaveLength(1);
    expect(messages[0]?.content).not.toContain("执行失败");
    expect(messages[0]?.content).not.toContain(failureMessage);
    expect(messages[0]?.contentParts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "tool_use" }),
        expect.objectContaining({
          type: "text",
          text: partialAnswer,
        }),
      ]),
    );
    expect(messages[0]?.isThinking).toBe(false);
    expect(messages[0]?.runtimeStatus).toBeUndefined();
    expect(removeQueuedTurnsFromProjection).not.toHaveBeenCalled();
    expect(setIsSending).toHaveBeenCalledWith(false);
    expect(disposeListener).toHaveBeenCalledTimes(1);
    expect(onError).not.toHaveBeenCalled();
    expect(mockToast.error).not.toHaveBeenCalled();
  });

  it("收到空 turn_completed 且没有真实产物信号时也应收起发送态并落失败态", () => {
    let messages: Message[] = [
      {
        id: "assistant-empty-turn-completed",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-07T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      currentTurnId: "turn-empty",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setIsSending = vi.fn();
    const disposeListener = vi.fn();
    const removeQueuedTurnsFromProjection = vi.fn();

    handleTurnStreamEvent({
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-empty",
          thread_id: "thread-empty",
          prompt_text: "整理今天的国际新闻",
          status: "completed",
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener,
        clearActiveStreamIfMatch: () => true,
        removeQueuedTurnsFromProjection,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-empty-turn-completed",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-empty-turn-completed",
      activeSessionId: "session-news",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "整理今天的国际新闻",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: setIsSending as never,
    });

    expect(messages[0]?.content).toContain("执行失败：");
    expect(messages[0]?.runtimeStatus).toMatchObject({
      phase: "failed",
      title: "当前处理失败",
      detail: "模型未输出最终答复，请重试",
    });
    expect(removeQueuedTurnsFromProjection).not.toHaveBeenCalled();
    expect(setIsSending).toHaveBeenCalledWith(false);
    expect(disposeListener).toHaveBeenCalledTimes(1);
    expect(mockToast.error).toHaveBeenCalledWith("模型未输出最终答复，请重试");
  });

  it("工具过程后没有 assistant 正文时不应把工具前开场白当最终答复", () => {
    let messages: Message[] = [
      {
        id: "assistant-search-no-final",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-24T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      hasFinalAnswerRequiredProcessBoundary: false,
      hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary: false,
      currentTurnId: "turn-search-no-final",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const removeQueuedTurnsFromProjection = vi.fn();
    const setIsSending = vi.fn();
    const disposeListener = vi.fn();
    const callbacks = {
      activateStream: () => {},
      isStreamActivated: () => true,
      clearOptimisticItem: () => {},
      clearOptimisticTurn: () => {},
      disposeListener,
      clearActiveStreamIfMatch: () => true,
      removeQueuedTurnsFromProjection,
      appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
        parts,
    };
    const baseOptions = {
      requestState,
      callbacks,
      eventName: "agent-runtime-search-no-final",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-search-no-final",
      activeSessionId: "session-news",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "整理今天的国际新闻",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: setIsSending as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: "我会先联网核实今天国际新闻的主要议题，再整理摘要。",
        itemId: "commentary-news-plan",
        phase: "commentary",
        sequence: 1,
        turn_id: "turn-search-no-final",
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        tool_id: "tool-web-search-1",
        tool_name: "WebSearch",
        arguments: JSON.stringify({ query: "international news today" }),
        sequence: 2,
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_end",
        tool_id: "tool-web-search-1",
        sequence: 3,
        result: {
          success: true,
          output: "2 results",
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        sequence: 4,
        turn: {
          id: "turn-search-no-final",
          thread_id: "thread-news",
          prompt_text: "整理今天的国际新闻",
          status: "completed",
          started_at: "2026-06-24T10:00:00.000Z",
          completed_at: "2026-06-24T10:00:01.000Z",
          created_at: "2026-06-24T10:00:00.000Z",
          updated_at: "2026-06-24T10:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(requestState.hasFinalAnswerRequiredProcessBoundary).toBe(true);
    expect(
      requestState.hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary,
    ).toBe(false);
    expect(messages[0]?.content).toContain("执行失败：");
    expect(messages[0]?.runtimeStatus).toMatchObject({
      phase: "failed",
      title: "当前处理失败",
      detail: "模型未输出最终答复，请重试",
    });
    expect(removeQueuedTurnsFromProjection).not.toHaveBeenCalled();
    expect(setIsSending).toHaveBeenCalledWith(false);
    expect(disposeListener).toHaveBeenCalledTimes(1);
    expect(mockToast.error).toHaveBeenCalledWith("模型未输出最终答复，请重试");
  });

  it("commentary text_delta 只进入 agent_message timeline，不再写回 Message", () => {
    let messages: Message[] = [
      {
        id: "assistant-commentary",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-24T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const requestState = {
      accumulatedContent: "",
      hasFinalAnswerRequiredProcessBoundary: false,
      hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary: false,
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );

    handleTurnStreamEvent({
      data: {
        type: "text_delta",
        text: "我会先搜索公开资料。",
        itemId: "item-commentary-1",
        phase: "commentary",
        sequence: 1,
        session_id: "session-commentary",
        thread_id: "thread-commentary",
        turn_id: "turn-commentary",
        timestamp: "2026-06-24T10:00:00.000Z",
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-commentary",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-commentary",
      activeSessionId: "session-commentary",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "搜索资料",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(requestState.accumulatedContent).toBe("");
    expect(messages[0]?.content).toBe("");
    expect(getAgentStreamTextOverlay("assistant-commentary")).toBeNull();
    expect(messages[0]?.contentParts).toEqual([]);
    expect(threadItems).toEqual([
      expect.objectContaining({
        id: "item-commentary-1",
        type: "agent_message",
        phase: "commentary",
        text: "我会先搜索公开资料。",
        turn_id: "turn-commentary",
      }),
    ]);
  });

  it("commentary delta 早于 Message 挂载时也不应重放进 contentParts", () => {
    let messages: Message[] = [];
    let threadItems: AgentThreadItem[] = [];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-commentary-race",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-commentary-race",
      activeSessionId: "session-commentary-race",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "搜索资料",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      getThreadItems: () => threadItems,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: "我先联网核实目标页面来源。\n",
        item_id: "item-commentary-race",
        itemId: "item-commentary-race",
        phase: "commentary",
        sequence: 1,
        session_id: "session-commentary-race",
        thread_id: "thread-commentary-race",
        turn_id: "turn-commentary-race",
        timestamp: "2026-06-24T10:00:00.000Z",
      } as AgentEvent,
    });

    messages = [
      {
        id: "assistant-commentary-race",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-24T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        tool_id: "tool-web-search-race",
        tool_name: "WebSearch",
        arguments: JSON.stringify({ query: "Lime WebSearch rendering" }),
        sequence: 2,
      } as AgentEvent,
    });

    expect(messages[0]?.contentParts?.map((part) => part.type)).toEqual([
      "tool_use",
    ]);
    expect(threadItems).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "item-commentary-race",
          type: "agent_message",
          text: "我先联网核实目标页面来源。\n",
        }),
      ]),
    );
  });

  it("工具过程后有 assistant 正文时应由结构化顺序正常完成", () => {
    let messages: Message[] = [
      {
        id: "assistant-search-final",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-24T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const requestState = {
      accumulatedContent: "",
      hasFinalAnswerRequiredProcessBoundary: false,
      hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary: false,
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const removeQueuedTurnsFromProjection = vi.fn();
    const onComplete = vi.fn();
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );
    const callbacks = {
      activateStream: () => {},
      isStreamActivated: () => true,
      clearOptimisticItem: () => {},
      clearOptimisticTurn: () => {},
      disposeListener: () => {},
      clearActiveStreamIfMatch: () => true,
      removeQueuedTurnsFromProjection,
      appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
        parts,
    };
    const baseOptions = {
      requestState,
      callbacks,
      observer: {
        onComplete,
      },
      eventName: "agent-runtime-search-final",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-search-final",
      activeSessionId: "session-news",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "整理今天的国际新闻",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: "我会先联网核实今天国际新闻的主要议题，再整理摘要。",
        sequence: 1,
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        tool_id: "tool-web-search-2",
        tool_name: "WebSearch",
        arguments: JSON.stringify({ query: "international news today" }),
        sequence: 2,
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_end",
        tool_id: "tool-web-search-2",
        sequence: 3,
        result: {
          success: true,
          output: "2 results",
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: "最终摘要：国际议题集中在安全、能源和供应链。",
        itemId: "final-news-summary",
        phase: "final_answer",
        sequence: 4,
        turn_id: "turn-search-final",
      } as AgentEvent,
    });
    expect(getAgentStreamTextOverlay("assistant-search-final")?.content).toBe(
      "最终摘要：国际议题集中在安全、能源和供应链。",
    );
    expect(
      messages[0]?.contentParts?.some(
        (part) =>
          part.type === "text" &&
          part.text === "最终摘要：国际议题集中在安全、能源和供应链。",
      ),
    ).toBe(false);
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        text: "最终摘要：国际议题集中在安全、能源和供应链。",
        sequence: 5,
        turn: {
          id: "turn-search-final",
          thread_id: "thread-news",
          prompt_text: "整理今天的国际新闻",
          status: "completed",
          started_at: "2026-06-24T10:00:00.000Z",
          completed_at: "2026-06-24T10:00:01.000Z",
          created_at: "2026-06-24T10:00:00.000Z",
          updated_at: "2026-06-24T10:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(
      requestState.hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary,
    ).toBe(true);
    expect(messages[0]?.content).toContain(
      "最终摘要：国际议题集中在安全、能源和供应链。",
    );
    expect(messages[0]?.contentParts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "text",
          text: "最终摘要：国际议题集中在安全、能源和供应链。",
          metadata: expect.objectContaining({
            itemId: "final-news-summary",
            phase: "final_answer",
            sequence: 4,
            source: "agent_text_delta",
            turnId: "turn-search-final",
          }),
        }),
      ]),
    );
    expect(messages[0]?.content).not.toContain(
      "我会先联网核实今天国际新闻的主要议题",
    );
    expect(messages[0]?.isThinking).toBe(false);
    expect(messages[0]?.runtimeStatus).toBeUndefined();
    expect(onComplete).toHaveBeenCalledWith(
      "最终摘要：国际议题集中在安全、能源和供应链。",
    );
    expect(threadItems).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "tool-web-search-2",
          type: "tool_call",
          status: "completed",
        }),
      ]),
    );
    expect(removeQueuedTurnsFromProjection).not.toHaveBeenCalled();
    expect(mockToast.error).not.toHaveBeenCalledWith(
      "模型未输出最终答复，请重试",
    );
  });

  it("过程事件后晚到的无 phase text_delta 应作为最终正文完成", () => {
    let messages: Message[] = [
      {
        id: "assistant-late-legacy-final",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-24T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      hasFinalAnswerRequiredProcessBoundary: false,
      hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary: false,
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const removeQueuedTurnsFromProjection = vi.fn();
    const onComplete = vi.fn();
    const callbacks = {
      activateStream: () => {},
      isStreamActivated: () => true,
      clearOptimisticItem: () => {},
      clearOptimisticTurn: () => {},
      disposeListener: () => {},
      clearActiveStreamIfMatch: () => true,
      removeQueuedTurnsFromProjection,
      appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
        parts,
    };
    const baseOptions = {
      requestState,
      callbacks,
      observer: {
        onComplete,
      },
      eventName: "agent-runtime-late-legacy-final",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-late-legacy-final",
      activeSessionId: "session-late-legacy-final",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "你好",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "reasoning_delta",
        text: "先理解用户问候。",
        sequence: 1,
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: "你好！有什么我可以帮你的吗？",
        sequence: 2,
        turn_id: "turn-late-legacy-final",
      } as AgentEvent,
    });

    expect(
      requestState.hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary,
    ).toBe(true);
    expect(
      getAgentStreamTextOverlay("assistant-late-legacy-final")?.content,
    ).toBe("你好！有什么我可以帮你的吗？");

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        sequence: 3,
        turn: {
          id: "turn-late-legacy-final",
          thread_id: "thread-late-legacy-final",
          prompt_text: "你好",
          status: "completed",
          started_at: "2026-06-24T10:00:00.000Z",
          completed_at: "2026-06-24T10:00:01.000Z",
          created_at: "2026-06-24T10:00:00.000Z",
          updated_at: "2026-06-24T10:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(messages[0]?.content).toBe("你好！有什么我可以帮你的吗？");
    expect(messages[0]?.contentParts).toEqual([
      expect.objectContaining({
        type: "text",
        text: "你好！有什么我可以帮你的吗？",
        metadata: expect.objectContaining({
          source: "agent_text_delta",
          sequence: 2,
          turnId: "turn-late-legacy-final",
        }),
      }),
    ]);
    expect(messages[0]?.runtimeStatus).toBeUndefined();
    expect(messages[0]?.isThinking).toBe(false);
    expect(onComplete).toHaveBeenCalledWith("你好！有什么我可以帮你的吗？");
    expect(removeQueuedTurnsFromProjection).not.toHaveBeenCalled();
    expect(mockToast.error).not.toHaveBeenCalledWith(
      "模型未输出最终答复，请重试",
    );
  });

  it("最终正文后收到 reasoning.final 归档时不应误判缺少最终答复", () => {
    let messages: Message[] = [
      {
        id: "assistant-reasoning-final-archive",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-12T07:07:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      hasFinalAnswerRequiredProcessBoundary: false,
      hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary: false,
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const onComplete = vi.fn();
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      observer: { onComplete },
      eventName: "agent-runtime-reasoning-final-archive",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-reasoning-final-archive",
      activeSessionId: "session-reasoning-final-archive",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "你好",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "reasoning_delta",
        text: "先理解用户问候。",
        sequence: 1,
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: "你好！有什么我可以帮你的吗？",
        sequence: 2,
        turn_id: "turn-reasoning-final-archive",
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "reasoning_final",
        text: "已完成问候意图分析。",
        sequence: 3,
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "reasoning_ended",
        status: "completed",
        sequence: 4,
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        sequence: 5,
        turn: {
          id: "turn-reasoning-final-archive",
          thread_id: "thread-reasoning-final-archive",
          prompt_text: "你好",
          status: "completed",
          started_at: "2026-07-12T07:07:00.000Z",
          completed_at: "2026-07-12T07:07:05.000Z",
          created_at: "2026-07-12T07:07:00.000Z",
          updated_at: "2026-07-12T07:07:05.000Z",
        },
      } as AgentEvent,
    });

    expect(messages[0]?.content).toBe("你好！有什么我可以帮你的吗？");
    expect(messages[0]?.runtimeStatus).toBeUndefined();
    expect(messages[0]?.isThinking).toBe(false);
    expect(onComplete).toHaveBeenCalledWith("你好！有什么我可以帮你的吗？");
    expect(mockToast.error).not.toHaveBeenCalledWith(
      "模型未输出最终答复，请重试",
    );
  });

  it("收到空 turn_completed 但已有真实产物信号时应软完成而不是等待 turn_completed", () => {
    let messages: Message[] = [
      {
        id: "assistant-artifact-turn-completed",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-07T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      hasMeaningfulCompletionSignal: true,
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setIsSending = vi.fn();
    const disposeListener = vi.fn();
    const onComplete = vi.fn();

    handleTurnStreamEvent({
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-artifact",
          thread_id: "thread-artifact",
          prompt_text: "生成代码产物",
          status: "completed",
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => false,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener,
        clearActiveStreamIfMatch: () => false,
        appendThinkingToParts: (parts) => parts,
      },
      observer: {
        onComplete,
      },
      eventName: "agent-runtime-artifact-turn-completed",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-artifact-turn-completed",
      activeSessionId: "session-artifact",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "生成代码产物",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: setIsSending as never,
    });

    expect(onComplete).toHaveBeenCalledWith(
      "本轮执行已完成，详细过程与产物已保留在当前对话中。",
    );
    expect(messages[0]).toMatchObject({
      content: "本轮执行已完成，详细过程与产物已保留在当前对话中。",
      isThinking: false,
    });
    expect(setIsSending).toHaveBeenCalledWith(false);
    expect(disposeListener).toHaveBeenCalledTimes(1);
    expect(mockToast.error).not.toHaveBeenCalled();
  });

  it("首个事件就是 turn_completed 时也应收起发送态", () => {
    let messages: Message[] = [
      {
        id: "assistant-final-first",
        role: "assistant",
        content: "整理完成",
        timestamp: new Date("2026-06-07T10:00:00.000Z"),
        isThinking: true,
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setIsSending = vi.fn();
    const disposeListener = vi.fn();

    handleTurnStreamEvent({
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-test",
          thread_id: "thread-test",
          prompt_text: "test",
          status: "completed",
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
      } as AgentEvent,
      requestState: {
        accumulatedContent: "整理完成",
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => false,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener,
        clearActiveStreamIfMatch: () => false,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-final-first",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-final-first",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "整理今天的国际新闻",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: setIsSending as never,
    });

    expect(messages[0]).toMatchObject({
      content: "整理完成",
      isThinking: false,
      runtimeStatus: undefined,
    });
    expect(setIsSending).toHaveBeenCalledWith(false);
    expect(disposeListener).toHaveBeenCalledTimes(1);
  });

  it("收到 turn_canceled 时应收起发送态并追加已停止终态", () => {
    let messages: Message[] = [
      {
        id: "assistant-canceled",
        role: "assistant",
        content: "已经输出的内容",
        timestamp: new Date("2026-06-07T10:00:00.000Z"),
        isThinking: true,
      },
    ];
    let threadItems: AgentThreadItem[] = [
      {
        id: "agent-message-canceled",
        thread_id: "thread-news",
        turn_id: "turn-canceled",
        sequence: 1,
        status: "in_progress",
        started_at: "2026-06-07T10:00:00.000Z",
        updated_at: "2026-06-07T10:00:00.500Z",
        type: "agent_message",
        text: "已经输出的内容",
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );
    const setIsSending = vi.fn();
    const disposeListener = vi.fn();

    handleTurnStreamEvent({
      data: {
        type: "turn_canceled",
        turn: {
          id: "turn-canceled",
          thread_id: "thread-news",
          prompt_text: "停止",
          status: "canceled",
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
      } as AgentEvent,
      requestState: {
        accumulatedContent: "已经输出的内容",
        currentTurnId: "turn-canceled",
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => false,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener,
        clearActiveStreamIfMatch: () => false,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-turn-canceled",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-canceled",
      activeSessionId: "session-news",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "停止",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending,
    });

    expect(messages[0]).toMatchObject({
      content: "已经输出的内容\n\n(已停止)",
      isThinking: false,
    });
    expect(threadItems[0]).toMatchObject({
      type: "agent_message",
      text: "已经输出的内容\n\n(已停止)",
      contentParts: [
        {
          type: "text",
          text: "(已停止)",
        },
      ],
    });
    expect(setIsSending).toHaveBeenCalledWith(false);
    expect(disposeListener).toHaveBeenCalledTimes(1);
  });

  it("陈旧 stream 的 turn_completed 不应误停新的发送态", () => {
    let messages: Message[] = [
      {
        id: "assistant-stale",
        role: "assistant",
        content: "旧请求完成",
        timestamp: new Date("2026-06-07T10:00:00.000Z"),
        isThinking: true,
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setIsSending = vi.fn();
    const disposeListener = vi.fn();

    handleTurnStreamEvent({
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-test",
          thread_id: "thread-test",
          prompt_text: "test",
          status: "completed",
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
      } as AgentEvent,
      requestState: {
        accumulatedContent: "旧请求完成",
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener,
        clearActiveStreamIfMatch: () => false,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-stale-final",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-stale",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "旧请求",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: setIsSending as never,
    });

    expect(messages[0]).toMatchObject({
      content: "旧请求完成",
      isThinking: false,
      runtimeStatus: undefined,
    });
    expect(setIsSending).not.toHaveBeenCalled();
    expect(disposeListener).toHaveBeenCalledTimes(1);
  });

  it("同一 stream 名下旧 turn_completed 晚到时不应清理新 turn", () => {
    let messages: Message[] = [
      {
        id: "assistant-current-turn",
        role: "assistant",
        content: "新请求正在输出",
        timestamp: new Date("2026-06-07T10:00:00.000Z"),
        isThinking: true,
      },
    ];
    const requestState = {
      accumulatedContent: "新请求正在输出",
      activeTextSegmentTurnId: "turn-current",
      currentTurnId: "turn-current",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const clearActiveStreamIfMatch = vi.fn(() => true);
    const clearOptimisticItem = vi.fn();
    const clearOptimisticTurn = vi.fn();
    const disposeListener = vi.fn();
    const setCurrentTurnId = vi.fn();
    const setIsSending = vi.fn();
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn();
    const setThreadTurns = vi.fn();

    handleTurnStreamEvent({
      data: {
        type: "turn_completed",
        text: "旧请求完成",
        turn: {
          id: "turn-old",
          thread_id: "thread-test",
          prompt_text: "old",
          status: "completed",
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem,
        clearOptimisticTurn,
        disposeListener,
        clearActiveStreamIfMatch,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-reused-stream",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-current-turn",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "新请求",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: setThreadTurns as never,
      setCurrentTurnId: setCurrentTurnId as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: setIsSending as never,
    });

    expect(clearActiveStreamIfMatch).not.toHaveBeenCalled();
    expect(clearOptimisticItem).not.toHaveBeenCalled();
    expect(clearOptimisticTurn).not.toHaveBeenCalled();
    expect(disposeListener).not.toHaveBeenCalled();
    expect(setCurrentTurnId).not.toHaveBeenCalled();
    expect(setIsSending).not.toHaveBeenCalled();
    expect(setMessages).not.toHaveBeenCalled();
    expect(setThreadItems).not.toHaveBeenCalled();
    expect(setThreadTurns).not.toHaveBeenCalled();
    expect(messages[0]).toMatchObject({
      content: "新请求正在输出",
      isThinking: true,
    });
    expect(requestState.currentTurnId).toBe("turn-current");
    expect(requestState.accumulatedContent).toBe("新请求正在输出");
  });

  it("收到完整 message 快照事件时应立即预填首屏文本", () => {
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-05-07T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const activateStream = vi.fn();

    handleTurnStreamEvent({
      data: {
        type: "message",
        message: {
          id: "msg-runtime-1",
          role: "assistant",
          content: [
            {
              type: "text",
              text: "完整快照会由后续 text_delta 渲染。",
            },
          ],
          timestamp: 1777284240,
        },
      } as AgentEvent,
      requestState: {
        accumulatedContent: "",
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      callbacks: {
        activateStream,
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-message-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "生成验收矩阵",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(activateStream).toHaveBeenCalledTimes(1);
    expect(setMessages).toHaveBeenCalledTimes(1);
    expect(messages[0]?.content).toBe("完整快照会由后续 text_delta 渲染。");
    expect(messages[0]?.contentParts).toEqual([
      {
        type: "text",
        text: "完整快照会由后续 text_delta 渲染。",
      },
    ]);
  });

  it("message 快照已预填正文时后续 text_delta 重放不应重复追加", () => {
    vi.useFakeTimers();
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-05-07T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const onTextDelta = vi.fn();
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      observer: {
        onTextDelta,
      },
      eventName: "agent-runtime-message-replay-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "生成验收矩阵",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "message",
        message: {
          id: "msg-runtime-1",
          role: "assistant",
          content: [{ type: "text", text: "先显示快照。" }],
          timestamp: 1777284240,
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: { type: "text_delta", text: "先显示" } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: { type: "text_delta", text: "快照。" } as AgentEvent,
    });

    expect(messages[0]?.content).toBe("先显示快照。");
    expect(messages[0]?.contentParts).toEqual([
      { type: "text", text: "先显示快照。" },
    ]);
    expect(onTextDelta).not.toHaveBeenCalled();

    handleTurnStreamEvent({
      ...baseOptions,
      data: { type: "text_delta", text: "继续输出。" } as AgentEvent,
    });

    vi.advanceTimersByTime(AGENT_STREAM_TEXT_DELTA_RENDER_FLUSH_MS);

    expect(messages[0]?.content).toBe("先显示快照。");
    expect(messages[0]?.contentParts).toEqual([
      { type: "text", text: "先显示快照。" },
    ]);
    expect(getAgentStreamTextOverlay("assistant-1")?.content).toBe(
      "先显示快照。继续输出。",
    );
    expect(onTextDelta).toHaveBeenCalledTimes(1);
    expect(onTextDelta).toHaveBeenCalledWith(
      "继续输出。",
      "先显示快照。继续输出。",
    );
  });

  it("tool_start 前的无 provenance 文本应按过程边界提交为普通 text part", () => {
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-05-29T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-interleave-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "分析一下项目",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: { type: "text_delta", text: "先分析。" } as AgentEvent,
    });
    expect(messages[0]?.contentParts).toEqual([]);

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        tool_id: "tool-1",
        tool_name: "Bash",
        arguments: JSON.stringify({ command: "pwd" }),
      } as AgentEvent,
    });

    expect(messages[0]?.content).toBe("先分析。");
    expect(getAgentStreamTextOverlay("assistant-1")).toBeNull();
    expect(messages[0]?.contentParts).toEqual([
      {
        type: "text",
        text: "先分析。",
      },
      expect.objectContaining({
        type: "tool_use",
        toolCall: expect.objectContaining({
          id: "tool-1",
          name: "Bash",
          status: "running",
        }),
      }),
    ]);
    expect(threadItems).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "agent_message",
          text: "先分析。",
        }),
      ]),
    );
    expect(requestState.accumulatedContent).toBe("先分析。");
  });

  it("process 后晚到的无 phase 文本应 live 显示，并允许 turn_completed.text 修正最终正文", () => {
    let messages: Message[] = [
      {
        id: "assistant-live-search",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-24T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-live-search-unphased",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-live-search",
      activeSessionId: "session-live-search",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "整理今天的国际新闻",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        tool_id: "tool-live-search",
        tool_name: "WebSearch",
        arguments: JSON.stringify({ query: "international news" }),
        sequence: 2,
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: "我",
        sequence: 3,
        turn_id: "turn-live-search",
      } as AgentEvent,
    });

    expect(requestState.accumulatedContent).toBe("我");
    expect(getAgentStreamTextOverlay("assistant-live-search")?.content).toBe(
      "我",
    );
    expect(messages[0]?.content).toBe("");
    expect(messages[0]?.contentParts?.map((part) => part.type)).toEqual([
      "tool_use",
    ]);
    expect(JSON.stringify(messages[0]?.contentParts)).not.toContain('"我"');
    expect(threadItems).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "agent_message",
          text: "我",
        }),
      ]),
    );

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        text: "最终摘要：已按来源整理国际新闻。",
        sequence: 9,
        turn: {
          id: "turn-live-search",
          thread_id: "session-live-search",
          prompt_text: "整理今天的国际新闻",
          status: "completed",
          started_at: "2026-06-24T10:00:00.000Z",
          completed_at: "2026-06-24T10:00:03.000Z",
          created_at: "2026-06-24T10:00:00.000Z",
          updated_at: "2026-06-24T10:00:03.000Z",
        },
      } as AgentEvent,
    });

    expect(messages[0]?.content).toBe("最终摘要：已按来源整理国际新闻。");
    expect(messages[0]?.content).not.toContain("我最终摘要");
    expect(messages[0]?.contentParts?.map((part) => part.type)).toEqual([
      "tool_use",
      "text",
    ]);
  });

  it("工具先到且文本乱序到达时不应把 process 前后文本合并成一句", () => {
    const firstText = "我会先联网核实今天的主要国际新闻。";
    const secondText = "我再补一个交叉对照，避免依赖单一来源。";
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-24T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-late-text-sequence-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "整理今天的国际新闻",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        tool_id: "tool-search",
        tool_name: "web_search",
        arguments: JSON.stringify({ query: "international news" }),
        sequence: 2,
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_end",
        tool_id: "tool-search",
        sequence: 5,
        result: {
          success: true,
          output: "搜索完成",
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: firstText,
        sequence: 1,
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: secondText,
        itemId: "final-news-cross-check",
        phase: "final_answer",
        sequence: 7,
        turn_id: "turn-news-order",
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-news-order",
          thread_id: "session-1",
          prompt_text: "整理今天的国际新闻",
          status: "completed",
          started_at: "2026-06-24T10:00:00.000Z",
          completed_at: "2026-06-24T10:00:03.000Z",
          created_at: "2026-06-24T10:00:00.000Z",
          updated_at: "2026-06-24T10:00:03.000Z",
        },
      } as AgentEvent,
    });

    const textParts = messages[0]?.contentParts?.filter(
      (
        part,
      ): part is Extract<
        NonNullable<Message["contentParts"]>[number],
        { type: "text" }
      > => part.type === "text",
    );
    expect(textParts?.map((part) => part.text)).toEqual([secondText]);
    expect(textParts?.[0]?.metadata).toMatchObject({
      source: "agent_text_delta",
      sequence: 7,
    });
    expect(threadItems).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "agent_message",
          text: firstText,
        }),
      ]),
    );
    expect(messages[0]?.contentParts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "tool_use",
          metadata: expect.objectContaining({ sequence: 2 }),
          toolCall: expect.objectContaining({
            id: "tool-search",
            status: "completed",
          }),
        }),
      ]),
    );
  });

  it("thinking 关闭时不应把 reasoning_delta 渲染进助手正文", () => {
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-04-30T10:00:00.000Z"),
        isThinking: true,
        contentParts: [{ type: "thinking", text: "隐藏推理" }],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const activateStream = vi.fn();
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream,
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: () => {
          throw new Error("thinking 关闭时不应追加 thinking part");
        },
      },
      eventName: "agent-runtime-thinking-disabled-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      surfaceThinkingDeltas: false,
      content: "只回复一个字：好",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "thinking_delta",
        text: "我们只：好。",
      } as AgentEvent,
    });

    expect(activateStream).toHaveBeenCalledTimes(1);
    expect(setMessages).toHaveBeenCalledTimes(1);
    expect(messages[0]?.contentParts).toEqual([
      { type: "thinking", text: "隐藏推理" },
    ]);

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: "好",
        phase: "final_answer",
      } as AgentEvent,
    });

    expect(messages[0]?.content).toBe("");
    expect(messages[0]?.thinkingContent).toBeUndefined();
    expect(messages[0]?.contentParts).toEqual([]);
    expect(getAgentStreamTextOverlay("assistant-1")?.content).toBe("好");
  });

  it("连续 text_delta 应合并到低频渲染，避免每个字符都刷新消息树", () => {
    vi.useFakeTimers();
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-04-30T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-text-batch-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "数数",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: { type: "text_delta", text: "1" } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: { type: "text_delta", text: "2" } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: { type: "text_delta", text: "3" } as AgentEvent,
    });

    expect(setMessages).not.toHaveBeenCalled();
    expect(messages[0]?.content).toBe("");
    expect(getAgentStreamTextOverlay("assistant-1")?.content).toBe("1");

    vi.advanceTimersByTime(AGENT_STREAM_TEXT_DELTA_RENDER_FLUSH_MS);

    expect(setMessages).not.toHaveBeenCalled();
    expect(messages[0]?.content).toBe("");
    expect(messages[0]?.contentParts).toEqual([]);
    expect(getAgentStreamTextOverlay("assistant-1")?.content).toBe("123");
  });

  it("图片生成轻卡应接纳模型自然 text_delta，并保留同一条消息里的预览", () => {
    let messages: Message[] = [
      {
        id: "assistant-image",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-05-12T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
        imageWorkbenchPreview: {
          taskId: "draft-image-1",
          prompt: "一张广州塔春天照片",
          mode: "generate",
          status: "running",
          modelName: "fal-ai/nano-banana-pro",
        },
      },
    ];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-image-draft-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      assistantFallbackContent: "",
      content: "@Nanobanana Pro 生成广州塔春天照片",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        tool_name: "Skill",
        tool_id: "tool-image-generate",
        arguments: JSON.stringify({ skill: "image_generate" }),
      } as AgentEvent,
    });

    expect(requestState.accumulatedContent).toBe("");
    expect(getAgentStreamTextOverlay("assistant-image")).toBeNull();

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: "我来为你生成这张照片。",
        phase: "final_answer",
      } as AgentEvent,
    });

    expect(requestState.accumulatedContent).toBe("我来为你生成这张照片。");
    expect(getAgentStreamTextOverlay("assistant-image")?.content).toBe(
      "我来为你生成这张照片。",
    );

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-test",
          thread_id: "thread-test",
          prompt_text: "test",
          status: "completed",
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(messages[0]?.content).toBe("我来为你生成这张照片。");
    expect(messages[0]?.content).not.toContain("先获取下工具参数");
    expect(messages[0]?.imageWorkbenchPreview?.taskId).toBe("draft-image-1");
    expect(messages[0]?.isThinking).toBe(false);
  });

  it("图片生成 final text_delta 应直接采用后端模型文案", () => {
    let messages: Message[] = [
      {
        id: "assistant-image",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-05-12T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
        imageWorkbenchPreview: {
          taskId: "draft-image-1",
          prompt:
            "用 Agnes 生成一张深圳夏天午后的城市照片，阳光明亮，街边绿树和高楼，真实摄影风格",
          mode: "generate",
          status: "running",
          modelName: "agnes-image-2.1-flash",
        },
      },
    ];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-image-text-sanitize-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      assistantFallbackContent: "",
      content:
        "@配图 用 Agnes 生成一张深圳夏天午后的城市照片，阳光明亮，街边绿树和高楼，真实摄影风格",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: "好啊，先来Generate深圳夏day午后的城市照片，阳光明亮，真实摄影Style。",
        phase: "final_answer",
      } as AgentEvent,
    });

    expect(requestState.accumulatedContent).toContain(
      "好啊，先来Generate深圳夏day午后的城市照片",
    );
    expect(requestState.accumulatedContent).toContain("真实摄影Style");
    expect(getAgentStreamTextOverlay("assistant-image")?.content).toBe(
      requestState.accumulatedContent,
    );

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-test",
          thread_id: "thread-test",
          prompt_text: "test",
          status: "completed",
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(messages[0]?.content).toContain(
      "好啊，先来Generate深圳夏day午后的城市照片",
    );
    expect(messages[0]?.content).toContain("真实摄影Style");
    expect(messages[0]?.imageWorkbenchPreview?.taskId).toBe("draft-image-1");
  });

  it("图片生成非 final text_delta 应保留在 canonical AgentMessage", () => {
    let messages: Message[] = [
      {
        id: "assistant-image-structured",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-05-12T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
        imageWorkbenchPreview: {
          taskId: "draft-image-structured",
          prompt:
            "用 Agnes 生成一张深圳夏天午后的城市照片，阳光明亮，街边绿树和高楼，真实摄影风格",
          mode: "generate",
          status: "running",
          modelName: "agnes-image-2.1-flash",
        },
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
      hasFinalAnswerRequiredProcessBoundary: true,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );

    handleTurnStreamEvent({
      data: {
        type: "text_delta",
        text: "好啊，先来Generate深圳夏day午后的城市照片，阳光明亮，真实摄影Style。",
        itemId: "item-image-structured",
        turn_id: "turn-image-structured",
        phase: "commentary",
        sequence: 3,
      } as AgentEvent,
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-image-structured-sanitize-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-image-structured",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      assistantFallbackContent: "",
      content:
        "@配图 用 Agnes 生成一张深圳夏天午后的城市照片，阳光明亮，街边绿树和高楼，真实摄影风格",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      getThreadItems: () => threadItems,
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(messages[0]?.content).toBe("");
    expect(messages[0]?.contentParts).toEqual([]);
    const firstThreadItem = threadItems[0];
    expect(firstThreadItem?.type).toBe("agent_message");
    const itemText =
      firstThreadItem?.type === "agent_message" ? firstThreadItem.text : "";
    expect(itemText).toContain("好啊，先来Generate深圳夏day午后的城市照片");
    expect(itemText).toContain("真实摄影Style");
  });

  it("text_delta_batch 应先写入 overlay，并在 turn_completed 时一次性 reconcile 回消息", () => {
    vi.useFakeTimers();
    let messages: Message[] = [
      {
        id: "assistant-batch",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-04-30T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-runtime-text-batch-protocol-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-batch",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "批量输出",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta_batch",
        text: "批量输出\n",
        chunks: ["批量", "输出", "\n"],
        boundary: "newline",
      } as AgentEvent,
    });

    expect(setMessages).not.toHaveBeenCalled();
    expect(messages[0]?.content).toBe("");
    expect(getAgentStreamTextOverlay("assistant-batch")?.content).toBe(
      "批量输出\n",
    );

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-test",
          thread_id: "thread-test",
          prompt_text: "test",
          status: "completed",
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(messages[0]).toMatchObject({
      content: "批量输出",
      contentParts: [{ type: "text", text: "批量输出" }],
      isThinking: false,
    });
    expect(getAgentStreamTextOverlay("assistant-batch")).toBeNull();
  });

  it("text flush 后仍应保留并继续累积 thinkingContent", () => {
    vi.useFakeTimers();
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-05-06T10:00:00.000Z"),
        isThinking: true,
        thinkingContent: "",
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (
          parts: NonNullable<Message["contentParts"]>,
          textDelta: string,
        ) => [...parts, { type: "thinking" as const, text: textDelta }],
      },
      eventName: "agent-runtime-thinking-retain-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      surfaceThinkingDeltas: true,
      content: "继续写正文",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: { type: "thinking_delta", text: "先想第一段。" } as AgentEvent,
    });

    expect(messages[0]?.thinkingContent).toBe("先想第一段。");

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: "正文一",
        phase: "final_answer",
      } as AgentEvent,
    });

    expect(messages[0]?.thinkingContent).toBe("先想第一段。");

    vi.advanceTimersByTime(AGENT_STREAM_TEXT_DELTA_RENDER_FLUSH_MS);

    expect(messages[0]?.content).toBe("");
    expect(messages[0]?.thinkingContent).toBe("先想第一段。");
    expect(getAgentStreamTextOverlay("assistant-1")?.content).toBe("正文一");

    handleTurnStreamEvent({
      ...baseOptions,
      data: { type: "thinking_delta", text: "再想第二段。" } as AgentEvent,
    });

    expect(messages[0]?.thinkingContent).toBe("先想第一段。再想第二段。");
  });

  it("reasoning item_updated 应持续刷新时间线思考内容", () => {
    const setThreadItems = vi.fn();
    const activateStream = vi.fn();

    handleTurnStreamEvent({
      data: {
        type: "item_updated",
        protocol_method: "item/started",
        protocol_revision: "fixture-revision",
        item: {
          id: "reasoning-1",
          thread_id: "session-1",
          turn_id: "turn-1",
          sequence: 2,
          type: "reasoning",
          text: "正在持续追加推理文本",
          status: "in_progress",
          started_at: "2026-04-27T10:00:00.000Z",
          updated_at: "2026-04-27T10:00:01.000Z",
        },
      } as AgentEvent,
      requestState: {
        accumulatedContent: "",
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      callbacks: {
        activateStream,
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-reasoning-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "生成验收矩阵",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: vi.fn() as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(activateStream).toHaveBeenCalledTimes(1);
    expect(setThreadItems).toHaveBeenCalledTimes(1);

    const updater = setThreadItems.mock.calls[0]?.[0];
    expect(typeof updater).toBe("function");
    const nextItems = typeof updater === "function" ? updater([]) : updater;
    expect(nextItems).toEqual([
      expect.objectContaining({
        id: "reasoning-1",
        type: "reasoning",
        text: "正在持续追加推理文本",
        status: "in_progress",
      }),
    ]);
  });

  it("thinking_delta 应同步生成当前 turn 的临时 reasoning 时间线项", () => {
    let threadItems: AgentThreadItem[] = [];
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );

    const requestState = {
      accumulatedContent: "",
      currentTurnId: "turn-1",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (
          parts: NonNullable<Message["contentParts"]>,
          textDelta: string,
        ) => [...parts, { type: "thinking" as const, text: textDelta }],
      },
      eventName: "agent-runtime-thinking-timeline-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      surfaceThinkingDeltas: true,
      content: "继续",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: vi.fn() as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: { type: "thinking_delta", text: "先分析。" } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: { type: "thinking_delta", text: "再查证。" } as AgentEvent,
    });

    expect(threadItems).toEqual([
      expect.objectContaining({
        id: "streamed-reasoning:turn-1:local-1",
        thread_id: "session-1",
        turn_id: "turn-1",
        sequence: 0,
        type: "reasoning",
        status: "in_progress",
        text: "先分析。再查证。",
      }),
    ]);

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_updated",
        protocol_method: "item/started",
        protocol_revision: "fixture-revision",
        item: {
          id: "reasoning-actual-1",
          thread_id: "session-1",
          turn_id: "turn-1",
          sequence: 3,
          type: "reasoning",
          text: "后端正式 reasoning。",
          status: "in_progress",
          started_at: "2026-06-17T08:00:00.000Z",
          updated_at: "2026-06-17T08:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(threadItems).toEqual([
      expect.objectContaining({
        id: "reasoning-actual-1",
        type: "reasoning",
        text: "后端正式 reasoning。",
      }),
    ]);
  });

  it("thinking 与工具交错时应按事件 sequence 分段展示", () => {
    let threadItems: AgentThreadItem[] = [];
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );
    const requestState = {
      accumulatedContent: "",
      currentTurnId: "turn-ordered",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (
          parts: NonNullable<Message["contentParts"]>,
          textDelta: string,
        ) => [...parts, { type: "thinking" as const, text: textDelta }],
      },
      eventName: "agent-runtime-thinking-ordered-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      surfaceThinkingDeltas: true,
      content: "继续",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: vi.fn() as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "thinking_delta",
        text: "先确认目标。",
        sequence: 1,
        timestamp: "2026-06-17T08:00:01.000Z",
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        tool_id: "tool-1",
        tool_name: "web_search",
        arguments: JSON.stringify({ query: "资料" }),
        sequence: 2,
        timestamp: "2026-06-17T08:00:02.000Z",
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "thinking_delta",
        text: "再整理结论。",
        sequence: 3,
        timestamp: "2026-06-17T08:00:03.000Z",
      } as AgentEvent,
    });

    expect(threadItems.map((item) => item.id)).toEqual([
      "streamed-reasoning:turn-ordered:1",
      "tool-1",
      "streamed-reasoning:turn-ordered:3",
    ]);
    expect(threadItems).toEqual([
      expect.objectContaining({
        id: "streamed-reasoning:turn-ordered:1",
        sequence: 1,
        status: "completed",
        text: "先确认目标。",
      }),
      expect.objectContaining({
        id: "tool-1",
        sequence: 2,
        type: "tool_call",
      }),
      expect.objectContaining({
        id: "streamed-reasoning:turn-ordered:3",
        sequence: 3,
        status: "in_progress",
        text: "再整理结论。",
      }),
    ]);
  });

  it("action_resolved 应同步收起 pending action 并回显已提交输入", () => {
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-17T08:00:00.000Z"),
        actionRequests: [
          {
            requestId: "ask-1",
            actionType: "ask_user",
            prompt: "请选择方向",
            status: "pending",
          },
        ],
        contentParts: [
          {
            type: "action_required",
            actionRequired: {
              requestId: "ask-1",
              actionType: "ask_user",
              prompt: "请选择方向",
              status: "pending",
            },
          },
        ],
      },
    ];
    let pendingActions = [
      {
        requestId: "ask-1",
        actionType: "ask_user" as const,
        prompt: "请选择方向",
        status: "pending" as const,
      },
    ];
    let threadItems: AgentThreadItem[] = [
      {
        id: "ask-1",
        thread_id: "session-1",
        turn_id: "turn-1",
        sequence: 4,
        type: "request_user_input",
        request_id: "ask-1",
        action_type: "ask_user",
        prompt: "请选择方向",
        status: "in_progress",
        started_at: "2026-06-17T08:00:04.000Z",
        updated_at: "2026-06-17T08:00:04.000Z",
      },
    ];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setPendingActions = vi.fn(
      (
        value:
          | typeof pendingActions
          | ((prev: typeof pendingActions) => typeof pendingActions),
      ) => {
        pendingActions =
          typeof value === "function" ? value(pendingActions) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );

    handleTurnStreamEvent({
      data: {
        type: "action_resolved",
        request_id: "ask-1",
        action_type: "ask_user",
        data: { answer: "极简" },
        scope: {
          session_id: "session-1",
          thread_id: "thread-1",
          turn_id: "turn-1",
        },
        sequence: 5,
        timestamp: "2026-06-17T08:00:05.000Z",
      } as AgentEvent,
      requestState: {
        accumulatedContent: "",
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-action-resolved-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "继续",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: setPendingActions as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(pendingActions).toEqual([]);
    expect(messages[0]?.actionRequests?.[0]).toMatchObject({
      requestId: "ask-1",
      status: "submitted",
      submittedResponse: '{"answer":"极简"}',
      submittedUserData: { answer: "极简" },
    });
    expect(
      messages[0]?.contentParts?.find(
        (part) => part.type === "action_required",
      ),
    ).toMatchObject({
      type: "action_required",
      actionRequired: {
        requestId: "ask-1",
        status: "submitted",
        submittedResponse: '{"answer":"极简"}',
      },
    });
    expect(threadItems[0]).toMatchObject({
      id: "ask-1",
      type: "request_user_input",
      status: "completed",
      response: { answer: "极简" },
    });
  });

  it("收到 turn_completed 时应剥离 assistant 正文中的工具协议残留", () => {
    let messages: Message[] = [
      {
        id: "assistant-2",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-04-07T10:00:00.000Z"),
        isThinking: true,
      },
    ];

    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );

    handleTurnStreamEvent({
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-test",
          thread_id: "thread-test",
          prompt_text: "test",
          status: "completed",
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
      } as AgentEvent,
      requestState: {
        accumulatedContent:
          '<tool_result>{"output":"saved"}</tool_result>\n\n已保存到项目目录。',
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-2",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(messages[0]?.content).toBe("已保存到项目目录。");
    expect(messages[0]?.contentParts).toEqual([
      { type: "text", text: "已保存到项目目录。" },
    ]);
  });

  it("收到空 turn_completed 且没有真实产物信号时应落成失败态", () => {
    let messages: Message[] = [
      {
        id: "assistant-3",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-04-07T10:00:00.000Z"),
        isThinking: true,
      },
    ];

    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );

    handleTurnStreamEvent({
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-test",
          thread_id: "thread-test",
          prompt_text: "test",
          status: "completed",
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
      } as AgentEvent,
      requestState: {
        accumulatedContent: "",
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-3",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(messages[0]?.content).toContain("执行失败：");
    expect(messages[0]?.runtimeStatus).toMatchObject({
      phase: "failed",
      title: "当前处理失败",
      detail: "模型未输出最终答复，请重试",
    });
    expect(mockToast.error).toHaveBeenCalledWith("模型未输出最终答复，请重试");
  });

  it("站点导出在 tool_end 已登记结果时，空 turn_completed 不应误报缺少最终答复", () => {
    let messages: Message[] = [
      {
        id: "assistant-site-export",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-04-07T10:00:00.000Z"),
        isThinking: true,
        toolCalls: [
          {
            id: "tool-site-export-1",
            name: "site_run_adapter",
            status: "running",
            startTime: new Date("2026-04-07T10:00:00.000Z"),
          },
        ],
      },
    ];

    const requestState = {
      accumulatedContent: "",
      hasMeaningfulCompletionSignal: false,
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };

    const callbacks = {
      activateStream: () => {},
      isStreamActivated: () => true,
      clearOptimisticItem: () => {},
      clearOptimisticTurn: () => {},
      disposeListener: () => {},
      clearActiveStreamIfMatch: () => true,
      appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
        parts,
    };

    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );

    handleTurnStreamEvent({
      data: {
        type: "tool_end",
        tool_id: "tool-site-export-1",
        result: {
          success: true,
          output: "exports/x-article-export/article/index.md",
          metadata: {
            tool_family: "site",
            saved_content: {
              content_id: "content-site-export-1",
              project_id: "project-site-export-1",
              markdown_relative_path:
                "exports/x-article-export/article/index.md",
            },
          },
        },
      } as AgentEvent,
      requestState,
      callbacks,
      eventName: "agent-runtime-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-site-export",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>([
        ["tool-site-export-1", "site_run_adapter"],
      ]),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(requestState.hasMeaningfulCompletionSignal).toBe(true);

    handleTurnStreamEvent({
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-test",
          thread_id: "thread-test",
          prompt_text: "test",
          status: "completed",
          started_at: "2026-06-07T10:00:00.000Z",
          completed_at: "2026-06-07T10:00:01.000Z",
          created_at: "2026-06-07T10:00:00.000Z",
          updated_at: "2026-06-07T10:00:01.000Z",
        },
      } as AgentEvent,
      requestState,
      callbacks,
      eventName: "agent-runtime-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-site-export",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(messages[0]?.content).toBe(
      "本轮执行已完成，详细过程与产物已保留在当前对话中。",
    );
    expect(mockToast.error).not.toHaveBeenCalledWith(
      "模型未输出最终答复，请重试",
    );
  });

  it("命中空最终答复错误但已有真实产物信号时仍应软完成", () => {
    let messages: Message[] = [
      {
        id: "assistant-site-export-error",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-04-07T10:00:00.000Z"),
        isThinking: true,
      },
    ];

    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );

    handleTurnStreamEvent({
      data: {
        type: "error",
        message:
          "已完成当前回合的工具执行，但模型未输出最终答复。\n尝试记录: site_run_adapter#tool-site-export-2:success",
      } as AgentEvent,
      requestState: {
        accumulatedContent: "",
        hasMeaningfulCompletionSignal: true,
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-site-export-error",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(messages[0]?.content).toBe(
      "本轮执行已完成，详细过程与产物已保留在当前对话中。",
    );
    expect(messages[0]?.runtimeStatus).toBeUndefined();
    expect(mockToast.error).not.toHaveBeenCalledWith(
      "模型未输出最终答复，请重试",
    );
  });

  it("provider stream 失败即使已有工具过程也应落失败态并保留过程卡", () => {
    const providerUnavailableMessage =
      "当前模型通道暂时不可用，请稍后重试；如果持续失败，请检查 Provider 状态或切换到其他可用模型。";
    let messages: Message[] = [
      {
        id: "assistant-provider-error",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-01T11:14:20.000Z"),
        isThinking: true,
        contentParts: [
          {
            type: "tool_use",
            toolCall: {
              id: "tool-search-1",
              name: "web_search",
              arguments: JSON.stringify({ query: "international news today" }),
              status: "completed",
              startTime: new Date("2026-06-01T11:14:11.000Z"),
              endTime: new Date("2026-06-01T11:14:19.000Z"),
              result: { success: true, output: "results" },
            },
          },
        ],
      },
    ];

    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );

    handleTurnStreamEvent({
      data: {
        type: "error",
        message:
          "[Agent][TTFT] provider stream request failed before body: provider=openai, model=gpt-5.5, elapsed_ms=8517, error=Server error: Server error (503 Service Unavailable): Service temporarily unavailable",
      } as AgentEvent,
      requestState: {
        accumulatedContent: "",
        hasMeaningfulCompletionSignal: true,
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-provider-error",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-provider-error",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(messages[0]?.content).toContain("执行失败：");
    expect(messages[0]?.contentParts).toEqual([
      expect.objectContaining({ type: "tool_use" }),
      expect.objectContaining({
        type: "text",
        text: expect.stringContaining("执行失败："),
      }),
    ]);
    expect(messages[0]?.runtimeStatus).toMatchObject({
      phase: "failed",
      detail: providerUnavailableMessage,
    });
    expect(mockToast.error).toHaveBeenCalledWith(providerUnavailableMessage);
    expect(mockToast.error).not.toHaveBeenCalledWith(
      "模型未输出最终答复，请重试",
    );
  });

  it("运行时权限确认等待错误应保留确认卡，不渲染失败正文", () => {
    const clearOptimisticItem = vi.fn();
    const clearActiveStreamIfMatch = vi.fn(() => true);
    const disposeListener = vi.fn();
    const setIsSending = vi.fn();
    let messages: Message[] = [
      {
        id: "assistant-permission-wait",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-05-07T10:00:00.000Z"),
        isThinking: true,
        actionRequests: [
          {
            requestId: "runtime_permission_confirmation:turn-1",
            actionType: "elicitation",
            prompt: "当前执行需要确认运行时权限：web_search。",
            status: "pending",
          },
        ],
        contentParts: [
          {
            type: "action_required",
            actionRequired: {
              requestId: "runtime_permission_confirmation:turn-1",
              actionType: "elicitation",
              prompt: "当前执行需要确认运行时权限：web_search。",
              status: "pending",
            },
          },
        ],
      },
    ];

    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );

    handleTurnStreamEvent({
      data: {
        type: "error",
        message:
          "运行时权限声明需要真实确认，当前 turn 已在模型执行前等待用户确认：confirmationStatus=not_requested，askProfileKeys=web_search。已创建真实权限确认请求；请确认后重试或恢复本轮执行。",
      } as AgentEvent,
      requestState: {
        accumulatedContent: "",
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      callbacks: {
        activateStream: () => {},
        isStreamActivated: () => true,
        clearOptimisticItem,
        clearOptimisticTurn: () => {},
        disposeListener,
        clearActiveStreamIfMatch,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-runtime-permission-wait",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-permission-wait",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: setIsSending as never,
    });

    expect(clearOptimisticItem).toHaveBeenCalledTimes(1);
    expect(clearActiveStreamIfMatch).toHaveBeenCalledWith(
      "agent-runtime-permission-wait",
    );
    expect(disposeListener).toHaveBeenCalledTimes(1);
    expect(setIsSending).toHaveBeenCalledWith(false);
    expect(messages[0]?.content).toBe("");
    expect(messages[0]?.isThinking).toBe(false);
    expect(messages[0]?.runtimeStatus).toBeUndefined();
    expect(messages[0]?.actionRequests?.[0]?.status).toBe("pending");
    expect(mockToast.error).not.toHaveBeenCalled();
  });
});
