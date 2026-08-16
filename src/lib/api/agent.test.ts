import { beforeEach, describe, expect, it, vi } from "vitest";

const {
  mockIsElectronHostCommandAvailable,
  mockLogAgentDebug,
  mockSafeListen,
  mockSafeInvoke,
} = vi.hoisted(() => ({
  mockIsElectronHostCommandAvailable: vi.fn(),
  mockLogAgentDebug: vi.fn(),
  mockSafeListen: vi.fn(),
  mockSafeInvoke: vi.fn(),
}));

vi.mock("@/lib/agentDebug", () => ({
  logAgentDebug: mockLogAgentDebug,
}));

vi.mock("@/lib/dev-bridge", () => ({
  safeInvoke: mockSafeInvoke,
  safeListen: mockSafeListen,
}));

vi.mock("@/lib/electron-host", () => ({
  isElectronHostCommandAvailable: mockIsElectronHostCommandAvailable,
}));

import {
  APP_SERVER_METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT,
  APP_SERVER_METHOD_AGENT_SESSION_HANDOFF_BUNDLE_EXPORT,
  APP_SERVER_METHOD_THREAD_READ,
  APP_SERVER_METHOD_THREAD_DELETE,
  APP_SERVER_METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT,
  APP_SERVER_METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE,
  APP_SERVER_METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT,
  APP_SERVER_METHOD_THREAD_START,
  APP_SERVER_METHOD_THREAD_RESUME,
  APP_SERVER_METHOD_AGENT_SESSION_TOOL_INVENTORY_READ,
  APP_SERVER_METHOD_TURN_START,
} from "./appServer";
import {
  generateAgentRuntimeTitleResult,
  generateAgentRuntimeTitle,
  generateAgentRuntimeSessionTitle,
} from "./agentRuntime/agentClient";
import {
  AGENT_RUNTIME_RENDERER_EVENT_NAME_CONTEXT_KEY,
  createApplicationAdditionalContext,
} from "./agentProtocolOps";
import {
  exportAgentRuntimeAnalysisHandoff,
  exportAgentRuntimeHandoffBundle,
  exportAgentRuntimeReplayCase,
  exportAgentRuntimeReviewDecisionTemplate,
  saveAgentRuntimeReviewDecision,
} from "./agentRuntime/exportClient";
import { getAgentRuntimeToolInventory } from "./agentRuntime/inventoryClient";
import {
  createAgentRuntimeSession,
  deleteAgentRuntimeSession,
  getAgentRuntimeSession,
  listAgentRuntimeSessions,
} from "./agentRuntime/sessionClient";
import {
  getAgentRuntimeThreadRead,
  replayAgentRuntimeRequest,
  resumeThread,
  respondAgentRuntimeAction,
  submitAgentRuntimeTurn,
} from "./agentRuntime/threadClient";

function line(value: unknown): string {
  return `${JSON.stringify(value)}\n`;
}

function canonicalThread(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    cliVersion: "1.111.0",
    cwd: "/tmp/workspace",
    modelProvider: "openai",
    source: "appServer",
    id: "thread-runtime",
    sessionId: "session-runtime",
    preview: "Runtime Session",
    ephemeral: false,
    createdAt: 1710000000,
    updatedAt: 1710000123,
    status: { type: "idle" },
    turns: [],
    ...overrides,
  };
}

type AppServerMockEnvelope =
  | { result: unknown }
  | { error: { code: number; message: string; data?: unknown } };

const appServerResponseQueue: AppServerMockEnvelope[] = [];

function mockAppServerResponse(result: unknown): void {
  appServerResponseQueue.push({ result });
}

function mockAppServerError(message: string, code = -32000): void {
  appServerResponseQueue.push({ error: { code, message } });
}

function installAppServerMock(): void {
  mockSafeInvoke.mockImplementation(async (command, args) => {
    if (command === "app_server_drain_events") {
      return { lines: [] };
    }
    if (command !== "app_server_handle_json_lines") {
      return undefined;
    }

    const envelope = appServerResponseQueue.shift();
    const requestLine = args?.request?.lines?.[0];
    const request =
      typeof requestLine === "string"
        ? (JSON.parse(requestLine) as { id: number | string })
        : { id: 1 };

    return {
      lines: [
        line({
          id: request.id,
          ...(envelope ?? { result: undefined }),
        }),
      ],
    };
  });
}

function expectAppServerRequest(
  callIndex: number,
  method: string,
  params: Record<string, unknown>,
): void {
  const call = mockSafeInvoke.mock.calls.filter(
    (safeInvokeCall) => safeInvokeCall[0] === "app_server_handle_json_lines",
  )[callIndex - 1];
  expect(call?.[0]).toBe("app_server_handle_json_lines");
  const requestLine = call?.[1]?.request?.lines?.[0];
  expect(typeof requestLine).toBe("string");
  const request = JSON.parse(requestLine as string);
  expect(request).toMatchObject({
    method,
    params,
  });
}

describe("Agent API 治理护栏", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    appServerResponseQueue.length = 0;
    installAppServerMock();
    mockIsElectronHostCommandAvailable.mockReturnValue(true);
    mockSafeListen.mockResolvedValue(vi.fn());
  });

  it("createAgentRuntimeSession 应经 Electron IPC 调 App Server thread/start", async () => {
    mockAppServerResponse({
      thread: canonicalThread({
        cwd: "/tmp/workspace-2",
        id: "thread-created",
        sessionId: "session-created",
        preview: "新会话",
      }),
    });

    await expect(
      createAgentRuntimeSession("workspace-2", "新会话", "react", {
        runStartHooks: false,
        workingDir: "/tmp/workspace-2",
        metadata: {
          providerSelector: "provider-runtime",
          modelName: "model-runtime",
        },
      }),
    ).resolves.toBe("session-created");

    expectAppServerRequest(1, APP_SERVER_METHOD_THREAD_START, {
      cwd: "/tmp/workspace-2",
      model: "model-runtime",
      modelProvider: "provider-runtime",
      serviceName: "新会话",
      threadSource: "appServer",
      historyMode: "paginated",
    });
  });

  it("submitAgentRuntimeTurn 应经 Electron IPC 调 App Server turn/start", async () => {
    mockAppServerResponse({
      turn: {
        id: "turn-runtime",
        status: "inProgress",
        items: [],
      },
    });

    await submitAgentRuntimeTurn({
      threadId: "thread-runtime",
      input: [{ type: "text", text: "runtime hello" }],
      additionalContext: createApplicationAdditionalContext({
        [AGENT_RUNTIME_RENDERER_EVENT_NAME_CONTEXT_KEY]: "event-runtime",
        metadata: { source: "hook-facade" },
      }),
    });

    expectAppServerRequest(1, APP_SERVER_METHOD_TURN_START, {
      threadId: "thread-runtime",
      input: [{ type: "text", text: "runtime hello" }],
      additionalContext: createApplicationAdditionalContext({
        metadata: { source: "hook-facade" },
      }),
    });
  });

  it("submitAgentRuntimeTurn 应通过 additionalContext 保留工具与排队元数据", async () => {
    mockAppServerResponse({
      turn: {
        id: "queued-turn-1",
        status: "inProgress",
        items: [],
      },
    });

    await submitAgentRuntimeTurn({
      threadId: "thread-runtime-search",
      input: [{ type: "text", text: "查一下今天的汇率" }],
      additionalContext: createApplicationAdditionalContext({
        [AGENT_RUNTIME_RENDERER_EVENT_NAME_CONTEXT_KEY]: "event-runtime-search",
        metadata: {
          workspaceId: "workspace-runtime-search",
          executionStrategy: "react",
          webSearch: true,
          queueIfBusy: true,
          queuedTurnId: "queued-turn-1",
        },
      }),
    });

    expectAppServerRequest(1, APP_SERVER_METHOD_TURN_START, {
      threadId: "thread-runtime-search",
      input: [{ type: "text", text: "查一下今天的汇率" }],
      additionalContext: createApplicationAdditionalContext({
        metadata: {
          workspaceId: "workspace-runtime-search",
          executionStrategy: "react",
          webSearch: true,
          queueIfBusy: true,
          queuedTurnId: "queued-turn-1",
        },
      }),
    });
  });

  it("submitAgentRuntimeTurn 应通过 canonical turn/start 支持模型与策略字段", async () => {
    mockAppServerResponse({
      turn: {
        id: "turn-runtime-preference",
        status: "inProgress",
        items: [],
      },
    });

    await submitAgentRuntimeTurn({
      threadId: "thread-runtime-preference",
      input: [{ type: "text", text: "请继续" }],
      model: "gpt-5.3-codex",
      effort: "high",
      approvalPolicy: "on-request",
      sandboxPolicy: "workspace-write",
      additionalContext: createApplicationAdditionalContext({
        [AGENT_RUNTIME_RENDERER_EVENT_NAME_CONTEXT_KEY]:
          "event-runtime-preference",
        metadata: {
          workspaceId: "workspace-runtime-preference",
          providerSelector: "custom-provider",
        },
      }),
    });

    expectAppServerRequest(1, APP_SERVER_METHOD_TURN_START, {
      threadId: "thread-runtime-preference",
      input: [{ type: "text", text: "请继续" }],
      model: "gpt-5.3-codex",
      effort: "high",
      approvalPolicy: "on-request",
      sandboxPolicy: "workspace-write",
      additionalContext: createApplicationAdditionalContext({
        metadata: {
          workspaceId: "workspace-runtime-preference",
          providerSelector: "custom-provider",
        },
      }),
    });
  });

  it("respondAgentRuntimeAction 缺少 typed pending 时应 fail closed，不发旧 action/respond", async () => {
    await expect(
      respondAgentRuntimeAction({
        session_id: "session-runtime",
        request_id: "req-runtime",
        action_type: "ask_user",
        confirmed: true,
      }),
    ).rejects.toThrow(
      "Typed server request is no longer pending; generic agentSession/action/respond is retired.",
    );
    expect(
      mockSafeInvoke.mock.calls.some(
        (call) => call[0] === "app_server_handle_json_lines",
      ),
    ).toBe(false);
  });

  it("resumeThread 应经 Electron IPC 调 App Server thread/resume", async () => {
    mockAppServerResponse({
      thread: canonicalThread({
        id: "thread-runtime-resume",
        sessionId: "session-runtime-resume",
      }),
      model: "gpt-5.4",
      modelProvider: "openai",
      cwd: "/tmp/workspace",
    });

    await expect(
      resumeThread({
        threadId: "thread-runtime-resume",
      }),
    ).resolves.toMatchObject({
      result: {
        thread: {
          id: "thread-runtime-resume",
        },
      },
    });

    expectAppServerRequest(1, APP_SERVER_METHOD_THREAD_RESUME, {
      threadId: "thread-runtime-resume",
      excludeTurns: true,
    });
  });

  it("replayAgentRuntimeRequest 无当前 typed pending 时应 fail closed", async () => {
    await expect(
      replayAgentRuntimeRequest({
        session_id: "session-runtime-replay",
        request_id: "req-runtime-replay",
      }),
    ).resolves.toBeNull();
  });

  it("getAgentRuntimeThreadRead 应经 Electron IPC 调 canonical thread/read", async () => {
    mockAppServerResponse({
      thread: canonicalThread({
        status: { type: "active", activeFlags: ["waitingOnUserInput"] },
        turns: [
          {
            id: "turn-runtime",
            status: "inProgress",
            items: [],
          },
          {
            id: "queued-turn-1",
            status: "inProgress",
            startedAt: 1711184400,
            queue: { state: "queued", position: 1 },
            items: [
              {
                id: "queued-user-1",
                type: "userMessage",
                content: [{ type: "text", text: "继续执行" }],
              },
            ],
          },
        ],
      }),
    });

    await expect(
      getAgentRuntimeThreadRead("session-runtime"),
    ).resolves.toMatchObject({
      thread_id: "thread-runtime",
      status: "waitingAction",
      profile_status: "blocked",
      active_turn_id: "turn-runtime",
    });

    expectAppServerRequest(1, APP_SERVER_METHOD_THREAD_READ, {
      threadId: "session-runtime",
      includeTurns: true,
    });
  });

  it("exportAgentRuntimeReplayCase 应经 Electron IPC 调 App Server replayCase/export", async () => {
    mockAppServerResponse({
      sessionId: "session-runtime-replay-case",
      threadId: "thread-runtime-replay-case",
      workspaceRoot: "/tmp/workspace",
      replayRelativeRoot:
        ".lime/harness/sessions/session-runtime-replay-case/replay",
      replayAbsoluteRoot:
        "/tmp/workspace/.lime/harness/sessions/session-runtime-replay-case/replay",
      handoffBundleRelativeRoot:
        ".lime/harness/sessions/session-runtime-replay-case",
      evidencePackRelativeRoot:
        ".lime/harness/sessions/session-runtime-replay-case/evidence",
      exportedAt: "2026-03-27T09:50:00.000Z",
      threadStatus: "waiting_request",
      pendingRequestCount: 1,
      queuedTurnCount: 1,
      linkedHandoffArtifactCount: 4,
      linkedEvidenceArtifactCount: 4,
      recentArtifactCount: 2,
      artifacts: [],
    });

    await expect(
      exportAgentRuntimeReplayCase("session-runtime-replay-case"),
    ).resolves.toMatchObject({
      replay_relative_root:
        ".lime/harness/sessions/session-runtime-replay-case/replay",
      linked_handoff_artifact_count: 4,
    });

    expectAppServerRequest(
      1,
      APP_SERVER_METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT,
      {
        sessionId: "session-runtime-replay-case",
      },
    );
  });

  it("listAgentRuntimeSessions 应返回现役 runtime 会话列表", async () => {
    mockAppServerResponse({
      data: [],
      nextCursor: null,
    });
    mockAppServerResponse({
      data: [
        canonicalThread({
          id: "thread-runtime-1",
          sessionId: "session-runtime-1",
          preview: "Runtime Session",
          modelProvider: "claude-sonnet-4-20250514",
          metadata: {
            messagesCount: 3,
            workspaceId: "workspace-1",
            workingDir: "/tmp/workspace-1",
            executionStrategy: "react",
            harness: {
              plugin_history_restore: {
                session_id: "session-runtime-1",
                plugin_id: "content-factory@limecloud",
              },
            },
          },
          cwd: "/tmp/workspace-1",
        }),
      ],
      nextCursor: null,
      backwardsCursor: null,
    });

    await expect(listAgentRuntimeSessions()).resolves.toEqual([
      {
        id: "session-runtime-1",
        thread_id: "thread-runtime-1",
        name: "Runtime Session",
        model: "claude-sonnet-4-20250514",
        created_at: 1710000000000,
        updated_at: 1710000123000,
        archived_at: null,
        messages_count: 3,
        workspace_id: "workspace-1",
        working_dir: "/tmp/workspace-1",
        execution_strategy: "react",
        session_business_object_ref_metadata: {
          messagesCount: 3,
          workspaceId: "workspace-1",
          workingDir: "/tmp/workspace-1",
          executionStrategy: "react",
          harness: {
            plugin_history_restore: {
              session_id: "session-runtime-1",
              plugin_id: "content-factory@limecloud",
            },
          },
        },
        thread_status: "idle",
        queued_turn_count: 0,
      },
    ]);
    expectAppServerRequest(1, "threadSection/list", {
      limit: 100,
    });
    expectAppServerRequest(2, "thread/list", {
      archived: false,
      limit: 100,
      sectionId: null,
    });
  });

  it("listAgentRuntimeSessions 应支持请求包含归档会话", async () => {
    mockAppServerResponse({
      data: [],
      nextCursor: null,
    });
    mockAppServerResponse({
      data: [],
      nextCursor: null,
      backwardsCursor: null,
    });
    mockAppServerResponse({
      data: [
        canonicalThread({
          id: "thread-runtime-archived",
          sessionId: "session-runtime-archived",
          preview: "Archived Runtime Session",
          modelProvider: "gpt-5.4",
          updatedAt: 1710000300,
          archived: true,
        }),
      ],
      nextCursor: null,
      backwardsCursor: null,
    });

    await expect(
      listAgentRuntimeSessions({ includeArchived: true }),
    ).resolves.toEqual([
      {
        id: "session-runtime-archived",
        thread_id: "thread-runtime-archived",
        name: "Archived Runtime Session",
        model: "gpt-5.4",
        created_at: 1710000000000,
        updated_at: 1710000300000,
        archived_at: 1710000300000,
        messages_count: 0,
        working_dir: "/tmp/workspace",
        thread_status: "idle",
        queued_turn_count: 0,
      },
    ]);

    expectAppServerRequest(1, "threadSection/list", {
      limit: 100,
    });
    expectAppServerRequest(2, "thread/list", {
      archived: false,
      limit: 100,
      sectionId: null,
    });
    expectAppServerRequest(3, "thread/list", {
      archived: true,
      limit: 100,
      sectionId: null,
    });
  });

  it("listAgentRuntimeSessions 应支持工作区限流与仅归档过滤", async () => {
    mockAppServerResponse({
      data: [],
      nextCursor: null,
    });
    mockAppServerResponse({
      data: [
        canonicalThread({
          id: "thread-runtime-archived",
          sessionId: "session-runtime-archived",
          preview: "Archived Runtime Session",
          modelProvider: "gpt-5.4",
          updatedAt: 1710000300,
          archived: true,
          metadata: { workspaceId: "workspace-1" },
        }),
      ],
      nextCursor: null,
      backwardsCursor: null,
    });

    await expect(
      listAgentRuntimeSessions({
        archivedOnly: true,
        workspaceId: "workspace-1",
        limit: 12,
      }),
    ).resolves.toEqual([
      {
        id: "session-runtime-archived",
        thread_id: "thread-runtime-archived",
        name: "Archived Runtime Session",
        model: "gpt-5.4",
        created_at: 1710000000000,
        updated_at: 1710000300000,
        archived_at: 1710000300000,
        workspace_id: "workspace-1",
        working_dir: "/tmp/workspace",
        messages_count: 0,
        session_business_object_ref_metadata: {
          workspaceId: "workspace-1",
        },
        thread_status: "idle",
        queued_turn_count: 0,
      },
    ]);

    expectAppServerRequest(1, "threadSection/list", {
      limit: 100,
    });
    expectAppServerRequest(2, "thread/list", {
      archived: true,
      limit: 12,
      sectionId: null,
    });
  });

  it("getAgentRuntimeSession 应返回现役 runtime 详情并排除 queued turn 历史", async () => {
    mockAppServerResponse({
      thread: canonicalThread({
        id: "thread-runtime-2",
        sessionId: "session-runtime-2",
        preview: "Runtime Detail",
        modelProvider: "gpt-5.4",
        cwd: "/tmp/workspace-2",
        createdAt: 1710001000,
        updatedAt: 1710002000,
        status: { type: "active", activeFlags: [] },
        metadata: {
          workspaceId: "workspace-2",
          workingDir: "/tmp/workspace-2",
          executionStrategy: "react",
        },
        turns: [
          {
            id: "turn-runtime-2",
            status: "completed",
            startedAt: 1710001000,
            completedAt: 1710002000,
            items: [
              {
                id: "item-user-2",
                type: "userMessage",
                content: [{ type: "text", text: "hello" }],
              },
              {
                id: "item-agent-2",
                type: "agentMessage",
                text: "world",
                phase: "final_answer",
              },
            ],
          },
          {
            id: "queued-2",
            status: "inProgress",
            startedAt: 1710001510,
            queue: { state: "queued", position: 1 },
            items: [
              {
                id: "item-queued-user-2",
                type: "userMessage",
                content: [
                  {
                    type: "text",
                    text: "线程读模型中的排队任务",
                  },
                ],
              },
            ],
          },
        ],
      }),
    });

    await expect(
      getAgentRuntimeSession("session-runtime-2"),
    ).resolves.toMatchObject({
      id: "session-runtime-2",
      thread_id: "thread-runtime-2",
      name: "Runtime Detail",
      model: "gpt-5.4",
      created_at: 1710001000000,
      updated_at: 1710002000000,
      workspace_id: "workspace-2",
      working_dir: "/tmp/workspace-2",
      execution_strategy: "react",
      thread_read: {
        thread_id: "thread-runtime-2",
        status: "running",
        profile_status: "running",
      },
      messages: [],
      items: [
        {
          id: "item-user-2",
          type: "user_message",
          content: "hello",
        },
        {
          id: "item-agent-2",
          type: "agent_message",
          text: "world",
        },
      ],
      turns: [
        expect.objectContaining({
          id: "turn-runtime-2",
          status: "completed",
        }),
      ],
    });
    expectAppServerRequest(1, APP_SERVER_METHOD_THREAD_READ, {
      threadId: "session-runtime-2",
      includeTurns: false,
    });
  });

  it("getAgentRuntimeSession 不应把 renderer resume hooks 标记泄漏到 thread/read", async () => {
    mockAppServerResponse({
      thread: canonicalThread({
        id: "thread-runtime-resume",
        sessionId: "session-runtime-resume",
        turns: [{ id: "turn-resume", status: "completed", items: [] }],
      }),
    });

    await expect(
      getAgentRuntimeSession("session-runtime-resume", {
        resumeSessionStartHooks: true,
      }),
    ).resolves.toMatchObject({
      id: "session-runtime-resume",
      messages: [],
    });

    expectAppServerRequest(1, APP_SERVER_METHOD_THREAD_READ, {
      threadId: "session-runtime-resume",
      includeTurns: false,
    });
  });

  it("getAgentRuntimeSession 不应把 renderer history tail 限制泄漏到 thread/read", async () => {
    mockAppServerResponse({
      thread: canonicalThread({
        id: "thread-runtime-tail",
        sessionId: "session-runtime-tail",
        turns: [{ id: "turn-tail", status: "completed", items: [] }],
      }),
    });

    await expect(
      getAgentRuntimeSession("session-runtime-tail", {
        historyLimit: 120,
      }),
    ).resolves.toMatchObject({
      id: "session-runtime-tail",
      messages: [],
    });

    expectAppServerRequest(1, APP_SERVER_METHOD_THREAD_READ, {
      threadId: "session-runtime-tail",
      includeTurns: false,
    });
  });

  it("getAgentRuntimeSession 遇到 transient DevBridge 读失败时只输出 warn 调试日志", async () => {
    mockSafeInvoke.mockRejectedValue(
      new Error(
        '[DevBridge] 浏览器模式无法连接后端桥接，命令 "app_server_handle_json_lines" 执行失败。原始错误: Failed to fetch (timeout after 20000ms)',
      ),
    );

    await expect(
      getAgentRuntimeSession("session-runtime-transient", {
        historyLimit: 40,
      }),
    ).rejects.toThrow("timeout after 20000ms");

    const errorDebugCall = mockLogAgentDebug.mock.calls.find(
      ([component, phase]) =>
        component === "AgentApi" && phase === "runtimeGetSession.error",
    );

    expect(errorDebugCall).toBeTruthy();
    expect(errorDebugCall?.[3]).toMatchObject({ level: "warn" });
  });

  it("exportAgentRuntimeHandoffBundle 应经 Electron IPC 调 App Server handoffBundle/export", async () => {
    mockAppServerResponse({
      sessionId: "session-runtime-3",
      threadId: "thread-runtime-3",
      workspaceRoot: "/tmp/workspace-3",
      bundleRelativeRoot: ".lime/harness/sessions/session-runtime-3",
      bundleAbsoluteRoot:
        "/tmp/workspace-3/.lime/harness/sessions/session-runtime-3",
      exportedAt: "2026-03-27T10:00:00Z",
      threadStatus: "running",
      latestTurnStatus: "completed",
      pendingRequestCount: 1,
      queuedTurnCount: 0,
      activeSubagentCount: 2,
      todoTotal: 3,
      todoPending: 1,
      todoInProgress: 1,
      todoCompleted: 1,
      artifacts: [
        {
          kind: "handoff",
          title: "交接摘要",
          relativePath: ".lime/harness/sessions/session-runtime-3/handoff.md",
          absolutePath:
            "/tmp/workspace-3/.lime/harness/sessions/session-runtime-3/handoff.md",
          bytes: 512,
        },
      ],
    });

    await expect(
      exportAgentRuntimeHandoffBundle("session-runtime-3"),
    ).resolves.toMatchObject({
      session_id: "session-runtime-3",
      thread_status: "running",
      pending_request_count: 1,
      artifacts: [
        expect.objectContaining({
          kind: "handoff",
          relative_path: ".lime/harness/sessions/session-runtime-3/handoff.md",
        }),
      ],
    });

    expectAppServerRequest(
      1,
      APP_SERVER_METHOD_AGENT_SESSION_HANDOFF_BUNDLE_EXPORT,
      {
        sessionId: "session-runtime-3",
      },
    );
  });

  it("exportAgentRuntimeAnalysisHandoff 应兼容 camelCase / snake_case 并经 App Server 导出", async () => {
    mockAppServerResponse({
      sessionId: "session-runtime-4a",
      threadId: "thread-runtime-4a",
      workspaceRoot: "/tmp/workspace-4a",
      analysisRelativeRoot:
        ".lime/harness/sessions/session-runtime-4a/analysis",
      analysisAbsoluteRoot:
        "/tmp/workspace-4a/.lime/harness/sessions/session-runtime-4a/analysis",
      handoffBundleRelativeRoot: ".lime/harness/sessions/session-runtime-4a",
      evidencePackRelativeRoot:
        ".lime/harness/sessions/session-runtime-4a/evidence",
      replayCaseRelativeRoot:
        ".lime/harness/sessions/session-runtime-4a/replay",
      exportedAt: "2026-03-27T10:08:00Z",
      title: "确认当前失败案例如何交给外部 AI 修复",
      threadStatus: "waiting_request",
      latestTurnStatus: "action_required",
      pendingRequestCount: 1,
      queuedTurnCount: 0,
      sanitizedWorkspaceRoot: "/workspace/lime",
      copyPrompt: "# Lime 外部诊断与修复任务",
      artifacts: [
        {
          kind: "analysis_brief",
          title: "外部分析简报",
          relativePath:
            ".lime/harness/sessions/session-runtime-4a/analysis/analysis-brief.md",
          absolutePath:
            "/tmp/workspace-4a/.lime/harness/sessions/session-runtime-4a/analysis/analysis-brief.md",
          bytes: 320,
        },
      ],
    });

    await expect(
      exportAgentRuntimeAnalysisHandoff("session-runtime-4a"),
    ).resolves.toMatchObject({
      session_id: "session-runtime-4a",
      thread_status: "waiting_request",
      copy_prompt: "# Lime 外部诊断与修复任务",
      artifacts: [
        expect.objectContaining({
          kind: "analysis_brief",
          relative_path:
            ".lime/harness/sessions/session-runtime-4a/analysis/analysis-brief.md",
        }),
      ],
    });

    expectAppServerRequest(
      1,
      APP_SERVER_METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT,
      {
        sessionId: "session-runtime-4a",
      },
    );
  });

  it("exportAgentRuntimeReviewDecisionTemplate 应兼容 camelCase / snake_case 并经 App Server 导出", async () => {
    mockAppServerResponse({
      sessionId: "session-runtime-4b",
      threadId: "thread-runtime-4b",
      workspaceRoot: "/tmp/workspace-4b",
      reviewRelativeRoot: ".lime/harness/sessions/session-runtime-4b/review",
      reviewAbsoluteRoot:
        "/tmp/workspace-4b/.lime/harness/sessions/session-runtime-4b/review",
      analysisRelativeRoot:
        ".lime/harness/sessions/session-runtime-4b/analysis",
      analysisAbsoluteRoot:
        "/tmp/workspace-4b/.lime/harness/sessions/session-runtime-4b/analysis",
      handoffBundleRelativeRoot: ".lime/harness/sessions/session-runtime-4b",
      evidencePackRelativeRoot:
        ".lime/harness/sessions/session-runtime-4b/evidence",
      replayCaseRelativeRoot:
        ".lime/harness/sessions/session-runtime-4b/replay",
      exportedAt: "2026-03-27T10:18:00Z",
      title: "记录人工审核决策",
      threadStatus: "waiting_request",
      latestTurnStatus: "action_required",
      pendingRequestCount: 1,
      queuedTurnCount: 0,
      defaultDecisionStatus: "pending_review",
      limitStatus: "user_locked_capability_gap",
      capabilityGap: "browser_reasoning_candidate_missing",
      userLockedCapabilitySummary:
        "显式用户模型锁定不满足当前 execution profile（capabilityGap=browser_reasoning_candidate_missing），不能作为成功交付证据。",
      permissionStatus: "requires_confirmation",
      permissionConfirmationStatus: "denied",
      permissionConfirmationRequestId: "approval-denied",
      permissionConfirmationSource: "runtime_action_required",
      permissionConfirmationSummary:
        "已拒绝（request_id=approval-denied, source=runtime_action_required），不能作为成功交付证据。",
      verificationSummary: {
        artifactValidator: {
          applicable: true,
          recordCount: 1,
          issueCount: 2,
          repairedCount: 1,
          fallbackUsedCount: 0,
          outcome: "blocking_failure",
        },
        focusVerificationFailureOutcomes: [
          "Artifact 校验存在 2 条未恢复 issues。",
        ],
        focusVerificationRecoveredOutcomes: [
          "Artifact 校验已恢复 1 个产物，fallback 0 次。",
        ],
        requestedFixExecutionResults: [
          {
            requestedFix:
              "复查 Artifact 校验相关产物，确认 issues / repaired / fallback 状态与最终结论一致。",
            requestedFixIndex: 2,
            executionStatus: "completed",
            regressionOutcome: "recovered",
            summaryPreview: "已复查并重新导出 evidence pack。",
            resultRef:
              "agent-runtime://session/session-runtime-4b/thread/thread-runtime-4b/turn/turn-review/item/item-fix-2",
            artifactPaths: [
              ".lime/harness/sessions/session-runtime-4b/evidence/runtime.json",
            ],
          },
        ],
      },
      decision: {
        decisionStatus: "pending_review",
        decisionSummary: "",
        chosenFixStrategy: "",
        riskLevel: "unknown",
        riskTags: [],
        humanReviewer: "",
        reviewedAt: null,
        followupActions: [
          "先对照 analysis-context.json / evidence/runtime.json 核对当前验证失败焦点，再决定是继续修复还是补证据。",
          "复查 Artifact 校验相关产物，确认 issues / repaired / fallback 状态与最终结论一致。",
        ],
        regressionRequirements: [
          "按 replay case 复现问题并确认修复后行为与预期一致。",
          "重新导出 evidence pack，确认 Artifact 校验摘要已更新。",
        ],
        notes: "",
      },
      decisionStatusOptions: [
        "accepted",
        "deferred",
        "rejected",
        "needs_more_evidence",
        "pending_review",
      ],
      riskLevelOptions: ["low", "medium", "high", "unknown"],
      reviewChecklist: ["先阅读 analysis-brief.md"],
      analysisArtifacts: [
        {
          kind: "analysis_brief",
          title: "外部分析简报",
          relativePath:
            ".lime/harness/sessions/session-runtime-4b/analysis/analysis-brief.md",
          absolutePath:
            "/tmp/workspace-4b/.lime/harness/sessions/session-runtime-4b/analysis/analysis-brief.md",
          bytes: 320,
        },
      ],
      artifacts: [
        {
          kind: "review_decision_json",
          title: "人工审核记录 JSON",
          relativePath:
            ".lime/harness/sessions/session-runtime-4b/review/review-decision.json",
          absolutePath:
            "/tmp/workspace-4b/.lime/harness/sessions/session-runtime-4b/review/review-decision.json",
          bytes: 256,
        },
      ],
    });

    await expect(
      exportAgentRuntimeReviewDecisionTemplate("session-runtime-4b"),
    ).resolves.toMatchObject({
      session_id: "session-runtime-4b",
      default_decision_status: "pending_review",
      limit_status: "user_locked_capability_gap",
      capability_gap: "browser_reasoning_candidate_missing",
      user_locked_capability_summary:
        "显式用户模型锁定不满足当前 execution profile（capabilityGap=browser_reasoning_candidate_missing），不能作为成功交付证据。",
      permission_status: "requires_confirmation",
      permission_confirmation_status: "denied",
      permission_confirmation_request_id: "approval-denied",
      permission_confirmation_source: "runtime_action_required",
      permission_confirmation_summary:
        "已拒绝（request_id=approval-denied, source=runtime_action_required），不能作为成功交付证据。",
      verification_summary: expect.objectContaining({
        artifact_validator: expect.objectContaining({
          outcome: "blocking_failure",
          issue_count: 2,
        }),
        focus_verification_failure_outcomes: [
          "Artifact 校验存在 2 条未恢复 issues。",
        ],
        focus_verification_recovered_outcomes: [
          "Artifact 校验已恢复 1 个产物，fallback 0 次。",
        ],
        requested_fix_execution_results: [
          expect.objectContaining({
            requested_fix_index: 2,
            execution_status: "completed",
            regression_outcome: "recovered",
            summary_preview: "已复查并重新导出 evidence pack。",
            result_ref:
              "agent-runtime://session/session-runtime-4b/thread/thread-runtime-4b/turn/turn-review/item/item-fix-2",
            artifact_paths: [
              ".lime/harness/sessions/session-runtime-4b/evidence/runtime.json",
            ],
          }),
        ],
      }),
      decision: expect.objectContaining({
        decision_status: "pending_review",
        risk_level: "unknown",
        followup_actions: [
          "先对照 analysis-context.json / evidence/runtime.json 核对当前验证失败焦点，再决定是继续修复还是补证据。",
          "复查 Artifact 校验相关产物，确认 issues / repaired / fallback 状态与最终结论一致。",
        ],
        regression_requirements: [
          "按 replay case 复现问题并确认修复后行为与预期一致。",
          "重新导出 evidence pack，确认 Artifact 校验摘要已更新。",
        ],
      }),
      decision_status_options: expect.arrayContaining(["accepted"]),
      risk_level_options: expect.arrayContaining(["medium"]),
      review_checklist: ["先阅读 analysis-brief.md"],
      analysis_artifacts: [
        expect.objectContaining({
          kind: "analysis_brief",
          relative_path:
            ".lime/harness/sessions/session-runtime-4b/analysis/analysis-brief.md",
        }),
      ],
      artifacts: [
        expect.objectContaining({
          kind: "review_decision_json",
          relative_path:
            ".lime/harness/sessions/session-runtime-4b/review/review-decision.json",
        }),
      ],
    });

    expectAppServerRequest(
      1,
      APP_SERVER_METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT,
      {
        sessionId: "session-runtime-4b",
      },
    );
  });

  it("saveAgentRuntimeReviewDecision 应经 App Server 保存并归一化返回结构", async () => {
    mockAppServerResponse({
      sessionId: "session-runtime-4c",
      threadId: "thread-runtime-4c",
      workspaceRoot: "/tmp/workspace-4c",
      reviewRelativeRoot: ".lime/harness/sessions/session-runtime-4c/review",
      reviewAbsoluteRoot:
        "/tmp/workspace-4c/.lime/harness/sessions/session-runtime-4c/review",
      analysisRelativeRoot:
        ".lime/harness/sessions/session-runtime-4c/analysis",
      analysisAbsoluteRoot:
        "/tmp/workspace-4c/.lime/harness/sessions/session-runtime-4c/analysis",
      handoffBundleRelativeRoot: ".lime/harness/sessions/session-runtime-4c",
      evidencePackRelativeRoot:
        ".lime/harness/sessions/session-runtime-4c/evidence",
      replayCaseRelativeRoot:
        ".lime/harness/sessions/session-runtime-4c/replay",
      exportedAt: "2026-03-27T10:25:00Z",
      title: "保存人工审核结论",
      threadStatus: "waiting_request",
      latestTurnStatus: "action_required",
      pendingRequestCount: 1,
      queuedTurnCount: 0,
      defaultDecisionStatus: "pending_review",
      limit_status: "normal",
      capability_gap: "",
      user_locked_capability_summary: "",
      permission_status: "requires_confirmation",
      permission_confirmation_status: "resolved",
      permission_confirmation_request_id: "approval-resolved",
      permission_confirmation_source: "runtime_action_required",
      permission_confirmation_summary:
        "已通过（request_id=approval-resolved, source=runtime_action_required）。",
      verificationSummary: {
        artifactValidator: {
          applicable: true,
          recordCount: 1,
          issueCount: 0,
          repairedCount: 1,
          fallbackUsedCount: 0,
          outcome: "recovered",
        },
        focusVerificationFailureOutcomes: [],
        focusVerificationRecoveredOutcomes: [
          "Artifact 校验已恢复 1 个产物，fallback 0 次。",
        ],
      },
      decision: {
        decisionStatus: "accepted",
        decisionSummary: "确认最小修复可接受。",
        chosenFixStrategy: "先收口 runtime 命令，再补 UI 回归。",
        riskLevel: "medium",
        riskTags: ["runtime", "ui"],
        humanReviewer: "Lime Maintainer",
        reviewedAt: "2026-03-27T10:25:00Z",
        followupActions: ["补充 HarnessStatusPanel 测试"],
        regressionRequirements: ["npm run test:contracts"],
        notes: "保持 review decision 主链单一。",
      },
      decisionStatusOptions: [
        "accepted",
        "deferred",
        "rejected",
        "needs_more_evidence",
        "pending_review",
      ],
      riskLevelOptions: ["low", "medium", "high", "unknown"],
      reviewChecklist: ["先阅读 analysis-brief.md"],
      analysisArtifacts: [
        {
          kind: "analysis_brief",
          title: "外部分析简报",
          relativePath:
            ".lime/harness/sessions/session-runtime-4c/analysis/analysis-brief.md",
          absolutePath:
            "/tmp/workspace-4c/.lime/harness/sessions/session-runtime-4c/analysis/analysis-brief.md",
          bytes: 320,
        },
      ],
      artifacts: [
        {
          kind: "review_decision_markdown",
          title: "人工审核记录",
          relativePath:
            ".lime/harness/sessions/session-runtime-4c/review/review-decision.md",
          absolutePath:
            "/tmp/workspace-4c/.lime/harness/sessions/session-runtime-4c/review/review-decision.md",
          bytes: 512,
        },
      ],
    });

    await expect(
      saveAgentRuntimeReviewDecision({
        session_id: "session-runtime-4c",
        decision_status: "accepted",
        decision_summary: "确认最小修复可接受。",
        chosen_fix_strategy: "先收口 runtime 命令，再补 UI 回归。",
        risk_level: "medium",
        risk_tags: ["runtime", "ui"],
        human_reviewer: "Lime Maintainer",
        reviewed_at: "2026-03-27T10:25:00Z",
        followup_actions: ["补充 HarnessStatusPanel 测试"],
        regression_requirements: ["npm run test:contracts"],
        notes: "保持 review decision 主链单一。",
      }),
    ).resolves.toMatchObject({
      session_id: "session-runtime-4c",
      permission_status: "requires_confirmation",
      limit_status: "normal",
      permission_confirmation_status: "resolved",
      permission_confirmation_request_id: "approval-resolved",
      permission_confirmation_source: "runtime_action_required",
      verification_summary: expect.objectContaining({
        artifact_validator: expect.objectContaining({
          outcome: "recovered",
          repaired_count: 1,
        }),
      }),
      decision: expect.objectContaining({
        decision_status: "accepted",
        risk_level: "medium",
        risk_tags: ["runtime", "ui"],
      }),
      artifacts: [
        expect.objectContaining({
          kind: "review_decision_markdown",
        }),
      ],
    });

    expectAppServerRequest(
      1,
      APP_SERVER_METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE,
      {
        sessionId: "session-runtime-4c",
        decisionStatus: "accepted",
        decisionSummary: "确认最小修复可接受。",
        chosenFixStrategy: "先收口 runtime 命令，再补 UI 回归。",
        riskLevel: "medium",
        riskTags: ["runtime", "ui"],
        humanReviewer: "Lime Maintainer",
        followupActions: ["补充 HarnessStatusPanel 测试"],
        regressionRequirements: ["npm run test:contracts"],
        notes: "保持 review decision 主链单一。",
      },
    );
  });

  it("saveAgentRuntimeReviewDecision 应透传 denied 权限确认阻止 accepted 的后端错误", async () => {
    mockAppServerError(
      "真实权限确认已被拒绝，不能把本次 review decision 保存为 accepted；请先处理真实权限确认，或改为 rejected / deferred / needs_more_evidence。",
    );

    await expect(
      saveAgentRuntimeReviewDecision({
        session_id: "session-runtime-4d",
        decision_status: "accepted",
        decision_summary: "错误接受被拒绝的权限确认。",
        chosen_fix_strategy: "直接接受。",
        risk_level: "low",
        risk_tags: ["permission"],
        human_reviewer: "Lime Maintainer",
        reviewed_at: undefined,
        followup_actions: [],
        regression_requirements: [],
        notes: "",
      }),
    ).rejects.toThrow("真实权限确认已被拒绝");

    expectAppServerRequest(
      1,
      APP_SERVER_METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE,
      {
        sessionId: "session-runtime-4d",
        decisionStatus: "accepted",
        decisionSummary: "错误接受被拒绝的权限确认。",
        chosenFixStrategy: "直接接受。",
        riskLevel: "low",
        riskTags: ["permission"],
        humanReviewer: "Lime Maintainer",
        followupActions: [],
        regressionRequirements: [],
        notes: "",
      },
    );
  });

  it("saveAgentRuntimeReviewDecision 应透传用户锁定能力缺口阻止 accepted 的后端错误", async () => {
    mockAppServerError(
      "显式用户模型锁定不满足当前 execution profile（capabilityGap=browser_reasoning_candidate_missing），不能把本次 review decision 保存为 accepted；请切换到满足 routingSlot 的模型或取消显式模型锁定并重新导出证据，或改为 rejected / deferred / needs_more_evidence。",
    );

    await expect(
      saveAgentRuntimeReviewDecision({
        session_id: "session-runtime-4e",
        decision_status: "accepted",
        decision_summary: "错误接受模型锁定能力缺口。",
        chosen_fix_strategy: "直接接受。",
        risk_level: "low",
        risk_tags: ["model-routing"],
        human_reviewer: "Lime Maintainer",
        reviewed_at: undefined,
        followup_actions: [],
        regression_requirements: [],
        notes: "",
      }),
    ).rejects.toThrow("显式用户模型锁定");

    expectAppServerRequest(
      1,
      APP_SERVER_METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE,
      {
        sessionId: "session-runtime-4e",
        decisionStatus: "accepted",
        decisionSummary: "错误接受模型锁定能力缺口。",
        chosenFixStrategy: "直接接受。",
        riskLevel: "low",
        riskTags: ["model-routing"],
        humanReviewer: "Lime Maintainer",
        followupActions: [],
        regressionRequirements: [],
        notes: "",
      },
    );
  });

  it("getAgentRuntimeToolInventory 应走统一 runtime inventory 命令", async () => {
    mockAppServerResponse({
      inventory: {
        request: {
          caller: "assistant",
          surface: {
            workbench: true,
            browser_assist: true,
          },
        },
        agent_initialized: true,
        warnings: [],
        mcp_servers: ["docs"],
        default_allowed_tools: ["ToolSearch"],
        counts: {
          catalog_total: 1,
          catalog_current_total: 1,
          catalog_compat_total: 0,
          catalog_deprecated_total: 0,
          default_allowed_total: 1,
          native_total: 1,
          native_visible_total: 1,
          native_catalog_unmapped_total: 0,
          extension_surface_total: 1,
          extension_mcp_bridge_total: 1,
          extension_runtime_total: 0,
          extension_tool_total: 1,
          extension_tool_visible_total: 1,
          mcp_server_total: 1,
          mcp_tool_total: 1,
          mcp_tool_visible_total: 1,
        },
        catalog_tools: [
          {
            name: "bash",
            profiles: ["core"],
            capabilities: ["execution"],
            lifecycle: "current",
            source: "agent_builtin",
            permission_plane: "parameter_restricted",
            workspace_default_allow: false,
            execution_warning_policy: "shell_command_risk",
            execution_warning_policy_source: "default",
            execution_restriction_profile: "workspace_shell_command",
            execution_restriction_profile_source: "runtime",
            execution_sandbox_profile: "workspace_command",
            execution_sandbox_profile_source: "persisted",
          },
        ],
        native_tools: [
          {
            name: "bash",
            description: "workspace bash",
            catalog_entry_name: "bash",
            catalog_source: "agent_builtin",
            catalog_lifecycle: "current",
            catalog_permission_plane: "parameter_restricted",
            catalog_workspace_default_allow: false,
            catalog_execution_warning_policy: "shell_command_risk",
            catalog_execution_warning_policy_source: "default",
            catalog_execution_restriction_profile: "workspace_shell_command",
            catalog_execution_restriction_profile_source: "runtime",
            catalog_execution_sandbox_profile: "workspace_command",
            catalog_execution_sandbox_profile_source: "persisted",
            deferred_loading: false,
            always_visible: true,
            allowed_callers: ["assistant"],
            tags: [],
            input_examples_count: 0,
            has_output_schema: false,
            caller_allowed: true,
            visible_in_context: true,
          },
        ],
        extension_surfaces: [],
        extension_tools: [],
        mcp_tools: [],
      },
    });

    await expect(
      getAgentRuntimeToolInventory({
        workbench: true,
        browserAssist: true,
        caller: "assistant",
      }),
    ).resolves.toMatchObject({
      request: {
        caller: "assistant",
        surface: {
          workbench: true,
          browser_assist: true,
        },
      },
      counts: {
        catalog_total: 1,
      },
      catalog_tools: [
        expect.objectContaining({
          execution_warning_policy_source: "default",
          execution_restriction_profile_source: "runtime",
          execution_sandbox_profile_source: "persisted",
        }),
      ],
    });

    expectAppServerRequest(
      1,
      APP_SERVER_METHOD_AGENT_SESSION_TOOL_INVENTORY_READ,
      {
        workbench: true,
        browserAssist: true,
        caller: "assistant",
      },
    );
  });

  it("getAgentRuntimeToolInventory 应透传 metadata 以计算 effective policy", async () => {
    mockAppServerResponse({
      inventory: {
        request: {
          caller: "assistant",
          surface: {
            workbench: false,
            browser_assist: false,
          },
        },
        agent_initialized: true,
        warnings: [],
        mcp_servers: [],
        default_allowed_tools: [],
        counts: {
          catalog_total: 0,
          catalog_current_total: 0,
          catalog_compat_total: 0,
          catalog_deprecated_total: 0,
          default_allowed_total: 0,
          native_total: 0,
          native_visible_total: 0,
          native_catalog_unmapped_total: 0,
          extension_surface_total: 0,
          extension_mcp_bridge_total: 0,
          extension_runtime_total: 0,
          extension_tool_total: 0,
          extension_tool_visible_total: 0,
          mcp_server_total: 0,
          mcp_tool_total: 0,
          mcp_tool_visible_total: 0,
        },
        catalog_tools: [],
        native_tools: [],
        extension_surfaces: [],
        extension_tools: [],
        mcp_tools: [],
      },
    });

    await getAgentRuntimeToolInventory({
      caller: "assistant",
      metadata: {
        harness: {
          executionPolicy: {
            toolOverrides: {
              bash: {
                warningPolicy: "none",
              },
            },
          },
        },
      },
    });

    expectAppServerRequest(
      1,
      APP_SERVER_METHOD_AGENT_SESSION_TOOL_INVENTORY_READ,
      {
        caller: "assistant",
        metadata: {
          harness: {
            executionPolicy: {
              toolOverrides: {
                bash: {
                  warningPolicy: "none",
                },
              },
            },
          },
        },
      },
    );
  });

  it("getAgentRuntimeToolInventory 默认请求应传空对象", async () => {
    mockAppServerResponse({
      inventory: {
        request: {
          caller: "assistant",
          surface: {
            workbench: false,
            browser_assist: false,
          },
        },
        agent_initialized: false,
        warnings: [],
        mcp_servers: [],
        default_allowed_tools: [],
        counts: {
          catalog_total: 0,
          catalog_current_total: 0,
          catalog_compat_total: 0,
          catalog_deprecated_total: 0,
          default_allowed_total: 0,
          native_total: 0,
          native_visible_total: 0,
          native_catalog_unmapped_total: 0,
          extension_surface_total: 0,
          extension_mcp_bridge_total: 0,
          extension_runtime_total: 0,
          extension_tool_total: 0,
          extension_tool_visible_total: 0,
          mcp_server_total: 0,
          mcp_tool_total: 0,
          mcp_tool_visible_total: 0,
        },
        catalog_tools: [],
        native_tools: [],
        extension_surfaces: [],
        extension_tools: [],
        mcp_tools: [],
      },
    });

    await getAgentRuntimeToolInventory();

    expectAppServerRequest(
      1,
      APP_SERVER_METHOD_AGENT_SESSION_TOOL_INVENTORY_READ,
      {},
    );
  });

  it("deleteAgentRuntimeSession 应走 current 边界，标题生成只做本地投影", async () => {
    mockAppServerResponse({
      data: [
        {
          cliVersion: "0.1.0",
          createdAt: 1780704000,
          cwd: "/tmp/workspace-1",
          ephemeral: false,
          id: "thread-runtime-3",
          modelProvider: "openai-compatible",
          preview: "Runtime Session 3",
          sessionId: "session-runtime-3",
          source: "appServer",
          status: { type: "idle" },
          turns: [],
          updatedAt: 1780704000,
        },
      ],
    });
    mockAppServerResponse({});
    await deleteAgentRuntimeSession("session-runtime-3");
    await expect(
      generateAgentRuntimeSessionTitle(
        "session-runtime-3",
        "user：新的智能标题\nassistant：正在整理",
      ),
    ).resolves.toBe("新的智能标题");

    expectAppServerRequest(1, "thread/list", {
      archived: false,
      limit: 100,
    });
    expectAppServerRequest(2, APP_SERVER_METHOD_THREAD_DELETE, {
      threadId: "thread-runtime-3",
    });
    expect(mockSafeInvoke).toHaveBeenCalledTimes(2);
  });

  it("generateAgentRuntimeTitle 应从图片任务预览文本生成本地标题", async () => {
    await expect(
      generateAgentRuntimeTitle({
        previewText: "赛博朋克风城市夜景主视觉",
        titleKind: "image_task",
      }),
    ).resolves.toBe("赛博朋克风城市夜景主视觉");

    expect(mockSafeInvoke).not.toHaveBeenCalled();
  });

  it("generateAgentRuntimeTitleResult 应返回本地 fallback 诊断且不调用旧标题命令", async () => {
    const result = await generateAgentRuntimeTitleResult({
      sessionId: "session-runtime-3",
      previewText:
        "user：整理今天的国际新闻，按地区归类并给出可执行摘要\nassistant：好的",
      titleKind: "session",
    });

    expect(result).toEqual({
      title: "整理今天的国际新闻，按地区归类并给出可执行摘要",
      sessionId: "session-runtime-3",
      executionRuntime: null,
      usedFallback: true,
      fallbackReason: "local_preview_title",
    });
    expect(mockSafeInvoke).not.toHaveBeenCalled();
  });

  it("generateAgentRuntimeTitleResult 应清理 Markdown 与角色前缀", async () => {
    await expect(
      generateAgentRuntimeTitleResult({
        previewText: "user：# `城市夜景主视觉`",
        titleKind: "image_task",
      }),
    ).resolves.toEqual({
      title: "城市夜景主视觉",
      sessionId: null,
      executionRuntime: null,
      usedFallback: true,
      fallbackReason: "local_preview_title",
    });
    expect(mockSafeInvoke).not.toHaveBeenCalled();
  });
});
