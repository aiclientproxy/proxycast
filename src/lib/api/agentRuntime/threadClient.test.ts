import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  APP_SERVER_METHOD_AGENT_SESSION_EVENT,
  AppServerRpcError,
  type AppServerJsonRpcNotification,
  type AppServerRequestResult,
} from "@/lib/api/appServer";
import type { TurnStartParams } from "@limecloud/app-server-client";
import { isAppServerBridgeAvailable } from "@/lib/api/appServerBridgeAvailability";
import { resetDefaultAppServerEventBusForTests } from "@/lib/api/appServerEventBus";
import {
  AGENT_RUNTIME_RENDERER_EVENT_NAME_CONTEXT_KEY,
  createApplicationAdditionalContext,
} from "@/lib/api/agentProtocolOps";
import { safeListen } from "@/lib/dev-bridge";
import { listenAgentRuntimeEvent } from "../agentRuntimeEvents";
import { resetAgentRuntimeEventSequenceGatesForTests } from "./eventSequenceGate";
import { projectAppServerAgentEventPayload } from "./appServerEventStream";
import {
  createThreadClient,
  type AgentRuntimeAppServerClient,
  type AgentRuntimeLifecycleClient,
} from "./threadClient";
import type { AgentRuntimeCommandInvoke } from "./transport";
import type {
  AgentRuntimeFileCheckpointDetail,
  AgentRuntimeFileCheckpointDiffResult,
  AgentRuntimeFileCheckpointListResult,
  AgentRuntimeFileCheckpointRestoreResult,
  AgentRuntimeFileCheckpointSummary,
} from "./sessionTypes";

vi.mock("@/lib/dev-bridge", () => ({
  safeInvoke: vi.fn(),
  safeListen: vi.fn(),
}));

vi.mock("@/lib/api/appServerBridgeAvailability", () => ({
  isAppServerBridgeAvailable: vi.fn(),
}));

function turnStartParams(options: {
  threadId?: string;
  text?: string;
  eventName: string;
  model?: TurnStartParams["model"];
  effort?: TurnStartParams["effort"];
  approvalPolicy?: TurnStartParams["approvalPolicy"];
  sandboxPolicy?: TurnStartParams["sandboxPolicy"];
  metadata?: Record<string, unknown>;
  attachments?: Array<{
    kind?: string;
    uri: string;
    metadata?: Record<string, unknown>;
  }>;
  outputSchema?: TurnStartParams["outputSchema"];
}): TurnStartParams {
  const {
    threadId = "session-1",
    text = "生成草稿",
    model,
    effort,
    approvalPolicy,
    sandboxPolicy,
    metadata,
    attachments,
    outputSchema,
  } = options;
  const additionalContext = createApplicationAdditionalContext({
    [AGENT_RUNTIME_RENDERER_EVENT_NAME_CONTEXT_KEY]: options.eventName,
    metadata,
  });
  return {
    threadId,
    input: [
      { type: "text", text },
      ...(attachments ?? []).map((attachment) => ({
        type: "image" as const,
        url: attachment.uri,
      })),
    ],
    ...(model !== undefined ? { model } : {}),
    ...(effort !== undefined ? { effort } : {}),
    ...(approvalPolicy !== undefined ? { approvalPolicy } : {}),
    ...(sandboxPolicy !== undefined ? { sandboxPolicy } : {}),
    ...(Object.keys(additionalContext).length > 0 ? { additionalContext } : {}),
    ...(outputSchema !== undefined ? { outputSchema } : {}),
  };
}

function appServerClientMock(): AgentRuntimeAppServerClient {
  return {
    readThread: vi.fn().mockResolvedValue({
      id: 2,
      result: {
        thread: {
          createdAt: 0.1,
          id: "thread-1",
          sessionId: "session-1",
          status: { type: "active", activeFlags: [] },
          turns: [],
          updatedAt: 0.2,
        },
      },
      response: { id: 2, result: {} },
      messages: [],
      notifications: [],
    }),
    setThreadName: vi.fn().mockResolvedValue({
      id: 6,
      result: {},
      response: { id: 6, result: {} },
      messages: [],
      notifications: [],
    }),
    updateThreadSettings: vi.fn().mockResolvedValue({
      id: 5,
      result: {},
      response: { id: 5, result: {} },
      messages: [],
      notifications: [],
    }),
    runThreadShellCommand: vi.fn().mockResolvedValue({
      id: 4,
      result: {},
      response: { id: 4, result: {} },
      messages: [],
      notifications: [],
    }),
    startReview: vi.fn().mockResolvedValue({
      id: 7,
      result: {
        reviewThreadId: "thread-1",
        turn: {
          id: "turn-review",
          items: [],
          itemsView: "notLoaded",
          status: "inProgress",
        },
      },
      response: { id: 7, result: {} },
      messages: [],
      notifications: [],
    }),
    startTurn: vi.fn().mockResolvedValue({}),
    steerTurn: vi.fn().mockResolvedValue({
      id: 3,
      result: { turnId: "turn-1" },
      response: { id: 3, result: { turnId: "turn-1" } },
      messages: [],
      notifications: [],
    }),
    cancelTurn: vi.fn().mockResolvedValue({}),
    startThreadCompaction: vi.fn().mockResolvedValue({
      id: 1,
      result: {},
      response: {
        id: 1,
        result: {},
      },
      messages: [],
      notifications: [],
    }),
    resumeThread: vi.fn().mockResolvedValue({
      id: 1,
      result: {
        thread: { id: "thread-1", sessionId: "session-1", turns: [] },
        model: "gpt-5.4",
        modelProvider: "openai",
        cwd: "/tmp/workspace",
      },
      response: {
        id: 1,
        result: {},
      },
      messages: [],
      notifications: [],
    }),
    drainEvents: vi.fn().mockResolvedValue([]),
    listAgentSessionFileCheckpoints: vi.fn().mockResolvedValue({
      id: 1,
      result: appServerCheckpointList,
      response: { id: 1, result: appServerCheckpointList },
      messages: [],
      notifications: [],
    }),
    getAgentSessionFileCheckpoint: vi.fn().mockResolvedValue({
      id: 1,
      result: appServerCheckpointDetail,
      response: { id: 1, result: appServerCheckpointDetail },
      messages: [],
      notifications: [],
    }),
    diffAgentSessionFileCheckpoint: vi.fn().mockResolvedValue({
      id: 1,
      result: appServerCheckpointDiff,
      response: { id: 1, result: appServerCheckpointDiff },
      messages: [],
      notifications: [],
    }),
    restoreAgentSessionFileCheckpoint: vi.fn().mockResolvedValue({
      id: 1,
      result: appServerCheckpointRestore,
      response: { id: 1, result: appServerCheckpointRestore },
      messages: [],
      notifications: [],
    }),
    listCapabilities: vi.fn().mockResolvedValue({
      id: 1,
      result: {
        capabilities: [
          {
            id: "agent.session",
            title: "Agent Session",
            methods: ["thread/start", "turn/start"],
          },
        ],
        runtimeCapabilityManifest: {
          schemaVersion: "lime-runtime-capability-manifest/v0.1",
          runtimeId: "app-server",
          sessionId: "session-1",
          generatedAt: "2026-06-12T00:00:00.000Z",
          capabilities: [
            {
              id: "transport.jsonrpc",
              status: "supported",
              scope: "runtime",
              title: "Agent Session",
            },
          ],
        },
      },
      response: { id: 1, result: {} },
      messages: [],
      notifications: [],
    }),
  };
}

function standardRuntimeClientMock(): AgentRuntimeLifecycleClient {
  return {
    startTurn: vi.fn().mockResolvedValue({
      id: 1,
      result: {
        turn: {
          turnId: "turn-1",
          sessionId: "session-1",
          threadId: "thread-1",
          status: "accepted",
        },
      },
      response: {
        id: 1,
        result: {
          turn: {
            turnId: "turn-1",
            sessionId: "session-1",
            threadId: "thread-1",
            status: "accepted",
          },
        },
      },
      messages: [],
      notifications: [],
    }),
    steerTurn: vi.fn().mockResolvedValue({
      id: 2,
      result: { turnId: "turn-1" },
      response: { id: 2, result: { turnId: "turn-1" } },
      messages: [],
      notifications: [],
    }),
    cancelTurn: vi.fn().mockResolvedValue({
      id: 2,
      result: {},
      response: { id: 2, result: {} },
      messages: [],
      notifications: [],
    }),
    readThread: vi.fn().mockResolvedValue({
      id: 4,
      result: {
        session: {
          sessionId: "session-1",
          threadId: "thread-1",
          appId: "agent-chat",
          status: "idle",
          createdAt: "2026-06-06T00:00:00.000Z",
          updatedAt: "2026-06-06T00:00:00.000Z",
        },
        turns: [],
      },
      response: {
        id: 4,
        result: {
          session: {
            sessionId: "session-1",
            threadId: "thread-1",
            appId: "agent-chat",
            status: "idle",
            createdAt: "2026-06-06T00:00:00.000Z",
            updatedAt: "2026-06-06T00:00:00.000Z",
          },
          turns: [],
        },
      },
      messages: [],
      notifications: [],
    }),
  } as AgentRuntimeLifecycleClient;
}

const appServerCheckpointSummary = {
  checkpointId: "checkpoint-1",
  turnId: "turn-1",
  path: "src/App.tsx",
  source: "tool_result",
  updatedAt: "2026-06-06T00:00:00.000Z",
  validationIssueCount: 0,
};

const appServerCheckpointList = {
  sessionId: "session-1",
  threadId: "thread-1",
  checkpointCount: 1,
  checkpoints: [appServerCheckpointSummary],
};

const appServerCheckpointDetail = {
  sessionId: "session-1",
  threadId: "thread-1",
  checkpoint: appServerCheckpointSummary,
  livePath: "/workspace/src/App.tsx",
  snapshotPath: "/workspace/.lime/checkpoints/checkpoint-1/App.tsx",
  versionHistory: [],
  validationIssues: [],
  content: "export default function App() {}",
};

const appServerCheckpointDiff = {
  sessionId: "session-1",
  threadId: "thread-1",
  checkpoint: appServerCheckpointSummary,
  currentVersionId: "version-current",
  previousVersionId: "version-previous",
  diff: [],
};

const appServerCheckpointRestore = {
  sessionId: "session-1",
  threadId: "thread-1",
  checkpoint: appServerCheckpointSummary,
  livePath: "/workspace/src/App.tsx",
  snapshotPath: "/workspace/.lime/checkpoints/checkpoint-1/App.tsx",
  backupPath: null,
  restoredAt: "2026-06-06T00:01:00.000Z",
};

const checkpointSummary: AgentRuntimeFileCheckpointSummary = {
  checkpoint_id: "checkpoint-1",
  turn_id: "turn-1",
  path: "src/App.tsx",
  source: "tool_result",
  updated_at: "2026-06-06T00:00:00.000Z",
  validation_issue_count: 0,
};

const checkpointList: AgentRuntimeFileCheckpointListResult = {
  session_id: "session-1",
  thread_id: "thread-1",
  checkpoint_count: 1,
  checkpoints: [checkpointSummary],
};

const checkpointDetail: AgentRuntimeFileCheckpointDetail = {
  session_id: "session-1",
  thread_id: "thread-1",
  checkpoint: checkpointSummary,
  live_path: "/workspace/src/App.tsx",
  snapshot_path: "/workspace/.lime/checkpoints/checkpoint-1/App.tsx",
  version_history: [],
  validation_issues: [],
  content: "export default function App() {}",
};

const checkpointDiff: AgentRuntimeFileCheckpointDiffResult = {
  session_id: "session-1",
  thread_id: "thread-1",
  checkpoint: checkpointSummary,
  current_version_id: "version-current",
  previous_version_id: "version-previous",
  diff: [],
};

const checkpointRestore: AgentRuntimeFileCheckpointRestoreResult = {
  session_id: "session-1",
  thread_id: "thread-1",
  checkpoint: checkpointSummary,
  live_path: "/workspace/src/App.tsx",
  snapshot_path: "/workspace/.lime/checkpoints/checkpoint-1/App.tsx",
  backup_path: null,
  restored_at: "2026-06-06T00:01:00.000Z",
};

function malformedAppServerResult<T>(
  result: unknown,
): AppServerRequestResult<T> {
  return {
    id: 1,
    result: result as T,
    response: {
      id: 1,
      result: result as T,
    },
    messages: [],
    notifications: [],
  };
}

describe("agentRuntime threadClient", () => {
  beforeEach(() => {
    resetDefaultAppServerEventBusForTests();
    vi.clearAllMocks();
    resetAgentRuntimeEventSequenceGatesForTests();
    vi.mocked(isAppServerBridgeAvailable).mockReturnValue(false);
    vi.mocked(safeListen).mockResolvedValue(vi.fn());
  });

  afterEach(() => {
    resetDefaultAppServerEventBusForTests();
    vi.useRealTimers();
  });

  it("用户 shell 命令应规范化 identity 并走 typed thread gateway", async () => {
    const appServerClient = appServerClientMock();
    const client = createThreadClient({ appServerClient });

    await expect(
      client.runUserShellCommand(
        { threadId: " thread-1 ", command: " printf ready " },
        " agentSession/event/session-1 ",
      ),
    ).resolves.toBeUndefined();

    expect(appServerClient.runThreadShellCommand).toHaveBeenCalledWith({
      threadId: "thread-1",
      command: "printf ready",
    });
  });

  it("用户 shell 命令缺少 command 时应在 gateway fail closed", async () => {
    const appServerClient = appServerClientMock();
    const client = createThreadClient({ appServerClient });

    await expect(
      client.runUserShellCommand(
        { threadId: "thread-1", command: "   " },
        "agentSession/event/session-1",
      ),
    ).rejects.toThrow("command is required");
    expect(appServerClient.runThreadShellCommand).not.toHaveBeenCalled();
  });

  it("replay request 无当前 typed server-request 时应 fail closed 且不调用旧 action/replay", async () => {
    const invokeCommand = vi.fn() as unknown as AgentRuntimeCommandInvoke;
    const appServerClient = appServerClientMock();
    const client = createThreadClient({
      appServerClient,
      invokeCommand,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    await expect(
      client.replayAgentRuntimeRequest({
        session_id: "session-1",
        request_id: "request-1",
      }),
    ).resolves.toBeNull();
    expect(invokeCommand).not.toHaveBeenCalled();
  });

  it("session control 应走 App Server current methods 且不调用 legacy command gateway", async () => {
    const invokeCommand = vi.fn() as unknown as AgentRuntimeCommandInvoke;
    const appServerClient = appServerClientMock();
    const client = createThreadClient({
      appServerClient,
      invokeCommand,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    await expect(
      client.compactAgentRuntimeSession({
        session_id: "session-1",
        event_name: "agentSession/event/session-1",
      }),
    ).resolves.toBeUndefined();
    await expect(
      client.resumeThread({ threadId: "thread-1" }),
    ).resolves.toMatchObject({
      result: {
        thread: { id: "thread-1" },
      },
    });
    expect(appServerClient.startThreadCompaction).toHaveBeenCalledWith({
      threadId: "session-1",
    });
    expect(appServerClient.resumeThread).toHaveBeenCalledWith({
      threadId: "thread-1",
      excludeTurns: true,
    });
    expect(invokeCommand).not.toHaveBeenCalled();
  });

  it("capability manifest 应消费 App Server current capability/list 合同", async () => {
    const appServerClient = appServerClientMock();
    const client = createThreadClient({
      appServerClient,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    await expect(
      client.getAgentRuntimeCapabilityManifest({
        app_id: "agent-chat",
        workspace_id: "workspace-1",
        session_id: "session-1",
        limit: 10,
      }),
    ).resolves.toMatchObject({
      schemaVersion: "lime-runtime-capability-manifest/v0.1",
      runtimeId: "app-server",
      sessionId: "session-1",
      capabilities: [
        {
          id: "transport.jsonrpc",
          status: "supported",
          scope: "runtime",
        },
      ],
    });
    expect(appServerClient.listCapabilities).toHaveBeenCalledWith({
      appId: "agent-chat",
      workspaceId: "workspace-1",
      sessionId: "session-1",
      limit: 10,
    });
  });

  it("replay current 缺少 typed pending request 时应返回 null", async () => {
    const appServerClient = appServerClientMock();
    const invokeCommand = vi.fn() as unknown as AgentRuntimeCommandInvoke;
    const client = createThreadClient({
      appServerClient,
      invokeCommand,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    await expect(
      client.replayAgentRuntimeRequest({
        session_id: "session-1",
        request_id: "request-1",
      }),
    ).resolves.toBeNull();
    expect(invokeCommand).not.toHaveBeenCalled();
  });

  it("file checkpoint 应走 App Server current methods，不复用 legacy command gateway", async () => {
    const appServerClient = appServerClientMock();
    const invokeCommand = vi.fn() as unknown as AgentRuntimeCommandInvoke;
    const client = createThreadClient({
      appServerClient,
      invokeCommand,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    await expect(
      client.listAgentRuntimeFileCheckpoints({
        session_id: "session-1",
      }),
    ).resolves.toEqual(checkpointList);
    await expect(
      client.getAgentRuntimeFileCheckpoint({
        session_id: "session-1",
        checkpoint_id: "checkpoint-1",
      }),
    ).resolves.toEqual(checkpointDetail);
    await expect(
      client.diffAgentRuntimeFileCheckpoint({
        session_id: "session-1",
        checkpoint_id: "checkpoint-1",
      }),
    ).resolves.toEqual(checkpointDiff);
    await expect(
      client.restoreAgentRuntimeFileCheckpoint({
        session_id: "session-1",
        checkpoint_id: "checkpoint-1",
        confirm_restore: true,
      }),
    ).resolves.toEqual(checkpointRestore);

    expect(
      appServerClient.listAgentSessionFileCheckpoints,
    ).toHaveBeenCalledWith({
      sessionId: "session-1",
    });
    expect(appServerClient.getAgentSessionFileCheckpoint).toHaveBeenCalledWith({
      sessionId: "session-1",
      checkpointId: "checkpoint-1",
    });
    expect(appServerClient.diffAgentSessionFileCheckpoint).toHaveBeenCalledWith(
      {
        sessionId: "session-1",
        checkpointId: "checkpoint-1",
      },
    );
    expect(
      appServerClient.restoreAgentSessionFileCheckpoint,
    ).toHaveBeenCalledWith({
      sessionId: "session-1",
      checkpointId: "checkpoint-1",
      confirmRestore: true,
      createBackup: undefined,
    });
    expect(invokeCommand).not.toHaveBeenCalled();
  });

  it("file checkpoint App Server current 收到假成功或缺字段结果时应 fail closed", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(
      appServerClient.listAgentSessionFileCheckpoints,
    ).mockResolvedValueOnce(
      malformedAppServerResult({
        ...appServerCheckpointList,
        checkpoints: [
          { ...appServerCheckpointSummary, validationIssueCount: "0" },
        ],
      }),
    );
    vi.mocked(
      appServerClient.getAgentSessionFileCheckpoint,
    ).mockResolvedValueOnce(
      malformedAppServerResult({
        ...appServerCheckpointDetail,
        checkpoint: { ...appServerCheckpointSummary, checkpointId: "" },
      }),
    );
    vi.mocked(
      appServerClient.diffAgentSessionFileCheckpoint,
    ).mockResolvedValueOnce(
      malformedAppServerResult({
        ...appServerCheckpointDiff,
        threadId: "",
      }),
    );
    vi.mocked(
      appServerClient.restoreAgentSessionFileCheckpoint,
    ).mockResolvedValueOnce(
      malformedAppServerResult({
        ...appServerCheckpointRestore,
        restoredAt: null,
      }),
    );
    const invokeCommand = vi.fn() as unknown as AgentRuntimeCommandInvoke;
    const client = createThreadClient({
      appServerClient,
      invokeCommand,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    await expect(
      client.listAgentRuntimeFileCheckpoints({ session_id: "session-1" }),
    ).rejects.toThrow(
      "agentSession/fileCheckpoint/list did not return file checkpoint list",
    );
    await expect(
      client.getAgentRuntimeFileCheckpoint({
        session_id: "session-1",
        checkpoint_id: "checkpoint-1",
      }),
    ).rejects.toThrow(
      "agentSession/fileCheckpoint/get did not return file checkpoint detail",
    );
    await expect(
      client.diffAgentRuntimeFileCheckpoint({
        session_id: "session-1",
        checkpoint_id: "checkpoint-1",
      }),
    ).rejects.toThrow(
      "agentSession/fileCheckpoint/diff did not return file checkpoint diff",
    );
    await expect(
      client.restoreAgentRuntimeFileCheckpoint({
        session_id: "session-1",
        checkpoint_id: "checkpoint-1",
        confirm_restore: true,
      }),
    ).rejects.toThrow(
      "agentSession/fileCheckpoint/restore did not return file checkpoint restore result",
    );
    expect(invokeCommand).not.toHaveBeenCalled();
  });

  it("App Server 可用时 submit 应进入 turn/start", async () => {
    const appServerClient = appServerClientMock();
    const invokeCommand = vi.fn() as unknown as AgentRuntimeCommandInvoke;
    const client = createThreadClient({
      appServerClient,
      invokeCommand,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    await client.submitAgentRuntimeTurn(
      turnStartParams({
        threadId: "session-1",
        eventName: "agentSession/event/session-1",
        model: "deepseek-v4-flash",
        metadata: { source: "chat" },
        attachments: [
          {
            kind: "image",
            uri: "data:image/png;base64,abc",
            metadata: {
              mediaType: "image/png",
              index: 0,
            },
          },
        ],
      }),
    );

    expect(appServerClient.startTurn).toHaveBeenCalledWith({
      threadId: "session-1",
      input: [
        { type: "text", text: "生成草稿" },
        {
          type: "image",
          url: "data:image/png;base64,abc",
        },
      ],
      model: "deepseek-v4-flash",
      additionalContext: createApplicationAdditionalContext({
        metadata: { source: "chat" },
      }),
    });
    const startTurnParams = appServerClient.startTurn.mock.calls[0]?.[0];
    expect(startTurnParams).not.toHaveProperty("sessionId");
    expect(startTurnParams).not.toHaveProperty("runtimeOptions");
    expect(invokeCommand).not.toHaveBeenCalled();
  });

  it("turn lifecycle 注入标准 runtime client 后应委托 facade，不直接调用 renderer App Server lifecycle", async () => {
    const appServerClient = appServerClientMock();
    const standardRuntimeClient = standardRuntimeClientMock();
    const invokeCommand = vi.fn() as unknown as AgentRuntimeCommandInvoke;
    const client = createThreadClient({
      appServerClient,
      standardRuntimeClient,
      invokeCommand,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    await expect(
      client.submitAgentRuntimeTurn(
        turnStartParams({
          threadId: "session-1",
          eventName: "agentSession/event/session-1",
        }),
      ),
    ).resolves.toBeUndefined();
    await expect(
      client.interruptAgentRuntimeTurn({
        session_id: "session-1",
        turn_id: "turn-1",
      }),
    ).resolves.toBe(true);
    await expect(
      client.steerAgentRuntimeTurn({
        threadId: "session-1",
        expectedTurnId: "turn-1",
        input: [{ type: "text", text: "补充约束" }],
      }),
    ).resolves.toEqual(
      expect.objectContaining({ result: { turnId: "turn-1" } }),
    );
    await expect(
      client.getAgentRuntimeThreadRead(" session-1 "),
    ).resolves.toEqual(
      expect.objectContaining({
        thread_id: "thread-1",
        status: "running",
      }),
    );

    expect(standardRuntimeClient.startTurn).toHaveBeenCalledWith({
      threadId: "session-1",
      input: [{ type: "text", text: "生成草稿" }],
    });
    expect(standardRuntimeClient.steerTurn).toHaveBeenCalledWith({
      threadId: "session-1",
      expectedTurnId: "turn-1",
      input: [{ type: "text", text: "补充约束" }],
    });
    expect(standardRuntimeClient.cancelTurn).toHaveBeenCalledWith({
      threadId: "session-1",
      turnId: "turn-1",
    });
    expect(appServerClient.readThread).toHaveBeenCalledWith({
      threadId: "session-1",
      includeTurns: true,
    });
    expect(appServerClient.startTurn).not.toHaveBeenCalled();
    expect(appServerClient.steerTurn).not.toHaveBeenCalled();
    expect(appServerClient.cancelTurn).not.toHaveBeenCalled();
    expect(standardRuntimeClient.readThread).not.toHaveBeenCalled();
    expect(invokeCommand).not.toHaveBeenCalled();
  });

  it("canonical thread read 应使用 hydrated threadId 和 full turns view", async () => {
    const appServerClient = appServerClientMock();
    const client = createThreadClient({
      appServerClient,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    await expect(client.readAgentRuntimeThread(" thread-1 ")).resolves.toEqual(
      expect.objectContaining({
        thread: expect.objectContaining({ id: "thread-1" }),
      }),
    );
    expect(appServerClient.readThread).toHaveBeenCalledWith({
      threadId: "thread-1",
      includeTurns: true,
    });
    await expect(client.readAgentRuntimeThread(" ")).rejects.toThrow(
      "threadId is required",
    );
  });

  it("model selection 应只提交一次 current thread/settings/update", async () => {
    const appServerClient = appServerClientMock();
    const client = createThreadClient({ appServerClient });

    await client.updateAgentRuntimeThreadSettings({
      threadId: " thread-1 ",
      modelProvider: "xai",
      model: "grok-4.5",
      effort: "xhigh",
    });

    expect(appServerClient.updateThreadSettings).toHaveBeenCalledTimes(1);
    expect(appServerClient.updateThreadSettings).toHaveBeenCalledWith({
      threadId: "thread-1",
      modelProvider: "xai",
      model: "grok-4.5",
      effort: "xhigh",
    });
  });

  it("thread rename 应只提交 current thread/setName", async () => {
    const appServerClient = appServerClientMock();
    const client = createThreadClient({ appServerClient });

    await client.setAgentRuntimeThreadName({
      threadId: " thread-1 ",
      name: " 新标题 ",
    });

    expect(appServerClient.setThreadName).toHaveBeenCalledWith({
      threadId: "thread-1",
      name: "新标题",
    });
  });

  it("canonical child Thread 导航应只读取身份并严格解析 sessionId", async () => {
    const appServerClient = appServerClientMock();
    const readThread = vi.mocked(appServerClient.readThread);
    readThread.mockResolvedValueOnce({
      id: 3,
      result: {
        thread: {
          createdAt: 0.1,
          id: "thread-child",
          sessionId: "agent-child",
          status: { type: "active", activeFlags: [] },
          updatedAt: 0.2,
        },
      },
      response: { id: 3, result: {} },
      messages: [],
      notifications: [],
    });
    const client = createThreadClient({
      appServerClient,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    await expect(client.readThreadSessionId(" thread-child ")).resolves.toBe(
      "agent-child",
    );
    expect(readThread).toHaveBeenCalledWith({
      threadId: "thread-child",
      includeTurns: false,
    });

    readThread.mockResolvedValueOnce({
      id: 4,
      result: {
        thread: {
          createdAt: 0.1,
          id: "thread-other",
          sessionId: "agent-other",
          status: { type: "active", activeFlags: [] },
          updatedAt: 0.2,
        },
      },
      response: { id: 4, result: {} },
      messages: [],
      notifications: [],
    });
    await expect(client.readThreadSessionId("thread-child")).rejects.toThrow(
      "mismatched threadId",
    );

    readThread.mockResolvedValueOnce({
      id: 5,
      result: {
        thread: {
          createdAt: 0.1,
          id: "thread-child",
          sessionId: " ",
          status: { type: "active", activeFlags: [] },
          updatedAt: 0.2,
        },
      },
      response: { id: 5, result: {} },
      messages: [],
      notifications: [],
    });
    await expect(client.readThreadSessionId("thread-child")).rejects.toThrow(
      "empty sessionId",
    );
    await expect(client.readThreadSessionId(" ")).rejects.toThrow(
      "threadId is required",
    );
  });

  it("浏览器 DevBridge 可用时 submit 应允许进入 App Server JSON-RPC", async () => {
    vi.mocked(isAppServerBridgeAvailable).mockReturnValue(true);
    const appServerClient = appServerClientMock();
    const invokeCommand = vi.fn() as unknown as AgentRuntimeCommandInvoke;
    const client = createThreadClient({
      appServerClient,
      invokeCommand,
    });

    await client.submitAgentRuntimeTurn(
      turnStartParams({
        text: "整理新闻",
        threadId: "session-1",
        eventName: "event-1",
      }),
    );

    expect(appServerClient.startTurn).toHaveBeenCalledWith({
      threadId: "session-1",
      input: [{ type: "text", text: "整理新闻" }],
    });
    expect(invokeCommand).not.toHaveBeenCalled();
  });

  it("App Server 可用时 steer 应直接进入 turn/steer", async () => {
    const appServerClient = appServerClientMock();
    const client = createThreadClient({
      appServerClient,
      isAppServerTurnLifecycleAvailable: () => true,
    });
    const request = {
      threadId: "thread-1",
      expectedTurnId: "turn-1",
      input: [{ type: "text" as const, text: "补充约束" }],
    };

    await expect(client.steerAgentRuntimeTurn(request)).resolves.toEqual(
      expect.objectContaining({ result: { turnId: "turn-1" } }),
    );
    expect(appServerClient.steerTurn).toHaveBeenCalledWith(request);
  });

  it("App Server submit 参数只保留 typed Turn 配置与业务 metadata", () => {
    const expectedOutput = {
      artifactKind: "content_batch",
      outputFormat: {
        type: "json_schema",
        schema: {
          type: "object",
          properties: {
            items: {
              type: "array",
            },
          },
          required: ["items"],
        },
        maxValidationRetries: 2,
      },
    };
    const structuredOutput = expectedOutput.outputFormat;
    const outputSchema = structuredOutput.schema;
    const params = turnStartParams({
      threadId: "session-claw",
      text: "继续执行完整 Claw 链路",
      eventName: "agent_stream_claw",
      model: "deepseek-v4-pro",
      effort: "high",
      approvalPolicy: "on-request",
      sandboxPolicy: "workspace-write",
      outputSchema,
      attachments: [
        {
          kind: "image",
          uri: "data:image/png;base64,claw",
          metadata: {
            mediaType: "image/png",
            index: 0,
          },
        },
      ],
      metadata: {
        harness: {
          source: "claw",
          workspace_skill_runtime_enable: {
            source: "manual_session_enable",
          },
        },
      },
    });

    expect(params).toEqual({
      threadId: "session-claw",
      input: [
        { type: "text", text: "继续执行完整 Claw 链路" },
        { type: "image", url: "data:image/png;base64,claw" },
      ],
      model: "deepseek-v4-pro",
      effort: "high",
      approvalPolicy: "on-request",
      sandboxPolicy: "workspace-write",
      additionalContext: createApplicationAdditionalContext({
        rendererEventName: "agent_stream_claw",
        metadata: {
          harness: {
            source: "claw",
            workspace_skill_runtime_enable: {
              source: "manual_session_enable",
            },
          },
        },
      }),
      outputSchema,
    });
  });

  it("App Server submit 返回未配对 tool.result 时不应投递到前端 stream event", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.startTurn).mockResolvedValueOnce({
      id: 1,
      result: {
        turn: {
          turnId: "turn-1",
          sessionId: "session-1",
          threadId: "thread-1",
          status: "accepted",
        },
      },
      response: {
        id: 1,
        result: {
          turn: {
            turnId: "turn-1",
            sessionId: "session-1",
            threadId: "thread-1",
            status: "accepted",
          },
        },
      },
      messages: [],
      notifications: [
        {
          method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
          params: {
            event: {
              eventId: "evt-orphan-tool-result",
              sequence: 1,
              sessionId: "session-1",
              threadId: "thread-1",
              turnId: "turn-1",
              type: "tool.result",
              timestamp: "2026-06-06T00:00:00.000Z",
              payload: {
                toolCallId: "tool-orphan",
                output: "should be blocked",
              },
            },
          },
        },
      ],
    });
    const client = createThreadClient({
      appServerClient,
      invokeCommand: vi.fn() as unknown as AgentRuntimeCommandInvoke,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    const listener = vi.fn();
    const unlisten = await listenAgentRuntimeEvent(
      "agent_stream_message-orphan",
      listener,
    );

    await client.submitAgentRuntimeTurn(
      turnStartParams({
        text: "生成草稿",
        threadId: "session-1",
        eventName: "agent_stream_message-orphan",
      }),
    );

    expect(listener).not.toHaveBeenCalled();
    unlisten();
  });

  it("App Server submit 不应把缺少 started 的 tool terminal 合成多个前端事件", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.startTurn).mockResolvedValueOnce({
      id: 1,
      result: {
        turn: {
          turnId: "turn-1",
          sessionId: "session-1",
          threadId: "thread-1",
          status: "accepted",
        },
      },
      response: {
        id: 1,
        result: {
          turn: {
            turnId: "turn-1",
            sessionId: "session-1",
            threadId: "thread-1",
            status: "accepted",
          },
        },
      },
      messages: [],
      notifications: [
        {
          method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
          params: {
            event: {
              eventId: "evt-tool-completed",
              sequence: 1,
              sessionId: "session-1",
              threadId: "thread-1",
              turnId: "turn-1",
              type: "tool.completed",
              timestamp: "2026-06-06T00:00:00.000Z",
              payload: {
                toolCallId: "tool-fanout",
                toolName: "search",
                output: "done",
              },
            },
          },
        },
      ],
    });
    const client = createThreadClient({
      appServerClient,
      invokeCommand: vi.fn() as unknown as AgentRuntimeCommandInvoke,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    const listener = vi.fn();
    const unlisten = await listenAgentRuntimeEvent(
      "agent_stream_message-fanout",
      listener,
    );

    await client.submitAgentRuntimeTurn(
      turnStartParams({
        text: "生成草稿",
        threadId: "session-1",
        eventName: "agent_stream_message-fanout",
      }),
    );

    expect(listener).not.toHaveBeenCalled();
    unlisten();
  });

  it("App Server submit 开启 drain route 后应只投递镜像 notification 一次", async () => {
    const appServerClient = appServerClientMock();
    const notification = {
      method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
      params: {
        event: {
          eventId: "evt-routing",
          sequence: 1,
          sessionId: "session-1",
          threadId: "thread-1",
          turnId: "turn-request",
          type: "runtime.status",
          timestamp: "2026-06-06T00:00:00.000Z",
          payload: {
            status: {
              phase: "routing",
              title: "正在路由",
              detail: "已进入 App Server",
            },
          },
        },
      },
    } satisfies AppServerJsonRpcNotification;
    vi.mocked(appServerClient.drainEvents)
      .mockResolvedValueOnce([notification])
      .mockResolvedValue([]);
    vi.mocked(appServerClient.startTurn).mockResolvedValueOnce({
      ...malformedAppServerResult({}),
      notifications: [notification],
    });
    const client = createThreadClient({
      appServerClient,
      invokeCommand: vi.fn() as unknown as AgentRuntimeCommandInvoke,
      isAppServerTurnLifecycleAvailable: () => true,
      enableAppServerEventDrain: true,
    });

    const listener = vi.fn();
    const unlisten = await listenAgentRuntimeEvent(
      "agent_stream_missing-turn",
      listener,
    );

    await client.submitAgentRuntimeTurn(
      turnStartParams({
        text: "生成草稿",
        threadId: "session-1",
        eventName: "agent_stream_missing-turn",
      }),
    );

    await vi.waitFor(() => {
      expect(listener).toHaveBeenCalledTimes(1);
      expect(listener).toHaveBeenCalledWith({
        payload: expect.objectContaining({
          type: "runtime_status",
          event_id: "evt-routing",
          session_id: "session-1",
          turn_id: "turn-request",
        }),
      });
    });
    unlisten();
  });

  it("App Server submit 应立即投递请求结果中的 direct v2 notification", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.startTurn).mockResolvedValueOnce({
      ...malformedAppServerResult({}),
      notifications: [
        {
          method: "item/agentMessage/delta",
          params: {
            delta: "请求结果中的最终答复",
            itemId: "item-result",
            threadId: "thread-result",
            turnId: "turn-result",
          },
        },
      ],
    });
    const client = createThreadClient({
      appServerClient,
      invokeCommand: vi.fn() as unknown as AgentRuntimeCommandInvoke,
      isAppServerTurnLifecycleAvailable: () => true,
      enableAppServerEventDrain: true,
    });
    const listener = vi.fn();
    const eventName = "agent_stream_result_notification";
    const unlisten = await listenAgentRuntimeEvent(eventName, listener);

    await client.submitAgentRuntimeTurn(
      turnStartParams({
        threadId: "thread-result",
        eventName,
      }),
    );

    expect(listener).toHaveBeenCalledWith({
      payload: expect.objectContaining({
        type: "text_delta",
        text: "请求结果中的最终答复",
        thread_id: "thread-result",
        turn_id: "turn-result",
      }),
    });
    unlisten();
  });

  it("App Server submit 注册 route 后仍应投递请求错误携带的 notification", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.startTurn).mockRejectedValueOnce(
      new AppServerRpcError(
        {
          id: 1,
          error: { code: -32000, message: "turn/start failed" },
        },
        [
          {
            method: "item/agentMessage/delta",
            params: {
              delta: "失败前已接收的文本",
              itemId: "item-error",
              threadId: "thread-error",
              turnId: "turn-error",
            },
          },
        ],
      ),
    );
    const client = createThreadClient({
      appServerClient,
      invokeCommand: vi.fn() as unknown as AgentRuntimeCommandInvoke,
      isAppServerTurnLifecycleAvailable: () => true,
      enableAppServerEventDrain: true,
    });
    const listener = vi.fn();
    const eventName = "agent_stream_error_notification";
    const unlisten = await listenAgentRuntimeEvent(eventName, listener);

    await expect(
      client.submitAgentRuntimeTurn(
        turnStartParams({ threadId: "thread-error", eventName }),
      ),
    ).rejects.toThrow("turn/start failed");

    expect(listener).toHaveBeenCalledWith({
      payload: expect.objectContaining({
        type: "text_delta",
        text: "失败前已接收的文本",
        thread_id: "thread-error",
        turn_id: "turn-error",
      }),
    });
    unlisten();
  });

  it("App Server review 应把 response notification 投递到当前 thread event route", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.startReview).mockResolvedValueOnce({
      id: 8,
      result: {
        reviewThreadId: "thread-review",
        turn: {
          id: "turn-review",
          items: [],
          itemsView: "notLoaded",
          status: "inProgress",
        },
      },
      response: { id: 8, result: {} },
      messages: [],
      notifications: [
        {
          method: "item/agentMessage/delta",
          params: {
            delta: "审查已开始",
            itemId: "item-review",
            threadId: "thread-review",
            turnId: "turn-review",
          },
        },
      ],
    });
    const client = createThreadClient({
      appServerClient,
      invokeCommand: vi.fn() as unknown as AgentRuntimeCommandInvoke,
      isAppServerTurnLifecycleAvailable: () => true,
      enableAppServerEventDrain: true,
    });
    const listener = vi.fn();
    const eventName = "agentSession/event/thread-review";
    const unlisten = await listenAgentRuntimeEvent(eventName, listener);

    await expect(
      client.submitAgentRuntimeReview({
        threadId: " thread-review ",
        delivery: "inline",
        target: { type: "uncommittedChanges" },
      }),
    ).resolves.toMatchObject({
      result: { reviewThreadId: "thread-review" },
    });

    expect(appServerClient.startReview).toHaveBeenCalledWith({
      threadId: "thread-review",
      delivery: "inline",
      target: { type: "uncommittedChanges" },
    });
    expect(listener).toHaveBeenCalledWith({
      payload: expect.objectContaining({
        type: "text_delta",
        text: "审查已开始",
        thread_id: "thread-review",
        turn_id: "turn-review",
      }),
    });
    unlisten();
  });

  it("App Server review 失败时应把 RPC error 投递到当前 thread event route", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.startReview).mockRejectedValueOnce(
      new AppServerRpcError(
        {
          id: 9,
          error: { code: -32000, message: "review failed" },
        },
        [
          {
            method: "error",
            params: {
              error: { message: "review failed" },
              threadId: "thread-review-error",
              turnId: "turn-review-error",
              willRetry: false,
            },
          },
        ],
      ),
    );
    const client = createThreadClient({
      appServerClient,
      invokeCommand: vi.fn() as unknown as AgentRuntimeCommandInvoke,
      isAppServerTurnLifecycleAvailable: () => true,
      enableAppServerEventDrain: true,
    });
    const listener = vi.fn();
    const eventName = "agentSession/event/thread-review-error";
    const unlisten = await listenAgentRuntimeEvent(eventName, listener);

    await expect(
      client.submitAgentRuntimeReview({
        threadId: "thread-review-error",
        delivery: "inline",
        target: { type: "uncommittedChanges" },
      }),
    ).rejects.toThrow("review failed");

    expect(listener).toHaveBeenCalledWith({
      payload: expect.objectContaining({
        type: "error",
        message: "review failed",
        thread_id: "thread-review-error",
        turn_id: "turn-review-error",
      }),
    });
    unlisten();
  });

  it("App Server runtime.status 应保留 retrying phase 供 GUI 展示恢复状态", () => {
    const payload = projectAppServerAgentEventPayload({
      method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
      params: {
        event: {
          eventId: "evt-retrying",
          sequence: 1,
          sessionId: "session-1",
          threadId: "thread-1",
          turnId: "turn-1",
          type: "runtime.status",
          timestamp: "2026-06-06T00:00:00.000Z",
          payload: {
            status: {
              phase: "retrying",
              title: "正在恢复模型输出",
              detail: "模型通道在尾段暂时中断，正在补齐最终答复。",
              metadata: {
                agentui: {
                  eventClass: "run.status",
                },
              },
            },
          },
        },
      },
    });

    expect(payload).toMatchObject({
      type: "runtime_status",
      event_id: "evt-retrying",
      renderer_event_received_at: expect.any(Number),
      server_event_emitted_at: Date.parse("2026-06-06T00:00:00.000Z"),
      status: {
        phase: "retrying",
        title: "正在恢复模型输出",
        detail: "模型通道在尾段暂时中断，正在补齐最终答复。",
      },
    });
  });

  it("App Server 不可用时 submit 应 fail closed，不回退 legacy command", async () => {
    const appServerClient = appServerClientMock();
    const invokeCommand = vi.fn().mockResolvedValue(undefined);
    const client = createThreadClient({
      appServerClient,
      invokeCommand: invokeCommand as unknown as AgentRuntimeCommandInvoke,
      isAppServerTurnLifecycleAvailable: () => false,
    });
    const request = {
      message: "继续",
      session_id: "session-1",
      event_name: "event-1",
      workspace_id: "workspace-1",
    };

    await expect(client.submitAgentRuntimeTurn(request)).rejects.toThrow(
      "App Server turn lifecycle is unavailable; Agent Runtime requires the App Server current lifecycle channel.",
    );

    expect(invokeCommand).not.toHaveBeenCalled();
    expect(appServerClient.startTurn).not.toHaveBeenCalled();
  });

  it("App Server 可用且 turn_id 存在时 interrupt 应进入 turn/interrupt", async () => {
    const appServerClient = appServerClientMock();
    const invokeCommand = vi.fn() as unknown as AgentRuntimeCommandInvoke;
    const client = createThreadClient({
      appServerClient,
      invokeCommand,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    await expect(
      client.interruptAgentRuntimeTurn({
        session_id: "session-1",
        turn_id: "turn-1",
      }),
    ).resolves.toBe(true);

    expect(appServerClient.cancelTurn).toHaveBeenCalledWith({
      threadId: "session-1",
      turnId: "turn-1",
    });
    expect(invokeCommand).not.toHaveBeenCalled();
  });

  it("App Server 不可用时 interrupt 应 fail closed，不回退 legacy command", async () => {
    const appServerClient = appServerClientMock();
    const invokeCommand = vi.fn().mockResolvedValue(true);
    const client = createThreadClient({
      appServerClient,
      invokeCommand: invokeCommand as unknown as AgentRuntimeCommandInvoke,
      isAppServerTurnLifecycleAvailable: () => false,
    });
    const request = { session_id: "session-1", turn_id: "turn-1" };

    await expect(client.interruptAgentRuntimeTurn(request)).rejects.toThrow(
      "App Server turn lifecycle is unavailable; Agent Runtime requires the App Server current lifecycle channel.",
    );

    expect(invokeCommand).not.toHaveBeenCalled();
    expect(appServerClient.cancelTurn).not.toHaveBeenCalled();
  });

  it("缺少 turn_id 时 interrupt 应 fail closed，不回退 legacy command", async () => {
    const appServerClient = appServerClientMock();
    const invokeCommand = vi.fn().mockResolvedValue(true);
    const client = createThreadClient({
      appServerClient,
      invokeCommand: invokeCommand as unknown as AgentRuntimeCommandInvoke,
      isAppServerTurnLifecycleAvailable: () => true,
    });
    const request = { session_id: "session-1" };

    await expect(client.interruptAgentRuntimeTurn(request)).rejects.toThrow(
      "turn_id is required to cancel App Server turn",
    );

    expect(invokeCommand).not.toHaveBeenCalled();
    expect(appServerClient.cancelTurn).not.toHaveBeenCalled();
  });

  it("typed pending 不存在时 respond action 应 fail closed，不回退旧 action/respond", async () => {
    const appServerClient = appServerClientMock();
    const invokeCommand = vi.fn() as unknown as AgentRuntimeCommandInvoke;
    const client = createThreadClient({
      appServerClient,
      invokeCommand,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    await expect(
      client.respondAgentRuntimeAction({
        session_id: "session-1",
        request_id: "req-1",
        action_type: "ask_user",
        confirmed: true,
      }),
    ).rejects.toThrow(
      "Typed server request is no longer pending; generic agentSession/action/respond is retired.",
    );
    expect(invokeCommand).not.toHaveBeenCalled();
  });

  it("App Server 可用时 thread read 应进入 thread/read 并投影 read model", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce({
      id: 3,
      result: {
        thread: {
          id: "thread-1",
          sessionId: "session-1",
          cwd: "/tmp/workspace",
          createdAt: 0.1,
          updatedAt: 0.3,
          status: { type: "active", activeFlags: ["waitingOnUserInput"] },
          turns: [{ id: "turn-1", status: "inProgress", items: [] }],
        },
      },
      response: {
        id: 3,
        result: {},
      },
      messages: [],
      notifications: [],
    });
    const invokeCommand = vi.fn().mockResolvedValue({});
    const client = createThreadClient({
      appServerClient,
      invokeCommand: invokeCommand as unknown as AgentRuntimeCommandInvoke,
      isAppServerTurnLifecycleAvailable: () => true,
    });

    await expect(
      client.getAgentRuntimeThreadRead("session-1"),
    ).resolves.toEqual(
      expect.objectContaining({
        thread_id: "thread-1",
        status: "waitingAction",
        profile_status: "blocked",
        turns: [
          expect.objectContaining({
            turn_id: "turn-1",
            status: "running",
            native_status: "running",
          }),
        ],
      }),
    );

    expect(appServerClient.readThread).toHaveBeenCalledWith({
      threadId: "session-1",
      includeTurns: true,
    });
    expect(invokeCommand).not.toHaveBeenCalled();
  });

  it("App Server 不可用时 thread read 应 fail closed，不回退 legacy command", async () => {
    const appServerClient = appServerClientMock();
    const invokeCommand = vi.fn().mockResolvedValue({});
    const client = createThreadClient({
      appServerClient,
      invokeCommand: invokeCommand as unknown as AgentRuntimeCommandInvoke,
      isAppServerTurnLifecycleAvailable: () => false,
    });

    await expect(client.getAgentRuntimeThreadRead("session-1")).rejects.toThrow(
      "App Server turn lifecycle is unavailable; Agent Runtime requires the App Server current lifecycle channel.",
    );

    expect(appServerClient.readThread).not.toHaveBeenCalled();
    expect(invokeCommand).not.toHaveBeenCalled();
  });

  it("App Server 不可用时 respond action 应 fail closed，不回退 legacy command", async () => {
    const appServerClient = appServerClientMock();
    const invokeCommand = vi.fn().mockResolvedValue(undefined);
    const client = createThreadClient({
      appServerClient,
      invokeCommand: invokeCommand as unknown as AgentRuntimeCommandInvoke,
      isAppServerTurnLifecycleAvailable: () => false,
    });

    await expect(
      client.respondAgentRuntimeAction({
        session_id: "session-1",
        request_id: "req-1",
        action_type: "ask_user",
        confirmed: true,
      }),
    ).rejects.toThrow(
      "App Server turn lifecycle is unavailable; Agent Runtime requires the App Server current lifecycle channel.",
    );

    expect(invokeCommand).not.toHaveBeenCalled();
  });

  it("request projection 应在未传可选项时保持精简 App Server 参数", () => {
    expect(
      turnStartParams({
        text: "继续",
        threadId: "session-1",
        eventName: "agentSession/event/session-1",
      }),
    ).toEqual({
      threadId: "session-1",
      input: [{ type: "text", text: "继续" }],
      additionalContext: createApplicationAdditionalContext({
        rendererEventName: "agentSession/event/session-1",
      }),
    });
  });
});
