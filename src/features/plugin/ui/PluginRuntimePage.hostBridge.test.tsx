import { describe, expect, it, vi } from "vitest";
import {
  compatiblePluginStandardVersions,
  currentPluginHostRuntimeVersion,
  currentPluginStandardVersion,
} from "../readiness/hostCapabilityProfile";
import {
  appServerClientMocks,
  apiMocks,
  buildReadyState,
  dispatchBridgeMessage,
  flush,
  getRuntimeFrame,
  renderPage,
  usePluginRuntimePageTestLifecycle,
} from "./PluginRuntimePage.testFixtures";

type PostMessageSpy = ReturnType<typeof vi.spyOn>;

function readHostBridgeResult(
  postMessage: PostMessageSpy,
  requestId: string,
): Record<string, unknown> & { taskId: string; traceId: string } {
  const call = postMessage.mock.calls.find(([message]) => {
    return (
      isRecord(message) &&
      message.type === "host:response" &&
      message.requestId === requestId
    );
  });
  const message = call?.[0];
  if (!isRecord(message) || !isRecord(message.payload)) {
    throw new Error(`未找到 Host Bridge 响应：${requestId}`);
  }
  const result = message.payload.result;
  if (
    !isRecord(result) ||
    typeof result.taskId !== "string" ||
    typeof result.traceId !== "string"
  ) {
    throw new Error(`Host Bridge 响应缺少任务标识：${requestId}`);
  }
  return result as Record<string, unknown> & {
    taskId: string;
    traceId: string;
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

describe("PluginRuntimePage Host Bridge", () => {
  usePluginRuntimePageTestLifecycle();

  it("App 可通过 lime.capabilities.getProfile 发现 Host capability profile", async () => {
    const container = await renderPage();
    await flush();
    const frame = getRuntimeFrame(container);
    const postMessage = vi.spyOn(frame.contentWindow!, "postMessage");

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.capabilities",
        method: "getProfile",
      },
      "profile-discovery",
    );

    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "profile-discovery",
        payload: expect.objectContaining({
          ok: true,
          result: expect.objectContaining({
            appRuntimeVersion: currentPluginHostRuntimeVersion,
            standardVersions: {
              current: currentPluginStandardVersion,
              compatible: compatiblePluginStandardVersions,
            },
            runtimeProfile: expect.objectContaining({
              runtimeId: `content-factory-app:in_lime:${currentPluginHostRuntimeVersion}`,
              runtimeVersion: currentPluginHostRuntimeVersion,
              shellKind: "desktop",
              installMode: "in_lime",
            }),
            runtimeCapabilities: expect.objectContaining({
              "lime.agent": expect.objectContaining({
                available: true,
                implementation: "adapter",
              }),
            }),
            standards: expect.objectContaining({
              layeredManifest: expect.objectContaining({
                version: "0.5",
                enabled: true,
              }),
              agentRuntime: expect.objectContaining({
                version: "0.6",
                enabled: true,
              }),
              requirementBoundary: expect.objectContaining({
                version: "0.7",
                enabled: false,
              }),
            }),
            capabilities: expect.objectContaining({
              "lime.capabilities": expect.objectContaining({
                enabled: true,
                implementation: "native",
              }),
              "lime.agent": expect.objectContaining({
                enabled: true,
              }),
              "lime.workflow": expect.objectContaining({
                enabled: false,
                implementation: "none",
              }),
              "lime.skills": expect.objectContaining({
                enabled: true,
                implementation: "adapter",
              }),
              "lime.memory": expect.objectContaining({
                enabled: true,
                implementation: "adapter",
              }),
              "lime.context": expect.objectContaining({
                enabled: true,
                implementation: "adapter",
              }),
              "lime.search": expect.objectContaining({
                enabled: true,
                implementation: "adapter",
              }),
              "lime.browser": expect.objectContaining({
                enabled: true,
                implementation: "adapter",
              }),
              "lime.documents": expect.objectContaining({
                enabled: true,
                implementation: "adapter",
              }),
              "lime.media": expect.objectContaining({
                enabled: true,
                implementation: "adapter",
              }),
              "lime.mcp": expect.objectContaining({
                enabled: true,
                implementation: "adapter",
              }),
              "lime.terminal": expect.objectContaining({
                enabled: true,
                implementation: "adapter",
              }),
              "lime.connectors": expect.objectContaining({
                enabled: true,
                implementation: "adapter",
              }),
            }),
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(postMessage).not.toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:error",
        requestId: "profile-discovery",
      }),
      expect.any(String),
    );
  });

  it("App 可通过 Lime 客户端 profile 读取 v0.6 AgentRuntime 合同", async () => {
    const agentRuntime = {
      agentTask: {
        eventSchema: "lime.agent-task-event.v1",
        resultSchema: "lime.agent-task-result.v1",
        structuredOutput: {
          type: "json_schema",
          schemaRef: "./artifacts/content-factory-workspace-patch.schema.json",
        },
        approval: { behavior: "host-mediated" },
        sessionPolicy: { modes: ["new", "resume", "continue", "fork"] },
        toolDiscovery: { mode: "on_demand" },
        checkpointScope: { workflowState: true },
        observability: { profileEvents: true },
      },
    };
    const container = await renderPage(
      buildReadyState({
        manifestPatch: {
          manifestVersion: "0.6.0",
          version: "0.6.0",
          requires: {
            sdk: "@lime/app-sdk@^0.6.0",
            capabilities: ["lime.agent", "lime.skills", "lime.usage"],
          },
          agentRuntime,
        },
      }),
    );
    await flush();
    const frame = getRuntimeFrame(container);
    const postMessage = vi.spyOn(frame.contentWindow!, "postMessage");

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.capabilities",
        method: "getProfile",
      },
      "profile-v06-runtime",
    );

    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "profile-v06-runtime",
        payload: expect.objectContaining({
          ok: true,
          result: expect.objectContaining({
            appRuntimeVersion: currentPluginHostRuntimeVersion,
            standardVersions: {
              current: currentPluginStandardVersion,
              compatible: compatiblePluginStandardVersions,
            },
            standards: expect.objectContaining({
              layeredManifest: expect.objectContaining({
                version: "0.5",
                enabled: true,
                layerFiles: expect.arrayContaining(["app.capabilities.yaml"]),
              }),
              agentRuntime: expect.objectContaining({
                version: "0.6",
                enabled: true,
                manifestVersion: "0.6",
                eventSchema: "lime.agent-task-event.v1",
                resultSchema: "lime.agent-task-result.v1",
                structuredOutput: true,
                approval: true,
                sessionPolicy: true,
                toolDiscovery: true,
                checkpointScope: true,
                observability: true,
              }),
              requirementBoundary: expect.objectContaining({
                version: "0.7",
                enabled: false,
              }),
            }),
            agentRuntime,
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
  });

  it("App 可通过 Lime 客户端 profile 读取 v0.7 需求边界与能力交接", async () => {
    const requirements = {
      requirements: [
        {
          id: "CF-R001",
          text: "生成可审核内容草稿",
          priority: "mvp",
        },
      ],
      nonGoals: ["不在 App 包内保存外部凭证"],
    };
    const boundary = {
      boundaries: [
        {
          requirementId: "CF-R001",
          planes: {
            app: { owns: ["workflow_state"] },
            host: { requires: ["lime.agent", "lime.evidence"] },
          },
        },
      ],
    };
    const integrations = [
      {
        key: "planning_table",
        provider: "cloud.table",
        executionPlane: "hybrid",
        hostCapability: "lime.connectors",
      },
    ];
    const operations = [
      {
        key: "write_external_draft",
        type: "external_write",
        sideEffect: "external_write",
        approvalRequired: true,
        dryRunRequired: true,
        evidenceRequired: true,
        autoExecute: false,
      },
    ];
    const container = await renderPage(
      buildReadyState({
        manifestPatch: {
          manifestVersion: "0.7.0",
          version: "0.7.0",
          requires: {
            sdk: "@lime/app-sdk@^0.7.0",
            capabilities: ["lime.agent", "lime.connectors", "lime.evidence"],
          },
          requirements,
          boundary,
          integrations,
          operations,
        },
      }),
    );
    await flush();
    const frame = getRuntimeFrame(container);
    const postMessage = vi.spyOn(frame.contentWindow!, "postMessage");

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.capabilities",
        method: "getProfile",
      },
      "profile-v07-handoff",
    );

    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "profile-v07-handoff",
        payload: expect.objectContaining({
          ok: true,
          result: expect.objectContaining({
            appRuntimeVersion: currentPluginHostRuntimeVersion,
            standardVersions: {
              current: currentPluginStandardVersion,
              compatible: compatiblePluginStandardVersions,
            },
            requirements,
            boundary,
            integrations,
            operations,
            standards: expect.objectContaining({
              requirementBoundary: expect.objectContaining({
                version: "0.7",
                enabled: true,
                manifestVersion: "0.7",
                requirementCount: 1,
                boundaryCount: 1,
                integrationCount: 1,
                operationCount: 1,
                hostCloudManagedExecution: true,
                externalSideEffectsRequireApproval: true,
              }),
            }),
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
  });

  it("Host Bridge 能在同一 App 作用域内 start / stream / get / cancel / retry Agent task", async () => {
    const container = await renderPage();
    await flush();
    const frame = getRuntimeFrame(container);
    const postMessage = vi.spyOn(frame.contentWindow!, "postMessage");

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.agent",
        method: "startTask",
        input: {
          title: "生成内容场景",
          prompt: "基于项目知识生成内容场景",
          taskKind: "content.scenario_planning",
          idempotencyKey: "dashboard:scenario",
          input: { projectId: "project-1" },
          expectedOutput: { artifactKind: "content_table" },
          humanReview: true,
        },
      },
      "task-start",
    );
    const startTaskResult = readHostBridgeResult(postMessage, "task-start");
    expect(startTaskResult).toEqual(
      expect.objectContaining({
        taskId: expect.stringMatching(/^plugin-task-/),
        traceId: expect.stringMatching(/^plugin-trace-plugin-task-/),
        status: "running",
        humanReview: true,
      }),
    );
    const taskId = startTaskResult.taskId;
    const traceId = startTaskResult.traceId;

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.agent",
        method: "streamTask",
        input: { taskId },
      },
      "task-stream",
    );
    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.agent",
        method: "getTask",
        input: { taskId },
      },
      "task-get",
    );
    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.agent",
        method: "submitHostResponse",
        input: {
          taskId,
          requestId: "runtime-request-1",
          actionType: "ask_user",
          response: "补充项目定位：高客单价咨询服务。",
        },
      },
      "task-host-response",
    );
    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.agent",
        method: "cancelTask",
        input: { taskId },
      },
      "task-cancel",
    );
    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.agent",
        method: "retryTask",
        input: { taskId },
      },
      "task-retry",
    );

    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "task-stream",
        payload: expect.objectContaining({
          ok: true,
          result: [
            expect.objectContaining({
              type: "task:queued",
              status: "running",
            }),
          ],
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "task-get",
        payload: expect.objectContaining({
          ok: true,
          result: expect.objectContaining({
            taskId,
            traceId,
            status: "running",
            taskKind: "content.scenario_planning",
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:error",
        requestId: "task-host-response",
        payload: expect.objectContaining({
          ok: false,
          message:
            "Typed server request is no longer pending; generic agentSession/action/respond is retired.",
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "task-cancel",
        payload: expect.objectContaining({
          ok: true,
          result: expect.objectContaining({
            taskId,
            status: "cancelled",
            events: expect.arrayContaining([
              expect.objectContaining({ type: "task:cancelled" }),
            ]),
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "task-retry",
        payload: expect.objectContaining({
          ok: true,
          result: expect.objectContaining({
            taskId: expect.stringMatching(/^plugin-task-/),
            traceId: expect.stringMatching(/^plugin-trace-plugin-task-/),
            retryOfTaskId: taskId,
            retryAttempt: 1,
            status: "running",
            humanReview: true,
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(appServerClientMocks.startSession).toHaveBeenCalledWith({
      serviceName: "生成内容场景",
      threadSource: "plugin",
      historyMode: "paginated",
    });
    expect(appServerClientMocks.startTurn).toHaveBeenCalledWith(
      expect.objectContaining({
        threadId: "plugin-thread-1",
        additionalContext: {
          metadata: expect.objectContaining({
            kind: "application",
            value: expect.stringContaining(`"retryOfTaskId":"${taskId}"`),
          }),
        },
        responsesapiClientMetadata: expect.objectContaining({
          appId: "content-factory-app",
          taskKind: "content.scenario_planning",
          workspaceId: "workspace-1",
        }),
      }),
    );
    expect(appServerClientMocks.readThread).toHaveBeenCalledWith({
      threadId: "plugin-thread-1",
      includeTurns: true,
    });
    expect(appServerClientMocks.cancelTurn).toHaveBeenCalledWith({
      threadId: "plugin-thread-1",
      turnId: "plugin-turn-1",
    });
  });

  it("Host Bridge 能把 App Server workflow/read 只读投影给 iframe", async () => {
    const container = await renderPage();
    await flush();
    const frame = getRuntimeFrame(container);
    const postMessage = vi.spyOn(frame.contentWindow!, "postMessage");

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.agent",
        method: "readWorkflow",
        input: { sessionId: "plugin-session-1" },
      },
      "workflow-read",
    );

    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "workflow-read",
        payload: expect.objectContaining({
          ok: true,
          result: expect.objectContaining({
            sessionId: "plugin-session-1",
            source: "app_server_workflow_read",
            workflow: expect.objectContaining({
              activeWorkflowRunId: "plugin-workflow-run-1",
            }),
            workflowRuns: [
              expect.objectContaining({
                workflowRunId: "plugin-workflow-run-1",
                status: "running",
              }),
            ],
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(appServerClientMocks.readWorkflow).toHaveBeenCalledWith({
      sessionId: "plugin-session-1",
    });
  });

  it("Host Bridge 支持 iframe 订阅 workflow read model 首次投影事件", async () => {
    const container = await renderPage();
    await flush();
    const frame = getRuntimeFrame(container);
    const postMessage = vi.spyOn(frame.contentWindow!, "postMessage");

    await dispatchBridgeMessage(
      frame,
      "capability:subscribe",
      {
        capability: "lime.agent",
        topic: "workflow",
        sessionId: "plugin-session-1",
        subscriptionId: "workflow-sub-1",
      },
      "workflow-subscribe",
    );
    await flush();

    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "workflow-subscribe",
        payload: expect.objectContaining({
          subscriptionId: "workflow-sub-1",
          capability: "lime.agent",
          topic: "workflow",
          sessionId: "plugin-session-1",
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "capability:event",
        payload: expect.objectContaining({
          subscriptionId: "workflow-sub-1",
          capability: "lime.agent",
          topic: "workflow",
          eventType: "workflow:readModel",
          sessionId: "plugin-session-1",
          workflowRead: expect.objectContaining({
            source: "app_server_workflow_read",
          }),
          workflow: expect.objectContaining({
            activeWorkflowRunId: "plugin-workflow-run-1",
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
  });

  it("Host Bridge 写回 artifact / evidence 时应拒绝未声明 subject", async () => {
    const container = await renderPage();
    await flush();
    const frame = getRuntimeFrame(container);
    const postMessage = vi.spyOn(frame.contentWindow!, "postMessage");

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.artifacts",
        method: "create",
        input: {
          kind: "content_table",
          title: "内容表",
          content: { rows: [] },
        },
      },
      "artifact-ok",
    );
    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.evidence",
        method: "record",
        input: {
          kind: "fact_grounding",
          message: "声明过的事实支撑证据。",
          refs: ["adapter-artifact-1"],
        },
      },
      "evidence-ok",
    );
    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.artifacts",
        method: "create",
        input: {
          kind: "undeclared_asset_pack",
          title: "未声明资产包",
          content: {},
        },
      },
      "artifact-blocked",
    );
    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.evidence",
        method: "record",
        input: {
          kind: "undeclared_evidence_subject",
          message: "未声明证据。",
        },
      },
      "evidence-blocked",
    );

    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "artifact-ok",
        payload: expect.objectContaining({
          ok: true,
          result: expect.objectContaining({
            kind: "content_table",
            title: "内容表",
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "evidence-ok",
        payload: expect.objectContaining({
          ok: true,
          result: expect.objectContaining({
            kind: "fact_grounding",
            refs: ["adapter-artifact-1"],
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:error",
        requestId: "artifact-blocked",
        payload: expect.objectContaining({
          ok: false,
          code: "upstream_failed",
          causeCode: "WRITEBACK_NOT_DECLARED",
          error: expect.objectContaining({
            code: "upstream_failed",
            causeCode: "WRITEBACK_NOT_DECLARED",
            capability: "lime.artifacts",
            method: "create",
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:error",
        requestId: "evidence-blocked",
        payload: expect.objectContaining({
          ok: false,
          code: "upstream_failed",
          causeCode: "WRITEBACK_NOT_DECLARED",
          error: expect.objectContaining({
            code: "upstream_failed",
            causeCode: "WRITEBACK_NOT_DECLARED",
            capability: "lime.evidence",
            method: "record",
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
  });

  it("Host Bridge 忽略非 runtime origin 的消息", async () => {
    const container = await renderPage();
    await flush();
    const frame = getRuntimeFrame(container);
    const postMessage = vi.spyOn(frame.contentWindow!, "postMessage");

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      { capability: "lime.storage" },
      "evil-req",
      "https://evil.example",
    );

    expect(postMessage).not.toHaveBeenCalledWith(
      expect.objectContaining({
        requestId: "evil-req",
      }),
      expect.any(String),
    );
  });

  it("runtime 启动失败时显示可重试错误态", async () => {
    apiMocks.startPluginUiRuntime.mockRejectedValueOnce(
      new Error("请从本地 APP.md 目录重新安装该 App。"),
    );
    const container = await renderPage();
    await flush();

    expect(container.textContent).toContain("App 打开失败");
    expect(container.textContent).toContain(
      "请从本地 APP.md 目录重新安装该 App。",
    );
    expect(container.textContent).toContain("重新打开");
    expect(
      container.querySelector('[data-testid="plugin-runtime-frame"]'),
    ).toBeNull();
  });
});
