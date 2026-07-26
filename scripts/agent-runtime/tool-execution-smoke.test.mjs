import fs from "node:fs";
import { describe, expect, it } from "vitest";
import {
  AGENT_CONTROL_FINAL_TEXT,
  AGENT_CONTROL_SUBAGENT_ACTIVITY_KINDS,
  AGENT_CONTROL_TOOL_NAMES,
  PARENT_OWNED_DIRECT_INPUT_ERROR,
  buildAgentControlVisibleDomAssertions,
} from "./agent-control-visible-dom-gate-b.mjs";
import {
  buildDeferredMcpVisibleDomAssertions,
  buildDeferredMcpToolSearchAssertions,
  buildDeferredMcpToolSearchFixtureResponses,
  DEFERRED_MCP_TOOL_CALL_ID,
  DEFERRED_MCP_TOOL_SEARCH_FINAL_TEXT,
  DEFERRED_MCP_TOOL_SEARCH_CALL_ID,
} from "./deferred-mcp-tool-search-gate-b.mjs";
import {
  buildAppServerRequestResponse,
  findPendingAppServerRequest,
} from "../lib/agent-runtime-smoke-core.mjs";

function readDeferredGateBSources() {
  return [
    "scripts/agent-runtime/tool-execution-smoke.mjs",
    "scripts/agent-runtime/tool-execution-managed-smoke.mjs",
    "scripts/agent-runtime/deferred-mcp-tool-search-gate-b.mjs",
  ]
    .map((filePath) => fs.readFileSync(filePath, "utf8"))
    .join("\n");
}

function readAgentControlGateBSources() {
  return [
    "scripts/agent-runtime/tool-execution-smoke.mjs",
    "scripts/agent-runtime/tool-execution-managed-smoke.mjs",
    "scripts/agent-runtime/tool-execution-managed-restart.mjs",
    "scripts/agent-runtime/agent-control-visible-dom-gate-b.mjs",
    "src/components/agent/chat/components/AgentThreadTimelineItemRenderers.tsx",
  ]
    .map((filePath) => fs.readFileSync(filePath, "utf8"))
    .join("\n");
}

describe("agent runtime tool execution smoke guard", () => {
  it("通过 typed server request outer response 处理 pending approval", () => {
    const request = findPendingAppServerRequest(
      [
        {
          id: "outer-stale",
          method: "item/commandExecution/requestApproval",
          params: { approvalId: "approval-stale" },
        },
        {
          id: "outer-current",
          method: "item/commandExecution/requestApproval",
          params: { approvalId: "approval-current" },
        },
      ],
      { requestId: "approval-current" },
    );

    expect(request).toMatchObject({ id: "outer-current" });
    expect(buildAppServerRequestResponse(request, { confirmed: true })).toEqual(
      { decision: "accept" },
    );
  });

  it("没有语义 request id 的 elicitation 以 current scope 匹配", () => {
    const request = findPendingAppServerRequest(
      [
        {
          id: "outer-elicitation",
          method: "mcpServer/elicitation/request",
          params: { threadId: "thread-1", turnId: "turn-1" },
        },
      ],
      {
        requestId: "legacy-pending-id",
        actionScope: { thread_id: "thread-1", turn_id: "turn-1" },
      },
    );

    expect(request).toMatchObject({ id: "outer-elicitation" });
    expect(
      buildAppServerRequestResponse(request, {
        confirmed: false,
        userData: { ignored: true },
      }),
    ).toEqual({ action: "decline" });
  });

  it("不再通过已退役 action/respond 提交 pending request", () => {
    const content = [
      "scripts/lib/agent-runtime-smoke-core.mjs",
      "scripts/agent-runtime/tool-execution-smoke.mjs",
      "scripts/agent-runtime/approval-sandbox-smoke.mjs",
    ]
      .map((filePath) => fs.readFileSync(filePath, "utf8"))
      .join("\n");

    expect(content).toContain("respondAgentServerRequestCurrent");
    expect(content).toContain('"app_server_drain_events"');
    expect(content).not.toContain("agentSession/action/respond");
  });

  it("keeps multi-agent execution on the six per-turn AgentControl tools", () => {
    const content = fs.readFileSync(
      "scripts/agent-runtime/tool-execution-smoke.mjs",
      "utf8",
    );

    for (const toolName of [
      "spawn_agent",
      "send_message",
      "followup_task",
      "wait_agent",
      "interrupt_agent",
      "list_agents",
    ]) {
      expect(content).toContain(`"${toolName}"`);
    }
    for (const retiredName of [
      "TeamCreate",
      "TeamDelete",
      "ListPeers",
      "SendMessage",
      "AgentTool",
    ]) {
      expect(content).not.toContain(`"${retiredName}"`);
    }

    expect(content).toContain("usesAppServerToolInventoryCurrent: true");
    expect(content).not.toContain("usesCompatToolInventoryCommand");
    expect(content).not.toContain("collabOperationToToolName");
    expect(content).not.toContain(
      'item?.type === "subagent_activity" ? item?.status_label',
    );
  });

  it("requires six completed AgentControl rows and canonical SubAgent activity in visible DOM", () => {
    const content = readAgentControlGateBSources();

    expect(content).toContain(AGENT_CONTROL_FINAL_TEXT);
    expect(content).toContain('method === "thread/read"');
    expect(content).toContain('transport === "electron-ipc"');
    expect(content).toContain("stage=cold-restart-electron");
    expect(content).toContain("--cold-restart");
    expect(content).toContain("visibleDomRestoredAfterColdRestart");
    expect(content).toContain("visibleDomToolIdentityStableAcrossRestart");
    expect(content).toContain("visibleDomSubAgentIdentityStableAcrossRestart");
    expect(content).toContain("visibleDomChildThreadStableAcrossRestart");
    expect(content).toContain("visibleDomParentOwnedComposerDisabled");
    expect(content).toContain("visibleDomParentOwnedServerRejectsDirectTurn");
    expect(content).toContain('"thread/read"');
    expect(content).toContain('"turn/start"');
    expect(content).toContain(PARENT_OWNED_DIRECT_INPUT_ERROR);
    expect(content).toContain('data-testid="subagent-activity-row"');
    expect(content).toContain('data-testid$=":subagent"');
    expect(content).toContain("visibleDomAllAgentControlToolRowsCompleted");
    expect(content).toContain("visibleDomCanonicalSubAgentActivitiesVisible");
    for (const toolName of AGENT_CONTROL_TOOL_NAMES) {
      expect(content).toContain(`"${toolName}"`);
    }
    for (const activityKind of AGENT_CONTROL_SUBAGENT_ACTIVITY_KINDS) {
      expect(content).toContain(`"${activityKind}"`);
    }
  });

  it("keeps the deferred MCP Gate B on the current Electron/App Server path", () => {
    const content = readDeferredGateBSources();

    expect(content).toContain("mcp-deferred-tool-search-gate-b");
    expect(content).toContain("deferred_loading: true");
    expect(content).toContain("always_visible: false");
    expect(content).toContain('"mcpServer/create"');
    expect(content).toContain('"mcpServer/start"');
    expect(content).toContain('"mcpServer/stop"');
    expect(content).toContain('"mcpServer/delete"');
    expect(content).toContain("startAgentSessionTurnCurrent");
  });

  it("requires the provider request boundary and new-Turn isolation assertions", () => {
    const content = readDeferredGateBSources();

    expect(content).toContain(
      "providerRequestBeforeSelectionHidesDeferredTool",
    );
    expect(content).toContain("sameTurnNextStepExposesDeferredTool");
    expect(content).toContain("newTurnDoesNotLeakDeferredTool");
    expect(content).toContain("runDeferredMcpNewTurnIsolation");
    expect(content).toContain("visibleDomToolSearchCompletedInReadModel");
    expect(content).toContain("visibleDomToolSearchStaysInternal");
    expect(content).not.toContain("visibleDomToolSearchRowCompleted");
    expect(content).toContain("visibleDomDeferredToolRowCompleted");
    expect(content).toContain(DEFERRED_MCP_TOOL_SEARCH_FINAL_TEXT);
    expect(content).toContain('method === "thread/read"');
    expect(content).toContain('transport === "electron-ipc"');
    expect(content).toContain('getAttribute("data-tool-name")');
    expect(content).toContain('getAttribute("data-tool-status")');
    expect(content).not.toContain("humanizeDeferredToolName");
    expect(content).not.toContain("tool-call-tool-search-result");
    expect(content).not.toContain("completedProcessIndicator");
    expect(content).toContain("message-list-historical-timeline-preview:");
    expect(content).toContain("agent-thread-block:");
  });

  it("accepts only the next-step and Turn-local deferred tool lifecycle", () => {
    const deferredToolName = "mcp__fixture__deferred_echo";
    const scriptedResponses = buildDeferredMcpToolSearchFixtureResponses({
      deferredToolName,
      toolCall: (name, id, argumentsPayload) => ({
        type: "tool_call",
        name,
        id,
        arguments: argumentsPayload,
      }),
    });
    const assertions = buildDeferredMcpToolSearchAssertions({
      deferredToolName,
      evidencePackText: deferredToolName,
      providerRequests: [
        { toolNames: ["tool_search"] },
        { toolNames: ["tool_search", deferredToolName] },
      ],
      runtimeContext: {
        serverCreated: true,
        deferredToolFoundByCurrentSearch: true,
        deferredToolSearchMetadata: { deferredLoading: true },
      },
      toolOutputText: `${deferredToolName}:LIME_DEFERRED_MCP_TOOL_OK`,
      newTurnProviderRequests: [{ toolNames: ["tool_search"] }],
    });

    expect(scriptedResponses).toHaveLength(3);
    expect(scriptedResponses[0]).toMatchObject({
      name: "tool_search",
      id: DEFERRED_MCP_TOOL_SEARCH_CALL_ID,
      arguments: { query: `select:${deferredToolName}` },
    });
    expect(scriptedResponses[1]).toMatchObject({
      name: deferredToolName,
      id: DEFERRED_MCP_TOOL_CALL_ID,
    });
    expect(Object.values(assertions)).toEqual(
      Array(Object.keys(assertions).length).fill(true),
    );
  });

  it("requires internal tool discovery, the completed deferred Tool row, final text, and current read trace", () => {
    const deferredToolName = "mcp__fixture__deferred_echo";
    const evidence = {
      runtime: {
        matrix: [
          { tool: "tool_search", status: "completed", success: true },
          { tool: deferredToolName, status: "completed", success: true },
        ],
      },
    };
    const snapshot = {
      electron: true,
      hasInvokeBridge: true,
      supportsAppServer: true,
      sessionId: "session-deferred",
      activeSessionId: "session-deferred",
      appServerCalls: [
        {
          method: "thread/read",
          transport: "electron-ipc",
          status: "success",
        },
      ],
      typedToolRows: [
        {
          id: "item-deferred",
          name: deferredToolName,
          status: "completed",
          visible: true,
        },
      ],
      deferredToolRow: {
        visible: true,
        completed: true,
        toolName: deferredToolName,
        toolStatus: "completed",
      },
      finalAssistantTextVisible: true,
      invokeErrorCount: 0,
      consoleErrorCount: 0,
    };

    const passing = buildDeferredMcpVisibleDomAssertions({
      deferredToolName,
      evidence,
      snapshot,
    });
    expect(Object.values(passing)).toEqual(
      Array(Object.keys(passing).length).fill(true),
    );

    const failing = buildDeferredMcpVisibleDomAssertions({
      deferredToolName,
      evidence,
      snapshot: {
        ...snapshot,
        activeSessionId: "session-other",
        finalAssistantTextVisible: false,
        deferredToolRow: {
          ...snapshot.deferredToolRow,
          toolStatus: "failed",
        },
      },
    });
    expect(failing.visibleDomNavigatedToTargetSession).toBe(false);
    expect(failing.visibleDomFinalAssistantTextVisible).toBe(false);
    expect(failing.visibleDomDeferredToolRowCompleted).toBe(false);

    const leakingInternalSearch = buildDeferredMcpVisibleDomAssertions({
      deferredToolName,
      evidence,
      snapshot: {
        ...snapshot,
        typedToolRows: [
          ...snapshot.typedToolRows,
          {
            id: "item-tool-search",
            name: "tool_search",
            status: "completed",
            visible: true,
          },
        ],
      },
    });
    expect(leakingInternalSearch.visibleDomToolSearchStaysInternal).toBe(false);
  });

  it("fails AgentControl visible DOM when a canonical row or activity identity is missing", () => {
    const evidence = {
      runtime: {
        matrix: AGENT_CONTROL_TOOL_NAMES.map((tool) => ({
          tool,
          status: "completed",
          success: true,
        })),
      },
    };
    const snapshot = {
      electron: true,
      hasInvokeBridge: true,
      supportsAppServer: true,
      coldRestart: {
        electronProcessReplaced: true,
      },
      sessionId: "session-agent-control",
      activeSessionId: "session-agent-control",
      appServerCalls: [
        {
          method: "thread/read",
          transport: "electron-ipc",
          status: "success",
        },
        {
          method: "thread/list",
          transport: "electron-ipc",
          status: "success",
        },
      ],
      typedToolRows: AGENT_CONTROL_TOOL_NAMES.map((name, index) => ({
        id: `tool-${index}`,
        name,
        status: "completed",
        visible: true,
      })),
      subagentActivityRows: AGENT_CONTROL_SUBAGENT_ACTIVITY_KINDS.map(
        (activityKind, index) => ({
          itemId: `activity-${index}`,
          activityKind,
          threadId: "thread-child",
          visible: true,
        }),
      ),
      parentOwnedChild: {
        childThreadId: "thread-child",
        canonicalThread: {
          id: "thread-child",
          sessionId: "session-child",
          parentThreadId: "session-agent-control",
          canAcceptDirectInput: false,
        },
        dom: {
          activeSessionId: "session-child",
          childThreadId: "thread-child",
          textareaVisible: true,
          textareaDisabled: true,
          placeholder: "此子线程由父线程管理，无法直接输入",
          controls: {
            sendDisabled: true,
            accessModeDisabled: true,
            modelSelectorCount: 1,
            modelSelectorsDisabled: true,
            taskModeDisabled: true,
          },
        },
        uiAttempt: {
          dispatchedEnter: true,
          clickedDisabledSend: true,
          turnStartCountBefore: 0,
          turnStartCountAfter: 0,
        },
        serverRejection: {
          code: -32600,
          message: PARENT_OWNED_DIRECT_INPUT_ERROR,
          hasResult: false,
        },
      },
      finalAssistantTextVisible: true,
      invokeErrorCount: 0,
      consoleErrorCount: 0,
    };
    snapshot.preRestart = {
      activeSessionId: snapshot.activeSessionId,
      typedToolRows: snapshot.typedToolRows,
      subagentActivityRows: snapshot.subagentActivityRows,
      finalAssistantTextVisible: true,
    };

    const passing = buildAgentControlVisibleDomAssertions({
      evidence,
      snapshot,
    });
    expect(Object.values(passing)).toEqual(
      Array(Object.keys(passing).length).fill(true),
    );

    const missingTool = buildAgentControlVisibleDomAssertions({
      evidence,
      snapshot: {
        ...snapshot,
        typedToolRows: snapshot.typedToolRows.slice(1),
      },
    });
    expect(missingTool.visibleDomAllAgentControlToolRowsPresentOnce).toBe(
      false,
    );
    expect(missingTool.visibleDomAllAgentControlToolRowsVisible).toBe(false);
    expect(missingTool.visibleDomToolIdentityStableAcrossRestart).toBe(false);

    const hotReloadOnly = buildAgentControlVisibleDomAssertions({
      evidence,
      snapshot: {
        ...snapshot,
        coldRestart: null,
      },
    });
    expect(hotReloadOnly.visibleDomRestoredAfterColdRestart).toBe(false);

    const missingIdentity = buildAgentControlVisibleDomAssertions({
      evidence,
      snapshot: {
        ...snapshot,
        subagentActivityRows: snapshot.subagentActivityRows.map((row, index) =>
          index === 0 ? { ...row, threadId: "" } : row,
        ),
      },
    });
    expect(missingIdentity.visibleDomCanonicalSubAgentActivitiesVisible).toBe(
      false,
    );
    expect(
      missingIdentity.visibleDomSubAgentActivitiesUseCanonicalIdentity,
    ).toBe(false);

    const directInputLeaked = buildAgentControlVisibleDomAssertions({
      evidence,
      snapshot: {
        ...snapshot,
        parentOwnedChild: {
          ...snapshot.parentOwnedChild,
          uiAttempt: {
            ...snapshot.parentOwnedChild.uiAttempt,
            turnStartCountAfter: 1,
          },
        },
      },
    });
    expect(directInputLeaked.visibleDomParentOwnedUiAttemptDidNotStartTurn).toBe(
      false,
    );

    const serverAccepted = buildAgentControlVisibleDomAssertions({
      evidence,
      snapshot: {
        ...snapshot,
        parentOwnedChild: {
          ...snapshot.parentOwnedChild,
          serverRejection: { code: null, message: null, hasResult: true },
        },
      },
    });
    expect(serverAccepted.visibleDomParentOwnedServerRejectsDirectTurn).toBe(
      false,
    );
  });
});
