import { describe, expect, it } from "vitest";
import type { AppServerAgentEvent } from "@/lib/api/appServer";
import { readCanonicalThreadItem } from "./appServerCanonicalItemReader";

const event: AppServerAgentEvent = {
  eventId: "event-1",
  sequence: 7,
  sessionId: "session-1",
  threadId: "thread-1",
  turnId: "turn-1",
  type: "item.started",
  timestamp: "2026-07-13T00:00:02.000Z",
  payload: {},
};

function item(value: Record<string, unknown>): Record<string, unknown> {
  return {
    id: "item-1",
    ...value,
  };
}

describe("readCanonicalThreadItem", () => {
  it("rejects internal snake_case detail items at the Codex v2 boundary", () => {
    for (const type of [
      "user_message",
      "agent_message",
      "command_execution",
      "file_artifact",
      "tool_call",
      "subagent_activity",
      "context_compaction",
      "web_search",
      "media",
    ]) {
      expect(readCanonicalThreadItem(item({ type }), event)).toBeNull();
    }
  });

  it("projects an unknown canonical v2 item without retaining raw values", () => {
    const projected = readCanonicalThreadItem(
      item({
        type: "futureCapability",
        label: "safe-but-unsupported-value",
        password: "secret-value",
        metadata: { rawPayload: "must-not-survive" },
        "invalid field": "must-not-survive-either",
      }),
      event,
    );

    expect(projected).toMatchObject({
      id: "item-1",
      type: "unknown_item",
      upstream_type: "futureCapability",
      field_names: ["[redacted]", "label", "metadata"],
      status: "in_progress",
    });
    expect(projected).not.toHaveProperty("metadata");
    expect(JSON.stringify(projected)).not.toContain(
      "safe-but-unsupported-value",
    );
    expect(JSON.stringify(projected)).not.toContain("secret-value");
    expect(JSON.stringify(projected)).not.toContain("must-not-survive");
  });

  it.each([
    [
      "userMessage",
      {
        type: "userMessage",
        clientId: "client-1",
        content: [
          {
            type: "text",
            text: "hello",
            text_elements: [
              {
                byteRange: { start: 0, end: 5 },
                placeholder: "greeting",
              },
            ],
          },
          {
            type: "image",
            url: "https://example.test/input",
            detail: "high",
          },
          {
            type: "image",
            url: "data:image/gif;base64,R0lGODlh",
            detail: "low",
          },
          { type: "localImage", path: "/tmp/local.webp", detail: "auto" },
          { type: "skill", name: "review", path: "/skills/review" },
          { type: "mention", name: "README", path: "README.md" },
        ],
      },
      {
        type: "user_message",
        content: "hello",
        content_parts: [
          {
            type: "text",
            text: "hello",
            text_elements: [
              {
                byte_range: { start: 0, end: 5 },
                placeholder: "greeting",
              },
            ],
          },
          {
            type: "image",
            mime_type: "image/*",
            data: "",
            uri: "https://example.test/input",
            detail: "high",
          },
          {
            type: "image",
            mime_type: "image/gif",
            data: "R0lGODlh",
            detail: "low",
          },
          {
            type: "image",
            mime_type: "image/webp",
            data: "",
            display_name: "local.webp",
            unavailable_reason: "host_reference_required",
            detail: "auto",
          },
          { type: "skill", name: "review", path: "review" },
          { type: "mention", name: "README", path: "README.md" },
        ],
        client_id: "client-1",
        status: "in_progress",
      },
    ],
    [
      "agentMessage",
      {
        type: "agentMessage",
        text: "answer",
        phase: "final_answer",
      },
      {
        type: "agent_message",
        text: "answer",
        phase: "final_answer",
        status: "in_progress",
      },
    ],
    [
      "plan",
      { type: "plan", text: "- [ ] 验证历史恢复" },
      {
        type: "plan",
        text: "- [ ] 验证历史恢复",
        status: "in_progress",
      },
    ],
    [
      "reasoning",
      { type: "reasoning", summary: ["summary"], content: ["a", "b"] },
      {
        type: "reasoning",
        text: "summary",
        summary: ["summary"],
        content: ["a", "b"],
        status: "in_progress",
      },
    ],
    [
      "commandExecution",
      {
        type: "commandExecution",
        command: "npm test",
        cwd: "/repo",
        status: "completed",
        pluginId: "plugin-docs",
        scriptPath: "scripts/test.mjs",
        processId: "process-1",
        source: "agent",
        commandActions: [{ type: "read", path: "package.json" }],
        aggregatedOutput: "passed",
        exitCode: 0,
        durationMs: 18,
      },
      {
        type: "command_execution",
        command: "npm test",
        cwd: "repo",
        plugin_id: "plugin-docs",
        script_path: "scripts/test.mjs",
        process_id: "process-1",
        source: "agent",
        command_actions: [{ type: "read", path: "package.json" }],
        aggregated_output: "passed",
        exit_code: 0,
        duration_ms: 18,
        status: "completed",
        completed_at: "2026-07-13T00:00:02.000Z",
      },
    ],
    [
      "fileChange",
      {
        type: "fileChange",
        changes: [
          { path: "src/app.ts", kind: { type: "add" }, diff: "+ok" },
          {
            path: "src/old.ts",
            kind: { type: "update", move_path: "src/new.ts" },
            diff: "-old\n+new",
          },
        ],
        status: "completed",
      },
      {
        type: "patch",
        changes: [
          { path: "src/app.ts", kind: { type: "add" }, diff: "+ok" },
          {
            path: "src/old.ts",
            kind: { type: "update", move_path: "src/new.ts" },
            diff: "-old\n+new",
          },
        ],
        paths: ["src/app.ts", "src/old.ts"],
        success: true,
        file_status: "completed",
        status: "completed",
        completed_at: "2026-07-13T00:00:02.000Z",
      },
    ],
    [
      "imageView",
      { type: "imageView", path: "/tmp/result.png" },
      {
        type: "media",
        uri: "",
        mime_type: "image/png",
        preview: "result.png",
        status: "in_progress",
      },
    ],
    [
      "imageView sidecar",
      {
        type: "imageView",
        path: "sidecar://media/output-deadbeef/fixture%20result.png",
      },
      {
        type: "media",
        uri: "sidecar://media/output-deadbeef/fixture%20result.png",
        mime_type: "image/png",
        preview: "fixture result.png",
        status: "in_progress",
      },
    ],
    [
      "hookPrompt",
      {
        type: "hookPrompt",
        fragments: [
          { hookRunId: "hook-run-1", text: "Run the focused checks." },
          { hookRunId: "hook-run-2", text: "Then summarize." },
        ],
      },
      {
        type: "hook_prompt",
        fragments: [
          { hook_run_id: "hook-run-1", text: "Run the focused checks." },
          { hook_run_id: "hook-run-2", text: "Then summarize." },
        ],
        status: "in_progress",
      },
    ],
    [
      "sleep",
      { type: "sleep", durationMs: 1250 },
      { type: "sleep", duration_ms: 1250, status: "in_progress" },
    ],
    [
      "enteredReviewMode",
      { type: "enteredReviewMode", review: "Review the current changes." },
      {
        type: "review_boundary",
        boundary: "entered",
        review: "Review the current changes.",
        status: "in_progress",
      },
    ],
    [
      "exitedReviewMode",
      { type: "exitedReviewMode", review: "No blocking findings." },
      {
        type: "review_boundary",
        boundary: "exited",
        review: "No blocking findings.",
        status: "in_progress",
      },
    ],
    [
      "contextCompaction",
      { type: "contextCompaction" },
      {
        type: "context_compaction",
        stage: "started",
        status: "in_progress",
      },
    ],
  ])(
    "projects canonical v2 %s without entity-envelope shadow fields",
    (_name, value, expected) => {
      const projected = readCanonicalThreadItem(item(value), event);
      expect(projected).toMatchObject({
        id: "item-1",
        thread_id: "thread-1",
        turn_id: "turn-1",
        sequence: 7,
        started_at: "2026-07-13T00:00:02.000Z",
        updated_at: "2026-07-13T00:00:02.000Z",
        ...(expected as Record<string, unknown>),
      });
      expect(projected).not.toHaveProperty("ordinal");
      expect(projected).not.toHaveProperty("metadata");
    },
  );

  it.each([
    { type: "audio", url: "https://example.test/input.wav" },
    { type: "localAudio", path: "/tmp/local.wav" },
  ])("rejects unsupported UserInput audio shape %#", (content) => {
    expect(
      readCanonicalThreadItem(
        item({ type: "userMessage", content: [content] }),
        event,
      ),
    ).toBeNull();
  });

  it.each([
    { type: "hookPrompt", fragments: [{ hookRunId: "", text: "x" }] },
    { type: "sleep", durationMs: -1 },
    { type: "sleep", durationMs: 1.5 },
    { type: "enteredReviewMode" },
  ])("rejects malformed typed canonical item %#", (value) => {
    expect(readCanonicalThreadItem(item(value), event)).toBeNull();
  });

  it("keeps raw-only reasoning protocol data out of the default display text", () => {
    expect(
      readCanonicalThreadItem(
        item({ type: "reasoning", summary: [], content: ["raw reasoning"] }),
        event,
      ),
    ).toMatchObject({
      type: "reasoning",
      text: "",
      summary: [],
      content: ["raw reasoning"],
    });
  });

  it("derives the current plan revision from a canonical v2 item identity", () => {
    const projected = readCanonicalThreadItem(
      item({
        id: "plan_turn-1_proposed_plan:fixture-1",
        type: "plan",
        text: "- [ ] 验证历史恢复",
      }),
      event,
    );
    expect(projected).toMatchObject({
      type: "plan",
      metadata: {
        revisionId: "proposed_plan:fixture-1",
        source: "proposed_plan",
      },
    });
  });

  it.each(["started", "interacted", "interrupted"] as const)(
    "projects canonical v2 SubAgent %s activity without rewriting child thread identity",
    (kind) => {
      const projected = readCanonicalThreadItem(
        item({
          type: "subAgentActivity",
          kind,
          agentThreadId: "thread-child",
          agentPath: "root/child",
        }),
        event,
      );
      expect(projected).toMatchObject({
        id: "item-1",
        type: "subagent_activity",
        session_id: "thread-child",
        status_label: kind,
        metadata: { agentPath: "root/child" },
      });
      expect(projected).not.toHaveProperty("summary");
    },
  );

  it("projects a canonical v2 dynamic tool call with stable UI identity", () => {
    const contentItems = [
      { type: "inputText", text: "score:9" },
      { type: "inputAudio", audioUrl: "https://example.test/score.wav" },
    ];
    const projected = readCanonicalThreadItem(
      item({
        type: "dynamicToolCall",
        namespace: "review",
        tool: "review",
        arguments: { score: 9 },
        status: "completed",
        contentItems,
        success: true,
        durationMs: 12,
      }),
      event,
    );

    expect(projected).toMatchObject({
      id: "item-1",
      type: "tool_call",
      tool_name: "review",
      arguments: { score: 9 },
      output: "score:9",
      duration_ms: 12,
      success: true,
      status: "completed",
      started_at: "2026-07-13T00:00:02.000Z",
      completed_at: "2026-07-13T00:00:02.000Z",
      metadata: {
        canonical_type: "dynamicToolCall",
        namespace: "review",
        content_items: contentItems,
        callId: "item-1",
      },
    });
    expect(projected).not.toHaveProperty("ordinal");
  });

  it("preserves canonical v2 MCP context and structured result blocks", () => {
    const content = [{ type: "text", text: "found" }];
    expect(
      readCanonicalThreadItem(
        item({
          type: "mcpToolCall",
          server: "docs",
          tool: "search",
          arguments: { query: "ThreadItem" },
          status: "completed",
          appContext: {
            connectorId: "connector-docs",
            resourceUri: "docs://codex",
          },
          pluginId: "plugin-docs",
          result: {
            content,
            structuredContent: { matches: 3 },
            _meta: { requestId: "request-1" },
          },
          durationMs: 24,
        }),
        event,
      ),
    ).toMatchObject({
      id: "item-1",
      type: "tool_call",
      tool_name: "search",
      output: "found",
      structured_content: { matches: 3 },
      duration_ms: 24,
      success: true,
      metadata: {
        callId: "item-1",
        canonical_type: "mcpToolCall",
        server: "docs",
        app_context: {
          connectorId: "connector-docs",
          resourceUri: "docs://codex",
        },
        plugin_id: "plugin-docs",
        result_content: content,
        result_meta: { requestId: "request-1" },
      },
    });
  });

  it("projects a canonical v2 collab wait lifecycle as a Tool item", () => {
    expect(
      readCanonicalThreadItem(
        item({
          type: "collabAgentToolCall",
          tool: "wait",
          status: "completed",
          senderThreadId: "thread-1",
          receiverThreadIds: ["thread-child"],
          prompt: null,
          model: null,
          reasoningEffort: null,
          agentsStates: {
            "thread-child": { status: "completed", message: "done" },
          },
        }),
        event,
      ),
    ).toMatchObject({
      id: "item-1",
      type: "tool_call",
      tool_name: "wait",
      status: "completed",
      success: true,
      metadata: {
        callId: "item-1",
        canonical_type: "collabAgentToolCall",
        sender_thread_id: "thread-1",
        receiver_thread_ids: ["thread-child"],
        agents_states: {
          "thread-child": { status: "completed", message: "done" },
        },
      },
    });
  });

  it("preserves canonical v2 WebSearch action and opaque results", () => {
    const action = {
      type: "search",
      query: "Codex ThreadItem",
      queries: ["Codex ThreadItem", "Codex app server v2"],
    };
    const results = [
      { title: "ThreadItem", url: "https://example.test/thread-item" },
    ];
    expect(
      readCanonicalThreadItem(
        item({
          type: "webSearch",
          query: "Codex ThreadItem",
          action,
          results,
        }),
        { ...event, type: "item.completed" },
      ),
    ).toMatchObject({
      id: "item-1",
      type: "web_search",
      query: "Codex ThreadItem",
      action: "search",
      action_data: action,
      results,
      status: "completed",
    });
  });

  it("projects canonical image generation as a dedicated read-model item", () => {
    expect(
      readCanonicalThreadItem(
        item({
          type: "imageGeneration",
          status: "completed",
          revisedPrompt: "a blue square",
          result: "Zm9v",
          savedPath: "/tmp/blue-square.png",
        }),
        { ...event, type: "item.completed" },
      ),
    ).toMatchObject({
      id: "item-1",
      type: "image_generation",
      status: "completed",
      generation_status: "completed",
      revised_prompt: "a blue square",
      result: "Zm9v",
      saved_path: "/tmp/blue-square.png",
    });
  });

  it("rejects image generation items without required status or result", () => {
    expect(
      readCanonicalThreadItem(
        item({ type: "imageGeneration", result: "Zm9v" }),
        event,
      ),
    ).toBeNull();
    expect(
      readCanonicalThreadItem(
        item({ type: "imageGeneration", status: "completed" }),
        event,
      ),
    ).toBeNull();
  });

  it("takes routing, sequence, and lifecycle timestamps from the raw event envelope", () => {
    const projected = readCanonicalThreadItem(
      item({ type: "agentMessage", text: "answer" }),
      {
        ...event,
        sequence: 41,
        threadId: "thread-envelope",
        turnId: "turn-envelope",
        timestamp: "2026-07-13T00:01:00.000Z",
      },
    );

    expect(projected).toMatchObject({
      id: "item-1",
      thread_id: "thread-envelope",
      turn_id: "turn-envelope",
      sequence: 41,
      started_at: "2026-07-13T00:01:00.000Z",
      updated_at: "2026-07-13T00:01:00.000Z",
    });
    expect(projected).not.toHaveProperty("ordinal");
    expect(projected).not.toHaveProperty("metadata");
    expect(projected).not.toHaveProperty("completed_at");
  });

  it("preserves declined command and patch outcomes without lowering them to failed", () => {
    const completedEvent = {
      ...event,
      type: "item.completed",
    } as const;

    expect(
      readCanonicalThreadItem(
        item({
          type: "commandExecution",
          command: "npm test",
          cwd: "/repo",
          status: "declined",
        }),
        completedEvent,
      ),
    ).toMatchObject({
      type: "command_execution",
      status: "completed",
      command_status: "declined",
      completed_at: event.timestamp,
    });
    expect(
      readCanonicalThreadItem(
        item({
          type: "fileChange",
          changes: [{ path: "src/app.ts" }],
          status: "declined",
        }),
        completedEvent,
      ),
    ).toMatchObject({
      type: "patch",
      status: "completed",
      file_status: "declined",
      success: false,
      completed_at: event.timestamp,
    });
  });

  it("derives terminal lifecycle from the raw event envelope", () => {
    expect(
      readCanonicalThreadItem(item({ type: "contextCompaction" }), {
        ...event,
        type: "item.completed",
        timestamp: "2026-07-13T00:02:00.000Z",
      }),
    ).toMatchObject({
      id: "item-1",
      type: "context_compaction",
      status: "completed",
      stage: "completed",
      started_at: "2026-07-13T00:02:00.000Z",
      updated_at: "2026-07-13T00:02:00.000Z",
      completed_at: "2026-07-13T00:02:00.000Z",
    });
  });

  it("fails closed on malformed canonical v2 variants", () => {
    for (const malformed of [
      { id: "", type: "agentMessage", text: "ignored" },
      { id: "item-1", type: "agentMessage" },
      { id: "item-1", type: "userMessage", content: { text: "ignored" } },
      { id: "item-1", type: "userMessage", content: [] },
      {
        id: "item-1",
        type: "userMessage",
        content: [{ type: "unknown", text: "ignored" }],
      },
      {
        id: "item-1",
        type: "userMessage",
        content: [{ type: "skill", name: "review" }],
      },
      {
        id: "item-1",
        type: "userMessage",
        content: [
          { type: "image", url: "data:image/png;base64,AA==", detail: "max" },
        ],
      },
      {
        id: "item-1",
        type: "userMessage",
        content: [
          {
            type: "text",
            text: "hello",
            text_elements: [{ byteRange: { start: 0, end: 6 } }],
          },
        ],
      },
      {
        id: "item-1",
        type: "userMessage",
        content: [
          {
            type: "text",
            text: "hello",
            textElements: [{ byteRange: { start: 0, end: 5 } }],
          },
        ],
      },
      { id: "item-1", type: "commandExecution", command: "npm test" },
      {
        id: "item-1",
        type: "mcpToolCall",
        tool: "search",
        arguments: {},
        status: "completed",
      },
      {
        id: "item-1",
        type: "dynamicToolCall",
        status: "declined",
        tool: "review",
        arguments: {},
      },
      {
        id: "item-1",
        type: "commandExecution",
        command: "npm test",
        cwd: "/repo",
        status: "running",
      },
      {
        id: "item-1",
        type: "collabAgentToolCall",
        tool: "wait",
        status: "completed",
        senderThreadId: "thread-1",
        receiverThreadIds: [],
      },
      {
        id: "item-1",
        type: "webSearch",
        query: "Codex",
        results: { title: "invalid" },
      },
    ]) {
      expect(readCanonicalThreadItem(malformed, event)).toBeNull();
    }
  });
});
