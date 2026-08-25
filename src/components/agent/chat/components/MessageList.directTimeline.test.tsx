import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  mockAgentThreadTimeline,
  mockStreamingRenderer,
  render,
} from "./MessageList.testHarness";
import type {
  AgentThreadItem,
  AgentThreadTurn,
  Message,
} from "./MessageList.testHarness";

const timestamp = "2026-07-29T04:00:00.000Z";

function turn(
  id: string,
  status: AgentThreadTurn["status"] = "completed",
): AgentThreadTurn {
  return {
    id,
    thread_id: "thread-direct",
    prompt_text: "",
    status,
    started_at: timestamp,
    completed_at: status === "running" ? undefined : timestamp,
    created_at: timestamp,
    updated_at: timestamp,
  };
}

function item(
  id: string,
  turnId: string,
  sequence: number,
  value: Partial<AgentThreadItem> & Pick<AgentThreadItem, "type">,
): AgentThreadItem {
  return {
    id,
    thread_id: "thread-direct",
    turn_id: turnId,
    sequence,
    status: "completed",
    started_at: timestamp,
    completed_at: timestamp,
    updated_at: timestamp,
    ...value,
  } as AgentThreadItem;
}

describe("MessageList direct canonical timeline", () => {
  it("已完成 canonical 用户回合应确认后按同一 Thread/Turn 恢复历史", async () => {
    const completedTurn = turn("turn-revert-target");
    const onRevertThread = vi.fn(async () => undefined);
    const container = render([], {
      currentTurnId: completedTurn.id,
      turns: [completedTurn],
      threadItems: [
        item("revert-user", completedTurn.id, 1, {
          type: "user_message",
          content: "从这里重新开始",
        }),
        item("revert-agent", completedTurn.id, 2, {
          type: "agent_message",
          text: "旧答复",
        }),
      ],
      threadRead: {
        thread_id: "thread-direct",
        can_accept_direct_input: true,
      },
      onRevertThread,
    });

    const trigger = container.querySelector<HTMLButtonElement>(
      '[data-testid="thread-revert-trigger"]',
    );
    expect(trigger?.dataset.threadId).toBe("thread-direct");
    expect(trigger?.dataset.beforeTurnId).toBe("turn-revert-target");

    act(() => trigger?.click());
    expect(document.body.textContent).toContain(
      "Local files are not rolled back",
    );
    expect(document.body.textContent).toContain("current thread stays");

    const confirm = document.querySelector<HTMLButtonElement>(
      '[data-testid="thread-revert-confirm"]',
    );
    await act(async () => {
      confirm?.click();
      await Promise.resolve();
    });

    expect(onRevertThread).toHaveBeenCalledWith({
      threadId: "thread-direct",
      beforeTurnId: "turn-revert-target",
    });
    expect(
      document
        .querySelector('[data-testid="thread-revert-status"]')
        ?.getAttribute("data-state"),
    ).toBe("success");
  });

  it("messages 为空时仍按 User -> Agent -> Process -> Agent -> Process 顺序渲染", () => {
    const currentTurn = turn("turn-direct", "running");
    const container = render([], {
      currentTurnId: currentTurn.id,
      turns: [currentTurn],
      threadItems: [
        item("tool-after", currentTurn.id, 5, {
          type: "tool_call",
          tool_name: "read_file",
        }),
        item("agent-final", currentTurn.id, 4, {
          type: "agent_message",
          text: "canonical final answer",
        }),
        item("command", currentTurn.id, 3, {
          type: "command_execution",
          command: "pwd",
          cwd: "/repo",
        }),
        item("agent-commentary", currentTurn.id, 2, {
          type: "agent_message",
          text: "canonical commentary",
          phase: "commentary",
        }),
        item("user", currentTurn.id, 1, {
          type: "user_message",
          content: "canonical user question",
        }),
      ],
    });

    const segments = Array.from(
      container.querySelectorAll<HTMLElement>("[data-direct-segment-kind]"),
    );
    expect(
      segments.map((segment) => segment.dataset.directSegmentKind),
    ).toEqual(["message", "message", "process", "message", "process"]);
    expect(segments.map((segment) => segment.textContent)).toEqual([
      expect.stringContaining("canonical user question"),
      expect.stringContaining("canonical commentary"),
      expect.stringContaining("执行轨迹"),
      expect.stringContaining("canonical final answer"),
      expect.stringContaining("执行轨迹"),
    ]);
    expect(
      container.textContent?.match(/canonical final answer/g),
    ).toHaveLength(1);

    const group = container.querySelector<HTMLElement>(
      '[data-testid="message-turn-group"]',
    );
    expect(group?.dataset.renderEntryKind).toBe("canonical_turn");
    expect(group?.dataset.runtimeTurnId).toBe(currentTurn.id);
    expect(
      container.querySelector('[data-thread-item-id="user"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-thread-item-id="agent-final"]'),
    ).not.toBeNull();

    const timelineCalls = mockAgentThreadTimeline.mock.calls.map(
      ([props]) =>
        (props as { showInlineStatusHint?: boolean }).showInlineStatusHint,
    );
    expect(timelineCalls).toEqual([true, false]);
  });

  it("已完成 canonical Turn 默认显示摘要，点击后按 Item identity 展开完整过程", () => {
    const completedTurn = turn("turn-direct-completed");
    const container = render([], {
      currentTurnId: completedTurn.id,
      turns: [completedTurn],
      threadItems: [
        item("completed-command", completedTurn.id, 1, {
          type: "command_execution",
          command: "npm test",
          cwd: "/repo",
        }),
      ],
    });

    const preview = container.querySelector<HTMLButtonElement>(
      '[data-testid="message-list-historical-timeline-preview:leading"]',
    );
    expect(preview).not.toBeNull();
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();

    act(() => preview?.click());

    expect(
      container.querySelector('[data-testid="agent-thread-timeline:leading"]'),
    ).not.toBeNull();
    expect(mockAgentThreadTimeline).toHaveBeenCalledWith(
      expect.objectContaining({
        expandCompletedProcessDetails: true,
        items: [expect.objectContaining({ id: "completed-command" })],
      }),
    );
  });

  it("已结束回合合并多个过程段，并只在最后一条 assistant 正文保留操作栏", () => {
    const completedTurn = turn("turn-merged-history");
    completedTurn.completed_at = "2026-07-29T04:00:02.000Z";
    completedTurn.updated_at = completedTurn.completed_at;
    const container = render([], {
      currentTurnId: completedTurn.id,
      onQuoteMessage: vi.fn(),
      turns: [completedTurn],
      threadItems: [
        item("merged-user", completedTurn.id, 1, {
          type: "user_message",
          content: "请检查并修复这个回合",
        }),
        item("merged-commentary", completedTurn.id, 2, {
          type: "agent_message",
          text: "我先检查相关文件。",
          phase: "commentary",
        }),
        item("merged-command", completedTurn.id, 3, {
          type: "command_execution",
          command: "npm test",
          cwd: "/repo",
        }),
        item("merged-final", completedTurn.id, 4, {
          type: "agent_message",
          text: "检查完成，问题已经修复。",
          phase: "final_answer",
        }),
        item("merged-tool", completedTurn.id, 5, {
          type: "tool_call",
          tool_name: "read_file",
        }),
      ],
    });

    const previews = container.querySelectorAll(
      '[data-testid="message-list-historical-timeline-preview:leading"]',
    );
    expect(previews).toHaveLength(1);
    expect(previews[0]?.textContent).toMatch(/(?:已处理|Processed)/);
    expect(
      container.querySelectorAll('[data-testid="message-actions"]'),
    ).toHaveLength(1);
  });

  it("已完成 canonical Turn 应常显脱敏后的未知 Item 诊断", () => {
    const completedTurn = turn("turn-direct-unknown-item");
    const container = render([], {
      currentTurnId: completedTurn.id,
      turns: [completedTurn],
      threadItems: [
        item("completed-unknown-item", completedTurn.id, 1, {
          type: "unknown_item",
          upstream_type: "futureCapability",
          field_names: ["[redacted]", "label", "status"],
        }),
      ],
    });

    const diagnostic = container.querySelector(
      '[data-testid="timeline-unsupported-item"]',
    );
    expect(diagnostic?.textContent).toContain("futureCapability");
    expect(diagnostic?.textContent).toContain("[redacted]");
    expect(diagnostic?.textContent).toContain("label");
    expect(diagnostic?.textContent).toContain("status");
    expect(container.textContent).not.toContain("unknown_item");
    expect(container.textContent).not.toContain("opaque-value-must-not-render");
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
  });

  it("聚焦 completed process Item 时直接展开并保留精确定位参数", () => {
    const completedTurn = turn("turn-focused-completed");
    render([], {
      currentTurnId: completedTurn.id,
      focusedTimelineItemId: "focused-command",
      timelineFocusRequestKey: 7,
      turns: [completedTurn],
      threadItems: [
        item("focused-command", completedTurn.id, 1, {
          type: "command_execution",
          command: "npm test",
          cwd: "/repo",
        }),
      ],
    });

    expect(mockAgentThreadTimeline).toHaveBeenCalledWith(
      expect.objectContaining({
        focusedItemId: "focused-command",
        focusRequestKey: 7,
      }),
    );
  });

  it("completed currentTurnId 的 canonical A2UI 与 action request 始终只读", () => {
    const completedTurn = turn("turn-completed-a2ui");
    const container = render([], {
      currentTurnId: completedTurn.id,
      turns: [completedTurn],
      threadItems: [
        item("completed-agent", completedTurn.id, 1, {
          type: "agent_message",
          text: "completed answer",
        }),
      ],
    });

    const renderer = container.querySelector<HTMLElement>(
      '[data-testid="streaming-renderer"]',
    );
    expect(renderer?.dataset.readOnlyA2ui).toBe("yes");
    expect(renderer?.dataset.readOnlyActionRequests).toBe("yes");
  });

  it("恢复窗口中的 direct canonical 长正文应先渲染有界预览并可显式展开", () => {
    const completedTurn = turn("turn-direct-long-history");
    const longContent = `# canonical long history\n\n${"bounded markdown row\n".repeat(1_600)}`;
    const container = render([], {
      currentTurnId: completedTurn.id,
      sessionHistoryWindow: {
        loadedMessages: 2,
        totalMessages: 480,
        isLoadingFull: false,
        error: null,
      },
      turns: [completedTurn],
      threadItems: [
        item("direct-long-agent", completedTurn.id, 1, {
          type: "agent_message",
          text: longContent,
        }),
      ],
    });

    const preview = container.querySelector<HTMLElement>(
      '[data-testid="message-list-long-history-preview"]',
    );
    expect(preview).not.toBeNull();
    expect(preview?.textContent).toContain("canonical long history");
    expect(preview?.textContent?.length).toBeLessThan(longContent.length);
    expect(mockStreamingRenderer).not.toHaveBeenCalledWith(
      expect.objectContaining({ content: longContent }),
    );

    act(() => {
      preview?.querySelector<HTMLButtonElement>("button")?.click();
    });

    expect(
      container.querySelector(
        '[data-testid="message-list-long-history-preview"]',
      ),
    ).toBeNull();
    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({
        content: longContent,
        isStreaming: false,
      }),
    );
  });

  it("direct User 保留 image、skill 与 mention content_parts", () => {
    const directTurn = turn("turn-user-parts", "running");
    const onOpenMessagePreview = vi.fn();
    const container = render([], {
      currentTurnId: directTurn.id,
      onOpenMessagePreview,
      sessionId: "session-user-parts",
      turns: [directTurn],
      threadItems: [
        item("user-parts", directTurn.id, 1, {
          type: "user_message",
          content: "inspect image",
          content_parts: [
            { type: "text", text: "inspect image" },
            { type: "skill", name: "image-review", path: "/skills/image" },
            { type: "mention", name: "reference.png", path: "/tmp/ref.png" },
            {
              type: "image",
              mime_type: "image/png",
              data: "aW1hZ2U=",
            },
            {
              type: "image",
              mime_type: "image/jpeg",
              data: "",
              source_path: "/tmp/imported-reference.jpg",
            },
          ],
        }),
      ],
    });

    expect(
      container.querySelectorAll('[data-testid="message-user-skill-content"]'),
    ).toHaveLength(2);
    expect(
      container.querySelectorAll('[data-testid^="message-image-attachment-"]'),
    ).toHaveLength(4);
    expect(
      container.querySelector('[data-testid="message-image-attachment-0"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="message-image-attachment-1"]'),
    ).not.toBeNull();

    act(() => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="message-image-attachment-open-0"]',
        )
        ?.click();
    });
    expect(onOpenMessagePreview).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "message_attachment",
        attachment: expect.objectContaining({ mediaType: "image/png" }),
      }),
      expect.objectContaining({
        id: "user-parts",
        role: "user",
        runtimeTurnId: directTurn.id,
      }),
    );
  });

  it("direct Agent 与独立 media Item 透传媒体预览入口", () => {
    const directTurn = turn("turn-direct-media", "running");
    const onOpenMessagePreview = vi.fn();
    const container = render([], {
      currentTurnId: directTurn.id,
      onOpenMessagePreview,
      turns: [directTurn],
      threadItems: [
        item("agent-media", directTurn.id, 1, {
          type: "agent_message",
          text: "media answer",
          contentParts: [
            {
              type: "media",
              kind: "image",
              reference: {
                uri: "https://example.com/agent.png",
                mime_type: "image/png",
              },
            },
          ],
        }),
        item("standalone-media", directTurn.id, 2, {
          type: "media",
          uri: "https://example.com/standalone.png",
          mime_type: "image/png",
        }),
        item("generated-media", directTurn.id, 3, {
          type: "image_generation",
          generation_status: "completed",
          result: "aW1hZ2U=",
        }),
      ],
    });

    expect(
      container.querySelectorAll('[data-direct-segment-kind="media"]'),
    ).toHaveLength(2);
    const mediaCalls = mockStreamingRenderer.mock.calls
      .map(([props]) => props)
      .filter(
        (props) =>
          Array.isArray(props.contentParts) &&
          props.contentParts.some(
            (part) =>
              typeof part === "object" &&
              part !== null &&
              "type" in part &&
              part.type === "media_reference",
          ),
      );
    expect(mediaCalls).toHaveLength(3);
    expect(mediaCalls.every((props) => props.onOpenMediaReference)).toBe(true);

    act(() => {
      mediaCalls[0]?.onOpenMediaReference?.(
        { uri: "https://example.com/agent.png", kind: "image" },
        0,
      );
    });
    expect(onOpenMessagePreview).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "media_reference" }),
      expect.objectContaining({
        id: "agent-media",
        runtimeTurnId: directTurn.id,
      }),
    );
  });

  it("独立 media Item 接管同引用并且不冒充最后一条 assistant 正文", () => {
    const directTurn = turn("turn-deduped-media", "running");
    const mediaUri = "https://example.com/shared.png";
    const container = render([], {
      currentTurnId: directTurn.id,
      turns: [directTurn],
      threadItems: [
        item("agent-with-media", directTurn.id, 1, {
          type: "agent_message",
          text: "media summary",
          contentParts: [
            {
              type: "media",
              kind: "image",
              reference: { uri: mediaUri, mime_type: "image/png" },
            },
          ],
        }),
        item("owned-media", directTurn.id, 2, {
          type: "media",
          uri: mediaUri,
          mime_type: "image/png",
        }),
      ],
    });

    const mediaReferenceCallCount = mockStreamingRenderer.mock.calls.filter(
      ([props]) =>
        props.contentParts?.some(
          (part) =>
            typeof part === "object" &&
            part !== null &&
            "type" in part &&
            part.type === "media_reference",
        ),
    ).length;

    expect(mockStreamingRenderer).toHaveBeenCalled();
    expect(mediaReferenceCallCount).toBe(1);
    expect(
      container.querySelectorAll('[data-message-role="assistant"]'),
    ).toHaveLength(1);
    expect(
      container.querySelectorAll('[data-message-role="media"]'),
    ).toHaveLength(1);
  });

  it("保留 imported/local rich residual，且 canonical process 不会被重复挂载", () => {
    const currentTurn = turn("turn-rich-residual");
    const messages: Message[] = [
      {
        id: "imported-message",
        role: "assistant",
        content: "imported residual content",
        timestamp: new Date(timestamp),
      },
      {
        id: "optimistic-rich-message",
        role: "assistant",
        content: "local rich residual content",
        timestamp: new Date(timestamp),
        runtimeTurnId: currentTurn.id,
        taskPreview: {
          kind: "typesetting",
          taskId: "local-task",
          taskType: "typesetting",
          prompt: "排版",
          status: "running",
        },
      },
    ];

    const container = render(messages, {
      currentTurnId: currentTurn.id,
      turns: [currentTurn],
      threadItems: [
        item("canonical-command", currentTurn.id, 2, {
          type: "command_execution",
          command: "pwd",
          cwd: "/repo",
        }),
        item("canonical-agent", currentTurn.id, 1, {
          type: "agent_message",
          text: "canonical direct content",
        }),
      ],
    });

    expect(
      container.querySelector('[data-message-id="imported-message"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-message-id="optimistic-rich-message"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-thread-item-id="canonical-agent"]'),
    ).not.toBeNull();
    expect(
      container.textContent?.match(/canonical direct content/g),
    ).toHaveLength(1);
    expect(
      container.querySelectorAll(
        '[data-testid="conversation-turn-process-segment"]',
      ),
    ).toHaveLength(1);
    expect(
      container.querySelectorAll('[data-render-entry-kind="message_group"]'),
    ).not.toHaveLength(0);
  });

  it("process-only canonical Turn 不会落入空态", () => {
    const processTurn = turn("turn-process-only");
    const container = render([], {
      emptyStateVariant: "task-center",
      turns: [processTurn],
      threadItems: [
        item("compaction", processTurn.id, 1, {
          type: "context_compaction",
          stage: "completed",
        }),
      ],
    });

    expect(
      container.querySelector('[data-testid="conversation-turn-timeline"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="message-list-empty-task-center"]'),
    ).toBeNull();
  });
});
