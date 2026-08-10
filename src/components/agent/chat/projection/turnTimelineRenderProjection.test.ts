import { describe, expect, it } from "vitest";
import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import type { MessageRenderGroupProjection } from "./messageTimelineRenderProjection";
import { buildTurnTimelineRenderProjection } from "./turnTimelineRenderProjection";
import { messageMatchesCanonicalUserMessage } from "../utils/importedUserMessageDedupe";

const timestamp = "2026-07-29T04:00:00.000Z";

function turn(id: string): AgentThreadTurn {
  return {
    id,
    thread_id: "thread-1",
    prompt_text: "",
    status: "completed",
    started_at: timestamp,
    completed_at: timestamp,
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
    thread_id: "thread-1",
    turn_id: turnId,
    sequence,
    status: "completed",
    started_at: timestamp,
    completed_at: timestamp,
    updated_at: timestamp,
    ...value,
  } as AgentThreadItem;
}

function message(
  id: string,
  role: Message["role"],
  runtimeTurnId?: string,
  extra: Partial<Message> = {},
): Message {
  return {
    id,
    role,
    content: id,
    timestamp: new Date(timestamp),
    runtimeTurnId,
    ...extra,
  };
}

function group(
  id: string,
  messages: Message[],
  timelineTurn?: AgentThreadTurn,
): MessageRenderGroupProjection {
  const assistants = messages.filter((entry) => entry.role === "assistant");
  return {
    id,
    messages,
    userMessage: messages.find((entry) => entry.role === "user") ?? null,
    assistantMessages: assistants,
    startedAt: messages[0]?.timestamp ?? new Date(timestamp),
    endedAt: messages.at(-1)?.timestamp ?? new Date(timestamp),
    lastAssistantId: assistants.at(-1)?.id ?? null,
    timelineMessageId: timelineTurn ? (assistants.at(-1)?.id ?? null) : null,
    timeline: timelineTurn
      ? {
          messageId: assistants.at(-1)?.id ?? "",
          turn: timelineTurn,
          items: [],
        }
      : null,
    isActiveGroup: false,
  };
}

describe("turnTimelineRenderProjection", () => {
  it("相同文本的不同 canonical 回合不会互相接管用户消息", () => {
    const firstMessage = message("first-message", "user", "turn-first", {
      content: "相同的问题",
    });
    const secondItem = item("second-user", "turn-second", 1, {
      type: "user_message",
      content: "相同的问题",
    });

    expect(messageMatchesCanonicalUserMessage(firstMessage, secondItem)).toBe(
      false,
    );
  });

  it("无 Message 锚点时仍按 canonical sequence 保留正文与过程交错顺序", () => {
    const currentTurn = turn("turn-direct");
    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [],
      renderedTurns: [currentTurn],
      currentTurnId: currentTurn.id,
      renderedThreadItems: [
        item("tool-after", currentTurn.id, 5, {
          type: "tool_call",
          tool_name: "read_file",
        }),
        item("agent-final", currentTurn.id, 4, {
          type: "agent_message",
          text: "最终答复",
        }),
        item("command", currentTurn.id, 3, {
          type: "command_execution",
          command: "pwd",
          cwd: "/repo",
        }),
        item("agent-commentary", currentTurn.id, 2, {
          type: "agent_message",
          text: "先检查目录",
          phase: "commentary",
        }),
        item("user", currentTurn.id, 1, {
          type: "user_message",
          content: "查看仓库",
        }),
      ],
    });

    expect(projection).toHaveLength(1);
    expect(projection[0]).toMatchObject({
      kind: "canonical_turn",
      isActive: true,
      segments: [
        { kind: "message", item: { id: "user" } },
        { kind: "message", item: { id: "agent-commentary" } },
        { kind: "process", items: [{ id: "command" }] },
        { kind: "message", item: { id: "agent-final" } },
        { kind: "process", items: [{ id: "tool-after" }] },
      ],
    });
  });

  it("canonical item id、client_id 与 runtimeTurnId 覆盖旧 Message 时只保留 direct Turn", () => {
    const currentTurn = turn("turn-owned");
    const messages = [
      message("client-user", "user", "pending-turn"),
      message("hydrated-assistant-shell", "assistant", currentTurn.id),
    ];
    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [group("legacy", messages, currentTurn)],
      renderedTurns: [currentTurn],
      renderedThreadItems: [
        item("canonical-user", currentTurn.id, 1, {
          type: "user_message",
          client_id: "client-user",
          content: "问题",
        }),
        item("canonical-agent", currentTurn.id, 2, {
          type: "agent_message",
          text: "答复",
        }),
      ],
    });

    expect(projection.map((entry) => entry.kind)).toEqual(["canonical_turn"]);
  });

  it("canonical reasoning 接管用户消息时应隐藏 pending assistant 首字占位", () => {
    const currentTurn = turn("turn-pending-assistant");
    currentTurn.status = "running";
    delete currentTurn.completed_at;
    const user = message("pending-user", "user", "pending-turn:1");
    const pendingAssistant = message(
      "pending-assistant",
      "assistant",
      "pending-turn:1",
      {
        content: "",
        isThinking: true,
        runtimeStatus: {
          phase: "routing",
          title: "正在生成回复",
          detail: "等待首个输出。",
        },
      },
    );

    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [group("pending", [user, pendingAssistant])],
      renderedTurns: [currentTurn],
      currentTurnId: currentTurn.id,
      renderedThreadItems: [
        item("canonical-user", currentTurn.id, 1, {
          type: "user_message",
          client_id: user.id,
          content: "你好",
        }),
        item("canonical-reasoning", currentTurn.id, 2, {
          type: "reasoning",
          status: "in_progress",
          text: "先确认用户意图。",
          summary: ["先确认用户意图。"],
        }),
      ],
    });

    expect(projection).toMatchObject([
      {
        kind: "canonical_turn",
        segments: [
          { kind: "message", item: { id: "canonical-user" } },
          { kind: "process", items: [{ id: "canonical-reasoning" }] },
        ],
      },
    ]);
  });

  it("reasoning Item 补全或追加后，过程段 identity 仍保持 turn 级稳定", () => {
    const currentTurn = turn("turn-stable-process-identity");
    const baseItems = [
      item("canonical-user-stable", currentTurn.id, 1, {
        type: "user_message",
        content: "验证思考展开",
      }),
      item("reasoning-live", currentTurn.id, 2, {
        type: "reasoning",
        status: "in_progress",
        text: "先确认用户意图。",
        summary: ["先确认用户意图。"],
      }),
      item("canonical-agent-stable", currentTurn.id, 3, {
        type: "agent_message",
        text: "最终答复",
      }),
    ];
    const baseProjection = buildTurnTimelineRenderProjection({
      messageGroups: [],
      renderedTurns: [currentTurn],
      renderedThreadItems: baseItems,
    });

    const persistedProjection = buildTurnTimelineRenderProjection({
      messageGroups: [],
      renderedTurns: [currentTurn],
      renderedThreadItems: [
        ...baseItems,
        item("reasoning-persisted", currentTurn.id, 4, {
          type: "reasoning",
          text: "先确认用户意图。",
          summary: ["先确认用户意图。"],
        }),
      ],
    });

    const baseProcess = baseProjection[0];
    const persistedProcess = persistedProjection[0];
    expect(baseProcess.kind).toBe("canonical_turn");
    expect(persistedProcess.kind).toBe("canonical_turn");
    expect(
      baseProcess.segments.find((segment) => segment.kind === "process")?.id,
    ).toBe("process:turn-stable-process-identity:0");
    expect(
      persistedProcess.segments.find((segment) => segment.kind === "process")
        ?.id,
    ).toBe("process:turn-stable-process-identity:0");
  });

  it("canonical turn 只有用户消息时仍保留 pending assistant 首字占位", () => {
    const currentTurn = turn("turn-pending-without-process");
    currentTurn.status = "running";
    delete currentTurn.completed_at;
    const user = message("pending-user-without-process", "user");
    const pendingAssistant = message(
      "pending-assistant-without-process",
      "assistant",
      "pending-turn:2",
      {
        content: "",
        isThinking: true,
        runtimeStatus: {
          phase: "routing",
          title: "正在生成回复",
          detail: "等待首个输出。",
        },
      },
    );

    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [
        group("pending-without-process", [user, pendingAssistant]),
      ],
      renderedTurns: [currentTurn],
      renderedThreadItems: [
        item("canonical-user-without-process", currentTurn.id, 1, {
          type: "user_message",
          client_id: user.id,
          content: "你好",
        }),
      ],
    });

    expect(projection).toMatchObject([
      { kind: "canonical_turn" },
      {
        kind: "message_group",
        group: {
          messages: [{ id: pendingAssistant.id }],
        },
      },
    ]);
  });

  it("canonical User Item 接管同一历史 Message 的图片附件", () => {
    const currentTurn = turn("turn-owned-user-image");
    const legacyUser = message("client-user-image", "user", currentTurn.id, {
      images: [
        {
          data: "aW1hZ2U=",
          mediaType: "image/png",
          sourceUri: "asset://fixture-user-image.png",
        },
      ],
    });
    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [group("legacy-user-image", [legacyUser], currentTurn)],
      renderedTurns: [currentTurn],
      renderedThreadItems: [
        item("canonical-user-image", currentTurn.id, 1, {
          type: "user_message",
          client_id: "client-user-image",
          content: "附带图片的提问",
          content_parts: [
            {
              type: "image",
              data: "aW1hZ2U=",
              mime_type: "image/png",
              uri: "asset://fixture-user-image.png",
            },
          ],
        }),
      ],
    });

    expect(projection).toHaveLength(1);
    expect(projection[0]).toMatchObject({
      kind: "canonical_turn",
      segments: [
        {
          kind: "message",
          item: { id: "canonical-user-image", type: "user_message" },
        },
      ],
    });
  });

  it("legacy image wrapper 与 canonical user_message 同图时只渲染一条用户消息", () => {
    const currentTurn = turn("turn-owned-user-image-wrapper");
    const imagePath = "/tmp/waveterm_paste_20260729.png";
    const legacyUser = message("legacy-user-image-wrapper", "user", undefined, {
      content: `<image name=[Image #1] path="${imagePath}"> [Image #1] 为什么会有这种两个同时回复的情况呢`,
      images: [
        {
          data: "",
          mediaType: "image/png",
          sourcePath: imagePath,
        },
      ],
    });
    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [group("legacy-user-image-wrapper", [legacyUser])],
      renderedTurns: [currentTurn],
      renderedThreadItems: [
        item("canonical-user-image-wrapper", currentTurn.id, 1, {
          type: "user_message",
          content: "[Image #1] 为什么会有这种两个同时回复的情况呢",
          content_parts: [
            {
              type: "image",
              data: "",
              mime_type: "image/png",
              source_path: imagePath,
            },
          ],
        }),
      ],
    });

    expect(projection).toHaveLength(1);
    expect(projection[0]).toMatchObject({
      kind: "canonical_turn",
      segments: [
        {
          kind: "message",
          item: { id: "canonical-user-image-wrapper", type: "user_message" },
        },
      ],
    });
  });

  it("没有 canonical Item 覆盖的 imported/local Message 保留 residual 路径", () => {
    const imported = message("imported-user", "user");
    const optimistic = message(
      "optimistic-assistant",
      "assistant",
      "turn-direct",
      {
        taskPreview: {
          kind: "typesetting",
          taskId: "task-local",
          taskType: "typesetting",
          prompt: "排版",
          status: "running",
        },
        thinkingContent: "旧 Message 推理",
        contentParts: [
          { type: "text", text: "旧 Message 正文" },
          {
            type: "tool_use",
            toolCall: {
              id: "legacy-tool",
              name: "read_file",
              status: "completed",
            },
          },
        ],
      },
    );
    const directTurn = turn("turn-direct");
    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [
        group("imported", [imported]),
        group("optimistic", [optimistic], directTurn),
      ],
      renderedTurns: [directTurn],
      renderedThreadItems: [
        item("canonical-agent", directTurn.id, 1, {
          type: "agent_message",
          text: "正文",
        }),
      ],
    });

    expect(projection).toMatchObject([
      {
        kind: "message_group",
        group: { messages: [{ id: "imported-user" }], timeline: null },
      },
      { kind: "canonical_turn" },
      {
        kind: "message_group",
        group: {
          messages: [
            {
              id: "optimistic-assistant",
              content: "",
              contentParts: undefined,
              thinkingContent: undefined,
            },
          ],
          timeline: null,
        },
      },
    ]);
  });

  it("无正文 Item 且无关联 Message 的 process-only Turn 仍直接可见", () => {
    const processTurn = turn("turn-process-only");
    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [],
      renderedTurns: [processTurn],
      renderedThreadItems: [
        item("compaction", processTurn.id, 1, {
          type: "context_compaction",
          stage: "completed",
        }),
      ],
    });

    expect(projection).toMatchObject([
      {
        kind: "canonical_turn",
        segments: [
          {
            kind: "process",
            items: [{ id: "compaction", type: "context_compaction" }],
          },
        ],
      },
    ]);
  });

  it("已有 Message owner 但尚无 canonical 正文 Item 时继续走旧路径", () => {
    const compatTurn = turn("turn-compat");
    const assistant = message("compat-assistant", "assistant", compatTurn.id);
    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [group("compat", [assistant], compatTurn)],
      renderedTurns: [compatTurn],
      renderedThreadItems: [
        item("compat-command", compatTurn.id, 1, {
          type: "command_execution",
          command: "pwd",
          cwd: "/repo",
        }),
      ],
    });

    expect(projection).toMatchObject([
      {
        kind: "message_group",
        group: {
          messages: [{ id: "compat-assistant" }],
          timeline: { turn: { id: "turn-compat" } },
        },
      },
    ]);
  });

  it("仅由旧 timeline 启发式关联的 Message 不触发 canonical 接管", () => {
    const compatTurn = turn("turn-heuristic-only");
    const assistant = message("legacy-assistant", "assistant");
    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [group("legacy", [assistant], compatTurn)],
      renderedTurns: [compatTurn],
      renderedThreadItems: [
        item("canonical-agent", compatTurn.id, 1, {
          type: "agent_message",
          text: "canonical answer",
        }),
      ],
    });

    expect(projection).toMatchObject([
      {
        kind: "message_group",
        group: {
          messages: [{ id: "legacy-assistant" }],
          timeline: { turn: { id: "turn-heuristic-only" } },
        },
      },
    ]);
  });

  it("其他 Turn 存在 Message 时仍保留无锚点的 process-only Turn", () => {
    const anchoredTurn = turn("turn-anchored");
    const orphanTurn = turn("turn-orphan-process");
    const assistant = message(
      "anchored-assistant",
      "assistant",
      anchoredTurn.id,
    );
    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [group("anchored", [assistant], anchoredTurn)],
      renderedTurns: [orphanTurn, anchoredTurn],
      renderedThreadItems: [
        item("orphan-command", orphanTurn.id, 1, {
          type: "command_execution",
          command: "pwd",
          cwd: "/repo",
        }),
        item("anchored-agent", anchoredTurn.id, 1, {
          type: "agent_message",
          text: "answer",
        }),
      ],
    });

    expect(projection.map((entry) => entry.id)).toEqual([
      "canonical-turn:turn-orphan-process",
      "canonical-turn:turn-anchored",
    ]);
  });

  it("canonical-only Turn 与 Message 锚定 Turn 始终服从 renderedTurns 顺序", () => {
    const earlierTurn = turn("turn-earlier");
    const anchoredTurn = turn("turn-later");
    const assistant = message(
      "anchored-assistant",
      "assistant",
      anchoredTurn.id,
    );
    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [group("anchored", [assistant], anchoredTurn)],
      renderedTurns: [earlierTurn, anchoredTurn],
      renderedThreadItems: [
        item("earlier-agent", earlierTurn.id, 1, {
          type: "agent_message",
          text: "earlier",
        }),
        item("later-agent", anchoredTurn.id, 1, {
          type: "agent_message",
          text: "later",
        }),
      ],
    });

    expect(
      projection
        .filter((entry) => entry.kind === "canonical_turn")
        .map((entry) => entry.turn.id),
    ).toEqual([earlierTurn.id, anchoredTurn.id]);
  });

  it("media 与 image_generation 作为独立 canonical 段保留 sequence", () => {
    const mediaTurn = turn("turn-media");
    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [],
      renderedTurns: [mediaTurn],
      renderedThreadItems: [
        item("command", mediaTurn.id, 1, {
          type: "command_execution",
          command: "pwd",
          cwd: "/repo",
        }),
        item("media", mediaTurn.id, 2, {
          type: "media",
          uri: "https://example.com/result.png",
          mime_type: "image/png",
        }),
        item("generated", mediaTurn.id, 3, {
          type: "image_generation",
          generation_status: "completed",
          result: "aW1hZ2U=",
        }),
        item("agent", mediaTurn.id, 4, {
          type: "agent_message",
          text: "done",
        }),
      ],
    });

    expect(projection).toMatchObject([
      {
        kind: "canonical_turn",
        segments: [
          { kind: "process", items: [{ id: "command" }] },
          { kind: "media", item: { id: "media" } },
          { kind: "media", item: { id: "generated" } },
          { kind: "message", item: { id: "agent" } },
        ],
      },
    ]);
  });

  it("exact canonical identity 只接管正文，Message-only rich surface 仍保留", () => {
    const richTurn = turn("turn-rich-exact");
    const assistant = message("canonical-agent", "assistant", richTurn.id, {
      content: "legacy duplicate",
      images: [{ data: "aW1hZ2U=", mediaType: "image/png" }],
      actionRequests: [
        {
          requestId: "request-rich",
          actionType: "ask",
          data: {},
        },
      ] as Message["actionRequests"],
      contentParts: [
        { type: "text", text: "legacy duplicate" },
        {
          type: "media_reference",
          reference: {
            uri: "https://example.com/rich.png",
            kind: "image",
          },
        },
      ],
    });
    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [group("rich-exact", [assistant], richTurn)],
      renderedTurns: [richTurn],
      renderedThreadItems: [
        item("canonical-agent", richTurn.id, 1, {
          type: "agent_message",
          text: "canonical answer",
        }),
      ],
    });

    expect(projection).toMatchObject([
      { kind: "canonical_turn" },
      {
        kind: "message_group",
        group: {
          messages: [
            {
              id: "canonical-agent",
              content: "",
              images: [{ data: "aW1hZ2U=" }],
              actionRequests: [{ requestId: "request-rich" }],
              contentParts: [{ type: "media_reference" }],
            },
          ],
        },
      },
    ]);
  });

  it("canonical media Item 删除同引用 residual media_reference", () => {
    const mediaTurn = turn("turn-media-residual");
    const mediaUri = "/tmp/fixture-media-reference.png";
    const assistant = message("legacy-media-shell", "assistant", mediaTurn.id, {
      content: "legacy duplicate",
      contentParts: [
        { type: "text", text: "legacy duplicate" },
        {
          type: "media_reference",
          reference: {
            uri: mediaUri,
            sourcePath: mediaUri,
            kind: "image",
          },
        },
      ],
    });
    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [group("media-residual", [assistant], mediaTurn)],
      renderedTurns: [mediaTurn],
      renderedThreadItems: [
        item("canonical-agent", mediaTurn.id, 1, {
          type: "agent_message",
          text: "canonical answer",
        }),
        item("canonical-media", mediaTurn.id, 2, {
          type: "media",
          uri: mediaUri,
          mime_type: "image/png",
        }),
      ],
    });

    expect(projection).toMatchObject([
      {
        kind: "canonical_turn",
        segments: [
          { kind: "message", item: { id: "canonical-agent" } },
          { kind: "media", item: { id: "canonical-media" } },
        ],
      },
    ]);
  });

  it("同组 residual User 必须排在 runtimeTurnId 接管的 canonical Agent 前", () => {
    const directTurn = turn("turn-mixed-owner");
    const user = message("residual-user", "user");
    const assistant = message("owned-assistant", "assistant", directTurn.id);
    const projection = buildTurnTimelineRenderProjection({
      messageGroups: [group("mixed", [user, assistant], directTurn)],
      renderedTurns: [directTurn],
      renderedThreadItems: [
        item("canonical-agent", directTurn.id, 1, {
          type: "agent_message",
          text: "answer",
        }),
      ],
    });

    expect(projection).toMatchObject([
      {
        kind: "message_group",
        group: { messages: [{ id: "residual-user" }], timeline: null },
      },
      { kind: "canonical_turn", turn: { id: "turn-mixed-owner" } },
    ]);
  });
});
