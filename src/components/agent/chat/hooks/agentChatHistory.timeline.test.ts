import { describe, expect, it } from "vitest";

import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";

import { hydrateSessionDetailMessages } from "./agentChatHistory";

describe("agentChatHistory timeline fallback", () => {
  it("应从历史消息的 thinking 字段恢复完整思考过程", () => {
    const detail: AgentSessionDetail = {
      id: "session-1",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "user",
          timestamp: 1710000000,
          content: [
            { type: "text", text: "请给我一版可直接使用的图片 Prompt" },
          ],
        },
        {
          role: "assistant",
          timestamp: 1710000005,
          content: [
            { type: "thinking", thinking: "先理解主题" } as never,
            { type: "thinking", thinking: "，再组织结构。\n" } as never,
            { type: "output_text", text: "下面是整理好的 Prompt。" } as never,
          ],
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(detail, "session-1");
    const assistantMessage = messages.find(
      (message) => message.role === "assistant",
    );

    expect(assistantMessage).toBeDefined();
    expect(assistantMessage?.content).toBe("下面是整理好的 Prompt。");
    expect(assistantMessage?.thinkingContent).toBe(
      "先理解主题，再组织结构。\n",
    );
    expect(assistantMessage?.contentParts).toEqual([
      {
        type: "thinking",
        text: "先理解主题，再组织结构。\n",
      },
      {
        type: "text",
        text: "下面是整理好的 Prompt。",
      },
    ]);
  });

  it("后端 detail.messages 为空但 timeline 有用户与助手消息时应恢复对话", () => {
    const detail: AgentSessionDetail = {
      id: "session-timeline-only",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "turn-timeline-only",
          thread_id: "session-timeline-only",
          prompt_text: "我来帮你搜索 OpenAI 最新模型",
          status: "completed",
          started_at: "2026-05-06T10:00:00.000Z",
          completed_at: "2026-05-06T10:00:03.000Z",
          created_at: "2026-05-06T10:00:00.000Z",
          updated_at: "2026-05-06T10:00:03.000Z",
        },
      ],
      items: [
        {
          id: "item-user",
          thread_id: "session-timeline-only",
          turn_id: "turn-timeline-only",
          sequence: 1,
          type: "user_message",
          content: "我来帮你搜索 OpenAI 最新模型",
          status: "completed",
          started_at: "2026-05-06T10:00:00.000Z",
          completed_at: "2026-05-06T10:00:00.000Z",
          updated_at: "2026-05-06T10:00:00.000Z",
        } as never,
        {
          id: "item-assistant",
          thread_id: "session-timeline-only",
          turn_id: "turn-timeline-only",
          sequence: 2,
          type: "agent_message",
          text: "已找到最新模型信息。",
          status: "completed",
          started_at: "2026-05-06T10:00:02.000Z",
          completed_at: "2026-05-06T10:00:03.000Z",
          updated_at: "2026-05-06T10:00:03.000Z",
        } as never,
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-timeline-only",
    );

    expect(messages).toHaveLength(2);
    expect(messages[0]).toMatchObject({
      id: "session-timeline-only-timeline-item-user",
      role: "user",
      content: "我来帮你搜索 OpenAI 最新模型",
    });
    expect(messages[1]).toMatchObject({
      id: "session-timeline-only-timeline-item-assistant",
      role: "assistant",
      content: "已找到最新模型信息。",
      contentParts: [
        {
          type: "text",
          text: "已找到最新模型信息。",
        },
      ],
    });
  });

  it("历史 timeline agent_message 只有 content 字段时仍应仅恢复最终正文", () => {
    const detail: AgentSessionDetail = {
      id: "session-timeline-content-final",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "turn-timeline-content-final",
          thread_id: "session-timeline-content-final",
          prompt_text:
            "读取 internal/roadmap/agent-workspace/README.md 并总结一下",
          status: "completed",
          started_at: "2026-06-08T10:00:00.000Z",
          completed_at: "2026-06-08T10:00:06.000Z",
          created_at: "2026-06-08T10:00:00.000Z",
          updated_at: "2026-06-08T10:00:06.000Z",
        },
      ],
      items: [
        {
          id: "item-user-content-final",
          thread_id: "session-timeline-content-final",
          turn_id: "turn-timeline-content-final",
          sequence: 1,
          type: "user_message",
          content: "读取 internal/roadmap/agent-workspace/README.md 并总结一下",
          status: "completed",
          started_at: "2026-06-08T10:00:00.000Z",
          completed_at: "2026-06-08T10:00:00.000Z",
          updated_at: "2026-06-08T10:00:00.000Z",
        } as never,
        {
          id: "item-read-file-content-final",
          thread_id: "session-timeline-content-final",
          turn_id: "turn-timeline-content-final",
          sequence: 2,
          type: "tool_call",
          tool_name: "Read",
          arguments: {
            file_path: "internal/roadmap/agent-workspace/README.md",
          },
          output: "# Agent Workspace\n\n主线说明。",
          success: true,
          status: "completed",
          started_at: "2026-06-08T10:00:01.000Z",
          completed_at: "2026-06-08T10:00:02.000Z",
          updated_at: "2026-06-08T10:00:02.000Z",
        } as never,
        {
          id: "item-assistant-content-final",
          thread_id: "session-timeline-content-final",
          turn_id: "turn-timeline-content-final",
          sequence: 3,
          type: "agent_message",
          content:
            "这个 README 主要说明 Agent Workspace 的目标、阶段和当前交付边界。",
          phase: "final_answer",
          status: "completed",
          started_at: "2026-06-08T10:00:05.000Z",
          completed_at: "2026-06-08T10:00:06.000Z",
          updated_at: "2026-06-08T10:00:06.000Z",
        } as never,
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-timeline-content-final",
    );

    expect(messages).toHaveLength(2);
    expect(messages[1]).toMatchObject({
      role: "assistant",
      content:
        "这个 README 主要说明 Agent Workspace 的目标、阶段和当前交付边界。",
    });
    expect(messages[1]?.toolCalls).toBeUndefined();
    expect(messages[1]?.contentParts?.map((part) => part.type)).toEqual(["text"]);
  });

  it("历史 timeline 的 canonical media item 不应二次合成为 Message 引用", () => {
    const detail: AgentSessionDetail = {
      id: "session-timeline-media-final",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "turn-media-final",
          thread_id: "session-timeline-media-final",
          prompt_text: "生成一张图",
          status: "completed",
          started_at: "2026-05-06T10:00:00.000Z",
          completed_at: "2026-05-06T10:00:03.000Z",
          created_at: "2026-05-06T10:00:00.000Z",
          updated_at: "2026-05-06T10:00:03.000Z",
        },
      ],
      items: [
        {
          id: "item-user-media",
          thread_id: "session-timeline-media-final",
          turn_id: "turn-media-final",
          sequence: 1,
          type: "user_message",
          content: "生成一张图",
          status: "completed",
          started_at: "2026-05-06T10:00:00.000Z",
          completed_at: "2026-05-06T10:00:00.000Z",
          updated_at: "2026-05-06T10:00:00.000Z",
        } as never,
        {
          id: "item-assistant-text",
          thread_id: "session-timeline-media-final",
          turn_id: "turn-media-final",
          sequence: 2,
          type: "agent_message",
          text: "图片已生成",
          phase: "final_answer",
          status: "completed",
          started_at: "2026-05-06T10:00:02.000Z",
          completed_at: "2026-05-06T10:00:03.000Z",
          updated_at: "2026-05-06T10:00:03.000Z",
        } as never,
        {
          id: "item-image-view",
          thread_id: "session-timeline-media-final",
          turn_id: "turn-media-final",
          sequence: 3,
          type: "media",
          uri: "/tmp/lime-media/image-1.png",
          mime_type: "image/png",
          status: "completed",
          started_at: "2026-05-06T10:00:02.500Z",
          completed_at: "2026-05-06T10:00:03.000Z",
          updated_at: "2026-05-06T10:00:03.000Z",
        } as never,
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-timeline-media-final",
    );

    expect(messages[1]).toMatchObject({
      id: "session-timeline-media-final-timeline-item-assistant-text",
      role: "assistant",
      content: "图片已生成",
    });
    expect(messages[1]?.contentParts).toEqual([
      expect.objectContaining({ type: "text", text: "图片已生成" }),
    ]);
  });

  it("历史 plan item 与 update_plan tool_call 均不应二次合成为 Message", () => {
    const detail: AgentSessionDetail = {
      id: "session-update-plan-history",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "turn-update-plan-history",
          thread_id: "session-update-plan-history",
          prompt_text: "先规划再执行",
          status: "completed",
          started_at: "2026-06-18T10:00:00.000Z",
          completed_at: "2026-06-18T10:00:06.000Z",
          created_at: "2026-06-18T10:00:00.000Z",
          updated_at: "2026-06-18T10:00:06.000Z",
        },
      ],
      items: [
        {
          id: "item-user-update-plan-history",
          thread_id: "session-update-plan-history",
          turn_id: "turn-update-plan-history",
          sequence: 1,
          type: "user_message",
          content: "先规划再执行",
          status: "completed",
          started_at: "2026-06-18T10:00:00.000Z",
          completed_at: "2026-06-18T10:00:00.000Z",
          updated_at: "2026-06-18T10:00:00.000Z",
        } as never,
        {
          id: "item-plan-update-plan-history",
          thread_id: "session-update-plan-history",
          turn_id: "turn-update-plan-history",
          sequence: 2,
          type: "plan",
          text: "- [x] 整理计划\n- [ ] 执行修改",
          metadata: {
            revisionId: "update_plan:item-update-plan-history",
            source: "update_plan",
          },
          status: "completed",
          started_at: "2026-06-18T10:00:01.000Z",
          completed_at: "2026-06-18T10:00:02.000Z",
          updated_at: "2026-06-18T10:00:02.000Z",
        } as never,
        {
          id: "item-tool-update-plan-history",
          thread_id: "session-update-plan-history",
          turn_id: "turn-update-plan-history",
          sequence: 3,
          type: "tool_call",
          tool_name: "UpdatePlanTool",
          arguments: {
            plan: [
              { step: "整理计划", status: "completed" },
              { step: "执行修改", status: "in_progress" },
            ],
          },
          output: "ok",
          success: true,
          status: "completed",
          started_at: "2026-06-18T10:00:03.000Z",
          completed_at: "2026-06-18T10:00:04.000Z",
          updated_at: "2026-06-18T10:00:04.000Z",
        } as never,
        {
          id: "item-assistant-update-plan-history",
          thread_id: "session-update-plan-history",
          turn_id: "turn-update-plan-history",
          sequence: 4,
          type: "agent_message",
          text: "计划已更新。",
          phase: "final_answer",
          status: "completed",
          started_at: "2026-06-18T10:00:05.000Z",
          completed_at: "2026-06-18T10:00:06.000Z",
          updated_at: "2026-06-18T10:00:06.000Z",
        } as never,
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-update-plan-history",
    );
    const assistantMessage = messages.find(
      (message) => message.role === "assistant",
    );

    expect(assistantMessage?.toolCalls).toBeUndefined();
    expect(assistantMessage?.contentParts).toEqual([
      {
        type: "text",
        text: "计划已更新。",
        metadata: {
          source: "agent_thread_item",
          threadItemId: "item-assistant-update-plan-history",
          turnId: "turn-update-plan-history",
          sequence: 4,
          phase: "final_answer",
        },
      },
    ]);
  });

  it("App Server 历史 turn 缺少旧 prompt_text 字段时不应中断会话恢复", () => {
    const detail: AgentSessionDetail = {
      id: "session-missing-legacy-text",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "turn-missing-prompt",
          thread_id: "session-missing-legacy-text",
          status: "failed",
          started_at: "2026-06-07T04:39:20.100Z",
          completed_at: "2026-06-07T04:42:05.905Z",
          created_at: "2026-06-07T04:39:20.100Z",
          updated_at: "2026-06-07T04:42:05.905Z",
        } as never,
      ],
      items: [
        {
          id: "item-user-missing-content",
          thread_id: "session-missing-legacy-text",
          turn_id: "turn-missing-prompt",
          sequence: 1,
          type: "user_message",
          status: "completed",
          started_at: "2026-06-07T04:39:20.100Z",
          updated_at: "2026-06-07T04:39:20.100Z",
        } as never,
        {
          id: "item-agent-missing-text",
          thread_id: "session-missing-legacy-text",
          turn_id: "turn-missing-prompt",
          sequence: 2,
          type: "agent_message",
          status: "failed",
          started_at: "2026-06-07T04:42:05.905Z",
          updated_at: "2026-06-07T04:42:05.905Z",
        } as never,
      ],
    };

    expect(() =>
      hydrateSessionDetailMessages(detail, "session-missing-legacy-text"),
    ).not.toThrow();
    expect(
      hydrateSessionDetailMessages(detail, "session-missing-legacy-text"),
    ).toEqual([]);
  });

  it("App Server 历史只有 artifact summary 时应恢复产物消息", () => {
    const detail: AgentSessionDetail = {
      id: "session-artifact-only",
      thread_id: "session-artifact-only-thread",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "turn-artifact-only",
          thread_id: "session-artifact-only-thread",
          prompt_text: "",
          status: "completed",
          started_at: "2026-06-07T06:17:13.000Z",
          completed_at: "2026-06-07T06:17:14.000Z",
          created_at: "2026-06-07T06:17:13.000Z",
          updated_at: "2026-06-07T06:17:14.000Z",
        },
      ],
      items: [],
      artifacts: [
        {
          artifactRef: "artifact-ref-1",
          eventId: "event-artifact-1",
          sequence: 1,
          turnId: "turn-artifact-only",
          artifactId: "code-artifact:greeting",
          path: ".lime/qc/code-artifact-workbench/src/greeting.ts",
          title: "greeting.ts",
          kind: "code",
          status: "complete",
          contentStatus: "available",
          metadata: {
            language: "typescript",
            previewText: "export const greeting = 'hello';",
          },
        },
      ],
      thread_read: {
        thread_id: "session-artifact-only-thread",
        status: "completed",
        profile_status: "completed",
        turns: [
          {
            turn_id: "turn-artifact-only",
            status: "completed",
            native_status: "completed",
          },
        ],
        pending_requests: [],
        incidents: [],
        artifacts: [
          {
            artifactRef: "artifact-ref-1",
            eventId: "event-artifact-1",
            sequence: 1,
            turnId: "turn-artifact-only",
            artifactId: "code-artifact:greeting",
            path: ".lime/qc/code-artifact-workbench/src/greeting.ts",
            title: "greeting.ts",
            kind: "code",
            status: "complete",
            contentStatus: "available",
            metadata: {
              language: "typescript",
              previewText: "export const greeting = 'hello';",
            },
          },
        ],
      } as never,
    } as AgentSessionDetail & { artifacts: unknown[] };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-artifact-only",
    );

    expect(messages).toHaveLength(1);
    expect(messages[0]).toMatchObject({
      id: "session-artifact-only-app-server-artifacts",
      role: "assistant",
      content: "已生成代码产物，可在工作台查看。",
      runtimeTurnId: "turn-artifact-only",
      artifacts: [
        {
          id: "code-artifact:greeting",
          type: "code",
          title: "greeting.ts",
          status: "complete",
          content: "export const greeting = 'hello';",
          meta: {
            filePath: ".lime/qc/code-artifact-workbench/src/greeting.ts",
            artifactPath: ".lime/qc/code-artifact-workbench/src/greeting.ts",
            previewText: "export const greeting = 'hello';",
            sessionId: "session-artifact-only",
            turnId: "turn-artifact-only",
            artifactRef: "artifact-ref-1",
            appServerArtifactSessionId: "session-artifact-only",
            appServerArtifactTurnId: "turn-artifact-only",
            appServerArtifactRef: "artifact-ref-1",
          },
        },
      ],
    });
  });

  it("历史应恢复文章 artifact document 且隐藏 workspace patch", () => {
    const detail: AgentSessionDetail = {
      id: "session-article-workspace-artifacts",
      thread_id: "session-article-workspace-artifacts-thread",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [],
      items: [],
      artifacts: [
        {
          artifactRef: "artifact-article-1",
          eventId: "event-article-workspace-artifact-1",
          sequence: 1,
          artifactId: "artifact-article-1",
          path: ".lime/artifacts/article-workspace/article.artifact.json",
          title: "公众号文章草稿",
          kind: "artifact_document",
          status: "complete",
          contentStatus: "available",
          metadata: {
            openedFrom: "app_server_article_workspace",
            artifactSchema: "artifact_document.v1",
            previewText: "历史恢复文章正文",
            artifactDocument: {
              id: "artifact-document:content-factory-app:artifact-article-1",
              title: "公众号文章草稿",
              blocks: [
                {
                  type: "rich_text",
                  content: "历史恢复文章正文",
                },
              ],
              metadata: {
                currentVersionNo: 1,
              },
            },
            articleWorkspace: {
              appId: "content-factory-app",
              sessionId: "session-article-workspace-artifacts",
              objectKind: "articleDraft",
              objectId: "article-1",
            },
          },
        },
        {
          artifactRef: "artifact-workspace-patch-current",
          eventId: "event-article-workspace-patch-current",
          sequence: 2,
          artifactId: "artifact-workspace-patch-current",
          path: ".lime/artifacts/article-workspace/workspace-patch.json",
          title: "Article Workspace Patch",
          kind: "workspace_patch",
          status: "complete",
          contentStatus: "available",
          metadata: {
            workspacePatch: {
              appId: "content-factory-app",
              sessionId: "session-article-workspace-artifacts",
              objects: [
                {
                  ref: {
                    appId: "content-factory-app",
                    kind: "articleDraft",
                    id: "article-1",
                    sessionId: "session-article-workspace-artifacts",
                  },
                  title: "公众号文章草稿",
                  status: "ready",
                },
              ],
            },
          },
        },
      ],
      thread_read: {
        thread_id: "session-article-workspace-artifacts-thread",
        status: "completed",
        profile_status: "completed",
        pending_requests: [],
        incidents: [],
        article_workspace: {
          schemaVersion: "article-workspace.v1",
          appId: "content-factory-app",
          sessionId: "session-article-workspace-artifacts",
          objects: [
            {
              ref: {
                appId: "content-factory-app",
                kind: "articleDraft",
                id: "article-1",
                sessionId: "session-article-workspace-artifacts",
              },
              title: "公众号文章草稿",
              status: "ready",
            },
          ],
        },
      } as never,
    } as AgentSessionDetail & { artifacts: unknown[] };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-article-workspace-artifacts",
    );

    expect(messages).toHaveLength(1);
    expect(messages[0]).toMatchObject({
      id: "session-article-workspace-artifacts-app-server-artifacts",
      role: "assistant",
      content: "已生成产物，可在工作台查看。",
      artifacts: [
        {
          id: "artifact-article-1",
          type: "document",
          title: "公众号文章草稿",
          status: "complete",
          content: "历史恢复文章正文",
          meta: {
            openedFrom: "app_server_article_workspace",
            artifactSchema: "artifact_document.v1",
            previewText: "历史恢复文章正文",
            filePath: ".lime/artifacts/article-workspace/article.artifact.json",
            artifactPath:
              ".lime/artifacts/article-workspace/article.artifact.json",
            appServerArtifactSessionId: "session-article-workspace-artifacts",
          },
        },
      ],
    });
    expect(messages[0]?.artifacts).toHaveLength(1);
    expect(messages[0]?.artifacts?.[0]?.id).toBe("artifact-article-1");
  });

  it("历史恢复不应把 commentary 阶段消息合并进最终正文", () => {
    const detail: AgentSessionDetail = {
      id: "session-commentary-final",
      created_at: 1,
      updated_at: 2,
      history_limit: 40,
      messages: [],
      turns: [
        {
          id: "turn-commentary-final",
          thread_id: "session-commentary-final",
          prompt_text: "整理今天的国际新闻",
          status: "completed",
          started_at: "2026-06-02T10:00:00.000Z",
          completed_at: "2026-06-02T10:00:20.000Z",
          created_at: "2026-06-02T10:00:00.000Z",
          updated_at: "2026-06-02T10:00:20.000Z",
        },
      ],
      items: [
        {
          id: "user-commentary-final",
          thread_id: "session-commentary-final",
          turn_id: "turn-commentary-final",
          sequence: 1,
          type: "user_message",
          content: "整理今天的国际新闻",
          status: "completed",
          started_at: "2026-06-02T10:00:00.000Z",
          completed_at: "2026-06-02T10:00:00.000Z",
          updated_at: "2026-06-02T10:00:00.000Z",
        } as never,
        {
          id: "assistant-commentary",
          thread_id: "session-commentary-final",
          turn_id: "turn-commentary-final",
          sequence: 2,
          type: "agent_message",
          text: "我会先检索多组来源并交叉核对。",
          phase: "commentary",
          status: "completed",
          started_at: "2026-06-02T10:00:01.000Z",
          completed_at: "2026-06-02T10:00:02.000Z",
          updated_at: "2026-06-02T10:00:02.000Z",
        } as never,
        {
          id: "assistant-final",
          thread_id: "session-commentary-final",
          turn_id: "turn-commentary-final",
          sequence: 3,
          type: "agent_message",
          text: "## 今日国际新闻简报\n\n- 第一条要闻。",
          phase: "final_answer",
          status: "completed",
          started_at: "2026-06-02T10:00:18.000Z",
          completed_at: "2026-06-02T10:00:20.000Z",
          updated_at: "2026-06-02T10:00:20.000Z",
        } as never,
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-commentary-final",
    );

    expect(messages).toHaveLength(2);
    expect(messages[1]).toMatchObject({
      role: "assistant",
      content: "## 今日国际新闻简报\n\n- 第一条要闻。",
    });
    expect(messages[1]?.content).not.toContain("我会先检索");
    expect(
      messages[1]?.contentParts
        ?.filter((part) => part.type === "text")
        .map((part) => part.text)
        .join("\n"),
    ).not.toContain("我会先检索");
    expect(
      messages[1]?.contentParts
        ?.filter((part) => part.type === "thinking")
        .map((part) => part.text)
        .join("\n"),
    ).not.toContain("我会先检索");
  });

  it("历史恢复应忽略旧无 phase 过程 Item 并只保留最后一条 agent_message 正文", () => {
    const detail: AgentSessionDetail = {
      id: "session-legacy-unphased-final",
      created_at: 1,
      updated_at: 2,
      history_limit: 40,
      messages: [],
      turns: [
        {
          id: "turn-legacy-unphased-final",
          thread_id: "session-legacy-unphased-final",
          prompt_text: "整理今天的国际新闻",
          status: "completed",
          started_at: "2026-06-02T10:00:00.000Z",
          completed_at: "2026-06-02T10:00:30.000Z",
          created_at: "2026-06-02T10:00:00.000Z",
          updated_at: "2026-06-02T10:00:30.000Z",
        },
      ],
      items: [
        {
          id: "user-legacy-unphased-final",
          thread_id: "session-legacy-unphased-final",
          turn_id: "turn-legacy-unphased-final",
          sequence: 1,
          type: "user_message",
          content: "整理今天的国际新闻",
          status: "completed",
          started_at: "2026-06-02T10:00:00.000Z",
          completed_at: "2026-06-02T10:00:00.000Z",
          updated_at: "2026-06-02T10:00:00.000Z",
        } as never,
        {
          id: "assistant-process-search",
          thread_id: "session-legacy-unphased-final",
          turn_id: "turn-legacy-unphased-final",
          sequence: 2,
          type: "agent_message",
          text: "我会先做几组中英文检索，覆盖多个新闻源。",
          status: "completed",
          started_at: "2026-06-02T10:00:01.000Z",
          completed_at: "2026-06-02T10:00:02.000Z",
          updated_at: "2026-06-02T10:00:02.000Z",
        } as never,
        {
          id: "tool-web-search",
          thread_id: "session-legacy-unphased-final",
          turn_id: "turn-legacy-unphased-final",
          sequence: 3,
          type: "tool_call",
          tool_name: "WebSearch",
          arguments: { query: "world news headlines" },
          output: "搜索结果摘要",
          success: true,
          status: "completed",
          started_at: "2026-06-02T10:00:03.000Z",
          completed_at: "2026-06-02T10:00:05.000Z",
          updated_at: "2026-06-02T10:00:05.000Z",
        } as never,
        {
          id: "assistant-process-fetch",
          thread_id: "session-legacy-unphased-final",
          turn_id: "turn-legacy-unphased-final",
          sequence: 4,
          type: "agent_message",
          text: "搜索结果里噪声较多，我再打开几个页面交叉核对。",
          status: "completed",
          started_at: "2026-06-02T10:00:06.000Z",
          completed_at: "2026-06-02T10:00:07.000Z",
          updated_at: "2026-06-02T10:00:07.000Z",
        } as never,
        {
          id: "tool-web-fetch-failed",
          thread_id: "session-legacy-unphased-final",
          turn_id: "turn-legacy-unphased-final",
          sequence: 5,
          type: "tool_call",
          tool_name: "WebFetch",
          arguments: { url: "https://example.invalid/news" },
          output: "",
          error: "请求失败",
          success: false,
          status: "failed",
          started_at: "2026-06-02T10:00:08.000Z",
          completed_at: "2026-06-02T10:00:09.000Z",
          updated_at: "2026-06-02T10:00:09.000Z",
        } as never,
        {
          id: "assistant-final-news",
          thread_id: "session-legacy-unphased-final",
          turn_id: "turn-legacy-unphased-final",
          sequence: 6,
          type: "agent_message",
          text: "## 今日国际新闻简报\n\n- 重点一：附来源。",
          status: "completed",
          started_at: "2026-06-02T10:00:28.000Z",
          completed_at: "2026-06-02T10:00:30.000Z",
          updated_at: "2026-06-02T10:00:30.000Z",
        } as never,
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-legacy-unphased-final",
    );

    expect(messages).toHaveLength(2);
    expect(messages[1]).toMatchObject({
      role: "assistant",
      content: "## 今日国际新闻简报\n\n- 重点一：附来源。",
    });
    expect(messages[1]?.content).not.toContain("我会先做");
    expect(messages[1]?.content).not.toContain("噪声较多");
    expect(messages[1]?.contentParts?.map((part) => part.type)).toEqual(["text"]);
    expect(
      messages[1]?.contentParts
        ?.filter((part) => part.type === "text")
        .map((part) => part.text),
    ).toEqual(["## 今日国际新闻简报\n\n- 重点一：附来源。"]);
    expect(messages[1]?.toolCalls).toBeUndefined();
  });

  it("后端 detail.messages 有正文时只从 timeline 恢复 canonical 用户与最终正文", () => {
    const detail: AgentSessionDetail = {
      id: "session-skill-timeline-process",
      created_at: 1,
      updated_at: 2,
      history_limit: 40,
      messages: [
        {
          role: "user",
          timestamp: 1778730438,
          content: [
            {
              type: "text",
              text: "@analysis 请只用一句话分析：E2E_SKILL_TRACE_1778730404446。",
            },
          ],
        },
        {
          role: "assistant",
          timestamp: 1778730447,
          content: [
            {
              type: "text",
              text: "该跟踪ID无上下文，无法判断具体含义。",
            },
          ],
        },
      ],
      turns: [
        {
          id: "turn-skill-process",
          thread_id: "session-skill-timeline-process",
          prompt_text:
            "@analysis 请只用一句话分析：E2E_SKILL_TRACE_1778730404446。",
          status: "completed",
          started_at: "2026-05-14T03:47:19.000Z",
          completed_at: "2026-05-14T03:47:27.000Z",
          created_at: "2026-05-14T03:47:19.000Z",
          updated_at: "2026-05-14T03:47:27.000Z",
        },
      ],
      items: [
        {
          id: "user:turn-skill-process",
          thread_id: "session-skill-timeline-process",
          turn_id: "turn-skill-process",
          sequence: 1,
          type: "user_message",
          content:
            "@analysis 请只用一句话分析：E2E_SKILL_TRACE_1778730404446。",
          status: "completed",
          started_at: "2026-05-14T03:47:19.000Z",
          completed_at: "2026-05-14T03:47:19.000Z",
          updated_at: "2026-05-14T03:47:19.000Z",
        } as never,
        {
          id: "skill:turn-skill-process",
          thread_id: "session-skill-timeline-process",
          turn_id: "turn-skill-process",
          sequence: 2,
          type: "tool_call",
          tool_name: "Skill",
          arguments: {
            skill: "analysis",
            source: "SKILL.md",
            version: "1.0.1",
          },
          output: "已从 SKILL.md 读取并执行 Skill：analysis",
          success: true,
          metadata: {
            tool_family: "skill",
            skill_source: "SKILL.md",
            markdown_content_bytes: 1633,
            skill_markdown_content:
              "---\nname: analysis\n---\n\n# Analysis Skill\n\n执行前必须读取本文件。",
          },
          status: "completed",
          started_at: "2026-05-14T03:47:19.100Z",
          completed_at: "2026-05-14T03:47:27.000Z",
          updated_at: "2026-05-14T03:47:27.000Z",
        } as never,
        {
          id: "reasoning:turn-skill-process",
          thread_id: "session-skill-timeline-process",
          turn_id: "turn-skill-process",
          sequence: 3,
          type: "reasoning",
          text: "先确认 Skill 指令，再基于可见上下文回答。",
          summary: ["先确认 Skill 指令，再基于可见上下文回答。"],
          status: "completed",
          started_at: "2026-05-14T03:47:20.000Z",
          completed_at: "2026-05-14T03:47:27.000Z",
          updated_at: "2026-05-14T03:47:27.000Z",
        } as never,
        {
          id: "assistant:turn-skill-process",
          thread_id: "session-skill-timeline-process",
          turn_id: "turn-skill-process",
          sequence: 4,
          type: "agent_message",
          text: "该跟踪ID无上下文，无法判断具体含义。",
          status: "completed",
          started_at: "2026-05-14T03:47:27.000Z",
          completed_at: "2026-05-14T03:47:27.000Z",
          updated_at: "2026-05-14T03:47:27.000Z",
        } as never,
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-skill-timeline-process",
      { compactCompletedHistory: true },
    );

    expect(messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
    ]);
    expect(messages[0]?.content).toBe(
      "@analysis 请只用一句话分析：E2E_SKILL_TRACE_1778730404446。",
    );
    expect(messages[1]).toMatchObject({
      role: "assistant",
      content: "该跟踪ID无上下文，无法判断具体含义。",
    });
    expect(messages[1]?.inlineProcessRetention).toBeUndefined();
    expect(messages[1]?.thinkingContent).toBeUndefined();
    expect(messages[1]?.toolCalls).toBeUndefined();
    expect(messages[1]?.contentParts?.map((part) => part.type)).toEqual(["text"]);
  });

  it("只有 turn 与 error Item 时不应回退合成用户 Message", () => {
    const detail: AgentSessionDetail = {
      id: "session-turn-only",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "turn-search",
          thread_id: "session-turn-only",
          prompt_text: "@搜索 OpenAI 最新模型公告，给我 3 条要点，并附来源。",
          status: "failed",
          error_message:
            "运行时权限声明需要真实确认，当前 turn 已在模型执行前等待用户确认：confirmationStatus=not_requested，askProfileKeys=web_search。已创建真实权限确认请求；请确认后重试或恢复本轮执行。",
          started_at: "2026-05-06T19:29:06.522Z",
          completed_at: "2026-05-06T19:29:06.862Z",
          created_at: "2026-05-06T19:29:06.522Z",
          updated_at: "2026-05-06T19:29:06.862Z",
        },
        {
          id: "auxiliary-runtime-projection-title",
          thread_id: "session-turn-only",
          prompt_text: "辅助标题生成 · 我来帮你搜索 OpenAI 最新模型...",
          status: "completed",
          started_at: "2026-05-06T19:29:55.849Z",
          completed_at: "2026-05-06T19:29:55.896Z",
          created_at: "2026-05-06T19:29:55.849Z",
          updated_at: "2026-05-06T19:29:55.896Z",
        },
      ],
      items: [
        {
          id: "permission-error",
          thread_id: "session-turn-only",
          turn_id: "turn-search",
          sequence: 3,
          status: "failed",
          type: "error",
          message:
            "运行时权限声明需要真实确认，当前 turn 已在模型执行前等待用户确认：confirmationStatus=not_requested，askProfileKeys=web_search。已创建真实权限确认请求；请确认后重试或恢复本轮执行。",
          started_at: "2026-05-06T19:29:06.862Z",
          completed_at: "2026-05-06T19:29:06.862Z",
          updated_at: "2026-05-06T19:29:06.862Z",
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(detail, "session-turn-only");

    expect(messages).toEqual([]);
  });
});
