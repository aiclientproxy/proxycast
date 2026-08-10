import { describe, expect, it } from "vitest";
import { buildUserInputSubmitOp } from "../utils/buildUserInputSubmitOp";
import { buildAgentStreamSubmitOp } from "./agentStreamSubmitOpController";

describe("agentStreamSubmitOpController", () => {
  it("应按 current stream submit 语义构造 runtime submitOp", () => {
    const op = buildAgentStreamSubmitOp({
      activeThreadId: "thread-fast-1",
      content: "只回答一个字：好",
      images: [],
      eventName: "agent_stream_fast",
      effectiveAccessMode: "current",
      effectiveProviderType: "deepseek",
      effectiveModel: "deepseek-chat",
    });

    expect(op).toEqual({
      type: "user_input",
      eventName: "agent_stream_fast",
      turn: {
        threadId: "thread-fast-1",
        input: [{ type: "text", text: "只回答一个字：好" }],
        model: "deepseek-chat",
        approvalPolicy: "on-request",
        permissions: ":workspace",
      },
    });
    for (const field of [
      "text",
      "images",
      "preferences",
      "sessionId",
      "threadId",
      "workspaceId",
      "turnId",
      "systemPrompt",
      "metadata",
    ]) {
      expect(op).not.toHaveProperty(field);
    }
  });

  it("应与底层 user_input builder 保持 payload 等价", () => {
    const streamOp = buildAgentStreamSubmitOp({
      activeThreadId: "thread-social-1",
      content: "继续生成社媒初稿",
      images: [
        {
          data: "base64-image",
          mediaType: "image/png",
        },
      ],
      eventName: "agent_stream_x",
      requestMetadata: {
        harness: {
          preferences: {
            web_search: false,
            thinking: true,
          },
          theme: "general",
          session_mode: "general_workbench",
          gate_key: "write_mode",
          run_title: "社媒初稿",
          content_id: "content-social-1",
        },
      },
      executionRuntime: {
        session_id: "session-social-1",
        source: "runtime_snapshot",
        provider_selector: "openai",
        model_name: "gpt-4.1",
        execution_strategy: "react",
        recent_preferences: {
          task: false,
          subagent: false,
        },
        recent_theme: "general",
        recent_session_mode: "general_workbench",
        recent_gate_key: "write_mode",
        recent_run_title: "社媒初稿",
        recent_content_id: "content-social-1",
      },
      syncedRecentPreferences: {
        task: false,
        subagent: false,
      },
      syncedSessionModelPreference: {
        providerType: "openai",
        model: "gpt-4.1",
      },
      effectiveAccessMode: "current",
      effectiveProviderType: "openai",
      effectiveModel: "gpt-4.1",
    });

    const directOp = buildUserInputSubmitOp({
      content: "继续生成社媒初稿",
      images: [
        {
          data: "base64-image",
          mediaType: "image/png",
        },
      ],
      threadId: "thread-social-1",
      eventName: "agent_stream_x",
      requestMetadata: {
        harness: {
          preferences: {
            web_search: false,
            thinking: true,
          },
          theme: "general",
          session_mode: "general_workbench",
          gate_key: "write_mode",
          run_title: "社媒初稿",
          content_id: "content-social-1",
        },
      },
      executionRuntime: {
        session_id: "session-social-1",
        source: "runtime_snapshot",
        provider_selector: "openai",
        model_name: "gpt-4.1",
        execution_strategy: "react",
        recent_preferences: {
          task: false,
          subagent: false,
        },
        recent_theme: "general",
        recent_session_mode: "general_workbench",
        recent_gate_key: "write_mode",
        recent_run_title: "社媒初稿",
        recent_content_id: "content-social-1",
      },
      syncedRecentPreferences: {
        task: false,
        subagent: false,
      },
      syncedSessionModelPreference: {
        providerType: "openai",
        model: "gpt-4.1",
      },
      effectiveAccessMode: "current",
      effectiveProviderType: "openai",
      effectiveModel: "gpt-4.1",
    });

    expect(streamOp).toEqual(directOp);
  });

  it("应把 reasoning effort 带入 typed turn", () => {
    const op = buildAgentStreamSubmitOp({
      activeThreadId: "thread-reasoning-1",
      content: "继续",
      images: [],
      eventName: "agent_stream_reasoning",
      effectiveAccessMode: "current",
      effectiveProviderType: "openai",
      effectiveModel: "o3-mini",
      reasoningEffort: " high ",
    });

    expect(op.turn.effort).toBe("high");
    expect(op).not.toHaveProperty("preferences");
  });

  it("搜索命令不应把旧 search 偏好写入 typed turn", () => {
    const op = buildAgentStreamSubmitOp({
      activeThreadId: "thread-search-1",
      content: "@搜索 关键词:AI 行业新闻",
      images: [],
      eventName: "agent_stream_search",
      effectiveAccessMode: "current",
      effectiveProviderType: "openai",
      effectiveModel: "gpt-5.5",
    });

    expect(op.turn.model).toBe("gpt-5.5");
    expect(op).not.toHaveProperty("preferences");
    expect(op.turn).not.toHaveProperty("webSearch");
    expect(op.turn).not.toHaveProperty("searchMode");
  });
});
