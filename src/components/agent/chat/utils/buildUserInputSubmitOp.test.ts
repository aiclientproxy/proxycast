import { describe, expect, it } from "vitest";
import { createAgentSessionTurnStartParamsFromUserInputOp } from "@/lib/api/agentProtocol";
import type { ModelCapabilitySummary } from "@/lib/model/inferModelCapabilities";
import { MODEL_INPUT_CAPABILITY_GAP_ERROR_PREFIX } from "@/lib/model/modelCapabilitySendGate";
import type { CollaborationMode } from "@limecloud/app-server-client";
import { buildUserInputSubmitOp } from "./buildUserInputSubmitOp";

type CurrentTurnStartParams = ReturnType<
  typeof createAgentSessionTurnStartParamsFromUserInputOp
>;
type CurrentUserInputOp = ReturnType<typeof buildUserInputSubmitOp>;

interface ExpectedCurrentTurnStartWire {
  threadId: string;
  text: string;
  imageUrls?: string[];
  mentions?: Array<{ name: string; path: string }>;
  model?: string;
  effort?: string;
  collaborationMode?: CollaborationMode;
  approvalPolicy: string;
  permissions: string;
  eventName: string;
  metadata?: Record<string, unknown>;
}

function createImageCommandMetadata(
  prompt: string,
  providerId: string,
  model: string,
) {
  return {
    harness: {
      image_command_intent: {
        image_task: {
          prompt,
          provider_id: providerId,
          model,
          runtime_contract: {
            contract_key: "image_generation",
            routing_slot: "image_generation_model",
          },
        },
      },
    },
  };
}

function createExpectedTurn(
  expected: ExpectedCurrentTurnStartWire,
  includeRendererEventName: boolean,
) {
  const additionalContext = {
    ...(includeRendererEventName
      ? {
          rendererEventName: {
            kind: "application" as const,
            value: expected.eventName,
          },
        }
      : {}),
    ...(expected.metadata
      ? {
          metadata: {
            kind: "application" as const,
            value: JSON.stringify(expected.metadata),
          },
        }
      : {}),
  };

  return {
    threadId: expected.threadId,
    input: [
      { type: "text" as const, text: expected.text },
      ...(expected.mentions ?? []).map((mention) => ({
        type: "mention" as const,
        ...mention,
      })),
      ...(expected.imageUrls ?? []).map((url) => ({
        type: "image" as const,
        url,
      })),
    ],
    ...(expected.model ? { model: expected.model } : {}),
    ...(expected.effort ? { effort: expected.effort } : {}),
    ...(expected.collaborationMode
      ? { collaborationMode: expected.collaborationMode }
      : {}),
    approvalPolicy: expected.approvalPolicy,
    permissions: expected.permissions,
    ...(Object.keys(additionalContext).length > 0 ? { additionalContext } : {}),
  };
}

function expectDeadSubmitFieldsAbsent(value: Record<string, unknown>): void {
  for (const field of [
    "runtimeOptions",
    "providerPreference",
    "providerConfig",
    "modelProvider",
    "webSearch",
    "searchMode",
    "thinking",
    "executionStrategy",
    "autoContinue",
    "queueIfBusy",
    "queuedTurnId",
    "skipPreSubmitResume",
    "systemPrompt",
    "sessionId",
    "workspaceId",
    "turnId",
    "metadata",
  ]) {
    expect(value).not.toHaveProperty(field);
  }
}

function expectCurrentUserInputOp(
  op: CurrentUserInputOp,
  expected: ExpectedCurrentTurnStartWire,
): void {
  expect(op).toEqual({
    type: "user_input",
    eventName: expected.eventName,
    turn: createExpectedTurn(expected, false),
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
  expectDeadSubmitFieldsAbsent(op.turn as unknown as Record<string, unknown>);
}

function expectCurrentTurnStartWire(
  request: CurrentTurnStartParams,
  expected: ExpectedCurrentTurnStartWire,
): void {
  expect(request).toEqual(createExpectedTurn(expected, true));
  expectDeadSubmitFieldsAbsent(request as unknown as Record<string, unknown>);
}

describe("buildUserInputSubmitOp", () => {
  const textOnlyModelCapabilitySummary: ModelCapabilitySummary = {
    capabilities: {
      vision: false,
      tools: true,
      streaming: true,
      json_mode: true,
      function_calling: true,
      reasoning: false,
    },
    task_families: ["chat"],
    input_modalities: ["text"],
    output_modalities: ["text"],
    runtime_features: ["streaming", "tool_calling"],
    supports_tools: true,
    supports_reasoning: false,
    supports_prompt_cache: false,
    supports_media_input: false,
    supports_media_output: false,
    context_length: 128000,
    max_output_tokens: 4096,
  };

  it("应构造最小 user_input op，并裁掉 steady-state 字段", () => {
    const op = buildUserInputSubmitOp({
      content: "继续生成社媒初稿",
      images: [
        {
          data: "base64-image",
          mediaType: "image/png",
        },
      ],
      inputMentions: [
        {
          type: "mention",
          name: "Browser",
          path: "plugin://browser",
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
          webSearch: false,
          thinking: true,
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

    expectCurrentUserInputOp(op, {
      threadId: "thread-social-1",
      text: "继续生成社媒初稿",
      mentions: [{ name: "Browser", path: "plugin://browser" }],
      imageUrls: ["data:image/png;base64,base64-image"],
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_x",
    });

    const request = createAgentSessionTurnStartParamsFromUserInputOp(op);
    expectCurrentTurnStartWire(request, {
      threadId: "thread-social-1",
      text: "继续生成社媒初稿",
      mentions: [{ name: "Browser", path: "plugin://browser" }],
      imageUrls: ["data:image/png;base64,base64-image"],
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_x",
    });
  });

  it("应把 renderer user message identity 透传到 typed turn/start", () => {
    const op = buildUserInputSubmitOp({
      content: "继续",
      images: [],
      threadId: "thread-identity",
      clientUserMessageId: "user-message-identity",
      eventName: "agent_stream_identity",
      effectiveAccessMode: "current",
      effectiveProviderType: "openai",
      effectiveModel: "gpt-5.4",
    });

    expect(op.turn.clientUserMessageId).toBe("user-message-identity");
    expect(
      createAgentSessionTurnStartParamsFromUserInputOp(op).clientUserMessageId,
    ).toBe("user-message-identity");
  });

  it("应把 typed Plan selection 降为完整 Codex collaboration mode", () => {
    const op = buildUserInputSubmitOp({
      content: "先制定计划",
      images: [],
      threadId: "thread-plan",
      eventName: "agent_stream_plan",
      collaborationMode: {
        name: "Plan",
        mode: "plan",
        model: null,
        reasoning_effort: "medium",
      },
      reasoningEffort: "high",
      requestMetadata: {
        harness: {
          plan_implementation_decision: { decision: "adjustment" },
        },
      },
      effectiveAccessMode: "current",
      effectiveProviderType: "openai",
      effectiveModel: "gpt-5.4",
    });
    const collaborationMode = {
      mode: "plan" as const,
      settings: {
        model: "gpt-5.4",
        reasoning_effort: "medium",
        developer_instructions: null,
      },
    };
    const expected = {
      threadId: "thread-plan",
      text: "先制定计划",
      model: "gpt-5.4",
      effort: "medium",
      collaborationMode,
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_plan",
      metadata: {
        harness: {
          plan_implementation_decision: { decision: "adjustment" },
        },
      },
    } satisfies ExpectedCurrentTurnStartWire;

    expectCurrentUserInputOp(op, expected);
    expectCurrentTurnStartWire(
      createAgentSessionTurnStartParamsFromUserInputOp(op),
      expected,
    );
    expect(JSON.stringify(op)).not.toContain("collaboration_mode");
  });

  it("应迁移尚未同步到 runtime 的显式偏好并保留其他 metadata", () => {
    const op = buildUserInputSubmitOp({
      content: "切到发布确认",
      images: [],
      threadId: "thread-social-1",
      eventName: "agent_stream_y",
      requestMetadata: {
        harness: {
          preferences: {
            thinking: true,
          },
          theme: "general",
          session_mode: "general_workbench",
          gate_key: "publish_confirm",
          run_title: "发布确认",
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
      },
      syncedRecentPreferences: {
        task: false,
        subagent: false,
      },
      syncedSessionModelPreference: {
        providerType: "openai",
        model: "gpt-4.1",
      },
      effectiveAccessMode: "full-access",
      effectiveProviderType: "openai",
      effectiveModel: "gpt-5",
      modelOverride: "gpt-5",
    });

    expectCurrentUserInputOp(op, {
      threadId: "thread-social-1",
      text: "切到发布确认",
      model: "gpt-5",
      approvalPolicy: "never",
      permissions: ":danger-full-access",
      eventName: "agent_stream_y",
      metadata: {
        harness: {
          gate_key: "publish_confirm",
          run_title: "发布确认",
        },
      },
    });
    const request = createAgentSessionTurnStartParamsFromUserInputOp(op);
    expectCurrentTurnStartWire(request, {
      threadId: "thread-social-1",
      text: "切到发布确认",
      model: "gpt-5",
      approvalPolicy: "never",
      permissions: ":danger-full-access",
      eventName: "agent_stream_y",
      metadata: {
        harness: {
          gate_key: "publish_confirm",
          run_title: "发布确认",
        },
      },
    });
  });

  it("图片输入不满足 selected model capability 时应在 submit op 边界 fail closed", () => {
    expect(() =>
      buildUserInputSubmitOp({
        content: "描述这张图",
        images: [
          {
            data: "base64-image",
            mediaType: "image/png",
          },
        ],
        threadId: "thread-image-1",
        eventName: "agent_stream_image",
        effectiveAccessMode: "current",
        effectiveProviderType: "openai",
        effectiveModel: "gpt-4.1-text",
        modelCapabilitySummary: textOnlyModelCapabilitySummary,
      }),
    ).toThrow(`${MODEL_INPUT_CAPABILITY_GAP_ERROR_PREFIX}:`);
  });

  it("图片输入缺少模型能力 summary 时不应阻断 submit op", () => {
    const op = buildUserInputSubmitOp({
      content: "描述这张图",
      images: [
        {
          data: "base64-image",
          mediaType: "image/png",
        },
      ],
      threadId: "thread-image-unknown",
      eventName: "agent_stream_image_unknown",
      effectiveAccessMode: "current",
      effectiveProviderType: "fixture-provider",
      effectiveModel: "fixture-model",
      modelCapabilitySummary: null,
    });

    expectCurrentUserInputOp(op, {
      threadId: "thread-image-unknown",
      text: "描述这张图",
      imageUrls: ["data:image/png;base64,base64-image"],
      model: "fixture-model",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_image_unknown",
    });
  });

  it("中途切换模型但会话尚未同步时应在 submit payload 带上当前模型", () => {
    const op = buildUserInputSubmitOp({
      content: "继续",
      images: [],
      threadId: "thread-model-pending",
      eventName: "agent_stream_model_pending",
      executionRuntime: {
        session_id: "session-model-pending",
        source: "runtime_snapshot",
        provider_selector: "deepseek",
        model_name: "deepseek-v4-flash",
        execution_strategy: "react",
      },
      syncedRecentPreferences: null,
      syncedSessionModelPreference: null,
      effectiveAccessMode: "current",
      effectiveProviderType: "deepseek",
      effectiveModel: "deepseek-v4-flash",
    });

    expectCurrentUserInputOp(op, {
      threadId: "thread-model-pending",
      text: "继续",
      model: "deepseek-v4-flash",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_model_pending",
    });
  });

  it("App Server current turn/start 投影应只提交 typed model，provider 不上 wire", () => {
    const op = buildUserInputSubmitOp({
      content: "继续",
      images: [],
      threadId: "thread-app-server-current",
      eventName: "agent_stream_app_server_current",
      executionRuntime: {
        session_id: "session-app-server-current",
        source: "runtime_snapshot",
        provider_selector: "openai",
        model_name: "gpt-4.1",
        execution_strategy: "react",
      },
      syncedRecentPreferences: null,
      syncedSessionModelPreference: null,
      effectiveAccessMode: "full-access",
      effectiveProviderType: "custom-provider",
      effectiveModel: "mimo-v2.5-pro",
    });

    const request = createAgentSessionTurnStartParamsFromUserInputOp(op);
    expectCurrentTurnStartWire(request, {
      threadId: "thread-app-server-current",
      text: "继续",
      model: "mimo-v2.5-pro",
      approvalPolicy: "never",
      permissions: ":danger-full-access",
      eventName: "agent_stream_app_server_current",
    });
  });

  it("显式 modelOverride 应解析为唯一 typed turn model", () => {
    const op = buildUserInputSubmitOp({
      content: "使用翻译服务模型",
      images: [],
      threadId: "thread-translation-1",
      eventName: "agent_stream_translation",
      executionRuntime: {
        session_id: "session-translation-1",
        source: "runtime_snapshot",
        provider_selector: "openai",
        model_name: "gpt-4.1",
        execution_strategy: "react",
        recent_preferences: {
          task: false,
          subagent: false,
        },
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
      effectiveProviderType: "translation-provider",
      effectiveModel: "gpt-4.1",
      modelOverride: "translation-model",
    });

    expectCurrentUserInputOp(op, {
      threadId: "thread-translation-1",
      text: "使用翻译服务模型",
      model: "translation-model",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_translation",
    });
  });

  it("首页首发不应把旧 submit 快路径标记带入 current turn/start", () => {
    const op = buildUserInputSubmitOp({
      content: "只回答一个字：好",
      images: [],
      threadId: "thread-fast-1",
      eventName: "agent_stream_fast",
      effectiveAccessMode: "current",
      effectiveProviderType: "deepseek",
      effectiveModel: "deepseek-chat",
    });

    const request = createAgentSessionTurnStartParamsFromUserInputOp(op);
    const expected = {
      threadId: "thread-fast-1",
      text: "只回答一个字：好",
      model: "deepseek-chat",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_fast",
    } satisfies ExpectedCurrentTurnStartWire;

    expectCurrentUserInputOp(op, expected);
    expectCurrentTurnStartWire(request, expected);
  });

  it("应透传 model slot metadata，并提交 typed model override", () => {
    const op = buildUserInputSubmitOp({
      content: "只回答一个字：好",
      images: [],
      threadId: "thread-fast-routing-1",
      eventName: "agent_stream_fast_routing",
      requestMetadata: {
        harness: {
          model_slots: {
            fast: {
              provider: "responsive-provider",
              model: "fast-chat",
              source: "service_models.responsive_chat",
              reason: "service_model_preference",
            },
          },
          browser_assist: {
            enabled: true,
            profile_key: "general_browser_assist",
          },
        },
      },
      executionRuntime: null,
      syncedRecentPreferences: null,
      syncedSessionModelPreference: null,
      effectiveAccessMode: "current",
      effectiveProviderType: "deepseek",
      effectiveModel: "deepseek-v4-pro",
    });

    expectCurrentUserInputOp(op, {
      threadId: "thread-fast-routing-1",
      text: "只回答一个字：好",
      model: "deepseek-v4-pro",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_fast_routing",
      metadata: {
        harness: {
          model_slots: {
            fast: {
              provider: "responsive-provider",
              model: "fast-chat",
              source: "service_models.responsive_chat",
              reason: "service_model_preference",
            },
          },
          browser_assist: {
            enabled: true,
            profile_key: "general_browser_assist",
          },
        },
      },
    });
  });

  it("空 provider 不应阻止提交 typed model", () => {
    const op = buildUserInputSubmitOp({
      content: "分析这个文件夹",
      images: [],
      threadId: "thread-partial-model",
      eventName: "agent_stream_partial_model",
      effectiveAccessMode: "current",
      effectiveProviderType: "",
      effectiveModel: "gpt-5.5",
    });

    expectCurrentUserInputOp(op, {
      threadId: "thread-partial-model",
      text: "分析这个文件夹",
      model: "gpt-5.5",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_partial_model",
    });
  });

  it("图片生成首发应提交聊天编排模型，避免 presentation 路由缺失或误锁图片模型槽位", () => {
    const op = buildUserInputSubmitOp({
      content: "@Nanobanana Pro 生成一张广州塔春天照片",
      images: [],
      threadId: "thread-image-1",
      eventName: "agent_stream_image",
      requestMetadata: createImageCommandMetadata(
        "一张广州塔春天照片",
        "fal",
        "fal-ai/nano-banana-pro",
      ),
      executionRuntime: null,
      syncedRecentPreferences: null,
      syncedSessionModelPreference: null,
      effectiveAccessMode: "current",
      effectiveProviderType: "deepseek",
      effectiveModel: "deepseek-v4-pro",
    });

    const expected = {
      threadId: "thread-image-1",
      text: "@Nanobanana Pro 生成一张广州塔春天照片",
      model: "deepseek-v4-pro",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_image",
      metadata: createImageCommandMetadata(
        "一张广州塔春天照片",
        "fal",
        "fal-ai/nano-banana-pro",
      ),
    } satisfies ExpectedCurrentTurnStartWire;

    expectCurrentUserInputOp(op, expected);
    const request = createAgentSessionTurnStartParamsFromUserInputOp(op);
    expectCurrentTurnStartWire(request, expected);
  });

  it("图片生成命令已同步会话模型时不应重复提交 model", () => {
    const op = buildUserInputSubmitOp({
      content: "@Nanobanana Pro 生成一张广州塔春天照片",
      images: [],
      threadId: "thread-image-2",
      eventName: "agent_stream_image_synced",
      requestMetadata: createImageCommandMetadata(
        "一张广州塔春天照片",
        "fal",
        "fal-ai/nano-banana-pro",
      ),
      executionRuntime: {
        session_id: "session-image-2",
        source: "runtime_snapshot",
        provider_selector: "deepseek",
        model_name: "deepseek-v4-pro",
        execution_strategy: "react",
      },
      syncedRecentPreferences: null,
      syncedSessionModelPreference: {
        providerType: "deepseek",
        model: "deepseek-v4-pro",
      },
      effectiveAccessMode: "current",
      effectiveProviderType: "deepseek",
      effectiveModel: "deepseek-v4-pro",
    });

    const expected = {
      threadId: "thread-image-2",
      text: "@Nanobanana Pro 生成一张广州塔春天照片",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_image_synced",
      metadata: createImageCommandMetadata(
        "一张广州塔春天照片",
        "fal",
        "fal-ai/nano-banana-pro",
      ),
    } satisfies ExpectedCurrentTurnStartWire;

    expectCurrentUserInputOp(op, expected);
    const request = createAgentSessionTurnStartParamsFromUserInputOp(op);
    expectCurrentTurnStartWire(request, expected);
  });

  it("图片生成命令当前有效模型为图片通道时应抑制 turn model", () => {
    const op = buildUserInputSubmitOp({
      content: "@Agnes Image 2.1 Flash 生成一张广州夏天照片",
      images: [],
      threadId: "thread-image-agnes",
      eventName: "agent_stream_image_agnes",
      requestMetadata: createImageCommandMetadata(
        "一张广州夏天照片",
        "agnes",
        "agnes-image-2.1-flash",
      ),
      executionRuntime: null,
      syncedRecentPreferences: null,
      syncedSessionModelPreference: {
        providerType: "deepseek",
        model: "deepseek-v4-pro",
      },
      effectiveAccessMode: "current",
      effectiveProviderType: "agnes",
      effectiveModel: "agnes-image-2.1-flash",
    });

    const expected = {
      threadId: "thread-image-agnes",
      text: "@Agnes Image 2.1 Flash 生成一张广州夏天照片",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_image_agnes",
      metadata: createImageCommandMetadata(
        "一张广州夏天照片",
        "agnes",
        "agnes-image-2.1-flash",
      ),
    } satisfies ExpectedCurrentTurnStartWire;

    expectCurrentUserInputOp(op, expected);
    const request = createAgentSessionTurnStartParamsFromUserInputOp(op);
    expectCurrentTurnStartWire(request, expected);
  });

  it("图片生成命令没有文本模型候选时不应提交图片 provider 作为编排模型", () => {
    const op = buildUserInputSubmitOp({
      content: "@Agnes Image 2.1 Flash 生成一张广州夏天照片",
      images: [],
      threadId: "thread-image-no-text-model",
      eventName: "agent_stream_image_no_text",
      requestMetadata: createImageCommandMetadata(
        "一张广州夏天照片",
        "agnes",
        "agnes-image-2.1-flash",
      ),
      executionRuntime: null,
      syncedRecentPreferences: null,
      syncedSessionModelPreference: null,
      effectiveAccessMode: "current",
      effectiveProviderType: "agnes",
      effectiveModel: "agnes-image-2.1-flash",
    });

    const expected = {
      threadId: "thread-image-no-text-model",
      text: "@Agnes Image 2.1 Flash 生成一张广州夏天照片",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_image_no_text",
      metadata: createImageCommandMetadata(
        "一张广州夏天照片",
        "agnes",
        "agnes-image-2.1-flash",
      ),
    } satisfies ExpectedCurrentTurnStartWire;

    expectCurrentUserInputOp(op, expected);
    const request = createAgentSessionTurnStartParamsFromUserInputOp(op);
    expectCurrentTurnStartWire(request, expected);
  });

  it("搜索请求不应把旧 search 控制字段带入 current turn/start wire", () => {
    const op = buildUserInputSubmitOp({
      content: "请搜索最新 AI 新闻",
      images: [],
      threadId: "thread-search-1",
      eventName: "agent_stream_search",
      effectiveAccessMode: "current",
      effectiveProviderType: "deepseek",
      effectiveModel: "deepseek-chat",
    });

    const expected = {
      threadId: "thread-search-1",
      text: "请搜索最新 AI 新闻",
      model: "deepseek-chat",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_search",
    } satisfies ExpectedCurrentTurnStartWire;

    expectCurrentUserInputOp(op, expected);
    const request = createAgentSessionTurnStartParamsFromUserInputOp(op);
    expectCurrentTurnStartWire(request, expected);
  });

  it("应把输入框推理强度透传到 current turn/start effort", () => {
    const op = buildUserInputSubmitOp({
      content: "先深入推理再给出实施计划",
      images: [],
      threadId: "thread-reasoning-effort-1",
      eventName: "agent_stream_reasoning_effort",
      effectiveAccessMode: "current",
      effectiveProviderType: "openai",
      effectiveModel: "gpt-5.5",
      reasoningEffort: " high ",
    });

    const expected = {
      threadId: "thread-reasoning-effort-1",
      text: "先深入推理再给出实施计划",
      model: "gpt-5.5",
      effort: "high",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_reasoning_effort",
    } satisfies ExpectedCurrentTurnStartWire;

    expectCurrentUserInputOp(op, expected);
    const request = createAgentSessionTurnStartParamsFromUserInputOp(op);
    expectCurrentTurnStartWire(request, expected);
  });

  it("旧 metadata 搜索和思考偏好不应进入 current turn/start wire", () => {
    const op = buildUserInputSubmitOp({
      content: "搜索并深度分析今天的 AI 新闻",
      images: [],
      threadId: "thread-search-thinking-1",
      eventName: "agent_stream_search_thinking",
      requestMetadata: {
        harness: {
          preferences: {
            web_search: true,
            thinking: true,
          },
        },
      },
      executionRuntime: {
        session_id: "session-search-thinking-1",
        source: "runtime_snapshot",
        recent_preferences: {
          task: false,
          subagent: false,
        },
      },
      syncedRecentPreferences: {
        task: false,
        subagent: false,
      },
      syncedSessionModelPreference: null,
      effectiveAccessMode: "current",
      effectiveProviderType: "openai",
      effectiveModel: "gpt-5.5",
    });

    const expected = {
      threadId: "thread-search-thinking-1",
      text: "搜索并深度分析今天的 AI 新闻",
      model: "gpt-5.5",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_search_thinking",
    } satisfies ExpectedCurrentTurnStartWire;

    expectCurrentUserInputOp(op, expected);
    const request = createAgentSessionTurnStartParamsFromUserInputOp(op);
    expectCurrentTurnStartWire(request, expected);
  });

  it("应清理旧 metadata 偏好并只把业务 metadata 投影到 additionalContext", () => {
    const op = buildUserInputSubmitOp({
      content: "启用搜索和思考",
      images: [],
      threadId: "thread-legacy-prefs-1",
      eventName: "agent_stream_legacy_prefs",
      requestMetadata: {
        harness: {
          preferences: {
            webSearchEnabled: true,
            thinkingEnabled: true,
          },
          turn_purpose: "content_review",
        },
      },
      executionRuntime: null,
      syncedRecentPreferences: null,
      syncedSessionModelPreference: null,
      effectiveAccessMode: "current",
      effectiveProviderType: "openai",
      effectiveModel: "gpt-5.5",
    });

    const expected = {
      threadId: "thread-legacy-prefs-1",
      text: "启用搜索和思考",
      model: "gpt-5.5",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_legacy_prefs",
      metadata: {
        harness: {
          turn_purpose: "content_review",
        },
      },
    } satisfies ExpectedCurrentTurnStartWire;

    expectCurrentUserInputOp(op, expected);
    const request = createAgentSessionTurnStartParamsFromUserInputOp(op);
    expectCurrentTurnStartWire(request, expected);
  });

  it("输入框自然语言新闻请求不应提交搜索、思考或旧执行策略选择", () => {
    const op = buildUserInputSubmitOp({
      content: "整理今天的国际新闻",
      images: [],
      threadId: "thread-news-1",
      eventName: "agent_stream_news",
      effectiveAccessMode: "current",
      effectiveProviderType: "openai",
      effectiveModel: "gpt-5.5",
    });

    const expected = {
      threadId: "thread-news-1",
      text: "整理今天的国际新闻",
      model: "gpt-5.5",
      approvalPolicy: "on-request",
      permissions: ":workspace",
      eventName: "agent_stream_news",
    } satisfies ExpectedCurrentTurnStartWire;

    expectCurrentUserInputOp(op, expected);
    const request = createAgentSessionTurnStartParamsFromUserInputOp(op);
    expectCurrentTurnStartWire(request, expected);
  });
});
