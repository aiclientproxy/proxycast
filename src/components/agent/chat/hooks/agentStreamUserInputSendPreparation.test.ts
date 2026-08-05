import { afterEach, describe, expect, it, vi } from "vitest";
import type { MutableRefObject } from "react";
import type { AgentSessionExecutionRuntime } from "@/lib/api/agentExecutionRuntime";
import type { ModelCapabilitySummary } from "@/lib/model/inferModelCapabilities";
import type { Message, MessageImage } from "../types";
import {
  prepareAgentStreamUserInputSend,
  type AgentStreamUserInputSendPreparationEnv,
} from "./agentStreamUserInputSendPreparation";

function createStateSetter<T>(getValue: () => T, setValue: (value: T) => void) {
  return (next: T | ((prev: T) => T)) => {
    setValue(
      typeof next === "function" ? (next as (prev: T) => T)(getValue()) : next,
    );
  };
}

function createModelCapabilitySummary(
  inputModalities: ModelCapabilitySummary["input_modalities"],
): ModelCapabilitySummary {
  const supportsMediaInput = inputModalities.some((modality) =>
    ["image", "audio", "video", "file"].includes(modality),
  );

  return {
    capabilities: {
      vision: inputModalities.includes("image"),
      tools: true,
      streaming: true,
      json_mode: true,
      function_calling: true,
      reasoning: false,
    },
    task_families: inputModalities.includes("image")
      ? ["chat", "vision_understanding"]
      : ["chat"],
    input_modalities: inputModalities,
    output_modalities: ["text"],
    runtime_features: ["streaming", "tool_calling"],
    supports_tools: true,
    supports_reasoning: false,
    supports_prompt_cache: false,
    supports_media_input: supportsMediaInput,
    supports_media_output: false,
    context_length: 128000,
    max_output_tokens: 4096,
  };
}

function createImageAttachment(): MessageImage {
  return {
    data: "data:image/png;base64,fixture",
    mediaType: "image/png",
  };
}

describe("agentStreamUserInputSendPreparation", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  function createEnv(options?: {
    sessionId?: string | null;
    providerType?: string;
    model?: string;
    reasoningEffort?: string;
    executionRuntime?: AgentSessionExecutionRuntime | null;
  }): AgentStreamUserInputSendPreparationEnv {
    let messages: Message[] = [];
    let isSending = false;

    return {
      executionStrategy: "react",
      providerTypeRef: {
        current: options?.providerType ?? "openai",
      } as MutableRefObject<string>,
      modelRef: {
        current: options?.model ?? "gpt-5.4",
      } as MutableRefObject<string>,
      reasoningEffortRef: {
        current: options?.reasoningEffort ?? "",
      } as MutableRefObject<string>,
      sessionIdRef: {
        current: options?.sessionId ?? "session-1",
      } as MutableRefObject<string | null>,
      executionRuntime: options?.executionRuntime ?? null,
      clawTraceEnabled: false,
      getWorkspaceIdForSubmit: () => "workspace-1",
      getSyncedSessionModelPreference: () => ({
        providerType: "openai",
        model: "gpt-5.4",
      }),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      setIsSending: createStateSetter(
        () => isSending,
        (value) => {
          isSending = value;
        },
      ),
    };
  }

  it("应归一化发送参数并注入 draft", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000001")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000002");

    let messages: Message[] = [];
    let isSending = false;
    const env = {
      ...createEnv(),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      setIsSending: createStateSetter(
        () => isSending,
        (value) => {
          isSending = value;
        },
      ),
    };

    const result = prepareAgentStreamUserInputSend({
      content: "继续生成提纲",
      images: [],
      webSearch: true,
      thinking: true,
      skipUserMessage: false,
      options: {
        requestMetadata: { source: "test" },
        threadGoal: { objective: "完成 Codex 主链", tokenBudget: 50_000 },
        inputMentions: [
          {
            type: "mention",
            name: "Browser",
            path: "plugin://browser",
          },
        ],
      },
      env,
    });

    expect(result.effectiveExecutionStrategy).toBe("react");
    expect(result.effectiveProviderType).toBe("openai");
    expect(result.effectiveModel).toBe("gpt-5.4");
    expect(result.syncedSessionModelPreference).toEqual({
      providerType: "openai",
      model: "gpt-5.4",
    });
    expect(result.webSearch).toBe(true);
    expect(result.threadGoal).toEqual({
      objective: "完成 Codex 主链",
      tokenBudget: 50_000,
    });
    expect(result.inputMentions).toEqual([
      {
        type: "mention",
        name: "Browser",
        path: "plugin://browser",
      },
    ]);
    expect(result.assistantMsgId).toBe("00000000-0000-0000-0000-000000000001");
    expect(result.userMsgId).toBe("00000000-0000-0000-0000-000000000002");
    expect(messages).toHaveLength(2);
    expect(isSending).toBe(true);
  });

  it("富输入恢复结构应随 current turn metadata 一起提交给 App Server", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000012")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000013");

    const result = prepareAgentStreamUserInputSend({
      content: "请结合截图和文件继续审查",
      images: [createImageAttachment()],
      skipUserMessage: false,
      options: {
        requestMetadata: {
          harness: {
            source: "inputbar",
          },
        },
        inputRestoreDraft: {
          text: "请结合截图和文件继续审查",
          images: [createImageAttachment()],
          pathReferences: [
            {
              id: "file:/project/report.md",
              path: "/project/report.md",
              name: "report.md",
              isDir: false,
              source: "file_manager",
            },
          ],
          textElements: [{ type: "text", text: "请结合截图和文件继续审查" }],
          inputCapabilityRoute: {
            kind: "installed_skill",
            skillKey: "code-review",
            skillName: "Code Review",
          },
        },
      },
      env: createEnv(),
    });

    expect(result.submittedDraft).toMatchObject({
      pathReferences: [
        {
          path: "/project/report.md",
        },
      ],
      textElements: [{ type: "text", text: "请结合截图和文件继续审查" }],
      inputCapabilityRoute: {
        kind: "installed_skill",
        skillKey: "code-review",
      },
    });
    expect(result.requestMetadata).toMatchObject({
      path_references: [
        {
          path: "/project/report.md",
        },
      ],
      text_elements: [{ type: "text", text: "请结合截图和文件继续审查" }],
      input_capability_route: {
        kind: "installed_skill",
        skillKey: "code-review",
      },
      harness: {
        source: "inputbar",
        file_references: [
          {
            path: "/project/report.md",
          },
        ],
        input_capability_route: {
          kind: "installed_skill",
          skillKey: "code-review",
        },
      },
    });
  });

  it("应优先使用 executionRuntime 回填空的 provider/model 发送偏好", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000010")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000011");

    let messages: Message[] = [];
    let isSending = false;
    const env = {
      ...createEnv({
        providerType: "",
        model: "",
        executionRuntime: {
          session_id: "session-runtime-fallback",
          source: "runtime_snapshot",
          provider_selector: "fixture-provider",
          model_name: "fixture-model",
          execution_strategy: "react",
        },
      }),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      setIsSending: createStateSetter(
        () => isSending,
        (value) => {
          isSending = value;
        },
      ),
      getSyncedSessionModelPreference: () => null,
    };

    const result = prepareAgentStreamUserInputSend({
      content: "继续",
      images: [],
      skipUserMessage: false,
      env,
    });

    expect(result.effectiveProviderType).toBe("fixture-provider");
    expect(result.effectiveModel).toBe("fixture-model");
    expect(messages).toHaveLength(2);
    expect(isSending).toBe(true);
  });

  it("current provider 有值但 model 为空时，应跳过半截偏好并使用 runtime 完整 pair", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000210")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000211");

    const result = prepareAgentStreamUserInputSend({
      content: "你好",
      images: [],
      skipUserMessage: false,
      env: createEnv({
        providerType: "lime-hub",
        model: "",
        executionRuntime: {
          session_id: "session-runtime-pair",
          source: "runtime_snapshot",
          provider_selector: "openai",
          model_name: "gpt-5.4",
          execution_strategy: "react",
        },
      }),
    });

    expect(result.effectiveProviderType).toBe("openai");
    expect(result.effectiveModel).toBe("gpt-5.4");
  });

  it("current provider 有值但 model 为空且 runtime 不完整时，应使用已同步 session 完整 pair", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000220")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000221");

    const result = prepareAgentStreamUserInputSend({
      content: "你好",
      images: [],
      skipUserMessage: false,
      env: createEnv({
        providerType: "lime-hub",
        model: "",
        executionRuntime: {
          session_id: "session-runtime-incomplete",
          source: "runtime_snapshot",
          provider_selector: "lime-hub",
          model_name: "",
          execution_strategy: "react",
        },
      }),
    });

    expect(result.effectiveProviderType).toBe("openai");
    expect(result.effectiveModel).toBe("gpt-5.4");
  });

  it("普通自然语言发送不应在前端注入搜索模式", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000301")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000302");

    const result = prepareAgentStreamUserInputSend({
      content: "整理今天的国际新闻",
      images: [],
      skipUserMessage: false,
      env: createEnv(),
    });

    expect(result.webSearch).toBeUndefined();
  });

  it("开启 Claw Trace 时应在发送准备阶段补齐 trace metadata", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000201")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000202")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000203")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000204")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000205")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000206")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000207");

    const result = prepareAgentStreamUserInputSend({
      content: "追踪首字延迟",
      images: [],
      skipUserMessage: false,
      options: {
        requestMetadata: { source: "test" },
      },
      env: {
        ...createEnv({ sessionId: "session-trace" }),
        clawTraceEnabled: true,
      },
    });

    expect(result.requestMetadata).toMatchObject({
      source: "test",
      agentUiPerformanceTrace: expect.objectContaining({
        requestId: expect.stringMatching(/^claw_request_/),
        runId: expect.stringMatching(/^claw_run_/),
        sessionId: "session-trace",
        source: "agent-chat",
        traceId: expect.stringMatching(/^claw_trace_/),
        workspaceId: "workspace-1",
        w3cTraceContext: expect.objectContaining({
          traceparent: expect.stringMatching(
            /^00-[0-9a-f]{32}-[0-9a-f]{16}-01$/,
          ),
          traceId: expect.stringMatching(/^[0-9a-f]{32}$/),
        }),
      }),
    });
  });

  it("默认 Claw 发送应生成 summary trace metadata", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000301")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000302")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000303")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000304")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000305")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000306")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000307");

    const result = prepareAgentStreamUserInputSend({
      content: "整理今天的国际新闻",
      images: [],
      skipUserMessage: false,
      env: createEnv(),
    });

    expect(result.assistantMsg.id).toBe("00000000-0000-0000-0000-000000000301");
    expect(result.userMsg?.id).toBe("00000000-0000-0000-0000-000000000302");
    expect(result.requestMetadata).toMatchObject({
      agentUiPerformanceTrace: expect.objectContaining({
        requestId: expect.stringMatching(/^claw_request_/),
        runId: expect.stringMatching(/^claw_run_/),
        sessionId: "session-1",
        source: "agent-chat",
        traceId: expect.stringMatching(/^claw_trace_/),
        workspaceId: "workspace-1",
        w3cTraceContext: expect.objectContaining({
          traceparent: expect.stringMatching(
            /^00-[0-9a-f]{32}-[0-9a-f]{16}-01$/,
          ),
        }),
      }),
    });
    expect(result.assistantMsg.requestMetadata).toBeUndefined();
    expect(result.userMsg?.requestMetadata).toBeUndefined();
  });

  it("已解析模型能力时应把纯文本发送 gate 写入 harness metadata", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000401")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000402");

    const result = prepareAgentStreamUserInputSend({
      content: "继续分析项目结构",
      images: [],
      skipUserMessage: false,
      options: {
        modelCapabilitySummary: createModelCapabilitySummary(["text"]),
      },
      env: createEnv(),
    });

    expect(result.modelInputCapabilityGate).toMatchObject({
      status: "allowed",
      requiredInputModalities: ["text"],
      missingInputModalities: [],
      reason: null,
    });
    expect(result.requestMetadata).toMatchObject({
      harness: {
        model_input_capability_gate: {
          status: "allowed",
          requiredInputModalities: ["text"],
          supportedInputModalities: ["text"],
        },
      },
    });
  });

  it("图片输入遇到 text-only 模型时应记录 blocked gate 且保留既有 metadata", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000411")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000412");

    const result = prepareAgentStreamUserInputSend({
      content: "描述这张图",
      images: [createImageAttachment()],
      skipUserMessage: false,
      options: {
        requestMetadata: {
          source: "test",
          harness: {
            existing_signal: true,
          },
        },
        modelCapabilitySummary: createModelCapabilitySummary(["text"]),
      },
      env: createEnv(),
    });

    expect(result.modelInputCapabilityGate).toMatchObject({
      status: "blocked",
      requiredInputModalities: ["text", "image"],
      supportedInputModalities: ["text"],
      missingInputModalities: ["image"],
      requiresMediaInput: true,
      reason: "missing_input_modalities",
    });
    expect(result.requestMetadata).toMatchObject({
      source: "test",
      harness: {
        existing_signal: true,
        model_input_capability_gate: {
          status: "blocked",
          missingInputModalities: ["image"],
        },
      },
    });
  });

  it("图片输入缺少模型能力 summary 时应记录 unknown gate 交给后续边界决策", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000421")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000422");

    const result = prepareAgentStreamUserInputSend({
      content: "分析这张图",
      images: [createImageAttachment()],
      skipUserMessage: false,
      env: createEnv(),
    });

    expect(result.modelInputCapabilityGate).toMatchObject({
      status: "unknown",
      requiredInputModalities: ["text", "image"],
      missingInputModalities: ["text", "image"],
      requiresMediaInput: true,
      reason: "missing_capability_summary",
    });
    expect(result.requestMetadata).toMatchObject({
      harness: {
        model_input_capability_gate: {
          status: "unknown",
          reason: "missing_capability_summary",
        },
      },
    });
  });

  it("displayContent 应透传给用户消息草稿", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000101")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000102");

    let messages: Message[] = [];
    let isSending = false;
    const env = {
      ...createEnv(),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      setIsSending: createStateSetter(
        () => isSending,
        (value) => {
          isSending = value;
        },
      ),
    };

    prepareAgentStreamUserInputSend({
      content: "/image_generate 生成 春日咖啡馆插画",
      images: [],
      skipUserMessage: false,
      options: {
        displayContent: "@配图 生成 春日咖啡馆插画",
      },
      env,
    });

    expect(messages[0]).toMatchObject({
      id: "00000000-0000-0000-0000-000000000102",
      role: "user",
      content: "@配图 生成 春日咖啡馆插画",
    });
    expect(isSending).toBe(true);
  });

  it("应允许跳过 user message 并覆盖 model 与 assistant draft", () => {
    vi.spyOn(crypto, "randomUUID").mockReturnValueOnce(
      "00000000-0000-0000-0000-000000000003",
    );

    let messages: Message[] = [];
    let isSending = false;
    const env = {
      ...createEnv({
        sessionId: "session-current",
        providerType: "claude",
        model: "sonnet",
      }),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      setIsSending: createStateSetter(
        () => isSending,
        (value) => {
          isSending = value;
        },
      ),
      getSyncedSessionModelPreference: () => null,
    };

    const result = prepareAgentStreamUserInputSend({
      content: "继续生成提纲",
      images: [],
      skipUserMessage: true,
      executionStrategyOverride: "react",
      modelOverride: "opus",
      options: {
        assistantDraft: {
          content: "正在整理",
        },
      },
      env,
    });

    expect(result.effectiveExecutionStrategy).toBe("react");
    expect(result.effectiveProviderType).toBe("claude");
    expect(result.effectiveModel).toBe("opus");
    expect(result.syncedSessionModelPreference).toBeNull();
    expect(result.userMsgId).toBeNull();
    expect(messages).toHaveLength(1);
    expect(messages[0]?.content).toBe("正在整理");
    expect(isSending).toBe(true);
  });

  it("应优先使用 options 里的 provider/model override", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000008")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000009");

    let messages: Message[] = [];
    let isSending = false;
    const env = {
      ...createEnv({
        providerType: "openai",
        model: "gpt-5.4-mini",
      }),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      setIsSending: createStateSetter(
        () => isSending,
        (value) => {
          isSending = value;
        },
      ),
    };

    const result = prepareAgentStreamUserInputSend({
      content: "改成翻译链路",
      images: [],
      skipUserMessage: false,
      options: {
        providerOverride: "translation-provider",
        modelOverride: "translation-model",
      },
      env,
    });

    expect(result.effectiveProviderType).toBe("translation-provider");
    expect(result.effectiveModel).toBe("translation-model");
    expect(result.modelOverride).toBe("translation-model");
    expect(messages).toHaveLength(2);
    expect(isSending).toBe(true);
  });

  it("应允许单次发送覆盖系统提示词", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000108")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000109");

    const result = prepareAgentStreamUserInputSend({
      content: "请快速回答",
      images: [],
      skipUserMessage: false,
      systemPrompt: "默认系统提示",
      options: {
        systemPromptOverride: "快速响应系统提示",
      },
      env: createEnv(),
    });

    expect(result.systemPrompt).toBe("快速响应系统提示");
  });

  it("Skill launch metadata 应标记助手草稿保留内联过程", () => {
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000128")
      .mockReturnValueOnce("00000000-0000-0000-0000-000000000129");

    let messages: Message[] = [];
    const env = {
      ...createEnv(),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
    };

    const result = prepareAgentStreamUserInputSend({
      content: "开始执行品牌知识库 Skill",
      images: [],
      skipUserMessage: false,
      options: {
        requestMetadata: {
          harness: {
            service_scene_launch: {
              kind: "local_service_skill",
              service_scene_run: {
                skill_id: "brand-product-knowledge-builder",
              },
            },
          },
        },
      },
      env,
    });

    expect(result.assistantMsg.inlineProcessRetention).toBe("skill");
    expect(messages[1]?.inlineProcessRetention).toBe("skill");
  });
});
