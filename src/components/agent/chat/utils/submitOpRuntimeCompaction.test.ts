import { existsSync } from "node:fs";
import { resolve } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";
import { buildSubmitOpRuntimeCompaction } from "./submitOpRuntimeCompaction";

const RETIRED_TOOL_PREFERENCE_PATH =
  "src/components/agent/chat/utils/submitOpToolPreferenceCompaction.ts";

function buildCompaction(
  options: Partial<Parameters<typeof buildSubmitOpRuntimeCompaction>[0]> = {},
) {
  return buildSubmitOpRuntimeCompaction({
    executionRuntime: null,
    syncedRecentPreferences: null,
    syncedSessionModelPreference: null,
    effectiveProviderType: "openai",
    effectiveModel: "gpt-5.4",
    ...options,
  });
}

describe("submitOpRuntimeCompaction", () => {
  it("应裁掉已经由 session/runtime 承接的 steady-state metadata", () => {
    const result = buildCompaction({
      requestMetadata: {
        harness: {
          turn_purpose: "content_review",
          preferences: {
            web_search: false,
            thinking: true,
            task: false,
            subagent: true,
          },
          theme: "general",
          session_mode: "general_workbench",
          gate_key: "write_mode",
          run_title: "社媒初稿",
          content_id: "content-social-1",
          selected_team_id: "team-social-1",
        },
      },
      executionRuntime: {
        session_id: "session-social-1",
        source: "runtime_snapshot",
        model_name: "gpt-5.4",
        recent_preferences: {
          webSearch: false,
          thinking: true,
          task: false,
          subagent: true,
        },
        recent_theme: "general",
        recent_session_mode: "general_workbench",
        recent_gate_key: "write_mode",
        recent_run_title: "社媒初稿",
        recent_content_id: "content-social-1",
      },
      syncedRecentPreferences: {
        task: false,
        subagent: true,
      },
      syncedSessionModelPreference: {
        providerType: "openai",
        model: "gpt-5.4",
      },
    });

    expect(result.metadata).toEqual({
      harness: {
        turn_purpose: "content_review",
      },
    });
    expect(result.shouldSubmitModel).toBe(false);
  });

  it("web/thinking 永久从 submit metadata 移除，且结果只暴露 typed model 标志", () => {
    const result = buildCompaction({
      requestMetadata: {
        harness: {
          preferences: {
            web_search: true,
            thinking: false,
            webSearchEnabled: true,
            thinkingEnabled: false,
            custom: "keep",
          },
        },
      },
    });

    expect(result.metadata).toEqual({
      harness: {
        preferences: {
          custom: "keep",
        },
      },
    });
    expect(result.shouldSubmitModel).toBe(true);
    expect(Object.keys(result).sort()).toEqual([
      "metadata",
      "shouldSubmitModel",
    ]);
  });

  it("task/subagent 与 runtime 已同步同值时应从 metadata 裁掉", () => {
    const result = buildCompaction({
      requestMetadata: {
        harness: {
          preferences: {
            task: true,
            subagent: false,
            custom: "keep",
          },
        },
      },
      executionRuntime: {
        session_id: "session-preferences",
        source: "runtime_snapshot",
        recent_preferences: {
          task: true,
          subagent: false,
        },
      },
      syncedRecentPreferences: {
        task: true,
        subagent: false,
      },
      syncedSessionModelPreference: {
        providerType: "openai",
        model: "gpt-5.4",
      },
    });

    expect(result.metadata).toEqual({
      harness: {
        preferences: {
          custom: "keep",
        },
      },
    });
  });

  it("session 模型尚未同步时应提交 typed model", () => {
    const result = buildCompaction({
      executionRuntime: {
        session_id: "session-model-pending",
        source: "runtime_snapshot",
        model_name: "gpt-5.4",
      },
      syncedSessionModelPreference: null,
    });

    expect(result.shouldSubmitModel).toBe(true);
  });

  it("typed model 与 session 当前值不同时应提交", () => {
    const result = buildCompaction({
      effectiveModel: "gpt-5.5",
      syncedSessionModelPreference: {
        providerType: "openai",
        model: "gpt-5.4",
      },
    });

    expect(result.shouldSubmitModel).toBe(true);
  });

  it("typed model 与 session 当前值相同时不应重复提交", () => {
    const result = buildCompaction({
      effectiveModel: "gpt-5.4",
      syncedSessionModelPreference: {
        providerType: "deepseek",
        model: "gpt-5.4",
      },
    });

    expect(result.shouldSubmitModel).toBe(false);
  });

  it("model 为空时不应提交半截 typed model", () => {
    const result = buildCompaction({
      effectiveModel: "",
      syncedSessionModelPreference: null,
    });

    expect(result.shouldSubmitModel).toBe(false);
  });

  it("image-only model 不应作为聊天 typed model 提交", () => {
    const result = buildCompaction({
      requestMetadata: {
        harness: {
          image_command_intent: {
            kind: "image_task",
            image_task: {
              prompt: "生成一张公众号封面",
              runtime_contract: {
                contract_key: "image_generation",
                routing_slot: "image_generation_model",
              },
            },
          },
        },
      },
      effectiveProviderType: "fal",
      effectiveModel: "fal-ai/nano-banana-pro",
      syncedSessionModelPreference: null,
    });

    expect(result.shouldSubmitModel).toBe(false);
  });

  it("retired image/team metadata 应清理，不得进入 submit", () => {
    const result = buildCompaction({
      requestMetadata: {
        harness: {
          trace_id: "trace-image-retired",
          image_skill_launch: {
            kind: "image_task",
            skill_name: "image_generate",
          },
          selected_team_disabled: false,
          preferred_team_preset_id: "social-preset",
          selected_team_id: "team-social-1",
          selected_team_source: "builtin",
          selected_team_label: "社媒执行团队",
          selected_team_description: "负责选题、写作和校对。",
          selected_team_summary: "负责选题、写作和校对。",
          selected_team_roles: [{ id: "writer" }],
        },
      },
    });

    expect(result.metadata).toEqual({
      harness: {
        trace_id: "trace-image-retired",
      },
    });
  });

  it("access/theme/session/gate/title/content 与 runtime 同值时应去重", () => {
    const result = buildCompaction({
      requestMetadata: {
        harness: {
          access_mode: "read-only",
          theme: "general",
          session_mode: "general_workbench",
          gate_key: "write_mode",
          run_title: "社媒初稿",
          content_id: "content-social-1",
          custom: "keep",
        },
      },
      executionRuntime: {
        session_id: "session-metadata",
        source: "runtime_snapshot",
        recent_access_mode: "read-only",
        recent_theme: "general",
        recent_session_mode: "theme_workbench",
        recent_gate_key: "write_mode",
        recent_run_title: "社媒初稿",
        recent_content_id: "content-social-1",
      },
    });

    expect(result.metadata).toEqual({
      harness: {
        custom: "keep",
      },
    });
  });

  it("历史专家 metadata 必须保留", () => {
    const result = buildCompaction({
      requestMetadata: {
        expert: {
          expertId: "code-literature",
          title: "代码文学专家",
          skillRefs: ["skill:capability-report"],
        },
        harness: {
          theme: "general",
          session_mode: "default",
          expert: {
            expert_id: "code-literature",
            title: "代码文学专家",
            skill_refs: ["skill:capability-report"],
          },
        },
      },
      executionRuntime: {
        session_id: "session-expert",
        source: "runtime_snapshot",
        recent_theme: "general",
        recent_session_mode: "default",
      },
    });

    expect(result.metadata).toEqual({
      expert: {
        expertId: "code-literature",
        title: "代码文学专家",
        skillRefs: ["skill:capability-report"],
      },
      harness: {
        expert: {
          expert_id: "code-literature",
          title: "代码文学专家",
          skill_refs: ["skill:capability-report"],
        },
      },
    });
  });

  it("已删除的 tool preference compaction 路径不得回流", () => {
    expect(
      existsSync(resolve(process.cwd(), RETIRED_TOOL_PREFERENCE_PATH)),
    ).toBe(false);
  });
});
