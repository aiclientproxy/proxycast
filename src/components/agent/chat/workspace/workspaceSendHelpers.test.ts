import { beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import type { AgentRuntimeWorkspaceSkillBinding } from "@/lib/api/agentRuntime/toolInventoryTypes";
import {
  buildGeneralWorkbenchSendBoundaryState,
  buildGeneralWorkbenchResumePromptFromRunState,
  buildInitialDispatchKey,
  buildSubmissionPreviewMessages,
  buildWorkspaceRequestMetadata,
  createSubmissionPreviewSnapshot,
  serviceSkillLaunchRequiresProject,
} from "./workspaceSendHelpers";

describe("workspaceSendHelpers", () => {
  beforeEach(async () => {
    await changeLimeLocale("zh-CN");
  });

  it("initialDispatchKey 应稳定编码首轮 prompt 与图片签名", () => {
    expect(
      buildInitialDispatchKey("写一篇文章", [
        { data: "abcdef1234567890", mediaType: "image/png" },
      ]),
    ).toContain("写一篇文章");
  });

  it("工作区首条创作意图不应再包装成旧内容写作 skill", () => {
    const boundary = buildGeneralWorkbenchSendBoundaryState({
      isThemeWorkbench: true,
      contentId: "content-1",
      initialDispatchKey: "dispatch-1",
      consumedInitialPromptKey: null,
      initialUserImages: [],
      mappedTheme: "general",
      sourceText: "请生成今天的社媒主稿",
    });

    expect(boundary).toMatchObject({
      sourceText: "请生成今天的社媒主稿",
      shouldConsumePendingGeneralWorkbenchInitialPrompt: true,
      shouldDismissGeneralWorkbenchEntryPrompt: true,
      browserRequirementMatch: null,
    });
  });

  it("普通 metadata 不应被判定为需要项目的服务技能启动", () => {
    expect(
      serviceSkillLaunchRequiresProject({
        harness: {
          theme: "general",
          session_mode: "default",
        },
      }),
    ).toBe(false);
  });

  it("缺少 project_id 的服务技能启动应要求先进入项目", () => {
    expect(
      serviceSkillLaunchRequiresProject({
        harness: {
          service_scene_launch: {
            service_scene_run: {
              skill_id: "x-article-export",
            },
          },
        },
      }),
    ).toBe(true);
  });

  it("浏览器任务应在 current send boundary 中保留 requirement 检测", () => {
    const boundary = buildGeneralWorkbenchSendBoundaryState({
      isThemeWorkbench: true,
      contentId: "content-1",
      initialDispatchKey: "dispatch-1",
      consumedInitialPromptKey: null,
      initialUserImages: [],
      mappedTheme: "general",
      sourceText: "帮我把这篇文章发布到微信公众号后台",
    });

    expect(boundary.sourceText).toBe("帮我把这篇文章发布到微信公众号后台");
    expect(boundary.browserRequirementMatch).toEqual(
      expect.objectContaining({
        requirement: "required_with_user_step",
        launchUrl: "https://mp.weixin.qq.com/",
        platformLabel: "微信公众号后台",
      }),
    );
  });

  it("浏览器 requirement 应进入工作区发送 metadata，而不是依赖组件挂载验证", () => {
    const metadata = buildWorkspaceRequestMetadata({
      effectiveToolPreferences: {
        task: false,
        subagent: false,
      },
      mappedTheme: "general",
      isThemeWorkbench: true,
      currentGateKey: "write_mode",
      contentId: "content-browser-required",
      browserRequirementMatch: {
        requirement: "required_with_user_step",
        reason: "需要在微信公众号后台完成发布流程。",
        launchUrl: "https://mp.weixin.qq.com/",
      },
      browserAssistProfileKey: "general_browser_assist",
    });

    expect(metadata).toMatchObject({
      harness: expect.objectContaining({
        theme: "general",
        session_mode: "general_workbench",
        content_id: "content-browser-required",
        browser_requirement: "required_with_user_step",
        browser_requirement_reason: "需要在微信公众号后台完成发布流程。",
        browser_launch_url: "https://mp.weixin.qq.com/",
        browser_user_step_required: true,
        browser_assist: expect.objectContaining({
          enabled: true,
          profile_key: "general_browser_assist",
        }),
      }),
    });
  });

  it("workspace skill runtime enable 应进入发送 metadata 且不打开 allow_model_skills", () => {
    const metadata = buildWorkspaceRequestMetadata({
      effectiveToolPreferences: {
        task: true,
        subagent: false,
      },
      mappedTheme: "general",
      isThemeWorkbench: false,
      currentGateKey: "default",
      workspaceSkillRuntimeEnable: {
        workspaceRoot: "/Users/demo/project",
        bindings: [
          {
            key: "workspace_skill:capability-report",
            name: "能力报告",
            description: "把能力输出整理成报告。",
            directory: "capability-report",
            registered_skill_directory:
              "/Users/demo/project/.agents/skills/capability-report",
            registration: {},
            permission_summary: ["Level 0 只读发现"],
            metadata: {},
            allowed_tools: ["read_file"],
            resource_summary: {},
            standard_compliance: {},
            runtime_binding_target: "workspace_skill",
            binding_status: "ready_for_manual_enable",
            binding_status_reason: "ready",
            next_gate: "manual_runtime_enable",
            query_loop_visible: false,
            tool_runtime_visible: false,
            launch_enabled: false,
            runtime_gate: "manual_runtime_enable",
          } satisfies AgentRuntimeWorkspaceSkillBinding,
        ],
      },
    });

    expect(metadata.harness).toMatchObject({
      workspace_skill_runtime_enable: {
        source: "manual_session_enable",
        approval: "manual",
        workspace_root: "/Users/demo/project",
        bindings: [
          expect.objectContaining({
            directory: "capability-report",
            skill: "project:capability-report",
          }),
        ],
      },
    });
    expect(metadata.allow_model_skills).toBeUndefined();
  });

  it("默认工作区 metadata 不应注入 Creator / Brand Voice", () => {
    const metadata = buildWorkspaceRequestMetadata({
      effectiveToolPreferences: {
        task: false,
        subagent: false,
      },
      mappedTheme: "general",
      isThemeWorkbench: true,
      currentGateKey: "write_mode",
      contentId: "content-no-voice",
    });

    expect(metadata.artifact).toBeUndefined();
    expect(metadata.harness).not.toHaveProperty("generation_brief");
    expect(metadata.harness).not.toHaveProperty("generationBrief");
  });

  it("plan mode 不得恢复 Renderer goal 状态 metadata", () => {
    const metadata = buildWorkspaceRequestMetadata({
      sendOptions: {
        collaborationMode: "plan",
        requestMetadata: {
          harness: {
            goal_mode_enabled: true,
            preferences: {
              goal: true,
              objective: true,
            },
            thread_goal: {
              enabled: true,
              source: "inputbar",
              status: "active",
              set: {
                threadId: "thread-workspace-plan-goal",
                objective: null,
                status: "active",
                tokenBudget: null,
              },
            },
          },
        },
        toolPreferencesOverride: {
          task: true,
          subagent: false,
        },
      },
      effectiveToolPreferences: {
        task: true,
        subagent: false,
      },
      mappedTheme: "general",
      isThemeWorkbench: true,
      currentGateKey: "write_mode",
      contentId: "content-goal-mode",
    });

    expect(metadata.harness).toMatchObject({
      preferences: {
        task: true,
        subagent: false,
      },
    });
    expect(metadata.harness).not.toHaveProperty("goal_mode_enabled");
    expect(metadata.harness).not.toHaveProperty("thread_goal");
    expect(metadata.harness.preferences).not.toHaveProperty("goal");
    expect(metadata.harness.preferences).not.toHaveProperty("objective");
  });

  it("应把显式 Generation Brief voice metadata 收敛到 artifact.generation_brief", () => {
    const metadata = buildWorkspaceRequestMetadata({
      workspaceRequestMetadataBase: {
        trace_id: "trace-voice-1",
      },
      sendOptions: {
        requestMetadata: {
          artifact: {
            generationBrief: {
              voiceSource: "brand_voice",
              voiceGuard: "user_explicit",
              brandVoiceId: "brand-voice-1",
              evidencePackId: "voice-pack-1",
              inheritsGlobalSoul: false,
              inheritsExpertPersona: false,
            },
          },
        },
      },
      effectiveToolPreferences: {
        task: false,
        subagent: false,
      },
      mappedTheme: "general",
      isThemeWorkbench: true,
      currentGateKey: "write_mode",
      contentId: "content-brand-voice",
    });

    expect(metadata).toMatchObject({
      trace_id: "trace-voice-1",
      artifact: {
        generation_brief: {
          voice_source: "brand_voice",
          voice_guard: "user_explicit",
          brand_voice_id: "brand-voice-1",
          evidence_pack_id: "voice-pack-1",
          inherits_global_soul: false,
          inherits_expert_persona: false,
        },
      },
      harness: expect.objectContaining({
        theme: "general",
        session_mode: "general_workbench",
        content_id: "content-brand-voice",
      }),
    });
    expect(metadata.artifact as Record<string, unknown>).not.toHaveProperty(
      "generationBrief",
    );
    expect(metadata.harness).not.toHaveProperty("generation_brief");
  });

  it("只带 generation_brief 时不应在前端补 Artifact Stage / Schema 合同", () => {
    const metadata = buildWorkspaceRequestMetadata({
      sendOptions: {
        requestMetadata: {
          generationBrief: {
            voiceSource: "creator_voice",
            creatorVoiceId: "creator-voice-1",
            evidenceRefs: ["memory:voice-1"],
          },
        },
      },
      effectiveToolPreferences: {
        task: false,
        subagent: false,
      },
      mappedTheme: "general",
      isThemeWorkbench: false,
      currentGateKey: "write_mode",
    });

    expect(metadata.artifact).toEqual({
      generation_brief: {
        voice_source: "creator_voice",
        creator_voice_id: "creator-voice-1",
        evidence_refs: ["memory:voice-1"],
      },
    });
    expect(metadata).not.toHaveProperty("generationBrief");
    expect(metadata.artifact as Record<string, unknown>).not.toHaveProperty(
      "artifact_mode",
    );
    expect(metadata.artifact as Record<string, unknown>).not.toHaveProperty(
      "artifact_stage",
    );
    expect(metadata.artifact as Record<string, unknown>).not.toHaveProperty(
      "artifact_kind",
    );
  });

  it("workspace base 与 sendOptions 的 artifact metadata 应深合并", () => {
    const metadata = buildWorkspaceRequestMetadata({
      workspaceRequestMetadataBase: {
        artifact: {
          artifact_mode: "draft",
          artifact_kind: "analysis",
        },
      },
      sendOptions: {
        requestMetadata: {
          artifact: {
            workbench_surface: "right_panel",
            generationBrief: {
              voiceSource: "brand_voice",
              evidencePackId: "voice-pack-1",
            },
          },
        },
      },
      effectiveToolPreferences: {
        task: false,
        subagent: false,
      },
      mappedTheme: "general",
      isThemeWorkbench: true,
      currentGateKey: "write_mode",
      contentId: "content-merged-artifact",
    });

    expect(metadata.artifact).toEqual({
      artifact_mode: "draft",
      artifact_kind: "analysis",
      workbench_surface: "right_panel",
      generation_brief: {
        voice_source: "brand_voice",
        evidence_pack_id: "voice-pack-1",
      },
    });
  });

  it("历史专家 session metadata 应保留到工作区发送 request metadata", () => {
    const metadata = buildWorkspaceRequestMetadata({
      workspaceRequestMetadataBase: {
        expert: {
          expertId: "code-literature",
          title: "代码文学专家",
          skillRefs: ["skill:capability-report"],
        },
        harness: {
          source: "history-session",
          expert: {
            expert_id: "code-literature",
            title: "代码文学专家",
            skill_refs: ["skill:capability-report"],
          },
        },
      },
      effectiveToolPreferences: {
        task: false,
        subagent: false,
      },
      mappedTheme: "general",
      isThemeWorkbench: false,
      currentGateKey: "default",
    });

    expect(metadata).toMatchObject({
      expert: {
        expertId: "code-literature",
        title: "代码文学专家",
        skillRefs: ["skill:capability-report"],
      },
      harness: expect.objectContaining({
        expert: {
          expert_id: "code-literature",
          title: "代码文学专家",
          skill_refs: ["skill:capability-report"],
        },
      }),
    });
  });

  it("保存的 Soul 创作声线无本轮显式声线时应作为发送 fallback", () => {
    const metadata = buildWorkspaceRequestMetadata({
      savedSoulArtifactVoiceGenerationBrief: {
        voice_source: "brand_voice",
        voice_guard: "user_explicit",
        brand_voice_id: "saved-brand-voice",
        evidence_source: "memory.soul.artifact_voice",
      },
      soulArtifactVoiceEnabledForTurn: true,
      effectiveToolPreferences: {
        task: false,
        subagent: false,
      },
      mappedTheme: "general",
      isThemeWorkbench: true,
      currentGateKey: "write_mode",
      contentId: "content-saved-voice",
    });

    expect(metadata).toMatchObject({
      artifact: {
        generation_brief: {
          voice_source: "brand_voice",
          voice_guard: "user_explicit",
          brand_voice_id: "saved-brand-voice",
          evidence_source: "memory.soul.artifact_voice",
        },
      },
      diagnostics: {
        soul_artifact_voice: {
          status: "saved_applied",
          enabled_for_turn: true,
          source: "memory.soul.artifact_voice",
          guard_result: "applied",
          voice_source: "brand_voice",
          voice_guard: "user_explicit",
          evidence_source: "memory.soul.artifact_voice",
        },
      },
    });
  });

  it("本轮关闭保存的 Soul 创作声线时不应注入 artifact metadata", () => {
    const metadata = buildWorkspaceRequestMetadata({
      savedSoulArtifactVoiceGenerationBrief: {
        voice_source: "brand_voice",
        voice_guard: "user_explicit",
        brand_voice_id: "saved-brand-voice",
        evidence_source: "memory.soul.artifact_voice",
      },
      soulArtifactVoiceEnabledForTurn: false,
      effectiveToolPreferences: {
        task: false,
        subagent: false,
      },
      mappedTheme: "general",
      isThemeWorkbench: true,
      currentGateKey: "write_mode",
      contentId: "content-saved-voice-disabled",
    });

    expect(metadata.artifact).toBeUndefined();
    expect(metadata).toMatchObject({
      diagnostics: {
        soul_artifact_voice: {
          status: "disabled_for_turn",
          enabled_for_turn: false,
          source: "memory.soul.artifact_voice",
          guard_result: "blocked_by_turn_override",
          voice_source: "brand_voice",
          voice_guard: "user_explicit",
          evidence_source: "memory.soul.artifact_voice",
        },
      },
    });
  });

  it("本轮显式声线应覆盖保存的 Soul 创作声线且不继承保存 evidence", () => {
    const metadata = buildWorkspaceRequestMetadata({
      savedSoulArtifactVoiceGenerationBrief: {
        voice_source: "brand_voice",
        voice_guard: "user_explicit",
        brand_voice_id: "saved-brand-voice",
        evidence_source: "memory.soul.artifact_voice",
      },
      soulArtifactVoiceEnabledForTurn: true,
      sendOptions: {
        requestMetadata: {
          generationBrief: {
            voiceSource: "creator_voice",
            creatorVoiceId: "turn-creator-voice",
            evidenceRefs: ["memory:turn-voice"],
          },
        },
      },
      effectiveToolPreferences: {
        task: false,
        subagent: false,
      },
      mappedTheme: "general",
      isThemeWorkbench: true,
      currentGateKey: "write_mode",
      contentId: "content-turn-voice",
    });

    expect(metadata).toMatchObject({
      artifact: {
        generation_brief: {
          voice_source: "creator_voice",
          creator_voice_id: "turn-creator-voice",
          evidence_refs: ["memory:turn-voice"],
        },
      },
      diagnostics: {
        soul_artifact_voice: {
          status: "turn_explicit",
          enabled_for_turn: true,
          source: "request_metadata.generation_brief",
          guard_result: "applied",
          voice_source: "creator_voice",
          evidence_refs: ["memory:turn-voice"],
          evidence_ref_count: 1,
        },
      },
    });
    expect(metadata.artifact).not.toMatchObject({
      generation_brief: {
        brand_voice_id: "saved-brand-voice",
        evidence_source: "memory.soul.artifact_voice",
      },
    });
  });

  it("自动首发 metadata 中已有浏览器协助参数时应继续保留", () => {
    const metadata = buildWorkspaceRequestMetadata({
      sendOptions: {
        requestMetadata: {
          harness: {
            browser_assist: {
              enabled: true,
              profile_key: "general_browser_assist",
              preferred_backend: "lime_extension_bridge",
              auto_launch: false,
            },
          },
        },
      },
      effectiveToolPreferences: {
        task: false,
        subagent: false,
      },
      mappedTheme: "general",
      isThemeWorkbench: true,
      currentGateKey: "write_mode",
      contentId: "content-browser-required-bootstrap",
      browserRequirementMatch: {
        requirement: "required_with_user_step",
        reason: "需要在微信公众号后台完成发布流程。",
        launchUrl: "https://mp.weixin.qq.com/",
      },
      browserAssistProfileKey: "general_browser_assist",
    });

    expect(metadata).toMatchObject({
      harness: expect.objectContaining({
        content_id: "content-browser-required-bootstrap",
        browser_requirement: "required_with_user_step",
        browser_user_step_required: true,
        browser_assist: expect.objectContaining({
          enabled: true,
          profile_key: "general_browser_assist",
          preferred_backend: "lime_extension_bridge",
          auto_launch: false,
        }),
      }),
    });
  });

  it("run-state 应生成 resume prompt", () => {
    const prompt = buildGeneralWorkbenchResumePromptFromRunState({
      run_state: "auto_running",
      current_gate_key: "write_mode",
      queue_items: [
        {
          run_id: "run-1",
          title: "撰写主稿",
          gate_key: "write_mode",
          status: "running",
          source: "skill",
          source_ref: null,
          started_at: new Date().toISOString(),
        },
      ],
      latest_terminal: null,
      recent_terminals: [],
      updated_at: new Date().toISOString(),
    });

    expect(prompt).toMatchObject({
      kind: "resume",
      title: "发现上次未完成任务",
      actionLabel: "继续上次生成",
      description: expect.stringContaining("撰写主稿"),
    });
  });

  it("提交预览应回显用户消息并保留 assistant 等待态", () => {
    vi.spyOn(Date, "now").mockReturnValue(1_710_000_000_000);

    const snapshot = createSubmissionPreviewSnapshot({
      key: "submission-preview-1",
      prompt: "/brand-product-knowledge-builder 继续处理当前任务",
      displayContent: "继续处理当前任务",
      inputCapabilityRoute: {
        kind: "installed_skill",
        skillKey: "brand-product-knowledge-builder",
        skillName: "产品知识库",
      },
      images: [],
    });

    expect(snapshot).toMatchObject({
      key: "submission-preview-1",
      prompt: "/brand-product-knowledge-builder 继续处理当前任务",
      displayContent: "继续处理当前任务",
      createdAt: 1_710_000_000_000,
    });

    const messages = buildSubmissionPreviewMessages(snapshot);
    expect(messages).toHaveLength(2);
    expect(messages[0]).toMatchObject({
      id: "submission-preview:submission-preview-1:user",
      role: "user",
      content: "继续处理当前任务",
      inputCapabilityRoute: {
        kind: "installed_skill",
        skillKey: "brand-product-knowledge-builder",
        skillName: "产品知识库",
      },
    });
    expect(messages[1]).toMatchObject({
      id: "submission-preview:submission-preview-1:assistant",
      role: "assistant",
      content: "",
      isThinking: true,
      runtimeStatus: {
        phase: "preparing",
      },
    });

    vi.restoreAllMocks();
  });
});
