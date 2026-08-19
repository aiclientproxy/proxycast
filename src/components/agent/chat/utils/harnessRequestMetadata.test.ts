import { describe, expect, it } from "vitest";
import type { AgentRuntimeWorkspaceSkillBinding } from "@/lib/api/agentRuntime/toolInventoryTypes";

import {
  buildHarnessRequestMetadata,
  extractExistingHarnessMetadata,
} from "./harnessRequestMetadata";
import { normalizeHarnessSessionMode } from "./harnessSessionMode";

describe("harnessRequestMetadata", () => {
  it("应保留已有 harness metadata 并覆盖当前字段", () => {
    const metadata = buildHarnessRequestMetadata({
      base: {
        trace_id: "trace-1",
        legacy_flag: true,
        browser_assist: { enabled: true },
        browserAssist: { enabled: true },
      },
      theme: "general",
      preferences: {
        task: true,
        subagent: false,
      },
      sessionMode: "default",
    });

    expect(metadata).toMatchObject({
      trace_id: "trace-1",
      legacy_flag: true,
      theme: "general",
      preferences: {
        task: true,
        subagent: false,
      },
    });
    expect(metadata.creation_mode).toBeUndefined();
    expect(metadata.chat_mode).toBeUndefined();
    expect(metadata.web_search_enabled).toBeUndefined();
    expect(metadata.task_mode_enabled).toBeUndefined();
    expect(metadata.browser_assist).toBeUndefined();
    expect(metadata.browserAssist).toBeUndefined();
  });

  it("应清理 base 中遗留的平铺状态字段", () => {
    const metadata = buildHarnessRequestMetadata({
      base: {
        creation_mode: "guided",
        chat_mode: "general",
        web_search_enabled: false,
        thinking_enabled: false,
        task_mode_enabled: false,
        subagent_mode_enabled: false,
        creationMode: "hybrid",
        chatMode: "workbench",
        webSearchEnabled: true,
        thinkingEnabled: true,
        taskModeEnabled: true,
        subagentModeEnabled: true,
        turn_team_decision: "team_prepared",
        turn_team_reason: "runtime_team_prepared",
        turn_team_blueprint: {
          label: "旧 Team 蓝图",
        },
        turnTeamDecision: "single_agent",
        turnTeamReason: "single_agent_direct",
        turnTeamBlueprint: {
          label: "legacy team blueprint",
        },
      },
      theme: "general",
      preferences: {
        task: false,
        subagent: false,
      },
      sessionMode: "default",
    });

    expect(metadata).toMatchObject({
      theme: "general",
      preferences: {
        task: false,
        subagent: false,
      },
    });
    expect(metadata.creation_mode).toBeUndefined();
    expect(metadata.chat_mode).toBeUndefined();
    expect(metadata.web_search_enabled).toBeUndefined();
    expect(metadata.thinking_enabled).toBeUndefined();
    expect(metadata.task_mode_enabled).toBeUndefined();
    expect(metadata.subagent_mode_enabled).toBeUndefined();
    expect(metadata.creationMode).toBeUndefined();
    expect(metadata.chatMode).toBeUndefined();
    expect(metadata.webSearchEnabled).toBeUndefined();
    expect(metadata.thinkingEnabled).toBeUndefined();
    expect(metadata.taskModeEnabled).toBeUndefined();
    expect(metadata.subagentModeEnabled).toBeUndefined();
    expect(metadata.turn_team_decision).toBeUndefined();
    expect(metadata.turn_team_reason).toBeUndefined();
    expect(metadata.turn_team_blueprint).toBeUndefined();
    expect(metadata.turnTeamDecision).toBeUndefined();
    expect(metadata.turnTeamReason).toBeUndefined();
    expect(metadata.turnTeamBlueprint).toBeUndefined();
  });

  it("应保留已有 plan preferences 别名并跟随当前 task 偏好", () => {
    const metadata = buildHarnessRequestMetadata({
      base: {
        preferences: {
          task: true,
          task_mode: true,
          subagent: false,
        },
      },
      theme: "general",
      preferences: {
        task: true,
        subagent: false,
      },
      sessionMode: "default",
    });

    expect(metadata.preferences).toMatchObject({
      task: true,
      task_mode: true,
      subagent: false,
    });
    expect(metadata.task_mode_enabled).toBe(true);

    const disabledMetadata = buildHarnessRequestMetadata({
      base: {
        preferences: {
          task: true,
          task_mode: true,
          subagent: false,
        },
      },
      theme: "general",
      preferences: {
        task: false,
        subagent: false,
      },
      sessionMode: "default",
    });

    expect(disabledMetadata.preferences).toMatchObject({
      task: false,
      task_mode: false,
      subagent: false,
    });
    expect(disabledMetadata.task_mode_enabled).toBeUndefined();
  });

  it("默认会话模式不应写入 gate_key", () => {
    const metadata = buildHarnessRequestMetadata({
      theme: "general",
      preferences: {
        task: false,
        subagent: false,
      },
      sessionMode: "default",
      gateKey: "topic_select",
    });

    expect(metadata.gate_key).toBeUndefined();
  });

  it("应透传当前发送用途，供后端统一决策运行时行为", () => {
    const metadata = buildHarnessRequestMetadata({
      theme: "general",
      turnPurpose: "content_review",
      preferences: {
        task: false,
        subagent: false,
      },
      sessionMode: "general_workbench",
    });

    expect(metadata.turn_purpose).toBe("content_review");
  });

  it("应将 legacy general workbench alias 会话模式归一为 general_workbench", () => {
    expect(normalizeHarnessSessionMode("theme_workbench")).toBe(
      "general_workbench",
    );

    const metadata = buildHarnessRequestMetadata({
      theme: "general",
      preferences: {
        task: false,
        subagent: false,
      },
      sessionMode: "theme_workbench",
      gateKey: "write_mode",
    });

    expect(metadata.session_mode).toBe("general_workbench");
    expect(metadata.gate_key).toBe("write_mode");
  });

  it("应以独立字段透传 Agent response language 且不复用 UI locale", () => {
    const metadata = buildHarnessRequestMetadata({
      base: {
        response_language: "ja-JP",
        agentResponseLanguage: "ko-KR",
      },
      theme: "general",
      preferences: {
        task: false,
        subagent: false,
      },
      sessionMode: "default",
      agentResponseLanguage: "en-US",
    });

    expect(metadata.agent_response_language).toBe("en-US");
    expect(metadata.language).toBeUndefined();
  });

  it("应兼容读取已有 response language alias 但 current 写入 snake_case", () => {
    const metadata = buildHarnessRequestMetadata({
      base: {
        response_language: " ja-JP ",
      },
      theme: "general",
      preferences: {
        task: false,
        subagent: false,
      },
      sessionMode: "default",
    });

    expect(metadata.agent_response_language).toBe("ja-JP");
  });

  it("应按 P3D contract 透传 workspace skill binding 候选且不打开模型技能", () => {
    const metadata = buildHarnessRequestMetadata({
      theme: "general",
      preferences: {
        task: true,
        subagent: false,
      },
      sessionMode: "default",
      workspaceSkillBindings: [
        {
          directory: "capability-report",
          name: "只读 CLI 报告",
          description: "把只读 CLI 输出整理成 Markdown 报告。",
          binding_status: "ready_for_manual_enable",
          next_gate: "manual_runtime_enable",
          query_loop_visible: false,
          tool_runtime_visible: false,
          launch_enabled: false,
          permission_summary: ["Level 0 只读发现"],
          registration: {
            source_draft_id: "capdraft-1",
            source_verification_report_id: "capver-1",
          },
        } as AgentRuntimeWorkspaceSkillBinding,
      ],
    });

    expect(metadata.workspace_skill_bindings).toEqual({
      source: "p3c_runtime_binding",
      bindings: [
        expect.objectContaining({
          directory: "capability-report",
          binding_status: "ready_for_manual_enable",
          launch_enabled: false,
          source_draft_id: "capdraft-1",
        }),
      ],
    });
    expect(metadata.allow_model_skills).toBeUndefined();
  });

  it("应按 P3E contract 透传显式 workspace skill runtime enable 且不写 allow_model_skills", () => {
    const metadata = buildHarnessRequestMetadata({
      theme: "general",
      preferences: {
        task: true,
        subagent: false,
      },
      sessionMode: "default",
      workspaceSkillRuntimeEnable: {
        workspaceRoot: "/tmp/work",
        bindings: [
          {
            directory: "capability-report",
            name: "只读 CLI 报告",
            description: "把只读 CLI 输出整理成 Markdown 报告。",
            registered_skill_directory:
              "/tmp/work/.agents/skills/capability-report",
            binding_status: "ready_for_manual_enable",
            next_gate: "manual_runtime_enable",
            query_loop_visible: false,
            tool_runtime_visible: false,
            launch_enabled: false,
            permission_summary: ["Level 0 只读发现"],
            registration: {
              source_draft_id: "capdraft-1",
              source_verification_report_id: "capver-1",
            },
          } as AgentRuntimeWorkspaceSkillBinding,
        ],
      },
    });

    expect(metadata.workspace_skill_runtime_enable).toEqual({
      source: "manual_session_enable",
      approval: "manual",
      workspace_root: "/tmp/work",
      bindings: [
        expect.objectContaining({
          directory: "capability-report",
          skill: "project:capability-report",
          source_draft_id: "capdraft-1",
          source_verification_report_id: "capver-1",
        }),
      ],
    });
    expect(metadata.allow_model_skills).toBeUndefined();
  });

  it("不应再写入旧 turn_team compat 字段", () => {
    const metadata = buildHarnessRequestMetadata({
      theme: "general",
      preferences: {
        task: true,
        subagent: true,
      },
      sessionMode: "default",
    });

    expect(metadata.turn_team_decision).toBeUndefined();
    expect(metadata.turn_team_reason).toBeUndefined();
    expect(metadata.turn_team_blueprint).toBeUndefined();
  });

  it("需要人工确认的浏览器任务应标记 user step", () => {
    const metadata = buildHarnessRequestMetadata({
      theme: "general",
      preferences: {
        task: true,
        subagent: true,
      },
      sessionMode: "general_workbench",
      gateKey: "publish_confirm",
      browserRequirement: "required_with_user_step",
      browserRequirementReason: "需要登录站点后继续",
      browserLaunchUrl: "https://example.com/publish",
    });

    expect(metadata).toMatchObject({
      gate_key: "publish_confirm",
      browser_requirement: "required_with_user_step",
      browser_requirement_reason: "需要登录站点后继续",
      browser_launch_url: "https://example.com/publish",
      browser_user_step_required: true,
    });
    expect(metadata.browser_assist).toBeUndefined();
  });

  it("应清理旧 Browser Assist 运行合同而只保留 current requirement", () => {
    const metadata = buildHarnessRequestMetadata({
      base: {
        browser_assist: {
          enabled: true,
          profile_key: "attached-xhs",
          preferred_backend: "lime_extension_bridge",
          auto_launch: false,
          runtime_contract: { contract_key: "browser_control" },
        },
        browserAssist: {
          enabled: true,
          sessionId: "legacy-session",
        },
      },
      theme: "general",
      preferences: {
        task: true,
        subagent: true,
      },
      sessionMode: "default",
      browserRequirement: "required",
      browserLaunchUrl: "https://example.com",
    });

    expect(metadata).toMatchObject({
      browser_requirement: "required",
      browser_launch_url: "https://example.com",
    });
    expect(metadata.browser_assist).toBeUndefined();
    expect(metadata.browserAssist).toBeUndefined();
  });

  it("应能从 request metadata 中提取已有 harness metadata", () => {
    expect(
      extractExistingHarnessMetadata({
        harness: {
          trace_id: "trace-2",
          theme: "general",
        },
      }),
    ).toEqual({
      trace_id: "trace-2",
      theme: "general",
    });

    expect(
      extractExistingHarnessMetadata({ harness: "invalid" }),
    ).toBeUndefined();
  });
});
