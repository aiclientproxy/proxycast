import { describe, expect, it } from "vitest";
import type { ScheduledTask } from "@/lib/api/scheduledTasks";
import type { AgentRuntimeWorkspaceSkillBinding } from "@/lib/api/agentRuntime/toolInventoryTypes";
import {
  buildWorkspaceSkillManagedAutomationPresentation,
  buildWorkspaceSkillAgentAutomationInitialValues,
  buildWorkspaceSkillAgentAutomationRequestMetadata,
  canBuildWorkspaceSkillAgentAutomationDraft,
  isWorkspaceSkillScheduledTaskForDirectory,
} from "./workspaceSkillAgentAutomationDraft";

function createBinding(
  overrides: Partial<AgentRuntimeWorkspaceSkillBinding> = {},
): AgentRuntimeWorkspaceSkillBinding {
  return {
    key: "workspace_skill:capability-report",
    name: "只读 CLI 报告",
    description: "把本地只读 CLI 输出整理成 Markdown 报告。",
    directory: "capability-report",
    registered_skill_directory: "/tmp/work/.agents/skills/capability-report",
    registration: {
      sourceDraftId: "capdraft-1",
      sourceVerificationReportId: "capver-1",
      registeredSkillDirectory: "/tmp/work/.agents/skills/capability-report",
    },
    permission_summary: ["Level 0 只读发现"],
    metadata: {},
    allowed_tools: [],
    resource_summary: {
      hasScripts: true,
    },
    standard_compliance: {
      isStandard: true,
    },
    runtime_binding_target: "workspace_skill",
    binding_status: "ready_for_manual_enable",
    binding_status_reason: "ready",
    next_gate: "manual_runtime_enable",
    query_loop_visible: false,
    tool_runtime_visible: false,
    launch_enabled: false,
    runtime_gate: "manual_runtime_enable",
    ...overrides,
  };
}

describe("workspaceSkillAgentAutomationDraft", () => {
  it("应为 ready binding 构建 automation job 初始值，并把执行绑定到 P3E runtime enable", () => {
    const initialValues = buildWorkspaceSkillAgentAutomationInitialValues({
      binding: createBinding(),
      workspaceRoot: "/tmp/work",
      workspaceId: "project-1",
    });

    expect(initialValues).toMatchObject({
      form: {
        title: "只读 CLI 报告｜Managed Agent 草案",
        projectId: "project-1",
        enabled: false,
        scheduleType: "daily",
        time: "09:00",
      },
    });
    expect(initialValues?.form.prompt).toContain("project:capability-report");
    expect(initialValues?.requestMetadata).toMatchObject({
      harness: {
        agent_envelope: {
          source: "skill_forge_p4_agent_envelope",
          state: "automation_draft",
          skill: "project:capability-report",
          source_draft_id: "capdraft-1",
          source_verification_report_id: "capver-1",
          authorization_scope: "scheduled_run_session",
        },
        workspace_skill_runtime_enable: {
          source: "manual_session_enable",
          approval: "manual",
          workspace_root: "/tmp/work",
          bindings: [
            {
              directory: "capability-report",
              skill: "project:capability-report",
              source_draft_id: "capdraft-1",
              source_verification_report_id: "capver-1",
            },
          ],
        },
      },
    });
    expect(JSON.stringify(initialValues?.requestMetadata)).not.toContain(
      "managed_objective",
    );
  });

  it("blocked 或缺少 verification provenance 时不能构建 managed job 草案", () => {
    expect(
      canBuildWorkspaceSkillAgentAutomationDraft(
        createBinding({ binding_status: "blocked" }),
      ),
    ).toBe(false);
    expect(
      buildWorkspaceSkillAgentAutomationRequestMetadata({
        binding: createBinding({
          registration: {
            sourceDraftId: "capdraft-1",
            sourceVerificationReportId: null,
            registeredSkillDirectory:
              "/tmp/work/.agents/skills/capability-report",
          },
        }),
        workspaceRoot: "/tmp/work",
      }),
    ).toBeNull();
  });

  it("应支持注入 Managed Job 初始值与 prompt 文案 copy", () => {
    const initialValues = buildWorkspaceSkillAgentAutomationInitialValues({
      binding: createBinding(),
      workspaceRoot: "/tmp/work",
      workspaceId: "project-1",
      copy: {
        descriptionPausedByDefault: "Review before enabling.",
        descriptionSource: "Source: envelope draft.",
        formatDescriptionProvenance: (draftId, reportId) =>
          `Provenance: ${draftId}/${reportId}`,
        formatDescriptionSkill: (skillName) => `Skill: ${skillName}`,
        formatName: (displayName) => `Managed draft for ${displayName}`,
        formatPromptIntro: (displayName, skillName) =>
          `Run ${displayName} with ${skillName}.`,
        promptNeedsInput: "Return needs_input when required data is missing.",
        promptReadRunbook: "Read the runbook before running.",
        promptResultEvidence: "Return summary and evidence.",
      },
    });

    expect(initialValues?.form.title).toBe("Managed draft for 只读 CLI 报告");
    expect(initialValues?.form.prompt).toContain(
      "Run 只读 CLI 报告 with project:capability-report.",
    );
    expect(JSON.stringify(initialValues?.requestMetadata)).not.toContain(
      "managed_objective",
    );
    expect(JSON.stringify(initialValues?.requestMetadata)).not.toContain(
      "agent_runtime_submit_turn",
    );
  });

  it("应识别 workspace skill 对应的 Scheduled Task 并生成状态摘要", () => {
    const task: ScheduledTask = {
      id: "job-1",
      title: "只读 CLI 报告｜Managed Agent 草案",
      prompt: "run",
      enabled: false,
      schedule: {
        type: "daily",
        time: "09:00",
        timezone: "Asia/Shanghai",
      },
      execution: {
        threadMode: "new_thread",
        projectId: "project-1",
        requestMetadata: {
          harness: {
            agent_envelope: {
              directory: "capability-report",
              skill: "project:capability-report",
            },
          },
        },
      },
      notificationPolicy: "failures",
      overlapPolicy: "skip_if_running",
      createdAt: "2026-05-06T10:00:00Z",
      updatedAt: "2026-05-06T10:00:00Z",
    };

    expect(
      isWorkspaceSkillScheduledTaskForDirectory(task, "capability-report"),
    ).toBe(true);
    expect(isWorkspaceSkillScheduledTaskForDirectory(task, "other-skill")).toBe(
      false,
    );

    const presentation = buildWorkspaceSkillManagedAutomationPresentation([
      task,
    ]);
    expect(presentation.statusLabel).toContain("草案暂停");
    expect(presentation.scheduleLabel).toContain("每天 09:00");
    expect(presentation.lastRunLabel).toContain("暂无");
  });
});
