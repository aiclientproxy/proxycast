import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { capabilityDraftsApi } from "@/lib/api/capabilityDrafts";
import { listWorkspaceSkillBindings } from "@/lib/api/agentRuntime/inventoryClient";
import { scheduledTasksApi } from "@/lib/api/scheduledTasks";
import { clearAgentUiProjectionEvents } from "@/components/agent/chat/projection/conversationProjectionStore";
import { WorkspaceRegisteredSkillsPanel } from "./WorkspaceRegisteredSkillsPanel";

const { mockUseTranslation } = vi.hoisted(() => {
  const mockTranslate = vi.fn((key: string, options?: unknown) => {
    if (typeof options === "string") {
      return options;
    }
    if (options && typeof options === "object") {
      const values = options as Record<string, unknown>;
      const template =
        typeof values.defaultValue === "string" ? values.defaultValue : key;
      return template.replace(/\{\{(\w+)\}\}/g, (_match, name: string) =>
        String(values[name] ?? ""),
      );
    }
    return key;
  });

  return {
    mockUseTranslation: vi.fn((_namespace?: string) => ({
      i18n: { language: "zh-CN" },
      t: mockTranslate,
    })),
  };
});

vi.mock("react-i18next", () => ({
  useTranslation: mockUseTranslation,
}));

vi.mock("@/lib/api/capabilityDrafts", () => ({
  capabilityDraftsApi: {
    listRegisteredSkills: vi.fn(),
  },
}));

vi.mock("@/lib/api/agentRuntime/inventoryClient", () => ({
  listWorkspaceSkillBindings: vi.fn(),
}));

vi.mock("@/lib/api/scheduledTasks", () => ({
  scheduledTasksApi: {
    listDetailed: vi.fn(),
    setEnabled: vi.fn(),
  },
}));

interface RenderResult {
  container: HTMLDivElement;
  root: Root;
}

const mountedRoots: RenderResult[] = [];

function renderPanel(
  props?: Parameters<typeof WorkspaceRegisteredSkillsPanel>[0],
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(<WorkspaceRegisteredSkillsPanel {...props} />);
  });
  mountedRoots.push({ container, root });
  return { container, root };
}

describe("WorkspaceRegisteredSkillsPanel", () => {
  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    vi.mocked(capabilityDraftsApi.listRegisteredSkills).mockReset();
    vi.mocked(listWorkspaceSkillBindings).mockReset();
    vi.mocked(scheduledTasksApi.listDetailed).mockReset();
    vi.mocked(scheduledTasksApi.listDetailed).mockResolvedValue([]);
    vi.mocked(scheduledTasksApi.setEnabled).mockReset();
    clearAgentUiProjectionEvents();
    vi.mocked(listWorkspaceSkillBindings).mockResolvedValue({
      request: {
        workspace_root: "/tmp/work",
        caller: "assistant",
        surface: {
          workbench: true,
          browser_assist: false,
        },
      },
      warnings: [],
      counts: {
        registered_total: 0,
        ready_for_manual_enable_total: 0,
        blocked_total: 0,
        query_loop_visible_total: 0,
        tool_runtime_visible_total: 0,
        launch_enabled_total: 0,
      },
      bindings: [],
    });
  });

  afterEach(() => {
    while (mountedRoots.length > 0) {
      const mounted = mountedRoots.pop();
      if (!mounted) {
        break;
      }
      act(() => {
        mounted.root.unmount();
      });
      mounted.container.remove();
    }
    vi.clearAllMocks();
    clearAgentUiProjectionEvents();
  });

  it("completion audit completed 时 Agent envelope 入口复用 Managed Job 草案创建链", async () => {
    const onCreateManagedAutomationDraft = vi.fn();
    vi.mocked(capabilityDraftsApi.listRegisteredSkills).mockResolvedValueOnce([
      {
        key: "workspace:capability-report",
        name: "只读 CLI 报告",
        description: "把本地只读 CLI 输出整理成 Markdown 报告。",
        directory: "capability-report",
        registeredSkillDirectory: "/tmp/work/.agents/skills/capability-report",
        registration: {
          registrationId: "capreg-1",
          registeredAt: "2026-05-05T01:10:00.000Z",
          skillDirectory: "capability-report",
          registeredSkillDirectory:
            "/tmp/work/.agents/skills/capability-report",
          sourceDraftId: "capdraft-1",
          sourceVerificationReportId: "capver-1",
          generatedFileCount: 4,
          permissionSummary: ["Level 0 只读发现"],
        },
        permissionSummary: ["Level 0 只读发现"],
        metadata: {},
        allowedTools: [],
        resourceSummary: {
          hasScripts: true,
          hasReferences: false,
          hasAssets: false,
        },
        standardCompliance: {
          isStandard: true,
          validationErrors: [],
          deprecatedFields: [],
        },
        launchEnabled: false,
        runtimeGate: "等待 runtime gate。",
      },
    ]);
    vi.mocked(listWorkspaceSkillBindings).mockResolvedValueOnce({
      request: {
        workspace_root: "/tmp/work",
        caller: "assistant",
        surface: {
          workbench: true,
          browser_assist: false,
        },
      },
      warnings: [],
      counts: {
        registered_total: 1,
        ready_for_manual_enable_total: 1,
        blocked_total: 0,
        query_loop_visible_total: 0,
        tool_runtime_visible_total: 0,
        launch_enabled_total: 0,
      },
      bindings: [
        {
          key: "workspace_skill:capability-report",
          name: "只读 CLI 报告",
          description: "把本地只读 CLI 输出整理成 Markdown 报告。",
          directory: "capability-report",
          registered_skill_directory:
            "/tmp/work/.agents/skills/capability-report",
          registration: {
            source_draft_id: "capdraft-1",
            source_verification_report_id: "capver-1",
            registered_skill_directory:
              "/tmp/work/.agents/skills/capability-report",
          },
          permission_summary: ["Level 0 只读发现"],
          metadata: {},
          allowed_tools: [],
          resource_summary: {
            has_scripts: true,
          },
          standard_compliance: {
            is_standard: true,
          },
          runtime_binding_target: "workspace_skill",
          binding_status: "ready_for_manual_enable",
          binding_status_reason: "已具备后续 runtime binding 候选资格。",
          next_gate: "manual_runtime_enable",
          query_loop_visible: false,
          tool_runtime_visible: false,
          launch_enabled: false,
          runtime_gate: "等待 P3E 显式启用。",
        },
      ],
    } as any);

    const { container } = renderPanel({
      workspaceRoot: "/tmp/work",
      workspaceId: "project-1",
      onCreateManagedAutomationDraft,
      completionAuditSummariesByDirectory: {
        "capability-report": {
          source: "runtime_evidence_pack_completion_audit",
          decision: "completed",
          owner_run_count: 1,
          successful_owner_run_count: 1,
          workspace_skill_tool_call_count: 1,
          artifact_count: 2,
          owner_audit_statuses: ["audit_input_ready"],
          required_evidence: {
            automation_owner: true,
            workspace_skill_tool_call: true,
            artifact_or_timeline: true,
          },
          blocking_reasons: [],
          notes: [],
        },
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const envelopeButton = container.querySelector(
      '[data-testid="workspace-registered-agent-envelope-action"]',
    ) as HTMLButtonElement | null;
    expect(envelopeButton).toBeTruthy();
    expect(envelopeButton?.disabled).toBe(false);
    expect(container.textContent).toContain("completion audit completed");

    await act(async () => {
      envelopeButton?.click();
      await Promise.resolve();
    });

    expect(onCreateManagedAutomationDraft).toHaveBeenCalledTimes(1);
    expect(onCreateManagedAutomationDraft.mock.calls[0]?.[0]).toMatchObject({
      directory: "capability-report",
      binding_status: "ready_for_manual_enable",
    });
  });

  it("Prompt-to-Artifact UI smoke 只在 completed audit 后启用 Agent 草案入口", async () => {
    const onCreateManagedAutomationDraft = vi.fn();
    vi.mocked(capabilityDraftsApi.listRegisteredSkills).mockResolvedValue([
      {
        key: "workspace:capability-report",
        name: "只读 CLI 报告",
        description: "把本地只读 CLI 输出整理成 Markdown 报告。",
        directory: "capability-report",
        registeredSkillDirectory: "/tmp/work/.agents/skills/capability-report",
        registration: {
          registrationId: "capreg-1",
          registeredAt: "2026-05-05T01:10:00.000Z",
          skillDirectory: "capability-report",
          registeredSkillDirectory:
            "/tmp/work/.agents/skills/capability-report",
          sourceDraftId: "capdraft-1",
          sourceVerificationReportId: "capver-1",
          generatedFileCount: 4,
          permissionSummary: ["Level 0 只读发现"],
        },
        permissionSummary: ["Level 0 只读发现"],
        metadata: {},
        allowedTools: [],
        resourceSummary: {
          hasScripts: true,
          hasReferences: false,
          hasAssets: false,
        },
        standardCompliance: {
          isStandard: true,
          validationErrors: [],
          deprecatedFields: [],
        },
        launchEnabled: false,
        runtimeGate: "等待 runtime gate。",
      },
    ]);
    vi.mocked(listWorkspaceSkillBindings).mockResolvedValue({
      request: {
        workspace_root: "/tmp/work",
        caller: "assistant",
        surface: {
          workbench: true,
          browser_assist: false,
        },
      },
      warnings: [],
      counts: {
        registered_total: 1,
        ready_for_manual_enable_total: 1,
        blocked_total: 0,
        query_loop_visible_total: 0,
        tool_runtime_visible_total: 0,
        launch_enabled_total: 0,
      },
      bindings: [
        {
          key: "workspace_skill:capability-report",
          name: "只读 CLI 报告",
          description: "把本地只读 CLI 输出整理成 Markdown 报告。",
          directory: "capability-report",
          registered_skill_directory:
            "/tmp/work/.agents/skills/capability-report",
          registration: {
            source_draft_id: "capdraft-1",
            source_verification_report_id: "capver-1",
            registered_skill_directory:
              "/tmp/work/.agents/skills/capability-report",
          },
          permission_summary: ["Level 0 只读发现"],
          metadata: {},
          allowed_tools: [],
          resource_summary: {
            has_scripts: true,
          },
          standard_compliance: {
            is_standard: true,
          },
          runtime_binding_target: "workspace_skill",
          binding_status: "ready_for_manual_enable",
          binding_status_reason: "已具备后续 runtime binding 候选资格。",
          next_gate: "manual_runtime_enable",
          query_loop_visible: false,
          tool_runtime_visible: false,
          launch_enabled: false,
          runtime_gate: "等待 P3E 显式启用。",
        },
      ],
    } as any);

    const incompleteAudit = {
      source: "runtime_evidence_pack_completion_audit",
      decision: "needs_input",
      owner_run_count: 1,
      successful_owner_run_count: 1,
      workspace_skill_tool_call_count: 0,
      artifact_count: 1,
      owner_audit_statuses: ["audit_input_ready"],
      required_evidence: {
        automation_owner: true,
        workspace_skill_tool_call: false,
        artifact_or_timeline: true,
      },
      blocking_reasons: ["missing_workspace_skill_tool_call"],
      notes: [],
    };
    const completedAudit = {
      ...incompleteAudit,
      decision: "completed",
      workspace_skill_tool_call_count: 1,
      required_evidence: {
        automation_owner: true,
        workspace_skill_tool_call: true,
        artifact_or_timeline: true,
      },
      blocking_reasons: [],
    };

    const { container, root } = renderPanel({
      workspaceRoot: "/tmp/work",
      workspaceId: "project-1",
      onCreateManagedAutomationDraft,
      completionAuditSummariesByDirectory: {
        "capability-report": incompleteAudit,
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const initialEnvelopeButton = container.querySelector(
      '[data-testid="workspace-registered-agent-envelope-action"]',
    ) as HTMLButtonElement | null;
    expect(initialEnvelopeButton).toBeTruthy();
    expect(initialEnvelopeButton?.disabled).toBe(true);
    expect(container.textContent).toContain("项目助手：试用通过后再保存。");
    expect(container.textContent).toContain("共享：先只对你可见。");
    expect(container.textContent).not.toContain(
      "项目助手：已准备好保存（capability-report）",
    );

    await act(async () => {
      root.render(
        <WorkspaceRegisteredSkillsPanel
          workspaceRoot="/tmp/work"
          workspaceId="project-1"
          onCreateManagedAutomationDraft={onCreateManagedAutomationDraft}
          completionAuditSummariesByDirectory={{
            "capability-report": completedAudit,
          }}
        />,
      );
      await Promise.resolve();
    });

    const completedEnvelopeButton = container.querySelector(
      '[data-testid="workspace-registered-agent-envelope-action"]',
    ) as HTMLButtonElement | null;
    expect(completedEnvelopeButton?.disabled).toBe(false);
    expect(container.textContent).toContain("completion audit completed");
    expect(container.textContent).toContain(
      "项目助手：已准备好保存（capability-report）",
    );
    expect(container.textContent).toContain("共享：可在当前项目内共享");
    expect(container.textContent).toContain(
      "团队可见性：当前项目成员可以使用这条技能。",
    );

    await act(async () => {
      completedEnvelopeButton?.click();
      await Promise.resolve();
    });

    expect(onCreateManagedAutomationDraft).toHaveBeenCalledTimes(1);
    expect(onCreateManagedAutomationDraft.mock.calls[0]?.[0]).toMatchObject({
      directory: "capability-report",
      binding_status: "ready_for_manual_enable",
    });
  });

  it("refreshSignal 变化时应重新读取已注册能力", async () => {
    vi.mocked(capabilityDraftsApi.listRegisteredSkills)
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([
        {
          key: "workspace:capability-new",
          name: "新注册能力",
          description: "刷新后出现。",
          directory: "capability-new",
          registeredSkillDirectory: "/tmp/work/.agents/skills/capability-new",
          registration: {
            registrationId: "capreg-2",
            registeredAt: "2026-05-05T01:20:00.000Z",
            skillDirectory: "capability-new",
            registeredSkillDirectory: "/tmp/work/.agents/skills/capability-new",
            sourceDraftId: "capdraft-2",
            sourceVerificationReportId: "capver-2",
            generatedFileCount: 3,
            permissionSummary: ["Level 0 只读发现"],
          },
          permissionSummary: ["Level 0 只读发现"],
          metadata: {},
          allowedTools: [],
          resourceSummary: {
            hasScripts: false,
            hasReferences: false,
            hasAssets: false,
          },
          standardCompliance: {
            isStandard: true,
            validationErrors: [],
            deprecatedFields: [],
          },
          launchEnabled: false,
          runtimeGate: "等待 runtime gate。",
        },
      ]);
    vi.mocked(listWorkspaceSkillBindings)
      .mockResolvedValueOnce({
        request: {
          workspace_root: "/tmp/work",
          caller: "assistant",
          surface: {
            workbench: true,
            browser_assist: false,
          },
        },
        warnings: [],
        counts: {
          registered_total: 0,
          ready_for_manual_enable_total: 0,
          blocked_total: 0,
          query_loop_visible_total: 0,
          tool_runtime_visible_total: 0,
          launch_enabled_total: 0,
        },
        bindings: [],
      })
      .mockResolvedValueOnce({
        request: {
          workspace_root: "/tmp/work",
          caller: "assistant",
          surface: {
            workbench: true,
            browser_assist: false,
          },
        },
        warnings: [],
        counts: {
          registered_total: 1,
          ready_for_manual_enable_total: 1,
          blocked_total: 0,
          query_loop_visible_total: 0,
          tool_runtime_visible_total: 0,
          launch_enabled_total: 0,
        },
        bindings: [
          {
            key: "workspace_skill:capability-new",
            name: "新注册能力",
            description: "刷新后出现。",
            directory: "capability-new",
            registered_skill_directory:
              "/tmp/work/.agents/skills/capability-new",
            registration: {
              registration_id: "capreg-2",
              registered_at: "2026-05-05T01:20:00.000Z",
              skill_directory: "capability-new",
              registered_skill_directory:
                "/tmp/work/.agents/skills/capability-new",
              source_draft_id: "capdraft-2",
              source_verification_report_id: "capver-2",
              generated_file_count: 3,
              permission_summary: ["Level 0 只读发现"],
            },
            permission_summary: ["Level 0 只读发现"],
            metadata: {},
            allowed_tools: [],
            resource_summary: {
              has_scripts: false,
              has_references: false,
              has_assets: false,
            },
            standard_compliance: {
              is_standard: true,
              validation_errors: [],
              deprecated_fields: [],
            },
            runtime_binding_target: "workspace_skill",
            binding_status: "ready_for_manual_enable",
            binding_status_reason: "已具备后续 runtime binding 候选资格。",
            next_gate: "manual_runtime_enable",
            query_loop_visible: false,
            tool_runtime_visible: false,
            launch_enabled: false,
            runtime_gate: "等待 P3C 后续绑定。",
          },
        ],
      });

    const { container, root } = renderPanel({
      workspaceRoot: "/tmp/work",
      refreshSignal: 0,
    });

    await act(async () => {
      await Promise.resolve();
    });
    expect(container.textContent).toContain("当前项目还没有已保存技能");

    await act(async () => {
      root.render(
        <WorkspaceRegisteredSkillsPanel
          workspaceRoot="/tmp/work"
          refreshSignal={1}
        />,
      );
      await Promise.resolve();
    });

    expect(capabilityDraftsApi.listRegisteredSkills).toHaveBeenCalledTimes(2);
    expect(listWorkspaceSkillBindings).toHaveBeenCalledTimes(2);
    expect(container.textContent).toContain("新注册能力");
  });
});
