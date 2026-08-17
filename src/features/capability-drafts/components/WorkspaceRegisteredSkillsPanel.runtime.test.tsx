import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { capabilityDraftsApi } from "@/lib/api/capabilityDrafts";
import { listWorkspaceSkillBindings } from "@/lib/api/agentRuntime/inventoryClient";
import {
  scheduledTasksApi,
  type ScheduledTask,
} from "@/lib/api/scheduledTasks";
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
  });

  it("显式传入 runtime enable handler 时，仅 ready binding 可触发本回合启用", async () => {
    const onEnableRuntime = vi.fn();
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
          verificationGates: [
            {
              checkId: "readonly_http_execution_preflight",
              label: "只读 HTTP 执行 preflight",
              evidence: [],
            },
          ],
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
            registration_id: "capreg-1",
            registered_at: "2026-05-05T01:10:00.000Z",
            skill_directory: "capability-report",
            registered_skill_directory:
              "/tmp/work/.agents/skills/capability-report",
            source_draft_id: "capdraft-1",
            source_verification_report_id: "capver-1",
            generated_file_count: 4,
            permission_summary: ["Level 0 只读发现"],
          },
          permission_summary: ["Level 0 只读发现"],
          metadata: {},
          allowed_tools: [],
          resource_summary: {
            has_scripts: true,
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
          runtime_gate: "等待 P3E 显式启用。",
        },
      ],
    });

    const { container } = renderPanel({
      workspaceRoot: "/tmp/work",
      onEnableRuntime,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const enableButton = container.querySelector(
      '[data-testid="workspace-registered-skill-enable-runtime"]',
    ) as HTMLButtonElement | null;
    expect(enableButton).toBeTruthy();
    expect(enableButton?.disabled).toBe(false);

    await act(async () => {
      enableButton?.click();
      await Promise.resolve();
    });

    expect(onEnableRuntime).toHaveBeenCalledTimes(1);
    expect(onEnableRuntime.mock.calls[0]?.[0]).toMatchObject({
      directory: "capability-report",
      binding_status: "ready_for_manual_enable",
    });
  });

  it("显式传入 managed automation handler 时，ready binding 可打开 Managed Job 草案", async () => {
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
          verificationGates: [
            {
              checkId: "readonly_http_execution_preflight",
              label: "只读 HTTP 执行 preflight",
              evidence: [],
            },
          ],
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
            registration_id: "capreg-1",
            registered_at: "2026-05-05T01:10:00.000Z",
            skill_directory: "capability-report",
            registered_skill_directory:
              "/tmp/work/.agents/skills/capability-report",
            source_draft_id: "capdraft-1",
            source_verification_report_id: "capver-1",
            generated_file_count: 4,
            permission_summary: ["Level 0 只读发现"],
          },
          permission_summary: ["Level 0 只读发现"],
          metadata: {},
          allowed_tools: [],
          resource_summary: {
            has_scripts: true,
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
          runtime_gate: "等待 P3E 显式启用。",
        },
      ],
    });
    const managedTask: ScheduledTask = {
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
    vi.mocked(scheduledTasksApi.listDetailed).mockResolvedValueOnce([
      managedTask,
    ]);
    vi.mocked(scheduledTasksApi.setEnabled).mockResolvedValueOnce({
      ...managedTask,
      enabled: true,
    });

    const { container } = renderPanel({
      workspaceRoot: "/tmp/work",
      workspaceId: "project-1",
      onCreateManagedAutomationDraft,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const managedButton = container.querySelector(
      '[data-testid="workspace-registered-agent-managed-automation"]',
    ) as HTMLButtonElement | null;
    const toggleButton = container.querySelector(
      '[data-testid="workspace-registered-agent-managed-automation-toggle"]',
    ) as HTMLButtonElement | null;
    expect(managedButton).toBeTruthy();
    expect(managedButton?.disabled).toBe(false);
    expect(container.textContent).toContain("项目助手：试用通过后再保存。");
    expect(container.textContent).toContain("共享：先只对你可见。");
    expect(container.textContent).toContain(
      "团队可见性：当前项目成员可以使用这条技能。",
    );
    expect(container.textContent).toContain("记录：已保留检查结果");
    expect(container.textContent).toContain("结果：试用后展示最近状态");
    expect(container.textContent).toContain("Managed Job：草案暂停");
    expect(container.textContent).toContain("Schedule：每天 09:00");
    expect(toggleButton).toBeTruthy();
    expect(toggleButton?.textContent).toContain("开启定时运行");
    expect(
      container.querySelector(
        '[data-testid="workspace-registered-agent-completion-audit"]',
      ),
    ).toBeNull();

    await act(async () => {
      managedButton?.click();
      await Promise.resolve();
    });

    expect(onCreateManagedAutomationDraft).toHaveBeenCalledTimes(1);
    expect(onCreateManagedAutomationDraft.mock.calls[0]?.[0]).toMatchObject({
      directory: "capability-report",
      binding_status: "ready_for_manual_enable",
      registration: {
        source_draft_id: "capdraft-1",
        source_verification_report_id: "capver-1",
      },
    });
    expect(onCreateManagedAutomationDraft.mock.calls[0]).toHaveLength(1);

    await act(async () => {
      toggleButton?.click();
      await Promise.resolve();
    });

    expect(scheduledTasksApi.setEnabled).toHaveBeenCalledWith("job-1", true);
    expect(container.textContent).toContain("Managed Job：已启用");

    const envelopeButton = container.querySelector(
      '[data-testid="workspace-registered-agent-envelope-action"]',
    ) as HTMLButtonElement | null;
    expect(envelopeButton?.disabled).toBe(true);

    await act(async () => {
      envelopeButton?.click();
      await Promise.resolve();
    });

    expect(onCreateManagedAutomationDraft).toHaveBeenCalledTimes(1);
  });
});
