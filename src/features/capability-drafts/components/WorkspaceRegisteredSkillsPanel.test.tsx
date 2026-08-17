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

  it("没有项目根目录时只显示选择项目提示，不读取已注册能力", () => {
    const { container } = renderPanel();

    expect(container.textContent).toContain("已保存技能");
    expect(container.textContent).toContain("选择或进入一个项目");
    expect(capabilityDraftsApi.listRegisteredSkills).not.toHaveBeenCalled();
    expect(listWorkspaceSkillBindings).not.toHaveBeenCalled();
    expect(scheduledTasksApi.listDetailed).not.toHaveBeenCalled();
  });

  it("应展示已注册能力来源和 runtime gate，且不提供运行入口", async () => {
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
          permissionSummary: ["Level 0 只读发现", "允许执行本地 CLI"],
          verificationGates: [
            {
              checkId: "readonly_http_execution_preflight",
              label: "只读 HTTP 执行 preflight",
              evidence: [
                { key: "preflightMode", value: "approval_request_only" },
                { key: "endpointSource", value: "runtime_input" },
                { key: "method", value: "GET" },
                {
                  key: "credentialReferenceId",
                  value: "readonly_api_session",
                },
                {
                  key: "evidenceSchema",
                  value:
                    "request_url_hash,request_method,response_status,response_sha256,executed_at",
                },
                {
                  key: "policyPath",
                  value: "policy/readonly-http-session.json",
                },
              ],
            },
          ],
          approvalRequests: [
            {
              approvalId: "capreg-1:readonly-http-session",
              status: "pending",
              sourceCheckId: "readonly_http_execution_preflight",
              skillDirectory: "capability-report",
              endpointSource: "runtime_input",
              method: "GET",
              credentialReferenceId: "readonly_api_session",
              evidenceSchema: [
                "request_url_hash",
                "request_method",
                "response_status",
                "response_sha256",
                "executed_at",
              ],
              policyPath: "policy/readonly-http-session.json",
              createdAt: "2026-05-05T01:10:00.000Z",
              consumptionGate: {
                status: "awaiting_session_approval",
                requiredInputs: [
                  "session_user_approval",
                  "runtime_endpoint_input",
                  "credential_reference:readonly_api_session",
                  "evidence_capture",
                ],
                runtimeExecutionEnabled: false,
                credentialStorageEnabled: false,
                blockedReason:
                  "等待当前 session 显式授权；本阶段不执行真实 HTTP，也不保存凭证。",
                nextAction:
                  "先消费 approval request artifact 并解析 session-scoped 输入，之后才能进入受控 GET 执行门禁。",
              },
              credentialResolver: {
                status: "awaiting_session_credential",
                referenceId: "readonly_api_session",
                scope: "session",
                source: "user_session_config",
                secretMaterialStatus: "not_requested",
                tokenPersisted: false,
                runtimeInjectionEnabled: false,
                blockedReason:
                  "等待当前 session 提供或确认凭证引用；本阶段不读取、不保存 token。",
                nextAction:
                  "后续只能在 session scope 内解析该 reference，并把解析结果直接交给受控 GET 门禁。",
              },
              consumptionInputSchema: {
                schemaId: "readonly_http_session_approval_v1",
                version: 1,
                fields: [
                  {
                    key: "session_user_approval",
                    label: "Session 授权确认",
                    kind: "boolean_confirmation",
                    required: true,
                    source: "user_confirmation",
                    secret: false,
                    description: "用户必须在当前 session 明确确认。",
                  },
                  {
                    key: "runtime_endpoint_input",
                    label: "运行时 Endpoint",
                    kind: "url",
                    required: true,
                    source: "runtime_input",
                    secret: false,
                    description: "运行时输入 endpoint。",
                  },
                  {
                    key: "credential_reference_confirmation",
                    label: "凭证引用确认",
                    kind: "credential_reference",
                    required: true,
                    source: "user_session_config",
                    secret: false,
                    description: "确认凭证引用。",
                  },
                  {
                    key: "evidence_capture_consent",
                    label: "Evidence 捕获确认",
                    kind: "boolean_confirmation",
                    required: true,
                    source: "user_confirmation",
                    secret: false,
                    description: "确认捕获 evidence。",
                  },
                ],
                uiSubmissionEnabled: false,
                runtimeExecutionEnabled: false,
                blockedReason:
                  "当前只定义 session 授权输入合同，尚未开放提交、凭证解析或真实 HTTP 执行。",
              },
              sessionInputIntake: {
                status: "awaiting_session_inputs",
                schemaId: "readonly_http_session_approval_v1",
                scope: "session",
                requiredFieldKeys: [
                  "session_user_approval",
                  "runtime_endpoint_input",
                  "credential_reference_confirmation",
                  "evidence_capture_consent",
                ],
                missingFieldKeys: [
                  "session_user_approval",
                  "runtime_endpoint_input",
                  "credential_reference_confirmation",
                  "evidence_capture_consent",
                ],
                collectedFieldKeys: [],
                credentialReferenceId: "readonly_api_session",
                endpointInputPersisted: false,
                secretMaterialStatus: "not_collected",
                tokenPersisted: false,
                uiSubmissionEnabled: false,
                runtimeExecutionEnabled: false,
                blockedReason:
                  "已声明当前 session 输入槽位，但尚未接入提交处理、凭证解析或真实 HTTP 执行。",
                nextAction:
                  "后续只允许在当前 session 收集一次性授权输入，再进入受控 GET 执行门禁。",
              },
              sessionInputSubmissionContract: {
                status: "submission_contract_declared",
                scope: "session",
                mode: "one_time_session_submission",
                acceptedFieldKeys: [
                  "session_user_approval",
                  "runtime_endpoint_input",
                  "credential_reference_confirmation",
                  "evidence_capture_consent",
                ],
                validationRules: [
                  {
                    fieldKey: "session_user_approval",
                    kind: "boolean_confirmation",
                    required: true,
                    source: "user_confirmation",
                    secretAllowed: false,
                    rule: "必须为显式 true。",
                  },
                  {
                    fieldKey: "runtime_endpoint_input",
                    kind: "url",
                    required: true,
                    source: "runtime_input",
                    secretAllowed: false,
                    rule: "必须是 http/https URL。",
                  },
                  {
                    fieldKey: "credential_reference_confirmation",
                    kind: "credential_reference",
                    required: true,
                    source: "user_session_config",
                    secretAllowed: false,
                    rule: "不接收 token 明文。",
                  },
                ],
                valueRetention: "none",
                endpointInputPersisted: false,
                secretMaterialAccepted: false,
                tokenPersisted: false,
                evidenceCaptureRequired: true,
                submissionHandlerEnabled: true,
                uiSubmissionEnabled: false,
                runtimeExecutionEnabled: false,
                blockedReason:
                  "已开放 session-scoped 输入校验 handler；本阶段仍不解析凭证、不执行真实 HTTP。",
                nextAction:
                  "后续可先提交一次性 session 输入做校验；校验通过后仍只进入受控 GET 执行门禁。",
              },
            },
          ],
        },
        permissionSummary: ["Level 0 只读发现", "允许执行本地 CLI"],
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
        runtimeGate:
          "已注册为 Workspace 本地 Skill 包；进入运行前还需要 P3C runtime binding 与 tool_runtime 授权。",
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
      warnings: [
        "P3C 当前只返回 runtime binding readiness；不会 reload Skill，也不会注入默认 tool surface。",
      ],
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
            permission_summary: ["Level 0 只读发现", "允许执行本地 CLI"],
          },
          permission_summary: ["Level 0 只读发现", "允许执行本地 CLI"],
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
          binding_status_reason:
            "已具备后续 workspace catalog binding 候选资格；当前仍未注入 Query Loop 或 tool_runtime。",
          next_gate: "manual_runtime_enable",
          query_loop_visible: false,
          tool_runtime_visible: false,
          launch_enabled: false,
          runtime_gate:
            "等待 P3C 后续把该 workspace skill 显式绑定到 Query Loop metadata 与 tool_runtime 授权裁剪。",
        },
      ],
    });

    const { container } = renderPanel({ workspaceRoot: "/tmp/work" });

    await act(async () => {
      await Promise.resolve();
    });

    expect(capabilityDraftsApi.listRegisteredSkills).toHaveBeenCalledWith({
      workspaceRoot: "/tmp/work",
    });
    expect(listWorkspaceSkillBindings).toHaveBeenCalledWith({
      workspaceRoot: "/tmp/work",
      caller: "assistant",
      workbench: true,
    });
    expect(container.textContent).toContain("只读 CLI 报告");
    expect(container.textContent).toContain("已注册");
    expect(container.textContent).toContain("可试用");
    expect(container.textContent).toContain("capdraft-1 / capver-1");
    expect(container.textContent).toContain("注册 provenance");
    expect(container.textContent).toContain("只读 HTTP 执行 preflight");
    expect(container.textContent).toContain("方法");
    expect(container.textContent).toContain("GET");
    expect(container.textContent).toContain("凭证引用");
    expect(container.textContent).toContain("readonly_api_session");
    expect(container.textContent).toContain("证据 Schema");
    expect(container.textContent).toContain(
      "Session approval request artifact",
    );
    expect(container.textContent).toContain("capreg-1:readonly-http-session");
    expect(container.textContent).toContain("pending / 未执行 / 未保存凭证");
    expect(container.textContent).toContain(
      "真实 API 执行前必须先消费这条授权请求 artifact",
    );
    expect(container.textContent).toContain("Endpoint");
    expect(container.textContent).toContain("runtime_input");
    expect(container.textContent).toContain("Policy");
    expect(container.textContent).toContain(
      "policy/readonly-http-session.json",
    );
    expect(container.textContent).toContain("不保存");
    expect(container.textContent).toContain("pending / 未执行 / 未保存凭证");
    expect(container.textContent).toContain("消费门禁");
    expect(container.textContent).toContain("awaiting_session_approval");
    expect(container.textContent).toContain("session_user_approval");
    expect(container.textContent).toContain("runtime_endpoint_input");
    expect(container.textContent).toContain(
      "credential_reference:readonly_api_session",
    );
    expect(container.textContent).toContain("runtimeExecution=false");
    expect(container.textContent).toContain("credentialStorage=false");
    expect(container.textContent).toContain("Session credential resolver");
    expect(container.textContent).toContain("awaiting_session_credential");
    expect(container.textContent).toContain("Reference");
    expect(container.textContent).toContain("Scope");
    expect(container.textContent).toContain("session");
    expect(container.textContent).toContain("Source");
    expect(container.textContent).toContain("user_session_config");
    expect(container.textContent).toContain("Secret");
    expect(container.textContent).toContain("not_requested");
    expect(container.textContent).toContain("tokenPersistedfalse");
    expect(container.textContent).toContain("runtimeInjectionfalse");
    expect(container.textContent).toContain(
      "Approval consumption input schema",
    );
    expect(container.textContent).toContain(
      "readonly_http_session_approval_v1",
    );
    expect(container.textContent).toContain("uiSubmission=false");
    expect(container.textContent).toContain("runtimeExecution=false");
    expect(container.textContent).toContain(
      "session_user_approval:boolean_confirmation:required",
    );
    expect(container.textContent).toContain(
      "runtime_endpoint_input:url:required",
    );
    expect(container.textContent).toContain(
      "credential_reference_confirmation:credential_reference:required",
    );
    expect(container.textContent).toContain("Session input intake");
    expect(container.textContent).toContain("awaiting_session_inputs");
    expect(container.textContent).toContain("Secret");
    expect(container.textContent).toContain("not_collected");
    expect(container.textContent).toContain("endpointPersistedfalse");
    expect(container.textContent).toContain("missing:session_user_approval");
    expect(container.textContent).toContain("missing:runtime_endpoint_input");
    expect(container.textContent).toContain(
      "missing:credential_reference_confirmation",
    );
    expect(container.textContent).toContain("missing:evidence_capture_consent");
    expect(container.textContent).toContain("Session submission contract");
    expect(container.textContent).toContain("submission_contract_declared");
    expect(container.textContent).toContain("one_time_session_submission");
    expect(container.textContent).toContain("Retention");
    expect(container.textContent).toContain("none");
    expect(container.textContent).toContain("submitHandlertrue");
    expect(container.textContent).toContain("secretAcceptedfalse");
    expect(container.textContent).toContain("evidenceRequiredtrue");
    expect(container.textContent).toContain(
      "validate:runtime_endpoint_input:url:required",
    );
    expect(container.textContent).toContain(
      "validate:credential_reference_confirmation:credential_reference:required",
    );
    expect(container.textContent).toContain(
      "Level 0 只读发现 / 允许执行本地 CLI",
    );
    expect(container.textContent).toContain("scripts");
    expect(container.textContent).toContain("Agent Skills 标准通过");
    expect(container.textContent).toContain(
      "当前仍未注入 Query Loop 或 tool_runtime",
    );
    expect(container.textContent).toContain("manual_runtime_enable");
    expect(container.textContent).toContain("项目助手");
    expect(container.textContent).toContain("待试用");
    expect(container.textContent).toContain(
      "试用结果没问题后，可以把它保存成当前项目里的助手。",
    );
    expect(container.textContent).toContain("最近结果：还没有试用结果。");
    expect(container.textContent).toContain("Managed Job：未创建");
    expect(container.textContent).not.toContain("立即运行");
    expect(container.textContent).not.toContain("创建自动化");
    expect(container.textContent).not.toContain("继续这套方法");
  });
});
