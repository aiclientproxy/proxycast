import { act, type ComponentProps, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, vi } from "vitest";
import type { AgentRuntimeToolInventory } from "@/lib/api/agentRuntime/toolInventoryTypes";
import {
  areLightweightRenderersRegistered,
  registerLightweightRenderers,
} from "@/components/artifact/renderers";
import { changeLimeLocale } from "@/i18n/createI18n";
import { HarnessStatusPanel } from "./HarnessStatusPanel";
import type { HarnessSessionState } from "../utils/harnessState";
import { clearAgentUiProjectionEvents } from "../projection/conversationProjectionStore";

const {
  exportAgentRuntimeAnalysisHandoffMock,
  exportAgentRuntimeHandoffBundleMock,
  exportAgentRuntimeReplayCaseMock,
  exportAgentRuntimeReviewDecisionTemplateMock,
  saveAgentRuntimeReviewDecisionMock,
  mockOpenExternalUrlWithSystemBrowser,
  mockToast,
} = vi.hoisted(() => ({
  exportAgentRuntimeAnalysisHandoffMock: vi.fn(),
  exportAgentRuntimeHandoffBundleMock: vi.fn(),
  exportAgentRuntimeReplayCaseMock: vi.fn(),
  exportAgentRuntimeReviewDecisionTemplateMock: vi.fn(),
  saveAgentRuntimeReviewDecisionMock: vi.fn(),
  mockOpenExternalUrlWithSystemBrowser: vi.fn().mockResolvedValue(undefined),
  mockToast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  },
}));

export function getHarnessPanelTestMocks() {
  return {
    exportAgentRuntimeAnalysisHandoffMock,
    exportAgentRuntimeHandoffBundleMock,
    exportAgentRuntimeReplayCaseMock,
    exportAgentRuntimeReviewDecisionTemplateMock,
    saveAgentRuntimeReviewDecisionMock,
    mockOpenExternalUrlWithSystemBrowser,
    mockToast,
  };
}

vi.mock("@/lib/api/agentRuntime/exportClient", async () => {
  const actual = await vi.importActual<
    typeof import("@/lib/api/agentRuntime/exportClient")
  >("@/lib/api/agentRuntime/exportClient");
  return {
    ...actual,
    exportAgentRuntimeAnalysisHandoff: exportAgentRuntimeAnalysisHandoffMock,
    exportAgentRuntimeHandoffBundle: exportAgentRuntimeHandoffBundleMock,
    exportAgentRuntimeReplayCase: exportAgentRuntimeReplayCaseMock,
    exportAgentRuntimeReviewDecisionTemplate:
      exportAgentRuntimeReviewDecisionTemplateMock,
    saveAgentRuntimeReviewDecision: saveAgentRuntimeReviewDecisionMock,
  };
});

vi.mock("sonner", () => ({
  toast: mockToast,
}));

vi.mock("@/lib/api/externalUrl", () => ({
  openExternalUrlWithSystemBrowser: mockOpenExternalUrlWithSystemBrowser,
}));

vi.mock("react-syntax-highlighter/dist/esm/prism", () => ({
  default: ({ children }: { children?: unknown }) => (
    <pre data-testid="syntax-highlighter-mock">{String(children ?? "")}</pre>
  ),
}));

vi.mock("react-syntax-highlighter/dist/esm/styles/prism", () => ({
  oneLight: {},
}));

export interface RenderResult {
  container: HTMLDivElement;
  root: Root;
}

const mountedRoots: RenderResult[] = [];
let originalClipboard: Clipboard | undefined;
let originalWindowOpen: typeof window.open;

export function createHarnessState(
  overrides: Partial<HarnessSessionState> = {},
): HarnessSessionState {
  return {
    runtimeStatus: null,
    pendingApprovals: [],
    latestContextTrace: [],
    plan: {
      phase: "idle",
      items: [],
    },
    activity: {
      planning: 0,
      filesystem: 1,
      execution: 0,
      web: 0,
      skills: 0,
      delegation: 0,
    },
    delegatedTasks: [],
    outputSignals: [],
    activeFileWrites: [],
    recentFileEvents: [],
    hasSignals: true,
    ...overrides,
  };
}

export function renderPanel(
  overrides: Partial<ComponentProps<typeof HarnessStatusPanel>> = {},
): RenderResult {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(
      <HarnessStatusPanel
        harnessState={createHarnessState()}
        environment={{
          skillsCount: 2,
          skillNames: ["read_file", "write_todos"],
          memorySignals: ["风格"],
          contextItemsCount: 2,
          activeContextCount: 1,
          contextItemNames: ["需求.md"],
          contextEnabled: true,
        }}
        {...overrides}
      />,
    );
  });

  const rendered = { container, root };
  mountedRoots.push(rendered);
  return rendered;
}

export function renderExpandedPanel(
  overrides: Partial<ComponentProps<typeof HarnessStatusPanel>> = {},
): RenderResult {
  return renderPanel({
    layout: "dialog",
    ...overrides,
  });
}

export function mountHarnessElement(element: ReactNode): RenderResult {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(element);
  });

  const rendered = { container, root };
  mountedRoots.push(rendered);
  return rendered;
}

export function setInputValue(
  input: HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement,
  value: string,
) {
  const prototype =
    input instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : input instanceof HTMLSelectElement
        ? HTMLSelectElement.prototype
        : HTMLInputElement.prototype;
  const descriptor = Object.getOwnPropertyDescriptor(prototype, "value");
  descriptor?.set?.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

export function findButtonByText(text: string): HTMLButtonElement | null {
  return Array.from(document.body.querySelectorAll("button")).find(
    (button): button is HTMLButtonElement =>
      button.textContent?.trim() === text,
  ) as HTMLButtonElement | null;
}

export async function flushUntilTextAppears(text: string): Promise<void> {
  for (let index = 0; index < 80; index += 1) {
    if (document.body.textContent?.includes(text)) {
      return;
    }
    await act(async () => {
      await Promise.resolve();
      await new Promise((resolve) => window.setTimeout(resolve, 0));
    });
  }
}

export function createToolInventory(): AgentRuntimeToolInventory {
  return {
    request: {
      caller: "assistant",
      surface: {
        workbench: false,
        browser_assist: true,
      },
    },
    agent_initialized: true,
    warnings: ["extension 搜索工具面存在延迟加载项"],
    mcp_servers: ["lime-browser"],
    default_allowed_tools: ["ToolSearch", "WebSearch"],
    counts: {
      catalog_total: 3,
      catalog_current_total: 3,
      catalog_compat_total: 0,
      catalog_deprecated_total: 0,
      default_allowed_total: 2,
      runtime_total: 4,
      runtime_visible_total: 3,
      native_total: 2,
      native_visible_total: 1,
      native_catalog_unmapped_total: 0,
      extension_surface_total: 1,
      extension_mcp_bridge_total: 1,
      extension_runtime_total: 0,
      extension_tool_total: 1,
      extension_tool_visible_total: 1,
      mcp_server_total: 1,
      mcp_tool_total: 1,
      mcp_tool_visible_total: 1,
    },
    catalog_tools: [
      {
        name: "bash",
        profiles: ["core"],
        capabilities: ["execution"],
        lifecycle: "current",
        source: "agent_builtin",
        permission_plane: "parameter_restricted",
        workspace_default_allow: false,
        execution_warning_policy: "shell_command_risk",
        execution_warning_policy_source: "runtime",
        execution_restriction_profile: "workspace_shell_command",
        execution_restriction_profile_source: "runtime",
        execution_sandbox_profile: "workspace_command",
        execution_sandbox_profile_source: "runtime",
      },
      {
        name: "write",
        profiles: ["core"],
        capabilities: ["workspace_io"],
        lifecycle: "current",
        source: "agent_builtin",
        permission_plane: "parameter_restricted",
        workspace_default_allow: false,
        execution_warning_policy: "none",
        execution_warning_policy_source: "persisted",
        execution_restriction_profile: "workspace_path_required",
        execution_restriction_profile_source: "persisted",
        execution_sandbox_profile: "none",
        execution_sandbox_profile_source: "default",
      },
      {
        name: "ToolSearch",
        profiles: ["core"],
        capabilities: ["web_search"],
        lifecycle: "current",
        source: "lime_injected",
        permission_plane: "session_allowlist",
        workspace_default_allow: true,
        execution_warning_policy: "none",
        execution_warning_policy_source: "default",
        execution_restriction_profile: "none",
        execution_restriction_profile_source: "default",
        execution_sandbox_profile: "none",
        execution_sandbox_profile_source: "default",
      },
    ],
    native_tools: [
      {
        name: "bash",
        description: "执行工作区命令",
        catalog_entry_name: "bash",
        catalog_source: "agent_builtin",
        catalog_lifecycle: "current",
        catalog_permission_plane: "parameter_restricted",
        catalog_workspace_default_allow: false,
        catalog_execution_warning_policy: "shell_command_risk",
        catalog_execution_warning_policy_source: "runtime",
        catalog_execution_restriction_profile: "workspace_shell_command",
        catalog_execution_restriction_profile_source: "runtime",
        catalog_execution_sandbox_profile: "workspace_command",
        catalog_execution_sandbox_profile_source: "runtime",
        deferred_loading: false,
        always_visible: false,
        allowed_callers: ["assistant"],
        tags: ["shell"],
        input_examples_count: 2,
        has_output_schema: false,
        caller_allowed: true,
        visible_in_context: true,
      },
      {
        name: "ToolSearch",
        description: "搜索工具目录",
        catalog_entry_name: "ToolSearch",
        catalog_source: "lime_injected",
        catalog_lifecycle: "current",
        catalog_permission_plane: "session_allowlist",
        catalog_workspace_default_allow: true,
        catalog_execution_warning_policy: "none",
        catalog_execution_warning_policy_source: "default",
        catalog_execution_restriction_profile: "none",
        catalog_execution_restriction_profile_source: "default",
        catalog_execution_sandbox_profile: "none",
        catalog_execution_sandbox_profile_source: "default",
        deferred_loading: true,
        always_visible: true,
        allowed_callers: [],
        tags: ["search"],
        input_examples_count: 1,
        has_output_schema: false,
        caller_allowed: false,
        visible_in_context: false,
      },
    ],
    runtime_tools: [
      {
        name: "Agent",
        description: "创建或调度子任务",
        source_kind: "current_surface",
        catalog_entry_name: "Agent",
        catalog_source: "agent_builtin",
        catalog_lifecycle: "current",
        catalog_permission_plane: "session_allowlist",
        catalog_workspace_default_allow: false,
        deferred_loading: false,
        always_visible: false,
        allowed_callers: [],
        tags: [],
        input_examples_count: 0,
        has_output_schema: false,
        caller_allowed: true,
        visible_in_context: true,
      },
      {
        name: "bash",
        description: "执行工作区命令",
        source_kind: "current_surface",
        catalog_entry_name: "bash",
        catalog_source: "agent_builtin",
        catalog_lifecycle: "current",
        catalog_permission_plane: "parameter_restricted",
        catalog_workspace_default_allow: false,
        deferred_loading: false,
        always_visible: false,
        allowed_callers: ["assistant"],
        tags: ["shell"],
        input_examples_count: 2,
        has_output_schema: false,
        caller_allowed: true,
        visible_in_context: true,
      },
      {
        name: "mcp__lime-browser__navigate",
        description: "打开网页",
        source_kind: "runtime_extension",
        source_label: "mcp__lime-browser",
        status: "loaded",
        deferred_loading: false,
        always_visible: false,
        allowed_callers: ["assistant"],
        tags: [],
        input_examples_count: 0,
        has_output_schema: true,
        caller_allowed: true,
        visible_in_context: true,
      },
      {
        name: "ToolSearch",
        description: "搜索工具目录",
        source_kind: "current_surface",
        catalog_entry_name: "ToolSearch",
        catalog_source: "lime_injected",
        catalog_lifecycle: "current",
        catalog_permission_plane: "session_allowlist",
        catalog_workspace_default_allow: true,
        deferred_loading: true,
        always_visible: true,
        allowed_callers: [],
        tags: ["search"],
        input_examples_count: 1,
        has_output_schema: false,
        caller_allowed: false,
        visible_in_context: false,
      },
    ],
    extension_surfaces: [
      {
        extension_name: "mcp__lime-browser",
        description: "浏览器桥接工具面",
        source_kind: "mcp_bridge",
        deferred_loading: true,
        allowed_caller: "assistant",
        available_tools: ["navigate", "click"],
        always_expose_tools: ["navigate"],
        loaded_tools: ["mcp__lime-browser__navigate"],
        searchable_tools: [
          "mcp__lime-browser__navigate",
          "mcp__lime-browser__click",
        ],
      },
    ],
    extension_tools: [
      {
        name: "mcp__lime-browser__navigate",
        description: "打开网页",
        extension_name: "mcp__lime-browser",
        source_kind: "mcp_bridge",
        deferred_loading: false,
        allowed_caller: "assistant",
        status: "loaded",
        caller_allowed: true,
        visible_in_context: true,
      },
    ],
    mcp_tools: [
      {
        server_name: "lime-browser",
        name: "mcp__lime-browser__navigate",
        description: "导航到指定页面",
        deferred_loading: false,
        always_visible: true,
        allowed_callers: ["assistant"],
        tags: ["browser", "navigation"],
        input_examples_count: 1,
        has_output_schema: true,
        caller_allowed: true,
        visible_in_context: true,
      },
    ],
  };
}

export function createAlignedRuntimeToolInventory(): AgentRuntimeToolInventory {
  const base = createToolInventory();

  return {
    ...base,
    counts: {
      ...base.counts,
      runtime_total: 10,
      runtime_visible_total: 9,
    },
    runtime_tools: [
      {
        name: "Agent",
        description: "创建或调度子任务",
        source_kind: "current_surface",
        catalog_entry_name: "Agent",
        catalog_source: "agent_builtin",
        catalog_lifecycle: "current",
        catalog_permission_plane: "session_allowlist",
        catalog_workspace_default_allow: false,
        deferred_loading: false,
        always_visible: false,
        allowed_callers: [],
        tags: [],
        input_examples_count: 0,
        has_output_schema: false,
        caller_allowed: true,
        visible_in_context: true,
      },
      {
        name: "SendMessage",
        description: "向子任务追加输入",
        source_kind: "current_surface",
        catalog_entry_name: "SendMessage",
        catalog_source: "agent_builtin",
        catalog_lifecycle: "current",
        catalog_permission_plane: "session_allowlist",
        catalog_workspace_default_allow: false,
        deferred_loading: false,
        always_visible: false,
        allowed_callers: [],
        tags: ["team"],
        input_examples_count: 0,
        has_output_schema: false,
        caller_allowed: true,
        visible_in_context: true,
      },
      {
        name: "TeamCreate",
        description: "创建子代理组",
        source_kind: "current_surface",
        catalog_entry_name: "TeamCreate",
        catalog_source: "agent_builtin",
        catalog_lifecycle: "current",
        catalog_permission_plane: "session_allowlist",
        catalog_workspace_default_allow: false,
        deferred_loading: false,
        always_visible: false,
        allowed_callers: [],
        tags: ["team"],
        input_examples_count: 0,
        has_output_schema: false,
        caller_allowed: true,
        visible_in_context: true,
      },
      {
        name: "TeamDelete",
        description: "删除子代理组",
        source_kind: "current_surface",
        catalog_entry_name: "TeamDelete",
        catalog_source: "agent_builtin",
        catalog_lifecycle: "current",
        catalog_permission_plane: "session_allowlist",
        catalog_workspace_default_allow: false,
        deferred_loading: false,
        always_visible: false,
        allowed_callers: [],
        tags: ["team"],
        input_examples_count: 0,
        has_output_schema: false,
        caller_allowed: true,
        visible_in_context: true,
      },
      {
        name: "ListPeers",
        description: "列出当前子代理成员",
        source_kind: "current_surface",
        catalog_entry_name: "ListPeers",
        catalog_source: "agent_builtin",
        catalog_lifecycle: "current",
        catalog_permission_plane: "session_allowlist",
        catalog_workspace_default_allow: false,
        deferred_loading: false,
        always_visible: false,
        allowed_callers: [],
        tags: ["team"],
        input_examples_count: 0,
        has_output_schema: false,
        caller_allowed: true,
        visible_in_context: true,
      },
      {
        name: "update_plan",
        description: "维护执行计划",
        source_kind: "current_surface",
        catalog_entry_name: "update_plan",
        catalog_source: "lime_injected",
        catalog_lifecycle: "current",
        catalog_permission_plane: "session_allowlist",
        catalog_workspace_default_allow: false,
        deferred_loading: false,
        always_visible: false,
        allowed_callers: [],
        tags: ["plan"],
        input_examples_count: 0,
        has_output_schema: false,
        caller_allowed: true,
        visible_in_context: true,
      },
      {
        name: "WebSearch",
        description: "执行联网检索",
        source_kind: "current_surface",
        catalog_entry_name: "WebSearch",
        catalog_source: "lime_injected",
        catalog_lifecycle: "current",
        catalog_permission_plane: "session_allowlist",
        catalog_workspace_default_allow: true,
        deferred_loading: false,
        always_visible: false,
        allowed_callers: [],
        tags: ["search"],
        input_examples_count: 0,
        has_output_schema: false,
        caller_allowed: true,
        visible_in_context: true,
      },
      ...(base.runtime_tools ?? []).filter((entry) => entry.name !== "Agent"),
    ],
  };
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  await changeLimeLocale("zh-CN");

  if (!areLightweightRenderersRegistered()) {
    registerLightweightRenderers();
  }

  originalClipboard = navigator.clipboard;
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: {
      writeText: vi.fn().mockResolvedValue(undefined),
    },
  });
  originalWindowOpen = window.open;
  Object.defineProperty(window, "open", {
    configurable: true,
    value: vi.fn(),
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
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: originalClipboard,
  });
  Object.defineProperty(window, "open", {
    configurable: true,
    value: originalWindowOpen,
  });
  clearAgentUiProjectionEvents();
  vi.clearAllMocks();
});
