import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";

const { mockUseComponentDebug } = vi.hoisted(() => ({
  mockUseComponentDebug: vi.fn(),
}));

const { mockGetConfig, mockSaveConfig } = vi.hoisted(() => ({
  mockGetConfig: vi.fn(),
  mockSaveConfig: vi.fn(),
}));

const {
  mockClearServiceSkillCatalogCache,
  mockGetServiceSkillCatalog,
  mockSubscribeServiceSkillCatalogChanged,
} = vi.hoisted(() => ({
  mockClearServiceSkillCatalogCache: vi.fn(),
  mockGetServiceSkillCatalog: vi.fn(),
  mockSubscribeServiceSkillCatalogChanged: vi.fn(),
}));

const { mockGetLogs, mockGetPersistedLogsTail } = vi.hoisted(() => ({
  mockGetLogs: vi.fn(),
  mockGetPersistedLogsTail: vi.fn(),
}));

const {
  mockGetServerDiagnostics,
  mockGetLogStorageDiagnostics,
  mockExportDiagnosticsTrace,
  mockExportSupportBundle,
  mockListDiagnosticsTraces,
  mockReadDiagnosticsTrace,
  mockGetWindowsStartupDiagnostics,
} = vi.hoisted(() => ({
  mockGetServerDiagnostics: vi.fn(),
  mockGetLogStorageDiagnostics: vi.fn(),
  mockExportDiagnosticsTrace: vi.fn(),
  mockExportSupportBundle: vi.fn(),
  mockListDiagnosticsTraces: vi.fn(),
  mockReadDiagnosticsTrace: vi.fn(),
  mockGetWindowsStartupDiagnostics: vi.fn(),
}));

const {
  mockBuildCrashDiagnosticPayload,
  mockClearCrashDiagnosticHistory,
  mockCollectRuntimeSnapshotForDiagnostic,
  mockCollectGeneralWorkbenchDocumentStateForDiagnostic,
  mockCopyTextToClipboard,
  mockCopyCrashDiagnosticJsonToClipboard,
  mockCopyCrashDiagnosticToClipboard,
  mockExportCrashDiagnosticToJson,
  mockIsClipboardPermissionDeniedError,
  mockNormalizeCrashReportingConfig,
  mockOpenCrashDiagnosticDownloadDirectory,
} = vi.hoisted(() => ({
  mockBuildCrashDiagnosticPayload: vi.fn(),
  mockClearCrashDiagnosticHistory: vi.fn(),
  mockCollectRuntimeSnapshotForDiagnostic: vi.fn(),
  mockCollectGeneralWorkbenchDocumentStateForDiagnostic: vi.fn(),
  mockCopyTextToClipboard: vi.fn(),
  mockCopyCrashDiagnosticJsonToClipboard: vi.fn(),
  mockCopyCrashDiagnosticToClipboard: vi.fn(),
  mockExportCrashDiagnosticToJson: vi.fn(),
  mockIsClipboardPermissionDeniedError: vi.fn(),
  mockNormalizeCrashReportingConfig: vi.fn(),
  mockOpenCrashDiagnosticDownloadDirectory: vi.fn(),
}));

const {
  mockEmitServiceSkillCatalogBootstrap,
  mockExtractServiceSkillCatalogFromBootstrapPayload,
} = vi.hoisted(() => ({
  mockEmitServiceSkillCatalogBootstrap: vi.fn(),
  mockExtractServiceSkillCatalogFromBootstrapPayload: vi.fn(),
}));

vi.mock("@/contexts/ComponentDebugContext", () => ({
  useComponentDebug: mockUseComponentDebug,
}));

vi.mock("@/lib/api/appConfig", () => ({
  getConfig: mockGetConfig,
  saveConfig: mockSaveConfig,
}));

vi.mock("@/lib/api/serviceSkills", () => ({
  clearServiceSkillCatalogCache: mockClearServiceSkillCatalogCache,
  getServiceSkillCatalog: mockGetServiceSkillCatalog,
  subscribeServiceSkillCatalogChanged: mockSubscribeServiceSkillCatalogChanged,
}));

vi.mock("@/lib/api/logs", () => ({
  getLogs: mockGetLogs,
  getPersistedLogsTail: mockGetPersistedLogsTail,
}));

vi.mock("@/lib/api/serverRuntime", () => ({
  getServerDiagnostics: mockGetServerDiagnostics,
  getLogStorageDiagnostics: mockGetLogStorageDiagnostics,
  exportDiagnosticsTrace: mockExportDiagnosticsTrace,
  exportSupportBundle: mockExportSupportBundle,
  listDiagnosticsTraces: mockListDiagnosticsTraces,
  readDiagnosticsTrace: mockReadDiagnosticsTrace,
  getWindowsStartupDiagnostics: mockGetWindowsStartupDiagnostics,
}));

vi.mock("@/lib/crashDiagnostic", () => ({
  buildCrashDiagnosticPayload: mockBuildCrashDiagnosticPayload,
  clearCrashDiagnosticHistory: mockClearCrashDiagnosticHistory,
  collectRuntimeSnapshotForDiagnostic: mockCollectRuntimeSnapshotForDiagnostic,
  collectGeneralWorkbenchDocumentStateForDiagnostic:
    mockCollectGeneralWorkbenchDocumentStateForDiagnostic,
  CLEAR_CRASH_DIAGNOSTIC_HISTORY_CONFIRM_TEXT: "确认清空诊断信息？",
  copyTextToClipboard: mockCopyTextToClipboard,
  copyCrashDiagnosticJsonToClipboard: mockCopyCrashDiagnosticJsonToClipboard,
  copyCrashDiagnosticToClipboard: mockCopyCrashDiagnosticToClipboard,
  exportCrashDiagnosticToJson: mockExportCrashDiagnosticToJson,
  isClipboardPermissionDeniedError: mockIsClipboardPermissionDeniedError,
  normalizeCrashReportingConfig: mockNormalizeCrashReportingConfig,
  openCrashDiagnosticDownloadDirectory:
    mockOpenCrashDiagnosticDownloadDirectory,
}));

vi.mock("@/lib/serviceSkillCatalogBootstrap", () => ({
  emitServiceSkillCatalogBootstrap: mockEmitServiceSkillCatalogBootstrap,
  extractServiceSkillCatalogFromBootstrapPayload:
    mockExtractServiceSkillCatalogFromBootstrapPayload,
}));

vi.mock("../shared/ClipboardPermissionGuideCard", () => ({
  ClipboardPermissionGuideCard: () => (
    <div>Clipboard permission card placeholder</div>
  ),
}));

vi.mock("../shared/WorkspaceRepairHistoryCard", () => ({
  WorkspaceRepairHistoryCard: () => <div>Repair history card placeholder</div>,
}));

import { DeveloperSettings } from ".";
import {
  clearAgentUiPerformanceMetrics,
  getAgentUiPerformanceMetrics,
  recordAgentUiPerformanceMetric,
} from "@/lib/agentUiPerformanceMetrics";
import {
  clearAgentUiPerformanceTraceHistory,
  listAgentUiPerformanceTraceHistory,
} from "@/lib/agentUiPerformanceTraceHistory";
import { clearClawTraceRegressionAlertChannel } from "@/lib/trace/clawTraceRegressionAlertChannel";
import { clearClawTraceRegressionTrend } from "@/lib/trace/clawTraceRegressionTrend";

interface Mounted {
  container: HTMLDivElement;
  root: Root;
}

const remoteCatalog = {
  version: "tenant-2026-03-24",
  tenantId: "tenant-demo",
  syncedAt: "2026-03-24T12:00:00.000Z",
  items: [
    {
      id: "tenant-skill-1",
      title: "租户技能 1",
    },
    {
      id: "tenant-skill-2",
      title: "租户技能 2",
    },
  ],
};

const seededCatalog = {
  version: "client-seed-2026-03-24",
  tenantId: "local-seeded",
  syncedAt: "2026-03-24T00:00:00.000Z",
  items: [
    {
      id: "seeded-skill-1",
      title: "Seeded 技能 1",
    },
  ],
};

const mounted: Mounted[] = [];

function renderComponent(): HTMLDivElement {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(<DeveloperSettings />);
  });

  mounted.push({ container, root });
  return container;
}

async function flushEffects(times = 8) {
  await act(async () => {
    for (let index = 0; index < times; index += 1) {
      await Promise.resolve();
    }
  });
}

function findButton(container: HTMLElement, text: string): HTMLButtonElement {
  const button = Array.from(container.querySelectorAll("button")).find((item) =>
    item.textContent?.includes(text),
  );
  if (!button) {
    throw new Error(`未找到按钮: ${text}`);
  }
  return button as HTMLButtonElement;
}

function findSwitch(
  container: HTMLElement,
  ariaLabel: string,
): HTMLButtonElement {
  const button = container.querySelector<HTMLButtonElement>(
    `button[aria-label="${ariaLabel}"]`,
  );
  if (!button) {
    throw new Error(`未找到开关: ${ariaLabel}`);
  }
  return button;
}

function findSelect(
  container: HTMLElement,
  ariaLabel: string,
): HTMLSelectElement {
  const select = container.querySelector<HTMLSelectElement>(
    `select[aria-label="${ariaLabel}"]`,
  );
  if (!select) {
    throw new Error(`未找到下拉框: ${ariaLabel}`);
  }
  return select;
}

function findInput(
  container: HTMLElement,
  ariaLabel: string,
): HTMLInputElement {
  const input = container.querySelector<HTMLInputElement>(
    `input[aria-label="${ariaLabel}"]`,
  );
  if (!input) {
    throw new Error(`未找到输入框: ${ariaLabel}`);
  }
  return input;
}

function findByTestId<TElement extends HTMLElement>(
  container: HTMLElement,
  testId: string,
): TElement {
  const element = container.querySelector<TElement>(
    `[data-testid="${testId}"]`,
  );
  if (!element) {
    throw new Error(`未找到测试元素: ${testId}`);
  }
  return element;
}

function findTextarea(
  container: HTMLElement,
  ariaLabel: string,
): HTMLTextAreaElement {
  const textarea = container.querySelector<HTMLTextAreaElement>(
    `textarea[aria-label="${ariaLabel}"]`,
  );
  if (!textarea) {
    throw new Error(`未找到输入框: ${ariaLabel}`);
  }
  return textarea;
}

async function clickButton(button: HTMLButtonElement) {
  await act(async () => {
    button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await flushEffects();
  });
}

async function setSelectValue(select: HTMLSelectElement, value: string) {
  const prototype = Object.getPrototypeOf(select) as HTMLSelectElement;
  const descriptor = Object.getOwnPropertyDescriptor(prototype, "value");
  const setValue = descriptor?.set;
  if (!setValue) {
    throw new Error("未找到 select value setter");
  }

  await act(async () => {
    setValue.call(select, value);
    select.dispatchEvent(new Event("change", { bubbles: true }));
    await flushEffects();
  });
}

async function setInputValue(input: HTMLInputElement, value: string) {
  const prototype = Object.getPrototypeOf(input) as HTMLInputElement;
  const descriptor = Object.getOwnPropertyDescriptor(prototype, "value");
  const setValue = descriptor?.set;
  if (!setValue) {
    throw new Error("未找到 input value setter");
  }

  await act(async () => {
    setValue.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    await flushEffects();
  });
}

async function waitForLazyPanels() {
  await flushEffects();
  if (typeof vi.dynamicImportSettled === "function") {
    await vi.dynamicImportSettled();
  }
  await flushEffects();
}

async function inputTextarea(textarea: HTMLTextAreaElement, value: string) {
  const prototype = Object.getPrototypeOf(textarea) as HTMLTextAreaElement;
  const descriptor = Object.getOwnPropertyDescriptor(prototype, "value");
  const setValue = descriptor?.set;
  if (!setValue) {
    throw new Error("未找到 textarea value setter");
  }

  await act(async () => {
    setValue.call(textarea, value);
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    await flushEffects();
  });
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  vi.clearAllMocks();
  clearAgentUiPerformanceTraceHistory();
  clearClawTraceRegressionAlertChannel();
  clearClawTraceRegressionTrend();
  await changeLimeLocale("en-US");

  mockUseComponentDebug.mockReturnValue({
    enabled: false,
    setEnabled: vi.fn(),
    componentInfo: null,
    showComponentInfo: vi.fn(),
    hideComponentInfo: vi.fn(),
  });

  mockGetConfig.mockResolvedValue({
    developer: {
      workspace_harness_enabled: false,
    },
    crash_reporting: {
      enabled: true,
      dsn: null,
      environment: "production",
      sample_rate: 1,
      send_pii: false,
    },
  });
  mockGetLogs.mockResolvedValue([{ level: "error", message: "boom" }]);
  mockSaveConfig.mockResolvedValue(undefined);
  mockGetPersistedLogsTail.mockResolvedValue([
    { level: "info", message: "ok" },
  ]);
  mockGetServerDiagnostics.mockResolvedValue({ ok: true });
  mockGetLogStorageDiagnostics.mockResolvedValue({ ok: true });
  mockListDiagnosticsTraces.mockResolvedValue({
    available: true,
    trace_root: null,
    traces: [
      {
        session_id: "session-a",
        trace_id: "trace-a",
        path: "sessions/session_session-a/trace_trace-a.jsonl",
        size_bytes: 128,
        event_count: 1,
      },
    ],
    redaction: {
      mode: "summary_only",
      raw_agent_event_payload: false,
      prompt_text: false,
      provider_payload: false,
    },
  });
  mockReadDiagnosticsTrace.mockResolvedValue({
    available: true,
    trace: {
      session_id: "session-a",
      trace_id: "trace-a",
      path: "sessions/session_session-a/trace_trace-a.jsonl",
      size_bytes: 128,
      event_count: 1,
    },
    events: [
      {
        schema_version: 1,
        seq: 1,
        wall_time_unix_ms: 1780000000000,
        trace_id: "trace-a",
        run_id: null,
        request_id: null,
        session_id: "session-a",
        thread_id: null,
        turn_id: null,
        event_id: "evt-a",
        event_sequence: 1,
        event_type: "message.delta",
        checkpoint: "app_server.message_delta.emitted",
        metrics: { text_chars: 4 },
        redaction: {
          mode: "summary_only",
          raw_agent_event_payload: false,
          prompt_text: false,
          provider_payload: false,
        },
      },
    ],
    redaction: {
      mode: "summary_only",
      raw_agent_event_payload: false,
      prompt_text: false,
      provider_payload: false,
    },
  });
  mockExportDiagnosticsTrace.mockResolvedValue({
    available: true,
    exported: true,
    trace: {
      session_id: "session-a",
      trace_id: "trace-a",
      path: "sessions/session_session-a/trace_trace-a.jsonl",
      size_bytes: 128,
      event_count: 1,
    },
    bundle_path: "/tmp/claw-trace-session-a-trace-a.zip",
    output_directory: "/tmp",
    generated_at: "2026-06-27T00:00:00.000Z",
    included_sections: [
      "meta/manifest.json",
      "meta/trace-summary.json",
      "trace/events.jsonl",
      "README.txt",
    ],
    omitted_sections: [
      "raw AgentEvent payload",
      "prompt text",
      "provider request/response payload",
      "assistant delta text",
      "unparsed raw JSONL bytes",
    ],
    redaction: {
      mode: "summary_only",
      raw_agent_event_payload: false,
      prompt_text: false,
      provider_payload: false,
    },
  });
  mockExportSupportBundle.mockResolvedValue({
    bundle_path: "/tmp/Lime-Support.zip",
    output_directory: "/tmp",
    generated_at: "2026-06-27T00:00:00.000Z",
    platform: "darwin",
    included_sections: [
      "meta/manifest.json",
      "trace-export/claw-trace-session-a-trace-a.zip",
    ],
    omitted_sections: ["raw trace event JSONL 原始字节"],
  });
  mockGetWindowsStartupDiagnostics.mockResolvedValue({ ok: true });

  mockCollectRuntimeSnapshotForDiagnostic.mockResolvedValue({
    runtimeSnapshot: { summary: "runtime" },
    collectionNotes: ["note-a"],
  });
  mockCollectGeneralWorkbenchDocumentStateForDiagnostic.mockResolvedValue({
    documentId: "doc-1",
  });
  mockNormalizeCrashReportingConfig.mockImplementation((config) => config);
  mockBuildCrashDiagnosticPayload.mockReturnValue({ payload: "diagnostic" });
  mockCopyTextToClipboard.mockResolvedValue(undefined);
  mockCopyCrashDiagnosticToClipboard.mockResolvedValue(undefined);
  mockCopyCrashDiagnosticJsonToClipboard.mockResolvedValue(undefined);
  mockExportCrashDiagnosticToJson.mockReturnValue({
    fileName: "diagnostic.json",
    locationHint: "/tmp",
  });
  mockOpenCrashDiagnosticDownloadDirectory.mockResolvedValue({
    openedPath: "/tmp",
  });
  mockClearCrashDiagnosticHistory.mockResolvedValue(undefined);
  mockIsClipboardPermissionDeniedError.mockReturnValue(false);
  mockGetServiceSkillCatalog.mockResolvedValue(remoteCatalog);
  mockSubscribeServiceSkillCatalogChanged.mockImplementation(() => vi.fn());
  mockClearServiceSkillCatalogCache.mockImplementation(() => undefined);
  mockExtractServiceSkillCatalogFromBootstrapPayload.mockReturnValue(
    remoteCatalog,
  );
});

afterEach(async () => {
  while (mounted.length > 0) {
    const target = mounted.pop();
    if (!target) {
      break;
    }

    act(() => {
      target.root.unmount();
    });
    target.container.remove();
  }

  vi.clearAllMocks();
  clearAgentUiPerformanceMetrics();
  clearAgentUiPerformanceTraceHistory();
  clearClawTraceRegressionAlertChannel();
  clearClawTraceRegressionTrend();
  await changeLimeLocale("zh-CN");
});

describe("DeveloperSettings", () => {
  it("应渲染清理后的Developer页和必要分区", async () => {
    const container = renderComponent();
    await flushEffects();

    const text = container.textContent ?? "";
    expect(text).toContain("Developer");
    expect(text).toContain(
      "Turn these on while troubleshooting, then turn them off again.",
    );
    expect(text).toContain("Debug Toggles");
    expect(text).toContain("Workspace debug info");
    expect(text).toContain("Claw stream trace");
    expect(text).toContain("Trace level");
    expect(text).toContain("Sample rate");
    expect(text).toContain("Regression alert channel");
    expect(text).toContain("Desktop alert notification");
    expect(text).toContain("Copy Trace summary");
    expect(text).toContain("Clear in-memory summary");
    expect(text).toContain("Trace history");
    expect(text).toContain("Save Trace snapshot");
    expect(text).toContain("Copy Trace history");
    expect(text).toContain("Clear Trace history");
    expect(text).toContain("Regression evidence");
    expect(text).toContain("Regression alert");
    expect(text).toContain("Alert channel: 0 saved alerts");
    expect(text).toContain("Save regression evidence");
    expect(text).toContain("Copy regression trend");
    expect(text).toContain("Clear regression trend");
    expect(text).toContain("Copy alert channel");
    expect(text).toContain("Clear alert channel");
    expect(text).toContain("App Server Trace");
    expect(text).toContain("Copy App Server Trace list");
    expect(text).toContain("Copy latest App Server Trace");
    expect(text).toContain("Export latest Trace");
    expect(text).toContain("Export support bundle with Trace");
    expect(text).toContain("Load Trace timeline");
    expect(text).toContain("Service Skill Catalog Debugging");
    expect(text).toContain("Component view debug");
    expect(text).toContain("Diagnostic Logs");
    expect(text).toContain("Workspace Repair History");
    expect(text).toContain("Repair history card placeholder");
    expect(text).not.toContain("settings.developer");
  });

  it("应移除开发页冗余说明卡片和提示噪音", async () => {
    renderComponent();
    await flushEffects();

    const text = document.body.textContent ?? "";
    expect(text).not.toContain("诊断建议");
    expect(text).not.toContain("动作清单");
    expect(text).not.toContain("首屏说明已收纳");
    expect(text).not.toContain("页面结构问题");
    expect(text).not.toContain("闪退或启动异常");
    expect(text).not.toContain("自愈链路核对");
    expect(text).not.toContain(
      "首屏先保留处理工作台、组件调试和诊断动作，目录联调、自愈记录与权限卡片按需加载，减少进入设置后的等待感。",
    );
    expect(
      document.body.querySelector(
        "button[aria-label='Developer settings hero details']",
      ),
    ).toBeNull();
    expect(
      document.body.querySelector(
        "button[aria-label='Workspace debug info details']",
      ),
    ).toBeNull();
  });

  it("切换组件Debug Toggles后应调用 setEnabled", async () => {
    const setEnabled = vi.fn();
    mockUseComponentDebug.mockReturnValue({
      enabled: false,
      setEnabled,
      componentInfo: null,
      showComponentInfo: vi.fn(),
      hideComponentInfo: vi.fn(),
    });

    const container = renderComponent();
    await clickButton(findSwitch(container, "Toggle component view debug"));

    expect(setEnabled).toHaveBeenCalledTimes(1);
    expect(setEnabled).toHaveBeenCalledWith(true);
  });

  it("切换处理工作台开关后应保存 developer.workspace_harness_enabled", async () => {
    const container = renderComponent();
    await flushEffects();

    await clickButton(findSwitch(container, "Toggle workspace debug info"));

    expect(mockSaveConfig).toHaveBeenCalledTimes(1);
    expect(mockSaveConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        developer: expect.objectContaining({
          workspace_harness_enabled: true,
        }),
      }),
    );
    expect(container.textContent).toContain(
      "Workspace debug collection is on. Tool inventory and environment summaries load when Harness opens.",
    );
  });

  it("切换 Claw Trace 开关后应保存独立 developer.claw_trace.enabled", async () => {
    const container = renderComponent();
    await flushEffects();

    await clickButton(findSwitch(container, "Toggle Claw stream trace"));

    expect(mockSaveConfig).toHaveBeenCalledTimes(1);
    expect(mockSaveConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        developer: expect.objectContaining({
          workspace_harness_enabled: false,
          claw_trace: expect.objectContaining({
            enabled: true,
            level: "summary",
            sample_rate: 1,
          }),
        }),
      }),
    );
    expect(container.textContent).toContain(
      "Claw trace is on. New turns record stream checkpoints; diagnostic exports include only latency summaries.",
    );
  });

  it("切换 Claw Trace 回归告警通道后应保存独立 developer.claw_trace.alert_enabled", async () => {
    const container = renderComponent();
    await flushEffects();

    await clickButton(
      findSwitch(container, "Toggle Claw trace regression alert channel"),
    );

    expect(mockSaveConfig).toHaveBeenCalledTimes(1);
    expect(mockSaveConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        developer: expect.objectContaining({
          workspace_harness_enabled: false,
          claw_trace: expect.objectContaining({
            alert_enabled: true,
            enabled: false,
            level: "summary",
            sample_rate: 1,
          }),
        }),
      }),
    );
    expect(container.textContent).toContain("Claw trace settings saved");
  });

  it("切换 Claw Trace 桌面告警通知后应保存独立 developer.claw_trace.alert_notification_enabled", async () => {
    const container = renderComponent();
    await flushEffects();

    await clickButton(
      findSwitch(container, "Toggle Claw trace desktop alert notification"),
    );

    expect(mockSaveConfig).toHaveBeenCalledTimes(1);
    expect(mockSaveConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        developer: expect.objectContaining({
          workspace_harness_enabled: false,
          claw_trace: expect.objectContaining({
            alert_enabled: false,
            alert_notification_enabled: true,
            enabled: false,
            level: "summary",
            sample_rate: 1,
          }),
        }),
      }),
    );
    expect(container.textContent).toContain("Claw trace settings saved");
  });

  it("修改 Claw Trace 级别和采样率后应保存到独立配置", async () => {
    mockGetConfig.mockResolvedValueOnce({
      developer: {
        workspace_harness_enabled: false,
        claw_trace: {
          enabled: true,
          level: "summary",
          sample_rate: 1,
        },
      },
      crash_reporting: {
        enabled: true,
        dsn: null,
        environment: "production",
        sample_rate: 1,
        send_pii: false,
      },
    });

    const container = renderComponent();
    await flushEffects();

    await setSelectValue(findSelect(container, "Claw trace level"), "debug");
    expect(mockSaveConfig).toHaveBeenLastCalledWith(
      expect.objectContaining({
        developer: expect.objectContaining({
          workspace_harness_enabled: false,
          claw_trace: expect.objectContaining({
            enabled: true,
            level: "debug",
            sample_rate: 1,
          }),
        }),
      }),
    );

    await setInputValue(findInput(container, "Claw trace sample rate"), "0.5");
    expect(mockSaveConfig).toHaveBeenLastCalledWith(
      expect.objectContaining({
        developer: expect.objectContaining({
          workspace_harness_enabled: false,
          claw_trace: expect.objectContaining({
            enabled: true,
            level: "debug",
            sample_rate: 0.5,
          }),
        }),
      }),
    );
  });

  it("复制和清空 Claw Trace 摘要时不导出 raw entries", async () => {
    recordAgentUiPerformanceMetric("agentStream.firstTextDelta", {
      sessionId: "session-a",
      providerWaitMs: 1200,
      raw_provider_payload: "secret-provider-payload",
    });

    const container = renderComponent();
    await flushEffects();

    await clickButton(findButton(container, "Copy Trace summary"));

    expect(mockCopyTextToClipboard).toHaveBeenCalledTimes(1);
    const copiedText = mockCopyTextToClipboard.mock.calls[0]?.[0] as string;
    expect(copiedText).toContain('"entry_count": 1');
    expect(copiedText).toContain('"providerWaitMs": 1200');
    expect(copiedText).not.toContain("secret-provider-payload");
    expect(container.textContent).toContain("Claw trace summary copied");

    await clickButton(findButton(container, "Clear in-memory summary"));

    expect(getAgentUiPerformanceMetrics()).toHaveLength(0);
    expect(container.textContent).toContain(
      "In-memory Claw trace summary cleared",
    );
  });

  it("保存、复制和清空 Claw Trace 历史时只导出 compact summary", async () => {
    recordAgentUiPerformanceMetric("agentStream.providerTrace", {
      providerWaitMs: 1200,
      raw_provider_payload: "secret-provider-payload",
      sessionId: "session-a",
      stage: "first_text_delta_received",
    });
    recordAgentUiPerformanceMetric("agentStream.firstTextDelta", {
      rendererEventReceivedDeltaMs: 4,
      serverToRendererDeltaMs: 18,
      sessionId: "session-a",
    });
    recordAgentUiPerformanceMetric("agentStream.firstTextPaint", {
      clientLocalOutputDeltaMs: 80,
      sessionId: "session-a",
    });

    const container = renderComponent();
    await flushEffects();

    await clickButton(findButton(container, "Save Trace snapshot"));

    expect(listAgentUiPerformanceTraceHistory()).toHaveLength(1);
    expect(container.textContent).toContain("Claw trace snapshot saved");
    expect(container.textContent).toContain("Trace history: 1/20 snapshots");

    await clickButton(findButton(container, "Copy Trace history"));

    expect(mockCopyTextToClipboard).toHaveBeenCalledTimes(1);
    const copiedText = mockCopyTextToClipboard.mock.calls[0]?.[0] as string;
    expect(copiedText).toContain('"mode": "compact_summary_only"');
    expect(copiedText).toContain('"providerWaitMs": 1200');
    expect(copiedText).not.toContain("secret-provider-payload");
    expect(copiedText).not.toContain("raw_provider_payload");
    expect(container.textContent).toContain("Claw trace history copied");

    await clickButton(findButton(container, "Clear Trace history"));

    expect(listAgentUiPerformanceTraceHistory()).toHaveLength(0);
    expect(container.textContent).toContain("Claw trace history cleared");
  });

  it("应展示 compact Trace baseline 对比入口且不暴露 raw payload", async () => {
    recordAgentUiPerformanceMetric("agentStream.providerTrace", {
      providerWaitMs: 900,
      raw_provider_payload: "secret-provider-payload",
      sessionId: "session-a",
      stage: "first_text_delta_received",
    });
    recordAgentUiPerformanceMetric("agentStream.firstTextDelta", {
      rendererEventReceivedDeltaMs: 4,
      serverToRendererDeltaMs: 18,
      sessionId: "session-a",
    });
    recordAgentUiPerformanceMetric("agentStream.firstTextPaint", {
      clientLocalOutputDeltaMs: 80,
      sessionId: "session-a",
    });

    const container = renderComponent();
    await flushEffects();

    const initialBaseline = findByTestId<HTMLDivElement>(
      container,
      "claw-trace-baseline-comparison",
    );
    expect(initialBaseline.textContent).toContain("Baseline compare");
    expect(initialBaseline.textContent).toContain("No baseline");
    expect(initialBaseline.textContent).not.toContain(
      "secret-provider-payload",
    );
    const initialRegression = findByTestId<HTMLDivElement>(
      container,
      "claw-trace-regression-report",
    );
    expect(initialRegression.textContent).toContain("Regression evidence");
    expect(initialRegression.textContent).toContain("No evidence");
    expect(initialRegression.textContent).toContain("Regression alert: Off");

    await clickButton(findButton(container, "Save Trace snapshot"));

    const baseline = findByTestId<HTMLDivElement>(
      container,
      "claw-trace-baseline-comparison",
    );
    const text = baseline.textContent ?? "";
    expect(text).toContain("Baseline compare");
    expect(text).toContain("Stable");
    expect(text).toContain("Window: 1 snapshots · oldest retained baseline");
    expect(text).toContain("Provider: 900 ms / 0 ms");
    expect(text).toContain("Client local: 80 ms / 0 ms");
    expect(text).not.toContain("raw_provider_payload");
    expect(text).not.toContain("secret-provider-payload");

    const regression = findByTestId<HTMLDivElement>(
      container,
      "claw-trace-regression-report",
    );
    expect(regression.textContent).toContain("Regression evidence");
    expect(regression.textContent).toContain("Stable");
    expect(regression.textContent).toContain(
      "Window: 1 compact snapshots · 0 App Server traces",
    );
    expect(regression.textContent).toContain(
      "Trend: 0 saved reports · latest No evidence",
    );
    expect(regression.textContent).toContain("Regression alert: Off");
    expect(regression.textContent).toContain(
      "Enable the regression alert channel",
    );
    expect(regression.textContent).toContain("Provider: 900 ms / 0 ms");

    await clickButton(findButton(container, "Save regression evidence"));

    expect(container.textContent).toContain(
      "Claw trace regression trend saved",
    );
    expect(regression.textContent).toContain(
      "Trend: 1 saved reports · latest Stable",
    );
    expect(regression.textContent).toContain("Regression alert: Off");

    await clickButton(findButton(container, "Copy regression trend"));

    expect(mockCopyTextToClipboard).toHaveBeenCalledTimes(1);
    const copiedTrendText = mockCopyTextToClipboard.mock
      .calls[0]?.[0] as string;
    expect(copiedTrendText).toContain("summary_only_regression_report");
    expect(copiedTrendText).toContain("providerWaitMs");
    expect(copiedTrendText).not.toContain("raw_provider_payload");
    expect(copiedTrendText).not.toContain("secret-provider-payload");

    await clickButton(findButton(container, "Clear regression trend"));

    expect(container.textContent).toContain(
      "Claw trace regression trend cleared",
    );
    expect(regression.textContent).toContain(
      "Trend: 0 saved reports · latest No evidence",
    );
  });

  it("复制 App Server Claw Trace 时应走 summary-only current 诊断 API", async () => {
    const container = renderComponent();
    await flushEffects();

    await clickButton(findButton(container, "Copy App Server Trace list"));

    expect(mockListDiagnosticsTraces).toHaveBeenCalledWith({ limit: 20 });
    expect(mockCopyTextToClipboard).toHaveBeenCalledTimes(1);
    const copiedListText = mockCopyTextToClipboard.mock.calls[0]?.[0] as string;
    expect(copiedListText).toContain('"mode": "summary_only"');
    expect(copiedListText).toContain('"trace_id": "trace-a"');
    expect(copiedListText).not.toContain("raw_provider_payload");
    expect(copiedListText).not.toContain('"provider_payload": true');
    expect(container.textContent).toContain("Claw trace list copied");

    await clickButton(findButton(container, "Copy latest App Server Trace"));

    expect(mockListDiagnosticsTraces).toHaveBeenLastCalledWith({ limit: 1 });
    expect(mockReadDiagnosticsTrace).toHaveBeenCalledWith({
      session_id: "session-a",
      trace_id: "trace-a",
      max_events: 200,
    });
    expect(mockCopyTextToClipboard).toHaveBeenCalledTimes(2);
    const copiedTraceText = mockCopyTextToClipboard.mock
      .calls[1]?.[0] as string;
    expect(copiedTraceText).toContain("app_server.message_delta.emitted");
    expect(copiedTraceText).toContain('"raw_agent_event_payload": false');
    expect(copiedTraceText).not.toContain('provider_payload": true');
    expect(container.textContent).toContain("Latest Claw trace copied");
  });

  it("导出最近 App Server Claw Trace 时应走显式 export current 诊断 API", async () => {
    const container = renderComponent();
    await flushEffects();

    await clickButton(findButton(container, "Export latest Trace"));

    expect(mockListDiagnosticsTraces).toHaveBeenCalledWith({ limit: 1 });
    expect(mockExportDiagnosticsTrace).toHaveBeenCalledWith({
      session_id: "session-a",
      trace_id: "trace-a",
    });
    expect(mockCopyTextToClipboard).not.toHaveBeenCalled();
    expect(container.textContent).toContain("Latest Claw trace exported");
    expect(container.textContent).toContain(
      "/tmp/claw-trace-session-a-trace-a.zip",
    );
  });

  it("导出附带最近 Claw Trace 的支持包时应走 current 支持包 API", async () => {
    const container = renderComponent();
    await flushEffects();

    await clickButton(
      findButton(container, "Export support bundle with Trace"),
    );

    expect(mockListDiagnosticsTraces).toHaveBeenCalledWith({ limit: 1 });
    expect(mockExportSupportBundle).toHaveBeenCalledWith({
      include_trace_export: {
        session_id: "session-a",
        trace_id: "trace-a",
      },
    });
    expect(mockExportDiagnosticsTrace).not.toHaveBeenCalled();
    expect(mockCopyTextToClipboard).not.toHaveBeenCalled();
    expect(container.textContent).toContain(
      "Support bundle exported with latest Claw trace",
    );
    expect(container.textContent).toContain("/tmp/Lime-Support.zip");
  });

  it("加载 App Server Claw Trace timeline 时应展示 phase span 和事件顺序", async () => {
    mockGetConfig.mockResolvedValueOnce({
      developer: {
        workspace_harness_enabled: false,
        claw_trace: {
          alert_enabled: true,
          enabled: true,
          level: "summary",
          sample_rate: 1,
        },
      },
      crash_reporting: {
        enabled: true,
        dsn: null,
        environment: "production",
        sample_rate: 1,
        send_pii: false,
      },
    });
    mockListDiagnosticsTraces.mockResolvedValueOnce({
      available: true,
      trace_root: null,
      traces: [
        {
          session_id: "session-a",
          trace_id: "trace-a",
          path: "sessions/session_session-a/trace_trace-a.jsonl",
          size_bytes: 256,
          event_count: 4,
        },
        {
          session_id: "session-a",
          trace_id: "trace-previous",
          path: "sessions/session_session-a/trace_trace-previous.jsonl",
          size_bytes: 224,
          event_count: 4,
        },
        {
          session_id: "session-a",
          trace_id: "trace-oldest",
          path: "sessions/session_session-a/trace_trace-oldest.jsonl",
          size_bytes: 192,
          event_count: 4,
        },
      ],
      redaction: {
        mode: "summary_only",
        raw_agent_event_payload: false,
        prompt_text: false,
        provider_payload: false,
      },
    });
    mockReadDiagnosticsTrace.mockResolvedValueOnce({
      available: true,
      trace: {
        session_id: "session-a",
        trace_id: "trace-a",
        path: "sessions/session_session-a/trace_trace-a.jsonl",
        size_bytes: 256,
        event_count: 4,
      },
      events: [
        {
          schema_version: 1,
          seq: 1,
          wall_time_unix_ms: 1780000000000,
          trace_id: "trace-a",
          run_id: null,
          request_id: null,
          session_id: "session-a",
          thread_id: null,
          turn_id: null,
          event_id: "evt-provider-start",
          event_sequence: 1,
          event_type: "provider.request.started",
          checkpoint: "provider.request.started",
          metrics: {
            provider: "anthropic",
            provider_request_id: "req-provider-1",
            raw_provider_payload: { should: "drop" },
          },
          redaction: {
            mode: "summary_only",
            raw_agent_event_payload: false,
            prompt_text: false,
            provider_payload: false,
          },
        },
        {
          schema_version: 1,
          seq: 2,
          wall_time_unix_ms: 1780000000090,
          trace_id: "trace-a",
          run_id: null,
          request_id: null,
          session_id: "session-a",
          thread_id: null,
          turn_id: null,
          event_id: "evt-provider-first-text",
          event_sequence: 2,
          event_type: "provider.first_text_delta.received",
          checkpoint: "provider.first_text_delta.received",
          metrics: { text_chars: 4 },
          redaction: {
            mode: "summary_only",
            raw_agent_event_payload: false,
            prompt_text: false,
            provider_payload: false,
          },
        },
        {
          schema_version: 1,
          seq: 3,
          wall_time_unix_ms: 1780000000120,
          trace_id: "trace-a",
          run_id: null,
          request_id: null,
          session_id: "session-a",
          thread_id: null,
          turn_id: null,
          event_id: "evt-message-delta",
          event_sequence: 3,
          event_type: "message.delta",
          checkpoint: "app_server.message_delta.emitted",
          metrics: { text_chars: 4 },
          redaction: {
            mode: "summary_only",
            raw_agent_event_payload: false,
            prompt_text: false,
            provider_payload: false,
          },
        },
        {
          schema_version: 1,
          seq: 4,
          wall_time_unix_ms: 1780000000160,
          trace_id: "trace-a",
          run_id: null,
          request_id: null,
          session_id: "session-a",
          thread_id: null,
          turn_id: null,
          event_id: "evt-terminal",
          event_sequence: 4,
          event_type: "turn.completed",
          checkpoint: "app_server.turn.terminal",
          metrics: { status: "completed" },
          redaction: {
            mode: "summary_only",
            raw_agent_event_payload: false,
            prompt_text: false,
            provider_payload: false,
          },
        },
      ],
      redaction: {
        mode: "summary_only",
        raw_agent_event_payload: false,
        prompt_text: false,
        provider_payload: false,
      },
    });
    mockReadDiagnosticsTrace.mockResolvedValueOnce({
      available: true,
      trace: {
        session_id: "session-a",
        trace_id: "trace-oldest",
        path: "sessions/session_session-a/trace_trace-oldest.jsonl",
        size_bytes: 192,
        event_count: 4,
      },
      events: [
        {
          schema_version: 1,
          seq: 1,
          wall_time_unix_ms: 1780000000000,
          trace_id: "trace-oldest",
          run_id: null,
          request_id: null,
          session_id: "session-a",
          thread_id: null,
          turn_id: null,
          event_id: "evt-provider-start-baseline",
          event_sequence: 1,
          event_type: "provider.request.started",
          checkpoint: "provider.request.started",
          metrics: {
            provider: "anthropic",
            raw_provider_payload: { should: "drop-baseline" },
          },
          redaction: {
            mode: "summary_only",
            raw_agent_event_payload: false,
            prompt_text: false,
            provider_payload: false,
          },
        },
        {
          schema_version: 1,
          seq: 2,
          wall_time_unix_ms: 1780000000040,
          trace_id: "trace-oldest",
          run_id: null,
          request_id: null,
          session_id: "session-a",
          thread_id: null,
          turn_id: null,
          event_id: "evt-provider-first-text-baseline",
          event_sequence: 2,
          event_type: "provider.first_text_delta.received",
          checkpoint: "provider.first_text_delta.received",
          metrics: { text_chars: 4 },
          redaction: {
            mode: "summary_only",
            raw_agent_event_payload: false,
            prompt_text: false,
            provider_payload: false,
          },
        },
        {
          schema_version: 1,
          seq: 3,
          wall_time_unix_ms: 1780000000050,
          trace_id: "trace-oldest",
          run_id: null,
          request_id: null,
          session_id: "session-a",
          thread_id: null,
          turn_id: null,
          event_id: "evt-message-delta-baseline",
          event_sequence: 3,
          event_type: "message.delta",
          checkpoint: "app_server.message_delta.emitted",
          metrics: { text_chars: 4 },
          redaction: {
            mode: "summary_only",
            raw_agent_event_payload: false,
            prompt_text: false,
            provider_payload: false,
          },
        },
        {
          schema_version: 1,
          seq: 4,
          wall_time_unix_ms: 1780000000090,
          trace_id: "trace-oldest",
          run_id: null,
          request_id: null,
          session_id: "session-a",
          thread_id: null,
          turn_id: null,
          event_id: "evt-terminal-baseline",
          event_sequence: 4,
          event_type: "turn.completed",
          checkpoint: "app_server.turn.terminal",
          metrics: { status: "completed" },
          redaction: {
            mode: "summary_only",
            raw_agent_event_payload: false,
            prompt_text: false,
            provider_payload: false,
          },
        },
      ],
      redaction: {
        mode: "summary_only",
        raw_agent_event_payload: false,
        prompt_text: false,
        provider_payload: false,
      },
    });

    const container = renderComponent();
    await flushEffects();

    await clickButton(findButton(container, "Load Trace timeline"));

    expect(mockListDiagnosticsTraces).toHaveBeenCalledWith({ limit: 20 });
    expect(mockReadDiagnosticsTrace).toHaveBeenCalledWith({
      session_id: "session-a",
      trace_id: "trace-a",
      max_events: 500,
    });
    expect(mockReadDiagnosticsTrace).toHaveBeenCalledWith({
      session_id: "session-a",
      trace_id: "trace-oldest",
      max_events: 500,
    });
    expect(mockReadDiagnosticsTrace).not.toHaveBeenCalledWith({
      session_id: "session-a",
      trace_id: "trace-previous",
      max_events: 500,
    });
    const text = container.textContent ?? "";
    expect(text).toContain("Claw trace timeline loaded");
    expect(text).toContain("App Server Trace compare");
    expect(text).toContain(
      "Window: 3 traces · oldest retained baseline · latest trace-a",
    );
    expect(text).toContain("Regression evidence");
    expect(text).toContain("Focus: Provider / API");
    expect(text).toContain("Window: 0 compact snapshots · 3 App Server traces");
    expect(text).toContain("Regressed");
    expect(text).toContain("Regression alert: Watch");
    expect(text).toContain("Alert channel: 1 saved alerts · latest Watch");
    expect(text).toContain(
      "Provider / API · +50 ms · 1/1 recent reports · Current report regressed",
    );
    expect(text).toContain("Provider first text: 90 ms / +50 ms");
    expect(text).toContain("Provider / API: +50 ms");
    expect(text).toContain("Root duration: 160 ms / +70 ms");
    expect(text).toContain("Trace timeline");
    expect(text).toContain("4 events · 160 ms · summary_only");
    expect(text).toContain("Provider / API");
    expect(text).toContain("App Server");
    expect(text).toContain("Terminal");
    expect(text).toContain("Diagnostics");
    expect(text).toContain("Selected event");
    expect(text).toContain("provider_request_id=req-provider-1");
    expect(text).toContain("Provider / API · 90 ms · 0-90 ms");
    expect(text).toContain(
      "provider.request.started -> provider.first_text_delta.received",
    );
    expect(text).toContain("provider.first_text_delta.received");
    expect(text).toContain("app_server.message_delta.emitted");
    expect(text).toContain("+90 ms");
    expect(text).not.toContain("Missing Provider / API phase");
    expect(text).not.toContain("raw_provider_payload");
    expect(text).not.toContain("should");
    expect(text).not.toContain("drop-baseline");

    await clickButton(findButton(container, "Copy alert channel"));

    expect(mockCopyTextToClipboard).toHaveBeenCalledWith(
      expect.stringContaining('"mode": "summary_only_alert"'),
    );
    expect(mockCopyTextToClipboard).toHaveBeenCalledWith(
      expect.not.stringContaining("raw_provider_payload"),
    );
    expect(container.textContent).toContain("Claw trace alert channel copied");

    await clickButton(findButton(container, "Clear alert channel"));

    expect(container.textContent).toContain("Claw trace alert channel cleared");
    expect(container.textContent).toContain(
      "Alert channel: 0 saved alerts · no alert recorded",
    );

    await clickButton(
      findByTestId<HTMLButtonElement>(container, "claw-trace-span-app_server"),
    );
    const selectedSpan = findByTestId<HTMLDivElement>(
      container,
      "claw-trace-selected-span",
    );
    expect(selectedSpan.textContent).toContain("Selected span");
    expect(selectedSpan.textContent).toContain("App Server · 0 ms");
    expect(selectedSpan.textContent).toContain("120-120 ms · 1 events");

    const eventList = findByTestId<HTMLDivElement>(
      container,
      "claw-trace-timeline-events",
    );
    const detail = findByTestId<HTMLDivElement>(
      container,
      "claw-trace-timeline-detail",
    );
    expect(eventList.textContent).toContain("app_server.message_delta.emitted");
    expect(eventList.textContent).not.toContain(
      "provider.first_text_delta.received",
    );
    expect(detail.textContent).toContain("app_server.message_delta.emitted");
    expect(detail.textContent).not.toContain("provider_request_id");

    await clickButton(
      findByTestId<HTMLButtonElement>(
        container,
        "claw-trace-filter-app_server",
      ),
    );
    expect(eventList.textContent).toContain("app_server.message_delta.emitted");
    expect(eventList.textContent).not.toContain(
      "provider.first_text_delta.received",
    );
    expect(detail.textContent).toContain("app_server.message_delta.emitted");
    expect(detail.textContent).not.toContain("provider_request_id");

    await clickButton(
      findByTestId<HTMLButtonElement>(container, "claw-trace-filter-slow"),
    );
    expect(eventList.textContent).toContain(
      "provider.first_text_delta.received",
    );
    expect(eventList.textContent).not.toContain(
      "app_server.message_delta.emitted",
    );
    expect(detail.textContent).toContain("provider.first_text_delta.received");
  });

  it("点击Copy diagnostics后应构建并复制诊断载荷", async () => {
    const container = renderComponent();

    await clickButton(findButton(container, "Copy diagnostics"));
    await flushEffects();

    expect(mockCollectRuntimeSnapshotForDiagnostic).toHaveBeenCalledTimes(1);
    expect(mockBuildCrashDiagnosticPayload).toHaveBeenCalledTimes(1);
    expect(mockBuildCrashDiagnosticPayload).toHaveBeenCalledWith(
      expect.objectContaining({
        agentUiPerformanceSnapshot: expect.objectContaining({
          sessions: expect.any(Array),
        }),
      }),
    );
    expect(mockCopyCrashDiagnosticToClipboard).toHaveBeenCalledWith({
      payload: "diagnostic",
    });
    expect(container.textContent).toContain(
      "Diagnostics copied. You can send them directly to developers.",
    );
  });

  it("复制诊断因剪贴板权限失败时应显示权限指引", async () => {
    const consoleErrorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});
    mockCopyCrashDiagnosticToClipboard.mockRejectedValueOnce(
      new Error("clipboard denied"),
    );
    mockIsClipboardPermissionDeniedError.mockReturnValue(true);

    try {
      const container = renderComponent();

      await clickButton(findButton(container, "Copy diagnostics"));
      await flushEffects();

      expect(container.textContent).toContain(
        "Clipboard permission card placeholder",
      );
      expect(container.textContent).toContain("clipboard denied");
    } finally {
      consoleErrorSpy.mockRestore();
    }
  });

  it("点击Load Current Catalog后应把 serviceSkillCatalog 写入调试输入框", async () => {
    const container = renderComponent();
    await waitForLazyPanels();

    await clickButton(findButton(container, "Load Current Catalog"));
    await flushEffects();

    const textarea = findTextarea(
      container,
      "Service skill catalog debug input",
    );
    expect(textarea.value).toContain('"tenantId": "tenant-demo"');
    expect(container.textContent).toContain(
      "Current catalog written to debug editor",
    );
  });

  it("输入 JSON 后Inject via Event应调用 bootstrap 桥接", async () => {
    const container = renderComponent();
    await waitForLazyPanels();
    const textarea = findTextarea(
      container,
      "Service skill catalog debug input",
    );

    await inputTextarea(
      textarea,
      JSON.stringify(
        {
          serviceSkillCatalog: remoteCatalog,
        },
        null,
        2,
      ),
    );
    await clickButton(findButton(container, "Inject via Event"));
    await flushEffects();

    expect(
      mockExtractServiceSkillCatalogFromBootstrapPayload,
    ).toHaveBeenCalledWith(
      expect.objectContaining({
        serviceSkillCatalog: expect.objectContaining({
          tenantId: "tenant-demo",
        }),
      }),
    );
    expect(mockEmitServiceSkillCatalogBootstrap).toHaveBeenCalledWith(
      expect.objectContaining({
        serviceSkillCatalog: expect.objectContaining({
          tenantId: "tenant-demo",
        }),
      }),
    );
    expect(container.textContent).toContain(
      "Injected catalog through bootstrap event: 2 items",
    );
  });

  it("Clear Catalog Cache后应回退 seeded 目录并展示提示", async () => {
    mockGetServiceSkillCatalog.mockResolvedValueOnce(remoteCatalog);
    mockGetServiceSkillCatalog.mockResolvedValueOnce(seededCatalog);

    const container = renderComponent();
    await waitForLazyPanels();

    await clickButton(findButton(container, "Clear Catalog Cache"));
    await flushEffects();

    expect(mockClearServiceSkillCatalogCache).toHaveBeenCalledTimes(1);
    expect(container.textContent).toContain(
      "Remote catalog cache cleared; falling back to seeded: 1 items",
    );
  });

});
