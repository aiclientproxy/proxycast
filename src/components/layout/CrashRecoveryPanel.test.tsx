import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import {
  buildCrashRecoveryReloadUrl,
  finalizeCrashRecoveryAutoReload,
  finalizeModuleImportAutoReload,
  isModuleImportFailureErrorMessage,
  isReactFastRefreshHookFailureErrorMessage,
  prepareModuleImportAutoReload,
  prepareReactFastRefreshHookAutoReload,
  stripCrashRecoveryReloadUrl,
} from "./CrashRecoveryPanel.helpers";

vi.mock("@/lib/api/appConfig", () => ({
  getConfig: vi.fn(async () => null),
}));

vi.mock("@/lib/api/logs", () => ({
  getLogs: vi.fn(async () => []),
  getPersistedLogsTail: vi.fn(async () => []),
}));

vi.mock("@/lib/crashDiagnostic", () => ({
  buildCrashDiagnosticPayload: vi.fn(() => ({})),
  clearCrashDiagnosticHistory: vi.fn(async () => undefined),
  collectGeneralWorkbenchDocumentStateForDiagnostic: vi.fn(async () => null),
  CLEAR_CRASH_DIAGNOSTIC_HISTORY_CONFIRM_TEXT: "confirm",
  copyCrashDiagnosticJsonToClipboard: vi.fn(async () => undefined),
  copyCrashDiagnosticToClipboard: vi.fn(async () => undefined),
  exportCrashDiagnosticToJson: vi.fn(() => ({
    fileName: "diagnostic.json",
    locationHint: "Downloads",
  })),
  isClipboardPermissionDeniedError: vi.fn(() => false),
  normalizeCrashReportingConfig: vi.fn(() => null),
  openCrashDiagnosticDownloadDirectory: vi.fn(async () => ({
    openedPath: "/tmp",
  })),
}));

vi.mock("@/lib/api/project", () => ({
  getProjectByRootPath: vi.fn(async () => null),
  updateProject: vi.fn(async () => undefined),
}));

vi.mock(
  "@/components/settings-v2/system/shared/ClipboardPermissionGuideCard",
  () => ({
    ClipboardPermissionGuideCard: ({ className }: { className?: string }) => (
      <div className={className}>clipboard-guide</div>
    ),
  }),
);

vi.mock("@/lib/desktop-host/dialog", () => ({
  open: vi.fn(async () => null),
}));

vi.mock("@/components/workspace/services/runtimeAgentsGuideService", () => ({
  notifyProjectRuntimeAgentsGuide: vi.fn(),
}));

import { CrashRecoveryPanel } from "./CrashRecoveryPanel";

interface RenderResult {
  container: HTMLDivElement;
  root: Root;
}

const mounted: RenderResult[] = [];

function renderPanel(error: Error | null) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(
      <CrashRecoveryPanel error={error} componentStack="" onRetry={vi.fn()} />,
    );
  });

  const rendered = { container, root };
  mounted.push(rendered);
  return rendered;
}

describe("CrashRecoveryPanel", () => {
  beforeEach(async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    await changeLimeLocale("en-US");
  });

  afterEach(async () => {
    while (mounted.length > 0) {
      const target = mounted.pop();
      if (!target) {
        continue;
      }
      act(() => {
        target.root.unmount();
      });
      target.container.remove();
    }

    vi.clearAllMocks();
    vi.unstubAllGlobals();
    await changeLimeLocale("zh-CN");
  });

  it("应识别模块脚本导入失败错误", () => {
    expect(
      isModuleImportFailureErrorMessage("Importing a module script failed."),
    ).toBe(true);
    expect(
      isModuleImportFailureErrorMessage(
        "Failed to fetch dynamically imported module: /src/app.tsx",
      ),
    ).toBe(true);
    expect(isModuleImportFailureErrorMessage("Random render error")).toBe(
      false,
    );
  });

  it("应识别 React Fast Refresh stale hook 队列错误", () => {
    expect(
      isReactFastRefreshHookFailureErrorMessage(
        "Should have a queue. This is likely a bug in React. Please file an issue.",
      ),
    ).toBe(true);
    expect(
      isReactFastRefreshHookFailureErrorMessage(
        "Rendered fewer hooks than expected. This may be caused by an accidental early return statement.",
      ),
    ).toBe(true);
    expect(
      isReactFastRefreshHookFailureErrorMessage(
        "Rendered more hooks than during the previous render.",
      ),
    ).toBe(true);
    expect(
      isReactFastRefreshHookFailureErrorMessage("Random render error"),
    ).toBe(false);
  });

  it("应为强制刷新资源构造带缓存刷新参数的地址", () => {
    expect(
      buildCrashRecoveryReloadUrl("http://127.0.0.1:1420/settings", "123456"),
    ).toBe("http://127.0.0.1:1420/settings?__lime_resource_reload=123456");

    const reloadUrl = buildCrashRecoveryReloadUrl(
      "http://127.0.0.1:1420/settings?tab=providers",
      "654321",
    );
    expect(reloadUrl).toContain("tab=providers");
    expect(reloadUrl).toContain("__lime_resource_reload=654321");
  });

  it("应在同一页面同一版本下只允许自动强制刷新一次", () => {
    const storage = {
      state: new Map<string, string>(),
      getItem(key: string) {
        return this.state.get(key) ?? null;
      },
      setItem(key: string, value: string) {
        this.state.set(key, value);
      },
      removeItem(key: string) {
        this.state.delete(key);
      },
    };

    const firstReloadUrl = prepareModuleImportAutoReload(
      "http://127.0.0.1:1420/agent?tab=chat",
      "1.19.0",
      storage,
    );
    const secondReloadUrl = prepareModuleImportAutoReload(
      "http://127.0.0.1:1420/agent?tab=chat",
      "1.19.0",
      storage,
    );

    expect(firstReloadUrl).toContain("__lime_resource_reload=");
    expect(secondReloadUrl).toBeNull();
  });

  it("成功启动后应移除自动刷新标记并清理 URL 参数", () => {
    const storage = {
      state: new Map<string, string>(),
      getItem(key: string) {
        return this.state.get(key) ?? null;
      },
      setItem(key: string, value: string) {
        this.state.set(key, value);
      },
      removeItem(key: string) {
        this.state.delete(key);
      },
    };
    const replaceState = vi.fn();
    const reloadUrl = prepareModuleImportAutoReload(
      "http://127.0.0.1:1420/settings?tab=providers",
      "1.19.0",
      storage,
    );

    expect(reloadUrl).toContain("__lime_resource_reload=");
    finalizeModuleImportAutoReload(reloadUrl!, "1.19.0", storage, {
      replaceState,
    });

    expect(replaceState).toHaveBeenCalledWith(
      null,
      "",
      stripCrashRecoveryReloadUrl(reloadUrl!),
    );
    expect(
      prepareModuleImportAutoReload(
        "http://127.0.0.1:1420/settings?tab=providers",
        "1.19.0",
        storage,
      ),
    ).toContain("__lime_resource_reload=");
  });

  it("成功启动后应同时移除 React hook 自动刷新标记", () => {
    const storage = {
      state: new Map<string, string>(),
      getItem(key: string) {
        return this.state.get(key) ?? null;
      },
      setItem(key: string, value: string) {
        this.state.set(key, value);
      },
      removeItem(key: string) {
        this.state.delete(key);
      },
    };
    const replaceState = vi.fn();
    const reloadUrl = prepareReactFastRefreshHookAutoReload(
      "http://127.0.0.1:1420/?tab=agent",
      "1.38.0",
      storage,
    );

    expect(reloadUrl).toContain("__lime_resource_reload=");
    expect(
      prepareReactFastRefreshHookAutoReload(
        "http://127.0.0.1:1420/?tab=agent",
        "1.38.0",
        storage,
      ),
    ).toBeNull();

    finalizeCrashRecoveryAutoReload(reloadUrl!, "1.38.0", storage, {
      replaceState,
    });

    expect(replaceState).toHaveBeenCalledWith(
      null,
      "",
      stripCrashRecoveryReloadUrl(reloadUrl!),
    );
    expect(
      prepareReactFastRefreshHookAutoReload(
        "http://127.0.0.1:1420/?tab=agent",
        "1.38.0",
        storage,
      ),
    ).toContain("__lime_resource_reload=");
  });

  it("普通恢复模式应通过 errors namespace 渲染英文外壳", () => {
    const { container } = renderPanel(new Error("Random render error"));
    const text = container.textContent ?? "";

    expect(text).toContain("The app hit an error and entered recovery mode");
    expect(text).toContain("Copy or export diagnostics first, then click");
    expect(text).toContain('"Retry recovery"');
    expect(text).toContain("Latest error: Random render error");
    expect(text).toContain("Clear Old Diagnostics");
    expect(text).toContain("Copy Diagnostics");
    expect(text).toContain("Copy Raw JSON");
    expect(text).toContain("Export Diagnostic JSON");
    expect(text).toContain("Open Downloads");
    expect(text).toContain("Retry Recovery");
    expect(text).not.toContain("应用发生错误");
    expect(text).not.toContain("errors.crashRecovery");
  });

  it("模块导入失败时应通过 errors namespace 展示强制刷新资源入口", () => {
    const { container } = renderPanel(
      new Error("Importing a module script failed."),
    );
    const text = container.textContent ?? "";

    expect(text).toContain("Force Resource Refresh");
    expect(text).toContain("Retry Only");
    expect(text).toContain("A frontend module resource failed to load.");
    expect(text).toContain("node_modules/.vite-electron");
    expect(text).not.toContain("强制刷新资源");
    expect(text).not.toContain("errors.crashRecovery");

    const codeTags = Array.from(container.querySelectorAll("code"));
    expect(codeTags).toHaveLength(2);
    expect(codeTags[0]?.className).toContain("bg-slate-100");
    expect(codeTags[0]?.className).not.toContain("bg-black/5");
  });
});
