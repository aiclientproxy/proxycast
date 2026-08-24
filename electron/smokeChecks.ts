/* global process */
import {
  AppServerClient,
  decodeMessage,
  encodeMessage,
  PROTOCOL_VERSION,
  SERVER_NAME,
  type InitializeResponse,
} from "@limecloud/app-server-client";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import type { BrowserWindow } from "./electronRuntime";
import {
  buildElectronSmokeSummary,
  isElectronSmokeStartupUrl,
  normalizeElectronSmokeRunId,
  sanitizeElectronSmokeLocation,
  type ElectronSmokeDiagnostics,
  type ElectronSmokeRendererEvidence,
  type ElectronSmokeRouteSnapshot,
  type ElectronSmokeSummary,
} from "./smokeEvidence";
import {
  waitForElectronSmokeMemorySettingsReady,
  type MemorySettingsSmokeSnapshot,
} from "./smokeMemorySettings";

interface AppServerSmokeHost {
  handleJsonLines(request: { lines: string[] }): Promise<{ lines: string[] }>;
}

interface ElectronSmokeRunnerOptions {
  window: BrowserWindow;
  appServerHost: AppServerSmokeHost;
  appVersion: string;
}

interface WorkbenchSmokeSnapshot {
  ok?: boolean;
  shellReady?: boolean;
  inputbarReady?: boolean;
  composerReady?: boolean;
  problemTexts?: unknown[];
  invokeErrors?: unknown[];
  traceErrors?: unknown[];
  visibleButtons?: unknown[];
  url?: string;
  title?: string;
  bodyStart?: string;
}

interface RendererSmokeCollection {
  renderer: ElectronSmokeRendererEvidence;
  traceFacts: Array<{
    command: string | null;
    transport: string | null;
    status: string | null;
    methods: string[];
  }>;
}

interface ElectronSmokeRunner {
  isStartupUrl(url: string): boolean;
  run(): Promise<ElectronSmokeSummary>;
  recordFailure(stage: string): Promise<ElectronSmokeSummary>;
}

const SUMMARY_FILE = "summary.json";
const TRACE_FILE = "trace-summary.json";
const SCREENSHOT_FILE = "settings-memory.png";
const LEGACY_SURFACE_CATALOG = "src/lib/governance/legacySurfaceCatalog.json";
const EMPTY_RENDERER_EVIDENCE: ElectronSmokeRendererEvidence = {
  electron: false,
  preloadInvoke: false,
  appServerCommandSupported: false,
  appServerIpcHitCount: 0,
  appServerMethods: [],
  invokeErrorCount: 0,
  traceErrorCount: 0,
  legacyCommandHitCount: 0,
  legacyCommands: [],
  mockFallbackHitCount: 0,
  pageErrorCount: 0,
};

export function createElectronSmokeRunner(
  options: ElectronSmokeRunnerOptions,
): ElectronSmokeRunner {
  const startedAtMs = Date.now();
  const startedAt = new Date(startedAtMs).toISOString();
  const runId = normalizeElectronSmokeRunId(
    process.env.LIME_GATE_RUN_ID,
    createStandaloneRunId(startedAt, process.pid),
  );
  const evidenceDir = path.resolve(
    process.env.LIME_ELECTRON_SMOKE_EVIDENCE_DIR?.trim() ||
      path.join(
        process.cwd(),
        ".lime/qc/project-gates",
        runId,
        "shell-01-electron-smoke",
      ),
  );
  const summaryPath = path.join(evidenceDir, SUMMARY_FILE);
  const tracePath = path.join(evidenceDir, TRACE_FILE);
  const screenshotPath = path.join(evidenceDir, SCREENSHOT_FILE);
  const diagnostics: ElectronSmokeDiagnostics = {
    consoleErrorCount: 0,
    rendererCrashCount: 0,
    rendererUnresponsiveCount: 0,
    preloadErrorCount: 0,
    rendererLoadErrorCount: 0,
  };
  const routes: ElectronSmokeRouteSnapshot[] = [];
  let hostAppServerInitialized = false;
  let hostAppServerProtocol: string | null = null;
  let rendererCollection: RendererSmokeCollection = {
    renderer: { ...EMPTY_RENDERER_EVIDENCE },
    traceFacts: [],
  };
  let screenshotCaptured = false;
  let pageErrorsBeforeReload = 0;
  let lastSummary: ElectronSmokeSummary | null = null;
  let startupCapture = Promise.resolve();

  installMainProcessDiagnostics(options.window, diagnostics);
  options.window.webContents.on("did-finish-load", () => {
    const loadedUrl = options.window.webContents.getURL();
    if (!isElectronSmokeStartupUrl(loadedUrl)) {
      return;
    }
    startupCapture = captureStartupRoute(options.window)
      .then((route) => {
        replaceRoute(routes, route);
      })
      .catch(() => {
        replaceRoute(routes, {
          stage: "startup",
          ready: false,
          location: sanitizeElectronSmokeLocation(loadedUrl),
        });
      });
  });

  async function writeSummary(
    failureStage: string | null,
  ): Promise<ElectronSmokeSummary> {
    await startupCapture;
    mkdirSync(evidenceDir, { recursive: true });
    const summary = buildElectronSmokeSummary({
      runId,
      startedAt,
      completedAt: new Date().toISOString(),
      appVersion: options.appVersion,
      backendMode:
        process.env.APP_SERVER_BACKEND_MODE?.trim() || "default-runtime",
      hostAppServerInitialized,
      hostAppServerProtocol,
      routes: [...routes],
      renderer: rendererCollection.renderer,
      diagnostics: { ...diagnostics },
      artifacts: {
        summary: SUMMARY_FILE,
        trace: existsSync(tracePath) ? TRACE_FILE : null,
        screenshot: screenshotCaptured ? SCREENSHOT_FILE : null,
        screenshotCaptured,
      },
      failureStage,
    });
    writeJsonAtomic(summaryPath, summary);
    lastSummary = summary;
    return summary;
  }

  return {
    isStartupUrl: isElectronSmokeStartupUrl,
    async run() {
      let stage = "renderer-error-capture";
      try {
        await installRendererErrorCapture(options.window);

        stage = "app-server-initialize";
        const client = new AppServerClient({ initialRequestId: 1 });
        const request = client.initialize({
          clientInfo: {
            name: "electron_smoke",
            title: "Electron smoke",
            version: options.appVersion,
          },
          capabilities: {
            eventMethods: ["agentSession/event"],
            experimentalApi: true,
          },
        });
        const response = await options.appServerHost.handleJsonLines({
          lines: [encodeMessage(request)],
        });
        const message = decodeMessage(response.lines[0] ?? "");
        if (!("result" in message)) {
          throw new Error("app-server initialize did not return a result");
        }
        const result = message.result as InitializeResponse;
        if (result.serverInfo.name !== SERVER_NAME) {
          throw new Error("app-server initialize returned an unexpected name");
        }
        if (result.serverInfo.protocolVersion !== PROTOCOL_VERSION) {
          throw new Error(
            "app-server initialize returned an unexpected protocol",
          );
        }
        hostAppServerInitialized = true;
        hostAppServerProtocol = result.serverInfo.protocolVersion;
        console.log(
          `[electron-smoke] app-server initialized protocol=${result.serverInfo.protocolVersion} version=${result.serverInfo.version}`,
        );

        stage = "workbench-ready";
        const workbench = await waitForElectronSmokeWorkbenchReady(
          options.window,
        );
        replaceRoute(routes, {
          stage: "workbench",
          ready: workbench.ok === true,
          location: sanitizeElectronSmokeLocation(workbench.url),
          title: workbench.title ?? null,
        });
        console.log("[electron-smoke] claw workbench shell ready");

        stage = "renderer-reload";
        pageErrorsBeforeReload = await collectRendererPageErrorCount(
          options.window,
        );
        await reloadElectronSmokeRenderer(options.window);
        await installRendererErrorCapture(options.window);
        const reloadedWorkbench = await waitForElectronSmokeWorkbenchReady(
          options.window,
        );
        replaceRoute(routes, {
          stage: "workbench-reload",
          ready: reloadedWorkbench.ok === true,
          location: sanitizeElectronSmokeLocation(reloadedWorkbench.url),
          title: reloadedWorkbench.title ?? null,
        });
        console.log("[electron-smoke] claw workbench shell ready after reload");

        stage = "settings-memory-ready";
        const memory = await waitForElectronSmokeMemorySettingsReady(
          options.window,
        );
        replaceRoute(routes, memoryRouteSnapshot(memory));
        console.log("[electron-smoke] memory settings ready");

        stage = "bridge-evidence";
        rendererCollection = await collectRendererSmokeEvidence(
          options.window,
          startedAtMs,
          loadRetiredCommands(),
        );
        rendererCollection.renderer.pageErrorCount += pageErrorsBeforeReload;
        mkdirSync(evidenceDir, { recursive: true });
        writeJsonAtomic(tracePath, {
          schemaVersion: 1,
          candidateRunId: runId,
          command: "app_server_handle_json_lines",
          methods: rendererCollection.renderer.appServerMethods,
          traceFacts: rendererCollection.traceFacts,
        });

        stage = "screenshot";
        const screenshot = await options.window.webContents.capturePage();
        if (screenshot.isEmpty()) {
          throw new Error("electron smoke screenshot was empty");
        }
        writeFileSync(screenshotPath, screenshot.toPNG());
        screenshotCaptured = existsSync(screenshotPath);

        stage = "contract-assertions";
        const summary = await writeSummary(null);
        if (summary.result !== "pass") {
          throw new Error(
            `electron smoke contract failed: ${summary.assertions.failed.join(", ")}`,
          );
        }
        console.log(
          `[electron-smoke] evidence ready run_id=${runId} summary=${summaryPath}`,
        );
        return summary;
      } catch (error) {
        if (!lastSummary || lastSummary.result !== "fail") {
          await writeSummary(stage);
        }
        throw error;
      }
    },
    async recordFailure(stage) {
      if (!options.window.isDestroyed()) {
        try {
          rendererCollection = await collectRendererSmokeEvidence(
            options.window,
            startedAtMs,
            loadRetiredCommands(),
          );
        } catch {
          rendererCollection = {
            renderer: { ...EMPTY_RENDERER_EVIDENCE },
            traceFacts: [],
          };
        }
      }
      return await writeSummary(stage);
    },
  };
}

function createStandaloneRunId(startedAt: string, pid: number): string {
  return `standalone-shell-01-${startedAt.replace(/[^0-9]/g, "").slice(0, 14)}-${pid}`;
}

function installMainProcessDiagnostics(
  window: BrowserWindow,
  diagnostics: ElectronSmokeDiagnostics,
): void {
  window.webContents.on("console-message", (_event, level) => {
    if (level === 3) {
      diagnostics.consoleErrorCount += 1;
    }
  });
  window.webContents.on("render-process-gone", () => {
    diagnostics.rendererCrashCount += 1;
  });
  window.webContents.on("unresponsive", () => {
    diagnostics.rendererUnresponsiveCount += 1;
  });
  window.webContents.on("preload-error", () => {
    diagnostics.preloadErrorCount += 1;
  });
  window.webContents.on(
    "did-fail-load",
    (_event, errorCode, _description, _validatedUrl, isMainFrame) => {
      if (isMainFrame !== false && errorCode !== -3) {
        diagnostics.rendererLoadErrorCount += 1;
      }
    },
  );
}

async function captureStartupRoute(
  window: BrowserWindow,
): Promise<ElectronSmokeRouteSnapshot> {
  const snapshot = (await window.webContents.executeJavaScript(
    `({
      ready: Boolean(document.querySelector("[data-lime-startup-shell]")),
      url: window.location.href,
      title: document.title || "",
    })`,
    true,
  )) as { ready?: boolean; url?: string; title?: string };
  return {
    stage: "startup",
    ready: snapshot.ready === true,
    location: sanitizeElectronSmokeLocation(snapshot.url),
    title: snapshot.title ?? null,
  };
}

async function installRendererErrorCapture(
  window: BrowserWindow,
): Promise<void> {
  await window.webContents.executeJavaScript(
    `(() => {
      const key = "__limeElectronSmokePageErrors";
      if (Array.isArray(globalThis[key])) return true;
      globalThis[key] = [];
      window.addEventListener("error", () => globalThis[key].push({ type: "error" }));
      window.addEventListener("unhandledrejection", () => globalThis[key].push({ type: "unhandledrejection" }));
      return true;
    })()`,
    true,
  );
}

async function collectRendererPageErrorCount(
  window: BrowserWindow,
): Promise<number> {
  return (await window.webContents.executeJavaScript(
    `Array.isArray(globalThis.__limeElectronSmokePageErrors)
      ? globalThis.__limeElectronSmokePageErrors.length
      : 0`,
    true,
  )) as number;
}

async function reloadElectronSmokeRenderer(
  window: BrowserWindow,
): Promise<void> {
  if (window.isDestroyed()) {
    throw new Error("main window was destroyed before renderer reload");
  }
  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error("renderer reload timed out"));
    }, 60_000);
    const onLoaded = () => {
      cleanup();
      resolve();
    };
    const onFailed = (
      _event: unknown,
      code: number,
      description: string,
      _url: string,
      isMainFrame: boolean,
    ) => {
      if (code === -3 || isMainFrame === false) {
        return;
      }
      cleanup();
      reject(new Error(`renderer reload failed: ${code} ${description}`));
    };
    const cleanup = () => {
      clearTimeout(timeout);
      window.webContents.off("did-finish-load", onLoaded);
      window.webContents.off("did-fail-load", onFailed);
    };
    window.webContents.once("did-finish-load", onLoaded);
    window.webContents.on("did-fail-load", onFailed);
    window.webContents.reload();
  });
}

async function collectRendererSmokeEvidence(
  window: BrowserWindow,
  startedAtMs: number,
  retiredCommands: string[],
): Promise<RendererSmokeCollection> {
  return (await window.webContents.executeJavaScript(
    `(() => {
      const startedAtMs = ${JSON.stringify(startedAtMs)};
      const retiredCommands = new Set(${JSON.stringify(retiredCommands)});
      const readJsonArray = (key) => {
        try {
          const value = JSON.parse(localStorage.getItem(key) || "[]");
          return Array.isArray(value) ? value : [];
        } catch {
          return [];
        }
      };
      const current = (entry) => {
        const timestamp = Date.parse(entry?.timestamp || "");
        return Number.isFinite(timestamp) && timestamp >= startedAtMs;
      };
      const methodsFor = (entry) => {
        const lines = entry?.args_preview?.request?.lines;
        if (!Array.isArray(lines)) return [];
        return lines.flatMap((line) => {
          try {
            const message = JSON.parse(String(line));
            return typeof message?.method === "string" ? [message.method] : [];
          } catch {
            return [];
          }
        });
      };
      const traces = readJsonArray("lime_invoke_trace_buffer_v1").filter(current);
      const invokeErrors = readJsonArray("lime_invoke_error_buffer_v1").filter(current);
      const appServerIpc = traces.filter((entry) =>
        entry?.command === "app_server_handle_json_lines" &&
        entry?.transport === "electron-ipc" &&
        entry?.status === "success"
      );
      const legacyCommands = traces
        .map((entry) => String(entry?.command || ""))
        .filter((command) => retiredCommands.has(command));
      const mockFallbackHits = traces.filter((entry) => {
        if (entry?.mock === true || entry?.mockFallback === true) return true;
        return [entry?.transport, entry?.source, entry?.fallback, entry?.fallbackMode]
          .some((value) => typeof value === "string" && value.toLowerCase().includes("mock"));
      });
      const pageErrors = Array.isArray(globalThis.__limeElectronSmokePageErrors)
        ? globalThis.__limeElectronSmokePageErrors
        : [];
      return {
        renderer: {
          electron: window.__LIME_ELECTRON__ === true,
          preloadInvoke: typeof window.electronAPI?.invoke === "function",
          appServerCommandSupported:
            typeof window.electronAPI?.supportsCommand === "function" &&
            window.electronAPI.supportsCommand("app_server_handle_json_lines"),
          appServerIpcHitCount: appServerIpc.length,
          appServerMethods: [...new Set(appServerIpc.flatMap(methodsFor))].sort(),
          invokeErrorCount: invokeErrors.length,
          traceErrorCount: traces.filter((entry) => entry?.status === "error").length,
          legacyCommandHitCount: legacyCommands.length,
          legacyCommands: [...new Set(legacyCommands)].sort(),
          mockFallbackHitCount: mockFallbackHits.length,
          pageErrorCount: pageErrors.length,
        },
        traceFacts: traces.map((entry) => ({
          command: typeof entry?.command === "string" ? entry.command : null,
          transport: typeof entry?.transport === "string" ? entry.transport : null,
          status: typeof entry?.status === "string" ? entry.status : null,
          methods: methodsFor(entry),
        })),
      };
    })()`,
    true,
  )) as RendererSmokeCollection;
}

async function waitForElectronSmokeWorkbenchReady(
  window: BrowserWindow,
): Promise<WorkbenchSmokeSnapshot> {
  if (window.isDestroyed()) {
    throw new Error("main window was destroyed before workbench smoke");
  }

  const result = (await window.webContents.executeJavaScript(
    `new Promise((resolve) => {
      const timeoutMs = 60000;
      const intervalMs = 250;
      const startedAt = Date.now();
      const problemPatterns = [
        /无法连接后端桥接/,
        /Desktop Host 尚未支持命令/,
        /Electron host command is not supported/,
        /Electron host command is not implemented/,
        /Unsupported command/,
        /未知命令/,
        /bridge cooldown active/,
        /加载.*失败/,
        /加载失败/,
        /调用失败/,
      ];
      const sanitize = (value) => String(value || "").replace(/\\s+/g, " ").trim();
      const readJsonArray = (key) => {
        try {
          const parsed = JSON.parse(localStorage.getItem(key) || "[]");
          return Array.isArray(parsed) ? parsed : [];
        } catch {
          return [];
        }
      };
      const isCurrentRunEntry = (entry) => {
        const timestamp = Date.parse(entry?.timestamp || "");
        return Number.isFinite(timestamp) && timestamp >= startedAt;
      };
      const visible = (element) => {
        if (!element) return false;
        const rect = element.getBoundingClientRect();
        const style = window.getComputedStyle(element);
        return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
      };
      const collect = () => {
        const text = document.body?.innerText || "";
        const problemTexts = problemPatterns.flatMap((pattern) => {
          const match = text.match(pattern);
          return match ? [match[0]] : [];
        });
        const textareas = Array.from(document.querySelectorAll('textarea[name="agent-chat-message"]'));
        const composer = textareas.find((item) => visible(item) && !item.disabled && item.getAttribute("aria-disabled") !== "true");
        const shellReady = Boolean(document.querySelector('[data-testid="workspace-shell-scene"]'));
        const inputbarReady = Boolean(document.querySelector('[data-testid="inputbar-core-container"]'));
        const invokeErrors = readJsonArray("lime_invoke_error_buffer_v1").filter(isCurrentRunEntry);
        const traceErrors = readJsonArray("lime_invoke_trace_buffer_v1").filter((entry) => entry && entry.status === "error" && isCurrentRunEntry(entry));
        return {
          ok: shellReady && inputbarReady && Boolean(composer) && problemTexts.length === 0 && invokeErrors.length === 0 && traceErrors.length === 0,
          shellReady,
          inputbarReady,
          composerReady: Boolean(composer),
          problemTexts,
          invokeErrors: invokeErrors.slice(-5).map((entry) => ({ command: entry?.command || null, transport: entry?.transport || null })),
          traceErrors: traceErrors.slice(-5).map((entry) => ({ command: entry?.command || null, transport: entry?.transport || null, status: entry?.status || null })),
          visibleButtons: Array.from(document.querySelectorAll("button"))
            .map((button, index) => {
              const rect = button.getBoundingClientRect();
              return {
                index,
                visible: rect.width > 0 && rect.height > 0,
                text: sanitize(button.textContent),
                aria: button.getAttribute("aria-label") || "",
                testId: button.getAttribute("data-testid") || "",
                disabled: button.disabled || button.getAttribute("aria-disabled") === "true",
              };
            })
            .filter((button) => button.visible && !button.disabled)
            .slice(0, 24),
          url: window.location.href,
          title: document.title,
          bodyStart: sanitize(text).slice(0, 500),
        };
      };
      const tick = () => {
        const snapshot = collect();
        if (snapshot.ok || Date.now() - startedAt >= timeoutMs) {
          resolve(snapshot);
          return;
        }
        setTimeout(tick, intervalMs);
      };
      tick();
    })`,
    true,
  )) as WorkbenchSmokeSnapshot;

  if (result?.ok) {
    return result;
  }

  throw new Error(
    `claw workbench shell not ready: ${JSON.stringify({
      shellReady: result?.shellReady ?? false,
      inputbarReady: result?.inputbarReady ?? false,
      composerReady: result?.composerReady ?? false,
      problemTexts: result?.problemTexts ?? [],
      invokeErrors: result?.invokeErrors ?? [],
      traceErrors: result?.traceErrors ?? [],
      visibleButtons: result?.visibleButtons ?? [],
      url: result?.url ?? "",
      title: result?.title ?? "",
      bodyStart: result?.bodyStart ?? "",
    })}`,
  );
}

function memoryRouteSnapshot(
  snapshot: MemorySettingsSmokeSnapshot,
): ElectronSmokeRouteSnapshot {
  return {
    stage: "settings-memory",
    ready: snapshot.ok === true && snapshot.stage === "done",
    location: sanitizeElectronSmokeLocation(snapshot.url),
    title: snapshot.title ?? null,
  };
}

function replaceRoute(
  routes: ElectronSmokeRouteSnapshot[],
  next: ElectronSmokeRouteSnapshot,
): void {
  const index = routes.findIndex((route) => route.stage === next.stage);
  if (index >= 0) {
    routes[index] = next;
    return;
  }
  routes.push(next);
}

function loadRetiredCommands(): string[] {
  const catalogPath = path.resolve(process.cwd(), LEGACY_SURFACE_CATALOG);
  const catalog = JSON.parse(readFileSync(catalogPath, "utf8")) as {
    commands?: Array<{
      classification?: string;
      commands?: string[];
    }>;
  };
  return (Array.isArray(catalog.commands) ? catalog.commands : [])
    .filter((entry) =>
      ["dead", "dead-candidate"].includes(entry.classification ?? ""),
    )
    .flatMap((entry) => (Array.isArray(entry.commands) ? entry.commands : []));
}

function writeJsonAtomic(filePath: string, value: unknown): void {
  mkdirSync(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.tmp`;
  writeFileSync(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  renameSync(temporaryPath, filePath);
}
