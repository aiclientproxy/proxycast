import {
  appServerCallFromPage,
  assert,
  parseInvokeTraceRaw,
  sleep,
} from "./mcp-config-fixture-smoke.mjs";

export const PLUGIN_V2_MCP_APP_RESOURCE_URI =
  "ui://mcp-elicitation-plugin/release-check.html";
export const PLUGIN_V2_MCP_APP_TITLE = "Plugin v2 Release Check";
export const PLUGIN_V2_MCP_APP_MARKER = "PLUGIN_V2_MCP_APP_GATE_B_READY";

export function pluginV2McpAppHtml() {
  return `<!doctype html>
<html>
  <head><meta charset="utf-8"><title>${PLUGIN_V2_MCP_APP_TITLE}</title></head>
  <body data-plugin-v2-mcp-app="${PLUGIN_V2_MCP_APP_MARKER}">
    <main><h1>${PLUGIN_V2_MCP_APP_TITLE}</h1><p>${PLUGIN_V2_MCP_APP_MARKER}</p></main>
  </body>
</html>`;
}

export function findPluginV2McpAppItems(value, items = []) {
  if (Array.isArray(value)) {
    value.forEach((entry) => findPluginV2McpAppItems(entry, items));
    return items;
  }
  if (!value || typeof value !== "object") return items;
  if (
    value.type === "mcpToolCall" &&
    value.mcpAppResourceUri === PLUGIN_V2_MCP_APP_RESOURCE_URI
  ) {
    items.push(value);
  }
  Object.values(value).forEach((entry) =>
    findPluginV2McpAppItems(entry, items),
  );
  return items;
}

export function pluginV2McpAppTraceEvidence(traceRaw, runtime) {
  const trace = parseInvokeTraceRaw(traceRaw);
  const resourceReadAttempts = trace.flatMap((entry) => {
    if (
      entry?.command !== "app_server_handle_json_lines" ||
      entry?.transport !== "electron-ipc"
    ) {
      return [];
    }
    const request = parseJsonRpcLine(entry?.args_preview?.request?.lines);
    if (
      request?.method === "mcpServer/resource/read" &&
      request?.params?.threadId === runtime.threadId &&
      request?.params?.sessionId === undefined &&
      request?.params?.uri === PLUGIN_V2_MCP_APP_RESOURCE_URI
    ) {
      return [
        {
          error:
            typeof entry?.error === "string" ? entry.error.slice(0, 500) : null,
          server: request?.params?.server ?? null,
          status: entry?.status ?? null,
        },
      ];
    }
    return [];
  });
  const embeddedBrowserAttempts = trace.flatMap((entry) => {
    if (
      ![
        "embedded_browser_view_load_html",
        "embedded_browser_view_mount",
        "embedded_browser_view_set_bounds",
        "embedded_browser_view_destroy",
      ].includes(entry?.command) ||
      entry?.transport !== "electron-ipc"
    ) {
      return [];
    }
    return [
      {
        command: entry.command,
        error:
          typeof entry?.error === "string" ? entry.error.slice(0, 500) : null,
        leaseId: entry?.args_preview?.leaseId ?? null,
        source: entry?.args_preview?.source ?? null,
        sourceUri: entry?.args_preview?.sourceUri ?? null,
        status: entry?.status ?? null,
        viewId: entry?.args_preview?.viewId ?? null,
        visible: entry?.args_preview?.visible ?? null,
      },
    ];
  });
  const htmlLoads = embeddedBrowserAttempts.filter(
    (entry) =>
      entry.command === "embedded_browser_view_load_html" &&
      entry.status === "success" &&
      entry.source === "mcpApp" &&
      entry.sourceUri === PLUGIN_V2_MCP_APP_RESOURCE_URI,
  );
  return {
    embeddedBrowserAttempts,
    htmlLoadCount: htmlLoads.length,
    htmlLoadViewIds: htmlLoads.map((entry) => entry.viewId),
    resourceReadAttempts,
    resourceReadCount: resourceReadAttempts.filter(
      (entry) => entry.status === "success",
    ).length,
  };
}

export async function installPluginV2EmbeddedBrowserLifecycleCapture(page) {
  await page.evaluate(() => {
    const stateKey = "__LIME_PLUGIN_V2_EMBEDDED_BROWSER_LIFECYCLE__";
    const listenerKey = "__LIME_PLUGIN_V2_EMBEDDED_BROWSER_UNLISTEN__";
    if (!Array.isArray(window[stateKey])) {
      window[stateKey] = [];
    }
    if (typeof window[listenerKey] === "function") {
      return;
    }
    window[listenerKey] = window.electronAPI?.listen?.(
      "embedded-browser-view-destroyed",
      (event) => {
        window[stateKey].push({
          observedAt: Date.now(),
          ...(event?.payload ?? {}),
        });
      },
    );
  });
}

export async function waitForPluginV2McpAppSurface({
  app,
  page,
  options,
  runtime,
}) {
  const { containerId, viewId } = await waitForPluginV2McpAppFrame({
    page,
    options,
    runtime,
  });

  const startedAt = Date.now();
  let latest = null;
  let latestTraceEvidence = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 60_000)) {
    latest = await readMcpAppWebContents(app);
    const traceRaw =
      (await page.evaluate(() =>
        window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      )) || "";
    latestTraceEvidence = pluginV2McpAppTraceEvidence(traceRaw, runtime);
    if (
      latest.some(
        (entry) =>
          entry.title === PLUGIN_V2_MCP_APP_TITLE &&
          entry.marker === PLUGIN_V2_MCP_APP_MARKER,
      )
    ) {
      if (
        latestTraceEvidence.resourceReadCount > 0 &&
        latestTraceEvidence.htmlLoadCount > 0
      ) {
        return {
          containerId,
          traceEvidence: latestTraceEvidence,
          viewId,
          webContents: latest,
        };
      }
    }
    await sleep(options.intervalMs);
  }
  const renderer = await readMcpAppRendererDiagnostics(page, runtime);
  throw new Error(
    `Plugin v2 MCP App surface 未完成: ${JSON.stringify({
      renderer,
      traceEvidence: latestTraceEvidence,
      webContents: latest,
    })}`,
  );
}

export async function waitForPluginV2McpAppHistoryUnavailable({
  app,
  page,
  options,
  runtime,
}) {
  const { containerId, frame, viewId } = await waitForPluginV2McpAppFrame({
    page,
    options,
    runtime,
  });
  const unavailable = page.locator(
    '[data-testid="workspace-plugin-surface-history-unavailable"]',
  );
  await unavailable.waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 30_000),
  });
  assert(
    (await frame.getAttribute("data-plugin-availability")) === "uninstalled" &&
      (await frame.getAttribute("data-mounted")) === "false",
    "卸载后历史 surface 不得挂载 Plugin WebContentsView",
  );

  const traceRaw =
    (await page.evaluate(() =>
      window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
    )) || "";
  const traceEvidence = pluginV2McpAppTraceEvidence(traceRaw, runtime);
  const webContents = await readMcpAppWebContents(app);
  assert(
    !webContents.some(
      (entry) =>
        entry.title === PLUGIN_V2_MCP_APP_TITLE ||
        entry.marker === PLUGIN_V2_MCP_APP_MARKER,
    ),
    "卸载后历史 surface 仍残留 Plugin MCP App WebContents",
  );

  return {
    containerId,
    message: (await unavailable.textContent())?.trim() ?? "",
    traceEvidence,
    viewId,
    webContents,
  };
}

async function waitForPluginV2McpAppFrame({ page, options, runtime }) {
  const frame = page.locator('[data-testid="workspace-plugin-surface-frame"]');
  const frameDeadline = Date.now() + Math.min(options.timeoutMs, 30_000);
  while (Date.now() < frameDeadline && !(await frame.isVisible())) {
    await sleep(options.intervalMs);
  }
  if (!(await frame.isVisible())) {
    throw new Error(
      `workspace-plugin-surface-frame 未出现: ${JSON.stringify(
        await readMcpAppRendererDiagnostics(page, runtime),
      )}`,
    );
  }
  const containerId = `mcp-app-${runtime.mcpAppItemId}`;
  const viewId = `plugin-surface-${containerId}`;
  assert(
    (await frame.getAttribute("data-view-id")) === viewId,
    "Right Surface 未使用 canonical MCP item identity",
  );
  return { containerId, frame, viewId };
}

async function readMcpAppRendererDiagnostics(page, runtime) {
  const itemPage = await appServerCallFromPage(page, "thread/items/list", {
    threadId: runtime.threadId,
    limit: 100,
    sortDirection: "desc",
  }).catch((error) => ({ error: String(error), result: null }));
  return await page.evaluate(
    ({ itemId, itemPage, resourceUri, sessionId, threadId }) => {
      const trace = JSON.parse(
        window.localStorage.getItem("lime_invoke_trace_buffer_v1") || "[]",
      );
      const methods = trace.flatMap((entry) => {
        if (entry?.command !== "app_server_handle_json_lines") return [];
        const lines = entry?.args_preview?.request?.lines;
        if (!Array.isArray(lines)) return [];
        return lines.flatMap((line) => {
          try {
            const request = JSON.parse(String(line));
            return typeof request?.method === "string" ? [request.method] : [];
          } catch {
            return [];
          }
        });
      });
      const switchSuccessEntries = (
        window.__LIME_AGENTUI_PERF__?.entries?.() ?? []
      )
        .filter(
          (entry) =>
            entry?.phase === "session.switch.success" &&
            entry?.sessionId === sessionId,
        )
        .slice(-5);
      const testIds = Array.from(document.querySelectorAll("[data-testid]"))
        .map((element) => element.getAttribute("data-testid"))
        .filter(
          (testId) =>
            typeof testId === "string" &&
            (testId.includes("right-surface") ||
              testId.includes("plugin-surface")),
        );
      const storedItemMatches = [
        ["session", window.sessionStorage],
        ["local", window.localStorage],
      ].flatMap(([storageKind, storage]) =>
        Object.keys(storage).flatMap((key) => {
          if (
            !key.includes("agent_thread_items") &&
            !key.includes("agent_session_snapshots")
          ) {
            return [];
          }
          let parsed;
          try {
            parsed = JSON.parse(storage.getItem(key) || "null");
          } catch {
            return [];
          }
          const matches = [];
          const visit = (value) => {
            if (Array.isArray(value)) {
              value.forEach(visit);
              return;
            }
            if (!value || typeof value !== "object") return;
            if (value.id === itemId) {
              matches.push({
                id: value.id,
                metadata: value.metadata ?? null,
                status: value.status ?? null,
                threadId: value.thread_id ?? null,
                type: value.type ?? null,
              });
            }
            Object.values(value).forEach(visit);
          };
          visit(parsed);
          return matches.length > 0 ? [{ key, matches, storageKind }] : [];
        }),
      );
      return {
        activeRightSurface:
          document
            .querySelector('[data-testid="workspace-right-surface-host"]')
            ?.getAttribute("data-surface") ?? null,
        activePanePresent: Boolean(
          document.querySelector(
            '[data-testid="workspace-right-surface-active-pane"]',
          ),
        ),
        appSurfaceTabPresent: Boolean(
          document.querySelector(
            '[data-testid="workspace-right-surface-tab-appSurface"]',
          ),
        ),
        expectedContainerId: `mcp-app-${itemId}`,
        embeddedBrowserLifecycle: Array.isArray(
          window.__LIME_PLUGIN_V2_EMBEDDED_BROWSER_LIFECYCLE__,
        )
          ? window.__LIME_PLUGIN_V2_EMBEDDED_BROWSER_LIFECYCLE__
          : [],
        frames: Array.from(
          document.querySelectorAll(
            '[data-testid="workspace-plugin-surface-frame"]',
          ),
        ).map((element) => ({
          active: element.getAttribute("data-active"),
          connected: element.isConnected,
          hostAvailable: element.getAttribute("data-host-available"),
          mcpResourceUri: element.getAttribute("data-mcp-resource-uri"),
          mcpServer: element.getAttribute("data-mcp-server"),
          mounted: element.getAttribute("data-mounted"),
          runtimeSessionId: element.getAttribute("data-runtime-session-id"),
          runtimeThreadId: element.getAttribute("data-runtime-thread-id"),
          viewId: element.getAttribute("data-view-id"),
        })),
        inputSessionIds: Array.from(
          document.querySelectorAll('textarea[name="agent-chat-message"]'),
        ).map((element) => element.getAttribute("data-session-id")),
        itemId,
        itemPage:
          itemPage?.result?.data?.map((entry) => ({
            id: entry?.item?.id ?? null,
            mcpAppResourceUri: entry?.item?.mcpAppResourceUri ?? null,
            pluginId: entry?.item?.pluginId ?? null,
            server: entry?.item?.server ?? null,
            status: entry?.item?.status ?? null,
            turnId: entry?.turnId ?? null,
            type: entry?.item?.type ?? null,
          })) ??
          itemPage?.error ??
          null,
        methodCounts: Object.fromEntries(
          Array.from(new Set(methods)).map((method) => [
            method,
            methods.filter((candidate) => candidate === method).length,
          ]),
        ),
        resourceUri,
        sessionId,
        storedItemMatches,
        switchSuccessEntries,
        testIds,
        threadId,
        visibleText: document.body?.innerText?.slice(-2_000) ?? "",
      };
    },
    {
      itemId: runtime.mcpAppItemId,
      itemPage,
      resourceUri: PLUGIN_V2_MCP_APP_RESOURCE_URI,
      sessionId: runtime.sessionId,
      threadId: runtime.threadId,
    },
  );
}

async function readMcpAppWebContents(app) {
  return await app.evaluate(
    async ({ BrowserWindow, webContents }, { marker, title }) => {
      const browserWindowWebContentsIds = new Set(
        BrowserWindow.getAllWindows().map((window) => window.webContents.id),
      );
      return await Promise.all(
        webContents
          .getAllWebContents()
          .filter(
            (entry) =>
              !entry.isDestroyed() &&
              !browserWindowWebContentsIds.has(entry.id),
          )
          .map(async (entry) => {
            let observedMarker = null;
            try {
              observedMarker = await Promise.race([
                entry.executeJavaScript(
                  "document.body?.dataset?.pluginV2McpApp ?? null",
                ),
                new Promise((resolve) =>
                  setTimeout(() => resolve(null), 1_000),
                ),
              ]);
            } catch {
              observedMarker = null;
            }
            const currentTitle = entry.getTitle();
            return {
              id: entry.id,
              isDestroyed: entry.isDestroyed(),
              isLoading: entry.isLoading(),
              marker: observedMarker === marker ? observedMarker : null,
              title: currentTitle.slice(0, 160) || null,
              titleMatches: currentTitle === title,
              type: entry.getType(),
              urlScheme: entry.getURL().split(":", 1)[0] || null,
            };
          }),
      );
    },
    { marker: PLUGIN_V2_MCP_APP_MARKER, title: PLUGIN_V2_MCP_APP_TITLE },
  );
}

function parseJsonRpcLine(lines) {
  const serialized = Array.isArray(lines)
    ? lines.filter((entry) => typeof entry === "string").join("\n")
    : lines;
  if (typeof serialized !== "string") return null;
  const line = serialized
    .split(/\r?\n/u)
    .map((entry) => entry.trim())
    .find(Boolean);
  if (!line) return null;
  try {
    return JSON.parse(line);
  } catch {
    return null;
  }
}
