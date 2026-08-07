#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { writeMcpFixture } from "../mcp/lib/current-smoke-fixture.mjs";
import {
  APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  LEGACY_MCP_COMMANDS,
  sanitizeJson,
  writeJsonFile,
} from "../mcp/lib/current-smoke-transport.mjs";
import {
  appServerCallFromPage,
  assert,
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  parseInvokeTraceRaw,
  parseJsonRpcRequestsFromInvokeTrace,
  sanitizeText,
  sleep,
  waitForPageCondition,
} from "./mcp-config-fixture-smoke.mjs";

const DEFAULTS = {
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "mcp-workspace-plugin-runtime-fixture",
  ),
  prefix: "mcp-workspace-plugin-runtime-fixture",
  timeoutMs: 180_000,
  intervalMs: 250,
  keepTemp: false,
};

const LOG_PREFIX = "[smoke:mcp-workspace-plugin-runtime-fixture]";
// 复用 shared MCP fixture 的 allowed_callers，避免点击 smoke 与 fixture caller 漂移。
const PLUGIN_ID = "mcp-current-plugin";
const REQUIRED_METHODS = [
  "thread/start",
  "mcpServer/create",
  "agentSession/toolInventory/read",
  "mcpServer/start",
  "mcpTool/listForContext",
  "mcpServer/tool/call",
];

function printHelp() {
  console.log(`
MCP Workspace Plugin Runtime Electron Fixture Smoke

用途:
  启动真实 Electron Desktop Host，创建临时 stdio MCP server，然后在页面内注入
  最小 Workspace Harness 点击面板，点击“准备 MCP”后经 preload
  app_server_handle_json_lines 调用 agentSession/toolInventory/read、
  mcpTool/listForContext 与带真实 Thread 的 mcpServer/tool/call。

边界:
  这是 Workspace 点击验收骨架：真实 Electron / preload / App Server / MCP
  current JSON-RPC 都会被验证，但插件安装、插件选择和完整 React Workspace
  Harness UX 仍需后续 Gate B 验收。本脚本不调用正式模型后端，不使用 mock
  backend / renderer fallback / 旧 MCP facade 作为成功证据。

用法:
  node scripts/electron/mcp-workspace-plugin-runtime-fixture-smoke.mjs

选项:
  --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
`);
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "-h" || arg === "--help") {
      printHelp();
      process.exit(0);
    }
    if (arg === "--evidence-dir" && next) {
      options.evidenceDir = path.resolve(next.trim());
      index += 1;
      continue;
    }
    if (arg === "--prefix" && next) {
      options.prefix = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--timeout-ms" && next) {
      options.timeoutMs = Number(next);
      index += 1;
      continue;
    }
    if (arg === "--interval-ms" && next) {
      options.intervalMs = Number(next);
      index += 1;
      continue;
    }
    if (arg === "--keep-temp") {
      options.keepTemp = true;
      continue;
    }
    throw new Error(`未知参数: ${arg}`);
  }

  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms 必须是 >= 30000 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
  }
  if (!options.evidenceDir || !options.prefix) {
    throw new Error("--evidence-dir / --prefix 均不能为空");
  }
  return options;
}

function logStage(stage) {
  console.log(`${LOG_PREFIX} stage=${stage}`);
}

function buildPluginRuntimeCapabilities({
  pluginId,
  serverName,
  includeCallProof,
}) {
  return {
    pluginId,
    skills: [],
    mcpBindings: [
      {
        serverId: serverName,
        toolKey: `${serverName}/echo`,
        provider: "mcp",
        required: true,
        ...(includeCallProof
          ? {
              callProof: {
                arguments: { message: "hello workspace MCP" },
              },
            }
          : {}),
      },
    ],
    workflowBindings: [],
  };
}

async function createStoppedMcpServer(page, fixture) {
  const serverId = `mcp-workspace-plugin-${Date.now()}`;
  const serverName = serverId.replace(/[^a-zA-Z0-9_-]/g, "-");
  const createResult = await appServerCallFromPage(page, "mcpServer/create", {
    server: {
      id: serverId,
      name: serverName,
      description: "Workspace plugin runtime MCP Electron fixture",
      server_config: {
        command: "node",
        args: [fixture.serverPath],
        cwd: fixture.root,
        timeout: 3,
      },
      enabled_lime: true,
      enabled_claude: false,
      enabled_codex: false,
      enabled_gemini: false,
      created_at: Date.now(),
    },
  });
  return { serverId, serverName, createResult };
}

async function cleanupMcpServer(page, server) {
  if (!page || !server?.serverName) {
    return;
  }
  await appServerCallFromPage(page, "mcpServer/stop", {
    name: server.serverName,
  }).catch((error) => {
    console.warn(
      `${LOG_PREFIX} fixture stop failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  });
  if (!server.serverId) {
    return;
  }
  await appServerCallFromPage(page, "mcpServer/delete", {
    id: server.serverId,
  }).catch((error) => {
    console.warn(
      `${LOG_PREFIX} fixture delete failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  });
}

async function injectWorkspaceHarnessClickSkeleton(page, { serverName }) {
  const capabilities = buildPluginRuntimeCapabilities({
    pluginId: PLUGIN_ID,
    serverName,
    includeCallProof: true,
  });
  await page.evaluate(
    ({ capabilities, command, pluginId, serverName }) => {
      const existing = document.querySelector(
        '[data-testid="harness-status-panel"]',
      );
      if (existing) {
        existing.remove();
      }

      const panel = document.createElement("section");
      panel.setAttribute("data-testid", "harness-status-panel");
      panel.setAttribute("data-fixture", "mcp-workspace-plugin-runtime");
      panel.style.position = "fixed";
      panel.style.right = "16px";
      panel.style.bottom = "16px";
      panel.style.zIndex = "2147483647";
      panel.style.maxWidth = "420px";
      panel.style.padding = "12px";
      panel.style.border = "1px solid rgba(0,0,0,.2)";
      panel.style.borderRadius = "8px";
      panel.style.background = "Canvas";
      panel.style.color = "CanvasText";
      panel.style.boxShadow = "0 8px 28px rgba(0,0,0,.18)";

      const title = document.createElement("div");
      title.textContent = "工具与权限";
      title.style.fontWeight = "600";
      title.style.marginBottom = "8px";

      const status = document.createElement("div");
      status.setAttribute("data-testid", "mcp-workspace-plugin-runtime-status");
      status.textContent = "等待点击准备 MCP";
      status.style.fontSize = "12px";
      status.style.marginTop = "8px";

      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "准备 MCP";
      button.setAttribute("aria-label", "准备插件 MCP 工具");
      button.setAttribute(
        "data-testid",
        "mcp-workspace-plugin-runtime-prepare",
      );

      const callJsonRpc = async (method, params = {}) => {
        window.__LIME_MCP_WORKSPACE_FIXTURE_REQUEST_METHODS__ = [
          ...(window.__LIME_MCP_WORKSPACE_FIXTURE_REQUEST_METHODS__ || []),
          method,
        ];
        const invoke = window.electronAPI?.invoke;
        if (typeof invoke !== "function") {
          throw new Error("Electron preload invoke bridge is unavailable");
        }
        const id = `mcp-workspace-plugin-${Date.now()}-${Math.random()
          .toString(16)
          .slice(2)}`;
        const response = await invoke(command, {
          request: {
            lines: [
              JSON.stringify({
                jsonrpc: "2.0",
                id,
                method,
                params,
              }),
            ],
          },
        });
        const messages = Array.isArray(response?.lines)
          ? response.lines
              .map((line) => {
                try {
                  return JSON.parse(line);
                } catch {
                  return null;
                }
              })
              .filter(Boolean)
          : [];
        const error = messages.find(
          (message) => message?.id === id && message.error,
        );
        if (error) {
          throw new Error(`${method} failed: ${JSON.stringify(error.error)}`);
        }
        const result = messages.find(
          (message) =>
            message?.id === id &&
            Object.prototype.hasOwnProperty.call(message, "result"),
        );
        if (!result) {
          throw new Error(`${method} did not return a JSON-RPC result`);
        }
        return result.result;
      };

      const toolMatches = (tools, expectedToolName) =>
        Array.isArray(tools) &&
        tools.some(
          (tool) =>
            String(tool?.name || "").toLowerCase() ===
            String(expectedToolName || "").toLowerCase(),
        );

      const findTarget = (inventory) => {
        const targets = Array.isArray(inventory?.plugin_mcp_targets)
          ? inventory.plugin_mcp_targets
          : [];
        return targets.find(
          (target) =>
            target?.pluginId === pluginId &&
            String(target?.expectedToolName || "").includes(serverName),
        );
      };

      button.addEventListener("click", async () => {
        button.disabled = true;
        status.textContent = "正在准备 MCP...";
        window.__LIME_MCP_WORKSPACE_FIXTURE_RESULT__ = {
          ok: false,
          stage: "started",
        };

        try {
          const threadResponse = await callJsonRpc("thread/start", {
            ephemeral: true,
          });
          const threadId = threadResponse?.thread?.id;
          if (typeof threadId !== "string" || threadId.length === 0) {
            throw new Error("thread/start 未返回真实 threadId");
          }
          const inventoryResponse = await callJsonRpc(
            "agentSession/toolInventory/read",
            {
              caller: "assistant",
              workbench: true,
              browserAssist: true,
              metadata: {
                harness: {
                  plugin_runtime_capabilities:
                    window.__LIME_MCP_WORKSPACE_FIXTURE_CAPABILITIES__,
                },
              },
            },
          );
          const target = findTarget(inventoryResponse?.inventory);
          if (!target) {
            throw new Error("未从 tool inventory 读到 plugin_mcp_targets");
          }

          const expectedToolName = target.expectedToolName;
          const prepareRequests = Array.isArray(target.prepareRequests)
            ? target.prepareRequests.filter(
                (request) => request?.status === "candidate",
              )
            : [];
          const executedPrepareMethods = [];
          let listProofToolCount = null;
          let listProofSeen = false;

          for (const request of prepareRequests) {
            const result = await callJsonRpc(request.method, request.params);
            executedPrepareMethods.push(request.method);
            if (request.method === "mcpTool/listForContext") {
              const tools = Array.isArray(result?.tools) ? result.tools : [];
              listProofToolCount = tools.length;
              listProofSeen = true;
              if (!toolMatches(tools, expectedToolName)) {
                throw new Error("caller-scoped list proof 未暴露目标工具");
              }
            }
          }

          if (!listProofSeen) {
            const result = await callJsonRpc(
              "mcpTool/listForContext",
              target.toolListRequest ?? {
                caller: `plugin:${pluginId}`,
                includeDeferred: true,
              },
            );
            const tools = Array.isArray(result?.tools) ? result.tools : [];
            listProofToolCount = tools.length;
            listProofSeen = true;
            if (!toolMatches(tools, expectedToolName)) {
              throw new Error("fallback list proof 未暴露目标工具");
            }
          }

          const callProofRequest = target.callProofRequest;
          if (callProofRequest?.method !== "mcpServer/tool/call") {
            throw new Error("tool inventory 未给出显式 call proof request");
          }
          const callProofResult = await callJsonRpc("mcpServer/tool/call", {
            threadId,
            ...callProofRequest.params,
          });
          if (callProofResult?.isError === true) {
            throw new Error("mcpServer/tool/call 返回 isError=true");
          }

          const callCountBeforeDefaultProof = (
            window.__LIME_MCP_WORKSPACE_FIXTURE_REQUEST_METHODS__ || []
          ).filter((method) => method === "mcpServer/tool/call").length;
          const defaultInventoryResponse = await callJsonRpc(
            "agentSession/toolInventory/read",
            {
              caller: "assistant",
              workbench: true,
              browserAssist: true,
              metadata: {
                harness: {
                  plugin_runtime_capabilities: {
                    pluginId,
                    skills: [],
                    mcpBindings: [
                      {
                        serverId: serverName,
                        toolKey: `${serverName}/echo`,
                        provider: "mcp",
                        required: true,
                      },
                    ],
                    workflowBindings: [],
                  },
                },
              },
            },
          );
          const defaultTarget = findTarget(defaultInventoryResponse?.inventory);
          if (!defaultTarget?.toolListRequest) {
            throw new Error("default proof 未给出 toolListRequest");
          }
          const defaultListResult = await callJsonRpc(
            "mcpTool/listForContext",
            defaultTarget.toolListRequest,
          );
          if (
            !toolMatches(
              Array.isArray(defaultListResult?.tools)
                ? defaultListResult.tools
                : [],
              defaultTarget.expectedToolName,
            )
          ) {
            throw new Error("default list proof 未暴露目标工具");
          }
          const callCountAfterDefaultProof = (
            window.__LIME_MCP_WORKSPACE_FIXTURE_REQUEST_METHODS__ || []
          ).filter((method) => method === "mcpServer/tool/call").length;

          const defaultProofDidNotCallTool =
            callCountAfterDefaultProof === callCountBeforeDefaultProof;
          if (!defaultProofDidNotCallTool) {
            throw new Error("default list proof 不应触发工具调用");
          }

          window.__LIME_MCP_WORKSPACE_FIXTURE_RESULT__ = {
            ok: true,
            stage: "completed",
            appServerCommand: command,
            requestMethods:
              window.__LIME_MCP_WORKSPACE_FIXTURE_REQUEST_METHODS__ || [],
            pluginId,
            serverName,
            expectedToolName,
            runtimeStatus: target.runtimeStatus,
            prepareStatus: target.prepareStatus,
            executedPrepareMethods,
            listProofSeen,
            listProofToolCount,
            explicitCallProofSeen: true,
            callProofContentCount: Array.isArray(callProofResult?.content)
              ? callProofResult.content.length
              : 0,
            defaultProofDidNotCallTool,
            defaultProofToolCount: Array.isArray(defaultListResult?.tools)
              ? defaultListResult.tools.length
              : 0,
          };
          status.textContent = "MCP 已准备";
        } catch (error) {
          window.__LIME_MCP_WORKSPACE_FIXTURE_RESULT__ = {
            ok: false,
            stage: "failed",
            error: error instanceof Error ? error.message : String(error),
          };
          status.textContent =
            error instanceof Error ? error.message : String(error);
        } finally {
          button.disabled = false;
        }
      });

      window.__LIME_MCP_WORKSPACE_FIXTURE_CAPABILITIES__ = capabilities;
      window.__LIME_MCP_WORKSPACE_FIXTURE_REQUEST_METHODS__ = [];

      panel.append(title, button, status);
      document.body.append(panel);
      return {
        harnessPanelVisible: true,
        buttonText: button.textContent,
        statusText: status.textContent,
      };
    },
    {
      capabilities,
      command: APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      pluginId: PLUGIN_ID,
      serverName,
    },
  );
}

async function clickPrepareButton(page, options) {
  await page
    .locator('[data-testid="mcp-workspace-plugin-runtime-prepare"]')
    .click({ timeout: Math.min(30_000, options.timeoutMs) });
  return await waitForPageCondition(
    page,
    options,
    () => {
      const result = window.__LIME_MCP_WORKSPACE_FIXTURE_RESULT__ || null;
      return result?.ok || result?.stage === "failed" ? result : null;
    },
    "MCP Workspace prepare click did not complete",
  );
}

function summarizeElectronEvidence({ traceRaw, createResult, clickResult }) {
  const entries = parseInvokeTraceRaw(traceRaw);
  const requests = parseJsonRpcRequestsFromInvokeTrace(traceRaw);
  const syntheticRequests = [
    createResult?.method,
    ...(Array.isArray(clickResult?.requestMethods)
      ? clickResult.requestMethods
      : []),
  ]
    .filter(Boolean)
    .map((method) => ({
      command: APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      transport: "electron-ipc",
      status: "success",
      durationMs: null,
      id: null,
      method,
      params: {},
      source: "workspace-harness-page-result",
    }));
  const commands = Array.from(
    new Set(
      [
        ...entries.map((entry) => entry?.command),
        createResult?.appServerCommand,
        clickResult?.appServerCommand,
      ].filter(Boolean),
    ),
  );
  const requestMethods = Array.from(
    new Set(
      [...requests, ...syntheticRequests]
        .map((request) => request.method)
        .filter(Boolean),
    ),
  );
  return {
    appServerHandleJsonLinesSeen: commands.includes(
      APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    ),
    requestMethods,
    missingRequiredMethods: REQUIRED_METHODS.filter(
      (method) => !requestMethods.includes(method),
    ),
    legacyMcpCommandsSeen: LEGACY_MCP_COMMANDS.filter((command) =>
      commands.includes(command),
    ),
    requests: [...requests, ...syntheticRequests],
  };
}

async function readGuiSnapshot(page) {
  return await page.evaluate(() => {
    const panel = document.querySelector(
      '[data-testid="harness-status-panel"]',
    );
    const button = document.querySelector(
      '[data-testid="mcp-workspace-plugin-runtime-prepare"]',
    );
    const status = document.querySelector(
      '[data-testid="mcp-workspace-plugin-runtime-status"]',
    );
    return {
      harnessPanelVisible: Boolean(panel),
      harnessPanelText: panel?.textContent || "",
      prepareButtonText: button?.textContent || "",
      prepareStatusText: status?.textContent || "",
      result: window.__LIME_MCP_WORKSPACE_FIXTURE_RESULT__ || null,
      traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
    };
  });
}

function assertElectronEvidence(evidence) {
  assert(
    evidence.appServerHandleJsonLinesSeen,
    "未观察到 app_server_handle_json_lines",
  );
  assert(
    evidence.missingRequiredMethods.length === 0,
    `缺少 App Server current method: ${evidence.missingRequiredMethods.join(", ")}`,
  );
  assert(
    evidence.legacyMcpCommandsSeen.length === 0,
    `观察到 legacy MCP 命令: ${evidence.legacyMcpCommandsSeen.join(", ")}`,
  );
}

export async function run() {
  const options = parseArgs(process.argv.slice(2));
  fs.mkdirSync(options.evidenceDir, { recursive: true });

  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const rawEvidencePath = path.join(
    options.evidenceDir,
    `${options.prefix}-raw.json`,
  );
  const screenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}.png`,
  );
  const failureScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-failure.png`,
  );

  const runtimeEnv = createTempRuntimeEnv();
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
    forceBuild: false,
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: {
      ...runtimeEnv.env,
      APP_SERVER_BIN: appServerBinary,
    },
  });

  const summary = {
    ok: false,
    checkedAt: new Date().toISOString(),
    backendMode: "runtime",
    proofLevel: "Gate B skeleton",
    claimBoundary:
      "真实 Electron/preload/App Server/MCP current JSON-RPC + page-level Workspace Harness click skeleton；不声称完整插件安装、插件选择或生产 React Workspace UX 已验收。",
    electronPreloadBridge: false,
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
    appServerBinary: options.keepTemp
      ? appServerBinary
      : path.basename(appServerBinary),
    pluginId: PLUGIN_ID,
    serverName: null,
    expectedToolName: null,
    guiHarnessPanelVisible: false,
    guiPrepareButtonClicked: false,
    explicitCallProofSeen: false,
    defaultProofDidNotCallTool: false,
    appServerHandleJsonLinesSeen: false,
    electronRequestMethods: [],
    missingRequiredMethods: [...REQUIRED_METHODS],
    legacyMcpCommandsSeen: [],
    consoleErrors: [],
    screenshot: null,
    rawEvidence: rawEvidencePath,
    summary: summaryPath,
  };

  const consoleErrors = [];
  const rawEvidence = {};
  let app = null;
  let page = null;
  let fixture = null;
  let server = null;

  try {
    logStage("write-mcp-fixture");
    fixture = await writeMcpFixture();

    logStage("launch-electron");
    const handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      backendMode: "runtime",
    });
    app = handle.app;
    page = handle.page;
    summary.electronPreloadBridge =
      handle.rendererSnapshot.electron &&
      handle.rendererSnapshot.hasInvokeBridge;

    logStage("create-stopped-mcp-server");
    server = await createStoppedMcpServer(page, fixture);
    summary.serverName = server.serverName;
    summary.expectedToolName = `mcp__${server.serverName}__echo`;
    rawEvidence.serverCreate = sanitizeJson(server.createResult);

    logStage("inject-workspace-harness-click-skeleton");
    rawEvidence.injectedHarness = sanitizeJson(
      await injectWorkspaceHarnessClickSkeleton(page, {
        serverName: server.serverName,
      }),
    );

    logStage("click-prepare-mcp");
    const clickResult = await clickPrepareButton(page, options);
    assert(
      clickResult?.ok === true,
      `准备 MCP 点击失败: ${clickResult?.error || "unknown"}`,
    );
    summary.guiPrepareButtonClicked = true;
    summary.explicitCallProofSeen = clickResult.explicitCallProofSeen === true;
    summary.defaultProofDidNotCallTool =
      clickResult.defaultProofDidNotCallTool === true;
    rawEvidence.clickResult = sanitizeJson(clickResult);

    logStage("read-gui-and-trace-evidence");
    const guiSnapshot = await readGuiSnapshot(page);
    rawEvidence.guiSnapshot = sanitizeJson({
      ...guiSnapshot,
      traceRaw: undefined,
      errorRaw: undefined,
    });
    assert(guiSnapshot.harnessPanelVisible, "未看到 harness-status-panel");
    assert(
      guiSnapshot.harnessPanelText.includes("工具与权限") &&
        guiSnapshot.harnessPanelText.includes("准备 MCP"),
      "Harness 点击骨架未展示工具与权限 / 准备 MCP",
    );
    summary.guiHarnessPanelVisible = true;

    const evidence = summarizeElectronEvidence({
      traceRaw: guiSnapshot.traceRaw,
      createResult: server.createResult,
      clickResult,
    });
    assertElectronEvidence(evidence);
    summary.appServerHandleJsonLinesSeen =
      evidence.appServerHandleJsonLinesSeen;
    summary.electronRequestMethods = evidence.requestMethods;
    summary.missingRequiredMethods = evidence.missingRequiredMethods;
    summary.legacyMcpCommandsSeen = evidence.legacyMcpCommandsSeen;
    rawEvidence.electronRequests = sanitizeJson(evidence.requests);

    await page.screenshot({ path: screenshotPath, fullPage: true });
    await cleanupMcpServer(page, server);
    server = null;
    await closeElectronFixture({ app });
    app = null;
    page = null;

    assert(
      consoleErrors.length === 0,
      `观察到 console error: ${consoleErrors.join(" | ")}`,
    );

    summary.consoleErrors = consoleErrors;
    summary.screenshot = screenshotPath;
    summary.ok = true;
    summary.completedAt = new Date().toISOString();
    writeJsonFile(rawEvidencePath, rawEvidence);
    writeJsonFile(summaryPath, summary);
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
    console.log(`${LOG_PREFIX} server=${summary.serverName ?? ""}`);
  } catch (error) {
    summary.error = error instanceof Error ? error.message : String(error);
    summary.consoleErrors = consoleErrors.map(sanitizeText);
    if (Object.keys(rawEvidence).length > 0) {
      writeJsonFile(rawEvidencePath, rawEvidence);
    }
    writeJsonFile(summaryPath, summary);
    if (page) {
      try {
        await page.screenshot({
          path: failureScreenshotPath,
          fullPage: true,
        });
        summary.failureScreenshot = failureScreenshotPath;
        writeJsonFile(summaryPath, summary);
      } catch {
        // 截图失败不覆盖原始错误。
      }
    }
    throw error;
  } finally {
    if (page && server) {
      await cleanupMcpServer(page, server);
    }
    if (app) {
      await closeElectronFixture({ app });
    }
    if (fixture && !options.keepTemp) {
      fs.rmSync(fixture.root, { recursive: true, force: true });
    }
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  run().catch((error) => {
    console.error(
      `${LOG_PREFIX} failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    process.exitCode = 1;
  });
}
