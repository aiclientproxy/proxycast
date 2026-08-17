#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { chromium } from "playwright";

const DEFAULTS = {
  appUrl: "http://127.0.0.1:1420/",
  healthUrl: "http://127.0.0.1:3030/health",
  invokeUrl: "http://127.0.0.1:3030/invoke",
  timeoutMs: 120_000,
  intervalMs: 1_000,
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "workspace-skill-bindings-current",
  ),
  prefix: "workspace-skill-bindings-current",
  headless: true,
};

const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const REQUIRED_APP_SERVER_METHODS = [
  "workspaceRegisteredSkills/list",
  "workspaceSkillBindings/list",
];
const LEGACY_WORKSPACE_SKILL_BINDING_COMMANDS = [
  "agent_runtime_list_workspace_skill_bindings",
];
const LEGACY_REGISTERED_SKILLS_DISCOVERY_COMMANDS = [
  "capability_draft_list_registered_skills",
];

function printHelp() {
  console.log(`
Workspace Skill Bindings Current Smoke

用途:
  通过真实 Skills 用户安装页和 DevBridge /invoke 网络请求，验证已保存技能面板的
  registered skills discovery 与 runtime binding readiness 读取分别走
  App Server JSON-RPC workspaceRegisteredSkills/list 和 workspaceSkillBindings/list
  current 主链，而不是 legacy capability_draft_list_registered_skills、
  agent_runtime_list_workspace_skill_bindings 或 mock。

用法:
  node scripts/workspace-skill-bindings-current-smoke.mjs

选项:
  --app-url <url>        前端地址，默认 http://127.0.0.1:1420/
  --health-url <url>     DevBridge 健康检查地址，默认 http://127.0.0.1:3030/health
  --invoke-url <url>     DevBridge invoke 地址，默认 http://127.0.0.1:3030/invoke
  --timeout-ms <ms>      总超时，默认 120000
  --interval-ms <ms>     健康检查轮询间隔，默认 1000
  --evidence-dir <path>  证据目录，默认 .lime/qc/gui-evidence/workspace-skill-bindings-current
  --prefix <name>        证据文件前缀，默认 workspace-skill-bindings-current
  --headed               使用有界面 Chrome
  -h, --help             显示帮助
`);
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--app-url" && argv[index + 1]) {
      options.appUrl = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--health-url" && argv[index + 1]) {
      options.healthUrl = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--invoke-url" && argv[index + 1]) {
      options.invokeUrl = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--timeout-ms" && argv[index + 1]) {
      options.timeoutMs = Number(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--interval-ms" && argv[index + 1]) {
      options.intervalMs = Number(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--evidence-dir" && argv[index + 1]) {
      options.evidenceDir = path.resolve(String(argv[index + 1]).trim());
      index += 1;
      continue;
    }
    if (arg === "--prefix" && argv[index + 1]) {
      options.prefix = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--headed") {
      options.headless = false;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
  }

  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms 必须是 >= 30000 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
  }
  if (!options.appUrl) {
    throw new Error("--app-url 不能为空");
  }
  if (!options.healthUrl) {
    throw new Error("--health-url 不能为空");
  }
  if (!options.invokeUrl) {
    throw new Error("--invoke-url 不能为空");
  }
  if (!options.evidenceDir) {
    throw new Error("--evidence-dir 不能为空");
  }
  if (!options.prefix) {
    throw new Error("--prefix 不能为空");
  }

  return options;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function logStage(stage) {
  console.log(`[smoke:workspace-skill-bindings-current] stage=${stage}`);
}

async function waitForHealth(options) {
  const startedAt = Date.now();
  let lastError = null;

  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      const response = await fetch(options.healthUrl, {
        signal: AbortSignal.timeout(Math.min(5_000, options.timeoutMs)),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      console.log(
        `[smoke:workspace-skill-bindings-current] DevBridge 已就绪 (${Date.now() - startedAt}ms) status=${payload?.status ?? response.status}`,
      );
      return payload;
    } catch (error) {
      lastError = error;
      await sleep(options.intervalMs);
    }
  }

  const detail =
    lastError instanceof Error
      ? lastError.message
      : String(lastError || "unknown error");
  throw new Error(
    `[smoke:workspace-skill-bindings-current] DevBridge 未就绪，请先启动 npm run electron:dev。最后错误: ${detail}`,
  );
}

function normalizeUrl(value) {
  try {
    const parsed = new URL(value);
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return value;
  }
}

function requestMatchesInvoke(url, invokeUrl) {
  const requestUrl = normalizeUrl(url);
  const expectedUrl = normalizeUrl(invokeUrl);
  return requestUrl === expectedUrl || requestUrl.endsWith("/invoke");
}

function parseJson(value) {
  if (!value) {
    return null;
  }
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function parseJsonRpcLine(line) {
  const trimmed = String(line || "").trim();
  if (!trimmed) {
    return null;
  }
  try {
    const parsed = JSON.parse(trimmed);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function decodeJsonRpcLines(lines) {
  return Array.isArray(lines)
    ? lines.map(parseJsonRpcLine).filter(Boolean)
    : [];
}

function sanitizeText(value) {
  const sanitized = String(value ?? "")
    .replace(
      /((?:api[_-]?key|authorization|password|secret|session|token)[^=\s]*=)(["']?)[^\s"']+/gi,
      "$1$2[redacted]",
    )
    .replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, "$1[redacted]");
  return sanitized.length > 1_200
    ? `${sanitized.slice(0, 1_200)}... [truncated ${sanitized.length - 1_200} chars]`
    : sanitized;
}

function sanitizeJson(value, depth = 0) {
  if (depth > 5) {
    return "[truncated-depth]";
  }
  if (typeof value === "string") {
    return sanitizeText(value);
  }
  if (
    value === null ||
    value === undefined ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value ?? null;
  }
  if (Array.isArray(value)) {
    return value.slice(0, 40).map((item) => sanitizeJson(item, depth + 1));
  }
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .slice(0, 80)
        .map(([key, item]) => [key, sanitizeJson(item, depth + 1)]),
    );
  }
  return sanitizeText(String(value));
}

function collectInvokeEntry(requestPayload, responsePayload, url) {
  const requestLines =
    requestPayload?.request?.lines ??
    requestPayload?.args?.request?.lines ??
    requestPayload?.payload?.lines ??
    requestPayload?.lines;
  const responseLines =
    responsePayload?.result?.lines ??
    responsePayload?.request?.lines ??
    responsePayload?.lines;
  const requestMessages = decodeJsonRpcLines(requestLines);
  const responseMessages = decodeJsonRpcLines(responseLines);
  const responseById = new Map(
    responseMessages
      .filter((message) => message && message.id !== undefined)
      .map((message) => [message.id, message]),
  );

  return {
    url: sanitizeText(url),
    cmd: requestPayload?.cmd ?? null,
    appServerRequests: requestMessages
      .filter((message) => typeof message?.method === "string")
      .map((message) => ({
        id: message.id ?? null,
        method: message.method,
        params: sanitizeJson(message.params ?? {}),
        response: sanitizeJson(responseById.get(message.id) ?? null),
      })),
    responseMessageCount: responseMessages.length,
    responseMessages: responseMessages.map(sanitizeJson),
  };
}

function summarizeInvokeEntries(entries) {
  const appServerRequests = entries.flatMap((entry) => entry.appServerRequests);
  const appServerMethodsSeen = Array.from(
    new Set(appServerRequests.map((request) => request.method)),
  ).sort();
  const workspaceSkillBindingsRequests = appServerRequests.filter(
    (request) => request.method === "workspaceSkillBindings/list",
  );
  const workspaceRegisteredSkillsRequests = appServerRequests.filter(
    (request) => request.method === "workspaceRegisteredSkills/list",
  );
  const workspaceSkillBindingsResponses = workspaceSkillBindingsRequests.map(
    (request) => {
      const bindingPayload = request.response?.result?.bindings;
      const bindings = Array.isArray(bindingPayload?.bindings)
        ? bindingPayload.bindings
        : Array.isArray(bindingPayload)
          ? bindingPayload
          : null;
      return {
        id: request.id,
        hasError: Boolean(request.response?.error),
        bindingsIsArray: Array.isArray(bindings),
        bindingCount: Array.isArray(bindings) ? bindings.length : null,
        hasCounts: Boolean(bindingPayload?.counts),
      };
    },
  );
  const workspaceRegisteredSkillsResponses =
    workspaceRegisteredSkillsRequests.map((request) => {
      const skills = request.response?.result?.skills;
      return {
        id: request.id,
        hasError: Boolean(request.response?.error),
        skillsIsArray: Array.isArray(skills),
        skillCount: Array.isArray(skills) ? skills.length : null,
      };
    });
  const legacyWorkspaceSkillBindingCommandsSeen = Array.from(
    new Set(
      entries
        .map((entry) =>
          typeof entry.cmd === "string" &&
          LEGACY_WORKSPACE_SKILL_BINDING_COMMANDS.includes(entry.cmd)
            ? entry.cmd
            : null,
        )
        .filter(Boolean),
    ),
  ).sort();
  const registeredSkillsDiscoveryCommandsSeen = Array.from(
    new Set(
      entries
        .map((entry) =>
          typeof entry.cmd === "string" &&
          LEGACY_REGISTERED_SKILLS_DISCOVERY_COMMANDS.includes(entry.cmd)
            ? entry.cmd
            : null,
        )
        .filter(Boolean),
    ),
  ).sort();
  const appServerHandleJsonLinesSeen = entries.some(
    (entry) => entry.cmd === APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  );

  return {
    appServerHandleJsonLinesSeen,
    appServerMethodsSeen,
    registeredSkillsDiscoveryCommandsSeen,
    legacyWorkspaceSkillBindingCommandsSeen,
    workspaceRegisteredSkillsRequestCount:
      workspaceRegisteredSkillsRequests.length,
    workspaceRegisteredSkillsRequests: workspaceRegisteredSkillsRequests.map(
      (request) => ({
        id: request.id,
        method: request.method,
        params: request.params,
      }),
    ),
    workspaceRegisteredSkillsResponses,
    workspaceRegisteredSkillsResponsesValid:
      workspaceRegisteredSkillsResponses.length > 0 &&
      workspaceRegisteredSkillsResponses.every(
        (response) => !response.hasError && response.skillsIsArray,
      ),
    workspaceSkillBindingsRequestCount: workspaceSkillBindingsRequests.length,
    workspaceSkillBindingsRequests: workspaceSkillBindingsRequests.map(
      (request) => ({
        id: request.id,
        method: request.method,
        params: request.params,
      }),
    ),
    workspaceSkillBindingsResponses,
    workspaceSkillBindingsResponsesValid:
      workspaceSkillBindingsResponses.length > 0 &&
      workspaceSkillBindingsResponses.every(
        (response) => !response.hasError && response.bindingsIsArray,
      ),
    missingRequiredAppServerMethods: REQUIRED_APP_SERVER_METHODS.filter(
      (method) => !appServerMethodsSeen.includes(method),
    ),
  };
}

async function waitForInvokeEvidence(entries, options) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < Math.min(30_000, options.timeoutMs)) {
    const observed = summarizeInvokeEntries(entries);
    if (
      observed.appServerHandleJsonLinesSeen &&
      observed.missingRequiredAppServerMethods.length === 0 &&
      observed.workspaceRegisteredSkillsRequestCount >= 1 &&
      observed.workspaceRegisteredSkillsResponsesValid &&
      observed.registeredSkillsDiscoveryCommandsSeen.length === 0 &&
      observed.workspaceSkillBindingsRequestCount >= 1 &&
      observed.workspaceSkillBindingsResponsesValid &&
      observed.legacyWorkspaceSkillBindingCommandsSeen.length === 0
    ) {
      return observed;
    }
    await sleep(250);
  }
  return summarizeInvokeEntries(entries);
}

function resolveLocalChromeExecutablePath() {
  const candidates = [
    process.env.PLAYWRIGHT_CHROME_EXECUTABLE,
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
  ].filter(Boolean);

  return candidates.find((candidate) => fs.existsSync(candidate)) ?? undefined;
}

function writeJsonFile(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

async function primeSidebarStorage(page, options) {
  await page.goto(options.appUrl, {
    waitUntil: "domcontentloaded",
    timeout: options.timeoutMs,
  });
  await page.evaluate(() => {
    localStorage.setItem("lime.app-sidebar.collapsed", "false");
  });
}

async function clickSidebarNav(page, options, matcher, label) {
  await primeSidebarStorage(page, options);
  await page.reload({
    waitUntil: "domcontentloaded",
    timeout: options.timeoutMs,
  });
  await page.waitForSelector('[data-testid="app-sidebar"]', {
    timeout: options.timeoutMs,
  });

  const clickNavButton = (scopeSelector) =>
    page.evaluate(
      ({ selector, pattern }) => {
        const scope = selector ? document.querySelector(selector) : document;
        if (!scope) {
          return false;
        }
        const regexp = new RegExp(pattern, "i");
        const buttons = Array.from(scope.querySelectorAll("button"));
        const target = buttons.find((button) => {
          const text = button.textContent || "";
          const aria = button.getAttribute("aria-label") || "";
          const title = button.getAttribute("title") || "";
          return regexp.test(`${text}\n${aria}\n${title}`);
        });
        if (!target) {
          return false;
        }
        target.click();
        return true;
      },
      {
        selector: scopeSelector,
        pattern: matcher.source,
      },
    );

  let clicked = await clickNavButton('[data-testid="app-sidebar"]');

  if (!clicked) {
    const accountButton = page.getByTestId("app-sidebar-account-button");
    if ((await accountButton.count()) > 0) {
      await accountButton.click();
      await page.waitForSelector('[data-testid="app-sidebar-account-menu"]', {
        timeout: options.timeoutMs,
      });
      clicked = await clickNavButton(
        '[data-testid="app-sidebar-account-menu"]',
      );
    }
  }

  assert(clicked, `未找到 ${label} 入口`);
}

async function openSkillsInstalledView(page, options) {
  await clickSidebarNav(page, options, /Skills|技能/i, "Skills");
  await page.waitForSelector("main", { timeout: options.timeoutMs });
  await page.waitForFunction(
    () =>
      Array.from(document.querySelectorAll('button[role="tab"]')).some(
        (button) => {
          const ariaControls = button.getAttribute("aria-controls") || "";
          return [
            "skills-store-view",
            "skills-builtin-view",
            "skills-installed-view",
          ].includes(ariaControls);
        },
      ),
    { timeout: Math.min(30_000, options.timeoutMs) },
  );

  const clickedInstalled = await page.evaluate(() => {
    const tabs = Array.from(document.querySelectorAll('button[role="tab"]'));
    const target = tabs.find((button) => {
      const ariaControls = button.getAttribute("aria-controls") || "";
      const text = button.textContent || "";
      return (
        ariaControls === "skills-installed-view" ||
        /用户安装|已安装|Installed|Manage/i.test(text)
      );
    });
    if (!target) {
      return false;
    }
    target.click();
    return true;
  });
  assert(clickedInstalled, "未找到 Skills 用户安装页 tab");

  await page.waitForSelector('[data-testid="skills-installed-view"]', {
    timeout: options.timeoutMs,
  });
  await page.waitForSelector(
    '[data-testid="workspace-registered-skills-panel"]',
    {
      timeout: options.timeoutMs,
    },
  );
}

async function waitForRegisteredPanelSettled(page, options) {
  await page.waitForFunction(
    () => {
      const panel = document.querySelector(
        '[data-testid="workspace-registered-skills-panel"]',
      );
      if (!panel) {
        return false;
      }
      const text = panel.textContent ?? "";
      return (
        /当前项目还没有已保存技能|还没有已保存技能|No saved skills|No registered skills|已保存技能暂时没读到|正在读取已保存技能|已保存技能|Saved skills/i.test(
          text,
        ) && !/选择或进入一个项目后/.test(text)
      );
    },
    { timeout: Math.min(30_000, options.timeoutMs) },
  );
}

async function run() {
  const options = parseArgs(process.argv.slice(2));
  fs.mkdirSync(options.evidenceDir, { recursive: true });

  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const networkPath = path.join(
    options.evidenceDir,
    `${options.prefix}-network-invoke.json`,
  );
  const screenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}.png`,
  );
  const failureScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-failure.png`,
  );

  let browser = null;
  let tmpProfileDir = null;
  const consoleErrors = [];
  const failedRequests = [];
  const invokeEntries = [];
  const summary = {
    ok: false,
    checkedAt: new Date().toISOString(),
    appUrl: options.appUrl,
    healthUrl: options.healthUrl,
    invokeUrl: options.invokeUrl,
    health: null,
    skillsInstalledViewMounted: false,
    workspaceRegisteredSkillsPanelMounted: false,
    workspaceRegisteredSkillsPanelSettled: false,
    appServerHandleJsonLinesSeen: false,
    appServerMethodsSeen: [],
    registeredSkillsDiscoveryCommandsSeen: [],
    legacyWorkspaceSkillBindingCommandsSeen: [],
    workspaceRegisteredSkillsRequestCount: 0,
    workspaceRegisteredSkillsRequests: [],
    workspaceRegisteredSkillsResponses: [],
    workspaceRegisteredSkillsResponsesValid: false,
    workspaceSkillBindingsRequestCount: 0,
    workspaceSkillBindingsRequests: [],
    workspaceSkillBindingsResponses: [],
    workspaceSkillBindingsResponsesValid: false,
    missingRequiredAppServerMethods: [...REQUIRED_APP_SERVER_METHODS],
    consoleErrors: [],
    failedRequests: [],
    screenshot: null,
    network: networkPath,
    summary: summaryPath,
  };

  try {
    logStage("wait-health");
    summary.health = await waitForHealth(options);

    logStage("launch-browser");
    tmpProfileDir = fs.mkdtempSync(
      path.join(os.tmpdir(), "workspace-skill-bindings-current-"),
    );
    browser = await chromium.launchPersistentContext(tmpProfileDir, {
      headless: options.headless,
      viewport: { width: 1440, height: 1000 },
      executablePath: resolveLocalChromeExecutablePath(),
    });
    const page = await browser.newPage();

    page.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push(sanitizeText(message.text()));
      }
    });
    page.on("requestfailed", (request) => {
      failedRequests.push({
        url: sanitizeText(request.url()),
        method: request.method(),
        failure: sanitizeText(request.failure()?.errorText || "unknown"),
      });
    });
    page.on("response", async (response) => {
      try {
        const request = response.request();
        if (
          request.method() !== "POST" ||
          !requestMatchesInvoke(request.url(), options.invokeUrl)
        ) {
          return;
        }
        const requestPayload = parseJson(request.postData() || "");
        const responsePayload = parseJson(await response.text());
        invokeEntries.push(
          collectInvokeEntry(requestPayload, responsePayload, request.url()),
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (message.includes("No resource with given identifier found")) {
          return;
        }
        console.warn(
          "[smoke:workspace-skill-bindings-current] 解析 invoke 网络证据失败:",
          message,
        );
      }
    });

    logStage("open-skills-installed-view");
    await openSkillsInstalledView(page, options);
    await waitForRegisteredPanelSettled(page, options);

    summary.skillsInstalledViewMounted =
      (await page.locator('[data-testid="skills-installed-view"]').count()) > 0;
    summary.workspaceRegisteredSkillsPanelMounted =
      (await page
        .locator('[data-testid="workspace-registered-skills-panel"]')
        .count()) > 0;
    summary.workspaceRegisteredSkillsPanelSettled = true;

    const observed = await waitForInvokeEvidence(invokeEntries, options);
    Object.assign(summary, observed);
    summary.consoleErrors = consoleErrors;
    summary.failedRequests = failedRequests.slice(0, 20);
    summary.screenshot = screenshotPath;

    await page.screenshot({ path: screenshotPath, fullPage: true });
    writeJsonFile(networkPath, {
      entries: invokeEntries,
      summary: observed,
    });

    assert(summary.skillsInstalledViewMounted, "Skills 用户安装页未挂载");
    assert(
      summary.workspaceRegisteredSkillsPanelMounted,
      "已保存技能面板未挂载",
    );
    assert(
      summary.workspaceRegisteredSkillsPanelSettled,
      "已保存技能面板未稳定到可读状态",
    );
    assert(
      summary.appServerHandleJsonLinesSeen,
      "未观察到 app_server_handle_json_lines",
    );
    assert(
      summary.missingRequiredAppServerMethods.length === 0,
      `缺少 App Server JSON-RPC 方法: ${summary.missingRequiredAppServerMethods.join(", ")}`,
    );
    assert(
      summary.workspaceRegisteredSkillsRequestCount >= 1,
      `workspaceRegisteredSkills/list 请求不足，实际 ${summary.workspaceRegisteredSkillsRequestCount}`,
    );
    assert(
      summary.workspaceRegisteredSkillsResponsesValid,
      "workspaceRegisteredSkills/list response 未返回 skills 数组",
    );
    assert(
      summary.workspaceSkillBindingsRequestCount >= 1,
      `workspaceSkillBindings/list 请求不足，实际 ${summary.workspaceSkillBindingsRequestCount}`,
    );
    assert(
      summary.workspaceSkillBindingsResponsesValid,
      "workspaceSkillBindings/list response 未返回 bindings 数组",
    );
    assert(
      summary.legacyWorkspaceSkillBindingCommandsSeen.length === 0,
      `观察到 legacy workspace skill binding 命令: ${summary.legacyWorkspaceSkillBindingCommandsSeen.join(", ")}`,
    );
    assert(
      summary.registeredSkillsDiscoveryCommandsSeen.length === 0,
      `观察到 legacy registered skills discovery 命令: ${summary.registeredSkillsDiscoveryCommandsSeen.join(", ")}`,
    );

    summary.ok = true;
    writeJsonFile(summaryPath, summary);
    console.log(
      `[smoke:workspace-skill-bindings-current] summary=${summaryPath}`,
    );
    console.log(
      `[smoke:workspace-skill-bindings-current] screenshot=${screenshotPath}`,
    );
  } catch (error) {
    summary.error = error instanceof Error ? error.message : String(error);
    summary.consoleErrors = consoleErrors;
    summary.failedRequests = failedRequests.slice(0, 20);
    const observed = summarizeInvokeEntries(invokeEntries);
    Object.assign(summary, observed);
    writeJsonFile(networkPath, {
      entries: invokeEntries,
      summary: observed,
    });
    writeJsonFile(summaryPath, summary);

    try {
      const pages = browser?.pages?.() ?? [];
      const page = pages.at(-1);
      if (page) {
        await page.screenshot({
          path: failureScreenshotPath,
          fullPage: true,
        });
        summary.screenshot = failureScreenshotPath;
        writeJsonFile(summaryPath, summary);
      }
    } catch {
      // 失败截图只是诊断证据，不能阻断错误上抛。
    }

    console.error(
      `[smoke:workspace-skill-bindings-current] summary=${summaryPath}`,
    );
    throw error;
  } finally {
    if (browser) {
      await browser.close().catch(() => {});
    }
    if (tmpProfileDir) {
      fs.rmSync(tmpProfileDir, { recursive: true, force: true });
    }
  }
}

run().catch((error) => {
  console.error(
    `[smoke:workspace-skill-bindings-current] failed: ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
