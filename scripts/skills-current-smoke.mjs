#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const DEFAULTS = {
  healthUrl: "http://127.0.0.1:3030/health",
  invokeUrl: "http://127.0.0.1:3030/invoke",
  timeoutMs: 120_000,
  intervalMs: 1_000,
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "skills-current",
  ),
  prefix: "skills-current",
  allowLiveProvider: false,
};

const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const REQUIRED_APP_SERVER_METHODS = [
  "skills/list",
  "skill/read",
  "skillManagement/list",
  "skillRepository/list",
  "skillInstalledDirectories/list",
];
const REQUIRED_SKILL_PACKAGE_APP_SERVER_METHODS = [
  "skillLocal/detail/inspect",
  "skillLocal/rename",
  "skillPackage/local/inspect",
  "skillPackage/local/install",
  "skillPackage/local/replace",
  "skillPackage/export",
  "skillMarketplace/install",
  "skillPackage/download/install",
];
const LEGACY_SKILL_READ_COMMANDS = [
  "list_executable_skills",
  "get_skill_detail",
];
const RETIRED_SKILL_MANAGEMENT_FACADE_COMMANDS = [
  "get_local_skills_for_app",
  "get_skills_for_app",
  "install_skill_for_app",
  "uninstall_skill_for_app",
  "get_skill_repos",
  "add_skill_repo",
  "remove_skill_repo",
  "refresh_skill_cache",
  "get_installed_lime_skills",
  "inspect_local_skill_for_app",
  "inspect_local_skill_detail_for_app",
  "rename_local_skill_for_app",
  "replace_local_skill_package_for_app",
  "create_skill_scaffold_for_app",
  "import_local_skill_for_app",
  "take_pending_skill_package_open_requests",
  "get_skill_package_file_association_status",
  "set_skill_package_file_association_default",
  "install_marketplace_skill_for_app",
  "install_skill_from_download_url_for_app",
  "inspect_remote_skill",
];
const LEGACY_SKILL_COMMANDS = [
  ...LEGACY_SKILL_READ_COMMANDS,
  ...RETIRED_SKILL_MANAGEMENT_FACADE_COMMANDS,
];
const FORBIDDEN_APP_SERVER_METHODS = ["turn/start"];
const FORBIDDEN_SIDE_EFFECT_COMMANDS = [
  "execute_skill",
  "agent_runtime_submit_turn",
];

function printHelp() {
  console.log(`
Skills Current Smoke

用途:
  直接通过 DevBridge /invoke 调用 app_server_handle_json_lines，验证
  skills/list、Skill 管理读链和 Skill package / marketplace current
  探针走 App Server JSON-RPC current 主链，而不是 legacy
  list_executable_skills / get_skill_detail / Skill 管理 / package 命令或 mock。
  普通 slash skill 当前已回到 Agent Runtime turn，不再作为 skills/list
  preflight 的成功证据。

用法:
  node scripts/skills-current-smoke.mjs

选项:
  --health-url <url>     DevBridge 健康检查地址，默认 http://127.0.0.1:3030/health
  --invoke-url <url>     DevBridge invoke 地址，默认 http://127.0.0.1:3030/invoke
  --timeout-ms <ms>      总超时，默认 120000
  --interval-ms <ms>     健康检查轮询间隔，默认 1000
  --evidence-dir <path>  证据目录，默认 .lime/qc/gui-evidence/skills-current
  --prefix <name>        证据文件前缀，默认 skills-current
  -h, --help             显示帮助
`);
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
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
    if ((arg === "--app-url" || arg === "--probe-command") && argv[index + 1]) {
      index += 1;
      continue;
    }
    if (arg === "--allow-live-provider") {
      options.allowLiveProvider = true;
      continue;
    }
    if (arg === "--headed") {
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
  console.log(`[smoke:skills-current] stage=${stage}`);
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
        `[smoke:skills-current] DevBridge 已就绪 (${Date.now() - startedAt}ms) status=${payload?.status ?? response.status}`,
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
    `[smoke:skills-current] DevBridge 未就绪，请先启动 npm run electron:dev。最后错误: ${detail}`,
  );
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

function sanitizeAppServerResponse(method, response) {
  const sanitized = sanitizeJson(response);
  if (method !== "skill/read" || !sanitized?.result?.skill) {
    return sanitized;
  }
  const skill = sanitized.result.skill;
  const metadata = skill.metadata ?? {};
  return {
    ...sanitized,
    result: {
      skill: {
        metadata: {
          skillId: metadata.skillId ?? null,
          name: metadata.name ?? null,
          scope: metadata.scope ?? null,
          source: metadata.source ?? null,
          authority: metadata.authority ?? null,
          enabled: metadata.enabled ?? null,
        },
        markdownContent: "[redacted]",
        workflowStepCount: Array.isArray(skill.workflowSteps)
          ? skill.workflowSteps.length
          : null,
      },
    },
  };
}

function collectInvokeEntry(requestPayload, responsePayload, url) {
  const requestLines =
    requestPayload?.request?.lines ??
    requestPayload?.args?.request?.lines ??
    requestPayload?.payload?.lines ??
    requestPayload?.lines;
  const responseLines =
    responsePayload?.result?.result?.lines ??
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
      .map((message) => {
        const response =
          responseById.get(message.id) ??
          (responsePayload?.error
            ? {
                id: message.id ?? null,
                error: {
                  message: String(responsePayload.error),
                },
              }
            : null);
        return {
          id: message.id ?? null,
          method: message.method,
          params: sanitizeJson(message.params ?? {}),
          response: sanitizeAppServerResponse(message.method, response),
        };
      }),
    responseMessageCount: responseMessages.length,
    responseMessages: responseMessages.map(sanitizeJson),
  };
}

function parseAppServerResponseMessages(responsePayload) {
  const responseLines =
    responsePayload?.result?.result?.lines ??
    responsePayload?.result?.lines ??
    responsePayload?.lines;
  return decodeJsonRpcLines(responseLines);
}

function summarizeInvokeEntries(entries) {
  const appServerRequests = entries.flatMap((entry) => entry.appServerRequests);
  const appServerMethodsSeen = Array.from(
    new Set(appServerRequests.map((request) => request.method)),
  ).sort();
  const summarizeAppServerRequest = (request) => ({
    id: request.id,
    method: request.method,
    params: request.params,
  });
  const skillListRequests = appServerRequests.filter(
    (request) => request.method === "skills/list",
  );
  const skillListResponses = skillListRequests.map((request) => ({
    id: request.id,
    hasError: Boolean(request.response?.error),
    dataIsArray: Array.isArray(request.response?.result?.data),
    skillCount: Array.isArray(request.response?.result?.data)
      ? request.response.result.data.reduce(
          (count, entry) =>
            count + (Array.isArray(entry?.skills) ? entry.skills.length : 0),
          0,
        )
      : null,
  }));
  const skillReadRequests = appServerRequests.filter(
    (request) => request.method === "skill/read",
  );
  const skillReadResponses = skillReadRequests.map((request) => {
    const requestedSkillId = request.params?.skillId ?? null;
    const responseSkillId =
      request.response?.result?.skill?.metadata?.skillId ?? null;
    return {
      id: request.id,
      requestedSkillId,
      responseSkillId,
      hasError: Boolean(request.response?.error),
      identityMatches:
        typeof requestedSkillId === "string" &&
        requestedSkillId.length > 0 &&
        requestedSkillId === responseSkillId,
    };
  });
  const skillPackageRequests = appServerRequests.filter((request) =>
    REQUIRED_SKILL_PACKAGE_APP_SERVER_METHODS.includes(request.method),
  );
  const skillPackageProbeResponses = skillPackageRequests.map((request) => ({
    id: request.id,
    method: request.method,
    failClosed: Boolean(request.response?.error),
    errorMessage:
      typeof request.response?.error?.message === "string"
        ? request.response.error.message
        : null,
  }));
  const forbiddenAppServerMethodsSeen = appServerMethodsSeen.filter((method) =>
    FORBIDDEN_APP_SERVER_METHODS.includes(method),
  );
  const forbiddenSideEffectCommandsSeen = Array.from(
    new Set(
      entries
        .map((entry) =>
          typeof entry.cmd === "string" &&
          FORBIDDEN_SIDE_EFFECT_COMMANDS.includes(entry.cmd)
            ? entry.cmd
            : null,
        )
        .filter(Boolean),
    ),
  ).sort();
  const legacySkillCommandsSeen = Array.from(
    new Set(
      entries
        .map((entry) =>
          typeof entry.cmd === "string" &&
          LEGACY_SKILL_COMMANDS.includes(entry.cmd)
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
    forbiddenAppServerMethodsSeen,
    forbiddenSideEffectCommandsSeen,
    legacySkillCommandsSeen,
    legacySkillCommandBoundary: {
      readCommands: [...LEGACY_SKILL_READ_COMMANDS],
      retiredManagementFacadeCommands: [
        ...RETIRED_SKILL_MANAGEMENT_FACADE_COMMANDS,
      ],
      currentSkillPackageMethods: [
        ...REQUIRED_SKILL_PACKAGE_APP_SERVER_METHODS,
      ],
      classification:
        "P10 Skill current smoke must not use retired Skill management, package, marketplace or file association commands as success evidence.",
    },
    skillListRequestCount: skillListRequests.length,
    skillListRequests: skillListRequests.map(summarizeAppServerRequest),
    skillListResponses,
    skillReadRequestCount: skillReadRequests.length,
    skillReadRequests: skillReadRequests.map(summarizeAppServerRequest),
    skillReadResponses,
    skillReadResponsesValid:
      skillReadResponses.length > 0 &&
      skillReadResponses.every(
        (response) => !response.hasError && response.identityMatches,
      ),
    skillPackageProbeRequestCount: skillPackageRequests.length,
    skillPackageProbeRequests: skillPackageRequests.map(
      summarizeAppServerRequest,
    ),
    skillPackageProbeResponses,
    missingRequiredAppServerMethods: REQUIRED_APP_SERVER_METHODS.filter(
      (method) => !appServerMethodsSeen.includes(method),
    ),
    missingRequiredSkillPackageAppServerMethods:
      REQUIRED_SKILL_PACKAGE_APP_SERVER_METHODS.filter(
        (method) => !appServerMethodsSeen.includes(method),
      ),
    skillPackageProbeResponsesFailClosed:
      skillPackageProbeResponses.length ===
        REQUIRED_SKILL_PACKAGE_APP_SERVER_METHODS.length &&
      skillPackageProbeResponses.every((response) => response.failClosed),
    skillListResponsesValid:
      skillListResponses.length > 0 &&
      skillListResponses.every(
        (response) => !response.hasError && response.dataIsArray,
      ),
  };
}

function writeJsonFile(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

async function invokeBridgeCommand(
  options,
  cmd,
  args,
  entries,
  { allowTopLevelError = false } = {},
) {
  const requestPayload = { cmd, args };
  const response = await fetch(options.invokeUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(requestPayload),
    signal: AbortSignal.timeout(options.timeoutMs),
  });
  const text = await response.text();
  const responsePayload = parseJson(text);
  if (responsePayload) {
    entries.push(
      collectInvokeEntry(requestPayload, responsePayload, options.invokeUrl),
    );
  }
  if (!response.ok) {
    throw new Error(`${cmd} HTTP ${response.status}: ${sanitizeText(text)}`);
  }
  if (responsePayload?.error && !allowTopLevelError) {
    throw new Error(`${cmd} error: ${sanitizeText(responsePayload.error)}`);
  }
  if (!responsePayload) {
    throw new Error(`${cmd} returned non-JSON response`);
  }
  return responsePayload;
}

let appServerRequestId = 1;

async function invokeAppServerMethod(options, method, params, entries) {
  const id = `skills-current-${appServerRequestId++}`;
  const request =
    params === undefined ? { id, method } : { id, method, params };
  const responsePayload = await invokeBridgeCommand(
    options,
    APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    { request: { lines: [`${JSON.stringify(request)}\n`] } },
    entries,
  );
  const messages = parseAppServerResponseMessages(responsePayload);
  const error = messages.find((message) => message.id === id && message.error);
  if (error) {
    throw new Error(
      `${method} error: ${error.error?.message || "App Server JSON-RPC error"}`,
    );
  }
  const response = messages.find(
    (message) => message.id === id && Object.hasOwn(message, "result"),
  );
  if (!response) {
    throw new Error(`${method} missing App Server response`);
  }
  return response.result;
}

async function invokeAppServerMethodExpectingError(
  options,
  method,
  params,
  entries,
) {
  const id = `skills-current-${appServerRequestId++}`;
  const request =
    params === undefined ? { id, method } : { id, method, params };
  const responsePayload = await invokeBridgeCommand(
    options,
    APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    { request: { lines: [`${JSON.stringify(request)}\n`] } },
    entries,
    { allowTopLevelError: true },
  );
  const messages = parseAppServerResponseMessages(responsePayload);
  const error = messages.find((message) => message.id === id && message.error);
  if (error) {
    return error.error;
  }
  if (responsePayload?.error) {
    return { message: String(responsePayload.error) };
  }
  throw new Error(`${method} expected fail-closed App Server error`);
}

async function invokeSkillPackageCurrentProbes(options, entries) {
  const invalidApp = "__skills_current_smoke_no_side_effect__";
  const missingSourcePath = path.join(
    options.evidenceDir,
    "__missing_skill_package__.skill",
  );
  const missingExportPath = path.join(
    options.evidenceDir,
    "__missing_skill_export__.skill",
  );
  const probes = [
    {
      stage: "invoke-skill-local-detail-inspect-probe",
      method: "skillLocal/detail/inspect",
      params: {
        app: invalidApp,
        directory: "__missing_skill__",
      },
    },
    {
      stage: "invoke-skill-local-rename-probe",
      method: "skillLocal/rename",
      params: {
        app: invalidApp,
        directory: "__missing_skill__",
        newDirectory: "__missing_skill_renamed__",
      },
    },
    {
      stage: "invoke-skill-package-local-inspect-probe",
      method: "skillPackage/local/inspect",
      params: {
        app: invalidApp,
        sourcePath: missingSourcePath,
      },
    },
    {
      stage: "invoke-skill-package-local-install-probe",
      method: "skillPackage/local/install",
      params: {
        app: invalidApp,
        sourcePath: missingSourcePath,
      },
    },
    {
      stage: "invoke-skill-package-local-replace-probe",
      method: "skillPackage/local/replace",
      params: {
        app: invalidApp,
        directory: "__missing_skill__",
        sourcePath: missingSourcePath,
      },
    },
    {
      stage: "invoke-skill-package-export-probe",
      method: "skillPackage/export",
      params: {
        app: invalidApp,
        directory: "__missing_skill__",
        targetPath: missingExportPath,
      },
    },
    {
      stage: "invoke-skill-marketplace-install-probe",
      method: "skillMarketplace/install",
      params: {
        app: invalidApp,
        manifestVersion: "agentskills.v1",
        name: "__missing_marketplace_skill__",
        aliases: [],
        version: "0.0.0",
        contentHash: "",
        fileCount: 1,
        files: [
          {
            path: "SKILL.md",
            content:
              "---\nname: Smoke Probe\ndescription: No side effect\n---\n",
          },
        ],
      },
    },
    {
      stage: "invoke-skill-package-download-install-probe",
      method: "skillPackage/download/install",
      params: {
        app: invalidApp,
        skillName: "__missing_download_skill__",
        downloadUrl: "https://example.invalid/__missing_skill__.skill",
      },
    },
  ];

  for (const probe of probes) {
    logStage(probe.stage);
    await invokeAppServerMethodExpectingError(
      options,
      probe.method,
      probe.params,
      entries,
    );
  }
}

async function run() {
  const options = parseArgs(process.argv.slice(2));
  if (!options.allowLiveProvider) {
    console.log(
      "[smoke:skills-current] live_provider_submission=status:not_submitted reason:默认未提交真实 Agent Runtime / Provider 请求；本 smoke 只验证 Skill list/read 与 package/marketplace current 探针。",
    );
  }
  fs.mkdirSync(options.evidenceDir, { recursive: true });

  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const networkPath = path.join(
    options.evidenceDir,
    `${options.prefix}-network-invoke.json`,
  );

  const invokeEntries = [];
  const summary = {
    ok: false,
    checkedAt: new Date().toISOString(),
    healthUrl: options.healthUrl,
    invokeUrl: options.invokeUrl,
    smokeMode: "direct-devbridge-app-server-json-rpc",
    retiredSlashPreflight:
      "ordinary slash skill now enters Agent Runtime turn and is not skills/list preflight evidence",
    health: null,
    appServerHandleJsonLinesSeen: false,
    appServerMethodsSeen: [],
    forbiddenAppServerMethodsSeen: [],
    forbiddenSideEffectCommandsSeen: [],
    legacySkillCommandsSeen: [],
    legacySkillCommandBoundary: {
      readCommands: [...LEGACY_SKILL_READ_COMMANDS],
      retiredManagementFacadeCommands: [
        ...RETIRED_SKILL_MANAGEMENT_FACADE_COMMANDS,
      ],
      currentSkillPackageMethods: [
        ...REQUIRED_SKILL_PACKAGE_APP_SERVER_METHODS,
      ],
      classification:
        "P10 Skill current smoke must not use retired Skill management, package, marketplace or file association commands as success evidence.",
    },
    skillListRequestCount: 0,
    skillListRequests: [],
    skillListResponses: [],
    skillListResponsesValid: false,
    skillReadRequestCount: 0,
    skillReadRequests: [],
    skillReadResponses: [],
    skillReadResponsesValid: false,
    skillPackageProbeRequestCount: 0,
    skillPackageProbeRequests: [],
    skillPackageProbeResponses: [],
    skillPackageProbeResponsesFailClosed: false,
    missingRequiredAppServerMethods: [...REQUIRED_APP_SERVER_METHODS],
    missingRequiredSkillPackageAppServerMethods: [
      ...REQUIRED_SKILL_PACKAGE_APP_SERVER_METHODS,
    ],
    network: networkPath,
    summary: summaryPath,
  };

  try {
    logStage("wait-health");
    summary.health = await waitForHealth(options);

    logStage("invoke-skill-list");
    const skillList = await invokeAppServerMethod(
      options,
      "skills/list",
      {},
      invokeEntries,
    );
    const readableSkill = skillList?.data
      ?.flatMap((entry) => (Array.isArray(entry?.skills) ? entry.skills : []))
      .find(
        (skill) =>
          skill?.enabled === true &&
          typeof skill?.name === "string" &&
          ["repo", "user", "system", "admin"].includes(skill?.scope),
      );
    const stableScope = {
      repo: "project",
      user: "user",
      system: "app",
      admin: "other",
    }[readableSkill?.scope];
    const readableSkillId =
      stableScope && readableSkill?.name
        ? `${stableScope}:${readableSkill.name.trim().toLowerCase()}`
        : null;
    assert(
      typeof readableSkillId === "string",
      "skills/list 未返回可用于 stable-id read probe 的 enabled skill",
    );

    logStage("invoke-skill-read");
    await invokeAppServerMethod(
      options,
      "skill/read",
      { skillId: readableSkillId },
      invokeEntries,
    );

    logStage("invoke-skill-management-list");
    await invokeAppServerMethod(
      options,
      "skillManagement/list",
      { app: "lime", refreshRemote: false },
      invokeEntries,
    );

    logStage("invoke-skill-repository-list");
    await invokeAppServerMethod(
      options,
      "skillRepository/list",
      {},
      invokeEntries,
    );

    logStage("invoke-skill-installed-directories-list");
    await invokeAppServerMethod(
      options,
      "skillInstalledDirectories/list",
      {},
      invokeEntries,
    );

    await invokeSkillPackageCurrentProbes(options, invokeEntries);

    const observed = summarizeInvokeEntries(invokeEntries);
    Object.assign(summary, observed);

    writeJsonFile(networkPath, {
      entries: invokeEntries,
      summary: observed,
    });

    assert(
      summary.health?.transport === "electron-host",
      `DevBridge transport 应为 electron-host，实际 ${summary.health?.transport ?? "unknown"}`,
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
      summary.missingRequiredSkillPackageAppServerMethods.length === 0,
      `缺少 Skill package / marketplace App Server JSON-RPC 方法: ${summary.missingRequiredSkillPackageAppServerMethods.join(", ")}`,
    );
    assert(
      summary.skillPackageProbeResponsesFailClosed,
      "Skill package / marketplace current 探针必须在 App Server current owner 内 fail closed",
    );
    assert(
      summary.skillListRequestCount >= 1,
      `skills/list 请求不足，实际 ${summary.skillListRequestCount}`,
    );
    assert(
      summary.skillListResponsesValid,
      "skills/list response 缺少 result.data 数组或返回错误",
    );
    assert(
      summary.skillReadRequestCount >= 1,
      `skill/read 请求不足，实际 ${summary.skillReadRequestCount}`,
    );
    assert(
      summary.skillReadResponsesValid,
      "skill/read 必须使用 stable skillId 且响应 metadata.skillId 必须一致",
    );
    assert(
      summary.forbiddenAppServerMethodsSeen.length === 0,
      `Skill list/read current smoke 不应启动 Agent turn: ${summary.forbiddenAppServerMethodsSeen.join(", ")}`,
    );
    assert(
      summary.forbiddenSideEffectCommandsSeen.length === 0,
      `Skill list/read current smoke 不应执行 side-effect 命令: ${summary.forbiddenSideEffectCommandsSeen.join(", ")}`,
    );
    assert(
      summary.legacySkillCommandsSeen.length === 0,
      `观察到 legacy skill 命令: ${summary.legacySkillCommandsSeen.join(", ")}`,
    );

    summary.ok = true;
    writeJsonFile(summaryPath, summary);
    console.log(`[smoke:skills-current] summary=${summaryPath}`);
  } catch (error) {
    summary.error = error instanceof Error ? error.message : String(error);
    const observed = summarizeInvokeEntries(invokeEntries);
    Object.assign(summary, observed);
    writeJsonFile(networkPath, {
      entries: invokeEntries,
      summary: observed,
    });
    writeJsonFile(summaryPath, summary);

    console.error(`[smoke:skills-current] summary=${summaryPath}`);
    throw error;
  }
}

run().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
