#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";

const DEFAULTS = {
  appUrl: "",
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "local-history-import-visual-audit",
  ),
  prefix: "local-history-import-visual-audit",
  timeoutMs: 180_000,
  keepTemp: false,
};

const LOG_PREFIX = "[smoke:local-history-import-visual-audit]";
const CLICK_THROUGH_SCRIPT = "scripts/electron/codex-import-click-through-fixture-smoke.mjs";
const CLICK_THROUGH_PREFIX = "local-history-import-visual-audit-click-through";
const SOURCE_BRAND_PATTERN = /\bcodex\b/i;
const ALLOWED_SOURCE_BRAND_CONTEXTS = [
  "sourceThreadId",
  "requiredMethods",
  "backend",
  "script",
  "trace",
  "rawEvidence",
  "backendLedger",
  "summary",
  "source",
  "rollout",
  "tempRoot",
  "backendPath",
  "backendLedgerPath",
  "appServerBinary",
  "importedDetailsSnapshot",
  "continuationSnapshot",
];
const VISIBLE_TEXT_KEYS = new Set([
  "bodyText",
  "popoverText",
  "taskRailText",
  "shelfText",
  "projectText",
  "recentText",
  "text",
]);

function printHelp() {
  console.log(`
Local History Import Visual Audit Smoke

用途:
  复用真实 Electron 本地历史导入点击闭环，再对采集到的 GUI 可见文本做
  产品边界审计：长历史会话页必须可读、输入框可用、terminal 历史只显示摘要，
  且除导入来源 / provenance / fixture / 协议枚举外，普通可见文本不得
  泄漏来源品牌字眼。

边界:
  该脚本不读取真实用户历史目录，不调用正式模型后端，不使用 renderer
  mock fallback。底层导入协议仍保留 sourceClient / provenance。

用法:
  node scripts/electron/local-history-import-visual-audit-smoke.mjs

选项:
  --app-url <url>        可选 renderer dev server，例如 http://127.0.0.1:1420/
  --evidence-dir <path>  证据目录
  --prefix <name>        证据文件前缀
  --timeout-ms <ms>      总超时，默认 180000
  --keep-temp            保留临时目录便于调试
  -h, --help             显示帮助
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
    if (arg === "--app-url" && next) {
      options.appUrl = next.trim();
      index += 1;
      continue;
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
    if (arg === "--keep-temp") {
      options.keepTemp = true;
      continue;
    }
    throw new Error(`未知参数: ${arg}`);
  }

  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 60_000) {
    throw new Error("--timeout-ms 必须是 >= 60000 的数字");
  }
  if (!options.evidenceDir || !options.prefix) {
    throw new Error("--evidence-dir / --prefix 均不能为空");
  }
  return options;
}

function readJsonFile(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJsonFile(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function isAllowedSourceBrandPath(pathParts) {
  const joined = pathParts.join(".");
  return ALLOWED_SOURCE_BRAND_CONTEXTS.some((fragment) =>
    joined.toLowerCase().includes(fragment.toLowerCase()),
  );
}

function collectVisibleTextLeaks(value, pathParts = []) {
  if (typeof value === "string") {
    const key = pathParts[pathParts.length - 1] || "";
    if (
      VISIBLE_TEXT_KEYS.has(key) &&
      SOURCE_BRAND_PATTERN.test(value) &&
      !isAllowedSourceBrandPath(pathParts)
    ) {
      return [
        {
          path: pathParts.join("."),
          sample: value
            .replace(/\s+/g, " ")
            .replace(/(.{0,100})(codex)(.{0,140})/i, "$1$2$3")
            .slice(0, 260),
        },
      ];
    }
    return [];
  }

  if (!value || typeof value !== "object") {
    return [];
  }

  if (Array.isArray(value)) {
    return value.flatMap((item, index) =>
      collectVisibleTextLeaks(item, [...pathParts, String(index)]),
    );
  }

  return Object.entries(value).flatMap(([key, item]) =>
    collectVisibleTextLeaks(item, [...pathParts, key]),
  );
}

function pickVisibleRawSnapshots(rawEvidence) {
  return {
    rendererSnapshots: rawEvidence.rendererSnapshots,
    previewSnapshot: rawEvidence.previewSnapshot,
    importedPageSnapshot: rawEvidence.importedPageSnapshot,
    importedDetailsSnapshot: rawEvidence.importedDetailsSnapshot,
    importedHistoryBannerSummary: rawEvidence.importedHistoryBannerSummary,
    environmentPopoverSummary: rawEvidence.environmentPopoverSummary,
    sidebarImportDiscoverabilitySummary:
      rawEvidence.sidebarImportDiscoverabilitySummary,
    continuationSnapshot: rawEvidence.continuationSnapshot,
    continuationSummary: rawEvidence.continuationSummary,
  };
}

function summarizeVisualAudit(summary) {
  const visualAudit = Array.isArray(summary.visualAudit)
    ? summary.visualAudit
    : [];
  return visualAudit.map((entry) => ({
    label: entry.label,
    inputbarVisible: entry.inputbarVisible === true,
    inputbarDisabled: entry.inputbarDisabled === true,
    inputbarOccludesMainContent: entry.inputbarOccludesMainContent === true,
    messageListVisible: entry.messageListVisible === true,
    hasImportedUserMessage: entry.hasImportedUserMessage === true,
    hasImportedAssistantMessage: entry.hasImportedAssistantMessage === true,
    hasContinueUserMessage: entry.hasContinueUserMessage === true,
    hasContinueAssistantMessage: entry.hasContinueAssistantMessage === true,
    historicalOperationalDetailsHidden:
      entry.historicalOperationalDetailsHidden === true,
    toolCallRowCount: Number(entry.toolCallRowCount || 0),
    operationalTimelineDetailsCount: Number(
      entry.operationalTimelineDetailsCount || 0,
    ),
    deferredHistoricalPreviewCount: Number(
      entry.deferredHistoricalPreviewCount || 0,
    ),
    historicalTimelinePreviewCount: Number(
      entry.historicalTimelinePreviewCount || 0,
    ),
    hasPatchText: entry.hasPatchText === true,
    importedBannerVisible: entry.importedBannerVisible === true,
    importedRunControlVisible: entry.importedRunControlVisible === true,
    hidesRawImportedCommand: entry.hidesRawImportedCommand === true,
    hidesSourceBrandText: entry.hidesSourceBrandText === true,
    screenshot: entry.screenshot || null,
  }));
}

function assertVisualAudit(visualAudit) {
  if (visualAudit.length < 3) {
    throw new Error("视觉审计缺少 desktop / compact / narrow 三视口证据");
  }

  for (const entry of visualAudit) {
    const label = entry.label || "unknown";
    if (!entry.inputbarVisible || entry.inputbarDisabled) {
      throw new Error(`${label} 视口输入框不可用`);
    }
    if (entry.inputbarOccludesMainContent) {
      throw new Error(`${label} 视口输入框遮挡主内容`);
    }
    if (!entry.messageListVisible) {
      throw new Error(`${label} 视口消息列表不可见`);
    }
    for (const key of [
      "hasImportedUserMessage",
      "hasImportedAssistantMessage",
      "hasContinueUserMessage",
      "hasContinueAssistantMessage",
      "historicalOperationalDetailsHidden",
      "hasPatchText",
      "hidesRawImportedCommand",
      "hidesSourceBrandText",
    ]) {
      if (entry[key] !== true) {
        throw new Error(`${label} 视口 ${key} 未通过`);
      }
    }
    if (
      entry.toolCallRowCount !== 0 ||
      entry.operationalTimelineDetailsCount !== 0 ||
      entry.deferredHistoricalPreviewCount !== 0 ||
      entry.historicalTimelinePreviewCount < 1
    ) {
      throw new Error(
        `${label} terminal 历史必须只保留摘要: ${JSON.stringify({
          toolCallRowCount: entry.toolCallRowCount,
          operationalTimelineDetailsCount: entry.operationalTimelineDetailsCount,
          deferredHistoricalPreviewCount: entry.deferredHistoricalPreviewCount,
          historicalTimelinePreviewCount: entry.historicalTimelinePreviewCount,
        })}`,
      );
    }
    if (entry.importedBannerVisible || entry.importedRunControlVisible) {
      throw new Error(`${label} 视口不应展示导入主线 banner / run control 卡`);
    }
  }
}

function runClickThroughFixture(options) {
  const clickThroughEvidenceDir = path.join(
    options.evidenceDir,
    "click-through",
  );
  fs.mkdirSync(clickThroughEvidenceDir, { recursive: true });

  const args = [
    CLICK_THROUGH_SCRIPT,
    "--evidence-dir",
    clickThroughEvidenceDir,
    "--prefix",
    CLICK_THROUGH_PREFIX,
    "--timeout-ms",
    String(options.timeoutMs),
  ];
  if (options.appUrl) {
    args.push("--app-url", options.appUrl);
  }
  if (options.keepTemp) {
    args.push("--keep-temp");
  }

  const result = spawnSync(process.execPath, args, {
    cwd: process.cwd(),
    env: process.env,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    timeout: options.timeoutMs + 30_000,
  });

  return {
    ok: result.status === 0,
    status: result.status,
    signal: result.signal,
    stdout: result.stdout || "",
    stderr: result.stderr || "",
    evidenceDir: clickThroughEvidenceDir,
    summaryPath: path.join(
      clickThroughEvidenceDir,
      `${CLICK_THROUGH_PREFIX}-summary.json`,
    ),
    rawPath: path.join(
      clickThroughEvidenceDir,
      `${CLICK_THROUGH_PREFIX}-raw.json`,
    ),
  };
}

function run() {
  const options = parseArgs(process.argv.slice(2));
  fs.mkdirSync(options.evidenceDir, { recursive: true });

  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const summary = {
    ok: false,
    checkedAt: new Date().toISOString(),
    appUrl: options.appUrl || null,
    clickThrough: null,
    visualAudit: [],
    visibleTextLeaks: [],
    summary: summaryPath,
  };

  try {
    console.log(`${LOG_PREFIX} stage=click-through-fixture`);
    const clickThrough = runClickThroughFixture(options);
    summary.clickThrough = {
      ok: clickThrough.ok,
      status: clickThrough.status,
      signal: clickThrough.signal,
      evidenceDir: clickThrough.evidenceDir,
      summaryPath: clickThrough.summaryPath,
      rawPath: clickThrough.rawPath,
    };

    if (!clickThrough.ok) {
      throw new Error(
        `点击闭环 fixture 失败 status=${clickThrough.status} signal=${clickThrough.signal}\n${clickThrough.stderr || clickThrough.stdout}`,
      );
    }

    const clickThroughSummary = readJsonFile(clickThrough.summaryPath);
    const clickThroughRaw = readJsonFile(clickThrough.rawPath);
    if (clickThroughSummary.ok !== true) {
      throw new Error("点击闭环 summary 未标记 ok=true");
    }

    console.log(`${LOG_PREFIX} stage=visual-boundary-audit`);
    const visualAudit = summarizeVisualAudit(clickThroughSummary);
    assertVisualAudit(visualAudit);
    const visibleTextLeaks = [
      ...collectVisibleTextLeaks(clickThroughSummary, ["clickThroughSummary"]),
      ...collectVisibleTextLeaks(pickVisibleRawSnapshots(clickThroughRaw), [
        "clickThroughRawVisible",
      ]),
    ];
    if (visibleTextLeaks.length > 0) {
      throw new Error(
        `GUI 可见文本仍泄漏来源品牌: ${JSON.stringify(visibleTextLeaks.slice(0, 5))}`,
      );
    }

    summary.visualAudit = visualAudit;
    summary.visibleTextLeaks = visibleTextLeaks;
    summary.ok = true;
    summary.completedAt = new Date().toISOString();
    writeJsonFile(summaryPath, summary);
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
  } catch (error) {
    summary.error = error instanceof Error ? error.message : String(error);
    writeJsonFile(summaryPath, summary);
    throw error;
  }
}

run();
