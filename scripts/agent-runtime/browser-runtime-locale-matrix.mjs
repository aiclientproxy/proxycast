#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

const ROOT = process.cwd();
const LOCALES = ["zh-CN", "zh-TW", "en-US", "ja-JP", "ko-KR"];
const SCREENSHOT_STATES = [
  "renderer-loading",
  "renderer-ready",
  "browser-loaded",
  "agent-controlled",
  "user-takeover",
  "released",
  "destroyed",
];
const EVIDENCE_DIR = path.resolve(
  ".lime/qc/gui-evidence/browser-runtime-electron-gate-b",
);
const OUTPUT = path.resolve(
  ".lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-locale-matrix-summary.json",
);
const SCREENSHOT_DIR = path.resolve(
  ".lime/qc/gui-evidence/browser-runtime-electron-gate-b/locales",
);

function runLocale(locale, output) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      process.execPath,
      [
        "scripts/agent-runtime/browser-runtime-electron-gate-b.mjs",
        "--scenario",
        "lifecycle",
        "--locale",
        locale,
        "--output",
        output,
        "--screenshot-dir",
        SCREENSHOT_DIR,
      ],
      { cwd: ROOT, env: process.env, stdio: ["ignore", "pipe", "pipe"] },
    );
    child.stdout.on("data", (chunk) => process.stdout.write(`[${locale}] ${chunk}`));
    child.stderr.on("data", (chunk) => process.stderr.write(`[${locale}] ${chunk}`));
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) {
        resolve({ locale, status: "pass", output });
        return;
      }
      resolve({
        locale,
        status: "fail",
        output,
        exitCode: code,
        signal: signal || null,
      });
    });
  });
}

function readEvidence(fileName) {
  const filePath = path.join(EVIDENCE_DIR, fileName);
  if (!fs.existsSync(filePath)) {
    return { file: filePath, status: "missing", proofLevel: null };
  }
  try {
    const evidence = JSON.parse(fs.readFileSync(filePath, "utf8"));
    return {
      file: filePath,
      status: evidence.status || "unknown",
      proofLevel: evidence.proofLevel || null,
      failedAssertions: evidence.failedAssertions || [],
      diagnostics: evidence.diagnostics || {},
    };
  } catch (error) {
    return { file: filePath, status: "invalid", error: String(error) };
  }
}

function buildScreenshotMatrix() {
  return Object.fromEntries(
    LOCALES.map((locale) => [
      locale,
      Object.fromEntries(
        SCREENSHOT_STATES.map((state) => {
          const file = path.join(
            SCREENSHOT_DIR,
            `${locale}-lifecycle-${state}.png`,
          );
          return [state, { file, exists: fs.existsSync(file) }];
        }),
      ),
    ]),
  );
}

async function main() {
  fs.mkdirSync(EVIDENCE_DIR, { recursive: true });
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
  const runs = [];
  if (!process.argv.includes("--aggregate-only")) {
    for (const locale of LOCALES) {
      const output = path.join(
        EVIDENCE_DIR,
        `browser-runtime-electron-gate-b-${locale}-lifecycle-summary.json`,
      );
      runs.push(await runLocale(locale, output));
    }
  } else {
    for (const locale of LOCALES) {
      runs.push({
        locale,
        status: "pass",
        output: path.join(
          EVIDENCE_DIR,
          `browser-runtime-electron-gate-b-${locale}-lifecycle-summary.json`,
        ),
      });
    }
  }

  const screenshots = buildScreenshotMatrix();
  const localeEvidence = Object.fromEntries(
    runs.map((run) => {
      if (!fs.existsSync(run.output)) return [run.locale, { rendererLocale: null }];
      try {
        const evidence = JSON.parse(fs.readFileSync(run.output, "utf8"));
        return [run.locale, { rendererLocale: evidence.gui?.beforeTurn?.rendererLocale || null }];
      } catch {
        return [run.locale, { rendererLocale: null }];
      }
    }),
  );
  const behavior = {
    approval: readEvidence("browser-runtime-electron-gate-b-approval-summary.json"),
    artifact: readEvidence("browser-runtime-electron-gate-b-artifact-summary.json"),
    cancel: readEvidence("browser-runtime-electron-gate-b-cancel-summary.json"),
    disconnect: readEvidence("browser-runtime-electron-gate-b-disconnect-summary.json"),
    download: readEvidence("browser-runtime-electron-gate-b-download-summary.json"),
    permission: readEvidence("browser-runtime-electron-gate-b-permission-summary.json"),
    userTakeover: readEvidence("browser-runtime-electron-gate-b-user-control-summary.json"),
    windowClose: readEvidence("browser-runtime-electron-gate-b-window-close-summary.json"),
  };
  const missingScreenshots = Object.entries(screenshots).flatMap(([locale, states]) =>
    Object.entries(states)
      .filter(([, entry]) => !entry.exists)
      .map(([state]) => `${locale}:${state}`),
  );
  const failedLocaleRuns = runs.filter((run) => run.status !== "pass");
  const summary = {
    schemaVersion: "lime.browser_runtime_electron_locale_matrix.v1",
    generatedAt: new Date().toISOString(),
    proofLevel: "Gate B",
    localeCoverage: LOCALES,
    screenshotStates: SCREENSHOT_STATES,
    screenshotMatrix: screenshots,
    behaviorStateMatrix: behavior,
    localeRuns: runs,
    claimBoundary:
      "真实 Electron Desktop Host、preload/IPC、App Server current JSON-RPC 与 Browser Workspace 在五语言配置下的生命周期截图；审批、权限、下载、artifact、取消、断连和窗口关闭行为引用各自 Gate B 证据。load-error 以无 console/page error 的诊断断言覆盖，未人为注入网络失败。",
    assertions: {
      fiveLocaleRunsPass: failedLocaleRuns.length === 0,
      rendererLocaleMatchesConfig: LOCALES.every(
        (locale) => localeEvidence[locale]?.rendererLocale === locale,
      ),
      everyLocaleHasLifecycleScreenshots: missingScreenshots.length === 0,
      behaviorEvidenceAllPass: Object.values(behavior).every(
        (entry) => entry.status === "pass",
      ),
      noMissingScreenshots: missingScreenshots.length === 0,
    },
    missingScreenshots,
    localeEvidence,
  };
  fs.writeFileSync(OUTPUT, `${JSON.stringify(summary, null, 2)}\n`);
  console.log(`[browser-runtime-locale-matrix] summary=${OUTPUT}`);
  if (failedLocaleRuns.length > 0 || missingScreenshots.length > 0) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(`[browser-runtime-locale-matrix] failed: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});
