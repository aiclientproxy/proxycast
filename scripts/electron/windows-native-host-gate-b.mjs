#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";
import process from "node:process";

const LOG_PREFIX = "[smoke:windows-native-host-gate-b]";
const DEFAULT_TIMEOUT_MS = 120_000;
const RESOURCE_MANIFEST = "desktop-resources.manifest.json";

export function parseArgs(argv) {
  const options = {
    electronExecutable: null,
    evidenceDir: path.resolve(
      ".lime/qc/gui-evidence/windows-native-host-gate-b",
    ),
    timeoutMs: DEFAULT_TIMEOUT_MS,
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "-h" || arg === "--help") {
      options.help = true;
      continue;
    }
    if (arg === "--electron-executable" && argv[index + 1]) {
      options.electronExecutable = path.resolve(argv[++index]);
      continue;
    }
    if (arg === "--evidence-dir" && argv[index + 1]) {
      options.evidenceDir = path.resolve(argv[++index]);
      continue;
    }
    if (arg === "--timeout-ms" && argv[index + 1]) {
      options.timeoutMs = Number(argv[++index]);
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (options.help) return options;
  if (process.platform !== "win32") {
    throw new Error("Windows native host Gate B requires a Windows runner");
  }
  if (!options.electronExecutable) {
    throw new Error("--electron-executable is required");
  }
  if (!existsSync(options.electronExecutable)) {
    throw new Error(
      `Installed Electron executable does not exist: ${options.electronExecutable}`,
    );
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms must be >= 30000");
  }
  return options;
}

export function resolveInstalledResourcesRoot(electronExecutable) {
  const executable = path.resolve(electronExecutable);
  const resourcesRoot = path.join(path.dirname(executable), "resources");
  if (!existsSync(resourcesRoot) || !statSync(resourcesRoot).isDirectory()) {
    throw new Error(
      `Installed Electron resources directory is missing: ${resourcesRoot}`,
    );
  }
  return resourcesRoot;
}

function sha256(filePath) {
  return createHash("sha256").update(readFileSync(filePath)).digest("hex");
}

function readInstalledHelper(resourcesRoot) {
  const manifestPath = path.join(resourcesRoot, RESOURCE_MANIFEST);
  if (!existsSync(manifestPath)) {
    throw new Error(
      `Installed desktop resource manifest is missing: ${manifestPath}`,
    );
  }
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  if (
    manifest.schemaVersion !== 1 ||
    manifest.applicationId !== "com.limecloud.lime" ||
    manifest.platform !== "win32" ||
    manifest.arch !== "x64" ||
    manifest.platformKey !== "win32-x64"
  ) {
    throw new Error("Installed desktop resource manifest identity is invalid");
  }
  const metadata = manifest.native?.windowsHelper;
  if (
    metadata?.id !== "windows-native-host" ||
    metadata.readOnly !== true ||
    metadata.path !== "native/windows/windows-native-host.exe"
  ) {
    throw new Error("Installed Windows native helper metadata is invalid");
  }
  const resource = manifest.resources?.find(
    (candidate) => candidate?.id === metadata.id,
  );
  if (!resource || resource.path !== metadata.path) {
    throw new Error(
      "Installed Windows native helper resource is not registered",
    );
  }
  const helperPath = path.resolve(resourcesRoot, metadata.path);
  if (!existsSync(helperPath) || !statSync(helperPath).isFile()) {
    throw new Error(
      `Installed Windows native helper is missing: ${helperPath}`,
    );
  }
  const packagedSha256 = sha256(helperPath);
  if (packagedSha256 !== resource.sha256) {
    throw new Error("Installed Windows native helper sha256 mismatch");
  }
  return {
    helperPath,
    manifest,
    sha256: packagedSha256,
  };
}

class NativeHostClient {
  #child;
  #buffer = "";
  #nextId = 1;
  #pending = new Map();
  #events = [];
  #timeoutMs;

  constructor(helperPath, timeoutMs) {
    this.#child = spawn(helperPath, [], {
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    this.#timeoutMs = timeoutMs;
    this.#child.stdout.setEncoding("utf8");
    this.#child.stdout.on("data", (chunk) => this.#consume(chunk));
    this.#child.stderr.setEncoding("utf8");
    this.#child.stderr.on("data", () => undefined);
    this.#child.once("error", (error) => this.#failAll(error));
    this.#child.once("exit", (code, signal) => {
      this.#failAll(
        new Error(
          `Windows native host exited code=${code ?? ""} signal=${signal ?? ""}`,
        ),
      );
    });
  }

  async invoke(method, params = {}) {
    const id = String(this.#nextId++);
    return await new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.#pending.delete(id);
        reject(new Error(`Windows native host request timed out: ${method}`));
      }, this.#timeoutMs);
      this.#pending.set(id, { resolve, reject, timer });
      this.#child.stdin.write(
        `${JSON.stringify({ id, method, params })}\n`,
        "utf8",
        (error) => {
          if (error) this.#settle(id, null, error);
        },
      );
    });
  }

  events() {
    return [...this.#events];
  }

  async close() {
    this.#child.stdin.end();
    if (!this.#child.killed) this.#child.kill();
    this.#failAll(new Error("Windows native host stopped"));
  }

  #consume(chunk) {
    this.#buffer += chunk;
    while (true) {
      const newline = this.#buffer.indexOf("\n");
      if (newline < 0) return;
      const line = this.#buffer.slice(0, newline).trim();
      this.#buffer = this.#buffer.slice(newline + 1);
      if (!line) continue;
      let response;
      try {
        response = JSON.parse(line);
      } catch {
        continue;
      }
      if (typeof response.event === "string") {
        this.#events.push({ event: response.event, payload: response.payload });
        continue;
      }
      if (typeof response.id !== "string") continue;
      if (response.ok === true) {
        this.#settle(response.id, response.result, null);
      } else {
        this.#settle(
          response.id,
          null,
          new Error(
            response.error?.message || "Windows native host request failed",
          ),
        );
      }
    }
  }

  #settle(id, value, error) {
    const pending = this.#pending.get(id);
    if (!pending) return;
    this.#pending.delete(id);
    clearTimeout(pending.timer);
    if (error) pending.reject(error);
    else pending.resolve(value);
  }

  #failAll(error) {
    for (const id of this.#pending.keys()) this.#settle(id, null, error);
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

export async function runWindowsNativeHostGateB(options) {
  const resourcesRoot = resolveInstalledResourcesRoot(
    options.electronExecutable,
  );
  const helper = readInstalledHelper(resourcesRoot);
  const client = new NativeHostClient(helper.helperPath, options.timeoutMs);
  const checks = [];
  let observedEventTypes = [];
  let failure = null;
  try {
    const capabilities = await client.invoke(
      "windows.uiAutomation.capabilities",
    );
    assert(capabilities?.readOnly === true, "native helper must be read-only");
    assert(
      capabilities?.uiAutomation === "ready",
      "UI Automation COM is unavailable",
    );
    assert(
      capabilities?.displayWatcher === "ready",
      "display watcher is unavailable",
    );
    checks.push({ name: "capabilities", status: "passed" });

    const windows = await client.invoke("windows.window.read");
    assert(
      Array.isArray(windows?.windows),
      "window enumeration returned no list",
    );
    const targetWindow = windows.windows.find(
      (window) => Number(window?.windowHandle) > 0,
    );
    assert(targetWindow, "window enumeration returned no native HWND");
    checks.push({
      name: "window.read",
      count: windows.windows.length,
      status: "passed",
    });

    const displays = await client.invoke("windows.display.read");
    assert(
      Array.isArray(displays?.displays) && displays.displays.length > 0,
      "display enumeration returned no monitors",
    );
    checks.push({
      name: "display.read",
      count: displays.displays.length,
      status: "passed",
    });

    const tree = await client.invoke("windows.uiAutomation.read", {
      windowHandle: Number(targetWindow.windowHandle),
      maxDepth: 2,
      maxNodes: 64,
    });
    assert(
      tree?.tree && Number(tree.nodeCount) >= 1,
      "UI Automation tree is empty",
    );
    checks.push({
      name: "uiAutomation.read",
      nodeCount: tree.nodeCount,
      status: "passed",
    });

    const displayWatcher = await client.invoke("windows.displayWatcher.start");
    assert(displayWatcher?.started === true, "display watcher did not start");
    await client.invoke("windows.displayWatcher.stop");
    checks.push({ name: "displayWatcher.start-stop", status: "passed" });

    const rawInput = await client.invoke("windows.bareModifierMonitor.start");
    assert(rawInput?.started === true, "Raw Input monitor did not start");
    await client.invoke("windows.bareModifierMonitor.stop");
    checks.push({ name: "rawInput.start-stop", status: "passed" });
  } catch (error) {
    failure = error instanceof Error ? error : new Error(String(error));
  } finally {
    observedEventTypes = [
      ...new Set(client.events().map((event) => event.event)),
    ];
    await client.close();
  }
  const summary = {
    result: failure ? "failed" : "passed",
    evidenceLevel: "gate-b",
    platform: "win32",
    arch: "x64",
    candidateRunId: process.env.LIME_GATE_RUN_ID?.trim() || null,
    candidateSha: process.env.LIME_CANDIDATE_SHA?.trim().toLowerCase() || null,
    electronExecutable: path.resolve(options.electronExecutable),
    checks,
    ...(failure ? { failure: failure.message } : {}),
    helper: {
      path: helper.helperPath,
      resourcesRoot,
      manifestSha256: helper.manifest.resources.find(
        (resource) => resource.id === "windows-native-host",
      )?.sha256,
      packagedSha256: helper.sha256,
      digestMatches: true,
      api: helper.manifest.native.windowsHelper.api,
      readOnly: true,
    },
    observedEventTypes,
  };
  await mkdir(options.evidenceDir, { recursive: true });
  const summaryPath = path.join(options.evidenceDir, "summary.json");
  await writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  if (failure) {
    throw failure;
  }
  return { summary, summaryPath };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(
      "Usage: node scripts/electron/windows-native-host-gate-b.mjs --electron-executable <path> [--evidence-dir <path>] [--timeout-ms <ms>]",
    );
    return;
  }
  const result = await runWindowsNativeHostGateB(options);
  console.log(`${LOG_PREFIX} result=passed summary=${result.summaryPath}`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(`${LOG_PREFIX} result=failed error=${error.message}`);
    process.exitCode = 1;
  });
}
