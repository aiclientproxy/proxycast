#!/usr/bin/env node

import { createHash } from "node:crypto";
import { execFileSync, spawn } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { runMacOSNativeHostElectronGateB } from "./lib/macos-native-host-electron-gate-b.mjs";
import {
  normalizeCandidateRunId,
  normalizeCandidateSha,
} from "./lib/release-candidate-identity.mjs";
import {
  readMacOSAppIdentity,
  verifyCodeSignature,
  verifyMacOSReleaseTrust,
} from "./lib/macos-release-trust.mjs";

const LOG_PREFIX = "[smoke:macos-native-host-gate-b]";
const DEFAULT_TIMEOUT_MS = 120_000;
const RESOURCE_MANIFEST = "desktop-resources.manifest.json";
const APPLICATION_ID = "com.limecloud.lime";
const HELPER_ID = "macos-native-host";
const PROTOCOL_VERSION = 1;

export function parseArgs(argv) {
  const options = {
    electronExecutable: null,
    evidenceDir: path.resolve(".lime/qc/gui-evidence/macos-native-host-gate-b"),
    timeoutMs: DEFAULT_TIMEOUT_MS,
    intervalMs: 250,
    arch: process.arch === "arm64" ? "arm64" : "x64",
    strictPermissions: false,
    releaseTrust: false,
    candidateSha: process.env.LIME_CANDIDATE_SHA || null,
    runId: process.env.LIME_GATE_RUN_ID || null,
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
    if (arg === "--interval-ms" && argv[index + 1]) {
      options.intervalMs = Number(argv[++index]);
      continue;
    }
    if (arg === "--arch" && argv[index + 1]) {
      options.arch = String(argv[++index]);
      continue;
    }
    if (arg === "--strict-permissions") {
      options.strictPermissions = true;
      continue;
    }
    if (arg === "--release-trust") {
      options.releaseTrust = true;
      continue;
    }
    if (arg === "--candidate-sha" && argv[index + 1]) {
      options.candidateSha = argv[++index];
      continue;
    }
    if (arg === "--run-id" && argv[index + 1]) {
      options.runId = argv[++index];
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (options.help) return options;
  if (process.platform !== "darwin") {
    throw new Error("macOS native host Gate B requires a macOS runner");
  }
  if (!options.electronExecutable) {
    throw new Error("--electron-executable is required");
  }
  if (!existsSync(options.electronExecutable)) {
    throw new Error(
      `Installed Electron executable does not exist: ${options.electronExecutable}`,
    );
  }
  if (!/^arm64$|^x64$/u.test(options.arch)) {
    throw new Error("--arch must be arm64 or x64");
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms must be >= 30000");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms must be >= 100");
  }
  options.candidateSha = normalizeCandidateSha(options.candidateSha);
  options.runId = normalizeCandidateRunId(options.runId);
  return options;
}

export function resolveInstalledResourcesRoot(electronExecutable) {
  const executable = path.resolve(electronExecutable);
  const appBundle = executable.match(
    /^(.*\.app)[/\\]Contents[/\\]MacOS[/\\][^/\\]+$/u,
  )?.[1];
  const resourcesRoot = appBundle
    ? path.join(appBundle, "Contents", "Resources")
    : path.join(path.dirname(executable), "..", "Resources");
  if (!existsSync(resourcesRoot) || !statSync(resourcesRoot).isDirectory()) {
    throw new Error(
      `Installed macOS app resources directory is missing: ${resourcesRoot}`,
    );
  }
  return resourcesRoot;
}

function sha256(filePath) {
  return createHash("sha256").update(readFileSync(filePath)).digest("hex");
}

function readInstalledHelper(resourcesRoot, arch) {
  const manifestPath = path.join(resourcesRoot, RESOURCE_MANIFEST);
  if (!existsSync(manifestPath)) {
    throw new Error(
      `Installed desktop resource manifest is missing: ${manifestPath}`,
    );
  }
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  if (
    manifest.schemaVersion !== 1 ||
    manifest.applicationId !== APPLICATION_ID ||
    manifest.platform !== "darwin" ||
    manifest.arch !== arch ||
    manifest.platformKey !== `darwin-${arch}`
  ) {
    throw new Error(
      "Installed macOS desktop resource manifest identity is invalid",
    );
  }
  const metadata = manifest.native?.helper;
  if (
    metadata?.id !== HELPER_ID ||
    metadata.protocolVersion !== PROTOCOL_VERSION ||
    metadata.path !==
      "native/macos/macos-native-host.app/Contents/MacOS/macos-native-host" ||
    metadata.bundlePath !== "native/macos/macos-native-host.app" ||
    metadata.bundleIdentifier !== `${APPLICATION_ID}.native-host`
  ) {
    throw new Error("Installed macOS native helper metadata is invalid");
  }
  const resource = manifest.resources?.find(
    (candidate) => candidate?.id === HELPER_ID,
  );
  if (!resource || resource.path !== metadata.path) {
    throw new Error("Installed macOS native helper resource is not registered");
  }
  const helperPath = path.resolve(resourcesRoot, metadata.path);
  const bundlePath = path.resolve(resourcesRoot, metadata.bundlePath);
  if (
    !helperPath.startsWith(`${resourcesRoot}${path.sep}`) ||
    !bundlePath.startsWith(`${resourcesRoot}${path.sep}`) ||
    !existsSync(helperPath) ||
    !statSync(helperPath).isFile() ||
    !existsSync(bundlePath) ||
    !statSync(bundlePath).isDirectory()
  ) {
    throw new Error("Installed macOS native helper path is invalid");
  }
  const packagedSha256 = sha256(helperPath);
  const digestMatches = packagedSha256 === resource.sha256;
  const signed = verifyCodeSignature(bundlePath);
  if (!digestMatches && !signed) {
    throw new Error(
      "Installed macOS native helper sha256 mismatch and signature is invalid",
    );
  }
  return {
    helperPath,
    bundlePath,
    manifest,
    sha256: packagedSha256,
    digestMatches,
    signed,
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
          `macOS native host exited code=${code ?? ""} signal=${signal ?? ""}`,
        ),
      );
    });
  }

  async invoke(method, params = {}) {
    const id = String(this.#nextId++);
    return await new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.#pending.delete(id);
        reject(new Error(`macOS native host request timed out: ${method}`));
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
    this.#failAll(new Error("macOS native host stopped"));
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
            response.error?.message || "macOS native host request failed",
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

function waitForExit(child, timeoutMs) {
  if (child.exitCode !== null) return Promise.resolve();
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve();
    };
    const timer = setTimeout(finish, timeoutMs);
    child.once("exit", finish);
  });
}

async function waitForFile(filePath, timeoutMs) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (existsSync(filePath)) return;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(`macOS window fixture did not become ready: ${filePath}`);
}

function compileWindowFixture(arch, bundlePath) {
  const sourcePath = path.resolve(
    "scripts/electron/macos-window-fixture.swift",
  );
  const target = arch === "arm64" ? "arm64" : "x86_64";
  const contentsPath = path.join(bundlePath, "Contents");
  const executablePath = path.join(contentsPath, "MacOS", "WindowFixture");
  mkdirSync(path.dirname(executablePath), { recursive: true });
  writeFileSync(
    path.join(contentsPath, "Info.plist"),
    `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key><string>WindowFixture</string>
  <key>CFBundleIdentifier</key><string>com.limecloud.lime.gate-b-window-fixture</string>
  <key>CFBundleName</key><string>Lime Gate B Window Fixture</string>
  <key>CFBundlePackageType</key><string>APPL</string>
</dict>
</plist>
`,
    "utf8",
  );
  execFileSync(
    "swiftc",
    [
      "-O",
      "-target",
      `${target}-apple-macos13.0`,
      sourcePath,
      "-o",
      executablePath,
    ],
    { stdio: "pipe" },
  );
  execFileSync("codesign", ["--force", "--deep", "--sign", "-", bundlePath], {
    stdio: "pipe",
  });
  return executablePath;
}

async function exerciseWindowControls(client, options) {
  const accessibility = await client.invoke("accessibility.read");
  if (accessibility?.status !== "ready") {
    if (options.strictPermissions) {
      throw new Error(
        "window orchestration requires Accessibility permission in strict mode",
      );
    }
    return {
      name: "window.anchor-stack-hideForTask",
      status: "skipped",
      reason: "Accessibility permission is not ready",
    };
  }

  const fixtureRoot = mkdtempSync(
    path.join(tmpdir(), "lime-macos-window-fixture-"),
  );
  const fixtureBundle = path.join(fixtureRoot, "WindowFixture.app");
  const fixtureRecord = path.join(fixtureRoot, "windows.json");
  let fixtureProcess = null;
  let fixtureOwnerPid = null;
  const taskId = `gate-b-${process.pid}`;
  let hideStarted = false;
  try {
    compileWindowFixture(options.arch, fixtureBundle);
    fixtureProcess = spawn(
      "/usr/bin/open",
      ["-n", "-W", fixtureBundle, "--args", fixtureRecord],
      {
        stdio: ["ignore", "ignore", "pipe"],
      },
    );
    fixtureProcess.stderr.setEncoding("utf8");
    fixtureProcess.stderr.on("data", () => undefined);
    await waitForFile(fixtureRecord, Math.min(options.timeoutMs, 30_000));
    const fixture = JSON.parse(readFileSync(fixtureRecord, "utf8"));
    const ownerPid = Number(fixture?.pid);
    assert(
      Number.isInteger(ownerPid) && ownerPid > 0,
      "window fixture PID is invalid",
    );
    fixtureOwnerPid = ownerPid;

    let fixtureWindows = [];
    const windowDeadline = Date.now() + Math.min(options.timeoutMs, 10_000);
    while (Date.now() < windowDeadline) {
      const windows = await client.invoke("window.read");
      fixtureWindows = (windows?.windows ?? []).filter(
        (window) =>
          Number(window?.ownerPid) === ownerPid &&
          /^Lime Gate B Fixture [12]$/u.test(String(window?.title ?? "")),
      );
      if (fixtureWindows.length === 2) break;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    assert(
      fixtureWindows.length === 2,
      `window fixture did not expose two windows (found ${fixtureWindows.length})`,
    );
    const [anchor, target] = fixtureWindows;
    const anchorResult = await client.invoke("window.anchor", {
      windowId: Number(target.windowId),
      anchorWindowId: Number(anchor.windowId),
      edge: "right",
      alignment: "start",
      gap: 12,
    });
    assert(anchorResult?.anchored === true, "window anchor did not complete");

    const stackResult = await client.invoke("window.stack", {
      windowIds: [Number(target.windowId), Number(anchor.windowId)],
    });
    assert(
      stackResult?.stacked === true &&
        JSON.stringify(stackResult.raisedOrder) ===
          JSON.stringify([Number(anchor.windowId), Number(target.windowId)]),
      "window stack order was not applied",
    );

    const hideResult = await client.invoke("window.hideForTask.start", {
      taskId,
      windowIds: [Number(anchor.windowId), Number(target.windowId)],
    });
    hideStarted = hideResult?.started === true;
    assert(
      hideStarted && hideResult.hiddenCount === 1,
      "hide-for-task did not hide the fixture owner",
    );
    const activeTasks = await client.invoke("window.hideForTask.read");
    assert(
      Array.isArray(activeTasks?.tasks) && activeTasks.tasks.includes(taskId),
      "hide-for-task lease was not observable",
    );
    const stopResult = await client.invoke("window.hideForTask.stop", {
      taskId,
    });
    hideStarted = false;
    assert(
      stopResult?.stopped === true && stopResult.restoredCount === 1,
      "hide-for-task did not restore the fixture owner",
    );
    const remainingTasks = await client.invoke("window.hideForTask.read");
    assert(
      Array.isArray(remainingTasks?.tasks) &&
        !remainingTasks.tasks.includes(taskId),
      "hide-for-task lease remained active after stop",
    );
    return {
      name: "window.anchor-stack-hideForTask",
      status: "passed",
      windowIds: [Number(anchor.windowId), Number(target.windowId)],
      ownerPid,
    };
  } catch (error) {
    if (!options.strictPermissions && error?.code === "ENOENT") {
      return {
        name: "window.anchor-stack-hideForTask",
        status: "skipped",
        reason: "Swift window fixture compiler is unavailable",
      };
    }
    throw error;
  } finally {
    if (hideStarted) {
      await client
        .invoke("window.hideForTask.stop", { taskId })
        .catch(() => undefined);
    }
    if (fixtureOwnerPid) {
      try {
        process.kill(fixtureOwnerPid, "SIGTERM");
      } catch {}
    }
    if (fixtureProcess && !fixtureProcess.killed) {
      fixtureProcess.kill("SIGTERM");
      await waitForExit(fixtureProcess, 5_000);
    }
    rmSync(fixtureRoot, { recursive: true, force: true });
  }
}

function permissionStatus(capabilities, key) {
  const value = capabilities?.[key];
  return {
    status: typeof value?.status === "string" ? value.status : "unavailable",
    reason: typeof value?.reason === "string" ? value.reason : null,
  };
}

export async function runMacOSNativeHostGateB(options) {
  const appIdentity = readMacOSAppIdentity(options.electronExecutable, {
    applicationId: APPLICATION_ID,
  });
  const resourcesRoot = resolveInstalledResourcesRoot(
    options.electronExecutable,
  );
  const helper = readInstalledHelper(resourcesRoot, options.arch);
  assert(
    helper.manifest.version === appIdentity.version,
    "Installed app and desktop resource manifest versions do not match",
  );
  const releaseTrust = options.releaseTrust
    ? verifyMacOSReleaseTrust(appIdentity.appBundlePath, helper.bundlePath)
    : { status: "not-requested" };
  const client = new NativeHostClient(helper.helperPath, options.timeoutMs);
  const checks = [];
  const permissions = {};
  let observedEventTypes = [];
  let failure = null;
  let electronGate = null;
  try {
    const capabilities = await client.invoke("capabilities.read");
    assert(
      capabilities?.protocolVersion === PROTOCOL_VERSION,
      "native helper protocol version is unsupported",
    );
    assert(capabilities?.helperId === HELPER_ID, "native helper id mismatch");
    assert(
      capabilities?.platform === "darwin",
      "native helper platform mismatch",
    );
    assert(
      capabilities?.applicationId === `${APPLICATION_ID}.native-host`,
      "native helper bundle identity mismatch",
    );
    checks.push({ name: "capabilities.read", status: "passed" });

    const windows = await client.invoke("window.read");
    assert(
      Array.isArray(windows?.windows),
      "window enumeration returned no list",
    );
    checks.push({
      name: "window.read",
      count: windows.windows.length,
      status: "passed",
    });

    const displays = await client.invoke("display.read");
    assert(
      Array.isArray(displays?.displays) && displays.displays.length > 0,
      "display enumeration returned no displays",
    );
    checks.push({
      name: "display.read",
      count: displays.displays.length,
      status: "passed",
    });

    const watcher = await client.invoke("display.watch.start");
    assert(watcher?.started === true, "display watcher did not start");
    await client.invoke("display.watch.stop");
    checks.push({ name: "display.watch.start-stop", status: "passed" });

    checks.push(await exerciseWindowControls(client, options));

    const bookmarkRoot = mkdtempSync(
      path.join(tmpdir(), "lime-macos-native-host-gate-b-"),
    );
    try {
      const created = await client.invoke("bookmark.create", {
        path: bookmarkRoot,
      });
      assert(
        typeof created?.bookmark === "string" && created.bookmark.length > 0,
        "security-scoped bookmark creation returned no encoded bookmark",
      );
      const resolved = await client.invoke("bookmark.resolve", {
        bookmark: created.bookmark,
      });
      assert(
        resolved?.isStale === false,
        "security-scoped bookmark resolved as stale",
      );
      assert(
        realpathSync(resolved.path) === realpathSync(bookmarkRoot),
        "security-scoped bookmark resolved to an unexpected path",
      );
      const started = await client.invoke("bookmark.start", {
        bookmark: created.bookmark,
      });
      assert(
        started?.started === true && typeof started.token === "string",
        "security-scoped bookmark access did not start",
      );
      const stopped = await client.invoke("bookmark.stop", {
        token: started.token,
      });
      assert(
        stopped?.stopped === true,
        "security-scoped bookmark did not stop",
      );
      checks.push({
        name: "bookmark.create-resolve-start-stop",
        stale: resolved.isStale,
        status: "passed",
      });
    } finally {
      rmSync(bookmarkRoot, { recursive: true, force: true });
    }

    const permissionCapabilities = [
      ["accessibility", "accessibility.read"],
      ["inputMonitoring", "inputMonitoring.read"],
      ["screenCapture", "screenCapture.read"],
    ];
    for (const [key, method] of permissionCapabilities) {
      const result = await client.invoke(method);
      permissions[key] = permissionStatus({ [key]: result }, key);
      if (options.strictPermissions) {
        assert(result?.status === "ready", `${key} permission is not ready`);
      }
    }
    const targets = await client.invoke("appleEvents.targets");
    assert(
      Array.isArray(targets?.targets),
      "Apple Events target enumeration returned no list",
    );
    const appleTarget =
      targets.targets.find(
        (target) => target?.bundleId === "com.apple.finder",
      ) ?? targets.targets[0];
    if (options.strictPermissions) {
      assert(
        typeof appleTarget?.bundleId === "string",
        "Apple Events has no running target application",
      );
      const appleEvents = await client.invoke("appleEvents.read", {
        targetBundleId: appleTarget.bundleId,
      });
      permissions.appleEvents = permissionStatus(
        { appleEvents },
        "appleEvents",
      );
      assert(
        appleEvents?.status === "ready",
        "Apple Events permission is not ready",
      );
    } else {
      permissions.appleEvents = {
        status: "observed",
        targetCount: targets.targets.length,
      };
    }
    checks.push({
      name: "permissions.read",
      statuses: Object.fromEntries(
        Object.entries(permissions).map(([key, value]) => [key, value.status]),
      ),
      strict: options.strictPermissions,
      status: "passed",
    });

    const appBundlePath = path.resolve(
      path.dirname(options.electronExecutable),
      "../..",
    );
    const bundle = await client.invoke("launchServices.bundleIdentifier", {
      path: appBundlePath,
    });
    assert(
      bundle?.bundleIdentifier === APPLICATION_ID,
      "Launch Services returned an unexpected Lime bundle identifier",
    );
    checks.push({ name: "launchServices.bundleIdentifier", status: "passed" });
  } catch (error) {
    failure = error instanceof Error ? error : new Error(String(error));
  } finally {
    observedEventTypes = [
      ...new Set(client.events().map((event) => event.event)),
    ];
    await client.close();
  }

  if (!failure) {
    try {
      electronGate = await runMacOSNativeHostElectronGateB(options);
    } catch (error) {
      failure = error instanceof Error ? error : new Error(String(error));
    }
  }

  const summary = {
    result: failure ? "failed" : "passed",
    evidenceLevel: "gate-b",
    platform: "darwin",
    arch: options.arch,
    permissionMode: options.strictPermissions ? "strict" : "observe",
    candidateRunId: options.runId,
    candidateSha: options.candidateSha,
    candidate: {
      platform: "darwin",
      arch: options.arch,
      version: helper.manifest.version,
      sha: options.candidateSha,
      runId: options.runId,
    },
    electronExecutable: path.resolve(options.electronExecutable),
    application: appIdentity,
    releaseTrust,
    checks,
    permissions,
    ...(failure ? { failure: failure.message } : {}),
    helper: {
      id: HELPER_ID,
      path: helper.helperPath,
      bundlePath: helper.bundlePath,
      resourcesRoot,
      manifestSha256: helper.manifest.resources.find(
        (resource) => resource.id === HELPER_ID,
      )?.sha256,
      packagedSha256: helper.sha256,
      digestMatches: helper.digestMatches,
      signed: helper.signed,
      protocolVersion: PROTOCOL_VERSION,
    },
    electron: electronGate,
    observedEventTypes,
  };
  mkdirSync(options.evidenceDir, { recursive: true });
  const summaryPath = path.join(options.evidenceDir, "summary.json");
  writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  if (failure) throw failure;
  return { summary, summaryPath };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(
      "Usage: node scripts/electron/macos-native-host-gate-b.mjs --electron-executable <path> --candidate-sha <sha> --run-id <id> [--arch <arm64|x64>] [--evidence-dir <path>] [--timeout-ms <ms>] [--interval-ms <ms>] [--strict-permissions] [--release-trust]",
    );
    return;
  }
  const result = await runMacOSNativeHostGateB(options);
  console.log(`${LOG_PREFIX} result=passed summary=${result.summaryPath}`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(`${LOG_PREFIX} result=failed error=${error.message}`);
    process.exitCode = 1;
  });
}
