import { spawn } from "node:child_process";
import {
  createReadStream,
  existsSync,
  mkdtempSync,
  readFileSync,
  statSync,
} from "node:fs";
import { createServer as createHttpServer } from "node:http";
import { createServer as createNetServer } from "node:net";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { chromium } from "playwright";

const PRODUCT_NAME = "Lime";
const SQUIRREL_PACKAGE_NAME = "lime";

export function normalizeVersion(value) {
  const version = String(value || "")
    .trim()
    .replace(/^v/, "");
  if (!/^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$/.test(version)) {
    throw new Error(`valid release version is required, got: ${String(value)}`);
  }
  return version;
}

export function compareVersions(left, right) {
  const parse = (value) => {
    const normalized = normalizeVersion(value);
    const [core, prerelease = ""] = normalized.split("-", 2);
    return {
      core: core.split(".").map(Number),
      prerelease,
    };
  };
  const leftVersion = parse(left);
  const rightVersion = parse(right);
  for (let index = 0; index < 3; index += 1) {
    const difference = leftVersion.core[index] - rightVersion.core[index];
    if (difference !== 0) {
      return Math.sign(difference);
    }
  }
  if (leftVersion.prerelease === rightVersion.prerelease) {
    return 0;
  }
  if (!leftVersion.prerelease) {
    return 1;
  }
  if (!rightVersion.prerelease) {
    return -1;
  }
  return leftVersion.prerelease.localeCompare(rightVersion.prerelease);
}

export function selectNMinusOneVersion({ candidateVersion, tags }) {
  const candidate = normalizeVersion(candidateVersion);
  const versions = [...new Set(tags)]
    .map((tag) => String(tag).trim())
    .filter((tag) => /^v?[0-9]+\.[0-9]+\.[0-9]+$/.test(tag))
    .map(normalizeVersion)
    .filter((version) => compareVersions(version, candidate) < 0)
    .sort((left, right) => compareVersions(right, left));
  if (versions.length === 0) {
    throw new Error(`no stable N-1 release tag exists below ${candidate}`);
  }
  return versions[0];
}

export function resolveSquirrelFeed({ feedDir, version }) {
  const root = path.resolve(feedDir);
  const metadataPath = path.join(root, "RELEASES");
  if (!existsSync(metadataPath)) {
    throw new Error(`candidate Squirrel feed is missing RELEASES: ${root}`);
  }
  const entries = readFileSync(metadataPath, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const match = /^([0-9a-f]{40})\s+(\S+)\s+([0-9]+)$/i.exec(line);
      if (!match) {
        throw new Error(`invalid Squirrel RELEASES entry: ${line}`);
      }
      const fileName = decodeURIComponent(match[2]);
      if (path.basename(fileName) !== fileName) {
        throw new Error(
          `Squirrel RELEASES entry must be a basename: ${fileName}`,
        );
      }
      const filePath = path.join(root, fileName);
      if (!existsSync(filePath)) {
        throw new Error(
          `Squirrel package referenced by RELEASES is missing: ${fileName}`,
        );
      }
      const size = Number(match[3]);
      const actualSize = statSync(filePath).size;
      if (actualSize !== size) {
        throw new Error(
          `Squirrel package size mismatch for ${fileName}: RELEASES=${size}, actual=${actualSize}`,
        );
      }
      return {
        fileName,
        filePath,
        sha1: match[1].toLowerCase(),
        size,
      };
    });
  const expectedFullPackage = `${SQUIRREL_PACKAGE_NAME}-${normalizeVersion(version)}-full.nupkg`;
  if (
    !entries.some(
      (entry) =>
        entry.fileName.toLowerCase() === expectedFullPackage.toLowerCase(),
    )
  ) {
    throw new Error(
      `candidate Squirrel feed does not reference ${expectedFullPackage}`,
    );
  }
  return { entries, metadataPath, root };
}

export function resolveInstalledSquirrelPaths({
  localAppData,
  version,
  packageName = SQUIRREL_PACKAGE_NAME,
}) {
  const packageRoot = path.resolve(localAppData, packageName);
  const appDirectory = path.join(
    packageRoot,
    `app-${normalizeVersion(version)}`,
  );
  return {
    appDirectory,
    executable: path.join(appDirectory, `${PRODUCT_NAME}.exe`),
    packageRoot,
    updateExecutable: path.join(packageRoot, "Update.exe"),
  };
}

export function isFinalElectronRendererUrl(value) {
  try {
    const url = new URL(String(value));
    return url.searchParams.get("nativeStartup") === "1";
  } catch {
    return false;
  }
}

export function buildNMinusOneLaunchEnv({
  baseEnv = process.env,
  feedUrl,
  userDataDir,
}) {
  const env = {
    ...baseEnv,
    APP_SERVER_BIN: "",
    ELECTRON_E2E_USER_DATA_DIR: userDataDir,
    LIME_ELECTRON_BRAND_DEV_APP: "0",
    LIME_ELECTRON_E2E: "1",
    LIME_ELECTRON_ENABLE_DEV_UPDATER: "1",
    LIME_ELECTRON_UPDATES_URL: feedUrl,
  };
  delete env.LIME_ELECTRON_SMOKE;
  delete env.NODE_OPTIONS;
  delete env.VITE_DEV_SERVER_URL;
  return env;
}

export async function findReadyElectronUpdaterPage(pages) {
  for (const candidate of pages) {
    if (!isFinalElectronRendererUrl(candidate.url())) {
      continue;
    }
    const ready = await candidate
      .evaluate(
        () =>
          window.__LIME_ELECTRON__ === true &&
          typeof window.electronAPI?.invoke === "function" &&
          window.electronAPI.supportsCommand("check_for_updates") &&
          window.electronAPI.supportsCommand("start_update_install_session"),
      )
      .catch(() => false);
    if (ready) {
      return candidate;
    }
  }
  return null;
}

export async function stopInstalledApp(executable) {
  const script = [
    "$target = [System.IO.Path]::GetFullPath($env:LIME_TARGET_EXECUTABLE)",
    "$processes = @(Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and [String]::Equals([System.IO.Path]::GetFullPath($_.ExecutablePath), $target, [StringComparison]::OrdinalIgnoreCase) })",
    "$processes | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop }",
    'Write-Output "stopped=$($processes.Count)"',
  ].join("; ");
  const result = await runProcess(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-Command", script],
    {
      env: { ...process.env, LIME_TARGET_EXECUTABLE: executable },
      timeoutMs: 30_000,
    },
  );
  if (result.exitCode !== 0) {
    throw new Error(
      `failed to stop installed app at ${executable}: exit ${result.exitCode}`,
    );
  }
  return { executable, exitCode: result.exitCode };
}

export async function uninstallInstalledSquirrel({
  updateExecutable,
  appDirectory,
  executable,
  shortcutPaths = [],
  timeoutMs = 60_000,
  runProcessImpl = runProcess,
  existsImpl = existsSync,
  waitForAbsentImpl = waitForAbsent,
} = {}) {
  const requiredPaths = [updateExecutable, appDirectory, executable].filter(
    (value) => typeof value === "string" && value.length > 0,
  );
  if (requiredPaths.length !== 3) {
    throw new Error(
      "Squirrel uninstall requires updateExecutable, appDirectory, and executable",
    );
  }
  const result = await runProcessImpl(updateExecutable, ["--uninstall"], {
    env: process.env,
    timeoutMs: 30_000,
  });
  if (result.exitCode !== 0) {
    throw new Error(
      `Squirrel uninstall exited with ${result.exitCode} at ${updateExecutable}`,
    );
  }
  await waitForAbsentImpl(
    [...requiredPaths, ...shortcutPaths].filter(
      (value) => typeof value === "string" && value.length > 0,
    ),
    { existsImpl, timeoutMs },
  );
  return {
    updateExecutable,
    appDirectory,
    executable,
    shortcutPaths: [...shortcutPaths],
    exitCode: result.exitCode,
    updateExecutableAbsent: !existsImpl(updateExecutable),
    appDirectoryAbsent: !existsImpl(appDirectory),
    executableAbsent: !existsImpl(executable),
    shortcutsAbsent: shortcutPaths.every((shortcut) => !existsImpl(shortcut)),
  };
}

async function waitForAbsent(
  paths,
  { existsImpl = existsSync, timeoutMs = 60_000, intervalMs = 250 } = {},
) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() <= deadline) {
    if (paths.every((filePath) => !existsImpl(filePath))) return;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  const remaining = paths.filter((filePath) => existsImpl(filePath));
  throw new Error(
    `Squirrel uninstall did not remove paths: ${remaining.join(", ")}`,
  );
}

export function buildWaitForWindowsProcessExitScript() {
  const matchingProcesses =
    "@(Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and [String]::Equals([System.IO.Path]::GetFullPath($_.ExecutablePath), $target, [StringComparison]::OrdinalIgnoreCase) })";
  return [
    "$ErrorActionPreference = 'Stop'",
    "$target = [System.IO.Path]::GetFullPath($env:LIME_TARGET_EXECUTABLE)",
    "$deadline = [DateTime]::UtcNow.AddMilliseconds([double]$env:LIME_PROCESS_WAIT_TIMEOUT_MS)",
    `while ([DateTime]::UtcNow -lt $deadline) { $processes = ${matchingProcesses}; if ($processes.Count -eq 0) { Write-Output "running=0"; exit 0 }; Start-Sleep -Milliseconds 250 }`,
    'Write-Error "timed out waiting for process exit: $target"',
    "exit 1",
  ].join("; ");
}

export async function waitForWindowsProcessExit(
  executable,
  { runProcessImpl = runProcess, timeoutMs = 60_000 } = {},
) {
  const result = await runProcessImpl(
    "powershell.exe",
    [
      "-NoProfile",
      "-NonInteractive",
      "-Command",
      buildWaitForWindowsProcessExitScript(),
    ],
    {
      env: {
        ...process.env,
        LIME_PROCESS_WAIT_TIMEOUT_MS: String(timeoutMs),
        LIME_TARGET_EXECUTABLE: executable,
      },
      timeoutMs: timeoutMs + 5_000,
    },
  );
  if (result.exitCode !== 0) {
    throw new Error(
      `timed out waiting for process exit at ${executable}: exit ${result.exitCode}`,
    );
  }
  return { executable, exitCode: result.exitCode, timeoutMs };
}

export async function exerciseNMinusOneUpdate({
  candidateFeedDir,
  candidateVersion,
  installed,
  nMinusOneVersion,
  timeoutMs,
}) {
  const feed = resolveSquirrelFeed({
    feedDir: candidateFeedDir,
    version: candidateVersion,
  });
  const candidateInstalled = resolveInstalledSquirrelPaths({
    localAppData: path.dirname(installed.packageRoot),
    version: candidateVersion,
  });
  if (existsSync(candidateInstalled.appDirectory)) {
    throw new Error(
      `candidate app directory existed before N-1 update: ${candidateInstalled.appDirectory}`,
    );
  }

  const baselineUpdaterQuiescence = await waitForWindowsProcessExit(
    installed.updateExecutable,
  );
  const staticFeed = await startSquirrelFeed(feed);
  const cdpPort = await reserveLocalPort();
  const cdpUrl = `http://127.0.0.1:${cdpPort}`;
  const userDataDir = mkdtempSync(
    path.join(os.tmpdir(), "lime-windows-n-minus-one-updater-"),
  );
  const launchEnv = buildNMinusOneLaunchEnv({
    feedUrl: staticFeed.url,
    userDataDir,
  });

  const child = spawn(
    installed.executable,
    [`--remote-debugging-port=${cdpPort}`, "--use-mock-keychain"],
    {
      env: launchEnv,
      shell: false,
      stdio: "inherit",
      windowsHide: true,
    },
  );
  let childError = null;
  child.once("error", (error) => {
    childError = error;
  });
  let browser = null;
  try {
    await waitFor(
      async () => {
        if (childError) {
          throw childError;
        }
        try {
          const response = await fetch(`${cdpUrl}/json/list`);
          return response.ok;
        } catch {
          return false;
        }
      },
      { label: "N-1 Electron CDP endpoint", timeoutMs: 60_000 },
    );
    browser = await chromium.connectOverCDP(cdpUrl);
    const page = await waitFor(
      async () => {
        const pages = browser.contexts().flatMap((context) => context.pages());
        return await findReadyElectronUpdaterPage(pages);
      },
      { label: "N-1 Electron updater bridge", timeoutMs: 60_000 },
    );

    const initialSession = await waitFor(
      () =>
        page
          .evaluate(() =>
            window.electronAPI.invoke("get_update_install_session"),
          )
          .catch(() => null),
      {
        accept: (session) => Boolean(session && session.stage !== "idle"),
        label: "N-1 automatic update check",
        timeoutMs: 60_000,
        intervalMs: 250,
      },
    );
    if (
      !["checking", "downloading", "completed", "failed"].includes(
        initialSession.stage,
      )
    ) {
      throw new Error(
        `N-1 automatic updater entered unexpected stage: ${String(initialSession.stage)}`,
      );
    }
    if (initialSession?.currentVersion !== nMinusOneVersion) {
      throw new Error(
        `update session current version ${String(initialSession?.currentVersion)}; expected ${nMinusOneVersion}`,
      );
    }

    const downloadedSession = await waitFor(
      () =>
        page
          .evaluate(() =>
            window.electronAPI.invoke("get_update_install_session"),
          )
          .catch(() => null),
      {
        accept: (session) =>
          session?.stage === "completed" || session?.stage === "failed",
        label: "candidate update download terminal",
        timeoutMs,
        intervalMs: 500,
      },
    );
    if (downloadedSession.stage !== "completed") {
      throw new Error(
        `candidate update download failed: ${downloadedSession.error || downloadedSession.message}`,
      );
    }
    if (downloadedSession.latestVersion !== candidateVersion) {
      throw new Error(
        `downloaded version ${String(downloadedSession.latestVersion)}; expected ${candidateVersion}`,
      );
    }

    const installSession = await page.evaluate(() =>
      window.electronAPI.invoke("start_update_install_session"),
    );
    if (installSession?.stage !== "restarting") {
      throw new Error(
        `updater did not enter restarting stage: ${String(installSession?.stage)}`,
      );
    }
    await waitFor(() => existsSync(candidateInstalled.executable), {
      label: "candidate app installed by updater",
      timeoutMs: 180_000,
    });
    await waitFor(() => child.exitCode !== null, {
      label: "N-1 Electron exit for updater",
      timeoutMs: 60_000,
    });

    const requestedPaths = staticFeed.requests
      .filter((request) => request.status === 200)
      .map((request) => request.path);
    const fullPackageName = `${SQUIRREL_PACKAGE_NAME}-${candidateVersion}-full.nupkg`;
    const candidateFeedServed =
      requestedPaths.includes("/RELEASES") &&
      requestedPaths.some(
        (requestPath) =>
          path.basename(requestPath).toLowerCase() ===
          fullPackageName.toLowerCase(),
      );
    if (!candidateFeedServed) {
      throw new Error(
        `candidate feed did not serve RELEASES and ${fullPackageName}`,
      );
    }
    return {
      baselineUpdaterQuiescence,
      candidateFeedServed,
      candidateInstalledByUpdater: existsSync(candidateInstalled.executable),
      candidateVersion,
      cdpUrl,
      downloadedSession,
      feed: {
        entries: feed.entries.map(({ fileName, sha1, size }) => ({
          fileName,
          sha1,
          size,
        })),
        requests: staticFeed.requests,
        url: staticFeed.url,
      },
      initialSession,
      installSession,
      nMinusOneVersion,
      updateDownloaded: downloadedSession.stage === "completed",
      updateInstallRequested: installSession.stage === "restarting",
      userDataDir,
    };
  } finally {
    await browser?.close().catch(() => undefined);
    if (child.exitCode === null) {
      child.kill();
    }
    await staticFeed.close();
  }
}

async function startSquirrelFeed(feed) {
  const requests = [];
  const allowed = new Map([
    ["RELEASES", feed.metadataPath],
    ...feed.entries.map((entry) => [entry.fileName, entry.filePath]),
  ]);
  const server = createHttpServer((request, response) => {
    const requestUrl = new URL(request.url || "/", "http://127.0.0.1");
    const fileName = decodeURIComponent(
      requestUrl.pathname.replace(/^\/+/, ""),
    );
    const filePath = allowed.get(fileName);
    const method = request.method || "GET";
    if (!filePath || path.basename(fileName) !== fileName) {
      requests.push({ method, path: requestUrl.pathname, status: 404 });
      response.writeHead(404).end();
      return;
    }
    const size = statSync(filePath).size;
    requests.push({ method, path: requestUrl.pathname, size, status: 200 });
    response.writeHead(200, {
      "Cache-Control": "no-store",
      "Content-Length": size,
      "Content-Type":
        fileName === "RELEASES" ? "text/plain" : "application/octet-stream",
    });
    if (method === "HEAD") {
      response.end();
      return;
    }
    createReadStream(filePath).pipe(response);
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("failed to resolve local Squirrel feed port");
  }
  return {
    close: () =>
      new Promise((resolve, reject) => {
        server.closeIdleConnections?.();
        server.close((error) => (error ? reject(error) : resolve()));
      }),
    requests,
    url: `http://127.0.0.1:${address.port}`,
  };
}

async function reserveLocalPort() {
  const server = createNetServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    server.close();
    throw new Error("failed to reserve a local CDP port");
  }
  const port = address.port;
  await new Promise((resolve, reject) =>
    server.close((error) => (error ? reject(error) : resolve())),
  );
  return port;
}

async function waitFor(
  read,
  { accept = Boolean, label, timeoutMs, intervalMs = 250 },
) {
  const deadline = Date.now() + timeoutMs;
  let lastValue;
  while (Date.now() <= deadline) {
    lastValue = await read();
    if (accept(lastValue)) {
      return lastValue;
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error(`timed out waiting for ${label}`);
}

function runProcess(command, args, { env, timeoutMs }) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      env,
      shell: false,
      stdio: "inherit",
      windowsHide: true,
    });
    let timedOut = false;
    const timeout = setTimeout(() => {
      timedOut = true;
      child.kill();
    }, timeoutMs);
    child.once("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.once("exit", (code, signal) => {
      clearTimeout(timeout);
      if (timedOut) {
        reject(
          new Error(`${path.basename(command)} timed out after ${timeoutMs}ms`),
        );
        return;
      }
      resolve({ exitCode: code ?? 1, signal });
    });
  });
}
