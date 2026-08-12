import { spawn, spawnSync } from "node:child_process";
import {
  existsSync,
  readdirSync,
  readFileSync,
  statSync,
  watch,
} from "node:fs";
import path from "node:path";
import { ensureMacBinaryRpath } from "../prepare-sherpa-onnx-runtime.mjs";
import { resolveRustyV8CargoEnv } from "./rusty-v8-artifacts.mjs";

export const APP_SERVER_WATCH_DEBOUNCE_MS = 800;

export function appServerBinaryName(platform = process.platform) {
  return platform === "win32" ? "app-server.exe" : "app-server";
}

export function codeModeHostBinaryName(platform = process.platform) {
  return platform === "win32" ? "code-mode-host.exe" : "code-mode-host";
}

export function localAppServerBinaryPath({
  repoRoot = process.cwd(),
  platform = process.platform,
  targetDirectory = resolveCargoTargetDirectory({ repoRoot, platform }),
} = {}) {
  return path.resolve(targetDirectory, "debug", appServerBinaryName(platform));
}

export function localCodeModeHostBinaryPath({
  repoRoot = process.cwd(),
  platform = process.platform,
  targetDirectory = resolveCargoTargetDirectory({ repoRoot, platform }),
} = {}) {
  return path.resolve(
    targetDirectory,
    "debug",
    codeModeHostBinaryName(platform),
  );
}

export function resolveCargoTargetDirectory({
  env = process.env,
  platform = process.platform,
  repoRoot = process.cwd(),
  readConfigFile = readFileSync,
  runner = spawnSync,
} = {}) {
  const metadataTargetDirectory = resolveCargoMetadataTargetDirectory({
    env,
    platform,
    repoRoot,
    runner,
  });
  if (metadataTargetDirectory) {
    return metadataTargetDirectory;
  }

  return resolveCargoTargetDirectoryFallback({
    env,
    repoRoot,
    readConfigFile,
  });
}

export function cargoMetadataArgs({ repoRoot = process.cwd() } = {}) {
  return [
    "metadata",
    "--manifest-path",
    path.resolve(repoRoot, "lime-rs", "Cargo.toml"),
    "--format-version",
    "1",
    "--no-deps",
  ];
}

export function resolveCargoMetadataTargetDirectory({
  env = process.env,
  platform = process.platform,
  repoRoot = process.cwd(),
  runner = spawnSync,
} = {}) {
  const cargoCommand = platform === "win32" ? "cargo.exe" : "cargo";
  try {
    const result = runner(cargoCommand, cargoMetadataArgs({ repoRoot }), {
      cwd: repoRoot,
      encoding: "utf8",
      env,
      shell: false,
      stdio: ["ignore", "pipe", "pipe"],
    });
    if (result.status !== 0 || !result.stdout) {
      return null;
    }
    const metadata = JSON.parse(String(result.stdout));
    const targetDirectory = String(metadata.target_directory || "").trim();
    return targetDirectory ? path.resolve(targetDirectory) : null;
  } catch {
    return null;
  }
}

export function resolveCargoTargetDirectoryFallback({
  env = process.env,
  repoRoot = process.cwd(),
  readConfigFile = readFileSync,
} = {}) {
  const fallback = path.resolve(repoRoot, "lime-rs", "target");
  const envTargetDirectory =
    env.CARGO_BUILD_TARGET_DIR?.trim() || env.CARGO_TARGET_DIR?.trim();
  if (envTargetDirectory) {
    return path.resolve(repoRoot, envTargetDirectory);
  }

  const configPath = path.resolve(repoRoot, ".cargo", "config.toml");
  try {
    const config = readConfigFile(configPath, "utf8");
    const match = config.match(/^\s*target-dir\s*=\s*["']([^"']+)["']/m);
    if (!match?.[1]?.trim()) {
      return fallback;
    }
    return path.resolve(repoRoot, match[1].trim());
  } catch {
    return fallback;
  }
}

export function resolveDevAppServerBinary({
  env = process.env,
  repoRoot = process.cwd(),
  platform = process.platform,
  targetDirectory,
  exists = existsSync,
  build = buildLocalAppServer,
  forceBuild = false,
  isUsableBinary = isUsableAppServerBinary,
  prepareBinary = prepareLocalAppServerBinary,
} = {}) {
  const envBinary = env.APP_SERVER_BIN?.trim();
  if (envBinary) {
    return envBinary;
  }

  const binaryPath = localAppServerBinaryPath({
    repoRoot,
    platform,
    ...(targetDirectory ? { targetDirectory } : {}),
  });
  const codeModeHostPath = localCodeModeHostBinaryPath({
    repoRoot,
    platform,
    ...(targetDirectory ? { targetDirectory } : {}),
  });
  const hasUsableBinaries = () =>
    isUsableBinary(binaryPath, { exists }) &&
    isUsableBinary(codeModeHostPath, { exists });
  if (forceBuild || !hasUsableBinaries()) {
    build({ repoRoot, platform, env });
  }

  if (!hasUsableBinaries()) {
    throw new Error(
      `app-server sidecar binaries were not created: ${binaryPath}, ${codeModeHostPath}`,
    );
  }

  prepareBinary({ binaryPath, platform });
  prepareBinary({ binaryPath: codeModeHostPath, platform });
  return binaryPath;
}

export function isUsableAppServerBinary(
  binaryPath,
  { exists = existsSync, getStats = statSync } = {},
) {
  if (!exists(binaryPath)) {
    return false;
  }
  try {
    const stats = getStats(binaryPath);
    if (typeof stats.isFile === "function" && !stats.isFile()) {
      return false;
    }
    return stats.size > 0;
  } catch {
    return exists !== existsSync ? exists(binaryPath) : false;
  }
}

export function prepareLocalAppServerBinary({
  binaryPath,
  platform = process.platform,
} = {}) {
  if (!binaryPath) {
    return;
  }
  ensureMacBinaryRpath(binaryPath, { platform });
}

export function resolveDevAppServerBackendEnv({
  env = process.env,
  backendCommand,
  defaultMode = "runtime",
  defaultTimeoutMs = 120_000,
} = {}) {
  const requestedMode = env.APP_SERVER_BACKEND_MODE?.trim();
  if (!requestedMode) {
    return {
      APP_SERVER_BACKEND_MODE: defaultMode,
    };
  }

  if (requestedMode === "runtime" || requestedMode === "unavailable") {
    return {
      APP_SERVER_BACKEND_MODE: requestedMode,
    };
  }

  if (requestedMode !== "external") {
    return {};
  }

  const resolvedBackendCommand =
    env.APP_SERVER_BACKEND_COMMAND?.trim() ||
    String(backendCommand || "").trim();

  return {
    APP_SERVER_BACKEND_MODE: "external",
    ...(env.APP_SERVER_BACKEND_COMMAND?.trim() || !resolvedBackendCommand
      ? {}
      : { APP_SERVER_BACKEND_COMMAND: resolvedBackendCommand }),
    ...(env.APP_SERVER_BACKEND_TIMEOUT_MS?.trim()
      ? {}
      : { APP_SERVER_BACKEND_TIMEOUT_MS: String(defaultTimeoutMs) }),
  };
}

export function cargoBuildAppServerArgs({ repoRoot = process.cwd() } = {}) {
  return [
    "build",
    "--manifest-path",
    path.resolve(repoRoot, "lime-rs", "Cargo.toml"),
    "-p",
    "app-server",
    "--bin",
    "app-server",
    "-p",
    "tool-runtime",
    "--bin",
    "code-mode-host",
  ];
}

export function buildLocalAppServer({
  repoRoot = process.cwd(),
  platform = process.platform,
  arch = process.arch,
  env = process.env,
  runner = spawnSync,
  prepareBinary = prepareLocalAppServerBinary,
  resolveV8Env = resolveRustyV8CargoEnv,
} = {}) {
  console.log("[electron-dev] building local app-server sidecar...");
  const cargoCommand = platform === "win32" ? "cargo.exe" : "cargo";
  const result = runner(cargoCommand, cargoBuildAppServerArgs({ repoRoot }), {
    cwd: repoRoot,
    env: {
      ...env,
      ...resolveV8Env({ env, repoRoot, platform, arch }),
    },
    stdio: "inherit",
    shell: false,
  });

  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`cargo build app-server failed with ${result.status}`);
  }
  prepareBinary({
    binaryPath: localAppServerBinaryPath({ repoRoot, platform }),
    platform,
  });
  prepareBinary({
    binaryPath: localCodeModeHostBinaryPath({ repoRoot, platform }),
    platform,
  });
}

export function buildLocalAppServerAsync({
  repoRoot = process.cwd(),
  platform = process.platform,
  arch = process.arch,
  env = process.env,
  runner = spawn,
  prepareBinary = prepareLocalAppServerBinary,
  resolveV8Env = resolveRustyV8CargoEnv,
} = {}) {
  const cargoCommand = platform === "win32" ? "cargo.exe" : "cargo";
  return new Promise((resolve, reject) => {
    const child = runner(cargoCommand, cargoBuildAppServerArgs({ repoRoot }), {
      cwd: repoRoot,
      env: {
        ...env,
        ...resolveV8Env({ env, repoRoot, platform, arch }),
      },
      stdio: "inherit",
      shell: false,
    });

    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) {
        try {
          prepareBinary({
            binaryPath: localAppServerBinaryPath({ repoRoot, platform }),
            platform,
          });
          prepareBinary({
            binaryPath: localCodeModeHostBinaryPath({ repoRoot, platform }),
            platform,
          });
        } catch (error) {
          reject(error);
          return;
        }
        resolve();
        return;
      }
      const detail = signal ? `signal=${signal}` : `code=${code ?? "unknown"}`;
      reject(new Error(`cargo build app-server failed (${detail})`));
    });
  });
}

export function appServerWatchPaths({ repoRoot = process.cwd() } = {}) {
  return [path.resolve(repoRoot, "lime-rs"), path.resolve(repoRoot, ".cargo")];
}

export function listAppServerWatchDirectories({
  repoRoot = process.cwd(),
  roots = appServerWatchPaths({ repoRoot }),
  readDirectory = readdirSync,
  getStats = statSync,
} = {}) {
  const directories = [];
  const ignoredDirectoryNames = new Set([
    "target",
    "node_modules",
    "dist",
    "dist-electron",
    ".git",
  ]);

  const visit = (directory) => {
    let stats;
    try {
      stats = getStats(directory);
    } catch {
      return;
    }
    if (!stats.isDirectory()) {
      return;
    }

    directories.push(directory);
    let entries;
    try {
      entries = readDirectory(directory, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (!entry.isDirectory() || ignoredDirectoryNames.has(entry.name)) {
        continue;
      }
      visit(path.join(directory, entry.name));
    }
  };

  for (const root of roots) {
    visit(root);
  }

  return directories;
}

export function shouldRebuildAppServer(filename) {
  if (!filename) {
    return true;
  }
  const normalized = String(filename).replaceAll("\\", "/");
  if (
    hasPathSegment(normalized, "target") ||
    hasPathSegment(normalized, "node_modules") ||
    hasPathSegment(normalized, "dist-electron")
  ) {
    return false;
  }

  const basename = path.basename(normalized);
  return (
    normalized.endsWith(".rs") ||
    basename === "Cargo.toml" ||
    basename === "Cargo.lock" ||
    basename === "build.rs" ||
    basename === "config.toml"
  );
}

function hasPathSegment(normalizedPath, segment) {
  return (
    normalizedPath === segment ||
    normalizedPath.startsWith(`${segment}/`) ||
    normalizedPath.includes(`/${segment}/`)
  );
}

export function watchAppServerSources({
  repoRoot = process.cwd(),
  debounceMs = APP_SERVER_WATCH_DEBOUNCE_MS,
  watchFn = watch,
  logger = console,
  onChange,
} = {}) {
  if (typeof onChange !== "function") {
    throw new Error("watchAppServerSources requires onChange");
  }

  const abortController = new AbortController();
  let timer = null;

  const schedule = (sourcePath, filename) => {
    if (!shouldRebuildAppServer(filename)) {
      return;
    }
    clearTimeout(timer);
    timer = setTimeout(() => {
      onChange({
        sourcePath,
        filename: filename ? String(filename) : "",
      });
    }, debounceMs);
  };

  const fallbackRoots = [];
  for (const sourcePath of appServerWatchPaths({ repoRoot })) {
    try {
      watchFn(
        sourcePath,
        {
          recursive: true,
          signal: abortController.signal,
        },
        (_eventType, filename) => schedule(sourcePath, filename),
      );
    } catch (error) {
      logger.warn(
        `[electron-dev] app-server recursive watcher skipped ${sourcePath}: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
      fallbackRoots.push(sourcePath);
    }
  }

  const fallbackDirectories = fallbackRoots.flatMap((root) =>
    listAppServerWatchDirectories({ roots: [root] }),
  );
  for (const sourcePath of fallbackDirectories) {
    try {
      watchFn(
        sourcePath,
        {
          signal: abortController.signal,
        },
        (_eventType, filename) => schedule(sourcePath, filename),
      );
    } catch {
      // 已尝试递归监听；单目录降级失败时忽略该目录，后续保存仍可由上层目录事件触发。
    }
  }

  return {
    close() {
      clearTimeout(timer);
      abortController.abort();
    },
  };
}
