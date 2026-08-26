import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import {
  chmod,
  copyFile,
  mkdir,
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import {
  appServerBinaryName,
  codeModeHostBinaryName,
  resolveDevAppServerBinary,
  windowsSandboxRunnerBinaryName,
} from "./electron-dev-sidecar.mjs";
import {
  ensureMacBinaryRpath,
  resolveRuntimeLibrarySource,
  resolveSherpaOnnxSysVersion,
  resolveSherpaRuntimePlan,
} from "../prepare-sherpa-onnx-runtime.mjs";

const execFileAsync = promisify(execFile);
const MACOS_LAUNCH_BLOCKING_XATTRS = [
  "com.apple.quarantine",
  "com.apple.provenance",
];

export const APP_SERVER_RELEASE_MANIFEST_NAME = "app-server.release.json";
export const APP_SERVER_PROTOCOL_VERSION = "appserver.v0";

export function appServerResourceBinaryName(platform = process.platform) {
  return appServerBinaryName(platform);
}

export function codeModeHostResourceBinaryName(platform = process.platform) {
  return codeModeHostBinaryName(platform);
}

export function windowsSandboxSetupResourceBinaryName(
  platform = process.platform,
) {
  return platform === "win32" ? "windows-sandbox-setup.exe" : null;
}

export function windowsSandboxRunnerResourceBinaryName(
  platform = process.platform,
) {
  return windowsSandboxRunnerBinaryName(platform);
}

export function appServerResourcePlatformKey(
  platform = process.platform,
  arch = process.arch,
) {
  if (platform === "win32") {
    return "win32-x64";
  }
  if (platform === "darwin" && arch === "arm64") {
    return "darwin-arm64";
  }
  if (platform === "darwin") {
    return "darwin-x64";
  }
  if (platform === "linux") {
    return "linux-x64";
  }
  return `${platform}-${arch}`;
}

export function electronAppServerResourcesRoot(repoRoot = process.cwd()) {
  return path.resolve(repoRoot, "dist-electron");
}

export function electronAppServerManifestPath({
  outputRoot = electronAppServerResourcesRoot(),
} = {}) {
  return path.resolve(outputRoot, APP_SERVER_RELEASE_MANIFEST_NAME);
}

export function electronAppServerBinaryDestination({
  outputRoot = electronAppServerResourcesRoot(),
  platform = process.platform,
  arch = process.arch,
} = {}) {
  const platformKey = appServerResourcePlatformKey(platform, arch);
  return path.resolve(
    outputRoot,
    "app-server",
    platformKey,
    appServerResourceBinaryName(platform),
  );
}

export function electronCodeModeHostBinaryDestination({
  outputRoot = electronAppServerResourcesRoot(),
  platform = process.platform,
  arch = process.arch,
} = {}) {
  const platformKey = appServerResourcePlatformKey(platform, arch);
  return path.resolve(
    outputRoot,
    "app-server",
    platformKey,
    codeModeHostResourceBinaryName(platform),
  );
}

export function electronWindowsSandboxSetupBinaryDestination({
  outputRoot = electronAppServerResourcesRoot(),
  platform = process.platform,
  arch = process.arch,
} = {}) {
  const name = windowsSandboxSetupResourceBinaryName(platform);
  if (!name) {
    return null;
  }
  const platformKey = appServerResourcePlatformKey(platform, arch);
  return path.resolve(outputRoot, "app-server", platformKey, name);
}

export function electronWindowsSandboxRunnerBinaryDestination({
  outputRoot = electronAppServerResourcesRoot(),
  platform = process.platform,
  arch = process.arch,
} = {}) {
  const name = windowsSandboxRunnerResourceBinaryName(platform);
  if (!name) {
    return null;
  }
  const platformKey = appServerResourcePlatformKey(platform, arch);
  return path.resolve(outputRoot, "app-server", platformKey, name);
}

export async function buildElectronAppServerReleaseManifest({
  binaryPath,
  codeModeHostBinaryPath,
  windowsSandboxSetupBinaryPath,
  windowsSandboxRunnerBinaryPath,
  version,
  platform = appServerResourcePlatformKey(),
  sha256File = hashFile,
}) {
  const normalizedBinaryPath = path.resolve(
    requiredValue(binaryPath, "binaryPath"),
  );
  const normalizedVersion = requiredValue(version, "version");
  const normalizedPlatform = requiredValue(platform, "platform");

  const artifact = {
    platform: normalizedPlatform,
    url: `app-resource://app-server/${normalizedPlatform}/${path.basename(normalizedBinaryPath)}`,
    sha256: await sha256File(normalizedBinaryPath),
    codeModeHostSha256: await sha256File(
      path.resolve(
        requiredValue(codeModeHostBinaryPath, "codeModeHostBinaryPath"),
      ),
    ),
  };
  if (normalizedPlatform === "win32-x64") {
    artifact.windowsSandboxSetupSha256 = await sha256File(
      path.resolve(
        requiredValue(
          windowsSandboxSetupBinaryPath,
          "windowsSandboxSetupBinaryPath",
        ),
      ),
    );
    artifact.windowsSandboxRunnerSha256 = await sha256File(
      path.resolve(
        requiredValue(
          windowsSandboxRunnerBinaryPath,
          "windowsSandboxRunnerBinaryPath",
        ),
      ),
    );
  }

  return {
    version: normalizedVersion,
    protocolVersion: APP_SERVER_PROTOCOL_VERSION,
    artifacts: [artifact],
  };
}

export async function prepareElectronAppServerAssets({
  repoRoot = process.cwd(),
  outputRoot = electronAppServerResourcesRoot(repoRoot),
  platform = process.platform,
  arch = process.arch,
  sourceBinary,
  sourceCodeModeHostBinary,
  sourceWindowsSandboxSetupBinary,
  sourceWindowsSandboxRunnerBinary,
  resolveBinary = resolveDevAppServerBinary,
  env = process.env,
  readPackageJson = readJsonFile,
  copy = copyFile,
  makeDir = mkdir,
  write = writeFile,
  getStat = stat,
  changeMode = chmod,
  sha256File = hashFile,
  clearLaunchBlockingXattrs = clearMacLaunchBlockingXattrs,
  prepareRuntimeBinary = ensureElectronAppServerRuntimeBinary,
  copyRuntimeLibraries = copyElectronAppServerRuntimeLibraries,
} = {}) {
  const packageJson = await readPackageJson(
    path.resolve(repoRoot, "package.json"),
  );
  const version = requiredValue(packageJson.version, "package version");
  const destination = electronAppServerBinaryDestination({
    outputRoot,
    platform,
    arch,
  });
  const codeModeHostDestination = electronCodeModeHostBinaryDestination({
    outputRoot,
    platform,
    arch,
  });
  const windowsSandboxSetupDestination =
    electronWindowsSandboxSetupBinaryDestination({
      outputRoot,
      platform,
      arch,
    });
  const windowsSandboxRunnerDestination =
    electronWindowsSandboxRunnerBinaryDestination({
      outputRoot,
      platform,
      arch,
    });
  const manifestPath = electronAppServerManifestPath({ outputRoot });
  const resolvedSourceBinary = path.resolve(
    sourceBinary ??
      resolveBinary({
        repoRoot,
        platform,
        forceBuild: true,
        env: withoutAppServerBin(env),
      }),
  );
  if (resolvedSourceBinary === destination) {
    throw new Error(
      `Electron app-server asset source must not equal packaged destination: ${destination}`,
    );
  }
  const resolvedSourceCodeModeHostBinary = path.resolve(
    sourceCodeModeHostBinary ??
      path.join(
        path.dirname(resolvedSourceBinary),
        codeModeHostResourceBinaryName(platform),
      ),
  );
  if (resolvedSourceCodeModeHostBinary === codeModeHostDestination) {
    throw new Error(
      `Electron code-mode-host asset source must not equal packaged destination: ${codeModeHostDestination}`,
    );
  }
  const resolvedSourceWindowsSandboxSetupBinary = windowsSandboxSetupDestination
    ? path.resolve(
        sourceWindowsSandboxSetupBinary ??
          path.join(
            path.dirname(resolvedSourceBinary),
            windowsSandboxSetupResourceBinaryName(platform),
          ),
      )
    : null;
  const resolvedSourceWindowsSandboxRunnerBinary =
    windowsSandboxRunnerDestination
      ? path.resolve(
          sourceWindowsSandboxRunnerBinary ??
            path.join(
              path.dirname(resolvedSourceBinary),
              windowsSandboxRunnerResourceBinaryName(platform),
            ),
        )
      : null;
  if (
    windowsSandboxSetupDestination &&
    resolvedSourceWindowsSandboxSetupBinary === windowsSandboxSetupDestination
  ) {
    throw new Error(
      `Windows sandbox setup asset source must not equal packaged destination: ${windowsSandboxSetupDestination}`,
    );
  }
  if (
    windowsSandboxRunnerDestination &&
    resolvedSourceWindowsSandboxRunnerBinary === windowsSandboxRunnerDestination
  ) {
    throw new Error(
      `Windows sandbox runner asset source must not equal packaged destination: ${windowsSandboxRunnerDestination}`,
    );
  }

  await makeDir(path.dirname(destination), { recursive: true });
  await rm(destination, { force: true });
  await rm(codeModeHostDestination, { force: true });
  if (windowsSandboxSetupDestination) {
    await rm(windowsSandboxSetupDestination, { force: true });
  }
  if (windowsSandboxRunnerDestination) {
    await rm(windowsSandboxRunnerDestination, { force: true });
  }
  await copy(resolvedSourceBinary, destination);
  await copy(resolvedSourceCodeModeHostBinary, codeModeHostDestination);
  if (windowsSandboxSetupDestination) {
    await copy(
      resolvedSourceWindowsSandboxSetupBinary,
      windowsSandboxSetupDestination,
    );
  }
  if (windowsSandboxRunnerDestination) {
    await copy(
      resolvedSourceWindowsSandboxRunnerBinary,
      windowsSandboxRunnerDestination,
    );
  }
  await clearLaunchBlockingXattrs(destination, platform);
  await clearLaunchBlockingXattrs(codeModeHostDestination, platform);
  const sourceStat = await getStat(resolvedSourceBinary);
  const codeModeHostSourceStat = await getStat(
    resolvedSourceCodeModeHostBinary,
  );
  const windowsSandboxSetupSourceStat = windowsSandboxSetupDestination
    ? await getStat(resolvedSourceWindowsSandboxSetupBinary)
    : null;
  const windowsSandboxRunnerSourceStat = windowsSandboxRunnerDestination
    ? await getStat(resolvedSourceWindowsSandboxRunnerBinary)
    : null;
  await changeMode(destination, sourceStat.mode);
  await changeMode(codeModeHostDestination, codeModeHostSourceStat.mode);
  if (windowsSandboxSetupDestination) {
    await changeMode(
      windowsSandboxSetupDestination,
      windowsSandboxSetupSourceStat.mode,
    );
  }
  if (windowsSandboxRunnerDestination) {
    await changeMode(
      windowsSandboxRunnerDestination,
      windowsSandboxRunnerSourceStat.mode,
    );
  }
  prepareRuntimeBinary({ binaryPath: destination, platform });
  prepareRuntimeBinary({ binaryPath: codeModeHostDestination, platform });
  if (windowsSandboxSetupDestination) {
    prepareRuntimeBinary({
      binaryPath: windowsSandboxSetupDestination,
      platform,
    });
  }
  if (windowsSandboxRunnerDestination) {
    prepareRuntimeBinary({
      binaryPath: windowsSandboxRunnerDestination,
      platform,
    });
  }
  const runtimeLibraries = await copyRuntimeLibraries({
    repoRoot,
    platform,
    arch,
    sourceBinary: resolvedSourceBinary,
    destinationDirectory: path.dirname(destination),
    copy,
    makeDir,
  });

  const manifest = await buildElectronAppServerReleaseManifest({
    binaryPath: destination,
    codeModeHostBinaryPath: codeModeHostDestination,
    windowsSandboxSetupBinaryPath: windowsSandboxSetupDestination,
    windowsSandboxRunnerBinaryPath: windowsSandboxRunnerDestination,
    version,
    platform: appServerResourcePlatformKey(platform, arch),
    sha256File,
  });
  await write(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

  return {
    sourceBinary: resolvedSourceBinary,
    sourceCodeModeHostBinary: resolvedSourceCodeModeHostBinary,
    sourceWindowsSandboxSetupBinary: resolvedSourceWindowsSandboxSetupBinary,
    sourceWindowsSandboxRunnerBinary: resolvedSourceWindowsSandboxRunnerBinary,
    binaryPath: destination,
    codeModeHostBinaryPath: codeModeHostDestination,
    windowsSandboxSetupBinaryPath: windowsSandboxSetupDestination,
    windowsSandboxRunnerBinaryPath: windowsSandboxRunnerDestination,
    manifestPath,
    manifest,
    runtimeLibraries,
  };
}

export function resolveElectronAppServerSherpaTargetTriple({
  platform = process.platform,
  arch = process.arch,
} = {}) {
  if (platform === "darwin" && arch === "arm64") {
    return "aarch64-apple-darwin";
  }
  if (platform === "darwin" && arch === "x64") {
    return "x86_64-apple-darwin";
  }
  if (platform === "win32") {
    return "x86_64-pc-windows-msvc";
  }
  return null;
}

export async function copyElectronAppServerRuntimeLibraries({
  repoRoot = process.cwd(),
  platform = process.platform,
  arch = process.arch,
  sourceBinary,
  destinationDirectory,
  readCargoLock = readFile,
  copy = copyFile,
  makeDir = mkdir,
  exists = existsSync,
  resolvePlan = resolveSherpaRuntimePlan,
  resolveLibrary = resolveRuntimeLibrarySource,
  targetTriple = resolveElectronAppServerSherpaTargetTriple({ platform, arch }),
} = {}) {
  if (!targetTriple) {
    return [];
  }

  const normalizedDestinationDirectory = path.resolve(
    requiredValue(destinationDirectory, "destinationDirectory"),
  );
  const cargoLockPath = path.resolve(repoRoot, "lime-rs", "Cargo.lock");
  const version = resolveSherpaOnnxSysVersion(
    String(await readCargoLock(cargoLockPath, "utf8")),
  );
  const plan = resolvePlan({ repoRoot, targetTriple, version });
  const runtimeLibraries = resolveElectronAppServerRuntimeLibrarySources({
    plan,
    platform,
    sourceBinary,
    exists,
    resolveLibrary,
  });

  await makeDir(normalizedDestinationDirectory, { recursive: true });
  const copied = [];
  for (const library of runtimeLibraries) {
    const destinationPath = path.join(
      normalizedDestinationDirectory,
      library.name,
    );
    if (path.resolve(library.sourcePath) !== path.resolve(destinationPath)) {
      await copy(library.sourcePath, destinationPath);
    }
    copied.push({
      ...library,
      destinationPath,
    });
  }
  return copied;
}

export function resolveElectronAppServerRuntimeLibrarySources({
  plan,
  platform = process.platform,
  sourceBinary,
  exists = existsSync,
  resolveLibrary = resolveRuntimeLibrarySource,
} = {}) {
  const requiredLibraries = plan?.libs ?? [];
  const optionalLibraries = optionalSherpaRuntimeLibraries(platform).filter(
    (name) => !requiredLibraries.includes(name),
  );
  const resolved = [];

  for (const name of requiredLibraries) {
    const sourcePath = resolvePackagedRuntimeLibrarySource({
      plan,
      name,
      sourceBinary,
      exists,
      resolveLibrary,
    });
    if (!sourcePath) {
      throw new Error(
        `Expected app-server runtime library missing for ${plan.targetTriple}: ${name}`,
      );
    }
    resolved.push({ name, sourcePath, required: true });
  }

  for (const name of optionalLibraries) {
    const sourcePath = resolvePackagedRuntimeLibrarySource({
      plan,
      name,
      sourceBinary,
      exists,
      resolveLibrary,
    });
    if (sourcePath) {
      resolved.push({ name, sourcePath, required: false });
    }
  }

  return resolved;
}

export function resolveElectronAppServerRuntimeEnv({
  env = process.env,
  repoRoot = process.cwd(),
  platform = process.platform,
  manifestPath = electronAppServerManifestPath({
    outputRoot: electronAppServerResourcesRoot(repoRoot),
  }),
  exists = existsSync,
  resolveBinary = resolveDevAppServerBinary,
  prepareRuntimeBinary = ensureElectronAppServerRuntimeBinary,
} = {}) {
  const envBinary = env.APP_SERVER_BIN?.trim();
  if (envBinary) {
    prepareRuntimeBinary({ binaryPath: envBinary, platform });
    return { APP_SERVER_BIN: envBinary };
  }
  if (exists(manifestPath)) {
    return {};
  }
  const appServerBin = resolveBinary({ env, repoRoot, platform });
  prepareRuntimeBinary({ binaryPath: appServerBin, platform });
  return {
    APP_SERVER_BIN: appServerBin,
  };
}

export function ensureElectronAppServerRuntimeBinary({
  binaryPath,
  platform = process.platform,
} = {}) {
  if (!binaryPath) {
    return;
  }
  ensureMacBinaryRpath(binaryPath, { platform });
}

async function hashFile(filePath) {
  const content = await readFile(filePath);
  return createHash("sha256").update(content).digest("hex");
}

async function readJsonFile(filePath) {
  return JSON.parse(await readFile(filePath, "utf8"));
}

async function clearMacLaunchBlockingXattrs(filePath, platform) {
  if (platform !== "darwin") {
    return;
  }

  for (const attribute of MACOS_LAUNCH_BLOCKING_XATTRS) {
    try {
      await execFileAsync("xattr", ["-d", attribute, filePath]);
    } catch (error) {
      const stderr = String(error?.stderr || "");
      if (stderr.includes("No such xattr") || stderr.includes("No such file")) {
        continue;
      }
      throw error;
    }
  }
}

function requiredValue(value, name) {
  const normalized = String(value || "").trim();
  if (!normalized) {
    throw new Error(`${name} is required`);
  }
  return normalized;
}

function withoutAppServerBin(env) {
  const nextEnv = { ...env };
  delete nextEnv.APP_SERVER_BIN;
  return nextEnv;
}

function resolvePackagedRuntimeLibrarySource({
  plan,
  name,
  sourceBinary,
  exists,
  resolveLibrary,
}) {
  const sourceBinaryPath = String(sourceBinary || "").trim();
  if (sourceBinaryPath) {
    const adjacentPath = path.join(
      path.dirname(path.resolve(sourceBinaryPath)),
      name,
    );
    if (exists(adjacentPath)) {
      return adjacentPath;
    }
  }
  return resolveLibrary(plan, name);
}

function optionalSherpaRuntimeLibraries(platform) {
  if (platform === "darwin") {
    return ["libsherpa-onnx-cxx-api.dylib"];
  }
  if (platform === "win32") {
    return ["sherpa-onnx-cxx-api.dll"];
  }
  return [];
}
