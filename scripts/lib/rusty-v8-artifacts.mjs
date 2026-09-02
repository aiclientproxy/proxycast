import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  unlinkSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

export const RUSTY_V8_ARTIFACT_PROFILE = "ptrcomp_sandbox_release";
export const RUSTY_V8_RELEASE_BASE_URL =
  "https://github.com/openai/codex/releases/download";

const RUSTY_V8_DOWNLOAD_CONNECT_TIMEOUT_SECS = "15";
const RUSTY_V8_DOWNLOAD_MAX_TIME_SECS = "120";

const SUPPORTED_TARGETS = new Set([
  "aarch64-apple-darwin",
  "x86_64-apple-darwin",
  "aarch64-unknown-linux-gnu",
  "x86_64-unknown-linux-gnu",
  "aarch64-unknown-linux-musl",
  "x86_64-unknown-linux-musl",
  "aarch64-pc-windows-msvc",
  "x86_64-pc-windows-msvc",
]);

export function resolveRustyV8Target({
  env = process.env,
  platform = process.platform,
  arch = process.arch,
} = {}) {
  const explicitTarget = env.CARGO_BUILD_TARGET?.trim();
  if (explicitTarget) {
    if (!SUPPORTED_TARGETS.has(explicitTarget)) {
      throw new Error(
        `No sandbox-enabled rusty_v8 artifact for ${explicitTarget}`,
      );
    }
    return explicitTarget;
  }

  const target =
    platform === "darwin"
      ? `${rustArch(arch)}-apple-darwin`
      : platform === "linux"
        ? `${rustArch(arch)}-unknown-linux-gnu`
        : platform === "win32"
          ? `${rustArch(arch)}-pc-windows-msvc`
          : null;
  if (!target || !SUPPORTED_TARGETS.has(target)) {
    throw new Error(
      `No sandbox-enabled rusty_v8 artifact for ${platform}-${arch}`,
    );
  }
  return target;
}

function rustArch(arch) {
  if (arch === "arm64") {
    return "aarch64";
  }
  if (arch === "x64") {
    return "x86_64";
  }
  throw new Error(`Unsupported rusty_v8 architecture: ${arch}`);
}

export function resolveV8CrateVersion(cargoLock) {
  const versions = new Set();
  for (const block of String(cargoLock).split(/^\[\[package\]\]\s*$/mu)) {
    if (!/^name\s*=\s*"v8"\s*$/mu.test(block)) {
      continue;
    }
    const match = block.match(/^version\s*=\s*"([^"]+)"\s*$/mu);
    if (match?.[1]) {
      versions.add(match[1]);
    }
  }
  if (versions.size !== 1) {
    throw new Error(
      `Expected exactly one resolved v8 crate version, found: ${[...versions].join(", ") || "none"}`,
    );
  }
  return [...versions][0];
}

export function rustyV8ArtifactNames(
  target,
  profile = RUSTY_V8_ARTIFACT_PROFILE,
) {
  if (!SUPPORTED_TARGETS.has(target)) {
    throw new Error(`No sandbox-enabled rusty_v8 artifact for ${target}`);
  }
  return {
    archive: target.endsWith("-pc-windows-msvc")
      ? `rusty_v8_${profile}_${target}.lib.gz`
      : `librusty_v8_${profile}_${target}.a.gz`,
    binding: `src_binding_${profile}_${target}.rs`,
    checksums: `rusty_v8_${profile}_${target}.sha256`,
  };
}

export function parseRustyV8Checksums(contents, expectedNames) {
  const names = new Set(expectedNames);
  const lines = String(contents).replaceAll("\r", "").trimEnd().split("\n");
  if (lines.length !== names.size) {
    throw new Error(
      `Expected exactly ${names.size} rusty_v8 checksums, found ${lines.length}`,
    );
  }
  const checksums = new Map();
  for (const line of lines) {
    const match = line.match(/^([0-9a-f]{64})\s+\*?([^\s/\\]+)$/u);
    if (!match || !names.has(match[2]) || checksums.has(match[2])) {
      throw new Error(`Invalid rusty_v8 checksum entry: ${line}`);
    }
    checksums.set(match[2], match[1]);
  }
  if (checksums.size !== names.size) {
    throw new Error("rusty_v8 checksum manifest does not cover both artifacts");
  }
  return checksums;
}

export function sha256File(filePath) {
  return createHash("sha256").update(readFileSync(filePath)).digest("hex");
}

export function defaultRustyV8CacheRoot({
  env = process.env,
  platform = process.platform,
  homeDirectory = os.homedir(),
} = {}) {
  const explicitCacheRoot = env.LIME_RUSTY_V8_CACHE_DIR?.trim();
  const pathApi = platform === "win32" ? path.win32 : path.posix;
  if (explicitCacheRoot) {
    return pathApi.resolve(explicitCacheRoot);
  }
  if (platform === "darwin") {
    return pathApi.join(homeDirectory, "Library", "Caches", "Lime", "rusty-v8");
  }
  if (platform === "win32") {
    const localAppData =
      env.LOCALAPPDATA?.trim() ||
      pathApi.join(homeDirectory, "AppData", "Local");
    return pathApi.join(localAppData, "Lime", "Cache", "rusty-v8");
  }
  const cacheHome =
    env.XDG_CACHE_HOME?.trim() || pathApi.join(homeDirectory, ".cache");
  return pathApi.join(cacheHome, "lime", "rusty-v8");
}

export function resolveRustyV8CargoEnv({
  env = process.env,
  repoRoot = process.cwd(),
  platform = process.platform,
  arch = process.arch,
  cacheRoot,
  download = downloadFile,
} = {}) {
  if (["1", "true", "yes"].includes(env.V8_FROM_SOURCE?.trim().toLowerCase())) {
    return {};
  }

  const archiveOverride = env.RUSTY_V8_ARCHIVE?.trim();
  const bindingOverride = env.RUSTY_V8_SRC_BINDING_PATH?.trim();
  if (Boolean(archiveOverride) !== Boolean(bindingOverride)) {
    throw new Error(
      "RUSTY_V8_ARCHIVE and RUSTY_V8_SRC_BINDING_PATH must be set together",
    );
  }

  const target = resolveRustyV8Target({ env, platform, arch });
  const version = resolveV8CrateVersion(
    readFileSync(path.resolve(repoRoot, "lime-rs", "Cargo.lock"), "utf8"),
  );
  const names = rustyV8ArtifactNames(target);
  const releaseUrl = `${RUSTY_V8_RELEASE_BASE_URL}/rusty-v8-v${version}`;
  const resolvedCacheRoot =
    cacheRoot || defaultRustyV8CacheRoot({ env, platform });
  const artifactDirectory = path.resolve(
    resolvedCacheRoot,
    `rusty-v8-v${version}-${target}`,
  );
  mkdirSync(artifactDirectory, { recursive: true });
  const checksumsPath = path.join(artifactDirectory, names.checksums);
  ensureDownloaded(checksumsPath, `${releaseUrl}/${names.checksums}`, download);
  const checksums = parseRustyV8Checksums(readFileSync(checksumsPath, "utf8"), [
    names.archive,
    names.binding,
  ]);

  const archivePath = path.resolve(
    archiveOverride || path.join(artifactDirectory, names.archive),
  );
  const bindingPath = path.resolve(
    bindingOverride || path.join(artifactDirectory, names.binding),
  );
  ensureVerifiedArtifact({
    filePath: archivePath,
    expectedSha256: checksums.get(names.archive),
    url: `${releaseUrl}/${names.archive}`,
    download: archiveOverride ? null : download,
  });
  ensureVerifiedArtifact({
    filePath: bindingPath,
    expectedSha256: checksums.get(names.binding),
    url: `${releaseUrl}/${names.binding}`,
    download: bindingOverride ? null : download,
  });

  const cargoEnv = {
    RUSTY_V8_ARCHIVE: archivePath,
    RUSTY_V8_SRC_BINDING_PATH: bindingPath,
  };
  if (target.endsWith("-pc-windows-msvc")) {
    cargoEnv.CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS =
      "-C target-feature=+crt-static";
  }
  return cargoEnv;
}

function ensureVerifiedArtifact({ filePath, expectedSha256, url, download }) {
  if (existsSync(filePath) && sha256File(filePath) === expectedSha256) {
    return;
  }
  if (!download) {
    throw new Error(
      `rusty_v8 override failed SHA-256 verification: ${filePath}`,
    );
  }
  removeFile(filePath);
  download(url, filePath);
  if (!existsSync(filePath) || sha256File(filePath) !== expectedSha256) {
    removeFile(filePath);
    throw new Error(`Downloaded rusty_v8 artifact failed SHA-256: ${filePath}`);
  }
}

function ensureDownloaded(filePath, url, download) {
  if (!existsSync(filePath)) {
    download(url, filePath);
  }
}

function downloadFile(url, destination) {
  mkdirSync(path.dirname(destination), { recursive: true });
  const temporaryPath = `${destination}.${process.pid}.${Date.now()}.tmp`;
  removeFile(temporaryPath);
  const command = process.platform === "win32" ? "curl.exe" : "curl";
  const result = spawnSync(
    command,
    [
      "--fail",
      "--silent",
      "--show-error",
      "--location",
      "--retry",
      "3",
      "--retry-delay",
      "1",
      "--connect-timeout",
      RUSTY_V8_DOWNLOAD_CONNECT_TIMEOUT_SECS,
      "--max-time",
      RUSTY_V8_DOWNLOAD_MAX_TIME_SECS,
      "--output",
      temporaryPath,
      url,
    ],
    {
      encoding: "utf8",
      shell: false,
      stdio: ["ignore", "ignore", "pipe"],
    },
  );
  if (result.error || result.status !== 0) {
    removeFile(temporaryPath);
    throw (
      result.error || new Error(`Failed to download ${url}: ${result.stderr}`)
    );
  }
  try {
    if (existsSync(destination)) {
      removeFile(temporaryPath);
    } else {
      renameSync(temporaryPath, destination);
    }
  } catch (error) {
    removeFile(temporaryPath);
    if (!existsSync(destination)) {
      throw error;
    }
  }
}

function removeFile(filePath) {
  try {
    unlinkSync(filePath);
  } catch (error) {
    if (error?.code !== "ENOENT") {
      throw error;
    }
  }
}

function main() {
  const cargoEnv = resolveRustyV8CargoEnv();
  if (process.argv.includes("--github-env")) {
    const githubEnv = process.env.GITHUB_ENV?.trim();
    if (!githubEnv) {
      throw new Error("--github-env requires GITHUB_ENV");
    }
    appendFileSync(
      githubEnv,
      `${Object.entries(cargoEnv)
        .map(([name, value]) => `${name}=${value}`)
        .join("\n")}\n`,
    );
    console.log("[rusty-v8] configured verified sandbox artifacts");
    return;
  }
  console.log(JSON.stringify(cargoEnv));
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  main();
}
