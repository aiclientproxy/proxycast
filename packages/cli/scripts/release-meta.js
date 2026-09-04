const VERSION = require("../package.json").version;

const REPO = "limecloud/lime";
const NAME = "lime";

const PLATFORM_MAP = {
  darwin: "darwin",
  linux: "linux",
  win32: "windows",
};

const ARCH_MAP = {
  x64: "amd64",
  arm64: "arm64",
};

const TARGET_TRIPLE_MAP = {
  "aarch64-apple-darwin": {
    platform: "darwin",
    arch: "arm64",
  },
  "x86_64-apple-darwin": {
    platform: "darwin",
    arch: "amd64",
  },
  "x86_64-pc-windows-msvc": {
    platform: "windows",
    arch: "amd64",
  },
  "aarch64-pc-windows-msvc": {
    platform: "windows",
    arch: "arm64",
  },
  "x86_64-unknown-linux-gnu": {
    platform: "linux",
    arch: "amd64",
  },
  "aarch64-unknown-linux-gnu": {
    platform: "linux",
    arch: "arm64",
  },
};

const SUPPORTED_RELEASE_ASSET_IDS = new Set([
  "darwin-arm64",
  "darwin-amd64",
  "windows-amd64",
  "linux-amd64",
]);

function resolvePlatformArch({
  targetTriple,
  platform = process.platform,
  arch = process.arch,
} = {}) {
  if (targetTriple) {
    const resolved = TARGET_TRIPLE_MAP[targetTriple];
    if (!resolved) {
      throw new Error(`Unsupported target triple: ${targetTriple}`);
    }
    return resolved;
  }

  const resolvedPlatform = PLATFORM_MAP[platform];
  const resolvedArch = ARCH_MAP[arch];

  if (!resolvedPlatform || !resolvedArch) {
    throw new Error(`Unsupported platform: ${platform}-${arch}`);
  }

  return {
    platform: resolvedPlatform,
    arch: resolvedArch,
  };
}

function resolveReleaseAssetMeta({
  version = VERSION,
  targetTriple,
  platform,
  arch,
  repo = REPO,
  name = NAME,
} = {}) {
  const resolved = resolvePlatformArch({
    targetTriple,
    platform,
    arch,
  });
  const isWindows = resolved.platform === "windows";
  const archiveExt = isWindows ? ".zip" : ".tar.gz";
  const binaryName = name + (isWindows ? ".exe" : "");
  const archiveName = `${name}-${version}-${resolved.platform}-${resolved.arch}${archiveExt}`;

  return {
    version,
    repo,
    name,
    platform: resolved.platform,
    arch: resolved.arch,
    isWindows,
    archiveExt,
    binaryName,
    archiveName,
    githubUrl: `https://github.com/${repo}/releases/download/v${version}/${archiveName}`,
  };
}

function isSupportedReleaseAssetMeta(meta) {
  if (!meta?.platform || !meta?.arch) {
    return false;
  }
  return SUPPORTED_RELEASE_ASSET_IDS.has(`${meta.platform}-${meta.arch}`);
}

function supportedReleaseAssetLabels() {
  return Array.from(SUPPORTED_RELEASE_ASSET_IDS).map((entry) =>
    entry.replace("-", "/"),
  );
}

module.exports = {
  NAME,
  REPO,
  isSupportedReleaseAssetMeta,
  resolveReleaseAssetMeta,
  supportedReleaseAssetLabels,
};
