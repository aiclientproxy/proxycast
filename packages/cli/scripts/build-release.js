const fs = require("fs");
const os = require("os");
const path = require("path");
const { execFileSync } = require("child_process");
const { resolveReleaseAssetMeta } = require("./release-meta");

const SCRIPT_DIR = __dirname;
const PACKAGE_DIR = path.resolve(SCRIPT_DIR, "..");
const REPO_ROOT = path.resolve(PACKAGE_DIR, "..", "..");

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const current = argv[index];
    if (!current.startsWith("--")) {
      continue;
    }

    const key = current.slice(2);
    if (key === "json" || key === "help") {
      parsed[key] = true;
      continue;
    }

    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      throw new Error(`Missing value for --${key}`);
    }
    parsed[key] = next;
    index += 1;
  }
  return parsed;
}

function printHelp() {
  console.log(`Usage:
  node packages/cli/scripts/build-release.js [options]

Options:
  --binary <path>         Explicit lime binary path
  --target-triple <triple> Rust target triple, e.g. x86_64-apple-darwin
  --platform <platform>   Host platform fallback, e.g. darwin/linux/win32
  --arch <arch>           Host arch fallback, e.g. x64/arm64
  --version <version>     Release version, defaults to packages/cli/package.json
  --out-dir <path>        Output directory, defaults to packages/cli/dist
  --json                  Print JSON result only
  --help                  Show this help`);
}

function resolveBinaryCandidates(meta, targetTriple, explicitBinary) {
  const candidates = [];
  const push = (value) => {
    if (!value) {
      return;
    }
    const absolute = path.resolve(value);
    if (!candidates.includes(absolute)) {
      candidates.push(absolute);
    }
  };

  push(explicitBinary);

  if (targetTriple) {
    push(
      path.join(
        REPO_ROOT,
        "lime-rs",
        "target",
        targetTriple,
        "release",
        meta.binaryName,
      ),
    );
    push(
      path.join(REPO_ROOT, "target", targetTriple, "release", meta.binaryName),
    );
  }

  push(path.join(REPO_ROOT, "lime-rs", "target", "release", meta.binaryName));
  push(path.join(REPO_ROOT, "target", "release", meta.binaryName));

  return candidates;
}

function resolveBinaryPath(meta, targetTriple, explicitBinary) {
  const candidates = resolveBinaryCandidates(
    meta,
    targetTriple,
    explicitBinary,
  );
  const resolved = candidates.find((candidate) => fs.existsSync(candidate));
  if (!resolved) {
    throw new Error(
      `Unable to find ${meta.binaryName}. Tried:\n${candidates.map((value) => `- ${value}`).join("\n")}`,
    );
  }
  return resolved;
}

function createTarGz(archivePath, stagingDir, binaryName) {
  execFileSync("tar", ["-czf", archivePath, "-C", stagingDir, binaryName], {
    stdio: "inherit",
  });
}

function createZip(archivePath, stagedBinaryPath) {
  if (process.platform === "win32") {
    execFileSync(
      "powershell",
      [
        "-NoProfile",
        "-Command",
        `Compress-Archive -Path '${stagedBinaryPath}' -DestinationPath '${archivePath}' -Force`,
      ],
      { stdio: "inherit" },
    );
    return;
  }

  execFileSync("zip", ["-j", "-q", archivePath, stagedBinaryPath], {
    stdio: "inherit",
  });
}

function createArchive(meta, binaryPath, outDir) {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "lime-release-"));
  try {
    const stagedBinaryPath = path.join(tempDir, meta.binaryName);
    fs.copyFileSync(binaryPath, stagedBinaryPath);
    if (!meta.isWindows) {
      fs.chmodSync(stagedBinaryPath, 0o755);
    }

    fs.mkdirSync(outDir, { recursive: true });
    const archivePath = path.join(outDir, meta.archiveName);

    if (meta.isWindows) {
      createZip(archivePath, stagedBinaryPath);
    } else {
      createTarGz(archivePath, tempDir, meta.binaryName);
    }

    return archivePath;
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printHelp();
    return;
  }

  const meta = resolveReleaseAssetMeta({
    version: args.version,
    targetTriple: args["target-triple"],
    platform: args.platform,
    arch: args.arch,
  });
  const outputDir = path.resolve(
    args["out-dir"] || path.join(PACKAGE_DIR, "dist"),
  );
  const binaryPath = resolveBinaryPath(
    meta,
    args["target-triple"],
    args.binary || process.env.LIME_CLI_BINARY_PATH,
  );
  const archivePath = createArchive(meta, binaryPath, outputDir);
  const result = {
    archiveName: meta.archiveName,
    archivePath,
    binaryPath,
    platform: meta.platform,
    arch: meta.arch,
    version: meta.version,
  };

  if (args.json) {
    console.log(JSON.stringify(result));
    return;
  }

  console.log(`Built ${meta.archiveName}`);
  console.log(archivePath);
}

try {
  main();
} catch (error) {
  console.error(`[lime release] ${error.message}`);
  process.exit(1);
}
