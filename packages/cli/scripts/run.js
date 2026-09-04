#!/usr/bin/env node
const fs = require("fs");
const { execFileSync, spawnSync } = require("child_process");
const path = require("path");

const ext = process.platform === "win32" ? ".exe" : "";
const packageDir = path.join(__dirname, "..");
const bundledBin = path.join(packageDir, "bin", "lime" + ext);

function isExecutableFile(filePath) {
  return (
    Boolean(filePath) &&
    fs.existsSync(filePath) &&
    fs.statSync(filePath).isFile()
  );
}

function resolveBinaryCandidates() {
  const candidates = [];
  const push = (candidate) => {
    if (!candidate) {
      return;
    }
    const absolute = path.resolve(candidate);
    if (!candidates.includes(absolute)) {
      candidates.push(absolute);
    }
  };

  push(process.env.LIME_CLI_BINARY_PATH);
  push(bundledBin);
  push(path.join(process.cwd(), "lime-rs", "target", "release", "lime" + ext));
  push(path.join(process.cwd(), "target", "release", "lime" + ext));
  push(
    path.join(
      packageDir,
      "..",
      "..",
      "lime-rs",
      "target",
      "release",
      "lime" + ext,
    ),
  );
  push(path.join(packageDir, "..", "..", "target", "release", "lime" + ext));

  return candidates;
}

function findBinary() {
  return (
    resolveBinaryCandidates().find((candidate) =>
      isExecutableFile(candidate),
    ) || null
  );
}

function cargoExists() {
  const result = spawnSync(
    process.platform === "win32" ? "where" : "which",
    ["cargo"],
    {
      stdio: "ignore",
    },
  );
  return result.status === 0;
}

function findManifestPath() {
  const candidates = [
    path.join(process.cwd(), "lime-rs", "Cargo.toml"),
    path.join(process.cwd(), "Cargo.toml"),
    path.join(packageDir, "..", "..", "lime-rs", "Cargo.toml"),
  ];
  return candidates.find((candidate) => fs.existsSync(candidate)) || null;
}

function runCargoFallback(manifestPath) {
  const args = [
    "run",
    "--quiet",
    "--manifest-path",
    manifestPath,
    "-p",
    "cli",
    "--",
    ...process.argv.slice(2),
  ];
  const result = spawnSync("cargo", args, {
    stdio: "inherit",
    cwd: path.dirname(manifestPath),
  });
  process.exit(result.status || 1);
}

const resolvedBinary = findBinary();

try {
  if (resolvedBinary) {
    execFileSync(resolvedBinary, process.argv.slice(2), { stdio: "inherit" });
    process.exit(0);
  }

  const manifestPath = findManifestPath();
  if (manifestPath && cargoExists()) {
    runCargoFallback(manifestPath);
  }

  console.error(
    [
      "未找到可执行的 lime 二进制。",
      "可选解决方案：",
      "1. 设置环境变量 LIME_CLI_BINARY_PATH 指向本地构建好的二进制",
      "2. 在 Lime 源码仓库内运行命令，让 wrapper 自动使用 cargo run",
      "3. 等待后续 GitHub Release 预编译资产可用后重新安装",
    ].join("\n"),
  );
  process.exit(1);
} catch (error) {
  process.exit(error.status || 1);
}
