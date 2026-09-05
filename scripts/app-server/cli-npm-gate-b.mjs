#!/usr/bin/env node

import { spawn } from "node:child_process";
import {
  access,
  copyFile,
  link,
  mkdir,
  mkdtemp,
  readdir,
  rm,
  symlink,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const rootDir = path.resolve(path.dirname(__filename), "../..");
const cliPackageDir = path.join(rootDir, "packages", "cli");
const buildScript = path.join(cliPackageDir, "scripts", "build_npm_package.py");

const PLATFORM = resolvePlatform();

async function main() {
  const tempDir = await mkdtemp(path.join(tmpdir(), "cli-npm-gate-b-"));
  try {
    const profileDir = path.join(rootDir, "lime-rs", "target", "debug");
    const vendorRoot = path.join(tempDir, "vendor");
    const vendorBin = path.join(vendorRoot, PLATFORM.targetTriple, "bin");
    await mkdir(vendorBin, { recursive: true });
    await stageRuntimePayload(profileDir, vendorBin);

    const rootStage = path.join(tempDir, "root-stage");
    const platformStage = path.join(tempDir, "platform-stage");
    await run("python3", [
      buildScript,
      "--package",
      "lime",
      "--version",
      "0.0.0-gate-b",
      "--staging-dir",
      rootStage,
    ]);
    await run("python3", [
      buildScript,
      "--package",
      PLATFORM.package,
      "--version",
      "0.0.0-gate-b",
      "--staging-dir",
      platformStage,
      "--vendor-src",
      vendorRoot,
    ]);

    const installScope = path.join(
      tempDir,
      "install",
      "node_modules",
      "@limecloud",
    );
    const installedRoot = path.join(installScope, "lime");
    const installedPlatform = path.join(installScope, PLATFORM.aliasName);
    await mkdir(installScope, { recursive: true });
    await copyPackageRoot(rootStage, installedRoot);
    await symlink(platformStage, installedPlatform, "junction");

    const executableSuffix = process.platform === "win32" ? ".exe" : "";
    const launcherPath = path.join(installedRoot, "bin", "lime.js");
    const packagedAppServer = path.join(
      platformStage,
      "vendor",
      PLATFORM.targetTriple,
      "bin",
      `app-server${executableSuffix}`,
    );
    await run(
      process.execPath,
      [path.join(rootDir, "scripts", "app-server", "cli-gate-b.mjs")],
      {
        APP_SERVER_BIN: packagedAppServer,
        LIME_CLI_BIN: launcherPath,
        LIME_CLI_GATE_B_USE_SIBLING_APP_SERVER: "1",
      },
    );
    await run(
      process.execPath,
      [path.join(rootDir, "scripts", "app-server", "tui-gate-b.mjs")],
      {
        APP_SERVER_BIN: packagedAppServer,
        LIME_CLI_BIN: launcherPath,
        LIME_TUI_GATE_B_SCENARIOS: "complete",
      },
    );

    console.log(
      `[smoke:cli-npm-gate-b] ok target=${PLATFORM.targetTriple} launcher=${launcherPath} appServer=sibling tui=complete`,
    );
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

function resolvePlatform() {
  if (process.platform === "darwin" && process.arch === "arm64") {
    return {
      aliasName: "lime-darwin-arm64",
      package: "lime-darwin-arm64",
      targetTriple: "aarch64-apple-darwin",
    };
  }
  if (process.platform === "darwin" && process.arch === "x64") {
    return {
      aliasName: "lime-darwin-x64",
      package: "lime-darwin-x64",
      targetTriple: "x86_64-apple-darwin",
    };
  }
  if (process.platform === "linux" && process.arch === "x64") {
    return {
      aliasName: "lime-linux-x64",
      package: "lime-linux-x64",
      targetTriple: "x86_64-unknown-linux-gnu",
    };
  }
  if (process.platform === "win32" && process.arch === "x64") {
    return {
      aliasName: "lime-win32-x64",
      package: "lime-win32-x64",
      targetTriple: "x86_64-pc-windows-msvc",
    };
  }
  throw new Error(
    `Unsupported CLI npm Gate B host: ${process.platform}/${process.arch}`,
  );
}

async function stageRuntimePayload(profileDir, vendorBin) {
  const suffix = process.platform === "win32" ? ".exe" : "";
  const binaryNames = [
    `lime${suffix}`,
    `app-server${suffix}`,
    `code-mode-host${suffix}`,
  ];
  if (process.platform === "win32") {
    binaryNames.push("windows-sandbox-setup.exe", "windows-sandbox-runner.exe");
  }

  const entries = await readdir(profileDir, { withFileTypes: true });
  const runtimeLibraries = entries
    .filter((entry) => entry.isFile() || entry.isSymbolicLink())
    .map((entry) => entry.name)
    .filter((name) => /(?:\.dylib|\.dll|\.so(?:\..*)?)$/u.test(name));
  for (const name of [...binaryNames, ...runtimeLibraries]) {
    const source = path.join(profileDir, name);
    const destination = path.join(vendorBin, name);
    await assertFile(source, name);
    try {
      await link(source, destination);
    } catch {
      await copyFile(source, destination);
    }
  }
}

async function copyPackageRoot(source, destination) {
  await mkdir(path.join(destination, "bin"), { recursive: true });
  await Promise.all([
    copyFile(
      path.join(source, "package.json"),
      path.join(destination, "package.json"),
    ),
    copyFile(
      path.join(source, "bin", "lime.js"),
      path.join(destination, "bin", "lime.js"),
    ),
  ]);
}

async function assertFile(filePath, label) {
  try {
    await access(filePath);
  } catch {
    throw new Error(
      `${label} is missing: ${filePath}\n` +
        '先构建：cargo build --manifest-path "lime-rs/Cargo.toml" -p cli -p app-server -p code-mode-host --bin code-mode-host',
    );
  }
}

async function run(command, args, extraEnv = {}) {
  await new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: rootDir,
      env: { ...process.env, ...extraEnv },
      stdio: "inherit",
      windowsHide: true,
    });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(
        new Error(
          `${command} exited with ${signal ? `signal ${signal}` : `code ${code ?? 1}`}`,
        ),
      );
    });
  });
}

main().catch((error) => {
  console.error(
    `[smoke:cli-npm-gate-b] failed: ${error instanceof Error ? error.message : String(error)}`,
  );
  process.exitCode = 1;
});
