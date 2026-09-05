#!/usr/bin/env node
// Unified entry point for the Lime CLI.

import { spawn } from "node:child_process";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const require = createRequire(import.meta.url);
const codexPackageRoot = realpathSync(path.join(__dirname, ".."));

const PLATFORM_PACKAGE_BY_TARGET = {
  "x86_64-unknown-linux-gnu": "@limecloud/lime-linux-x64",
  "x86_64-apple-darwin": "@limecloud/lime-darwin-x64",
  "aarch64-apple-darwin": "@limecloud/lime-darwin-arm64",
  "x86_64-pc-windows-msvc": "@limecloud/lime-win32-x64",
};

const { platform, arch } = process;

let targetTriple = null;
switch (platform) {
  case "linux":
    if (arch === "x64") {
      targetTriple = "x86_64-unknown-linux-gnu";
    }
    break;
  case "darwin":
    if (arch === "x64") {
      targetTriple = "x86_64-apple-darwin";
    } else if (arch === "arm64") {
      targetTriple = "aarch64-apple-darwin";
    }
    break;
  case "win32":
    if (arch === "x64") {
      targetTriple = "x86_64-pc-windows-msvc";
    }
    break;
  default:
    break;
}

if (!targetTriple) {
  throw new Error(`Unsupported platform: ${platform} (${arch})`);
}

const platformPackage = PLATFORM_PACKAGE_BY_TARGET[targetTriple];
if (!platformPackage) {
  throw new Error(`Unsupported target triple: ${targetTriple}`);
}

function findCodexExecutable() {
  let vendorRoot;
  try {
    const packageJsonPath = require.resolve(`${platformPackage}/package.json`);
    vendorRoot = path.join(path.dirname(packageJsonPath), "vendor");
  } catch {
    vendorRoot = path.join(__dirname, "..", "vendor");
  }

  const limeExecutable = path.join(
    vendorRoot,
    targetTriple,
    "bin",
    process.platform === "win32" ? "lime.exe" : "lime",
  );
  if (existsSync(limeExecutable)) {
    return limeExecutable;
  }

  const packageManager = detectPackageManager();
  const updateCommand =
    packageManager === "bun"
      ? "bun install -g @limecloud/lime@latest"
      : packageManager === "pnpm"
        ? "pnpm add -g @limecloud/lime@latest"
        : packageManager === "vite-plus"
          ? "vp install -g @limecloud/lime@latest"
          : "npm install -g @limecloud/lime@latest";
  throw new Error(
    `Missing optional dependency ${platformPackage}. Reinstall Lime: ${updateCommand}`,
  );
}

function isPnpmOwnedCodexInstall(nodeModulesDir) {
  if (!existsSync(path.join(nodeModulesDir, ".modules.yaml"))) {
    return false;
  }

  try {
    return (
      realpathSync(path.join(nodeModulesDir, "@limecloud", "lime")) ===
      codexPackageRoot
    );
  } catch {
    return false;
  }
}

function isVitePlusOwnedCodexInstall(packagesDir) {
  if (path.basename(packagesDir) !== "packages") {
    return false;
  }

  try {
    const metadata = JSON.parse(
      readFileSync(path.join(packagesDir, "@limecloud", "lime.json"), "utf8"),
    );
    if (metadata.name !== "@limecloud/lime") {
      return false;
    }

    const installId = metadata.installId || "";
    const installDir = installId.startsWith("#")
      ? path.join(packagesDir, `@limecloud/lime${installId}`)
      : path.join(packagesDir, "@limecloud/lime", installId);
    for (const nodeModulesDir of [
      path.join(installDir, "lib", "node_modules"),
      path.join(installDir, "node_modules"),
    ]) {
      const packageRoot = path.join(nodeModulesDir, "@limecloud", "lime");
      if (
        existsSync(packageRoot) &&
        realpathSync(packageRoot) === codexPackageRoot
      ) {
        return true;
      }
    }
  } catch {
    // Missing ownership metadata must not prevent Lime from starting.
  }
  return false;
}

function detectPackageManager() {
  const entrypointDir = path.dirname(path.resolve(process.argv[1]));
  for (const startDir of new Set([codexPackageRoot, entrypointDir])) {
    const filesystemRoot = path.parse(startDir).root;
    for (
      let currentDir = startDir;
      currentDir !== filesystemRoot;
      currentDir = path.dirname(currentDir)
    ) {
      if (isVitePlusOwnedCodexInstall(currentDir)) {
        return "vite-plus";
      }
      if (isPnpmOwnedCodexInstall(path.join(currentDir, "node_modules"))) {
        return "pnpm";
      }
    }

    if (isPnpmOwnedCodexInstall(path.join(filesystemRoot, "node_modules"))) {
      return "pnpm";
    }
  }

  const userAgent = process.env.npm_config_user_agent || "";
  if (/\bbun\//u.test(userAgent)) {
    return "bun";
  }

  const execPath = process.env.npm_execpath || "";
  if (execPath.includes("bun")) {
    return "bun";
  }

  if (
    __dirname.includes(".bun/install/global") ||
    __dirname.includes(".bun\\install\\global")
  ) {
    return "bun";
  }

  return userAgent ? "npm" : null;
}

const binaryPath = findCodexExecutable();
const packageManager = detectPackageManager();
const packageManagerEnvVar =
  packageManager === "bun"
    ? "LIME_MANAGED_BY_BUN"
    : packageManager === "pnpm"
      ? "LIME_MANAGED_BY_PNPM"
      : packageManager === "vite-plus"
        ? "LIME_MANAGED_BY_VITE_PLUS"
        : "LIME_MANAGED_BY_NPM";
const env = {
  ...process.env,
  LIME_MANAGED_PACKAGE_ROOT: codexPackageRoot,
};
delete env.LIME_MANAGED_BY_NPM;
delete env.LIME_MANAGED_BY_BUN;
delete env.LIME_MANAGED_BY_PNPM;
delete env.LIME_MANAGED_BY_VITE_PLUS;
env[packageManagerEnvVar] = "1";

// macOS strips DYLD_* variables when Node launches the native child. The
// bundled App Server loads its @rpath libraries from this directory, so pass
// the package-local runtime directory explicitly to the native process.
const nativeRuntimeDir = path.dirname(binaryPath);
if (process.platform === "darwin") {
  env.DYLD_LIBRARY_PATH = [nativeRuntimeDir, env.DYLD_LIBRARY_PATH]
    .filter(Boolean)
    .join(path.delimiter);
} else if (process.platform === "linux") {
  env.LD_LIBRARY_PATH = [nativeRuntimeDir, env.LD_LIBRARY_PATH]
    .filter(Boolean)
    .join(path.delimiter);
}

const child = spawn(binaryPath, process.argv.slice(2), {
  stdio: "inherit",
  env,
});

child.on("error", (error) => {
  console.error(error);
  process.exit(1);
});

const forwardSignal = (signal) => {
  if (child.killed) {
    return;
  }
  try {
    child.kill(signal);
  } catch {
    // The exit handler below remains the single parent termination owner.
  }
};

const signalHandlers = new Map();
for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  const handler = () => forwardSignal(signal);
  signalHandlers.set(signal, handler);
  process.on(signal, handler);
}

const childResult = await new Promise((resolve) => {
  child.on("exit", (code, signal) => {
    if (signal) {
      resolve({ type: "signal", signal });
    } else {
      resolve({ type: "code", exitCode: code ?? 1 });
    }
  });
});

if (childResult.type === "signal") {
  for (const [signal, handler] of signalHandlers) {
    process.removeListener(signal, handler);
  }
  process.kill(process.pid, childResult.signal);
} else {
  process.exit(childResult.exitCode);
}
