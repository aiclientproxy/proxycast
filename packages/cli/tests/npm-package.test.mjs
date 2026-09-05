import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const PACKAGE_ROOT = path.resolve(
  fileURLToPath(new URL("..", import.meta.url)),
);
const LAUNCHER_SOURCE = path.join(PACKAGE_ROOT, "bin", "lime.js");
const BUILD_SCRIPT = path.join(PACKAGE_ROOT, "scripts", "build_npm_package.py");

function currentPlatform() {
  if (process.platform === "darwin" && process.arch === "arm64") {
    return {
      packageName: "lime-darwin-arm64",
      packageAlias: "@limecloud/lime-darwin-arm64",
      targetTriple: "aarch64-apple-darwin",
    };
  }
  if (process.platform === "darwin" && process.arch === "x64") {
    return {
      packageName: "lime-darwin-x64",
      packageAlias: "@limecloud/lime-darwin-x64",
      targetTriple: "x86_64-apple-darwin",
    };
  }
  if (process.platform === "linux" && process.arch === "x64") {
    return {
      packageName: "lime-linux-x64",
      packageAlias: "@limecloud/lime-linux-x64",
      targetTriple: "x86_64-unknown-linux-gnu",
    };
  }
  if (process.platform === "win32" && process.arch === "x64") {
    return {
      packageName: "lime-win32-x64",
      packageAlias: "@limecloud/lime-win32-x64",
      targetTriple: "x86_64-pc-windows-msvc",
    };
  }
  return null;
}

function createInstalledPackage(t, { platformPackage = true } = {}) {
  const config = currentPlatform();
  assert.ok(config, "test host must be a supported Lime npm target");
  const root = mkdtempSync(path.join(os.tmpdir(), "lime-npm-launcher-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));

  const packageRoot = path.join(root, "node_modules", "@limecloud", "lime");
  const launcherPath = path.join(packageRoot, "bin", "lime.js");
  mkdirSync(path.dirname(launcherPath), { recursive: true });
  copyFileSync(LAUNCHER_SOURCE, launcherPath);
  writeFileSync(
    path.join(packageRoot, "package.json"),
    JSON.stringify({ name: "@limecloud/lime", type: "module" }),
  );

  let vendorRoot = path.join(packageRoot, "vendor");
  if (platformPackage) {
    const aliasRoot = path.join(
      root,
      "node_modules",
      ...config.packageAlias.split("/"),
    );
    mkdirSync(aliasRoot, { recursive: true });
    writeFileSync(
      path.join(aliasRoot, "package.json"),
      JSON.stringify({ name: config.packageAlias, version: "1.2.3" }),
    );
    vendorRoot = path.join(aliasRoot, "vendor");
  }

  const nativePath = path.join(
    vendorRoot,
    config.targetTriple,
    "bin",
    process.platform === "win32" ? "lime.exe" : "lime",
  );
  mkdirSync(path.dirname(nativePath), { recursive: true });
  return { config, launcherPath, nativePath, packageRoot, root };
}

function installFakeNative(tree, source) {
  const scriptPath = path.join(tree.root, "fake-native.cjs");
  writeFileSync(scriptPath, source);
  if (process.platform === "win32") {
    copyFileSync(process.execPath, tree.nativePath);
    return [scriptPath];
  }
  writeFileSync(tree.nativePath, `#!/usr/bin/env node\n${source}`);
  chmodSync(tree.nativePath, 0o755);
  return [];
}

function runLauncher(tree, args = [], env = {}) {
  return spawnSync(process.execPath, [tree.launcherPath, ...args], {
    encoding: "utf8",
    env: { ...process.env, ...env },
  });
}

test("launcher resolves the optional platform package and mirrors exit code", (t) => {
  const tree = createInstalledPackage(t);
  const prefixArgs = installFakeNative(
    tree,
    `console.log(JSON.stringify({
      args: process.argv.slice(2),
      root: process.env.LIME_MANAGED_PACKAGE_ROOT,
      npm: process.env.LIME_MANAGED_BY_NPM,
      pnpm: process.env.LIME_MANAGED_BY_PNPM
    }));
    process.exit(Number(process.env.FAKE_EXIT_CODE || 0));`,
  );
  const result = runLauncher(tree, [...prefixArgs, "alpha", "beta"], {
    FAKE_EXIT_CODE: "23",
    npm_config_user_agent: "npm/10.0.0 node/v22.0.0",
  });

  assert.equal(result.status, 23, result.stderr);
  const payload = JSON.parse(result.stdout.trim());
  assert.deepEqual(payload.args, ["alpha", "beta"]);
  assert.equal(payload.root, realpathSync(tree.packageRoot));
  assert.equal(payload.npm, "1");
  assert.equal(payload.pnpm, undefined);
});

test("launcher uses the root vendor tree only as a development fallback", (t) => {
  const tree = createInstalledPackage(t, { platformPackage: false });
  const prefixArgs = installFakeNative(tree, "console.log('root-vendor');");
  const result = runLauncher(tree, prefixArgs);

  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), "root-vendor");
});

test("launcher fails closed when the optional package payload is missing", (t) => {
  const tree = createInstalledPackage(t);
  rmSync(tree.nativePath, { force: true });
  const result = runLauncher(tree);

  assert.equal(result.status, 1);
  assert.match(result.stderr, /Missing optional dependency @limecloud\/lime-/u);
  assert.match(result.stderr, /Reinstall Lime:/u);
});

test(
  "launcher mirrors native signal termination",
  { skip: process.platform === "win32" },
  (t) => {
    const tree = createInstalledPackage(t);
    writeFileSync(tree.nativePath, "#!/bin/sh\nkill -TERM $$\n");
    chmodSync(tree.nativePath, 0o755);
    const result = runLauncher(tree);

    assert.equal(result.status, null);
    assert.equal(result.signal, "SIGTERM");
  },
);

test(
  "launcher forwards parent termination signals to the native process",
  { skip: process.platform === "win32" },
  async (t) => {
    const tree = createInstalledPackage(t);
    const readyPath = path.join(tree.root, "ready");
    const signalPath = path.join(tree.root, "signal");
    installFakeNative(
      tree,
      `const fs = require('node:fs');
      fs.writeFileSync(process.env.FAKE_READY_PATH, 'ready');
      process.on('SIGTERM', () => {
        fs.writeFileSync(process.env.FAKE_SIGNAL_PATH, 'SIGTERM');
        process.exit(0);
      });
      setInterval(() => {}, 1000);`,
    );

    const child = spawn(process.execPath, [tree.launcherPath], {
      env: {
        ...process.env,
        FAKE_READY_PATH: readyPath,
        FAKE_SIGNAL_PATH: signalPath,
      },
      stdio: "ignore",
    });
    t.after(() => {
      if (child.exitCode === null) {
        child.kill("SIGKILL");
      }
    });
    await waitFor(() => existsSync(readyPath));
    child.kill("SIGTERM");
    const result = await new Promise((resolve) => {
      child.once("exit", (code, signal) => resolve({ code, signal }));
    });

    assert.deepEqual(result, { code: 0, signal: null });
    assert.equal(readFileSync(signalPath, "utf8"), "SIGTERM");
  },
);

test("staging creates Codex-style root aliases and a real npm tarball", (t) => {
  const root = mkdtempSync(path.join(os.tmpdir(), "lime-npm-stage-test-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const staging = path.join(root, "root-package");
  const tarball = path.join(root, "lime-npm-1.2.3.tgz");
  const result = spawnSync(
    "python3",
    [
      BUILD_SCRIPT,
      "--package",
      "lime",
      "--version",
      "1.2.3",
      "--staging-dir",
      staging,
      "--pack-output",
      tarball,
    ],
    { encoding: "utf8" },
  );

  assert.equal(result.status, 0, result.stderr);
  const packageJson = JSON.parse(
    readFileSync(path.join(staging, "package.json"), "utf8"),
  );
  assert.equal(packageJson.bin.lime, "bin/lime.js");
  assert.equal(packageJson.packageManager, "pnpm@9.15.9");
  assert.equal(packageJson.scripts, undefined);
  assert.deepEqual(packageJson.optionalDependencies, {
    "@limecloud/lime-linux-x64": "npm:@limecloud/lime@1.2.3-linux-x64",
    "@limecloud/lime-darwin-x64": "npm:@limecloud/lime@1.2.3-darwin-x64",
    "@limecloud/lime-darwin-arm64": "npm:@limecloud/lime@1.2.3-darwin-arm64",
    "@limecloud/lime-win32-x64": "npm:@limecloud/lime@1.2.3-win32-x64",
  });
  assert.ok(existsSync(tarball));
});

test("platform staging requires the complete App Server runtime payload", (t) => {
  const root = mkdtempSync(path.join(os.tmpdir(), "lime-npm-native-test-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const vendor = path.join(root, "vendor");
  const binDir = path.join(vendor, "aarch64-apple-darwin", "bin");
  mkdirSync(binDir, { recursive: true });
  for (const name of [
    "lime",
    "app-server",
    "code-mode-host",
    "libsherpa-onnx-c-api.dylib",
    "libonnxruntime.1.24.4.dylib",
  ]) {
    writeFileSync(path.join(binDir, name), name);
  }

  const staging = path.join(root, "platform-package");
  const tarball = path.join(root, "lime-npm-darwin-arm64-1.2.3.tgz");
  const success = spawnSync(
    "python3",
    [
      BUILD_SCRIPT,
      "--package",
      "lime-darwin-arm64",
      "--version",
      "1.2.3",
      "--staging-dir",
      staging,
      "--vendor-src",
      vendor,
      "--pack-output",
      tarball,
    ],
    { encoding: "utf8" },
  );
  assert.equal(success.status, 0, success.stderr);
  const packageJson = JSON.parse(
    readFileSync(path.join(staging, "package.json"), "utf8"),
  );
  assert.equal(packageJson.name, "@limecloud/lime");
  assert.equal(packageJson.version, "1.2.3-darwin-arm64");
  assert.deepEqual(packageJson.os, ["darwin"]);
  assert.deepEqual(packageJson.cpu, ["arm64"]);
  assert.equal(packageJson.packageManager, "pnpm@9.15.9");
  assert.ok(existsSync(tarball));
  assert.ok(
    existsSync(
      path.join(staging, "vendor", "aarch64-apple-darwin", "bin", "app-server"),
    ),
  );

  rmSync(staging, { recursive: true, force: true });
  rmSync(path.join(binDir, "app-server"));
  const failure = spawnSync(
    "python3",
    [
      BUILD_SCRIPT,
      "--package",
      "lime-darwin-arm64",
      "--version",
      "1.2.3",
      "--staging-dir",
      staging,
      "--vendor-src",
      vendor,
    ],
    { encoding: "utf8" },
  );
  assert.notEqual(failure.status, 0);
  assert.match(failure.stderr, /missing app-server/u);
});

test("Windows platform staging fails closed without sandbox helpers", (t) => {
  const root = mkdtempSync(path.join(os.tmpdir(), "lime-npm-windows-test-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const vendor = path.join(root, "vendor");
  const binDir = path.join(vendor, "x86_64-pc-windows-msvc", "bin");
  mkdirSync(binDir, { recursive: true });
  for (const name of [
    "lime.exe",
    "app-server.exe",
    "code-mode-host.exe",
    "sherpa-onnx-c-api.dll",
    "onnxruntime.dll",
  ]) {
    writeFileSync(path.join(binDir, name), name);
  }

  const result = spawnSync(
    "python3",
    [
      BUILD_SCRIPT,
      "--package",
      "lime-win32-x64",
      "--version",
      "1.2.3",
      "--staging-dir",
      path.join(root, "platform-package"),
      "--vendor-src",
      vendor,
    ],
    { encoding: "utf8" },
  );

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /windows-sandbox-setup\.exe/u);
  assert.match(result.stderr, /windows-sandbox-runner\.exe/u);
});

async function waitFor(predicate, timeoutMs = 3_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  throw new Error(`condition did not become true within ${timeoutMs}ms`);
}
