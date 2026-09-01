import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import {
  electronAppServerBinaryDestination,
  electronCodeModeHostBinaryDestination,
} from "./electron-app-server-assets.mjs";

const BUILD_READY_ENV = "LIME_ELECTRON_FIXTURE_BUILD_READY";
const BUILD_LOCK_DIR = path.join(".lime", "electron-fixture-build.lock");
const BUILD_LOCK_WAIT_MS = 1_000;
const BUILD_LOCK_TIMEOUT_MS = 20 * 60_000;
const BUILD_LOCK_STALE_MS = 30 * 60_000;
const RENDERER_SOURCE_DIRS = [
  "src",
  "public",
  "packages/app-server-client/src",
  "packages/agent-runtime-client/src",
  "packages/agent-runtime-projection/src",
  "packages/agent-runtime-ui/src",
  "packages/agent-ui-contracts/src",
];
const RENDERER_SOURCE_FILES = [
  "index.html",
  "package-lock.json",
  "package.json",
  "tsconfig.json",
  "tsconfig.renderer.json",
  "vite.config.ts",
  "scripts/electron/build-renderer-smoke.mjs",
  "scripts/electron/build-renderer.mjs",
  "scripts/electron/renderer-build-env.mjs",
];
const HOST_SOURCE_DIRS = [
  "electron",
  "packages/app-server-client/src",
  "assets/plugins",
];
const HOST_SOURCE_FILES = [
  "package-lock.json",
  "package.json",
  "tsconfig.electron.json",
  "tsconfig.json",
  "tsconfig.node.json",
  "packages/app-server-client/package.json",
  "packages/app-server-client/tsconfig.json",
  "scripts/electron/build-host.mjs",
  "scripts/electron/copy-desktop-assets.mjs",
  "scripts/lib/electron-desktop-resources.mjs",
];
const APP_SERVER_SOURCE_DIRS = ["lime-rs/crates", "lime-rs/vendor"];
const APP_SERVER_SOURCE_FILES = [
  "package.json",
  "lime-rs/Cargo.lock",
  "lime-rs/Cargo.toml",
  "scripts/electron/prepare-app-server-assets.mjs",
  "scripts/lib/electron-app-server-assets.mjs",
  "scripts/lib/electron-dev-sidecar.mjs",
];
const SOURCE_EXTENSIONS = new Set([
  ".cjs",
  ".cpp",
  ".css",
  ".html",
  ".ico",
  ".gif",
  ".js",
  ".jpeg",
  ".jpg",
  ".json",
  ".json5",
  ".jsx",
  ".less",
  ".md",
  ".mdx",
  ".mjs",
  ".mp3",
  ".png",
  ".rs",
  ".sass",
  ".scss",
  ".swift",
  ".svg",
  ".toml",
  ".ts",
  ".tsx",
  ".wav",
  ".wasm",
  ".webp",
  ".woff",
  ".woff2",
  ".yaml",
  ".yml",
]);
const IGNORED_DIR_NAMES = new Set([
  ".git",
  ".next",
  ".turbo",
  "coverage",
  "dist",
  "dist-electron",
  "node_modules",
  "target",
]);
const IGNORED_SOURCE_FILE_PATTERNS = [
  /(^|[/\\])__snapshots__([/\\]|$)/,
  /(^|[/\\])tests([/\\]|$)/,
  /\.spec\.[cm]?[jt]sx?$/,
  /\.test\.[cm]?[jt]sx?$/,
  /_test\.rs$/,
];

function npmCommand() {
  return process.platform === "win32" ? "npm.cmd" : "npm";
}

function sleepSync(ms) {
  const buffer = new SharedArrayBuffer(4);
  Atomics.wait(new Int32Array(buffer), 0, 0, ms);
}

function nodeCommand() {
  return process.execPath;
}

function electronFixtureBuildSegments({ rootDir }) {
  return [
    {
      id: "renderer",
      label: "renderer",
      requiredFiles: [path.join(rootDir, "dist", "index.html")],
      sourceDirs: RENDERER_SOURCE_DIRS,
      sourceFiles: RENDERER_SOURCE_FILES,
      buildSteps: [
        {
          label: "build renderer",
          command: npmCommand(),
          args: ["run", "build:renderer:electron:smoke"],
        },
      ],
    },
    {
      id: "host",
      label: "Electron host",
      requiredFiles: [
        path.join(rootDir, "dist-electron", "main", "main.js"),
        path.join(rootDir, "dist-electron", "preload", "preload.cjs"),
        path.join(
          rootDir,
          "dist-electron",
          "plugins",
          "openai-bundled",
          ".agents",
          "plugins",
          "marketplace.json",
        ),
      ],
      sourceDirs: HOST_SOURCE_DIRS,
      sourceFiles: HOST_SOURCE_FILES,
      buildSteps: [
        {
          label: "build app-server client",
          command: npmCommand(),
          args: ["--prefix", "packages/app-server-client", "run", "build"],
        },
        {
          label: "typecheck Electron host",
          command: npmCommand(),
          args: ["run", "typecheck:electron"],
        },
        {
          label: "build Electron host",
          command: nodeCommand(),
          args: ["scripts/electron/build-host.mjs"],
        },
        {
          label: "copy desktop assets",
          command: npmCommand(),
          args: ["run", "electron:build:assets"],
        },
      ],
    },
    {
      id: "appServer",
      label: "app-server sidecar",
      requiredFiles: [
        path.join(rootDir, "dist-electron", "app-server.release.json"),
        path.join(rootDir, "dist-electron", "desktop-resources.manifest.json"),
        electronAppServerBinaryDestination({
          outputRoot: path.join(rootDir, "dist-electron"),
        }),
        electronCodeModeHostBinaryDestination({
          outputRoot: path.join(rootDir, "dist-electron"),
        }),
      ],
      sourceDirs: APP_SERVER_SOURCE_DIRS,
      sourceFiles: APP_SERVER_SOURCE_FILES,
      buildSteps: [
        {
          label: "prepare app-server sidecar",
          command: npmCommand(),
          args: ["run", "electron:build:app-server-assets"],
        },
      ],
    },
  ];
}

export function electronFixtureBuildReadyEnv() {
  return BUILD_READY_ENV;
}

export function electronFixtureBuildRequiredFiles({ rootDir }) {
  return electronFixtureBuildSegments({ rootDir }).flatMap(
    (segment) => segment.requiredFiles,
  );
}

function isIgnoredSourceFile(filePath) {
  return IGNORED_SOURCE_FILE_PATTERNS.some((pattern) => pattern.test(filePath));
}

function statFile(filePath) {
  try {
    return statSync(filePath);
  } catch {
    return null;
  }
}

function createMtimeEntry(rootDir, filePath, stats) {
  return {
    path: filePath,
    relativePath: path.relative(rootDir, filePath),
    mtimeMs: stats.mtimeMs,
  };
}

function pickOlder(left, right) {
  if (!left) {
    return right;
  }
  if (!right) {
    return left;
  }
  return right.mtimeMs < left.mtimeMs ? right : left;
}

function pickNewer(left, right) {
  if (!left) {
    return right;
  }
  if (!right) {
    return left;
  }
  return right.mtimeMs > left.mtimeMs ? right : left;
}

function findOldestRequiredArtifact(rootDir, requiredFiles) {
  return requiredFiles.reduce((oldest, filePath) => {
    const stats = statFile(filePath);
    if (!stats?.isFile()) {
      return oldest;
    }
    return pickOlder(oldest, createMtimeEntry(rootDir, filePath, stats));
  }, null);
}

function findNewestBuildInputInDir(rootDir, sourceDir) {
  const absoluteDir = path.join(rootDir, sourceDir);
  const rootStats = statFile(absoluteDir);
  if (!rootStats?.isDirectory()) {
    return null;
  }

  const stack = [absoluteDir];
  let newest = null;
  while (stack.length > 0) {
    const currentDir = stack.pop();
    const entries = readdirSync(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      const entryPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        if (!IGNORED_DIR_NAMES.has(entry.name)) {
          stack.push(entryPath);
        }
        continue;
      }
      if (!entry.isFile()) {
        continue;
      }
      if (
        !SOURCE_EXTENSIONS.has(path.extname(entry.name)) ||
        isIgnoredSourceFile(entryPath)
      ) {
        continue;
      }
      const stats = statFile(entryPath);
      if (stats?.isFile()) {
        newest = pickNewer(newest, createMtimeEntry(rootDir, entryPath, stats));
      }
    }
  }
  return newest;
}

function findNewestBuildInput(rootDir, { sourceDirs, sourceFiles }) {
  let newest = null;
  for (const sourceFile of sourceFiles) {
    const filePath = path.join(rootDir, sourceFile);
    const stats = statFile(filePath);
    if (stats?.isFile()) {
      newest = pickNewer(newest, createMtimeEntry(rootDir, filePath, stats));
    }
  }
  for (const sourceDir of sourceDirs) {
    newest = pickNewer(newest, findNewestBuildInputInDir(rootDir, sourceDir));
  }
  return newest;
}

function inspectElectronFixtureBuildSegmentFreshness(rootDir, segment) {
  const requiredFiles = segment.requiredFiles;
  const missingFiles = requiredFiles.filter(
    (filePath) => !existsSync(filePath),
  );
  if (missingFiles.length > 0) {
    return {
      id: segment.id,
      label: segment.label,
      ready: false,
      reason: "missing-artifacts",
      requiredFiles,
      missingFiles,
      newestSource: findNewestBuildInput(rootDir, segment),
      oldestArtifact: null,
    };
  }

  const oldestArtifact = findOldestRequiredArtifact(rootDir, requiredFiles);
  const newestSource = findNewestBuildInput(rootDir, segment);
  if (
    oldestArtifact &&
    newestSource &&
    newestSource.mtimeMs > oldestArtifact.mtimeMs
  ) {
    return {
      id: segment.id,
      label: segment.label,
      ready: false,
      reason: "stale-source",
      requiredFiles,
      missingFiles: [],
      newestSource,
      oldestArtifact,
    };
  }

  return {
    id: segment.id,
    label: segment.label,
    ready: true,
    reason: "fresh-artifacts",
    requiredFiles,
    missingFiles: [],
    newestSource,
    oldestArtifact,
  };
}

export function inspectElectronFixtureBuildFreshness({ rootDir }) {
  const segmentFreshness = electronFixtureBuildSegments({ rootDir }).map(
    (segment) => inspectElectronFixtureBuildSegmentFreshness(rootDir, segment),
  );
  const requiredFiles = segmentFreshness.flatMap(
    (segment) => segment.requiredFiles,
  );
  const missingFiles = segmentFreshness.flatMap((segment) =>
    segment.missingFiles.map((filePath) => filePath),
  );
  const staleSegments = segmentFreshness.filter((segment) => !segment.ready);
  const aggregateSegments =
    staleSegments.length > 0 ? staleSegments : segmentFreshness;
  const oldestArtifact = aggregateSegments.reduce(
    (oldest, segment) => pickOlder(oldest, segment.oldestArtifact),
    null,
  );
  const newestSource = aggregateSegments.reduce(
    (newest, segment) => pickNewer(newest, segment.newestSource),
    null,
  );
  const reason =
    missingFiles.length > 0
      ? "missing-artifacts"
      : staleSegments.length > 0
        ? "stale-source"
        : "fresh-artifacts";

  return {
    ready: staleSegments.length === 0,
    reason,
    requiredFiles,
    missingFiles,
    newestSource,
    oldestArtifact,
    segments: Object.fromEntries(
      segmentFreshness.map((segment) => [segment.id, segment]),
    ),
    staleSegments: staleSegments.map((segment) => segment.id),
  };
}

function buildReadyResult(logPrefix, freshness) {
  console.log(
    `[${logPrefix}] reusing fresh packaged fixture assets; oldest=${freshness.oldestArtifact?.relativePath ?? "unknown"} newestSource=${freshness.newestSource?.relativePath ?? "none"}`,
  );
  process.env[BUILD_READY_ENV] = "1";
  return {
    status: "reused",
    reason: freshness.reason,
    freshness,
  };
}

function runCommandSync({ command, args, cwd, env, stdio }) {
  return spawnSync(command, args, {
    cwd,
    stdio,
    env,
  });
}

function runBuildStep(step, { rootDir, env, logPrefix, runCommand }) {
  console.log(`[${logPrefix}] > ${step.label}`);
  const result = runCommand({
    command: step.command,
    args: step.args,
    cwd: rootDir,
    stdio: "inherit",
    env,
  });

  if (result.error) {
    throw result.error;
  }
  if (typeof result.status === "number" && result.status !== 0) {
    const error = new Error(
      `[${logPrefix}] ${step.label} 失败: ${step.command} ${step.args.join(
        " ",
      )}`,
    );
    error.exitCode = result.status;
    throw error;
  }
}

function buildStaleElectronFixtureSegments({
  freshness,
  logPrefix,
  rootDir,
  runCommand,
}) {
  const staleSegmentIds = new Set(freshness.staleSegments);
  const segmentsToBuild = electronFixtureBuildSegments({ rootDir }).filter(
    (segment) => staleSegmentIds.has(segment.id),
  );
  if (segmentsToBuild.length === 0) {
    return;
  }

  const buildEnv = {
    ...process.env,
    LIME_ALLOW_LIVE_PROVIDER_SMOKE: "0",
    LIME_REAL_API_TEST: "0",
  };
  const preflightSteps = [
    {
      label: "verify app version",
      command: npmCommand(),
      args: ["run", "verify:app-version"],
    },
  ];

  for (const step of preflightSteps) {
    runBuildStep(step, {
      rootDir,
      env: buildEnv,
      logPrefix,
      runCommand,
    });
  }

  for (const segment of segmentsToBuild) {
    console.log(`[${logPrefix}] rebuilding ${segment.label} fixture assets`);
    for (const step of segment.buildSteps) {
      runBuildStep(step, {
        rootDir,
        env: buildEnv,
        logPrefix,
        runCommand,
      });
    }
  }
}

function acquireElectronFixtureBuildLock({ rootDir, logPrefix }) {
  const lockDir = path.join(rootDir, BUILD_LOCK_DIR);
  const startedAt = Date.now();
  let loggedWait = false;

  while (Date.now() - startedAt < BUILD_LOCK_TIMEOUT_MS) {
    mkdirSync(path.dirname(lockDir), { recursive: true });
    try {
      mkdirSync(lockDir);
      writeFileSync(
        path.join(lockDir, "owner.json"),
        `${JSON.stringify(
          {
            pid: process.pid,
            createdAt: new Date().toISOString(),
          },
          null,
          2,
        )}\n`,
      );
      return () => {
        rmSync(lockDir, { force: true, recursive: true });
      };
    } catch (error) {
      if (error?.code !== "EEXIST") {
        throw error;
      }
    }

    const lockStats = statFile(lockDir);
    if (lockStats && Date.now() - lockStats.mtimeMs > BUILD_LOCK_STALE_MS) {
      console.warn(
        `[${logPrefix}] removing stale Electron fixture build lock after ${Math.round(
          (Date.now() - lockStats.mtimeMs) / 1000,
        )}s: ${path.relative(rootDir, lockDir)}`,
      );
      rmSync(lockDir, { force: true, recursive: true });
      continue;
    }

    if (!loggedWait) {
      console.log(
        `[${logPrefix}] waiting for Electron fixture packaged renderer/assets build lock`,
      );
      loggedWait = true;
    }
    sleepSync(BUILD_LOCK_WAIT_MS);
  }

  throw new Error(
    `[${logPrefix}] timed out waiting for Electron fixture packaged renderer/assets build lock: ${path.relative(
      rootDir,
      lockDir,
    )}`,
  );
}

export function ensureElectronFixtureBuild({
  appUrl = "",
  logPrefix,
  rootDir,
  runCommand = runCommandSync,
}) {
  if (appUrl) {
    return {
      status: "skipped",
      reason: "app-url",
    };
  }

  if (process.env[BUILD_READY_ENV] === "1") {
    return {
      status: "skipped",
      reason: BUILD_READY_ENV,
    };
  }

  const freshness = inspectElectronFixtureBuildFreshness({ rootDir });
  if (freshness.ready) {
    return buildReadyResult(logPrefix, freshness);
  }

  const releaseBuildLock = acquireElectronFixtureBuildLock({
    rootDir,
    logPrefix,
  });
  try {
    const lockedFreshness = inspectElectronFixtureBuildFreshness({ rootDir });
    if (lockedFreshness.ready) {
      return buildReadyResult(logPrefix, lockedFreshness);
    }

    console.log(
      `\n[${logPrefix}] > Electron fixture packaged renderer/assets build`,
    );
    if (lockedFreshness.missingFiles.length > 0) {
      console.log(
        `[${logPrefix}] missing build artifacts: ${lockedFreshness.missingFiles
          .map((filePath) => path.relative(rootDir, filePath))
          .join(", ")}`,
      );
    } else {
      console.log(
        `[${logPrefix}] rebuilding stale packaged fixture assets; segments=${lockedFreshness.staleSegments.join(",")} newestSource=${lockedFreshness.newestSource?.relativePath ?? "unknown"} oldestArtifact=${lockedFreshness.oldestArtifact?.relativePath ?? "unknown"}`,
      );
    }

    buildStaleElectronFixtureSegments({
      freshness: lockedFreshness,
      logPrefix,
      rootDir,
      runCommand,
    });

    process.env[BUILD_READY_ENV] = "1";
    return {
      status: lockedFreshness.missingFiles.length > 0 ? "built" : "rebuilt",
      missingFiles: lockedFreshness.missingFiles,
      freshness: lockedFreshness,
      rebuiltSegments: lockedFreshness.staleSegments,
    };
  } finally {
    releaseBuildLock();
  }
}
