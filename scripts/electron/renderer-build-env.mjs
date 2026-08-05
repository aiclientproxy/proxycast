import { mkdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

const RENDERER_BUILD_NODE_OPTIONS = "--max-old-space-size=8192";
const DEFAULT_HEARTBEAT_INTERVAL_MS = 60_000;
const RENDERER_BUILD_LOCK_PATH = path.join(
  ".lime",
  "electron-renderer-build.lock",
);
const RENDERER_BUILD_LOCK_WAIT_MS = 1_000;
const RENDERER_BUILD_LOCK_TIMEOUT_MS = 20 * 60_000;
const RENDERER_BUILD_LOCK_STALE_MS = 30 * 60_000;

export function rendererBuildEnv(env = process.env) {
  const nodeOptions = env.NODE_OPTIONS ?? "";
  const hasOldSpaceSize = /(?:^|\s)--max-old-space-size(?:=|\s|$)/.test(
    nodeOptions,
  );
  return {
    ...env,
    NODE_OPTIONS: hasOldSpaceSize
      ? nodeOptions
      : [nodeOptions, RENDERER_BUILD_NODE_OPTIONS].filter(Boolean).join(" "),
  };
}

export function acquireRendererBuildLock({
  rootDir = process.cwd(),
  logPrefix = "electron-renderer-build",
} = {}) {
  const lockPath = path.join(rootDir, RENDERER_BUILD_LOCK_PATH);
  const startedAt = Date.now();
  let loggedWait = false;

  while (Date.now() - startedAt < RENDERER_BUILD_LOCK_TIMEOUT_MS) {
    mkdirSync(path.dirname(lockPath), { recursive: true });
    try {
      mkdirSync(lockPath);
      writeFileSync(
        path.join(lockPath, "owner.json"),
        `${JSON.stringify(
          { pid: process.pid, createdAt: new Date().toISOString() },
          null,
          2,
        )}\n`,
      );
      return () => rmSync(lockPath, { force: true, recursive: true });
    } catch (error) {
      if (error?.code !== "EEXIST") {
        throw error;
      }
    }

    let lockStats = null;
    try {
      lockStats = statSync(lockPath);
    } catch {
      continue;
    }
    if (Date.now() - lockStats.mtimeMs > RENDERER_BUILD_LOCK_STALE_MS) {
      console.warn(
        `[${logPrefix}] removing stale renderer build lock after ${Math.round(
          (Date.now() - lockStats.mtimeMs) / 1000,
        )}s: ${path.relative(rootDir, lockPath)}`,
      );
      rmSync(lockPath, { force: true, recursive: true });
      continue;
    }

    if (!loggedWait) {
      console.log(`[${logPrefix}] waiting for renderer build lock`);
      loggedWait = true;
    }
    const buffer = new SharedArrayBuffer(4);
    Atomics.wait(new Int32Array(buffer), 0, 0, RENDERER_BUILD_LOCK_WAIT_MS);
  }

  throw new Error(
    `[${logPrefix}] timed out waiting for renderer build lock: ${path.relative(
      rootDir,
      lockPath,
    )}`,
  );
}

export function startRendererBuildHeartbeat({
  intervalMs = DEFAULT_HEARTBEAT_INTERVAL_MS,
  now = () => new Date(),
  write = (message) => process.stdout.write(message),
} = {}) {
  const startedAt = Date.now();
  const timer = setInterval(() => {
    const elapsedSeconds = Math.round((Date.now() - startedAt) / 1000);
    write(
      `[electron-renderer-build] still running after ${elapsedSeconds}s at ${now().toISOString()}\n`,
    );
  }, intervalMs);
  timer.unref?.();
  return () => clearInterval(timer);
}
