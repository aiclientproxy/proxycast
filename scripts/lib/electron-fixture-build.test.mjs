import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  electronFixtureBuildReadyEnv,
  electronFixtureBuildRequiredFiles,
  ensureElectronFixtureBuild,
  inspectElectronFixtureBuildFreshness,
} from "./electron-fixture-build.mjs";

const tempRoots = [];

function createTempRoot() {
  const rootDir = fs.mkdtempSync(
    path.join(os.tmpdir(), "electron-fixture-build-"),
  );
  tempRoots.push(rootDir);
  return rootDir;
}

function writeFile(filePath, content, mtime) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content);
  fs.utimesSync(filePath, mtime, mtime);
}

function writeRequiredFiles(rootDir, mtime) {
  for (const filePath of electronFixtureBuildRequiredFiles({ rootDir })) {
    writeFile(filePath, "fixture artifact", mtime);
  }
}

function createRecordingRunner() {
  const calls = [];
  return {
    calls,
    runCommand(step) {
      calls.push({
        command: step.command,
        args: step.args,
      });
      return { status: 0 };
    },
  };
}

afterEach(() => {
  while (tempRoots.length > 0) {
    const rootDir = tempRoots.pop();
    fs.rmSync(rootDir, { force: true, recursive: true });
  }
});

describe("electron fixture build freshness", () => {
  it("serializes stale packaged fixture rebuilds behind a repository lock", () => {
    const content = fs.readFileSync(
      "scripts/lib/electron-fixture-build.mjs",
      "utf8",
    );

    expect(content).toContain("electron-fixture-build.lock");
    expect(content).toContain("acquireElectronFixtureBuildLock");
    expect(content).toContain("waiting for Electron fixture packaged");
    expect(content).toContain("const lockedFreshness");
    expect(content).toContain("inspectElectronFixtureBuildFreshness");
    expect(content).toContain("buildStaleElectronFixtureSegments");
  });

  it("reuses packaged artifacts when build inputs are older", () => {
    const rootDir = createTempRoot();
    writeFile(
      path.join(rootDir, "src", "App.tsx"),
      "export const app = true;",
      new Date("2026-01-01T00:00:00.000Z"),
    );
    writeRequiredFiles(rootDir, new Date("2026-01-01T00:10:00.000Z"));

    const freshness = inspectElectronFixtureBuildFreshness({ rootDir });

    expect(freshness).toMatchObject({
      ready: true,
      reason: "fresh-artifacts",
      missingFiles: [],
    });
    expect(freshness.newestSource.relativePath).toBe(
      path.join("src", "App.tsx"),
    );
  });

  it("marks the current process build-ready when fresh artifacts are reused", () => {
    const rootDir = createTempRoot();
    const envKey = electronFixtureBuildReadyEnv();
    const previousEnv = process.env[envKey];
    const previousLog = console.log;
    delete process.env[envKey];
    writeFile(
      path.join(rootDir, "src", "App.tsx"),
      "export const app = true;",
      new Date("2026-01-01T00:00:00.000Z"),
    );
    writeRequiredFiles(rootDir, new Date("2026-01-01T00:10:00.000Z"));

    try {
      console.log = () => {};
      const result = ensureElectronFixtureBuild({
        rootDir,
        logPrefix: "electron-fixture-build-test",
      });

      expect(result).toMatchObject({
        status: "reused",
        reason: "fresh-artifacts",
      });
      expect(process.env[envKey]).toBe("1");
    } finally {
      if (previousEnv === undefined) {
        delete process.env[envKey];
      } else {
        process.env[envKey] = previousEnv;
      }
      console.log = previousLog;
    }
  });

  it("marks packaged artifacts stale when a build input is newer", () => {
    const rootDir = createTempRoot();
    writeRequiredFiles(rootDir, new Date("2026-01-01T00:00:00.000Z"));
    writeFile(
      path.join(rootDir, "electron", "main.ts"),
      "export const main = true;",
      new Date("2026-01-01T00:10:00.000Z"),
    );

    const freshness = inspectElectronFixtureBuildFreshness({ rootDir });

    expect(freshness).toMatchObject({
      ready: false,
      reason: "stale-source",
      missingFiles: [],
      staleSegments: ["host"],
    });
    expect(freshness.segments.host.ready).toBe(false);
    expect(freshness.segments.renderer.ready).toBe(true);
    expect(freshness.segments.appServer.ready).toBe(true);
    expect(freshness.newestSource.relativePath).toBe(
      path.join("electron", "main.ts"),
    );
  });

  it("does not mark renderer stale when only Rust app-server sources are newer", () => {
    const rootDir = createTempRoot();
    writeRequiredFiles(rootDir, new Date("2026-01-01T00:10:00.000Z"));
    writeFile(
      path.join(rootDir, "lime-rs", "crates", "tool-runtime", "src", "lib.rs"),
      "pub fn changed() {}",
      new Date("2026-01-01T00:20:00.000Z"),
    );

    const freshness = inspectElectronFixtureBuildFreshness({ rootDir });

    expect(freshness).toMatchObject({
      ready: false,
      reason: "stale-source",
      missingFiles: [],
      staleSegments: ["appServer"],
    });
    expect(freshness.segments.renderer.ready).toBe(true);
    expect(freshness.segments.host.ready).toBe(true);
    expect(freshness.segments.appServer.ready).toBe(false);
    expect(freshness.segments.appServer.newestSource.relativePath).toBe(
      path.join("lime-rs", "crates", "tool-runtime", "src", "lib.rs"),
    );
  });

  it("only rebuilds the app-server segment for Rust-only fixture staleness", () => {
    const rootDir = createTempRoot();
    const envKey = electronFixtureBuildReadyEnv();
    const previousEnv = process.env[envKey];
    const previousLog = console.log;
    const runner = createRecordingRunner();
    delete process.env[envKey];
    writeRequiredFiles(rootDir, new Date("2026-01-01T00:10:00.000Z"));
    writeFile(
      path.join(rootDir, "lime-rs", "crates", "app-server", "src", "lib.rs"),
      "pub fn changed() {}",
      new Date("2026-01-01T00:20:00.000Z"),
    );

    try {
      console.log = () => {};
      const result = ensureElectronFixtureBuild({
        rootDir,
        logPrefix: "electron-fixture-build-test",
        runCommand: runner.runCommand,
      });
      const commandLines = runner.calls.map((call) =>
        [path.basename(call.command), ...call.args].join(" "),
      );

      expect(result).toMatchObject({
        status: "rebuilt",
        rebuiltSegments: ["appServer"],
      });
      expect(commandLines).toContain("npm run verify:app-version");
      expect(commandLines).toContain(
        "npm run electron:build:app-server-assets",
      );
      expect(commandLines).not.toContain(
        "npm run build:renderer:electron:smoke",
      );
      expect(commandLines).not.toContain("npm run typecheck:electron");
      expect(process.env[envKey]).toBe("1");
    } finally {
      if (previousEnv === undefined) {
        delete process.env[envKey];
      } else {
        process.env[envKey] = previousEnv;
      }
      console.log = previousLog;
    }
  });

  it("treats app-server asset helper changes as build inputs", () => {
    const rootDir = createTempRoot();
    writeRequiredFiles(rootDir, new Date("2026-01-01T00:00:00.000Z"));
    writeFile(
      path.join(rootDir, "scripts", "lib", "electron-app-server-assets.mjs"),
      "export const changed = true;",
      new Date("2026-01-01T00:10:00.000Z"),
    );

    const freshness = inspectElectronFixtureBuildFreshness({ rootDir });

    expect(freshness).toMatchObject({
      ready: false,
      reason: "stale-source",
      missingFiles: [],
      staleSegments: ["appServer"],
    });
    expect(freshness.newestSource.relativePath).toBe(
      path.join("scripts", "lib", "electron-app-server-assets.mjs"),
    );
  });

  it("requires preload and packaged app-server artifacts", () => {
    const rootDir = createTempRoot();
    writeFile(
      path.join(rootDir, "dist", "index.html"),
      "<html></html>",
      new Date("2026-01-01T00:00:00.000Z"),
    );
    writeFile(
      path.join(rootDir, "dist-electron", "main", "main.js"),
      "console.log('main');",
      new Date("2026-01-01T00:00:00.000Z"),
    );
    writeFile(
      path.join(rootDir, "dist-electron", "app-server.release.json"),
      "{}",
      new Date("2026-01-01T00:00:00.000Z"),
    );

    const freshness = inspectElectronFixtureBuildFreshness({ rootDir });
    const missingRelativePaths = freshness.missingFiles.map((filePath) =>
      path.relative(rootDir, filePath),
    );

    expect(freshness.ready).toBe(false);
    expect(freshness.reason).toBe("missing-artifacts");
    expect(missingRelativePaths).toContain(
      path.join("dist-electron", "preload", "preload.cjs"),
    );
    expect(
      missingRelativePaths.some((filePath) =>
        filePath.startsWith(path.join("dist-electron", "app-server")),
      ),
    ).toBe(true);
  });
});
