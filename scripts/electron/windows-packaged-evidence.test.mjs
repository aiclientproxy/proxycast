import { createHash } from "node:crypto";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  buildWindowsPackagedEvidenceSummary,
  parseArgs,
  validateWindowsPackagedEvidence,
} from "./windows-packaged-evidence.mjs";

const roots = [];

afterEach(() => {
  while (roots.length > 0) {
    rmSync(roots.pop(), { recursive: true, force: true });
  }
});

function createFixture() {
  const root = mkdtempSync(
    path.join(tmpdir(), "lime-windows-packaged-evidence-"),
  );
  roots.push(root);
  const version = "1.2.3";
  const packageRoot = path.join(root, "lime");
  const appDirectory = path.join(packageRoot, `app-${version}`);
  const executable = path.join(appDirectory, "Lime.exe");
  const resourcesRoot = path.join(appDirectory, "resources");
  const resourceFiles = {
    "app-server": "app-server.exe",
    "code-mode-host": "code-mode-host.exe",
    "windows-sandbox-setup": "windows-sandbox-setup.exe",
    "windows-sandbox-runner": "windows-sandbox-runner.exe",
    "windows-native-host": "windows-native-host.exe",
  };
  mkdirSync(resourcesRoot, { recursive: true });
  writeFileSync(executable, "lime executable");
  for (const [id, fileName] of Object.entries(resourceFiles)) {
    const relativePath =
      id === "windows-native-host"
        ? "native/windows/windows-native-host.exe"
        : `app-server/win32-x64/${fileName}`;
    const filePath = path.join(resourcesRoot, relativePath);
    mkdirSync(path.dirname(filePath), { recursive: true });
    writeFileSync(filePath, `${id} binary\0`);
  }
  const resources = Object.entries(resourceFiles).map(([id]) => {
    const relativePath =
      id === "windows-native-host"
        ? "native/windows/windows-native-host.exe"
        : `app-server/win32-x64/${resourceFiles[id]}`;
    const filePath = path.join(resourcesRoot, relativePath);
    return {
      id,
      path: relativePath,
      sha256: createHash("sha256").update(readFileSync(filePath)).digest("hex"),
      required: true,
    };
  });
  writeFileSync(
    path.join(resourcesRoot, "desktop-resources.manifest.json"),
    JSON.stringify({
      schemaVersion: 1,
      applicationId: "com.limecloud.lime",
      platform: "win32",
      arch: "x64",
      platformKey: "win32-x64",
      resources,
    }),
  );

  const candidateRunId = "windows-run-1";
  const squirrelSummary = {
    scenarioId: "PLT-02-windows-squirrel-rc",
    result: "pass",
    candidateRunId,
    platform: { os: "win32", arch: "x64", appVersion: version },
    evidence: {
      installation: {
        executable,
        appDirectory,
        packageRoot,
        updateExecutable: path.join(packageRoot, "Update.exe"),
      },
    },
  };
  const codeModeSummary = {
    status: "pass",
    candidateRunId,
    packagedExecutable: true,
    packagedExecutablePath: executable,
    processes: {
      appServerPid: 101,
      codeModeHostParentPid: 101,
      appServerCommand: `"${path.join(resourcesRoot, "app-server/win32-x64/app-server.exe")}" --stdio`,
      codeModeHostCommand: path.join(
        resourcesRoot,
        "app-server/win32-x64/code-mode-host.exe",
      ),
    },
  };
  const nativeHostSummary = {
    result: "passed",
    evidenceLevel: "gate-b",
    platform: "win32",
    arch: "x64",
    candidateRunId,
    electronExecutable: executable,
    helper: {
      path: path.join(resourcesRoot, "native/windows/windows-native-host.exe"),
      resourcesRoot,
      readOnly: true,
      digestMatches: true,
    },
  };
  return {
    version,
    executable,
    resourcesRoot,
    squirrelSummary,
    codeModeSummary,
    nativeHostSummary,
  };
}

describe("Windows packaged evidence identity", () => {
  it("解析 camelCase 参数并要求三个 summary", () => {
    expect(
      parseArgs([
        "--version",
        "v1.2.3",
        "--squirrel-summary",
        "squirrel.json",
        "--code-mode-summary",
        "code.json",
        "--native-host-summary",
        "native.json",
        "--output",
        "result.json",
      ]),
    ).toMatchObject({
      version: "1.2.3",
      squirrelSummary: path.resolve("squirrel.json"),
      codeModeSummary: path.resolve("code.json"),
      nativeHostSummary: path.resolve("native.json"),
      output: path.resolve("result.json"),
    });
    expect(() => parseArgs(["--version", "1.2.3"])).toThrow(
      "--squirrel-summary is required",
    );
  });

  it("把同一 Squirrel 安装、sidecar 进程和 native helper 收口为 passed", () => {
    const fixture = createFixture();
    const summary = validateWindowsPackagedEvidence(fixture);

    expect(summary.result).toBe("passed");
    expect(summary.candidate).toEqual({
      version: fixture.version,
      runId: "windows-run-1",
      executable: fixture.executable,
      resourcesRoot: fixture.resourcesRoot,
    });
    expect(summary.checks.map((check) => check.name)).toEqual([
      "squirrel-summary",
      "installed-resource-manifest",
      "code-mode-summary",
      "native-host-summary",
    ]);
  });

  it("候选 run、安装路径或开发 sidecar 不一致时 fail closed", () => {
    const fixture = createFixture();
    fixture.codeModeSummary = {
      ...fixture.codeModeSummary,
      candidateRunId: "stale-run",
      packagedExecutablePath: path.join(
        path.dirname(fixture.executable),
        "old-Lime.exe",
      ),
      processes: {
        ...fixture.codeModeSummary.processes,
        appServerCommand: "/repo/lime-rs/target/debug/app-server --stdio",
      },
    };

    const summary = buildWindowsPackagedEvidenceSummary(fixture);
    expect(summary.result).toBe("failed");
    expect(summary.failures.map((failure) => failure.name)).toEqual([
      "code-mode-summary",
    ]);
    expect(
      summary.checks.find((check) => check.name === "code-mode-summary"),
    ).toMatchObject({
      status: "failed",
    });
  });

  it("缺少 Gate B 文件仍输出结构化失败结果", () => {
    const fixture = createFixture();
    const summary = buildWindowsPackagedEvidenceSummary({
      version: fixture.version,
      squirrelSummary: fixture.squirrelSummary,
      codeModeSummary: null,
      nativeHostSummary: null,
      fileExists: (filePath) =>
        filePath === fixture.executable ||
        filePath.endsWith("desktop-resources.manifest.json"),
      readFile: (filePath) => {
        if (filePath.endsWith("desktop-resources.manifest.json")) {
          return readFileSync(filePath);
        }
        return Buffer.from("binary");
      },
    });

    expect(summary.result).toBe("failed");
    expect(summary.failures.map((failure) => failure.name)).toEqual([
      "installed-resource-manifest",
      "code-mode-summary",
      "native-host-summary",
    ]);
  });
});
