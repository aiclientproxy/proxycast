import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { runCli } from "./i18n-patch-retirement-gate.mjs";

const tempDirs: string[] = [];

function createTempDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "lime-i18n-patch-gate-"));
  tempDirs.push(dir);
  return dir;
}

function writeJson(root: string, relativePath: string, value: unknown): string {
  const filePath = path.join(root, relativePath);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
  return filePath;
}

afterEach(() => {
  vi.restoreAllMocks();
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true });
  }
});

describe("i18n patch retirement gate", () => {
  it("应在 Patch no-hit 且 legacy surface 零违规时通过", () => {
    const root = createTempDir();
    const patchReportPath = writeJson(root, "patch-report.json", {
      generatedAt: "2026-05-23T00:00:00.000Z",
      recommendations: [
        "当前样本未命中 legacy Patch，可进入 current 主路径依赖审计。",
      ],
      recentRuns: [],
      retirementCandidate: true,
      schemaVersion: "lime.i18n.patchMetricsReport.v1",
      sourcePath: ".lime/i18n/patch-metrics.json",
      status: "no-hit",
      summary: {
        averagePatchTimeMs: 0,
        languageChanges: 0,
        lastRunAt: null,
        lastRunLanguage: null,
        lastRunRootKind: null,
        recentRunCount: 0,
        slowestPatchTimeMs: 0,
        totalMatchedSegments: 0,
        totalReplacedNodes: 0,
        totalRuns: 3,
      },
      thresholdIssues: [],
      thresholds: {},
    });
    const legacyReportPath = writeJson(root, "legacy-report.json", {
      repoRoot: "/Users/coso/Documents/dev/ai/aiclientproxy/lime",
      summary: {
        classificationDriftCandidates: [],
        runtimeSourceCount: 1,
        rustRuntimeSourceCount: 0,
        rustTestSourceCount: 0,
        testSourceCount: 0,
        violations: [],
        zeroReferenceCandidates: [],
      },
    });

    const writeSpy = vi
      .spyOn(process.stdout, "write")
      .mockImplementation(() => true);
    const exitCode = runCli([
      "--check",
      "--format",
      "json",
      "--patch-report",
      patchReportPath,
      "--legacy-report",
      legacyReportPath,
    ]);

    expect(exitCode).toBe(0);
    expect(writeSpy).toHaveBeenCalledTimes(1);
    const report = JSON.parse(String(writeSpy.mock.calls[0]?.[0] ?? "")) as {
      advisoryIssues: string[];
      gateIssues: string[];
      legacy: { violationCount: number };
      patch: { retirementCandidate: boolean; status: string };
      retirementReady: boolean;
      schemaVersion: string;
    };

    expect(report.schemaVersion).toBe("lime.i18n.patchRetirementGate.v1");
    expect(report.retirementReady).toBe(true);
    expect(report.advisoryIssues).toEqual([]);
    expect(report.gateIssues).toEqual([]);
    expect(report.patch.status).toBe("no-hit");
    expect(report.patch.retirementCandidate).toBe(true);
    expect(report.legacy.violationCount).toBe(0);
  });

  it("应在 Patch 非 no-hit 或 legacy 有违规时失败", () => {
    const root = createTempDir();
    const patchReportPath = writeJson(root, "patch-report.json", {
      generatedAt: "2026-05-23T00:00:00.000Z",
      recommendations: [
        "仍有 legacy Patch 命中，优先把页面文案迁入 key-based namespace。",
      ],
      recentRuns: [],
      retirementCandidate: false,
      schemaVersion: "lime.i18n.patchMetricsReport.v1",
      sourcePath: ".lime/i18n/patch-metrics.json",
      status: "active-patch",
      summary: {
        averagePatchTimeMs: 1,
        languageChanges: 1,
        lastRunAt: 1716422400000,
        lastRunLanguage: "zh",
        lastRunRootKind: "document",
        recentRunCount: 1,
        slowestPatchTimeMs: 1,
        totalMatchedSegments: 2,
        totalReplacedNodes: 1,
        totalRuns: 1,
      },
      thresholdIssues: [
        {
          actual: 2,
          field: "totalMatchedSegments",
          message: "命中文本段数 2 超过门限 0",
          threshold: 0,
        },
      ],
      thresholds: { maxMatchedSegments: 0 },
    });
    const legacyReportPath = writeJson(root, "legacy-report.json", {
      summary: {
        classificationDriftCandidates: [
          "retired-active-launch-surface -> dead / 受控",
        ],
        runtimeSourceCount: 1,
        rustRuntimeSourceCount: 0,
        rustTestSourceCount: 0,
        testSourceCount: 0,
        violations: [
          "retired-active-launch-surface -> src/lib/retired-runtime/launchBridge.ts",
        ],
        zeroReferenceCandidates: [
          "retired-active-launch-surface (retired active launch surface)",
        ],
      },
    });

    const writeSpy = vi
      .spyOn(process.stdout, "write")
      .mockImplementation(() => true);
    const exitCode = runCli([
      "--check",
      "--format",
      "json",
      "--patch-report",
      patchReportPath,
      "--legacy-report",
      legacyReportPath,
    ]);

    expect(exitCode).toBe(1);
    expect(writeSpy).toHaveBeenCalledTimes(1);
    const report = JSON.parse(String(writeSpy.mock.calls[0]?.[0] ?? "")) as {
      advisoryIssues: string[];
      gateIssues: string[];
      legacy: {
        classificationDriftCandidateCount: number;
        violationCount: number;
      };
      patch: {
        retirementCandidate: boolean;
        status: string;
        thresholdIssueCount: number;
      };
      retirementReady: boolean;
    };

    expect(report.retirementReady).toBe(false);
    expect(report.patch.status).toBe("active-patch");
    expect(report.patch.thresholdIssueCount).toBe(1);
    expect(report.legacy.violationCount).toBe(1);
    expect(report.legacy.classificationDriftCandidateCount).toBe(1);
    expect(report.advisoryIssues).toEqual(
      expect.arrayContaining([expect.stringContaining("分类漂移候选")]),
    );
    expect(report.gateIssues).toEqual(
      expect.arrayContaining([
        expect.stringContaining("Patch status 必须为 no-hit"),
        expect.stringContaining("Legacy surface report 存在 1 个违规引用"),
      ]),
    );
  });
});
