import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { buildExpertSkillsLiveGateReport } from "./expert-skills-live-gate.mjs";

const CORE_ASSERTIONS = {
  electronPreloadBridge: true,
  appServerJsonRpcUsed: true,
  liveProviderNotUsed: true,
  expertSkillsRuntimePromptReachedBackend: true,
  expertSkillsRuntimeMetadataReachedBackend: true,
  expertDeclaredSkillRefsObserved: true,
  expertSelectedSkillObserved: true,
  expertInvokedSkillObserved: true,
  readModelExpertSkillsRuntimeCompleted: true,
  readModelExpertSkillSearchObserved: true,
  readModelExpertSkillInvocationObserved: true,
  readModelExpertSkillBodyReadObserved: true,
  readModelExpertSkillGateObserved: true,
  expertSkillSearchBeforeSkillInvocation: true,
};

const PANEL_ASSERTIONS = {
  expertPanelSecondTurnPromptReachedBackend: true,
  expertPanelSkillRefsOverrideReachedBackend: true,
  expertPanelReadModelCompleted: true,
  expertPanelReadModelSkillBodyReadObserved: true,
  expertPanelReadModelSkillGateObserved: true,
  expertPanelReadModelSkillSearchObserved: true,
  expertPanelReadModelSkillInvocationObserved: true,
  expertPanelSkillSearchBeforeSkillInvocation: true,
};

function writeSummary(root, name, overrides = {}) {
  const {
    assertions,
    readModelExpertPanelSkillsRuntime,
    ...summaryOverrides
  } = overrides;
  const summary = {
    ok: true,
    scenario: "expert-panel-skills-runtime",
    provider: "fixture-provider",
    model: "fixture-model",
    assertions: {
      liveProviderUsed: false,
      liveProviderNotUsed: true,
      ...PANEL_ASSERTIONS,
      ...(assertions ?? {}),
    },
    readModelExpertPanelSkillsRuntime: {
      hasThreadRead: true,
      skillSearchCount: 1,
      skillInvocationCount: 1,
      skillBodyReadObserved: true,
      skillGateObserved: true,
      expertDeclaredObserved: true,
      expertSelectedObserved: true,
      expertInvokedObserved: true,
      skillSearchBeforeSkillInvocation: true,
      ...(readModelExpertPanelSkillsRuntime ?? {}),
    },
    ...summaryOverrides,
  };
  const filePath = path.join(root, name);
  fs.writeFileSync(filePath, `${JSON.stringify(summary, null, 2)}\n`);
  return filePath;
}

function withTempDir(callback) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "expert-skills-gate-"));
  try {
    return callback(dir);
  } finally {
    fs.rmSync(dir, { force: true, recursive: true });
  }
}

describe("expert skills live gate", () => {
  it("确定性专家 Skills 证据完整但缺 live 时返回 pending", () =>
    withTempDir((dir) => {
      const deterministicSummary = writeSummary(dir, "deterministic.json");

      const report = buildExpertSkillsLiveGateReport({
        deterministicSummary,
        allowMissingLive: true,
      });

      expect(report.status).toBe("pending_live_provider");
      expect(report.completion).toEqual({
        deterministicExpertSkillsReady: true,
        liveProviderExpertSkillsReady: false,
        overallGoalReady: false,
      });
      expect(report.nextRequired).toContain("live Provider");
    }));

  it("没有显式 allow-missing-live 时缺 live 证据应失败", () =>
    withTempDir((dir) => {
      const deterministicSummary = writeSummary(dir, "deterministic.json");

      const report = buildExpertSkillsLiveGateReport({
        deterministicSummary,
      });

      expect(report.status).toBe("fail");
      expect(report.live.status).toBe("missing");
    }));

  it("live summary 仍使用 fixture provider 时不能通过整体门禁", () =>
    withTempDir((dir) => {
      const deterministicSummary = writeSummary(dir, "deterministic.json");
      const liveSummary = writeSummary(dir, "live-fixture.json", {
        assertions: {
          liveProviderUsed: true,
          liveProviderNotUsed: false,
        },
      });

      const report = buildExpertSkillsLiveGateReport({
        deterministicSummary,
        liveSummary,
      });

      expect(report.status).toBe("fail");
      expect(report.live.issues).toContain(
        "summary still uses fixture provider/model",
      );
    }));

  it("live summary 有真实 provider 声明和专家技能证据时通过", () =>
    withTempDir((dir) => {
      const deterministicSummary = writeSummary(dir, "deterministic.json");
      const liveSummary = writeSummary(dir, "live.json", {
        provider: "deepseek",
        model: "deepseek-v4-flash",
        assertions: {
          liveProviderUsed: true,
          liveProviderNotUsed: false,
        },
      });

      const report = buildExpertSkillsLiveGateReport({
        deterministicSummary,
        liveSummary,
      });

      expect(report.status).toBe("pass");
      expect(report.completion.overallGoalReady).toBe(true);
    }));
});
