import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";

import {
  buildExpertSkillsLiveRuntimeMetadata,
  buildLiveRuntimeSummary,
  ensureExpertSkillsLiveWorkspaceSkill,
  normalizeLiveSummaryFromSource,
  parseArgs,
  runExpertSkillsLiveRunner,
} from "./expert-skills-live-runner.mjs";

const CORE_ASSERTIONS = {
  liveProviderUsed: true,
  liveProviderNotUsed: false,
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

async function withTempDir(callback) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "expert-skills-live-"));
  try {
    return await callback(dir);
  } finally {
    fs.rmSync(dir, { force: true, recursive: true });
  }
}

function writeSummary(dir, fileName, overrides = {}) {
  const {
    assertions,
    readModelExpertPanelSkillsRuntime,
    readModelExpertSkillsRuntime,
    ...summaryOverrides
  } = overrides;
  const summary = {
    ok: true,
    scenario: "expert-panel-skills-runtime",
    provider: "deepseek",
    model: "deepseek-v4-flash",
    assertions: {
      liveProviderUsed: true,
      liveProviderNotUsed: false,
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
    readModelExpertSkillsRuntime,
    ...summaryOverrides,
  };
  const filePath = path.join(dir, fileName);
  fs.writeFileSync(filePath, `${JSON.stringify(summary, null, 2)}\n`);
  return filePath;
}

function writeDeterministicSummary(dir) {
  return writeSummary(dir, "deterministic.json", {
    provider: "fixture-provider",
    model: "fixture-model",
    assertions: {
      liveProviderUsed: false,
      liveProviderNotUsed: true,
    },
  });
}

describe("expert skills live runner", () => {
  it("默认不允许 live Provider 验收执行", async () => {
    await expect(
      runExpertSkillsLiveRunner({
        allowLiveProvider: false,
        liveSummary: "",
        executeLiveRuntime: false,
      }),
    ).rejects.toThrow("默认禁止执行");
  });

  it("parseArgs 只有显式参数才打开 live runtime 执行", () => {
    const options = parseArgs([
      "--allow-live-provider",
      "--execute-live-runtime",
      "--live-workspace-root",
      ".lime/qc/live-test-workspace",
      "--provider-preference",
      "deepseek",
      "--model-preference",
      "deepseek-v4-flash",
      "--settled-grace-ms",
      "45000",
    ]);

    expect(options.allowLiveProvider).toBe(true);
    expect(options.executeLiveRuntime).toBe(true);
    expect(options.liveWorkspaceRoot).toContain(".lime/qc/live-test-workspace");
    expect(options.providerPreference).toBe("deepseek");
    expect(options.modelPreference).toBe("deepseek-v4-flash");
    expect(options.settledGraceMs).toBe(45000);
  });

  it("live runtime 执行前应准备 workspace-local skill 和 runtime enable metadata", () =>
    withTempDir((dir) => {
      const workspaceSkill = ensureExpertSkillsLiveWorkspaceSkill(dir);
      const metadata = buildExpertSkillsLiveRuntimeMetadata(workspaceSkill);

      expect(fs.existsSync(workspaceSkill.skillFilePath)).toBe(true);
      expect(fs.existsSync(workspaceSkill.registrationFilePath)).toBe(true);
      expect(metadata.expert.skillRefs).toContain("skill:capability-report");
      expect(metadata.harness.expert.skill_refs).toContain(
        "skill:capability-report",
      );
      expect(metadata.harness.workspace_skill_runtime_enable).toMatchObject({
        source: "manual_session_enable",
        approval: "manual",
        workspace_root: dir,
        bindings: [
          {
            directory: "capability-report",
            skill: "project:capability-report",
            registered_skill_directory: workspaceSkill.skillDirectory,
            source_draft_id: "capdraft-live-capability-report",
            source_verification_report_id: "capver-live-capability-report",
          },
        ],
      });
    }));

  it("live runtime summary 应产出专家 skill search before invocation 断言", () => {
    const summary = buildLiveRuntimeSummary({
      provider: "deepseek",
      model: "deepseek-v4-flash",
      sessionId: "session-live",
      turnId: "turn-live",
      threadRead: {
        result: {
          session: {
            status: "completed",
          },
          turns: [
            {
              status: "completed",
              events: [
                {
                  type: "tool.started",
                  payload: {
                    tool_name: "skill_search",
                  },
                },
                {
                  type: "tool.result",
                  payload: {
                    tool_name: "skill_search",
                    output: "selected project:capability-report",
                  },
                },
                {
                  type: "tool.started",
                  payload: {
                    tool_name: "Skill",
                    arguments: {
                      skill: "capability-report",
                    },
                  },
                },
                {
                  type: "tool.result",
                  payload: {
                    tool_name: "Skill",
                    output: "skill_invocation capability-report completed",
                  },
                },
              ],
            },
          ],
        },
      },
    });

    expect(summary.assertions.expertSkillSearchBeforeSkillInvocation).toBe(
      true,
    );
    expect(summary.readModelExpertSkillsRuntime).toMatchObject({
      skillSearchBeforeSkillInvocation: true,
      skillSearchCount: expect.any(Number),
      skillInvocationCount: expect.any(Number),
    });
  });

  it("live runtime summary 不应把 Provider 失败误判成完成或 Skill 调用", () => {
    const summary = buildLiveRuntimeSummary({
      provider: "deepseek",
      model: "deepseek-v4-flash",
      sessionId: "session-live",
      turnId: "turn-live",
      threadRead: {
        status: "failed",
        diagnostics: {
          latest_turn_status: "failed",
          latest_turn_error_message:
            "execution backend error: Request failed with status 402 Payment Required",
        },
        turns: [{ status: "failed" }],
        tool_calls: [],
      },
    });

    expect(summary.ok).toBe(false);
    expect(summary.assertions.readModelExpertSkillsRuntimeCompleted).toBe(
      false,
    );
    expect(summary.assertions.readModelExpertSkillSearchObserved).toBe(false);
    expect(summary.assertions.readModelExpertSkillInvocationObserved).toBe(
      false,
    );
    expect(summary.runtime.providerFailureMessage).toContain("402");
  });

  it("拒绝把 fixture summary 归一化成 live summary", () =>
    withTempDir((dir) => {
      const fixture = writeSummary(dir, "fixture.json", {
        provider: "fixture-provider",
        model: "fixture-model",
      });

      expect(() =>
        normalizeLiveSummaryFromSource(
          JSON.parse(fs.readFileSync(fixture, "utf8")),
          fixture,
        ),
      ).toThrow("fixture provider/model");
    }));

  it("读取已有 live summary 后写出 gate 可消费的 summary", async () =>
    withTempDir(async (dir) => {
      const deterministicSummary = writeDeterministicSummary(dir);
      const liveSummary = writeSummary(dir, "live.json");
      const output = path.join(dir, "normalized-live.json");

      const result = await runExpertSkillsLiveRunner({
        allowLiveProvider: true,
        liveSummary,
        executeLiveRuntime: false,
        deterministicSummary,
        output,
      });

      expect(result.output).toBe(output);
      expect(result.summary.liveProviderUsed).toBe(true);
      expect(result.gateReport.status).toBe("pass");
      expect(fs.existsSync(output)).toBe(true);
    }));
});
