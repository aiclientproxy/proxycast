import { describe, expect, it } from "vitest";

import {
  evaluateVitestLayerBudget,
  parseArgs,
  renderBudgetResultText,
} from "./check-vitest-layer-budget.mjs";

function buildReport(candidateFiles) {
  return {
    componentUnitMigrationCandidates: {
      total: candidateFiles.length,
      files: candidateFiles,
    },
  };
}

describe("check-vitest-layer-budget", () => {
  it("应解析 component 候选预算参数", () => {
    expect(parseArgs([])).toMatchObject({
      json: false,
      maxComponentCandidates: 37,
    });
    expect(
      parseArgs(["--max-component-candidates", "8", "--json"]),
    ).toMatchObject({
      json: true,
      maxComponentCandidates: 8,
    });
    expect(parseArgs(["--max-component-candidates=0"])).toMatchObject({
      maxComponentCandidates: 0,
    });
  });

  it("应在候选数不超过预算时通过", () => {
    expect(
      evaluateVitestLayerBudget(
        buildReport([{ file: "a.test.tsx", hints: [] }]),
        {
          maxComponentCandidates: 1,
        },
      ),
    ).toMatchObject({
      ok: true,
      componentCandidates: 1,
      overBudgetBy: 0,
    });
  });

  it("应在候选数超过预算时输出候选文件", () => {
    const result = evaluateVitestLayerBudget(
      buildReport([
        { file: "a.test.tsx", hints: ["large-component-file"] },
        { file: "b.test.tsx", hints: ["large-component-suite"] },
      ]),
      { maxComponentCandidates: 1 },
    );

    expect(result).toMatchObject({
      ok: false,
      componentCandidates: 2,
      overBudgetBy: 1,
    });
    expect(renderBudgetResultText(result)).toContain(
      "- a.test.tsx (large-component-file)",
    );
  });
});
