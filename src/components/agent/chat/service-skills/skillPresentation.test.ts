import { describe, expect, it } from "vitest";
import {
  buildServiceSkillCapabilityDescription,
  getServiceSkillActionLabel,
  getServiceSkillOutputDestination,
  getServiceSkillTypeLabel,
  summarizeServiceSkillRequiredInputs,
} from "./skillPresentation";
import type { ServiceSkillItem } from "./types";

function createServiceSkill(
  overrides: Partial<ServiceSkillItem> = {},
): ServiceSkillItem {
  return {
    id: "deep-research",
    title: "深度研究",
    summary: "综合多来源信息并给出归纳后的结论。",
    category: "调研",
    outputHint: "研究摘要",
    source: "cloud_catalog",
    runnerType: "instant",
    defaultExecutorBinding: "agent_turn",
    executionLocation: "client_default",
    slotSchema: [],
    version: "2026-04-09",
    ...overrides,
  };
}

describe("skillPresentation", () => {
  it("本地即时技能的动作文案应指向开始这一步，而不是旧工作区心智", () => {
    expect(getServiceSkillActionLabel(createServiceSkill())).toBe("开始这一步");
  });

  it("service skill 的必填输入摘要应优先使用槽位标签，并在无必填项时给出稳定兜底", () => {
    expect(summarizeServiceSkillRequiredInputs(createServiceSkill())).toBe(
      "当前无必填信息",
    );

    expect(
      summarizeServiceSkillRequiredInputs(
        createServiceSkill({
          slotSchema: [
            {
              key: "query",
              label: "检索词",
              type: "text",
              required: true,
              placeholder: "例如 AI Agent",
            },
            {
              key: "platform",
              label: "目标平台",
              type: "text",
              required: true,
              placeholder: "例如 GitHub",
            },
          ],
        }),
      ),
    ).toBe("检索词、目标平台");
  });

  it("service skill 的能力描述应收口成 promise + 需要 + 交付 的统一合同", () => {
    expect(
      buildServiceSkillCapabilityDescription(
        createServiceSkill({
          entryHint: "补一个主题，我先整理一版研究结论。",
          slotSchema: [
            {
              key: "query",
              label: "检索词",
              type: "text",
              required: true,
              placeholder: "例如 AI Agent",
            },
          ],
        }),
      ),
    ).toBe(
      "补一个主题，我先整理一版研究结论。 · 需要：检索词 · 交付：研究摘要",
    );

    expect(
      buildServiceSkillCapabilityDescription(createServiceSkill(), {
        includeSummary: false,
      }),
    ).toBe("需要：当前无必填信息 · 交付：研究摘要");
  });

  it("带必填槽位的即时技能应提示先补齐这一步", () => {
    expect(
      getServiceSkillActionLabel(
        createServiceSkill({
          slotSchema: [
            {
              key: "query",
              label: "检索词",
              type: "text",
              required: true,
              placeholder: "例如 AI Agent",
            },
          ],
        }),
      ),
    ).toBe("补齐这一步");
  });

  it("允许调用方注入本地化 presentation copy", () => {
    const copy = {
      actionLabels: { instant: "Start this step" },
      fallbackRequiredInputs: "No required inputs",
      formatFactItems: (visibleItems: string[], totalCount: number) =>
        visibleItems.length < totalCount
          ? `${visibleItems.join(", ")} + ${totalCount - visibleItems.length} more`
          : visibleItems.join(", "),
      outputDefault: "The result returns to Generate.",
      outputPrefix: "Delivers: ",
      requiredPrefix: "Needs: ",
      typeLabels: { service: "Creation workflow" },
    };
    const skill = createServiceSkill({
      slotSchema: [
        {
          key: "topic",
          label: "Topic",
          placeholder: "Topic",
          type: "text",
          required: true,
        },
        {
          key: "audience",
          label: "Audience",
          placeholder: "Audience",
          type: "text",
          required: true,
        },
        {
          key: "tone",
          label: "Tone",
          placeholder: "Tone",
          type: "text",
          required: true,
        },
      ],
    });

    expect(
      summarizeServiceSkillRequiredInputs(skill, {
        copy,
        limit: 2,
      }),
    ).toBe("Topic, Audience + 1 more");
    expect(
      buildServiceSkillCapabilityDescription(
        createServiceSkill({ outputHint: "Research brief" }),
        {
          copy,
          includeSummary: false,
        },
      ),
    ).toBe("Needs: No required inputs · Delivers: Research brief");
    expect(getServiceSkillActionLabel(createServiceSkill(), { copy })).toBe(
      "Start this step",
    );
    expect(
      getServiceSkillOutputDestination(createServiceSkill(), { copy }),
    ).toBe("The result returns to Generate.");
    expect(getServiceSkillTypeLabel(createServiceSkill(), { copy })).toBe(
      "Creation workflow",
    );
  });
});
