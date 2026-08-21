import { describe, expect, it } from "vitest";

import {
  buildGeneralAgentSystemPrompt,
  isGeneralResearchTheme,
  resolveAgentChatMode,
} from "./generalAgentPrompt";

describe("generalAgentPrompt", () => {
  it("应识别通用对话主题", () => {
    expect(isGeneralResearchTheme("general")).toBe(true);
    expect(isGeneralResearchTheme(" custom-theme ")).toBe(false);
    expect(isGeneralResearchTheme(undefined)).toBe(false);
  });

  it("工作台模式应优先返回 workbench", () => {
    expect(resolveAgentChatMode("general", true)).toBe("workbench");
    expect(resolveAgentChatMode("general", false)).toBe("general");
    expect(resolveAgentChatMode("custom-theme", false)).toBe("agent");
  });

  it("通用主题 Prompt 应避免编程和落盘默认倾向", () => {
    const prompt = buildGeneralAgentSystemPrompt("general", {
      now: new Date("2026-03-12T12:00:00+08:00"),
      toolPreferences: {
        task: true,
        subagent: true,
      },
    });

    expect(prompt).toContain("不要把自己限制为编程助手");
    expect(prompt).toContain("不主动落盘");
    expect(prompt).toContain("需求澄清");
    expect(prompt).toContain("当前能力开关");
    expect(prompt).toContain("执行车道");
    expect(prompt).toContain("计划执行：已开启");
    expect(prompt).toContain("Subagents：已开启");
    expect(prompt).toContain("统一使用 WebSearch");
    expect(prompt).toContain("不要混用 search/search_query/ToolSearch");
    expect(prompt).toContain("不要通过前端关键词、固定模板或别名改写强制搜索");
    expect(prompt).not.toContain("只有在以下场景才主动联网核实");
    expect(prompt).not.toMatch(/3-4\s+组 WebSearch/);
    expect(prompt).toContain("1 个当前最关键的问题");
    expect(prompt).toContain("合理假设补齐");
    expect(prompt).toContain("每轮最多只保留 1 个最关键问题");
    expect(prompt).not.toContain("多代理");
    expect(prompt).toContain("没有显式路径或保存要求时");
    expect(prompt).toContain("写计划 / 写方案 / 写报告");
    expect(prompt).toContain("输出契约");
    expect(prompt).toContain("标准 Markdown");
    expect(prompt).toContain("工具调用过程");
    expect(prompt).toContain("不得写进最终正文");
    expect(prompt).toContain("可核验来源");
    expect(prompt).toContain("Markdown 链接");
    expect(prompt).toContain("粘贴原始工具输出");
  });

  it("通用主题 Prompt 应回落到统一的主题说明", () => {
    const prompt = buildGeneralAgentSystemPrompt("general");

    expect(prompt).toContain("当前主题：通用对话");
    expect(prompt).toContain("优先处理需求澄清");
    expect(prompt).toContain("不要一上来就走重链路");
    expect(prompt).toContain("必要扩搜");
  });

  it("通用主题 Prompt 仍应强调执行升级与任务分工边界", () => {
    const prompt = buildGeneralAgentSystemPrompt("general");

    expect(prompt).toContain("计划执行");
    expect(prompt).toContain("Subagents");
    expect(prompt).toContain("如果进入计划执行或 Subagents");
    expect(prompt).toContain("Subagents 模式下");
    expect(prompt).not.toContain("Team 协作");
  });

  it("工作区编排场景应注入 harness 上下文，并兼容旧 general workbench alias", () => {
    const prompt = buildGeneralAgentSystemPrompt("general", {
      harness: {
        sessionMode: "theme_workbench",
        gateKey: "research_mode",
        runTitle: "行业分析",
        contentId: "content-1",
      },
    });

    expect(prompt).toContain("工作区编排场景");
    expect(prompt).toContain("当前 gate：research_mode");
    expect(prompt).toContain("当前任务标题：行业分析");
    expect(prompt).toContain("当前内容 ID：content-1");
    expect(prompt).toContain("<proposed_plan>");
  });

  it("启用 Browser Assist 时应强制网页任务走 Lime 浏览器会话", () => {
    const prompt = buildGeneralAgentSystemPrompt("general", {
      harness: {
        browserAssistEnabled: true,
        browserAssistProfileKey: "general_browser_assist",
      },
    });

    expect(prompt).toContain("当前通用对话已启用 Browser Assist");
    expect(prompt).toContain("general_browser_assist");
    expect(prompt).toContain("Playwright code");
    expect(prompt).toContain("浏览器工作台");
    expect(prompt).toContain("browser session");
    expect(prompt).toContain("显式给出 URL");
    expect(prompt).toContain("不得先退化成 WebSearch");
    expect(prompt).toContain("一旦给了 URL，先打开页面");
    expect(prompt).toContain("service_skill_launch");
    expect(prompt).toContain("对话内 A2UI");
    expect(prompt).toContain("站点技能启动上下文");
    expect(prompt).toContain("lime_site_run");
    expect(prompt).toContain("旧外部浏览器工具");
    expect(prompt).not.toContain("mcp__lime-browser__");
    expect(prompt).toContain("严格 JSON 对象");
    expect(prompt).toContain("attached_session_required");
  });

  it("紧凑 Prompt 应保留核心边界并减少首轮输入体积", () => {
    const fullPrompt = buildGeneralAgentSystemPrompt("general", {
      now: new Date("2026-03-12T12:00:00+08:00"),
      harness: {
        browserAssistEnabled: true,
        browserAssistProfileKey: "general_browser_assist",
      },
    });
    const compactPrompt = buildGeneralAgentSystemPrompt("general", {
      now: new Date("2026-03-12T12:00:00+08:00"),
      compact: true,
      harness: {
        browserAssistEnabled: true,
        browserAssistProfileKey: "general_browser_assist",
      },
    });

    expect(compactPrompt.length).toBeLessThan(fullPrompt.length);
    expect(compactPrompt).toContain("当前日期：2026年3月12日");
    expect(compactPrompt).toContain("能直接回答就直接回答");
    expect(compactPrompt).toContain("WebSearch");
    expect(compactPrompt).toContain("lime_site_run");
    expect(compactPrompt).toContain("Playwright code");
    expect(compactPrompt).toContain("general_browser_assist");
    expect(compactPrompt).toContain("输出契约");
    expect(compactPrompt).toContain("标准 Markdown");
    expect(compactPrompt).toContain("runtime 状态");
    expect(compactPrompt).toContain("最终正文");
    expect(compactPrompt).toContain("Markdown 链接");
  });
});
