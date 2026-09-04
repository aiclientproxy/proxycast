import { beforeEach, describe, expect, it } from "vitest";

import type { AgentToolCallState } from "@/lib/api/agentProtocol";
import { changeLimeLocale } from "@/i18n/createI18n";

import { resolveToolProcessNarrative } from "./toolProcessSummary";

function createToolCall(
  overrides: Partial<AgentToolCallState>,
): AgentToolCallState {
  return {
    id: "tool-1",
    name: "ConfigTool",
    status: "completed",
    startTime: new Date("2026-04-14T00:00:00.000Z"),
    ...overrides,
  };
}

describe("toolProcessSummary", () => {
  beforeEach(async () => {
    await changeLimeLocale("zh-CN");
  });

  it("Agent-only 工作树工具不再提供 current 专用过程文案", () => {
    const enterNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "EnterWorktreeTool",
        status: "running",
      }),
    );
    const exitNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "ExitWorktreeTool",
        status: "completed",
      }),
    );

    expect(enterNarrative.preSummary).not.toBe("先进入隔离工作树");
    expect(enterNarrative.summary).not.toBe("先进入隔离工作树");
    expect(exitNarrative.postSummary).not.toBe("已回到主工作区");
    expect(exitNarrative.summary).not.toBe("已回到主工作区");
  });

  it("Agent-only 配置与工作流工具不再提供 current 专用过程文案", () => {
    const configNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "ConfigTool",
        status: "completed",
      }),
    );
    const workflowNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "WorkflowTool",
        status: "completed",
      }),
    );

    expect(configNarrative.preSummary).not.toBe("先查看或调整运行配置");
    expect(configNarrative.postSummary).not.toBe("已更新运行配置");
    expect(workflowNarrative.preSummary).not.toBe("先执行预设工作流");
    expect(workflowNarrative.postSummary).not.toBe("已执行工作流");
  });

  it("应为等待工具提供显式完成文案", () => {
    const narrative = resolveToolProcessNarrative(
      createToolCall({
        name: "sleep",
        status: "completed",
      }),
    );
    const deletedLegacyNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "SleepTool",
        status: "completed",
      }),
    );

    expect(narrative.preSummary).toBe("先等待一段时间再继续");
    expect(narrative.postSummary).toBe("已完成等待");
    expect(narrative.summary).toBe("已完成等待");
    expect(deletedLegacyNarrative.postSummary).not.toBe("已完成等待");
  });

  it("应为图片查看工具提供稳定过程文案，避免展示 raw output", () => {
    const runningNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "ViewImageTool",
        status: "running",
        arguments: JSON.stringify({ path: "/workspace/assets/sample.png" }),
      }),
    );
    const completedNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "view_image",
        status: "completed",
        arguments: JSON.stringify({ path: "/workspace/assets/sample.png" }),
        result: {
          success: true,
          output:
            "Viewed image: /workspace/assets/sample.png\nFormat: image/png\nImage content is attached to this tool result.",
        },
      }),
    );
    expect(runningNarrative.preSummary).toBe("先查看图片 sample.png");
    expect(runningNarrative.summary).toBe("先查看图片 sample.png");
    expect(completedNarrative.postSummary).toBe("已查看图片 sample.png");
    expect(completedNarrative.summary).toBe("已查看图片 sample.png");
    expect(completedNarrative.summary).not.toContain("Viewed image");
  });

  it("图片任务创建结果应显示任务状态文案而不是 raw JSON", () => {
    const narrative = resolveToolProcessNarrative(
      createToolCall({
        name: "mediaTaskArtifact/image/create",
        status: "completed",
        result: {
          success: true,
          output: JSON.stringify({
            success: true,
            task_id: "task-image-1",
            task_type: "image_generate",
            task_family: "image",
            status: "pending_submit",
            normalized_status: "pending",
            artifact_path: ".lime/tasks/image_generate/task-image-1.json",
            record: {
              payload: {
                prompt: "画一张广州夏天的图",
              },
            },
          }),
        },
      }),
    );

    expect(narrative.postSummary).toBe("正在生成图片。");
    expect(narrative.summary).toBe("正在生成图片。");
    expect(narrative.summary).not.toContain("task_id");
  });

  it("v2 image_generation task_family 结果应显示任务状态文案", () => {
    const narrative = resolveToolProcessNarrative(
      createToolCall({
        name: "mediaTaskArtifact/image/create",
        status: "completed",
        result: {
          success: true,
          output: JSON.stringify({
            success: true,
            task_id: "task-image-v2-family",
            task_family: "image_generation",
            status: "pending_submit",
            normalized_status: "pending",
            artifact_path:
              ".lime/tasks/image_generate/task-image-v2-family.json",
            record: {
              payload: {
                prompt: "画一张广州夏天的图",
              },
            },
          }),
        },
      }),
    );

    expect(narrative.postSummary).toBe("正在生成图片。");
    expect(narrative.summary).toBe("正在生成图片。");
    expect(narrative.summary).not.toContain("task_id");
  });

  it("应为计划模式与最终答复提供专用文案", () => {
    const updatePlanNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "update_plan",
        status: "completed",
      }),
    );
    const finalNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "SyntheticOutputTool",
        status: "completed",
      }),
    );

    expect(updatePlanNarrative.preSummary).toBe("已更新计划");
    expect(updatePlanNarrative.postSummary).toBe("已更新计划");
    expect(updatePlanNarrative.summary).toBe("已更新计划");
    expect(finalNarrative.preSummary).toBe("先整理最终答复");
    expect(finalNarrative.postSummary).toBe("已整理最终答复");
    expect(finalNarrative.summary).toBe("已整理最终答复");
  });

  it("应区分不同任务与计划工具的过程文案", () => {
    const taskCreateNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "TaskCreateTool",
        status: "completed",
        arguments: JSON.stringify({ title: "每日趋势摘要" }),
      }),
    );
    const taskListNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "TaskListTool",
        status: "completed",
      }),
    );
    const taskOutputNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "TaskOutputTool",
        status: "completed",
      }),
    );
    const taskStopNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "TaskStopTool",
        status: "completed",
        arguments: JSON.stringify({ task_id: "task-123" }),
      }),
    );
    expect(taskCreateNarrative.preSummary).toBe("先开始 每日趋势摘要");
    expect(taskCreateNarrative.postSummary).toBe("已开始 每日趋势摘要");
    expect(taskListNarrative.preSummary).toBe("先查看任务列表");
    expect(taskListNarrative.postSummary).toBe("已查看任务列表");
    expect(taskOutputNarrative.preSummary).toBe("先查看任务结果");
    expect(taskOutputNarrative.postSummary).toBe("已查看任务结果");
    expect(taskStopNarrative.preSummary).toBe("先终止任务 task-123");
    expect(taskStopNarrative.postSummary).toBe("已终止任务 task-123");
  });

  it("Agent-only gated runtime 工具不再保留专用主体，但失败摘要仍要净化", () => {
    const cronCreateNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "CronCreateTool",
        status: "completed",
        arguments: JSON.stringify({
          id: "morning-news",
          prompt: "整理国际新闻",
        }),
      }),
    );
    const cronDeleteNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "CronDeleteTool",
        status: "completed",
        arguments: JSON.stringify({ id: "morning-news" }),
      }),
    );
    const remoteTriggerNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "RemoteTriggerTool",
        status: "failed",
        arguments: JSON.stringify({ trigger_id: "remote-daily-news" }),
        result: {
          success: false,
          error: "-32603: -32002: remote trigger runtime is not configured",
          output: "",
        },
      }),
    );
    expect(cronCreateNarrative.preSummary).not.toBe(
      "先创建定时触发器 morning-news",
    );
    expect(cronCreateNarrative.postSummary).not.toBe(
      "已创建定时触发器 morning-news",
    );
    expect(cronDeleteNarrative.postSummary).not.toBe(
      "已删除定时触发器 morning-news",
    );
    expect(remoteTriggerNarrative.summary).toBe(
      "执行失败：remote trigger runtime is not configured",
    );
    expect(remoteTriggerNarrative.summary).not.toContain("-32603");
  });

  it("应为外部信息与结构化数据工具生成可读过程文案", () => {
    const searchNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "SearchQuery",
        status: "completed",
        arguments: JSON.stringify({ q: "2026-06-03 international news" }),
      }),
    );
    const imageNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "ImageQuery",
        status: "running",
        arguments: JSON.stringify({ query: "product screenshot" }),
      }),
    );
    const financeNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "finance",
        status: "completed",
        arguments: JSON.stringify({ ticker: "AAPL" }),
      }),
    );
    const weatherNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "weather",
        status: "completed",
        arguments: JSON.stringify({ location: "Tokyo" }),
      }),
    );
    const timeNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "time",
        status: "running",
        arguments: JSON.stringify({ utc_offset: "+09:00" }),
      }),
    );
    const resolveLibraryNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "ResolveLibraryId",
        status: "completed",
        arguments: JSON.stringify({ libraryName: "Next.js" }),
      }),
    );
    const queryDocsNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "QueryDocs",
        status: "completed",
        arguments: JSON.stringify({ query: "React useEffect cleanup" }),
      }),
    );

    expect(searchNarrative.preSummary).toBe(
      "先搜索 2026-06-03 international news",
    );
    expect(searchNarrative.postSummary).toBe(
      "已搜索 2026-06-03 international news",
    );
    expect(imageNarrative.preSummary).toBe("先搜索 product screenshot");
    expect(imageNarrative.summary).toBe("先搜索 product screenshot");
    expect(financeNarrative.postSummary).toBe("已获取 AAPL 内容");
    expect(weatherNarrative.postSummary).toBe("已获取 Tokyo 内容");
    expect(timeNarrative.preSummary).toBe("先获取 +09:00 内容");
    expect(resolveLibraryNarrative.postSummary).toBe("已搜索 Next.js");
    expect(queryDocsNarrative.postSummary).toBe(
      "已查看 React useEffect cleanup",
    );
  });

  it("应为 MCP 搜索与读取工具生成可读过程文案", () => {
    const searchNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "mcp__github__search_code",
        status: "running",
        arguments: JSON.stringify({ query: "repo:lime tool runtime" }),
      }),
    );
    const readNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "mcp__github__get_file_contents",
        status: "completed",
        arguments: JSON.stringify({ path: "docs/guide.md" }),
      }),
    );

    expect(searchNarrative.preSummary).toBe("先搜索 repo:lime tool runtime");
    expect(searchNarrative.summary).toBe("先搜索 repo:lime tool runtime");
    expect(readNarrative.preSummary).toBe("先查看 guide.md");
    expect(readNarrative.postSummary).toBe("已查看 guide.md");
    expect(readNarrative.summary).toBe("已查看 guide.md");
  });

  it("应为 compat dynamic aliases 生成专用过程文案", () => {
    const mcpNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "MCPTool",
        status: "completed",
      }),
    );
    const authNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "McpAuthTool",
        status: "completed",
      }),
    );
    const replNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "REPLTool",
        status: "running",
      }),
    );
    const listSkillsNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "ListSkills",
        status: "completed",
      }),
    );
    const loadSkillNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "LoadSkill",
        status: "completed",
        arguments: JSON.stringify({ name: "browser" }),
      }),
    );
    const waitNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "WaitAgent",
        status: "completed",
        arguments: JSON.stringify({ id: "agent-1" }),
      }),
    );
    const resumeNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "ResumeAgent",
        status: "running",
        arguments: JSON.stringify({ id: "agent-1" }),
      }),
    );
    const closeNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "CloseAgent",
        status: "completed",
        arguments: JSON.stringify({ id: "agent-1" }),
      }),
    );

    expect(mcpNarrative.preSummary).toBe("先调用 MCP 工具");
    expect(mcpNarrative.postSummary).toBe("已完成 MCP 工具调用");
    expect(authNarrative.preSummary).toBe("先完成 MCP 授权");
    expect(authNarrative.postSummary).toBe("已完成 MCP 授权");
    expect(replNarrative.summary).toBe("先运行命令确认当前状态");
    expect(listSkillsNarrative.postSummary).toBe("已查看可用技能");
    expect(loadSkillNarrative.postSummary).toBe("已加载技能 browser");
    expect(waitNarrative.postSummary).toBe("已查看子任务 agent-1 进展");
    expect(resumeNarrative.preSummary).toBe("先继续子任务 agent-1");
    expect(closeNarrative.postSummary).toBe("已暂停子任务 agent-1");
  });

  it("应为 MCP resource 与 mutation 工具生成可读过程文案", () => {
    const listResourceNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "ListMcpResourcesTool",
        status: "completed",
        arguments: JSON.stringify({ server: "docs" }),
      }),
    );
    const readResourceNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "ReadMcpResourceTool",
        status: "completed",
        arguments: JSON.stringify({
          server: "docs",
          uri: "file:///guide.md",
        }),
      }),
    );
    const mutationNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "mcp__github__create_issue",
        status: "completed",
        arguments: JSON.stringify({ title: "修复工具渲染" }),
        result: {
          success: true,
          output: JSON.stringify({ id: 123, title: "修复工具渲染" }),
        },
      }),
    );

    expect(listResourceNarrative.preSummary).toBe("先查看 docs");
    expect(listResourceNarrative.postSummary).toBe("已查看 docs");
    expect(readResourceNarrative.preSummary).toBe("先读取 file:///guide.md");
    expect(readResourceNarrative.postSummary).toBe("已读取 file:///guide.md");
    expect(mutationNarrative.preSummary).toBe("先处理 修复工具渲染");
    expect(mutationNarrative.postSummary).toBe("已处理 修复工具渲染");
    expect(mutationNarrative.summary).not.toContain("{");
  });

  it("应为站点目录、搜索、详情与执行工具生成站点语义文案", () => {
    const siteListNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_site_list",
        status: "completed",
      }),
    );
    const siteSearchNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_site_search",
        status: "running",
        arguments: JSON.stringify({ query: "GitHub issue 搜索" }),
      }),
    );
    const siteInfoNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_site_info",
        status: "completed",
        arguments: JSON.stringify({ adapter_name: "github/search" }),
      }),
    );
    const siteRunNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_site_run",
        status: "running",
        arguments: JSON.stringify({ adapter_name: "github/search" }),
      }),
    );

    expect(siteListNarrative.preSummary).toBe("先查看可用站点能力");
    expect(siteListNarrative.postSummary).toBe("已查看可用站点能力");
    expect(siteSearchNarrative.preSummary).toBe(
      "先搜索 GitHub issue 搜索 相关站点能力",
    );
    expect(siteSearchNarrative.summary).toBe(
      "先搜索 GitHub issue 搜索 相关站点能力",
    );
    expect(siteInfoNarrative.postSummary).toBe(
      "已确认 github/search 的参数与登录要求",
    );
    expect(siteRunNarrative.preSummary).toBe("先执行站点能力 github/search");
    expect(siteRunNarrative.summary).toBe("先执行站点能力 github/search");
  });

  it("应为服务技能与站点推荐工具生成专用过程文案", () => {
    const serviceSkillNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_run_service_skill",
        status: "running",
        arguments: JSON.stringify({ skill_title: "渠道预览" }),
      }),
    );
    const siteRecommendNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_site_recommend",
        status: "completed",
        arguments: JSON.stringify({ query: "GitHub issue 搜索" }),
      }),
    );

    expect(serviceSkillNarrative.preSummary).toBe("先执行服务技能 渠道预览");
    expect(serviceSkillNarrative.summary).toBe("先执行服务技能 渠道预览");
    expect(serviceSkillNarrative.summary).not.toContain("兼容");
    expect(siteRecommendNarrative.preSummary).toBe(
      "先推荐适合 GitHub issue 搜索 的站点能力",
    );
    expect(siteRecommendNarrative.postSummary).toBe(
      "已推荐适合 GitHub issue 搜索 的站点能力",
    );
  });

  it("应为新补齐的任务工具生成更贴近当前前台的发起文案", () => {
    const audioNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_create_audio_generation_task",
        status: "completed",
        arguments: JSON.stringify({ prompt: "温暖的播客旁白" }),
      }),
    );
    const transcriptionNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_create_transcription_task",
        status: "completed",
        arguments: JSON.stringify({ sourcePath: "/tmp/interview.mp4" }),
      }),
    );
    const resourceNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_create_modal_resource_search_task",
        status: "running",
        arguments: JSON.stringify({ query: "科技播客 BGM" }),
      }),
    );
    const urlParseNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_create_url_parse_task",
        status: "completed",
        arguments: JSON.stringify({ url: "https://example.com/report" }),
      }),
    );
    const typesettingNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_create_typesetting_task",
        status: "running",
        arguments: JSON.stringify({ targetPlatform: "小红书" }),
      }),
    );
    const coverImageNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "social_generate_cover_image",
        status: "completed",
        arguments: JSON.stringify({ subject: "开发 Lime 的经验" }),
      }),
    );

    expect(audioNarrative.postSummary).toBe("已发起 温暖的播客旁白 的配音生成");
    expect(transcriptionNarrative.preSummary).toBe(
      "先发起 /tmp/interview.mp4 的转写",
    );
    expect(transcriptionNarrative.postSummary).toBe(
      "已发起 /tmp/interview.mp4 的转写",
    );
    expect(resourceNarrative.preSummary).toBe("先发起 科技播客 BGM 的素材检索");
    expect(resourceNarrative.summary).toBe("先发起 科技播客 BGM 的素材检索");
    expect(urlParseNarrative.postSummary).toBe(
      "已发起 https://example.com/report 的链接解析",
    );
    expect(typesettingNarrative.preSummary).toBe("先发起 小红书 的排版");
    expect(coverImageNarrative.preSummary).toBe(
      "先生成 开发 Lime 的经验 的封面图",
    );
    expect(coverImageNarrative.postSummary).toBe(
      "已生成 开发 Lime 的经验 的封面图",
    );
  });

  it("generic、vision、站点和错误摘要应随当前语言切换", async () => {
    await changeLimeLocale("en-US");

    const readNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "Read",
        status: "completed",
        arguments: JSON.stringify({ file_path: "src/app.tsx" }),
      }),
    );
    const writeNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "Write",
        status: "running",
        arguments: JSON.stringify({ file_path: "src/output.md" }),
      }),
    );
    const visionNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "view_image",
        status: "completed",
        arguments: JSON.stringify({ path: "/workspace/assets/sample.png" }),
      }),
    );
    const siteNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_site_search",
        status: "running",
        arguments: JSON.stringify({ query: "GitHub issue search" }),
      }),
    );
    const failedNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "Bash",
        status: "failed",
        result: {
          success: false,
          error: "-32603: -32002: sandbox execution failed",
          output: "",
        },
      }),
    );

    expect(readNarrative.summary).toBe("Reviewed app.tsx");
    expect(writeNarrative.summary).toBe("Preparing to write output.md");
    expect(visionNarrative.summary).toBe("Image sample.png viewed");
    expect(siteNarrative.summary).toBe(
      "Searching site capabilities related to GitHub issue search first",
    );
    expect(failedNarrative.summary).toBe(
      "Run failed: sandbox execution failed",
    );
    expect(readNarrative.summary).not.toContain("已查看");
    expect(writeNarrative.summary).not.toContain("准备写入");
    expect(visionNarrative.summary).not.toContain("图片");
    expect(siteNarrative.summary).not.toContain("站点能力");
    expect(failedNarrative.summary).not.toContain("执行失败");
  });

  it("内容任务过程摘要应随当前语言切换且不回退中文 defaultValue", async () => {
    await changeLimeLocale("en-US");

    const audioNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_create_audio_generation_task",
        status: "completed",
        arguments: JSON.stringify({ prompt: "warm podcast narration" }),
      }),
    );
    const resourceNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_create_modal_resource_search_task",
        status: "running",
        arguments: JSON.stringify({ query: "podcast BGM" }),
      }),
    );
    const coverImageNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "social_generate_cover_image",
        status: "completed",
        arguments: JSON.stringify({ subject: "release recap" }),
      }),
    );

    expect(audioNarrative.postSummary).toBe(
      "Started Voice generation for warm podcast narration",
    );
    expect(resourceNarrative.preSummary).toBe(
      "Start Asset search for podcast BGM",
    );
    expect(resourceNarrative.summary).toBe(
      "Start Asset search for podcast BGM",
    );
    expect(coverImageNarrative.preSummary).toBe(
      "Generate cover image for release recap",
    );
    expect(coverImageNarrative.postSummary).toBe(
      "Generated cover image for release recap",
    );
    expect(
      [
        audioNarrative.postSummary,
        resourceNarrative.preSummary,
        coverImageNarrative.preSummary,
        coverImageNarrative.postSummary,
      ].join(" "),
    ).not.toMatch(/先发起|已发起|先生成|已生成|配音生成|素材检索|封面图/);
  });

  it("应把 WebSearch 协议错误翻译成可操作提示", () => {
    const narrative = resolveToolProcessNarrative(
      createToolCall({
        name: "WebSearch",
        status: "failed",
        result: {
          success: false,
          error: "-32603: -32002: WebSearch",
          output: "",
        },
      }),
    );

    expect(narrative.postSummary).toBe("搜索结果暂时无法读取");
    expect(narrative.summary).toBe("搜索结果暂时无法读取");
  });

  it("应把 WebFetch 失败降级成弱提示", () => {
    const narrative = resolveToolProcessNarrative(
      createToolCall({
        name: "WebFetch",
        status: "failed",
        result: {
          success: false,
          error: "404 Not Found",
          output: "",
        },
      }),
    );

    expect(narrative.postSummary).toBe("来源暂时无法读取");
    expect(narrative.summary).toBe("来源暂时无法读取");
  });

  it("WebFetch 返回 RSS/XML 或超时诊断时应降级成弱提示", () => {
    const rssNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "WebFetch",
        status: "completed",
        result: {
          success: true,
          output:
            '<?xml version="1.0"?><rss><channel><title>News</title></channel></rss>',
        },
      }),
    );
    const timeoutNarrative = resolveToolProcessNarrative(
      createToolCall({
        name: "WebSearch",
        status: "completed",
        result: {
          success: true,
          output: "Timeout while reading https://example.com/rss.xml",
        },
      }),
    );

    expect(rssNarrative.postSummary).toBe("来源暂时无法读取");
    expect(rssNarrative.summary).toBe("来源暂时无法读取");
    expect(timeoutNarrative.postSummary).toBe("搜索结果暂时无法读取");
    expect(timeoutNarrative.summary).toBe("搜索结果暂时无法读取");
  });

  it("Bash 协议错误摘要应保留底层原因而不是只展示错误码", () => {
    const narrative = resolveToolProcessNarrative(
      createToolCall({
        name: "Bash",
        status: "failed",
        arguments: JSON.stringify({
          command: "set -e\np='/Users/coso/.yansu-agent'\nls \"$p\"",
        }),
        result: {
          success: false,
          error: "-32603: -32002: sandbox 执行失败: Operation not permitted",
          output: "",
        },
      }),
    );

    expect(narrative.postSummary).toBe(
      "执行失败：sandbox 执行失败: Operation not permitted",
    );
    expect(narrative.summary).toBe(
      "执行失败：sandbox 执行失败: Operation not permitted",
    );
    expect(narrative.summary).not.toContain("-32603");
    expect(narrative.summary).not.toContain("-32002");
  });

  it("Bash 空协议错误应给出短失败说明而不是裸错误码", () => {
    const narrative = resolveToolProcessNarrative(
      createToolCall({
        name: "Bash",
        status: "failed",
        result: {
          success: false,
          error: "-32603: -32002:",
          output: "",
        },
      }),
    );

    expect(narrative.summary).toBe(
      "执行失败：命令执行失败，底层错误未返回详细信息",
    );
    expect(narrative.summary).not.toContain("-32603");
    expect(narrative.summary).not.toContain("-32002");
  });

  it("图片生成任务失败不应把内部协议错误带进过程摘要", () => {
    const narrative = resolveToolProcessNarrative(
      createToolCall({
        name: "lime_create_image_generation_task",
        arguments: JSON.stringify({
          prompt: "A comic book style illustration of a formal statue",
        }),
        status: "failed",
        result: {
          success: false,
          error: "-32603: -32002: lime_create_image_generation_task",
          output: "",
        },
      }),
    );

    expect(narrative.postSummary).toBe("生成失败");
    expect(narrative.summary).toBe("生成失败");
    expect(narrative.summary).not.toContain("-32603");
    expect(narrative.summary).not.toContain(
      "lime_create_image_generation_task",
    );
  });

  it("内容工作台任务失败不应泄露内部协议错误", () => {
    const cases = [
      {
        name: "video_generate",
        expected: "视频生成失败",
      },
      {
        name: "lime_create_audio_generation_task",
        expected: "配音生成失败",
      },
      {
        name: "lime_create_transcription_task",
        expected: "转写失败",
      },
      {
        name: "lime_create_modal_resource_search_task",
        expected: "素材检索失败",
      },
    ] as const;

    for (const { name, expected } of cases) {
      const narrative = resolveToolProcessNarrative(
        createToolCall({
          name,
          status: "failed",
          result: {
            success: false,
            error: `-32603: -32002: ${name}`,
            output: "",
          },
        }),
      );

      expect(narrative.summary).toBe(expected);
      expect(narrative.summary).not.toContain("-32603");
      expect(narrative.summary).not.toContain("-32002");
      expect(narrative.summary).not.toContain(name);
    }
  });
});
