import { afterEach, describe, expect, it } from "vitest";

import {
  buildToolHeadline,
  buildToolGroupHeadline,
  extractSearchQueryLabel,
  getToolDisplayInfo,
  normalizeToolNameKey,
  parseToolCallArguments,
  resolveToolFilePath,
  resolveToolPrimarySubject,
  resolveUserFacingToolDisplayLabel,
  resolveToolDisplayLabel,
} from "./toolDisplayInfo";
import { loadNamespaceResource } from "@/i18n/loadNamespace";
import { SUPPORTED_LOCALES } from "@/i18n/locales";

const resetDocumentLanguage = () => {
  document.documentElement.lang = "";
};

afterEach(resetDocumentLanguage);

const REFERENCE_JS_TOOL_NAME_MAPPINGS = [
  ["AgentTool", "agent"],
  ["BashTool", "bash"],
  ["developer__shell", "bash"],
  ["mcp__system__shell", "bash"],
  ["shell_command", "bash"],
  ["exec_command", "bash"],
  ["local_shell_call", "bash"],
  ["request_user_input", "requestuserinput"],
  ["RequestUserInputTool", "requestuserinput"],
  ["clock.sleep", "sleep"],
  ["sleep", "sleep"],
  ["update_plan", "updateplan"],
  ["UpdatePlanTool", "updateplan"],
  ["FileReadTool", "read"],
  ["read_file", "read"],
  ["developer__read", "read"],
  ["mcp__system__read_file", "read"],
  ["GlobTool", "glob"],
  ["mcp__system__glob", "glob"],
  ["GrepTool", "grep"],
  ["mcp__system__grep", "grep"],
  ["ListMcpResourcesTool", "listmcpresources"],
  ["MCPTool", "mcp"],
  ["McpAuthTool", "mcpauth"],
  ["PowerShellTool", "powershell"],
  ["ReadMcpResourceTool", "readmcpresource"],
  ["REPLTool", "repl"],
  ["ListSkills", "listskills"],
  ["LoadSkill", "loadskill"],
  ["WaitAgent", "waitagent"],
  ["ResumeAgent", "resumeagent"],
  ["CloseAgent", "closeagent"],
  ["SendMessageTool", "sendmessage"],
  ["SkillTool", "skill"],
  ["SyntheticOutputTool", "structuredoutput"],
  ["TaskCreateTool", "taskcreate"],
  ["TaskGetTool", "taskget"],
  ["TaskListTool", "tasklist"],
  ["TaskOutputTool", "taskoutput"],
  ["TaskStopTool", "taskstop"],
  ["KillShell", "taskstop"],
  ["TaskUpdateTool", "taskupdate"],
  ["TeamCreateTool", "teamcreate"],
  ["TeamDeleteTool", "teamdelete"],
  ["ListPeersTool", "listpeers"],
  ["ToolSearchTool", "toolsearch"],
  ["tool_search", "toolsearch"],
  ["mcp__system__tool_search", "toolsearch"],
  ["WebFetchTool", "webfetch"],
  ["web_fetch", "webfetch"],
  ["mcp__system__web_fetch", "webfetch"],
  ["WebSearchTool", "websearch"],
  ["web_search", "websearch"],
  ["mcp__system__web_search", "websearch"],
  ["ViewImageTool", "viewimage"],
] as const;

describe("toolDisplayInfo", () => {
  it("应把参考 JS 工具目录名归一化为现役展示键", () => {
    for (const [toolName, expected] of REFERENCE_JS_TOOL_NAME_MAPPINGS) {
      expect(normalizeToolNameKey(toolName)).toBe(expected);
    }

    expect(normalizeToolNameKey("AskUserQuestionTool")).toBe(
      "askuserquestiontool",
    );
    expect(normalizeToolNameKey("AgentOutputTool")).toBe("taskoutput");
    expect(normalizeToolNameKey("BashOutputTool")).toBe("taskoutput");
    expect(normalizeToolNameKey("ConfigTool")).toBe("configtool");
    expect(normalizeToolNameKey("EnterWorktreeTool")).toBe("enterworktreetool");
    expect(normalizeToolNameKey("ExitWorktreeTool")).toBe("exitworktreetool");
    expect(normalizeToolNameKey("NotebookEditTool")).toBe("notebookedittool");
    expect(normalizeToolNameKey("RemoteTriggerTool")).toBe("remotetriggertool");
    expect(normalizeToolNameKey("ScheduleCronTool")).toBe("schedulecrontool");
    expect(normalizeToolNameKey("SleepTool")).toBe("sleeptool");
  });

  it("应兼容导入来源中被双层 JSON 编码的工具参数", () => {
    const encoded = JSON.stringify(
      JSON.stringify({
        path: "/workspace/imported-local-history/docs/imported-preview.md",
      }),
    );

    expect(parseToolCallArguments(encoded)).toEqual({
      path: "/workspace/imported-local-history/docs/imported-preview.md",
    });
  });

  it("应兼容导入来源中已结构化的工具参数对象", () => {
    const parsed = parseToolCallArguments({
      path: "/workspace/imported-local-history/docs/imported-preview.md",
    });

    expect(parsed).toEqual({
      path: "/workspace/imported-local-history/docs/imported-preview.md",
    });
    expect(resolveToolFilePath(parsed)).toBe(
      "/workspace/imported-local-history/docs/imported-preview.md",
    );
  });

  it("应为参考 JS 工具目录名解析出当前展示文案", () => {
    expect(resolveToolDisplayLabel("request_user_input")).toBe("用户输入");
    expect(resolveToolDisplayLabel("developer__shell")).toBe("命令执行");
    expect(resolveToolDisplayLabel("exec_command")).toBe("命令执行");
    expect(resolveToolDisplayLabel("FileReadTool")).toBe("文件读取");
    expect(resolveToolDisplayLabel("mcp__system__read_file")).toBe("文件读取");
    expect(resolveToolDisplayLabel("update_plan")).toBe("计划更新");
    expect(resolveToolDisplayLabel("sleep")).toBe("等待");
    expect(resolveToolDisplayLabel("PowerShellTool")).toBe("PowerShell");
    expect(resolveToolDisplayLabel("MCPTool")).toBe("MCP 工具");
    expect(resolveToolDisplayLabel("McpAuthTool")).toBe("MCP 授权");
    expect(resolveToolDisplayLabel("REPLTool")).toBe("REPL 执行");
    expect(resolveToolDisplayLabel("ListSkills")).toBe("技能列表");
    expect(resolveToolDisplayLabel("LoadSkill")).toBe("技能加载");
    expect(resolveToolDisplayLabel("WaitAgent")).toBe("查看任务进展");
    expect(resolveToolDisplayLabel("ResumeAgent")).toBe("继续处理");
    expect(resolveToolDisplayLabel("CloseAgent")).toBe("暂停处理");
    expect(resolveToolDisplayLabel("SearchQuery")).toBe("网络搜索");
    expect(resolveToolDisplayLabel("ImageQuery")).toBe("图片搜索");
    expect(resolveToolDisplayLabel("finance")).toBe("行情查询");
    expect(resolveToolDisplayLabel("weather")).toBe("天气查询");
    expect(resolveToolDisplayLabel("sports")).toBe("体育查询");
    expect(resolveToolDisplayLabel("time")).toBe("时间查询");
    expect(resolveToolDisplayLabel("ResolveLibraryId")).toBe("库解析");
    expect(resolveToolDisplayLabel("QueryDocs")).toBe("文档查询");
    expect(resolveToolDisplayLabel("lime_search_web_images")).toBe("联网搜图");
    expect(resolveToolDisplayLabel("AgentTool")).toBe("创建子任务");
    expect(resolveToolDisplayLabel("SendMessageTool")).toBe("补充说明");
    expect(resolveToolDisplayLabel("ListPeersTool")).toBe("子任务");
    expect(resolveToolDisplayLabel("TeamCreateTool")).toBe("创建子代理组");
    expect(resolveToolDisplayLabel("TeamDeleteTool")).toBe("删除子代理组");
    expect(resolveToolDisplayLabel("SyntheticOutputTool")).toBe("最终答复");
    expect(resolveToolDisplayLabel("AgentOutputTool")).toBe("任务输出");
    expect(resolveToolDisplayLabel("BashOutputTool")).toBe("任务输出");
    expect(resolveToolDisplayLabel("ViewImageTool")).toBe("图片查看");
    expect(resolveToolDisplayLabel("lime_create_audio_generation_task")).toBe(
      "配音生成",
    );
    expect(resolveToolDisplayLabel("lime_create_transcription_task")).toBe(
      "转写",
    );
    expect(
      resolveToolDisplayLabel("lime_create_modal_resource_search_task"),
    ).toBe("素材检索");
    expect(resolveToolDisplayLabel("lime_run_service_skill")).toBe(
      "服务技能执行",
    );
    expect(getToolDisplayInfo("lime_run_service_skill", "running").family).toBe(
      "skill",
    );
    expect(getToolDisplayInfo("lime_run_service_skill", "running").action).toBe(
      "执行服务技能中",
    );
    expect(
      getToolDisplayInfo("lime_run_service_skill", "completed").action,
    ).toBe("已执行服务技能");
    expect(getToolDisplayInfo("lime_run_service_skill", "failed").action).toBe(
      "服务技能执行失败",
    );
    expect(resolveToolDisplayLabel("social_generate_cover_image")).toBe(
      "封面图生成",
    );
    expect(resolveToolDisplayLabel("lime_site_recommend")).toBe("站点能力推荐");
    expect(resolveToolDisplayLabel("mcp__github__search_code")).toBe(
      "MCP 搜索",
    );
    expect(resolveToolDisplayLabel("mcp__docs__list_pages")).toBe("MCP 列表");
    expect(resolveToolDisplayLabel("mcp__github__get_file_contents")).toBe(
      "MCP 读取",
    );
    expect(resolveToolDisplayLabel("mcp__playwright__browser_screenshot")).toBe(
      "页面截图",
    );
    expect(resolveToolDisplayLabel("mcp__github__create_issue")).toBe(
      "MCP 工具",
    );
  });

  it("应为用户可见场景提供更自然的工具标签", () => {
    expect(resolveUserFacingToolDisplayLabel("FileReadTool")).toBe("查看文件");
    expect(resolveUserFacingToolDisplayLabel("apply_patch")).toBe("修改文件");
    expect(resolveUserFacingToolDisplayLabel("PowerShellTool")).toBe(
      "运行命令",
    );
    expect(resolveUserFacingToolDisplayLabel("MCPTool")).toBe("调用 MCP 工具");
    expect(resolveUserFacingToolDisplayLabel("McpAuthTool")).toBe(
      "完成 MCP 授权",
    );
    expect(resolveUserFacingToolDisplayLabel("REPLTool")).toBe("运行命令");
    expect(resolveUserFacingToolDisplayLabel("ListSkills")).toBe("查看技能");
    expect(resolveUserFacingToolDisplayLabel("LoadSkill")).toBe("加载技能");
    expect(resolveUserFacingToolDisplayLabel("WaitAgent")).toBe("查看任务进展");
    expect(resolveUserFacingToolDisplayLabel("ResumeAgent")).toBe("继续处理");
    expect(resolveUserFacingToolDisplayLabel("CloseAgent")).toBe("暂停处理");
    expect(resolveUserFacingToolDisplayLabel("SearchQuery")).toBe("搜索网页");
    expect(resolveUserFacingToolDisplayLabel("ImageQuery")).toBe("搜索图片");
    expect(resolveUserFacingToolDisplayLabel("finance")).toBe("获取数据");
    expect(resolveUserFacingToolDisplayLabel("weather")).toBe("获取数据");
    expect(resolveUserFacingToolDisplayLabel("sports")).toBe("获取数据");
    expect(resolveUserFacingToolDisplayLabel("time")).toBe("获取数据");
    expect(resolveUserFacingToolDisplayLabel("ResolveLibraryId")).toBe(
      "查找内容",
    );
    expect(resolveUserFacingToolDisplayLabel("QueryDocs")).toBe("查看文档");
    expect(resolveUserFacingToolDisplayLabel("lime_run_service_skill")).toBe(
      "运行服务技能",
    );
    expect(resolveUserFacingToolDisplayLabel("lime_site_recommend")).toBe(
      "推荐站点能力",
    );
    expect(resolveUserFacingToolDisplayLabel("mcp__github__search_code")).toBe(
      "搜索内容",
    );
    expect(resolveUserFacingToolDisplayLabel("mcp__docs__list_pages")).toBe(
      "查看内容",
    );
    expect(
      resolveUserFacingToolDisplayLabel("mcp__github__get_file_contents"),
    ).toBe("查看内容");
    expect(
      resolveUserFacingToolDisplayLabel("mcp__playwright__browser_screenshot"),
    ).toBe("页面截图");
    expect(resolveUserFacingToolDisplayLabel("mcp__github__create_issue")).toBe(
      "调用 MCP 工具",
    );
    expect(resolveUserFacingToolDisplayLabel("TaskOutput")).toBe(
      "查看任务结果",
    );
    expect(resolveUserFacingToolDisplayLabel("TaskCreateTool")).toBe(
      "开始这一步",
    );
    expect(
      resolveUserFacingToolDisplayLabel("mcp__playwright__browser_click"),
    ).toBe("页面点击");
    expect(resolveUserFacingToolDisplayLabel("ViewImageTool")).toBe("查看图片");
    expect(resolveUserFacingToolDisplayLabel("video_generate")).toBe(
      "生成视频",
    );
    expect(
      resolveUserFacingToolDisplayLabel("lime_create_audio_generation_task"),
    ).toBe("生成配音");
    expect(resolveUserFacingToolDisplayLabel("generate_image")).toBe(
      "生成图片",
    );
    expect(
      resolveUserFacingToolDisplayLabel("lime_create_image_generation_task"),
    ).toBe("生成图片");
    expect(
      resolveUserFacingToolDisplayLabel("social_generate_cover_image"),
    ).toBe("生成封面图");
  });

  it("应为站点与任务工具提取更贴近主链的主体对象", () => {
    expect(
      resolveToolPrimarySubject(
        "lime_create_transcription_task",
        { sourceUrl: "https://example.com/interview.mp4" },
        null,
      ),
    ).toBe("https://example.com/interview.mp4");
    expect(
      resolveToolPrimarySubject(
        "lime_create_modal_resource_search_task",
        { query: "科技播客 BGM" },
        null,
      ),
    ).toBe("科技播客 BGM");
    expect(resolveToolPrimarySubject("lime_site_list", {}, null)).toBe(
      "站点能力目录",
    );
    expect(
      getToolDisplayInfo("lime_create_typesetting_task", "running").family,
    ).toBe("task");
    expect(
      getToolDisplayInfo("lime_create_audio_generation_task", "completed")
        .groupTitle,
    ).toBe("内容任务");
    expect(
      getToolDisplayInfo("lime_create_audio_generation_task", "completed")
        .action,
    ).toBe("已发起");
  });

  it("本地历史导入的 exec_command 应展示真实命令而不是只显示工具名", () => {
    const subject = resolveToolPrimarySubject(
      "exec_command",
      { command: "npm test", cwd: "/workspace/imported-codex" },
      null,
    );

    expect(subject).toBe("npm test");
    expect(
      resolveToolPrimarySubject(
        "functions.exec_command",
        { cmd: "rg -n thinking src" },
        null,
      ),
    ).toBe("rg -n thinking src");
    expect(
      buildToolHeadline({
        toolDisplay: getToolDisplayInfo("exec_command", "completed"),
        toolName: "exec_command",
        subject,
      }),
    ).toContain("npm test");
  });

  it("应为外部信息与结构化数据工具提取主体对象", () => {
    expect(
      resolveToolPrimarySubject(
        "SearchQuery",
        { q: "2026-06-03 international news" },
        null,
      ),
    ).toBe("2026-06-03 international news");
    expect(
      resolveToolPrimarySubject(
        "ImageQuery",
        { query: "product screenshot" },
        null,
      ),
    ).toBe("product screenshot");
    expect(resolveToolPrimarySubject("finance", { ticker: "AAPL" }, null)).toBe(
      "AAPL",
    );
    expect(
      resolveToolPrimarySubject("weather", { location: "Tokyo" }, null),
    ).toBe("Tokyo");
    expect(
      resolveToolPrimarySubject("sports", { team: "GSW", league: "nba" }, null),
    ).toBe("GSW");
    expect(
      resolveToolPrimarySubject("time", { utc_offset: "+09:00" }, null),
    ).toBe("+09:00");
    expect(
      resolveToolPrimarySubject(
        "ResolveLibraryId",
        { libraryName: "Next.js" },
        null,
      ),
    ).toBe("Next.js");
    expect(
      resolveToolPrimarySubject(
        "QueryDocs",
        { query: "React useEffect cleanup" },
        null,
      ),
    ).toBe("React useEffect cleanup");
  });

  it("应为 MCP resource 与动态 MCP 工具提取用户可读主体对象", () => {
    expect(
      resolveToolPrimarySubject(
        "ListMcpResourcesTool",
        { server: "docs" },
        null,
      ),
    ).toBe("docs");
    expect(
      resolveToolPrimarySubject(
        "ReadMcpResourceTool",
        { server: "docs", uri: "file:///guide.md" },
        null,
      ),
    ).toBe("file:///guide.md");
    expect(
      resolveToolPrimarySubject(
        "mcp__github__create_issue",
        { title: "修复工具渲染" },
        null,
      ),
    ).toBe("修复工具渲染");
    expect(
      resolveToolPrimarySubject("LoadSkill", { name: "browser" }, null),
    ).toBe("browser");
    expect(
      resolveToolPrimarySubject("WaitAgent", { id: "agent-1" }, null),
    ).toBe("agent-1");
    expect(
      resolveToolPrimarySubject("ResumeAgent", { session_id: "agent-2" }, null),
    ).toBe("agent-2");
    expect(
      resolveToolPrimarySubject("CloseAgent", { ids: ["agent-3"] }, null),
    ).toBe("agent-3");
  });

  it("应隐藏 ToolSearch 中的内部协议查询词", () => {
    expect(
      extractSearchQueryLabel({
        id: "tool-1",
        name: "ToolSearch",
        arguments: JSON.stringify({ query: "select:StructuredOutput" }),
        status: "completed",
        startTime: new Date("2026-04-09T00:00:00.000Z"),
      }),
    ).toBe("工具入口");
    expect(resolveToolPrimarySubject("ToolSearch", {}, null)).toBe("工具入口");
    expect(resolveUserFacingToolDisplayLabel("ToolSearch")).toBe(
      "确认工具入口",
    );
  });

  it("无主体对象时应直接展示动作句，避免重复拼接工具类别", () => {
    expect(
      buildToolHeadline({
        toolDisplay: getToolDisplayInfo("TaskList", "completed"),
        toolName: "TaskList",
      }),
    ).toBe("已获取任务列表");

    expect(
      buildToolHeadline({
        toolDisplay: getToolDisplayInfo("ListSkills", "completed"),
        toolName: "ListSkills",
      }),
    ).toBe("已获取技能列表");
  });

  it("应为查看类与计划类批次生成更自然的标题", () => {
    expect(
      buildToolGroupHeadline([
        {
          id: "tool-read-1",
          name: "Read",
          arguments: JSON.stringify({ file_path: "docs/guide.md" }),
          status: "completed",
          result: { success: true, output: "ok" },
          startTime: new Date("2026-04-13T00:00:00.000Z"),
          endTime: new Date("2026-04-13T00:00:01.000Z"),
        },
        {
          id: "tool-glob-1",
          name: "glob",
          arguments: JSON.stringify({ pattern: "src/**/*.tsx" }),
          status: "completed",
          result: { success: true, output: "ok" },
          startTime: new Date("2026-04-13T00:00:02.000Z"),
          endTime: new Date("2026-04-13T00:00:03.000Z"),
        },
      ]),
    ).toBe("已查看");

    expect(
      buildToolGroupHeadline([
        {
          id: "tool-task-list-1",
          name: "TaskList",
          arguments: JSON.stringify({}),
          status: "completed",
          result: { success: true, output: "[]" },
          startTime: new Date("2026-04-13T00:00:04.000Z"),
          endTime: new Date("2026-04-13T00:00:05.000Z"),
        },
        {
          id: "tool-task-update-1",
          name: "TaskUpdate",
          arguments: JSON.stringify({ task_id: "task-1" }),
          status: "completed",
          result: { success: true, output: "{}" },
          startTime: new Date("2026-04-13T00:00:06.000Z"),
          endTime: new Date("2026-04-13T00:00:07.000Z"),
        },
      ]),
    ).toBe("已处理 2 项安排");
  });

  it("工具批次标题应走 agent namespace 资源而不是中文硬编码", () => {
    document.documentElement.lang = "en-US";

    const commandInfo = getToolDisplayInfo("exec_command", "completed");
    expect(commandInfo.label).toBe("Command run");
    expect(commandInfo.groupTitle).toBe("Command");
    expect(commandInfo.verb).toBe("Run");
    expect(commandInfo.action).toBe("Ran");

    const mcpInfo = getToolDisplayInfo("McpAuthTool", "completed");
    expect(mcpInfo.label).toBe("MCP authorization");
    expect(mcpInfo.groupTitle).toBe("MCP");
    expect(mcpInfo.action).toBe("MCP authorization completed");

    expect(
      buildToolGroupHeadline([
        {
          id: "tool-command-1",
          name: "bash",
          arguments: JSON.stringify({ command: "npm test" }),
          status: "completed",
          result: { success: true, output: "ok" },
          startTime: new Date("2026-04-13T00:00:00.000Z"),
          endTime: new Date("2026-04-13T00:00:01.000Z"),
        },
        {
          id: "tool-command-2",
          name: "exec_command",
          arguments: JSON.stringify({ command: "npm run lint" }),
          status: "completed",
          result: { success: true, output: "ok" },
          startTime: new Date("2026-04-13T00:00:02.000Z"),
          endTime: new Date("2026-04-13T00:00:03.000Z"),
        },
      ]),
    ).toBe("Ran 2 commands");

    expect(
      buildToolGroupHeadline([
        {
          id: "tool-browser-1",
          name: "browser_click",
          arguments: JSON.stringify({ element: "Import" }),
          status: "running",
          startTime: new Date("2026-04-13T00:00:04.000Z"),
        },
      ]),
    ).toBe("1 page operation steps running");
  });

  it("i18n 未初始化时工具批次标题也应读取非中英语言资源", () => {
    document.documentElement.lang = "ja-JP";

    const browserInfo = getToolDisplayInfo(
      "mcp__playwright__browser_screenshot",
      "running",
    );
    expect(browserInfo.label).toBe("ページスクリーンショット");
    expect(browserInfo.groupTitle).toBe("ブラウザー");
    expect(browserInfo.verb).toBe("スクリーンショット");
    expect(browserInfo.action).toBe("スクリーンショット取得中");

    expect(
      buildToolGroupHeadline([
        {
          id: "tool-command-ja-1",
          name: "bash",
          arguments: JSON.stringify({ command: "npm test" }),
          status: "completed",
          result: { success: true, output: "ok" },
          startTime: new Date("2026-04-13T00:00:00.000Z"),
          endTime: new Date("2026-04-13T00:00:01.000Z"),
        },
      ]),
    ).toBe("1 件のコマンドを実行済み");

    document.documentElement.lang = "ko-KR";

    const siteInfo = getToolDisplayInfo("lime_site_recommend", "completed");
    expect(siteInfo.label).toBe("사이트 기능 추천");
    expect(siteInfo.groupTitle).toBe("사이트");
    expect(siteInfo.verb).toBe("추천");
    expect(siteInfo.action).toBe("추천됨");

    expect(
      buildToolGroupHeadline([
        {
          id: "tool-write-ko-1",
          name: "write_file",
          arguments: JSON.stringify({ path: "docs/result.md" }),
          status: "running",
          startTime: new Date("2026-04-13T00:00:02.000Z"),
        },
      ]),
    ).toBe("파일 1개 저장 중");
  });

  it("应为图片查看批次生成查看语义标题", () => {
    expect(
      buildToolGroupHeadline([
        {
          id: "tool-view-image-1",
          name: "ViewImageTool",
          arguments: JSON.stringify({ path: "assets/sample.png" }),
          status: "completed",
          result: { success: true, output: "Viewed image: assets/sample.png" },
          startTime: new Date("2026-04-13T00:00:00.000Z"),
          endTime: new Date("2026-04-13T00:00:01.000Z"),
        },
      ]),
    ).toBe("已查看 1 张图片");
  });

  it("应为内容工作台批次生成发起和生成语义标题", () => {
    expect(
      buildToolGroupHeadline([
        {
          id: "tool-video-1",
          name: "video_generate",
          arguments: JSON.stringify({ prompt: "产品演示短片" }),
          status: "completed",
          result: { success: true, output: "{}" },
          startTime: new Date("2026-04-13T00:00:00.000Z"),
          endTime: new Date("2026-04-13T00:00:01.000Z"),
        },
      ]),
    ).toBe("已发起视频生成");

    expect(
      buildToolGroupHeadline([
        {
          id: "tool-audio-1",
          name: "lime_create_audio_generation_task",
          arguments: JSON.stringify({ prompt: "播客旁白" }),
          status: "completed",
          result: { success: true, output: "{}" },
          startTime: new Date("2026-04-13T00:00:00.000Z"),
          endTime: new Date("2026-04-13T00:00:01.000Z"),
        },
        {
          id: "tool-resource-1",
          name: "lime_create_modal_resource_search_task",
          arguments: JSON.stringify({ query: "科技 BGM" }),
          status: "completed",
          result: { success: true, output: "{}" },
          startTime: new Date("2026-04-13T00:00:02.000Z"),
          endTime: new Date("2026-04-13T00:00:03.000Z"),
        },
      ]),
    ).toBe("已发起 2 个内容任务");

    expect(
      buildToolGroupHeadline([
        {
          id: "tool-cover-1",
          name: "social_generate_cover_image",
          arguments: JSON.stringify({ subject: "开发 Lime 的经验" }),
          status: "completed",
          result: { success: true, output: "{}" },
          startTime: new Date("2026-04-13T00:00:04.000Z"),
          endTime: new Date("2026-04-13T00:00:05.000Z"),
        },
      ]),
    ).toBe("已生成封面图");
  });

  it("应按站点工具族生成批次标题，不依赖展示分组中文", () => {
    expect(
      buildToolGroupHeadline([
        {
          id: "tool-site-search-1",
          name: "lime_site_search",
          arguments: JSON.stringify({ query: "GitHub issue 搜索" }),
          status: "completed",
          result: { success: true, output: "{}" },
          startTime: new Date("2026-04-13T00:00:00.000Z"),
          endTime: new Date("2026-04-13T00:00:01.000Z"),
        },
      ]),
    ).toBe("已搜索站点能力");

    expect(
      buildToolGroupHeadline([
        {
          id: "tool-site-list-1",
          name: "lime_site_list",
          arguments: JSON.stringify({}),
          status: "running",
          result: { success: true, output: "{}" },
          startTime: new Date("2026-04-13T00:00:02.000Z"),
        },
      ]),
    ).toBe("站点浏览中");

    expect(
      buildToolGroupHeadline([
        {
          id: "tool-site-run-1",
          name: "lime_site_run",
          arguments: JSON.stringify({ adapter_name: "github/search" }),
          status: "completed",
          result: { success: true, output: "{}" },
          startTime: new Date("2026-04-13T00:00:03.000Z"),
          endTime: new Date("2026-04-13T00:00:04.000Z"),
        },
        {
          id: "tool-site-info-1",
          name: "lime_site_info",
          arguments: JSON.stringify({ adapter_name: "github/search" }),
          status: "completed",
          result: { success: true, output: "{}" },
          startTime: new Date("2026-04-13T00:00:05.000Z"),
          endTime: new Date("2026-04-13T00:00:06.000Z"),
        },
      ]),
    ).toBe("已完成 2 项站点操作");
  });

  it("工具批次标题资源应覆盖所有支持语言", () => {
    const zhCNResource = loadNamespaceResource("zh-CN", "agent");
    const requiredKeys = Object.keys(zhCNResource).filter(
      (key) =>
        key.startsWith("agentChat.toolCall.group.") &&
        !key.endsWith(".hiddenItems") &&
        !key.endsWith(".collapseWork") &&
        !key.endsWith(".expandWork") &&
        !key.endsWith(".collapseSearch") &&
        !key.endsWith(".expandSearch") &&
        !key.endsWith(".hiddenSearchGroups"),
    );

    expect(requiredKeys).toContain(
      "agentChat.toolCall.group.command.completed",
    );

    for (const locale of SUPPORTED_LOCALES) {
      const resource = loadNamespaceResource(locale, "agent");
      for (const key of requiredKeys) {
        expect(resource[key], `${locale} missing ${key}`).toBeTruthy();
      }
    }
  });
});
