import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import type { AgentToolCallState } from "@/lib/api/agentProtocol";
import type { SkillDetailInfo } from "@/lib/api/skill-execution";
import type { Message } from "../types";
import type { GeneralWorkbenchActivityLogGroup } from "./generalWorkbenchWorkflowData";
import {
  buildGeneralWorkbenchExecLogEntries,
  filterGeneralWorkbenchExecLogEntries,
} from "./generalWorkbenchExecLogData";

function skillDetail(
  overrides: Partial<SkillDetailInfo> = {},
): SkillDetailInfo {
  return {
    name: "content_post_with_cover",
    display_name: "社媒主稿与封面",
    description: "生成内容主稿，并补齐封面素材。",
    execution_mode: "prompt",
    has_workflow: true,
    workflow_steps: [
      {
        id: "outline",
        name: "提炼内容主线",
        dependencies: [],
      },
      {
        id: "cover",
        name: "生成封面提示词",
        dependencies: ["outline"],
      },
    ],
    allowed_tools: ["read_file", "generate_image"],
    when_to_use: "适合需要主稿与封面同时产出的社媒场景。",
    markdown_content: "",
    ...overrides,
  };
}

function activityGroup(
  overrides: Partial<GeneralWorkbenchActivityLogGroup> = {},
): GeneralWorkbenchActivityLogGroup {
  return {
    key: "run:run-1",
    runId: "run-1",
    status: "completed",
    source: "skill",
    timeLabel: "10:30",
    artifactPaths: ["content-posts/research.md"],
    logs: [
      {
        id: "log-1",
        name: "content_post_with_cover",
        status: "completed",
        timeLabel: "10:30",
        source: "skill",
        sourceRef: "content_post_with_cover",
      },
    ],
    ...overrides,
  };
}

function assistantMessage(overrides: Partial<Message> = {}): Message {
  return {
    id: "assistant-1",
    role: "assistant",
    content: "已完成。",
    timestamp: new Date("2026-03-12T10:35:00.000Z"),
    ...overrides,
  };
}

function toolCall(
  overrides: Omit<AgentToolCallState, "startTime"> &
    Partial<Pick<AgentToolCallState, "startTime">>,
): AgentToolCallState {
  return {
    startTime: new Date("2026-03-12T10:36:00.000Z"),
    ...overrides,
  };
}

describe("generalWorkbenchExecLogData", () => {
  beforeEach(async () => {
    await changeLimeLocale("zh-CN");
  });

  afterEach(async () => {
    await changeLimeLocale("zh-CN");
  });

  it("应把消息、thinking、工具调用和响应投影成按时间排序的执行日志", () => {
    const entries = buildGeneralWorkbenchExecLogEntries({
      messages: [
        {
          id: "user-1",
          role: "user",
          content: "请检查文件",
          timestamp: new Date("2026-03-12T10:34:00.000Z"),
        },
        assistantMessage({
          thinkingContent: "先读取文件。",
          toolCalls: [
            {
              id: "tool-1",
              name: "read_file",
              arguments: JSON.stringify({ path: "/tmp/a.txt", limit: 50 }),
              status: "failed",
              result: {
                success: false,
                output: "",
                error: "文件不存在",
              },
              startTime: new Date("2026-03-12T10:34:30.000Z"),
            },
          ],
        }),
      ],
      groupedActivityLogs: [],
      groupedCreationTaskEvents: [],
      skillDetailMap: {},
    });

    expect(entries.map((entry) => [entry.type, entry.typeLabel])).toEqual([
      ["user", "用户请求"],
      ["tool", "查看文件"],
      ["thinking", "深度思考"],
      ["response", "AI 响应"],
    ]);
    expect(entries.find((entry) => entry.type === "tool")).toMatchObject({
      content: "path: /tmp/a.txt · limit: 50",
      meta: "❌ 文件不存在",
      status: "failed",
      detail: {
        kind: "tool",
        argumentsText: '{\n  "path": "/tmp/a.txt",\n  "limit": 50\n}',
        errorText: "文件不存在",
      },
    });
  });

  it("应复用工具展示语义而不是旧标签映射", () => {
    const entries = buildGeneralWorkbenchExecLogEntries({
      messages: [
        assistantMessage({
          id: "assistant-tool-labels",
          content: "",
          toolCalls: [
            toolCall({
              id: "tool-browser-1",
              name: "mcp__site__navigate",
              arguments: JSON.stringify({ url: "https://example.com/docs" }),
              status: "completed",
              result: { success: true, output: "ok" },
            }),
            toolCall({
              id: "tool-task-output-1",
              name: "TaskOutput",
              arguments: JSON.stringify({ task_id: "video-task-1" }),
              status: "completed",
              result: { success: true, output: "done" },
            }),
            toolCall({
              id: "tool-input-1",
              name: "request_user_input",
              arguments: JSON.stringify({ question: "需要继续吗？" }),
              status: "running",
            }),
          ],
        }),
      ],
      groupedActivityLogs: [],
      groupedCreationTaskEvents: [],
      skillDetailMap: {},
    });

    expect(entries.map((entry) => entry.typeLabel)).toEqual([
      "页面打开",
      "查看任务结果",
      "用户输入",
    ]);
  });

  it("应构造技能运行详情、工具列表和产物路径", () => {
    const entries = buildGeneralWorkbenchExecLogEntries({
      messages: [],
      groupedActivityLogs: [activityGroup()],
      groupedCreationTaskEvents: [],
      skillDetailMap: {
        content_post_with_cover: skillDetail(),
      },
    });

    expect(entries[0]).toMatchObject({
      type: "run",
      typeLabel: "技能：社媒主稿与封面",
      content: "执行技能 社媒主稿与封面",
      meta: expect.stringContaining("技能标识：content_post_with_cover"),
      detail: {
        kind: "skill",
        sourceRef: "content_post_with_cover",
        description: "生成内容主稿，并补齐封面素材。",
        workflowSteps: ["提炼内容主线", "生成封面提示词"],
        allowedTools: ["查看文件", "生成图片"],
        whenToUse: "适合需要主稿与封面同时产出的社媒场景。",
        artifactPaths: ["content-posts/research.md"],
      },
    });
  });

  it("应把 creation task 转为任务提交日志并支持清空时间过滤", () => {
    vi.setSystemTime(new Date("2026-03-12T10:00:00.000Z"));
    const entries = buildGeneralWorkbenchExecLogEntries({
      messages: [],
      groupedActivityLogs: [],
      groupedCreationTaskEvents: [
        {
          key: "video_generate",
          taskType: "video_generate",
          label: "视频生成",
          latestTimeLabel: "09:30",
          tasks: [
            {
              taskId: "video-1",
              taskType: "video_generate",
              path: "tasks/video-1.json",
              createdAt: new Date("2026-03-12T09:30:00.000Z").getTime(),
              timeLabel: "09:30",
            },
          ],
        },
      ],
      skillDetailMap: {},
    });

    expect(entries[0]).toMatchObject({
      id: "task-video_generate",
      type: "task",
      typeLabel: "任务提交",
      content: "视频生成",
      status: "completed",
    });
    expect(filterGeneralWorkbenchExecLogEntries(entries, null)).toEqual(
      entries,
    );
    expect(
      filterGeneralWorkbenchExecLogEntries(
        entries,
        new Date("2026-03-12T09:45:00.000Z").getTime(),
      ),
    ).toEqual([]);
    vi.useRealTimers();
  });
});
