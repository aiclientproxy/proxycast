import { describe, expect, it } from "vitest";
import {
  buildGeneralWorkbenchActivityLogGroups,
  parseGeneralWorkbenchRunMetadataSummary,
} from "./generalWorkbenchWorkflowData";

describe("generalWorkbenchWorkflowData", () => {
  it("应通过 artifact protocol 统一解析运行元数据里的产物路径", () => {
    expect(
      parseGeneralWorkbenchRunMetadataSummary(
        JSON.stringify({
          workflow: "social",
          execution_id: "exec-1",
          version_id: "version-1",
          stages: ["write_mode"],
          artifact_paths: [" content-posts/demo.md "],
          artifactPath: "content-posts\\demo-cover.png",
          filePath: "content-posts/summary.md",
        }),
      ),
    ).toEqual({
      workflow: "social",
      executionId: "exec-1",
      versionId: "version-1",
      stages: ["write_mode"],
      artifactPaths: [
        "content-posts/summary.md",
        "content-posts/demo-cover.png",
        "content-posts/demo.md",
      ],
      curatedTask: null,
    });
  });

  it("应解析 harness 里的 curated task 并补齐结果去向与下一步", () => {
    expect(
      parseGeneralWorkbenchRunMetadataSummary(
        JSON.stringify({
          harness: {
            curated_task: {
              task_id: "daily-trend-briefing",
              task_title: "每日趋势摘要",
            },
          },
        }),
      ),
    ).toMatchObject({
      curatedTask: {
        taskId: "daily-trend-briefing",
        taskTitle: "每日趋势摘要",
        resultDestination: "趋势摘要会先写回当前内容，方便继续展开选题和主稿。",
        followUpActions: ["继续展开其中一个选题", "生成首条内容主稿"],
      },
    });
  });

  it("应继续解析 curated task 的启动参数与灵感引用", () => {
    expect(
      parseGeneralWorkbenchRunMetadataSummary(
        JSON.stringify({
          harness: {
            curated_task: {
              task_id: "daily-trend-briefing",
              task_title: "每日趋势摘要",
              launch_input_values: {
                theme_target: "AI 内容创作",
                platform_region: "X 与 TikTok 北美区",
              },
              reference_memory_ids: ["memory-1"],
              reference_entries: [
                {
                  id: "memory-1",
                  title: "品牌风格样本",
                  summary: "保留轻盈但专业的表达。",
                  category: "context",
                  tags: ["品牌", "语气"],
                },
              ],
            },
          },
        }),
      ),
    ).toMatchObject({
      curatedTask: {
        taskId: "daily-trend-briefing",
        launchInputValues: {
          theme_target: "AI 内容创作",
          platform_region: "X 与 TikTok 北美区",
        },
        referenceMemoryIds: ["memory-1"],
        referenceEntries: [
          {
            id: "memory-1",
            title: "品牌风格样本",
            summary: "保留轻盈但专业的表达。",
            category: "context",
            categoryLabel: "参考",
            tags: ["品牌", "语气"],
          },
        ],
      },
    });
  });

  it("应把结果参考统一解析为 memory reference", () => {
    expect(
      parseGeneralWorkbenchRunMetadataSummary(
        JSON.stringify({
          harness: {
            curated_task: {
              task_id: "account-project-review",
              task_title: "复盘这个账号/项目",
              reference_entries: [
                {
                  id: "memory:content-pack:run:1",
                  source_kind: "memory",
                  title: "AI 内容周报",
                  summary: "当前已有一轮运行结果，可直接作为复盘基线。",
                  category: "experience",
                  tags: ["复盘"],
                  task_prefill_by_task_id: {
                    "account-project-review": {
                      project_goal: "AI 内容周报",
                      existing_results: "当前已有一轮运行结果。",
                    },
                  },
                },
              ],
            },
          },
        }),
      ),
    ).toMatchObject({
      curatedTask: {
        taskId: "account-project-review",
        referenceEntries: [
          {
            id: "memory:content-pack:run:1",
            sourceKind: "memory",
            title: "AI 内容周报",
            category: "experience",
            taskPrefillByTaskId: {
              "account-project-review": {
                project_goal: "AI 内容周报",
                existing_results: "当前已有一轮运行结果。",
              },
            },
          },
        ],
      },
    });
    expect(
      parseGeneralWorkbenchRunMetadataSummary(
        JSON.stringify({
          harness: {
            curated_task: {
              task_id: "account-project-review",
              reference_entries: [
                {
                  id: "memory:content-pack:run:1",
                  source_kind: "memory",
                  title: "AI 内容周报",
                  summary: "当前已有一轮运行结果，可直接作为复盘基线。",
                  category: "experience",
                  tags: ["复盘"],
                },
              ],
            },
          },
        }),
      ).curatedTask?.referenceMemoryIds,
    ).toEqual(["memory:content-pack:run:1"]);
  });

  it("应合并同一运行日志中的产物路径并保持规范化", () => {
    expect(
      buildGeneralWorkbenchActivityLogGroups([
        {
          id: "log-1",
          name: "排版优化",
          status: "running",
          timeLabel: "10:30",
          runId: "run-1",
          artifactPaths: [" content-posts/demo.md "],
        },
        {
          id: "log-2",
          name: "排版优化",
          status: "completed",
          timeLabel: "10:31",
          runId: "run-1",
          artifactPaths: [
            "content-posts/demo.md",
            "content-posts/demo-cover.png",
          ],
        },
      ])[0]?.artifactPaths,
    ).toEqual(["content-posts/demo.md", "content-posts/demo-cover.png"]);
  });
});
