import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import { renderTool, renderToolList } from "./ToolCallDisplay.testFixtures";

describe("ToolCallDisplay command output", () => {
  it("命令行显示已发送输入的脱敏摘要", () => {
    const { container } = renderTool({
      id: "tool-terminal-interaction",
      name: "exec_command",
      arguments: JSON.stringify({ command: "read input" }),
      status: "completed",
      result: {
        success: true,
        output: "done",
        metadata: {
          terminal_interactions: [
            { process_id: "unified-exec-1000", stdin: "sent 9 chars" },
            {
              process_id: "unified-exec-1000",
              stdin: "raw-input-must-not-render",
            },
          ],
        },
      },
      startTime: new Date("2026-03-20T11:59:58.000Z"),
      endTime: new Date("2026-03-20T11:59:59.000Z"),
    });

    expect(
      container.querySelector('[data-testid="tool-call-terminal-interactions"]')
        ?.textContent,
    ).toContain("已发送输入：sent 9 chars");
    expect(container.textContent).not.toContain("raw-input-must-not-render");
  });

  it("连续完成的命令工具应聚合成一个 work group", () => {
    const { container } = renderToolList({
      toolCalls: [
        {
          id: "tool-exec-1",
          name: "bash",
          arguments: JSON.stringify({ command: "pwd" }),
          status: "completed",
          result: { success: true, output: "/workspace\n" },
          startTime: new Date("2026-03-20T12:00:00.000Z"),
          endTime: new Date("2026-03-20T12:00:01.000Z"),
        },
        {
          id: "tool-exec-2",
          name: "bash",
          arguments: JSON.stringify({ command: "ls -la" }),
          status: "completed",
          result: { success: true, output: "file-a\nfile-b\n" },
          startTime: new Date("2026-03-20T12:00:02.000Z"),
          endTime: new Date("2026-03-20T12:00:03.000Z"),
        },
      ],
    });

    const groups = container.querySelectorAll(
      '[data-testid="tool-call-work-group"]',
    );
    expect(groups).toHaveLength(1);
    expect(container.textContent).toContain("已运行 2 条命令");
    expect(container.textContent).toContain("2");
    expect(container.textContent).toContain("pwd");
    expect(container.textContent).toContain("ls -la");

    act(() => {
      const groupToggle = groups[0]?.querySelector(
        "button",
      ) as HTMLButtonElement | null;
      groupToggle?.click();
    });

    expect(container.textContent).toContain("运行 pwd");
    expect(container.textContent).toContain("运行 ls -la");
    expect(container.textContent).not.toContain("pwd · ls -la");
  });

  it("命令结果应进入代码块渲染，而不是裸文本标题重复", () => {
    const { container } = renderTool({
      id: "tool-exec-render-1",
      name: "bash",
      arguments: JSON.stringify({ command: "ls -la" }),
      status: "completed",
      result: {
        success: true,
        output: "/tmp\nfile-a\nfile-b\nfile-c\n",
        metadata: {
          exit_code: 0,
          cwd: "/workspace",
          stdout_length: 24,
          stderr_length: 0,
          sandboxed: true,
          sandbox_type: "workspace-write",
        },
      },
      startTime: new Date("2026-03-20T12:10:00.000Z"),
      endTime: new Date("2026-03-20T12:10:01.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(container.textContent).toContain("已运行 ls -la");
    expect(container.textContent).not.toContain("已运行已运行");
    expect(container.textContent).toContain("命令摘要");
    expect(container.textContent).toContain("命令");
    expect(container.textContent).toContain("ls -la");
    expect(container.textContent).toContain("目录");
    expect(container.textContent).toContain("/workspace");
    expect(container.textContent).toContain("退出码：0");
    expect(container.textContent).toContain("stdout：24");
    expect(container.textContent).toContain("stderr：0");
    expect(container.textContent).toContain("沙箱：workspace-write");
    expect(
      container.querySelector('[data-testid="tool-call-rendered-result"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="tool-call-command-summary"]'),
    ).toBeTruthy();
    expect(container.textContent).toContain("text");
    expect(container.textContent).toContain("复制");
  });

  it("命令结果带 stdout/stderr 分流时应展示友好的输出分区", () => {
    const { container } = renderTool({
      id: "tool-exec-streams-1",
      name: "bash",
      arguments: JSON.stringify({ command: "npm test" }),
      status: "failed",
      result: {
        success: false,
        output: JSON.stringify({
          stdout: "✓ parser.test.ts\n✓ renderer.test.ts",
          stderr: "FAIL src/runtime.test.ts\nExpected status 0",
        }),
        metadata: {
          exit_code: 1,
          cwd: "/workspace/lime",
          stdout_length: 37,
          stderr_length: 42,
          sandboxed: true,
          sandbox_type: "workspace-write",
        },
      },
      startTime: new Date("2026-03-20T12:11:00.000Z"),
      endTime: new Date("2026-03-20T12:11:01.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(container.textContent).toContain("命令摘要");
    expect(container.textContent).toContain("退出码：1");
    expect(container.textContent).toContain("命令输出");
    expect(
      container.querySelector(
        '[data-testid="tool-call-command-output-streams"]',
      ),
    ).toBeTruthy();
    expect(
      container.querySelector(
        '[data-testid="tool-call-command-output-stdout"]',
      ),
    ).toBeTruthy();
    expect(
      container.querySelector(
        '[data-testid="tool-call-command-output-stderr"]',
      ),
    ).toBeTruthy();
    expect(container.textContent).toContain("✓ parser.test.ts");
    expect(container.textContent).toContain("FAIL src/runtime.test.ts");
    expect(
      container.querySelector('[data-testid="tool-call-rendered-result"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain('"stdout"');
  });

  it("Bash JSON stdout 即使包含 metadata/result 也不应被协议包络过滤吞掉", () => {
    const { container } = renderTool({
      id: "tool-exec-json-stdout-1",
      name: "bash",
      arguments: JSON.stringify({ command: "node scripts/check.mjs" }),
      status: "completed",
      result: {
        success: true,
        output: JSON.stringify({
          result: {
            ok: true,
          },
          metadata: {
            durationMs: 42,
          },
        }),
        metadata: {
          exit_code: 0,
          stdout_length: 48,
          stderr_length: 0,
        },
      },
      startTime: new Date("2026-06-21T12:08:00.000Z"),
      endTime: new Date("2026-06-21T12:08:01.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(
      container.querySelector('[data-testid="tool-call-rendered-result"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("命令摘要");
    expect(container.textContent).toContain("durationMs");
    expect(container.textContent).toContain("ok");
  });

  it("apply_patch 工具应把补丁参数渲染为文件级变更审阅", () => {
    const { container } = renderTool({
      id: "tool-apply-patch-review-1",
      name: "apply_patch",
      arguments: JSON.stringify({
        patch: [
          "*** Begin Patch",
          "*** Update File: src/components/App.tsx",
          "@@",
          '-const title = "Old";',
          '+const title = "New";',
          '+const subtitle = "Ready";',
          "*** End Patch",
        ].join("\n"),
      }),
      status: "completed",
      result: {
        success: true,
        output: "Patch applied successfully",
      },
      startTime: new Date("2026-03-20T12:11:30.000Z"),
      endTime: new Date("2026-03-20T12:11:31.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(
      container.querySelector('[data-testid="tool-call-diff-review"]'),
    ).toBeTruthy();
    expect(container.textContent).toContain("src/components/App.tsx");
    expect(container.textContent).toContain("+2");
    expect(container.textContent).toContain("-1");
    expect(container.textContent).not.toContain("变更范围");
    expect(container.textContent).toContain('const title = "Old";');
    expect(container.textContent).toContain('const subtitle = "Ready";');
    expect(
      container.querySelector('[data-testid="tool-call-rendered-result"]'),
    ).toBeNull();
  });

  it("文件级变更审阅应可作为 diff 审阅内容打开到画布", () => {
    const onFileClick = vi.fn();
    const { container } = renderTool(
      {
        id: "tool-apply-patch-review-canvas-1",
        name: "apply_patch",
        arguments: JSON.stringify({
          patch: [
            "*** Begin Patch",
            "*** Update File: src/components/App.tsx",
            "@@",
            '-const title = "Old";',
            '+const title = "New";',
            '+const subtitle = "Ready";',
            "*** End Patch",
          ].join("\n"),
        }),
        status: "completed",
        result: {
          success: true,
          output: "Patch applied successfully",
        },
        startTime: new Date("2026-03-20T12:11:30.000Z"),
        endTime: new Date("2026-03-20T12:11:31.000Z"),
      },
      { onFileClick },
    );

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    act(() => {
      const openButton = container.querySelector(
        'button[aria-label="在画布中打开变更审阅：src/components/App.tsx"]',
      ) as HTMLButtonElement | null;
      openButton?.click();
    });

    expect(onFileClick).toHaveBeenCalledTimes(1);
    expect(onFileClick.mock.calls[0]?.[0]).toBe(
      "src/components/App.tsx.diff.md",
    );
    expect(onFileClick.mock.calls[0]?.[1]).toContain(
      "# src/components/App.tsx 的变更审阅",
    );
    expect(onFileClick.mock.calls[0]?.[1]).toContain("- 状态：修改");
    expect(onFileClick.mock.calls[0]?.[1]).toContain("- +2 行");
    expect(onFileClick.mock.calls[0]?.[1]).toContain("- -1 行");
    expect(onFileClick.mock.calls[0]?.[1]).toContain("- 1 处变更");
    expect(onFileClick.mock.calls[0]?.[1]).toContain("````diff");
    expect(onFileClick.mock.calls[0]?.[1]).toContain('-const title = "Old";');
    expect(onFileClick.mock.calls[0]?.[1]).toContain('+const title = "New";');
  });

  it("长补丁应默认保留文件级预览，并支持展开完整变更", () => {
    const { container } = renderTool({
      id: "tool-apply-patch-review-long-1",
      name: "apply_patch",
      arguments: JSON.stringify({
        patch: [
          "*** Begin Patch",
          "*** Update File: src/runtime/longPatch.ts",
          "@@",
          " const stable = true;",
          '-const oldValue = "legacy";',
          '+const nextValue1 = "current";',
          '+const nextValue2 = "current";',
          '+const nextValue3 = "current";',
          '+const nextValue4 = "current";',
          '+const nextValue5 = "current";',
          '+const nextValue6 = "current";',
          '+const nextValue7 = "current";',
          '+const nextValue8 = "current";',
          '+const nextValue9 = "current";',
          '+const nextValue10 = "current";',
          "*** End Patch",
        ].join("\n"),
      }),
      status: "completed",
      result: {
        success: true,
        output: "Patch applied successfully",
      },
      startTime: new Date("2026-03-20T12:11:32.000Z"),
      endTime: new Date("2026-03-20T12:11:33.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(container.textContent).toContain("src/runtime/longPatch.ts");
    expect(container.textContent).toContain("展开其余 5 行");
    expect(container.textContent).toContain('const nextValue5 = "current";');
    expect(container.textContent).not.toContain(
      'const nextValue10 = "current";',
    );

    act(() => {
      const expandButton = Array.from(
        container.querySelectorAll("button"),
      ).find((button) => button.textContent?.includes("展开其余"));
      expandButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(container.textContent).toContain("收起完整变更");
    expect(container.textContent).toContain('const nextValue10 = "current";');
    expect(
      container.querySelector(
        '[data-testid="tool-call-diff-review-file-lines"]',
      ),
    ).toBeTruthy();
  });

  it("命令 stdout 返回 unified diff 时应同时保留输出分区并展示变更审阅", () => {
    const { container } = renderTool({
      id: "tool-exec-diff-review-1",
      name: "bash",
      arguments: JSON.stringify({ command: "git diff -- src/app.ts" }),
      status: "completed",
      result: {
        success: true,
        output: JSON.stringify({
          stdout: [
            "diff --git a/src/app.ts b/src/app.ts",
            "--- a/src/app.ts",
            "+++ b/src/app.ts",
            "@@ -1,2 +1,3 @@",
            "-const count = 1;",
            "+const count = 2;",
            "+const enabled = true;",
            ' export const name = "lime";',
          ].join("\n"),
        }),
        metadata: {
          exit_code: 0,
          stdout_length: 168,
          stderr_length: 0,
        },
      },
      startTime: new Date("2026-03-20T12:11:40.000Z"),
      endTime: new Date("2026-03-20T12:11:41.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(
      container.querySelector(
        '[data-testid="tool-call-command-output-stdout"]',
      ),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="tool-call-diff-review"]'),
    ).toBeTruthy();
    expect(container.textContent).toContain("src/app.ts");
    expect(container.textContent).toContain("+2");
    expect(container.textContent).toContain("-1");
    expect(container.textContent).toContain("const enabled = true;");
    expect(
      container.querySelector('[data-testid="tool-call-rendered-result"]'),
    ).toBeNull();
  });

  it("多文件补丁应按通用路径结构聚合变更范围", () => {
    const { container } = renderTool({
      id: "tool-apply-patch-review-scope-1",
      name: "apply_patch",
      arguments: JSON.stringify({
        patch: [
          "*** Begin Patch",
          "*** Update File: src/components/App.tsx",
          "@@",
          '-const title = "Old";',
          '+const title = "New";',
          "*** Update File: src/components/Panel.tsx",
          "@@",
          "+export const ready = true;",
          "*** Update File: lime-rs/src/app/runner.rs",
          "@@",
          '-let mode = "legacy";',
          '+let mode = "current";',
          "+let enabled = true;",
          "*** Update File: README.md",
          "@@",
          "+Lime supports code review from tool results.",
          "*** End Patch",
        ].join("\n"),
      }),
      status: "completed",
      result: {
        success: true,
        output: "Patch applied successfully",
      },
      startTime: new Date("2026-03-20T12:11:42.000Z"),
      endTime: new Date("2026-03-20T12:11:43.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    const scope = container.querySelector(
      '[data-testid="tool-call-diff-review-scope"]',
    );
    const scopeItems = Array.from(
      container.querySelectorAll(
        '[data-testid="tool-call-diff-review-scope-item"]',
      ),
    );

    expect(scope).toBeTruthy();
    expect(scope?.textContent).toContain("变更范围");
    expect(scopeItems).toHaveLength(3);
    expect(
      scopeItems.some((item) => item.textContent?.includes("src/components")),
    ).toBe(true);
    expect(
      scopeItems.some((item) => item.textContent?.includes("lime-rs/src")),
    ).toBe(true);
    expect(
      scopeItems.some((item) => item.textContent?.includes("仓库根目录")),
    ).toBe(true);
    expect(
      scopeItems.some((item) =>
        item.textContent?.includes("src/components2 个文件+2-1"),
      ),
    ).toBe(true);
    expect(
      scopeItems.some((item) =>
        item.textContent?.includes("lime-rs/src1 个文件+2-1"),
      ),
    ).toBe(true);
    expect(
      scopeItems.some((item) =>
        item.textContent?.includes("仓库根目录1 个文件+1-0"),
      ),
    ).toBe(true);
  });

  it("结果区应压缩内部元信息与长路径提示", () => {
    const { container } = renderTool({
      id: "tool-exec-render-2",
      name: "bash",
      arguments: JSON.stringify({ command: "generate-report" }),
      status: "failed",
      result: {
        success: false,
        output: "报告生成失败，请检查参数后重试。",
        metadata: {
          exit_code: 2,
          lime_offloaded: true,
          output_truncated: true,
          output_file: "exports/reports/final-result.md",
        },
      },
      startTime: new Date("2026-03-20T12:12:00.000Z"),
      endTime: new Date("2026-03-20T12:12:01.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(container.textContent).toContain("内容较长，已省略部分文本");
    expect(container.textContent).toContain("命令返回错误");
    expect(container.textContent).toContain("命令摘要");
    expect(container.textContent).toContain("退出码：2");
    expect(container.textContent).toContain("输出已截断");
    expect(container.textContent).toContain("结果文件: final-result.md");
    expect(container.textContent).not.toContain("完整输出已转存");
    expect(container.textContent).not.toContain("输出文件:");
    expect(container.textContent).not.toContain(
      "exports/reports/final-result.md",
    );
  });

  it("语义成功的非零退出码不应继续显示命令错误提示", () => {
    const { container } = renderTool({
      id: "tool-exec-render-3",
      name: "bash",
      arguments: JSON.stringify({ command: "rg missing src" }),
      status: "completed",
      result: {
        success: true,
        output: "No matches found",
        metadata: {
          exit_code: 1,
          stdout_length: 0,
          stderr_length: 0,
          reported_success: true,
        },
      },
      startTime: new Date("2026-04-14T10:00:00.000Z"),
      endTime: new Date("2026-04-14T10:00:01.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(container.textContent).toContain("No matches found");
    expect(container.textContent).not.toContain("命令返回错误");
  });

  it("正式工具卡不应额外展示原始工具名", () => {
    const { container } = renderTool({
      id: "tool-ask-user-1",
      name: "request_user_input",
      arguments: JSON.stringify({ question: "需要继续吗？" }),
      status: "completed",
      result: {
        success: true,
        output: "用户已确认继续。",
      },
      startTime: new Date("2026-04-13T10:31:00.000Z"),
      endTime: new Date("2026-04-13T10:31:01.000Z"),
    });

    expect(container.textContent).toContain("已收集 需要继续吗？");
    expect(container.textContent).not.toContain("Ask User Question");
  });
});
