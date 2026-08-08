import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { CodingWorkbenchView } from "@limecloud/agent-runtime-projection";
import type { ConfirmResponse } from "../types";
import {
  CodingWorkbenchOutputPanel,
  type CodingWorkbenchCommandProcessControls,
} from "./CodingWorkbenchOutputPanel";
import type {
  CodingWorkbenchRecoveryContext,
  CodingWorkbenchRecoverySignal,
} from "./codingWorkbenchRecovery";

vi.mock("react-i18next", async () => {
  const agentZhCN = (await import("@/i18n/resources/zh-CN/agent.json"))
    .default as Record<string, string>;

  return {
    useTranslation: () => ({
      t: (key: string, options?: Record<string, unknown>) => {
        const template = agentZhCN[key] ?? key;
        return template.replace(/{{\s*([^}]+?)\s*}}/g, (_, name: string) =>
          String(options?.[name.trim()] ?? ""),
        );
      },
    }),
  };
});

interface MountedHarness {
  container: HTMLDivElement;
  root: Root;
}

const mountedRoots: MountedHarness[] = [];

function createCodingView(
  overrides: Partial<CodingWorkbenchView> = {},
): CodingWorkbenchView {
  return {
    runtime: { status: "running" },
    mainObject: {
      id: "turn-1",
      title: "Coding Workbench",
      status: "running",
      activeCommandId: "command-1",
      activeTestRunId: "test-1",
    },
    files: [],
    changes: [],
    patches: [],
    commands: [
      {
        commandId: "command-1",
        status: "running",
        title: "npm test",
        command: "bash -lc 'npm test'",
        canonicalCommand: "npm test",
        commandSummary: "npm test",
        commandArgv: ["npm", "test"],
        commandArgvSource: "argv",
        cwd: "app",
        outputRefs: ["output://command-1"],
        preview: "running tests",
        processId: "process-1",
        executionProcessStatus: "running",
        executionSurface: "execution_process",
        sourceEventIds: ["event-command-1"],
      },
    ],
    tests: [
      {
        testRunId: "test-1",
        status: "running",
        title: "unit",
        suite: "unit",
        commandId: "command-1",
        passed: 3,
        failed: 0,
        outputRefs: [],
        sourceEventIds: ["event-test-1"],
      },
    ],
    actions: [
      {
        id: "event-action-1",
        actionId: "action-1",
        source: {
          id: "event-action-1",
          runtimeId: "session-1",
          threadId: "thread-1",
          turnId: "turn-1",
          kind: "action",
          status: "blocked",
          eventClass: "action.required",
          title: "确认执行命令",
          actionId: "action-1",
          payload: {
            actionKind: "approve-command",
            targetModule: "coding-workbench",
            command: "npm test",
            controls: ["approve", "reject"],
          },
          createdAt: "2026-06-13T00:00:00.000Z",
        },
        surface: "human-action",
        title: "确认执行命令",
        status: "blocked",
        displayStatusKey: "agent.status.actionRequired",
        resolved: false,
        actionKind: "approval",
        targetModule: "coding-workbench",
      },
    ],
    artifacts: [],
    evidence: [],
    diagnostics: [
      {
        id: "diagnostic-1",
        sourceEventId: "event-command-1",
        title: "命令失败",
        detail: "exit=1",
        status: "failed",
      },
    ],
    ui: {
      preferredTab: "outputs",
      stale: false,
    },
    ...overrides,
  };
}

function renderPanel(codingView = createCodingView()) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(<CodingWorkbenchOutputPanel codingView={codingView} />);
  });

  mountedRoots.push({ container, root });
  return container;
}

function renderPanelWithProps({
  codingView = createCodingView(),
  fileCheckpointSummary,
  submittedActionsInFlight = [],
  processControls,
  onRespondToAction,
  onSubmitRecoveryPrompt,
}: {
  codingView?: CodingWorkbenchView;
  fileCheckpointSummary?: Parameters<
    typeof CodingWorkbenchOutputPanel
  >[0]["fileCheckpointSummary"];
  submittedActionsInFlight?: Parameters<
    typeof CodingWorkbenchOutputPanel
  >[0]["submittedActionsInFlight"];
  processControls?: CodingWorkbenchCommandProcessControls;
  onRespondToAction?: (response: ConfirmResponse) => void | Promise<void>;
  onSubmitRecoveryPrompt?: Parameters<
    typeof CodingWorkbenchOutputPanel
  >[0]["onSubmitRecoveryPrompt"];
}) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(
      <CodingWorkbenchOutputPanel
        codingView={codingView}
        fileCheckpointSummary={fileCheckpointSummary}
        submittedActionsInFlight={submittedActionsInFlight}
        processControls={processControls}
        onRespondToAction={onRespondToAction}
        onSubmitRecoveryPrompt={onSubmitRecoveryPrompt}
      />,
    );
  });

  mountedRoots.push({ container, root });
  return container;
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) break;
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.clearAllMocks();
});

describe("CodingWorkbenchOutputPanel", () => {
  it("应从 CodingWorkbenchView 渲染命令、测试和待确认动作", () => {
    const container = renderPanel();

    expect(
      container.querySelector('[data-testid="coding-workbench-command"]')
        ?.textContent,
    ).toContain("npm test");
    expect(
      container.querySelector('[data-testid="coding-workbench-command"]')
        ?.textContent,
    ).toContain("bash -lc 'npm test'");
    expect(container.textContent).toContain("running tests");
    expect(
      container.querySelector('[data-testid="coding-workbench-test"]')
        ?.textContent,
    ).toContain("3 通过，0 失败");
    expect(
      container.querySelector('[data-testid="coding-workbench-action"]')
        ?.textContent,
    ).toContain("确认执行命令");
    expect(container.textContent).toContain("npm test");
    expect(container.textContent).toContain("进程 process-1");
    expect(container.textContent).toContain("状态：running");
    expect(
      container.querySelector('[data-testid="coding-workbench-diagnostic"]')
        ?.textContent,
    ).toContain("命令失败");
    expect(container.textContent).toContain("失败即停止");
    expect(container.textContent).toContain("event-command-1");
  });

  it("应为 live command 渲染后台终端终止按钮并传递 itemId", async () => {
    const processControls: CodingWorkbenchCommandProcessControls = {
      onTerminateBackgroundTerminal: vi.fn(),
    };
    const container = renderPanelWithProps({ processControls });

    expect(
      container.querySelector(
        '[data-testid="coding-workbench-command-process-controls"]',
      ),
    ).not.toBeNull();

    const terminate = container.querySelector(
      'button[aria-label="终止进程 process-1"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      terminate?.click();
    });

    expect(processControls.onTerminateBackgroundTerminal).toHaveBeenCalledWith(
      "command-1",
    );
  });

  it("已结束或缺少 processId 的命令不应显示进程控制按钮", () => {
    const processControls: CodingWorkbenchCommandProcessControls = {
      onTerminateBackgroundTerminal: vi.fn(),
    };
    const terminalCommand = {
      ...createCodingView().commands[0],
      commandId: "command-terminal",
      processId: "process-terminal",
      executionProcessStatus: "exited",
    };
    const nonProcessCommand = {
      ...createCodingView().commands[0],
      commandId: "command-no-process",
      processId: undefined,
      executionProcessStatus: "running",
    };
    const container = renderPanelWithProps({
      processControls,
      codingView: createCodingView({
        commands: [terminalCommand, nonProcessCommand],
        tests: [],
        actions: [],
        diagnostics: [],
      }),
    });

    expect(
      container.querySelector(
        '[data-testid="coding-workbench-command-process-controls"]',
      ),
    ).toBeNull();
  });

  it("应允许从 coding action projection 直接提交命令确认", () => {
    const onRespondToAction =
      vi.fn<(response: ConfirmResponse) => void | Promise<void>>();
    const container = renderPanelWithProps({ onRespondToAction });
    const buttons = Array.from(container.querySelectorAll("button"));

    act(() => {
      buttons[0]?.click();
    });
    expect(onRespondToAction).toHaveBeenCalledWith({
      requestId: "action-1",
      actionType: "tool_confirmation",
      confirmed: true,
      response: "approved",
    });

    act(() => {
      buttons[1]?.click();
    });
    expect(onRespondToAction).toHaveBeenLastCalledWith({
      requestId: "action-1",
      actionType: "tool_confirmation",
      confirmed: false,
      response: "rejected",
    });
  });

  it("提交中的 action 应禁用确认按钮并显示提交中", () => {
    const container = renderPanelWithProps({
      submittedActionsInFlight: [
        {
          requestId: "action-1",
          actionType: "tool_confirmation",
          status: "submitted",
        },
      ],
      onRespondToAction: vi.fn(),
    });
    const buttons = Array.from(container.querySelectorAll("button"));

    expect(container.textContent).toContain("提交中");
    expect((buttons[0] as HTMLButtonElement | undefined)?.disabled).toBe(true);
    expect((buttons[1] as HTMLButtonElement | undefined)?.disabled).toBe(true);
  });

  it("无法映射成 current action response 的动作不应伪造提交按钮", () => {
    const onRespondToAction = vi.fn();
    const codingView = createCodingView({
      actions: [
        {
          id: "event-action-unknown",
          actionId: "action-unknown",
          source: {
            id: "event-action-unknown",
            kind: "action",
            status: "blocked",
            eventClass: "action.required",
            title: "等待人工处理",
            actionId: "action-unknown",
            payload: {
              actionKind: "open-settings",
              targetModule: "coding-workbench",
            },
            createdAt: "2026-06-13T00:00:00.000Z",
          },
          surface: "human-action",
          title: "等待人工处理",
          status: "blocked",
          displayStatusKey: "agent.status.actionRequired",
          resolved: false,
          actionKind: "open-settings",
          targetModule: "coding-workbench",
        },
      ],
    });

    const container = renderPanelWithProps({ codingView, onRespondToAction });

    expect(container.querySelector("button")).toBeNull();
    expect(container.textContent).toContain(
      "请在对话里的待处理请求中继续处理。",
    );
    expect(onRespondToAction).not.toHaveBeenCalled();
  });

  it("失败输出应显示继续修复入口并提交包含失败事实的 prompt", async () => {
    const onSubmitRecoveryPrompt = vi.fn();
    const codingView = createCodingView({
      changes: [
        {
          id: "change-1",
          path: "src/App.tsx",
          status: "completed",
          changeKind: "modified",
          artifactRefs: [],
          checkpointRef: "checkpoint-app",
          preview: "render app",
          sourceEventId: "event-change-1",
        },
      ],
      patches: [
        {
          patchId: "patch-1",
          status: "failed",
          title: "apply patch",
          path: "src/App.tsx",
          toolCallId: "tool-patch-1",
          diffRef: "diff://patch-1",
          failureCategory: "patch_failed",
          sourceEventIds: ["event-patch-1"],
        },
      ],
      commands: [
        {
          commandId: "command-1",
          status: "failed",
          title: "npm test",
          command: "npm test",
          cwd: "app",
          exitCode: 1,
          outputRefs: ["output://command-1"],
          preview: "App.test.tsx failed",
          sourceEventIds: ["event-command-1"],
        },
      ],
      tests: [
        {
          testRunId: "test-1",
          status: "failed",
          title: "unit",
          suite: "unit",
          commandId: "command-1",
          result: "failed",
          passed: 3,
          failed: 1,
          outputRefs: [],
          failureCategory: "assertion_failed",
          sourceEventIds: ["event-test-1"],
        },
      ],
      actions: [],
      diagnostics: [
        {
          id: "diagnostic-1",
          sourceEventId: "event-command-1",
          title: "命令失败",
          detail: "exit=1",
          status: "failed",
        },
      ],
    });

    const container = renderPanelWithProps({
      codingView,
      fileCheckpointSummary: {
        count: 1,
        latest_checkpoint: {
          checkpoint_id: "checkpoint-1",
          turn_id: "turn-1",
          path: "src/App.tsx",
          source: "runtime",
          updated_at: "2026-06-14T00:00:00.000Z",
          validation_issue_count: 0,
        },
      },
      onSubmitRecoveryPrompt,
    });

    expect(
      container.querySelector('[data-testid="coding-workbench-recovery"]')
        ?.textContent,
    ).toContain("需要继续修复");
    expect(container.textContent).toContain("失败命令");
    expect(container.textContent).toContain("失败测试");
    expect(container.textContent).toContain("失败补丁");
    expect(container.textContent).toContain("src/App.tsx");

    const recoveryButton = Array.from(
      container.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("继续修复"));
    await act(async () => {
      recoveryButton?.click();
    });

    expect(onSubmitRecoveryPrompt).toHaveBeenCalledWith(
      expect.stringContaining("请继续修复本轮编程任务中的失败输出。"),
      expect.objectContaining({
        schemaVersion: "coding-workbench-recovery/v1",
        failureKind: "patch",
        sourceIds: {
          commandId: "command-1",
          testRunId: "test-1",
          patchId: "patch-1",
          toolCallId: "tool-patch-1",
        },
        refs: expect.objectContaining({
          outputRefs: ["output://command-1"],
          diffRef: "diff://patch-1",
        }),
        relatedFiles: ["src/App.tsx"],
        latestCheckpointPath: "src/App.tsx",
      }),
    );
    const prompt = onSubmitRecoveryPrompt.mock.calls[0]?.[0] as string;
    const context = onSubmitRecoveryPrompt.mock.calls[0]?.[1] as
      | CodingWorkbenchRecoveryContext
      | undefined;
    expect(prompt).toContain("npm test");
    expect(prompt).toContain("unit");
    expect(prompt).toContain("命令失败");
    expect(prompt).toContain("App.test.tsx failed");
    expect(prompt).toContain("相关文件: src/App.tsx");
    expect(prompt).toContain("最近文件快照: src/App.tsx");
    expect(
      context?.signals.map(
        (signal: CodingWorkbenchRecoverySignal) => signal.kind,
      ),
    ).toEqual(["command", "test", "patch", "diagnostic"]);
  });

  it("没有失败事实时不应显示继续修复入口", () => {
    const container = renderPanel(
      createCodingView({
        patches: [],
        commands: [
          {
            commandId: "command-1",
            status: "completed",
            title: "npm test",
            command: "npm test",
            cwd: "app",
            exitCode: 0,
            outputRefs: [],
            preview: "passed",
            sourceEventIds: ["event-command-1"],
          },
        ],
        tests: [
          {
            testRunId: "test-1",
            status: "completed",
            title: "unit",
            suite: "unit",
            commandId: "command-1",
            result: "passed",
            passed: 4,
            failed: 0,
            outputRefs: [],
            sourceEventIds: ["event-test-1"],
          },
        ],
        actions: [],
        diagnostics: [],
      }),
    );

    expect(
      container.querySelector('[data-testid="coding-workbench-recovery"]'),
    ).toBeNull();
  });

  it("没有 projection 输出时应渲染稳定空态", () => {
    const container = renderPanel(
      createCodingView({
        commands: [],
        tests: [],
        actions: [],
        diagnostics: [],
      }),
    );

    expect(
      container.querySelector(
        '[data-testid="coding-workbench-output-projection"]',
      )?.textContent,
    ).toContain("本轮还没有可展示的输出。");
  });
});
