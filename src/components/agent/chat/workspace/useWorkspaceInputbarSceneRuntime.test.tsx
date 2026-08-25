import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import { useWorkspaceInputbarSceneRuntime } from "./useWorkspaceInputbarSceneRuntime";

const mockInputbarRender = vi.hoisted(() => vi.fn());
const mockPendingInteractionRuntime = vi.hoisted(() => ({
  interactions: [] as unknown[],
  respond: vi.fn(() => ({ accepted: true })),
}));
const mockStrictReviewRuntime = vi.hoisted(() => ({
  status: null as null | {
    startedAtMs: number;
    threadId: string;
    turnId: string;
  },
}));

vi.mock("../components/Inputbar", () => ({
  Inputbar: (props: { overlayAccessory?: React.ReactNode }) => {
    mockInputbarRender(props);
    return <div data-testid="inputbar-mock">{props.overlayAccessory}</div>;
  },
}));

vi.mock("../hooks/usePendingInteractions", () => ({
  usePendingInteractions: () => mockPendingInteractionRuntime,
}));

vi.mock("../hooks/useStrictReviewStatus", () => ({
  useStrictReviewStatus: () => mockStrictReviewRuntime.status,
}));

vi.mock("./WorkspaceHarnessDialogs", () => ({
  GeneralWorkbenchDialogSection: () => (
    <div data-testid="general-workbench-dialog-mock" />
  ),
}));

vi.mock("./knowledge/useWorkspaceKnowledgeRuntime", () => ({
  useWorkspaceKnowledgeRuntime: () => ({
    knowledgePackSelection: null,
    knowledgePackOptions: [],
    onToggleKnowledgePack: vi.fn(),
    onSelectKnowledgePack: vi.fn(),
    onToggleKnowledgeCompanionPack: vi.fn(),
    onStartKnowledgeOrganize: vi.fn(),
    onManageKnowledgePacks: vi.fn(),
    onImportPathReferenceAsKnowledge: vi.fn(),
    onImportTextAsKnowledge: vi.fn(),
  }),
}));

type HookProps = Parameters<typeof useWorkspaceInputbarSceneRuntime>[0];
type HarnessPanelBaseProps = HookProps["generalWorkbenchHarnessPanelBaseProps"];
type HookPropsOverrides = Partial<
  Omit<HookProps, "generalWorkbenchHarnessPanelBaseProps">
> &
  Partial<HarnessPanelBaseProps> & {
    generalWorkbenchHarnessPanelBaseProps?: Partial<HarnessPanelBaseProps>;
  };

const mountedRoots: Array<{ container: HTMLDivElement; root: Root }> = [];

function createDefaultHarnessPanelBaseProps(
  noop: ReturnType<typeof vi.fn>,
  overrides: Partial<HarnessPanelBaseProps> = {},
): HarnessPanelBaseProps {
  return {
    environment: {
      skillsCount: 0,
      skillNames: [],
      memorySignals: [],
      contextItemsCount: 0,
      activeContextCount: 0,
      contextItemNames: [],
      contextEnabled: false,
    },
    canonicalChildren: [],
    threadRead: null,
    turns: [],
    threadItems: [],
    currentTurnId: null,
    pendingActions: [],
    submittedActionsInFlight: [],
    onRespondToAction: noop,
    messages: [],
    canInterrupt: false,
    onInterruptCurrentTurn: noop,
    onReplayPendingRequest: noop,
    onObjectiveChanged: noop,
    onManageProviders: noop,
    onOpenExecutionPolicySettings: noop,
    diagnosticRuntimeContext: {
      sessionId: null,
      workspaceId: "project-1",
      workingDir: "/tmp/project-1",
      providerType: "openai",
      model: "gpt-5",
      executionStrategy: "react",
      activeTheme: "general",
    },
    toolInventory: null,
    toolInventoryLoading: false,
    toolInventoryError: null,
    onRefreshToolInventory: noop,
    onOpenSubagentSession: noop,
    onLoadFilePreview: noop,
    onOpenFile: noop,
    ...overrides,
  };
}

function createDefaultProps(overrides: HookPropsOverrides = {}): HookProps {
  const noop = vi.fn();
  const {
    generalWorkbenchHarnessPanelBaseProps,
    pendingActions,
    submittedActionsInFlight,
    onRespondToAction,
    ...hookOverrides
  } = overrides;

  return {
    setMentionedCharacters: noop,
    isThemeWorkbench: true,
    sessionId: "session-1",
    generalWorkbenchHarnessPanelBaseProps: createDefaultHarnessPanelBaseProps(
      noop,
      {
        ...generalWorkbenchHarnessPanelBaseProps,
        ...(pendingActions ? { pendingActions } : {}),
        ...(submittedActionsInFlight ? { submittedActionsInFlight } : {}),
        ...(onRespondToAction ? { onRespondToAction } : {}),
      },
    ),
    currentSessionTitle: "当前会话",
    handleStopSending: noop,
    input: "",
    setInput: noop,
    currentGate: null,
    generalWorkbenchWorkflowSteps: [],
    steps: [],
    workflowRunState: "idle",
    handleSend: vi.fn().mockResolvedValue(true),
    isPreparingSend: false,
    isSending: false,
    providerType: "openai",
    setProviderType: noop,
    model: "gpt-5",
    setModel: noop,
    reasoningEffort: "",
    setReasoningEffort: noop,
    sessionExecutionRuntime: null,
    projectId: "project-1",
    projectRootPath: "/tmp/project-1",
    openedProjects: [],
    accessMode: "default" as any,
    setAccessMode: noop,
    activeTheme: "general",
    navigationActions: {
      handleManageProviders: noop,
      handleOpenExecutionPolicySettings: noop,
      handleOpenRuntimeMemoryWorkbench: noop,
      handleOpenKnowledgeManagement: noop,
      handleProjectChange: noop,
    },
    characters: [],
    skills: [],
    serviceSkills: [],
    serviceSkillGroups: [],
    skillsLoading: false,
    onSelectServiceSkill: noop,
    initialInputCapability: undefined,
    initialKnowledgePackSelection: undefined,
    setChatToolPreferences: noop,
    handleNavigateToSkillSettings: noop,
    handleRefreshSkills: noop,
    soulArtifactVoiceGenerationBrief: null,
    soulArtifactVoiceEnabledForTurn: true,
    onSoulArtifactVoiceEnabledForTurnChange: noop,
    generalWorkbenchEntryPrompt: null,
    handleRestartGeneralWorkbenchEntryPrompt: noop,
    handleContinueGeneralWorkbenchEntryPrompt: noop,
    generalWorkbenchEnabled: false,
    harnessPanelVisible: false,
    setHarnessPanelVisible: noop,
    harnessState: {
      runtimeStatus: null,
      pendingApprovals: [],
      latestContextTrace: [],
      plan: {
        phase: "idle",
        items: [],
      },
      activity: {
        planning: 0,
        filesystem: 0,
        execution: 0,
        web: 0,
        skills: 0,
        delegation: 0,
      },
      delegatedTasks: [],
      outputSignals: [],
      activeFileWrites: [],
      recentFileEvents: [],
      hasSignals: false,
    },
    mappedTheme: "general",
    activeRuntimeStatusTitle: null,
    chatToolPreferences: {
      task: false,
      subagent: false,
    },
    defaultCuratedTaskReferenceMemoryIds: [],
    defaultCuratedTaskReferenceEntries: [],
    pathReferences: [],
    onAddPathReferences: noop,
    onRemovePathReference: noop,
    onClearPathReferences: noop,
    fileManagerOpen: false,
    onToggleFileManager: noop,
    inputCompletionEnabled: true,
    ...hookOverrides,
  };
}

function renderHookNode(props: HookProps): HTMLDivElement {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  function Probe() {
    const runtime = useWorkspaceInputbarSceneRuntime(props);
    return <>{runtime.inputbarNode}</>;
  }

  act(() => {
    root.render(<Probe />);
  });

  mountedRoots.push({ container, root });
  return container;
}

function getLatestInputbarProps(): {
  disabled?: boolean;
  disabledPlaceholder?: string;
  isLoading?: boolean;
  inputRestoreRequest?: unknown;
  onInputRestoreRequestHandled?: (requestId: string) => void;
  onSend?: (payload?: {
    images?: unknown[];
    textOverride?: string;
    sendOptions?: {
      requestMetadata?: Record<string, unknown>;
      targetSessionId?: string;
      skipSessionRestore?: boolean;
    };
  }) => void | Promise<boolean> | boolean;
  toolStates?: Record<string, boolean>;
  onToolStatesChange?: (states: Record<string, boolean>) => void;
} {
  const latestCall = mockInputbarRender.mock.calls.at(-1);
  expect(latestCall).toBeTruthy();
  return latestCall?.[0] as {
    disabled?: boolean;
    disabledPlaceholder?: string;
    isLoading?: boolean;
    inputRestoreRequest?: unknown;
    onInputRestoreRequestHandled?: (requestId: string) => void;
    onSend?: (payload?: {
      images?: unknown[];
      textOverride?: string;
      sendOptions?: {
        requestMetadata?: Record<string, unknown>;
        targetSessionId?: string;
        skipSessionRestore?: boolean;
      };
    }) => void | Promise<boolean> | boolean;
    toolStates?: Record<string, boolean>;
    onToolStatesChange?: (states: Record<string, boolean>) => void;
  };
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  mockPendingInteractionRuntime.interactions = [];
  mockStrictReviewRuntime.status = null;
  await changeLimeLocale("zh-CN");
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) {
      break;
    }
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.clearAllMocks();
});

describe("useWorkspaceInputbarSceneRuntime", () => {
  it("strict review 状态应位于 Composer 上方且不替换 Inputbar", () => {
    mockStrictReviewRuntime.status = {
      startedAtMs: 1_783_814_400_100,
      threadId: "thread-1",
      turnId: "turn-1",
    };
    const container = renderHookNode(
      createDefaultProps({ threadId: "thread-1" }),
    );

    const status = container.querySelector(
      '[data-testid="strict-review-status"]',
    );
    const inputbar = container.querySelector('[data-testid="inputbar-mock"]');
    expect(status).not.toBeNull();
    expect(inputbar).not.toBeNull();
    expect(
      Boolean(
        status && inputbar && status.compareDocumentPosition(inputbar) & 4,
      ),
    ).toBe(true);
  });

  it("typed MCP interaction 应位于 Composer 上方且不替换 Inputbar", () => {
    mockPendingInteractionRuntime.interactions = [
      {
        id: "mcp_elicitation:thread-1:turn-1:1",
        thread_id: "thread-1",
        turn_id: "turn-1",
        kind: "mcp_elicitation",
        status: "pending",
        payload: {
          message: "请确认发布参数",
          requestedSchema: {
            type: "object",
            properties: { confirmed: { type: "boolean" } },
          },
          serverName: "release-tools",
        },
      },
    ];
    const container = renderHookNode(
      createDefaultProps({ threadId: "thread-1" }),
    );

    const layer = container.querySelector(
      '[data-testid="pending-interaction-layer"]',
    );
    const inputbar = container.querySelector('[data-testid="inputbar-mock"]');
    expect(layer).not.toBeNull();
    expect(inputbar).not.toBeNull();
    expect(
      Boolean(layer && inputbar && layer.compareDocumentPosition(inputbar) & 4),
    ).toBe(true);
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    expect(
      container.querySelector('[data-testid="inputbar-approval-replacement"]'),
    ).toBeNull();
  });

  it("Inputbar 发送应绑定当前会话，避免落到旧 active session", async () => {
    const handleSend = vi.fn().mockResolvedValue(true);
    renderHookNode(
      createDefaultProps({
        sessionId: "session-current-inputbar",
        handleSend,
      }),
    );

    await act(async () => {
      await getLatestInputbarProps().onSend?.({
        textOverride: "整理今天的国际新闻",
        sendOptions: {
          requestMetadata: {
            harness: {
              scenario: "home-hotpath",
            },
          },
        },
      });
    });

    expect(handleSend).toHaveBeenCalledWith(
      undefined,
      undefined,
      undefined,
      "整理今天的国际新闻",
      "react",
      undefined,
      expect.objectContaining({
        requestMetadata: {
          harness: {
            scenario: "home-hotpath",
          },
        },
        targetSessionId: "session-current-inputbar",
      }),
    );
  });

  it("Inputbar 显式 targetSessionId 应优先于当前会话", async () => {
    const handleSend = vi.fn().mockResolvedValue(true);
    renderHookNode(
      createDefaultProps({
        sessionId: "session-current-inputbar",
        handleSend,
      }),
    );

    await act(async () => {
      await getLatestInputbarProps().onSend?.({
        textOverride: "继续处理草稿",
        sendOptions: {
          targetSessionId: "session-explicit-target",
          skipSessionRestore: true,
        },
      });
    });

    expect(handleSend).toHaveBeenCalledWith(
      undefined,
      undefined,
      undefined,
      "继续处理草稿",
      "react",
      undefined,
      expect.objectContaining({
        targetSessionId: "session-explicit-target",
        skipSessionRestore: true,
      }),
    );
  });

  it("有已保存创作声线时应在输入区显示本轮开关并响应切换", () => {
    const onSoulArtifactVoiceEnabledForTurnChange = vi.fn();
    const container = renderHookNode(
      createDefaultProps({
        soulArtifactVoiceGenerationBrief: {
          voice_source: "brand_voice",
          brand_voice_id: "brand-voice-1",
        },
        soulArtifactVoiceEnabledForTurn: true,
        onSoulArtifactVoiceEnabledForTurnChange,
      }),
    );

    expect(
      container.querySelector('[data-testid="soul-artifact-voice-turn-toggle"]')
        ?.textContent,
    ).toContain("创作声线");
    expect(container.textContent).toContain("本轮使用");

    act(() => {
      (
        container.querySelector(
          '[data-testid="soul-artifact-voice-turn-switch"]',
        ) as HTMLButtonElement
      ).click();
    });

    expect(onSoulArtifactVoiceEnabledForTurnChange).toHaveBeenCalledWith(false);
  });

  it("没有已保存创作声线时不应显示本轮开关", () => {
    const container = renderHookNode(createDefaultProps());

    expect(
      container.querySelector(
        '[data-testid="soul-artifact-voice-turn-toggle"]',
      ),
    ).toBeNull();
  });

  it("应在 Plan 确认态替换底部输入框而不是叠加渲染", () => {
    const container = renderHookNode(
      createDefaultProps({
        planDecisionAccessory: (
          <div data-testid="plan-composer-decision-panel">计划确认</div>
        ),
      }),
    );

    expect(
      container.querySelector('[data-testid="plan-composer-decision-panel"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("计划确认");
    expect(container.querySelector('[data-testid="inputbar-mock"]')).toBeNull();
  });

  it("Plan 确认态与 pending approval 同时存在时应优先展示 approval", () => {
    const container = renderHookNode(
      createDefaultProps({
        planDecisionAccessory: (
          <div data-testid="plan-composer-decision-panel">计划确认</div>
        ),
        pendingActions: [
          {
            requestId: "approval-over-plan-1",
            actionType: "tool_confirmation",
            status: "pending",
            toolName: "functions.exec_command",
            prompt: "允许执行当前命令？",
          },
        ],
      }),
    );

    expect(
      container.querySelector('[data-testid="inputbar-approval-replacement"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="inputbar-approval-prompt"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="plan-composer-decision-panel"]'),
    ).toBeNull();
    expect(container.querySelector('[data-testid="inputbar-mock"]')).toBeNull();
  });

  it("pending approval 应在输入区替换普通输入框并提交 tool_confirmation 响应", async () => {
    const onRespondToAction = vi.fn().mockResolvedValue(undefined);
    const container = renderHookNode(
      createDefaultProps({
        pendingActions: [
          {
            requestId: "approval-inputbar-1",
            actionType: "tool_confirmation",
            status: "pending",
            toolName: "functions.exec_command",
            prompt: "允许执行当前命令？",
            arguments: {
              command: "npm test -- --runInBand",
              cwd: "/tmp/project-1",
            },
          },
        ],
        onRespondToAction,
      }),
    );

    expect(
      container.querySelector('[data-testid="inputbar-approval-replacement"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="inputbar-approval-prompt"]'),
    ).not.toBeNull();
    expect(container.querySelector('[data-testid="inputbar-mock"]')).toBeNull();
    expect(container.textContent).toContain("允许执行当前命令？");
    expect(container.textContent).not.toContain("functions.exec_command");
    expect(container.textContent).not.toContain("npm test -- --runInBand");

    const allowButton = container.querySelector(
      'button[data-decision="allow_once"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      allowButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onRespondToAction).toHaveBeenCalledWith({
      requestId: "approval-inputbar-1",
      decision: "allow_once",
      response: "允许本次工具操作",
      actionType: "tool_confirmation",
    });
  });

  it("pending approval 应只展示后端声明可用的 decision 动作", async () => {
    const onRespondToAction = vi.fn().mockResolvedValue(undefined);
    const container = renderHookNode(
      createDefaultProps({
        pendingActions: [
          {
            requestId: "approval-cancel-only-1",
            actionType: "tool_confirmation",
            status: "pending",
            toolName: "functions.exec_command",
            prompt: "允许执行高风险命令？",
            availableDecisions: ["decline", "cancel"],
          },
        ],
        onRespondToAction,
      }),
    );

    expect(
      container.querySelector('button[data-decision="allow_once"]'),
    ).toBeNull();
    expect(
      container.querySelector('button[data-decision="allow_for_session"]'),
    ).toBeNull();
    expect(
      container.querySelector('button[data-decision="decline"]'),
    ).not.toBeNull();
    const cancelButton = container.querySelector(
      'button[data-decision="cancel"]',
    ) as HTMLButtonElement | null;
    expect(cancelButton).not.toBeNull();

    await act(async () => {
      cancelButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onRespondToAction).toHaveBeenCalledWith({
      requestId: "approval-cancel-only-1",
      decision: "cancel",
      response: "用户取消了当前任务",
      actionType: "tool_confirmation",
    });
  });

  it("已提交中的 approval 不应继续占用输入区替换槽", () => {
    const container = renderHookNode(
      createDefaultProps({
        pendingActions: [
          {
            requestId: "approval-submitted-1",
            actionType: "tool_confirmation",
            status: "pending",
            toolName: "Bash",
            prompt: "允许执行当前命令？",
          },
        ],
        submittedActionsInFlight: [
          {
            requestId: "approval-submitted-1",
            actionType: "tool_confirmation",
            status: "submitted",
          },
        ],
      }),
    );

    expect(
      container.querySelector('[data-testid="inputbar-approval-prompt"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="inputbar-mock"]'),
    ).not.toBeNull();
  });

  it("approval 提交中释放输入区后应回到 Plan 确认态且不误发 approval response", () => {
    const onPlanContinue = vi.fn();
    const onRespondToAction = vi.fn();
    const container = renderHookNode(
      createDefaultProps({
        planDecisionAccessory: (
          <button
            type="button"
            data-testid="plan-composer-continue"
            onClick={onPlanContinue}
          >
            确认计划
          </button>
        ),
        pendingActions: [
          {
            requestId: "approval-submitted-plan-1",
            actionType: "tool_confirmation",
            status: "pending",
            toolName: "Bash",
            prompt: "允许执行当前命令？",
          },
        ],
        submittedActionsInFlight: [
          {
            requestId: "approval-submitted-plan-1",
            actionType: "tool_confirmation",
            status: "submitted",
          },
        ],
        onRespondToAction,
      }),
    );

    expect(
      container.querySelector('[data-testid="inputbar-approval-prompt"]'),
    ).toBeNull();
    expect(
      container.querySelector(
        '[data-testid="plan-decision-inputbar-replacement"]',
      ),
    ).not.toBeNull();
    const continueButton = container.querySelector(
      '[data-testid="plan-composer-continue"]',
    ) as HTMLButtonElement | null;

    act(() => {
      continueButton?.click();
    });

    expect(onPlanContinue).toHaveBeenCalledTimes(1);
    expect(onRespondToAction).not.toHaveBeenCalled();
  });

  it("非 Plan 决策态仍应把附加面板作为输入区 overlay accessory 渲染", () => {
    const container = renderHookNode(
      createDefaultProps({
        generalWorkbenchEntryPrompt: {
          kind: "resume",
          signature: "resume:continue-last-work",
          title: "继续上次工作",
          description: "补充上下文后继续。",
          actionLabel: "继续",
          prompt: "继续上次工作",
        },
      }),
    );

    expect(
      container.querySelector('[data-testid="inputbar-mock"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("继续上次工作");
  });

  it("应把工作区 task 偏好映射为 Inputbar plan 受控状态", () => {
    const setChatToolPreferences = vi.fn();
    renderHookNode(
      createDefaultProps({
        chatToolPreferences: {
          task: true,
          subagent: false,
        },
        setChatToolPreferences,
      }),
    );

    const inputbarProps = getLatestInputbarProps();

    expect(inputbarProps.toolStates).toMatchObject({
      plan: true,
      subagent: false,
    });

    act(() => {
      inputbarProps.onToolStatesChange?.({
        plan: false,
      });
    });

    const updater = setChatToolPreferences.mock.calls[0]?.[0] as (previous: {
      task: boolean;
      subagent: boolean;
    }) => { task: boolean; subagent: boolean };
    expect(updater({ task: true, subagent: true })).toEqual({
      task: false,
      subagent: true,
    });
  });

  it("应把工作区 goal 会话态映射为 Inputbar objective 受控状态", () => {
    renderHookNode(
      createDefaultProps({
        objectiveEnabled: true,
      }),
    );

    expect(getLatestInputbarProps().toolStates).toMatchObject({
      objective: true,
    });
  });

  it("Inputbar goal 变更应只更新 goal 会话态，不写入 ChatToolPreferences", () => {
    const setChatToolPreferences = vi.fn();
    const onObjectiveEnabledChange = vi.fn();
    renderHookNode(
      createDefaultProps({
        setChatToolPreferences,
        onObjectiveEnabledChange,
      }),
    );

    act(() => {
      getLatestInputbarProps().onToolStatesChange?.({
        objective: true,
      });
    });

    expect(setChatToolPreferences).not.toHaveBeenCalled();
    expect(onObjectiveEnabledChange).toHaveBeenCalledWith(true);
  });

  it("任务中心 detached 会话没有 projectId 时仍应允许继续输入", () => {
    renderHookNode(
      createDefaultProps({
        contextVariant: "task-center",
        projectId: null,
        isPreparingSend: false,
      }),
    );

    expect(getLatestInputbarProps().disabled).toBe(false);
  });

  it("会话恢复中应禁用任务中心输入，避免消息落到新会话", () => {
    renderHookNode(
      createDefaultProps({
        contextVariant: "task-center",
        projectId: null,
        isPreparingSend: false,
        isSessionRestoring: true,
      }),
    );

    expect(getLatestInputbarProps().disabled).toBe(true);
  });

  it("parent-owned thread 应保留草稿并拒绝发送与设置变更", async () => {
    const handleSend = vi.fn().mockResolvedValue(true);
    const setChatToolPreferences = vi.fn();
    renderHookNode(
      createDefaultProps({
        canAcceptDirectInput: false,
        handleSend,
        setChatToolPreferences,
      }),
    );

    const inputbarProps = getLatestInputbarProps();
    expect(inputbarProps.disabled).toBe(true);
    expect(inputbarProps.disabledPlaceholder).toBe(
      "此子线程由父线程管理，无法直接输入",
    );

    expect(inputbarProps.onSend?.({ textOverride: "不得直接提交" })).toBe(
      false,
    );
    act(() => {
      inputbarProps.onToolStatesChange?.({ plan: true });
    });

    expect(handleSend).not.toHaveBeenCalled();
    expect(setChatToolPreferences).not.toHaveBeenCalled();
  });

  it("普通会话没有 projectId 但已有 sessionId 时仍应允许继续输入", () => {
    renderHookNode(
      createDefaultProps({
        contextVariant: "default",
        projectId: null,
        sessionId: "session-detached",
        isPreparingSend: false,
      }),
    );

    expect(getLatestInputbarProps().disabled).toBe(false);
  });

  it("普通工作区没有 projectId 且没有 sessionId 时仍应禁用输入", () => {
    renderHookNode(
      createDefaultProps({
        contextVariant: "default",
        projectId: null,
        sessionId: null,
        isPreparingSend: false,
      }),
    );

    expect(getLatestInputbarProps().disabled).toBe(true);
  });

  it("应把中断输入恢复请求透传给 inline Inputbar", () => {
    const onInputRestoreRequestHandled = vi.fn();
    const inputRestoreRequest = {
      requestId: "restore-1",
      reason: "thinking_only_cancelled_turn" as const,
      draft: {
        text: "恢复这段输入",
        images: [],
        pathReferences: [],
        inputCapabilityRoute: null,
      },
    };

    renderHookNode(
      createDefaultProps({
        inputRestoreRequest,
        onInputRestoreRequestHandled,
      }),
    );

    expect(getLatestInputbarProps().inputRestoreRequest).toBe(
      inputRestoreRequest,
    );
    getLatestInputbarProps().onInputRestoreRequestHandled?.("restore-1");
    expect(onInputRestoreRequestHandled).toHaveBeenCalledWith("restore-1");
  });
});
