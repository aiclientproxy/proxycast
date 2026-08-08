import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  useWorkspaceConversationLandingSessionRuntime,
  useWorkspaceConversationLandingSurfaceRuntime,
} from "./useWorkspaceConversationLandingSurfaceRuntime";

type HookProps = Parameters<
  typeof useWorkspaceConversationLandingSurfaceRuntime
>[0];

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

function createDefaultProps(overrides: Partial<HookProps> = {}): HookProps {
  return {
    accessMode: "default",
    activeTheme: "general",
    artifacts: [],
    browserAssistLoading: false,
    chatToolPreferences: {
      task: false,
      subagent: false,
    },
    contentId: null,
    creationMode: "guided",
    creationReplaySurface: null,
    emptyStateDisabled: false,
    emptyStateIsLoading: false,
    emptyStateSendOnPointerDown: true,
    entryBannerMessage: undefined,
    entryBannerVisible: false,
    generalCanvasContent: "",
    handleSendFromEmptyState: vi.fn(),
    input: "",
    inputbarScene: {
      runtimeToolAvailability: undefined,
      knowledgePackSelection: undefined,
      knowledgePackOptions: [],
      onToggleKnowledgePack: vi.fn(),
      onSelectKnowledgePack: vi.fn(),
      onToggleKnowledgeCompanionPack: vi.fn(),
      onStartKnowledgeOrganize: vi.fn(),
      onManageKnowledgePacks: vi.fn(),
      onImportPathReferenceAsKnowledge: vi.fn(),
    },
    lockTheme: false,
    model: "mock-model",
    onDismissEntryBanner: vi.fn(),
    projectCharacters: [],
    projectId: "project-1",
    providerType: "mock-provider",
    resolvedCanvasState: null,
    serviceSkillGroups: [],
    serviceSkills: [],
    setActiveTheme: vi.fn(),
    setChatToolPreferences: vi.fn(),
    setCreationMode: vi.fn(),
    setInput: vi.fn(),
    setModel: vi.fn(),
    setProviderType: vi.fn(),
    skills: [],
    skillsLoading: false,
    selectedText: "",
    ...overrides,
  };
}

function renderHook(props: HookProps) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latestValue: ReturnType<
    typeof useWorkspaceConversationLandingSurfaceRuntime
  > | null = null;

  function Probe(currentProps: HookProps) {
    latestValue = useWorkspaceConversationLandingSurfaceRuntime(currentProps);
    return null;
  }

  act(() => {
    root.render(<Probe {...props} />);
  });
  mountedRoots.push({ root, container });

  return {
    getValue: () => {
      if (!latestValue) {
        throw new Error("hook 尚未初始化");
      }
      return latestValue;
    },
  };
}

type SessionHookProps = Parameters<
  typeof useWorkspaceConversationLandingSessionRuntime
>[0];

function renderSessionHook(props: SessionHookProps) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latestValue: ReturnType<
    typeof useWorkspaceConversationLandingSessionRuntime
  > | null = null;

  function Probe(currentProps: SessionHookProps) {
    latestValue = useWorkspaceConversationLandingSessionRuntime(currentProps);
    return null;
  }

  act(() => {
    root.render(<Probe {...props} />);
  });
  mountedRoots.push({ root, container });

  return {
    getValue: () => {
      if (!latestValue) {
        throw new Error("session hook 尚未初始化");
      }
      return latestValue;
    },
  };
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

describe("useWorkspaceConversationLandingSurfaceRuntime", () => {
  it("应集中产出入口 banner、leading card 与空态技能 props", () => {
    const onSelectServiceSkill = vi.fn();
    const serviceSkills = [
      {
        id: "daily-trend-briefing",
        title: "每日趋势摘要",
      },
    ];
    const { getValue } = renderHook(
      createDefaultProps({
        entryBannerVisible: true,
        entryBannerMessage: "继续上次的内容",
        serviceSkills,
        onSelectServiceSkill,
      }),
    );

    const runtime = getValue();
    expect(runtime.entryBannerVisible).toBe(true);
    expect(runtime.entryBannerMessage).toBe("继续上次的内容");
    expect(runtime.emptyStateProps.serviceSkills).toBe(serviceSkills);
    expect(runtime.emptyStateProps.onSelectServiceSkill).toBe(
      onSelectServiceSkill,
    );
  });

  it("Task Center 空态应隐藏最近会话恢复入口", () => {
    const { getValue } = renderHook(
      createDefaultProps({
        recentSessionTitle: "品牌发布节奏整理",
        recentSessionSummary: "上次整理到发布节奏",
        recentSessionActionLabel: "继续",
        homeRecoverySession: {
          sessionId: "session-1",
          title: "品牌发布节奏整理",
          status: "running",
        },
        onResumeRecentSession: vi.fn(),
        projectConversationGroups: [
          {
            label: "今天",
            conversations: [],
          },
        ],
        suppressRecentSessionRecovery: true,
      }),
    );

    const emptyStateProps = getValue().emptyStateProps;
    expect(emptyStateProps.recentSessionTitle).toBeUndefined();
    expect(emptyStateProps.recentSessionSummary).toBeUndefined();
    expect(emptyStateProps.recentSessionActionLabel).toBeUndefined();
    expect(emptyStateProps.homeRecoverySession).toBeUndefined();
    expect(emptyStateProps.onResumeRecentSession).toBeUndefined();
    expect(emptyStateProps.projectConversationGroups).toBeUndefined();
  });

  it("空态工具偏好变更应回写 chat tool preferences", () => {
    const setChatToolPreferences = vi.fn();
    const { getValue } = renderHook(
      createDefaultProps({
        setChatToolPreferences,
      }),
    );

    getValue().emptyStateProps.onTaskEnabledChange?.(true);

    expect(setChatToolPreferences).toHaveBeenCalledTimes(1);
    const updater = setChatToolPreferences.mock.calls[0]?.[0] as (
      previous: HookProps["chatToolPreferences"],
    ) => HookProps["chatToolPreferences"];
    expect(
      updater({
        task: false,
        subagent: false,
      }),
    ).toEqual({
      task: true,
      subagent: false,
    });
  });

  it("输入栏知识包与路径导入入口应由 inputbar scene current owner 透传", () => {
    const onSelectKnowledgePack = vi.fn();
    const onImportPathReferenceAsKnowledge = vi.fn();
    const knowledgePackSelection = {
      workingDir: "/tmp/project",
      packName: "品牌资料",
    };
    const knowledgePackOptions = [
      {
        workingDir: "/tmp/project",
        packName: "品牌资料",
      },
    ];
    const { getValue } = renderHook(
      createDefaultProps({
        inputbarScene: {
          ...createDefaultProps().inputbarScene,
          knowledgePackSelection,
          knowledgePackOptions,
          onSelectKnowledgePack,
          onImportPathReferenceAsKnowledge,
        },
      }),
    );

    const emptyStateProps = getValue().emptyStateProps;
    expect(emptyStateProps.knowledgePackSelection).toBe(knowledgePackSelection);
    expect(emptyStateProps.knowledgePackOptions).toBe(knowledgePackOptions);
    expect(emptyStateProps.onSelectKnowledgePack).toBe(onSelectKnowledgePack);
    expect(emptyStateProps.onImportPathReferenceAsKnowledge).toBe(
      onImportPathReferenceAsKnowledge,
    );
  });
});

describe("useWorkspaceConversationLandingSessionRuntime", () => {
  it("非聊天布局应由 landing owner 补齐任务卡和运行时入口", () => {
    const onOpenMemoryWorkbench = vi.fn();
    const onOpenChannels = vi.fn();
    const onOpenChromeRelay = vi.fn();
    const baseRuntime = renderHook(createDefaultProps()).getValue();
    const { getValue } = renderSessionHook({
      landingSurface: baseRuntime,
      showChatLayout: false,
      messages: [
        {
          id: "user-1",
          role: "user",
          content: "整理项目进度",
          timestamp: new Date("2026-07-10T10:00:00.000Z"),
        },
        {
          id: "assistant-1",
          role: "assistant",
          content: "",
          timestamp: new Date("2026-07-10T10:00:01.000Z"),
          toolCalls: [
            {
              id: "tool-1",
              name: "Read",
              arguments: JSON.stringify({ file_path: "/tmp/project-1" }),
              status: "running",
              startTime: new Date("2026-07-10T10:00:01.000Z"),
            },
          ],
        },
      ],
      turns: [
        {
          id: "turn-1",
          thread_id: "thread-1",
          prompt_text: "整理项目进度",
          status: "running",
          started_at: "2026-07-10T10:00:00.000Z",
          created_at: "2026-07-10T10:00:00.000Z",
          updated_at: "2026-07-10T10:00:00.000Z",
        },
      ],
      currentTurnId: "turn-1",
      threadRead: {
        thread_id: "thread-1",
        status: "running",
      },
      canonicalChildren: [
        {
          name: "执行",
          parentThreadId: "thread-1",
          sessionId: "subagent-1",
          status: "running",
          threadId: "thread-subagent-1",
          updatedAtMs: 1783699200000,
        },
      ],
      isSending: true,
      sessionId: "session-1",
      projectRootPath: "/tmp/project-1",
      currentUserMessage: "整理项目进度",
      onOpenMemoryWorkbench,
      onOpenChannels,
      onOpenChromeRelay,
    });

    const emptyStateProps = getValue().emptyStateProps;
    expect(emptyStateProps.runtimeTaskCard).toEqual(
      expect.objectContaining({
        taskId: "turn-1",
        title: "整理项目进度",
      }),
    );

    emptyStateProps.onOpenMemoryWorkbench?.();
    emptyStateProps.onOpenChannels?.();
    emptyStateProps.onOpenChromeRelay?.();

    expect(onOpenMemoryWorkbench).toHaveBeenCalledWith({
      sessionId: "session-1",
      workingDir: "/tmp/project-1",
      userMessage: "整理项目进度",
    });
    expect(onOpenChannels).toHaveBeenCalledTimes(1);
    expect(onOpenChromeRelay).toHaveBeenCalledTimes(1);
  });

  it("聊天布局不应生成重复的首页任务卡", () => {
    const baseRuntime = renderHook(createDefaultProps()).getValue();
    const { getValue } = renderSessionHook({
      landingSurface: baseRuntime,
      showChatLayout: true,
      messages: [],
      onOpenMemoryWorkbench: vi.fn(),
      onOpenChannels: vi.fn(),
      onOpenChromeRelay: vi.fn(),
    });

    expect(getValue().emptyStateProps.runtimeTaskCard).toBeNull();
  });
});
