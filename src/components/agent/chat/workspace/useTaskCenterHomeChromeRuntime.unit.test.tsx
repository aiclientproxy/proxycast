import { act, useMemo, type Dispatch, type SetStateAction } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Topic } from "../hooks/agentChatShared";
import type { TaskCenterDraftSendRequest } from "../homePendingPreview";
import {
  useTaskCenterHomeChromeRuntime,
  type TaskCenterHomeChromeRuntime,
} from "./useTaskCenterHomeChromeRuntime";

interface HarnessProps {
  topics?: Topic[];
  sessionId?: string | null;
  detachedTopicId?: string | null;
  transitionTopicId?: string | null;
  renameTopic?: (topicId: string, title: string) => void | Promise<void>;
  displayMessageCount?: number;
  threadItemCount?: number;
  draftSendRequest?: TaskCenterDraftSendRequest | null;
  hasLocalSessionOverride?: boolean;
  isHomeSessionBackgroundRecovery?: boolean;
  embeddedHomeSessionIds?: ReadonlySet<string>;
  shouldUseBrowserWorkspaceHomeChrome?: boolean;
  harnessPanelVisible?: boolean;
  setHarnessPanelVisible?: Dispatch<SetStateAction<boolean>>;
  clearEmbeddedHomeSession?: (sessionId: string) => void;
}

let container: HTMLDivElement;
let root: Root;
let latestRuntime: TaskCenterHomeChromeRuntime | null = null;

function createTopic(id: string, overrides: Partial<Topic> = {}): Topic {
  return {
    id,
    title: overrides.title ?? id,
    createdAt: new Date("2026-06-22T00:00:00.000Z"),
    updatedAt: new Date("2026-06-22T01:00:00.000Z"),
    workspaceId: "workspace-a",
    workingDir: null,
    messagesCount: 1,
    executionStrategy: "react",
    status: "done",
    statusReason: "default",
    lastPreview: `${id} preview`,
    hasUnread: false,
    tag: null,
    sourceSessionId: id,
    ...overrides,
  };
}

function Harness({
  topics = [createTopic("topic-a", { title: "旧标题" })],
  sessionId = "topic-a",
  detachedTopicId = null,
  transitionTopicId = null,
  renameTopic = () => undefined,
  displayMessageCount = 0,
  threadItemCount = 0,
  draftSendRequest = null,
  hasLocalSessionOverride = false,
  isHomeSessionBackgroundRecovery = false,
  embeddedHomeSessionIds = new Set(),
  shouldUseBrowserWorkspaceHomeChrome = false,
  harnessPanelVisible = false,
  setHarnessPanelVisible = () => undefined,
  clearEmbeddedHomeSession = () => undefined,
}: HarnessProps) {
  const topicById = useMemo(
    () => new Map(topics.map((topic) => [topic.id, topic])),
    [topics],
  );
  latestRuntime = useTaskCenterHomeChromeRuntime({
    agentEntry: "claw",
    sessionId,
    detachedTopicId,
    transitionTopicId,
    topicById,
    untitledTaskLabel: "未命名任务",
    renamePromptLabel: "重命名任务",
    renameTopic,
    draftSurfaceActive: false,
    draftTabActive: false,
    shouldSuppressDraftContent: false,
    draftSendRequest,
    normalizedInitialSessionId: null,
    isHomeSessionBackgroundRecovery,
    displayMessageCount,
    threadItemCount,
    hasPendingA2UIForm: false,
    isPreparingSend: false,
    isSending: false,
    isHomePendingPreviewActive: false,
    queuedTurnCount: 0,
    hasLocalSessionOverride,
    embeddedHomeSessionIds,
    isAutoRestoringSession: false,
    isSessionHydrating: false,
    shouldUseBrowserWorkspaceHomeChrome,
    harnessPanelVisible,
    setHarnessPanelVisible,
    clearEmbeddedHomeSession,
  });
  return null;
}

function renderHarness(props?: HarnessProps): TaskCenterHomeChromeRuntime {
  act(() => {
    root.render(<Harness {...props} />);
  });
  if (!latestRuntime) {
    throw new Error("hook 尚未初始化");
  }
  return latestRuntime;
}

describe("useTaskCenterHomeChromeRuntime", () => {
  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    latestRuntime = null;
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
    latestRuntime = null;
  });

  it("重命名普通任务话题时应使用当前标题并提交修剪后的新标题", async () => {
    const renameTopic = vi.fn().mockResolvedValue(undefined);
    const prompt = vi.spyOn(window, "prompt").mockReturnValue(" 新标题 ");
    const runtime = renderHarness({ renameTopic });

    await act(async () => {
      await runtime.handleRenameTaskTopic("topic-a");
    });

    expect(prompt).toHaveBeenCalledWith("重命名任务", "旧标题");
    expect(renameTopic).toHaveBeenCalledWith("topic-a", "新标题");
  });

  it("重命名草稿页签时不应弹出 prompt 或提交更新", async () => {
    const renameTopic = vi.fn().mockResolvedValue(undefined);
    const prompt = vi.spyOn(window, "prompt").mockReturnValue("新标题");
    const runtime = renderHarness({ renameTopic });

    await act(async () => {
      await runtime.handleRenameTaskTopic("task-draft-local");
    });

    expect(prompt).not.toHaveBeenCalled();
    expect(renameTopic).not.toHaveBeenCalled();
  });

  it("embedded home 会话出现真实内容后应清理首页占位标记", () => {
    const clearEmbeddedHomeSession = vi.fn();

    renderHarness({
      sessionId: "topic-a",
      displayMessageCount: 1,
      embeddedHomeSessionIds: new Set(["topic-a"]),
      clearEmbeddedHomeSession,
    });

    expect(clearEmbeddedHomeSession).toHaveBeenCalledWith("topic-a");
  });

  it("本地物化会话应保留为首页 chrome 的前台会话活动", () => {
    const clearEmbeddedHomeSession = vi.fn();
    const runtime = renderHarness({
      sessionId: "topic-a",
      hasLocalSessionOverride: true,
      isHomeSessionBackgroundRecovery: true,
      embeddedHomeSessionIds: new Set(["topic-a"]),
      clearEmbeddedHomeSession,
    });

    expect(runtime.hasHomeConversationActivity).toBe(true);
    expect(runtime.shouldRenderTaskCenterEmbeddedHome).toBe(false);
    expect(
      runtime.taskCenterHomeSurfaceState.shouldHideCurrentSessionContent,
    ).toBe(false);
    expect(runtime.taskCenterHomeSurfaceState.sceneSessionId).toBe("topic-a");
    expect(clearEmbeddedHomeSession).toHaveBeenCalledWith("topic-a");
  });

  it("首页 chrome 隐藏导航工具动作时应同步关闭 harness 面板", () => {
    const setHarnessPanelVisible = vi.fn();
    const runtime = renderHarness({
      shouldUseBrowserWorkspaceHomeChrome: true,
      harnessPanelVisible: true,
      setHarnessPanelVisible,
    });

    expect(runtime.suppressHomeNavbarUtilityActions).toBe(true);
    expect(setHarnessPanelVisible).toHaveBeenCalledWith(false);
  });

  it("切换中的话题应作为 tab 预览目标并标记 session switch pending", () => {
    const runtime = renderHarness({
      sessionId: "topic-a",
      transitionTopicId: "topic-b",
    });

    expect(runtime.taskCenterPreviewTopicId).toBe("topic-b");
    expect(runtime.taskCenterSessionSwitchPending).toBe(true);
  });
});
