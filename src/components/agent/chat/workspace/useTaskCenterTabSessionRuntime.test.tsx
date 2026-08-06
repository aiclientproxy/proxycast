import { act, useMemo, useRef, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { Topic } from "../hooks/agentChatShared";
import type { TaskCenterDraftSendRequest } from "../homePendingPreview";
import {
  TASK_CENTER_OPEN_TAB_IDS_STORAGE_KEY,
  type TaskCenterLocalSessionOverride,
  type TaskCenterWorkspaceTabMap,
} from "../utils/taskCenterTabs";
import type { TaskCenterDraftTab } from "./agentChatWorkspaceHelpers";
import {
  useTaskCenterTabSessionRuntime,
  type TaskCenterTabSessionRuntimeState,
} from "./useTaskCenterTabSessionRuntime";

interface HarnessProps {
  agentEntry?: "new-task" | "claw" | null;
  normalizedInitialSessionId?: string | null;
  newChatAt?: number;
  sessionId?: string | null;
  taskCenterWorkspaceId?: string | null;
  topics?: Topic[];
}

interface LatestState {
  activeDraftTabId: string | null;
  draftSendRequest: TaskCenterDraftSendRequest | null;
  draftSurfaceActive: boolean;
  draftTabs: TaskCenterDraftTab[];
  homePendingPreviewRequest: TaskCenterDraftSendRequest | null;
  runtime: TaskCenterTabSessionRuntimeState;
}

let container: HTMLDivElement;
let root: Root;
let latest: LatestState | null = null;

function createTopic(id: string, overrides?: Partial<Topic>): Topic {
  return {
    id,
    title: id,
    createdAt: new Date("2026-06-18T00:00:00.000Z"),
    updatedAt: new Date("2026-06-18T01:00:00.000Z"),
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

function createDraft(id: string): TaskCenterDraftTab {
  const now = new Date("2026-06-18T00:00:00.000Z");
  return {
    id,
    title: id,
    createdAt: now,
    updatedAt: now,
    status: "draft",
  };
}

function createDraftSendRequest(id: string): TaskCenterDraftSendRequest {
  return {
    id,
    draftTabId: id,
    text: "生成任务",
    images: [],
    submittedAt: Date.now(),
    materializeDraft: false,
    source: "empty-state",
  };
}

function Harness({
  agentEntry = "claw",
  normalizedInitialSessionId = "topic-a",
  newChatAt,
  sessionId = "topic-a",
  taskCenterWorkspaceId = "workspace-a",
  topics = [createTopic("topic-a")],
}: HarnessProps) {
  const taskCenterDraftSurfaceActiveRef = useRef(false);
  const [activeDraftTabId, setActiveDraftTabId] = useState<string | null>(
    "draft-a",
  );
  const [homePendingPreviewRequest, setHomePendingPreviewRequest] =
    useState<TaskCenterDraftSendRequest | null>(
      createDraftSendRequest("pending-a"),
    );
  const [draftSendRequest, setDraftSendRequest] =
    useState<TaskCenterDraftSendRequest | null>(
      createDraftSendRequest("send-a"),
    );
  const [draftTabs, setDraftTabs] = useState<TaskCenterDraftTab[]>([
    createDraft("draft-a"),
  ]);
  const topicById = useMemo(
    () => new Map(topics.map((topic) => [topic.id, topic])),
    [topics],
  );
  const runtime = useTaskCenterTabSessionRuntime({
    agentEntry,
    normalizedInitialSessionId,
    newChatAt,
    sessionId,
    taskCenterDraftSurfaceActiveRef,
    taskCenterWorkspaceId,
    topicById,
    topics,
    setActiveTaskCenterDraftTabId: setActiveDraftTabId,
    setHomePendingPreviewRequest,
    setTaskCenterDraftSendRequest: setDraftSendRequest,
    setTaskCenterDraftTabs: setDraftTabs,
  });

  latest = {
    activeDraftTabId,
    draftSendRequest,
    draftSurfaceActive: taskCenterDraftSurfaceActiveRef.current,
    draftTabs,
    homePendingPreviewRequest,
    runtime,
  };
  return null;
}

function renderHarness(props?: HarnessProps): LatestState {
  act(() => {
    root.render(<Harness {...props} />);
  });
  if (!latest) {
    throw new Error("hook 尚未初始化");
  }
  return latest;
}

describe("useTaskCenterTabSessionRuntime", () => {
  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    latest = null;
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
    window.localStorage.clear();
    latest = null;
  });

  it("claw 初始路由会替换当前工作区标签并写回持久化", () => {
    const persisted: TaskCenterWorkspaceTabMap = {
      "workspace-a": ["old-topic"],
      "workspace-b": ["other-topic"],
    };
    window.localStorage.setItem(
      TASK_CENTER_OPEN_TAB_IDS_STORAGE_KEY,
      JSON.stringify(persisted),
    );

    const state = renderHarness({
      normalizedInitialSessionId: "topic-a",
      sessionId: "topic-a",
      topics: [createTopic("topic-a")],
    });

    expect(state.runtime.taskCenterOpenTabIds).toEqual(["topic-a"]);
    expect(state.runtime.isTaskCenterEntry).toBe(true);
    expect(
      JSON.parse(
        window.localStorage.getItem(TASK_CENTER_OPEN_TAB_IDS_STORAGE_KEY) ||
          "{}",
      ),
    ).toMatchObject({
      "workspace-a": ["topic-a"],
      "workspace-b": ["other-topic"],
    });
  });

  it("new-task 初始路由也会替换当前工作区标签，避免会话回到首页", () => {
    const persisted: TaskCenterWorkspaceTabMap = {
      "workspace-a": ["old-topic"],
    };
    window.localStorage.setItem(
      TASK_CENTER_OPEN_TAB_IDS_STORAGE_KEY,
      JSON.stringify(persisted),
    );

    const state = renderHarness({
      agentEntry: "new-task",
      normalizedInitialSessionId: "topic-a",
      sessionId: "topic-a",
      topics: [createTopic("topic-a")],
    });

    expect(state.runtime.taskCenterOpenTabIds).toEqual(["topic-a"]);
    expect(state.runtime.isTaskCenterEntry).toBe(true);
  });

  it("new-task 路由 session 暂未成为已知 topic 时同样标记 detached", () => {
    renderHarness({
      agentEntry: "new-task",
      normalizedInitialSessionId: "topic-missing",
      sessionId: "topic-missing",
      topics: [],
    });

    expect(latest?.runtime.taskCenterDetachedTopicId).toBe("topic-missing");
  });

  it("new-task 显式新建时应清空当前工作区打开标签和 draft UI 态", () => {
    const persisted: TaskCenterWorkspaceTabMap = {
      "workspace-a": ["topic-a", "topic-b"],
    };
    window.localStorage.setItem(
      TASK_CENTER_OPEN_TAB_IDS_STORAGE_KEY,
      JSON.stringify(persisted),
    );

    renderHarness({
      agentEntry: "new-task",
      normalizedInitialSessionId: null,
      sessionId: "topic-a",
      topics: [createTopic("topic-a"), createTopic("topic-b")],
    });

    expect(latest?.runtime.taskCenterOpenTabIds).toEqual([
      "topic-a",
      "topic-b",
    ]);

    renderHarness({
      agentEntry: "new-task",
      normalizedInitialSessionId: null,
      newChatAt: 123,
      sessionId: "topic-a",
      topics: [createTopic("topic-a"), createTopic("topic-b")],
    });

    expect(latest?.runtime.taskCenterOpenTabIds).toEqual([]);
    expect(latest?.activeDraftTabId).toBeNull();
    expect(latest?.draftTabs).toEqual([]);
    expect(latest?.draftSendRequest).toBeNull();
    expect(latest?.homePendingPreviewRequest).toBeNull();
    expect(latest?.draftSurfaceActive).toBe(false);
    expect(latest?.runtime.taskCenterDetachedTopicId).toBeNull();
    expect(latest?.runtime.taskCenterTransitionTopicId).toBeNull();
  });

  it("可维护打开标签、嵌入 home 标记和本地 session override", () => {
    renderHarness({
      topics: [createTopic("topic-a"), createTopic("topic-b"), createTopic("topic-c")],
    });

    act(() => {
      latest?.runtime.upsertTaskCenterOpenTab("topic-b", " workspace-a ");
      latest?.runtime.markTaskCenterEmbeddedHomeSession("topic-b");
      latest?.runtime.markTaskCenterLocalSessionOverride("topic-b");
    });

    expect(latest?.runtime.taskCenterOpenTabIds).toEqual([
      "topic-a",
      "topic-b",
    ]);
    expect(latest?.runtime.taskCenterEmbeddedHomeSessionIds.has("topic-b")).toBe(
      true,
    );
    expect(latest?.runtime.taskCenterLocalSessionOverride).toEqual({
      sessionId: "topic-b",
      routeSessionId: "topic-a",
    } satisfies TaskCenterLocalSessionOverride);

    act(() => {
      latest?.runtime.clearTaskCenterEmbeddedHomeSession("topic-b");
      latest?.runtime.replaceTaskCenterOpenTabs("topic-c", "workspace-a");
    });

    expect(latest?.runtime.taskCenterEmbeddedHomeSessionIds.has("topic-b")).toBe(
      false,
    );
    expect(latest?.runtime.taskCenterOpenTabIds).toEqual(["topic-c"]);
  });

  it("路由 session 暂未成为已知 topic 时标记 detached，topic 回来后清理", () => {
    renderHarness({
      normalizedInitialSessionId: "topic-missing",
      sessionId: "topic-missing",
      topics: [],
    });

    expect(latest?.runtime.taskCenterDetachedTopicId).toBe("topic-missing");

    renderHarness({
      normalizedInitialSessionId: "topic-missing",
      sessionId: "topic-missing",
      topics: [createTopic("topic-missing")],
    });

    expect(latest?.runtime.taskCenterDetachedTopicId).toBeNull();
  });

  it("离开 task center 入口时清理 draft 与 pending 请求", () => {
    const state = renderHarness({
      agentEntry: null,
      normalizedInitialSessionId: "topic-a",
      sessionId: "topic-a",
    });

    expect(state.runtime.isTaskCenterEntry).toBe(false);
    expect(state.activeDraftTabId).toBeNull();
    expect(state.draftTabs).toEqual([]);
    expect(state.draftSendRequest).toBeNull();
    expect(state.homePendingPreviewRequest).toBeNull();
    expect(state.draftSurfaceActive).toBe(false);
    expect(state.runtime.taskCenterLocalSessionOverride).toBeNull();
  });
});
