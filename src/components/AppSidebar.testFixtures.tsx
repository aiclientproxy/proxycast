/* eslint-disable react-refresh/only-export-components */
import React from "react";
import { act as reactAct } from "react";
import { createRoot, type Root } from "react-dom/client";
import { vi } from "vitest";
import type { AgentPageParams, Page, PageParams } from "@/types/page";
import { SettingsTabs as LimeSettingsTabs } from "@/types/settings";
import { AppSidebar as AppSidebarComponent } from "./AppSidebar";
import { changeLimeLocale as changeLimeLocaleImpl } from "@/i18n/createI18n";
import { TASK_CENTER_CREATE_DRAFT_TASK_EVENT as TASK_CENTER_CREATE_DRAFT_TASK_EVENT_VALUE } from "@/components/agent/chat/taskCenterDraftTaskEvents";
import { LIME_COLOR_SCHEME_STORAGE_KEY as LIME_COLOR_SCHEME_STORAGE_KEY_VALUE } from "@/lib/appearance/colorSchemes";
import { LIME_THEME_STORAGE_KEY as LIME_THEME_STORAGE_KEY_VALUE } from "@/lib/appearance/themeMode";
import {
  getStoredOemCloudSessionState as getStoredOemCloudSessionStateImpl,
  setOemCloudBootstrapSnapshot as setOemCloudBootstrapSnapshotImpl,
  setStoredOemCloudSessionState as setStoredOemCloudSessionStateImpl,
} from "@/lib/oemCloudSession";
import type {
  ConversationImportJob,
  ConversationImportThreadCommitResponse,
  ConversationImportThreadPreviewResponse,
  ImportedThreadSummary,
} from "@/lib/api/conversationImport";
import type { AgentBackgroundSessionRuntimeSnapshot } from "@/components/agent/chat";

export const act = reactAct;
export const AppSidebar = AppSidebarComponent;
export const changeLimeLocale = changeLimeLocaleImpl;
export const SettingsTabs = LimeSettingsTabs;
export const LIME_COLOR_SCHEME_STORAGE_KEY =
  LIME_COLOR_SCHEME_STORAGE_KEY_VALUE;
export const LIME_THEME_STORAGE_KEY = LIME_THEME_STORAGE_KEY_VALUE;
export const TASK_CENTER_CREATE_DRAFT_TASK_EVENT =
  TASK_CENTER_CREATE_DRAFT_TASK_EVENT_VALUE;
export const getStoredOemCloudSessionState = getStoredOemCloudSessionStateImpl;
export const setOemCloudBootstrapSnapshot = setOemCloudBootstrapSnapshotImpl;
export const setStoredOemCloudSessionState = setStoredOemCloudSessionStateImpl;

const {
  mockGetConfig,
  mockSaveConfig,
  mockSubscribeAppConfigChanged,
  mockListAgentRuntimeSessions,
  mockListInstalledPlugins,
  mockSelectPluginDirectory,
  mockGetProject,
  mockUpdateProject,
  mockDeleteProject,
  mockEnsureProjectWorkspace,
  mockCreateProjectGitWorktree,
  mockRevealPathInFinder,
  mockArchiveAgentRuntimeSession,
  mockUnarchiveAgentRuntimeSession,
  mockSetAgentRuntimeThreadName,
  mockListThreadSections,
  mockCreateThreadSection,
  mockUpdateThreadSection,
  mockDeleteThreadSection,
  mockMoveThreadToSection,
  mockDeleteAgentRuntimeSession,
  mockScanConversationImportSource,
  mockPreviewConversationImportThread,
  mockCommitConversationImportThread,
  mockReadConversationImportJob,
  mockWaitForConversationImportJob,
  mockSetI18nLanguage,
  mockScheduleMinimumDelayIdleTask,
  mockLogoutClient,
  mockGetConfiguredOemCloudTarget,
  mockBuildOemCloudUserCenterUrl,
  mockCreateExternalBrowserOpenTarget,
  mockOpenExternalUrl,
  mockStartOemCloudLogin,
  mockGetClientReferralDashboard,
  mockClearSiteAdapterCatalogCache,
  mockToastSuccess,
  mockToastError,
  mockToastInfo,
  mockRecordAgentUiPerformanceMetric,
  mockSubscribeAgentUiPerformanceMetricRecorded,
  mockAgentUiPerformanceMetricListeners,
  mockCheckForUpdates,
  mockGetUpdateInstallSession,
  mockListenUpdateInstallSession,
  mockOpenUpdateWindow,
  mockRecordUpdateNotificationAction,
  mockRemindUpdateLater,
  mockStartUpdateInstallSession,
} = vi.hoisted(() => ({
  mockGetConfig: vi.fn(),
  mockSaveConfig: vi.fn(),
  mockSubscribeAppConfigChanged: vi.fn(),
  mockListAgentRuntimeSessions: vi.fn(),
  mockListInstalledPlugins: vi.fn(),
  mockSelectPluginDirectory: vi.fn(),
  mockGetProject: vi.fn(),
  mockUpdateProject: vi.fn(),
  mockDeleteProject: vi.fn(),
  mockEnsureProjectWorkspace: vi.fn(),
  mockCreateProjectGitWorktree: vi.fn(),
  mockRevealPathInFinder: vi.fn(),
  mockArchiveAgentRuntimeSession: vi.fn(),
  mockUnarchiveAgentRuntimeSession: vi.fn(),
  mockSetAgentRuntimeThreadName: vi.fn(),
  mockListThreadSections: vi.fn(),
  mockCreateThreadSection: vi.fn(),
  mockUpdateThreadSection: vi.fn(),
  mockDeleteThreadSection: vi.fn(),
  mockMoveThreadToSection: vi.fn(),
  mockDeleteAgentRuntimeSession: vi.fn(),
  mockScanConversationImportSource: vi.fn(),
  mockPreviewConversationImportThread: vi.fn(),
  mockCommitConversationImportThread: vi.fn(),
  mockReadConversationImportJob: vi.fn(),
  mockWaitForConversationImportJob: vi.fn(),
  mockSetI18nLanguage: vi.fn(),
  mockScheduleMinimumDelayIdleTask: vi.fn((task: () => void) => {
    task();
    return () => undefined;
  }),
  mockLogoutClient: vi.fn(),
  mockGetConfiguredOemCloudTarget: vi.fn(),
  mockBuildOemCloudUserCenterUrl: vi.fn(
    (baseUrl: string, path = "") => `${baseUrl}${path}`,
  ),
  mockCreateExternalBrowserOpenTarget: vi.fn(() => null),
  mockOpenExternalUrl: vi.fn(),
  mockStartOemCloudLogin: vi.fn(),
  mockGetClientReferralDashboard: vi.fn(),
  mockClearSiteAdapterCatalogCache: vi.fn(),
  mockToastSuccess: vi.fn(),
  mockToastError: vi.fn(),
  mockToastInfo: vi.fn(),
  mockRecordAgentUiPerformanceMetric: vi.fn(),
  mockSubscribeAgentUiPerformanceMetricRecorded: vi.fn(),
  mockAgentUiPerformanceMetricListeners: [] as Array<
    (detail: {
      id: number;
      phase: string;
      sessionId?: string | null;
      source?: string | null;
      workspaceId?: string | null;
    }) => void
  >,
  mockCheckForUpdates: vi.fn(),
  mockGetUpdateInstallSession: vi.fn(),
  mockListenUpdateInstallSession: vi.fn(),
  mockOpenUpdateWindow: vi.fn(),
  mockRecordUpdateNotificationAction: vi.fn(),
  mockRemindUpdateLater: vi.fn(),
  mockStartUpdateInstallSession: vi.fn(),
}));

export {
  mockBuildOemCloudUserCenterUrl,
  mockClearSiteAdapterCatalogCache,
  mockCheckForUpdates,
  mockCreateExternalBrowserOpenTarget,
  mockDeleteAgentRuntimeSession,
  mockGetUpdateInstallSession,
  mockGetClientReferralDashboard,
  mockGetConfig,
  mockGetConfiguredOemCloudTarget,
  mockListenUpdateInstallSession,
  mockListAgentRuntimeSessions,
  mockListInstalledPlugins,
  mockSelectPluginDirectory,
  mockGetProject,
  mockUpdateProject,
  mockDeleteProject,
  mockEnsureProjectWorkspace,
  mockCreateProjectGitWorktree,
  mockRevealPathInFinder,
  mockScanConversationImportSource,
  mockLogoutClient,
  mockPreviewConversationImportThread,
  mockCommitConversationImportThread,
  mockReadConversationImportJob,
  mockWaitForConversationImportJob,
  mockOpenExternalUrl,
  mockOpenUpdateWindow,
  mockRecordUpdateNotificationAction,
  mockRecordAgentUiPerformanceMetric,
  mockSubscribeAgentUiPerformanceMetricRecorded,
  mockRemindUpdateLater,
  mockSaveConfig,
  mockScheduleMinimumDelayIdleTask,
  mockSetI18nLanguage,
  mockStartOemCloudLogin,
  mockStartUpdateInstallSession,
  mockSubscribeAppConfigChanged,
  mockToastError,
  mockToastInfo,
  mockToastSuccess,
  mockArchiveAgentRuntimeSession,
  mockUnarchiveAgentRuntimeSession,
  mockSetAgentRuntimeThreadName,
  mockListThreadSections,
  mockCreateThreadSection,
  mockUpdateThreadSection,
  mockDeleteThreadSection,
  mockMoveThreadToSection,
};

vi.mock("@/lib/api/appConfig", () => ({
  getConfig: mockGetConfig,
  saveConfig: mockSaveConfig,
  subscribeAppConfigChanged: mockSubscribeAppConfigChanged,
}));

vi.mock("@/i18n/legacy-patch/I18nPatchProvider", () => ({
  useI18nPatch: () => ({
    language: "zh",
    setLanguage: mockSetI18nLanguage,
  }),
}));

vi.mock("@/lib/api/agentRuntime/sessionClient", () => ({
  AGENT_RUNTIME_SESSIONS_CHANGED_EVENT: "lime:agent-runtime-sessions-changed",
  archiveAgentRuntimeSession: mockArchiveAgentRuntimeSession,
  deleteAgentRuntimeSession: mockDeleteAgentRuntimeSession,
  listAgentRuntimeSessions: mockListAgentRuntimeSessions,
  unarchiveAgentRuntimeSession: mockUnarchiveAgentRuntimeSession,
}));

vi.mock("@/lib/api/agentRuntime/threadClient", () => ({
  setAgentRuntimeThreadName: mockSetAgentRuntimeThreadName,
}));

vi.mock("@/lib/api/threadSections", () => {
  const pinnedSection = {
    id: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
    name: "Pinned",
  };
  return {
    PINNED_THREAD_SECTION: pinnedSection,
    isPinnedThreadSession: (session: { section?: { id?: string } }) =>
      session.section?.id === pinnedSection.id,
    listThreadSections: mockListThreadSections,
    createThreadSection: mockCreateThreadSection,
    updateThreadSection: mockUpdateThreadSection,
    deleteThreadSection: mockDeleteThreadSection,
    moveThreadToSection: mockMoveThreadToSection,
  };
});

vi.mock("@/lib/api/conversationImport", () => ({
  scanConversationImportSource: mockScanConversationImportSource,
  previewConversationImportThread: mockPreviewConversationImportThread,
  commitConversationImportThread: mockCommitConversationImportThread,
  readConversationImportJob: mockReadConversationImportJob,
  waitForConversationImportJob: mockWaitForConversationImportJob,
}));

vi.mock("@/lib/api/plugins", () => ({
  PLUGINS_CHANGED_EVENT: "lime:plugins-changed",
  listInstalledPlugins: mockListInstalledPlugins,
  selectPluginDirectory: mockSelectPluginDirectory,
}));

vi.mock("@/lib/api/project", () => ({
  getProject: mockGetProject,
  updateProject: mockUpdateProject,
  deleteProject: mockDeleteProject,
  ensureProjectWorkspace: mockEnsureProjectWorkspace,
  extractErrorMessage: (error: unknown) =>
    error instanceof Error ? error.message : String(error),
}));

vi.mock("@/lib/api/projectGit", () => ({
  createProjectGitWorktree: mockCreateProjectGitWorktree,
}));

vi.mock("@/lib/api/fileSystem", () => ({
  revealPathInFinder: mockRevealPathInFinder,
}));

vi.mock("@/lib/api/appUpdate", () => ({
  checkForUpdates: mockCheckForUpdates,
  getUpdateInstallSession: mockGetUpdateInstallSession,
  listenUpdateInstallSession: mockListenUpdateInstallSession,
  openUpdateWindow: mockOpenUpdateWindow,
  recordUpdateNotificationAction: mockRecordUpdateNotificationAction,
  remindUpdateLater: mockRemindUpdateLater,
  startUpdateInstallSession: mockStartUpdateInstallSession,
  isUpdateInstallSessionActive: (
    session:
      | {
          stage?: string;
          isActive?: boolean;
        }
      | null
      | undefined,
  ) =>
    Boolean(
      session?.isActive &&
      (
        ["checking", "downloading", "installing", "restarting"] as string[]
      ).includes(session.stage ?? ""),
    ),
}));

vi.mock("@/lib/api/oemCloudControlPlane", () => ({
  logoutClient: mockLogoutClient,
  getConfiguredOemCloudTarget: mockGetConfiguredOemCloudTarget,
  getClientReferralDashboard: mockGetClientReferralDashboard,
}));

vi.mock("@/lib/oemCloudLoginLauncher", () => ({
  buildOemCloudUserCenterUrl: mockBuildOemCloudUserCenterUrl,
  createExternalBrowserOpenTarget: mockCreateExternalBrowserOpenTarget,
  openExternalUrl: mockOpenExternalUrl,
  startOemCloudLogin: mockStartOemCloudLogin,
}));

vi.mock("sonner", () => ({
  toast: {
    success: mockToastSuccess,
    error: mockToastError,
    info: mockToastInfo,
  },
}));

vi.mock("@/lib/siteAdapterCatalogBootstrap", () => ({
  clearSiteAdapterCatalogCache: mockClearSiteAdapterCatalogCache,
}));

vi.mock("@/lib/utils/scheduleMinimumDelayIdleTask", () => ({
  scheduleMinimumDelayIdleTask: mockScheduleMinimumDelayIdleTask,
}));

type MockAgentUiPerformanceMetricDetail = {
  id: number;
  phase: string;
  sessionId?: string | null;
  source?: string | null;
  workspaceId?: string | null;
};

export function emitMockAgentUiPerformanceMetricRecorded(
  detail: MockAgentUiPerformanceMetricDetail,
) {
  for (const listener of [...mockAgentUiPerformanceMetricListeners]) {
    listener(detail);
  }
}

vi.mock("@/lib/agentUiPerformanceMetrics", () => ({
  recordAgentUiPerformanceMetric: mockRecordAgentUiPerformanceMetric,
  subscribeAgentUiPerformanceMetricRecorded:
    mockSubscribeAgentUiPerformanceMetricRecorded.mockImplementation(
      (listener: (detail: MockAgentUiPerformanceMetricDetail) => void) => {
        mockAgentUiPerformanceMetricListeners.push(listener);
        return () => {
          const index = mockAgentUiPerformanceMetricListeners.indexOf(listener);
          if (index >= 0) {
            mockAgentUiPerformanceMetricListeners.splice(index, 1);
          }
        };
      },
    ),
}));

interface MountedSidebar {
  container: HTMLDivElement;
  root: Root;
}

const mountedSidebars: MountedSidebar[] = [];
export const APP_SIDEBAR_COLLAPSED_STORAGE_KEY = "lime.app-sidebar.collapsed";
export const APP_SIDEBAR_ENABLED_ITEMS_STORAGE_KEY =
  "lime.app-sidebar.enabled-items";

export function mountSidebar(options?: {
  currentPage?: Page;
  currentPageParams?: PageParams;
  activeAgentSessionId?: string | null;
  activeAgentStreaming?: boolean;
  backgroundAgentSessionRuntime?: AgentBackgroundSessionRuntimeSnapshot | null;
  requestedPage?: Page;
  requestedPageParams?: PageParams;
  onNavigate?: (page: Page, params?: PageParams) => void;
}): MountedSidebar {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(
      <AppSidebar
        currentPage={options?.currentPage ?? "agent"}
        currentPageParams={options?.currentPageParams}
        activeAgentSessionId={options?.activeAgentSessionId}
        activeAgentStreaming={options?.activeAgentStreaming}
        backgroundAgentSessionRuntime={options?.backgroundAgentSessionRuntime}
        requestedPage={options?.requestedPage}
        requestedPageParams={options?.requestedPageParams}
        onNavigate={options?.onNavigate ?? vi.fn()}
      />,
    );
  });

  const mounted = { container, root };
  mountedSidebars.push(mounted);
  return mounted;
}

export function mountSidebarContainer(options?: {
  currentPage?: Page;
  currentPageParams?: PageParams;
  activeAgentSessionId?: string | null;
  activeAgentStreaming?: boolean;
  backgroundAgentSessionRuntime?: AgentBackgroundSessionRuntimeSnapshot | null;
  requestedPage?: Page;
  requestedPageParams?: PageParams;
  onNavigate?: (page: Page, params?: PageParams) => void;
}) {
  return mountSidebar(options).container;
}

export async function flushEffects(times = 1) {
  for (let index = 0; index < times; index += 1) {
    await act(async () => {
      await Promise.resolve();
    });
  }
}

export function setInputValue(input: HTMLInputElement, value: string) {
  const prototype = Object.getPrototypeOf(input) as HTMLInputElement;
  const valueSetter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
  if (valueSetter) {
    valueSetter.call(input, value);
  } else {
    input.value = value;
  }
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

export async function openConversationMenu(title: string) {
  await act(async () => {
    document
      .querySelector<HTMLButtonElement>(
        `button[aria-label="打开 ${title} 操作菜单"]`,
      )
      ?.click();
    await Promise.resolve();
  });

  return document.body.querySelector<HTMLElement>(
    '[data-testid="app-sidebar-conversation-menu"]',
  );
}

export async function openProjectMenu(title: string) {
  await act(async () => {
    document
      .querySelector<HTMLButtonElement>(
        `button[aria-label="打开 ${title} 项目菜单"]`,
      )
      ?.click();
    await Promise.resolve();
  });

  return document.body.querySelector<HTMLElement>(
    '[data-testid="app-sidebar-project-menu"]',
  );
}

export async function clickConversationMenuItem(testId: string) {
  await act(async () => {
    document.body
      .querySelector<HTMLButtonElement>(`[data-testid="${testId}"]`)
      ?.click();
    await Promise.resolve();
  });
}

export async function openAccountMenu(container: HTMLElement) {
  await act(async () => {
    container
      .querySelector<HTMLButtonElement>(
        '[data-testid="app-sidebar-account-button"]',
      )
      ?.click();
    await Promise.resolve();
  });
}

export async function clickAccountMenuItem(
  container: HTMLElement,
  label: string,
) {
  await openAccountMenu(container);

  await act(async () => {
    container
      .querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`)
      ?.click();
    await Promise.resolve();
  });
}

export function buildMockReferralDashboard() {
  return {
    code: {
      id: "refcode-001",
      tenantId: "tenant-0001",
      userId: "user-001",
      code: "LIME-2026",
      landingUrl: "https://limeai.run/invite?code=LIME-2026",
      status: "active",
      createdAt: "2026-04-28T00:00:00.000Z",
      updatedAt: "2026-04-28T00:00:00.000Z",
    },
    policy: {
      enabled: true,
      rewardCredits: 600,
      referrerRewardCredits: 480,
      inviteeRewardCredits: 120,
      claimWindowDays: 30,
      autoClaimEnabled: true,
      allowManualClaimFallback: true,
      riskReviewEnabled: false,
    },
    summary: {
      totalInvites: 0,
      successfulInvites: 0,
      totalRewardCredits: 0,
      referrerRewardCreditsTotal: 0,
      inviteeRewardCreditsTotal: 0,
    },
    events: [],
    rewards: [],
    invitedBy: {},
    share: {
      brandName: "Lime",
      code: "LIME-2026",
      landingUrl: "https://limeai.run/invite?code=LIME-2026",
      downloadUrl: "https://limeai.run",
      shareText:
        "邀请你体验Lime，让AI做牛做马，我们来做牛人！前往 https://limeai.run 下载客户端，复制邀请码 LIME-2026 激活并注册账号参与内测",
      headline: "登录后自动领取奖励",
      rules: "复制邀请码后完成注册即可参与内测。",
    },
  };
}

export function seedCloudSessionWithReferral(options?: {
  referralEnabled?: boolean;
  referral?: ReturnType<typeof buildMockReferralDashboard> | null;
}) {
  setStoredOemCloudSessionState({
    token: "session-token",
    tenant: { id: "tenant-0001", name: "Lime Cloud" },
    user: {
      id: "user-001",
      displayName: "晚风",
      email: "wanfeng@example.com",
    },
    session: { id: "session-001", provider: "google" },
  });
  setOemCloudBootstrapSnapshot({
    session: {
      tenant: { id: "tenant-0001", name: "Lime Cloud" },
    },
    features: {
      referralEnabled: options?.referralEnabled ?? true,
    },
    ...(options?.referral !== null
      ? { referral: options?.referral ?? buildMockReferralDashboard() }
      : {}),
  });
}

export type { AgentPageParams, Page, PageParams };

export function buildMockImportedThreadSummary(
  overrides: Partial<ImportedThreadSummary> = {},
): ImportedThreadSummary {
  return {
    sourceClient: "codex",
    sourceThreadId: "codex-thread-1",
    title: "本地历史修复记录",
    createdAt: "2026-06-15T00:00:00.000Z",
    updatedAt: "2026-06-16T00:00:00.000Z",
    cwd: "/repo/project-1",
    source: "cli",
    modelProvider: "openai",
    archived: false,
    sourcePath: "/Users/example/.codex/sessions/codex-thread-1.jsonl",
    importStatus: "not_imported",
    ...overrides,
  };
}

export function buildMockConversationImportPreview(
  overrides: Partial<ConversationImportThreadPreviewResponse> = {},
): ConversationImportThreadPreviewResponse {
  const thread = buildMockImportedThreadSummary(overrides.thread);
  return {
    source: {
      sourceClient: "codex",
      status: "ready",
      sourceRoot: "/Users/example/.codex",
      readable: true,
      threadCount: 1,
      sourceHomeExists: true,
      stateDbReadable: true,
      rolloutFileCount: 1,
      indexedAt: "2026-06-16T00:00:00.000Z",
      statePath: "/Users/example/.codex/state_5.sqlite",
    },
    thread,
    summary: {
      lineCount: 8,
      messageCount: 2,
      rolloutEventItems: 2,
      unsupportedCount: 1,
      dryRun: {
        willCreateSession: thread.importStatus !== "imported",
        willAppendToExistingSession: thread.importStatus === "imported",
        willImportMessages: 2,
        willImportTurns: 1,
        willImportTimelineItems: 4,
        willImportAttachments: 1,
        unsupportedItems: 1,
      },
      fidelity: {
        messages: 2,
        reasoning: 0,
        tools: 2,
        commands: 1,
        patches: 1,
        approvals: 0,
        mcp: 0,
        webSearch: 0,
        attachments: 1,
        unsupported: 1,
        provenanceOnly: 1,
        budgetDropped: 0,
      },
      truncated: false,
      warnings: ["工具事件会作为来源信息保留，不会伪造成 Lime 工具时间线。"],
    },
    messages: [
      {
        role: "user",
        text: "请帮我修复运行时问题",
        attachments: [
          {
            kind: "image",
            uri: "data:image/png;base64,preview",
            metadata: {
              sourceType: "event_msg",
              codexField: "images",
              mediaType: "image/png",
            },
          },
        ],
        truncated: false,
        omittedBytes: 0,
        timestamp: "2026-06-16T00:00:00.000Z",
        sourceType: "event_msg",
        provenance: {
          sourceClient: "codex",
          sourceThreadId: thread.sourceThreadId,
          sourcePath: thread.sourcePath,
          sourceEventType: "event_msg",
          sourceEventSeq: 2,
          sourcePayloadType: "user_message",
        },
      },
      {
        role: "assistant",
        text: "已完成修复并补充测试。",
        attachments: [],
        truncated: false,
        omittedBytes: 0,
        timestamp: "2026-06-16T00:00:01.000Z",
        sourceType: "event_msg",
        provenance: {
          sourceClient: "codex",
          sourceThreadId: thread.sourceThreadId,
          sourcePath: thread.sourcePath,
          sourceEventType: "event_msg",
          sourceEventSeq: 3,
          sourcePayloadType: "agent_message",
        },
      },
    ],
    events: [],
    ...overrides,
  };
}

export async function resetAppSidebarTest() {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  await changeLimeLocale("zh-CN");
  mockAgentUiPerformanceMetricListeners.splice(
    0,
    mockAgentUiPerformanceMetricListeners.length,
  );
  localStorage.clear();
  delete window.__LIME_BOOTSTRAP__;
  delete window.__LIME_OEM_CLOUD__;
  delete window.__LIME_SESSION_TOKEN__;
  document.documentElement.classList.remove("dark");
  document.documentElement.removeAttribute("data-lime-theme");
  document.documentElement.removeAttribute("data-lime-color-scheme");
  document.documentElement.removeAttribute("style");
  mockGetConfig.mockResolvedValue({});
  mockSaveConfig.mockResolvedValue(undefined);
  mockListAgentRuntimeSessions.mockResolvedValue([]);
  mockListInstalledPlugins.mockResolvedValue({ states: [], issues: [] });
  mockSelectPluginDirectory.mockResolvedValue({
    path: null,
    cancelled: true,
  });
  mockGetProject.mockResolvedValue(null);
  mockUpdateProject.mockResolvedValue({});
  mockDeleteProject.mockResolvedValue(true);
  mockEnsureProjectWorkspace.mockResolvedValue({
    id: "project-worktree",
    name: "project-worktree",
    rootPath: "/tmp/project-worktree",
  });
  mockCreateProjectGitWorktree.mockResolvedValue({
    worktreePath: "/tmp/project-worktree",
    branch: "worktree",
    status: {
      rootPath: "/tmp/project",
      hasGitRepository: true,
      currentBranch: "main",
      branches: ["main"],
      uncommittedFileCount: 0,
    },
  });
  mockRevealPathInFinder.mockResolvedValue(undefined);
  mockArchiveAgentRuntimeSession.mockResolvedValue(undefined);
  mockUnarchiveAgentRuntimeSession.mockResolvedValue(undefined);
  mockSetAgentRuntimeThreadName.mockResolvedValue(undefined);
  mockListThreadSections.mockResolvedValue([
    {
      id: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
      name: "Pinned",
    },
  ]);
  mockCreateThreadSection.mockResolvedValue({
    id: "section-created",
    name: "新分组",
  });
  mockUpdateThreadSection.mockImplementation(
    async ({ sectionId, name }: { sectionId: string; name: string }) => ({
      id: sectionId,
      name,
    }),
  );
  mockDeleteThreadSection.mockResolvedValue(undefined);
  mockMoveThreadToSection.mockResolvedValue(undefined);
  mockDeleteAgentRuntimeSession.mockResolvedValue(undefined);
  mockScanConversationImportSource.mockResolvedValue({
    source: {
      sourceClient: "codex",
      status: "ready",
      sourceRoot: "/Users/example/.codex",
      readable: true,
      threadCount: 1,
      indexedAt: "2026-06-16T00:00:00.000Z",
      statePath: "/Users/example/.codex/state_5.sqlite",
    },
    threads: [
      {
        sourceClient: "codex",
        sourceThreadId: "codex-thread-1",
        title: "本地历史修复记录",
        createdAt: "2026-06-15T00:00:00.000Z",
        updatedAt: "2026-06-16T00:00:00.000Z",
        cwd: "/repo/project-1",
        source: "cli",
        modelProvider: "openai",
        archived: false,
        sourcePath: "/Users/example/.codex/sessions/codex-thread-1.jsonl",
        importStatus: "not_imported",
      },
      {
        sourceClient: "codex",
        sourceThreadId: "codex-thread-2",
        title: "本地历史第二条记录",
        createdAt: "2026-06-15T01:00:00.000Z",
        updatedAt: "2026-06-16T01:00:00.000Z",
        cwd: "/repo/project-1",
        source: "cli",
        modelProvider: "openai",
        archived: false,
        sourcePath: "/Users/example/.codex/sessions/codex-thread-2.jsonl",
        importStatus: "not_imported",
      },
    ],
  });
  mockPreviewConversationImportThread.mockResolvedValue(
    buildMockConversationImportPreview(),
  );
  mockWaitForConversationImportJob.mockImplementation(
    async (job: ConversationImportJob) => {
      if (job.status !== "completed" || !job.result) {
        throw new Error("Test import job did not complete");
      }
      return job.result;
    },
  );
  mockCommitConversationImportThread.mockImplementation(async (params) => {
    const response: ConversationImportThreadCommitResponse = {
      session: {
        sessionId:
          params.sourceThreadId === "codex-thread-2"
            ? "session-imported-2"
            : "session-imported",
        threadId:
          params.sourceThreadId === "codex-thread-2"
            ? "thread-imported-2"
            : "thread-imported",
        appId: "content-studio",
        workspaceId: "project-1",
        status: "completed",
        createdAt: "2026-06-16T00:00:00.000Z",
        updatedAt: "2026-06-16T00:00:01.000Z",
      },
      thread: {
        sourceClient: "codex",
        sourceThreadId: params.sourceThreadId ?? "codex-thread-1",
        title:
          params.sourceThreadId === "codex-thread-2"
            ? "本地历史第二条记录"
            : "本地历史修复记录",
        createdAt: "2026-06-15T00:00:00.000Z",
        updatedAt: "2026-06-16T00:00:00.000Z",
        cwd: "/repo/project-1",
        source: "cli",
        modelProvider: "openai",
        archived: false,
        sourcePath:
          params.sourcePath ??
          "/Users/example/.codex/sessions/codex-thread-1.jsonl",
        importStatus: "imported",
      },
      summary: {
        lineCount: 8,
        messageCount: 2,
        rolloutEventItems: 2,
        unsupportedCount: 1,
        dryRun: {
          willCreateSession: true,
          willAppendToExistingSession: false,
          willImportMessages: 2,
          willImportTurns: 1,
          willImportTimelineItems: 4,
          willImportAttachments: 1,
          unsupportedItems: 1,
        },
        fidelity: {
          messages: 2,
          reasoning: 0,
          tools: 2,
          commands: 1,
          patches: 1,
          approvals: 0,
          mcp: 0,
          webSearch: 0,
          attachments: 1,
          unsupported: 1,
          provenanceOnly: 1,
          budgetDropped: 0,
        },
        truncated: false,
        warnings: [],
      },
      importedMessages: 2,
      importedTurns: 1,
      canContinue: true,
      warnings: [],
    };
    return {
      job: {
        jobId: `import-job-${params.sourceThreadId ?? "codex-thread-1"}`,
        sourceClient: "codex",
        sourceThreadId: params.sourceThreadId ?? "codex-thread-1",
        status: "completed",
        progress: {
          phase: "completed",
          completedItems: response.summary.rolloutEventItems,
          totalItems: response.summary.rolloutEventItems,
          completedTurns: response.importedTurns,
          totalTurns: response.importedTurns,
        },
        result: response,
        createdAt: "2026-06-16T00:00:00.000Z",
        updatedAt: "2026-06-16T00:00:01.000Z",
      },
    };
  });
  mockCheckForUpdates.mockResolvedValue({
    current: "1.57.0",
    latest: null,
    hasUpdate: false,
    downloadUrl: null,
    releaseNotesUrl: null,
    releaseNotes: null,
    pubDate: null,
    error: null,
  });
  mockGetUpdateInstallSession.mockResolvedValue({
    sessionId: "idle",
    stage: "idle",
    currentVersion: "1.57.0",
    latestVersion: null,
    downloadUrl: null,
    downloadedBytes: 0,
    totalBytes: null,
    percent: 0,
    message: "idle",
    error: null,
    startedAt: 0,
    updatedAt: 0,
    completedAt: null,
    canCloseWindow: true,
    isActive: false,
  });
  mockListenUpdateInstallSession.mockResolvedValue(() => undefined);
  mockOpenUpdateWindow.mockResolvedValue(undefined);
  mockRecordUpdateNotificationAction.mockResolvedValue(undefined);
  mockRemindUpdateLater.mockResolvedValue(0);
  mockStartUpdateInstallSession.mockResolvedValue({
    sessionId: "installing",
    stage: "downloading",
    currentVersion: "1.57.0",
    latestVersion: "1.58.0",
    downloadUrl: "https://example.com/lime",
    downloadedBytes: 20,
    totalBytes: 100,
    percent: 0.2,
    message: "downloading",
    error: null,
    startedAt: 0,
    updatedAt: 0,
    completedAt: null,
    canCloseWindow: true,
    isActive: true,
  });
  mockLogoutClient.mockResolvedValue(undefined);
  mockGetConfiguredOemCloudTarget.mockReturnValue({
    baseUrl: "https://user.limeai.run",
    tenantId: "tenant-0001",
  });
  mockBuildOemCloudUserCenterUrl.mockImplementation(
    (baseUrl: string, path = "") => `${baseUrl}${path}`,
  );
  mockOpenExternalUrl.mockResolvedValue(undefined);
  mockStartOemCloudLogin.mockResolvedValue({
    mode: "login_url",
    openedUrl: "https://user.limeai.run/login",
  });
  mockGetClientReferralDashboard.mockResolvedValue(
    buildMockReferralDashboard(),
  );
  mockClearSiteAdapterCatalogCache.mockResolvedValue(null);
  mockScheduleMinimumDelayIdleTask.mockImplementation((task: () => void) => {
    task();
    return () => undefined;
  });
  mockSubscribeAppConfigChanged.mockImplementation((listener: () => void) => {
    (
      globalThis as typeof globalThis & { __appConfigListener?: () => void }
    ).__appConfigListener = listener;
    return () => {
      (
        globalThis as typeof globalThis & {
          __appConfigListener?: () => void;
        }
      ).__appConfigListener = undefined;
    };
  });
}

export function cleanupAppSidebarTest() {
  while (mountedSidebars.length > 0) {
    const mounted = mountedSidebars.pop();
    if (!mounted) {
      continue;
    }

    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }

  vi.clearAllMocks();
  mockAgentUiPerformanceMetricListeners.splice(
    0,
    mockAgentUiPerformanceMetricListeners.length,
  );
  vi.unstubAllGlobals();
  delete window.__LIME_BOOTSTRAP__;
  delete window.__LIME_OEM_CLOUD__;
  delete window.__LIME_SESSION_TOKEN__;
  document.documentElement.classList.remove("dark");
  document.documentElement.removeAttribute("data-lime-theme");
  document.documentElement.removeAttribute("data-lime-color-scheme");
  document.documentElement.removeAttribute("style");
  (
    globalThis as typeof globalThis & {
      __appConfigListener?: () => void;
    }
  ).__appConfigListener = undefined;
}
