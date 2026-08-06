import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type MouseEvent,
} from "react";
import { useTranslation } from "react-i18next";
import {
  FileInput,
  FolderPlus,
  MessageSquarePlus,
  Pencil,
  Trash2,
} from "lucide-react";
import type {
  AgentSessionInfo,
  AgentSessionSection,
} from "@/lib/api/agentRuntime/sessionTypes";
import type { AgentBackgroundSessionRuntimeSnapshot } from "@/components/agent/chat";
import {
  resolveUnfinishedSessionProjection,
  type AgentUnfinishedSessionStatus,
} from "@/components/agent/chat/projection/unfinishedSessionProjection";
import {
  formatSidebarSessionMeta,
  resolveSidebarSessionTitle,
} from "@/components/app-sidebar/sidebarSessionFormatting";
import { AppSidebarConversationRow } from "@/components/app-sidebar/AppSidebarConversationRow";
import { AppSidebarConversationEmptyState } from "@/components/app-sidebar/AppSidebarConversationEmptyState";
import { AppSidebarProjectConversationGroups } from "@/components/app-sidebar/AppSidebarProjectConversationGroups";
import {
  AppSidebarConversationMenus,
  CONVERSATION_MENU_APPROX_HEIGHT,
  CONVERSATION_MENU_VIEWPORT_MARGIN,
  CONVERSATION_MENU_WIDTH,
  type ConversationMenuState,
  type ProjectMenuState,
} from "@/components/app-sidebar/AppSidebarConversationMenus";
import {
  buildSidebarConversationGroups,
  type SidebarOpenedProjectSummary,
} from "@/components/app-sidebar/sidebarConversationGroups";
import { resolveSidebarFloatingMenuPosition } from "@/components/app-sidebar/sidebarFloatingMenuPosition";
import { PINNED_THREAD_SECTION } from "@/lib/api/threadSections";
import { useAppSidebarThreadSections } from "@/components/app-sidebar/useAppSidebarThreadSections";
import {
  ConversationActionButton,
  ConversationList,
  ConversationListMoreButton,
  ConversationSection,
  ConversationSectionActions,
  ConversationSectionHeader,
  ConversationSectionTitle,
  ConversationShelf,
} from "@/components/app-sidebar/AppSidebarConversationShelf.styles";

interface AppSidebarConversationShelfProps {
  openedProjects?: SidebarOpenedProjectSummary[];
  recentSessions: AgentSessionInfo[];
  currentSessionId?: string | null;
  activeAgentStreaming?: boolean;
  backgroundAgentSessionRuntime?: AgentBackgroundSessionRuntimeSnapshot | null;
  recentLoading: boolean;
  hasMoreRecent: boolean;
  actionSessionId: string | null;
  onCreateConversation: (project?: SidebarOpenedProjectSummary) => void;
  onImportConversation?: (project?: SidebarOpenedProjectSummary) => void;
  onNavigateToConversation: (session: AgentSessionInfo) => void;
  onRenameConversation?: (session: AgentSessionInfo) => void;
  onDeleteConversation?: (session: AgentSessionInfo) => void;
  onToggleArchive: (session: AgentSessionInfo, archived: boolean) => void;
  onTogglePinned: (session: AgentSessionInfo) => void;
  onMoveToSection: (
    session: AgentSessionInfo,
    section: AgentSessionSection | null,
  ) => void;
  onToggleProjectPin?: (project: SidebarOpenedProjectSummary) => void;
  onRevealProject?: (project: SidebarOpenedProjectSummary) => void;
  onCreateProjectWorktree?: (project: SidebarOpenedProjectSummary) => void;
  onRenameProject?: (project: SidebarOpenedProjectSummary) => void;
  onRemoveProject?: (project: SidebarOpenedProjectSummary) => void;
  onRefreshConversations?: () => Promise<void>;
  onShowMoreRecent: () => void;
}

const TERMINAL_SIDEBAR_SESSION_STATUSES = new Set([
  "completed",
  "failed",
  "canceled",
  "aborted",
]);

function normalizeSidebarRuntimeStatus(value?: string | null): string | null {
  const normalized = value
    ?.trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
  if (!normalized) {
    return null;
  }
  return normalized === "cancelled" ? "canceled" : normalized;
}

function hasTerminalSidebarRuntimeStatus(session: AgentSessionInfo): boolean {
  const threadStatus = normalizeSidebarRuntimeStatus(session.thread_status);
  const latestTurnStatus = normalizeSidebarRuntimeStatus(
    session.latest_turn_status,
  );
  return Boolean(
    (threadStatus && TERMINAL_SIDEBAR_SESSION_STATUSES.has(threadStatus)) ||
    (latestTurnStatus &&
      TERMINAL_SIDEBAR_SESSION_STATUSES.has(latestTurnStatus)),
  );
}

function resolveBackgroundSidebarRuntimeStatus(
  session: AgentSessionInfo,
  backgroundAgentSessionRuntime?: AgentBackgroundSessionRuntimeSnapshot | null,
): AgentUnfinishedSessionStatus | null {
  const backgroundSessionId = backgroundAgentSessionRuntime?.sessionId.trim();
  if (
    !backgroundAgentSessionRuntime ||
    !backgroundSessionId ||
    backgroundSessionId !== session.id ||
    hasTerminalSidebarRuntimeStatus(session)
  ) {
    return null;
  }

  switch (backgroundAgentSessionRuntime.status) {
    case "waiting":
      return "waitingAction";
    case "queued":
      return "queued";
    case "running":
      return "running";
  }
}

export function AppSidebarConversationShelf({
  openedProjects = [],
  recentSessions,
  currentSessionId,
  activeAgentStreaming = false,
  backgroundAgentSessionRuntime = null,
  recentLoading,
  hasMoreRecent,
  actionSessionId,
  onCreateConversation,
  onImportConversation,
  onNavigateToConversation,
  onRenameConversation,
  onDeleteConversation,
  onToggleArchive,
  onTogglePinned,
  onMoveToSection,
  onToggleProjectPin,
  onRevealProject,
  onCreateProjectWorktree,
  onRenameProject,
  onRemoveProject,
  onRefreshConversations,
  onShowMoreRecent,
}: AppSidebarConversationShelfProps) {
  const { t, i18n } = useTranslation("navigation");
  const {
    createSection,
    pendingSectionId,
    removeSection,
    renameSection,
    sections: threadSectionCatalog,
  } = useAppSidebarThreadSections({
    onSectionsChanged: onRefreshConversations,
  });
  const conversationUntitledLabel = t(
    "navigation.sidebar.conversations.untitled",
    "未命名对话",
  );
  const resolveLocalizedSessionTitle = useCallback(
    (session: AgentSessionInfo) =>
      resolveSidebarSessionTitle(session, conversationUntitledLabel),
    [conversationUntitledLabel],
  );
  const formatLocalizedSessionMeta = useCallback(
    (session: AgentSessionInfo) =>
      formatSidebarSessionMeta(session, {
        locale: i18n.language,
      }),
    [i18n.language],
  );
  const activeConversationGroups = useMemo(
    () =>
      buildSidebarConversationGroups({
        sessions: recentSessions,
        openedProjects,
        threadSections: threadSectionCatalog,
      }),
    [openedProjects, recentSessions, threadSectionCatalog],
  );
  const [menuState, setMenuState] = useState<ConversationMenuState>(null);
  const [projectMenuState, setProjectMenuState] =
    useState<ProjectMenuState>(null);
  const [collapsedProjectIds, setCollapsedProjectIds] = useState<Set<string>>(
    () => new Set(),
  );
  const activeProjectIdKey = useMemo(
    () =>
      activeConversationGroups.projectSections
        .map((section) => section.project.id)
        .join("\u0000"),
    [activeConversationGroups.projectSections],
  );

  useEffect(() => {
    const activeProjectIds = new Set(
      activeProjectIdKey ? activeProjectIdKey.split("\u0000") : [],
    );

    setCollapsedProjectIds((current) => {
      const next = new Set(
        [...current].filter((projectId) => activeProjectIds.has(projectId)),
      );
      return next.size === current.size ? current : next;
    });
  }, [activeProjectIdKey]);

  useEffect(() => {
    if (!menuState && !projectMenuState) {
      return;
    }

    const closeMenu = () => {
      setMenuState(null);
      setProjectMenuState(null);
    };
    window.addEventListener("click", closeMenu);
    window.addEventListener("resize", closeMenu);
    window.addEventListener("scroll", closeMenu, true);

    return () => {
      window.removeEventListener("click", closeMenu);
      window.removeEventListener("resize", closeMenu);
      window.removeEventListener("scroll", closeMenu, true);
    };
  }, [menuState, projectMenuState]);

  const openConversationMenu = useCallback(
    (event: MouseEvent<HTMLButtonElement>, session: AgentSessionInfo) => {
      event.stopPropagation();
      const rect = event.currentTarget.getBoundingClientRect();
      setMenuState({
        session,
        ...resolveSidebarFloatingMenuPosition(rect, window, {
          menuWidth: CONVERSATION_MENU_WIDTH,
          menuApproxHeight: CONVERSATION_MENU_APPROX_HEIGHT,
          viewportMargin: CONVERSATION_MENU_VIEWPORT_MARGIN,
        }),
      });
    },
    [],
  );

  const openProjectMenu = useCallback(
    (
      event: MouseEvent<HTMLButtonElement>,
      project: SidebarOpenedProjectSummary,
    ) => {
      event.stopPropagation();
      const rect = event.currentTarget.getBoundingClientRect();
      setProjectMenuState({
        project,
        ...resolveSidebarFloatingMenuPosition(rect, window, {
          menuWidth: CONVERSATION_MENU_WIDTH,
          menuApproxHeight: CONVERSATION_MENU_APPROX_HEIGHT,
          viewportMargin: CONVERSATION_MENU_VIEWPORT_MARGIN,
        }),
      });
    },
    [],
  );

  const toggleProjectCollapsed = useCallback((projectId: string) => {
    setCollapsedProjectIds((current) => {
      const next = new Set(current);
      if (next.has(projectId)) {
        next.delete(projectId);
      } else {
        next.add(projectId);
      }
      return next;
    });
  }, []);

  const closeMenus = useCallback(() => {
    setMenuState(null);
    setProjectMenuState(null);
  }, []);

  const projectsTitleLabel = t(
    "navigation.sidebar.conversations.projectsTitle",
    "项目",
  );
  const standaloneTitleLabel = t(
    "navigation.sidebar.conversations.standaloneTitle",
    "对话",
  );
  const pinnedTitleLabel = t(
    "navigation.sidebar.conversations.pinnedTitle",
    "置顶",
  );
  const newConversationLabel = t(
    "navigation.sidebar.conversations.newConversation",
    "新建对话",
  );
  const newProjectConversationLabel = t(
    "navigation.sidebar.conversations.newProjectConversation",
    "在此项目新建对话",
  );
  const importConversationLabel = t(
    "navigation.sidebar.conversations.importConversation",
    "Import Conversation",
  );
  const importProjectConversationLabel = t(
    "navigation.sidebar.conversations.importProjectConversation",
    "Import Conversation",
  );
  const loadingRecentLabel = t(
    "navigation.sidebar.conversations.loadingRecent",
    "正在加载对话",
  );
  const emptyStandaloneLabel = t(
    "navigation.sidebar.conversations.emptyStandalone",
    "暂无聊天",
  );
  const moreRecentLabel = t(
    "navigation.sidebar.conversations.moreRecent",
    "查看更多对话",
  );
  const moreActionsLabel = t(
    "navigation.sidebar.conversations.moreActions",
    "更多操作",
  );
  const renameActionLabel = t(
    "navigation.sidebar.conversations.menu.rename",
    "重命名",
  );
  const pinActionLabel = t("navigation.sidebar.conversations.menu.pin", "置顶");
  const unpinActionLabel = t(
    "navigation.sidebar.conversations.menu.unpin",
    "取消置顶",
  );
  const archiveActionLabel = t(
    "navigation.sidebar.conversations.menu.archive",
    "归档",
  );
  const moveToSectionActionLabel = t(
    "navigation.sidebar.conversations.menu.moveToSection",
    "移动到分组",
  );
  const moveToSectionBackLabel = t(
    "navigation.sidebar.conversations.menu.moveToSectionBack",
    "选择分组",
  );
  const unsectionedLabel = t(
    "navigation.sidebar.conversations.menu.unsectioned",
    "不分组",
  );
  const createSectionLabel = t(
    "navigation.sidebar.conversations.section.create.action",
    "新建分组",
  );
  const deleteActionLabel = t(
    "navigation.sidebar.conversations.menu.delete",
    "删除",
  );
  const projectPinActionLabel = t(
    "navigation.sidebar.conversations.projectMenu.pin",
    "置顶项目",
  );
  const projectUnpinActionLabel = t(
    "navigation.sidebar.conversations.projectMenu.unpin",
    "取消置顶",
  );
  const projectRevealActionLabel = t(
    "navigation.sidebar.conversations.projectMenu.reveal",
    "显示位置",
  );
  const projectWorktreeActionLabel = t(
    "navigation.sidebar.conversations.projectMenu.createWorktree",
    "创建永久工作树",
  );
  const projectRenameActionLabel = t(
    "navigation.sidebar.conversations.projectMenu.rename",
    "重命名项目",
  );
  const projectRemoveActionLabel = t(
    "navigation.sidebar.conversations.projectMenu.remove",
    "移除",
  );
  const projectMoreActionsLabel = t(
    "navigation.sidebar.conversations.projectMenu.moreActions",
    "项目操作",
  );
  const runtimeStatusLabels: Record<AgentUnfinishedSessionStatus, string> = {
    running: t("navigation.sidebar.conversations.status.running", "正在输出"),
    queued: t("navigation.sidebar.conversations.status.queued", "排队中"),
    waitingAction: t(
      "navigation.sidebar.conversations.status.waitingAction",
      "等待确认",
    ),
  };

  const renderConversationRow = (session: AgentSessionInfo) => {
    const active = currentSessionId === session.id;
    const title = resolveLocalizedSessionTitle(session);
    const runtimeProjection = resolveUnfinishedSessionProjection(session);
    const terminalRuntimeStatus = hasTerminalSidebarRuntimeStatus(session);
    const backgroundRuntimeStatus = resolveBackgroundSidebarRuntimeStatus(
      session,
      backgroundAgentSessionRuntime,
    );
    const activeRuntimeStatus: AgentUnfinishedSessionStatus | null =
      active && activeAgentStreaming && !terminalRuntimeStatus
        ? "running"
        : null;
    const runtimeStatus: AgentUnfinishedSessionStatus | null =
      runtimeProjection?.status ??
      backgroundRuntimeStatus ??
      activeRuntimeStatus;
    return (
      <AppSidebarConversationRow
        key={session.id}
        session={session}
        title={title}
        meta={formatLocalizedSessionMeta(session)}
        active={active}
        runtimeStatus={runtimeStatus}
        runtimeStatusLabel={
          runtimeStatus ? runtimeStatusLabels[runtimeStatus] : null
        }
        actionDisabled={actionSessionId === session.id}
        moreActionsLabel={moreActionsLabel}
        openActionMenuLabel={t(
          "navigation.sidebar.conversations.openActionMenu",
          {
            title,
            defaultValue: "打开 {{title}} 操作菜单",
          },
        )}
        onNavigate={onNavigateToConversation}
        onOpenMenu={openConversationMenu}
      />
    );
  };

  const threadSections = activeConversationGroups.threadSections
    .filter(
      ({ section, sessions }) =>
        section.id !== PINNED_THREAD_SECTION.id || sessions.length > 0,
    )
    .map(({ section, sessions }) => (
      <ConversationSection
        key={section.id}
        $compact
        data-testid="app-sidebar-thread-section"
        data-section-id={section.id}
      >
        <ConversationSectionHeader>
          <ConversationSectionTitle>
            {section.id === PINNED_THREAD_SECTION.id
              ? pinnedTitleLabel
              : section.name}
          </ConversationSectionTitle>
          {section.id !== PINNED_THREAD_SECTION.id ? (
            <ConversationSectionActions>
              <ConversationActionButton
                type="button"
                disabled={pendingSectionId !== null}
                onClick={() => void renameSection(section)}
                aria-label={t(
                  "navigation.sidebar.conversations.section.rename.action",
                  {
                    name: section.name,
                    defaultValue: "重命名分组 {{name}}",
                  },
                )}
                title={t(
                  "navigation.sidebar.conversations.section.rename.action",
                  {
                    name: section.name,
                    defaultValue: "重命名分组 {{name}}",
                  },
                )}
                data-testid="app-sidebar-thread-section-rename"
              >
                <Pencil />
              </ConversationActionButton>
              <ConversationActionButton
                type="button"
                disabled={pendingSectionId !== null}
                onClick={() => void removeSection(section)}
                aria-label={t(
                  "navigation.sidebar.conversations.section.delete.action",
                  {
                    name: section.name,
                    defaultValue: "删除分组 {{name}}",
                  },
                )}
                title={t(
                  "navigation.sidebar.conversations.section.delete.action",
                  {
                    name: section.name,
                    defaultValue: "删除分组 {{name}}",
                  },
                )}
                data-testid="app-sidebar-thread-section-delete"
              >
                <Trash2 />
              </ConversationActionButton>
            </ConversationSectionActions>
          ) : null}
        </ConversationSectionHeader>
        <ConversationList>
          {sessions.map((session) => renderConversationRow(session))}
        </ConversationList>
      </ConversationSection>
    ));

  const projectsSection = (
    <ConversationSection>
      <ConversationSectionHeader>
        <ConversationSectionTitle>
          {projectsTitleLabel}
        </ConversationSectionTitle>
      </ConversationSectionHeader>
      <ConversationList data-testid="app-sidebar-project-conversations">
        {recentLoading ? (
          <AppSidebarConversationEmptyState text={loadingRecentLabel} />
        ) : (
          <AppSidebarProjectConversationGroups
            projectSections={activeConversationGroups.projectSections}
            collapsedProjectIds={collapsedProjectIds}
            newProjectConversationLabel={newProjectConversationLabel}
            projectMoreActionsLabel={projectMoreActionsLabel}
            formatNewProjectConversationForLabel={(projectName) =>
              t("navigation.sidebar.conversations.newProjectConversationFor", {
                title: projectName,
                defaultValue: "在 {{title}} 新建对话",
              })
            }
            formatOpenProjectMenuLabel={(projectName) =>
              t("navigation.sidebar.conversations.projectMenu.open", {
                title: projectName,
                defaultValue: "打开 {{title}} 项目菜单",
              })
            }
            renderConversationRow={renderConversationRow}
            onCreateConversation={onCreateConversation}
            onToggleProjectCollapsed={toggleProjectCollapsed}
            onOpenProjectMenu={openProjectMenu}
          />
        )}
      </ConversationList>
    </ConversationSection>
  );

  const conversationsSection = (
    <ConversationSection>
      <ConversationSectionHeader>
        <ConversationSectionTitle>
          {standaloneTitleLabel}
        </ConversationSectionTitle>
        <ConversationSectionActions>
          <ConversationActionButton
            type="button"
            disabled={pendingSectionId !== null}
            onClick={() => void createSection()}
            aria-label={createSectionLabel}
            title={createSectionLabel}
            data-testid="app-sidebar-new-thread-section-button"
          >
            <FolderPlus />
          </ConversationActionButton>
          {onImportConversation ? (
            <ConversationActionButton
              type="button"
              onClick={() => onImportConversation()}
              aria-label={importConversationLabel}
              title={importConversationLabel}
              data-testid="app-sidebar-import-conversation-button"
            >
              <FileInput />
            </ConversationActionButton>
          ) : null}
          <ConversationActionButton
            type="button"
            onClick={() => onCreateConversation()}
            aria-label={newConversationLabel}
            title={newConversationLabel}
            data-testid="app-sidebar-new-conversation-button"
          >
            <MessageSquarePlus />
          </ConversationActionButton>
        </ConversationSectionActions>
      </ConversationSectionHeader>
      <ConversationList data-testid="app-sidebar-recent-conversations">
        {recentLoading ? (
          <AppSidebarConversationEmptyState text={loadingRecentLabel} />
        ) : activeConversationGroups.standaloneSessions.length > 0 ? (
          activeConversationGroups.standaloneSessions.map((session) =>
            renderConversationRow(session),
          )
        ) : (
          <AppSidebarConversationEmptyState text={emptyStandaloneLabel} />
        )}
        {hasMoreRecent ? (
          <ConversationListMoreButton type="button" onClick={onShowMoreRecent}>
            {moreRecentLabel}
          </ConversationListMoreButton>
        ) : null}
      </ConversationList>
    </ConversationSection>
  );

  return (
    <ConversationShelf data-testid="app-sidebar-conversation-shelf">
      {threadSections}
      {projectsSection}
      {conversationsSection}

      <AppSidebarConversationMenus
        conversationMenuState={menuState}
        projectMenuState={projectMenuState}
        resolveSessionTitle={resolveLocalizedSessionTitle}
        onCloseMenus={closeMenus}
        onTogglePinned={onTogglePinned}
        onMoveToSection={onMoveToSection}
        onRenameConversation={onRenameConversation}
        onDeleteConversation={onDeleteConversation}
        onToggleArchive={onToggleArchive}
        onToggleProjectPin={onToggleProjectPin}
        onRevealProject={onRevealProject}
        onCreateProjectWorktree={onCreateProjectWorktree}
        onRenameProject={onRenameProject}
        onRemoveProject={onRemoveProject}
        onImportConversation={onImportConversation}
        threadSections={activeConversationGroups.threadSections.map(
          ({ section }) => section,
        )}
        conversationLabels={{
          ariaLabel: (title) =>
            t("navigation.sidebar.conversations.menu.ariaLabel", {
              title,
              defaultValue: "{{title}} 操作菜单",
            }),
          rename: renameActionLabel,
          pin: pinActionLabel,
          unpin: unpinActionLabel,
          archive: archiveActionLabel,
          moveToSection: moveToSectionActionLabel,
          moveToSectionBack: moveToSectionBackLabel,
          unsectioned: unsectionedLabel,
          delete: deleteActionLabel,
        }}
        projectLabels={{
          ariaLabel: (title) =>
            t("navigation.sidebar.conversations.projectMenu.ariaLabel", {
              title,
              defaultValue: "{{title}} 项目菜单",
            }),
          pin: projectPinActionLabel,
          unpin: projectUnpinActionLabel,
          reveal: projectRevealActionLabel,
          createWorktree: projectWorktreeActionLabel,
          importConversation: importProjectConversationLabel,
          importConversationFor: (title) =>
            t("navigation.sidebar.conversations.importProjectConversationFor", {
              title,
              defaultValue: "Import local history to {{title}}",
            }),
          rename: projectRenameActionLabel,
          remove: projectRemoveActionLabel,
        }}
      />
    </ConversationShelf>
  );
}
