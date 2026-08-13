import { useCallback, type MutableRefObject } from "react";
import { toast } from "sonner";
import {
  archiveAgentRuntimeSession,
  deleteAgentRuntimeSession,
  unarchiveAgentRuntimeSession,
} from "@/lib/api/agentRuntime/sessionClient";
import { setAgentRuntimeThreadName } from "@/lib/api/agentRuntime/threadClient";
import {
  isPinnedThreadSession,
  moveThreadToSection,
  PINNED_THREAD_SECTION,
} from "@/lib/api/threadSections";
import type {
  AgentSessionInfo,
  AgentSessionSection,
} from "@/lib/api/agentRuntime/sessionTypes";
import { recordAgentUiPerformanceMetric } from "@/lib/agentUiPerformanceMetrics";
import {
  buildClawAgentParams,
  buildHomeAgentParams,
} from "@/lib/workspace/navigation";
import { requestTaskCenterDraftTask } from "@/components/agent/chat/taskCenterDraftTaskEvents";
import { requestExplicitInitialSessionNavigation } from "@/components/agent/chat/workspace/useWorkspaceInitialSessionNavigation";
import type { Page, PageParams } from "@/types/page";
import type { SidebarOpenedProjectSummary } from "@/components/app-sidebar/sidebarConversationGroups";
import {
  isSameSidebarNavigationTarget,
  serializeNavigationParams,
  type SidebarNavigationTarget,
} from "@/components/app-sidebar/sidebarNavigationTarget";

interface UseAppSidebarConversationActionsParams {
  currentProjectId: string | null;
  currentSessionId: string | null;
  conversationDisplayProjects: SidebarOpenedProjectSummary[];
  isAgentWorkspace: boolean;
  projectScopedNavigationProjectId: string | null;
  requestedNavigationTargetRef: MutableRefObject<SidebarNavigationTarget>;
  onNavigate: (page: Page, params?: PageParams) => void;
  closeSidebarSearchDialog: () => void;
  deferConversationNavigation: () => void;
  beginSidebarSessionAction: (sessionId: string) => void;
  clearSidebarSessionAction: (sessionId: string) => void;
  refreshSidebarSessions: () => Promise<void>;
  renameSidebarSessionOptimistically: (session: AgentSessionInfo) => void;
  moveSidebarSessionArchiveStateOptimistically: (
    session: AgentSessionInfo,
  ) => void;
  moveSidebarSessionSectionOptimistically: (session: AgentSessionInfo) => void;
  removeSidebarSessionOptimistically: (sessionId: string) => void;
  resolveLocalizedSessionTitle: (session: AgentSessionInfo) => string;
  renameConversationPromptLabel: string;
  renameConversationSuccessLabel: string;
  renameConversationErrorLabel: string;
  formatDeleteConversationConfirm: (title: string) => string;
  deleteConversationSuccessLabel: string;
  deleteConversationErrorLabel: string;
}

function normalizeSidebarPath(value?: string | null): string | null {
  const normalized = value?.trim().replace(/[\\/]+$/u, "");
  return normalized ? normalized : null;
}

function resolveProjectIdForSession(
  session: AgentSessionInfo,
  projects: SidebarOpenedProjectSummary[],
): string | null {
  const sessionWorkspaceId = session.workspace_id?.trim() || null;
  if (sessionWorkspaceId) {
    const matchedProject = projects.find(
      (project) => project.id === sessionWorkspaceId,
    );
    if (matchedProject) {
      return matchedProject.id;
    }
  }

  const sessionCwd = normalizeSidebarPath(session.working_dir);
  if (!sessionCwd) {
    return null;
  }
  return (
    projects.find(
      (project) => normalizeSidebarPath(project.rootPath) === sessionCwd,
    )?.id ?? null
  );
}

function navigateIfNeeded(params: {
  target: SidebarNavigationTarget;
  requestedNavigationTargetRef: MutableRefObject<SidebarNavigationTarget>;
  onNavigate: (page: Page, params?: PageParams) => void;
}) {
  const { target, requestedNavigationTargetRef, onNavigate } = params;
  if (
    isSameSidebarNavigationTarget(
      target,
      requestedNavigationTargetRef.current.page,
      requestedNavigationTargetRef.current.rawParams,
    )
  ) {
    return;
  }

  requestedNavigationTargetRef.current = target;
  onNavigate(target.page, target.rawParams);
}

export function useAppSidebarConversationActions({
  currentProjectId,
  currentSessionId,
  conversationDisplayProjects,
  isAgentWorkspace,
  projectScopedNavigationProjectId,
  requestedNavigationTargetRef,
  onNavigate,
  closeSidebarSearchDialog,
  deferConversationNavigation,
  beginSidebarSessionAction,
  clearSidebarSessionAction,
  refreshSidebarSessions,
  renameSidebarSessionOptimistically,
  moveSidebarSessionArchiveStateOptimistically,
  moveSidebarSessionSectionOptimistically,
  removeSidebarSessionOptimistically,
  resolveLocalizedSessionTitle,
  renameConversationPromptLabel,
  renameConversationSuccessLabel,
  renameConversationErrorLabel,
  formatDeleteConversationConfirm,
  deleteConversationSuccessLabel,
  deleteConversationErrorLabel,
}: UseAppSidebarConversationActionsParams) {
  const tryOpenTaskCenterDraftFromSidebar = useCallback(
    (projectId?: string | null) => {
      return (
        isAgentWorkspace &&
        requestTaskCenterDraftTask({ source: "sidebar", projectId })
      );
    },
    [isAgentWorkspace],
  );

  const navigateToHome = useCallback(() => {
    if (tryOpenTaskCenterDraftFromSidebar(currentProjectId)) {
      return;
    }

    const targetParams = buildHomeAgentParams({
      projectId: currentProjectId ?? undefined,
    });
    navigateIfNeeded({
      target: {
        page: "agent",
        rawParams: targetParams,
        paramsKey: serializeNavigationParams(targetParams),
      },
      requestedNavigationTargetRef,
      onNavigate,
    });
  }, [
    currentProjectId,
    onNavigate,
    requestedNavigationTargetRef,
    tryOpenTaskCenterDraftFromSidebar,
  ]);

  const navigateToWorkbench = useCallback(
    (fallbackSessionId?: string | null) => {
      const targetSessionId =
        currentSessionId ?? fallbackSessionId ?? undefined;
      const targetParams = buildClawAgentParams({
        projectId: projectScopedNavigationProjectId ?? undefined,
        initialSessionId: targetSessionId,
      });
      navigateIfNeeded({
        target: {
          page: "agent",
          rawParams: targetParams,
          paramsKey: serializeNavigationParams(targetParams),
        },
        requestedNavigationTargetRef,
        onNavigate,
      });
    },
    [
      currentSessionId,
      onNavigate,
      projectScopedNavigationProjectId,
      requestedNavigationTargetRef,
    ],
  );

  const navigateToConversation = useCallback(
    (session: AgentSessionInfo) => {
      deferConversationNavigation();
      requestExplicitInitialSessionNavigation(session.id);

      const sessionProjectId = resolveProjectIdForSession(
        session,
        conversationDisplayProjects,
      );
      const targetParams = buildClawAgentParams({
        ...(sessionProjectId ? { projectId: sessionProjectId } : {}),
        initialSessionId: session.id,
        initialSessionName: resolveLocalizedSessionTitle(session),
      });
      navigateIfNeeded({
        target: {
          page: "agent",
          rawParams: targetParams,
          paramsKey: serializeNavigationParams(targetParams),
        },
        requestedNavigationTargetRef,
        onNavigate,
      });
    },
    [
      conversationDisplayProjects,
      deferConversationNavigation,
      onNavigate,
      requestedNavigationTargetRef,
      resolveLocalizedSessionTitle,
    ],
  );

  const navigateToNewTask = useCallback(
    (projectId?: string | null) => {
      const normalizedProjectId = projectId?.trim() || null;
      if (
        isAgentWorkspace &&
        requestTaskCenterDraftTask({
          source: "sidebar",
          projectId: normalizedProjectId,
        })
      ) {
        return;
      }

      const targetParams = buildHomeAgentParams({
        projectId: normalizedProjectId ?? undefined,
      });
      navigateIfNeeded({
        target: {
          page: "agent",
          rawParams: targetParams,
          paramsKey: serializeNavigationParams(targetParams),
        },
        requestedNavigationTargetRef,
        onNavigate,
      });
    },
    [isAgentWorkspace, onNavigate, requestedNavigationTargetRef],
  );

  const navigateToProjectNewTask = useCallback(
    (project: SidebarOpenedProjectSummary) => {
      navigateToNewTask(project.id);
    },
    [navigateToNewTask],
  );

  const navigateToStandaloneConversation = useCallback(() => {
    navigateToNewTask(null);
  }, [navigateToNewTask]);

  const createConversationFromSearch = useCallback(() => {
    closeSidebarSearchDialog();
    navigateToStandaloneConversation();
  }, [closeSidebarSearchDialog, navigateToStandaloneConversation]);

  const navigateToConversationFromSearch = useCallback(
    (session: AgentSessionInfo) => {
      closeSidebarSearchDialog();
      recordAgentUiPerformanceMetric("sidebar.conversation.click", {
        sessionId: session.id,
        source: "sidebar_search",
        cwd: session.working_dir ?? null,
        projectId:
          resolveProjectIdForSession(session, conversationDisplayProjects) ??
          null,
      });
      navigateToConversation(session);
    },
    [
      closeSidebarSearchDialog,
      conversationDisplayProjects,
      navigateToConversation,
    ],
  );

  const renameConversation = useCallback(
    async (session: AgentSessionInfo) => {
      const currentTitle = resolveLocalizedSessionTitle(session);
      const nextTitle = window
        .prompt(renameConversationPromptLabel, currentTitle)
        ?.trim();
      if (!nextTitle || nextTitle === currentTitle) {
        return;
      }

      const nextUpdatedAt = Math.floor(Date.now() / 1000);
      const nextSession = {
        ...session,
        name: nextTitle,
        updated_at: nextUpdatedAt,
      } satisfies AgentSessionInfo;
      beginSidebarSessionAction(session.id);
      renameSidebarSessionOptimistically(nextSession);

      try {
        const threadId = session.thread_id?.trim();
        if (!threadId) {
          throw new Error(
            `canonical thread identity is missing for session ${session.id}`,
          );
        }
        await setAgentRuntimeThreadName({
          threadId,
          name: nextTitle,
        });
        toast.success(renameConversationSuccessLabel);
        await refreshSidebarSessions();
      } catch (error) {
        console.error("重命名会话失败:", error);
        toast.error(renameConversationErrorLabel);
        await refreshSidebarSessions();
      } finally {
        clearSidebarSessionAction(session.id);
      }
    },
    [
      beginSidebarSessionAction,
      clearSidebarSessionAction,
      refreshSidebarSessions,
      renameConversationErrorLabel,
      renameConversationPromptLabel,
      renameConversationSuccessLabel,
      renameSidebarSessionOptimistically,
      resolveLocalizedSessionTitle,
    ],
  );

  const toggleSessionArchive = useCallback(
    async (session: AgentSessionInfo, archived: boolean) => {
      const nextUpdatedAt = Math.floor(Date.now() / 1000);
      const nextSession = {
        ...session,
        updated_at: nextUpdatedAt,
        archived_at: archived ? nextUpdatedAt : null,
      } satisfies AgentSessionInfo;
      beginSidebarSessionAction(session.id);
      moveSidebarSessionArchiveStateOptimistically(nextSession);

      try {
        if (archived) {
          await archiveAgentRuntimeSession(session.id);
        } else {
          await unarchiveAgentRuntimeSession(session.id);
        }
        await refreshSidebarSessions();
      } catch (error) {
        console.error(archived ? "归档会话失败:" : "恢复会话失败:", error);
        await refreshSidebarSessions();
      } finally {
        clearSidebarSessionAction(session.id);
      }
    },
    [
      beginSidebarSessionAction,
      clearSidebarSessionAction,
      moveSidebarSessionArchiveStateOptimistically,
      refreshSidebarSessions,
    ],
  );

  const moveSessionToSection = useCallback(
    async (session: AgentSessionInfo, section: AgentSessionSection | null) => {
      const threadId = session.thread_id?.trim();
      if (!threadId) {
        console.error(
          `移动会话分组失败: canonical thread identity is missing for session ${session.id}`,
        );
        return;
      }
      const currentSectionId = session.section?.id.trim() || null;
      const targetSectionId = section?.id.trim() || null;
      if (currentSectionId === targetSectionId) {
        return;
      }
      const nextSession = {
        ...session,
        section: section ? { ...section } : undefined,
        section_entered_at: section ? Date.now() : undefined,
      } satisfies AgentSessionInfo;
      beginSidebarSessionAction(session.id);
      moveSidebarSessionSectionOptimistically(nextSession);

      try {
        await moveThreadToSection({
          threadId,
          sectionId: targetSectionId,
        });
        await refreshSidebarSessions();
      } catch (error) {
        console.error("移动会话分组失败:", error);
        await refreshSidebarSessions();
      } finally {
        clearSidebarSessionAction(session.id);
      }
    },
    [
      beginSidebarSessionAction,
      clearSidebarSessionAction,
      moveSidebarSessionSectionOptimistically,
      refreshSidebarSessions,
    ],
  );

  const toggleSessionPinned = useCallback(
    (session: AgentSessionInfo) =>
      moveSessionToSection(
        session,
        isPinnedThreadSession(session) ? null : PINNED_THREAD_SECTION,
      ),
    [moveSessionToSection],
  );

  const deleteConversation = useCallback(
    async (session: AgentSessionInfo) => {
      const title = resolveLocalizedSessionTitle(session);
      const confirmed = window.confirm(formatDeleteConversationConfirm(title));
      if (!confirmed) {
        return;
      }

      beginSidebarSessionAction(session.id);
      removeSidebarSessionOptimistically(session.id);

      try {
        await deleteAgentRuntimeSession(session.id);
        toast.success(deleteConversationSuccessLabel);
        if (currentSessionId === session.id) {
          navigateToStandaloneConversation();
        } else {
          await refreshSidebarSessions();
        }
      } catch (error) {
        console.error("删除会话失败:", error);
        toast.error(deleteConversationErrorLabel);
        await refreshSidebarSessions();
      } finally {
        clearSidebarSessionAction(session.id);
      }
    },
    [
      beginSidebarSessionAction,
      clearSidebarSessionAction,
      currentSessionId,
      deleteConversationErrorLabel,
      deleteConversationSuccessLabel,
      formatDeleteConversationConfirm,
      navigateToStandaloneConversation,
      refreshSidebarSessions,
      removeSidebarSessionOptimistically,
      resolveLocalizedSessionTitle,
    ],
  );

  return {
    createConversationFromSearch,
    deleteConversation,
    navigateToConversation,
    navigateToConversationFromSearch,
    navigateToHome,
    navigateToNewTask,
    navigateToProjectNewTask,
    navigateToStandaloneConversation,
    navigateToWorkbench,
    moveSessionToSection,
    toggleSessionArchive,
    toggleSessionPinned,
    tryOpenTaskCenterDraftFromSidebar,
    renameConversation,
  };
}
