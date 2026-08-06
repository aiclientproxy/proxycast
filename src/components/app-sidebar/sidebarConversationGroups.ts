import type {
  AgentSessionInfo,
  AgentSessionSection,
} from "@/lib/api/agentRuntime/sessionTypes";
import { PINNED_THREAD_SECTION } from "@/lib/api/threadSections";

export interface SidebarOpenedProjectSummary {
  id: string;
  name: string;
  rootPath?: string | null;
  isFavorite?: boolean;
}

export interface SidebarConversationProjectSection {
  project: SidebarOpenedProjectSummary;
  sessions: AgentSessionInfo[];
}

export interface SidebarConversationThreadSection {
  section: AgentSessionSection;
  sessions: AgentSessionInfo[];
}

interface BuildSidebarConversationGroupsParams {
  sessions: AgentSessionInfo[];
  openedProjects: SidebarOpenedProjectSummary[];
  threadSections?: AgentSessionSection[] | null;
}

export interface SidebarConversationGroups {
  threadSections: SidebarConversationThreadSection[];
  projectSections: SidebarConversationProjectSection[];
  standaloneSessions: AgentSessionInfo[];
}

function normalizeId(value?: string | null): string | null {
  const normalized = value?.trim();
  return normalized ? normalized : null;
}

function normalizePath(value?: string | null): string | null {
  const normalized = value?.trim().replace(/[\\/]+$/u, "");
  return normalized ? normalized : null;
}

function sessionBelongsToProject(
  session: AgentSessionInfo,
  project: SidebarOpenedProjectSummary,
): boolean {
  if (normalizeId(session.workspace_id) === normalizeId(project.id)) {
    return true;
  }

  const projectRoot = normalizePath(project.rootPath);
  const sessionCwd = normalizePath(session.working_dir);
  return Boolean(projectRoot && sessionCwd && projectRoot === sessionCwd);
}

function dedupeOpenedProjects(
  openedProjects: SidebarOpenedProjectSummary[],
): SidebarOpenedProjectSummary[] {
  const seen = new Set<string>();
  return openedProjects.filter((project) => {
    const projectKey =
      normalizePath(project.rootPath) ?? normalizeId(project.id);
    if (!projectKey || seen.has(projectKey)) {
      return false;
    }
    seen.add(projectKey);
    return true;
  });
}

export function buildSidebarConversationGroups({
  sessions,
  openedProjects,
  threadSections: threadSectionCatalog = null,
}: BuildSidebarConversationGroupsParams): SidebarConversationGroups {
  const normalizedOpenedProjects = dedupeOpenedProjects(openedProjects);
  const scopedSessions = sessions.filter((session) => !session.archived_at);
  const sessionSectionsById = new Map<
    string,
    SidebarConversationThreadSection
  >();
  for (const session of scopedSessions) {
    const sectionId = normalizeId(session.section?.id);
    const sectionName = session.section?.name.trim();
    if (!sectionId || !sectionName) {
      continue;
    }
    const current = sessionSectionsById.get(sectionId);
    if (current) {
      current.sessions.push(session);
    } else {
      sessionSectionsById.set(sectionId, {
        section: { id: sectionId, name: sectionName },
        sessions: [session],
      });
    }
  }
  const normalizedThreadSectionCatalog = threadSectionCatalog
    ?.map((section) => ({
      id: normalizeId(section.id),
      name: section.name.trim(),
    }))
    .filter((section): section is AgentSessionSection =>
      Boolean(section.id && section.name),
    );
  const threadSections = normalizedThreadSectionCatalog
    ? normalizedThreadSectionCatalog.map((section) => ({
        section,
        sessions: sessionSectionsById.get(section.id)?.sessions ?? [],
      }))
    : [...sessionSectionsById.values()].sort(
        (left, right) =>
          Number(right.section.id === PINNED_THREAD_SECTION.id) -
          Number(left.section.id === PINNED_THREAD_SECTION.id),
      );
  const knownThreadSectionIds = new Set(
    threadSections.map(({ section }) => section.id),
  );
  const unsectionedSessions = scopedSessions.filter((session) => {
    const sectionId = normalizeId(session.section?.id);
    return !sectionId || !knownThreadSectionIds.has(sectionId);
  });

  const projectSections = normalizedOpenedProjects.map((project) => {
    return {
      project,
      sessions: unsectionedSessions.filter((session) =>
        sessionBelongsToProject(session, project),
      ),
    };
  });

  const projectSessionIds = new Set(
    projectSections.flatMap((section) =>
      section.sessions.map((session) => session.id),
    ),
  );
  const standaloneSessions = unsectionedSessions.filter(
    (session) =>
      !normalizeId(session.workspace_id) && !projectSessionIds.has(session.id),
  );

  return {
    threadSections,
    projectSections,
    standaloneSessions,
  };
}

export function flattenSidebarConversationGroups({
  threadSections,
  projectSections,
  standaloneSessions,
}: SidebarConversationGroups): AgentSessionInfo[] {
  return [
    ...threadSections.flatMap((section) => section.sessions),
    ...projectSections.flatMap((section) => section.sessions),
    ...standaloneSessions,
  ];
}
