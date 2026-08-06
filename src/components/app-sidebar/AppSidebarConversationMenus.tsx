import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import styled from "styled-components";
import {
  Archive,
  ArrowLeft,
  Check,
  ChevronRight,
  FileInput,
  FolderInput,
  FolderOpen,
  FolderPlus,
  Pencil,
  Pin,
  Trash2,
} from "lucide-react";
import type {
  AgentSessionInfo,
  AgentSessionSection,
} from "@/lib/api/agentRuntime/sessionTypes";
import type { SidebarOpenedProjectSummary } from "@/components/app-sidebar/sidebarConversationGroups";
import { resolveProjectDisplayName } from "@/components/app-sidebar/sidebarProjectDisplayName";
import {
  isPinnedThreadSession,
  PINNED_THREAD_SECTION,
} from "@/lib/api/threadSections";

export const CONVERSATION_MENU_WIDTH = 188;
export const CONVERSATION_MENU_APPROX_HEIGHT = 252;
export const CONVERSATION_MENU_VIEWPORT_MARGIN = 12;

export type ConversationMenuState = {
  session: AgentSessionInfo;
  top: number;
  left: number;
} | null;

export type ProjectMenuState = {
  project: SidebarOpenedProjectSummary;
  top: number;
  left: number;
} | null;

interface ConversationMenuLabels {
  ariaLabel: (title: string) => string;
  rename: string;
  pin: string;
  unpin: string;
  archive: string;
  moveToSection: string;
  moveToSectionBack: string;
  unsectioned: string;
  delete: string;
}

interface ProjectMenuLabels {
  ariaLabel: (title: string) => string;
  pin: string;
  unpin: string;
  reveal: string;
  createWorktree: string;
  importConversation: string;
  importConversationFor: (title: string) => string;
  rename: string;
  remove: string;
}

interface AppSidebarConversationMenusProps {
  conversationMenuState: ConversationMenuState;
  projectMenuState: ProjectMenuState;
  resolveSessionTitle: (session: AgentSessionInfo) => string;
  onCloseMenus: () => void;
  onTogglePinned: (session: AgentSessionInfo) => void;
  onMoveToSection: (
    session: AgentSessionInfo,
    section: AgentSessionSection | null,
  ) => void;
  onRenameConversation?: (session: AgentSessionInfo) => void;
  onDeleteConversation?: (session: AgentSessionInfo) => void;
  onToggleArchive: (session: AgentSessionInfo, archived: boolean) => void;
  onToggleProjectPin?: (project: SidebarOpenedProjectSummary) => void;
  onRevealProject?: (project: SidebarOpenedProjectSummary) => void;
  onCreateProjectWorktree?: (project: SidebarOpenedProjectSummary) => void;
  onRenameProject?: (project: SidebarOpenedProjectSummary) => void;
  onRemoveProject?: (project: SidebarOpenedProjectSummary) => void;
  onImportConversation?: (project?: SidebarOpenedProjectSummary) => void;
  threadSections: AgentSessionSection[];
  conversationLabels: ConversationMenuLabels;
  projectLabels: ProjectMenuLabels;
}

const ConversationMenuSurface = styled.div`
  position: fixed;
  z-index: 110;
  width: ${CONVERSATION_MENU_WIDTH}px;
  padding: 8px;
  border-radius: 16px;
  border: 1px solid var(--lime-card-subtle-border, rgba(226, 240, 226, 0.9));
  background: var(--lime-surface, #ffffff);
  color: var(--lime-text-strong, #0f172a);
  max-height: ${CONVERSATION_MENU_APPROX_HEIGHT}px;
  overflow-y: auto;
  box-shadow:
    0 22px 64px rgba(15, 23, 42, 0.18),
    0 1px 0 rgba(255, 255, 255, 0.76) inset;
`;

const ConversationMenuItemTrailing = styled.span`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-left: auto;
`;

const ConversationMenuItem = styled.button<{ $danger?: boolean }>`
  width: 100%;
  min-height: 36px;
  border: none;
  border-radius: 10px;
  background: transparent;
  color: ${({ $danger }) =>
    $danger
      ? "var(--lime-danger, #b91c1c)"
      : "var(--lime-text-strong, #0f172a)"};
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 10px;
  cursor: pointer;
  text-align: left;
  font-size: 13px;
  font-weight: 650;
  transition:
    background-color 0.16s ease,
    color 0.16s ease;

  &:hover {
    background: ${({ $danger }) =>
      $danger
        ? "var(--lime-danger-soft, #fff1f2)"
        : "var(--lime-surface-hover, #f4fdf4)"};
  }

  &:disabled {
    color: var(--sidebar-muted);
    cursor: not-allowed;
    opacity: 0.52;
  }

  &:disabled:hover {
    background: transparent;
  }

  svg {
    width: 15px;
    height: 15px;
    flex-shrink: 0;
    color: ${({ $danger }) =>
      $danger ? "var(--lime-danger, #b91c1c)" : "var(--sidebar-muted)"};
  }
`;

export function AppSidebarConversationMenus({
  conversationMenuState,
  projectMenuState,
  resolveSessionTitle,
  onCloseMenus,
  onTogglePinned,
  onMoveToSection,
  onRenameConversation,
  onDeleteConversation,
  onToggleArchive,
  onToggleProjectPin,
  onRevealProject,
  onCreateProjectWorktree,
  onRenameProject,
  onRemoveProject,
  onImportConversation,
  threadSections,
  conversationLabels,
  projectLabels,
}: AppSidebarConversationMenusProps) {
  const [sectionPickerOpen, setSectionPickerOpen] = useState(false);
  const conversationSessionId = conversationMenuState?.session.id ?? null;

  useEffect(() => {
    setSectionPickerOpen(false);
  }, [conversationSessionId]);

  if (typeof document === "undefined") {
    return null;
  }

  const runMenuAction = (action: () => void) => {
    onCloseMenus();
    action();
  };

  const customThreadSections = threadSections.filter(
    (section) => section.id !== PINNED_THREAD_SECTION.id,
  );

  const sectionPicker = conversationMenuState ? (
    <>
      <ConversationMenuItem
        type="button"
        role="menuitem"
        data-testid="app-sidebar-conversation-menu-section-back"
        onClick={() => setSectionPickerOpen(false)}
      >
        <ArrowLeft />
        {conversationLabels.moveToSectionBack}
      </ConversationMenuItem>
      {[null, ...customThreadSections].map((section) => {
        const sectionId = section?.id ?? null;
        const selected =
          (conversationMenuState.session.section?.id ?? null) === sectionId;
        return (
          <ConversationMenuItem
            key={sectionId ?? "unsectioned"}
            type="button"
            role="menuitemradio"
            aria-checked={selected}
            data-testid="app-sidebar-conversation-menu-section-option"
            data-section-id={sectionId ?? "unsectioned"}
            onClick={() =>
              runMenuAction(() =>
                onMoveToSection(conversationMenuState.session, section),
              )
            }
          >
            <FolderInput />
            {section?.name ?? conversationLabels.unsectioned}
            {selected ? (
              <ConversationMenuItemTrailing>
                <Check />
              </ConversationMenuItemTrailing>
            ) : null}
          </ConversationMenuItem>
        );
      })}
    </>
  ) : null;

  const conversationMenu = conversationMenuState
    ? createPortal(
        <ConversationMenuSurface
          role="menu"
          aria-label={conversationLabels.ariaLabel(
            resolveSessionTitle(conversationMenuState.session),
          )}
          style={{
            top: conversationMenuState.top,
            left: conversationMenuState.left,
          }}
          data-testid="app-sidebar-conversation-menu"
          onClick={(event) => event.stopPropagation()}
        >
          {sectionPickerOpen ? sectionPicker : null}
          {!sectionPickerOpen && onRenameConversation ? (
            <ConversationMenuItem
              type="button"
              role="menuitem"
              data-testid="app-sidebar-conversation-menu-rename"
              onClick={() =>
                runMenuAction(() =>
                  onRenameConversation(conversationMenuState.session),
                )
              }
            >
              <Pencil />
              {conversationLabels.rename}
            </ConversationMenuItem>
          ) : null}
          {!sectionPickerOpen ? (
            <ConversationMenuItem
              type="button"
              role="menuitem"
              aria-pressed={isPinnedThreadSession(
                conversationMenuState.session,
              )}
              data-testid="app-sidebar-conversation-menu-pin"
              onClick={() =>
                runMenuAction(() =>
                  onTogglePinned(conversationMenuState.session),
                )
              }
            >
              <Pin />
              {isPinnedThreadSession(conversationMenuState.session)
                ? conversationLabels.unpin
                : conversationLabels.pin}
            </ConversationMenuItem>
          ) : null}
          {!sectionPickerOpen && customThreadSections.length > 0 ? (
            <ConversationMenuItem
              type="button"
              role="menuitem"
              data-testid="app-sidebar-conversation-menu-move-section"
              onClick={() => setSectionPickerOpen(true)}
            >
              <FolderInput />
              {conversationLabels.moveToSection}
              <ConversationMenuItemTrailing>
                <ChevronRight />
              </ConversationMenuItemTrailing>
            </ConversationMenuItem>
          ) : null}
          {!sectionPickerOpen ? (
            <ConversationMenuItem
              type="button"
              role="menuitem"
              data-testid="app-sidebar-conversation-menu-archive"
              onClick={() =>
                runMenuAction(() =>
                  onToggleArchive(conversationMenuState.session, true),
                )
              }
            >
              <Archive />
              {conversationLabels.archive}
            </ConversationMenuItem>
          ) : null}
          {!sectionPickerOpen && onDeleteConversation ? (
            <ConversationMenuItem
              type="button"
              role="menuitem"
              $danger
              data-testid="app-sidebar-conversation-menu-delete"
              onClick={() =>
                runMenuAction(() =>
                  onDeleteConversation(conversationMenuState.session),
                )
              }
            >
              <Trash2 />
              {conversationLabels.delete}
            </ConversationMenuItem>
          ) : null}
        </ConversationMenuSurface>,
        document.body,
      )
    : null;

  const projectMenu = projectMenuState
    ? createPortal(
        <ConversationMenuSurface
          role="menu"
          aria-label={projectLabels.ariaLabel(
            resolveProjectDisplayName(projectMenuState.project),
          )}
          style={{ top: projectMenuState.top, left: projectMenuState.left }}
          data-testid="app-sidebar-project-menu"
          onClick={(event) => event.stopPropagation()}
        >
          {onToggleProjectPin ? (
            <ConversationMenuItem
              type="button"
              role="menuitem"
              data-testid="app-sidebar-project-menu-pin"
              onClick={() =>
                runMenuAction(() =>
                  onToggleProjectPin(projectMenuState.project),
                )
              }
            >
              <Pin />
              {projectMenuState.project.isFavorite
                ? projectLabels.unpin
                : projectLabels.pin}
            </ConversationMenuItem>
          ) : null}
          {onRevealProject ? (
            <ConversationMenuItem
              type="button"
              role="menuitem"
              data-testid="app-sidebar-project-menu-reveal"
              onClick={() =>
                runMenuAction(() => onRevealProject(projectMenuState.project))
              }
            >
              <FolderOpen />
              {projectLabels.reveal}
            </ConversationMenuItem>
          ) : null}
          {onCreateProjectWorktree ? (
            <ConversationMenuItem
              type="button"
              role="menuitem"
              data-testid="app-sidebar-project-menu-worktree"
              onClick={() =>
                runMenuAction(() =>
                  onCreateProjectWorktree(projectMenuState.project),
                )
              }
            >
              <FolderPlus />
              {projectLabels.createWorktree}
            </ConversationMenuItem>
          ) : null}
          {onImportConversation ? (
            <ConversationMenuItem
              type="button"
              role="menuitem"
              aria-label={projectLabels.importConversationFor(
                resolveProjectDisplayName(projectMenuState.project),
              )}
              title={projectLabels.importConversationFor(
                resolveProjectDisplayName(projectMenuState.project),
              )}
              data-testid="app-sidebar-project-menu-import-conversation"
              onClick={() =>
                runMenuAction(() =>
                  onImportConversation(projectMenuState.project),
                )
              }
            >
              <FileInput />
              {projectLabels.importConversation}
            </ConversationMenuItem>
          ) : null}
          {onRenameProject ? (
            <ConversationMenuItem
              type="button"
              role="menuitem"
              data-testid="app-sidebar-project-menu-rename"
              onClick={() =>
                runMenuAction(() => onRenameProject(projectMenuState.project))
              }
            >
              <Pencil />
              {projectLabels.rename}
            </ConversationMenuItem>
          ) : null}
          {onRemoveProject ? (
            <ConversationMenuItem
              type="button"
              role="menuitem"
              $danger
              data-testid="app-sidebar-project-menu-remove"
              onClick={() =>
                runMenuAction(() => onRemoveProject(projectMenuState.project))
              }
            >
              <Trash2 />
              {projectLabels.remove}
            </ConversationMenuItem>
          ) : null}
        </ConversationMenuSurface>,
        document.body,
      )
    : null;

  return (
    <>
      {conversationMenu}
      {projectMenu}
    </>
  );
}
