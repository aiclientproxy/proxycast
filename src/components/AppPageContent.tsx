/**
 * 应用页面分发层
 *
 * 负责根据当前页面类型渲染对应主内容，避免主入口继续膨胀。
 */

import { lazy, useCallback } from "react";
import type {
  AgentPageParams,
  PluginsPageParams,
  ExpertsPageParams,
  KnowledgePageParams,
  Page,
  PageParams,
  ResourcesPageParams,
  ScheduledTasksPageParams,
  SettingsPageParams,
  SkillsPageParams,
} from "@/types/page";
import type { AgentBackgroundSessionRuntimeSnapshot } from "./agent/chat";
import { ScheduledTasksPage } from "./scheduled-tasks/ScheduledTasksPage";
import { ImConfigPage } from "./channels/ImConfigPage";
import { SettingsPageV2 } from "./settings-v2";
import { PluginWorkspaceTabs } from "@/features/plugin/ui/PluginWorkspaceTabs";

const columnPageStyle = {
  flex: 1,
  minHeight: 0,
  display: "flex",
  flexDirection: "column",
} as const;

const loadResourcesPage = () =>
  import("./resources").then((module) => ({
    default: module.ResourcesPage,
  }));
const loadSkillsWorkspacePage = () =>
  import("./skills").then((module) => ({
    default: module.SkillsWorkspacePage,
  }));
const loadKnowledgePage = () =>
  import("@/features/knowledge").then((module) => ({
    default: module.KnowledgePage,
  }));
const loadPluginsPage = () =>
  import("@/features/plugin/ui/PluginCatalogPage").then((module) => ({
    default: module.PluginCatalogPage,
  }));

const loadExpertPlazaPage = () =>
  import("./experts").then((module) => ({
    default: module.ExpertPlazaPage,
  }));
const loadAgentChatPage = () =>
  import("./agent/chat").then((module) => ({
    default: module.AgentChatPage,
  }));

const ResourcesPage = lazy(loadResourcesPage);
const SkillsWorkspacePage = lazy(loadSkillsWorkspacePage);
const KnowledgePage = lazy(loadKnowledgePage);
const PluginsPage = lazy(loadPluginsPage);
const ExpertPlazaPage = lazy(loadExpertPlazaPage);
const AgentChatPage = lazy(loadAgentChatPage);

interface AppPageContentProps {
  currentPage: Page;
  pageParams: PageParams;
  requestedPage?: Page;
  requestedPageParams?: PageParams;
  onNavigate: (page: Page, params?: PageParams) => void;
  onAgentHasMessagesChange: (hasMessages: boolean) => void;
  onAgentSessionChange?: (sessionId: string | null) => void;
  onAgentStreamingChange?: (isStreaming: boolean) => void;
  onBackgroundSessionRuntimeChange?: (
    snapshot: AgentBackgroundSessionRuntimeSnapshot | null,
  ) => void;
}

type PluginWorkspacePage = Extract<Page, "plugins" | "skills" | "experts">;

function resolvePluginWorkspaceProjectId(
  activePage: PluginWorkspacePage,
  pageParams: PageParams,
): string | undefined {
  if (activePage === "skills") {
    return (pageParams as SkillsPageParams).creationProjectId;
  }
  if (activePage === "experts") {
    const expertsPageParams = pageParams as ExpertsPageParams;
    return expertsPageParams.currentProjectId ?? expertsPageParams.projectId;
  }
  if (activePage === "plugins") {
    return (pageParams as PluginsPageParams).currentProjectId;
  }
  return undefined;
}

function buildPluginWorkspacePageParams(
  activePage: PluginWorkspacePage,
  pageParams: PageParams,
  targetPage: PluginWorkspacePage,
): PageParams | undefined {
  const projectId = resolvePluginWorkspaceProjectId(activePage, pageParams);
  if (!projectId) {
    return undefined;
  }
  if (targetPage === "skills") {
    return { creationProjectId: projectId } satisfies SkillsPageParams;
  }
  if (targetPage === "experts") {
    return {
      currentProjectId: projectId,
      projectId,
    } satisfies ExpertsPageParams;
  }
  return undefined;
}

export function AppPageContent({
  currentPage,
  pageParams,
  requestedPage,
  requestedPageParams,
  onNavigate,
  onAgentHasMessagesChange,
  onAgentSessionChange,
  onAgentStreamingChange,
  onBackgroundSessionRuntimeChange,
}: AppPageContentProps) {
  const activePage = requestedPage ?? currentPage;
  const activePageParams = requestedPageParams ?? pageParams;
  const handlePluginWorkspaceNavigate = (targetPage: PluginWorkspacePage) => {
    onNavigate(
      targetPage,
      buildPluginWorkspacePageParams(
        activePage as PluginWorkspacePage,
        activePageParams,
        targetPage,
      ),
    );
  };
  const handleAgentSessionChange = useCallback(
    (sessionId: string | null) => {
      const normalizedSessionId = sessionId?.trim();
      onAgentSessionChange?.(normalizedSessionId || null);
    },
    [onAgentSessionChange],
  );

  if (activePage === "scheduled-tasks") {
    return (
      <div style={columnPageStyle}>
        <ScheduledTasksPage
          onNavigate={onNavigate}
          pageParams={activePageParams as ScheduledTasksPageParams}
        />
      </div>
    );
  }

  if (activePage === "channels") {
    return (
      <div style={columnPageStyle}>
        <div className="flex-1 overflow-auto px-6 py-6">
          <div className="mx-auto w-full max-w-[1440px]">
            <ImConfigPage />
          </div>
        </div>
      </div>
    );
  }

  if (activePage === "agent") {
    const agentPageParams = activePageParams as AgentPageParams;

    return (
      <div style={columnPageStyle}>
        <AgentChatPage
          onNavigate={onNavigate}
          projectId={agentPageParams.projectId}
          contentId={agentPageParams.contentId}
          initialSessionId={agentPageParams.initialSessionId}
          initialSceneAppExecutionSummary={
            agentPageParams.initialSceneAppExecutionSummary
          }
          initialRequestMetadata={agentPageParams.initialRequestMetadata}
          initialAutoSendRequestMetadata={
            agentPageParams.initialAutoSendRequestMetadata
          }
          autoRunInitialPromptOnMount={
            agentPageParams.autoRunInitialPromptOnMount
          }
          initialUserPrompt={agentPageParams.initialUserPrompt}
          initialUserImages={agentPageParams.initialUserImages}
          initialCreationMode={agentPageParams.initialCreationMode}
          initialSessionName={agentPageParams.initialSessionName}
          entryBannerMessage={agentPageParams.entryBannerMessage}
          immersiveHome={agentPageParams.immersiveHome}
          openBrowserAssistOnMount={agentPageParams.openBrowserAssistOnMount}
          initialSiteSkillLaunch={agentPageParams.initialSiteSkillLaunch}
          initialPendingServiceSkillLaunch={
            agentPageParams.initialPendingServiceSkillLaunch
          }
          initialInputCapability={agentPageParams.initialInputCapability}
          preferHomeForInitialInputCapability={
            agentPageParams.preferHomeForInitialInputCapability
          }
          initialKnowledgePackSelection={
            agentPageParams.initialKnowledgePackSelection
          }
          initialProjectFileOpenTarget={
            agentPageParams.initialProjectFileOpenTarget
          }
          theme={agentPageParams.theme}
          lockTheme={agentPageParams.lockTheme}
          fromResources={agentPageParams.fromResources}
          agentEntry={agentPageParams.agentEntry}
          showChatPanel={
            agentPageParams.agentEntry !== "new-task" &&
            !agentPageParams.immersiveHome
          }
          newChatAt={agentPageParams.newChatAt}
          expertAgentLaunch={agentPageParams.expertAgentLaunch}
          onHasMessagesChange={onAgentHasMessagesChange}
          onSessionChange={handleAgentSessionChange}
          onAgentStreamingChange={onAgentStreamingChange}
          onBackgroundSessionRuntimeChange={onBackgroundSessionRuntimeChange}
        />
      </div>
    );
  }

  if (activePage === "resources") {
    return (
      <div style={columnPageStyle}>
        <ResourcesPage
          onNavigate={onNavigate}
          pageParams={activePageParams as ResourcesPageParams}
        />
      </div>
    );
  }

  if (activePage === "skills") {
    return (
      <PluginWorkspaceTabs
        activePage="skills"
        onNavigate={handlePluginWorkspaceNavigate}
      >
        <SkillsWorkspacePage
          onNavigate={onNavigate}
          pageParams={activePageParams as SkillsPageParams}
        />
      </PluginWorkspaceTabs>
    );
  }

  if (activePage === "plugins") {
    return (
      <PluginWorkspaceTabs
        activePage="plugins"
        onNavigate={handlePluginWorkspaceNavigate}
      >
        <PluginsPage pageParams={activePageParams as PluginsPageParams} />
      </PluginWorkspaceTabs>
    );
  }

  if (activePage === "experts") {
    const expertsPageParams = activePageParams as ExpertsPageParams;
    return (
      <PluginWorkspaceTabs
        activePage="experts"
        onNavigate={handlePluginWorkspaceNavigate}
      >
        <ExpertPlazaPage
          onNavigate={onNavigate}
          currentProjectId={
            expertsPageParams.currentProjectId ?? expertsPageParams.projectId
          }
        />
      </PluginWorkspaceTabs>
    );
  }

  if (activePage === "knowledge") {
    return (
      <div style={{ ...columnPageStyle, overflow: "hidden" }}>
        <KnowledgePage
          onNavigate={onNavigate}
          pageParams={activePageParams as KnowledgePageParams}
        />
      </div>
    );
  }

  if (activePage === "settings") {
    return (
      <div style={columnPageStyle}>
        <SettingsPageV2
          onNavigate={onNavigate}
          initialTab={(activePageParams as SettingsPageParams).tab}
          initialProviderView={
            (activePageParams as SettingsPageParams).providerView
          }
          initialProviderFocus={
            (activePageParams as SettingsPageParams).providerFocus
          }
          initialExecutionPolicyFocus={
            (activePageParams as SettingsPageParams).executionPolicyFocus
          }
        />
      </div>
    );
  }

  return null;
}
