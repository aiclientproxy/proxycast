/**
 * 设置页面主布局组件
 *
 * 采用左侧边栏 + 右侧内容的布局
 * 参考成熟产品的设置布局设计
 */

import {
  useState,
  lazy,
  Suspense,
  type ReactNode,
  useEffect,
  useRef,
  useCallback,
} from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import styled from "styled-components";
import { SettingsSidebar } from "./SettingsSidebar";
import { SettingsTabs } from "@/types/settings";
import {
  Page,
  PageParams,
  type ExecutionPolicyFocusContext,
  type ProviderSettingsFocusContext,
  type SettingsProviderView,
} from "@/types/page";
import { buildHomeAgentParams } from "@/lib/workspace/navigation";
import { shouldReserveMacWindowControls } from "@/lib/windowControls";
import { SettingsHomePage } from "../home";
import { resolveOemCloudRuntimeContext } from "@/lib/api/oemCloudRuntime";
import { Home } from "lucide-react";

const SETTINGS_SIDEBAR_WIDTH_PX = 240;

const AppearanceSettings = lazy(() =>
  import("../general/appearance").then((module) => ({
    default: module.AppearanceSettings,
  })),
);
const MemorySettings = lazy(() =>
  import("../general/memory").then((module) => ({
    default: module.MemorySettings,
  })),
);
const ArchivedConversationsSettings = lazy(() =>
  import("../general/archived-conversations").then((module) => ({
    default: module.ArchivedConversationsSettings,
  })),
);
const DeveloperLabSettings = lazy(() =>
  import("../system/developer-lab").then((module) => ({
    default: module.DeveloperLabSettings,
  })),
);
const AboutSection = lazy(() =>
  import("../system/about").then((module) => ({
    default: module.AboutSection,
  })),
);
const MediaServicesSettings = lazy(() =>
  import("../agent/media-services").then((module) => ({
    default: module.MediaServicesSettings,
  })),
);
const StatsSettings = lazy(() =>
  import("../account/stats").then((module) => ({
    default: module.StatsSettings,
  })),
);
const ProfileSettings = lazy(() =>
  import("../account/profile").then((module) => ({
    default: module.ProfileSettings,
  })),
);
const UserCenterSessionSettings = lazy(() =>
  import("../account/user-center-session").then((module) => ({
    default: module.UserCenterSessionSettings,
  })),
);
const CloudProviderSettings = lazy(() =>
  import("../agent/providers").then((module) => ({
    default: module.CloudProviderSettings,
  })),
);
const McpPanel = lazy(() =>
  import("@/components/mcp").then((module) => ({
    default: module.McpPanel,
  })),
);
const EnvironmentSettings = lazy(() =>
  import("../system/environment").then((module) => ({
    default: module.EnvironmentSettings,
  })),
);
const ExecutionPolicySettings = lazy(() =>
  import("../system/execution-policy").then((module) => ({
    default: module.ExecutionPolicySettings,
  })),
);
const WebSearchSettings = lazy(() =>
  import("../system/web-search").then((module) => ({
    default: module.WebSearchSettings,
  })),
);
const ChromeRelaySettings = lazy(() =>
  import("../system/chrome-relay").then((module) => ({
    default: module.ChromeRelaySettings,
  })),
);

const LayoutContainer = styled.div`
  display: flex;
  flex: 1;
  min-height: 0;
  background: var(--lime-app-bg, hsl(var(--background)));

  @media (max-width: 1200px) {
    flex-direction: column;
  }
`;

const HeaderBar = styled.div<{ $reserveWindowControls: boolean }>`
  display: grid;
  grid-template-columns: ${SETTINGS_SIDEBAR_WIDTH_PX}px;
  align-items: center;
  gap: 0;
  min-height: 86px;
  padding: ${({ $reserveWindowControls }) =>
    $reserveWindowControls ? "34px 24px 14px 0" : "24px 24px 14px 0"};
  border-bottom: 1px solid var(--lime-surface-border, hsl(var(--border)));
  background: var(--lime-app-bg, hsl(var(--background)));

  @media (max-width: 1200px) {
    grid-template-columns: minmax(0, 1fr);
    gap: 8px;
    padding: ${({ $reserveWindowControls }) =>
      $reserveWindowControls ? "34px 20px 14px" : "24px 20px 14px"};
  }

  @media (max-width: 640px) {
    grid-template-columns: minmax(0, 1fr);
    gap: 10px;
    padding: ${({ $reserveWindowControls }) =>
      $reserveWindowControls ? "34px 14px 14px" : "24px 14px 14px"};
  }
`;

const HeaderHomeButton = styled.button`
  -webkit-app-region: no-drag;
  justify-self: start;
  margin-left: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 36px;
  padding: 0 14px;
  border: 1px solid var(--lime-surface-border, hsl(var(--border)));
  border-radius: 999px;
  background: var(--lime-surface, hsl(var(--card)));
  color: var(--lime-text-muted, hsl(var(--muted-foreground)));
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  transition:
    background 0.15s ease,
    border-color 0.15s ease,
    color 0.15s ease,
    transform 0.15s ease;

  &:hover {
    border-color: var(
      --lime-surface-border-strong,
      hsl(var(--foreground) / 0.18)
    );
    background: var(--lime-surface-hover, hsl(var(--accent)));
    color: var(--lime-text-strong, hsl(var(--foreground)));
  }

  &:active {
    transform: translateY(1px);
  }

  svg {
    width: 15px;
    height: 15px;
  }

  @media (max-width: 1200px) {
    justify-self: start;
    margin-left: 0;
  }
`;

const ContentContainer = styled.main`
  flex: 1;
  min-width: 0;
  position: relative;
  isolation: isolate;
  overflow-y: auto;
  padding: 24px 32px;
  background: var(
    --lime-stage-surface-soft,
    linear-gradient(
      180deg,
      rgba(248, 250, 252, 0.96) 0%,
      rgba(244, 249, 247, 0.92) 100%
    )
  );

  &::-webkit-scrollbar {
    width: 6px;
  }

  &::-webkit-scrollbar-track {
    background: transparent;
  }

  &::-webkit-scrollbar-thumb {
    background: hsl(var(--border));
    border-radius: 3px;
  }

  @media (max-width: 1200px) {
    padding: 20px;
  }

  @media (max-width: 640px) {
    padding: 16px 12px 24px;
  }
`;

const ContentWrapper = styled.div<{ $wide: boolean }>`
  position: relative;
  z-index: 1;
  width: 100%;
  min-width: 0;
  max-width: ${({ $wide }) => ($wide ? "1440px" : "800px")};
`;

const ContentAtmosphere = styled.div`
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background:
    radial-gradient(
      circle at 8% 0%,
      var(--lime-home-glow-primary, rgba(16, 185, 129, 0.1)) 0%,
      rgba(16, 185, 129, 0) 34%
    ),
    radial-gradient(
      circle at 92% 4%,
      var(--lime-home-glow-secondary, rgba(56, 189, 248, 0.1)) 0%,
      rgba(56, 189, 248, 0) 30%
    );

  @media (max-width: 640px) {
    background:
      radial-gradient(
        circle at 10% 0%,
        var(--lime-home-glow-primary, rgba(16, 185, 129, 0.08)) 0%,
        rgba(16, 185, 129, 0) 36%
      ),
      radial-gradient(
        circle at 92% 2%,
        var(--lime-home-glow-secondary, rgba(56, 189, 248, 0.08)) 0%,
        rgba(56, 189, 248, 0) 32%
      );
  }
`;

const PlaceholderPage = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 300px;
  color: hsl(var(--muted-foreground));
  text-align: center;

  p {
    margin-top: 8px;
    font-size: 14px;
  }
`;

const LoadingPanel = styled.div`
  border: 1px solid hsl(var(--border));
  border-radius: 20px;
  background: hsl(var(--card));
  padding: 18px 20px;
  color: hsl(var(--muted-foreground));
  font-size: 14px;
  line-height: 1.6;
`;

function SettingsContentFallback({ label }: { label: string }) {
  return (
    <LoadingPanel
      role="status"
      aria-live="polite"
      aria-busy="true"
      data-testid="settings-page-loading"
    >
      {label}
    </LoadingPanel>
  );
}

function withSettingsContentFallback(
  node: ReactNode,
  label: string,
): ReactNode {
  return (
    <Suspense fallback={<SettingsContentFallback label={label} />}>
      {node}
    </Suspense>
  );
}

const ACTIVE_SETTINGS_TABS = new Set<SettingsTabs>([
  SettingsTabs.Home,
  SettingsTabs.Profile,
  SettingsTabs.Stats,
  SettingsTabs.Appearance,
  SettingsTabs.Memory,
  SettingsTabs.ArchivedConversations,
  SettingsTabs.Providers,
  SettingsTabs.MediaServices,
  SettingsTabs.McpServer,
  SettingsTabs.WebSearch,
  SettingsTabs.Environment,
  SettingsTabs.ExecutionPolicy,
  SettingsTabs.ChromeRelay,
  SettingsTabs.Developer,
  SettingsTabs.About,
]);

function resolveActiveSettingsTab(tab?: SettingsTabs): SettingsTabs {
  const canonicalTab =
    tab === SettingsTabs.Experimental ? SettingsTabs.Developer : tab;

  if (!canonicalTab || !ACTIVE_SETTINGS_TABS.has(canonicalTab)) {
    return SettingsTabs.Home;
  }
  return canonicalTab;
}

function preloadSettingsTab(tab: SettingsTabs): Promise<unknown> | null {
  switch (resolveActiveSettingsTab(tab)) {
    case SettingsTabs.Home:
      return null;
    case SettingsTabs.Profile:
      return Promise.all([
        import("../account/profile"),
        import("../account/user-center-session"),
      ]);
    case SettingsTabs.Stats:
      return import("../account/stats");
    case SettingsTabs.Appearance:
      return import("../general/appearance");
    case SettingsTabs.Memory:
      return import("../general/memory");
    case SettingsTabs.ArchivedConversations:
      return import("../general/archived-conversations");
    case SettingsTabs.Providers:
      return import("../agent/providers");
    case SettingsTabs.MediaServices:
      return import("../agent/media-services");
    case SettingsTabs.McpServer:
      return import("@/components/mcp");
    case SettingsTabs.WebSearch:
      return import("../system/web-search");
    case SettingsTabs.Environment:
      return import("../system/environment");
    case SettingsTabs.ExecutionPolicy:
      return import("../system/execution-policy");
    case SettingsTabs.ChromeRelay:
      return import("../system/chrome-relay");
    case SettingsTabs.Developer:
      return Promise.all([
        import("../system/developer-lab"),
        import("../system/developer/preload").then((module) =>
          module.preloadDeveloperDefaultSections(),
        ),
      ]);
    case SettingsTabs.About:
      return import("../system/about");
    default:
      return null;
  }
}

/**
 * 渲染设置内容
 */

function renderSettingsContent(
  tab: SettingsTabs,
  onTabChange: (tab: SettingsTabs) => void,
  t: TFunction<"settings", undefined>,
  onTabPrefetch?: (tab: SettingsTabs) => void,
  onNavigate?: (page: Page, params?: PageParams) => void,
  initialProviderView?: SettingsProviderView,
  initialProviderFocus?: ProviderSettingsFocusContext | null,
  initialExecutionPolicyFocus?: ExecutionPolicyFocusContext | null,
  activeDeveloperLabTab: "developer" | "experimental" = "developer",
): ReactNode {
  const hasManagedAccountProfile = Boolean(resolveOemCloudRuntimeContext());

  switch (tab) {
    case SettingsTabs.Home:
      return (
        <SettingsHomePage
          onTabChange={onTabChange}
          onTabPrefetch={onTabPrefetch}
          onNavigate={onNavigate}
        />
      );

    // 账号组
    case SettingsTabs.Profile:
      return withSettingsContentFallback(
        <>
          <UserCenterSessionSettings />
          {!hasManagedAccountProfile ? <ProfileSettings /> : null}
        </>,
        t("settings.layout.loading.profile"),
      );

    case SettingsTabs.Stats:
      return withSettingsContentFallback(
        <StatsSettings />,
        t("settings.layout.loading.stats"),
      );

    // 通用组
    case SettingsTabs.Appearance:
      return withSettingsContentFallback(
        <AppearanceSettings />,
        t("settings.layout.loading.appearance"),
      );

    case SettingsTabs.Memory:
      return withSettingsContentFallback(
        <MemorySettings />,
        t("settings.layout.loading.memory"),
      );

    case SettingsTabs.ArchivedConversations:
      return withSettingsContentFallback(
        <ArchivedConversationsSettings />,
        t("settings.layout.loading.archivedConversations"),
      );

    // 智能体组
    case SettingsTabs.Providers:
      return withSettingsContentFallback(
        <CloudProviderSettings
          initialView={initialProviderView}
          initialFocus={initialProviderFocus}
        />,
        t("settings.layout.loading.providers"),
      );

    case SettingsTabs.MediaServices:
      return withSettingsContentFallback(
        <MediaServicesSettings />,
        t("settings.layout.loading.mediaServices"),
      );

    // 系统组
    case SettingsTabs.McpServer:
      return withSettingsContentFallback(
        <McpPanel hideHeader />,
        t("settings.layout.loading.mcpServer"),
      );

    case SettingsTabs.WebSearch:
      return withSettingsContentFallback(
        <WebSearchSettings />,
        t("settings.layout.loading.webSearch"),
      );

    case SettingsTabs.Environment:
      return withSettingsContentFallback(
        <EnvironmentSettings />,
        t("settings.layout.loading.environment"),
      );

    case SettingsTabs.ExecutionPolicy:
      return withSettingsContentFallback(
        <ExecutionPolicySettings focus={initialExecutionPolicyFocus ?? null} />,
        t("settings.layout.loading.executionPolicy"),
      );

    case SettingsTabs.ChromeRelay:
      return withSettingsContentFallback(
        <ChromeRelaySettings />,
        t("settings.layout.loading.chromeRelay"),
      );

    case SettingsTabs.Developer:
      return withSettingsContentFallback(
        <DeveloperLabSettings initialTab={activeDeveloperLabTab} />,
        t("settings.layout.loading.developerLab"),
      );

    case SettingsTabs.About:
      return withSettingsContentFallback(
        <AboutSection />,
        t("settings.layout.loading.about"),
      );

    default:
      return (
        <PlaceholderPage>
          <p>{t("settings.layout.placeholder.notFound")}</p>
        </PlaceholderPage>
      );
  }
}

/**
 * 设置页面主组件
 */
interface SettingsLayoutV2Props {
  onNavigate?: (page: Page, params?: PageParams) => void;
  initialTab?: SettingsTabs;
  initialProviderView?: SettingsProviderView;
  initialProviderFocus?: ProviderSettingsFocusContext | null;
  initialExecutionPolicyFocus?: ExecutionPolicyFocusContext | null;
}

const WIDE_CONTENT_TABS = ACTIVE_SETTINGS_TABS;
type DeveloperLabInitialTab = "developer" | "experimental";

function resolveDeveloperLabInitialTab(
  tab?: SettingsTabs,
): DeveloperLabInitialTab {
  return tab === SettingsTabs.Experimental ? "experimental" : "developer";
}

export function SettingsLayoutV2({
  onNavigate,
  initialTab,
  initialProviderView,
  initialProviderFocus,
  initialExecutionPolicyFocus,
}: SettingsLayoutV2Props) {
  const { t } = useTranslation("settings");
  const [activeTab, setActiveTab] = useState<SettingsTabs>(
    resolveActiveSettingsTab(initialTab),
  );
  const [activeProviderView, setActiveProviderView] = useState<
    SettingsProviderView | undefined
  >(initialProviderView);
  const [activeDeveloperLabTab, setActiveDeveloperLabTab] =
    useState<DeveloperLabInitialTab>(resolveDeveloperLabInitialTab(initialTab));
  const contentContainerRef = useRef<HTMLElement | null>(null);
  const prefetchedTabsRef = useRef<Set<SettingsTabs>>(new Set());
  const reserveWindowControls = shouldReserveMacWindowControls();

  const handleTabChange = useCallback((tab: SettingsTabs) => {
    const nextTab = resolveActiveSettingsTab(tab);
    setActiveTab(nextTab);
    if (nextTab === SettingsTabs.Developer) {
      setActiveDeveloperLabTab(resolveDeveloperLabInitialTab(tab));
    }
    if (nextTab !== SettingsTabs.Providers) {
      setActiveProviderView(undefined);
    }
  }, []);

  const handleTabPrefetch = useCallback((tab: SettingsTabs) => {
    if (prefetchedTabsRef.current.has(tab)) {
      return;
    }

    const preloadTask = preloadSettingsTab(tab);
    if (!preloadTask) {
      return;
    }

    prefetchedTabsRef.current.add(tab);
    void preloadTask.catch(() => {
      prefetchedTabsRef.current.delete(tab);
    });
  }, []);

  const handleBackHome = useCallback(() => {
    onNavigate?.("agent", buildHomeAgentParams());
  }, [onNavigate]);

  useEffect(() => {
    setActiveTab(resolveActiveSettingsTab(initialTab));
    setActiveDeveloperLabTab(resolveDeveloperLabInitialTab(initialTab));
  }, [initialTab]);

  useEffect(() => {
    if (!initialTab && !initialProviderView && !initialProviderFocus) {
      return;
    }

    if ((initialTab ?? SettingsTabs.Providers) === SettingsTabs.Providers) {
      setActiveProviderView(
        initialProviderView ?? (initialProviderFocus ? "settings" : undefined),
      );
      return;
    }

    setActiveProviderView(undefined);
  }, [initialProviderFocus, initialProviderView, initialTab]);

  useEffect(() => {
    contentContainerRef.current?.scrollTo?.({ top: 0, behavior: "auto" });
  }, [activeTab]);

  return (
    <>
      {/* 设置内容 */}
      <HeaderBar
        className="lime-settings-theme-scope"
        $reserveWindowControls={reserveWindowControls}
        data-testid="settings-top-header"
        data-window-controls-reserved={String(reserveWindowControls)}
      >
        <HeaderHomeButton
          type="button"
          onClick={handleBackHome}
          aria-label={t("settings.layout.action.backHome")}
          data-testid="settings-home-button"
        >
          <Home />
          {t("settings.layout.action.backHome")}
        </HeaderHomeButton>
      </HeaderBar>
      <LayoutContainer className="lime-settings-theme-scope">
        <SettingsSidebar
          activeTab={activeTab}
          onTabChange={handleTabChange}
          onTabPrefetch={handleTabPrefetch}
        />
        <ContentContainer ref={contentContainerRef}>
          <ContentAtmosphere data-testid="settings-content-atmosphere" />
          <ContentWrapper $wide={WIDE_CONTENT_TABS.has(activeTab)}>
            {renderSettingsContent(
              activeTab,
              handleTabChange,
              t,
              handleTabPrefetch,
              onNavigate,
              activeProviderView,
              initialProviderFocus,
              initialExecutionPolicyFocus,
              activeDeveloperLabTab,
            )}
          </ContentWrapper>
        </ContentContainer>
      </LayoutContainer>
    </>
  );
}
