import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { Page } from "@/types/page";

const PLUGIN_WORKSPACE_TABS = [
  {
    page: "plugins",
    labelKey: "navigation.pluginWorkspace.tabs.plugins",
    fallbackLabel: "插件",
  },
  {
    page: "skills",
    labelKey: "navigation.pluginWorkspace.tabs.skills",
    fallbackLabel: "Skills",
  },
  {
    page: "experts",
    labelKey: "navigation.pluginWorkspace.tabs.experts",
    fallbackLabel: "专家",
  },
] as const satisfies ReadonlyArray<{
  page: Extract<Page, "plugins" | "skills" | "experts">;
  labelKey: string;
  fallbackLabel: string;
}>;

type PluginWorkspacePage = (typeof PLUGIN_WORKSPACE_TABS)[number]["page"];

interface PluginWorkspaceTabsProps {
  activePage: PluginWorkspacePage;
  children: ReactNode;
  onNavigate: (page: PluginWorkspacePage) => void;
}

export function PluginWorkspaceTabs({
  activePage,
  children,
  onNavigate,
}: PluginWorkspaceTabsProps) {
  const { t } = useTranslation("navigation");

  return (
    <div className="lime-workbench-theme-scope flex min-h-0 flex-1 flex-col overflow-hidden bg-[color:var(--lime-app-bg)]">
      <nav className="flex h-12 shrink-0 items-center border-b border-[color:var(--lime-surface-border)] bg-[color:var(--lime-app-bg)] px-5 lg:px-8">
        <div
          className="relative z-[1001] flex items-center gap-1 [app-region:no-drag] [-webkit-app-region:no-drag]"
          role="tablist"
          aria-label={t("navigation.pluginWorkspace.tabsLabel", "插件内容")}
          data-testid="plugin-workspace-tabs"
        >
          {PLUGIN_WORKSPACE_TABS.map((tab) => {
            const active = activePage === tab.page;
            const label = t(tab.labelKey, tab.fallbackLabel);

            return (
              <button
                key={tab.page}
                type="button"
                role="tab"
                aria-selected={active}
                className={`inline-flex h-8 items-center justify-center rounded-md px-3 text-sm font-medium transition-colors ${
                  active
                    ? "bg-[color:var(--lime-surface-hover)] text-[color:var(--lime-text-strong)]"
                    : "text-[color:var(--lime-text-muted)] hover:bg-[color:var(--lime-surface-hover)] hover:text-[color:var(--lime-text-strong)]"
                }`}
                data-testid={`plugin-workspace-tab-${tab.page}`}
                onClick={() => {
                  if (!active) {
                    onNavigate(tab.page);
                  }
                }}
              >
                {label}
              </button>
            );
          })}
        </div>
      </nav>
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        {children}
      </div>
    </div>
  );
}
