import { memo, useEffect, useMemo, useState } from "react";
import {
  Code2,
  FileText,
  FolderOpen,
  GitCompare,
  Globe2,
  ListChecks,
  Plus,
  TerminalSquare,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { resolveCanvasWorkbenchToolTabKind } from "../CanvasWorkbenchLayoutState";
import type {
  CanvasWorkbenchNewToolTab,
  CanvasWorkbenchTab,
} from "../../CanvasWorkbenchLayout";

type CanvasWorkbenchTranslation = (
  key: string,
  options?: Record<string, unknown>,
) => string;

export interface CanvasWorkbenchTopTab {
  key: CanvasWorkbenchTab;
  label: string;
  badge?: string;
  badgeTone?: "slate" | "sky" | "rose";
  disabled?: boolean;
  closable?: boolean;
}

export interface CanvasWorkbenchNewTabAction {
  key: CanvasWorkbenchNewToolTab;
  label: string;
  shortcut?: string;
}

interface CanvasWorkbenchTopTabsProps {
  activeTab: CanvasWorkbenchTab;
  tabs: CanvasWorkbenchTopTab[];
  newTabActions: CanvasWorkbenchNewTabAction[];
  translateWorkbench: CanvasWorkbenchTranslation;
  onSelectTab: (tab: CanvasWorkbenchTab) => void;
  onNewToolTab: (tab: CanvasWorkbenchNewToolTab) => void;
  onCloseTab?: (tab: CanvasWorkbenchTab) => void;
}

function resolveTabIcon(key: CanvasWorkbenchTab | CanvasWorkbenchNewToolTab) {
  const toolTabKind = resolveCanvasWorkbenchToolTabKind(key);
  if (key === "markdown") {
    return <FileText className="h-3.5 w-3.5 shrink-0" />;
  }
  if (key === "html") {
    return <Globe2 className="h-3.5 w-3.5 shrink-0" />;
  }
  if (key === "code") {
    return <Code2 className="h-3.5 w-3.5 shrink-0" />;
  }
  if (key === "changes") {
    return <GitCompare className="h-3.5 w-3.5 shrink-0" />;
  }
  if (key === "outputs" || toolTabKind === "terminal") {
    return <TerminalSquare className="h-3.5 w-3.5 shrink-0" />;
  }
  if (key === "logs") {
    return <ListChecks className="h-3.5 w-3.5 shrink-0" />;
  }
  if (key === "workspace" || toolTabKind === "project-files") {
    return <FolderOpen className="h-3.5 w-3.5 shrink-0" />;
  }
  return <FileText className="h-3.5 w-3.5 shrink-0" />;
}

function resolveBadgeClassName(tone?: "slate" | "sky" | "rose") {
  if (tone === "rose") {
    return "bg-rose-50 text-rose-700";
  }
  if (tone === "sky") {
    return "bg-sky-50 text-sky-700";
  }
  return "bg-slate-100 text-slate-600";
}

export const CanvasWorkbenchTopTabs = memo(function CanvasWorkbenchTopTabs({
  activeTab,
  tabs,
  newTabActions,
  translateWorkbench,
  onSelectTab,
  onNewToolTab,
  onCloseTab,
}: CanvasWorkbenchTopTabsProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const visibleTabs = useMemo(() => tabs, [tabs]);

  useEffect(() => {
    setMenuOpen(false);
  }, [activeTab]);

  if (visibleTabs.length === 0) {
    return null;
  }

  const renderTabButton = ({
    key,
    label,
    badge,
    badgeTone,
    disabled = false,
    closable = false,
  }: CanvasWorkbenchTopTab) => {
    const active = activeTab === key;
    const toolTabKind = resolveCanvasWorkbenchToolTabKind(key);
    const tabClassName = cn(
      "inline-flex h-8 shrink-0 items-center gap-1.5 rounded-[7px] border text-[13px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-45",
      active && !disabled
        ? "border-slate-300 bg-white text-slate-950 shadow-sm shadow-slate-950/5"
        : "border-transparent bg-transparent text-slate-600 hover:border-slate-200 hover:bg-white/70 hover:text-slate-950",
      closable ? "max-w-[220px] px-1.5 pr-1" : "px-2.5",
    );
    const tabContent = (
      <>
        <span className={cn(active ? "text-slate-500" : "text-slate-400")}>
          {resolveTabIcon(key)}
        </span>
        <span className="truncate">{label}</span>
        {badge ? (
          <span
            className={cn(
              "rounded-[5px] px-1.5 py-0.5 text-[10px] font-semibold",
              resolveBadgeClassName(badgeTone),
            )}
          >
            {badge}
          </span>
        ) : null}
      </>
    );

    if (closable && onCloseTab) {
      return (
        <div
          key={key}
          className={tabClassName}
          data-canvas-tab-key={key}
          data-canvas-tab-kind={toolTabKind || key}
        >
          <button
            type="button"
            aria-label={translateWorkbench(
              "agentChat.canvasWorkbench.tabs.switchAria",
              { label },
            )}
            role="tab"
            aria-selected={active}
            disabled={disabled}
            title={label}
            onClick={() => {
              if (disabled) {
                return;
              }
              onSelectTab(key);
              setMenuOpen(false);
            }}
            className="inline-flex min-w-0 flex-1 items-center gap-1.5 outline-none"
          >
            {tabContent}
          </button>
          <button
            type="button"
            aria-label={translateWorkbench(
              "agentChat.canvasWorkbench.tabs.closeAria",
              { label },
            )}
            title={translateWorkbench(
              "agentChat.canvasWorkbench.tabs.closeAria",
              { label },
            )}
            onClick={(event) => {
              event.stopPropagation();
              onCloseTab(key);
              setMenuOpen(false);
            }}
            className="ml-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-[5px] text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-700"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      );
    }

    return (
      <button
        key={key}
        type="button"
        aria-label={translateWorkbench(
          "agentChat.canvasWorkbench.tabs.switchAria",
          { label },
        )}
        role="tab"
        aria-selected={active}
        disabled={disabled}
        title={label}
        onClick={() => {
          if (disabled) {
            return;
          }
          onSelectTab(key);
          setMenuOpen(false);
        }}
        data-canvas-tab-key={key}
        data-canvas-tab-kind={toolTabKind || key}
        className={tabClassName}
      >
        {tabContent}
      </button>
    );
  };

  return (
    <div
      role="tablist"
      data-testid="canvas-workbench-top-tabs"
      className="inline-flex h-9 max-w-full items-center gap-1"
    >
      <div
        className="flex min-w-0 items-center gap-1 overflow-x-auto overflow-y-visible"
        data-testid="canvas-workbench-direct-tabs"
      >
        {visibleTabs.map((tab) => renderTabButton(tab))}
      </div>
      <div className="relative shrink-0">
        <button
          type="button"
          aria-label={translateWorkbench(
            "agentChat.canvasWorkbench.tabs.newTabAria",
          )}
          title={translateWorkbench(
            "agentChat.canvasWorkbench.tabs.newTabAria",
          )}
          aria-haspopup="menu"
          aria-expanded={menuOpen}
          data-testid="canvas-workbench-tab-menu-trigger"
          onClick={() => setMenuOpen((open) => !open)}
          className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[7px] border border-transparent text-slate-500 transition-colors hover:border-slate-200 hover:bg-white/70 hover:text-slate-950"
        >
          <Plus className="h-3.5 w-3.5" />
        </button>
        {menuOpen ? (
          <div
            role="menu"
            className="absolute left-0 top-[calc(100%+5px)] z-[80] w-[250px] rounded-[12px] border border-slate-200/90 bg-white p-1.5 shadow-xl shadow-slate-950/10"
            data-testid="canvas-workbench-tab-menu"
          >
            {newTabActions.map((action) => (
              <button
                key={action.key}
                type="button"
                aria-label={translateWorkbench(
                  "agentChat.canvasWorkbench.tabs.newToolAria",
                  { label: action.label },
                )}
                onClick={() => {
                  onNewToolTab(action.key);
                  setMenuOpen(false);
                }}
                role="menuitem"
                className="flex h-8 w-full items-center justify-between gap-3 rounded-[7px] px-2 text-left text-[13px] font-medium text-slate-600 transition-colors hover:bg-slate-50 hover:text-slate-950"
              >
                <span className="flex min-w-0 items-center gap-2">
                  <span className="text-slate-400">
                    {resolveTabIcon(action.key)}
                  </span>
                  <span className="truncate">{action.label}</span>
                </span>
                {action.shortcut ? (
                  <span className="shrink-0 font-mono text-[11px] font-medium text-slate-400">
                    {action.shortcut}
                  </span>
                ) : null}
              </button>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
});

CanvasWorkbenchTopTabs.displayName = "CanvasWorkbenchTopTabs";
