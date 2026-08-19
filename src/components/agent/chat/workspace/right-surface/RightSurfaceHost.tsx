import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";
import type {
  RightSurfaceDefinition,
  WorkspaceRightSurfaceKind,
} from "./rightSurfaceTypes";

interface RightSurfaceHostProps {
  activeSurface: WorkspaceRightSurfaceKind | null;
  definitions: readonly RightSurfaceDefinition[];
  openSurfaces?: readonly WorkspaceRightSurfaceKind[];
  onSelectSurface?: (kind: WorkspaceRightSurfaceKind) => void;
  tabMode?: "auto" | "hidden";
}

const RIGHT_SURFACE_TAB_LABELS: Record<
  WorkspaceRightSurfaceKind,
  { key: string; fallback: string }
> = {
  workbench: {
    key: "agentChat.rightSurface.tabs.workbench",
    fallback: "工作台",
  },
  appSurface: {
    key: "agentChat.rightSurface.tabs.appSurface",
    fallback: "Plugin",
  },
  articleWorkspace: {
    key: "agentChat.rightSurface.tabs.articleWorkspace",
    fallback: "文章编辑器",
  },
  expertInfo: {
    key: "agentChat.rightSurface.tabs.expertInfo",
    fallback: "专家信息",
  },
  browser: {
    key: "agentChat.rightSurface.tabs.browser",
    fallback: "浏览器",
  },
  files: {
    key: "agentChat.rightSurface.tabs.files",
    fallback: "文件",
  },
  shell: {
    key: "agentChat.rightSurface.tabs.shell",
    fallback: "Shell",
  },
  harness: {
    key: "agentChat.rightSurface.tabs.harness",
    fallback: "运行",
  },
  trace: {
    key: "agentChat.rightSurface.tabs.trace",
    fallback: "Trace",
  },
};

export function RightSurfaceHost({
  activeSurface,
  definitions,
  openSurfaces,
  onSelectSurface,
  tabMode = "auto",
}: RightSurfaceHostProps): ReactNode {
  const { t } = useTranslation("agent");
  if (!activeSurface) {
    return null;
  }

  const definition = definitions.find((item) => item.kind === activeSurface);
  if (!definition) {
    return null;
  }
  const definitionByKind = new Map(
    definitions.map((item) => [item.kind, item] as const),
  );
  const tabs = (openSurfaces?.length ? openSurfaces : [activeSurface]).filter(
    (kind, index, surfaces) =>
      definitionByKind.has(kind) && surfaces.indexOf(kind) === index,
  );

  return (
    <div
      className="flex h-full min-h-0 w-full flex-col overflow-hidden"
      data-testid="workspace-right-surface-host"
      data-surface={activeSurface}
    >
      {tabMode !== "hidden" && tabs.length > 1 ? (
        <div
          className="flex h-10 shrink-0 items-center gap-1 overflow-x-auto border-b border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] px-2"
          data-testid="workspace-right-surface-tabs"
          role="tablist"
        >
          {tabs.map((kind) => {
            const label = t(RIGHT_SURFACE_TAB_LABELS[kind].key, {
              defaultValue: RIGHT_SURFACE_TAB_LABELS[kind].fallback,
            });
            const tabLabel = definitionByKind.get(kind)?.label?.trim() || label;
            const active = kind === activeSurface;
            return (
              <button
                key={kind}
                type="button"
                className={cn(
                  "inline-flex h-7 shrink-0 items-center rounded-xl border px-2.5 text-xs font-medium transition",
                  active
                    ? "border-[color:var(--lime-surface-border-strong)] bg-[color:var(--lime-chrome-tab-active-surface)] text-[color:var(--lime-text-strong)]"
                    : "border-transparent bg-transparent text-[color:var(--lime-text-muted)] hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-text-strong)]",
                )}
                aria-selected={active}
                aria-label={tabLabel}
                data-testid={`workspace-right-surface-tab-${kind}`}
                role="tab"
                onClick={() => {
                  if (!active) {
                    onSelectSurface?.(kind);
                  }
                }}
              >
                {tabLabel}
              </button>
            );
          })}
        </div>
      ) : null}
      <div
        className="min-h-0 flex-1 w-full overflow-hidden"
        data-testid="workspace-right-surface-active-pane"
      >
        {definition.render({ activeSurface })}
      </div>
    </div>
  );
}
