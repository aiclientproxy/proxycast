import { memo, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import type { BrowserTabState } from "@/lib/api/browserTab";
import type { BrowserWorkspaceOwner } from "@/lib/api/browserWorkspace";
import { resolveWorkspaceBrowserControlPresentation } from "../../workspaceBrowserControlMode";
import { BrowserWorkspace } from "./BrowserWorkspace";

export interface RightSurfaceBrowserPanelProps {
  ensureOwner?: () => Promise<BrowserWorkspaceOwner | null>;
  initialUrl?: string | null;
  runtimeSessionId?: string | null;
  threadId: string;
  controlMode?: string | null;
  lifecycleState?: string | null;
  active?: boolean;
  onNavigate?: (url: string, title?: string | null) => void;
}

export const RightSurfaceBrowserPanel = memo(function RightSurfaceBrowserPanel({
  ensureOwner,
  initialUrl,
  runtimeSessionId,
  threadId,
  controlMode,
  lifecycleState,
  active = true,
  onNavigate,
}: RightSurfaceBrowserPanelProps) {
  const { t } = useTranslation("agent");
  const [selectedState, setSelectedState] = useState<BrowserTabState | null>(
    null,
  );

  const control = useMemo(
    () =>
      resolveWorkspaceBrowserControlPresentation({
        controlMode:
          selectedState?.controlOwner === "human_takeover"
            ? "human_takeover"
            : selectedState?.controlOwner === "user"
              ? "human"
              : (selectedState?.controlOwner ?? controlMode),
        lifecycleState,
      }),
    [controlMode, lifecycleState, selectedState?.controlOwner],
  );

  if (!active) {
    return null;
  }

  const overlayClassName =
    control.owner === "human"
      ? "border-amber-300 bg-amber-50/95 text-amber-950"
      : "border-sky-300 bg-sky-50/95 text-sky-950";

  return (
    <div
      className="relative h-full min-h-0 bg-[color:var(--lime-surface)]"
      data-testid="right-surface-browser-panel"
      data-browser-control-mode={control.rawControlMode ?? ""}
      data-browser-control-owner={control.owner}
      data-browser-human-takeover={control.humanTakeover ? "true" : "false"}
      data-browser-lifecycle-state={control.rawLifecycleState ?? ""}
      data-browser-session-id={selectedState?.browserSessionId ?? ""}
      data-browser-tab-id={selectedState?.tabId ?? ""}
      data-browser-thread-id={threadId}
      data-browser-web-contents-id={selectedState?.webContentsId ?? ""}
    >
      {control.overlayVisible && control.labelKey && control.detailKey ? (
        <div
          aria-live="polite"
          className={`pointer-events-none absolute bottom-3 right-3 z-10 max-w-[min(360px,calc(100%-24px))] rounded-md border px-3 py-2 text-xs shadow-sm ${overlayClassName}`}
          data-testid="right-surface-browser-control-overlay"
        >
          <div className="font-medium leading-4">
            {t(control.labelKey as never)}
          </div>
          <div className="mt-0.5 text-[11px] leading-4 opacity-80">
            {t(control.detailKey as never)}
          </div>
        </div>
      ) : null}
      <BrowserWorkspace
        active={active}
        ensureOwner={ensureOwner}
        initialUrl={initialUrl}
        runtimeSessionId={runtimeSessionId}
        threadId={threadId}
        onNavigate={onNavigate}
        onSelectedStateChange={setSelectedState}
      />
    </div>
  );
});
