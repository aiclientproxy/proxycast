import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import {
  ArrowLeft,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Globe2,
  Lock,
  Plus,
  RotateCw,
  Search,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  closeBrowserTab,
  findInBrowserTab,
  goBackBrowserTab,
  goForwardBrowserTab,
  isBrowserTabHostAvailable,
  listenBrowserTabClosed,
  listenBrowserTabDownload,
  listenBrowserTabLoadFailed,
  listenBrowserTabPermissionRequest,
  listenBrowserTabState,
  mountBrowserTab,
  navigateBrowserTab,
  reloadBrowserTab,
  selectBrowserTab,
  setBrowserTabBounds,
  setBrowserTabZoom,
  stopBrowserTab,
  stopFindInBrowserTab,
  type BrowserTabDownloadEvent,
  type BrowserTabPermissionRequestEvent,
  type BrowserTabState,
} from "@/lib/api/browserTab";
import {
  createBrowserWorkspaceTabIdentity,
  openBrowserWorkspaceIdentity,
  type BrowserWorkspaceOwner,
} from "@/lib/api/browserWorkspace";
import type { EmbeddedBrowserBounds } from "@/lib/api/embeddedBrowser";
import { cn } from "@/lib/utils";
import {
  BROWSER_ZOOM_STEP,
  clampBrowserZoom,
  DEFAULT_BROWSER_URL,
  browserBoundsEqual,
  normalizeBrowserAddress,
  resolveBrowserAddressValue,
  resolveElementBounds,
} from "./browserWorkspaceModel";
import {
  BrowserWorkspaceDownloadShelf,
  BrowserWorkspaceErrorBanner,
  BrowserWorkspaceHostUnavailable,
  BrowserWorkspaceLoading,
  BrowserWorkspacePermissionBanner,
  type BrowserWorkspaceError,
} from "./BrowserWorkspaceStatus";

interface BrowserWorkspaceProps {
  active?: boolean;
  ensureOwner?: () => Promise<BrowserWorkspaceOwner | null>;
  initialUrl?: string | null;
  runtimeSessionId?: string | null;
  threadId: string;
  onNavigate?: (url: string, title?: string | null) => void;
  onSelectedStateChange?: (state: BrowserTabState | null) => void;
}

type Translate = (key: string, options?: Record<string, unknown>) => string;

export const BrowserWorkspace = memo(function BrowserWorkspace({
  active = true,
  ensureOwner,
  initialUrl,
  runtimeSessionId,
  threadId,
  onNavigate,
  onSelectedStateChange,
}: BrowserWorkspaceProps) {
  const { t } = useTranslation("agent");
  const translate = useCallback<Translate>(
    (key, options) => String(t(key as never, options as never)),
    [t],
  );
  const hostAvailable = isBrowserTabHostAvailable();
  const resolvedInitialUrl = useMemo(
    () => normalizeBrowserAddress(initialUrl || DEFAULT_BROWSER_URL),
    [initialUrl],
  );
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const lastBoundsRef = useRef<EmbeddedBrowserBounds | null>(null);
  const selectedRef = useRef<BrowserTabState | null>(null);
  const activeRef = useRef(active);
  const lastNavigationRef = useRef("");
  const [owner, setOwner] = useState<BrowserWorkspaceOwner | null>(null);
  const [identity, setIdentity] = useState<{
    browserSessionId: string;
    primaryTabId: string;
  } | null>(null);
  const [tabs, setTabs] = useState<BrowserTabState[]>([]);
  const [address, setAddress] = useState(
    resolveBrowserAddressValue(resolvedInitialUrl),
  );
  const [findVisible, setFindVisible] = useState(false);
  const [findValue, setFindValue] = useState("");
  const [download, setDownload] = useState<BrowserTabDownloadEvent | null>(
    null,
  );
  const [permission, setPermission] =
    useState<BrowserTabPermissionRequestEvent | null>(null);
  const [error, setError] = useState<BrowserWorkspaceError | null>(null);
  activeRef.current = active;

  const selected = tabs.find((tab) => tab.selected) ?? null;
  selectedRef.current = selected;

  const acceptState = useCallback(
    (state: BrowserTabState) => {
      if (
        !identity ||
        state.browserSessionId !== identity.browserSessionId ||
        state.threadId !== owner?.threadId
      ) {
        return;
      }
      setTabs((current) => {
        const normalized = state.selected
          ? current.map((tab) =>
              tab.tabId === state.tabId ? tab : { ...tab, selected: false },
            )
          : current;
        const index = normalized.findIndex((tab) => tab.tabId === state.tabId);
        if (index < 0) {
          return [...normalized, state];
        }
        const next = [...normalized];
        next[index] = state;
        return next;
      });
      setError((current) => (current?.source === "host" ? null : current));
    },
    [identity, owner?.threadId],
  );

  const run = useCallback(
    async (operation: () => Promise<BrowserTabState>) => {
      try {
        acceptState(await operation());
      } catch (reason) {
        setError({
          body: reason instanceof Error ? reason.message : String(reason),
          source: "host",
          title: translate("agentChat.browserWorkspace.loadFailedTitle"),
        });
      }
    },
    [acceptState, translate],
  );

  const syncBounds = useCallback(
    async (force = false) => {
      const element = viewportRef.current;
      const tab = selectedRef.current;
      if (!hostAvailable || !element || !tab) {
        return;
      }
      const bounds = resolveElementBounds(element);
      if (!force && browserBoundsEqual(lastBoundsRef.current, bounds)) {
        return;
      }
      lastBoundsRef.current = bounds;
      await run(() =>
        setBrowserTabBounds({
          tabId: tab.tabId,
          bounds,
          visible: activeRef.current && bounds.width > 0 && bounds.height > 0,
        }),
      );
    },
    [hostAvailable, run],
  );

  useEffect(() => {
    let disposed = false;
    if (!hostAvailable) {
      return;
    }
    void (async () => {
      const resolvedOwner =
        runtimeSessionId?.trim() && threadId.trim()
          ? {
              runtimeSessionId: runtimeSessionId.trim(),
              threadId: threadId.trim(),
            }
          : await ensureOwner?.();
      if (!resolvedOwner?.runtimeSessionId || !resolvedOwner.threadId) {
        throw new Error("Browser workspace requires a canonical conversation");
      }
      const next = await openBrowserWorkspaceIdentity(resolvedOwner);
      return { next, owner: resolvedOwner };
    })()
      .then(({ next, owner: nextOwner }) => {
        if (!disposed) {
          setOwner(nextOwner);
          setIdentity({
            browserSessionId: next.browserSessionId,
            primaryTabId: next.tabId,
          });
        }
      })
      .catch((reason) => {
        if (!disposed) {
          setError({
            body: reason instanceof Error ? reason.message : String(reason),
            source: "host",
            title: translate("agentChat.browserWorkspace.loadFailedTitle"),
          });
        }
      });
    return () => {
      disposed = true;
    };
  }, [ensureOwner, hostAvailable, runtimeSessionId, threadId, translate]);

  useEffect(() => {
    let disposed = false;
    const cleanup: Array<() => void> = [];
    const register = async (promise: Promise<() => void>) => {
      const stop = await promise;
      if (disposed) {
        stop();
      } else {
        cleanup.push(stop);
      }
    };
    void register(listenBrowserTabState(acceptState));
    void register(
      listenBrowserTabClosed((event) => {
        if (
          identity &&
          event.browserSessionId === identity.browserSessionId &&
          event.threadId === owner?.threadId
        ) {
          setTabs((current) =>
            current.filter((tab) => tab.tabId !== event.tabId),
          );
        }
      }),
    );
    void register(
      listenBrowserTabLoadFailed((event) => {
        if (
          identity &&
          event.browserSessionId === identity.browserSessionId &&
          event.threadId === owner?.threadId
        ) {
          setError(
            resolveLoadError(
              event.failureCategory,
              event.errorDescription,
              translate,
            ),
          );
        }
      }),
    );
    void register(
      listenBrowserTabDownload((event) => {
        if (
          identity &&
          event.browserSessionId === identity.browserSessionId &&
          event.threadId === owner?.threadId
        ) {
          setDownload(event);
        }
      }),
    );
    void register(
      listenBrowserTabPermissionRequest((event) => {
        if (
          identity &&
          event.browserSessionId === identity.browserSessionId &&
          event.threadId === owner?.threadId
        ) {
          setPermission(event);
        }
      }),
    );
    return () => {
      disposed = true;
      cleanup.forEach((stop) => stop());
    };
  }, [acceptState, identity, owner?.threadId, translate]);

  useEffect(() => {
    if (!hostAvailable || !identity || !owner) {
      return;
    }
    const frame = requestAnimationFrame(() => {
      const element = viewportRef.current;
      if (!element) {
        return;
      }
      const bounds = resolveElementBounds(element);
      lastBoundsRef.current = bounds;
      void run(() =>
        mountBrowserTab({
          browserSessionId: identity.browserSessionId,
          bounds,
          selected: true,
          tabId: identity.primaryTabId,
          threadId: owner.threadId,
          url: resolvedInitialUrl,
          visible: active && bounds.width > 0 && bounds.height > 0,
        }),
      );
    });
    return () => {
      cancelAnimationFrame(frame);
      const tab = selectedRef.current;
      const bounds = lastBoundsRef.current;
      if (tab && bounds) {
        void setBrowserTabBounds({ tabId: tab.tabId, bounds, visible: false });
      }
    };
  }, [active, hostAvailable, identity, owner, resolvedInitialUrl, run]);

  useEffect(() => {
    const element = viewportRef.current;
    if (!element || typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(() => void syncBounds());
    observer.observe(element);
    return () => observer.disconnect();
  }, [syncBounds]);

  useEffect(() => {
    if (selected) {
      setAddress(resolveBrowserAddressValue(selected.url));
    }
    onSelectedStateChange?.(selected);
    const navigationKey = selected
      ? `${selected.tabId}\u0000${selected.url}\u0000${selected.title}`
      : "";
    if (selected && navigationKey !== lastNavigationRef.current) {
      lastNavigationRef.current = navigationKey;
      onNavigate?.(selected.url, selected.title || null);
    }
  }, [onNavigate, onSelectedStateChange, selected]);

  useEffect(() => {
    if (selected?.tabId) {
      void syncBounds(true);
    }
  }, [selected?.tabId, syncBounds]);

  const openTab = useCallback(async () => {
    const element = viewportRef.current;
    if (!element || !identity || !owner) {
      return;
    }
    const bounds = resolveElementBounds(element);
    try {
      const next = await createBrowserWorkspaceTabIdentity({
        browserSessionId: identity.browserSessionId,
        runtimeSessionId: owner.runtimeSessionId,
        threadId: owner.threadId,
      });
      await run(() =>
        mountBrowserTab({
          browserSessionId: next.browserSessionId,
          bounds,
          selected: true,
          tabId: next.tabId,
          threadId: owner.threadId,
          url: DEFAULT_BROWSER_URL,
          visible: active,
        }),
      );
    } catch (reason) {
      setError({
        body: reason instanceof Error ? reason.message : String(reason),
        source: "host",
        title: translate("agentChat.browserWorkspace.loadFailedTitle"),
      });
    }
  }, [active, identity, owner, run, translate]);

  const selectTab = useCallback(
    async (tabId: string) => {
      const element = viewportRef.current;
      if (!element) {
        return;
      }
      await run(() =>
        selectBrowserTab({ tabId, bounds: resolveElementBounds(element) }),
      );
    },
    [run],
  );

  const submitAddress = useCallback(
    (event: FormEvent) => {
      event.preventDefault();
      if (selected) {
        const url = normalizeBrowserAddress(address);
        setAddress(url);
        void run(() => navigateBrowserTab(selected.tabId, url));
      }
    },
    [address, run, selected],
  );

  const submitFind = useCallback(
    (forward: boolean, findNext: boolean) => {
      if (selected && findValue.trim()) {
        void run(() =>
          findInBrowserTab({
            tabId: selected.tabId,
            text: findValue,
            forward,
            findNext,
          }),
        );
      }
    },
    [findValue, run, selected],
  );

  const closeFind = useCallback(() => {
    setFindVisible(false);
    setFindValue("");
    if (selected) {
      void run(() => stopFindInBrowserTab(selected.tabId));
    }
  }, [run, selected]);

  const iconButton =
    "inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-[color:var(--lime-text-muted)] transition hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-text-strong)] disabled:pointer-events-none disabled:opacity-35";
  const zoom = selected?.zoomFactor ?? 1;

  return (
    <section
      className="flex h-full min-h-0 flex-col overflow-hidden bg-[color:var(--lime-surface)]"
      data-testid="browser-workspace"
      data-browser-active-turn-id={selected?.activeTurnId ?? ""}
      data-browser-control-owner={selected?.controlOwner ?? ""}
      data-browser-page-revision={selected?.pageRevision ?? ""}
      data-browser-session-id={identity?.browserSessionId ?? ""}
      data-browser-tab-id={selected?.tabId ?? ""}
      data-browser-thread-id={owner?.threadId ?? ""}
      data-browser-view-id={selected?.viewId ?? ""}
      data-browser-web-contents-id={selected?.webContentsId ?? ""}
    >
      <div
        className="flex h-9 shrink-0 items-end gap-1 overflow-x-auto border-b border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface-muted)] px-2 pt-1"
        data-testid="browser-workspace-tabs"
        role="tablist"
      >
        {tabs.map((tab) => (
          <div
            className={cn(
              "flex h-7 min-w-[112px] max-w-[220px] items-center rounded-t-md border border-b-0",
              tab.selected
                ? "border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] text-[color:var(--lime-text-strong)]"
                : "border-transparent text-[color:var(--lime-text-muted)] hover:bg-[color:var(--lime-chrome-tab-hover)]",
            )}
            key={tab.tabId}
          >
            <button
              aria-selected={tab.selected}
              className="flex h-full min-w-0 flex-1 items-center gap-1.5 px-2 text-left text-xs"
              onClick={() => void selectTab(tab.tabId)}
              role="tab"
              type="button"
            >
              <Globe2 className="h-3.5 w-3.5 shrink-0" />
              <span className="truncate">
                {tab.title || translate("agentChat.browserWorkspace.newTab")}
              </span>
            </button>
            <button
              aria-label={translate("agentChat.browserWorkspace.closeTab")}
              className={cn(iconButton, "h-6 w-6")}
              disabled={tabs.length <= 1}
              onClick={() => void closeBrowserTab(tab.tabId)}
              title={translate("agentChat.browserWorkspace.closeTab")}
              type="button"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
        <button
          aria-label={translate("agentChat.browserWorkspace.newTab")}
          className={iconButton}
          onClick={() => void openTab()}
          title={translate("agentChat.browserWorkspace.newTab")}
          type="button"
        >
          <Plus className="h-4 w-4" />
        </button>
      </div>

      <div className="flex h-10 shrink-0 items-center gap-1 border-b border-[color:var(--lime-surface-border)] px-2">
        <button
          aria-label={translate("agentChat.browserWorkspace.back")}
          className={iconButton}
          disabled={!selected?.canGoBack}
          onClick={() =>
            selected && void run(() => goBackBrowserTab(selected.tabId))
          }
          title={translate("agentChat.browserWorkspace.back")}
          type="button"
        >
          <ArrowLeft className="h-4 w-4" />
        </button>
        <button
          aria-label={translate("agentChat.browserWorkspace.forward")}
          className={iconButton}
          disabled={!selected?.canGoForward}
          onClick={() =>
            selected && void run(() => goForwardBrowserTab(selected.tabId))
          }
          title={translate("agentChat.browserWorkspace.forward")}
          type="button"
        >
          <ArrowRight className="h-4 w-4" />
        </button>
        <button
          aria-label={translate(
            selected?.isLoading
              ? "agentChat.browserWorkspace.stop"
              : "agentChat.browserWorkspace.refresh",
          )}
          className={iconButton}
          disabled={!selected}
          onClick={() =>
            selected &&
            void run(() =>
              selected.isLoading
                ? stopBrowserTab(selected.tabId)
                : reloadBrowserTab(selected.tabId),
            )
          }
          title={translate(
            selected?.isLoading
              ? "agentChat.browserWorkspace.stop"
              : "agentChat.browserWorkspace.refresh",
          )}
          type="button"
        >
          {selected?.isLoading ? (
            <X className="h-4 w-4" />
          ) : (
            <RotateCw className="h-4 w-4" />
          )}
        </button>
        <form
          className="flex h-7 min-w-0 flex-1 items-center gap-1.5 rounded-md border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface-muted)] px-2 focus-within:border-[color:var(--lime-accent)]"
          onSubmit={submitAddress}
        >
          {selected?.url.startsWith("https://") ? (
            <Lock className="h-3.5 w-3.5 shrink-0 text-emerald-600" />
          ) : (
            <Globe2 className="h-3.5 w-3.5 shrink-0 text-[color:var(--lime-text-muted)]" />
          )}
          <input
            aria-label={translate("agentChat.browserWorkspace.address")}
            className="min-w-0 flex-1 bg-transparent text-xs text-[color:var(--lime-text-strong)] outline-none"
            onChange={(event) => setAddress(event.target.value)}
            placeholder={translate(
              "agentChat.browserWorkspace.addressPlaceholder",
            )}
            value={address}
          />
        </form>
        <button
          aria-label={translate("agentChat.browserWorkspace.find")}
          className={iconButton}
          disabled={!selected}
          onClick={() => setFindVisible((visible) => !visible)}
          title={translate("agentChat.browserWorkspace.find")}
          type="button"
        >
          <Search className="h-4 w-4" />
        </button>
        <button
          aria-label={translate("agentChat.browserWorkspace.zoomOut")}
          className={iconButton}
          disabled={!selected}
          onClick={() =>
            selected &&
            void run(() =>
              setBrowserTabZoom(
                selected.tabId,
                clampBrowserZoom(zoom - BROWSER_ZOOM_STEP),
              ),
            )
          }
          title={translate("agentChat.browserWorkspace.zoomOut")}
          type="button"
        >
          <ZoomOut className="h-4 w-4" />
        </button>
        <button
          className="h-7 min-w-10 px-1 text-[11px] text-[color:var(--lime-text-muted)] hover:text-[color:var(--lime-text-strong)]"
          disabled={!selected}
          onClick={() =>
            selected && void run(() => setBrowserTabZoom(selected.tabId, 1))
          }
          title={translate("agentChat.browserWorkspace.zoomReset")}
          type="button"
        >
          {Math.round(zoom * 100)}%
        </button>
        <button
          aria-label={translate("agentChat.browserWorkspace.zoomIn")}
          className={iconButton}
          disabled={!selected}
          onClick={() =>
            selected &&
            void run(() =>
              setBrowserTabZoom(
                selected.tabId,
                clampBrowserZoom(zoom + BROWSER_ZOOM_STEP),
              ),
            )
          }
          title={translate("agentChat.browserWorkspace.zoomIn")}
          type="button"
        >
          <ZoomIn className="h-4 w-4" />
        </button>
      </div>

      {findVisible ? (
        <form
          className="flex h-9 shrink-0 items-center gap-1 border-b border-[color:var(--lime-surface-border)] px-2"
          onSubmit={(event) => {
            event.preventDefault();
            submitFind(true, true);
          }}
        >
          <Search className="h-3.5 w-3.5 text-[color:var(--lime-text-muted)]" />
          <input
            aria-label={translate("agentChat.browserWorkspace.findInput")}
            autoFocus
            className="h-7 min-w-0 flex-1 bg-transparent text-xs outline-none"
            onChange={(event) => {
              setFindValue(event.target.value);
              if (!event.target.value) {
                closeFind();
              }
            }}
            placeholder={translate(
              "agentChat.browserWorkspace.findPlaceholder",
            )}
            value={findValue}
          />
          <span className="min-w-10 text-center text-[11px] text-[color:var(--lime-text-muted)]">
            {translate("agentChat.browserWorkspace.findMatchCount", {
              active: selected?.find?.activeMatchOrdinal ?? 0,
              total: selected?.find?.matches ?? 0,
            })}
          </span>
          <button
            className={iconButton}
            onClick={() => submitFind(false, true)}
            title={translate("agentChat.browserWorkspace.findPrevious")}
            type="button"
          >
            <ChevronUp className="h-4 w-4" />
          </button>
          <button
            className={iconButton}
            onClick={() => submitFind(true, true)}
            title={translate("agentChat.browserWorkspace.findNext")}
            type="button"
          >
            <ChevronDown className="h-4 w-4" />
          </button>
          <button
            className={iconButton}
            onClick={closeFind}
            title={translate("agentChat.browserWorkspace.closeFind")}
            type="button"
          >
            <X className="h-4 w-4" />
          </button>
        </form>
      ) : null}

      <div
        className="relative min-h-0 flex-1 bg-white"
        data-testid="browser-workspace-viewport"
        ref={viewportRef}
      >
        {!hostAvailable ? (
          <BrowserWorkspaceHostUnavailable t={translate} />
        ) : !selected && !error ? (
          <BrowserWorkspaceLoading t={translate} />
        ) : null}
        {error ? <BrowserWorkspaceErrorBanner error={error} /> : null}
        {permission ? (
          <BrowserWorkspacePermissionBanner
            permission={permission}
            t={translate}
          />
        ) : null}
        {download ? (
          <BrowserWorkspaceDownloadShelf download={download} t={translate} />
        ) : null}
      </div>
    </section>
  );
});

function resolveLoadError(
  category: string,
  message: string,
  t: Translate,
): BrowserWorkspaceError {
  const suffix =
    category === "dns"
      ? "Dns"
      : category === "tls"
        ? "Tls"
        : category === "blocked"
          ? "Blocked"
          : category === "aborted"
            ? "Aborted"
            : "";
  return {
    title: t(`agentChat.browserWorkspace.loadFailed${suffix}Title`),
    body: t(`agentChat.browserWorkspace.loadFailed${suffix}Body`, { message }),
    source: "load",
  };
}
