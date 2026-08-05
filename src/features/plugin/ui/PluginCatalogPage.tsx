import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Check,
  ChevronDown,
  CircleAlert,
  ExternalLink,
  FolderOpen,
  LoaderCircle,
  MoreHorizontal,
  PackageCheck,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Unplug,
  X,
} from "lucide-react";
import { toast } from "sonner";
import type {
  AppServerPluginCatalogDetail,
  AppServerPluginCatalogSummary,
} from "@/lib/api/appServerTypes";
import {
  installPluginCatalog,
  listPluginCatalog,
  readPluginCatalog,
  selectPluginCatalogSource,
  setPluginCatalogEnabled,
  uninstallPluginCatalog,
} from "@/lib/api/pluginCatalog";
import type { Page, PageParams, PluginsPageParams } from "@/types/page";
import type { PluginRightSurfaceLaunchTarget } from "./pluginRightSurfaceLaunch";
import {
  detailCapabilityCount,
  filterPluginCatalog,
  listPluginCatalogSources,
  mergePluginCatalogSummary,
  type PluginCatalogView,
} from "./PluginCatalogPageViewModel";

interface InstallCandidate {
  sourcePath: string;
  summary: AppServerPluginCatalogSummary;
}

function sourceLabel(source: string, t: (key: string) => string): string {
  const key = `plugin.catalog.v2.source.${source}`;
  const translated = t(key);
  return translated === key ? source : translated;
}

function pathLeaf(path: string): string {
  const segments = path.split(/[\\/]/).filter(Boolean);
  return segments.at(-1) ?? path;
}

function formatCapabilityCount(
  count: number,
  t: (key: string, options?: Record<string, unknown>) => string,
): string {
  return t("plugin.catalog.v2.capabilityCount", { count });
}

export function PluginCatalogPage({
  pageParams,
}: {
  onNavigate?: (page: Page, params?: PageParams) => void;
  pageParams?: PluginsPageParams;
  rightSurfaceTarget?: PluginRightSurfaceLaunchTarget | null;
  rightSurfaceTargets?: PluginRightSurfaceLaunchTarget[] | null;
}) {
  const { t } = useTranslation("agent");
  const [plugins, setPlugins] = useState<AppServerPluginCatalogSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(
    pageParams?.selectedPluginId ?? null,
  );
  const [detail, setDetail] = useState<AppServerPluginCatalogDetail | null>(
    null,
  );
  const [view, setView] = useState<PluginCatalogView>(
    pageParams?.statusFilter === "installed" ? "installed" : "all",
  );
  const [query, setQuery] = useState(pageParams?.query ?? "");
  const [source, setSource] = useState("all");
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionMenuId, setActionMenuId] = useState<string | null>(null);
  const [installCandidates, setInstallCandidates] = useState<
    InstallCandidate[]
  >([]);
  const [selectedInstallId, setSelectedInstallId] = useState<string | null>(
    null,
  );
  const [uninstallTarget, setUninstallTarget] =
    useState<AppServerPluginCatalogSummary | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listPluginCatalog();
      setPlugins(response.plugins);
      setSelectedId((current) => {
        if (
          current &&
          response.plugins.some((plugin) => plugin.id === current)
        ) {
          return current;
        }
        return response.plugins[0]?.id ?? null;
      });
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    const selected = plugins.find((plugin) => plugin.id === selectedId);
    if (!selected?.installed) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    void readPluginCatalog({ pluginId: selectedId })
      .then((response) => {
        if (!cancelled) {
          setDetail(response.plugin);
        }
      })
      .catch((cause) => {
        if (!cancelled) {
          setDetail(null);
          setError(cause instanceof Error ? cause.message : String(cause));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [plugins, selectedId]);

  const filteredPlugins = useMemo(
    () => filterPluginCatalog(plugins, { query, source, view }),
    [plugins, query, source, view],
  );
  const sources = useMemo(() => listPluginCatalogSources(plugins), [plugins]);
  const selectedSummary =
    plugins.find((plugin) => plugin.id === selectedId) ?? null;
  const selectedInstallCandidate =
    installCandidates.find(
      (candidate) => candidate.summary.id === selectedInstallId,
    ) ??
    installCandidates[0] ??
    null;

  const handleInstallFromPath = useCallback(async () => {
    try {
      const sourcePath = await selectPluginCatalogSource({
        title: t("plugin.catalog.v2.selectSource"),
      });
      if (!sourcePath) {
        return;
      }
      const response = await listPluginCatalog({
        marketplacePaths: [sourcePath],
      });
      const candidates = response.plugins.map((summary) => ({
        summary,
        sourcePath: summary.sourceUri,
      }));
      if (!candidates.length) {
        throw new Error(t("plugin.catalog.v2.noManifest"));
      }
      setInstallCandidates(candidates);
      setSelectedInstallId(candidates[0].summary.id);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [t]);

  const handleInstall = useCallback(async () => {
    if (!selectedInstallCandidate) {
      return;
    }
    const pluginId = selectedInstallCandidate.summary.id;
    setBusyId(pluginId);
    try {
      const response = await installPluginCatalog({
        sourcePath: selectedInstallCandidate.sourcePath,
        marketplaceId: selectedInstallCandidate.summary.marketplaceId,
        source: selectedInstallCandidate.summary.source,
        expectedDigest: selectedInstallCandidate.summary.contentDigest,
      });
      setPlugins((current) =>
        mergePluginCatalogSummary(current, response.plugin),
      );
      setSelectedId(response.plugin.id);
      setInstallCandidates([]);
      setSelectedInstallId(null);
      toast.success(t("plugin.catalog.v2.installSuccess"));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusyId(null);
    }
  }, [selectedInstallCandidate, t]);

  const handleToggle = useCallback(
    async (plugin: AppServerPluginCatalogSummary) => {
      setBusyId(plugin.id);
      setActionMenuId(null);
      try {
        const response = await setPluginCatalogEnabled({
          pluginId: plugin.id,
          enabled: !plugin.enabled,
        });
        setPlugins((current) =>
          mergePluginCatalogSummary(current, response.plugin),
        );
        if (detail?.summary.id === plugin.id) {
          setDetail((current) =>
            current ? { ...current, summary: response.plugin } : current,
          );
        }
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        setBusyId(null);
      }
    },
    [detail],
  );

  const handleUninstall = useCallback(async () => {
    if (!uninstallTarget) {
      return;
    }
    const pluginId = uninstallTarget.id;
    setBusyId(pluginId);
    try {
      const response = await uninstallPluginCatalog({ pluginId });
      if (response.uninstalled) {
        setPlugins((current) =>
          current.filter((plugin) => plugin.id !== pluginId),
        );
        setSelectedId((current) => (current === pluginId ? null : current));
        setDetail(null);
        toast.success(t("plugin.catalog.v2.uninstallSuccess"));
      }
      setUninstallTarget(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusyId(null);
    }
  }, [t, uninstallTarget]);

  const clearError = () => setError(null);

  return (
    <div className="lime-workbench-theme-scope lime-workbench-surface-scope flex min-h-0 flex-1 flex-col overflow-auto bg-[color:var(--lime-surface-soft)]">
      <div className="mx-auto flex w-full max-w-[1440px] flex-1 flex-col gap-6 px-6 py-7 xl:px-8">
        <header className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--lime-text-muted)]">
              <PackageCheck size={15} />
              {t("plugin.catalog.v2.eyebrow")}
            </div>
            <h1 className="mt-2 text-[28px] font-semibold text-[color:var(--lime-text-strong)]">
              {t("plugin.catalog.v2.title")}
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-[color:var(--lime-text-muted)]">
              {t("plugin.catalog.v2.description")}
            </p>
          </div>
          <button
            type="button"
            className="inline-flex h-9 shrink-0 items-center justify-center gap-2 rounded-full bg-[color:var(--lime-text-strong)] px-5 text-sm font-semibold text-[color:var(--lime-surface)] transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
            onClick={() => void handleInstallFromPath()}
            disabled={Boolean(busyId)}
            data-testid="plugin-v2-install-local"
          >
            <FolderOpen size={16} />
            {t("plugin.catalog.v2.installLocal")}
          </button>
        </header>

        <section className="flex flex-col gap-3 border-b border-[color:var(--lime-surface-border)] pb-4 xl:flex-row xl:items-center xl:justify-between">
          <div className="flex flex-wrap items-center gap-2">
            {(["all", "installed"] as const).map((nextView) => (
              <button
                key={nextView}
                type="button"
                className={`inline-flex h-9 items-center gap-2 rounded-full px-4 text-sm font-semibold transition ${
                  view === nextView
                    ? "bg-[color:var(--lime-text-strong)] text-[color:var(--lime-surface)]"
                    : "text-[color:var(--lime-text-muted)] hover:bg-[color:var(--lime-surface-hover)] hover:text-[color:var(--lime-text-strong)]"
                }`}
                onClick={() => setView(nextView)}
                data-testid={`plugin-v2-view-${nextView}`}
              >
                {t(`plugin.catalog.v2.view.${nextView}`)}
                <span className="text-xs opacity-70">
                  {nextView === "all"
                    ? plugins.length
                    : plugins.filter((plugin) => plugin.installed).length}
                </span>
              </button>
            ))}
          </div>
          <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-center">
            <label className="relative min-w-0 sm:w-[310px]">
              <Search
                size={17}
                className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-[color:var(--lime-text-muted)]"
              />
              <input
                value={query}
                onChange={(event) => setQuery(event.currentTarget.value)}
                className="h-9 w-full rounded-full border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] pl-10 pr-4 text-sm text-[color:var(--lime-text-strong)] outline-none placeholder:text-[color:var(--lime-text-muted)] focus:border-[color:var(--lime-surface-border-strong)]"
                placeholder={t("plugin.catalog.v2.searchPlaceholder")}
                aria-label={t("plugin.catalog.v2.searchLabel")}
                data-testid="plugin-v2-search"
              />
            </label>
            <label className="relative flex min-w-[150px] items-center">
              <SlidersHorizontal
                size={15}
                className="pointer-events-none absolute left-3 text-[color:var(--lime-text-muted)]"
              />
              <select
                value={source}
                onChange={(event) => setSource(event.currentTarget.value)}
                className="h-9 w-full appearance-none rounded-full border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] pl-9 pr-8 text-sm font-medium text-[color:var(--lime-text-strong)] outline-none focus:border-[color:var(--lime-surface-border-strong)]"
                aria-label={t("plugin.catalog.v2.sourceLabel")}
                data-testid="plugin-v2-source-filter"
              >
                <option value="all">{t("plugin.catalog.v2.source.all")}</option>
                {sources.map((item) => (
                  <option key={item} value={item}>
                    {sourceLabel(item, t)}
                  </option>
                ))}
              </select>
              <ChevronDown
                size={15}
                className="pointer-events-none absolute right-3 text-[color:var(--lime-text-muted)]"
              />
            </label>
            <button
              type="button"
              className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] text-[color:var(--lime-text-muted)] transition hover:bg-[color:var(--lime-surface-hover)] hover:text-[color:var(--lime-text-strong)] disabled:cursor-not-allowed disabled:opacity-50"
              onClick={() => void refresh()}
              disabled={loading}
              title={t("plugin.catalog.v2.refresh")}
              aria-label={t("plugin.catalog.v2.refresh")}
              data-testid="plugin-v2-refresh"
            >
              <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
            </button>
          </div>
        </section>

        {error ? (
          <div
            className="flex items-start justify-between gap-3 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800"
            role="alert"
            data-testid="plugin-v2-error"
          >
            <div className="flex min-w-0 items-start gap-2">
              <CircleAlert size={17} className="mt-0.5 shrink-0" />
              <span className="break-words">{error}</span>
            </div>
            <button
              type="button"
              className="shrink-0 rounded-full p-1 hover:bg-rose-100"
              onClick={clearError}
              title={t("plugin.catalog.v2.dismissError")}
              aria-label={t("plugin.catalog.v2.dismissError")}
            >
              <X size={15} />
            </button>
          </div>
        ) : null}

        {loading ? (
          <div
            className="flex min-h-[280px] items-center justify-center text-sm text-[color:var(--lime-text-muted)]"
            data-testid="plugin-v2-loading"
          >
            <LoaderCircle size={18} className="mr-2 animate-spin" />
            {t("plugin.catalog.v2.loading")}
          </div>
        ) : filteredPlugins.length === 0 ? (
          <div
            className="flex min-h-[280px] flex-col items-center justify-center rounded-lg border border-dashed border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] px-6 text-center"
            data-testid="plugin-v2-empty"
          >
            <Sparkles
              size={24}
              className="text-[color:var(--lime-text-muted)]"
            />
            <p className="mt-3 text-sm font-semibold text-[color:var(--lime-text-strong)]">
              {plugins.length === 0
                ? t("plugin.catalog.v2.empty.noPlugins")
                : t("plugin.catalog.v2.empty.noMatches")}
            </p>
            <p className="mt-1 max-w-md text-sm leading-6 text-[color:var(--lime-text-muted)]">
              {t("plugin.catalog.v2.empty.description")}
            </p>
          </div>
        ) : (
          <div className="grid min-w-0 grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(320px,380px)]">
            <section
              className="grid min-w-0 grid-cols-1 content-start gap-3 md:grid-cols-2"
              data-testid="plugin-v2-list"
            >
              {filteredPlugins.map((plugin) => {
                const isSelected = selectedId === plugin.id;
                const isBusy = busyId === plugin.id;
                return (
                  <article
                    key={`${plugin.id}:${plugin.version}`}
                    className={`min-w-0 rounded-lg border bg-[color:var(--lime-surface)] p-4 transition ${
                      isSelected
                        ? "border-[color:var(--lime-surface-border-strong)] shadow-sm"
                        : "border-[color:var(--lime-surface-border)] hover:border-[color:var(--lime-surface-border-strong)]"
                    }`}
                    data-testid={`plugin-v2-card-${plugin.id}`}
                  >
                    <div className="flex min-w-0 items-start justify-between gap-3">
                      <button
                        type="button"
                        className="min-w-0 text-left"
                        onClick={() => setSelectedId(plugin.id)}
                        data-testid={`plugin-v2-select-${plugin.id}`}
                      >
                        <h2 className="truncate text-base font-semibold text-[color:var(--lime-text-strong)]">
                          {plugin.name}
                        </h2>
                        <p className="mt-1 truncate text-xs text-[color:var(--lime-text-muted)]">
                          {plugin.id} · {t("plugin.catalog.v2.version", {
                            version: plugin.version,
                          })}
                        </p>
                      </button>
                      {plugin.installed ? (
                        <div className="relative shrink-0">
                          <button
                            type="button"
                            className="inline-flex h-8 w-8 items-center justify-center rounded-full text-[color:var(--lime-text-muted)] hover:bg-[color:var(--lime-surface-hover)] hover:text-[color:var(--lime-text-strong)]"
                            onClick={() =>
                              setActionMenuId((current) =>
                                current === plugin.id ? null : plugin.id,
                              )
                            }
                            aria-haspopup="menu"
                            aria-expanded={actionMenuId === plugin.id}
                            aria-label={t("plugin.catalog.v2.moreActions")}
                            data-testid={`plugin-v2-actions-${plugin.id}`}
                          >
                            <MoreHorizontal size={17} />
                          </button>
                          {actionMenuId === plugin.id ? (
                            <div
                              className="absolute right-0 top-9 z-10 min-w-[150px] overflow-hidden rounded-lg border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] py-1 shadow-lg shadow-slate-950/10"
                              role="menu"
                            >
                              <button
                                type="button"
                                className="flex h-9 w-full items-center gap-2 px-3 text-left text-sm font-medium hover:bg-[color:var(--lime-surface-hover)]"
                                onClick={() => void handleToggle(plugin)}
                                role="menuitem"
                                data-testid={`plugin-v2-toggle-${plugin.id}`}
                              >
                                {plugin.enabled ? (
                                  <Unplug size={15} />
                                ) : (
                                  <Check size={15} />
                                )}
                                {plugin.enabled
                                  ? t("plugin.catalog.v2.disable")
                                  : t("plugin.catalog.v2.enable")}
                              </button>
                              <button
                                type="button"
                                className="flex h-9 w-full items-center gap-2 px-3 text-left text-sm font-medium text-rose-700 hover:bg-rose-50"
                                onClick={() => {
                                  setActionMenuId(null);
                                  setUninstallTarget(plugin);
                                }}
                                role="menuitem"
                                data-testid={`plugin-v2-uninstall-${plugin.id}`}
                              >
                                <X size={15} />
                                {t("plugin.catalog.v2.uninstall")}
                              </button>
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                    <p className="mt-3 line-clamp-2 min-h-[40px] text-sm leading-5 text-[color:var(--lime-text-muted)]">
                      {plugin.description ||
                        t("plugin.catalog.v2.descriptionFallback")}
                    </p>
                    <div className="mt-4 flex flex-wrap items-center gap-2 text-xs text-[color:var(--lime-text-muted)]">
                      <span className="rounded-full bg-[color:var(--lime-surface-hover)] px-2.5 py-1 font-medium">
                        {sourceLabel(plugin.source, t)}
                      </span>
                      {plugin.installed ? (
                        <span
                          className={
                            plugin.enabled
                              ? "text-emerald-700"
                              : "text-amber-700"
                          }
                        >
                          {plugin.enabled
                            ? t("plugin.catalog.v2.status.enabled")
                            : t("plugin.catalog.v2.status.disabled")}
                        </span>
                      ) : null}
                    </div>
                    <div className="mt-4 grid grid-cols-3 gap-2 border-t border-[color:var(--lime-surface-border)] pt-3 text-xs text-[color:var(--lime-text-muted)]">
                      <span>
                        {t("plugin.catalog.v2.metric.skills", {
                          count: plugin.skillsCount,
                        })}
                      </span>
                      <span>
                        {t("plugin.catalog.v2.metric.mcp", {
                          count: plugin.mcpServersCount,
                        })}
                      </span>
                      <span>
                        {t("plugin.catalog.v2.metric.hooks", {
                          count: plugin.hooksCount,
                        })}
                      </span>
                    </div>
                    <div className="mt-4 flex items-center justify-between gap-3">
                      <span className="text-xs text-[color:var(--lime-text-muted)]">
                        {plugin.authPolicy === "ON_USE"
                          ? t("plugin.catalog.v2.auth.onUse")
                          : plugin.authPolicy}
                      </span>
                      {plugin.installed ? (
                        <button
                          type="button"
                          className="inline-flex h-8 items-center gap-1 rounded-full border border-[color:var(--lime-surface-border)] px-3 text-xs font-semibold text-[color:var(--lime-text-strong)] hover:bg-[color:var(--lime-surface-hover)]"
                          onClick={() => setSelectedId(plugin.id)}
                          data-testid={`plugin-v2-details-${plugin.id}`}
                        >
                          {t("plugin.catalog.v2.details")}
                          <ExternalLink size={13} />
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="inline-flex h-8 items-center gap-1 rounded-full bg-[color:var(--lime-text-strong)] px-3 text-xs font-semibold text-[color:var(--lime-surface)] hover:opacity-90"
                          onClick={() => {
                            setInstallCandidates([
                              { sourcePath: plugin.sourceUri, summary: plugin },
                            ]);
                            setSelectedInstallId(plugin.id);
                          }}
                          data-testid={`plugin-v2-install-${plugin.id}`}
                        >
                          {isBusy ? (
                            <LoaderCircle size={13} className="animate-spin" />
                          ) : null}
                          {t("plugin.catalog.v2.install")}
                        </button>
                      )}
                    </div>
                  </article>
                );
              })}
            </section>

            <PluginCatalogDetailPanel
              summary={selectedSummary}
              detail={detail}
              t={t}
              onClose={() => setSelectedId(null)}
            />
          </div>
        )}
      </div>

      {installCandidates.length > 0 ? (
        <div
          className="fixed inset-0 z-40 flex items-center justify-center bg-slate-950/25 px-4"
          role="presentation"
        >
          <div
            className="w-full max-w-[520px] rounded-xl border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] p-5 shadow-xl"
            role="dialog"
            aria-modal="true"
            aria-labelledby="plugin-v2-install-title"
            data-testid="plugin-v2-install-review"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--lime-text-muted)]">
                  {t("plugin.catalog.v2.review.eyebrow")}
                </p>
                <h2
                  id="plugin-v2-install-title"
                  className="mt-1 text-lg font-semibold text-[color:var(--lime-text-strong)]"
                >
                  {t("plugin.catalog.v2.review.title")}
                </h2>
              </div>
              <button
                type="button"
                className="rounded-full p-1.5 text-[color:var(--lime-text-muted)] hover:bg-[color:var(--lime-surface-hover)]"
                onClick={() => setInstallCandidates([])}
                aria-label={t("plugin.catalog.v2.close")}
              >
                <X size={17} />
              </button>
            </div>
            {installCandidates.length > 1 ? (
              <label className="mt-5 block text-sm font-medium text-[color:var(--lime-text-strong)]">
                {t("plugin.catalog.v2.review.choosePackage")}
                <select
                  value={selectedInstallId ?? ""}
                  onChange={(event) =>
                    setSelectedInstallId(event.currentTarget.value)
                  }
                  className="mt-2 h-10 w-full rounded-lg border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] px-3 text-sm outline-none"
                  data-testid="plugin-v2-install-package-select"
                >
                  {installCandidates.map((candidate) => (
                    <option
                      key={candidate.summary.id}
                      value={candidate.summary.id}
                    >
                      {candidate.summary.name} ·{" "}
                      {t("plugin.catalog.v2.version", {
                        version: candidate.summary.version,
                      })}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            {selectedInstallCandidate ? (
              <div className="mt-5 rounded-lg border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface-soft)] p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h3 className="truncate font-semibold text-[color:var(--lime-text-strong)]">
                      {selectedInstallCandidate.summary.name}
                    </h3>
                    <p className="mt-1 text-xs text-[color:var(--lime-text-muted)]">
                      {selectedInstallCandidate.summary.id} ·{" "}
                      {t("plugin.catalog.v2.version", {
                        version: selectedInstallCandidate.summary.version,
                      })}
                    </p>
                  </div>
                  <ShieldCheck
                    size={19}
                    className="shrink-0 text-emerald-700"
                  />
                </div>
                <p className="mt-3 text-sm leading-6 text-[color:var(--lime-text-muted)]">
                  {selectedInstallCandidate.summary.description ||
                    t("plugin.catalog.v2.descriptionFallback")}
                </p>
                <div className="mt-3 flex flex-wrap gap-2 text-xs text-[color:var(--lime-text-muted)]">
                  <span>
                    {t("plugin.catalog.v2.metric.skills", {
                      count: selectedInstallCandidate.summary.skillsCount,
                    })}
                  </span>
                  <span>
                    {t("plugin.catalog.v2.metric.mcp", {
                      count: selectedInstallCandidate.summary.mcpServersCount,
                    })}
                  </span>
                  <span>
                    {t("plugin.catalog.v2.metric.hooks", {
                      count: selectedInstallCandidate.summary.hooksCount,
                    })}
                  </span>
                </div>
                <p className="mt-3 truncate text-xs text-[color:var(--lime-text-muted)]">
                  {t("plugin.catalog.v2.review.source", {
                    name: pathLeaf(selectedInstallCandidate.sourcePath),
                  })}
                </p>
              </div>
            ) : null}
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                className="h-9 rounded-full border border-[color:var(--lime-surface-border)] px-4 text-sm font-semibold text-[color:var(--lime-text-strong)] hover:bg-[color:var(--lime-surface-hover)]"
                onClick={() => setInstallCandidates([])}
              >
                {t("plugin.catalog.v2.cancel")}
              </button>
              <button
                type="button"
                className="inline-flex h-9 items-center gap-2 rounded-full bg-[color:var(--lime-text-strong)] px-5 text-sm font-semibold text-[color:var(--lime-surface)] hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
                onClick={() => void handleInstall()}
                disabled={!selectedInstallCandidate || Boolean(busyId)}
                data-testid="plugin-v2-confirm-install"
              >
                {busyId ? (
                  <LoaderCircle size={15} className="animate-spin" />
                ) : (
                  <PackageCheck size={15} />
                )}
                {t("plugin.catalog.v2.confirmInstall")}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {uninstallTarget ? (
        <div
          className="fixed inset-0 z-40 flex items-center justify-center bg-slate-950/25 px-4"
          role="presentation"
        >
          <div
            className="w-full max-w-[420px] rounded-xl border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] p-5 shadow-xl"
            role="dialog"
            aria-modal="true"
            data-testid="plugin-v2-uninstall-confirm"
          >
            <h2 className="text-lg font-semibold text-[color:var(--lime-text-strong)]">
              {t("plugin.catalog.v2.uninstallTitle", {
                name: uninstallTarget.name,
              })}
            </h2>
            <p className="mt-2 text-sm leading-6 text-[color:var(--lime-text-muted)]">
              {t("plugin.catalog.v2.uninstallDescription")}
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                className="h-9 rounded-full border border-[color:var(--lime-surface-border)] px-4 text-sm font-semibold text-[color:var(--lime-text-strong)] hover:bg-[color:var(--lime-surface-hover)]"
                onClick={() => setUninstallTarget(null)}
              >
                {t("plugin.catalog.v2.cancel")}
              </button>
              <button
                type="button"
                className="h-9 rounded-full bg-rose-700 px-4 text-sm font-semibold text-white hover:bg-rose-800 disabled:cursor-not-allowed disabled:opacity-60"
                onClick={() => void handleUninstall()}
                disabled={Boolean(busyId)}
              >
                {t("plugin.catalog.v2.confirmUninstall")}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function PluginCatalogDetailPanel({
  summary,
  detail,
  t,
  onClose,
}: {
  summary: AppServerPluginCatalogSummary | null;
  detail: AppServerPluginCatalogDetail | null;
  t: (key: string, options?: Record<string, unknown>) => string;
  onClose: () => void;
}) {
  if (!summary) {
    return (
      <aside className="hidden min-h-[300px] rounded-lg border border-dashed border-[color:var(--lime-surface-border)] px-5 py-6 text-sm text-[color:var(--lime-text-muted)] xl:block">
        {t("plugin.catalog.v2.detail.empty")}
      </aside>
    );
  }
  return (
    <aside
      className="min-w-0 self-start rounded-lg border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] p-5 xl:sticky xl:top-0"
      data-testid="plugin-v2-detail"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--lime-text-muted)]">
            {t("plugin.catalog.v2.detail.eyebrow")}
          </p>
          <h2 className="mt-1 truncate text-lg font-semibold text-[color:var(--lime-text-strong)]">
            {summary.name}
          </h2>
          <p className="mt-1 text-xs text-[color:var(--lime-text-muted)]">
            {summary.id} ·{" "}
            {t("plugin.catalog.v2.version", { version: summary.version })}
          </p>
        </div>
        <button
          type="button"
          className="rounded-full p-1.5 text-[color:var(--lime-text-muted)] hover:bg-[color:var(--lime-surface-hover)]"
          onClick={onClose}
          title={t("plugin.catalog.v2.close")}
          aria-label={t("plugin.catalog.v2.close")}
        >
          <X size={16} />
        </button>
      </div>
      <p className="mt-4 text-sm leading-6 text-[color:var(--lime-text-muted)]">
        {summary.description || t("plugin.catalog.v2.descriptionFallback")}
      </p>
      <div className="mt-5 grid grid-cols-2 gap-2 text-xs">
        <DetailMetric
          label={t("plugin.catalog.v2.detail.skills")}
          value={detail?.skills.length ?? summary.skillsCount}
        />
        <DetailMetric
          label={t("plugin.catalog.v2.detail.mcp")}
          value={detail?.mcpServers.length ?? summary.mcpServersCount}
        />
        <DetailMetric
          label={t("plugin.catalog.v2.detail.apps")}
          value={detail?.apps.length ?? 0}
        />
        <DetailMetric
          label={t("plugin.catalog.v2.detail.hooks")}
          value={detail?.hooks.length ?? summary.hooksCount}
        />
      </div>
      {!summary.installed ? (
        <div className="mt-5 rounded-lg bg-[color:var(--lime-surface-soft)] p-3 text-sm leading-6 text-[color:var(--lime-text-muted)]">
          {t("plugin.catalog.v2.detail.installToInspect")}
        </div>
      ) : detail ? (
        <div className="mt-5 space-y-4 border-t border-[color:var(--lime-surface-border)] pt-4">
          <CapabilityGroup
            title={t("plugin.catalog.v2.detail.skills")}
            items={detail.skills.map((item) => item.name)}
          />
          <CapabilityGroup
            title={t("plugin.catalog.v2.detail.mcp")}
            items={detail.mcpServers.map((item) => item.name)}
          />
          <CapabilityGroup
            title={t("plugin.catalog.v2.detail.apps")}
            items={detail.apps.map((item) => item.name)}
          />
          <CapabilityGroup
            title={t("plugin.catalog.v2.detail.hooks")}
            items={detail.hooks.map((item) => item.event)}
          />
          <p className="flex items-center gap-2 text-xs text-[color:var(--lime-text-muted)]">
            <ShieldCheck size={14} className="text-emerald-700" />
            {formatCapabilityCount(detailCapabilityCount(detail), t)}
          </p>
        </div>
      ) : (
        <div className="mt-5 flex items-center text-sm text-[color:var(--lime-text-muted)]">
          <LoaderCircle size={15} className="mr-2 animate-spin" />
          {t("plugin.catalog.v2.detail.loading")}
        </div>
      )}
    </aside>
  );
}

function DetailMetric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md bg-[color:var(--lime-surface-soft)] px-3 py-2">
      <div className="text-[color:var(--lime-text-muted)]">{label}</div>
      <div className="mt-1 text-base font-semibold text-[color:var(--lime-text-strong)]">
        {value}
      </div>
    </div>
  );
}

function CapabilityGroup({ title, items }: { title: string; items: string[] }) {
  if (!items.length) {
    return null;
  }
  return (
    <div>
      <h3 className="text-xs font-semibold text-[color:var(--lime-text-strong)]">
        {title}
      </h3>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {items.map((item) => (
          <span
            key={item}
            className="max-w-full truncate rounded-full border border-[color:var(--lime-surface-border)] px-2 py-1 text-xs text-[color:var(--lime-text-muted)]"
          >
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}
