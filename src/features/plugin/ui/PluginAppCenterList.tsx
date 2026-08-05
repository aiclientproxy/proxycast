import { useEffect, useRef, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  FolderOpen,
  MoreHorizontal,
  RefreshCw,
  Search,
  ShieldCheck,
  UploadCloud,
} from "lucide-react";
import type {
  HostCapabilityProfile,
  InstalledPluginState,
  ProjectedEntry,
} from "../types";
import { PluginPublishWorkbench } from "../publish/PluginPublishWorkbench";
import { PluginReleaseReviewWorkbench } from "../publish/PluginReleaseReviewWorkbench";
import type {
  PluginLaunchTargetMode,
  PluginLaunchTargetPolicy,
} from "./pluginLaunchTargetPolicy";
import { PluginAppCard } from "./PluginAppCard";
import { PluginLaunchTargetControl } from "./PluginLaunchTargetControl";
import type { PluginDynamicTranslation } from "./PluginsPagePresentation";
import type {
  AppCenterFilterCounts,
  AppCenterItem,
  AppCenterSourceFilter,
  AppCenterStatusFilter,
} from "./PluginsPageViewModel";

const STATUS_FILTERS: readonly AppCenterStatusFilter[] = [
  "all",
  "installed",
  "installable",
  "attention",
];

const SOURCE_FILTERS: readonly AppCenterSourceFilter[] = [
  "all",
  "cloud",
  "local",
];

export function PluginAppCenterList({
  appItemsCount,
  busyAction,
  currentPage,
  filterCounts,
  filteredItemsCount,
  launchTargetPolicy,
  loading,
  pagedItems,
  profile,
  publishWorkbenchOpen,
  releaseReviewWorkbenchOpen,
  searchQuery,
  selectedRightSurfaceTargetId,
  sourceFilter,
  statusFilter,
  t,
  totalPages,
  issueCount,
  onCloudAction,
  onClosePublishWorkbench,
  onCloseReleaseReviewWorkbench,
  onInstallLocal,
  onLaunchEntry,
  onNextPage,
  onOpenDetail,
  onPluginPublishStateChanged,
  onPreviousPage,
  onPrimaryAction,
  onRefresh,
  onSearchQueryChange,
  onSelectedTargetIdChange,
  onSourceFilterChange,
  onStatusFilterChange,
  onTargetModeChange,
  onTogglePublishWorkbench,
  onToggleReleaseReviewWorkbench,
}: {
  appItemsCount: number;
  busyAction: string | null;
  currentPage: number;
  filterCounts: AppCenterFilterCounts;
  filteredItemsCount: number;
  launchTargetPolicy: PluginLaunchTargetPolicy;
  loading: boolean;
  pagedItems: AppCenterItem[];
  profile: HostCapabilityProfile;
  publishWorkbenchOpen: boolean;
  releaseReviewWorkbenchOpen: boolean;
  searchQuery: string;
  selectedRightSurfaceTargetId: string | null;
  sourceFilter: AppCenterSourceFilter;
  statusFilter: AppCenterStatusFilter;
  t: PluginDynamicTranslation;
  totalPages: number;
  issueCount: number;
  onCloudAction: (item: AppCenterItem) => void | Promise<void>;
  onClosePublishWorkbench: () => void;
  onCloseReleaseReviewWorkbench: () => void;
  onInstallLocal: () => void | Promise<void>;
  onLaunchEntry: (
    state: InstalledPluginState,
    entry: ProjectedEntry,
  ) => void | Promise<void>;
  onNextPage: () => void;
  onOpenDetail: (appId: string) => void;
  onPluginPublishStateChanged: () => void;
  onPreviousPage: () => void;
  onPrimaryAction: (item: AppCenterItem) => void | Promise<void>;
  onRefresh: () => void | Promise<void>;
  onSearchQueryChange: (value: string) => void;
  onSelectedTargetIdChange: (targetId: string | null) => void;
  onSourceFilterChange: (filter: AppCenterSourceFilter) => void;
  onStatusFilterChange: (filter: AppCenterStatusFilter) => void;
  onTargetModeChange: (mode: PluginLaunchTargetMode) => void;
  onTogglePublishWorkbench: () => void;
  onToggleReleaseReviewWorkbench: () => void;
}) {
  const [moreActionsOpen, setMoreActionsOpen] = useState(false);
  const moreActionsRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!moreActionsOpen) {
      return;
    }

    const handlePointerDown = (event: PointerEvent) => {
      if (!moreActionsRef.current?.contains(event.target as Node)) {
        setMoreActionsOpen(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMoreActionsOpen(false);
      }
    };

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [moreActionsOpen]);

  const closeMoreActions = () => setMoreActionsOpen(false);

  return (
    <>
      <header className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
        <div className="min-w-0">
          <h1 className="text-[28px] font-semibold text-[color:var(--lime-text-strong)]">
            {t("plugin.apps.center.title")}
          </h1>
          {issueCount > 0 ? (
            <p
              className="mt-2 text-sm font-medium text-amber-700"
              data-testid="plugins-load-issues"
            >
              {t("plugin.apps.installed.issues", { count: issueCount })}
            </p>
          ) : null}
        </div>
        <div className="flex min-w-0 flex-col gap-3 xl:flex-row xl:items-center">
          <label className="relative w-full min-w-0 xl:w-[360px]">
            <Search
              className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-[color:var(--lime-text-muted)]"
              size={18}
            />
            <input
              className="h-9 w-full rounded-full border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] pl-10 pr-4 text-sm font-semibold text-[color:var(--lime-text-strong)] shadow-none outline-none transition placeholder:text-[color:var(--lime-text-muted)] focus:border-[color:var(--lime-surface-border-strong)]"
              value={searchQuery}
              onChange={(event) => onSearchQueryChange(event.target.value)}
              onInput={(event) =>
                onSearchQueryChange(event.currentTarget.value)
              }
              placeholder={t("plugin.apps.center.searchPlaceholder")}
              data-testid="plugins-search"
            />
          </label>
          <div className="flex min-w-0 flex-wrap gap-3">
            <button
              type="button"
              className="inline-flex h-9 shrink-0 items-center gap-2 whitespace-nowrap rounded-full bg-[color:var(--lime-text-strong)] px-5 text-sm font-semibold text-[color:var(--lime-surface)] shadow-none transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
              disabled={Boolean(busyAction)}
              onClick={() => void onInstallLocal()}
              data-testid="plugins-install-local"
            >
              <FolderOpen size={16} />
              {t("plugin.apps.center.installLocal")}
            </button>
            <div className="relative shrink-0" ref={moreActionsRef}>
              <button
                type="button"
                className="inline-flex h-9 shrink-0 items-center gap-2 whitespace-nowrap rounded-full border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] px-4 text-sm font-semibold text-[color:var(--lime-text-strong)] shadow-none transition hover:bg-[color:var(--lime-surface-hover)]"
                aria-haspopup="menu"
                aria-expanded={moreActionsOpen}
                aria-controls="plugins-more-actions-menu"
                onClick={() => setMoreActionsOpen((current) => !current)}
                data-testid="plugins-more-actions"
              >
                <MoreHorizontal size={16} />
                {t("plugin.apps.center.moreActions")}
                <ChevronDown
                  className={`transition-transform ${moreActionsOpen ? "rotate-180" : ""}`}
                  size={14}
                />
              </button>
              {moreActionsOpen ? (
                <div
                  id="plugins-more-actions-menu"
                  className="absolute right-0 top-full z-30 mt-2 min-w-[188px] overflow-hidden rounded-xl border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] py-1 text-[color:var(--lime-text-strong)] shadow-lg shadow-slate-950/10"
                  role="menu"
                  aria-label={t("plugin.apps.center.moreActions")}
                  data-testid="plugins-more-actions-menu"
                >
                  <button
                    type="button"
                    className="flex h-9 w-full items-center gap-2 whitespace-nowrap px-3 text-left text-sm font-semibold transition hover:bg-[color:var(--lime-surface-hover)]"
                    onClick={() => {
                      closeMoreActions();
                      onTogglePublishWorkbench();
                    }}
                    role="menuitem"
                    data-testid="plugins-open-publish"
                  >
                    <UploadCloud size={16} />
                    {t("plugin.apps.center.publish")}
                  </button>
                  <button
                    type="button"
                    className="flex h-9 w-full items-center gap-2 whitespace-nowrap px-3 text-left text-sm font-semibold transition hover:bg-[color:var(--lime-surface-hover)]"
                    onClick={() => {
                      closeMoreActions();
                      onToggleReleaseReviewWorkbench();
                    }}
                    role="menuitem"
                    data-testid="plugins-open-release-review"
                  >
                    <ShieldCheck size={16} />
                    {t("plugin.apps.center.review")}
                  </button>
                  <button
                    type="button"
                    className="flex h-9 w-full items-center gap-2 whitespace-nowrap px-3 text-left text-sm font-semibold transition hover:bg-[color:var(--lime-surface-hover)] disabled:cursor-not-allowed disabled:opacity-60"
                    onClick={() => {
                      closeMoreActions();
                      void onRefresh();
                    }}
                    disabled={loading}
                    role="menuitem"
                    data-testid="plugins-refresh"
                  >
                    <RefreshCw size={16} />
                    {t("plugin.apps.center.refresh")}
                  </button>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </header>

      {publishWorkbenchOpen ? (
        <PluginPublishWorkbench
          profile={profile}
          onClose={onClosePublishWorkbench}
          onSubmissionCreated={onPluginPublishStateChanged}
        />
      ) : null}
      {releaseReviewWorkbenchOpen ? (
        <PluginReleaseReviewWorkbench
          onClose={onCloseReleaseReviewWorkbench}
          onPublished={onPluginPublishStateChanged}
        />
      ) : null}

      <section className="flex flex-wrap items-center gap-5">
        {STATUS_FILTERS.map((filter) => (
          <button
            key={filter}
            type="button"
            className={`inline-flex h-8 items-center gap-2 rounded-full text-base font-semibold transition ${
              statusFilter === filter
                ? "text-[color:var(--lime-text-strong)]"
                : "text-[color:var(--lime-text-muted)] hover:text-[color:var(--lime-text-strong)]"
            }`}
            onClick={() => onStatusFilterChange(filter)}
            data-testid={`plugins-status-filter-${filter}`}
          >
            {t(`plugin.apps.center.filter.${filter}`)}
            <span className="text-xs text-[color:var(--lime-text-muted)]">
              {filterCounts[filter]}
            </span>
          </button>
        ))}
      </section>

      <section className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-[color:var(--lime-text-muted)]">
          <div className="flex flex-wrap items-center gap-3">
            <span>{t("plugin.apps.center.source.label")}：</span>
            {SOURCE_FILTERS.map((filter) => (
              <button
                key={filter}
                type="button"
                className={`h-8 rounded-lg border px-3 text-xs font-semibold transition ${
                  sourceFilter === filter
                    ? "border-[color:var(--lime-surface-border-strong)] bg-[color:var(--lime-surface-hover)] text-[color:var(--lime-text-strong)]"
                    : "border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] text-[color:var(--lime-text)] hover:bg-[color:var(--lime-surface-hover)]"
                }`}
                onClick={() => onSourceFilterChange(filter)}
                data-testid={`plugins-source-filter-${filter}`}
              >
                {t(`plugin.apps.center.source.${filter}`)}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <span>{t("plugin.apps.center.status.label")}：</span>
            <button
              type="button"
              className={`font-semibold ${
                statusFilter === "all"
                  ? "text-[color:var(--lime-text-strong)]"
                  : "text-[color:var(--lime-text-muted)]"
              }`}
              onClick={() => onStatusFilterChange("all")}
            >
              {t("plugin.apps.center.status.all")}
            </button>
            <span className="text-slate-300">/</span>
            <button
              type="button"
              className="font-medium text-[color:var(--lime-text-muted)] hover:text-[color:var(--lime-text-strong)]"
              onClick={() => onStatusFilterChange("attention")}
            >
              {t("plugin.apps.center.status.updateShort")}
            </button>
            <span className="text-slate-300">/</span>
            <button
              type="button"
              className="font-medium text-[color:var(--lime-text-muted)] hover:text-[color:var(--lime-text-strong)]"
              onClick={() => onStatusFilterChange("attention")}
            >
              {t("plugin.apps.center.status.authorizationShort")}
            </button>
          </div>
          <div className="text-[color:var(--lime-text-muted)]">
            {t("plugin.apps.center.sort.label")}：
            <span className="ml-2 font-medium text-[color:var(--lime-text)]">
              {t("plugin.apps.center.sort.recent")}
            </span>
          </div>
        </div>
        <PluginLaunchTargetControl
          policy={launchTargetPolicy}
          selectedTargetId={selectedRightSurfaceTargetId}
          onModeChange={onTargetModeChange}
          onSelectedTargetIdChange={onSelectedTargetIdChange}
        />
        <main className="min-w-0">
          <div
            className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3"
            data-testid="plugins-list"
          >
            {pagedItems.map((item) => (
              <PluginAppCard
                key={item.appId}
                item={item}
                busyAction={busyAction}
                t={t}
                dynamicT={t}
                onOpenDetail={onOpenDetail}
                onPrimaryAction={onPrimaryAction}
                onCloudAction={onCloudAction}
                onLaunchEntry={onLaunchEntry}
              />
            ))}
            {pagedItems.length === 0 ? (
              <div className="col-span-full flex min-h-[260px] items-center justify-center rounded-lg border border-dashed border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface-soft)] px-4 py-8 text-center">
                <div>
                  <p className="text-sm font-semibold text-[color:var(--lime-text-strong)]">
                    {appItemsCount === 0
                      ? t("plugin.apps.center.empty.noApps")
                      : t("plugin.apps.center.empty.noMatches")}
                  </p>
                  <p className="mt-2 text-sm text-[color:var(--lime-text-muted)]">
                    {t("plugin.apps.center.empty.helper")}
                  </p>
                </div>
              </div>
            ) : null}
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-[color:var(--lime-surface-border)] pt-4">
            <p className="text-xs text-[color:var(--lime-text-muted)]">
              {t("plugin.apps.center.pagination.summary", {
                page: currentPage,
                total: totalPages,
                count: filteredItemsCount,
              })}
            </p>
            <div className="flex gap-2">
              <button
                type="button"
                className="inline-flex h-8 items-center gap-1 rounded-full border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] px-4 text-sm font-medium text-[color:var(--lime-text)] transition hover:bg-[color:var(--lime-surface-hover)] disabled:cursor-not-allowed disabled:opacity-50"
                disabled={currentPage <= 1}
                onClick={onPreviousPage}
                data-testid="plugins-pagination-prev"
              >
                <ChevronLeft size={14} />
                {t("plugin.apps.center.pagination.previous")}
              </button>
              <button
                type="button"
                className="inline-flex h-8 items-center gap-1 rounded-full border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] px-4 text-sm font-medium text-[color:var(--lime-text)] transition hover:bg-[color:var(--lime-surface-hover)] disabled:cursor-not-allowed disabled:opacity-50"
                disabled={currentPage >= totalPages}
                onClick={onNextPage}
                data-testid="plugins-pagination-next"
              >
                {t("plugin.apps.center.pagination.next")}
                <ChevronRight size={14} />
              </button>
            </div>
          </div>
        </main>
      </section>
    </>
  );
}
