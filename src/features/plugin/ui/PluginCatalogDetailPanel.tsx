import { LoaderCircle, ShieldCheck, X } from "lucide-react";
import type {
  AppServerPluginCatalogDetail,
  AppServerPluginCatalogSummary,
} from "@/lib/api/appServerTypes";
import { PluginCatalogAppsSection } from "./PluginCatalogAppsSection";
import { detailCapabilityCount } from "./PluginCatalogPageViewModel";

export function PluginCatalogDetailPanel({
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
      data-testid="plugin-catalog-detail"
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
          value={detail?.apps.length ?? summary.appsCount}
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
          <PluginCatalogAppsSection capabilities={detail.apps} t={t} />
          <CapabilityGroup
            title={t("plugin.catalog.v2.detail.hooks")}
            items={detail.hooks.map((item) => item.event)}
          />
          <p className="flex items-center gap-2 text-xs text-[color:var(--lime-text-muted)]">
            <ShieldCheck size={14} className="text-emerald-700" />
            {t("plugin.catalog.v2.capabilityCount", {
              count: detailCapabilityCount(detail),
            })}
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
