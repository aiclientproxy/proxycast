import { useEffect, useMemo, useState } from "react";
import { CircleAlert, LoaderCircle } from "lucide-react";
import type {
  AppServerAppInfo,
  AppServerInstalledApp,
  AppServerPluginCatalogCapability,
} from "@/lib/api/appServerTypes";
import { readAppsReadiness, subscribeAppsListUpdates } from "@/lib/api/apps";
import { projectPluginCatalogApps } from "./PluginCatalogPageViewModel";

interface AppsSnapshot {
  apps: AppServerAppInfo[];
  installed: AppServerInstalledApp[];
}

export function PluginCatalogAppsSection({
  capabilities,
  t,
}: {
  capabilities: AppServerPluginCatalogCapability[];
  t: (key: string, options?: Record<string, unknown>) => string;
}) {
  const [snapshot, setSnapshot] = useState<AppsSnapshot | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const next = await readAppsReadiness();
        if (active) {
          setSnapshot({ apps: next.apps, installed: next.installed });
          setFailed(false);
        }
      } catch {
        if (active) {
          setFailed(true);
        }
      }
    };

    void refresh();
    const unsubscribe = subscribeAppsListUpdates({
      onError: () => {
        if (active) {
          setFailed(true);
        }
      },
      onUpdate: () => void refresh(),
    });
    return () => {
      active = false;
      unsubscribe();
    };
  }, []);

  const items = useMemo(
    () =>
      projectPluginCatalogApps(
        capabilities,
        snapshot?.apps ?? [],
        snapshot?.installed ?? [],
      ),
    [capabilities, snapshot],
  );

  if (!capabilities.length) {
    return null;
  }

  return (
    <section data-testid="plugin-catalog-apps-readiness">
      <h3 className="text-xs font-semibold text-[color:var(--lime-text-strong)]">
        {t("plugin.catalog.v2.detail.apps")}
      </h3>
      {!snapshot && !failed ? (
        <div className="mt-2 flex items-center text-xs text-[color:var(--lime-text-muted)]">
          <LoaderCircle size={13} className="mr-2 animate-spin" />
          {t("plugin.catalog.v2.detail.loading")}
        </div>
      ) : (
        <div className="mt-2 divide-y divide-[color:var(--lime-surface-border)] border-y border-[color:var(--lime-surface-border)]">
          {items.map((item) => (
            <div
              key={item.id}
              className="flex min-w-0 items-center justify-between gap-3 py-2.5 text-xs"
              data-testid={`plugin-catalog-app-readiness-${item.id}`}
              data-enabled={String(item.enabled)}
              data-callable={String(item.callable)}
              data-status={item.status}
            >
              <span className="min-w-0 truncate font-medium text-[color:var(--lime-text-strong)]">
                {item.name}
              </span>
              <span className={statusClassName(item.status)}>
                {statusLabel(item.status, t)}
              </span>
            </div>
          ))}
        </div>
      )}
      {failed ? (
        <p
          className="mt-2 flex items-center gap-1.5 text-xs text-amber-700"
          role="status"
          data-testid="plugin-catalog-apps-readiness-error"
        >
          <CircleAlert size={13} />
          {t("plugin.apps.center.status.partial")}
        </p>
      ) : null}
    </section>
  );
}

function statusLabel(
  status: "disabled" | "pending" | "ready",
  t: (key: string) => string,
): string {
  if (status === "disabled") {
    return t("plugin.catalog.v2.status.disabled");
  }
  return t(
    status === "ready"
      ? "plugin.apps.center.host.status.ready"
      : "plugin.apps.center.host.status.planned",
  );
}

function statusClassName(status: "disabled" | "pending" | "ready"): string {
  const color =
    status === "ready"
      ? "text-emerald-700"
      : status === "disabled"
        ? "text-[color:var(--lime-text-muted)]"
        : "text-amber-700";
  return `shrink-0 font-medium ${color}`;
}
