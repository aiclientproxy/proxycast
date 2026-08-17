import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { scheduledTasksApi } from "@/lib/api/scheduledTasks";
import {
  getSkillCatalog,
  isSeededSkillCatalog,
  refreshSkillCatalogFromRemote,
  subscribeSkillCatalogChanged,
  type SkillCatalog,
  type SkillCatalogItem,
  type SkillCatalogGroup,
} from "@/lib/api/skillCatalog";
import {
  buildServiceSkillAutomationStatusMap,
  listServiceSkillAutomationLinks,
  resolveServiceSkillAutomationLinks,
  subscribeServiceSkillAutomationLinksChanged,
} from "./automationLinkStorage";
import {
  getServiceSkillActionLabel,
  getServiceSkillRunnerDescription,
  getServiceSkillRunnerLabel,
  getServiceSkillRunnerTone,
} from "./skillPresentation";
import { shouldExposeServiceSkillHomeItem } from "./homeEntrySkills";
import {
  buildServiceSkillHomeCopy,
  type ServiceSkillHomeCopy,
} from "./homeCopy";
import { scheduleMinimumDelayIdleTask } from "@/lib/utils/scheduleMinimumDelayIdleTask";
import { supportsServiceSkillLocalAutomation } from "./automationDraft";
import {
  getServiceSkillUsageMap,
  recordServiceSkillUsage,
  subscribeServiceSkillUsageChanged,
} from "./storage";
import type {
  RecordServiceSkillUsageInput,
  ServiceSkillAutomationStatus,
  ServiceSkillCatalogMeta,
  ServiceSkillHomeItem,
} from "./types";
import type { AgentI18nResource } from "@/i18n/agentResources";

const SERVICE_SKILLS_IDLE_TIMEOUT_MS = 1_500;

type AgentI18nKey = keyof AgentI18nResource;

function getSkillBadge(
  item: SkillCatalogItem,
  isRecent: boolean,
  copy: ServiceSkillHomeCopy,
): string {
  if (isRecent) {
    return copy.badge.recent;
  }
  if (
    item.defaultExecutorBinding === "browser_assist" ||
    item.execution.kind === "site_adapter"
  ) {
    return copy.badge.browserAssist;
  }

  return copy.badge.readyMade;
}

function buildHomeItems(
  items: SkillCatalogItem[],
  automationStatusMap: Record<string, ServiceSkillAutomationStatus>,
  copy: ServiceSkillHomeCopy,
): ServiceSkillHomeItem[] {
  const usageMap = getServiceSkillUsageMap();
  const mapped: Array<ServiceSkillHomeItem & { _sortIndex: number }> =
    items.map((item, index) => {
      const recent = usageMap.get(item.id);
      const recentUsedAt = recent?.usedAt ?? null;
      const isRecent = typeof recentUsedAt === "number";

      return {
        ...item,
        groupKey: item.groupKey,
        executionKind: item.execution.kind,
        badge: getSkillBadge(item, isRecent, copy),
        recentUsedAt,
        isRecent,
        runnerLabel: getServiceSkillRunnerLabel(item),
        runnerTone: getServiceSkillRunnerTone(item),
        runnerDescription: getServiceSkillRunnerDescription(item),
        actionLabel: getServiceSkillActionLabel(item),
        automationStatus: automationStatusMap[item.id] ?? null,
        _sortIndex: index,
      };
    });

  return mapped
    .sort((left, right) => {
      if (left.recentUsedAt && right.recentUsedAt) {
        if (left.recentUsedAt !== right.recentUsedAt) {
          return right.recentUsedAt - left.recentUsedAt;
        }
      } else if (left.recentUsedAt) {
        return -1;
      } else if (right.recentUsedAt) {
        return 1;
      }

      return left._sortIndex - right._sortIndex;
    })
    .map(({ _sortIndex, ...item }) => item);
}

function buildCatalogMeta(
  catalog: SkillCatalog,
  copy: ServiceSkillHomeCopy,
): ServiceSkillCatalogMeta {
  const isSeeded = isSeededSkillCatalog(catalog);

  return {
    tenantId: catalog.tenantId,
    version: catalog.version,
    syncedAt: catalog.syncedAt,
    itemCount: catalog.items.length,
    groupCount: catalog.groups.length,
    sourceLabel: isSeeded
      ? copy.catalogSource.seeded
      : copy.catalogSource.synced,
    isSeeded,
  };
}

interface UseServiceSkillsResult {
  skills: ServiceSkillHomeItem[];
  groups: SkillCatalogGroup[];
  catalogMeta: ServiceSkillCatalogMeta | null;
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  recordUsage: (input: RecordServiceSkillUsageInput) => void;
}

interface UseServiceSkillsOptions {
  enabled?: boolean;
  autoLoad?: boolean;
  loadMode?: "immediate" | "deferred";
  deferredDelayMs?: number;
}

export function useServiceSkills(
  options: UseServiceSkillsOptions | boolean = true,
): UseServiceSkillsResult {
  const { t } = useTranslation("agent");
  const normalizedOptions =
    typeof options === "boolean" ? { enabled: options } : options;
  const enabled = normalizedOptions.enabled ?? true;
  const autoLoad = normalizedOptions.autoLoad ?? true;
  const loadMode = normalizedOptions.loadMode ?? "immediate";
  const deferredDelayMs =
    normalizedOptions.deferredDelayMs ?? SERVICE_SKILLS_IDLE_TIMEOUT_MS;
  const [items, setItems] = useState<SkillCatalogItem[]>([]);
  const [groups, setGroups] = useState<SkillCatalogGroup[]>([]);
  const [automationStatusMap, setAutomationStatusMap] = useState<
    Record<string, ServiceSkillAutomationStatus>
  >({});
  const [catalogMeta, setCatalogMeta] =
    useState<ServiceSkillCatalogMeta | null>(null);
  const [isLoading, setIsLoading] = useState(enabled && autoLoad);
  const [error, setError] = useState<string | null>(null);
  const [usageVersion, setUsageVersion] = useState(0);
  const [automationLinkCount, setAutomationLinkCount] = useState(0);
  const [catalogLoaded, setCatalogLoaded] = useState(false);
  const serviceSkillHomeCopy = useMemo(
    () => buildServiceSkillHomeCopy((key) => t(key as AgentI18nKey, {})),
    [t],
  );

  const applyCatalogSnapshot = useCallback(
    async (catalog: SkillCatalog) => {
      const visibleItems = catalog.items.filter(
        shouldExposeServiceSkillHomeItem,
      );
      const visibleGroupKeys = new Set(
        visibleItems.map((item) => item.groupKey),
      );
      const visibleGroups = catalog.groups.filter((group) =>
        visibleGroupKeys.has(group.key),
      );
      const automationLinks = listServiceSkillAutomationLinks();
      let automationStatuses: Record<string, ServiceSkillAutomationStatus> = {};
      let resolvedAutomationLinkCount = automationLinks.length;

      const hasLocalAutomationSkills = visibleItems.some((item) =>
        supportsServiceSkillLocalAutomation(item),
      );

      if (automationLinks.length > 0 || hasLocalAutomationSkills) {
        try {
          const scheduledTasks = await scheduledTasksApi.listDetailed({
            limit: 200,
          });
          automationStatuses =
            buildServiceSkillAutomationStatusMap(scheduledTasks);
          resolvedAutomationLinkCount =
            resolveServiceSkillAutomationLinks(scheduledTasks).length;
        } catch {
          automationStatuses = {};
        }
      }

      setItems(visibleItems);
      setGroups(visibleGroups);
      setAutomationLinkCount(resolvedAutomationLinkCount);
      setAutomationStatusMap(automationStatuses);
      setCatalogMeta(
        buildCatalogMeta(
          {
            ...catalog,
            groups: visibleGroups,
            items: visibleItems,
          },
          serviceSkillHomeCopy,
        ),
      );
    },
    [serviceSkillHomeCopy],
  );

  const loadCurrentCatalog = useCallback(async () => {
    const catalog = await getSkillCatalog();
    await applyCatalogSnapshot(catalog);
    return catalog;
  }, [applyCatalogSnapshot]);

  const refresh = useCallback(async () => {
    if (!enabled) {
      setItems([]);
      setGroups([]);
      setAutomationStatusMap({});
      setAutomationLinkCount(0);
      setCatalogMeta(null);
      setCatalogLoaded(false);
      setError(null);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    let currentCatalog: SkillCatalog | null = null;

    try {
      currentCatalog = await loadCurrentCatalog();
      setCatalogLoaded(true);
      setError(null);
      setIsLoading(false);

      try {
        const remoteCatalog = await refreshSkillCatalogFromRemote();
        if (remoteCatalog) {
          await applyCatalogSnapshot(remoteCatalog);
        }
      } catch (reason) {
        if (!currentCatalog) {
          throw reason;
        }
      }
    } catch (reason) {
      setItems([]);
      setGroups([]);
      setAutomationStatusMap({});
      setAutomationLinkCount(0);
      setCatalogMeta(null);
      setCatalogLoaded(false);
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setIsLoading(false);
    }
  }, [applyCatalogSnapshot, enabled, loadCurrentCatalog]);

  useEffect(() => {
    if (!autoLoad) {
      return;
    }

    if (loadMode === "deferred") {
      return scheduleMinimumDelayIdleTask(
        () => {
          void refresh();
        },
        {
          minimumDelayMs: deferredDelayMs,
          idleTimeoutMs: SERVICE_SKILLS_IDLE_TIMEOUT_MS,
        },
      );
    }

    void refresh();
    return;
  }, [autoLoad, deferredDelayMs, loadMode, refresh]);

  useEffect(() => {
    if (!enabled || !catalogLoaded) {
      return;
    }

    const unsubscribeCatalog = subscribeSkillCatalogChanged(() => {
      void loadCurrentCatalog();
    });
    const unsubscribeAutomationLinks =
      subscribeServiceSkillAutomationLinksChanged(() => {
        void loadCurrentCatalog();
      });

    return () => {
      unsubscribeCatalog();
      unsubscribeAutomationLinks();
    };
  }, [catalogLoaded, enabled, loadCurrentCatalog]);

  useEffect(() => {
    if (!enabled || !catalogLoaded || automationLinkCount === 0) {
      return;
    }

    const timer = window.setInterval(() => {
      void refresh();
    }, 15_000);

    return () => {
      window.clearInterval(timer);
    };
  }, [automationLinkCount, catalogLoaded, enabled, refresh]);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    return subscribeServiceSkillUsageChanged(() => {
      setUsageVersion((previous) => previous + 1);
    });
  }, [enabled]);

  const recordUsageAndRefresh = useCallback(
    (input: RecordServiceSkillUsageInput) => {
      recordServiceSkillUsage(input);
    },
    [],
  );

  const skills = useMemo(() => {
    void usageVersion;
    return buildHomeItems(items, automationStatusMap, serviceSkillHomeCopy);
  }, [items, usageVersion, automationStatusMap, serviceSkillHomeCopy]);

  return {
    skills,
    groups,
    catalogMeta,
    isLoading,
    error,
    refresh,
    recordUsage: recordUsageAndRefresh,
  };
}
