import {
  ArrowRight,
  Brain,
  Image as ImageIcon,
  Palette,
  Settings2,
  ShieldCheck,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import { useMemo } from "react";
import { WorkbenchInfoTip } from "@/components/media/WorkbenchInfoTip";
import type { Page, PageParams } from "@/types/page";
import {
  useSettingsCategory,
  type CategoryGroup,
  type CategoryItem,
} from "../hooks/useSettingsCategory";
import { SettingsGroupKey, SettingsTabs } from "@/types/settings";
import { useTranslation } from "react-i18next";

interface SettingsHomePageProps {
  onTabChange: (tab: SettingsTabs) => void;
  onTabPrefetch?: (tab: SettingsTabs) => void;
  onNavigate?: (page: Page, params?: PageParams) => void;
}

type DisplayGroupKey = Exclude<SettingsGroupKey, SettingsGroupKey.Overview>;
type DisplayGroup = CategoryGroup & { key: DisplayGroupKey };

function isDisplayGroup(group: CategoryGroup): group is DisplayGroup {
  return group.key !== SettingsGroupKey.Overview;
}

const groupMeta = {
  account: {
    descriptionKey: "settings.home.group.account.description",
    accentClassName: "from-slate-200/70 via-white to-white",
    iconClassName: "border-slate-200 bg-slate-100 text-slate-700",
    icon: Settings2,
  },
  general: {
    descriptionKey: "settings.home.group.general.description",
    accentClassName: "from-sky-200/60 via-white to-white",
    iconClassName: "border-sky-200 bg-sky-100 text-sky-700",
    icon: Palette,
  },
  agent: {
    descriptionKey: "settings.home.group.agent.description",
    accentClassName: "from-emerald-200/70 via-white to-white",
    iconClassName: "border-emerald-200 bg-emerald-100 text-emerald-700",
    icon: Brain,
  },
  system: {
    descriptionKey: "settings.home.group.system.description",
    accentClassName: "from-amber-200/65 via-white to-white",
    iconClassName: "border-amber-200 bg-amber-100 text-amber-700",
    icon: ShieldCheck,
  },
} as const satisfies Record<
  DisplayGroupKey,
  {
    descriptionKey: string;
    accentClassName: string;
    iconClassName: string;
    icon: LucideIcon;
  }
>;

const quickAccessMeta = {
  [SettingsTabs.Appearance]: {
    titleKey: "settings.home.quickAccess.appearance.title",
    descriptionKey: "settings.home.quickAccess.appearance.description",
    icon: Palette,
  },
  [SettingsTabs.Providers]: {
    titleKey: "settings.home.quickAccess.providers.title",
    descriptionKey: "settings.home.quickAccess.providers.description",
    icon: Brain,
  },
  [SettingsTabs.MediaServices]: {
    titleKey: "settings.home.quickAccess.mediaServices.title",
    descriptionKey: "settings.home.quickAccess.mediaServices.description",
    icon: ImageIcon,
  },
} as const satisfies Partial<
  Record<
    SettingsTabs,
    {
      titleKey: string;
      descriptionKey: string;
      icon: LucideIcon;
    }
  >
>;

type QuickAccessTab = keyof typeof quickAccessMeta;
type QuickAccessItem = CategoryItem & { key: QuickAccessTab };

function hasQuickAccessMeta(item: CategoryItem): item is QuickAccessItem {
  return Object.prototype.hasOwnProperty.call(quickAccessMeta, item.key);
}

export function SettingsHomePage({
  onTabChange,
  onTabPrefetch,
  onNavigate,
}: SettingsHomePageProps) {
  const groups = useSettingsCategory();
  const { t } = useTranslation("settings");

  const overview = useMemo(() => {
    const visibleGroups = groups.filter(isDisplayGroup);
    const totalItems = visibleGroups.reduce(
      (count, group) => count + group.items.length,
      0,
    );
    const experimentalCount = visibleGroups.reduce(
      (count, group) =>
        count + group.items.filter((item) => item.experimental).length,
      0,
    );
    const quickAccessItems = visibleGroups
      .flatMap((group) => group.items)
      .filter(hasQuickAccessMeta)
      .slice(0, 4);

    return {
      visibleGroups,
      totalItems,
      experimentalCount,
      quickAccessItems,
    };
  }, [groups]);

  return (
    <div className="space-y-6 pb-8">
      <section className="rounded-[26px] border border-slate-200/80 bg-white px-5 py-4 shadow-sm shadow-slate-950/5">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-[24px] font-semibold tracking-tight text-slate-900">
                {t("settings.home.title")}
              </h1>
              <WorkbenchInfoTip
                ariaLabel={t("settings.home.hero.tipAria")}
                content={t("settings.home.hero.tip")}
                tone="mint"
              />
            </div>
            <p className="text-sm text-slate-500">
              {t("settings.home.description")}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 xl:justify-end">
            <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-600">
              {t("settings.home.summary.groups", {
                count: overview.visibleGroups.length,
              })}
            </span>
            <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-600">
              {t("settings.home.summary.items", {
                count: overview.totalItems,
              })}
            </span>
            <span
              className={`rounded-full border px-3 py-1.5 text-xs font-medium ${
                overview.experimentalCount > 0
                  ? "border-amber-200 bg-amber-50 text-amber-700"
                  : "border-emerald-200 bg-emerald-50 text-emerald-700"
              }`}
            >
              {t("settings.home.summary.experimental", {
                count: overview.experimentalCount,
              })}
            </span>
            <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-600">
              {t("settings.home.summary.quickAccess", {
                count: overview.quickAccessItems.length,
              })}
            </span>
          </div>
        </div>
      </section>

      {overview.quickAccessItems.length > 0 ? (
        <section className="rounded-[26px] border border-slate-200/80 bg-white p-5 shadow-sm shadow-slate-950/5">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                <Sparkles className="h-4 w-4 text-emerald-600" />
                {t("settings.home.quickAccess.title")}
                <WorkbenchInfoTip
                  ariaLabel={t("settings.home.quickAccess.tipAria")}
                  content={t("settings.home.quickAccess.tip")}
                  tone="slate"
                />
              </div>
              <p className="mt-1 text-sm text-slate-500">
                {t("settings.home.quickAccess.description")}
              </p>
            </div>

            <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-500">
              {t("settings.home.quickAccess.count", {
                count: overview.quickAccessItems.length,
              })}
            </span>
          </div>

          <div className="mt-5 grid gap-3 lg:grid-cols-2 xl:grid-cols-4">
            {overview.quickAccessItems.map((item) => {
              const meta = quickAccessMeta[item.key];
              if (!meta) {
                return null;
              }
              const ItemIcon = meta.icon;
              const title = t(meta.titleKey);
              const description = t(meta.descriptionKey);
              return (
                <article
                  key={item.key}
                  className="group rounded-[22px] border border-slate-200 bg-white p-4 shadow-sm transition hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-md"
                >
                  <button
                    type="button"
                    onMouseEnter={() => onTabPrefetch?.(item.key)}
                    onMouseDown={() => onTabPrefetch?.(item.key)}
                    onFocus={() => onTabPrefetch?.(item.key)}
                    onClick={() => onTabChange(item.key)}
                    className="w-full text-left"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-slate-200 bg-slate-100 text-slate-700">
                        <ItemIcon className="h-5 w-5" />
                      </div>
                      <ArrowRight className="h-4 w-4 text-slate-400 transition group-hover:translate-x-0.5 group-hover:text-slate-600" />
                    </div>
                    <p className="mt-4 text-base font-semibold text-slate-900">
                      {title}
                    </p>
                  </button>

                  <div className="mt-3 flex justify-end">
                    <WorkbenchInfoTip
                      ariaLabel={t("settings.home.quickAccess.cardTipAria", {
                        title,
                      })}
                      content={description}
                      tone="slate"
                    />
                  </div>
                </article>
              );
            })}
          </div>
        </section>
      ) : null}

      <section
        data-testid="settings-home-entry-migration"
        className="rounded-[26px] border border-sky-200/80 bg-[linear-gradient(135deg,rgba(239,246,255,0.96)_0%,rgba(255,255,255,0.98)_100%)] p-5 shadow-sm shadow-slate-950/5"
      >
        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
              <Sparkles className="h-4 w-4 text-sky-600" />
              {t("settings.home.current.title")}
              <WorkbenchInfoTip
                ariaLabel={t("settings.home.current.tipAria")}
                content={t("settings.home.current.tip")}
                tone="slate"
              />
            </div>
            <p className="text-sm leading-6 text-slate-600">
              {t("settings.home.current.description")}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 xl:justify-end">
            {onNavigate ? (
              <button
                type="button"
                onClick={() => onNavigate("scheduled-tasks")}
                className="inline-flex items-center rounded-2xl border border-slate-200 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-100 hover:text-slate-900"
              >
                {t("settings.home.current.actions.scheduledTasks")}
              </button>
            ) : null}
            {onNavigate ? (
              <button
                type="button"
                onClick={() => onNavigate("channels")}
                className="inline-flex items-center rounded-2xl border border-slate-200 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-100 hover:text-slate-900"
              >
                {t("settings.home.current.actions.channels")}
              </button>
            ) : null}
            {onNavigate ? (
              <button
                type="button"
                onClick={() => onNavigate("resources")}
                className="inline-flex items-center rounded-2xl border border-slate-200 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-100 hover:text-slate-900"
              >
                {t("settings.home.current.actions.resources")}
              </button>
            ) : null}
          </div>
        </div>

        <div className="mt-5 grid gap-3 xl:grid-cols-2">
          <article className="rounded-[22px] border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-950/5">
            <div className="text-sm font-semibold text-slate-900">
              {t("settings.home.current.scheduledTasks.title")}
            </div>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              {t("settings.home.current.scheduledTasks.description")}
            </p>
          </article>

          <article className="rounded-[22px] border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-950/5">
            <div className="text-sm font-semibold text-slate-900">
              {t("settings.home.current.channels.title")}
            </div>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              {t("settings.home.current.channels.description")}
            </p>
          </article>

          <article className="rounded-[22px] border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-950/5">
            <div className="text-sm font-semibold text-slate-900">
              {t("settings.home.current.resources.title")}
            </div>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              {t("settings.home.current.resources.description")}
            </p>
          </article>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-2">
        {overview.visibleGroups.map((group) => {
          const meta = groupMeta[group.key];
          const GroupIcon = meta.icon;
          const groupDescription = t(meta.descriptionKey);

          return (
            <article
              key={group.key}
              className="relative overflow-hidden rounded-[26px] border border-slate-200/80 bg-white shadow-sm shadow-slate-950/5"
            >
              <div
                className={`absolute inset-x-0 top-0 h-24 bg-gradient-to-br ${meta.accentClassName}`}
              />
              <div className="relative p-5">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-start gap-3">
                    <div
                      className={`flex h-12 w-12 items-center justify-center rounded-2xl border ${meta.iconClassName}`}
                    >
                      <GroupIcon className="h-5 w-5" />
                    </div>
                    <div>
                      <h2 className="text-lg font-semibold tracking-tight text-slate-900">
                        {group.title}
                      </h2>
                      <div className="mt-1">
                        <WorkbenchInfoTip
                          ariaLabel={t("settings.home.group.tipAria", {
                            title: group.title,
                          })}
                          content={groupDescription}
                          tone="slate"
                        />
                      </div>
                    </div>
                  </div>
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
                    {t("settings.home.group.count", {
                      count: group.items.length,
                    })}
                  </span>
                </div>

                <div className="mt-5 grid gap-3 sm:grid-cols-2">
                  {group.items.map((item) => (
                    <button
                      key={item.key}
                      onClick={() => onTabChange(item.key)}
                      className="group flex items-center justify-between gap-3 rounded-2xl border border-slate-200 bg-white/90 px-4 py-3 text-left transition hover:border-slate-300 hover:bg-slate-50"
                    >
                      <div className="flex min-w-0 items-center gap-3">
                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-slate-200 bg-slate-50 text-slate-600">
                          <item.icon className="h-4 w-4" />
                        </div>
                        <div className="min-w-0">
                          <div className="truncate text-sm font-medium text-slate-800">
                            {item.label}
                          </div>
                          <div className="mt-0.5 text-xs text-slate-500">
                            {item.experimental
                              ? t("settings.home.item.experimental")
                              : t("settings.home.item.configure")}
                          </div>
                        </div>
                      </div>

                      <ArrowRight className="h-4 w-4 shrink-0 text-slate-400 transition group-hover:translate-x-0.5 group-hover:text-slate-600" />
                    </button>
                  ))}
                </div>
              </div>
            </article>
          );
        })}
      </section>
    </div>
  );
}
