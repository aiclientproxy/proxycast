import React, { Suspense, useCallback, useMemo, useRef, useState } from "react";
import { FolderOpen, Loader2, RefreshCw, Zap } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { Skill } from "@/lib/api/skills";
import { toast } from "sonner";
import { filterMentionableServiceSkills } from "@/components/agent/chat/service-skills/entryAdapter";
import type {
  ServiceSkillGroup,
  ServiceSkillHomeItem,
} from "@/components/agent/chat/service-skills/types";
import type { BuiltinInputCommand } from "./builtinCommands";
import {
  LazyCharacterMentionPanel,
  preloadCharacterMentionPanel,
} from "./characterMentionPanelLoader";
import type { InputCapabilityDescriptor } from "./inputCapabilitySections";
import type { SelectInputCapabilityHandler } from "./inputCapabilitySelection";
import {
  getActiveSkillDisplayLabel,
  SKILL_SELECTION_DISPLAY_COPY,
} from "./skillSelectionDisplay";
import { partitionMentionableSkills } from "./skillQuery";
import { useIdleModulePreload } from "./useIdleModulePreload";

interface SkillSelectorProps {
  skills?: Skill[];
  serviceSkills?: ServiceSkillHomeItem[];
  serviceSkillGroups?: ServiceSkillGroup[];
  activeSkill?: Skill | null;
  isLoading?: boolean;
  renderMode?: "button" | "inline";
  onSelectInputCapability: SelectInputCapabilityHandler;
  onClearSkill?: () => void;
  onNavigateToSettings?: () => void;
  onImportSkill?: () => void | Promise<void>;
  onRefreshSkills?: () => void | Promise<void>;
}

interface SkillSelectorContentProps {
  activeSkill: Skill | null;
  installedSkills: Skill[];
  availableSkills: Skill[];
  mentionServiceSkills?: ServiceSkillHomeItem[];
  serviceSkillGroups?: ServiceSkillGroup[];
  query: string;
  refreshBusy: boolean;
  hasResults: boolean;
  canRefresh: boolean;
  canImport: boolean;
  importing: boolean;
  commandRef: React.RefObject<HTMLDivElement>;
  onQueryChange: (query: string) => void;
  onSelectInputCapability: SelectInputCapabilityHandler;
  onSelectAvailableSkill: (skill: Skill) => void;
  onClearSkill?: () => void;
  onNavigateToSettings?: () => void;
  onRefresh?: () => void;
  onImport?: () => void;
}

export const SkillSelectorContent: React.FC<SkillSelectorContentProps> = ({
  activeSkill,
  installedSkills,
  availableSkills,
  mentionServiceSkills = [],
  serviceSkillGroups = [],
  query,
  refreshBusy,
  hasResults,
  canRefresh,
  canImport,
  importing,
  commandRef,
  onQueryChange,
  onSelectInputCapability,
  onSelectAvailableSkill,
  onClearSkill,
  onNavigateToSettings,
  onRefresh,
  onImport,
}) => {
  const { t } = useTranslation("agent");
  const activeSkillLabel = getActiveSkillDisplayLabel(activeSkill);
  const handleSelectCapability = React.useCallback(
    (item: InputCapabilityDescriptor) => {
      switch (item.kind) {
        case "service_skill":
          onSelectInputCapability({
            kind: "service_skill",
            skill: item.skill,
          });
          return;
        case "installed_skill":
          onSelectInputCapability({
            kind: "installed_skill",
            skill: item.skill,
          });
          return;
        case "available_skill":
          onSelectAvailableSkill(item.skill);
          return;
        default:
          return;
      }
    },
    [onSelectAvailableSkill, onSelectInputCapability],
  );

  return (
    <Suspense
      fallback={
        <div className="px-4 py-7 text-center text-sm text-slate-500">
          {t("agentChat.inputbar.skillSelector.loading")}
        </div>
      }
    >
      <div className="bg-white">
        <div className="border-b border-slate-200/80 bg-[linear-gradient(180deg,rgb(255,255,255)_0%,rgb(248,250,252)_100%)] px-4 py-3">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-[11px] font-semibold tracking-[0.08em] text-slate-500">
                {SKILL_SELECTION_DISPLAY_COPY.titleLabel}
              </div>
              <div className="mt-1 text-sm text-slate-700">
                {activeSkillLabel ??
                  SKILL_SELECTION_DISPLAY_COPY.emptySelectionLabel}
              </div>
            </div>
            {activeSkill && onClearSkill ? (
              <button
                type="button"
                data-testid="skill-selector-clear"
                onClick={onClearSkill}
                className="rounded-full border border-slate-200 px-3 py-1.5 text-xs text-slate-600 transition hover:border-slate-300 hover:bg-white hover:text-slate-900"
              >
                {SKILL_SELECTION_DISPLAY_COPY.clearActionLabel}
              </button>
            ) : null}
          </div>
          {refreshBusy && !hasResults ? (
            <div className="mt-2 text-xs text-slate-500">
              {SKILL_SELECTION_DISPLAY_COPY.loadingLabel}
            </div>
          ) : null}
        </div>
        <LazyCharacterMentionPanel
          mode="mention"
          mentionQuery={query}
          builtinCommands={[] satisfies BuiltinInputCommand[]}
          sceneCommands={[]}
          slashCommands={[]}
          mentionServiceSkills={mentionServiceSkills}
          serviceSkillGroups={serviceSkillGroups}
          filteredCharacters={[]}
          installedSkills={installedSkills}
          availableSkills={availableSkills}
          commandRef={commandRef}
          onQueryChange={onQueryChange}
          onSelectCapability={handleSelectCapability}
          onSelectProjectFile={() => undefined}
          onNavigateToSettings={onNavigateToSettings}
        />
        {canRefresh || canImport ? (
          <div className="border-t border-slate-200/80 p-1.5">
            <div className="flex items-center gap-1.5">
              {canRefresh ? (
                <button
                  type="button"
                  data-testid="skill-selector-refresh"
                  onClick={onRefresh}
                  disabled={refreshBusy}
                  className="flex flex-1 items-center justify-center gap-2 rounded-xl px-3 py-2.5 text-sm text-slate-600 transition hover:bg-slate-50 hover:text-slate-900 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {refreshBusy ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                  <span>{refreshBusy ? "刷新中..." : "刷新技能"}</span>
                </button>
              ) : null}
              {canImport ? (
                <button
                  type="button"
                  data-testid="skill-selector-import"
                  onClick={onImport}
                  disabled={importing}
                  className="flex flex-1 items-center justify-center gap-2 rounded-xl px-3 py-2.5 text-sm text-slate-600 transition hover:bg-slate-50 hover:text-slate-900 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {importing ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <FolderOpen className="h-4 w-4" />
                  )}
                  <span>{importing ? "导入中..." : "导入技能"}</span>
                </button>
              ) : null}
            </div>
          </div>
        ) : null}
        {!hasResults && !canRefresh && !canImport ? (
          <div className="px-4 pb-3 text-xs text-slate-400">
            {t("agentChat.inputbar.skillSelector.emptyActions")}
          </div>
        ) : null}
      </div>
    </Suspense>
  );
};

export const SkillSelector: React.FC<SkillSelectorProps> = ({
  skills = [],
  serviceSkills = [],
  serviceSkillGroups = [],
  activeSkill = null,
  isLoading = false,
  renderMode = "button",
  onSelectInputCapability,
  onClearSkill,
  onNavigateToSettings,
  onImportSkill,
  onRefreshSkills,
}) => {
  const { t } = useTranslation("agent");
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [importing, setImporting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [autoRefreshTriggered, setAutoRefreshTriggered] = useState(false);
  const commandRef = useRef<HTMLDivElement>(null);
  const inlineMode = renderMode === "inline";
  const effectiveOpen = inlineMode || open;

  useIdleModulePreload(() => {
    void preloadCharacterMentionPanel();
  });

  React.useEffect(() => {
    if (!effectiveOpen) {
      setQuery("");
      setAutoRefreshTriggered(false);
    }
  }, [effectiveOpen]);

  const { installedSkills, availableSkills } = useMemo(
    () => partitionMentionableSkills(skills, query),
    [query, skills],
  );

  const filteredServiceSkills = useMemo(
    () => filterMentionableServiceSkills(serviceSkills, query),
    [query, serviceSkills],
  );

  const hasResults =
    filteredServiceSkills.length > 0 ||
    installedSkills.length > 0 ||
    availableSkills.length > 0 ||
    Boolean(activeSkill && onClearSkill);
  const canImport = Boolean(onImportSkill || onNavigateToSettings);
  const canRefresh = Boolean(onRefreshSkills);
  const refreshBusy = isLoading || refreshing;

  const handleRefresh = useCallback(async () => {
    if (!onRefreshSkills || refreshBusy) {
      return;
    }

    try {
      setRefreshing(true);
      await onRefreshSkills();
    } finally {
      setRefreshing(false);
    }
  }, [onRefreshSkills, refreshBusy]);

  React.useEffect(() => {
    if (
      !effectiveOpen ||
      autoRefreshTriggered ||
      !onRefreshSkills ||
      refreshBusy ||
      skills.length > 0
    ) {
      return;
    }

    setAutoRefreshTriggered(true);
    void handleRefresh();
  }, [
    autoRefreshTriggered,
    handleRefresh,
    onRefreshSkills,
    effectiveOpen,
    refreshBusy,
    skills.length,
  ]);

  const handleSelectCapability = (
    capability: Parameters<SelectInputCapabilityHandler>[0],
  ) => {
    onSelectInputCapability(capability);
    setOpen(false);
  };

  const handleClearSkill = () => {
    onClearSkill?.();
    setOpen(false);
  };

  const handleSelectAvailableSkill = (skill: Skill) => {
    setOpen(false);
    toast.info(`技能「${skill.name}」尚未安装`, {
      action: onNavigateToSettings
        ? {
            label: "去技能中心",
            onClick: onNavigateToSettings,
          }
        : undefined,
    });
  };

  const handleImport = async () => {
    if (!canImport || importing) {
      return;
    }

    setOpen(false);

    if (!onImportSkill) {
      onNavigateToSettings?.();
      return;
    }

    try {
      setImporting(true);
      await onImportSkill();
    } finally {
      setImporting(false);
    }
  };

  const content = (
    <SkillSelectorContent
      activeSkill={activeSkill}
      installedSkills={installedSkills}
      availableSkills={availableSkills}
      mentionServiceSkills={filteredServiceSkills}
      serviceSkillGroups={serviceSkillGroups}
      query={query}
      refreshBusy={refreshBusy}
      hasResults={hasResults}
      canRefresh={canRefresh}
      canImport={canImport}
      importing={importing}
      commandRef={commandRef}
      onQueryChange={setQuery}
      onSelectInputCapability={handleSelectCapability}
      onSelectAvailableSkill={handleSelectAvailableSkill}
      onClearSkill={handleClearSkill}
      onNavigateToSettings={
        onNavigateToSettings
          ? () => {
              setOpen(false);
              onNavigateToSettings();
            }
          : undefined
      }
      onRefresh={() => void handleRefresh()}
      onImport={() => void handleImport()}
    />
  );

  if (inlineMode) {
    return (
      <div
        className="w-full overflow-hidden rounded-lg border border-slate-100 bg-white"
        data-testid="skill-selector-inline"
      >
        {content}
      </div>
    );
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          data-testid="skill-selector-trigger"
          className={`inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-xs font-medium shadow-none transition-colors ${
            activeSkill
              ? "border-emerald-300 bg-emerald-50 text-emerald-700 hover:border-emerald-300 hover:bg-emerald-50 hover:text-emerald-700"
              : "border-slate-200/80 bg-white/92 text-slate-600 hover:border-slate-300 hover:bg-white hover:text-slate-900"
          }`}
        >
          <Zap className="h-3.5 w-3.5" />
          <span>{t("agentChat.inputbar.skillSelector.trigger")}</span>
        </button>
      </PopoverTrigger>
      <PopoverContent
        className="w-80 overflow-hidden rounded-[22px] border border-slate-200/80 bg-white p-0 shadow-xl shadow-slate-950/8 opacity-100"
        side="top"
        align="start"
        sideOffset={8}
      >
        {open ? content : null}
      </PopoverContent>
    </Popover>
  );
};
