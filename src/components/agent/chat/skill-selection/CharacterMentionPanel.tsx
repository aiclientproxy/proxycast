import React from "react";
import {
  Blocks,
  Command as CommandIcon,
  File,
  ImagePlus,
  Sparkles,
  User,
  Zap,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  Command,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { cn } from "@/lib/utils";
import type { Character } from "@/lib/api/projectMemory";
import type { Skill } from "@/lib/api/skills";
import type {
  ServiceSkillGroup,
  ServiceSkillHomeItem,
} from "@/components/agent/chat/service-skills/types";
import type { SlashCommandDefinition } from "../commands";
import type {
  BuiltinInputCommand,
  RuntimeSceneSlashCommand,
} from "./builtinCommands";
import type { InputbarPluginCapability } from "../components/Inputbar/pluginInputCapability";
import {
  buildInputCapabilitySections,
  buildInputCapabilitySectionsCopy,
  type InputCapabilityDescriptor,
} from "./inputCapabilitySections";
import { buildCuratedTaskTemplateCopy } from "../utils/curatedTaskTemplates";
import type { CuratedTaskReferenceEntry } from "../utils/curatedTaskReferenceSelection";
import type { AgentI18nResource } from "@/i18n/agentResources";
import type { ProjectFileSearchResult } from "@/lib/api/fuzzyFileSearch";
import type { ProjectFileMentionSearchStatus } from "./useProjectFileMentionSearch";

type AgentI18nKey = keyof AgentI18nResource;

interface CharacterMentionPanelProps {
  mode: "mention" | "slash";
  mentionQuery: string;
  builtinCommands: BuiltinInputCommand[];
  slashCommands: SlashCommandDefinition[];
  sceneCommands: RuntimeSceneSlashCommand[];
  mentionServiceSkills: ServiceSkillHomeItem[];
  pluginSuggestions?: readonly InputbarPluginCapability[];
  serviceSkillGroups?: ServiceSkillGroup[];
  filteredCharacters: Character[];
  installedSkills: Skill[];
  availableSkills: Skill[];
  projectFiles?: ProjectFileSearchResult[];
  projectFileSearchStatus?: ProjectFileMentionSearchStatus;
  projectId?: string | null;
  sessionId?: string | null;
  referenceEntries?: CuratedTaskReferenceEntry[];
  curatedTaskTemplatesVersion?: number;
  curatedTaskRecommendationSignalsVersion?: number;
  mentionEntryUsageVersion?: number;
  slashEntryUsageVersion?: number;
  commandRef: React.RefObject<HTMLDivElement>;
  onQueryChange: (query: string) => void;
  onSelectCapability: (item: InputCapabilityDescriptor) => void;
  onSelectProjectFile?: (file: ProjectFileSearchResult) => void;
  onNavigateToSettings?: () => void;
}

export const CharacterMentionPanel: React.FC<CharacterMentionPanelProps> = ({
  mode,
  mentionQuery,
  builtinCommands,
  slashCommands,
  sceneCommands,
  mentionServiceSkills,
  pluginSuggestions = [],
  serviceSkillGroups = [],
  filteredCharacters,
  installedSkills,
  availableSkills,
  projectFiles = [],
  projectFileSearchStatus = "idle",
  projectId,
  sessionId,
  referenceEntries = [],
  curatedTaskTemplatesVersion = 0,
  curatedTaskRecommendationSignalsVersion = 0,
  mentionEntryUsageVersion = 0,
  slashEntryUsageVersion = 0,
  commandRef,
  onQueryChange,
  onSelectCapability,
  onSelectProjectFile,
  onNavigateToSettings,
}) => {
  const { t } = useTranslation("agent");
  const translateAgentCopyKey = React.useCallback(
    (key: string, values?: Record<string, number | string>) =>
      t(key as AgentI18nKey, values ?? {}),
    [t],
  );
  const curatedTaskTemplateCopy = React.useMemo(
    () => buildCuratedTaskTemplateCopy(translateAgentCopyKey),
    [translateAgentCopyKey],
  );
  const inputCapabilityCopy = React.useMemo(
    () => buildInputCapabilitySectionsCopy(translateAgentCopyKey),
    [translateAgentCopyKey],
  );
  const sections = React.useMemo(() => {
    void curatedTaskTemplatesVersion;
    void curatedTaskRecommendationSignalsVersion;
    void mentionEntryUsageVersion;
    void slashEntryUsageVersion;
    return buildInputCapabilitySections({
      mode,
      mentionQuery,
      builtinCommands,
      slashCommands,
      sceneCommands,
      mentionServiceSkills,
      pluginSuggestions,
      serviceSkillGroups,
      filteredCharacters,
      installedSkills,
      availableSkills,
      projectId,
      sessionId,
      referenceEntries,
      curatedTaskTemplateCopy,
      inputCapabilityCopy,
    });
  }, [
    availableSkills,
    builtinCommands,
    curatedTaskTemplateCopy,
    curatedTaskTemplatesVersion,
    curatedTaskRecommendationSignalsVersion,
    filteredCharacters,
    inputCapabilityCopy,
    installedSkills,
    mentionEntryUsageVersion,
    mentionQuery,
    mentionServiceSkills,
    pluginSuggestions,
    mode,
    projectId,
    referenceEntries,
    sceneCommands,
    serviceSkillGroups,
    sessionId,
    slashEntryUsageVersion,
    slashCommands,
  ]);
  const hasFilteredResults = sections.some(
    (section) => section.items.length > 0,
  );
  const showProjectFileSection =
    mode === "mention" &&
    mentionQuery.trim().length > 0 &&
    (projectFiles.length > 0 ||
      projectFileSearchStatus === "loading" ||
      projectFileSearchStatus === "ready" ||
      projectFileSearchStatus === "error");
  const isEmptySlashQuery =
    mode === "slash" && mentionQuery.trim().length === 0;
  const isEmptyMentionQuery =
    mode === "mention" && mentionQuery.trim().length === 0;
  const isRegistryLanding = isEmptySlashQuery || isEmptyMentionQuery;

  const resolveSectionTone = (sectionKey: string) => {
    if (sectionKey === "result-templates") {
      return "primary" as const;
    }

    if (sectionKey === "recent-mention") {
      return "continuation" as const;
    }

    if (sectionKey === "recent-slash-continuations") {
      return "continuation" as const;
    }

    if (sectionKey.startsWith("builtin-commands:")) {
      return "primary" as const;
    }

    if (
      sectionKey === "featured-service-skills" ||
      sectionKey.startsWith("service-skill-group:") ||
      sectionKey === "characters"
    ) {
      return "supporting" as const;
    }

    if (
      sectionKey === "installed-skills" ||
      sectionKey === "available-skills"
    ) {
      return "methods" as const;
    }

    if (
      sectionKey === "supported-slash-commands:workspace-action" ||
      sectionKey === "recent-slash-operations"
    ) {
      return "supporting" as const;
    }

    return "default" as const;
  };

  const resolveSectionHelperText = (sectionKey: string): string | null => {
    if (isEmptyMentionQuery) {
      if (sectionKey === "recent-mention") {
        return t("inputCapabilities.panel.helper.recentMention");
      }

      if (sectionKey === "featured-service-skills") {
        return t("inputCapabilities.panel.helper.featuredServiceSkills");
      }

      if (sectionKey === "plugins") {
        return t("inputCapabilities.panel.helper.plugins");
      }

      if (sectionKey === "installed-skills") {
        return t("inputCapabilities.panel.helper.installedSkillsMention");
      }

      if (sectionKey === "available-skills") {
        return t("inputCapabilities.panel.helper.availableSkills");
      }

      if (sectionKey === "characters") {
        return t("inputCapabilities.panel.helper.characters");
      }
    }

    if (!isEmptySlashQuery) {
      return null;
    }

    if (sectionKey === "supported-slash-commands:workspace-action") {
      return t("inputCapabilities.panel.helper.workspaceAction");
    }

    if (sectionKey === "recent-slash-operations") {
      return t("inputCapabilities.panel.helper.recentOperations");
    }

    if (sectionKey === "recent-slash-continuations") {
      return t("inputCapabilities.panel.helper.recentContinuations");
    }

    if (sectionKey === "installed-skills") {
      return t("inputCapabilities.panel.helper.installedSkillsSlash");
    }

    return null;
  };

  const shouldCompactSectionItems = (sectionKey: string): boolean =>
    isEmptySlashQuery &&
    sectionKey === "supported-slash-commands:workspace-action";

  const shouldSubdueMethodItems = (sectionKey: string): boolean =>
    (isEmptySlashQuery && sectionKey === "installed-skills") ||
    (isEmptyMentionQuery &&
      ([
        "featured-service-skills",
        "plugins",
        "installed-skills",
        "available-skills",
        "characters",
      ].includes(sectionKey) ||
        sectionKey.startsWith("service-skill-group:")));

  const shouldHighlightContinuationItems = (sectionKey: string): boolean =>
    (isEmptySlashQuery && sectionKey === "recent-slash-continuations") ||
    (isEmptyMentionQuery && sectionKey === "recent-mention");

  const resolveVisibleKindLabel = (
    item: InputCapabilityDescriptor,
  ): string | null => {
    if (item.kind !== "slash_command") {
      return null;
    }

    const normalizedKindLabel = item.kindLabel?.trim();
    if (!normalizedKindLabel) {
      return null;
    }

    return normalizedKindLabel === item.title.trim()
      ? null
      : normalizedKindLabel;
  };

  const handleSelectCapability = (item: InputCapabilityDescriptor) => {
    onSelectCapability(item);
  };

  const renderItemIcon = (
    item: InputCapabilityDescriptor,
    sectionKey?: string,
  ) => {
    const iconClassName = cn(
      item.iconClassName,
      sectionKey && shouldSubdueMethodItems(sectionKey) && "text-slate-400",
    );

    switch (item.icon) {
      case "blocks":
        return <Blocks className={iconClassName} />;
      case "command":
        return <CommandIcon className={iconClassName} />;
      case "image-plus":
        return <ImagePlus className={iconClassName} />;
      case "sparkles":
        return <Sparkles className={iconClassName} />;
      case "user":
        return <User className={iconClassName} />;
      case "zap":
        return <Zap className={iconClassName} />;
      default:
        return null;
    }
  };

  return (
    <Command ref={commandRef} className="bg-background" shouldFilter={false}>
      <CommandInput
        placeholder={
          mode === "slash"
            ? t("inputCapabilities.panel.searchPlaceholder.slash")
            : t("inputCapabilities.panel.searchPlaceholder.mention")
        }
        value={mentionQuery}
        onValueChange={onQueryChange}
      />
      <CommandList>
        {!hasFilteredResults && !showProjectFileSection ? (
          <div className="px-3 py-6 text-center text-sm text-muted-foreground">
            <div>
              {mode === "slash"
                ? t("inputCapabilities.panel.empty.slash")
                : t("inputCapabilities.panel.empty.mention")}
            </div>
            {onNavigateToSettings ? (
              <button
                type="button"
                className="mt-2 text-primary hover:underline"
                onMouseDown={(e) => e.preventDefault()}
                onClick={onNavigateToSettings}
              >
                {t("inputCapabilities.panel.action.skillCenter")}
              </button>
            ) : null}
          </div>
        ) : null}
        {showProjectFileSection ? (
          <CommandGroup heading={t("inputCapabilities.heading.projectFiles")}>
            {projectFileSearchStatus === "loading" &&
            projectFiles.length === 0 ? (
              <div
                className="px-2 py-2 text-xs text-slate-500"
                data-testid="project-file-search-loading"
              >
                {t("inputCapabilities.projectFiles.loading")}
              </div>
            ) : null}
            {projectFileSearchStatus === "error" ? (
              <div
                className="px-2 py-2 text-xs text-rose-600"
                data-testid="project-file-search-error"
              >
                {t("inputCapabilities.projectFiles.error")}
              </div>
            ) : null}
            {projectFileSearchStatus === "ready" &&
            projectFiles.length === 0 ? (
              <div
                className="px-2 py-2 text-xs text-slate-500"
                data-testid="project-file-search-empty"
              >
                {t("inputCapabilities.projectFiles.empty")}
              </div>
            ) : null}
            {projectFiles.map((file) => (
              <CommandItem
                key={`${file.root}:${file.path}`}
                value={`project-file:${file.path}`}
                className="items-start gap-3 px-2 py-2.5"
                data-testid="project-file-search-result"
                onSelect={() => onSelectProjectFile?.(file)}
              >
                <File className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" />
                <div className="min-w-0 flex-1 text-left">
                  <div className="truncate text-sm font-medium text-slate-900">
                    {file.file_name}
                  </div>
                  <div className="truncate text-[11px] leading-5 text-slate-500">
                    {file.path}
                  </div>
                </div>
              </CommandItem>
            ))}
          </CommandGroup>
        ) : null}
        {sections.map((section) => (
          <CommandGroup
            key={section.key}
            heading={section.heading}
            className={cn(
              resolveSectionTone(section.key) === "supporting" &&
                isRegistryLanding &&
                "[&_[cmdk-group-heading]]:pb-0.5 [&_[cmdk-group-heading]]:text-[11px] [&_[cmdk-group-heading]]:font-normal [&_[cmdk-group-heading]]:text-slate-400",
              resolveSectionTone(section.key) === "methods" &&
                isRegistryLanding &&
                "[&_[cmdk-group-heading]]:pb-0.5 [&_[cmdk-group-heading]]:text-[11px] [&_[cmdk-group-heading]]:font-normal [&_[cmdk-group-heading]]:text-slate-500",
              resolveSectionTone(section.key) === "continuation" &&
                isRegistryLanding &&
                "[&_[cmdk-group-heading]]:pb-0.5 [&_[cmdk-group-heading]]:text-[11px] [&_[cmdk-group-heading]]:font-medium [&_[cmdk-group-heading]]:text-emerald-700",
            )}
          >
            {section.banner ? (
              <div
                className="mx-2 mb-2 rounded-[16px] border border-sky-200/80 bg-[linear-gradient(180deg,rgba(248,250,252,0.98),rgba(239,246,255,0.92))] px-3 py-3"
                data-testid={`input-capability-section-banner-${section.key}`}
              >
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      {section.banner.badge ? (
                        <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-700">
                          {section.banner.badge}
                        </span>
                      ) : null}
                      <div className="text-xs font-semibold leading-5 text-slate-900">
                        {section.banner.title}
                      </div>
                    </div>
                    <div className="mt-1.5 text-[11px] leading-5 text-slate-600">
                      {section.banner.summary}
                    </div>
                    {section.banner.footnote ? (
                      <div className="mt-1 text-[11px] leading-5 text-sky-700">
                        {section.banner.footnote}
                      </div>
                    ) : null}
                  </div>
                  {section.banner.actionLabel &&
                  section.banner.actionItemKey ? (
                    <button
                      type="button"
                      className="rounded-full border border-sky-200 bg-white px-3 py-1 text-[11px] font-medium leading-5 text-slate-700 transition-colors hover:border-sky-300 hover:bg-sky-50"
                      data-testid={`input-capability-section-banner-action-${section.key}`}
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={() => {
                        const actionItem = section.items.find(
                          (item) => item.key === section.banner?.actionItemKey,
                        );
                        if (actionItem) {
                          handleSelectCapability(actionItem);
                        }
                      }}
                    >
                      {section.banner.actionLabel}
                    </button>
                  ) : null}
                </div>
              </div>
            ) : null}
            {resolveSectionHelperText(section.key) ? (
              <div
                className="px-2 pb-1.5 text-[11px] leading-5 text-slate-400"
                data-testid={`input-capability-section-helper-${section.key}`}
              >
                {resolveSectionHelperText(section.key)}
              </div>
            ) : null}
            {section.items.map((item) => {
              const visibleKindLabel = resolveVisibleKindLabel(item);
              const disabled =
                item.kind === "available_skill" ||
                (item.kind === "plugin" && item.disabled === true);

              return (
                <CommandItem
                  key={item.key}
                  disabled={disabled}
                  onSelect={() => {
                    if (!disabled) {
                      handleSelectCapability(item);
                    }
                  }}
                  className={cn(
                    disabled ? "cursor-pointer opacity-60" : "cursor-pointer",
                    shouldCompactSectionItems(section.key) && "min-h-0 py-1.5",
                    shouldSubdueMethodItems(section.key) && "py-1.5",
                    isRegistryLanding &&
                      resolveSectionTone(section.key) === "primary" &&
                      "py-2",
                    shouldHighlightContinuationItems(section.key) &&
                      "rounded-md border border-emerald-100/80 bg-emerald-50/50",
                  )}
                >
                  {renderItemIcon(item, section.key)}
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <div
                        className={cn(
                          "font-medium",
                          shouldCompactSectionItems(section.key) &&
                            "text-sm font-normal text-slate-700",
                          shouldSubdueMethodItems(section.key) &&
                            "text-sm font-normal text-slate-700",
                        )}
                      >
                        {item.title}
                      </div>
                      {visibleKindLabel ? (
                        <span
                          className={cn(
                            "text-[10px] text-muted-foreground",
                            shouldCompactSectionItems(section.key) &&
                              "text-[10px] text-slate-400",
                            shouldSubdueMethodItems(section.key) &&
                              "text-[10px] text-slate-400",
                          )}
                        >
                          {visibleKindLabel}
                        </span>
                      ) : null}
                    </div>
                    {!shouldCompactSectionItems(section.key) ? (
                      <div
                        className={cn(
                          "text-xs text-muted-foreground line-clamp-1",
                          isRegistryLanding &&
                            resolveSectionTone(section.key) === "primary" &&
                            "text-[11px] leading-5",
                          shouldSubdueMethodItems(section.key) &&
                            "text-[11px] leading-5 text-slate-500",
                        )}
                      >
                        {item.description}
                      </div>
                    ) : null}
                  </div>
                </CommandItem>
              );
            })}
          </CommandGroup>
        ))}
      </CommandList>
    </Command>
  );
};
