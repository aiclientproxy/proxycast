import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { open as openDialog } from "@/lib/desktop-host/dialog";
import { toast } from "sonner";
import { useTranslation } from "react-i18next";
import { useSkills } from "@/hooks/useSkills";
import {
  skillsApi,
  type CreateSkillScaffoldRequest,
  type Skill,
} from "@/lib/api/skills";
import type {
  Page,
  PageParams,
  SkillScaffoldDraft,
  SkillsPageParams,
} from "@/types/page";
import type { AgentRuntimeWorkspaceSkillBinding } from "@/lib/api/agentRuntime/toolInventoryTypes";
import { buildServiceSkillRecommendationBuckets } from "@/components/agent/chat/service-skills/recommendedServiceSkills";
import { resolveServiceSkillLaunchPrefill } from "@/components/agent/chat/service-skills/serviceSkillLaunchPrefill";
import type { ServiceSkillHomeItem } from "@/components/agent/chat/service-skills/types";
import { useServiceSkills } from "@/components/agent/chat/service-skills/useServiceSkills";
import { buildHomeAgentParams } from "@/lib/workspace/navigation";
import { buildSkillScaffoldCreationReplayRequestMetadata } from "@/components/agent/chat/utils/creationReplayMetadata";
import { recordSlashEntryUsage } from "@/components/agent/chat/skill-selection/slashEntryUsage";
import { useOfficialSkillMarketplace } from "@/hooks/useOfficialSkillMarketplace";
import { installOfficialMarketplaceSkill } from "@/lib/api/officialSkillMarketplace";
import {
  buildInstalledLocalSkills,
  buildSkillStoreItems,
  getVisibleBuiltinLocalSkills,
  getVisibleInstalledLocalSkills,
  getVisibleSkillStoreItems,
  getVisibleUserInstalledSkills,
  isMarketplaceSkillInstalledAsLocalSkill,
  splitFeaturedSkillStoreItems,
  type SkillsWorkspaceView,
  type SkillStoreItem,
} from "./SkillsWorkspacePageViewModel";
import { basenameFromPath } from "./SkillsWorkspacePageContent";
import type {
  MarketplaceSkillActionState,
  SkillsWorkspaceViewTab,
} from "./SkillsWorkspacePageTypes";
import { useSkillsWorkspaceCopy } from "./SkillsWorkspacePageCopy";
import { SkillsWorkspacePageView } from "./SkillsWorkspacePageView";
import { useSkillsWorkspaceProject } from "./useSkillsWorkspaceProject";
import { useSkillsWorkspaceDetailContent } from "./useSkillsWorkspaceDetailContent";
import { useSkillsWorkspaceLocalSkillActions } from "./useSkillsWorkspaceLocalSkillActions";
import { SkillScaffoldDialog } from "./SkillScaffoldDialog";
import { SkillPackageInstallDialog } from "./SkillPackageInstallDialog";
import {
  buildSkillScaffoldCreationSeed,
  buildSkillScaffoldReplayText,
} from "./skillScaffoldCreationSeed";
import { buildWorkspaceSkillRuntimeLaunchParams } from "./workspaceSkillRuntimeLaunch";

interface SkillsWorkspacePageProps {
  onNavigate: (page: Page, params?: PageParams) => void;
  pageParams?: SkillsPageParams;
}

export function SkillsWorkspacePage({
  onNavigate,
  pageParams,
}: SkillsWorkspacePageProps) {
  const { t, i18n } = useTranslation("agent");
  const {
    installedSkillPresentationCopy,
    serviceSkillLaunchPrefillCopy,
    serviceSkillPresentationCopy,
  } = useSkillsWorkspaceCopy(t, i18n.language);
  const {
    skills: serviceSkills = [],
    error: serviceSkillsError,
    refresh: refreshServiceSkills,
  } = useServiceSkills();
  const {
    skills: officialMarketplaceSkills = [],
    isLoading: officialMarketplaceLoading,
    error: officialMarketplaceError,
    refresh: refreshOfficialMarketplace,
  } = useOfficialSkillMarketplace();
  const {
    skills: localSkills = [],
    loading: localSkillsLoading,
    error: localSkillsError,
    refresh: refreshLocalSkills,
    uninstall: uninstallLocalSkill,
  } = useSkills("lime", { includeRepos: false });

  const [searchQuery, setSearchQuery] = useState("");
  const [activeView, setActiveView] = useState<SkillsWorkspaceView>(
    pageParams?.initialView === "manage" ||
      pageParams?.initialView === "installed"
      ? "installed"
      : "store",
  );
  const [refreshing, setRefreshing] = useState(false);
  const [installingMarketplaceSkillName, setInstallingMarketplaceSkillName] =
    useState<string | null>(null);
  const [selectedMarketplaceSkillName, setSelectedMarketplaceSkillName] =
    useState<string | null>(null);
  const [detailMarketplaceSkillName, setDetailMarketplaceSkillName] = useState<
    string | null
  >(null);
  const [detailInstalledSkillDirectory, setDetailInstalledSkillDirectory] =
    useState<string | null>(null);
  const [scaffoldDialogOpen, setScaffoldDialogOpen] = useState(false);
  const [scaffoldDialogDraft, setScaffoldDialogDraft] =
    useState<SkillScaffoldDraft | null>(null);
  const [scaffoldCreating, setScaffoldCreating] = useState(false);
  const [registeredSkillsRefreshSignal, setRegisteredSkillsRefreshSignal] =
    useState(0);
  const [localPackageDialogOpen, setLocalPackageDialogOpen] = useState(false);
  const [localPackageSourcePath, setLocalPackageSourcePath] = useState<
    string | null
  >(null);
  const [localPackageSourceName, setLocalPackageSourceName] = useState<
    string | null
  >(null);
  const [selectingLocalSkillPackage, setSelectingLocalSkillPackage] =
    useState(false);
  const [
    highlightedInstalledSkillDirectory,
    setHighlightedInstalledSkillDirectory,
  ] = useState<string | null>(null);
  const [optimisticInstalledSkill, setOptimisticInstalledSkill] =
    useState<Skill | null>(null);
  const [
    optimisticallyHiddenSkillDirectories,
    setOptimisticallyHiddenSkillDirectories,
  ] = useState<Set<string>>(() => new Set());
  const [consumedScaffoldRequestKey, setConsumedScaffoldRequestKey] = useState<
    number | null
  >(null);
  const lastHandledScaffoldRequestKeyRef = useRef<number | null>(null);
  const lastHandledSkillPackageRequestKeyRef = useRef<number | string | null>(
    null,
  );
  const lastHandledInitialSearchRequestKeyRef = useRef<number | string | null>(
    null,
  );
  const creationProjectId = pageParams?.creationProjectId?.trim() || undefined;

  const { currentProjectState } = useSkillsWorkspaceProject({
    activeView,
    creationProjectId,
    localSkillsLoading,
  });
  const {
    exportingSkillDirectory,
    handleExportLocalSkillPackage,
    handleRenameLocalSkill,
    handleReplaceLocalSkillPackage,
    handleRevealLocalSkill,
    handleSkillAutoLoadChange,
    handleUninstallLocalSkill,
    renamingSkillDirectory,
    replacingSkillDirectory,
    revealingSkillDirectory,
    skillAutoLoadPreferences,
    uninstallingSkillDirectory,
  } = useSkillsWorkspaceLocalSkillActions({
    detailInstalledSkillDirectory,
    highlightedInstalledSkillDirectory,
    optimisticInstalledSkillDirectory:
      optimisticInstalledSkill?.directory ?? null,
    refreshLocalSkills,
    setDetailInstalledSkillDirectory,
    setHighlightedInstalledSkillDirectory,
    setOptimisticInstalledSkill,
    setOptimisticallyHiddenSkillDirectories,
    t,
    uninstallLocalSkill,
  });

  const installedLocalSkills = useMemo(
    () => buildInstalledLocalSkills(localSkills, optimisticInstalledSkill),
    [localSkills, optimisticInstalledSkill],
  );
  useEffect(() => {
    if (optimisticallyHiddenSkillDirectories.size === 0) {
      return;
    }

    setOptimisticallyHiddenSkillDirectories((previous) => {
      const next = new Set(
        [...previous].filter((directory) =>
          localSkills.some((skill) => skill.directory === directory),
        ),
      );
      return next.size === previous.size ? previous : next;
    });
  }, [localSkills, optimisticallyHiddenSkillDirectories]);

  const activeInstalledLocalSkills = useMemo(
    () =>
      installedLocalSkills.filter(
        (skill) => !optimisticallyHiddenSkillDirectories.has(skill.directory),
      ),
    [installedLocalSkills, optimisticallyHiddenSkillDirectories],
  );
  const findInstalledMarketplaceLocalSkill = useCallback(
    (item: SkillStoreItem): Skill | undefined => {
      if (item.source !== "official") {
        return undefined;
      }
      return activeInstalledLocalSkills.find((localSkill) =>
        isMarketplaceSkillInstalledAsLocalSkill({
          marketplaceSkill: item.skill,
          localSkill,
        }),
      );
    },
    [activeInstalledLocalSkills],
  );
  const serviceSkillRecommendationBuckets = useMemo(
    () =>
      buildServiceSkillRecommendationBuckets(serviceSkills, {
        featuredLimit: 0,
        surface: "workspace",
      }),
    [serviceSkills],
  );
  const recentServiceSkills = serviceSkillRecommendationBuckets.recentSkills;
  const nonRecentServiceSkills =
    serviceSkillRecommendationBuckets.remainingSkills;
  const workspaceServiceSkills = useMemo(
    () => [...recentServiceSkills, ...nonRecentServiceSkills],
    [nonRecentServiceSkills, recentServiceSkills],
  );
  const scaffoldCreationReplay = useMemo(() => {
    if (!pageParams?.initialScaffoldDraft) {
      return undefined;
    }

    return buildSkillScaffoldCreationReplayRequestMetadata(
      pageParams.initialScaffoldDraft,
      {
        projectId: creationProjectId,
      },
    ).harness.creation_replay;
  }, [creationProjectId, pageParams?.initialScaffoldDraft]);

  useEffect(() => {
    if (
      pageParams?.initialView === "manage" ||
      pageParams?.initialView === "installed"
    ) {
      setActiveView("installed");
      return;
    }
    if (pageParams?.initialView === "builtin") {
      setActiveView("builtin");
      return;
    }
    if (pageParams?.initialView === "store") {
      setActiveView("store");
    }
  }, [pageParams?.initialView]);

  useEffect(() => {
    const initialSearchQuery = pageParams?.initialSearchQuery?.trim();
    if (!initialSearchQuery) {
      return;
    }

    const requestKey =
      pageParams?.initialSearchRequestKey ?? initialSearchQuery;
    if (lastHandledInitialSearchRequestKeyRef.current === requestKey) {
      return;
    }

    lastHandledInitialSearchRequestKeyRef.current = requestKey;
    setActiveView("installed");
    setSearchQuery(initialSearchQuery);
  }, [pageParams?.initialSearchQuery, pageParams?.initialSearchRequestKey]);

  useEffect(() => {
    const requestKey = pageParams?.initialScaffoldRequestKey ?? null;
    if (
      !pageParams?.initialScaffoldDraft ||
      requestKey === null ||
      lastHandledScaffoldRequestKeyRef.current === requestKey
    ) {
      return;
    }

    lastHandledScaffoldRequestKeyRef.current = requestKey;
    setScaffoldDialogDraft(pageParams.initialScaffoldDraft);
    setScaffoldDialogOpen(true);
    setActiveView("installed");
  }, [pageParams?.initialScaffoldDraft, pageParams?.initialScaffoldRequestKey]);

  useEffect(() => {
    const sourcePath = pageParams?.initialSkillPackagePath?.trim();
    if (!sourcePath) {
      return;
    }

    const requestKey = pageParams?.initialSkillPackageRequestKey ?? sourcePath;
    if (lastHandledSkillPackageRequestKeyRef.current === requestKey) {
      return;
    }

    lastHandledSkillPackageRequestKeyRef.current = requestKey;
    setLocalPackageSourcePath(sourcePath);
    setLocalPackageSourceName(
      pageParams?.initialSkillPackageName?.trim() || null,
    );
    setLocalPackageDialogOpen(true);
    setActiveView("installed");
    setSearchQuery("");
  }, [
    pageParams?.initialSkillPackageName,
    pageParams?.initialSkillPackagePath,
    pageParams?.initialSkillPackageRequestKey,
  ]);

  const visibleInstalledLocalSkills = useMemo(
    () =>
      getVisibleInstalledLocalSkills({
        installedLocalSkills: activeInstalledLocalSkills,
        searchQuery,
        highlightedInstalledSkillDirectory,
        copy: installedSkillPresentationCopy,
      }),
    [
      activeInstalledLocalSkills,
      highlightedInstalledSkillDirectory,
      installedSkillPresentationCopy,
      searchQuery,
    ],
  );
  const skillStoreItems = useMemo<SkillStoreItem[]>(
    () =>
      buildSkillStoreItems({
        officialMarketplaceSkills,
        workspaceServiceSkills,
      }),
    [officialMarketplaceSkills, workspaceServiceSkills],
  );
  const visibleStoreItems = useMemo(
    () =>
      getVisibleSkillStoreItems({
        skillStoreItems,
        searchQuery,
        serviceSkillPresentationCopy,
      }),
    [searchQuery, serviceSkillPresentationCopy, skillStoreItems],
  );
  const { featuredStoreItems, otherStoreItems } = useMemo(
    () => splitFeaturedSkillStoreItems(visibleStoreItems),
    [visibleStoreItems],
  );
  const selectedStoreItem = useMemo(() => {
    if (!selectedMarketplaceSkillName) {
      return null;
    }
    return (
      visibleStoreItems.find(
        (item) => item.skill.name === selectedMarketplaceSkillName,
      ) ?? null
    );
  }, [selectedMarketplaceSkillName, visibleStoreItems]);
  const detailStoreItem = useMemo(() => {
    if (!detailMarketplaceSkillName) {
      return null;
    }
    return (
      skillStoreItems.find(
        (item) => item.skill.name === detailMarketplaceSkillName,
      ) ?? null
    );
  }, [detailMarketplaceSkillName, skillStoreItems]);
  const detailInstalledSkill = useMemo(() => {
    if (!detailInstalledSkillDirectory) {
      return null;
    }
    return (
      activeInstalledLocalSkills.find(
        (skill) => skill.directory === detailInstalledSkillDirectory,
      ) ?? null
    );
  }, [activeInstalledLocalSkills, detailInstalledSkillDirectory]);
  const {
    detailContentState,
    installedDetailContentState,
    installedDetailSelectedFilePath,
    setInstalledDetailSelectedFilePath,
  } = useSkillsWorkspaceDetailContent({
    detailInstalledSkill,
    detailStoreItem,
  });
  const visibleBuiltinLocalSkills = useMemo(
    () => getVisibleBuiltinLocalSkills({ localSkills, searchQuery }),
    [localSkills, searchQuery],
  );
  const visibleUserInstalledSkills = useMemo(
    () => getVisibleUserInstalledSkills(visibleInstalledLocalSkills),
    [visibleInstalledLocalSkills],
  );
  const skillStoreCount = skillStoreItems.length;
  const viewTabs: SkillsWorkspaceViewTab[] = [
    {
      key: "store",
      label: t("skills.workspace.view.store"),
      count: skillStoreCount,
    },
    {
      key: "builtin",
      label: t("skills.workspace.view.builtin"),
      count: visibleBuiltinLocalSkills.length,
    },
    {
      key: "installed",
      label: t("skills.workspace.view.installed"),
      count: visibleUserInstalledSkills.length,
    },
  ];
  const activeScaffoldDraft =
    pageParams?.initialScaffoldRequestKey === consumedScaffoldRequestKey
      ? null
      : (pageParams?.initialScaffoldDraft ?? null);
  const activeScaffoldTitle = useMemo(
    () =>
      activeScaffoldDraft?.name?.trim() ||
      t("skills.workspace.scaffold.defaultTitle"),
    [activeScaffoldDraft, t],
  );

  const handleRefreshAll = useCallback(async () => {
    setRefreshing(true);
    try {
      await Promise.allSettled([
        refreshOfficialMarketplace(),
        refreshServiceSkills(),
        refreshLocalSkills(),
      ]);
      toast.success(t("skills.workspace.feedback.refreshSuccess"));
    } catch (error) {
      toast.error(
        t("skills.workspace.feedback.refreshError", {
          message: String(error),
        }),
      );
    } finally {
      setRefreshing(false);
    }
  }, [refreshLocalSkills, refreshOfficialMarketplace, refreshServiceSkills, t]);

  const handleServiceSkillSelect = useCallback(
    (skill: ServiceSkillHomeItem) => {
      const prefill = resolveServiceSkillLaunchPrefill({
        skill,
        creationReplay: scaffoldCreationReplay,
        copy: serviceSkillLaunchPrefillCopy,
      });
      onNavigate("agent", {
        ...buildHomeAgentParams({
          projectId: creationProjectId,
        }),
        initialPendingServiceSkillLaunch: {
          skillId: skill.id,
          requestKey: Date.now(),
          initialSlotValues: prefill?.slotValues,
          prefillHint: prefill?.hint,
          launchUserInput: prefill?.launchUserInput,
        },
      });
    },
    [
      creationProjectId,
      onNavigate,
      scaffoldCreationReplay,
      serviceSkillLaunchPrefillCopy,
    ],
  );

  const handleInstalledSkillSelect = useCallback(
    (skill: Skill) => {
      onNavigate("agent", {
        ...buildHomeAgentParams({
          projectId: creationProjectId,
        }),
        initialInputCapability: {
          capabilityRoute: {
            kind: "installed_skill",
            skillKey: skill.key,
            skillName: skill.name,
          },
          requestKey: Date.now(),
        },
        preferHomeForInitialInputCapability: true,
      });
    },
    [creationProjectId, onNavigate],
  );

  const handleEnableRegisteredSkillRuntime = useCallback(
    (binding: AgentRuntimeWorkspaceSkillBinding) => {
      const params = buildWorkspaceSkillRuntimeLaunchParams({
        workspaceRoot: currentProjectState.rootPath,
        projectId: creationProjectId,
        binding,
        prompt: t("skills.workspace.runtimeEnable.initialPrompt", {
          name: binding.name || binding.directory,
          directory: binding.directory,
        }),
      });

      if (!params) {
        toast.error(
          t("skills.workspace.runtimeEnable.unavailable", {
            name: binding.name || binding.directory,
          }),
        );
        return;
      }

      onNavigate("agent", params);
    },
    [creationProjectId, currentProjectState.rootPath, onNavigate, t],
  );

  const handleBringScaffoldToCreation = useCallback(
    (draft: SkillScaffoldDraft) => {
      const seed = buildSkillScaffoldCreationSeed(draft);
      onNavigate(
        "agent",
        buildHomeAgentParams({
          projectId: creationProjectId,
          initialUserPrompt: seed.initialUserPrompt,
          entryBannerMessage: seed.entryBannerMessage,
          initialRequestMetadata:
            buildSkillScaffoldCreationReplayRequestMetadata(draft, {
              projectId: creationProjectId,
            }),
        }),
      );
    },
    [creationProjectId, onNavigate],
  );

  const handleOpenScaffoldDialog = useCallback(() => {
    setScaffoldDialogDraft(null);
    setScaffoldDialogOpen(true);
  }, []);

  const resolveMarketplaceSkillActionState = useCallback(
    (item: SkillStoreItem): MarketplaceSkillActionState => {
      if (item.source === "local_fallback") {
        return "local_fallback";
      }
      if (installingMarketplaceSkillName === item.skill.name) {
        return "installing";
      }

      const localSkill = findInstalledMarketplaceLocalSkill(item);
      if (!localSkill) {
        return "not_installed";
      }
      if (uninstallingSkillDirectory === localSkill.directory) {
        return "uninstalling";
      }
      if (localSkill.sourceKind === "builtin") {
        return "builtin";
      }
      return "installed";
    },
    [
      installingMarketplaceSkillName,
      findInstalledMarketplaceLocalSkill,
      uninstallingSkillDirectory,
    ],
  );

  const getMarketplaceSkillActionLabel = useCallback(
    (state: MarketplaceSkillActionState) => {
      switch (state) {
        case "installing":
          return t("skills.workspace.marketplace.action.installing");
        case "installed":
          return t("skills.workspace.marketplace.action.useInstalled");
        case "builtin":
          return t("skills.workspace.marketplace.action.useBuiltin");
        case "uninstalling":
          return t("skills.workspace.marketplace.action.uninstalling");
        case "local_fallback":
          return t("skills.workspace.marketplace.action.useLocal");
        default:
          return t("skills.workspace.marketplace.action.install");
      }
    },
    [t],
  );

  const handleMarketplaceSkillInstall = useCallback(
    async (item: SkillStoreItem) => {
      if (item.source !== "official") {
        return;
      }
      setInstallingMarketplaceSkillName(item.skill.name);
      try {
        const result = await installOfficialMarketplaceSkill(
          item.skill.name,
          "lime",
        );
        setOptimisticallyHiddenSkillDirectories((previous) => {
          if (!previous.has(result.directory)) {
            return previous;
          }
          const next = new Set(previous);
          next.delete(result.directory);
          return next;
        });
        setOptimisticInstalledSkill({
          key: `local:${result.directory}`,
          name: item.skill.title,
          description: item.skill.summary,
          directory: result.directory,
          installed: true,
          sourceKind: "other",
          catalogSource: "user",
          license: result.inspection.license,
          compatibility: result.inspection.compatibility,
          metadata: result.inspection.metadata,
          allowedTools: result.inspection.allowedTools,
          resourceSummary: result.inspection.resourceSummary,
          standardCompliance: result.inspection.standardCompliance,
        });
        await refreshLocalSkills();
        setActiveView("installed");
        setSearchQuery("");
        setHighlightedInstalledSkillDirectory(result.directory);
        toast.success(
          t("skills.workspace.marketplace.installSuccess", {
            title: item.skill.title,
          }),
        );
      } catch (error) {
        toast.error(
          t("skills.workspace.marketplace.installFailed", {
            message: error instanceof Error ? error.message : String(error),
          }),
        );
      } finally {
        setInstallingMarketplaceSkillName(null);
      }
    },
    [refreshLocalSkills, t],
  );

  const handleMarketplaceSkillPrimaryAction = useCallback(
    (item: SkillStoreItem) => {
      const state = resolveMarketplaceSkillActionState(item);
      if (item.source === "local_fallback") {
        handleServiceSkillSelect(item.serviceSkill);
        return;
      }

      const localSkill = findInstalledMarketplaceLocalSkill(item);
      if (localSkill && (state === "installed" || state === "builtin")) {
        handleInstalledSkillSelect(localSkill);
        return;
      }

      if (state === "not_installed") {
        void handleMarketplaceSkillInstall(item);
      }
    },
    [
      handleInstalledSkillSelect,
      handleMarketplaceSkillInstall,
      handleServiceSkillSelect,
      findInstalledMarketplaceLocalSkill,
      resolveMarketplaceSkillActionState,
    ],
  );

  const handleMarketplaceSkillUninstall = useCallback(
    (item: SkillStoreItem) => {
      if (item.source !== "official") {
        return;
      }
      const localSkill = findInstalledMarketplaceLocalSkill(item);
      if (!localSkill || localSkill.sourceKind === "builtin") {
        return;
      }
      void handleUninstallLocalSkill(localSkill);
    },
    [findInstalledMarketplaceLocalSkill, handleUninstallLocalSkill],
  );

  const handleMarketplaceSkillDetailOpen = useCallback((skillName: string) => {
    setSelectedMarketplaceSkillName(skillName);
    setDetailMarketplaceSkillName(skillName);
  }, []);

  const handleScaffoldCreated = useCallback(
    async (skill: Skill) => {
      const scaffoldReplayText = pageParams?.initialScaffoldDraft
        ? buildSkillScaffoldReplayText(pageParams.initialScaffoldDraft)
        : undefined;

      setOptimisticallyHiddenSkillDirectories((previous) => {
        if (!previous.has(skill.directory)) {
          return previous;
        }
        const next = new Set(previous);
        next.delete(skill.directory);
        return next;
      });
      setOptimisticInstalledSkill(skill);
      try {
        await refreshLocalSkills();
      } catch (error) {
        toast.error(
          t("skills.workspace.feedback.refreshError", {
            message: String(error),
          }),
        );
      }

      if (scaffoldReplayText) {
        recordSlashEntryUsage({
          kind: "skill",
          entryId: skill.key,
          replayText: scaffoldReplayText,
        });
      }

      setSearchQuery("");
      setHighlightedInstalledSkillDirectory(skill.directory);
      setActiveView("installed");
      if (skill.catalogSource === "project") {
        setRegisteredSkillsRefreshSignal((value) => value + 1);
      }
      setConsumedScaffoldRequestKey(
        pageParams?.initialScaffoldRequestKey ?? null,
      );
      toast.success(
        t("skills.workspace.scaffold.created", {
          name: skill.name,
        }),
      );
    },
    [
      pageParams?.initialScaffoldDraft,
      pageParams?.initialScaffoldRequestKey,
      refreshLocalSkills,
      t,
    ],
  );

  const handleCreateScaffold = useCallback(
    async (request: CreateSkillScaffoldRequest) => {
      setScaffoldCreating(true);
      try {
        const inspection = await skillsApi.createSkillScaffold(request, "lime");
        const createdSkill: Skill = {
          key: `local:${request.directory}`,
          name: request.name,
          description: request.description,
          directory: request.directory,
          installed: true,
          sourceKind: "other",
          catalogSource: request.target,
          license: inspection.license,
          metadata: inspection.metadata,
          allowedTools: inspection.allowedTools,
          resourceSummary: inspection.resourceSummary,
          standardCompliance: inspection.standardCompliance,
        };
        setScaffoldDialogOpen(false);
        setScaffoldDialogDraft(null);
        await handleScaffoldCreated(createdSkill);
      } catch (error) {
        toast.error(
          t("skills.workspace.scaffold.createFailed", {
            message: error instanceof Error ? error.message : String(error),
          }),
        );
      } finally {
        setScaffoldCreating(false);
      }
    },
    [handleScaffoldCreated, t],
  );

  const handleLocalSkillPackageInstalled = useCallback(
    async (directory: string) => {
      setOptimisticallyHiddenSkillDirectories((previous) => {
        if (!previous.has(directory)) {
          return previous;
        }
        const next = new Set(previous);
        next.delete(directory);
        return next;
      });
      try {
        await refreshLocalSkills();
      } catch (error) {
        toast.error(
          t("skills.workspace.feedback.refreshError", {
            message: String(error),
          }),
        );
      }

      setOptimisticInstalledSkill(null);
      setActiveView("installed");
      setSearchQuery("");
      setHighlightedInstalledSkillDirectory(directory);
    },
    [refreshLocalSkills, t],
  );

  const handleSelectLocalSkillPackage = useCallback(async () => {
    let selected: string | string[] | null;
    setSelectingLocalSkillPackage(true);
    try {
      selected = await openDialog({
        directory: false,
        multiple: false,
        title: t("skills.localPackage.open.dialogTitle"),
        filters: [
          {
            name: t("skills.workspace.export.filterName"),
            extensions: ["skills", "skill"],
          },
        ],
      });
    } catch (error) {
      toast.error(
        t("skills.workspace.import.failed", {
          message: error instanceof Error ? error.message : String(error),
        }),
      );
      return;
    } finally {
      setSelectingLocalSkillPackage(false);
    }

    if (!selected || Array.isArray(selected)) {
      return;
    }

    setLocalPackageSourcePath(selected);
    setLocalPackageSourceName(basenameFromPath(selected));
    setLocalPackageDialogOpen(true);
    setActiveView("installed");
    setSearchQuery("");
  }, [t]);

  return (
    <>
      <SkillsWorkspacePageView
        activeScaffoldDraft={activeScaffoldDraft}
        activeScaffoldTitle={activeScaffoldTitle}
        activeView={activeView}
        currentProjectState={currentProjectState}
        detailContentState={detailContentState}
        detailInstalledSkill={detailInstalledSkill}
        detailStoreItem={detailStoreItem}
        exportingSkillDirectory={exportingSkillDirectory}
        featuredStoreItems={featuredStoreItems}
        findInstalledMarketplaceLocalSkill={findInstalledMarketplaceLocalSkill}
        getMarketplaceSkillActionLabel={getMarketplaceSkillActionLabel}
        highlightedInstalledSkillDirectory={highlightedInstalledSkillDirectory}
        installedDetailContentState={installedDetailContentState}
        installedDetailSelectedFilePath={installedDetailSelectedFilePath}
        installedSkillPresentationCopy={installedSkillPresentationCopy}
        isRefreshing={refreshing}
        isSelectingLocalSkillPackage={selectingLocalSkillPackage}
        localSkillsError={localSkillsError}
        officialMarketplaceError={officialMarketplaceError}
        officialMarketplaceLoading={officialMarketplaceLoading}
        officialMarketplaceSkillCount={officialMarketplaceSkills.length}
        otherStoreItems={otherStoreItems}
        registeredSkillsRefreshSignal={registeredSkillsRefreshSignal}
        renamingSkillDirectory={renamingSkillDirectory}
        replacingSkillDirectory={replacingSkillDirectory}
        resolveMarketplaceSkillActionState={resolveMarketplaceSkillActionState}
        revealingSkillDirectory={revealingSkillDirectory}
        searchQuery={searchQuery}
        selectedStoreItem={selectedStoreItem}
        serviceSkillsError={serviceSkillsError}
        skillAutoLoadPreferences={skillAutoLoadPreferences}
        skillStoreCount={skillStoreCount}
        uninstallingSkillDirectory={uninstallingSkillDirectory}
        viewTabs={viewTabs}
        visibleBuiltinLocalSkills={visibleBuiltinLocalSkills}
        visibleStoreItems={visibleStoreItems}
        visibleUserInstalledSkills={visibleUserInstalledSkills}
        onActiveViewChange={setActiveView}
        onBringScaffoldToCreation={handleBringScaffoldToCreation}
        onEnableRegisteredSkillRuntime={handleEnableRegisteredSkillRuntime}
        onExportLocalSkillPackage={(skill) =>
          void handleExportLocalSkillPackage(skill)
        }
        onInstalledDetailClose={() => setDetailInstalledSkillDirectory(null)}
        onInstalledDetailSelectedFilePathChange={
          setInstalledDetailSelectedFilePath
        }
        onInstalledSkillDetailOpen={setDetailInstalledSkillDirectory}
        onInstalledSkillSelect={handleInstalledSkillSelect}
        onMarketplaceDetailClose={() => setDetailMarketplaceSkillName(null)}
        onMarketplaceSkillDetailOpen={handleMarketplaceSkillDetailOpen}
        onMarketplaceSkillPrimaryAction={handleMarketplaceSkillPrimaryAction}
        onMarketplaceSkillUninstall={handleMarketplaceSkillUninstall}
        onOpenScaffoldDialog={handleOpenScaffoldDialog}
        onRefreshAll={() => void handleRefreshAll()}
        onRenameLocalSkill={(skill) => void handleRenameLocalSkill(skill)}
        onReplaceLocalSkillPackage={(skill) =>
          void handleReplaceLocalSkillPackage(skill)
        }
        onRevealLocalSkill={(skill) => void handleRevealLocalSkill(skill)}
        onSearchQueryChange={setSearchQuery}
        onSelectLocalSkillPackage={() => void handleSelectLocalSkillPackage()}
        onSkillAutoLoadChange={handleSkillAutoLoadChange}
        onUninstallLocalSkill={(skill) => void handleUninstallLocalSkill(skill)}
      />

      <SkillScaffoldDialog
        open={scaffoldDialogOpen}
        onOpenChange={(open) => {
          setScaffoldDialogOpen(open);
          if (!open) {
            setScaffoldDialogDraft(null);
          }
        }}
        onCreate={handleCreateScaffold}
        creating={scaffoldCreating}
        allowProjectTarget={Boolean(creationProjectId)}
        initialValues={scaffoldDialogDraft}
        sourceHint={scaffoldDialogDraft?.sourceExcerpt ?? null}
        onBringBackToCreation={handleBringScaffoldToCreation}
      />

      <SkillPackageInstallDialog
        open={localPackageDialogOpen}
        sourcePath={localPackageSourcePath}
        sourceName={localPackageSourceName}
        onOpenChange={(open) => {
          setLocalPackageDialogOpen(open);
          if (!open) {
            setLocalPackageSourcePath(null);
            setLocalPackageSourceName(null);
          }
        }}
        onInstalled={handleLocalSkillPackageInstalled}
      />
    </>
  );
}
