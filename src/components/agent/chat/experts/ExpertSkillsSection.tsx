import { useEffect, useLayoutEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Check,
  ChevronDown,
  ChevronUp,
  Loader2,
  Plus,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import { buildExpertSkillRuntimeCandidates } from "@/features/experts";
import {
  getCurrentSkillCatalogSnapshot,
  subscribeSkillCatalogChanged,
} from "@/lib/api/skillCatalog";
import type { AgentRuntimeWorkspaceSkillBinding } from "@/lib/api/agentRuntime/toolInventoryTypes";
import type { ServiceSkillItem } from "@/lib/api/serviceSkills";
import type { Skill } from "@/lib/api/skills";
import type { SkillScaffoldDraft } from "@/types/page";
import type { AgentThreadItem } from "../types";
import {
  buildSkillCandidates,
  filterSkillCandidates,
  resolveSkillLabel,
  sanitizeSkillRefTestId,
  skillCandidateInitial,
  type ExpertSkillCandidate,
} from "./expertSkillCandidates";
import {
  addExpertSkillRef,
  dedupeExpertSkillRefs,
  normalizeExpertSkillRefKey,
  removeExpertSkillRef,
  replaceExpertSkillRef,
} from "./expertSkillRefEditing";
import {
  buildExpertSkillRuntimeActionViewModels,
  buildExpertSkillRuntimeChipViewModels,
  buildExpertSkillRuntimeInvocationViewModel,
  buildExpertSkillRuntimeSummaryViewModel,
  buildExpertSkillRuntimeTraceViewModel,
  type ExpertSkillRuntimeRecoveryKind,
} from "./expertSkillRuntimeViewModel";
import { buildExpertSkillRuntimeTimelineViewModel } from "./expertSkillRuntimeTimelineViewModel";
import {
  CandidateAddButton,
  Chip,
  ChipLabel,
  ChipList,
  ChipRemoveButton,
  EmptyCard,
  IconActionButton,
  Section,
  SectionHeaderRow,
  SectionTitle,
  SectionToggleButton,
  SkillCandidateAvatar,
  SkillCandidateCard,
  SkillCandidateContent,
  SkillCandidateList,
  SkillCandidateMeta,
  SkillCandidateSummary,
  SkillCandidateTitle,
  SkillDialog,
  SkillDialogBackdrop,
  SkillDialogBody,
  SkillDialogHeader,
  SkillDialogSubtitle,
  SkillDialogTitle,
  SkillEmptyState,
  SkillReadinessBadge,
  SkillRuntimeActionButton,
  SkillRuntimeActionItem,
  SkillRuntimeActionList,
  SkillRuntimeActionReason,
  SkillRuntimeActionTitle,
  SkillRuntimeEditNotice,
  SkillRuntimeInvocationDetail,
  SkillRuntimeInvocationRow,
  SkillRuntimeInvocationStatus,
  SkillRuntimeSummaryCard,
  SkillRuntimeSummaryCount,
  SkillRuntimeSummaryDetail,
  SkillRuntimeSummaryHeader,
  SkillRuntimeSummaryTitle,
  SkillRuntimeTimelineDetail,
  SkillRuntimeTimelineEmpty,
  SkillRuntimeTimelineItem,
  SkillRuntimeTimelineLabel,
  SkillRuntimeTimelineList,
  SkillRuntimeTimelineText,
  SkillRuntimeTraceBadge,
  SkillSearchBox,
  DialogCloseButton,
} from "./ExpertInfoPanel.styles";

export interface ExpertSkillsSectionProps {
  collapsed: boolean;
  expertId: string;
  baseSkillRefs: string[];
  localSkills?: Skill[];
  serviceSkills?: ServiceSkillItem[];
  workspaceSkillBindings?: AgentRuntimeWorkspaceSkillBinding[];
  skillsLoading?: boolean;
  threadItems?: readonly AgentThreadItem[];
  skillRefsEdited?: boolean;
  enabledWorkspaceSkillRuntimeCount?: number;
  onToggle: () => void;
  onSkillRefsChange?: (skillRefs: string[]) => void;
  onEnableWorkspaceSkillRuntime?: (ref: string) => void;
  onOpenSkillsManage?: (options?: ExpertSkillsManageOptions) => void;
}

export interface ExpertSkillsManageOptions {
  searchQuery?: string;
  scaffoldDraft?: SkillScaffoldDraft;
}

function normalizeScaffoldDirectory(value: string): string {
  return (
    value
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "expert-skill"
  );
}

export function ExpertSkillsSection({
  collapsed,
  expertId,
  baseSkillRefs,
  localSkills,
  serviceSkills,
  workspaceSkillBindings,
  skillsLoading = false,
  threadItems,
  skillRefsEdited = false,
  enabledWorkspaceSkillRuntimeCount = 0,
  onToggle,
  onSkillRefsChange,
  onEnableWorkspaceSkillRuntime,
  onOpenSkillsManage,
}: ExpertSkillsSectionProps) {
  const { t } = useTranslation("agent");
  const [editedSkillRefs, setEditedSkillRefs] = useState<string[] | null>(null);
  const [skillQuery, setSkillQuery] = useState("");
  const [skillPickerOpen, setSkillPickerOpen] = useState(false);
  const [skillPickerReplacementRef, setSkillPickerReplacementRef] = useState<
    string | null
  >(null);
  const [skillPickerRecoveryKind, setSkillPickerRecoveryKind] =
    useState<ExpertSkillRuntimeRecoveryKind | null>(null);
  const [skillCatalog, setSkillCatalog] = useState(() => {
    try {
      return getCurrentSkillCatalogSnapshot();
    } catch {
      return null;
    }
  });
  const candidates = useMemo(
    () =>
      buildSkillCandidates({
        localSkills,
        serviceSkills,
        catalog: skillCatalog,
      }),
    [localSkills, serviceSkills, skillCatalog],
  );
  const filteredCandidates = useMemo(
    () => filterSkillCandidates(candidates, skillQuery),
    [candidates, skillQuery],
  );
  const dedupedBaseSkillRefs = useMemo(
    () => dedupeExpertSkillRefs(baseSkillRefs),
    [baseSkillRefs],
  );
  const dedupedBaseSkillRefsKey = useMemo(
    () => dedupedBaseSkillRefs.map(normalizeExpertSkillRefKey).join("\n"),
    [dedupedBaseSkillRefs],
  );
  const effectiveSkillRefs = editedSkillRefs ?? dedupedBaseSkillRefs;
  const effectiveSkillRefKeys = useMemo(
    () => new Set(effectiveSkillRefs.map(normalizeExpertSkillRefKey)),
    [effectiveSkillRefs],
  );
  useEffect(
    () =>
      subscribeSkillCatalogChanged(() => {
        try {
          setSkillCatalog(getCurrentSkillCatalogSnapshot());
        } catch {
          setSkillCatalog(null);
        }
      }),
    [],
  );
  const runtimeCandidates = useMemo(
    () =>
      buildExpertSkillRuntimeCandidates(effectiveSkillRefs, {
        catalog: skillCatalog,
        localSkills,
        serviceSkills,
        workspaceSkillBindings,
      }),
    [
      effectiveSkillRefs,
      localSkills,
      serviceSkills,
      skillCatalog,
      workspaceSkillBindings,
    ],
  );
  const skillChipViewModels = useMemo(
    () =>
      buildExpertSkillRuntimeChipViewModels({
        skillRefs: effectiveSkillRefs,
        candidates: runtimeCandidates,
        resolveLabel: (ref) => resolveSkillLabel(ref, candidates),
        copy: {
          ready: t("agentExperts.info.skills.readiness.ready", "可运行"),
          needsMapping: t(
            "agentExperts.info.skills.readiness.needsMapping",
            "待映射",
          ),
          needsRegistration: t(
            "agentExperts.info.skills.readiness.needsRegistration",
            "待注册",
          ),
          needsEnable: t(
            "agentExperts.info.skills.readiness.needsEnable",
            "待启用",
          ),
          blocked: t("agentExperts.info.skills.readiness.blocked", "不可用"),
        },
      }),
    [candidates, effectiveSkillRefs, runtimeCandidates, t],
  );
  const skillRuntimeSummary = useMemo(
    () =>
      buildExpertSkillRuntimeSummaryViewModel(skillChipViewModels, {
        readyTitle: t(
          "agentExperts.info.skills.runtime.readyTitle",
          "本轮运行准备就绪",
        ),
        readyDetail: t(
          "agentExperts.info.skills.runtime.readyDetail",
          "已绑定技能都能进入按需加载链路；下一轮请求会先检索，再只读取命中的说明。",
        ),
        partialTitle: t(
          "agentExperts.info.skills.runtime.partialTitle",
          "部分技能还需处理",
        ),
        partialDetail: t(
          "agentExperts.info.skills.runtime.partialDetail",
          "可运行技能会继续生效；其余技能需要补映射或注册后，才能进入同一条按需加载链路。",
        ),
        blockedTitle: t(
          "agentExperts.info.skills.runtime.blockedTitle",
          "技能还不能运行",
        ),
        blockedDetail: t(
          "agentExperts.info.skills.runtime.blockedDetail",
          "当前绑定还没有可读取的本地说明或目录映射，需要先补齐后再试用。",
        ),
        emptyTitle: t(
          "agentExperts.info.skills.runtime.emptyTitle",
          "还没有绑定技能",
        ),
        emptyDetail: t(
          "agentExperts.info.skills.runtime.emptyDetail",
          "添加一个技能后，专家会在下一轮请求中按需检索并只加载命中的说明。",
        ),
      }),
    [skillChipViewModels, t],
  );
  const skillRuntimeActions = useMemo(
    () =>
      buildExpertSkillRuntimeActionViewModels({
        candidates: runtimeCandidates,
        resolveLabel: (ref) => resolveSkillLabel(ref, candidates),
        copy: {
          ready: t("agentExperts.info.skills.action.ready", "可直接试用"),
          needsMapping: t(
            "agentExperts.info.skills.action.needsMapping",
            "补目录映射",
          ),
          needsRegistration: t(
            "agentExperts.info.skills.action.needsRegistration",
            "完成注册",
          ),
          needsEnable: t(
            "agentExperts.info.skills.action.needsEnable",
            "启用运行",
          ),
          blocked: t("agentExperts.info.skills.action.blocked", "检查引用"),
        },
      }),
    [candidates, runtimeCandidates, t],
  );
  const skillRuntimeTrace = useMemo(
    () =>
      buildExpertSkillRuntimeTraceViewModel({
        threadItems,
        copy: {
          none: t(
            "agentExperts.info.skills.trace.none",
            "最近还没有技能加载记录",
          ),
          bodyRead: t(
            "agentExperts.info.skills.trace.bodyRead",
            "最近已按需读取技能说明",
          ),
          gateReady: t(
            "agentExperts.info.skills.trace.gateReady",
            "最近已完成技能授权",
          ),
          gateBlocked: t(
            "agentExperts.info.skills.trace.gateBlocked",
            "最近授权未放行",
          ),
          search: t(
            "agentExperts.info.skills.trace.search",
            "最近已检索技能候选",
          ),
        },
      }),
    [threadItems, t],
  );
  const skillRuntimeInvocation = useMemo(
    () =>
      buildExpertSkillRuntimeInvocationViewModel({
        threadItems,
        copy: {
          none: t(
            "agentExperts.info.skills.invocation.none",
            "最近还没有技能执行记录",
          ),
          running: t(
            "agentExperts.info.skills.invocation.running",
            "最近正在执行技能",
          ),
          completed: t(
            "agentExperts.info.skills.invocation.completed",
            "最近已执行技能",
          ),
          failed: t(
            "agentExperts.info.skills.invocation.failed",
            "最近技能执行失败",
          ),
          unknown: t(
            "agentExperts.info.skills.invocation.unknown",
            "最近技能执行状态待确认",
          ),
        },
      }),
    [threadItems, t],
  );
  const skillRuntimeTimeline = useMemo(
    () =>
      buildExpertSkillRuntimeTimelineViewModel({
        threadItems,
        copy: {
          empty: t(
            "agentExperts.info.skills.timeline.empty",
            "最近还没有完整技能轨迹",
          ),
          search: t("agentExperts.info.skills.timeline.search", "检索候选"),
          bodyRead: t("agentExperts.info.skills.timeline.bodyRead", "读取说明"),
          runtimeEnable: t(
            "agentExperts.info.skills.timeline.runtimeEnable",
            "运行启用",
          ),
          runtimeEnableWithCount: (count) =>
            t("agentExperts.info.skills.timeline.runtimeEnableCount", {
              count,
              defaultValue: "运行启用 {{count}} 个绑定",
            }),
          gateReady: t(
            "agentExperts.info.skills.timeline.gateReady",
            "授权放行",
          ),
          gateBlocked: t(
            "agentExperts.info.skills.timeline.gateBlocked",
            "授权未放行",
          ),
          invocationRunning: t(
            "agentExperts.info.skills.timeline.invocationRunning",
            "执行中",
          ),
          invocationCompleted: t(
            "agentExperts.info.skills.timeline.invocationCompleted",
            "执行完成",
          ),
          invocationFailed: t(
            "agentExperts.info.skills.timeline.invocationFailed",
            "执行失败",
          ),
          invocationUnknown: t(
            "agentExperts.info.skills.timeline.invocationUnknown",
            "执行待确认",
          ),
        },
      }),
    [threadItems, t],
  );
  const handleRuntimeActionClick = (
    item: (typeof skillRuntimeActions)[number],
  ) => {
    const openOptions: ExpertSkillsManageOptions = {
      searchQuery: item.searchQuery,
      scaffoldDraft:
        item.readiness === "needs_registration"
          ? {
              target: "project",
              directory: normalizeScaffoldDirectory(item.searchQuery),
              name: item.label,
              description: t(
                "agentExperts.info.skills.registrationDraft.description",
                { title: item.label, ref: item.ref },
              ),
              whenToUse: [
                t("agentExperts.info.skills.registrationDraft.whenToUse", {
                  title: item.label,
                }),
              ],
              inputs: [
                t("agentExperts.info.skills.registrationDraft.inputs", {
                  title: item.label,
                }),
              ],
              outputs: [
                t("agentExperts.info.skills.registrationDraft.outputs", {
                  title: item.label,
                }),
              ],
              steps: [
                t("agentExperts.info.skills.registrationDraft.steps", {
                  title: item.label,
                }),
              ],
              fallbackStrategy: [
                t(
                  "agentExperts.info.skills.registrationDraft.fallbackStrategy",
                ),
              ],
              sourceExcerpt: item.ref,
            }
          : undefined,
    };
    if (item.recoveryKind === "enable_workspace_skill") {
      if (onEnableWorkspaceSkillRuntime) {
        onEnableWorkspaceSkillRuntime(item.ref);
        return;
      }
      if (onOpenSkillsManage) {
        onOpenSkillsManage(openOptions);
        return;
      }
    }
    if (item.recoveryKind === "open_skills_manage" && onOpenSkillsManage) {
      onOpenSkillsManage(openOptions);
      return;
    }
    setSkillQuery(item.searchQuery);
    setSkillPickerReplacementRef(item.ref);
    setSkillPickerRecoveryKind(item.recoveryKind);
    setSkillPickerOpen(true);
  };

  useLayoutEffect(() => {
    if (editedSkillRefs === null) {
      return;
    }
    onSkillRefsChange?.(editedSkillRefs);
  }, [editedSkillRefs, onSkillRefsChange]);

  useEffect(() => {
    setEditedSkillRefs(null);
    setSkillQuery("");
    setSkillPickerOpen(false);
    setSkillPickerReplacementRef(null);
    setSkillPickerRecoveryKind(null);
  }, [dedupedBaseSkillRefsKey, expertId]);

  const replacingSkill = Boolean(skillPickerReplacementRef);
  const skillPickerDialogTitle = replacingSkill
    ? skillPickerRecoveryKind === "map_skill_ref"
      ? t("agentExperts.info.skills.mapDialogTitle", "补齐技能目录映射")
      : t("agentExperts.info.skills.replaceDialogTitle", "替换当前技能引用")
    : t("agentExperts.info.skills.dialogTitle", "为当前 Agent 添加技能");
  const skillPickerDialogSubtitle = replacingSkill
    ? skillPickerRecoveryKind === "map_skill_ref"
      ? t(
          "agentExperts.info.skills.mapDialogSubtitle",
          "选择一个已安装或目录可识别的技能来替换待映射引用，下一轮会进入同一条按需加载链路。",
        )
      : t(
          "agentExperts.info.skills.replaceDialogSubtitle",
          "选择后会替换当前不可运行的技能引用，并随下一轮专家请求进入运行准备。",
        )
    : t(
        "agentExperts.info.skills.dialogSubtitle",
        "选择后会加入当前专家对话的 skillRefs，并随下一轮 Agent 请求进入运行时上下文。",
      );
  const hasEditedSkillRefs = editedSkillRefs !== null || skillRefsEdited;
  const hasEnabledWorkspaceSkillRuntime = enabledWorkspaceSkillRuntimeCount > 0;
  const handleAddSkill = (ref: string) => {
    setEditedSkillRefs((current) => {
      const refs = current ?? dedupedBaseSkillRefs;
      if (skillPickerReplacementRef) {
        return replaceExpertSkillRef(refs, skillPickerReplacementRef, ref);
      }
      return addExpertSkillRef(refs, ref);
    });
    if (skillPickerReplacementRef) {
      setSkillPickerOpen(false);
      setSkillQuery("");
      setSkillPickerReplacementRef(null);
      setSkillPickerRecoveryKind(null);
    }
  };
  const handleRemoveAddedSkill = (ref: string) => {
    setEditedSkillRefs((current) =>
      removeExpertSkillRef(current ?? dedupedBaseSkillRefs, ref),
    );
  };
  const handleCloseSkillPicker = () => {
    setSkillPickerOpen(false);
    setSkillPickerReplacementRef(null);
    setSkillPickerRecoveryKind(null);
  };
  const resolveCandidateSourceLabel = (candidate: ExpertSkillCandidate) => {
    if (candidate.source === "local") {
      return candidate.sourceLabel === "remote"
        ? t("agentExperts.info.skills.source.remote", "技能市场")
        : t("agentExperts.info.skills.source.local", "本地技能");
    }
    if (candidate.source === "service") {
      return (
        candidate.sourceLabel ||
        t("agentExperts.info.skills.source.service", "服务技能")
      );
    }
    return (
      candidate.sourceLabel ||
      t("agentExperts.info.skills.source.catalog", "技能目录")
    );
  };

  return (
    <Section data-testid="expert-info-section-skills">
      <SectionHeaderRow>
        <SectionToggleButton
          type="button"
          aria-expanded={!collapsed}
          onClick={onToggle}
        >
          <SectionTitle>
            <Sparkles size={14} />
            {t("agentExperts.info.sections.skills", "技能")}
          </SectionTitle>
          {collapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
        </SectionToggleButton>
        <IconActionButton
          type="button"
          aria-label={t(
            "agentExperts.info.skills.addAria",
            "为当前 Agent 添加技能",
          )}
          title={t("agentExperts.info.skills.add", "添加技能")}
          data-testid="expert-info-skills-add"
          onClick={() => setSkillPickerOpen(true)}
        >
          <Plus size={15} />
        </IconActionButton>
      </SectionHeaderRow>
      {collapsed ? null : (
        <>
          <SkillRuntimeSummaryCard
            $tone={skillRuntimeSummary.tone}
            data-testid="expert-info-skills-runtime-summary"
          >
            <SkillRuntimeSummaryHeader>
              <SkillRuntimeSummaryTitle>
                {skillRuntimeSummary.title}
              </SkillRuntimeSummaryTitle>
              <SkillRuntimeSummaryCount $tone={skillRuntimeSummary.tone}>
                {t("agentExperts.info.skills.runtime.count", {
                  ready: skillRuntimeSummary.readyCount,
                  total: skillRuntimeSummary.totalCount,
                  defaultValue: "{{ready}}/{{total}} 可运行",
                })}
              </SkillRuntimeSummaryCount>
            </SkillRuntimeSummaryHeader>
            <SkillRuntimeSummaryDetail>
              {skillRuntimeSummary.detail}
            </SkillRuntimeSummaryDetail>
            {hasEditedSkillRefs ? (
              <SkillRuntimeEditNotice data-testid="expert-info-skills-edit-notice">
                {t(
                  "agentExperts.info.skills.editNotice",
                  "已更新：下一条消息会使用当前技能设置。",
                )}
              </SkillRuntimeEditNotice>
            ) : null}
            {hasEnabledWorkspaceSkillRuntime ? (
              <SkillRuntimeEditNotice data-testid="expert-info-skills-enable-notice">
                {t("agentExperts.info.skills.enableNotice", {
                  count: enabledWorkspaceSkillRuntimeCount,
                  defaultValue:
                    "已选择启用 {{count}} 个工作区技能：下一条消息会把它们加入本轮运行允许列表。",
                })}
              </SkillRuntimeEditNotice>
            ) : null}
            <SkillRuntimeTraceBadge
              $tone={skillRuntimeTrace.tone}
              data-testid="expert-info-skills-runtime-trace"
            >
              {skillRuntimeTrace.label}
            </SkillRuntimeTraceBadge>
            <SkillRuntimeInvocationRow data-testid="expert-info-skills-runtime-invocation">
              <SkillRuntimeInvocationStatus $tone={skillRuntimeInvocation.tone}>
                {skillRuntimeInvocation.label}
              </SkillRuntimeInvocationStatus>
              {skillRuntimeInvocation.skillName ? (
                <SkillRuntimeInvocationDetail>
                  {skillRuntimeInvocation.skillName}
                </SkillRuntimeInvocationDetail>
              ) : null}
            </SkillRuntimeInvocationRow>
            {skillRuntimeTimeline.steps.length > 0 ? (
              <SkillRuntimeTimelineList data-testid="expert-info-skills-runtime-timeline">
                {skillRuntimeTimeline.steps.map((step) => (
                  <SkillRuntimeTimelineItem key={step.id} $tone={step.tone}>
                    <SkillRuntimeTimelineText>
                      <SkillRuntimeTimelineLabel>
                        {step.label}
                      </SkillRuntimeTimelineLabel>
                      {step.detail ? (
                        <SkillRuntimeTimelineDetail>
                          {step.detail}
                        </SkillRuntimeTimelineDetail>
                      ) : null}
                    </SkillRuntimeTimelineText>
                  </SkillRuntimeTimelineItem>
                ))}
              </SkillRuntimeTimelineList>
            ) : (
              <SkillRuntimeTimelineEmpty data-testid="expert-info-skills-runtime-timeline">
                {skillRuntimeTimeline.emptyLabel}
              </SkillRuntimeTimelineEmpty>
            )}
            {skillRuntimeActions.length > 0 ? (
              <SkillRuntimeActionList data-testid="expert-info-skills-runtime-actions">
                {skillRuntimeActions.slice(0, 3).map((item) => (
                  <SkillRuntimeActionItem key={item.ref}>
                    <div>
                      <SkillRuntimeActionTitle>
                        {item.label}
                      </SkillRuntimeActionTitle>
                      <SkillRuntimeActionReason>
                        {item.reason}
                      </SkillRuntimeActionReason>
                    </div>
                    <SkillRuntimeActionButton
                      type="button"
                      aria-label={t(
                        "agentExperts.info.skills.action.openAria",
                        {
                          action: item.actionLabel,
                          title: item.label,
                          defaultValue: "{{action}}：{{title}}",
                        },
                      )}
                      data-testid={`expert-info-skills-runtime-action-${sanitizeSkillRefTestId(
                        item.ref,
                      )}`}
                      onClick={() => handleRuntimeActionClick(item)}
                    >
                      {item.actionLabel}
                    </SkillRuntimeActionButton>
                  </SkillRuntimeActionItem>
                ))}
              </SkillRuntimeActionList>
            ) : null}
          </SkillRuntimeSummaryCard>
          {effectiveSkillRefs.length > 0 ? (
            <ChipList data-testid="expert-info-skills">
              {skillChipViewModels.map((item) => (
                <Chip
                  key={item.ref}
                  title={item.title}
                  data-testid={`expert-info-skill-chip-${sanitizeSkillRefTestId(
                    item.ref,
                  )}`}
                >
                  <ChipLabel>{item.label}</ChipLabel>
                  <SkillReadinessBadge
                    $tone={item.readinessTone}
                    data-testid={`expert-info-skill-readiness-${sanitizeSkillRefTestId(
                      item.ref,
                    )}`}
                  >
                    {item.readinessLabel}
                  </SkillReadinessBadge>
                  <ChipRemoveButton
                    type="button"
                    aria-label={t("agentExperts.info.skills.remove", {
                      title: item.label,
                      defaultValue: "移除 {{title}}",
                    })}
                    onClick={() => handleRemoveAddedSkill(item.ref)}
                  >
                    <X size={11} />
                  </ChipRemoveButton>
                </Chip>
              ))}
            </ChipList>
          ) : (
            <EmptyCard>
              {t("agentExperts.info.skills.empty", "暂无绑定技能")}
            </EmptyCard>
          )}
        </>
      )}

      {skillPickerOpen ? (
        <SkillDialogBackdrop
          role="presentation"
          data-testid="expert-skill-picker-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              handleCloseSkillPicker();
            }
          }}
        >
          <SkillDialog
            role="dialog"
            aria-modal="true"
            aria-labelledby="expert-skill-picker-title"
            data-testid="expert-skill-picker-dialog"
          >
            <SkillDialogHeader>
              <div>
                <SkillDialogTitle id="expert-skill-picker-title">
                  {skillPickerDialogTitle}
                </SkillDialogTitle>
                <SkillDialogSubtitle>
                  {skillPickerDialogSubtitle}
                </SkillDialogSubtitle>
              </div>
              <DialogCloseButton
                type="button"
                aria-label={t("agentExperts.info.skills.close", "关闭")}
                onClick={handleCloseSkillPicker}
              >
                <X size={15} />
              </DialogCloseButton>
            </SkillDialogHeader>
            <SkillDialogBody>
              <SkillSearchBox>
                <Search size={14} />
                <input
                  value={skillQuery}
                  placeholder={t(
                    "agentExperts.info.skills.searchPlaceholder",
                    "搜索技能名称、描述或引用",
                  )}
                  onChange={(event) => setSkillQuery(event.target.value)}
                />
              </SkillSearchBox>

              {skillsLoading && candidates.length === 0 ? (
                <SkillEmptyState>
                  <Loader2 size={14} />
                  {t("agentExperts.info.skills.loading", "正在加载技能目录...")}
                </SkillEmptyState>
              ) : filteredCandidates.length > 0 ? (
                <SkillCandidateList>
                  {filteredCandidates.map((candidate) => {
                    const added = effectiveSkillRefKeys.has(
                      normalizeExpertSkillRefKey(candidate.ref),
                    );
                    return (
                      <SkillCandidateCard
                        key={candidate.ref}
                        data-testid={`expert-skill-candidate-${sanitizeSkillRefTestId(
                          candidate.ref,
                        )}`}
                      >
                        <SkillCandidateAvatar aria-hidden="true">
                          {skillCandidateInitial(candidate.title)}
                        </SkillCandidateAvatar>
                        <SkillCandidateContent>
                          <SkillCandidateTitle>
                            {candidate.title}
                          </SkillCandidateTitle>
                          <SkillCandidateSummary>
                            {candidate.summary}
                          </SkillCandidateSummary>
                          <SkillCandidateMeta>
                            {resolveCandidateSourceLabel(candidate)} ·{" "}
                            {candidate.ref}
                          </SkillCandidateMeta>
                        </SkillCandidateContent>
                        <CandidateAddButton
                          type="button"
                          $added={!replacingSkill && added}
                          disabled={!replacingSkill && added}
                          data-testid={`expert-skill-add-${sanitizeSkillRefTestId(
                            candidate.ref,
                          )}`}
                          onClick={() => handleAddSkill(candidate.ref)}
                        >
                          {!replacingSkill && added ? (
                            <Check size={13} />
                          ) : (
                            <Plus size={13} />
                          )}
                          {replacingSkill
                            ? t("agentExperts.info.skills.replace", "替换")
                            : added
                              ? t("agentExperts.info.skills.added", "已添加")
                              : t("agentExperts.info.skills.add", "添加")}
                        </CandidateAddButton>
                      </SkillCandidateCard>
                    );
                  })}
                </SkillCandidateList>
              ) : (
                <SkillEmptyState>
                  {t(
                    "agentExperts.info.skills.noCandidates",
                    "没有找到可添加的技能。",
                  )}
                </SkillEmptyState>
              )}
            </SkillDialogBody>
          </SkillDialog>
        </SkillDialogBackdrop>
      ) : null}
    </Section>
  );
}
