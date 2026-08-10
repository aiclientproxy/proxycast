/**
 * 角色与技能引用组件
 *
 * 在输入框中检测 @ 或 / 符号，显示角色、技能与命令列表供选择
 */

import React, {
  Suspense,
  useEffect,
  useState,
  useMemo,
  useRef,
  useCallback,
} from "react";
import { createPortal } from "react-dom";
import { useTranslation } from "react-i18next";
import type { Character } from "@/lib/api/projectMemory";
import type { Skill } from "@/lib/api/skills";
import {
  listServiceSkills,
  type ServiceSkillItem,
} from "@/lib/api/serviceSkills";
import { filterMentionableServiceSkills } from "@/components/agent/chat/service-skills/entryAdapter";
import {
  getServiceSkillActionLabel,
  getServiceSkillRunnerDescription,
  getServiceSkillRunnerLabel,
  getServiceSkillRunnerTone,
} from "@/components/agent/chat/service-skills/skillPresentation";
import { matchesRuntimeSceneCommandToServiceSkill } from "@/components/agent/chat/service-skills/runtimeSceneBinding";
import type {
  ServiceSkillGroup,
  ServiceSkillHomeItem,
} from "@/components/agent/chat/service-skills/types";
import { toast } from "sonner";
import { filterSlashCommands, type SlashCommandDefinition } from "../commands";
import {
  filterBuiltinCommands,
  filterRuntimeSceneSlashCommands,
  type BuiltinInputCommand,
  type RuntimeSceneSlashCommand,
} from "./builtinCommands";
import type {
  InputCapabilityActivationOptions,
  SelectInputCapabilityHandler,
} from "./inputCapabilitySelection";
import {
  LazyCharacterMentionPanel,
  preloadCharacterMentionPanel,
} from "./characterMentionPanelLoader";
import {
  isCompleteInputbarPluginTriggerQuery,
  type InputbarPluginSelectionOptions,
  type InputbarPluginCapability,
  type InputbarPluginSkillCapability,
} from "../components/Inputbar/pluginInputCapability";
import type { InputCapabilityDescriptor } from "./inputCapabilitySections";
import {
  recordSlashEntryUsage,
  subscribeSlashEntryUsageChanged,
} from "./slashEntryUsage";
import { subscribeMentionEntryUsageChanged } from "./mentionEntryUsage";
import { partitionMentionableSkills } from "./skillQuery";
import { useRuntimeInputCapabilityCatalog } from "./runtimeInputCapabilityCatalog";
import { useIdleModulePreload } from "./useIdleModulePreload";
import {
  buildCuratedTaskLaunchPrompt,
  recordCuratedTaskTemplateUsage,
  subscribeCuratedTaskTemplateUsageChanged,
  type CuratedTaskInputValues,
  type CuratedTaskTemplateItem,
} from "../utils/curatedTaskTemplates";
import { subscribeCuratedTaskRecommendationSignalsChanged } from "../utils/curatedTaskRecommendationSignals";
import { CuratedTaskLauncherDialog } from "../components/CuratedTaskLauncherDialog";
import {
  extractCuratedTaskReferenceMemoryIds,
  mergeCuratedTaskReferenceEntries,
  normalizeCuratedTaskReferenceMemoryIds,
} from "../utils/curatedTaskReferenceSelection";
import type {
  CuratedTaskReferenceEntry,
  CuratedTaskReferenceSelection,
} from "../utils/curatedTaskReferenceSelection";
import type { ProjectFileSearchResult } from "@/lib/api/fuzzyFileSearch";
import { replaceProjectFileMentionToken } from "./projectFileMention";
import { useProjectFileMentionSearch } from "./useProjectFileMentionSearch";

interface CharacterMentionProps {
  /** 角色列表 */
  characters: Character[];
  /** 技能列表 */
  skills?: Skill[];
  /** 技能目录项列表 */
  serviceSkills?: ServiceSkillHomeItem[];
  /** 技能目录分组 */
  serviceSkillGroups?: ServiceSkillGroup[];
  /** 输入框 ref */
  inputRef: React.RefObject<HTMLTextAreaElement>;
  /** 当前输入值 */
  value: string;
  /** 输入值变更回调 */
  onChange: (value: string) => void;
  /** 选择角色回调 */
  onSelectCharacter?: (character: Character) => void;
  /** 统一选择输入能力回调 */
  onSelectInputCapability?: SelectInputCapabilityHandler;
  /** 选择结果模板回调 */
  onSelectCuratedTask?: (
    task: CuratedTaskTemplateItem,
    options?: {
      launchInputValues?: CuratedTaskInputValues;
      referenceMemoryIds?: string[];
      referenceEntries?: CuratedTaskReferenceEntry[];
      launcherPrefillHint?: string;
    },
  ) => void;
  /** 选择 Plugin 插件回调 */
  onSelectPlugin?: (
    plugin: InputbarPluginCapability,
    skill?: InputbarPluginSkillCapability,
    options?: InputbarPluginSelectionOptions,
  ) => void;
  /** Plugin 插件候选 */
  pluginSuggestions?: readonly InputbarPluginCapability[];
  /** 需要加载技能候选时触发 */
  onSkillSuggestionsNeeded?: () => void;
  /** 需要加载 Plugin 插件候选时触发 */
  onPluginSuggestionsNeeded?: () => void;
  projectId?: string | null;
  sessionId?: string | null;
  /** 当前默认带入的灵感引用 */
  defaultCuratedTaskReferenceMemoryIds?: string[];
  /** 当前默认带入的灵感引用对象 */
  defaultCuratedTaskReferenceEntries?: CuratedTaskReferenceEntry[];
  /** 跳转到设置页安装技能 */
  onNavigateToSettings?: () => void;
  /** 是否启用输入自动补全与联想面板 */
  inputCompletionEnabled?: boolean;
}

type TriggerMode = "mention" | "slash";

interface ActiveTrigger {
  mode: TriggerMode;
  triggerIndex: number;
  query: string;
}

interface PendingCuratedTaskLaunch {
  task: CuratedTaskTemplateItem;
  leadingText: string;
  trailingText: string;
  restoreCursorPos: number;
  initialInputValues?: CuratedTaskInputValues;
  initialReferenceMemoryIds?: string[];
  initialReferenceEntries?: CuratedTaskReferenceEntry[];
  launcherPrefillHint?: string;
}

function resolveMentionTrigger(textBeforeCursor: string): ActiveTrigger | null {
  const lastAtIndex = textBeforeCursor.lastIndexOf("@");
  if (lastAtIndex === -1) {
    return null;
  }

  const textAfterAt = textBeforeCursor.slice(lastAtIndex + 1);
  if (textAfterAt.includes(" ") || textAfterAt.includes("\n")) {
    return null;
  }

  return {
    mode: "mention",
    triggerIndex: lastAtIndex,
    query: textAfterAt,
  };
}

function resolveSlashTrigger(textBeforeCursor: string): ActiveTrigger | null {
  const slashMatch = textBeforeCursor.match(/(?:^|[\s\n])(\/[^\s\n/]*)$/);
  if (!slashMatch) {
    return null;
  }

  const slashToken = slashMatch[1];
  const triggerIndex = textBeforeCursor.length - slashToken.length;
  return {
    mode: "slash",
    triggerIndex,
    query: slashToken.slice(1),
  };
}

function resolveActiveTrigger(
  value: string,
  cursorPos: number,
): ActiveTrigger | null {
  const textBeforeCursor = value.slice(0, cursorPos);
  const mentionTrigger = resolveMentionTrigger(textBeforeCursor);
  const slashTrigger = resolveSlashTrigger(textBeforeCursor);

  if (!mentionTrigger) {
    return slashTrigger;
  }
  if (!slashTrigger) {
    return mentionTrigger;
  }

  return mentionTrigger.triggerIndex > slashTrigger.triggerIndex
    ? mentionTrigger
    : slashTrigger;
}

function normalizeReplayText(value: string | undefined): string | null {
  const trimmed = value?.trim();
  return trimmed ? trimmed : null;
}

function mergeTriggerSelectionText(params: {
  leadingText: string;
  insertedText: string;
  trailingText: string;
}): { value: string; cursorPos: number } {
  const normalizedInsertedText = params.insertedText.trim();
  if (!normalizedInsertedText) {
    return {
      value: `${params.leadingText}${params.trailingText}`,
      cursorPos: params.leadingText.length,
    };
  }

  const needsLeadingSpace =
    params.leadingText.length > 0 && !/[\s\n]$/.test(params.leadingText);
  const needsTrailingSpace =
    params.trailingText.length > 0 && !/^[\s\n]/.test(params.trailingText);
  const mergedInsertedText = `${needsLeadingSpace ? " " : ""}${normalizedInsertedText}${
    needsTrailingSpace ? " " : ""
  }`;

  return {
    value: `${params.leadingText}${mergedInsertedText}${params.trailingText}`,
    cursorPos: params.leadingText.length + mergedInsertedText.length,
  };
}

function toServiceSkillHomeItem(skill: ServiceSkillItem): ServiceSkillHomeItem {
  return {
    ...skill,
    badge: "",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: getServiceSkillRunnerLabel(skill),
    runnerTone: getServiceSkillRunnerTone(skill),
    runnerDescription: getServiceSkillRunnerDescription(skill),
    actionLabel: getServiceSkillActionLabel(skill),
    automationStatus: null,
  };
}

export function CharacterMention({
  characters,
  skills = [],
  serviceSkills = [],
  serviceSkillGroups = [],
  inputRef,
  value,
  onChange,
  onSelectCharacter,
  onSelectInputCapability,
  onSelectCuratedTask,
  onSelectPlugin,
  pluginSuggestions = [],
  onSkillSuggestionsNeeded,
  onPluginSuggestionsNeeded,
  projectId,
  sessionId,
  defaultCuratedTaskReferenceMemoryIds = [],
  defaultCuratedTaskReferenceEntries = [],
  onNavigateToSettings,
  inputCompletionEnabled = true,
}: CharacterMentionProps) {
  const { t } = useTranslation("agent");
  const [showMentions, setShowMentions] = useState(false);
  const [mentionQuery, setMentionQuery] = useState("");
  const [triggerMode, setTriggerMode] = useState<TriggerMode>("mention");
  const [pendingCuratedTaskLaunch, setPendingCuratedTaskLaunch] =
    useState<PendingCuratedTaskLaunch | null>(null);
  const [panelAnchor, setPanelAnchor] = useState({
    top: 0,
    left: 0,
    width: 320,
    bottom: 0,
    maxHeight: 320,
  });
  const commandRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const activeTriggerRef = useRef<ActiveTrigger | null>(null);
  const [
    curatedTaskRecommendationSignalsVersion,
    setCuratedTaskRecommendationSignalsVersion,
  ] = useState(0);
  const [curatedTaskTemplatesVersion, setCuratedTaskTemplatesVersion] =
    useState(0);
  const [mentionEntryUsageVersion, setMentionEntryUsageVersion] = useState(0);
  const [slashEntryUsageVersion, setSlashEntryUsageVersion] = useState(0);
  const {
    builtinCommands: runtimeBuiltinCommands,
    sceneCommands: runtimeSceneCommands,
  } = useRuntimeInputCapabilityCatalog();

  useIdleModulePreload(() => {
    void preloadCharacterMentionPanel();
  });

  useEffect(() => {
    if (showMentions) {
      onSkillSuggestionsNeeded?.();
      onPluginSuggestionsNeeded?.();
    }
  }, [onPluginSuggestionsNeeded, onSkillSuggestionsNeeded, showMentions]);

  const filteredBuiltinCommands = useMemo(
    () => filterBuiltinCommands(mentionQuery, runtimeBuiltinCommands),
    [mentionQuery, runtimeBuiltinCommands],
  );
  const filteredServiceSkills = useMemo(
    () => filterMentionableServiceSkills(serviceSkills, mentionQuery),
    [mentionQuery, serviceSkills],
  );
  const filteredSlashCommands = useMemo(
    () => filterSlashCommands(mentionQuery),
    [mentionQuery],
  );
  const filteredRuntimeSceneCommands = useMemo(
    () => filterRuntimeSceneSlashCommands(mentionQuery, runtimeSceneCommands),
    [mentionQuery, runtimeSceneCommands],
  );

  const filteredCharacters = useMemo(() => {
    if (!mentionQuery) return characters;
    const query = mentionQuery.toLowerCase();
    return characters.filter(
      (char) =>
        char.name.toLowerCase().includes(query) ||
        char.description?.toLowerCase().includes(query),
    );
  }, [characters, mentionQuery]);

  const { installedSkills, availableSkills } = useMemo(
    () => partitionMentionableSkills(skills, mentionQuery),
    [skills, mentionQuery],
  );
  const projectFileSearch = useProjectFileMentionSearch({
    active: showMentions && triggerMode === "mention",
    projectId,
    query: mentionQuery,
  });

  const updateMentionState = useCallback(() => {
    if (!inputCompletionEnabled) {
      setMentionQuery("");
      setShowMentions(false);
      return;
    }

    const textarea = inputRef.current;
    if (!textarea) {
      setShowMentions(false);
      return;
    }

    const cursorPos = textarea.selectionStart ?? textarea.value.length;
    const activeTrigger = resolveActiveTrigger(textarea.value, cursorPos);
    if (!activeTrigger) {
      activeTriggerRef.current = null;
      setShowMentions(false);
      return;
    }

    if (
      activeTrigger.mode === "mention" &&
      isCompleteInputbarPluginTriggerQuery({
        query: activeTrigger.query,
        plugins: pluginSuggestions,
      })
    ) {
      activeTriggerRef.current = null;
      setMentionQuery("");
      setShowMentions(false);
      return;
    }

    activeTriggerRef.current = activeTrigger;
    setMentionQuery(activeTrigger.query);
    setTriggerMode(activeTrigger.mode);
    setShowMentions(true);

    const rect = textarea.getBoundingClientRect();
    const viewportWidth =
      window.innerWidth || document.documentElement.clientWidth || 1280;
    const viewportHeight =
      window.innerHeight || document.documentElement.clientHeight || 800;
    const screenPadding = viewportWidth < 640 ? 12 : 16;
    const panelGap = viewportWidth < 640 ? 6 : 8;
    const panelWidth = Math.min(
      rect.width,
      Math.max(viewportWidth - screenPadding * 2, 240),
    );
    const left = Math.min(
      Math.max(rect.left, screenPadding),
      viewportWidth - panelWidth - screenPadding,
    );
    setPanelAnchor({
      top: rect.top,
      left,
      width: panelWidth,
      bottom: Math.max(viewportHeight - rect.top + panelGap, panelGap),
      maxHeight: Math.max(
        Math.min(rect.top - panelGap - screenPadding, 420),
        120,
      ),
    });
  }, [inputCompletionEnabled, inputRef, pluginSuggestions]);

  useEffect(() => {
    if (!inputCompletionEnabled) {
      setShowMentions(false);
      return;
    }

    const textarea = inputRef.current;
    if (!textarea) return;
    textarea.addEventListener("input", updateMentionState);
    textarea.addEventListener("click", updateMentionState);
    textarea.addEventListener("keyup", updateMentionState);

    return () => {
      textarea.removeEventListener("input", updateMentionState);
      textarea.removeEventListener("click", updateMentionState);
      textarea.removeEventListener("keyup", updateMentionState);
    };
  }, [inputCompletionEnabled, inputRef, updateMentionState]);

  useEffect(() => {
    updateMentionState();
  }, [updateMentionState, value]);

  useEffect(() => {
    if (!showMentions) {
      return;
    }

    const handleReposition = () => {
      updateMentionState();
    };

    window.addEventListener("resize", handleReposition);
    window.addEventListener("scroll", handleReposition, true);
    return () => {
      window.removeEventListener("resize", handleReposition);
      window.removeEventListener("scroll", handleReposition, true);
    };
  }, [showMentions, updateMentionState]);

  useEffect(() => {
    if (!showMentions) {
      return;
    }

    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) {
        return;
      }
      if (
        panelRef.current?.contains(target) ||
        inputRef.current?.contains(target)
      ) {
        return;
      }
      setShowMentions(false);
    };

    document.addEventListener("mousedown", handlePointerDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
    };
  }, [inputRef, showMentions]);

  useEffect(() => {
    return subscribeCuratedTaskTemplateUsageChanged(() => {
      setCuratedTaskTemplatesVersion((previous) => previous + 1);
    });
  }, []);

  useEffect(() => {
    return subscribeCuratedTaskRecommendationSignalsChanged(() => {
      setCuratedTaskRecommendationSignalsVersion((previous) => previous + 1);
    });
  }, []);

  useEffect(() => {
    return subscribeMentionEntryUsageChanged(() => {
      setMentionEntryUsageVersion((previous) => previous + 1);
    });
  }, []);

  useEffect(() => {
    return subscribeSlashEntryUsageChanged(() => {
      setSlashEntryUsageVersion((previous) => previous + 1);
    });
  }, []);

  const handleSelectCharacter = (character: Character) => {
    const textarea = inputRef.current;
    if (!textarea) return;

    const currentValue = textarea.value || value;
    const cursorPos = textarea.selectionStart ?? currentValue.length;
    const textAfterCursor = currentValue.slice(cursorPos);
    const activeTrigger = resolveActiveTrigger(currentValue, cursorPos);
    if (!activeTrigger || activeTrigger.mode !== "mention") {
      return;
    }

    const newValue =
      currentValue.slice(0, activeTrigger.triggerIndex) +
      `@${character.name} ` +
      textAfterCursor;

    onChange(newValue);
    setShowMentions(false);
    onSelectCharacter?.(character);

    setTimeout(() => {
      textarea.focus();
      const newCursorPos =
        activeTrigger.triggerIndex + character.name.length + 2;
      textarea.setSelectionRange(newCursorPos, newCursorPos);
    }, 0);
  };

  const handleSelectProjectFile = (file: ProjectFileSearchResult) => {
    const textarea = inputRef.current;
    if (!textarea) return;

    const currentValue = textarea.value || value;
    const cursorPos = textarea.selectionStart ?? currentValue.length;
    const activeTrigger =
      activeTriggerRef.current?.mode === "mention"
        ? activeTriggerRef.current
        : resolveActiveTrigger(currentValue, cursorPos);
    if (!activeTrigger || activeTrigger.mode !== "mention") {
      return;
    }

    const selection = replaceProjectFileMentionToken({
      value: currentValue,
      tokenStart: activeTrigger.triggerIndex,
      tokenEnd: activeTrigger.triggerIndex + activeTrigger.query.length + 1,
      path: file.path,
    });
    onChange(selection.value);
    setShowMentions(false);

    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(selection.cursorPos, selection.cursorPos);
    }, 0);
  };

  const handleSelectInstalledSkill = (
    skill: Skill,
    options?: InputCapabilityActivationOptions,
  ) => {
    const textarea = inputRef.current;
    if (!textarea) return;

    const currentValue = textarea.value || value;
    const cursorPos = textarea.selectionStart ?? currentValue.length;
    const textAfterCursor = currentValue.slice(cursorPos);
    const activeTrigger = resolveActiveTrigger(currentValue, cursorPos);
    if (!activeTrigger) {
      return;
    }

    if (activeTrigger.mode === "slash") {
      const replayText = normalizeReplayText(options?.replayText);
      if (onSelectInputCapability) {
        const leadingText = currentValue.slice(0, activeTrigger.triggerIndex);
        const nextSelection = replayText
          ? mergeTriggerSelectionText({
              leadingText,
              insertedText: replayText,
              trailingText: textAfterCursor,
            })
          : {
              value: `${leadingText}${textAfterCursor}`,
              cursorPos: Math.max(0, activeTrigger.triggerIndex),
            };
        const normalizedValue =
          nextSelection.value.trimEnd() === "" ? "" : nextSelection.value;

        onChange(normalizedValue);
        setShowMentions(false);
        onSelectInputCapability(
          {
            kind: "installed_skill",
            skill,
          },
          replayText ? { replayText } : undefined,
        );

        setTimeout(() => {
          textarea.focus();
          const newCursorPos = Math.min(
            nextSelection.cursorPos,
            normalizedValue.length,
          );
          textarea.setSelectionRange(newCursorPos, newCursorPos);
        }, 0);
        return;
      }

      const nextSelection = replayText
        ? mergeTriggerSelectionText({
            leadingText: currentValue.slice(0, activeTrigger.triggerIndex),
            insertedText: `/${skill.key} ${replayText}`,
            trailingText: textAfterCursor,
          })
        : {
            value:
              currentValue.slice(0, activeTrigger.triggerIndex) +
              `/${skill.key} ` +
              textAfterCursor,
            cursorPos: activeTrigger.triggerIndex + skill.key.length + 2,
          };
      onChange(nextSelection.value);
      setShowMentions(false);
      recordSlashEntryUsage({
        kind: "skill",
        entryId: skill.key,
        replayText: replayText || undefined,
      });

      setTimeout(() => {
        textarea.focus();
        const newCursorPos = nextSelection.cursorPos;
        textarea.setSelectionRange(newCursorPos, newCursorPos);
      }, 0);
      return;
    }

    if (onSelectInputCapability) {
      const newValue =
        currentValue.slice(0, activeTrigger.triggerIndex) + textAfterCursor;
      onChange(newValue.trimEnd() === "" ? "" : newValue);
      setShowMentions(false);
      onSelectInputCapability({
        kind: "installed_skill",
        skill,
      });

      setTimeout(() => {
        textarea.focus();
        const newCursorPos = Math.max(0, activeTrigger.triggerIndex);
        textarea.setSelectionRange(newCursorPos, newCursorPos);
      }, 0);
      return;
    }

    const newValue =
      currentValue.slice(0, activeTrigger.triggerIndex) +
      `/${skill.key} ` +
      textAfterCursor;
    onChange(newValue);
    setShowMentions(false);

    setTimeout(() => {
      textarea.focus();
      const newCursorPos = activeTrigger.triggerIndex + skill.key.length + 2;
      textarea.setSelectionRange(newCursorPos, newCursorPos);
    }, 0);
  };

  const handleSelectAvailableSkill = (skill: Skill) => {
    setShowMentions(false);

    toast.info(`技能「${skill.name}」尚未安装`, {
      action: onNavigateToSettings
        ? {
            label: "去技能中心",
            onClick: onNavigateToSettings,
          }
        : undefined,
    });
  };

  const handleSelectBuiltinCommand = (
    command: BuiltinInputCommand,
    options?: InputCapabilityActivationOptions,
  ) => {
    const textarea = inputRef.current;
    if (!textarea) return;

    const currentValue = textarea.value || value;
    const cursorPos = textarea.selectionStart ?? currentValue.length;
    const textAfterCursor = currentValue.slice(cursorPos);
    const activeTrigger = resolveActiveTrigger(currentValue, cursorPos);
    if (!activeTrigger || activeTrigger.mode !== "mention") {
      return;
    }

    const leadingText = currentValue.slice(0, activeTrigger.triggerIndex);
    const replayText = normalizeReplayText(options?.replayText);

    if (onSelectInputCapability) {
      const nextSelection = replayText
        ? mergeTriggerSelectionText({
            leadingText,
            insertedText: replayText,
            trailingText: textAfterCursor,
          })
        : {
            value: `${leadingText}${textAfterCursor}`,
            cursorPos: Math.max(0, activeTrigger.triggerIndex),
          };
      const normalizedValue =
        nextSelection.value.trimEnd() === "" ? "" : nextSelection.value;

      onChange(normalizedValue);
      setShowMentions(false);
      onSelectInputCapability(
        {
          kind: "builtin_command",
          command,
        },
        replayText ? { replayText } : undefined,
      );

      setTimeout(() => {
        textarea.focus();
        const newCursorPos = Math.min(
          nextSelection.cursorPos,
          normalizedValue.length,
        );
        textarea.setSelectionRange(newCursorPos, newCursorPos);
      }, 0);
      return;
    }

    const nextSelection = replayText
      ? mergeTriggerSelectionText({
          leadingText,
          insertedText: `${command.commandPrefix} ${replayText}`,
          trailingText: textAfterCursor,
        })
      : {
          value: `${leadingText}${command.commandPrefix} ${textAfterCursor}`,
          cursorPos:
            activeTrigger.triggerIndex + command.commandPrefix.length + 1,
        };

    onChange(nextSelection.value);
    setShowMentions(false);

    setTimeout(() => {
      textarea.focus();
      const newCursorPos = nextSelection.cursorPos;
      textarea.setSelectionRange(newCursorPos, newCursorPos);
    }, 0);
  };

  const handleSelectServiceSkill = (skill: ServiceSkillHomeItem) => {
    const textarea = inputRef.current;
    if (!textarea) return;

    const currentValue = textarea.value || value;
    const cursorPos = textarea.selectionStart ?? currentValue.length;
    const textAfterCursor = currentValue.slice(cursorPos);
    const activeTrigger = resolveActiveTrigger(currentValue, cursorPos);
    if (!activeTrigger || activeTrigger.mode !== "mention") {
      return;
    }

    if (onSelectInputCapability) {
      const newValue =
        currentValue.slice(0, activeTrigger.triggerIndex) + textAfterCursor;
      onChange(newValue.trimEnd() === "" ? "" : newValue);
      setShowMentions(false);
      onSelectInputCapability({
        kind: "service_skill",
        skill,
      });

      setTimeout(() => {
        textarea.focus();
        const newCursorPos = Math.max(0, activeTrigger.triggerIndex);
        textarea.setSelectionRange(newCursorPos, newCursorPos);
      }, 0);
      return;
    }

    const newValue =
      currentValue.slice(0, activeTrigger.triggerIndex) +
      `@${skill.title} ` +
      textAfterCursor;
    onChange(newValue);
    setShowMentions(false);

    setTimeout(() => {
      textarea.focus();
      const newCursorPos = activeTrigger.triggerIndex + skill.title.length + 2;
      textarea.setSelectionRange(newCursorPos, newCursorPos);
    }, 0);
  };

  const handleSelectPlugin = (
    plugin: InputbarPluginCapability,
    skill?: InputbarPluginSkillCapability,
  ) => {
    const textarea = inputRef.current;
    if (!textarea) return;

    const currentValue = textarea.value || value;
    const fallbackCursorPos = textarea.selectionStart ?? currentValue.length;
    const fallbackTrigger = resolveActiveTrigger(
      currentValue,
      fallbackCursorPos,
    );
    const activeTrigger =
      activeTriggerRef.current?.mode === "mention"
        ? activeTriggerRef.current
        : fallbackTrigger;
    if (!activeTrigger || activeTrigger.mode !== "mention") {
      return;
    }

    const mentionEnd =
      activeTrigger.triggerIndex + activeTrigger.query.length + 1;
    const textAfterCursor = currentValue.slice(mentionEnd);
    const nextSelection = mergeTriggerSelectionText({
      leadingText: currentValue.slice(0, activeTrigger.triggerIndex),
      insertedText: "",
      trailingText: textAfterCursor,
    });
    const normalizedValue =
      nextSelection.value.trimEnd() === "" ? "" : nextSelection.value;

    onChange(normalizedValue);
    setShowMentions(false);
    onSelectPlugin?.(plugin, skill, {
      inputOverride: normalizedValue,
      preserveInputOverride: true,
    });

    setTimeout(() => {
      textarea.focus();
      const newCursorPos = Math.min(
        nextSelection.cursorPos,
        normalizedValue.length,
      );
      textarea.setSelectionRange(newCursorPos, newCursorPos);
    }, 0);
  };

  const handleSelectSlashCommand = (
    command: SlashCommandDefinition,
    options?: { replayText?: string },
  ) => {
    const textarea = inputRef.current;
    if (!textarea) return;

    const currentValue = textarea.value || value;
    const cursorPos = textarea.selectionStart ?? currentValue.length;
    const textAfterCursor = currentValue.slice(cursorPos);
    const activeTrigger = resolveActiveTrigger(currentValue, cursorPos);
    if (!activeTrigger || activeTrigger.mode !== "slash") {
      return;
    }

    const replayText = normalizeReplayText(options?.replayText);
    const nextSelection = replayText
      ? mergeTriggerSelectionText({
          leadingText: currentValue.slice(0, activeTrigger.triggerIndex),
          insertedText: `${command.commandPrefix} ${replayText}`,
          trailingText: textAfterCursor,
        })
      : {
          value:
            currentValue.slice(0, activeTrigger.triggerIndex) +
            `${command.commandPrefix} ` +
            textAfterCursor,
          cursorPos:
            activeTrigger.triggerIndex + command.commandPrefix.length + 1,
        };

    onChange(nextSelection.value);
    setShowMentions(false);
    if (command.support === "supported") {
      recordSlashEntryUsage({
        kind: "command",
        entryId: command.key,
        replayText: replayText || undefined,
      });
    }

    setTimeout(() => {
      textarea.focus();
      const newCursorPos = nextSelection.cursorPos;
      textarea.setSelectionRange(newCursorPos, newCursorPos);
    }, 0);
  };

  const handleSelectRuntimeSceneCommand = async (
    command: RuntimeSceneSlashCommand,
    options?: { replayText?: string },
  ) => {
    const replayText = normalizeReplayText(options?.replayText);

    const matchedVisibleSkill = serviceSkills.find((skill) =>
      matchesRuntimeSceneCommandToServiceSkill(skill, command),
    );
    const shouldResolveServiceSkill =
      !replayText &&
      onSelectInputCapability &&
      (matchedVisibleSkill || command.linkedSkillId);

    if (shouldResolveServiceSkill) {
      const matchedSkill =
        matchedVisibleSkill ||
        (await (async () => {
          try {
            const runtimeSkills = await listServiceSkills();
            const runtimeMatchedSkill = runtimeSkills.find((skill) =>
              matchesRuntimeSceneCommandToServiceSkill(skill, command),
            );
            return runtimeMatchedSkill
              ? toServiceSkillHomeItem(runtimeMatchedSkill)
              : null;
          } catch {
            return null;
          }
        })());

      if (
        matchedSkill &&
        matchedSkill.slotSchema.some((slot) => slot.required)
      ) {
        onSelectInputCapability({
          kind: "service_skill",
          skill: matchedSkill,
        });
        setShowMentions(false);
        recordSlashEntryUsage({
          kind: "scene",
          entryId: command.key,
        });
        return;
      }
    }

    const textarea = inputRef.current;
    if (!textarea) return;

    const currentValue = textarea.value || value;
    const cursorPos = textarea.selectionStart ?? currentValue.length;
    const textAfterCursor = currentValue.slice(cursorPos);
    const activeTrigger = resolveActiveTrigger(currentValue, cursorPos);
    if (!activeTrigger || activeTrigger.mode !== "slash") {
      return;
    }

    if (onSelectInputCapability) {
      const leadingText = currentValue.slice(0, activeTrigger.triggerIndex);
      const nextSelection = replayText
        ? mergeTriggerSelectionText({
            leadingText,
            insertedText: replayText,
            trailingText: textAfterCursor,
          })
        : {
            value: `${leadingText}${textAfterCursor}`,
            cursorPos: Math.max(0, activeTrigger.triggerIndex),
          };
      const normalizedValue =
        nextSelection.value.trimEnd() === "" ? "" : nextSelection.value;

      onChange(normalizedValue);
      setShowMentions(false);
      onSelectInputCapability(
        {
          kind: "runtime_scene",
          command,
        },
        replayText ? { replayText } : undefined,
      );

      setTimeout(() => {
        textarea.focus();
        const newCursorPos = Math.min(
          nextSelection.cursorPos,
          normalizedValue.length,
        );
        textarea.setSelectionRange(newCursorPos, newCursorPos);
      }, 0);
      return;
    }

    const nextSelection = replayText
      ? mergeTriggerSelectionText({
          leadingText: currentValue.slice(0, activeTrigger.triggerIndex),
          insertedText: `${command.commandPrefix} ${replayText}`,
          trailingText: textAfterCursor,
        })
      : {
          value:
            currentValue.slice(0, activeTrigger.triggerIndex) +
            `${command.commandPrefix} ` +
            textAfterCursor,
          cursorPos:
            activeTrigger.triggerIndex + command.commandPrefix.length + 1,
        };

    onChange(nextSelection.value);
    setShowMentions(false);
    recordSlashEntryUsage({
      kind: "scene",
      entryId: command.key,
      replayText: replayText || undefined,
    });

    setTimeout(() => {
      textarea.focus();
      const newCursorPos = nextSelection.cursorPos;
      textarea.setSelectionRange(newCursorPos, newCursorPos);
    }, 0);
  };

  const handleSelectCuratedTask = (
    task: CuratedTaskTemplateItem,
    options?: {
      launchInputValues?: CuratedTaskInputValues;
      referenceMemoryIds?: string[];
      referenceEntries?: CuratedTaskReferenceEntry[];
      launcherPrefillHint?: string;
    },
  ) => {
    const textarea = inputRef.current;
    if (!textarea) return;

    const currentValue = textarea.value || value;
    const cursorPos = textarea.selectionStart ?? currentValue.length;
    const textAfterCursor = currentValue.slice(cursorPos);
    const activeTrigger = resolveActiveTrigger(currentValue, cursorPos);
    if (!activeTrigger || activeTrigger.mode !== "slash") {
      return;
    }

    const mergedReferenceEntries = mergeCuratedTaskReferenceEntries([
      ...(options?.referenceEntries ?? []),
      ...defaultCuratedTaskReferenceEntries,
    ]);
    const mergedReferenceMemoryIds =
      normalizeCuratedTaskReferenceMemoryIds([
        ...(options?.referenceMemoryIds ?? []),
        ...(extractCuratedTaskReferenceMemoryIds(mergedReferenceEntries) ?? []),
        ...defaultCuratedTaskReferenceMemoryIds,
      ]) ?? [];

    setShowMentions(false);
    setPendingCuratedTaskLaunch({
      task,
      leadingText: currentValue.slice(0, activeTrigger.triggerIndex),
      trailingText: textAfterCursor,
      restoreCursorPos: cursorPos,
      initialInputValues: options?.launchInputValues,
      initialReferenceMemoryIds: mergedReferenceMemoryIds,
      initialReferenceEntries: mergedReferenceEntries,
      launcherPrefillHint: options?.launcherPrefillHint,
    });
  };

  const handleSelectCapabilityDescriptor = (
    item: InputCapabilityDescriptor,
  ) => {
    switch (item.kind) {
      case "builtin_command":
        handleSelectBuiltinCommand(item.command, {
          replayText: item.replayText,
        });
        return;
      case "service_skill":
        handleSelectServiceSkill(item.skill);
        return;
      case "plugin":
        handleSelectPlugin(item.plugin, item.skill);
        return;
      case "slash_command":
        handleSelectSlashCommand(item.command, {
          replayText: item.replayText,
        });
        return;
      case "scene_command":
        void handleSelectRuntimeSceneCommand(item.command, {
          replayText: item.replayText,
        });
        return;
      case "curated_task":
        handleSelectCuratedTask(item.task, {
          launchInputValues: item.launchInputValues,
          referenceMemoryIds: item.referenceMemoryIds,
          referenceEntries: item.referenceEntries,
          launcherPrefillHint: item.launcherPrefillHint,
        });
        return;
      case "character":
        handleSelectCharacter(item.character);
        return;
      case "installed_skill":
        handleSelectInstalledSkill(item.skill, {
          replayText: item.replayText,
        });
        return;
      case "available_skill":
        handleSelectAvailableSkill(item.skill);
        return;
      default:
        return;
    }
  };

  const handleCuratedTaskLauncherOpenChange = useCallback(
    (open: boolean) => {
      if (open) {
        return;
      }

      const pendingLaunch = pendingCuratedTaskLaunch;
      setPendingCuratedTaskLaunch(null);

      const textarea = inputRef.current;
      if (!textarea || !pendingLaunch) {
        return;
      }

      setTimeout(() => {
        textarea.focus();
        const nextCursorPos = Math.min(
          pendingLaunch.restoreCursorPos,
          textarea.value.length,
        );
        textarea.setSelectionRange(nextCursorPos, nextCursorPos);
      }, 0);
    },
    [inputRef, pendingCuratedTaskLaunch],
  );
  const handleApplyCuratedTaskReviewSuggestion = useCallback(
    (
      task: CuratedTaskTemplateItem,
      options: {
        inputValues: CuratedTaskInputValues;
        referenceSelection: CuratedTaskReferenceSelection;
      },
    ) => {
      setPendingCuratedTaskLaunch((current) => {
        if (!current) {
          return current;
        }

        return {
          ...current,
          task,
          initialInputValues: options.inputValues,
          initialReferenceMemoryIds:
            normalizeCuratedTaskReferenceMemoryIds(
              options.referenceSelection.referenceMemoryIds,
            ) ?? [],
          initialReferenceEntries: mergeCuratedTaskReferenceEntries(
            options.referenceSelection.referenceEntries,
          ),
          launcherPrefillHint:
            "已按最近判断切到更适合的结果模板，你可以继续改后再发。",
        };
      });
    },
    [],
  );

  const handleConfirmCuratedTaskLaunch = useCallback(
    (
      task: CuratedTaskTemplateItem,
      inputValues: CuratedTaskInputValues,
      referenceSelection: CuratedTaskReferenceSelection,
    ) => {
      const pendingLaunch = pendingCuratedTaskLaunch;
      const textarea = inputRef.current;
      if (!pendingLaunch || !textarea) {
        return;
      }

      const launchPrompt = buildCuratedTaskLaunchPrompt({
        task,
        inputValues,
        referenceEntries: referenceSelection.referenceEntries,
      });
      const nextSelection = mergeTriggerSelectionText({
        leadingText: pendingLaunch.leadingText,
        insertedText: launchPrompt,
        trailingText: pendingLaunch.trailingText,
      });
      const launchedTask = {
        ...task,
        prompt: launchPrompt,
      };

      onChange(nextSelection.value);
      recordCuratedTaskTemplateUsage({
        templateId: task.id,
        launchInputValues: inputValues,
        referenceMemoryIds: referenceSelection.referenceMemoryIds,
        referenceEntries: referenceSelection.referenceEntries,
      });
      setPendingCuratedTaskLaunch(null);

      if (onSelectInputCapability) {
        onSelectInputCapability(
          {
            kind: "curated_task",
            task: launchedTask,
            launchInputValues: inputValues,
            referenceMemoryIds: referenceSelection.referenceMemoryIds,
            referenceEntries: referenceSelection.referenceEntries,
          },
          {
            replayText: launchPrompt,
          },
        );
      } else {
        onSelectCuratedTask?.(launchedTask);
      }

      setTimeout(() => {
        textarea.focus();
        textarea.setSelectionRange(
          nextSelection.cursorPos,
          nextSelection.cursorPos,
        );
      }, 0);
    },
    [
      inputRef,
      onChange,
      onSelectCuratedTask,
      onSelectInputCapability,
      pendingCuratedTaskLaunch,
    ],
  );

  useEffect(() => {
    const textarea = inputRef.current;
    if (!textarea || !showMentions) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const composing =
        (e as KeyboardEvent & { isComposing?: boolean }).isComposing ||
        e.key === "Process" ||
        e.keyCode === 229;
      if (composing) {
        return;
      }

      if (e.key === "Escape") {
        setShowMentions(false);
        e.preventDefault();
        return;
      }

      if (e.key === "ArrowUp" || e.key === "ArrowDown" || e.key === "Enter") {
        e.preventDefault();
        e.stopPropagation();
        const cmdkRoot = commandRef.current;
        if (cmdkRoot) {
          cmdkRoot.dispatchEvent(
            new KeyboardEvent("keydown", {
              key: e.key,
              bubbles: true,
              cancelable: true,
            }),
          );
        }
      }
    };

    textarea.addEventListener("keydown", handleKeyDown);
    return () => textarea.removeEventListener("keydown", handleKeyDown);
  }, [showMentions, inputRef]);

  if (!showMentions && !pendingCuratedTaskLaunch) return null;

  return createPortal(
    <>
      {showMentions ? (
        <>
          <div
            data-testid="mention-anchor"
            style={{
              position: "fixed",
              top: panelAnchor.top,
              left: panelAnchor.left,
              width: panelAnchor.width,
              height: 1,
              pointerEvents: "none",
              zIndex: 79,
            }}
          />
          <div
            ref={panelRef}
            data-testid="mention-popover-content"
            data-side="top"
            data-align="start"
            data-avoid-collisions="false"
            className="border bg-background p-0 shadow-md"
            style={{
              position: "fixed",
              left: panelAnchor.left,
              bottom: panelAnchor.bottom,
              width: `${panelAnchor.width}px`,
              maxWidth: "calc(100vw - 24px)",
              maxHeight: `${panelAnchor.maxHeight}px`,
              overflow: "auto",
              zIndex: 80,
            }}
          >
            <Suspense
              fallback={
                <div className="px-3 py-6 text-center text-sm text-muted-foreground">
                  {t("agentChat.characterMention.loading", "加载中...")}
                </div>
              }
            >
              <LazyCharacterMentionPanel
                mode={triggerMode}
                mentionQuery={mentionQuery}
                builtinCommands={filteredBuiltinCommands}
                slashCommands={filteredSlashCommands}
                sceneCommands={filteredRuntimeSceneCommands}
                mentionServiceSkills={filteredServiceSkills}
                pluginSuggestions={pluginSuggestions}
                serviceSkillGroups={serviceSkillGroups}
                filteredCharacters={filteredCharacters}
                installedSkills={installedSkills}
                availableSkills={availableSkills}
                projectFiles={projectFileSearch.files}
                projectFileSearchStatus={projectFileSearch.status}
                projectId={projectId}
                sessionId={sessionId}
                referenceEntries={defaultCuratedTaskReferenceEntries}
                curatedTaskTemplatesVersion={curatedTaskTemplatesVersion}
                curatedTaskRecommendationSignalsVersion={
                  curatedTaskRecommendationSignalsVersion
                }
                mentionEntryUsageVersion={mentionEntryUsageVersion}
                slashEntryUsageVersion={slashEntryUsageVersion}
                commandRef={commandRef}
                onQueryChange={setMentionQuery}
                onSelectCapability={handleSelectCapabilityDescriptor}
                onSelectProjectFile={handleSelectProjectFile}
                onNavigateToSettings={
                  onNavigateToSettings
                    ? () => {
                        setShowMentions(false);
                        onNavigateToSettings();
                      }
                    : undefined
                }
              />
            </Suspense>
          </div>
        </>
      ) : null}
      <CuratedTaskLauncherDialog
        open={Boolean(pendingCuratedTaskLaunch)}
        task={pendingCuratedTaskLaunch?.task ?? null}
        projectId={projectId}
        sessionId={sessionId}
        initialInputValues={pendingCuratedTaskLaunch?.initialInputValues}
        initialReferenceMemoryIds={
          pendingCuratedTaskLaunch?.initialReferenceMemoryIds ??
          normalizeCuratedTaskReferenceMemoryIds([
            ...defaultCuratedTaskReferenceMemoryIds,
            ...(extractCuratedTaskReferenceMemoryIds(
              defaultCuratedTaskReferenceEntries,
            ) ?? []),
          ])
        }
        initialReferenceEntries={
          pendingCuratedTaskLaunch?.initialReferenceEntries ??
          mergeCuratedTaskReferenceEntries(defaultCuratedTaskReferenceEntries)
        }
        prefillHint={pendingCuratedTaskLaunch?.launcherPrefillHint}
        onOpenChange={handleCuratedTaskLauncherOpenChange}
        onApplyReviewSuggestion={handleApplyCuratedTaskReviewSuggestion}
        onConfirm={handleConfirmCuratedTaskLaunch}
      />
    </>,
    document.body,
  );
}
