import type { ModelReasoningEffortLevel } from "@/lib/types/modelRegistry";

export type InputbarComposerSectionCopyKey =
  | "agentChat.inputbar.composer.advancedSettings.label"
  | "agentChat.inputbar.composer.advancedSettings.expand"
  | "agentChat.inputbar.composer.advancedSettings.collapse"
  | "agentChat.inputbar.composer.currentModel.label"
  | "agentChat.inputbar.composer.currentModel.title"
  | "agentChat.inputbar.composer.fileManager.open"
  | "agentChat.inputbar.composer.fileManager.close"
  | "agentChat.inputbar.planStatus.label"
  | "agentChat.inputbar.planStatus.model"
  | "agentChat.inputbar.planStatus.modelFallback"
  | "agentChat.inputbar.planStatus.reasoning"
  | "agentChat.inputbar.planStatus.reasoning.default"
  | "agentChat.inputbar.planStatus.reasoning.none"
  | "agentChat.inputbar.planStatus.reasoning.minimal"
  | "agentChat.inputbar.planStatus.reasoning.low"
  | "agentChat.inputbar.planStatus.reasoning.medium"
  | "agentChat.inputbar.planStatus.reasoning.high"
  | "agentChat.inputbar.planStatus.reasoning.xhigh"
  | "agentChat.inputbar.planStatus.reasoning.max"
  | "agentChat.inputbar.planStatus.reasoning.ultra"
  | "agentChat.inputbar.plusMenu.open"
  | "agentChat.inputbar.plusMenu.addFiles"
  | "agentChat.inputbar.plusMenu.attachKnowledge"
  | "agentChat.inputbar.plusMenu.planMode"
  | "agentChat.inputbar.tools.subagent.label"
  | "agentChat.inputbar.plusMenu.objective"
  | "agentChat.inputbar.plusMenu.plugins"
  | "agentChat.inputbar.plusMenu.skills"
  | "agentChat.inputbar.plusMenu.unavailable"
  | "agentChat.inputbar.pluginChip.empty"
  | "agentChat.inputbar.pluginChip.error"
  | "agentChat.inputbar.pluginChip.loading"
  | "agentChat.inputbar.pluginChip.remove"
  | "agentChat.inputbar.pluginChip.skillPrefix"
  | "agentChat.inputbar.pluginChip.selectorTitle"
  | "agentChat.inputbar.pluginChip.unavailable"
  | "agentChat.inputbar.composer.workspacePlaceholder.waiting"
  | "agentChat.inputbar.composer.workspacePlaceholder.taskCenter"
  | "agentChat.inputbar.composer.workspacePlaceholder.default";

type InputbarComposerSectionCopyValue = number | string;

export type InputbarComposerSectionCopyTranslate = (
  key: InputbarComposerSectionCopyKey,
  values?: Record<string, InputbarComposerSectionCopyValue>,
) => string;

export interface InputbarComposerSectionCopy {
  advancedSettings: {
    label: string;
    expand: string;
    collapse: string;
  };
  currentModel: {
    label: string;
    title: (model: string) => string;
  };
  fileManager: {
    open: string;
    close: string;
  };
  planStatus: {
    label: string;
    model: (model: string) => string;
    modelFallback: string;
    reasoning: (level: string) => string;
    reasoningDefault: string;
    reasoningLevels: Record<ModelReasoningEffortLevel, string>;
  };
  plusMenu: {
    open: string;
    addFiles: string;
    attachKnowledge: string;
    planMode: string;
    subagent: string;
    objective: string;
    plugins: string;
    skills: string;
    unavailable: string;
  };
  pluginChip: {
    empty: string;
    error: string;
    loading: string;
    remove: (name: string) => string;
    skillPrefix: string;
    selectorTitle: string;
    unavailable: string;
  };
  workspacePlaceholder: {
    waiting: string;
    taskCenter: string;
    default: string;
  };
}

export function buildInputbarComposerSectionCopy(
  translate: InputbarComposerSectionCopyTranslate,
): InputbarComposerSectionCopy {
  return {
    advancedSettings: {
      label: translate("agentChat.inputbar.composer.advancedSettings.label"),
      expand: translate("agentChat.inputbar.composer.advancedSettings.expand"),
      collapse: translate(
        "agentChat.inputbar.composer.advancedSettings.collapse",
      ),
    },
    currentModel: {
      label: translate("agentChat.inputbar.composer.currentModel.label"),
      title: (model) =>
        translate("agentChat.inputbar.composer.currentModel.title", { model }),
    },
    fileManager: {
      open: translate("agentChat.inputbar.composer.fileManager.open"),
      close: translate("agentChat.inputbar.composer.fileManager.close"),
    },
    planStatus: {
      label: translate("agentChat.inputbar.planStatus.label"),
      model: (model) =>
        translate("agentChat.inputbar.planStatus.model", { model }),
      modelFallback: translate("agentChat.inputbar.planStatus.modelFallback"),
      reasoning: (level) =>
        translate("agentChat.inputbar.planStatus.reasoning", { level }),
      reasoningDefault: translate(
        "agentChat.inputbar.planStatus.reasoning.default",
      ),
      reasoningLevels: {
        none: translate("agentChat.inputbar.planStatus.reasoning.none"),
        minimal: translate("agentChat.inputbar.planStatus.reasoning.minimal"),
        low: translate("agentChat.inputbar.planStatus.reasoning.low"),
        medium: translate("agentChat.inputbar.planStatus.reasoning.medium"),
        high: translate("agentChat.inputbar.planStatus.reasoning.high"),
        xhigh: translate("agentChat.inputbar.planStatus.reasoning.xhigh"),
        max: translate("agentChat.inputbar.planStatus.reasoning.max"),
        ultra: translate("agentChat.inputbar.planStatus.reasoning.ultra"),
      },
    },
    plusMenu: {
      open: translate("agentChat.inputbar.plusMenu.open"),
      addFiles: translate("agentChat.inputbar.plusMenu.addFiles"),
      attachKnowledge: translate("agentChat.inputbar.plusMenu.attachKnowledge"),
      planMode: translate("agentChat.inputbar.plusMenu.planMode"),
      subagent: translate("agentChat.inputbar.tools.subagent.label"),
      objective: translate("agentChat.inputbar.plusMenu.objective"),
      plugins: translate("agentChat.inputbar.plusMenu.plugins"),
      skills: translate("agentChat.inputbar.plusMenu.skills"),
      unavailable: translate("agentChat.inputbar.plusMenu.unavailable"),
    },
    pluginChip: {
      empty: translate("agentChat.inputbar.pluginChip.empty"),
      error: translate("agentChat.inputbar.pluginChip.error"),
      loading: translate("agentChat.inputbar.pluginChip.loading"),
      remove: (name) =>
        translate("agentChat.inputbar.pluginChip.remove", { name }),
      skillPrefix: translate("agentChat.inputbar.pluginChip.skillPrefix"),
      selectorTitle: translate("agentChat.inputbar.pluginChip.selectorTitle"),
      unavailable: translate("agentChat.inputbar.pluginChip.unavailable"),
    },
    workspacePlaceholder: {
      waiting: translate(
        "agentChat.inputbar.composer.workspacePlaceholder.waiting",
      ),
      taskCenter: translate(
        "agentChat.inputbar.composer.workspacePlaceholder.taskCenter",
      ),
      default: translate(
        "agentChat.inputbar.composer.workspacePlaceholder.default",
      ),
    },
  };
}
