import { useCallback } from "react";
import type { MessageImage, MessagePathReference } from "../../../types";
import type { InputbarKnowledgePackSelection } from "../types";
import type { InputbarSendHandler } from "../inputbarSendPayload";
import {
  buildInputbarPluginMention,
  resolveInputbarPluginSubmissionText,
  type InputbarPluginSelection,
} from "../pluginInputCapability";
import { buildKnowledgeRequestMetadata } from "@/features/knowledge/agent/knowledgeMetadata";
import { recordCuratedTaskTemplateUsage } from "../../../utils/curatedTaskTemplates";
import { buildPathReferenceRequestMetadata } from "../../../utils/pathReferences";
import {
  resolveInputCapabilityDispatch,
  type InputCapabilitySelection,
} from "../../../skill-selection/inputCapabilitySelection";
import { buildInputbarToolPreferencesOverride } from "../utils/inputbarModeRequestMetadata";
import { recordAgentUiPerformanceMetric } from "@/lib/agentUiPerformanceMetrics";
import type {
  BaseComposerSendMetadata,
  ComposerController,
  ComposerDraftSnapshot,
  ComposerSubmitTarget,
} from "@/components/input-kit";

interface UseInputbarSendParams {
  input: string;
  pendingImages: MessageImage[];
  pathReferences: MessagePathReference[];
  activeCapability: InputCapabilitySelection | null;
  activePluginSelection?: InputbarPluginSelection | null;
  knowledgePackSelection?: InputbarKnowledgePackSelection | null;
  activeTools?: Record<string, boolean>;
  projectId?: string | null;
  sessionId?: string | null;
  isLoading?: boolean;
  onSend: InputbarSendHandler;
  clearPendingImages: () => void;
  clearPathReferences?: () => void;
  clearActiveCapability: () => void;
  getInputRestoreEpoch?: () => number;
  composerController?: ComposerController;
  onComposerCommitted?: (text: string, draft?: ComposerDraftSnapshot) => void;
  submitTarget?: ComposerSubmitTarget;
}

export function useInputbarSend({
  input,
  pendingImages,
  pathReferences,
  activeCapability,
  activePluginSelection = null,
  knowledgePackSelection,
  activeTools = {},
  projectId,
  sessionId,
  isLoading = false,
  onSend,
  clearPendingImages,
  clearPathReferences,
  clearActiveCapability,
  getInputRestoreEpoch,
  composerController,
  onComposerCommitted,
  submitTarget,
}: UseInputbarSendParams) {
  return useCallback(
    async (triggerMetadata?: BaseComposerSendMetadata) => {
      const handlerEnteredAt = Date.now();
      const triggeredAt =
        typeof triggerMetadata?.triggeredAt === "number" &&
        Number.isFinite(triggerMetadata.triggeredAt)
          ? triggerMetadata.triggeredAt
          : handlerEnteredAt;
      const triggerSource = triggerMetadata?.triggerSource ?? "adapter";
      recordAgentUiPerformanceMetric("inputbar.send.enter", {
        durationMs: Math.max(0, handlerEnteredAt - triggeredAt),
        hasTriggerMetadata: Boolean(triggerMetadata),
        imageCount: pendingImages.length,
        inputLength: input.trim().length,
        pathReferenceCount: pathReferences.length,
        sessionId: sessionId ?? null,
        source: "inputbar",
        triggerSource,
        workspaceId: projectId ?? null,
      });
      const sendRestoreEpoch = getInputRestoreEpoch?.() ?? 0;
      const resolvedSubmitTarget: ComposerSubmitTarget =
        triggerMetadata?.submitTarget ??
        submitTarget ??
        (isLoading ? "steer" : "start");
      composerController?.setAttachments(pendingImages);
      composerController?.setPathReferences(pathReferences);
      const composerReceipt = composerController?.submit(resolvedSubmitTarget);
      if (composerReceipt?.kind === "empty") {
        return;
      }
      const composerDraft = composerReceipt?.draft;
      const submittedInput = resolveInputbarPluginSubmissionText({
        input: composerDraft?.text ?? input,
        selection: activePluginSelection,
      });
      const inputMentions = activePluginSelection
        ? [buildInputbarPluginMention(activePluginSelection.plugin)]
        : [];
      if (
        !submittedInput.trim() &&
        pendingImages.length === 0 &&
        pathReferences.length === 0
      ) {
        return;
      }

      const hasRuntimeMode =
        Boolean(activeTools["objective_mode"]) ||
        Boolean(activeTools["task_mode"]) ||
        Boolean(activeTools["subagent_mode"]);
      const canUsePlainTextFastPath =
        submittedInput.trim() &&
        pendingImages.length === 0 &&
        pathReferences.length === 0 &&
        !activeCapability &&
        !activePluginSelection &&
        !knowledgePackSelection?.enabled &&
        !hasRuntimeMode &&
        resolvedSubmitTarget !== "queue";
      if (canUsePlainTextFastPath) {
        recordAgentUiPerformanceMetric("inputbar.send.plainTextFastPath", {
          elapsedMs: Math.max(0, Date.now() - triggeredAt),
          inputLength: submittedInput.trim().length,
          sessionId: sessionId ?? null,
          source: "inputbar",
          triggerSource,
          workspaceId: projectId ?? null,
        });
        const result = await onSend({
          images: undefined,
          textOverride: submittedInput,
          sendOptions: undefined,
          ...(composerReceipt?.kind === "accepted"
            ? {
                composerIntent: composerReceipt.intent,
                composerTarget: composerReceipt.target,
                composerDraft,
              }
            : {}),
          ...(triggerMetadata
            ? {
                triggeredAt,
                triggerSource,
              }
            : {}),
        });
        if (result === false) {
          return;
        }
        if (
          (getInputRestoreEpoch?.() ?? sendRestoreEpoch) !== sendRestoreEpoch
        ) {
          return;
        }
        if (
          composerReceipt?.kind === "accepted" &&
          !composerController?.commit(composerReceipt)
        ) {
          return;
        }
        clearPendingImages();
        clearPathReferences?.();
        clearActiveCapability();
        if (composerReceipt?.kind === "accepted") {
          onComposerCommitted?.(
            composerController?.getDocument().text ?? "",
            composerDraft,
          );
        }
        return;
      }

      const capabilityDispatch = resolveInputCapabilityDispatch(
        activeCapability,
        submittedInput,
      );
      const baseRequestMetadata = buildPathReferenceRequestMetadata(
        capabilityDispatch.requestMetadata,
        pathReferences,
      );
      const knowledgeRequestMetadata =
        knowledgePackSelection?.enabled &&
        knowledgePackSelection.packName.trim() &&
        knowledgePackSelection.workingDir.trim()
          ? {
              ...(baseRequestMetadata || {}),
              ...buildKnowledgeRequestMetadata({
                workingDir: knowledgePackSelection.workingDir.trim(),
                packName: knowledgePackSelection.packName.trim(),
                packs: knowledgePackSelection.companionPacks,
                source: "inputbar",
              }),
            }
          : baseRequestMetadata;
      const inputbarModeState = {
        planEnabled: Boolean(activeTools["task_mode"]),
        subagentEnabled: Boolean(activeTools["subagent_mode"]),
      };
      const requestMetadata = knowledgeRequestMetadata;
      const threadGoal =
        activeTools["objective_mode"] && submittedInput.trim()
          ? { objective: submittedInput.trim() }
          : undefined;
      const toolPreferencesOverride =
        buildInputbarToolPreferencesOverride(inputbarModeState);
      const collaborationMode = inputbarModeState.planEnabled
        ? ("plan" as const)
        : undefined;
      const hasPathReferences = pathReferences.length > 0;
      const textOverride = submittedInput.trim()
        ? submittedInput
        : hasPathReferences
          ? "请查看这些文件或文件夹。"
          : undefined;
      const inputRestoreDraft = {
        text: submittedInput.trim() ? submittedInput : "",
        images: [...pendingImages],
        pathReferences: [...pathReferences],
        textElements: submittedInput.trim()
          ? [{ type: "text", text: submittedInput }]
          : [],
        inputCapabilityRoute: capabilityDispatch.capabilityRoute,
      };
      const sendOptions =
        capabilityDispatch.capabilityRoute ||
        capabilityDispatch.displayContent ||
        requestMetadata ||
        threadGoal ||
        toolPreferencesOverride ||
        collaborationMode ||
        inputMentions.length > 0
          ? {
              ...(capabilityDispatch.capabilityRoute
                ? { capabilityRoute: capabilityDispatch.capabilityRoute }
                : {}),
              inputRestoreDraft,
              ...(inputMentions.length > 0 ? { inputMentions } : {}),
              ...(capabilityDispatch.displayContent || submittedInput.trim()
                ? {
                    displayContent:
                      capabilityDispatch.displayContent ||
                      (submittedInput.trim() ? submittedInput : undefined),
                  }
                : {}),
              ...(requestMetadata ? { requestMetadata } : {}),
              ...(threadGoal ? { threadGoal } : {}),
              ...(toolPreferencesOverride ? { toolPreferencesOverride } : {}),
              ...(collaborationMode ? { collaborationMode } : {}),
            }
          : undefined;

      try {
        const result = await onSend({
          images: pendingImages.length > 0 ? pendingImages : undefined,
          textOverride,
          sendOptions,
          ...(composerReceipt?.kind === "accepted"
            ? {
                composerIntent: composerReceipt.intent,
                composerTarget: composerReceipt.target,
                composerDraft,
              }
            : {}),
          ...(triggerMetadata
            ? {
                triggeredAt,
                triggerSource,
              }
            : {}),
        });
        if (result === false) {
          return;
        }
        if (activeCapability?.kind === "curated_task") {
          recordCuratedTaskTemplateUsage({
            templateId: activeCapability.task.id,
            launchInputValues: activeCapability.launchInputValues,
            referenceMemoryIds: activeCapability.referenceMemoryIds,
            referenceEntries: activeCapability.referenceEntries,
          });
        }
        if (
          (getInputRestoreEpoch?.() ?? sendRestoreEpoch) !== sendRestoreEpoch
        ) {
          return;
        }
        if (
          composerReceipt?.kind === "accepted" &&
          !composerController?.commit(composerReceipt)
        ) {
          return;
        }
        clearPendingImages();
        clearPathReferences?.();
        clearActiveCapability();
        if (composerReceipt?.kind === "accepted") {
          onComposerCommitted?.(
            composerController?.getDocument().text ?? "",
            composerDraft,
          );
        }
      } catch {
        // 发送失败时保留图片与技能，交由上层 toast / 恢复逻辑处理。
      }
    },
    [
      activeCapability,
      activePluginSelection,
      activeTools,
      clearActiveCapability,
      composerController,
      clearPendingImages,
      clearPathReferences,
      getInputRestoreEpoch,
      input,
      isLoading,
      knowledgePackSelection,
      onComposerCommitted,
      projectId,
      submitTarget,
      sessionId,
      onSend,
      pendingImages,
      pathReferences,
    ],
  );
}
