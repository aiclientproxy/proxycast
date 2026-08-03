import {
  areArtifactProtocolPathsEquivalent,
  resolveArtifactProtocolFilePath,
} from "@/lib/artifact-protocol";
import { isArticleArtifactFrameArtifact } from "./articleArtifactProjection";
import { resolveKnowledgeSourceFromArtifacts } from "./messageListKnowledgeSource";
import { isHiddenConversationArtifactPath } from "../utils/internalArtifactVisibility";
import { resolveLatestProjectFileSavedSiteContentTargetFromMessage } from "../utils/latestSavedSiteContentTarget";
import {
  resolveSiteSavedContentTargetDisplayName,
  resolveSiteSavedContentTargetRelativePath,
} from "../utils/siteToolResultSummary";
import type { Message } from "../types";

interface ResolveMessageListItemArtifactProjectionParams {
  actionContent: string;
  alreadyRenderedArtifactPaths: string[];
  canOpenSavedSiteContent: boolean;
  hasTrailingArtifactTimelineItems: boolean;
  isImportedHistoryTurn: boolean;
  message: Message;
  shouldSuppressImageProcessFlow: boolean;
}

export function resolveMessageListItemArtifactProjection({
  actionContent,
  alreadyRenderedArtifactPaths,
  canOpenSavedSiteContent,
  hasTrailingArtifactTimelineItems,
  isImportedHistoryTurn,
  message,
  shouldSuppressImageProcessFlow,
}: ResolveMessageListItemArtifactProjectionParams) {
  const knowledgeArtifactSource =
    message.role === "assistant"
      ? resolveKnowledgeSourceFromArtifacts(message.artifacts)
      : null;
  const knowledgeSaveContent =
    knowledgeArtifactSource?.content.trim() || actionContent;
  const canSaveMessageAsKnowledge = Boolean(
    message.role === "assistant" &&
    !message.imageWorkbenchPreview &&
    !message.isThinking &&
    knowledgeSaveContent &&
    knowledgeSaveContent.length >= 24,
  );
  const messageSavedSiteContentTarget =
    message.role === "assistant"
      ? resolveLatestProjectFileSavedSiteContentTargetFromMessage(message)
      : null;
  const visibleAssistantArtifacts =
    message.role === "assistant" &&
    !shouldSuppressImageProcessFlow &&
    !isImportedHistoryTurn
      ? (message.artifacts || []).filter((artifact) => {
          const artifactPath = resolveArtifactProtocolFilePath(artifact);
          if (isHiddenConversationArtifactPath(artifactPath)) {
            return false;
          }

          return !alreadyRenderedArtifactPaths.some((renderedPath) =>
            areArtifactProtocolPathsEquivalent(artifactPath, renderedPath),
          );
        })
      : [];
  const hasArticleArtifactFrame = visibleAssistantArtifacts.some(
    isArticleArtifactFrameArtifact,
  );
  const shouldRenderMessageCanvasShortcut = Boolean(
    messageSavedSiteContentTarget &&
    canOpenSavedSiteContent &&
    !message.imageWorkbenchPreview &&
    !hasTrailingArtifactTimelineItems,
  );
  const messageCanvasShortcutTitle = messageSavedSiteContentTarget
    ? resolveSiteSavedContentTargetDisplayName(messageSavedSiteContentTarget) ||
      "导出稿"
    : "文件";
  const messageCanvasShortcutPath = messageSavedSiteContentTarget
    ? resolveSiteSavedContentTargetRelativePath(messageSavedSiteContentTarget)
    : null;

  return {
    canSaveMessageAsKnowledge,
    hasArticleArtifactFrame,
    knowledgeArtifactSource,
    messageCanvasShortcutPath,
    messageCanvasShortcutTitle,
    messageSavedSiteContentTarget,
    shouldRenderMessageCanvasShortcut,
    visibleAssistantArtifacts,
  };
}
