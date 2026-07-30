import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { readThreadSessionId } from "@/lib/api/agentRuntime/threadClient";
import type { CanonicalChildThreadSummary } from "../projection/canonicalChildThreadSummary";

interface UseWorkspaceSubagentNavigationRuntimeParams {
  canonicalChildren: CanonicalChildThreadSummary[];
  deferSessionRecentMetadataSyncForNavigation: (sessionId: string) => void;
  switchTopic: (
    sessionId: string,
    options?: { allowDetachedSession?: boolean },
  ) => Promise<unknown> | void;
}

interface OpenWorkspaceSubagentTargetParams extends UseWorkspaceSubagentNavigationRuntimeParams {
  readSessionId?: (threadId: string) => Promise<string>;
  targetId: string;
}

export async function openWorkspaceSubagentTarget({
  canonicalChildren,
  deferSessionRecentMetadataSyncForNavigation,
  readSessionId = readThreadSessionId,
  switchTopic,
  targetId,
}: OpenWorkspaceSubagentTargetParams): Promise<void> {
  const normalizedTargetId = targetId.trim();
  if (!normalizedTargetId) {
    return;
  }
  const canonicalSessionId = canonicalChildren
    .find((child) => child.threadId.trim() === normalizedTargetId)
    ?.sessionId?.trim();
  const sessionId =
    canonicalSessionId || (await readSessionId(normalizedTargetId));
  deferSessionRecentMetadataSyncForNavigation(sessionId);
  await switchTopic(sessionId, { allowDetachedSession: true });
}

export function useWorkspaceSubagentNavigationRuntime({
  canonicalChildren,
  deferSessionRecentMetadataSyncForNavigation,
  switchTopic,
}: UseWorkspaceSubagentNavigationRuntimeParams) {
  const { t } = useTranslation("agent");
  const handleOpenSubagentSession = useCallback(
    (targetId: string) => {
      void openWorkspaceSubagentTarget({
        canonicalChildren,
        deferSessionRecentMetadataSyncForNavigation,
        switchTopic,
        targetId,
      }).catch(() => {
        toast.error(t("agentChat.threadTimeline.alert.failed"));
      });
    },
    [
      canonicalChildren,
      deferSessionRecentMetadataSyncForNavigation,
      switchTopic,
      t,
    ],
  );

  return { handleOpenSubagentSession };
}
