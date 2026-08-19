import { useCallback } from "react";
import { toast } from "sonner";
import type { Artifact } from "@/lib/artifact/types";
import { requestWorkspaceRightSurface } from "@/lib/api/workspaceRightSurface";

interface UseWorkspaceBrowserAssistRequestRuntimeParams {
  projectId?: string | null;
  sessionId?: string | null;
}

export function useWorkspaceBrowserAssistRequestRuntime({
  projectId,
  sessionId,
}: UseWorkspaceBrowserAssistRequestRuntimeParams) {
  const handleOpenBrowserRuntimeForBrowserAssist = useCallback(
    async (artifact?: Artifact) => {
      const launchUrl = readArtifactString(artifact, ["url", "launchUrl"]);
      const title = artifact?.title?.trim() || null;
      try {
        await requestWorkspaceRightSurface({
          surfaceKind: "browser",
          origin: "user",
          priority: "foreground",
          reason: "open_browser_workspace",
          sessionId: sessionId ?? null,
          workspaceId: projectId ?? null,
          candidateId: artifact?.id ?? launchUrl ?? null,
          metadata: {
            browser: {
              launchUrl,
              title,
            },
          },
        });
      } catch (error) {
        toast.error(
          error instanceof Error
            ? error.message
            : "无法打开浏览器工作区，请稍后重试。",
        );
      }
    },
    [projectId, sessionId],
  );

  return {
    handleOpenBrowserRuntimeForBrowserAssist,
  };
}

function readArtifactString(
  artifact: Artifact | undefined,
  keys: readonly string[],
): string | null {
  for (const key of keys) {
    const value = artifact?.meta?.[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return null;
}
