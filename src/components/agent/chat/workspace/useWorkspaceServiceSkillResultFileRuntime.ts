import { useCallback, useMemo } from "react";
import { toast } from "sonner";
import type { AgentThreadItem } from "../types";
import type { TaskFile } from "../components/TaskFiles";
import {
  hasPreferredServiceSkillResultFileTargetSignals,
  resolvePreferredServiceSkillResultFileTarget,
} from "./serviceSkillResultFileTarget";
import { doesWorkspaceFileCandidateMatch } from "./workspaceFilePathMatch";
import { resolveAbsoluteWorkspacePath } from "./workspacePath";

interface UseWorkspaceServiceSkillResultFileRuntimeParams {
  currentTurnId?: string | null;
  effectiveThreadItems: AgentThreadItem[];
  handleWorkspaceFileClick: (fileName: string, content: string) => void;
  openProjectFilePreviewInCanvas: (params: {
    relativePath?: string | null;
    absolutePath: string;
    isCancelled?: () => boolean;
  }) => Promise<boolean>;
  projectRootPath?: string | null;
  taskFiles: TaskFile[];
}

export function useWorkspaceServiceSkillResultFileRuntime({
  currentTurnId,
  effectiveThreadItems,
  handleWorkspaceFileClick,
  openProjectFilePreviewInCanvas,
  projectRootPath,
  taskFiles,
}: UseWorkspaceServiceSkillResultFileRuntimeParams) {
  const currentTurnThreadItems = useMemo(
    () =>
      currentTurnId &&
      hasPreferredServiceSkillResultFileTargetSignals({
        currentTurnId,
        threadItems: effectiveThreadItems,
      })
        ? effectiveThreadItems.filter((item) => item.turn_id === currentTurnId)
        : [],
    [currentTurnId, effectiveThreadItems],
  );

  const preferredServiceSkillResultFileTarget = useMemo(
    () =>
      currentTurnThreadItems.length > 0
        ? resolvePreferredServiceSkillResultFileTarget({
            threadItems: currentTurnThreadItems,
          })
        : null,
    [currentTurnThreadItems],
  );

  const handleOpenServiceSkillResultFile = useCallback(
    async (relativePath: string) => {
      const normalizedPath = relativePath.trim();
      if (!normalizedPath) {
        return;
      }

      const absolutePath = resolveAbsoluteWorkspacePath(
        projectRootPath,
        normalizedPath,
      );
      if (absolutePath) {
        const opened = await openProjectFilePreviewInCanvas({
          relativePath: normalizedPath,
          absolutePath,
        });
        if (opened) {
          return;
        }
      }

      const matchedTaskFile = taskFiles.find((file) =>
        doesWorkspaceFileCandidateMatch(file.name, normalizedPath),
      );
      if (matchedTaskFile) {
        handleWorkspaceFileClick(
          matchedTaskFile.name,
          matchedTaskFile.content ?? "",
        );
        return;
      }

      if (absolutePath) {
        return;
      }

      toast.error("打开结果文件失败：当前工作区里还没有同步到这份文件");
    },
    [
      handleWorkspaceFileClick,
      openProjectFilePreviewInCanvas,
      projectRootPath,
      taskFiles,
    ],
  );

  return {
    handleOpenServiceSkillResultFile,
    preferredServiceSkillResultFileTarget,
  };
}
