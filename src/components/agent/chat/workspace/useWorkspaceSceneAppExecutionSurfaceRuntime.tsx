import { useCallback, useMemo, type ReactNode } from "react";
import { toast } from "sonner";
import type { SessionFile } from "@/lib/api/session-files";
import type { Artifact } from "@/lib/artifact/types";
import { resolveArtifactProtocolFilePath } from "@/lib/artifact-protocol";
import type {
  SceneAppExecutionSummaryViewModel,
} from "@/lib/agent/legacySceneAppExecutionSummary";
import { SCENEAPP_QUICK_REVIEW_ACTIONS } from "@/lib/agent/legacySceneAppExecutionSummary";
import type { Page, PageParams } from "@/types/page";
import type { TaskFile } from "../components/TaskFiles";
import { RuntimeReviewDecisionDialog } from "../components/RuntimeReviewDecisionDialog";
import type { GeneralWorkbenchFollowUpActionPayload } from "../components/generalWorkbenchSidebarContract";
import type { CuratedTaskReferenceEntry } from "../utils/curatedTaskReferenceSelection";
import {
  buildCuratedTaskReferenceEntryFromSceneAppExecution,
  buildDefaultCuratedTaskReferenceSelection,
  buildSceneAppExecutionCuratedTaskFollowUpAction,
  buildSceneAppExecutionReviewFollowUpAction,
} from "../utils/sceneAppCuratedTaskReference";
import {
  buildSceneAppExecutionContentPostEntries,
  type SceneAppExecutionContentPostEntry,
} from "./sceneAppExecutionContentPosts";
import { SceneAppExecutionSummaryCard } from "./SceneAppExecutionSummaryCard";
import { useSceneAppExecutionSummaryRuntime } from "./useSceneAppExecutionSummaryRuntime";
import { useSceneAppReviewDecisionRuntime } from "./useSceneAppReviewDecisionRuntime";
import { doesWorkspaceFileCandidateMatch } from "./workspaceFilePathMatch";

interface UseWorkspaceSceneAppExecutionSurfaceRuntimeParams {
  artifacts: Artifact[];
  initialSummary?: SceneAppExecutionSummaryViewModel | null;
  isSending: boolean;
  onApplyFollowUpAction: (payload: GeneralWorkbenchFollowUpActionPayload) => void;
  onNavigate?: (page: Page, params?: PageParams) => void;
  onOpenArtifact: (artifact: Artifact) => void;
  onOpenTaskFile: (file: TaskFile) => void;
  onOpenWorkspaceFile: (fileName: string, content: string) => void;
  projectId?: string | null;
  readSessionFile: (fileName: string) => Promise<string | null>;
  replayReferenceEntries?: CuratedTaskReferenceEntry[];
  replayReferenceMemoryIds?: string[];
  sessionFiles: SessionFile[];
  sessionId?: string | null;
  taskFiles: TaskFile[];
}

interface WorkspaceSceneAppExecutionSurfaceRuntime {
  defaultCuratedTaskReferenceEntries: CuratedTaskReferenceEntry[];
  defaultCuratedTaskReferenceMemoryIds: string[];
  reviewDecisionDialogNode: ReactNode;
  summaryCard: ReactNode;
}

export function useWorkspaceSceneAppExecutionSurfaceRuntime({
  artifacts,
  initialSummary,
  isSending,
  onApplyFollowUpAction,
  onNavigate,
  onOpenArtifact,
  onOpenTaskFile,
  onOpenWorkspaceFile,
  projectId,
  readSessionFile,
  replayReferenceEntries,
  replayReferenceMemoryIds,
  sessionFiles,
  sessionId,
  taskFiles,
}: UseWorkspaceSceneAppExecutionSurfaceRuntimeParams): WorkspaceSceneAppExecutionSurfaceRuntime {
  const sceneAppExecutionSummaryState = useSceneAppExecutionSummaryRuntime({
    initialSummary,
    sessionId,
    isSending,
  });
  const sceneAppSummaryAvailable = Boolean(
    sceneAppExecutionSummaryState?.summary,
  );
  const sceneAppExecutionReferenceEntry = useMemo(
    () =>
      sceneAppSummaryAvailable
        ? buildCuratedTaskReferenceEntryFromSceneAppExecution({
            summary: sceneAppExecutionSummaryState?.summary,
          })
        : null,
    [sceneAppExecutionSummaryState?.summary, sceneAppSummaryAvailable],
  );
  const defaultCuratedTaskReferenceSelection = useMemo(
    () =>
      buildDefaultCuratedTaskReferenceSelection({
        replayReferenceEntries,
        replayReferenceMemoryIds,
        sceneAppReferenceEntry: sceneAppExecutionReferenceEntry,
      }),
    [
      replayReferenceEntries,
      replayReferenceMemoryIds,
      sceneAppExecutionReferenceEntry,
    ],
  );
  const defaultCuratedTaskReferenceEntries =
    defaultCuratedTaskReferenceSelection.referenceEntries;
  const defaultCuratedTaskReferenceMemoryIds =
    defaultCuratedTaskReferenceSelection.referenceMemoryIds;
  const handleReviewCurrentSceneAppExecution = useCallback(() => {
    const followUpAction = buildSceneAppExecutionReviewFollowUpAction({
      referenceEntries: defaultCuratedTaskReferenceEntries,
    });
    if (!followUpAction) {
      toast.error("当前还没有足够的项目结果基线，暂时无法直接进入下一步判断。");
      return;
    }

    onApplyFollowUpAction(followUpAction);
  }, [defaultCuratedTaskReferenceEntries, onApplyFollowUpAction]);
  const handleContinueSceneAppReviewFeedback = useCallback(
    (taskId: string) => {
      const followUpAction = buildSceneAppExecutionCuratedTaskFollowUpAction({
        referenceEntries: defaultCuratedTaskReferenceEntries,
        taskId,
      });
      if (!followUpAction) {
        toast.error("当前判断建议还缺少可继续的结果模板。");
        return;
      }

      onApplyFollowUpAction(followUpAction);
    },
    [defaultCuratedTaskReferenceEntries, onApplyFollowUpAction],
  );
  const sceneAppExecutionContentPostEntries = useMemo(
    () =>
      sceneAppSummaryAvailable
        ? buildSceneAppExecutionContentPostEntries({
            taskFiles,
            sessionFiles,
            artifacts,
          })
        : [],
    [artifacts, sceneAppSummaryAvailable, sessionFiles, taskFiles],
  );
  const sceneAppReviewDecisionRuntime = useSceneAppReviewDecisionRuntime({
    enabled: sceneAppSummaryAvailable,
    projectId,
    sessionId,
    sceneAppExecutionSummaryState,
    onNavigate,
  });
  const handleOpenSceneAppExecutionContentPost = useCallback(
    (entry: SceneAppExecutionContentPostEntry) => {
      if (entry.source.kind === "task_file") {
        onOpenTaskFile(entry.source.file);
        return;
      }

      if (entry.source.kind === "artifact") {
        onOpenArtifact(entry.source.artifact);
        return;
      }

      void (async () => {
        try {
          const matchedTaskFile = taskFiles.find((file) =>
            doesWorkspaceFileCandidateMatch(file.name, entry.pathLabel),
          );
          if (matchedTaskFile) {
            onOpenTaskFile(matchedTaskFile);
            return;
          }

          const matchedArtifact = artifacts.find((artifact) =>
            doesWorkspaceFileCandidateMatch(
              resolveArtifactProtocolFilePath(artifact),
              entry.pathLabel,
            ),
          );
          if (matchedArtifact) {
            onOpenArtifact(matchedArtifact);
            return;
          }

          const matchedSessionFile = sessionFiles.find((file) =>
            doesWorkspaceFileCandidateMatch(file.name, entry.pathLabel),
          );
          if (!matchedSessionFile) {
            toast.error("当前发布产物已不存在，暂时无法打开。");
            return;
          }

          const content = await readSessionFile(matchedSessionFile.name);
          if (typeof content !== "string" || !content.trim()) {
            toast.info("该发布产物当前没有可直接预览的正文内容。");
            return;
          }

          onOpenWorkspaceFile(matchedSessionFile.name, content);
        } catch (error) {
          console.error("[AgentChatPage] 打开发布产物失败:", error);
          toast.error("打开发布产物失败，请稍后重试。");
        }
      })();
    },
    [
      artifacts,
      onOpenArtifact,
      onOpenTaskFile,
      onOpenWorkspaceFile,
      readSessionFile,
      sessionFiles,
      taskFiles,
    ],
  );
  const summaryCard = useMemo(
    () =>
      sceneAppExecutionSummaryState?.summary ? (
        <SceneAppExecutionSummaryCard
          summary={sceneAppExecutionSummaryState.summary}
          latestReviewFeedbackSignal={
            sceneAppReviewDecisionRuntime.latestReviewFeedbackSignal
          }
          onContinueReviewFeedback={handleContinueSceneAppReviewFeedback}
          onReviewCurrentProject={handleReviewCurrentSceneAppExecution}
          onSaveAsSkill={sceneAppReviewDecisionRuntime.handleSaveAsSkill}
          humanReviewAvailable={
            sceneAppReviewDecisionRuntime.humanReviewAvailable
          }
          humanReviewLoading={sceneAppReviewDecisionRuntime.loading}
          quickReviewActions={SCENEAPP_QUICK_REVIEW_ACTIONS}
          quickReviewPending={sceneAppReviewDecisionRuntime.quickReviewPending}
          onOpenHumanReview={
            sceneAppReviewDecisionRuntime.handleOpenHumanReview
          }
          onApplyQuickReview={
            sceneAppReviewDecisionRuntime.handleApplyQuickReview
          }
          contentPostEntries={sceneAppExecutionContentPostEntries}
          onContentPostAction={handleOpenSceneAppExecutionContentPost}
        />
      ) : null,
    [
      handleContinueSceneAppReviewFeedback,
      handleOpenSceneAppExecutionContentPost,
      handleReviewCurrentSceneAppExecution,
      sceneAppExecutionContentPostEntries,
      sceneAppExecutionSummaryState,
      sceneAppReviewDecisionRuntime,
    ],
  );
  const reviewDecisionDialogNode = (
    <RuntimeReviewDecisionDialog
      open={sceneAppReviewDecisionRuntime.dialogOpen}
      template={sceneAppReviewDecisionRuntime.template}
      saving={sceneAppReviewDecisionRuntime.saving}
      onOpenChange={sceneAppReviewDecisionRuntime.setDialogOpen}
      onSave={sceneAppReviewDecisionRuntime.handleSaveHumanReview}
    />
  );

  return {
    defaultCuratedTaskReferenceEntries,
    defaultCuratedTaskReferenceMemoryIds,
    reviewDecisionDialogNode,
    summaryCard,
  };
}
