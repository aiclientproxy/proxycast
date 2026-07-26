import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import type { CanvasStateUnion } from "@/components/workspace/canvas/canvasUtils";
import { getMediaTaskArtifact } from "@/lib/api/mediaTasks";
import { writeAgentRuntimeArtifactSnapshot } from "@/lib/api/agentRuntime/appServerArtifactClient";
import type { MediaTaskArtifactOutput } from "@/lib/api/agentRuntime/mediaTaskTypes";
import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import { logAgentDebug } from "@/lib/agentDebug";
import type { Message } from "../types";
import type { SessionImageWorkbenchState } from "./imageWorkbenchHelpers";
import {
  shouldEnableWorkspaceImageTaskPreviewRuntime,
  useWorkspaceImageTaskPreviewRuntime,
} from "./useWorkspaceImageTaskPreviewRuntime";
import { useWorkspaceImageTaskExecutorRuntime } from "./useWorkspaceImageTaskExecutorRuntime";
import {
  attachWorkspaceArticleWorkspacePreviewArtifactToMessages,
  buildWorkspaceArticleWorkspaceFromMessageArtifacts,
  hasWorkspaceArticleWorkspaceMessageArtifactSignals,
} from "./workspaceArticleWorkspaceMessageArtifacts";
import {
  applyWorkspaceArticleEditedDraft,
  buildWorkspaceArticleEditedDraftArtifactWriteParams,
  buildWorkspaceArticleEditedDraftFromChange,
  readWorkspaceArticleObjectMarkdown,
  shouldRejectWorkspaceArticleEditedDraftChange,
  type WorkspaceArticleEditedDraft,
  type WorkspaceArticleMarkdownChange,
} from "./workspaceArticleWorkspaceEditedDraft";
import {
  applyWorkspaceArticleInlineHostCommandSyncResult,
  buildWorkspaceArticleInlineHostCommandSync,
} from "./workspaceArticleInlineHostCommandSync";
import {
  applyWorkspaceArticleInlineImageTaskSyncResult,
  buildWorkspaceArticleInlineImageTaskSync,
  collectWorkspaceArticleInlineImageTaskRecoveryMarkdowns,
  collectWorkspaceArticleInlineImageTaskRecoveryMarkdownsFromMessages,
  selectWorkspaceArticleInlineImageTaskIds,
  suppressWorkspaceArticleInlineImageTaskPreviewMessages,
} from "./workspaceArticleInlineImageTaskSync";
import {
  buildWorkspaceArticleWorkspaceFromThreadRead,
  hasWorkspaceArticleFinalDocument,
  hasWorkspaceArticleWorkspaceThreadReadMetadata,
  type WorkspaceArticleWorkspace,
  type WorkspaceArticleWorkspaceImageSlotIntent,
} from "./workspaceArticleWorkspaceModel";

interface UseWorkspaceArticleEditorRightSurfaceRuntimeParams {
  activeArticleWorkspace: WorkspaceArticleWorkspace | null;
  canvasState: CanvasStateUnion | null;
  canvasWorkbenchRootPath: string | null;
  contentId?: string | null;
  currentImageWorkbenchState: SessionImageWorkbenchState;
  imageWorkbenchSessionKey?: string | null;
  messages: Message[];
  onImageSlotIntent?: (
    intent: WorkspaceArticleWorkspaceImageSlotIntent,
  ) => void | Promise<void>;
  projectId?: string | null;
  runtimeWorkspaceId?: string | null;
  sceneDisplayMessages: Message[];
  sceneIsPreparingSend: boolean;
  sceneIsSending: boolean;
  sceneSessionId?: string | null;
  sceneThreadRead: AgentRuntimeThreadReadModel | null;
  setCanvasState: Dispatch<SetStateAction<CanvasStateUnion | null>>;
  setChatMessages: Dispatch<SetStateAction<Message[]>>;
  shouldDeferWorkspaceAuxiliaryLoads: boolean;
  shouldHideCurrentSessionContent: boolean;
  shouldRestoreImageTasksFromWorkspace: boolean;
  updateCurrentImageWorkbenchState: (
    updater: (
      current: SessionImageWorkbenchState,
    ) => SessionImageWorkbenchState,
  ) => void;
}

interface UseWorkspaceArticleEditorRightSurfaceRuntimeResult {
  articleEditorRightSurface: WorkspaceArticleWorkspace | null;
  articleEditorRightSurfaceAvailable: boolean;
  handleArticleWorkspaceMarkdownChange: (
    change: WorkspaceArticleMarkdownChange,
  ) => void;
  sceneDisplayMessagesWithArticleWorkspaceArtifact: Message[];
}

export function useWorkspaceArticleEditorRightSurfaceRuntime({
  activeArticleWorkspace,
  canvasState,
  canvasWorkbenchRootPath,
  contentId,
  currentImageWorkbenchState,
  imageWorkbenchSessionKey,
  messages,
  onImageSlotIntent,
  projectId,
  runtimeWorkspaceId,
  sceneDisplayMessages,
  sceneIsPreparingSend,
  sceneIsSending,
  sceneSessionId,
  sceneThreadRead,
  setCanvasState,
  setChatMessages,
  shouldDeferWorkspaceAuxiliaryLoads,
  shouldHideCurrentSessionContent,
  shouldRestoreImageTasksFromWorkspace,
  updateCurrentImageWorkbenchState,
}: UseWorkspaceArticleEditorRightSurfaceRuntimeParams): UseWorkspaceArticleEditorRightSurfaceRuntimeResult {
  const [activeArticleEditedDraft, setActiveArticleEditedDraft] =
    useState<WorkspaceArticleEditedDraft | null>(null);
  const inlineHostCommandDispatchSignatureRef = useRef<string | null>(null);
  const onImageSlotIntentRef = useRef(onImageSlotIntent);
  onImageSlotIntentRef.current = onImageSlotIntent;

  useEffect(() => {
    setActiveArticleEditedDraft(null);
    inlineHostCommandDispatchSignatureRef.current = null;
  }, [runtimeWorkspaceId, sceneSessionId]);

  const articleWorkspaceFromThreadRead = useMemo(
    () =>
      hasWorkspaceArticleWorkspaceThreadReadMetadata(sceneThreadRead)
        ? buildWorkspaceArticleWorkspaceFromThreadRead(sceneThreadRead)
        : null,
    [sceneThreadRead],
  );
  const articleWorkspaceFromMessageArtifacts = useMemo(
    () =>
      hasWorkspaceArticleWorkspaceMessageArtifactSignals(sceneDisplayMessages)
        ? buildWorkspaceArticleWorkspaceFromMessageArtifacts(
            sceneDisplayMessages,
          )
        : null,
    [sceneDisplayMessages],
  );
  const sceneThreadId = sceneThreadRead?.thread_id;
  const rawArticleEditorRightSurface =
    articleWorkspaceFromThreadRead ??
    articleWorkspaceFromMessageArtifacts ??
    activeArticleWorkspace;
  const baseArticleEditorRightSurface = useMemo(
    () =>
      applyWorkspaceArticleEditedDraft(
        rawArticleEditorRightSurface,
        activeArticleEditedDraft,
      ),
    [activeArticleEditedDraft, rawArticleEditorRightSurface],
  );
  const articleInlineHostCommandSyncResult = useMemo(
    () =>
      buildWorkspaceArticleInlineHostCommandSync({
        articleWorkspace: baseArticleEditorRightSurface,
        editedDraft: activeArticleEditedDraft,
      }),
    [activeArticleEditedDraft, baseArticleEditorRightSurface],
  );
  const articleInlineHostMaterializedRightSurface = useMemo(
    () =>
      applyWorkspaceArticleInlineHostCommandSyncResult(
        baseArticleEditorRightSurface,
        articleInlineHostCommandSyncResult,
      ),
    [articleInlineHostCommandSyncResult, baseArticleEditorRightSurface],
  );

  useEffect(() => {
    const syncResult = articleInlineHostCommandSyncResult;
    if (!syncResult || !baseArticleEditorRightSurface) {
      return;
    }
    if (sceneIsSending || sceneIsPreparingSend) {
      return;
    }

    const signature = [
      syncResult.object.ref.appId,
      syncResult.object.ref.sessionId,
      syncResult.object.ref.kind,
      syncResult.object.ref.id,
      syncResult.markdown,
    ].join(":");
    if (inlineHostCommandDispatchSignatureRef.current === signature) {
      return;
    }
    inlineHostCommandDispatchSignatureRef.current = signature;

    const change: WorkspaceArticleMarkdownChange = {
      articleWorkspace: baseArticleEditorRightSurface,
      markdown: syncResult.markdown,
      object: syncResult.object,
    };
    const editedDraft = buildWorkspaceArticleEditedDraftFromChange(change);
    if (editedDraft) {
      setActiveArticleEditedDraft((previous) =>
        previous?.objectKey === editedDraft.objectKey &&
        previous.markdown === editedDraft.markdown
          ? previous
          : editedDraft,
      );

      const request = buildWorkspaceArticleEditedDraftArtifactWriteParams(
        change,
        editedDraft,
        sceneThreadId,
      );
      if (request) {
        void writeAgentRuntimeArtifactSnapshot(request).catch((error) => {
          console.warn(
            "[AgentChatWorkspace] Article Editor 配图占位写回失败:",
            error,
          );
        });
      }
    }

    syncResult.imageSlotIntents.forEach((intent) => {
      void onImageSlotIntentRef.current?.({
        ...intent,
        articleWorkspace: baseArticleEditorRightSurface,
      });
    });
  }, [
    articleInlineHostCommandSyncResult,
    baseArticleEditorRightSurface,
    sceneIsPreparingSend,
    sceneIsSending,
    sceneThreadId,
  ]);

  const articleInlineImageTaskSyncResult = useMemo(
    () =>
      buildWorkspaceArticleInlineImageTaskSync({
        articleWorkspace: articleInlineHostMaterializedRightSurface,
        editedDraft: activeArticleEditedDraft,
        imageWorkbenchState: currentImageWorkbenchState,
      }),
    [
      activeArticleEditedDraft,
      articleInlineHostMaterializedRightSurface,
      currentImageWorkbenchState,
    ],
  );

  useEffect(() => {
    const hasInlineRecoverySignal =
      collectWorkspaceArticleInlineImageTaskRecoveryMarkdowns({
        articleWorkspace: articleInlineHostMaterializedRightSurface,
        editedDraft: activeArticleEditedDraft,
      }).length > 0;
    const documentInlineTasks = currentImageWorkbenchState.tasks.filter(
      (task) =>
        task.applyTarget?.kind === "canvas-insert" &&
        task.applyTarget.canvasType === "document",
    );
    if (
      !hasInlineRecoverySignal &&
      documentInlineTasks.length === 0 &&
      !articleInlineImageTaskSyncResult
    ) {
      return;
    }

    logAgentDebug(
      "AgentChatWorkspace",
      "articleInlineImageSync.state",
      {
        consumedTaskIds:
          articleInlineImageTaskSyncResult?.consumedTaskIds ?? [],
        documentInlineSlotIds: documentInlineTasks.map((task) =>
          task.applyTarget?.kind === "canvas-insert"
            ? task.applyTarget.slotId || null
            : null,
        ),
        hasInlineRecoverySignal,
        hasSyncResult: Boolean(articleInlineImageTaskSyncResult),
        outputCount: currentImageWorkbenchState.outputs.length,
        taskCount: currentImageWorkbenchState.tasks.length,
      },
      { level: "debug", throttleMs: 1000 },
    );
  }, [
    activeArticleEditedDraft,
    articleInlineHostMaterializedRightSurface,
    articleInlineImageTaskSyncResult,
    currentImageWorkbenchState,
  ]);

  const articleEditorRightSurface = useMemo(
    () =>
      applyWorkspaceArticleInlineImageTaskSyncResult(
        articleInlineHostMaterializedRightSurface,
        articleInlineImageTaskSyncResult,
      ),
    [
      articleInlineHostMaterializedRightSurface,
      articleInlineImageTaskSyncResult,
    ],
  );
  const articleInlineImageTaskIds = useMemo(
    () =>
      selectWorkspaceArticleInlineImageTaskIds({
        articleWorkspace: articleEditorRightSurface,
        editedDraft: activeArticleEditedDraft,
        imageWorkbenchState: currentImageWorkbenchState,
      }),
    [
      activeArticleEditedDraft,
      articleEditorRightSurface,
      currentImageWorkbenchState,
    ],
  );
  const articleInlineImageTaskRecoveryMarkdowns = useMemo(() => {
    const markdowns = new Set<string>();
    collectWorkspaceArticleInlineImageTaskRecoveryMarkdowns({
      articleWorkspace: articleEditorRightSurface,
      editedDraft: activeArticleEditedDraft,
    }).forEach((markdown) => markdowns.add(markdown));
    collectWorkspaceArticleInlineImageTaskRecoveryMarkdownsFromMessages(
      sceneDisplayMessages,
    ).forEach((markdown) => markdowns.add(markdown));
    return [...markdowns];
  }, [
    activeArticleEditedDraft,
    articleEditorRightSurface,
    sceneDisplayMessages,
  ]);
  const shouldRestoreCurrentImageTasksFromWorkspace =
    shouldRestoreImageTasksFromWorkspace ||
    articleInlineImageTaskRecoveryMarkdowns.length > 0;
  const imageTaskPreviewRuntimeEnabled = useMemo(
    () =>
      shouldEnableWorkspaceImageTaskPreviewRuntime({
        shouldDeferWorkspaceAuxiliaryLoads,
        restoreFromWorkspace: shouldRestoreCurrentImageTasksFromWorkspace,
        messages,
        imageWorkbenchState: currentImageWorkbenchState,
        canvasState,
        documentMarkdowns: articleInlineImageTaskRecoveryMarkdowns,
      }),
    [
      articleInlineImageTaskRecoveryMarkdowns,
      canvasState,
      currentImageWorkbenchState,
      messages,
      shouldDeferWorkspaceAuxiliaryLoads,
      shouldRestoreCurrentImageTasksFromWorkspace,
    ],
  );

  useEffect(() => {
    const hasInlineRecoverySignal =
      articleInlineImageTaskRecoveryMarkdowns.some(
        (markdown) =>
          markdown.includes("pending-image-task://") ||
          markdown.includes("lime:image-task-slot:"),
      );
    if (
      !hasInlineRecoverySignal &&
      !shouldRestoreCurrentImageTasksFromWorkspace &&
      !imageTaskPreviewRuntimeEnabled
    ) {
      return;
    }

    logAgentDebug(
      "AgentChatWorkspace",
      "articleInlineImageRecovery.state",
      {
        canvasWorkbenchRootPath: canvasWorkbenchRootPath || null,
        documentMarkdownCount: articleInlineImageTaskRecoveryMarkdowns.length,
        hasArticleEditorRightSurface: Boolean(articleEditorRightSurface),
        hasArticleWorkspaceFromMessageArtifacts: Boolean(
          articleWorkspaceFromMessageArtifacts,
        ),
        hasArticleWorkspaceFromThreadRead: Boolean(
          articleWorkspaceFromThreadRead,
        ),
        hasInlineRecoverySignal,
        imageTaskPreviewRuntimeEnabled,
        sceneDisplayMessagesCount: sceneDisplayMessages.length,
        shouldDeferWorkspaceAuxiliaryLoads,
        shouldRestoreCurrentImageTasksFromWorkspace,
      },
      { level: "debug", throttleMs: 1000 },
    );
  }, [
    articleEditorRightSurface,
    articleInlineImageTaskRecoveryMarkdowns,
    articleWorkspaceFromMessageArtifacts,
    articleWorkspaceFromThreadRead,
    canvasWorkbenchRootPath,
    imageTaskPreviewRuntimeEnabled,
    sceneDisplayMessages.length,
    shouldDeferWorkspaceAuxiliaryLoads,
    shouldRestoreCurrentImageTasksFromWorkspace,
  ]);

  useWorkspaceImageTaskPreviewRuntime({
    enabled: imageTaskPreviewRuntimeEnabled,
    sessionId: imageWorkbenchSessionKey,
    projectId,
    contentId,
    projectRootPath: canvasWorkbenchRootPath,
    restoreFromWorkspace: shouldRestoreCurrentImageTasksFromWorkspace,
    messages,
    documentMarkdowns: articleInlineImageTaskRecoveryMarkdowns,
    currentImageWorkbenchState,
    canvasState,
    setCanvasState,
    setChatMessages,
    updateCurrentImageWorkbenchState,
  });
  useWorkspaceImageTaskExecutorRuntime({
    enabled: imageTaskPreviewRuntimeEnabled,
    projectRootPath: canvasWorkbenchRootPath,
    currentImageWorkbenchState,
    getImageTask: getMediaTaskArtifact as (request: {
      projectRootPath: string;
      taskRef: string;
    }) => Promise<MediaTaskArtifactOutput>,
  });

  useEffect(() => {
    const syncResult = articleInlineImageTaskSyncResult;
    if (!syncResult || !articleInlineHostMaterializedRightSurface) {
      return;
    }

    const change: WorkspaceArticleMarkdownChange = {
      articleWorkspace: articleInlineHostMaterializedRightSurface,
      markdown: syncResult.markdown,
      object: syncResult.object,
    };
    const editedDraft = buildWorkspaceArticleEditedDraftFromChange(change);
    if (!editedDraft) {
      logAgentDebug(
        "AgentChatWorkspace",
        "articleInlineImageSync.persistSkipped",
        {
          reason: "missing_edited_draft",
          consumedTaskIds: syncResult.consumedTaskIds,
        },
        { level: "warn", throttleMs: 1000 },
      );
      return;
    }

    setActiveArticleEditedDraft((previous) =>
      previous?.objectKey === editedDraft.objectKey &&
      previous.markdown === editedDraft.markdown
        ? previous
        : editedDraft,
    );

    const request = buildWorkspaceArticleEditedDraftArtifactWriteParams(
      change,
      editedDraft,
      sceneThreadId,
    );
    if (!request) {
      logAgentDebug(
        "AgentChatWorkspace",
        "articleInlineImageSync.persistSkipped",
        {
          reason: "missing_artifact_write_request",
          consumedTaskIds: syncResult.consumedTaskIds,
          threadId: sceneThreadId ?? null,
        },
        { level: "warn", throttleMs: 1000 },
      );
      return;
    }
    logAgentDebug(
      "AgentChatWorkspace",
      "articleInlineImageSync.persistStart",
      {
        consumedTaskIds: syncResult.consumedTaskIds,
        markdownIncludesPending: syncResult.markdown.includes(
          "pending-image-task://",
        ),
        markdownIncludesResolvedImage:
          /!\[[^\]]*]\((?!pending-image-task:\/\/)(?:https?:\/\/|file:\/\/|asset:\/\/|data:image\/)/i.test(
            syncResult.markdown,
          ),
        threadId: request.threadId,
      },
      { level: "debug", throttleMs: 1000 },
    );
    void writeAgentRuntimeArtifactSnapshot(request).catch((error) => {
      console.warn(
        "[AgentChatWorkspace] Article Editor 配图回填写回失败:",
        error,
      );
    });
  }, [
    articleInlineImageTaskSyncResult,
    articleInlineHostMaterializedRightSurface,
    sceneThreadId,
  ]);

  const sceneDisplayMessagesWithoutArticleInlineImageTasks = useMemo(
    () =>
      suppressWorkspaceArticleInlineImageTaskPreviewMessages(
        sceneDisplayMessages,
        articleInlineImageTaskIds,
      ),
    [articleInlineImageTaskIds, sceneDisplayMessages],
  );
  const articleEditorRightSurfaceAvailable = hasWorkspaceArticleFinalDocument(
    articleEditorRightSurface,
  );
  const sceneDisplayMessagesWithArticleWorkspaceArtifact = useMemo(
    () =>
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages: sceneDisplayMessagesWithoutArticleInlineImageTasks,
        articleWorkspace: shouldHideCurrentSessionContent
          ? null
          : articleEditorRightSurface,
        status:
          sceneIsSending || sceneIsPreparingSend ? "streaming" : "complete",
      }),
    [
      articleEditorRightSurface,
      sceneDisplayMessagesWithoutArticleInlineImageTasks,
      sceneIsPreparingSend,
      sceneIsSending,
      shouldHideCurrentSessionContent,
    ],
  );

  const handleArticleWorkspaceMarkdownChange = useCallback(
    (change: WorkspaceArticleMarkdownChange) => {
      const editedDraft = buildWorkspaceArticleEditedDraftFromChange(change);
      if (
        shouldRejectWorkspaceArticleEditedDraftChange({
          currentDraft: activeArticleEditedDraft,
          currentMarkdown: readWorkspaceArticleObjectMarkdown(change.object),
          nextDraft: editedDraft,
        })
      ) {
        return;
      }
      setActiveArticleEditedDraft(editedDraft);
      const request = buildWorkspaceArticleEditedDraftArtifactWriteParams(
        change,
        editedDraft,
        sceneThreadId,
      );
      if (!request) {
        return;
      }
      void writeAgentRuntimeArtifactSnapshot(request).catch((error) => {
        console.warn(
          "[AgentChatWorkspace] Article Editor 编辑正文写回失败:",
          error,
        );
      });
    },
    [activeArticleEditedDraft, sceneThreadId],
  );

  return {
    articleEditorRightSurface,
    articleEditorRightSurfaceAvailable,
    handleArticleWorkspaceMarkdownChange,
    sceneDisplayMessagesWithArticleWorkspaceArtifact,
  };
}
