import {
  startTransition,
  useCallback,
  useEffect,
  useRef,
  type Dispatch,
  type SetStateAction,
} from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import {
  createInitialVideoState,
  type CanvasStateUnion,
} from "@/components/workspace/canvas/canvasUtils";
import type { CanvasState as GeneralCanvasState } from "@/components/general-chat/bridge";
import { normalizeArtifactProtocolPath } from "@/lib/artifact-protocol";
import type { SessionFile } from "@/lib/api/session-files";
import type { Artifact } from "@/lib/artifact/types";
import { requestWorkspaceRightSurface } from "@/lib/api/workspaceRightSurface";
import { createPreviewArtifact } from "@/lib/artifact/previewArtifact";
import type { ArtifactDocumentV1 } from "@/lib/artifact-document";
import type { LayoutMode, ThemeType } from "@/lib/workspace/workbenchContract";
import type { AgentChatWorkspaceProps } from "../agentChatWorkspaceContract";
import type { CanvasWorkbenchLayoutMode } from "../components/CanvasWorkbenchLayout";
import type { TaskFile } from "../components/TaskFiles";
import type { WorkspaceWorkbenchRequestsController } from "../hooks/useWorkspaceWorkbenchRequests";
import type { ArtifactTimelineOpenTarget } from "../utils/artifactTimelineNavigation";
import type { SearchResultPreviewItem } from "../utils/searchResultPreview";
import type {
  AgentThreadItem,
  Message,
  MessagePreviewTarget,
  SiteSavedContentTarget,
} from "../types";
import {
  normalizeVideoAspectRatio,
  normalizeVideoResolution,
  resolveTaskPreviewArtifact,
  resolveVideoCanvasStatusFromPreview,
} from "./agentChatWorkspaceHelpers";
import { openCanvasForReason } from "./canvasOpenPolicy";
import { MediaReferencePreviewPaginationActions } from "./mediaReferencePreviewToolbarActions";
import { resolveMediaReferencePreviewPageRequest } from "./mediaReferencePreviewToolbarState";
import type { GeneralArtifactSyncResult } from "./useWorkspaceGeneralResourceSync";
import type { ApplyArtifactViewMode } from "./useWorkspaceArtifactViewModeControl";
import {
  useWorkspaceArtifactDocumentSaveRuntime,
  type WorkspaceArtifactWriteFile,
} from "./useWorkspaceArtifactDocumentSaveRuntime";
import { useWorkspaceArtifactPreviewActions } from "./useWorkspaceArtifactPreviewActions";
import { useWorkspaceArtifactWorkbenchActions } from "./useWorkspaceArtifactWorkbenchActions";
import { useWorkspaceMediaReferencePreviewRuntime } from "./useWorkspaceMediaReferencePreviewRuntime";
import { useWorkspaceRightSurfaceArtifactOpenRuntime } from "./useWorkspaceRightSurfaceArtifactOpenRuntime";
import { useWorkspaceServiceSkillResultFileRuntime } from "./useWorkspaceServiceSkillResultFileRuntime";
import { resolveImageWorkbenchStateForPreviewSelection } from "./imageWorkbenchPreviewSelection";
import type { WorkspaceArticleWorkspace } from "./workspaceArticleWorkspaceModel";
import { buildGeneralCanvasStateFromWorkspaceFile } from "./workspaceFilePreview";
import { doesWorkspaceFileCandidateMatch } from "./workspaceFilePathMatch";
import {
  isAbsoluteWorkspacePath,
  resolveAbsoluteWorkspacePath,
} from "./workspacePath";
import type { WorkspaceImageWorkbenchSessionRuntimeState } from "./useWorkspaceImageWorkbenchSessionRuntime";

type SetBoolean = (value: boolean) => void;

export interface UseWorkspaceArtifactOpenRuntimeParams {
  activeTheme: string;
  artifacts: Artifact[];
  contentId?: string | null;
  currentCanvasArtifact: Artifact | null;
  currentTurnId?: string | null;
  effectiveThreadItems: AgentThreadItem[];
  generalCanvasState: GeneralCanvasState;
  handleWriteFile: WorkspaceArtifactWriteFile;
  initialProjectFileOpenTarget?: AgentChatWorkspaceProps["initialProjectFileOpenTarget"];
  isInitialContentLoading: boolean;
  isThemeWorkbench: boolean;
  layoutMode: LayoutMode;
  mappedTheme: ThemeType;
  messages: Message[];
  onNavigate?: AgentChatWorkspaceProps["onNavigate"];
  openArticleWorkspaceRightSurface: (
    articleWorkspace: WorkspaceArticleWorkspace,
  ) => void;
  projectId?: string | null;
  projectRootPath?: string | null;
  readSessionFile: (fileName: string) => Promise<string | null>;
  sessionFiles: SessionFile[];
  threadId?: string | null;
  setArtifactViewMode: ApplyArtifactViewMode;
  setCanvasState: Dispatch<SetStateAction<CanvasStateUnion | null>>;
  setCanvasWorkbenchLayoutMode: Dispatch<
    SetStateAction<CanvasWorkbenchLayoutMode>
  >;
  setExpertInfoPanelCollapsed: SetBoolean;
  setGeneralCanvasState: Dispatch<SetStateAction<GeneralCanvasState>>;
  setHarnessPanelVisible: SetBoolean;
  setLayoutMode: Dispatch<SetStateAction<LayoutMode>>;
  setSelectedArtifactId: (artifactId: string | null) => void;
  setSelectedFileId: (fileId: string) => void;
  setTaskFiles: Dispatch<SetStateAction<TaskFile[]>>;
  syncGeneralArtifactToResource: (input: {
    rawFilePath: string;
    preferredName?: string;
  }) => Promise<GeneralArtifactSyncResult>;
  taskFiles: TaskFile[];
  updateCurrentImageWorkbenchState: WorkspaceImageWorkbenchSessionRuntimeState["updateCurrentImageWorkbenchState"];
  upsertGeneralArtifact: (artifact: Artifact) => void;
  workbenchRequests: WorkspaceWorkbenchRequestsController;
}

export function useWorkspaceArtifactOpenRuntime({
  activeTheme,
  artifacts,
  contentId,
  currentCanvasArtifact,
  currentTurnId,
  effectiveThreadItems,
  generalCanvasState,
  handleWriteFile,
  initialProjectFileOpenTarget,
  isInitialContentLoading,
  isThemeWorkbench,
  layoutMode,
  mappedTheme,
  messages,
  onNavigate,
  openArticleWorkspaceRightSurface,
  projectId,
  projectRootPath,
  readSessionFile,
  sessionFiles,
  threadId,
  setArtifactViewMode,
  setCanvasState,
  setCanvasWorkbenchLayoutMode,
  setExpertInfoPanelCollapsed,
  setGeneralCanvasState,
  setHarnessPanelVisible,
  setLayoutMode,
  setSelectedArtifactId,
  setSelectedFileId,
  setTaskFiles,
  syncGeneralArtifactToResource,
  taskFiles,
  updateCurrentImageWorkbenchState,
  upsertGeneralArtifact,
  workbenchRequests,
}: UseWorkspaceArtifactOpenRuntimeParams) {
  const { t } = useTranslation("agent");
  const handledInitialProjectFileOpenSignatureRef = useRef("");
  const {
    clearFocusedArtifactBlock,
    focusArtifactBlock,
    requestCanvasWorkbenchPreviewOpen,
  } = workbenchRequests;

  const handleSaveArtifactDocument = useWorkspaceArtifactDocumentSaveRuntime({
    handleWriteFile,
  });

  const { renderToolbarActions: renderBaseArtifactWorkbenchToolbarActions } =
    useWorkspaceArtifactWorkbenchActions({
      activeTheme,
      projectId,
      syncGeneralArtifactToResource,
      onSaveArtifactDocument: handleSaveArtifactDocument,
    });

  const {
    handleHarnessLoadFilePreview,
    openArtifactInWorkbench: openWorkspaceArtifactInWorkbench,
    handleArtifactClick,
    handleFileClick,
    handleCodeBlockClick,
    shouldCollapseCodeBlocks,
    shouldCollapseCodeBlockInChat,
    handleTaskFileClick,
  } = useWorkspaceArtifactPreviewActions({
    activeTheme,
    mappedTheme,
    layoutMode,
    isThemeWorkbench,
    isGeneralCanvasOpen: generalCanvasState.isOpen,
    artifacts,
    currentCanvasArtifact,
    taskFiles,
    sessionFiles,
    readSessionFile,
    onRequestCanvasPreviewOpen: requestCanvasWorkbenchPreviewOpen,
    upsertGeneralArtifact,
    setSelectedArtifactId,
    setArtifactViewMode,
    setLayoutMode,
    setTaskFiles,
    setSelectedFileId,
    setGeneralCanvasState,
    setCanvasState,
  });

  const {
    bindArticleEditorRightSurface,
    bindRightSurfacePendingActions,
    handleWorkspaceArtifactClick,
  } = useWorkspaceRightSurfaceArtifactOpenRuntime({
    clearFocusedArtifactBlock,
    fallbackOpenArtifact: handleArtifactClick,
    openArticleWorkspaceRightSurface,
    setExpertInfoPanelCollapsed,
    setHarnessPanelVisible,
  });

  const handleWorkspaceFileClick = useCallback(
    (fileName: string, content: string) => {
      clearFocusedArtifactBlock();
      const normalizedFileName = fileName.trim();
      if (content.trim() || !normalizedFileName) {
        handleFileClick(fileName, content);
        return;
      }

      void (async () => {
        const absolutePath =
          resolveAbsoluteWorkspacePath(projectRootPath, normalizedFileName) ||
          normalizedFileName;
        const preview = await handleHarnessLoadFilePreview(absolutePath);
        if (preview.error) {
          toast.error(
            t("agentChat.filePreview.openFailed", {
              message: preview.error,
            }),
          );
          return;
        }
        const nextContent =
          !preview.isBinary && typeof preview.content === "string"
            ? preview.content
            : "";
        const nextFilePath = isAbsoluteWorkspacePath(normalizedFileName)
          ? preview.path || normalizedFileName
          : normalizedFileName;
        startTransition(() => {
          if (activeTheme === "general") {
            setSelectedArtifactId(null);
            requestCanvasWorkbenchPreviewOpen({
              filePath: preview.path || absolutePath,
              selectionKey: `default-preview:${
                nextFilePath || "current-canvas"
              }`,
            });
            setGeneralCanvasState(
              buildGeneralCanvasStateFromWorkspaceFile(
                nextFilePath,
                nextContent,
                { sourcePath: preview.path || absolutePath },
              ),
            );
            openCanvasForReason("user_open_file", setLayoutMode);
            return;
          }
          handleFileClick(nextFilePath, nextContent);
        });
      })();
    },
    [
      activeTheme,
      clearFocusedArtifactBlock,
      handleFileClick,
      handleHarnessLoadFilePreview,
      projectRootPath,
      requestCanvasWorkbenchPreviewOpen,
      setGeneralCanvasState,
      setLayoutMode,
      setSelectedArtifactId,
      t,
    ],
  );

  const openProjectFilePreviewInCanvas = useCallback(
    async ({
      relativePath,
      absolutePath,
      isCancelled,
    }: {
      relativePath?: string | null;
      absolutePath: string;
      isCancelled?: () => boolean;
    }) => {
      const preview = await handleHarnessLoadFilePreview(absolutePath);
      if (isCancelled?.()) {
        return false;
      }

      if (preview.error) {
        toast.error(`打开导出文件失败: ${preview.error}`);
        return false;
      }

      const nextContent =
        !preview.isBinary && typeof preview.content === "string"
          ? preview.content
          : "";
      const nextFilePath = relativePath?.trim() || preview.path || absolutePath;
      startTransition(() => {
        if (activeTheme === "general") {
          setSelectedArtifactId(null);
          requestCanvasWorkbenchPreviewOpen({
            filePath: preview.path || absolutePath,
            selectionKey: `default-preview:${nextFilePath || "current-canvas"}`,
          });
          setGeneralCanvasState(
            buildGeneralCanvasStateFromWorkspaceFile(
              nextFilePath,
              nextContent,
              { sourcePath: preview.path || absolutePath },
            ),
          );
          openCanvasForReason("user_open_file", setLayoutMode);
          return;
        }
        handleWorkspaceFileClick(nextFilePath, nextContent);
      });
      return true;
    },
    [
      activeTheme,
      handleHarnessLoadFilePreview,
      handleWorkspaceFileClick,
      requestCanvasWorkbenchPreviewOpen,
      setGeneralCanvasState,
      setLayoutMode,
      setSelectedArtifactId,
    ],
  );

  const handleOpenSavedSiteContent = useCallback(
    async ({
      projectId: targetProjectId,
      contentId: targetContentId,
      preferredTarget,
      projectFile,
    }: SiteSavedContentTarget) => {
      const relativePath = projectFile?.relativePath?.trim() || "";
      const canOpenInlineInCurrentWorkspace =
        preferredTarget === "project_file" &&
        Boolean(relativePath) &&
        Boolean(projectRootPath) &&
        Boolean(projectId) &&
        targetProjectId === projectId;

      if (canOpenInlineInCurrentWorkspace) {
        const absolutePath = resolveAbsoluteWorkspacePath(
          projectRootPath,
          relativePath,
        );
        if (absolutePath) {
          const opened = await openProjectFilePreviewInCanvas({
            relativePath,
            absolutePath,
          });
          if (opened) {
            return;
          }
        }
      }

      onNavigate?.("agent", {
        projectId: targetProjectId,
        contentId: targetContentId,
        lockTheme: true,
        fromResources: true,
        ...(preferredTarget === "project_file" && relativePath
          ? {
              initialProjectFileOpenTarget: {
                relativePath,
                requestKey: Date.now(),
              },
            }
          : {}),
      });
    },
    [onNavigate, openProjectFilePreviewInCanvas, projectId, projectRootPath],
  );

  const openMessageAttachmentPreview = useCallback(
    (
      target: Extract<MessagePreviewTarget, { kind: "message_attachment" }>,
      message: Message,
    ) => {
      const attachment = target.attachment;
      const sourceRef =
        attachment.sourcePath?.trim() ||
        attachment.sourceUri?.trim() ||
        `${message.id}:attachment:${target.index}`;
      const sourceUri = attachment.sourceUri?.trim();
      const canUseSourceUriAsPreview =
        Boolean(sourceUri) &&
        (/^(data|https?|file|asset):/u.test(sourceUri || "") ||
          sourceUri?.startsWith("//"));
      const previewUrl =
        attachment.previewUrl?.trim() ||
        (canUseSourceUriAsPreview ? sourceUri : undefined) ||
        (attachment.data.trim()
          ? `data:${attachment.mediaType || "image/png"};base64,${attachment.data.trim()}`
          : undefined);
      const sourcePath = attachment.sourcePath?.trim() || sourceRef;
      const projection = createPreviewArtifact({
        source: "session_file",
        sourceRef,
        path: sourcePath,
        title: `attachment-${target.index + 1}`,
        content: "",
        isBinary: true,
        mimeType: attachment.mediaType,
        previewUrl,
        meta: {
          openedFrom: "message-attachment",
          messageId: message.id,
          attachmentIndex: target.index,
        },
      });
      upsertGeneralArtifact(projection.artifact);
      handleWorkspaceArtifactClick(projection.artifact);
      requestCanvasWorkbenchPreviewOpen({
        filePath: sourcePath,
        selectionKey: `artifact:${projection.artifact.id}`,
      });
    },
    [
      handleWorkspaceArtifactClick,
      requestCanvasWorkbenchPreviewOpen,
      upsertGeneralArtifact,
    ],
  );

  const { openMediaReferencePreview, openMediaReferencePreviewPage } =
    useWorkspaceMediaReferencePreviewRuntime({
      artifacts,
      handleWorkspaceArtifactClick,
      requestCanvasWorkbenchPreviewOpen,
      threadId,
      setCanvasWorkbenchLayoutMode,
      setLayoutMode,
      t,
      upsertGeneralArtifact,
    });

  const renderMediaReferencePaginationActions = useCallback(
    (artifact: Artifact) => {
      const pageRequest = resolveMediaReferencePreviewPageRequest(
        artifact,
        messages,
      );
      if (!pageRequest) {
        return null;
      }

      return (
        <MediaReferencePreviewPaginationActions
          artifact={artifact}
          onOpenPage={(page) => {
            void openMediaReferencePreviewPage(
              pageRequest.target,
              pageRequest.message,
              page,
            );
          }}
        />
      );
    },
    [messages, openMediaReferencePreviewPage],
  );

  const renderArtifactWorkbenchToolbarActions = useCallback(
    (params: { artifact: Artifact; document: ArtifactDocumentV1 | null }) => {
      const baseActions = renderBaseArtifactWorkbenchToolbarActions(params);
      const paginationActions = renderMediaReferencePaginationActions(
        params.artifact,
      );
      if (!baseActions && !paginationActions) {
        return null;
      }
      return (
        <>
          {baseActions}
          {paginationActions}
        </>
      );
    },
    [
      renderBaseArtifactWorkbenchToolbarActions,
      renderMediaReferencePaginationActions,
    ],
  );

  const handleOpenUrlPreview = useCallback(
    (item: SearchResultPreviewItem) => {
      const url = item.url.trim();
      if (!url) {
        return;
      }
      void requestWorkspaceRightSurface({
        surfaceKind: "browser",
        origin: "user",
        priority: "foreground",
        reason: "open_url_preview",
        sessionId: threadId ?? null,
        workspaceId: projectId ?? null,
        candidateId: item.id,
        metadata: {
          browser: {
            launchUrl: url,
            title: item.title.trim() || null,
          },
        },
      }).catch((error) => {
        toast.error(
          error instanceof Error
            ? error.message
            : "无法打开浏览器工作区，请稍后重试。",
        );
      });
    },
    [
      projectId,
      threadId,
    ],
  );

  const handleOpenMessagePreview = useCallback(
    (target: MessagePreviewTarget, message: Message) => {
      if (target.kind === "image_workbench") {
        updateCurrentImageWorkbenchState((current) =>
          resolveImageWorkbenchStateForPreviewSelection({
            current,
            messages,
            preview: target.preview,
            selection: target.selection,
          }),
        );
        openCanvasForReason("user_open_message_preview", setLayoutMode);
        return;
      }

      if (target.kind === "message_attachment") {
        openMessageAttachmentPreview(target, message);
        return;
      }

      if (target.kind === "media_reference") {
        void openMediaReferencePreview(target, message);
        return;
      }

      if (target.preview.kind === "video_generate") {
        const preview = target.preview;
        const initialState = createInitialVideoState(preview.prompt);
        setCanvasState({
          ...initialState,
          providerId: preview.providerId?.trim() || "",
          model: preview.model?.trim() || "",
          duration: preview.durationSeconds || initialState.duration,
          aspectRatio: normalizeVideoAspectRatio(preview.aspectRatio),
          resolution: normalizeVideoResolution(preview.resolution),
          status: resolveVideoCanvasStatusFromPreview(target),
          selectedTaskId: preview.taskId,
          videoUrl: preview.videoUrl || undefined,
          errorMessage:
            preview.status === "failed" || preview.status === "cancelled"
              ? preview.statusMessage?.trim() || "视频任务未成功完成"
              : undefined,
        });
        openCanvasForReason("user_open_message_preview", setLayoutMode);
        return;
      }

      const matchedArtifact = resolveTaskPreviewArtifact(message, target);
      if (matchedArtifact) {
        handleWorkspaceArtifactClick(matchedArtifact);
        return;
      }

      const normalizedArtifactPath = normalizeArtifactProtocolPath(
        target.preview.artifactPath || null,
      );
      if (normalizedArtifactPath) {
        const matchedTaskFile = taskFiles.find((file) =>
          doesWorkspaceFileCandidateMatch(file.name, normalizedArtifactPath),
        );
        if (matchedTaskFile?.content?.trim()) {
          handleWorkspaceFileClick(
            matchedTaskFile.name,
            matchedTaskFile.content,
          );
          return;
        }
      }

      toast.info("当前任务产物还未同步完成，请稍后再试");
    },
    [
      handleWorkspaceArtifactClick,
      handleWorkspaceFileClick,
      messages,
      openMediaReferencePreview,
      openMessageAttachmentPreview,
      setCanvasState,
      setLayoutMode,
      taskFiles,
      updateCurrentImageWorkbenchState,
    ],
  );

  const handleOpenArtifactFromTimeline = useCallback(
    (target: ArtifactTimelineOpenTarget) => {
      void (async () => {
        let content = target.content;
        if (!content.trim()) {
          const absolutePath = resolveAbsoluteWorkspacePath(
            projectRootPath,
            target.filePath,
          );
          if (absolutePath) {
            const preview = await handleHarnessLoadFilePreview(absolutePath);
            if (preview.error) {
              toast.error(`打开产物失败: ${preview.error}`);
              return;
            }
            if (preview.isBinary) {
              toast.info("该产物是二进制格式，暂不支持在工作台预览");
              return;
            }
            content =
              typeof preview.content === "string" ? preview.content : "";
          }
        }

        handleWorkspaceFileClick(target.filePath, content);

        const normalizedBlockId = target.blockId?.trim();
        if (!normalizedBlockId) {
          return;
        }

        focusArtifactBlock(normalizedBlockId);
      })();
    },
    [
      focusArtifactBlock,
      handleHarnessLoadFilePreview,
      handleWorkspaceFileClick,
      projectRootPath,
    ],
  );

  const {
    handleOpenServiceSkillResultFile,
    preferredServiceSkillResultFileTarget,
  } = useWorkspaceServiceSkillResultFileRuntime({
    currentTurnId,
    effectiveThreadItems,
    handleWorkspaceFileClick,
    openProjectFilePreviewInCanvas,
    projectRootPath,
    taskFiles,
  });

  useEffect(() => {
    const relativePath = initialProjectFileOpenTarget?.relativePath?.trim();
    if (!relativePath) {
      handledInitialProjectFileOpenSignatureRef.current = "";
      return;
    }

    if (contentId && isInitialContentLoading) {
      return;
    }

    if (!projectRootPath && !isAbsoluteWorkspacePath(relativePath)) {
      return;
    }

    const absolutePath = resolveAbsoluteWorkspacePath(
      projectRootPath,
      relativePath,
    );
    if (!absolutePath) {
      return;
    }

    const signature = JSON.stringify({
      projectId: projectId ?? "",
      contentId: contentId ?? "",
      relativePath,
      requestKey: initialProjectFileOpenTarget?.requestKey ?? 0,
    });
    if (handledInitialProjectFileOpenSignatureRef.current === signature) {
      return;
    }
    handledInitialProjectFileOpenSignatureRef.current = signature;

    let cancelled = false;
    void (async () => {
      await openProjectFilePreviewInCanvas({
        relativePath,
        absolutePath,
        isCancelled: () => cancelled,
      });
    })();

    return () => {
      cancelled = true;
    };
  }, [
    contentId,
    initialProjectFileOpenTarget,
    isInitialContentLoading,
    openProjectFilePreviewInCanvas,
    projectRootPath,
    projectId,
  ]);

  return {
    bindArticleEditorRightSurface,
    bindRightSurfacePendingActions,
    handleArtifactClick,
    handleCodeBlockClick,
    handleFileClick,
    handleHarnessLoadFilePreview,
    handleOpenArtifactFromTimeline,
    handleOpenMessagePreview,
    handleOpenSavedSiteContent,
    handleOpenServiceSkillResultFile,
    handleOpenUrlPreview,
    handleSaveArtifactDocument,
    handleTaskFileClick,
    handleWorkspaceArtifactClick,
    handleWorkspaceFileClick,
    openProjectFilePreviewInCanvas,
    openWorkspaceArtifactInWorkbench,
    preferredServiceSkillResultFileTarget,
    renderArtifactWorkbenchToolbarActions,
    shouldCollapseCodeBlockInChat,
    shouldCollapseCodeBlocks,
  };
}
