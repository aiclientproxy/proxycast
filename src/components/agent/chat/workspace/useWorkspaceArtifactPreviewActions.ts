import {
  useCallback,
  useMemo,
  type Dispatch,
  type SetStateAction,
} from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { createInitialDocumentState } from "@/components/workspace/canvas/canvasUtils";
import type { CanvasStateUnion } from "@/components/workspace/canvas/canvasUtils";
import type { LayoutMode, ThemeType } from "@/lib/workspace/workbenchContract";
import {
  hasAgentRuntimeArtifactPreviewScope,
  readAgentRuntimeArtifactPreviewContent,
} from "@/lib/api/agentRuntime/appServerArtifactClient";
import { readFilePreview } from "@/lib/api/fileBrowser";
import type { SessionFile } from "@/lib/api/session-files";
import { createPreviewArtifactFromFile } from "@/lib/artifact/previewArtifact";
import type { Artifact } from "@/lib/artifact/types";
import { resolveArtifactProtocolFilePath } from "@/lib/artifact-protocol";
import {
  createLayeredDesignArtifact,
  type LayeredDesignDocumentInput,
} from "@/lib/layered-design";
import type { TaskFile } from "../components/TaskFiles";
import type { HarnessFilePreviewResult } from "../components/HarnessStatusPanel";
import { useArtifactAutoPreviewSync } from "../hooks/useArtifactAutoPreviewSync";
import { resolveDefaultArtifactViewMode } from "../utils/messageArtifacts";
import { openCanvasForReason } from "./canvasOpenPolicy";
import type { ApplyArtifactViewMode } from "./useWorkspaceArtifactViewModeControl";
import {
  isRenderableTaskFile,
  looksLikeSocialPublishPayload,
  normalizeSessionTaskFileType,
  resolveTaskFileType,
} from "./generalWorkbenchHelpers";
import { doesWorkspaceFileCandidateMatch } from "./workspaceFilePathMatch";
import { extractFileNameFromPath } from "./workspacePath";
import type { CanvasState as GeneralCanvasState } from "@/components/general-chat/bridge";

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function readNonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}

function omitUndefined<T extends Record<string, unknown>>(value: T): T {
  return Object.fromEntries(
    Object.entries(value).filter(([, entry]) => entry !== undefined),
  ) as T;
}

function resolveArtifactSelectionFilePath(artifact: Artifact): string | null {
  return (
    readNonEmptyString(artifact.meta.filePath) ??
    readNonEmptyString(artifact.meta.sourcePath) ??
    readNonEmptyString(artifact.title)
  );
}

function shouldUseProjectedPreviewArtifact(artifact: Artifact): boolean {
  if (artifact.meta.previewArtifact !== true) {
    return false;
  }

  const source =
    typeof artifact.meta.source === "string" ? artifact.meta.source : "";
  const contentKind =
    typeof artifact.meta.contentKind === "string"
      ? artifact.meta.contentKind
      : "";
  const renderMode =
    typeof artifact.meta.renderMode === "string"
      ? artifact.meta.renderMode
      : "";
  return (
    source === "url" ||
    source === "database_record" ||
    source === "app" ||
    contentKind === "app_shell" ||
    renderMode === "media" ||
    renderMode === "system_open" ||
    renderMode === "unsupported"
  );
}

function shouldRequestCanvasDocumentSelection(artifact: Artifact): boolean {
  return !shouldUseProjectedPreviewArtifact(artifact);
}

function hasLayeredDesignDocumentShape(
  value: unknown,
): value is LayeredDesignDocumentInput {
  if (!isRecord(value)) {
    return false;
  }

  const canvas = value.canvas;
  return (
    typeof value.id === "string" &&
    typeof value.title === "string" &&
    isRecord(canvas) &&
    typeof canvas.width === "number" &&
    typeof canvas.height === "number" &&
    Array.isArray(value.layers)
  );
}

function createWorkspaceLayeredDesignArtifactId(fileName: string): string {
  const normalized = fileName.trim() || "design.json";
  let hash = 0;
  for (let index = 0; index < normalized.length; index += 1) {
    hash = (hash * 31 + normalized.charCodeAt(index)) >>> 0;
  }

  return `artifact-workspace-layered-design-${hash.toString(36)}`;
}

function createLayeredDesignArtifactFromWorkspaceFile(
  fileName: string,
  content: string,
): Artifact | null {
  const trimmed = content.trim();
  if (!trimmed.startsWith("{")) {
    return null;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return null;
  }

  if (!hasLayeredDesignDocumentShape(parsed)) {
    return null;
  }

  const filename = extractFileNameFromPath(fileName) || "design.json";
  try {
    return createLayeredDesignArtifact(parsed, {
      artifactId: createWorkspaceLayeredDesignArtifactId(fileName),
      artifactTitle: parsed.title.trim() || filename,
      meta: {
        filePath: fileName,
        filename,
        openedFrom: "general-workbench-file",
      },
    });
  } catch {
    return null;
  }
}

function buildCanvasStateFromContent(params: {
  previous: CanvasStateUnion | null;
  mappedTheme: ThemeType;
  content: string;
}): CanvasStateUnion {
  const { previous, mappedTheme: _mappedTheme, content } = params;

  if (!previous || previous.type !== "document") {
    return createInitialDocumentState(content);
  }

  return {
    ...previous,
    content,
  };
}

interface UseWorkspaceArtifactPreviewActionsParams {
  activeTheme: string;
  mappedTheme: ThemeType;
  layoutMode: LayoutMode;
  isThemeWorkbench: boolean;
  isGeneralCanvasOpen: boolean;
  artifacts: Artifact[];
  currentCanvasArtifact: Artifact | null;
  taskFiles: TaskFile[];
  sessionFiles: SessionFile[];
  readSessionFile: (fileName: string) => Promise<string | null>;
  onRequestCanvasPreviewOpen?: (request: {
    filePath?: string | null;
    selectionKey?: string | null;
  }) => void;
  upsertGeneralArtifact: (artifact: Artifact) => void;
  setSelectedArtifactId: (artifactId: string | null) => void;
  setArtifactViewMode: ApplyArtifactViewMode;
  setLayoutMode: Dispatch<SetStateAction<LayoutMode>>;
  setTaskFiles: Dispatch<SetStateAction<TaskFile[]>>;
  setSelectedFileId: (fileId: string) => void;
  setGeneralCanvasState: Dispatch<SetStateAction<GeneralCanvasState>>;
  setCanvasState: Dispatch<SetStateAction<CanvasStateUnion | null>>;
}

interface WorkspaceArtifactPreviewActionsResult {
  handleHarnessLoadFilePreview: (
    path: string,
    artifact?: Artifact,
  ) => Promise<HarnessFilePreviewResult>;
  openArtifactInWorkbench: (artifact: Artifact) => Promise<void>;
  handleArtifactClick: (artifact: Artifact) => void;
  handleFileClick: (fileName: string, content: string) => void;
  handleCodeBlockClick: (language: string, code: string) => void;
  shouldCollapseCodeBlocks: boolean;
  shouldCollapseCodeBlockInChat: (language: string, code: string) => boolean;
  handleTaskFileClick: (file: TaskFile) => void;
}

export function useWorkspaceArtifactPreviewActions({
  activeTheme,
  mappedTheme,
  layoutMode,
  isThemeWorkbench,
  isGeneralCanvasOpen,
  artifacts,
  currentCanvasArtifact,
  taskFiles,
  sessionFiles,
  readSessionFile,
  onRequestCanvasPreviewOpen,
  upsertGeneralArtifact,
  setSelectedArtifactId,
  setArtifactViewMode,
  setLayoutMode,
  setTaskFiles,
  setSelectedFileId,
  setGeneralCanvasState,
  setCanvasState,
}: UseWorkspaceArtifactPreviewActionsParams): WorkspaceArtifactPreviewActionsResult {
  const { t } = useTranslation("agent");
  const handleHarnessLoadFilePreview = useCallback(
    async (
      path: string,
      artifact?: Artifact,
    ): Promise<HarnessFilePreviewResult> => {
      const normalizedPath = path.trim();
      const createFallbackResult = (
        overrides: Partial<HarnessFilePreviewResult> = {},
      ): HarnessFilePreviewResult => ({
        path: normalizedPath,
        content: null,
        isBinary: false,
        size: 0,
        error: null,
        ...overrides,
      });

      if (!normalizedPath) {
        return createFallbackResult({ error: "文件路径为空" });
      }

      if (
        artifact &&
        hasAgentRuntimeArtifactPreviewScope(artifact, normalizedPath)
      ) {
        try {
          const appServerContent = await readAgentRuntimeArtifactPreviewContent(
            artifact,
            normalizedPath,
          );
          if (!appServerContent) {
            return createFallbackResult({
              error: "App Server artifact 内容不可用",
            });
          }

          return createFallbackResult(
            omitUndefined({
              artifactId: appServerContent.artifactId,
              artifactRef: appServerContent.artifactRef,
              path: appServerContent.filePath || normalizedPath,
              content: appServerContent.content,
              metadata: appServerContent.metadata,
              size: appServerContent.content.length,
              title: appServerContent.title,
            }),
          );
        } catch (error) {
          return createFallbackResult({
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }

      const matchedTaskFile = taskFiles.find((file) =>
        doesWorkspaceFileCandidateMatch(file.name, normalizedPath),
      );
      if (matchedTaskFile) {
        const content = matchedTaskFile.content ?? "";
        return createFallbackResult({
          path: matchedTaskFile.name,
          content,
          size: content.length,
        });
      }

      const matchedSessionFile = sessionFiles.find((file) =>
        doesWorkspaceFileCandidateMatch(file.name, normalizedPath),
      );
      if (matchedSessionFile) {
        const content = await readSessionFile(matchedSessionFile.name);
        if (content !== null) {
          return createFallbackResult({
            path: matchedSessionFile.name,
            content,
            size: content.length,
          });
        }
      }

      try {
        const result = await readFilePreview(normalizedPath, 64 * 1024);

        return createFallbackResult({
          path: result.path || normalizedPath,
          content: result.content ?? null,
          isBinary: result.isBinary ?? false,
          size: result.size ?? 0,
          error: result.error ?? null,
        });
      } catch (error) {
        return createFallbackResult({
          error: error instanceof Error ? error.message : String(error),
        });
      }
    },
    [readSessionFile, sessionFiles, taskFiles],
  );

  useArtifactAutoPreviewSync({
    enabled: activeTheme === "general",
    artifact: currentCanvasArtifact,
    loadPreview: handleHarnessLoadFilePreview,
    onSyncArtifact: upsertGeneralArtifact,
  });

  const openArtifactInWorkbench = useCallback(
    async (artifact: Artifact) => {
      if (activeTheme === "general") {
        setGeneralCanvasState((previous) =>
          previous.isOpen || previous.content.trim() || previous.filename
            ? {
                ...previous,
                isOpen: false,
                contentType: "empty",
                content: "",
                filename: undefined,
                sourcePath: undefined,
                isEditing: false,
              }
            : previous,
        );
      }

      let nextArtifact = artifact;
      const artifactPath = resolveArtifactProtocolFilePath(artifact);
      const shouldLoadPreview =
        artifact.content.length === 0 &&
        artifactPath &&
        !shouldUseProjectedPreviewArtifact(artifact);

      if (shouldLoadPreview) {
        const preview = await handleHarnessLoadFilePreview(
          artifactPath,
          artifact,
        );
        if (preview.error) {
          toast.error(
            t("agentChat.workspace.artifactPreview.toast.readFailed", {
              error: preview.error,
            }),
          );
        } else if (preview.isBinary) {
          toast.info(
            t("agentChat.workspace.artifactPreview.toast.binaryUnsupported"),
          );
        } else if (typeof preview.content === "string") {
          nextArtifact = {
            ...artifact,
            content: preview.content,
            meta: {
              ...artifact.meta,
              filePath: preview.path || artifactPath,
              filename:
                artifact.meta.filename ||
                extractFileNameFromPath(preview.path || artifactPath),
            },
            updatedAt: Date.now(),
          };
        }
      }

      if (activeTheme === "general") {
        upsertGeneralArtifact(nextArtifact);
      }

      setSelectedArtifactId(nextArtifact.id);
      onRequestCanvasPreviewOpen?.({
        filePath: shouldRequestCanvasDocumentSelection(nextArtifact)
          ? resolveArtifactSelectionFilePath(nextArtifact)
          : null,
        selectionKey: `artifact:${nextArtifact.id}`,
      });
      setArtifactViewMode(
        resolveDefaultArtifactViewMode(nextArtifact, {
          preferSourceWhenStreaming: true,
        }),
        { artifactId: nextArtifact.id },
      );
      openCanvasForReason("user_open_artifact", setLayoutMode);
    },
    [
      activeTheme,
      handleHarnessLoadFilePreview,
      onRequestCanvasPreviewOpen,
      setArtifactViewMode,
      setGeneralCanvasState,
      setLayoutMode,
      setSelectedArtifactId,
      t,
      upsertGeneralArtifact,
    ],
  );

  const handleArtifactClick = useCallback(
    (artifact: Artifact) => {
      void openArtifactInWorkbench(artifact);
    },
    [openArtifactInWorkbench],
  );

  const findArtifactForCodeBlock = useCallback(
    (code: string) => {
      const normalizedCode = code.replace(/\r\n/g, "\n").trimEnd();
      if (!normalizedCode) {
        return undefined;
      }

      return artifacts.find((artifact) => {
        if (typeof artifact.content !== "string") {
          return false;
        }
        return (
          artifact.content.replace(/\r\n/g, "\n").trimEnd() === normalizedCode
        );
      });
    },
    [artifacts],
  );

  const applyContentToCanvas = useCallback(
    (content: string) => {
      setCanvasState((previous) =>
        buildCanvasStateFromContent({
          previous,
          mappedTheme,
          content,
        }),
      );
      openCanvasForReason("user_open_file", setLayoutMode);
    },
    [mappedTheme, setCanvasState, setLayoutMode],
  );

  const openFilePreviewArtifact = useCallback(
    (params: {
      fileName: string;
      content: string;
      sourcePath?: string | null;
      isBinary?: boolean | null;
      size?: number | null;
      error?: string | null;
    }) => {
      const projection = createPreviewArtifactFromFile({
        filePath: params.fileName,
        path: params.sourcePath || params.fileName,
        content: params.content,
        isBinary: params.isBinary,
        size: params.size,
        error: params.error,
        meta: {
          openedFrom: "general-workbench-file",
        },
      });
      void openArtifactInWorkbench(projection.artifact);
    },
    [openArtifactInWorkbench],
  );

  const handleFileClick = useCallback(
    (fileName: string, content: string) => {
      if (activeTheme === "general") {
        const layeredDesignArtifact =
          createLayeredDesignArtifactFromWorkspaceFile(fileName, content);
        if (layeredDesignArtifact) {
          void openArtifactInWorkbench(layeredDesignArtifact);
          return;
        }

        if (!content.trim()) {
          void (async () => {
            const preview = await handleHarnessLoadFilePreview(fileName);
            const nextContent =
              !preview.error && !preview.isBinary && preview.content !== null
                ? preview.content || ""
                : content;
            openFilePreviewArtifact({
              fileName,
              content: nextContent,
              sourcePath: !preview.error ? preview.path || fileName : fileName,
              isBinary: preview.isBinary,
              size: preview.size,
              error: preview.error,
            });
          })();
          return;
        }

        openFilePreviewArtifact({ fileName, content });
        return;
      }

      const nextFileType = resolveTaskFileType(fileName, content);
      const fallbackTaskFileId = crypto.randomUUID();
      const existingTaskFile = taskFiles.find((file) => file.name === fileName);
      const selectedTaskFileId = existingTaskFile?.id ?? fallbackTaskFileId;
      setSelectedFileId(selectedTaskFileId);
      setTaskFiles((previous) => {
        const existingFile = previous.find((file) => file.name === fileName);
        if (existingFile) {
          return previous;
        }

        const nextFile: TaskFile = {
          id: fallbackTaskFileId,
          name: fileName,
          type: nextFileType,
          content,
          version: 1,
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        return [...previous, nextFile];
      });

      if (
        !isRenderableTaskFile(
          { name: fileName, type: nextFileType },
          isThemeWorkbench,
        )
      ) {
        toast.info(
          t("agentChat.workspace.artifactPreview.toast.auxiliaryNotRendered"),
        );
        return;
      }

      onRequestCanvasPreviewOpen?.({
        filePath: fileName,
        selectionKey: `task:${selectedTaskFileId}`,
      });
      applyContentToCanvas(content);
    },
    [
      activeTheme,
      applyContentToCanvas,
      handleHarnessLoadFilePreview,
      isThemeWorkbench,
      onRequestCanvasPreviewOpen,
      openFilePreviewArtifact,
      openArtifactInWorkbench,
      setSelectedFileId,
      setTaskFiles,
      taskFiles,
      t,
    ],
  );

  const hydrateTaskFileContent = useCallback(
    async (file: TaskFile): Promise<TaskFile | null> => {
      if (typeof file.content === "string") {
        return file;
      }

      const matchedSessionFile = sessionFiles.find((candidate) =>
        doesWorkspaceFileCandidateMatch(candidate.name, file.name),
      );
      if (!matchedSessionFile) {
        return file;
      }

      const content = await readSessionFile(matchedSessionFile.name);
      if (content === null) {
        return null;
      }

      const hydratedFile: TaskFile = {
        ...file,
        name: matchedSessionFile.name,
        type: normalizeSessionTaskFileType(
          matchedSessionFile.fileType ?? file.type,
          matchedSessionFile.name,
          content,
        ),
        content,
        metadata: file.metadata ?? matchedSessionFile.metadata,
        createdAt: file.createdAt || matchedSessionFile.createdAt || Date.now(),
        updatedAt:
          matchedSessionFile.updatedAt ||
          file.updatedAt ||
          matchedSessionFile.createdAt ||
          Date.now(),
      };

      setTaskFiles((previous) => {
        const matchedIndex = previous.findIndex(
          (item) =>
            item.id === file.id ||
            doesWorkspaceFileCandidateMatch(item.name, matchedSessionFile.name),
        );
        if (matchedIndex < 0) {
          return [...previous, hydratedFile];
        }

        const next = [...previous];
        next[matchedIndex] = hydratedFile;
        return next;
      });

      return hydratedFile;
    },
    [readSessionFile, sessionFiles, setTaskFiles],
  );

  const handleCodeBlockClick = useCallback(
    (language: string, code: string) => {
      console.log("[AgentChatPage] 代码块点击:", language);

      const matchingArtifact = findArtifactForCodeBlock(code);
      if (!matchingArtifact) {
        console.warn(
          "[AgentChatPage] 代码块未匹配到 artifact，保持内联渲染:",
          language,
        );
        return;
      }

      console.log("[AgentChatPage] 找到匹配的 artifact:", matchingArtifact.id);
      void openArtifactInWorkbench(matchingArtifact);
    },
    [findArtifactForCodeBlock, openArtifactInWorkbench],
  );

  const shouldCollapseCodeBlocks = useMemo(() => {
    if (activeTheme !== "general") {
      return false;
    }
    if (layoutMode === "chat") {
      return false;
    }
    return artifacts.length > 0 || isGeneralCanvasOpen;
  }, [activeTheme, artifacts.length, isGeneralCanvasOpen, layoutMode]);

  const shouldCollapseCodeBlockInChat = useCallback(
    (language: string, code: string) => {
      if (!shouldCollapseCodeBlocks) {
        return false;
      }

      const normalizedLanguage = language.trim().toLowerCase();
      if (
        ["", "text", "plaintext", "plain", "txt", "markdown", "md"].includes(
          normalizedLanguage,
        )
      ) {
        return false;
      }

      return Boolean(findArtifactForCodeBlock(code));
    },
    [findArtifactForCodeBlock, shouldCollapseCodeBlocks],
  );

  const handleTaskFileClick = useCallback(
    (file: TaskFile) => {
      void (async () => {
        const resolvedFile = await hydrateTaskFileContent(file);
        if (!resolvedFile) {
          toast.error(
            t(
              "agentChat.workspace.artifactPreview.toast.sessionFileReadFailed",
            ),
          );
          return;
        }

        if (activeTheme === "general") {
          if (!resolvedFile.content?.trim()) {
            toast.info(
              t(
                "agentChat.workspace.artifactPreview.toast.auxiliaryNotRendered",
              ),
            );
            return;
          }

          openFilePreviewArtifact({
            fileName: resolvedFile.name,
            content: resolvedFile.content ?? "",
          });
          return;
        }

        setSelectedFileId(resolvedFile.id);

        if (
          !isRenderableTaskFile(resolvedFile, isThemeWorkbench) ||
          looksLikeSocialPublishPayload(resolvedFile.content || "") ||
          !resolvedFile.content?.trim()
        ) {
          toast.info(
            t("agentChat.workspace.artifactPreview.toast.auxiliaryNotRendered"),
          );
          return;
        }

        applyContentToCanvas(resolvedFile.content ?? "");
      })();
    },
    [
      activeTheme,
      applyContentToCanvas,
      hydrateTaskFileContent,
      isThemeWorkbench,
      openFilePreviewArtifact,
      setSelectedFileId,
      t,
    ],
  );

  return {
    handleHarnessLoadFilePreview,
    openArtifactInWorkbench,
    handleArtifactClick,
    handleFileClick,
    handleCodeBlockClick,
    shouldCollapseCodeBlocks,
    shouldCollapseCodeBlockInChat,
    handleTaskFileClick,
  };
}
