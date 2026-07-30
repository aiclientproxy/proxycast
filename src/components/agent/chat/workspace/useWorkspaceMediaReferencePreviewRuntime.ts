import {
  useCallback,
  useEffect,
  useRef,
  type Dispatch,
  type SetStateAction,
} from "react";
import { createAppServerClient } from "@/lib/api/appServer";
import { logAgentDebug } from "@/lib/agentDebug";
import type { Artifact } from "@/lib/artifact/types";
import type { LayoutMode } from "@/lib/workspace/workbenchContract";
import type { CanvasWorkbenchLayoutMode } from "../components/CanvasWorkbenchLayout";
import type { CanvasWorkbenchPreviewRequestInput } from "../hooks/useWorkspaceWorkbenchRequests";
import type { Message, MessagePreviewTarget } from "../types";
import { openCanvasForReason } from "./canvasOpenPolicy";
import {
  createMediaReferenceChunkedObjectUrlPreviewArtifact,
  createMediaReferenceProgressPreviewArtifact,
  createMediaReferencePreviewArtifact,
  type MediaReferencePreviewProgress,
} from "./mediaReferencePreviewArtifacts";
import { createMediaReferencePagedPreviewArtifact } from "./mediaReferencePreviewPagination";

export const WORKSPACE_MEDIA_REFERENCE_PREVIEW_RUNTIME_POLICY = {
  objectUrlMaxCount: 4,
  objectUrlMaxBytes: 64 * 1024 * 1024,
} as const;

type Translate = (key: string, options?: Record<string, unknown>) => string;

export interface UseWorkspaceMediaReferencePreviewRuntimeParams {
  artifacts: Artifact[];
  threadId?: string | null;
  t: Translate;
  setLayoutMode: Dispatch<SetStateAction<LayoutMode>>;
  setCanvasWorkbenchLayoutMode: Dispatch<
    SetStateAction<CanvasWorkbenchLayoutMode>
  >;
  upsertGeneralArtifact: (artifact: Artifact) => void;
  handleWorkspaceArtifactClick: (artifact: Artifact) => void;
  requestCanvasWorkbenchPreviewOpen: (
    request: CanvasWorkbenchPreviewRequestInput,
  ) => void;
}

export interface WorkspaceMediaReferencePreviewRuntime {
  openMediaReferencePreview: (
    target: Extract<MessagePreviewTarget, { kind: "media_reference" }>,
    message: Message,
  ) => Promise<void>;
  openMediaReferencePreviewPage: (
    target: Extract<MessagePreviewTarget, { kind: "media_reference" }>,
    message: Message,
    page: { offset: number; length?: number },
  ) => Promise<void>;
}

function readMediaPreviewObjectUrl(artifact: Artifact | undefined): string {
  const value = artifact?.meta?.mediaPreviewObjectUrl;
  return typeof value === "string" ? value.trim() : "";
}

function totalTrackedObjectUrlBytes(objectUrls: Map<string, number>): number {
  let totalBytes = 0;
  for (const bytes of objectUrls.values()) {
    totalBytes += bytes;
  }
  return totalBytes;
}

export function useWorkspaceMediaReferencePreviewRuntime({
  artifacts,
  handleWorkspaceArtifactClick,
  requestCanvasWorkbenchPreviewOpen,
  threadId,
  setCanvasWorkbenchLayoutMode,
  setLayoutMode,
  t,
  upsertGeneralArtifact,
}: UseWorkspaceMediaReferencePreviewRuntimeParams): WorkspaceMediaReferencePreviewRuntime {
  const activePreviewRequestIdRef = useRef(0);
  const activePreviewAbortControllerRef = useRef<AbortController | null>(null);
  const isDisposedRef = useRef(false);
  const mediaPreviewObjectUrlsRef = useRef<Map<string, number>>(new Map());

  const isPreviewRequestCurrent = useCallback((requestId: number) => {
    return (
      !isDisposedRef.current && activePreviewRequestIdRef.current === requestId
    );
  }, []);

  const enforceMediaPreviewObjectUrlBudget = useCallback(() => {
    const objectUrls = mediaPreviewObjectUrlsRef.current;
    let totalBytes = totalTrackedObjectUrlBytes(objectUrls);
    while (
      objectUrls.size > 1 &&
      (objectUrls.size >
        WORKSPACE_MEDIA_REFERENCE_PREVIEW_RUNTIME_POLICY.objectUrlMaxCount ||
        totalBytes >
          WORKSPACE_MEDIA_REFERENCE_PREVIEW_RUNTIME_POLICY.objectUrlMaxBytes)
    ) {
      const oldest = objectUrls.entries().next().value as
        | [string, number]
        | undefined;
      if (!oldest) {
        break;
      }
      const [objectUrl, bytes] = oldest;
      objectUrls.delete(objectUrl);
      totalBytes -= bytes;
      URL.revokeObjectURL(objectUrl);
    }
  }, []);

  const registerMediaPreviewObjectUrl = useCallback(
    (blob: Blob) => {
      const objectUrl = URL.createObjectURL(blob);
      if (isDisposedRef.current) {
        URL.revokeObjectURL(objectUrl);
        return objectUrl;
      }
      mediaPreviewObjectUrlsRef.current.set(objectUrl, blob.size);
      enforceMediaPreviewObjectUrlBudget();
      return objectUrl;
    },
    [enforceMediaPreviewObjectUrlBudget],
  );

  const revokeMediaPreviewObjectUrl = useCallback((objectUrl: string) => {
    const normalized = objectUrl.trim();
    if (!normalized || !mediaPreviewObjectUrlsRef.current.delete(normalized)) {
      return;
    }
    URL.revokeObjectURL(normalized);
  }, []);

  useEffect(
    () => () => {
      isDisposedRef.current = true;
      activePreviewRequestIdRef.current += 1;
      activePreviewAbortControllerRef.current?.abort(
        "media preview runtime disposed",
      );
      activePreviewAbortControllerRef.current = null;
      for (const objectUrl of mediaPreviewObjectUrlsRef.current) {
        URL.revokeObjectURL(objectUrl[0]);
      }
      mediaPreviewObjectUrlsRef.current.clear();
    },
    [],
  );

  const openPreviewArtifact = useCallback(
    (previewArtifact: Artifact) => {
      openCanvasForReason("user_open_message_preview", setLayoutMode);
      setCanvasWorkbenchLayoutMode("split");
      upsertGeneralArtifact(previewArtifact);
      handleWorkspaceArtifactClick(previewArtifact);
      const artifactFilePath =
        typeof previewArtifact.meta?.filePath === "string"
          ? previewArtifact.meta.filePath
          : previewArtifact.title;
      requestCanvasWorkbenchPreviewOpen({
        filePath: artifactFilePath,
        selectionKey: `artifact:${previewArtifact.id}`,
      });
    },
    [
      handleWorkspaceArtifactClick,
      requestCanvasWorkbenchPreviewOpen,
      setCanvasWorkbenchLayoutMode,
      setLayoutMode,
      upsertGeneralArtifact,
    ],
  );

  const beginPreviewRequest = useCallback(() => {
    const requestId = activePreviewRequestIdRef.current + 1;
    activePreviewRequestIdRef.current = requestId;
    activePreviewAbortControllerRef.current?.abort(
      "media preview request superseded",
    );
    const abortController = new AbortController();
    activePreviewAbortControllerRef.current = abortController;
    return { abortController, requestId };
  }, []);

  const openMediaReferencePreview = useCallback(
    async (
      target: Extract<MessagePreviewTarget, { kind: "media_reference" }>,
      message: Message,
    ) => {
      const { abortController, requestId } = beginPreviewRequest();
      let artifact = createMediaReferencePreviewArtifact({
        message,
        target,
        t,
      });
      const updateProgressArtifact = (
        progress: MediaReferencePreviewProgress,
      ) => {
        if (!isPreviewRequestCurrent(requestId)) {
          return;
        }
        openPreviewArtifact(
          createMediaReferenceProgressPreviewArtifact({
            message,
            progress,
            target,
            t,
          }),
        );
      };
      try {
        let client: ReturnType<typeof createAppServerClient> | null = null;
        const mediaArtifact =
          await createMediaReferenceChunkedObjectUrlPreviewArtifact({
            threadId,
            target,
            message,
            t,
            createObjectUrl: registerMediaPreviewObjectUrl,
            onProgress: updateProgressArtifact,
            shouldContinue: () => isPreviewRequestCurrent(requestId),
            readMedia: async (request) => {
              if (!isPreviewRequestCurrent(requestId)) {
                throw new Error("media preview request superseded");
              }
              client ??= createAppServerClient();
              const response = await client.readMedia(request, {
                signal: abortController.signal,
              });
              if (!isPreviewRequestCurrent(requestId)) {
                throw new Error("media preview request superseded");
              }
              return response.result;
            },
          });
        if (!isPreviewRequestCurrent(requestId)) {
          if (mediaArtifact) {
            revokeMediaPreviewObjectUrl(
              readMediaPreviewObjectUrl(mediaArtifact),
            );
          }
          return;
        }
        if (mediaArtifact) {
          artifact = mediaArtifact;
          const previousObjectUrl = readMediaPreviewObjectUrl(
            artifacts.find((candidate) => candidate.id === artifact.id),
          );
          const nextObjectUrl = readMediaPreviewObjectUrl(artifact);
          if (previousObjectUrl && previousObjectUrl !== nextObjectUrl) {
            revokeMediaPreviewObjectUrl(previousObjectUrl);
          }
        }
      } catch (error) {
        if (!isPreviewRequestCurrent(requestId)) {
          return;
        }
        logAgentDebug(
          "AgentChatWorkspace",
          "media_reference_sidecar_read_failed",
          {
            threadId,
            uri: target.reference.uri,
            error,
          },
          { level: "warn", throttleMs: 5_000 },
        );
      }
      if (!isPreviewRequestCurrent(requestId)) {
        revokeMediaPreviewObjectUrl(readMediaPreviewObjectUrl(artifact));
        return;
      }
      openPreviewArtifact(artifact);
      if (activePreviewAbortControllerRef.current === abortController) {
        activePreviewAbortControllerRef.current = null;
      }
    },
    [
      artifacts,
      beginPreviewRequest,
      isPreviewRequestCurrent,
      openPreviewArtifact,
      registerMediaPreviewObjectUrl,
      revokeMediaPreviewObjectUrl,
      threadId,
      t,
    ],
  );

  const openMediaReferencePreviewPage = useCallback(
    async (
      target: Extract<MessagePreviewTarget, { kind: "media_reference" }>,
      message: Message,
      page: { offset: number; length?: number },
    ) => {
      const { abortController, requestId } = beginPreviewRequest();
      let artifact = createMediaReferencePreviewArtifact({
        message,
        target,
        t,
      });
      try {
        let client: ReturnType<typeof createAppServerClient> | null = null;
        const pageArtifact = await createMediaReferencePagedPreviewArtifact({
          threadId,
          target,
          message,
          t,
          offset: page.offset,
          length: page.length,
          shouldContinue: () => isPreviewRequestCurrent(requestId),
          readMedia: async (request) => {
            if (!isPreviewRequestCurrent(requestId)) {
              throw new Error("media preview request superseded");
            }
            client ??= createAppServerClient();
            const response = await client.readMedia(request, {
              signal: abortController.signal,
            });
            if (!isPreviewRequestCurrent(requestId)) {
              throw new Error("media preview request superseded");
            }
            return response.result;
          },
        });
        if (!isPreviewRequestCurrent(requestId)) {
          return;
        }
        if (pageArtifact) {
          artifact = pageArtifact;
        }
      } catch (error) {
        if (!isPreviewRequestCurrent(requestId)) {
          return;
        }
        logAgentDebug(
          "AgentChatWorkspace",
          "media_reference_sidecar_page_read_failed",
          {
            offset: page.offset,
            threadId,
            uri: target.reference.uri,
            error,
          },
          { level: "warn", throttleMs: 5_000 },
        );
      }
      if (!isPreviewRequestCurrent(requestId)) {
        return;
      }
      openPreviewArtifact(artifact);
      if (activePreviewAbortControllerRef.current === abortController) {
        activePreviewAbortControllerRef.current = null;
      }
    },
    [
      beginPreviewRequest,
      isPreviewRequestCurrent,
      openPreviewArtifact,
      threadId,
      t,
    ],
  );

  return { openMediaReferencePreview, openMediaReferencePreviewPage };
}
