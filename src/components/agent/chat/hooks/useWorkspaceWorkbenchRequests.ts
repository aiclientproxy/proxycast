import { useCallback, useMemo, useRef, useState } from "react";

import type { CanvasWorkbenchPreviewOpenRequest } from "../components/CanvasWorkbenchLayout";

export interface CanvasWorkbenchPreviewRequestInput {
  filePath?: string | null;
  selectionKey?: string | null;
}

export interface WorkspaceWorkbenchRequestsController {
  canvasWorkbenchPreviewOpenRequest: CanvasWorkbenchPreviewOpenRequest | null;
  focusedArtifactBlockId: string | null;
  artifactBlockFocusRequestKey: number;
  focusedTimelineItemId: string | null;
  timelineFocusRequestKey: number;
  requestCanvasWorkbenchPreviewOpen: (
    request: CanvasWorkbenchPreviewRequestInput,
  ) => void;
  handleCanvasWorkbenchPreviewOpenRequestHandled: (
    requestKey: string | number,
  ) => void;
  clearFocusedArtifactBlock: () => void;
  focusArtifactBlock: (blockId: string | null | undefined) => void;
  jumpToTimelineItem: (itemId: string | null | undefined) => boolean;
}

export function useWorkspaceWorkbenchRequests(): WorkspaceWorkbenchRequestsController {
  const [
    canvasWorkbenchPreviewOpenRequest,
    setCanvasWorkbenchPreviewOpenRequest,
  ] = useState<CanvasWorkbenchPreviewOpenRequest | null>(null);
  const [focusedArtifactBlockId, setFocusedArtifactBlockId] = useState<
    string | null
  >(null);
  const [artifactBlockFocusRequestKey, setArtifactBlockFocusRequestKey] =
    useState(0);
  const [focusedTimelineItemId, setFocusedTimelineItemId] = useState<
    string | null
  >(null);
  const [timelineFocusRequestKey, setTimelineFocusRequestKey] = useState(0);
  const canvasWorkbenchPreviewRequestKeyRef = useRef(0);

  const requestCanvasWorkbenchPreviewOpen = useCallback(
    (request: CanvasWorkbenchPreviewRequestInput) => {
      canvasWorkbenchPreviewRequestKeyRef.current += 1;
      setCanvasWorkbenchPreviewOpenRequest({
        requestKey: canvasWorkbenchPreviewRequestKeyRef.current,
        filePath: request.filePath || null,
        selectionKey: request.selectionKey || null,
      });
    },
    [],
  );

  const handleCanvasWorkbenchPreviewOpenRequestHandled = useCallback(
    (requestKey: string | number) => {
      setCanvasWorkbenchPreviewOpenRequest((current) =>
        current?.requestKey === requestKey ? null : current,
      );
    },
    [],
  );

  const clearFocusedArtifactBlock = useCallback(() => {
    setFocusedArtifactBlockId(null);
  }, []);

  const focusArtifactBlock = useCallback(
    (blockId: string | null | undefined) => {
      const normalizedBlockId = blockId?.trim();
      if (!normalizedBlockId) {
        return;
      }

      setFocusedArtifactBlockId(normalizedBlockId);
      setArtifactBlockFocusRequestKey((current) => current + 1);
    },
    [],
  );

  const jumpToTimelineItem = useCallback(
    (itemId: string | null | undefined) => {
      const normalizedItemId = itemId?.trim();
      if (!normalizedItemId) {
        return false;
      }

      setFocusedTimelineItemId(normalizedItemId);
      setTimelineFocusRequestKey((current) => current + 1);
      return true;
    },
    [],
  );

  return useMemo(
    () => ({
      canvasWorkbenchPreviewOpenRequest,
      focusedArtifactBlockId,
      artifactBlockFocusRequestKey,
      focusedTimelineItemId,
      timelineFocusRequestKey,
      requestCanvasWorkbenchPreviewOpen,
      handleCanvasWorkbenchPreviewOpenRequestHandled,
      clearFocusedArtifactBlock,
      focusArtifactBlock,
      jumpToTimelineItem,
    }),
    [
      artifactBlockFocusRequestKey,
      canvasWorkbenchPreviewOpenRequest,
      clearFocusedArtifactBlock,
      focusedArtifactBlockId,
      focusedTimelineItemId,
      focusArtifactBlock,
      handleCanvasWorkbenchPreviewOpenRequestHandled,
      jumpToTimelineItem,
      requestCanvasWorkbenchPreviewOpen,
      timelineFocusRequestKey,
    ],
  );
}
