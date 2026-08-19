import { useEffect } from "react";
import type { ComponentProps, ReactNode } from "react";
import {
  ArtifactCanvasOverlay,
  ArtifactRenderer,
  ArtifactToolbar,
} from "@/components/artifact";
import { CanvasFactory } from "@/components/workspace/canvas/CanvasFactory";
import type { ThemeType } from "@/lib/workspace/workbenchContract";
import { CanvasPanel as GeneralCanvasPanel } from "@/components/general-chat/bridge";
import type { Artifact } from "@/lib/artifact/types";
import { ImageTaskViewer } from "../components/ImageTaskViewer";
import { ArtifactWorkbenchShell } from "./ArtifactWorkbenchShell";
import { hasRenderableGeneralCanvasPreview } from "./generalCanvasPreviewState";
import { wrapPreviewWithWorkbenchTrigger } from "./workbenchPreviewHelpers";
import { resolveArtifactProtocolDocumentPayload } from "@/lib/artifact-protocol";
import type { ArtifactDocumentV1 } from "@/lib/artifact-document";
import type { AgentThreadItem } from "../types";
import type {
  ArtifactBlockRewriteCompletion,
  ArtifactBlockRewriteRunPayload,
} from "./artifactWorkbenchRewrite";
import {
  type ArtifactWorkbenchDocumentController,
  useArtifactWorkbenchDocumentController,
} from "./artifactWorkbenchDocument";

interface ArtifactWorkbenchPreviewProps {
  artifact: Artifact;
  currentCanvasArtifact: Artifact | null;
  displayedCanvasArtifact: Artifact | null;
  artifactOverlay:
    | ComponentProps<typeof ArtifactCanvasOverlay>["overlay"]
    | null;
  showPreviousVersionBadge: boolean;
  artifactViewMode: ComponentProps<typeof ArtifactToolbar>["viewMode"];
  onArtifactViewModeChange: NonNullable<
    ComponentProps<typeof ArtifactToolbar>["onViewModeChange"]
  >;
  artifactPreviewSize: ComponentProps<typeof ArtifactToolbar>["previewSize"];
  onArtifactPreviewSizeChange: NonNullable<
    ComponentProps<typeof ArtifactToolbar>["onPreviewSizeChange"]
  >;
  onSaveArtifactDocument?: (
    artifact: Artifact,
    document: ArtifactDocumentV1,
  ) => Promise<void> | void;
  onArtifactBlockRewriteRun?: (
    payload: ArtifactBlockRewriteRunPayload,
  ) =>
    | Promise<ArtifactBlockRewriteCompletion>
    | ArtifactBlockRewriteCompletion
    | void;
  threadItems?: AgentThreadItem[];
  focusedBlockId?: string | null;
  blockFocusRequestKey?: number;
  onJumpToTimelineItem?: (itemId: string) => void;
  onCloseCanvas: () => void;
  stackedWorkbenchTrigger?: ReactNode;
  onArtifactDocumentControllerChange?: (
    controller: ArtifactWorkbenchDocumentController | null,
  ) => void;
  canvasFactoryProps?: ComponentProps<
    typeof ArtifactRenderer
  >["canvasFactoryProps"];
  renderToolbarActions?: (params: {
    artifact: Artifact;
    document: ArtifactDocumentV1 | null;
  }) => ReactNode;
}

export function ArtifactWorkbenchPreview({
  artifact,
  currentCanvasArtifact,
  displayedCanvasArtifact,
  artifactOverlay,
  showPreviousVersionBadge,
  artifactViewMode,
  onArtifactViewModeChange,
  artifactPreviewSize,
  onArtifactPreviewSizeChange,
  onSaveArtifactDocument,
  onArtifactBlockRewriteRun,
  threadItems,
  focusedBlockId,
  blockFocusRequestKey,
  onJumpToTimelineItem,
  onCloseCanvas,
  stackedWorkbenchTrigger,
  onArtifactDocumentControllerChange,
  canvasFactoryProps,
  renderToolbarActions,
}: ArtifactWorkbenchPreviewProps) {
  const isLiveSelectedArtifact =
    currentCanvasArtifact?.id === artifact.id &&
    displayedCanvasArtifact !== null;
  const toolbarArtifact =
    isLiveSelectedArtifact && currentCanvasArtifact
      ? currentCanvasArtifact
      : artifact;
  const previewArtifact =
    isLiveSelectedArtifact && displayedCanvasArtifact
      ? displayedCanvasArtifact
      : artifact;
  const isArtifactStreaming = Boolean(
    isLiveSelectedArtifact &&
    currentCanvasArtifact &&
    displayedCanvasArtifact &&
    currentCanvasArtifact.id === displayedCanvasArtifact.id &&
    currentCanvasArtifact.id === previewArtifact.id &&
    currentCanvasArtifact.status === "streaming",
  );
  const artifactDocument = resolveArtifactProtocolDocumentPayload({
    content: previewArtifact.content,
    metadata: previewArtifact.meta,
  });
  const documentController = useArtifactWorkbenchDocumentController({
    artifact: previewArtifact,
    onSaveArtifactDocument,
    threadItems,
    focusedBlockId,
    blockFocusRequestKey,
    onJumpToTimelineItem,
  });

  useEffect(() => {
    if (!onArtifactDocumentControllerChange) {
      return;
    }

    onArtifactDocumentControllerChange(
      artifactDocument ? documentController : null,
    );
    return () => {
      onArtifactDocumentControllerChange(null);
    };
  }, [
    artifactDocument,
    documentController,
    onArtifactDocumentControllerChange,
  ]);

  const combinedActionsSlot = (
    <>
      {renderToolbarActions?.({
        artifact: previewArtifact,
        document: artifactDocument,
      })}
      {stackedWorkbenchTrigger}
    </>
  );

  if (artifactDocument) {
    return (
      <ArtifactWorkbenchShell
        artifact={previewArtifact}
        artifactOverlay={isLiveSelectedArtifact ? artifactOverlay : null}
        isStreaming={isArtifactStreaming}
        showPreviousVersionBadge={
          isLiveSelectedArtifact && showPreviousVersionBadge
        }
        viewMode={artifactViewMode}
        onViewModeChange={onArtifactViewModeChange}
        previewSize={artifactPreviewSize}
        onPreviewSizeChange={onArtifactPreviewSizeChange}
        onArtifactBlockRewriteRun={onArtifactBlockRewriteRun}
        onCloseCanvas={onCloseCanvas}
        actionsSlot={combinedActionsSlot}
        documentController={documentController}
      />
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-white">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <ArtifactToolbar
          artifact={toolbarArtifact}
          onClose={onCloseCanvas}
          isStreaming={Boolean(
            isLiveSelectedArtifact &&
            currentCanvasArtifact?.status === "streaming",
          )}
          viewMode={artifactViewMode}
          onViewModeChange={onArtifactViewModeChange}
          previewSize={artifactPreviewSize}
          onPreviewSizeChange={onArtifactPreviewSizeChange}
          tone="light"
          displayBadgeLabel={
            isLiveSelectedArtifact && showPreviousVersionBadge
              ? "预览上一版本"
              : undefined
          }
          actionsSlot={combinedActionsSlot}
        />
        <div className="relative min-h-0 flex-1 overflow-auto bg-white">
          <ArtifactRenderer
            artifact={previewArtifact}
            isStreaming={isArtifactStreaming}
            hideToolbar={true}
            viewMode={artifactViewMode}
            previewSize={artifactPreviewSize}
            tone="light"
            canvasFactoryProps={canvasFactoryProps}
          />
          {isLiveSelectedArtifact && artifactOverlay ? (
            <ArtifactCanvasOverlay overlay={artifactOverlay} />
          ) : null}
        </div>
      </div>
    </div>
  );
}

interface WorkspaceLiveCanvasPreviewProps {
  currentImageWorkbenchActive: boolean;
  imageWorkbenchProps: ComponentProps<typeof ImageTaskViewer>;
  onCloseCanvas: () => void;
  canvasRenderTheme: ThemeType;
  liveArtifact: Artifact | null;
  hasDisplayedLiveArtifact: boolean;
  renderArtifactPreview: (
    artifact: Artifact,
    options?: {
      stackedWorkbenchTrigger?: ReactNode;
      onArtifactDocumentControllerChange?: (
        controller: ArtifactWorkbenchDocumentController | null,
      ) => void;
    },
  ) => ReactNode;
  generalCanvasPanelProps: Omit<
    ComponentProps<typeof GeneralCanvasPanel>,
    "toolbarActions"
  > | null;
  shouldShowCanvasLoadingState: boolean;
  canvasLoadingLabel: string;
  canvasFactoryProps: ComponentProps<typeof CanvasFactory> | null;
  stackedWorkbenchTrigger?: ReactNode;
}

export function WorkspaceLiveCanvasPreview({
  currentImageWorkbenchActive,
  imageWorkbenchProps,
  onCloseCanvas,
  canvasRenderTheme,
  liveArtifact,
  hasDisplayedLiveArtifact,
  renderArtifactPreview,
  generalCanvasPanelProps,
  shouldShowCanvasLoadingState,
  canvasLoadingLabel,
  canvasFactoryProps,
  stackedWorkbenchTrigger,
}: WorkspaceLiveCanvasPreviewProps) {
  if (currentImageWorkbenchActive) {
    return wrapPreviewWithWorkbenchTrigger(
      <div className="flex h-full min-h-0 flex-col pt-4">
        <div className="min-h-0 flex-1">
          <ImageTaskViewer {...imageWorkbenchProps} onClose={onCloseCanvas} />
        </div>
      </div>,
      stackedWorkbenchTrigger,
    );
  }

  if (
    canvasRenderTheme === "general" &&
    liveArtifact &&
    hasDisplayedLiveArtifact
  ) {
    return renderArtifactPreview(liveArtifact, {
      stackedWorkbenchTrigger,
    });
  }

  if (canvasRenderTheme === "general") {
    if (
      generalCanvasPanelProps &&
      hasRenderableGeneralCanvasPreview(generalCanvasPanelProps.state)
    ) {
      return (
        <GeneralCanvasPanel
          {...generalCanvasPanelProps}
          chrome="embedded"
          toolbarActions={stackedWorkbenchTrigger}
        />
      );
    }
    return null;
  }

  if (shouldShowCanvasLoadingState) {
    return wrapPreviewWithWorkbenchTrigger(
      <div
        data-testid="canvas-loading-state"
        className="flex h-full items-center justify-center rounded-[10px] border border-dashed border-slate-200 bg-slate-50 text-sm text-slate-500"
      >
        {canvasLoadingLabel}
      </div>,
      stackedWorkbenchTrigger,
    );
  }

  if (!canvasFactoryProps) {
    return null;
  }

  return wrapPreviewWithWorkbenchTrigger(
    <CanvasFactory {...canvasFactoryProps} />,
    stackedWorkbenchTrigger,
  );
}
