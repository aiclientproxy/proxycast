import { useEffect, useMemo, useState } from "react";
import type { Artifact } from "@/lib/artifact/types";
import { WorkspaceArticleEditorSurface } from "./WorkspaceArticleEditorSurface";
import type {
  WorkspaceArticleObject,
  WorkspaceArticleObjectRef,
  WorkspaceArticleWorkspace,
  WorkspaceArticleWorkspaceActionIntent,
} from "./workspaceArticleWorkspaceModel";
import {
  buildWorkspaceArticleWorkspaceViewModel,
  selectWorkspaceArticleDraftObject,
} from "./workspaceArticleWorkspaceModel";
import { buildWorkspaceArticleWorkspacePreviewArtifact } from "./workspaceArticleWorkspacePreviewArtifact";
import {
  buildWorkspaceArticleObjectKey,
  readWorkspaceArticleWorkspaceSelectedObjectKey,
  writeWorkspaceArticleWorkspaceSelectedObjectKey,
} from "./workspaceArticleWorkspaceSelection";
import type { WorkspaceArticleMarkdownChange } from "./workspaceArticleWorkspaceEditedDraft";

interface WorkspaceArticleWorkspaceSelectionChange {
  articleWorkspace: WorkspaceArticleWorkspace;
  object: WorkspaceArticleObject;
}

interface WorkspaceArticleEditorRightSurfaceProps {
  articleWorkspace: WorkspaceArticleWorkspace;
  actionsDisabled?: boolean;
  onActionIntent?: (intent: WorkspaceArticleWorkspaceActionIntent) => void;
  onOpenPreviewArtifact?: (artifact: Artifact) => void;
  onArticleMarkdownChange?: (change: WorkspaceArticleMarkdownChange) => void;
  onSelectedObjectChange?: (
    change: WorkspaceArticleWorkspaceSelectionChange,
  ) => void;
}

export function WorkspaceArticleEditorRightSurface({
  actionsDisabled = false,
  articleWorkspace,
  onArticleMarkdownChange,
  onActionIntent,
  onOpenPreviewArtifact,
  onSelectedObjectChange,
}: WorkspaceArticleEditorRightSurfaceProps) {
  const [selectedObjectKey, setSelectedObjectKey] = useState<string | null>(
    null,
  );
  const articleWorkspaceSelectionSignature = useMemo(
    () =>
      [
        articleWorkspace.workspaceId ?? "",
        articleWorkspace.sessionId,
        articleWorkspace.appId,
        articleWorkspace.objects.map(buildWorkspaceArticleObjectKey).join("|"),
      ].join("::"),
    [articleWorkspace],
  );

  useEffect(() => {
    setSelectedObjectKey(null);
  }, [articleWorkspaceSelectionSignature]);

  const selectedObject = useMemo(
    () =>
      resolveArticleEditorObject(
        articleWorkspace,
        selectedObjectKey,
      ),
    [articleWorkspace, selectedObjectKey],
  );
  const activeArticleWorkspace = useMemo(() => {
    if (!selectedObject) {
      return articleWorkspace;
    }
    return {
      ...articleWorkspace,
      selectedObjectRef: selectedObject.ref,
    };
  }, [articleWorkspace, selectedObject]);
  const viewModel = buildWorkspaceArticleWorkspaceViewModel(
    activeArticleWorkspace,
  );
  const selectedObjectKeyForRender = buildWorkspaceArticleObjectKey(
    viewModel.selectedObject,
  );
  const previewArtifact = onOpenPreviewArtifact
    ? buildWorkspaceArticleWorkspacePreviewArtifact({
        artifactIds: viewModel.selectedArtifactIds,
        layout: viewModel.selectedSurface.layout,
        object: viewModel.selectedObject,
        preview: viewModel.selectedPreview,
        articleWorkspace: activeArticleWorkspace,
      })
    : null;

  const handleSelectObject = (object: WorkspaceArticleObject) => {
    const nextObjectKey = buildWorkspaceArticleObjectKey(object);
    setSelectedObjectKey(nextObjectKey);
    writeWorkspaceArticleWorkspaceSelectedObjectKey(
      articleWorkspace,
      nextObjectKey,
    );
    onSelectedObjectChange?.({
      articleWorkspace: activeArticleWorkspace,
      object,
    });
  };

  return (
    <WorkspaceArticleEditorSurface
      actions={viewModel.selectedActions}
      actionsDisabled={actionsDisabled}
      artifactIds={viewModel.selectedArtifactIds}
      compact
      object={viewModel.selectedObject}
      objects={viewModel.objects}
      onActionIntent={onActionIntent}
      onArticleMarkdownChange={onArticleMarkdownChange}
      onOpenPreviewArtifact={onOpenPreviewArtifact}
      onSelectObject={handleSelectObject}
      preview={viewModel.selectedPreview}
      previewArtifact={previewArtifact}
      articleWorkspace={activeArticleWorkspace}
      selectedObjectKey={selectedObjectKeyForRender}
      updatedAt={viewModel.updatedAt}
    />
  );
}

function resolveArticleEditorObject(
  articleWorkspace: WorkspaceArticleWorkspace,
  selectedObjectKey: string | null,
): WorkspaceArticleObject | null {
  const preferredArticleDraft = selectWorkspaceArticleDraftObject(
    articleWorkspace.objects,
  );
  return (
    findDocumentObjectByKey(articleWorkspace, selectedObjectKey) ??
    findDocumentObjectByKey(
      articleWorkspace,
      articleWorkspace.editedDraft?.objectKey,
    ) ??
    preferredArticleDraft ??
    findDocumentObjectByRef(
      articleWorkspace,
      articleWorkspace.primaryObjectRef,
    ) ??
    findDocumentObjectByKey(
      articleWorkspace,
      readWorkspaceArticleWorkspaceSelectedObjectKey(articleWorkspace),
    ) ??
    findDocumentObjectByRef(
      articleWorkspace,
      articleWorkspace.selectedObjectRef,
    ) ??
    findObjectByRef(articleWorkspace, articleWorkspace.primaryObjectRef) ??
    findObjectByKey(
      articleWorkspace,
      readWorkspaceArticleWorkspaceSelectedObjectKey(articleWorkspace),
    ) ??
    findObjectByRef(articleWorkspace, articleWorkspace.selectedObjectRef) ??
    articleWorkspace.objects[0] ??
    null
  );
}

function isArticleEditorDocumentObject(
  object: WorkspaceArticleObject,
): boolean {
  return (
    object.ref.kind === "articleDraft" || object.ref.kind === "videoScript"
  );
}

function findDocumentObjectByKey(
  articleWorkspace: WorkspaceArticleWorkspace,
  objectKey: string | null | undefined,
): WorkspaceArticleObject | null {
  const object = findObjectByKey(articleWorkspace, objectKey);
  return object && isArticleEditorDocumentObject(object) ? object : null;
}

function findDocumentObjectByRef(
  articleWorkspace: WorkspaceArticleWorkspace,
  ref: WorkspaceArticleObjectRef | null | undefined,
): WorkspaceArticleObject | null {
  const object = findObjectByRef(articleWorkspace, ref);
  return object && isArticleEditorDocumentObject(object) ? object : null;
}

function findObjectByKey(
  articleWorkspace: WorkspaceArticleWorkspace,
  objectKey: string | null | undefined,
): WorkspaceArticleObject | null {
  if (!objectKey) {
    return null;
  }
  return (
    articleWorkspace.objects.find(
      (object) => buildWorkspaceArticleObjectKey(object) === objectKey,
    ) ?? null
  );
}

function findObjectByRef(
  articleWorkspace: WorkspaceArticleWorkspace,
  ref: WorkspaceArticleObjectRef | null | undefined,
): WorkspaceArticleObject | null {
  if (!ref) {
    return null;
  }
  return (
    articleWorkspace.objects.find(
      (object) =>
        object.ref.appId === ref.appId &&
        object.ref.sessionId === ref.sessionId &&
        object.ref.kind === ref.kind &&
        object.ref.id === ref.id,
    ) ?? null
  );
}
