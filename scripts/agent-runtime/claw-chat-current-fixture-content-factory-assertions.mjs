import {
  APP_SERVER_METHOD_PLUGIN_INSTALLED_SAVE,
  APP_SERVER_METHOD_ARTIFACT_READ,
  APP_SERVER_METHOD_ARTIFACT_WRITE,
  APP_SERVER_METHOD_SESSION_UPDATE,
  APP_SERVER_METHOD_SESSION_TURN_START,
  APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
  CONTENT_FACTORY_ARTICLE_WORKSPACE_ARTICLE_ARTIFACT_ID,
  CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_TITLE,
} from "./claw-chat-current-fixture-constants.mjs";

export function buildContentFactoryArticleWorkspaceScenarioAssertions({
  appServerRequestMethods,
  backendLedger,
  pageText,
  summary,
}) {
  const gui = summary.contentFactoryArticleWorkspaceGui ?? {};
  const readModel = summary.contentFactoryArticleWorkspaceReadModel ?? {};
  const artifactRead = summary.contentFactoryArticleWorkspaceArtifactRead ?? {};
  const identity =
    summary.contentFactoryArticleWorkspaceSessionCreation?.identity ?? {};
  const storyboardRendererContract =
    readModel.storyboardArtifact?.rendererContract ?? {};
  return {
    contentFactoryArticleWorkspaceArtifactWritePersisted:
      appServerRequestMethods.includes(APP_SERVER_METHOD_ARTIFACT_WRITE) &&
      summary.contentFactoryArticleWorkspaceArtifactWrite?.threadId ===
        identity.threadId &&
      summary.contentFactoryArticleWorkspaceArtifactWrite?.artifactRef ===
        "artifact-workspace-patch-1" &&
      Boolean(summary.contentFactoryArticleWorkspaceArtifactWrite?.eventId) &&
      summary.contentFactoryArticleWorkspaceArtifactWrite?.sequence > 0 &&
      summary.contentFactoryArticleWorkspaceArtifactWrite?.contentStatus ===
        "available" &&
      Boolean(
        summary.contentFactoryArticleWorkspaceArtifactWrite
          ?.sidecarRelativePath,
      ),
    contentFactoryArticleWorkspaceRightSurfaceRequested:
      appServerRequestMethods.includes(
        APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
      ) &&
      summary.contentFactoryArticleWorkspaceRightSurfaceRequest?.surfaceKind ===
        "articleWorkspace" &&
      summary.contentFactoryArticleWorkspaceRightSurfaceRequest?.origin ===
        "runtime" &&
      summary.contentFactoryArticleWorkspaceRightSurfaceRequest?.status ===
        "pending",
    contentFactoryArticleWorkspaceSessionOpenedFromSidebar:
      summary.contentFactoryArticleWorkspaceSessionCreation?.sessionId ===
        identity.sessionId &&
      summary.guiContentFactoryArticleWorkspaceSessionVisible
        ?.hasSessionTitle === true &&
      summary.guiContentFactoryArticleWorkspaceSessionOpened?.readModel
        ?.sessionId === identity.sessionId &&
      pageText.includes(CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_TITLE),
    contentFactoryArticleWorkspaceRightSurfaceVisible:
      summary.contentFactoryArticleWorkspaceRightSurface?.activeSurface ===
        "articleWorkspace" &&
      summary.contentFactoryArticleWorkspaceRightSurface?.rootVisible ===
        true &&
      gui.activeSurface === "articleWorkspace" &&
      gui.rootVisible === true,
    contentFactoryArticleWorkspaceFinalArticleFrameVisible:
      summary.contentFactoryArticleWorkspaceArtifactFrame?.visible === true &&
      summary.contentFactoryArticleWorkspaceArtifactFrame
        ?.hasArticlePreviewContent === true &&
      gui.hasArticleDraftObject === true &&
      (gui.hasArticleCanvasContent === true ||
        gui.hasFixtureOnlyArticleHidden === true),
    contentFactoryArticleWorkspacePageShowsObjects:
      gui.hasArticleEditorTitle === true &&
      gui.hasArticleDraftObject === true &&
      (gui.hasArticleCanvasContent === true ||
        gui.hasFixtureOnlyArticleHidden === true) &&
      readModel.hasImageSetObject === true &&
      readModel.hasStoryboardObject === true &&
      readModel.hasChecklistObject === true,
    contentFactoryArticleWorkspaceReadModelProjected:
      readModel.hasArticleWorkspace === true &&
      readModel.appId === "content-factory-app" &&
      readModel.sessionId === identity.sessionId &&
      readModel.objectCount >= 2 &&
      readModel.hasArticleObject === true &&
      readModel.hasImageSetObject === true &&
      readModel.hasStoryboardObject === true &&
      readModel.hasChecklistObject === true,
    contentFactoryArticleWorkspaceArtifactsProjected:
      readModel.articleArtifact?.artifactRef ===
        CONTENT_FACTORY_ARTICLE_WORKSPACE_ARTICLE_ARTIFACT_ID &&
      readModel.articleArtifact?.kind === "artifact_document" &&
      readModel.articleArtifact?.artifactSchema === "artifact_document.v1" &&
      readModel.articleArtifact?.artifactDocumentId ===
        "artifact-document:content-factory-app:artifact-article-1" &&
      readModel.articleArtifact?.articleWorkspaceObjectKind === "articleDraft",
    contentFactoryArticleWorkspaceRendererArtifactsProjected:
      readModel.storyboardArtifact?.artifactRef ===
        "artifact-video-storyboard" &&
      readModel.storyboardArtifact?.kind === "artifact_document" &&
      readModel.storyboardArtifact?.surfaceKind === "storyboard" &&
      readModel.storyboardArtifact?.articleWorkspaceObjectKind ===
        "videoStoryboard" &&
      readModel.storyboardArtifact?.articleWorkspaceSurfaceKind ===
        "storyboard" &&
      readModel.checklistArtifact?.artifactRef ===
        "artifact-delivery-checklist" &&
      readModel.checklistArtifact?.kind === "artifact_document" &&
      readModel.checklistArtifact?.surfaceKind === "checklist" &&
      readModel.checklistArtifact?.articleWorkspaceObjectKind ===
        "deliveryChecklist" &&
      readModel.checklistArtifact?.articleWorkspaceSurfaceKind === "checklist",
    contentFactoryArticleWorkspaceArtifactReadContent:
      appServerRequestMethods.includes(APP_SERVER_METHOD_ARTIFACT_READ) &&
      artifactRead.artifactRef ===
        (readModel.workerArticleObject?.previewArtifactId ||
          CONTENT_FACTORY_ARTICLE_WORKSPACE_ARTICLE_ARTIFACT_ID) &&
      artifactRead.kind === "artifact_document" &&
      artifactRead.contentStatus === "available" &&
      artifactRead.contentIncludesSchema === true &&
      artifactRead.contentIncludesDocumentId === true &&
      artifactRead.documentObjectKind === "articleDraft" &&
      artifactRead.documentBlockCount >= 1 &&
      artifactRead.documentRichTextLength > 160 &&
      artifactRead.contentIncludesArticleTitle === true &&
      artifactRead.richTextHasForbiddenTemplate !== true &&
      artifactRead.contentIncludesWorkerArticle === true,
    contentFactoryArticleWorkspaceArticleCanvasSurfaceVisible:
      summary.contentFactoryArticleWorkspaceArticleObjectSelection?.selected ===
        true &&
      summary.contentFactoryArticleWorkspaceArticleObjectSelection
        ?.objectKind === "articleDraft" &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.rootVisible === true &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.documentCanvasVisible === true &&
      (summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.hasDocumentPreview === true ||
        summary.contentFactoryArticleWorkspaceArticleCanvasSurface
          ?.fixtureOnlyArticleHidden === true) &&
      (summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.hasFullArticleCanvas === true ||
        summary.contentFactoryArticleWorkspaceArticleCanvasSurface
          ?.fixtureOnlyArticleHidden === true) &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.articleCanvasHasForbiddenTemplate !== true &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.metadataPanelsHidden === true &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.structurePresent === false &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.researchPresent === false &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.outlinePresent === false &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.citationsPresent === false &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.imageSlotsPresent === false &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.takeawaysPresent === false &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.writingPlanPresent === false &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.reviewPresent === false &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.workflowUiRailHidden === true &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.contentFactoryOrchestrationVisible === false &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.contentFactoryOrchestrationStepCount === 0 &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.hasVisibleContentFactoryOrchestration === false &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.hasVisibleSubagents === false &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.hasVisibleSkillRef === false &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.hasVisibleConnectors === false &&
      summary.contentFactoryArticleWorkspaceArticleCanvasSurface
        ?.hasVisibleHooks === false,
    contentFactoryArticleWorkspaceEditedDraftRestored:
      appServerRequestMethods.includes(APP_SERVER_METHOD_SESSION_UPDATE) &&
      summary.contentFactoryArticleWorkspaceEditedDraftUpdate?.sessionId ===
        identity.sessionId &&
      summary.contentFactoryArticleWorkspaceEditedDraftUpdate?.objectRef
        ?.kind === "articleDraft" &&
      summary.contentFactoryArticleWorkspaceEditedDraftUpdate
        ?.markdownMarker === "E2E_EDITED_ARTICLE_DRAFT_RESTORED" &&
      summary.contentFactoryArticleWorkspaceEditedDraftReload?.renderer
        ?.supportsAppServer === true &&
      summary.contentFactoryArticleWorkspaceEditedDraftReload?.sessionVisible
        ?.hasSessionTitle === true &&
      summary.contentFactoryArticleWorkspaceEditedDraftSessionReopened
        ?.readModel?.sessionId === identity.sessionId &&
      summary.contentFactoryArticleWorkspaceEditedDraftArtifactFrame
        ?.visible === true &&
      summary.contentFactoryArticleWorkspaceEditedDraftArtifactFrame
        ?.hasEditedDraftMarker === true &&
      summary.contentFactoryArticleWorkspaceEditedDraftRestored
        ?.canvasVisible === true &&
      summary.contentFactoryArticleWorkspaceEditedDraftRestored
        ?.markerVisibleInCanvas === true &&
      summary.contentFactoryArticleWorkspaceEditedDraftRestored
        ?.hasEditedTitle === true &&
      readModel.editedDraft?.markdownIncludesEditedDraftMarker === true &&
      readModel.editedDraft?.objectRef?.kind === "articleDraft" &&
      readModel.workerArticleObject?.markdownIncludesEditedDraftMarker ===
        true &&
      readModel.workerArticleObject?.sourceEdited === true,
    contentFactoryArticleWorkspaceWorkerTurnExecuted:
      appServerRequestMethods.includes(
        APP_SERVER_METHOD_PLUGIN_INSTALLED_SAVE,
      ) &&
      appServerRequestMethods.includes(APP_SERVER_METHOD_SESSION_TURN_START) &&
      summary.contentFactoryArticleWorkspaceInstalledStateSave?.appId ===
        "content-factory-app" &&
      summary.contentFactoryArticleWorkspaceWorkerTurnStart?.turnStatus ===
        "inProgress" &&
      summary.contentFactoryArticleWorkspaceWorkerTurnStart?.taskId ===
        identity.workerTaskId &&
      summary.contentFactoryArticleWorkspaceWorkerTurnStart?.readModel
        ?.workerTurnStatus === "completed" &&
      summary.contentFactoryArticleWorkspaceWorkerTurnStart?.readModel
        ?.workerTurnId === identity.workerTurnId &&
      readModel.workerDogfoodEvidence?.taskId === identity.workerTaskId &&
      readModel.workerDogfoodEvidence?.taskKind ===
        "content.article.generate" &&
      readModel.workerDogfoodEvidence?.status === "completed" &&
      readModel.workerDogfoodEvidence?.artifactKind ===
        "content_factory.workspace_patch" &&
      readModel.workerArticleObject?.sourceTaskId === identity.workerTaskId &&
      readModel.workerArticleObject?.markdownIncludesResearch === true &&
      readModel.workerArticleObject?.markdownIncludesDraft === true &&
      readModel.workerArticleObject?.hostManagedGenerationStatus ===
        "completed" &&
      !readModel.workerArticleObject?.hostManagedGenerationReasonCode &&
      readModel.workerArticleObject?.hostManagedGenerationOutputIds?.includes(
        "article-draft-document",
      ) === true &&
      summary.contentFactoryArticleWorkspaceWorkerTurnStart
        ?.hostGenerationFixture?.requestCount >= 1 &&
      readModel.workerArticleObject?.researchRoundCount >= 3 &&
      readModel.workerArticleObject?.imageSlotCount >= 3,
    contentFactoryArticleWorkspaceWorkerAuditFactsHidden:
      !readModel.workerDogfoodEvidence?.workflowKey &&
      (readModel.workerDogfoodEvidence?.subagents?.length ?? 0) === 0 &&
      (readModel.workerDogfoodEvidence?.skillRefs?.length ?? 0) === 0 &&
      (readModel.workerDogfoodEvidence?.cliRefs?.length ?? 0) === 0 &&
      (readModel.workerDogfoodEvidence?.connectorRefs?.length ?? 0) === 0 &&
      (readModel.workerDogfoodEvidence?.hookRefs?.length ?? 0) === 0 &&
      (readModel.workerDogfoodEvidence?.orchestrationStepCount ?? 0) === 0,
    contentFactoryArticleWorkspaceActionResultPatchProjected:
      summary.contentFactoryArticleWorkspaceActionResultArtifactWrite
        ?.threadId === identity.threadId &&
      summary.contentFactoryArticleWorkspaceActionResultArtifactWrite
        ?.artifactRef === "artifact-image-regenerate-workspace-patch" &&
      Boolean(
        summary.contentFactoryArticleWorkspaceActionResultArtifactWrite
          ?.eventId,
      ) &&
      readModel.completedActionWorkerEvidence?.taskId ===
        "image_regenerate_job_1" &&
      readModel.completedActionWorkerEvidence?.status === "completed" &&
      readModel.actionResultArtifacts?.some(
        (artifact) =>
          artifact.artifactRef === "artifact-image-regenerated" &&
          artifact.kind === "artifact_document",
      ) === true &&
      readModel.actionResultArtifacts?.some(
        (artifact) =>
          artifact.artifactRef ===
            "artifact-image-regenerate-workspace-patch" &&
          artifact.kind === "content_factory.workspace_patch",
      ) === true,
    contentFactoryArticleWorkspaceStoryboardRendererContractPreserved:
      summary.contentFactoryArticleWorkspaceStoryboardObjectSelection
        ?.objectKind === "videoStoryboard" &&
      (summary.contentFactoryArticleWorkspaceStoryboardObjectSelection
        ?.selected === true ||
        summary.contentFactoryArticleWorkspaceStoryboardObjectSelection
          ?.candidatePresent === true) &&
      storyboardRendererContract.pluginId === "content-factory-app" &&
      storyboardRendererContract.rendererKind === "app_declared" &&
      storyboardRendererContract.executionMode === "host_placeholder" &&
      storyboardRendererContract.reasonCode ===
        "app_declared_renderer_placeholder_only" &&
      storyboardRendererContract.entry === "./renderer/storyboard.tsx" &&
      storyboardRendererContract.allowedOutputArtifactKinds?.includes(
        "content_factory.workspace_patch",
      ) === true,
    contentFactoryArticleWorkspaceDoesNotUseModelTurn: backendLedger.every(
      (entry) => entry.kind !== "turnStart",
    ),
  };
}
