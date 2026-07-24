import fs from "node:fs";
import { CONTENT_FACTORY_ARTICLE_WORKSPACE_ASSERTION_KEYS } from "./claw-chat-current-fixture-constants.mjs";

function expectAllToContain(expect, content, fragments) {
  for (const fragment of fragments) expect(content).toContain(fragment);
}

function expectAllNotToContain(expect, content, fragments) {
  for (const fragment of fragments) expect(content).not.toContain(fragment);
}

function readImageCommandScript() {
  return fs.readFileSync(
    "scripts/agent-runtime/claw-chat-current-fixture-image-command.mjs",
    "utf8",
  );
}

function readImageCommandWorkflowReadScript() {
  return fs.readFileSync(
    "scripts/agent-runtime/claw-chat-current-fixture-image-command-workflow-read.mjs",
    "utf8",
  );
}

function readRpcScript() {
  return fs.readFileSync(
    "scripts/agent-runtime/claw-chat-current-fixture-rpc.mjs",
    "utf8",
  );
}

function readContentFactoryArticleWorkspaceScript() {
  return fs.readFileSync(
    "scripts/agent-runtime/claw-chat-current-fixture-content-factory-article-workspace.mjs",
    "utf8",
  );
}

export function registerImageContentSmokeGuards({
  expect,
  it,
  readSmokeScript,
  removeContentFactoryForbiddenMarkerGuard,
}) {
  it("covers Claw @配图 through ImageCommandWorkflow and current task artifact", () => {
    const content = readSmokeScript();
    const imageCommandContent = readImageCommandScript();
    const imageCommandWorkflowReadContent =
      readImageCommandWorkflowReadScript();
    const rpcContent = readRpcScript();

    expectAllToContain(expect, content, [
      "image-command",
      "plain-image-intent",
      "IMAGE_COMMAND_SCENARIO",
      "PLAIN_IMAGE_INTENT_SCENARIO",
      "@配图 E2E 图片命令路由测试，请生成一张青柠插画",
      "画一张广州夏天的图",
      "@配图 ${PLAIN_IMAGE_INTENT_PROMPT}",
      "ensure-fixture-image-provider",
      "image_command_intent",
      "imageCommandLegacySkillLaunchNotSubmitted",
      "image_task",
      "lime_create_image_generation_task",
      "mediaTaskArtifact/image/create",
      "mediaTaskArtifact/get",
      "mediaTaskArtifact/list",
      'APP_SERVER_BACKEND_MODE: "runtime"',
      "media_runtime_worker",
      "lime-image-api-worker",
      ".lime/tasks/image_generate",
      "runImageCommandScenario",
      "waitForImageCommandWorkflowTaskArtifact",
      "isImageIntentScenario",
      "waitForGuiImageCommandCompleted",
      "waitForGuiImageCommandTerminal",
      "waitForSessionReadImageCommandCompleted",
      "waitForImageCommandTaskArtifactTerminal",
      "imageCommandTaskArtifactTerminalPatch",
      "completeMethodUsed",
      "imageCommandTaskArtifactTerminal",
      "imageCommandTaskArtifactAfterReload",
      "imageCommandTaskAuditLog",
      "EXPECTED_IMAGE_TASK_AUDIT_EVENTS",
      "worker_loaded",
      "request_slot_succeeded",
      "task_succeeded",
      "guiImageCommandRestoredAfterReload",
      "agentUiPerformanceTracePreReload",
      "collectAgentUiPerformanceTraceEvidence",
      "image-workbench-message-preview-${taskId}",
      "page.reload",
      "imageCommandTaskArtifact",
      "imageCommandPromptReachedBackend",
      "imageCommandMetadataReachedBackend",
      "imageCommandUsedCurrentMediaTaskArtifactMethods",
      "imageCommandTaskArtifactWritten",
      "imageCommandTaskArtifactSameTaskUpdated",
      "imageCommandTaskAuditLogWritten",
      "imageCommandTaskAuditLogEventSequence",
      "imageCommandTaskAuditLogNoSensitiveTokens",
      "readImageCommandWorkflowAudit",
      "APP_SERVER_METHOD_WORKFLOW_READ",
      "workflow/read",
      "imageCommandWorkflowRead",
      "imageCommandWorkflowAuditReadModelProjected",
      "imageCommandWorkflowAuditStepsProjected",
      "imageCommandWorkflowAuditSummaryRedacted",
      "image-command-run-${turnId}",
      "imageCommandWorkerUsedFixtureProviderAndModel",
      "imageCommandFixtureProvider",
      "bodyIncludesModel",
      "headerProviderId",
      "imageCommandWorkflowToolObserved",
      "imageCommandCreateTaskToolObserved",
      "guiImageCommandToolProcessVisible",
      "guiImageCommandTaskCardVisible",
      "guiImageCommandTaskCardTerminal",
      "guiImageCommandSingleTaskCard",
      "hasLoadedVisiblePreviewImage",
      "guiImageCommandNoDraftCard",
      "guiImageCommandNoTemplateTaskId",
      "readModelImageCommandTaskPreviewObserved",
      "IMAGE_COMMAND_ASSERTION_KEYS",
      "draft-image-",
      "{task_id}",
    ]);
    expectAllToContain(expect, imageCommandContent, [
      "expectedSessionId: options.sessionId",
      "sessionId: options.sessionId",
      "sessionId === expectedSessionId",
      "threadId: options.threadId",
      "includeTurns: true",
    ]);
    expectAllNotToContain(expect, imageCommandContent, [
      "SESSION_ID",
      "THREAD_ID",
    ]);
    expectAllToContain(expect, imageCommandWorkflowReadContent, [
      "sessionId,",
      "sessionId",
    ]);
    expectAllNotToContain(expect, imageCommandWorkflowReadContent, [
      "SESSION_ID",
    ]);
    expectAllToContain(expect, imageCommandContent, [
      "expectedSessionId: options.sessionId",
      'entrySource: "at_image_command"',
      "image_command_workflow",
      "snapshot.hasVisibleImageTaskProcess",
      "hasPresentationIntroInAssistantText",
      "hasPresentationCaptionAfterCard",
      "cardHasPresentationCaption",
      "completionAfterCard",
    ]);
    expectAllToContain(expect, rpcContent, [
      "modelProvider/create",
      "modelProviderKey/create",
      "media_defaults",
    ]);
    expectAllNotToContain(expect, content, [
      "createImageCommandTaskArtifact",
      "completeImageCommandTaskArtifact",
      "completeImageCommandTaskArtifactFile",
      "execute_skill",
      "agent_runtime_submit_turn",
    ]);
    expectAllNotToContain(expect, imageCommandContent, [
      'entrySource: "plain_image_intent"',
      "IMAGE_COMMAND_SKILL_NAME",
      "IMAGE_COMMAND_SKILL_TOOL_CALL_ID",
      "(snapshot.hasAssistantSummary || snapshot.hasDoneText) &&",
      "cardText.includes(presentationCaption) ||",
    ]);
  });

  it("covers content factory Article Workspace through typed artifact write and read", () => {
    const content = readSmokeScript();
    const contentFactoryScenario = readContentFactoryArticleWorkspaceScript();

    expectAllToContain(expect, content, [
      "content-factory-article-workspace",
      "content-factory-inline-image-article-workspace",
      "runContentFactoryArticleWorkspaceScenario",
      "runContentFactoryInlineImageArticleWorkspaceScenario",
      "pluginInstalled/save",
      "turn/start",
      "artifact/write",
      "artifact/read",
      "content_factory.workspace_patch",
      "contentFactoryWorkspacePatch",
      "内容工厂 Article Editor Fixture",
      "公众号文章草稿",
      "配图组",
      "视频分镜",
      "交付检查清单",
      "artifact-article-1",
      "artifact-image-1",
      "artifact-video-storyboard",
      "artifact-delivery-checklist",
      "artifact-image-regenerate-workspace-patch",
      "artifact-image-regenerated",
      "image_regenerate_job_1",
      "worker_dogfood",
      "contentFactoryArticleWorkspaceWorkerTurnStart",
      "contentFactoryArticleWorkspaceWorkerHostGenerationFixture",
      "contentFactoryHostGenerationAgentRuntimeRequest",
      "startContentFactoryHostGenerationFixture",
      "fixture-openai",
      "article-draft-document",
      "contentFactoryArticleWorkspaceWorkerTurnExecuted",
      "contentFactoryArticleWorkspaceWorkerAuditFactsHidden",
      "content.article.generate",
      "options.scenario === CONTENT_FACTORY_ARTICLE_WORKSPACE_SCENARIO",
      "options.scenario !== CONTENT_FACTORY_INLINE_IMAGE_ARTICLE_WORKSPACE_SCENARIO",
      "CONTENT_FACTORY_INLINE_IMAGE_SLOT_ID",
      "contentFactoryInlineImageTaskEventEmitted",
      "contentFactoryInlineImageArticleRestored",
      "mediaTaskArtifact/image/create",
      "mediaTaskArtifact/image/complete",
      "readModel.workerArticleObject?.hostManagedGenerationStatus ===",
      '"completed"',
      "contentFactoryArticleWorkspaceStoryboardObjectSelection",
      "contentFactoryArticleWorkspaceArticleObjectSelection",
      "contentFactoryArticleWorkspaceArticleCanvasSurface",
      "contentFactoryArticleWorkspaceArticleCanvasSurfaceVisible",
      "contentFactoryArticleWorkspaceEditedDraftUpdate",
      "contentFactoryArticleWorkspaceEditedDraftSessionReopened",
      "contentFactoryArticleWorkspaceEditedDraftReload",
      "contentFactoryArticleWorkspaceEditedDraftArtifactFrame",
      "contentFactoryArticleWorkspaceEditedDraftRestored",
      "E2E_EDITED_ARTICLE_DRAFT_RESTORED",
      "workspace-article-editor-related-articleDraft",
      "workspace-article-editor-related-videoStoryboard",
      "workspace-article-editor-title-candidates",
      "workspace-article-editor-research",
      "workspace-article-editor-outline",
      "workspace-article-editor-citations",
      "workspace-article-editor-image-slots",
      "workspace-article-editor-canvas",
      "documentCanvasText.includes",
      "snapshot.hasArticleCanvasContent",
      "readModel.hasImageSetObject",
      "readModel.hasStoryboardObject",
      "readModel.hasChecklistObject",
      "workspace-article-workspace-app-declared-renderer",
      "app_declared",
      "host_placeholder",
      "host_placeholder_only",
      "rendererContract",
      "not_loaded",
      "rendererExecutionModelVisible",
      "entryLoadPolicyVisible",
      "executableHostAbsent",
      "app_declared_renderer_placeholder_only",
      "./renderer/storyboard.tsx",
      "open_storyboard",
      "contentFactoryArticleWorkspaceStoryboardRendererContractPreserved",
      "已重新生成 2 张候选图",
      "workspace-article-editor-surface",
      "workspace-right-surface-host",
      "artifact_document.v1",
      "contentFactoryArticleWorkspaceDoesNotUseModelTurn",
      "contentFactoryArticleWorkspaceActionResultPatchProjected",
    ]);
    expectAllToContain(expect, contentFactoryScenario, [
      "APP_SERVER_METHOD_ARTIFACT_WRITE",
      "writeContentFactoryWorkspacePatchArtifact",
      "writeContentFactoryArticleWorkspaceActionResultArtifact",
      "summarizeArtifactWrite",
      "articleCanvasHasForbiddenTemplate",
      "reloadContentFactoryArticleWorkspaceSession",
      "reloadRendererDocument",
      "updateContentFactoryArticleWorkspaceEditedDraft",
      "waitForContentFactoryArticleWorkspaceEditedDraftRestored",
      "readContentFactoryArticleDraftObjectRef",
      "article-artifact-frame",
      "clickContentFactoryArticleArtifactFrame",
      "waitForContentFactoryArticleEditorOpened",
      "FORBIDDEN_CONTENT_FACTORY_ARTICLE_TEMPLATE_MARKERS",
      "metadataPanelsHidden",
      "snapshot.metadataPanelsHidden",
      "snapshot.hasFullArticleCanvas",
    ]);

    const contentWithoutForbiddenMarkerGuard =
      removeContentFactoryForbiddenMarkerGuard(content);
    expectAllNotToContain(expect, contentWithoutForbiddenMarkerGuard, [
      "受控宿主生成标题",
      "内容工厂插件化写作：让文章生产可审计",
    ]);
    expectAllNotToContain(expect, contentFactoryScenario, [
      "agentSession/runtimeEvents/append",
      "APP_SERVER_METHOD_AGENT_SESSION_RUNTIME_EVENTS_APPEND",
      'type: "workflow.run.started"',
      'type: "runtime.error"',
      'toggleTestId: "task-center-object-canvas-toggle"',
      "researchText.includes",
      "takeawaysText.length",
      "writingPlanText.length",
    ]);
    for (const assertionKey of CONTENT_FACTORY_ARTICLE_WORKSPACE_ASSERTION_KEYS) {
      expect(content).toContain(assertionKey);
    }
    expectAllNotToContain(expect, content, [
      'readModel.workerArticleObject?.hostManagedGenerationStatus ===\n        "unavailable"',
      "APP_SERVER_METHOD_CONTENT_FACTORY",
      "content_factory/start",
      "content_factory/generate",
      "BrowserView",
    ]);
  });
}
