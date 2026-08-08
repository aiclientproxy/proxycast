import {
  DEFAULT_FIXTURE_API_KEY,
  DEFAULT_FIXTURE_MODEL,
  startOpenAiCompatibleFixtureServer,
} from "../lib/openai-compatible-fixture-server.mjs";
import {
  APP_SERVER_METHOD_ARTIFACT_READ,
  APP_SERVER_METHOD_ARTIFACT_WRITE,
  APP_SERVER_METHOD_SESSION_READ,
  APP_SERVER_METHOD_SESSION_START,
  APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
  CONTENT_FACTORY_ARTICLE_WORKSPACE_ARTICLE_ARTIFACT_ID,
  CONTENT_FACTORY_ARTICLE_WORKSPACE_IMAGE_ARTIFACT_ID,
  CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_TITLE,
} from "./claw-chat-current-fixture-constants.mjs";
import {
  buildContentFactoryActionResultWorkspacePatch,
  buildWorkspacePatch,
} from "./claw-chat-current-fixture-content-factory-workspace-patches.mjs";
import {
  ensureFixtureTextProvider,
  invokeAppServerFromPage,
  reloadRendererDocument,
  waitForRendererReady,
} from "./claw-chat-current-fixture-rpc.mjs";
import {
  openSessionFromSidebar,
  waitForGuiSessionVisible,
} from "./claw-chat-current-fixture-session.mjs";
import {
  assert,
  sanitizeJson,
  sleep,
} from "./claw-chat-current-fixture-utils.mjs";

const CONTENT_FACTORY_APP_ID = "content-factory-app";
const IMAGE_SET_OBJECT_ID = "image-set-1";
const CONTENT_FACTORY_ARTICLE_WORKSPACE_STORYBOARD_ARTIFACT_ID =
  "artifact-video-storyboard";
const CONTENT_FACTORY_ARTICLE_WORKSPACE_CHECKLIST_ARTIFACT_ID =
  "artifact-delivery-checklist";
const CONTENT_FACTORY_ARTICLE_WORKSPACE_EDITED_DRAFT_MARKER =
  "E2E_EDITED_ARTICLE_DRAFT_RESTORED";
const FORBIDDEN_CONTENT_FACTORY_ARTICLE_TEMPLATE_MARKERS = [
  "受控宿主生成标题",
  "内容工厂插件化写作：让文章生产可审计",
  "从基础语法到工程实战",
  "## 请求摘要",
  "## 资料检索",
  "## 交付检查",
  "targetObjectKind",
  "outputField",
];
const CONTENT_FACTORY_ARTICLE_WORKSPACE_EDITED_DRAFT_MARKDOWN = [
  "# 内容工厂首版文章 - 编辑后恢复稿",
  "",
  "## 编辑恢复验证",
  "",
  `这是一段来自 Article Editor 画布编辑后的唯一正文标记：${CONTENT_FACTORY_ARTICLE_WORKSPACE_EDITED_DRAFT_MARKER}。`,
  "",
  "## 三轮资料检索",
  "",
  "- 第一轮：确认写作主题与读者预期。",
  "- 第二轮：补齐案例、引用与反方风险。",
  "- 第三轮：把资料收束成可发布正文。",
  "",
  "## 正文草稿",
  "",
  "编辑后的正文必须覆盖 artifact 首稿，并且刷新或重新打开会话后仍回到 Article Editor 画布。",
].join("\n");
const CONTENT_FACTORY_ARTICLE_WORKSPACE_EDITED_DRAFT_UPDATED_AT =
  "2026-06-29T10:00:00.000Z";

export async function runContentFactoryArticleWorkspaceScenario({
  page,
  options,
  workspace,
  appServerRequests,
}) {
  const providerFixture = await startOpenAiCompatibleFixtureServer({
    content: "Article Workspace fixture provider ready.",
  });
  try {
    const fixtureProvider = await ensureFixtureTextProvider(
      page,
      appServerRequests,
      {
        apiHost: providerFixture.baseUrl,
        apiKey: DEFAULT_FIXTURE_API_KEY,
        modelId: DEFAULT_FIXTURE_MODEL,
      },
    );
    const sessionCreation = await createContentFactoryArticleWorkspaceSession(
      page,
      workspace,
      appServerRequests,
      fixtureProvider,
    );
    const identity = createContentFactoryScenarioIdentity(
      sessionCreation.identity,
    );
    const artifactWrite = await writeWorkspacePatchArtifact(
      page,
      workspace,
      appServerRequests,
      identity,
    );
    const actionResultArtifactWrite =
      await writeContentFactoryArticleWorkspaceActionResultArtifact(
        page,
        workspace,
        appServerRequests,
        identity,
      );
    const rightSurfaceRequest =
      await requestContentFactoryArticleWorkspaceSurface(
        page,
        workspace,
        appServerRequests,
        identity,
      );

    await notifySessionChanged(page, workspace.workspaceId, identity.sessionId);

    const guiSessionVisible = await waitForGuiSessionVisible(
      page,
      options,
      CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_TITLE,
    );
    const guiSessionOpened = await openSessionFromSidebar(
      page,
      options,
      appServerRequests,
      {
        sessionId: identity.sessionId,
        threadId: identity.threadId,
        title: CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_TITLE,
      },
    );

    const articleObjectRef = await readContentFactoryArticleDraftObjectRef(
      page,
      options,
      appServerRequests,
      identity,
    );
    const workerArticleArtifactRef = Array.isArray(articleObjectRef.artifactIds)
      ? articleObjectRef.artifactIds.find(
          (value) => typeof value === "string" && value.trim(),
        )
      : null;
    assert(
      workerArticleArtifactRef,
      "内容工厂 Article Editor 缺少 worker articleDraft artifact ref",
    );
    const articleArtifactFrame = await clickContentFactoryArticleArtifactFrame(
      page,
      options,
      workerArticleArtifactRef,
    );
    const rightSurface = await waitForContentFactoryArticleEditorOpened(
      page,
      options,
    );
    const articleObjectSelection = await selectContentFactoryArticleObject(
      page,
      options,
    );
    const articleCanvasSurface =
      await waitForContentFactoryArticleCanvasSurface(page, options);
    const articleEditedDraftUpdate =
      await updateContentFactoryArticleWorkspaceEditedDraft(
        page,
        options,
        appServerRequests,
        identity,
      );
    const articleEditedDraftReload =
      await reloadContentFactoryArticleWorkspaceSession(
        page,
        options,
        workspace,
        identity,
      );
    const articleEditedDraftSessionReopened = await openSessionFromSidebar(
      page,
      options,
      appServerRequests,
      {
        sessionId: identity.sessionId,
        threadId: identity.threadId,
        title: CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_TITLE,
      },
    );
    const articleEditedDraftArtifactFrame =
      await clickContentFactoryArticleArtifactFrame(
        page,
        options,
        workerArticleArtifactRef,
      );
    const articleEditedDraftRestored =
      await waitForContentFactoryArticleWorkspaceEditedDraftRestored(
        page,
        options,
      );
    const storyboardObjectSelection =
      await selectContentFactoryStoryboardObject(page, options);
    const gui = await waitForContentFactoryArticleWorkspaceGui(page, options);

    const readModel = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      {
        threadId: identity.threadId,
        includeTurns: true,
      },
      appServerRequests,
    );
    const readModelSummary = summarizeContentFactoryArticleWorkspaceReadModel(
      readModel.result,
      identity,
    );
    const readModelArticleArtifactRef =
      readModelSummary.workerArticleObject?.previewArtifactId ||
      CONTENT_FACTORY_ARTICLE_WORKSPACE_ARTICLE_ARTIFACT_ID;
    assert(
      readModelArticleArtifactRef === workerArticleArtifactRef,
      `内容工厂 Article Editor artifact ref 漂移: expected=${workerArticleArtifactRef} actual=${readModelArticleArtifactRef}`,
    );

    const artifactRead = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_ARTIFACT_READ,
      {
        sessionId: identity.sessionId,
        artifactRef: workerArticleArtifactRef,
        includeContent: true,
        limit: 1,
      },
      appServerRequests,
    );
    const artifactReadSummary = summarizeContentFactoryArtifactRead(
      artifactRead.result,
    );
    return sanitizeJson({
      contentFactoryArticleWorkspaceSessionCreation: {
        ...sessionCreation,
        identity,
      },
      contentFactoryArticleWorkspaceArtifactWrite: summarizeArtifactWrite(
        artifactWrite.result,
      ),
      contentFactoryArticleWorkspaceProviderFixture: {
        requestCount: providerFixture.requests.length,
      },
      contentFactoryArticleWorkspaceActionResultArtifactWrite:
        summarizeArtifactWrite(actionResultArtifactWrite.result),
      contentFactoryArticleWorkspaceRightSurfaceRequest:
        summarizeRightSurfaceRequest(rightSurfaceRequest.result),
      guiContentFactoryArticleWorkspaceSessionVisible: guiSessionVisible,
      guiContentFactoryArticleWorkspaceSessionOpened: guiSessionOpened,
      contentFactoryArticleWorkspaceArticleArtifactIdentity: {
        artifactRef: workerArticleArtifactRef,
        objectRef: articleObjectRef,
        readModelArtifactRef: readModelArticleArtifactRef,
      },
      contentFactoryArticleWorkspaceArtifactFrame: articleArtifactFrame,
      contentFactoryArticleWorkspaceRightSurface: rightSurface,
      contentFactoryArticleWorkspaceArticleObjectSelection:
        articleObjectSelection,
      contentFactoryArticleWorkspaceArticleCanvasSurface: articleCanvasSurface,
      contentFactoryArticleWorkspaceEditedDraftUpdate: articleEditedDraftUpdate,
      contentFactoryArticleWorkspaceEditedDraftReload: articleEditedDraftReload,
      contentFactoryArticleWorkspaceEditedDraftSessionReopened:
        articleEditedDraftSessionReopened,
      contentFactoryArticleWorkspaceEditedDraftArtifactFrame:
        articleEditedDraftArtifactFrame,
      contentFactoryArticleWorkspaceEditedDraftRestored:
        articleEditedDraftRestored,
      contentFactoryArticleWorkspaceStoryboardObjectSelection:
        storyboardObjectSelection,
      contentFactoryArticleWorkspaceGui: gui,
      contentFactoryArticleWorkspaceReadModel: readModelSummary,
      contentFactoryArticleWorkspaceArtifactRead: artifactReadSummary,
    });
  } finally {
    await providerFixture.close();
  }
}

async function reloadContentFactoryArticleWorkspaceSession(
  page,
  options,
  workspace,
  identity,
) {
  const reload = await reloadRendererDocument(page, options);
  const renderer = await waitForRendererReady(page, options);
  await notifySessionChanged(page, workspace.workspaceId, identity.sessionId);
  const sessionVisible = await waitForGuiSessionVisible(
    page,
    options,
    CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_TITLE,
  );
  return sanitizeJson({
    reload,
    renderer,
    sessionVisible,
  });
}

async function updateContentFactoryArticleWorkspaceEditedDraft(
  page,
  options,
  requestLog,
  identity,
) {
  const objectRef = await readContentFactoryArticleDraftObjectRef(
    page,
    options,
    requestLog,
    identity,
  );
  const editedDraft = {
    objectKey: `${objectRef.appId}:${objectRef.sessionId}:${objectRef.kind}:${objectRef.id}`,
    objectRef,
    markdown: CONTENT_FACTORY_ARTICLE_WORKSPACE_EDITED_DRAFT_MARKDOWN,
    updatedAt: CONTENT_FACTORY_ARTICLE_WORKSPACE_EDITED_DRAFT_UPDATED_AT,
  };
  const artifactRef =
    objectRef.artifactIds?.find(
      (value) => typeof value === "string" && value.trim(),
    ) || editedDraft.objectKey;
  const workspacePatch = {
    objects: [
      {
        ref: objectRef,
        title: "Article Editor edited draft",
        status: "draft",
        source: {
          documentText: editedDraft.markdown,
          finalMarkdown: editedDraft.markdown,
          updatedAt: editedDraft.updatedAt,
        },
      },
    ],
    updatedAt: editedDraft.updatedAt,
  };
  const response = await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_ARTIFACT_WRITE,
    {
      threadId: identity.threadId,
      artifact: {
        artifactRef,
        artifactDocumentId: objectRef.id,
        title: "Article Editor edited draft",
        kind: "article_workspace_draft",
        status: "draft",
        content: editedDraft.markdown,
        metadata: {
          workspacePatch,
          editedDraft,
        },
      },
    },
    requestLog,
  );

  return sanitizeJson({
    sessionId: identity.sessionId,
    threadId: response.result?.threadId ?? identity.threadId,
    eventId: response.result?.eventId ?? null,
    objectKey: editedDraft.objectKey,
    objectRef,
    markdownMarker: CONTENT_FACTORY_ARTICLE_WORKSPACE_EDITED_DRAFT_MARKER,
    updatedAt: editedDraft.updatedAt,
  });
}

async function readContentFactoryArticleDraftObjectRef(
  page,
  options,
  requestLog,
  identity,
) {
  const startedAt = Date.now();
  let lastSummary = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const readModel = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      {
        threadId: identity.threadId,
        includeTurns: true,
      },
      requestLog,
    );
    const summary = summarizeContentFactoryArticleWorkspaceReadModel(
      readModel.result,
      identity,
    );
    lastSummary = summary;
    const objectRef = summary.workerArticleObject?.objectRef;
    if (
      objectRef?.appId &&
      objectRef?.sessionId &&
      objectRef?.kind === "articleDraft" &&
      objectRef?.id
    ) {
      return objectRef;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `内容工厂 Article Editor 未找到可编辑文章对象: ${JSON.stringify(
      sanitizeJson(lastSummary),
    )}`,
  );
}

async function waitForContentFactoryArticleWorkspaceEditedDraftRestored(
  page,
  options,
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await page.evaluate(
      ({ marker }) => {
        const canvases = Array.from(
          document.querySelectorAll(
            '[data-testid="workspace-article-editor-canvas"]',
          ),
        );
        const canvas = canvases.find(isVisible) ?? null;
        const canvasContents = Array.from(
          document.querySelectorAll(
            '[data-testid="workspace-article-editor-canvas-content"]',
          ),
        );
        const canvasContent = canvasContents.find(isVisible) ?? null;
        const status = document.querySelector(
          '[data-testid="workspace-article-editor-canvas-status"]',
        );
        const bodyText = document.body?.innerText || "";
        const canvasText =
          canvasContent?.textContent || canvas?.textContent || "";
        return {
          canvasVisible: isVisible(canvas),
          canvasDirty: canvas?.getAttribute("data-dirty") ?? null,
          canvasText: canvasText.slice(0, 1600),
          statusText: status?.textContent ?? "",
          markerVisibleInCanvas: canvasText.includes(marker),
          markerVisibleInBody: bodyText.includes(marker),
          hasEditedTitle:
            canvasText.includes("内容工厂首版文章 - 编辑后恢复稿"),
          hasWorkerMetadataStillVisible:
            bodyText.includes("标题候选") &&
            bodyText.includes("关键观点") &&
            bodyText.includes("配图"),
        };

        function isVisible(node) {
          const rect = node?.getBoundingClientRect();
          const style = node ? window.getComputedStyle(node) : null;
          return Boolean(
            node &&
            rect &&
            rect.width > 8 &&
            rect.height > 8 &&
            style?.display !== "none" &&
            style?.visibility !== "hidden",
          );
        }
      },
      { marker: CONTENT_FACTORY_ARTICLE_WORKSPACE_EDITED_DRAFT_MARKER },
    );
    lastSnapshot = snapshot;
    if (
      snapshot?.canvasVisible &&
      snapshot.markerVisibleInCanvas &&
      snapshot.hasEditedTitle
    ) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `内容工厂 Article Editor 编辑草稿未从会话恢复: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function clickContentFactoryArticleArtifactFrame(
  page,
  options,
  articleArtifactRef,
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await page.evaluate((artifactRef) => {
      const frames = Array.from(
        document.querySelectorAll(
          '[data-testid="article-artifact-frame"], [data-frame-kind="articleArtifacts"]',
        ),
      );
      const targetFrames = frames.filter(
        (candidate) =>
          candidate.getAttribute("data-artifact-ref") === artifactRef,
      );
      const frame = targetFrames.find(isVisible) ?? targetFrames[0] ?? null;
      const buttons = Array.from(frame?.querySelectorAll("button") ?? []);
      const button =
        buttons.find((candidate) => {
          const text = candidate.textContent?.replace(/\s+/g, " ").trim() ?? "";
          return text.includes("展开右侧编辑器");
        }) ??
        buttons.find((candidate) => {
          const text = candidate.textContent?.replace(/\s+/g, " ").trim() ?? "";
          return text.includes("打开编辑器");
        }) ??
        buttons.at(-1) ??
        buttons[0] ??
        null;
      const bodyText = document.body?.innerText || "";
      const frameText = frame?.textContent ?? "";
      const rect = frame?.getBoundingClientRect();
      const style = frame ? window.getComputedStyle(frame) : null;
      const visible = Boolean(
        frame &&
        rect &&
        rect.width > 8 &&
        rect.height > 8 &&
        style?.display !== "none" &&
        style?.visibility !== "hidden",
      );
      const buttonDisabled =
        button instanceof HTMLButtonElement
          ? button.disabled || button.getAttribute("aria-disabled") === "true"
          : null;
      if (visible && button && buttonDisabled === false) {
        button.click();
      }
      return {
        visible,
        frameCount: frames.length,
        visibleFrameCount: frames.filter(isVisible).length,
        candidateArtifactIds: frames.map((candidate) =>
          candidate.getAttribute("data-artifact-id"),
        ),
        candidateArtifactRefs: frames.map((candidate) =>
          candidate.getAttribute("data-artifact-ref"),
        ),
        targetArtifactRef: artifactRef,
        targetFrameCount: targetFrames.length,
        selectedArtifactId: frame?.getAttribute("data-artifact-id") ?? null,
        selectedArtifactRef: frame?.getAttribute("data-artifact-ref") ?? null,
        buttonPresent: Boolean(button),
        buttonText: button?.textContent?.replace(/\s+/g, " ").trim() ?? "",
        buttonDisabled,
        frameText: frameText.slice(0, 500),
        hasArticleTitle:
          bodyText.includes("公众号文章草稿") ||
          frameText.includes("articleDraft") ||
          frameText.length > 200,
        hasArticleDraftObject:
          frameText.includes("articleDraft") || frameText.length > 200,
        hasArticlePreviewContent:
          frameText.length > 200 &&
          !frameText.includes("等待正文产物") &&
          !frameText.includes("文章正文生成后会出现在这里"),
        hasLegacyWorkerTemplateText:
          bodyText.includes("三轮资料检索") ||
          frameText.includes("先把目标定清楚") ||
          frameText.includes("学习路线"),
        hasArticlePreviewBodyContent:
          frameText.length > 200 && frameText.includes("打开文档"),
        hasEditedDraftMarker: bodyText.includes(
          "E2E_EDITED_ARTICLE_DRAFT_RESTORED",
        ),
        rect: rect ? rectToJson(rect) : null,
      };

      function rectToJson(rect) {
        return {
          x: Math.round(rect.x),
          y: Math.round(rect.y),
          width: Math.round(rect.width),
          height: Math.round(rect.height),
          top: Math.round(rect.top),
          left: Math.round(rect.left),
          right: Math.round(rect.right),
          bottom: Math.round(rect.bottom),
        };
      }

      function isVisible(node) {
        const rect = node?.getBoundingClientRect();
        const style = node ? window.getComputedStyle(node) : null;
        return Boolean(
          node &&
          rect &&
          rect.width > 8 &&
          rect.height > 8 &&
          style?.display !== "none" &&
          style?.visibility !== "hidden",
        );
      }
    }, articleArtifactRef);
    lastSnapshot = snapshot;
    if (
      snapshot?.visible &&
      snapshot.buttonPresent &&
      snapshot.buttonDisabled === false
    ) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `内容工厂 ArticleArtifactFrame 不可点击: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function waitForContentFactoryArticleEditorOpened(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await page.evaluate(() => {
      const host = document.querySelector(
        '[data-testid="workspace-right-surface-host"]',
      );
      const hostPane = document.querySelector(
        '[data-testid="workspace-right-surface-active-pane"]',
      );
      const layoutRoot = document.querySelector(
        '[data-testid="layout-transition-root"]',
      );
      const layoutCanvasPanel = document.querySelector(
        '[data-testid="layout-canvas-panel"]',
      );
      const roots = Array.from(
        document.querySelectorAll(
          '[data-testid="workspace-article-editor-surface"]',
        ),
      );
      const root = roots.find(isVisible) ?? roots[0] ?? null;
      const canvases = Array.from(
        document.querySelectorAll(
          '[data-testid="workspace-article-editor-canvas"]',
        ),
      );
      const canvas = canvases.find(isVisible) ?? canvases[0] ?? null;
      const fixtureOnlyNotice = document.querySelector(
        '[data-testid="workspace-article-editor-fixture-only"]',
      );
      const bodyText = document.body?.innerText || "";
      const canvasText = canvas?.innerText || "";
      const articleDraftButtons = Array.from(
        document.querySelectorAll(
          '[data-testid="workspace-article-editor-related-articleDraft"]',
        ),
      );
      const articleHeaderTitle =
        root?.querySelector("h2")?.textContent?.replace(/\s+/g, " ").trim() ??
        "";
      const fixtureOnlyArticleHidden =
        isVisible(fixtureOnlyNotice) &&
        !canvasText.includes("fixtureOnlyHostGeneration") &&
        !canvasText.includes("fixturePromptFingerprint");
      const hasArticleDraftObject =
        articleDraftButtons.length > 0 ||
        (articleHeaderTitle.length > 0 &&
          (fixtureOnlyArticleHidden ||
            (canvasText.length > 160 &&
              !canvasText.includes("文章正文生成后会出现在这里"))));
      const hasArticleCanvasContent =
        fixtureOnlyArticleHidden ||
        (canvasText.length > 160 &&
          !canvasText.includes("文章正文生成后会出现在这里"));
      return {
        activeSurface: host?.getAttribute("data-surface") ?? null,
        hostVisible: isVisible(host),
        hostPaneVisible: isVisible(hostPane),
        layoutRootVisible: isVisible(layoutRoot),
        layoutCanvasPanelVisible: isVisible(layoutCanvasPanel),
        hostRect: rectToJson(host?.getBoundingClientRect() ?? null),
        hostPaneRect: rectToJson(hostPane?.getBoundingClientRect() ?? null),
        layoutRootRect: rectToJson(layoutRoot?.getBoundingClientRect() ?? null),
        layoutCanvasPanelRect: rectToJson(
          layoutCanvasPanel?.getBoundingClientRect() ?? null,
        ),
        layoutRootStyle: readStyle(layoutRoot),
        layoutCanvasPanelStyle: readStyle(layoutCanvasPanel),
        hostStyle: readStyle(host),
        hostPaneStyle: readStyle(hostPane),
        rootRect: rectToJson(root?.getBoundingClientRect() ?? null),
        canvasRect: rectToJson(canvas?.getBoundingClientRect() ?? null),
        rootStyle: readStyle(root),
        canvasStyle: readStyle(canvas),
        rootVisible: isVisible(root),
        canvasVisible: isVisible(canvas),
        visibleRootCount: roots.filter(isVisible).length,
        visibleCanvasCount: canvases.filter(isVisible).length,
        hasArticleEditorTitle:
          bodyText.includes("文章编辑器") ||
          bodyText.includes("Article Editor"),
        hasArticleTitle:
          bodyText.includes("公众号文章草稿") || hasArticleDraftObject,
        hasArticleDraftObject,
        hasArticleCanvasContent,
        hasFixtureOnlyArticleHidden: fixtureOnlyArticleHidden,
        articleHeaderTitle,
        articleDraftObjectCount: articleDraftButtons.length,
        hasLoadedDraftStatus:
          bodyText.includes("已载入产物正文") ||
          bodyText.includes("测试夹具正文未载入"),
        hasLegacyWorkerTemplateText:
          bodyText.includes("三轮资料检索") ||
          canvasText.includes("先把目标定清楚") ||
          canvasText.includes("学习路线"),
        hasArticleCanvasBodyContent: hasArticleCanvasContent,
        hasArticleArtifactFrame: Boolean(
          document.querySelector('[data-testid="article-artifact-frame"]') ??
          document.querySelector('[data-frame-kind="articleArtifacts"]'),
        ),
        bodyTextSample: bodyText.slice(0, 1600),
      };

      function rectToJson(rect) {
        if (!rect) {
          return null;
        }
        return {
          x: Math.round(rect.x),
          y: Math.round(rect.y),
          width: Math.round(rect.width),
          height: Math.round(rect.height),
          top: Math.round(rect.top),
          left: Math.round(rect.left),
          right: Math.round(rect.right),
          bottom: Math.round(rect.bottom),
        };
      }

      function readStyle(node) {
        if (!node) {
          return null;
        }
        const style = window.getComputedStyle(node);
        return {
          display: style.display,
          width: style.width,
          height: style.height,
          minWidth: style.minWidth,
          minHeight: style.minHeight,
          flex: style.flex,
          flexBasis: style.flexBasis,
          flexGrow: style.flexGrow,
          flexShrink: style.flexShrink,
          position: style.position,
          overflow: style.overflow,
          boxSizing: style.boxSizing,
        };
      }

      function isVisible(node) {
        const rect = node?.getBoundingClientRect();
        const style = node ? window.getComputedStyle(node) : null;
        return Boolean(
          node &&
          rect &&
          rect.width > 8 &&
          rect.height > 8 &&
          style?.display !== "none" &&
          style?.visibility !== "hidden",
        );
      }
    });
    lastSnapshot = snapshot;
    if (
      snapshot?.activeSurface === "articleWorkspace" &&
      snapshot.rootVisible &&
      snapshot.canvasVisible &&
      snapshot.hasArticleEditorTitle &&
      snapshot.hasArticleDraftObject &&
      snapshot.hasArticleCanvasContent
    ) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `点击 ArticleArtifactFrame 后 Article Editor 未展开: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function requestContentFactoryArticleWorkspaceSurface(
  page,
  workspace,
  requestLog,
  identity,
) {
  return await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
    {
      workspaceId: workspace.workspaceId,
      workspaceRoot: workspace.rootPath,
      sessionId: identity.sessionId,
      surfaceKind: "articleWorkspace",
      origin: "runtime",
      priority: "foreground",
      candidateId: "content-factory-article-workspace",
      reason: "article_workspace_ready",
      ttlMs: 120_000,
      metadata: {
        fixtureOrigin: "content-factory-article-workspace",
        workspacePatch: buildWorkspacePatch(workspace, identity),
      },
    },
    requestLog,
  );
}

async function createContentFactoryArticleWorkspaceSession(
  page,
  workspace,
  requestLog,
  fixtureProvider,
) {
  assert(
    workspace?.rootPath,
    "内容工厂 Article Editor fixture 缺少 workspace rootPath",
  );
  assert(
    workspace?.workspaceId,
    "内容工厂 Article Editor fixture 缺少 workspaceId",
  );

  const session = await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_SESSION_START,
    {
      model: fixtureProvider.modelId,
      modelProvider: fixtureProvider.providerId,
      cwd: workspace.rootPath,
      runtimeWorkspaceRoots: [workspace.rootPath],
      serviceName: CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_TITLE,
      threadSource: "appServer",
      historyMode: "paginated",
    },
    requestLog,
  );
  const thread = session.result?.thread ?? {};
  const sessionId = String(thread.sessionId ?? "").trim();
  const threadId = String(thread.id ?? "").trim();
  assert(sessionId, "thread/start 未返回内容工厂 canonical sessionId");
  assert(threadId, "thread/start 未返回内容工厂 canonical thread.id");

  return sanitizeJson({
    sessionId,
    threadId,
    identity: { sessionId, threadId },
    providerId: fixtureProvider.providerId,
    modelId: fixtureProvider.modelId,
    appId: CONTENT_FACTORY_APP_ID,
    title: CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_TITLE,
  });
}

function createContentFactoryScenarioIdentity(sessionIdentity) {
  const sessionId = String(sessionIdentity?.sessionId ?? "").trim();
  const threadId = String(sessionIdentity?.threadId ?? "").trim();
  assert(sessionId && threadId, "内容工厂场景缺少 canonical thread identity");

  return {
    sessionId,
    threadId,
    sourceTaskId: "workspace-patch-fixture",
  };
}

async function writeWorkspacePatchArtifact(
  page,
  workspace,
  requestLog,
  identity,
) {
  const workspacePatch = buildWorkspacePatch(workspace, identity);
  return await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_ARTIFACT_WRITE,
    {
      threadId: identity.threadId,
      artifact: {
        artifactRef: "artifact-workspace-patch-1",
        path: ".lime/artifacts/workspace-patch.json",
        title: "内容工厂工作区补丁",
        kind: "workspace_patch",
        status: "ready",
        content: JSON.stringify(workspacePatch),
        metadata: { workspacePatch },
      },
    },
    requestLog,
  );
}

async function writeContentFactoryArticleWorkspaceActionResultArtifact(
  page,
  workspace,
  requestLog,
  identity,
) {
  const workspacePatch = buildContentFactoryActionResultWorkspacePatch(
    workspace,
    identity,
  );
  return await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_ARTIFACT_WRITE,
    {
      threadId: identity.threadId,
      artifact: {
        artifactRef: "artifact-image-regenerate-workspace-patch",
        path: ".lime/artifacts/article-workspace/image-regenerate-workspace-patch.json",
        title: "配图组重新生成结果",
        kind: "workspace_patch",
        status: "ready",
        content: JSON.stringify(workspacePatch),
        metadata: { workspacePatch },
      },
    },
    requestLog,
  );
}

async function notifySessionChanged(page, workspaceId, sessionId) {
  await page.evaluate(
    ({ sessionId, workspaceId }) => {
      window.dispatchEvent(
        new CustomEvent("lime:agent-runtime-sessions-changed", {
          detail: {
            reason: "external",
            sessionId,
            workspaceId,
          },
        }),
      );
      window.dispatchEvent(new Event("focus"));
    },
    {
      sessionId,
      workspaceId,
    },
  );
}

async function waitForContentFactoryArticleWorkspaceGui(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await page.evaluate(() => {
      const host = document.querySelector(
        '[data-testid="workspace-right-surface-host"]',
      );
      const roots = Array.from(
        document.querySelectorAll(
          '[data-testid="workspace-article-editor-surface"]',
        ),
      );
      const root = roots.find(isVisible) ?? roots[0] ?? null;
      const rendererHost = document.querySelector(
        '[data-testid="workspace-article-workspace-app-declared-renderer"]',
      );
      const rendererExecutableHost =
        rendererHost?.querySelector("iframe, webview");
      const toggle = document.querySelector(
        '[data-testid="task-center-object-canvas-toggle"]',
      );
      const bodyText = document.body?.innerText || "";
      const articleDraftButtons = Array.from(
        document.querySelectorAll(
          '[data-testid="workspace-article-editor-related-articleDraft"]',
        ),
      );
      const canvas = document.querySelector(
        '[data-testid="workspace-article-editor-canvas"]',
      );
      const fixtureOnlyNotice = document.querySelector(
        '[data-testid="workspace-article-editor-fixture-only"]',
      );
      const canvasText = canvas?.innerText || "";
      const articleHeaderTitle =
        root?.querySelector("h2")?.textContent?.replace(/\s+/g, " ").trim() ??
        "";
      const fixtureOnlyArticleHidden =
        isVisible(fixtureOnlyNotice) &&
        !canvasText.includes("fixtureOnlyHostGeneration") &&
        !canvasText.includes("fixturePromptFingerprint");
      const hasArticleDraftObject =
        articleDraftButtons.length > 0 ||
        (articleHeaderTitle.length > 0 &&
          (fixtureOnlyArticleHidden ||
            (canvasText.length > 160 &&
              !canvasText.includes("文章正文生成后会出现在这里"))));
      const hasArticleCanvasContent =
        fixtureOnlyArticleHidden ||
        (canvasText.length > 160 &&
          !canvasText.includes("文章正文生成后会出现在这里"));
      return {
        activeSurface: host?.getAttribute("data-surface") ?? null,
        rootVisible: isVisible(root),
        visibleRootCount: roots.filter(isVisible).length,
        toggleTitle: toggle?.getAttribute("title") ?? "",
        toggleAria: toggle?.getAttribute("aria-label") ?? "",
        hasArticleEditorTitle:
          bodyText.includes("文章编辑器") ||
          bodyText.includes("Article Editor"),
        hasArticleTitle:
          bodyText.includes("公众号文章草稿") || hasArticleDraftObject,
        hasArticleDraftObject,
        hasArticleCanvasContent,
        hasFixtureOnlyArticleHidden: fixtureOnlyArticleHidden,
        articleHeaderTitle,
        articleDraftObjectCount: articleDraftButtons.length,
        hasImageSetTitle: bodyText.includes("配图组"),
        hasStoryboardTitle: bodyText.includes("视频分镜"),
        hasChecklistTitle: bodyText.includes("交付检查清单"),
        hasWorkerEvidenceTitle:
          bodyText.includes("运行记录") || bodyText.includes("写作计划"),
        hasLegacyWorkerTemplateText:
          bodyText.includes("三轮资料检索") ||
          canvasText.includes("先把目标定清楚") ||
          canvasText.includes("学习路线"),
        hasArticleCanvasBodyContent: hasArticleCanvasContent,
        hasArticlePreview: hasArticleCanvasContent,
        hasImagePrompt: bodyText.includes("明亮的中文内容工厂主图"),
        hasRegeneratedImageSummary: bodyText.includes("已重新生成 2 张候选图"),
        hasRegeneratedImagePrompt:
          bodyText.includes("厨房台面主图，明亮自然光"),
        rendererHost: {
          visible: isVisible(rendererHost),
          pluginVisible: bodyText.includes("content-factory-app"),
          rendererKindVisible: bodyText.includes("app_declared"),
          executionModeVisible: bodyText.includes("host_placeholder"),
          rendererExecutionModelVisible: bodyText.includes(
            "host_placeholder_only",
          ),
          entryLoadPolicyVisible: bodyText.includes("not_loaded"),
          executableHostAbsent: !rendererExecutableHost,
          reasonVisible: bodyText.includes(
            "app_declared_renderer_placeholder_only",
          ),
          allowedOutputVisible: bodyText.includes(
            "workspace_patch",
          ),
          entryVisible: bodyText.includes("./renderer/storyboard.tsx"),
          actionVisible: bodyText.includes("open_storyboard"),
        },
        bodyTextSample: bodyText.slice(0, 2000),
      };

      function isVisible(node) {
        const rect = node?.getBoundingClientRect();
        const style = node ? window.getComputedStyle(node) : null;
        return Boolean(
          node &&
          rect &&
          rect.width > 8 &&
          rect.height > 8 &&
          style?.display !== "none" &&
          style?.visibility !== "hidden",
        );
      }
    });
    lastSnapshot = snapshot;
    if (
      snapshot?.activeSurface === "articleWorkspace" &&
      snapshot.rootVisible &&
      snapshot.hasArticleEditorTitle &&
      snapshot.hasArticleDraftObject &&
      snapshot.hasArticleCanvasContent
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `内容工厂 Article Editor GUI 未就绪: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function selectContentFactoryArticleObject(page, options) {
  return await selectContentFactoryArticleWorkspaceObject(page, options, {
    objectKind: "articleDraft",
    selector: '[data-testid="workspace-article-editor-related-articleDraft"]',
    failureLabel: "文章对象",
  });
}

async function waitForContentFactoryArticleCanvasSurface(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await page.evaluate((forbiddenArticleTemplateMarkers) => {
      const root = document.querySelector(
        '[data-testid="workspace-article-editor-surface"]',
      );
      const structure = document.querySelector(
        '[data-testid="workspace-article-editor-title-candidates"]',
      );
      const research = document.querySelector(
        '[data-testid="workspace-article-editor-research"]',
      );
      const outline = document.querySelector(
        '[data-testid="workspace-article-editor-outline"]',
      );
      const citations = document.querySelector(
        '[data-testid="workspace-article-editor-citations"]',
      );
      const imageSlots = document.querySelector(
        '[data-testid="workspace-article-editor-image-slots"]',
      );
      const documentCanvas = document.querySelector(
        '[data-testid="workspace-article-editor-canvas"]',
      );
      const fixtureOnlyNotice = document.querySelector(
        '[data-testid="workspace-article-editor-fixture-only"]',
      );
      const documentImageSlots = document.querySelector(
        '[data-testid="workspace-article-editor-image-slots"]',
      );
      const takeaways = document.querySelector(
        '[data-testid="workspace-article-editor-takeaways"]',
      );
      const writingPlan = document.querySelector(
        '[data-testid="workspace-article-editor-writing-plan"]',
      );
      const contentFactoryOrchestration = document.querySelector(
        '[data-testid="workspace-article-editor-content-factory-orchestration"]',
      );
      const orchestrationSteps = Array.from(
        document.querySelectorAll(
          '[data-testid="workspace-article-editor-orchestration-step"]',
        ),
      );
      const orchestrationSubagents = Array.from(
        document.querySelectorAll(
          '[data-testid="workspace-article-editor-orchestration-subagent"]',
        ),
      );
      const orchestrationSkillRefs = Array.from(
        document.querySelectorAll(
          '[data-testid="workspace-article-editor-orchestration-skill-ref"], [data-testid="workspace-article-editor-orchestration-skill"]',
        ),
      );
      const orchestrationConnectors = Array.from(
        document.querySelectorAll(
          '[data-testid="workspace-article-editor-orchestration-connector"]',
        ),
      );
      const orchestrationHooks = Array.from(
        document.querySelectorAll(
          '[data-testid="workspace-article-editor-orchestration-hook"]',
        ),
      );
      const review = document.querySelector(
        '[data-testid="workspace-article-editor-review"]',
      );
      const documentCanvasText = documentCanvas?.innerText || "";
      const fixtureOnlyNoticeText = fixtureOnlyNotice?.textContent || "";
      const contentFactoryOrchestrationText =
        contentFactoryOrchestration?.textContent || "";
      const orchestrationSkillRefsText = orchestrationSkillRefs
        .map((item) => item.textContent || "")
        .join("\n");
      const orchestrationSubagentValues = orchestrationSubagents
        .map((item) => item.getAttribute("data-subagent-ref") || "")
        .filter(Boolean);
      const orchestrationSkillRefValues = orchestrationSkillRefs
        .map((item) => item.getAttribute("data-skill-ref") || "")
        .filter(Boolean);
      const orchestrationConnectorValues = orchestrationConnectors
        .map((item) => item.getAttribute("data-connector-ref") || "")
        .filter(Boolean);
      const orchestrationHookValues = orchestrationHooks
        .map((item) => item.getAttribute("data-hook-ref") || "")
        .filter(Boolean);
      const articleCanvasHasForbiddenTemplate =
        forbiddenArticleTemplateMarkers.some((marker) =>
          documentCanvasText.includes(marker),
        ) ||
        documentCanvasText.includes("fixtureOnlyHostGeneration") ||
        documentCanvasText.includes("fixturePromptFingerprint");
      const hasArticleDocumentPreview =
        documentCanvasText.length > 160 &&
        !documentCanvasText.includes("文章正文生成后会出现在这里");
      const fixtureOnlyArticleHidden =
        isVisible(fixtureOnlyNotice) &&
        fixtureOnlyNoticeText.length > 0 &&
        !documentCanvasText.includes("fixtureOnlyHostGeneration") &&
        !documentCanvasText.includes("fixturePromptFingerprint");
      const metadataPanelsHidden =
        !structure &&
        !research &&
        !outline &&
        !citations &&
        !imageSlots &&
        !takeaways &&
        !writingPlan &&
        !review;
      return {
        rootVisible: isVisible(root),
        structurePresent: Boolean(structure),
        researchPresent: Boolean(research),
        outlinePresent: Boolean(outline),
        citationsPresent: Boolean(citations),
        imageSlotsPresent: Boolean(imageSlots),
        documentImageSlotsPresent: Boolean(documentImageSlots),
        takeawaysPresent: Boolean(takeaways),
        writingPlanPresent: Boolean(writingPlan),
        contentFactoryOrchestrationPresent: Boolean(
          contentFactoryOrchestration,
        ),
        reviewPresent: Boolean(review),
        structureVisible: isVisible(structure),
        researchVisible: isVisible(research),
        outlineVisible: isVisible(outline),
        citationsVisible: isVisible(citations),
        imageSlotsVisible: isVisible(imageSlots),
        documentCanvasVisible: isVisible(documentCanvas),
        documentImageSlotsVisible: isVisible(documentImageSlots),
        contentFactoryOrchestrationVisible: isVisible(
          contentFactoryOrchestration,
        ),
        contentFactoryOrchestrationStepCount: orchestrationSteps.length,
        workflowUiRailHidden:
          !contentFactoryOrchestration &&
          orchestrationSteps.length === 0 &&
          orchestrationSubagents.length === 0 &&
          orchestrationSkillRefs.length === 0 &&
          orchestrationConnectors.length === 0 &&
          orchestrationHooks.length === 0,
        metadataPanelsHidden,
        hasDocumentPreview: hasArticleDocumentPreview,
        hasVisibleContentFactoryOrchestration:
          contentFactoryOrchestrationText.length > 0 &&
          isVisible(contentFactoryOrchestration),
        hasVisibleSubagents: orchestrationSubagentValues.length > 0,
        hasVisibleSkillRef:
          orchestrationSkillRefValues.length > 0 &&
          orchestrationSkillRefsText.trim().length > 0,
        hasVisibleConnectors: orchestrationConnectorValues.length > 0,
        hasVisibleHooks: orchestrationHookValues.length > 0,
        visibleSubagents: orchestrationSubagentValues,
        visibleSkillRefs: orchestrationSkillRefValues,
        visibleConnectors: orchestrationConnectorValues,
        visibleHooks: orchestrationHookValues,
        articleCanvasHasForbiddenTemplate,
        hasFullArticleCanvas:
          hasArticleDocumentPreview && !articleCanvasHasForbiddenTemplate,
        fixtureOnlyNoticeVisible: isVisible(fixtureOnlyNotice),
        fixtureOnlyArticleHidden,
        articleCanvasTextSample: documentCanvasText.slice(0, 2000),
      };

      function isVisible(node) {
        const rect = node?.getBoundingClientRect();
        const style = node ? window.getComputedStyle(node) : null;
        return Boolean(
          node &&
          rect &&
          rect.width > 8 &&
          rect.height > 8 &&
          style?.display !== "none" &&
          style?.visibility !== "hidden",
        );
      }
    }, FORBIDDEN_CONTENT_FACTORY_ARTICLE_TEMPLATE_MARKERS);
    lastSnapshot = snapshot;
    if (
      snapshot?.rootVisible &&
      snapshot.documentCanvasVisible &&
      (snapshot.hasDocumentPreview || snapshot.fixtureOnlyArticleHidden) &&
      snapshot.metadataPanelsHidden &&
      snapshot.workflowUiRailHidden &&
      (snapshot.hasFullArticleCanvas || snapshot.fixtureOnlyArticleHidden)
    ) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `内容工厂 Article Editor 正文画布未按 current 口径渲染: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function selectContentFactoryStoryboardObject(page, options) {
  return await selectContentFactoryArticleWorkspaceObject(page, options, {
    objectKind: "videoStoryboard",
    selector:
      '[data-testid="workspace-article-editor-related-videoStoryboard"]',
    failureLabel: "分镜对象",
    allowHiddenCandidate: true,
  });
}

async function selectContentFactoryArticleWorkspaceObject(
  page,
  options,
  {
    allowHiddenCandidate = false,
    failureLabel,
    objectKind,
    preferredText,
    selector,
  },
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await page.evaluate(
      ({ objectKind, preferredText, selector }) => {
        const roots = Array.from(
          document.querySelectorAll(
            '[data-testid="workspace-article-editor-surface"]',
          ),
        );
        const root = roots.find(isVisible) ?? roots[0] ?? null;
        const queryRoot = root ?? document;
        const buttons = Array.from(queryRoot.querySelectorAll(selector));
        const preferredButton = preferredText
          ? buttons.find((candidate) =>
              (candidate.textContent || "").includes(preferredText),
            )
          : null;
        const button = preferredText
          ? preferredButton
          : (buttons.find(isVisible) ?? buttons[0]);
        if (button && !isVisible(button)) {
          button.scrollIntoView({ block: "center", inline: "nearest" });
        }
        const buttonVisible = isVisible(button);
        if (buttonVisible) {
          button.click();
        }
        const canvas = queryRoot.querySelector(
          '[data-testid="workspace-article-editor-canvas"]',
        );
        const fixtureOnlyNotice = queryRoot.querySelector(
          '[data-testid="workspace-article-editor-fixture-only"]',
        );
        const canvasText = canvas?.textContent ?? "";
        const fixtureOnlyArticleHidden =
          isVisible(fixtureOnlyNotice) &&
          !canvasText.includes("fixtureOnlyHostGeneration") &&
          !canvasText.includes("fixturePromptFingerprint");
        const alreadySelected =
          objectKind === "articleDraft" &&
          isVisible(root) &&
          isVisible(canvas) &&
          (fixtureOnlyArticleHidden ||
            (canvasText.length > 160 &&
              !canvasText.includes("文章正文生成后会出现在这里")));
        const visible = buttonVisible || alreadySelected;
        return {
          candidateCount: buttons.length,
          preferredFound: Boolean(preferredButton),
          alreadySelected,
          buttonVisible,
          visible,
          text: button?.textContent ?? "",
        };

        function isVisible(node) {
          const rect = node?.getBoundingClientRect();
          const style = node ? window.getComputedStyle(node) : null;
          return Boolean(
            node &&
            rect &&
            rect.width > 8 &&
            rect.height > 8 &&
            style?.display !== "none" &&
            style?.visibility !== "hidden",
          );
        }
      },
      { objectKind, preferredText, selector },
    );
    lastSnapshot = snapshot;
    if (snapshot?.visible) {
      return sanitizeJson({
        selected: true,
        objectKind,
        text: snapshot.text,
      });
    }
    if (allowHiddenCandidate && snapshot?.candidateCount > 0) {
      return sanitizeJson({
        selected: false,
        objectKind,
        candidatePresent: true,
        skippedReason: "compact_hidden_related_object",
        text: snapshot.text,
      });
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `内容工厂 Article Editor ${failureLabel}不可选: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

function summarizeArtifactWrite(result) {
  const sidecar = asRecord(result?.sidecar) ?? {};
  return sanitizeJson({
    threadId: readString(result?.threadId, result?.thread_id),
    turnId: readString(result?.turnId, result?.turn_id),
    artifactRef: readString(result?.artifactRef, result?.artifact_ref),
    eventId: readString(result?.eventId, result?.event_id),
    sequence: readNumber(result?.sequence),
    persistedAt: readString(result?.persistedAt, result?.persisted_at),
    sidecarRelativePath: readString(
      sidecar.relativePath,
      sidecar.relative_path,
    ),
    contentStatus: readString(sidecar.contentStatus, sidecar.content_status),
    contentBytes: readNumber(sidecar.bytes),
    contentSha256: readString(sidecar.sha256),
  });
}

function summarizeRightSurfaceRequest(result) {
  const pending = asRecord(result?.pending) ?? {};
  return sanitizeJson({
    requestId: readString(result?.requestId, pending.requestId),
    status: readString(result?.status, pending.status),
    surfaceKind: readString(pending.surfaceKind, pending.surface_kind),
    candidateId: readString(pending.candidateId, pending.candidate_id),
    origin: readString(pending.origin),
  });
}

function selectContentFactoryWorkerDogfoodEvidence(workerEvidence, identity) {
  const taskEvidenceItems = workerEvidence.filter(
    (evidence) =>
      readString(evidence?.taskId, evidence?.task_id) === identity.sourceTaskId,
  );
  const completedTaskEvidence = taskEvidenceItems.find((evidence) =>
    isCompletedContentFactoryWorkerEvidence(evidence),
  );
  if (completedTaskEvidence) {
    return completedTaskEvidence;
  }
  const terminalTaskEvidence = taskEvidenceItems.find((evidence) =>
    ["completed", "failed"].includes(readString(evidence?.status)),
  );
  if (terminalTaskEvidence) {
    return terminalTaskEvidence;
  }

  const completedEvidence = workerEvidence.find((evidence) =>
    isCompletedContentFactoryWorkerEvidence(evidence),
  );
  if (completedEvidence) {
    return completedEvidence;
  }

  return taskEvidenceItems[0];
}

function isCompletedContentFactoryWorkerEvidence(evidence) {
  if (readString(evidence?.status) !== "completed") {
    return false;
  }
  if (
    readString(evidence?.eventType, evidence?.event_type) !==
    "artifact.snapshot"
  ) {
    return false;
  }
  if (
    readString(evidence?.taskKind, evidence?.task_kind) !==
    "content.article.generate"
  ) {
    return false;
  }
  return (
    readString(evidence?.artifactKind, evidence?.artifact_kind) ===
      "workspace_patch" ||
    readNumber(evidence?.outputObjectCount, evidence?.output_object_count) > 0
  );
}

function summarizeContentFactoryArticleWorkspaceReadModel(result, identity) {
  const detail = asRecord(result?.detail) ?? asRecord(result);
  const thread = asRecord(result?.thread) ?? {};
  const threadExtra = asRecord(thread.extra) ?? {};
  const threadRead =
    asRecord(detail?.threadRead) ??
    asRecord(detail?.thread_read) ??
    asRecord(threadExtra.threadRead) ??
    asRecord(threadExtra.thread_read) ??
    thread;
  const workflowRuns = readArray(
    threadRead.workflowRuns,
    threadRead.workflow_runs,
    detail.workflowRuns,
    detail.workflow_runs,
  );
  const workflowSteps = readArray(
    threadRead.workflowSteps,
    threadRead.workflow_steps,
    detail.workflowSteps,
    detail.workflow_steps,
  );
  const articleWorkspace =
    asRecord(threadRead.articleWorkspace) ??
    asRecord(threadRead.article_workspace) ??
    asRecord(detail?.articleWorkspace) ??
    asRecord(detail?.article_workspace) ??
    asRecord(threadExtra.articleWorkspace) ??
    asRecord(threadExtra.article_workspace) ??
    {};
  const objects = readArray(articleWorkspace.objects);
  const workerEvidence = readArray(
    articleWorkspace.workerEvidence,
    articleWorkspace.worker_evidence,
  );
  const editedDraft =
    asRecord(articleWorkspace.editedDraft) ??
    asRecord(articleWorkspace.edited_draft) ??
    {};
  const artifacts = readArray(
    threadRead.artifacts,
    detail?.artifacts,
    threadExtra.artifacts,
  );
  const articleArtifact = findArtifactByRef(
    artifacts,
    CONTENT_FACTORY_ARTICLE_WORKSPACE_ARTICLE_ARTIFACT_ID,
  );
  const storyboardArtifact = findArtifactByRef(
    artifacts,
    CONTENT_FACTORY_ARTICLE_WORKSPACE_STORYBOARD_ARTIFACT_ID,
  );
  const checklistArtifact = findArtifactByRef(
    artifacts,
    CONTENT_FACTORY_ARTICLE_WORKSPACE_CHECKLIST_ARTIFACT_ID,
  );
  const articleMetadata = asRecord(articleArtifact?.metadata) ?? {};
  const articleArticleWorkspace =
    asRecord(articleMetadata.articleWorkspace) ?? {};
  const storyboardMetadata = asRecord(storyboardArtifact?.metadata) ?? {};
  const storyboardArticleWorkspace =
    asRecord(storyboardMetadata.articleWorkspace) ?? {};
  const checklistMetadata = asRecord(checklistArtifact?.metadata) ?? {};
  const checklistArticleWorkspace =
    asRecord(checklistMetadata.articleWorkspace) ?? {};
  const failedEvidence = workerEvidence.find(
    (evidence) =>
      readString(evidence?.status) === "failed" &&
      readString(evidence?.errorCode, evidence?.error_code) ===
        "worker_invalid_json_output",
  );
  const completedActionEvidence = workerEvidence.find(
    (evidence) =>
      readString(evidence?.taskId, evidence?.task_id) ===
      "image_regenerate_job_1",
  );
  const workerDogfoodEvidence = selectContentFactoryWorkerDogfoodEvidence(
    workerEvidence,
    identity,
  );
  const workerArticleObject = objects.find((object) => {
    if (productObjectKind(object) !== "articleDraft") {
      return false;
    }
    const source = asRecord(object?.source) ?? {};
    return readString(source.taskId, source.task_id) === identity.sourceTaskId;
  });
  const workerArticleSource = asRecord(workerArticleObject?.source) ?? {};
  const workerArticleSourceText = [
    readString(workerArticleSource.markdown, workerArticleSource.markdown_text),
    readString(
      workerArticleSource.processMarkdown,
      workerArticleSource.process_markdown,
    ),
    readString(
      workerArticleSource.documentText,
      workerArticleSource.document_text,
    ),
    readString(
      workerArticleSource.finalMarkdown,
      workerArticleSource.final_markdown,
    ),
  ]
    .filter(Boolean)
    .join("\n\n");
  const workerArticleResearchRoundCount = readArray(
    workerArticleSource.researchRounds,
    workerArticleSource.research_rounds,
  ).length;
  const workerArticleImageSlotCount = readArray(
    workerArticleSource.imageSlots,
    workerArticleSource.image_slots,
  ).length;
  const imageObject = objects.find((object) => {
    const ref = asRecord(object?.ref) ?? asRecord(object?.objectRef) ?? {};
    return readString(ref.id) === IMAGE_SET_OBJECT_ID;
  });
  const storyboardObject = objects.find(
    (object) => productObjectKind(object) === "videoStoryboard",
  );
  const storyboardObjectSource =
    asRecord(storyboardObject?.source) ??
    asRecord(storyboardObject?.source_metadata) ??
    {};
  const storyboardRendererContract =
    asRecord(storyboardObjectSource.rendererContract) ??
    asRecord(storyboardObjectSource.renderer_contract) ??
    {};
  const storyboardRuntimeAuthorization =
    asRecord(storyboardRendererContract.runtimeAuthorization) ??
    asRecord(storyboardRendererContract.runtime_authorization) ??
    {};
  const actionResultArtifacts = artifacts.filter((artifact) => {
    const artifactRef = readString(
      artifact?.artifactRef,
      artifact?.artifact_ref,
    );
    return (
      artifactRef === "artifact-image-regenerated" ||
      artifactRef === "artifact-image-regenerate-workspace-patch"
    );
  });

  return sanitizeJson({
    hasArticleWorkspace: Object.keys(articleWorkspace).length > 0,
    schemaVersion: readString(
      articleWorkspace.schemaVersion,
      articleWorkspace.schema_version,
    ),
    appId: readString(articleWorkspace.appId, articleWorkspace.app_id),
    sessionId: readString(
      articleWorkspace.sessionId,
      articleWorkspace.session_id,
    ),
    workspaceId: readString(
      articleWorkspace.workspaceId,
      articleWorkspace.workspace_id,
    ),
    objectCount: objects.length,
    objectTitles: objects.map((object) => readString(object?.title)),
    hasArticleObject: objects.some(
      (object) => productObjectKind(object) === "articleDraft",
    ),
    hasImageSetObject: objects.some(
      (object) => productObjectKind(object) === "imageGenerationSet",
    ),
    hasStoryboardObject: objects.some(
      (object) => productObjectKind(object) === "videoStoryboard",
    ),
    hasChecklistObject: objects.some(
      (object) => productObjectKind(object) === "deliveryChecklist",
    ),
    imageObject: imageObject
      ? {
          status: readString(imageObject.status),
          summary: readString(imageObject.summary),
          previewArtifactId: readString(
            imageObject.previewArtifactId,
            imageObject.preview_artifact_id,
          ),
          sourceTaskId: readString(
            imageObject.ref?.sourceTaskId,
            imageObject.ref?.source_task_id,
          ),
        }
      : null,
    workerEvidenceCount: workerEvidence.length,
    workflowRunCount: workflowRuns.length,
    workflowStepCount: workflowSteps.length,
    workflowUiFactsHidden:
      workflowRuns.length === 0 && workflowSteps.length === 0,
    failedWorkerEvidence: failedEvidence
      ? {
          taskId: readString(failedEvidence.taskId, failedEvidence.task_id),
          status: readString(failedEvidence.status),
          errorCode: readString(
            failedEvidence.errorCode,
            failedEvidence.error_code,
          ),
          failureCategory: readString(
            failedEvidence.failureCategory,
            failedEvidence.failure_category,
          ),
          retryable: readBoolean(failedEvidence.retryable),
          retryAdvice: readString(
            failedEvidence.retryAdvice,
            failedEvidence.retry_advice,
          ),
          retryAttempt: readNumber(
            failedEvidence.retryAttempt,
            failedEvidence.retry_attempt,
          ),
          retryMaxAttempts: readNumber(
            failedEvidence.retryMaxAttempts,
            failedEvidence.retry_max_attempts,
          ),
        }
      : null,
    completedActionWorkerEvidence: completedActionEvidence
      ? {
          taskId: readString(
            completedActionEvidence.taskId,
            completedActionEvidence.task_id,
          ),
          status: readString(completedActionEvidence.status),
          artifactRef: readString(
            completedActionEvidence.artifactRef,
            completedActionEvidence.artifact_ref,
          ),
        }
      : null,
    workerDogfoodEvidence: workerDogfoodEvidence
      ? {
          taskId: readString(
            workerDogfoodEvidence.taskId,
            workerDogfoodEvidence.task_id,
          ),
          taskKind: readString(
            workerDogfoodEvidence.taskKind,
            workerDogfoodEvidence.task_kind,
          ),
          status: readString(workerDogfoodEvidence.status),
          artifactRef: readString(
            workerDogfoodEvidence.artifactRef,
            workerDogfoodEvidence.artifact_ref,
          ),
          artifactKind: readString(
            workerDogfoodEvidence.artifactKind,
            workerDogfoodEvidence.artifact_kind,
          ),
          outputObjectCount: readNumber(
            workerDogfoodEvidence.outputObjectCount,
            workerDogfoodEvidence.output_object_count,
          ),
          workflowKey: readString(
            workerDogfoodEvidence.workflowKey,
            workerDogfoodEvidence.workflow_key,
          ),
          subagents: readStringArray(
            workerDogfoodEvidence.subagents,
            workerDogfoodEvidence.sub_agents,
          ),
          skillRefs: readStringArray(
            workerDogfoodEvidence.skillRefs,
            workerDogfoodEvidence.skill_refs,
          ),
          cliRefs: readStringArray(
            workerDogfoodEvidence.cliRefs,
            workerDogfoodEvidence.cli_refs,
          ),
          connectorRefs: readStringArray(
            workerDogfoodEvidence.connectorRefs,
            workerDogfoodEvidence.connector_refs,
          ),
          hookRefs: readHookPolicyLabels(
            workerDogfoodEvidence.hookPolicy,
            workerDogfoodEvidence.hook_policy,
          ),
          orchestrationStepCount: readArray(workerDogfoodEvidence.orchestration)
            .length,
          errorCode: readString(
            workerDogfoodEvidence.errorCode,
            workerDogfoodEvidence.error_code,
          ),
          errorMessage: readString(
            workerDogfoodEvidence.errorMessage,
            workerDogfoodEvidence.error_message,
            workerDogfoodEvidence.message,
          ),
          failureCategory: readString(
            workerDogfoodEvidence.failureCategory,
            workerDogfoodEvidence.failure_category,
          ),
        }
      : null,
    workerArticleObject: workerArticleObject
      ? {
          objectRef: summarizeArticleObjectRef(workerArticleObject),
          title: readString(workerArticleObject.title),
          summary: readString(workerArticleObject.summary),
          previewArtifactId: readString(
            workerArticleObject.previewArtifactId,
            workerArticleObject.preview_artifact_id,
          ),
          sourceTaskId: readString(
            workerArticleSource.taskId,
            workerArticleSource.task_id,
          ),
          markdownIncludesResearch:
            workerArticleSourceText.includes("## 三轮资料检索") ||
            workerArticleSourceText.includes("## 检索轮次") ||
            workerArticleResearchRoundCount >= 3,
          markdownIncludesDraft:
            workerArticleSourceText.includes("## 正文草稿") ||
            readString(
              workerArticleSource.documentText,
              workerArticleSource.document_text,
              workerArticleSource.finalMarkdown,
              workerArticleSource.final_markdown,
            ).length > 300,
          markdownIncludesEditedDraftMarker: workerArticleSourceText.includes(
            CONTENT_FACTORY_ARTICLE_WORKSPACE_EDITED_DRAFT_MARKER,
          ),
          updatedAt: readString(
            workerArticleSource.updatedAt,
            workerArticleSource.updated_at,
          ),
          researchRoundCount: workerArticleResearchRoundCount,
          imageSlotCount: workerArticleImageSlotCount,
        }
      : null,
    editedDraft:
      Object.keys(editedDraft).length > 0
        ? {
            objectKey: readString(
              editedDraft.objectKey,
              editedDraft.object_key,
            ),
            objectRef: summarizeArticleRefRecord(
              asRecord(editedDraft.objectRef) ??
                asRecord(editedDraft.object_ref) ??
                {},
            ),
            markdownIncludesEditedDraftMarker: readString(
              editedDraft.markdown,
            ).includes(CONTENT_FACTORY_ARTICLE_WORKSPACE_EDITED_DRAFT_MARKER),
            updatedAt: readString(
              editedDraft.updatedAt,
              editedDraft.updated_at,
            ),
          }
        : null,
    artifactCount: artifacts.length,
    actionResultArtifacts: actionResultArtifacts.map((artifact) => ({
      artifactRef: readString(artifact.artifactRef, artifact.artifact_ref),
      kind: readString(artifact.kind),
      title: readString(artifact.title),
    })),
    articleArtifact: articleArtifact
      ? {
          artifactRef: readString(
            articleArtifact.artifactRef,
            articleArtifact.artifact_ref,
          ),
          kind: readString(articleArtifact.kind),
          title: readString(articleArtifact.title),
          artifactSchema: readString(articleMetadata.artifactSchema),
          artifactDocumentId: readString(articleMetadata.artifactDocumentId),
          articleWorkspaceObjectKind: readString(
            articleArticleWorkspace.objectKind,
            articleArticleWorkspace.object_kind,
          ),
        }
      : null,
    storyboardArtifact: storyboardArtifact
      ? {
          artifactRef: readString(
            storyboardArtifact.artifactRef,
            storyboardArtifact.artifact_ref,
          ),
          kind: readString(storyboardArtifact.kind),
          title: readString(storyboardArtifact.title),
          artifactSchema: readString(storyboardMetadata.artifactSchema),
          artifactDocumentId: readString(storyboardMetadata.artifactDocumentId),
          surfaceKind: readString(storyboardMetadata.surfaceKind),
          articleWorkspaceObjectKind: readString(
            storyboardArticleWorkspace.objectKind,
            storyboardArticleWorkspace.object_kind,
          ),
          articleWorkspaceSurfaceKind: readString(
            storyboardArticleWorkspace.surfaceKind,
            storyboardArticleWorkspace.surface_kind,
          ),
          rendererContract: {
            pluginId: readString(
              storyboardRendererContract.pluginId,
              storyboardRendererContract.plugin_id,
            ),
            rendererKind: readString(
              storyboardRendererContract.rendererKind,
              storyboardRendererContract.renderer_kind,
            ),
            entry: readString(storyboardRendererContract.entry),
            executionMode: readString(
              storyboardRuntimeAuthorization.executionMode,
              storyboardRuntimeAuthorization.execution_mode,
            ),
            reasonCode: readString(
              storyboardRuntimeAuthorization.reasonCode,
              storyboardRuntimeAuthorization.reason_code,
            ),
            allowedOutputArtifactKinds: readArray(
              storyboardRuntimeAuthorization.allowedOutputArtifactKinds,
              storyboardRuntimeAuthorization.allowed_output_artifact_kinds,
            )
              .map((value) => readString(value))
              .filter(Boolean),
          },
        }
      : null,
    checklistArtifact: checklistArtifact
      ? {
          artifactRef: readString(
            checklistArtifact.artifactRef,
            checklistArtifact.artifact_ref,
          ),
          kind: readString(checklistArtifact.kind),
          title: readString(checklistArtifact.title),
          artifactSchema: readString(checklistMetadata.artifactSchema),
          artifactDocumentId: readString(checklistMetadata.artifactDocumentId),
          surfaceKind: readString(checklistMetadata.surfaceKind),
          articleWorkspaceObjectKind: readString(
            checklistArticleWorkspace.objectKind,
            checklistArticleWorkspace.object_kind,
          ),
          articleWorkspaceSurfaceKind: readString(
            checklistArticleWorkspace.surfaceKind,
            checklistArticleWorkspace.surface_kind,
          ),
        }
      : null,
  });
}

function summarizeArticleObjectRef(object) {
  const ref = asRecord(object?.ref) ?? asRecord(object?.objectRef) ?? {};
  return summarizeArticleRefRecord(ref);
}

function summarizeArticleRefRecord(ref) {
  return sanitizeJson({
    appId: readString(ref.appId, ref.app_id),
    kind: readString(ref.kind),
    id: readString(ref.id),
    sessionId: readString(ref.sessionId, ref.session_id),
    artifactIds: readArray(ref.artifactIds, ref.artifact_ids)
      .map((value) => readString(value))
      .filter(Boolean),
    sourceTurnId: readString(ref.sourceTurnId, ref.source_turn_id),
    sourceTaskId: readString(ref.sourceTaskId, ref.source_task_id),
    version: readString(ref.version),
  });
}

function findArtifactByRef(artifacts, artifactRef) {
  return artifacts.find(
    (artifact) =>
      readString(artifact?.artifactRef, artifact?.artifact_ref) === artifactRef,
  );
}

function productObjectKind(object) {
  const ref = asRecord(object?.ref) ?? asRecord(object?.objectRef) ?? {};
  return readString(ref.kind);
}

function summarizeContentFactoryArtifactRead(result) {
  const artifacts = readArray(result?.artifacts);
  const artifact = artifacts[0] ?? null;
  const artifactRef = readString(artifact?.artifactRef, artifact?.artifact_ref);
  const content = readString(artifact?.content);
  const document = parseJsonObject(content);
  const documentArtifactId = readString(
    document?.artifactId,
    document?.artifact_id,
  );
  const documentMetadata = asRecord(document?.metadata) ?? {};
  const documentArticleWorkspace =
    asRecord(documentMetadata.articleWorkspace) ??
    asRecord(documentMetadata.article_workspace) ??
    {};
  const documentBlocks = readArray(document?.blocks);
  const richTextMarkdown = documentBlocks
    .map((block) => readString(block?.markdown, block?.content))
    .filter(Boolean)
    .join("\n\n");
  const richTextHasForbiddenTemplate =
    FORBIDDEN_CONTENT_FACTORY_ARTICLE_TEMPLATE_MARKERS.some((marker) =>
      richTextMarkdown.includes(marker),
    );
  return sanitizeJson({
    artifactCount: artifacts.length,
    artifactRef,
    kind: readString(artifact?.kind),
    title: readString(artifact?.title),
    documentTitle: readString(document?.title),
    documentKind: readString(document?.kind),
    documentObjectKind: readString(
      documentArticleWorkspace.objectKind,
      documentArticleWorkspace.object_kind,
    ),
    documentBlockCount: documentBlocks.length,
    documentRichTextLength: richTextMarkdown.length,
    contentStatus: readString(
      artifact?.contentStatus,
      artifact?.content_status,
    ),
    contentIncludesSchema: content.includes("artifact_document.v1"),
    contentIncludesDocumentId: documentArtifactId.startsWith(
      `artifact-document:${CONTENT_FACTORY_APP_ID}:`,
    ),
    contentIncludesArticleTitle: Boolean(readString(document?.title)),
    richTextHasForbiddenTemplate,
    contentIncludesCanonicalArticle:
      richTextMarkdown.length > 160 && !richTextHasForbiddenTemplate,
    contentIncludesEditedDraftMarker: richTextMarkdown.includes(
      CONTENT_FACTORY_ARTICLE_WORKSPACE_EDITED_DRAFT_MARKER,
    ),
  });
}

function parseJsonObject(value) {
  if (!value) {
    return {};
  }
  try {
    const parsed = JSON.parse(value);
    return asRecord(parsed) ?? {};
  } catch {
    return {};
  }
}

function asRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value
    : null;
}

function readArray(...values) {
  for (const value of values) {
    if (Array.isArray(value)) {
      return value;
    }
  }
  return [];
}

function readStringArray(...values) {
  const seen = new Set();
  const items = [];
  for (const value of values) {
    for (const item of readArray(value)) {
      const text = readString(item, item?.id, item?.key, item?.title);
      if (!text || seen.has(text)) {
        continue;
      }
      seen.add(text);
      items.push(text);
    }
  }
  return items;
}

function readHookPolicyLabels(...values) {
  const policy = values.map(asRecord).find(Boolean);
  if (!policy) {
    return [];
  }
  return Object.entries(policy).flatMap(([scope, hooks]) =>
    readStringArray(hooks).map((hook) => `${scope}:${hook}`),
  );
}

function readString(...values) {
  for (const value of values) {
    if (typeof value === "string") {
      return value;
    }
  }
  return "";
}

function readNumber(...values) {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function readBoolean(...values) {
  for (const value of values) {
    if (typeof value === "boolean") {
      return value;
    }
  }
  return null;
}
