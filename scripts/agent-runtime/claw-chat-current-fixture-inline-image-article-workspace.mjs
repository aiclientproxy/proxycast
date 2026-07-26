import fs from "node:fs";
import {
  APP_SERVER_METHOD_ARTIFACT_WRITE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE,
  APP_SERVER_METHOD_SESSION_READ,
  CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_TITLE,
  CONTENT_FACTORY_INLINE_IMAGE_FILE_PATH,
  CONTENT_FACTORY_INLINE_IMAGE_SLOT_ID,
  CONTENT_FACTORY_INLINE_IMAGE_TASK_PROMPT,
  CONTENT_FACTORY_INLINE_IMAGE_URL,
  FIXTURE_MODEL,
  FIXTURE_PROVIDER,
} from "./claw-chat-current-fixture-constants.mjs";
import { runContentFactoryArticleWorkspaceScenario } from "./claw-chat-current-fixture-content-factory-article-workspace.mjs";
import {
  evaluatePageSnapshot,
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

const INLINE_SECTION_TITLE = "核心观点";
const INLINE_ANCHOR_TEXT = "核心观点段落";
const INLINE_PENDING_MARKDOWN = [
  "# Inline 配图恢复验证",
  "",
  `## ${INLINE_SECTION_TITLE}`,
  INLINE_ANCHOR_TEXT,
  "",
  `![${CONTENT_FACTORY_INLINE_IMAGE_TASK_PROMPT}](pending-image-task://content-factory-inline-image-task?status=running&prompt=${encodeURIComponent(
    CONTENT_FACTORY_INLINE_IMAGE_TASK_PROMPT,
  )})`,
  `<!-- lime:image-task-slot:${CONTENT_FACTORY_INLINE_IMAGE_SLOT_ID} -->`,
].join("\n");

export async function runContentFactoryInlineImageArticleWorkspaceScenario({
  page,
  options,
  workspace,
  summary,
  appServerRequests,
}) {
  const base = await runContentFactoryArticleWorkspaceScenario({
    page,
    options,
    workspace,
    appServerRequests,
  });
  Object.assign(summary, base);
  const identity = base.contentFactoryArticleWorkspaceSessionCreation?.identity;
  assert(identity?.sessionId, "inline image fixture 缺少 canonical sessionId");
  assert(identity?.threadId, "inline image fixture 缺少 canonical threadId");
  assert(identity?.workerTaskId, "inline image fixture 缺少 worker taskId");

  const objectRef = await waitForArticleDraftObjectRef(
    page,
    options,
    appServerRequests,
    identity,
  );
  const editedDraftUpdate = await writeInlinePendingEditedDraft({
    page,
    requestLog: appServerRequests,
    objectRef,
    identity,
  });
  summary.contentFactoryInlineImageEditedDraftUpdate = editedDraftUpdate;
  const created = await createInlineImageTask({
    page,
    workspace,
    requestLog: appServerRequests,
    identity,
  });
  summary.contentFactoryInlineImageTaskCreated =
    summarizeMediaTaskArtifact(created);
  const submittedEvent = await emitInlineTaskSubmittedEvent({
    page,
    workspace,
    created,
    identity,
  });
  summary.contentFactoryInlineImageTaskSubmittedEvent = submittedEvent;
  const completed = await completeInlineImageTask({
    page,
    workspace,
    requestLog: appServerRequests,
    created,
    identity,
  });
  summary.contentFactoryInlineImageTaskCompleted =
    summarizeMediaTaskArtifact(completed);
  const restored = await reloadAndOpenInlineArticleWorkspace({
    page,
    options,
    workspace,
    requestLog: appServerRequests,
    identity,
  });
  summary.contentFactoryInlineImageReload = restored;
  const readModel = await waitForInlineImageReadModelReplacement(
    page,
    options,
    appServerRequests,
    identity,
  );
  summary.contentFactoryInlineImageReadModel = readModel;
  const canvas = await waitForInlineImageReplacement(page, options);
  summary.contentFactoryInlineImageCanvas = canvas;

  return sanitizeJson({
    ...base,
    contentFactoryInlineImageEditedDraftUpdate: editedDraftUpdate,
    contentFactoryInlineImageTaskCreated: summarizeMediaTaskArtifact(created),
    contentFactoryInlineImageTaskSubmittedEvent: submittedEvent,
    contentFactoryInlineImageTaskCompleted:
      summarizeMediaTaskArtifact(completed),
    contentFactoryInlineImageReadModel: readModel,
    contentFactoryInlineImageReload: restored,
    contentFactoryInlineImageCanvas: canvas,
  });
}

async function waitForArticleDraftObjectRef(
  page,
  options,
  requestLog,
  identity,
) {
  const startedAt = Date.now();
  let lastSummary = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      {
        threadId: identity.threadId,
        includeTurns: true,
      },
      requestLog,
    );
    const objectRef = findArticleDraftObjectRef(
      read.result,
      identity.workerTaskId,
    );
    lastSummary = summarizeArticleWorkspaceRead(
      read.result,
      identity.workerTaskId,
    );
    if (objectRef) {
      return objectRef;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `inline image fixture 未找到文章对象: ${JSON.stringify(
      sanitizeJson(lastSummary),
    )}`,
  );
}

function findArticleDraftObjectRef(result, workerTaskId) {
  const objects = readArticleWorkspaceObjects(result);
  const articleObjects = objects
    .map((object) => ({
      ref: normalizeArticleObjectRef(object),
      sourceTaskId: readArticleObjectSourceTaskId(object),
    }))
    .filter(({ ref }) => ref?.kind === "articleDraft");
  const selected =
    articleObjects.find(
      ({ ref, sourceTaskId }) =>
        ref.sourceTaskId === workerTaskId || sourceTaskId === workerTaskId,
    ) ?? articleObjects[0];
  return selected?.ref ?? null;
}

function readArticleWorkspaceObjects(result) {
  const detail = asRecord(result?.detail) ?? asRecord(result) ?? {};
  const thread = asRecord(result?.thread) ?? {};
  const threadExtra = asRecord(thread.extra) ?? {};
  const threadRead =
    asRecord(detail.threadRead) ??
    asRecord(detail.thread_read) ??
    asRecord(threadExtra.threadRead) ??
    asRecord(threadExtra.thread_read) ??
    thread;
  const articleWorkspace =
    asRecord(threadRead.articleWorkspace) ??
    asRecord(threadRead.article_workspace) ??
    asRecord(detail.articleWorkspace) ??
    asRecord(detail.article_workspace) ??
    asRecord(threadExtra.articleWorkspace) ??
    asRecord(threadExtra.article_workspace) ??
    {};
  return readArray(articleWorkspace.objects);
}

function normalizeArticleObjectRef(object) {
  const ref = mergeRecords(
    asRecord(object),
    asRecord(object?.object_ref),
    asRecord(object?.objectRef),
    asRecord(object?.ref),
  );
  const source = asRecord(object?.source) ?? {};
  if (readString(ref?.kind) !== "articleDraft") {
    return null;
  }
  const normalized = {
    appId: readString(ref.appId, ref.app_id),
    sessionId: readString(ref.sessionId, ref.session_id),
    kind: readString(ref.kind),
    id: readString(ref.id),
    version: readString(ref.version) || undefined,
    artifactIds: readArray(ref.artifactIds, ref.artifact_ids)
      .map((item) => readString(item))
      .filter(Boolean),
    sourceTurnId: readString(ref.sourceTurnId, ref.source_turn_id) || undefined,
    sourceTaskId:
      readString(
        ref.sourceTaskId,
        ref.source_task_id,
        source.taskId,
        source.task_id,
        source.sourceTaskId,
        source.source_task_id,
      ) || undefined,
  };
  if (
    !normalized.appId ||
    !normalized.sessionId ||
    !normalized.kind ||
    !normalized.id
  ) {
    return null;
  }
  return normalized;
}

function readArticleObjectSourceTaskId(object) {
  const source = asRecord(object?.source) ?? {};
  const ref = mergeRecords(
    asRecord(object),
    asRecord(object?.object_ref),
    asRecord(object?.objectRef),
    asRecord(object?.ref),
  );
  return readString(
    ref.sourceTaskId,
    ref.source_task_id,
    source.taskId,
    source.task_id,
    source.sourceTaskId,
    source.source_task_id,
  );
}

function summarizeArticleWorkspaceRead(result, workerTaskId) {
  const objects = readArticleWorkspaceObjects(result);
  return sanitizeJson({
    hasArticleWorkspace: objects.length > 0,
    objectCount: objects.length,
    objectKinds: objects.map((object) => readString(object?.ref?.kind)),
    articleObjects: objects
      .map((object) => ({
        ref: normalizeArticleObjectRef(object),
        sourceTaskId: readArticleObjectSourceTaskId(object),
      }))
      .filter(({ ref }) => ref?.kind === "articleDraft"),
    expectedWorkerTaskId: workerTaskId,
  });
}

async function writeInlinePendingEditedDraft({
  page,
  requestLog,
  objectRef,
  identity,
}) {
  assert(objectRef?.appId, "inline image fixture 缺少 article object appId");
  assert(
    objectRef?.sessionId,
    "inline image fixture 缺少 article object sessionId",
  );
  assert(objectRef?.kind, "inline image fixture 缺少 article object kind");
  assert(objectRef?.id, "inline image fixture 缺少 article object id");

  const updatedAt = "2026-07-04T00:00:00.000Z";
  const objectKey = `${objectRef.appId}:${objectRef.sessionId}:${objectRef.kind}:${objectRef.id}`;
  const artifactRef =
    objectRef.artifactIds?.find(
      (value) => typeof value === "string" && value.trim(),
    ) || objectKey;
  const editedDraft = {
    objectKey,
    objectRef,
    markdown: INLINE_PENDING_MARKDOWN,
    updatedAt,
  };
  const workspacePatch = {
    objects: [
      {
        ref: objectRef,
        title: "Inline image edited draft",
        status: "draft",
        source: {
          documentText: INLINE_PENDING_MARKDOWN,
          finalMarkdown: INLINE_PENDING_MARKDOWN,
          updatedAt,
        },
      },
    ],
    updatedAt,
  };
  const response = await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_ARTIFACT_WRITE,
    {
      threadId: identity.threadId,
      artifact: {
        artifactRef,
        artifactDocumentId: objectRef.id,
        title: "Inline image edited draft",
        kind: "article_workspace_draft",
        status: "draft",
        content: INLINE_PENDING_MARKDOWN,
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
    objectKey,
    marker: CONTENT_FACTORY_INLINE_IMAGE_SLOT_ID,
    hasPendingImage: INLINE_PENDING_MARKDOWN.includes("pending-image-task://"),
    updatedAt,
  });
}

async function createInlineImageTask({
  page,
  workspace,
  requestLog,
  identity,
}) {
  return await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE,
    {
      projectRootPath: workspace.rootPath,
      prompt: CONTENT_FACTORY_INLINE_IMAGE_TASK_PROMPT,
      title: "Inline 配图恢复任务",
      count: 1,
      providerId: FIXTURE_PROVIDER,
      model: FIXTURE_MODEL,
      executorMode: "images_api",
      usage: "document-inline",
      slotId: CONTENT_FACTORY_INLINE_IMAGE_SLOT_ID,
      anchorSectionTitle: INLINE_SECTION_TITLE,
      anchorText: INLINE_ANCHOR_TEXT,
      sessionId: identity.sessionId,
      projectId: workspace.workspaceId,
      entrySource: "article_inline_fixture",
      modalityContractKey: "image_generation",
      modality: "image",
      routingSlot: "image_generation_model",
      requestedTarget: "generate",
      size: "1024x1024",
    },
    requestLog,
  );
}

async function emitInlineTaskSubmittedEvent({
  page,
  workspace,
  created,
  identity,
}) {
  const result = created?.result ?? {};
  const payload = {
    task_id: readString(result.task_id, result.taskId),
    task_type:
      readString(result.task_type, result.taskType) || "image_generate",
    task_family: "image",
    status: readString(result.status) || "pending_submit",
    path: readString(result.path, result.artifact_path, result.artifactPath),
    absolute_path: readString(result.absolute_path, result.absolutePath),
    prompt: CONTENT_FACTORY_INLINE_IMAGE_TASK_PROMPT,
    size: "1024x1024",
    mode: "generate",
    provider_id: FIXTURE_PROVIDER,
    model: FIXTURE_MODEL,
    count: 1,
    session_id: identity.sessionId,
    project_id: workspace.workspaceId,
    entry_source: "article_inline_fixture",
    requested_target: "generate",
    slot_id: CONTENT_FACTORY_INLINE_IMAGE_SLOT_ID,
    anchor_hint: "section_end",
    anchor_section_title: INLINE_SECTION_TITLE,
    anchor_text: INLINE_ANCHOR_TEXT,
  };
  assert(payload.task_id, "inline image fixture 任务事件缺少 task_id");
  assert(
    payload.absolute_path,
    "inline image fixture 任务事件缺少 absolute_path",
  );
  await page.evaluate(
    async ({ eventName, payload }) => {
      const emit = window.electronAPI?.emit;
      if (typeof emit !== "function") {
        throw new Error("Electron event emit bridge is unavailable");
      }
      await emit(eventName, payload);
    },
    {
      eventName: "lime://creation_task_submitted",
      payload,
    },
  );
  return sanitizeJson({
    emitted: true,
    taskId: payload.task_id,
    taskType: payload.task_type,
    absolutePath: payload.absolute_path,
    slotId: payload.slot_id,
  });
}

async function completeInlineImageTask({
  page,
  workspace,
  requestLog,
  created,
  identity,
}) {
  const taskRef =
    readString(created.result?.artifact_path, created.result?.artifactPath) ||
    readString(created.result?.task_id, created.result?.taskId);
  assert(taskRef, "inline image fixture 创建任务后缺少 taskRef");
  assert(
    fs.existsSync(CONTENT_FACTORY_INLINE_IMAGE_FILE_PATH),
    `inline image fixture 文件不存在: ${CONTENT_FACTORY_INLINE_IMAGE_FILE_PATH}`,
  );
  return await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE,
    {
      projectRootPath: workspace.rootPath,
      taskRef,
      providerId: FIXTURE_PROVIDER,
      model: FIXTURE_MODEL,
      executorMode: "images_api",
      responseId: "inline-image-fixture-response",
      images: [
        {
          url: CONTENT_FACTORY_INLINE_IMAGE_URL,
          prompt: CONTENT_FACTORY_INLINE_IMAGE_TASK_PROMPT,
          revisedPrompt: CONTENT_FACTORY_INLINE_IMAGE_TASK_PROMPT,
          slotId: CONTENT_FACTORY_INLINE_IMAGE_SLOT_ID,
          slotIndex: 1,
          slotPrompt: CONTENT_FACTORY_INLINE_IMAGE_TASK_PROMPT,
        },
      ],
    },
    requestLog,
  );
}

async function reloadAndOpenInlineArticleWorkspace({
  page,
  options,
  workspace,
  requestLog,
  identity,
}) {
  const reload = await reloadRendererDocument(page, options);
  const renderer = await waitForRendererReady(page, options);
  await page.evaluate(
    ({ workspaceId, sessionId }) => {
      window.dispatchEvent(
        new CustomEvent("lime:agent-runtime-sessions-changed", {
          detail: { reason: "external", workspaceId, sessionId },
        }),
      );
    },
    { workspaceId: workspace.workspaceId, sessionId: identity.sessionId },
  );
  const visible = await waitForGuiSessionVisible(
    page,
    options,
    CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_TITLE,
  );
  const opened = await openSessionFromSidebar(page, options, requestLog, {
    sessionId: identity.sessionId,
    title: CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_TITLE,
  });
  return sanitizeJson({ reload, renderer, visible, opened });
}

async function waitForInlineImageReadModelReplacement(
  page,
  options,
  requestLog,
  identity,
) {
  const startedAt = Date.now();
  let lastSummary = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      {
        threadId: identity.threadId,
        includeTurns: true,
      },
      requestLog,
    );
    const summary = summarizeInlineImageReadModel(read.result);
    lastSummary = summary;
    if (
      summary.hasInlineTitle &&
      summary.hasAnchorText &&
      summary.hasImageUrl &&
      !summary.hasPendingProtocol
    ) {
      return summary;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `inline image fixture read model 未持久化已回填正文: ${JSON.stringify(
      sanitizeJson(lastSummary),
    )}`,
  );
}

function collectStringValues(value, output = []) {
  if (typeof value === "string") {
    output.push(value);
    return output;
  }
  if (Array.isArray(value)) {
    value.forEach((item) => collectStringValues(item, output));
    return output;
  }
  if (value && typeof value === "object") {
    Object.values(value).forEach((item) => collectStringValues(item, output));
  }
  return output;
}

function summarizeInlineImageReadModel(result) {
  const strings = collectStringValues(result);
  const targetStrings = strings.filter(
    (value) =>
      value.includes("Inline 配图恢复验证") &&
      value.includes(INLINE_ANCHOR_TEXT) &&
      value.includes(CONTENT_FACTORY_INLINE_IMAGE_SLOT_ID),
  );
  const resolvedTargetStrings = targetStrings.filter(
    (value) =>
      value.includes(CONTENT_FACTORY_INLINE_IMAGE_URL) &&
      !value.includes("pending-image-task://"),
  );
  const matchingStrings = (
    targetStrings.length > 0 ? targetStrings : strings
  ).filter(
    (value) =>
      value.includes("Inline 配图恢复验证") ||
      value.includes(INLINE_ANCHOR_TEXT) ||
      value.includes(CONTENT_FACTORY_INLINE_IMAGE_URL) ||
      value.includes("pending-image-task://") ||
      value.includes(CONTENT_FACTORY_INLINE_IMAGE_SLOT_ID),
  );
  const joined = matchingStrings.join("\n");
  return sanitizeJson({
    hasInlineTitle: joined.includes("Inline 配图恢复验证"),
    hasAnchorText: joined.includes(INLINE_ANCHOR_TEXT),
    hasImageUrl: resolvedTargetStrings.length > 0,
    hasPendingProtocol:
      resolvedTargetStrings.length === 0 &&
      targetStrings.some((value) => value.includes("pending-image-task://")),
    globalHasPendingProtocol: strings.some((value) =>
      value.includes("pending-image-task://"),
    ),
    hasSlotMarker: joined.includes(CONTENT_FACTORY_INLINE_IMAGE_SLOT_ID),
    matchingStringCount: matchingStrings.length,
    targetStringCount: targetStrings.length,
    resolvedTargetStringCount: resolvedTargetStrings.length,
    matchingStrings: matchingStrings.slice(0, 8),
  });
}

async function waitForInlineImageReplacement(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      (expected) => {
        const expectedImageUrl = expected.imageUrl;
        const text = document.body?.innerText || "";
        const editableText =
          Array.from(
            document.querySelectorAll("textarea, [contenteditable='true']"),
          )
            .map((element) => element.value || element.textContent || "")
            .join("\n") || text;
        const images = Array.from(document.querySelectorAll("img")).map(
          (image) => ({
            src: image.getAttribute("src") || "",
            resolvedSrc: image.src || "",
            alt: image.getAttribute("alt") || "",
            complete: image.complete,
            naturalWidth: image.naturalWidth,
            naturalHeight: image.naturalHeight,
          }),
        );
        const imageDiagnosticSources = Array.from(
          document.querySelectorAll("[data-markdown-image-src]"),
        ).map(
          (element) => element.getAttribute("data-markdown-image-src") || "",
        );
        const matchingImages = images.filter(
          (image) =>
            (image.src === expectedImageUrl ||
              image.resolvedSrc === expectedImageUrl) &&
            image.complete &&
            image.naturalWidth > 0,
        );
        const matchingImageContexts = Array.from(
          document.querySelectorAll("img"),
        )
          .filter(
            (image) =>
              (image.getAttribute("src") === expectedImageUrl ||
                image.src === expectedImageUrl) &&
              image.complete &&
              image.naturalWidth > 0,
          )
          .map((image) => {
            let node = image.parentElement;
            for (let depth = 0; node && depth < 8; depth += 1) {
              const contextText = node.innerText || "";
              if (
                contextText.includes(expected.anchorText) ||
                contextText.includes(expected.sectionTitle)
              ) {
                return {
                  depth,
                  text: contextText.slice(0, 1000),
                };
              }
              node = node.parentElement;
            }
            return {
              depth: null,
              text: image.parentElement?.innerText?.slice(0, 1000) || "",
            };
          });
        const hasRenderedImageNearAnchor = matchingImageContexts.some(
          (context) =>
            context.text.includes(expected.anchorText) ||
            context.text.includes(expected.sectionTitle),
        );
        const expectedUnavailablePlaceholders = Array.from(
          document.querySelectorAll(
            '[data-testid="markdown-image-unavailable"]',
          ),
        ).filter((element) => {
          const source =
            element
              .closest("[data-markdown-image-src]")
              ?.getAttribute("data-markdown-image-src") || "";
          return source === expectedImageUrl;
        });
        return {
          url: window.location.href,
          hasArticleWorkspace:
            text.includes("Article") || text.includes("文章"),
          hasSlotMarker:
            editableText.includes("article-inline-image-slot-e2e") ||
            text.includes("article-inline-image-slot-e2e"),
          hasPendingProtocol:
            editableText.includes("pending-image-task://") ||
            text.includes("pending-image-task://"),
          hasImageUrl:
            editableText.includes(expectedImageUrl) ||
            text.includes(expectedImageUrl) ||
            images.some(
              (image) =>
                image.src === expectedImageUrl ||
                image.resolvedSrc === expectedImageUrl,
            ) ||
            imageDiagnosticSources.includes(expectedImageUrl),
          hasRenderedImage: matchingImages.length > 0,
          hasRenderedImageNearAnchor,
          hasUnavailablePlaceholderForExpected:
            expectedUnavailablePlaceholders.length > 0,
          imageCount: images.length,
          matchingImages,
          matchingImageContexts,
          imageDiagnosticSources: imageDiagnosticSources.filter(
            (source) => source === expectedImageUrl,
          ),
          bodyText: text.slice(0, 3000),
        };
      },
      {
        imageUrl: CONTENT_FACTORY_INLINE_IMAGE_URL,
        anchorText: INLINE_ANCHOR_TEXT,
        sectionTitle: INLINE_SECTION_TITLE,
      },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.hasRenderedImage &&
      snapshot.hasRenderedImageNearAnchor &&
      !snapshot.hasUnavailablePlaceholderForExpected &&
      !snapshot.hasPendingProtocol
    ) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `inline image fixture 未看到原位替换结果: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

function summarizeMediaTaskArtifact(invocation) {
  const result = invocation?.result ?? {};
  const record = asRecord(result.record) ?? {};
  const payload = asRecord(record.payload) ?? {};
  const relationships = asRecord(record.relationships) ?? {};
  const resultRecord = asRecord(record.result) ?? {};
  const images = readArray(resultRecord.images);
  return sanitizeJson({
    taskId: readString(result.task_id, result.taskId),
    taskType: readString(result.task_type, result.taskType),
    status: readString(result.status),
    normalizedStatus: readString(
      result.normalized_status,
      result.normalizedStatus,
    ),
    artifactPath: readString(result.artifact_path, result.artifactPath),
    absolutePath: readString(result.absolute_path, result.absolutePath),
    payloadUsage: readString(payload.usage),
    payloadSlotId: readString(payload.slot_id, payload.slotId),
    relationshipSlotId: readString(relationships.slot_id, relationships.slotId),
    firstImageUrl: readString(images[0]?.url),
  });
}

function asRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value
    : null;
}

function mergeRecords(...records) {
  return Object.assign({}, ...records.filter(Boolean));
}

function readArray(...values) {
  for (const value of values) {
    if (Array.isArray(value)) {
      return value;
    }
  }
  return [];
}

function readString(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}
