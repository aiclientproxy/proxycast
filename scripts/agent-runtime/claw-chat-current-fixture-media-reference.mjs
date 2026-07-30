import fs from "node:fs";
import path from "node:path";
import { APP_SERVER_METHOD_SESSION_READ } from "./claw-chat-current-fixture-constants.mjs";
import { sendPromptFromGui } from "./claw-chat-current-fixture-gui-actions.mjs";
import { waitForGuiChatCompleted } from "./claw-chat-current-fixture-gui-completion-waits.mjs";
import { readModelLatestTurnStatus } from "./claw-chat-current-fixture-read-model-core.mjs";
import {
  evaluatePageSnapshot,
  invokeAppServerFromPage,
} from "./claw-chat-current-fixture-rpc.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

export const MEDIA_REFERENCE_SCENARIO = "media-reference";
export const MEDIA_REFERENCE_PROMPT = "验证媒体引用展示";
export const MEDIA_REFERENCE_DONE_TEXT = "CLAW_MEDIA_REFERENCE_FIXTURE_DONE";
export const MEDIA_REFERENCE_SUMMARY_TEXT = "媒体引用已进入对话";
export const MEDIA_REFERENCE_CAPTION = "fixture-media-reference.png";
export const MEDIA_REFERENCE_URI = "fixture-media-reference.png";
export const MEDIA_REFERENCE_SAFE_URI_PREFIX = "sidecar://media/output-";
export const MEDIA_REFERENCE_MIME_TYPE = "image/png";
export const MEDIA_REFERENCE_TITLE = "fixture-media-reference.png";
export const MEDIA_REFERENCE_SHA256 = "sha256-fixture-image-1";
export const MEDIA_REFERENCE_BYTE_SIZE = 2048;
const APP_SERVER_METHOD_MEDIA_READ = "media/read";
const MEDIA_READ_MAX_BYTES = 8 * 1024 * 1024;

export const MEDIA_REFERENCE_ASSERTION_KEYS = [
  "mediaReferencePromptReachedBackend",
  "guiMediaReferenceInputSubmitted",
  "guiMediaReferenceCardVisible",
  "guiMediaReferenceDoesNotExposeInlinePayload",
  "guiMediaReferenceUsesSafeSidecarHandle",
  "guiMediaReferenceSourcePathNotExposed",
  "guiMediaReferencePreviewOpened",
  "appServerMediaReadV2Succeeded",
  "appServerMediaReadThreadScoped",
  "guiMediaReferenceUnavailableFallbackVisible",
  "readModelMediaReferenceCompleted",
  "readModelMediaReferenceObserved",
];

function collectImageViews(value, collector = []) {
  if (Array.isArray(value)) {
    for (const item of value) {
      collectImageViews(item, collector);
    }
    return collector;
  }
  if (!value || typeof value !== "object") {
    return collector;
  }

  if (value.type === "imageView" && typeof value.path === "string") {
    collector.push({ path: value.path });
  }

  for (const item of Object.values(value)) {
    collectImageViews(item, collector);
  }
  return collector;
}

export function summarizeReadModelMediaReference(
  readModel,
  referenceUri,
  sourcePath = null,
) {
  const serialized = JSON.stringify(readModel || {});
  const imageViews = collectImageViews(readModel);
  const expectedReferenceUri =
    typeof referenceUri === "string" ? referenceUri : "";
  const matchingViews = imageViews.filter(
    (item) => item.path === expectedReferenceUri,
  );
  return sanitizeJson({
    detailItemCount: Array.isArray(readModel?.detail?.items)
      ? readModel.detail.items.length
      : null,
    latestTurnStatus: readModelLatestTurnStatus(readModel),
    includesPrompt: serialized.includes(MEDIA_REFERENCE_PROMPT),
    includesAssistantDone: serialized.includes(MEDIA_REFERENCE_DONE_TEXT),
    includesAssistantSummary: serialized.includes(MEDIA_REFERENCE_SUMMARY_TEXT),
    contentPartsKeyObserved: false,
    imageViewCount: imageViews.length,
    imageViewPaths: imageViews.map((item) => item.path),
    matchingImageViewCount: matchingViews.length,
    hasMediaReference: matchingViews.length > 0,
    hasReferenceUri:
      expectedReferenceUri.length > 0 &&
      serialized.includes(expectedReferenceUri),
    usesSafeSidecarHandle: expectedReferenceUri.startsWith(
      MEDIA_REFERENCE_SAFE_URI_PREFIX,
    ),
    sourcePathNotExposed:
      typeof sourcePath !== "string" ||
      sourcePath.length === 0 ||
      !serialized.includes(sourcePath),
    hasMimeType: false,
    hasCaption: false,
    hasSourceOwner: matchingViews.some(
      (item) => item.path === expectedReferenceUri,
    ),
    noInlinePayload:
      !serialized.includes("data:image") && !serialized.includes("base64,"),
  });
}

export async function waitForSessionReadMediaReferenceCompleted(
  page,
  options,
  requestLog,
  referenceUri,
  sourcePath,
  threadId,
) {
  const startedAt = Date.now();
  let lastSummary = null;
  let lastRead = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      {
        threadId,
        includeTurns: true,
      },
      requestLog,
    );
    lastRead = read.result;
    lastSummary = summarizeReadModelMediaReference(
      lastRead,
      referenceUri,
      sourcePath,
    );
    if (
      lastSummary.includesPrompt === true &&
      (lastSummary.includesAssistantDone === true ||
        lastSummary.includesAssistantSummary === true) &&
      lastSummary.hasMediaReference === true &&
      lastSummary.hasSourceOwner === true &&
      lastSummary.usesSafeSidecarHandle === true &&
      lastSummary.sourcePathNotExposed === true &&
      lastSummary.noInlinePayload === true
    ) {
      return {
        readModel: lastRead,
        summary: lastSummary,
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `App Server media reference read model 未完成闭环: ${JSON.stringify(
      sanitizeJson({
        summary: lastSummary,
        readModel: lastRead,
      }),
    )}`,
  );
}

export async function summarizeGuiMediaReferenceSnapshot(
  page,
  referenceUri = null,
  sourcePath = null,
) {
  return await evaluatePageSnapshot(
    page,
    ({
      caption,
      visibleUri,
      mimeType,
      expectedReferenceUri,
      safeUriPrefix,
      sourcePath: localSourcePath,
    }) => {
      const text = document.body?.innerText || "";
      const cards = Array.from(
        document.querySelectorAll(
          '[data-testid="streaming-media-reference-card"]',
        ),
      );
      const matchingCard =
        (expectedReferenceUri
          ? cards.find(
              (card) =>
                card.getAttribute("data-reference-uri") ===
                expectedReferenceUri,
            )
          : null) ??
        cards[0] ??
        null;
      const cardText = matchingCard?.textContent || "";
      const cardReferenceUri = matchingCard?.getAttribute("data-reference-uri");
      const shell = document.querySelector(
        [
          '[data-testid="artifact-workbench-shell"]',
          '[data-testid="canvas-workbench-shell"]',
          '[data-testid="canvas-workbench-preview-mode-panel"]',
          '[data-testid="canvas-workbench-markdown-preview"]',
        ].join(", "),
      );
      const shellRect = shell?.getBoundingClientRect();
      const shellStyle = shell ? window.getComputedStyle(shell) : null;
      const shellVisible = Boolean(
        shell &&
        shellRect &&
        shellRect.width > 32 &&
        shellRect.height > 32 &&
        shellStyle?.visibility !== "hidden" &&
        shellStyle?.display !== "none",
      );
      const mainArea = document.querySelector(
        '[data-testid="workspace-main-area"]',
      );
      const layoutMode =
        mainArea instanceof HTMLElement
          ? mainArea.dataset.layoutMode || null
          : null;
      const previewTextIncludesSidecarSource = text.includes(
        "media sidecar source",
      );
      const previewTextIncludesReference = text.includes(visibleUri);
      const previewImage = document.querySelector(
        '[data-testid="preview-artifact-image"]',
      );
      const previewImageRect = previewImage?.getBoundingClientRect();
      const previewImageStyle = previewImage
        ? window.getComputedStyle(previewImage)
        : null;
      const previewImageVisible = Boolean(
        previewImage &&
        previewImageRect &&
        previewImageRect.width > 8 &&
        previewImageRect.height > 8 &&
        previewImageStyle?.visibility !== "hidden" &&
        previewImageStyle?.display !== "none",
      );
      const markdownPreview = document.querySelector(
        '[data-testid="canvas-workbench-markdown-preview"]',
      );
      const markdownPreviewRect = markdownPreview?.getBoundingClientRect();
      const markdownPreviewStyle = markdownPreview
        ? window.getComputedStyle(markdownPreview)
        : null;
      const markdownPreviewVisible = Boolean(
        markdownPreview &&
        markdownPreviewRect &&
        markdownPreviewRect.width > 8 &&
        markdownPreviewRect.height > 8 &&
        markdownPreviewStyle?.visibility !== "hidden" &&
        markdownPreviewStyle?.display !== "none",
      );
      const canvasWorkbenchVisible =
        shellVisible || layoutMode === "chat-canvas";
      const workbenchPreviewVisible =
        canvasWorkbenchVisible &&
        (previewImageVisible ||
          (previewTextIncludesSidecarSource && previewTextIncludesReference));
      return {
        url: window.location.href,
        cardCount: cards.length,
        hasCard: Boolean(matchingCard),
        hasCaption: cardText.includes(caption),
        hasUri: cardText.includes(visibleUri),
        hasMimeType: cardText.includes(mimeType),
        referenceUri: cardReferenceUri,
        referenceUriMatchesExpected:
          !expectedReferenceUri || cardReferenceUri === expectedReferenceUri,
        referenceUriUsesSafeSidecar:
          typeof cardReferenceUri === "string" &&
          cardReferenceUri.startsWith(safeUriPrefix),
        cardTextIncludesSafeHandle: cardText.includes(safeUriPrefix),
        bodyTextIncludesSourcePath:
          typeof localSourcePath === "string" &&
          localSourcePath.length > 0 &&
          text.includes(localSourcePath),
        cardReferenceIncludesSourcePath:
          typeof localSourcePath === "string" &&
          localSourcePath.length > 0 &&
          typeof cardReferenceUri === "string" &&
          cardReferenceUri.includes(localSourcePath),
        cardText,
        bodyTextIncludesInlinePayload:
          text.includes("data:image") || text.includes("base64,"),
        workbenchShellVisible: shellVisible,
        canvasWorkbenchVisible,
        workbenchPreviewVisible,
        previewImageVisible,
        markdownPreviewVisible,
        previewImageSrc:
          previewImage instanceof HTMLImageElement
            ? previewImage.getAttribute("src")
            : null,
        previewImageIncludesSourcePath:
          typeof localSourcePath === "string" &&
          localSourcePath.length > 0 &&
          previewImage instanceof HTMLImageElement &&
          (previewImage.getAttribute("src") || "").includes(localSourcePath),
        layoutMode,
        previewTextIncludesSidecarSource,
        previewTextIncludesReference,
        bodyText: text,
      };
    },
    {
      caption: MEDIA_REFERENCE_CAPTION,
      visibleUri: MEDIA_REFERENCE_URI,
      mimeType: MEDIA_REFERENCE_MIME_TYPE,
      expectedReferenceUri: referenceUri,
      safeUriPrefix: MEDIA_REFERENCE_SAFE_URI_PREFIX,
      sourcePath,
    },
  );
}

async function readMediaRequestTrace(page, expectedThreadId) {
  return await evaluatePageSnapshot(
    page,
    ({ method, threadId }) => {
      let entries = [];
      try {
        const parsed = JSON.parse(
          window.localStorage.getItem("lime_invoke_trace_buffer_v1") || "[]",
        );
        entries = Array.isArray(parsed) ? parsed : [];
      } catch {
        entries = [];
      }
      const requests = entries.flatMap((entry) => {
        if (entry?.command !== "app_server_handle_json_lines") {
          return [];
        }
        const lines = entry?.args_preview?.request?.lines;
        if (!Array.isArray(lines)) {
          return [];
        }
        return lines.flatMap((line) => {
          try {
            const request = JSON.parse(String(line));
            if (request?.method !== method) {
              return [];
            }
            const params =
              request.params && typeof request.params === "object"
                ? request.params
                : {};
            return [
              {
                command: entry.command,
                status: entry.status ?? null,
                method: request.method,
                threadId: params.threadId ?? null,
                hasSessionId: Object.prototype.hasOwnProperty.call(
                  params,
                  "sessionId",
                ),
                stream: params.stream ?? null,
                uri: params.uri ?? null,
              },
            ];
          } catch {
            return [];
          }
        });
      });
      return {
        requestCount: requests.length,
        requests,
        allUseExpectedThread:
          requests.length > 0 &&
          requests.every((request) => request.threadId === threadId),
        noLegacySessionIdentity:
          requests.length > 0 &&
          requests.every((request) => request.hasSessionId === false),
      };
    },
    { method: APP_SERVER_METHOD_MEDIA_READ, threadId: expectedThreadId },
  );
}

function summarizeMediaReadSuccess(result, expectedThreadId) {
  const sidecarRef = result?.sidecarRef;
  const relativePath = sidecarRef?.relativePath;
  if (
    result?.threadId !== expectedThreadId ||
    typeof result?.contentBase64 !== "string" ||
    result.contentBase64.length === 0 ||
    typeof relativePath !== "string" ||
    relativePath.length === 0
  ) {
    throw new Error(
      `media/read 未返回 thread-scoped 可读 sidecar: ${JSON.stringify(sanitizeJson(result))}`,
    );
  }
  return sanitizeJson({
    threadId: result.threadId,
    uri: result.uri ?? null,
    mimeType: result.mimeType ?? null,
    bytes: result.bytes ?? null,
    totalBytes: result.totalBytes ?? null,
    offset: result.offset ?? null,
    length: result.length ?? null,
    hasMore: result.hasMore ?? null,
    sha256: result.sha256 ?? null,
    sidecarRef,
    contentBase64Present: true,
  });
}

function makeMediaSidecarUnavailable(runtimeEnv, relativePath) {
  const sidecarRoot = path.resolve(runtimeEnv.agentRoot, "runtime", "sidecar");
  const sidecarPath = path.resolve(sidecarRoot, relativePath);
  const relativeToRoot = path.relative(sidecarRoot, sidecarPath);
  if (
    !relativeToRoot ||
    relativeToRoot.startsWith("..") ||
    path.isAbsolute(relativeToRoot) ||
    !fs.existsSync(sidecarPath)
  ) {
    throw new Error(`media sidecar 临时文件不可用: ${sidecarPath}`);
  }
  const originalBytes = fs.statSync(sidecarPath).size;
  fs.writeFileSync(sidecarPath, Buffer.alloc(0));
  return sanitizeJson({
    sidecarPath,
    relativePath,
    originalBytes,
    unavailableBytes: fs.statSync(sidecarPath).size,
  });
}

export async function openGuiMediaReferencePreview(
  page,
  options,
  referenceUri,
  sourcePath,
) {
  const clickSnapshot = await evaluatePageSnapshot(
    page,
    ({ uri }) => {
      const cards = Array.from(
        document.querySelectorAll(
          '[data-testid="streaming-media-reference-card"]',
        ),
      );
      const card =
        cards.find(
          (candidate) => candidate.getAttribute("data-reference-uri") === uri,
        ) ??
        cards[0] ??
        null;
      if (card instanceof HTMLElement) {
        card.click();
        return {
          clicked: true,
          cardText: card.textContent || "",
          referenceUri: card.getAttribute("data-reference-uri"),
        };
      }
      return {
        clicked: false,
        cardText: "",
        referenceUri: null,
      };
    },
    { uri: referenceUri },
  );

  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await summarizeGuiMediaReferenceSnapshot(
      page,
      referenceUri,
      sourcePath,
    );
    lastSnapshot = snapshot;
    if (
      snapshot?.workbenchPreviewVisible === true &&
      snapshot.previewImageVisible === true &&
      snapshot.previewTextIncludesSidecarSource === false
    ) {
      return sanitizeJson({
        click: clickSnapshot,
        preview: snapshot,
      });
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `media reference Workbench 预览未打开: ${JSON.stringify(
      sanitizeJson({
        click: clickSnapshot,
        preview: lastSnapshot,
      }),
    )}`,
  );
}

async function openGuiMediaReferenceUnavailableFallback(
  page,
  options,
  referenceUri,
  sourcePath,
) {
  const click = await evaluatePageSnapshot(
    page,
    ({ uri }) => {
      const card = Array.from(
        document.querySelectorAll(
          '[data-testid="streaming-media-reference-card"]',
        ),
      ).find(
        (candidate) => candidate.getAttribute("data-reference-uri") === uri,
      );
      if (!(card instanceof HTMLElement)) {
        return { clicked: false, referenceUri: null };
      }
      card.click();
      return {
        clicked: true,
        referenceUri: card.getAttribute("data-reference-uri"),
      };
    },
    { uri: referenceUri },
  );

  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await summarizeGuiMediaReferenceSnapshot(
      page,
      referenceUri,
      sourcePath,
    );
    lastSnapshot = snapshot;
    if (
      snapshot?.workbenchPreviewVisible === true &&
      snapshot.markdownPreviewVisible === true &&
      snapshot.previewImageVisible === false &&
      snapshot.previewTextIncludesReference === true &&
      snapshot.bodyTextIncludesInlinePayload === false
    ) {
      return sanitizeJson({ click, preview: snapshot });
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `media/read 不可用态未显示 Workbench fallback: ${JSON.stringify(
      sanitizeJson({ click, preview: lastSnapshot }),
    )}`,
  );
}

export async function runMediaReferenceScenario({
  page,
  options,
  appServerRequests,
  runtimeEnv,
}) {
  const result = {};
  const sourcePath = runtimeEnv.mediaReferenceSourcePath;

  result.mediaReferenceInputSend = sanitizeJson(
    await sendPromptFromGui(page, options, MEDIA_REFERENCE_PROMPT),
  );

  result.guiMediaReferenceCompleted = sanitizeJson(
    await waitForGuiChatCompleted(page, options, {
      prompt: MEDIA_REFERENCE_PROMPT,
      doneText: MEDIA_REFERENCE_DONE_TEXT,
      summaryText: MEDIA_REFERENCE_SUMMARY_TEXT,
      requiredVisibleTexts: [MEDIA_REFERENCE_TITLE, MEDIA_REFERENCE_MIME_TYPE],
      disallowedVisibleTexts: ["data:image", "base64,"],
    }),
  );
  const guiSnapshot = await summarizeGuiMediaReferenceSnapshot(
    page,
    null,
    sourcePath,
  );
  result.guiMediaReferenceSnapshot = sanitizeJson(guiSnapshot);
  const referenceUri = guiSnapshot?.referenceUri;

  const readModel = await waitForSessionReadMediaReferenceCompleted(
    page,
    options,
    appServerRequests,
    referenceUri,
    sourcePath,
    options.threadId,
  );
  result.readModelMediaReferenceCompleted = readModel.summary;

  const mediaReadParams = {
    threadId: options.threadId,
    uri: referenceUri,
    offset: 0,
    length: MEDIA_READ_MAX_BYTES,
    maxBytes: MEDIA_READ_MAX_BYTES,
    stream: true,
  };
  const mediaRead = await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_MEDIA_READ,
    mediaReadParams,
    appServerRequests,
  );
  result.mediaReadV2Success = summarizeMediaReadSuccess(
    mediaRead.result,
    options.threadId,
  );

  result.guiMediaReferencePreview = sanitizeJson(
    await openGuiMediaReferencePreview(page, options, referenceUri, sourcePath),
  );

  result.mediaReadUnavailableMutation = makeMediaSidecarUnavailable(
    runtimeEnv,
    result.mediaReadV2Success.sidecarRef.relativePath,
  );
  result.guiMediaReferenceUnavailableFallback =
    await openGuiMediaReferenceUnavailableFallback(
      page,
      options,
      referenceUri,
      sourcePath,
    );
  result.mediaReadV2Trace = sanitizeJson(
    await readMediaRequestTrace(page, options.threadId),
  );

  return result;
}
