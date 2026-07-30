import fs from "node:fs";
import path from "node:path";
import {
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_GET,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_LIST,
  APP_SERVER_METHOD_SESSION_READ,
  IMAGE_COMMAND_CREATE_TASK_TOOL_NAME,
  IMAGE_COMMAND_DONE_TEXT,
  IMAGE_COMMAND_IMAGE_PROMPT,
  IMAGE_COMMAND_PRESENTATION_CAPTION,
  IMAGE_COMMAND_PRESENTATION_INTRO,
  IMAGE_COMMAND_PROMPT,
  PLAIN_IMAGE_INTENT_IMAGE_PROMPT,
  PLAIN_IMAGE_INTENT_PROMPT,
  PLAIN_IMAGE_INTENT_ROUTED_PROMPT,
  PLAIN_IMAGE_INTENT_SCENARIO,
  IMAGE_FIXTURE_MODEL,
} from "./claw-chat-current-fixture-constants.mjs";
import { collectAgentUiPerformanceTraceEvidence } from "./claw-chat-current-fixture-agent-ui-trace.mjs";
import { sendPromptFromGui } from "./claw-chat-current-fixture-gui-actions.mjs";
import { readImageCommandWorkflowAudit } from "./claw-chat-current-fixture-image-command-workflow-read.mjs";
import {
  collectReadModelToolCalls,
  readModelLatestTurnStatus,
} from "./claw-chat-current-fixture-read-model-core.mjs";
import {
  evaluatePageSnapshot,
  invokeAppServerFromPage,
  reloadRendererDocument,
  waitForRendererReady,
} from "./claw-chat-current-fixture-rpc.mjs";
import { openFixtureSessionFromSidebar } from "./claw-chat-current-fixture-session.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

const IMAGE_COMMAND_TERMINAL_STATUS = "succeeded";
const IMAGE_COMMAND_WORKER_ID = "lime-image-api-worker";
const IMAGE_COMMAND_TOOL_LABEL = "Image Generation";
const IMAGE_COMMAND_TOOL_LABELS = [IMAGE_COMMAND_TOOL_LABEL, "图片生成"];
const IMAGE_COMMAND_MODEL_LABEL = "Nanobanana Pro";
const IMAGE_COMMAND_TOKEN_LABEL = "31.0K Tokens";
const EXPECTED_IMAGE_TASK_AUDIT_EVENTS = [
  "worker_loaded",
  "task_queued",
  "task_running",
  "request_slot_started",
  "request_slot_succeeded",
  "task_succeeded",
];
const IMAGE_TASK_AUDIT_FORBIDDEN_MARKERS = [
  "Authorization",
  "Bearer",
  "api_key",
  "apikey",
  "x-api-key",
  "LOCAL_IMAGE_SERVER_API_KEY",
  "test-key",
];

export function resolveImageIntentScenario(scenario) {
  if (scenario === PLAIN_IMAGE_INTENT_SCENARIO) {
    return {
      scenario,
      inputPrompt: PLAIN_IMAGE_INTENT_PROMPT,
      routedPrompt: PLAIN_IMAGE_INTENT_ROUTED_PROMPT,
      imagePrompt: PLAIN_IMAGE_INTENT_IMAGE_PROMPT,
      taskTitle: "广州夏天 E2E",
      entrySource: "at_image_command",
    };
  }

  return {
    scenario: "image-command",
    inputPrompt: IMAGE_COMMAND_PROMPT,
    routedPrompt: IMAGE_COMMAND_PROMPT,
    imagePrompt: IMAGE_COMMAND_IMAGE_PROMPT,
    taskTitle: "青柠插画 E2E",
    entrySource: "at_image_command",
  };
}

function traceEvidenceHasProviderAndClient(evidence) {
  return (
    evidence?.hasProviderWaitMs === true &&
    evidence?.hasClientLocalOutputMs === true
  );
}

function taskOutputCandidate(taskArtifact) {
  return (
    taskArtifact?.result ?? taskArtifact?.taskArtifact ?? taskArtifact ?? {}
  );
}

function imageTaskId(taskArtifact) {
  const task = taskOutputCandidate(taskArtifact);
  return task.task_id ?? task.taskId ?? task.record?.task_id ?? null;
}

function imageTaskPath(taskArtifact) {
  const task = taskOutputCandidate(taskArtifact);
  return (
    task.absolute_path ??
    task.absolutePath ??
    task.absolute_artifact_path ??
    task.absoluteArtifactPath ??
    task.path ??
    task.artifact_path ??
    null
  );
}

function imageTaskArtifactPath(taskArtifact) {
  const task = taskOutputCandidate(taskArtifact);
  return task.artifact_path ?? task.artifactPath ?? task.path ?? null;
}

export function serializeReadModelSummary(
  readModel,
  scenarioConfig = resolveImageIntentScenario("image-command"),
) {
  const serialized = JSON.stringify(readModel || {});
  const createTaskToolCall = collectReadModelToolCalls(readModel).find(
    (toolCall) =>
      String(toolCall.tool_name ?? toolCall.toolName ?? toolCall.name ?? "") ===
      IMAGE_COMMAND_CREATE_TASK_TOOL_NAME,
  );
  const createTaskOutputValue =
    createTaskToolCall?.output ??
    createTaskToolCall?.output_preview ??
    createTaskToolCall?.outputPreview ??
    "";
  const createTaskOutput =
    typeof createTaskOutputValue === "string"
      ? createTaskOutputValue
      : JSON.stringify(createTaskOutputValue);

  return sanitizeJson({
    detailItemCount: Array.isArray(readModel?.detail?.items)
      ? readModel.detail.items.length
      : null,
    toolCallCount: collectReadModelToolCalls(readModel).length,
    latestTurnStatus: readModelLatestTurnStatus(readModel),
    includesPrompt: serialized.includes(scenarioConfig.routedPrompt),
    includesAssistantDone: serialized.includes(IMAGE_COMMAND_DONE_TEXT),
    includesWorkflowSource: serialized.includes("image_command_workflow"),
    includesCreateTaskTool: Boolean(createTaskToolCall),
    includesTaskArtifactPath:
      serialized.includes(".lime/tasks/image_generate") ||
      serialized.includes("image_generate"),
    includesTaskIdPlaceholder: serialized.includes("{task_id}"),
    includesDraftTask: serialized.includes("draft-image-"),
    createTaskToolStatus: createTaskToolCall?.status ?? null,
    createTaskOutputContainsTaskId: createTaskOutput.includes("task_id"),
    createTaskOutputContainsTaskFile:
      createTaskOutput.includes(".lime/tasks/image_generate") ||
      createTaskOutput.includes("image_generate"),
  });
}

export async function waitForGuiImageCommandCompleted(
  page,
  options,
  scenarioConfig = resolveImageIntentScenario(options.scenario),
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({
        prompt,
        imagePrompt,
        doneText,
        createTaskToolName,
        toolLabel,
        presentationIntro,
        presentationCaption,
      }) => {
        const text = document.body?.innerText || "";
        const textarea = document.querySelector(
          'textarea[name="agent-chat-message"]',
        );
        const rect = textarea?.getBoundingClientRect();
        const style = textarea ? window.getComputedStyle(textarea) : null;
        const textareaVisible = Boolean(
          textarea &&
          rect &&
          rect.width > 16 &&
          rect.height > 16 &&
          style?.visibility !== "hidden" &&
          style?.display !== "none",
        );
        const buttons = Array.from(document.querySelectorAll("button")).map(
          (button) => ({
            title: button.getAttribute("title") || "",
            text: button.textContent || "",
            aria: button.getAttribute("aria-label") || "",
            disabled: button.disabled,
          }),
        );
        const stopButtonVisible = buttons.some((button) => {
          const label = [button.title, button.text, button.aria].join("\n");
          return (
            !button.disabled &&
            (label.includes("停止") ||
              label.includes("终止") ||
              /\bStop\b/i.test(label))
          );
        });
        const processGroups = Array.from(
          document.querySelectorAll('[data-testid="streaming-process-group"]'),
        ).map((group) => ({
          text: group.textContent || "",
          expanded:
            group.querySelector("button")?.getAttribute("aria-expanded") || "",
        }));
        const inlineToolSteps = Array.from(
          document.querySelectorAll('[data-testid="inline-tool-process-step"]'),
        ).map((step) => ({
          text: step.textContent || "",
          grouped: step.getAttribute("data-grouped") || "",
        }));
        const combinedProcessText = [...processGroups, ...inlineToolSteps]
          .map((entry) => entry.text)
          .join("\n");
        const bodyLower = text.toLowerCase();
        const hasPrompt =
          text.includes(prompt) ||
          (text.includes("@配图") && text.includes(imagePrompt));
        const hasVisibleImageTaskProcess =
          text.includes("图片生成") ||
          text.includes(toolLabel) ||
          combinedProcessText.includes("图片生成") ||
          combinedProcessText.includes(toolLabel);
        const hasAssistantSummary =
          text.includes(presentationIntro) || text.includes(presentationCaption);
        const imageTaskCardVisible =
          text.includes(".lime/tasks/image_generate") ||
          text.includes("image_generate") ||
          bodyLower.includes("pending_submit") ||
          bodyLower.includes("图片任务") ||
          hasVisibleImageTaskProcess;

        return {
          url: window.location.href,
          hasPrompt,
          hasAssistantSummary,
          hasPresentationIntro: text.includes(presentationIntro),
          hasPresentationCaption: text.includes(presentationCaption),
          hasDoneText: text.includes(doneText),
          hasWorkflowSource:
            text.includes("image_command_workflow") ||
            combinedProcessText.includes("image_command_workflow"),
          hasCreateTaskTool:
            text.includes(createTaskToolName) ||
            combinedProcessText.includes(createTaskToolName) ||
            hasVisibleImageTaskProcess,
          hasVisibleImageTaskProcess,
          hasTaskArtifactPath: text.includes(".lime/tasks/image_generate"),
          imageTaskCardVisible,
          processGroupCount: processGroups.length,
          inlineToolStepCount: inlineToolSteps.length,
          processText: combinedProcessText,
          templateTaskIdVisible: text.includes("{task_id}"),
          draftImageVisible: text.includes("draft-image-"),
          textareaVisible,
          textareaDisabled:
            textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
          textareaValue:
            textarea instanceof HTMLTextAreaElement ? textarea.value : null,
          stopButtonVisible,
          bodyText: text,
        };
      },
      {
        prompt: scenarioConfig.inputPrompt,
        imagePrompt: scenarioConfig.imagePrompt,
        doneText: IMAGE_COMMAND_DONE_TEXT,
        createTaskToolName: IMAGE_COMMAND_CREATE_TASK_TOOL_NAME,
        toolLabel: IMAGE_COMMAND_TOOL_LABEL,
        presentationIntro: IMAGE_COMMAND_PRESENTATION_INTRO,
        presentationCaption: IMAGE_COMMAND_PRESENTATION_CAPTION,
      },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.hasPrompt &&
      snapshot.hasCreateTaskTool &&
      snapshot.hasVisibleImageTaskProcess &&
      snapshot.imageTaskCardVisible &&
      snapshot.templateTaskIdVisible === false &&
      snapshot.draftImageVisible === false &&
      snapshot.textareaVisible &&
      snapshot.textareaDisabled === false &&
      snapshot.stopButtonVisible === false
    ) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Claw GUI 未完成 @配图 current 主链验收: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

export async function waitForSessionReadImageCommandCompleted(
  page,
  options,
  requestLog,
  scenarioConfig = resolveImageIntentScenario(options.scenario),
) {
  const startedAt = Date.now();
  let lastRead = null;
  let lastSummary = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      {
        threadId: options.threadId,
        includeTurns: true,
      },
      requestLog,
    );
    lastRead = read.result;
    lastSummary = serializeReadModelSummary(lastRead, scenarioConfig);
    if (
      JSON.stringify(lastRead || {}).includes(scenarioConfig.routedPrompt) &&
      lastSummary.latestTurnStatus === "completed" &&
      lastSummary.includesCreateTaskTool === true &&
      lastSummary.createTaskOutputContainsTaskId === true &&
      lastSummary.createTaskOutputContainsTaskFile === true &&
      lastSummary.includesTaskIdPlaceholder === false &&
      lastSummary.includesDraftTask === false
    ) {
      return {
        readModel: lastRead,
        summary: lastSummary,
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `App Server image-command read model 未完成工具链闭环: ${JSON.stringify(
      sanitizeJson({
        summary: lastSummary,
        readModel: lastRead,
      }),
    )}`,
  );
}

function imageTaskRecord(taskArtifact) {
  const task = taskOutputCandidate(taskArtifact);
  return task.record &&
    typeof task.record === "object" &&
    !Array.isArray(task.record)
    ? task.record
    : null;
}

function imageTaskPayload(taskArtifact) {
  const payload = imageTaskRecord(taskArtifact)?.payload;
  return payload && typeof payload === "object" && !Array.isArray(payload)
    ? payload
    : {};
}

function optionalPayloadString(payload, keys) {
  for (const key of keys) {
    const value = payload?.[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function imageTaskMatchesScenario(
  taskArtifact,
  scenarioConfig,
  providerId,
  expectedSessionId,
) {
  const task = taskOutputCandidate(taskArtifact);
  if (
    task.task_type !== "image_generate" &&
    task.taskType !== "image_generate"
  ) {
    return false;
  }
  const payload = imageTaskPayload(taskArtifact);
  const sessionId = optionalPayloadString(payload, ["session_id", "sessionId"]);
  const rawText = optionalPayloadString(payload, ["raw_text", "rawText"]);
  const prompt = optionalPayloadString(payload, ["prompt"]);
  const entrySource = optionalPayloadString(payload, [
    "entry_source",
    "entrySource",
  ]);
  const provider = optionalPayloadString(payload, [
    "provider_id",
    "providerId",
  ]);
  const model = optionalPayloadString(payload, ["model"]);
  return (
    sessionId === expectedSessionId &&
    rawText === scenarioConfig.routedPrompt &&
    prompt === scenarioConfig.imagePrompt &&
    entrySource === scenarioConfig.entrySource &&
    (!providerId || provider === providerId) &&
    model === IMAGE_FIXTURE_MODEL
  );
}

async function waitForImageCommandWorkflowTaskArtifact({
  page,
  workspace,
  appServerRequests,
  options,
  imageFixtureProvider,
  scenarioConfig,
}) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const [listResult, sessionRead] = await Promise.all([
      invokeAppServerFromPage(
        page,
        APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_LIST,
        {
          projectRootPath: workspace.rootPath,
          taskType: "image_generate",
          limit: 20,
        },
        appServerRequests,
      ),
      invokeAppServerFromPage(
        page,
        APP_SERVER_METHOD_SESSION_READ,
        {
          threadId: options.threadId,
          includeTurns: true,
        },
        appServerRequests,
      ),
    ]);
    const tasks = Array.isArray(listResult?.result?.tasks)
      ? listResult.result.tasks
      : [];
    const matched = tasks.find((task) =>
      imageTaskMatchesScenario(
        task,
        scenarioConfig,
        imageFixtureProvider?.providerId ?? null,
        options.sessionId,
      ),
    );
    lastSnapshot = sanitizeJson({
      taskCount: tasks.length,
      matchedTaskId: imageTaskId(matched),
      workflowTaskIds: tasks.map((task) => imageTaskId(task)).filter(Boolean),
      providerId: imageFixtureProvider?.providerId ?? null,
      model: IMAGE_FIXTURE_MODEL,
      prompt: scenarioConfig.imagePrompt,
      rawText: scenarioConfig.routedPrompt,
      entrySource: scenarioConfig.entrySource,
      listResult: listResult?.result ?? null,
      createFailure: summarizeImageCommandCreateFailure(sessionRead?.result),
    });
    if (matched) {
      return {
        response: matched,
        listSummary: lastSnapshot,
      };
    }
    if (lastSnapshot.createFailure) {
      throw new Error(
        `ImageCommandWorkflow 创建 image task 失败: ${JSON.stringify(
          lastSnapshot.createFailure,
        )}`,
      );
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `ImageCommandWorkflow 未创建匹配的 image task artifact: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

export function summarizeImageCommandCreateFailure(readModel) {
  const failedItem = collectReadModelToolCalls(readModel).find((item) => {
    const toolName =
      item?.tool_name ?? item?.toolName ?? item?.tool ?? item?.name ?? null;
    return (
      toolName === IMAGE_COMMAND_CREATE_TASK_TOOL_NAME &&
      String(item?.status ?? "").toLowerCase() === "failed"
    );
  });
  const failedEvent = findImageTaskCreateFailedEvent(readModel);
  if (!failedItem && !failedEvent) {
    return null;
  }

  const metadata =
    failedItem?.metadata &&
    typeof failedItem.metadata === "object" &&
    !Array.isArray(failedItem.metadata)
      ? failedItem.metadata
      : {};
  const failurePayload =
    failedEvent?.payload &&
    typeof failedEvent.payload === "object" &&
    !Array.isArray(failedEvent.payload)
      ? failedEvent.payload
      : failedEvent ?? {};
  return sanitizeJson({
    reasonCode:
      optionalPayloadString(failurePayload, ["reasonCode", "reason_code"]) ??
      optionalPayloadString(metadata, ["reasonCode", "reason_code"]),
    message:
      optionalPayloadString(failedItem, [
        "error",
        "output",
        "outputPreview",
        "output_preview",
      ]) ?? optionalPayloadString(failurePayload, ["message", "error"]),
    itemId: optionalPayloadString(failedItem, ["id", "item_id", "itemId"]),
    turnId:
      optionalPayloadString(failedItem, ["turn_id", "turnId"]) ??
      optionalPayloadString(failurePayload, ["turn_id", "turnId"]),
    toolName: IMAGE_COMMAND_CREATE_TASK_TOOL_NAME,
  });
}

function findImageTaskCreateFailedEvent(value, depth = 0) {
  if (depth > 8 || value == null) {
    return null;
  }
  if (Array.isArray(value)) {
    return value
      .map((entry) => findImageTaskCreateFailedEvent(entry, depth + 1))
      .find(Boolean);
  }
  if (typeof value !== "object") {
    return null;
  }
  if (
    [value.type, value.event_type, value.eventType].includes(
      "image_task.create_failed",
    )
  ) {
    return value;
  }
  return Object.values(value)
    .map((entry) => findImageTaskCreateFailedEvent(entry, depth + 1))
    .find(Boolean);
}

function readTaskArtifactFile(taskPath) {
  return JSON.parse(fs.readFileSync(taskPath, "utf8"));
}

function hasEventSequence(events, expectedEvents) {
  let searchFrom = 0;
  for (const expectedEvent of expectedEvents) {
    const nextIndex = events.findIndex(
      (event, index) => index >= searchFrom && event === expectedEvent,
    );
    if (nextIndex < 0) {
      return false;
    }
    searchFrom = nextIndex + 1;
  }
  return true;
}

function resolveTaskAuditLogPath(workspaceRoot, logsRef) {
  if (typeof logsRef !== "string" || logsRef.trim().length === 0) {
    return null;
  }
  return path.isAbsolute(logsRef) ? logsRef : path.join(workspaceRoot, logsRef);
}

function readImageCommandTaskAuditLog({ workspace, taskPath }) {
  const record = readTaskArtifactFile(taskPath);
  const attempts = Array.isArray(record.attempts) ? record.attempts : [];
  const currentAttemptIndex = resolveCurrentAttemptIndex(record);
  const currentAttempt =
    currentAttemptIndex >= 0 ? attempts[currentAttemptIndex] : null;
  const logsRef = currentAttempt?.logs_ref ?? currentAttempt?.logsRef ?? null;
  const logsPath = resolveTaskAuditLogPath(workspace.rootPath, logsRef);
  const exists = Boolean(logsPath && fs.existsSync(logsPath));
  const rawText = exists ? fs.readFileSync(logsPath, "utf8") : "";
  const events = [];
  const taskIds = new Set();
  let parseError = null;

  if (rawText.trim().length > 0) {
    for (const line of rawText.trim().split(/\r?\n/)) {
      if (!line.trim()) {
        continue;
      }
      try {
        const event = JSON.parse(line);
        events.push(typeof event?.event === "string" ? event.event : null);
        if (typeof event?.task_id === "string") {
          taskIds.add(event.task_id);
        }
      } catch (error) {
        parseError = String(error?.message ?? error);
      }
    }
  }

  const forbiddenMarkerHits = IMAGE_TASK_AUDIT_FORBIDDEN_MARKERS.filter(
    (marker) => rawText.toLowerCase().includes(marker.toLowerCase()),
  );

  return sanitizeJson({
    taskId: record.task_id ?? record.taskId ?? null,
    attemptId: currentAttempt?.attempt_id ?? currentAttempt?.attemptId ?? null,
    attemptIndex: currentAttemptIndex >= 0 ? currentAttemptIndex + 1 : null,
    logsRef,
    logsRefLooksLikeTaskLog:
      typeof logsRef === "string" &&
      logsRef.startsWith(".lime/task-logs/") &&
      logsRef.endsWith(".jsonl"),
    logsPath,
    exists,
    lineCount:
      rawText.trim().length > 0 ? rawText.trim().split(/\r?\n/).length : 0,
    events,
    expectedEvents: EXPECTED_IMAGE_TASK_AUDIT_EVENTS,
    hasExpectedEventSequence: hasEventSequence(
      events,
      EXPECTED_IMAGE_TASK_AUDIT_EVENTS,
    ),
    hasNoSensitiveTokenMarkers: forbiddenMarkerHits.length === 0,
    forbiddenMarkerHits,
    allEventTaskIdsMatch:
      taskIds.size === 1 && taskIds.has(record.task_id ?? record.taskId ?? ""),
    parseError,
  });
}

function resolveCurrentAttemptIndex(record) {
  const attempts = Array.isArray(record.attempts) ? record.attempts : [];
  if (attempts.length === 0) {
    return -1;
  }
  const currentAttemptId =
    typeof record.current_attempt_id === "string"
      ? record.current_attempt_id
      : null;
  if (currentAttemptId) {
    const matchedIndex = attempts.findIndex(
      (attempt) => attempt?.attempt_id === currentAttemptId,
    );
    if (matchedIndex >= 0) {
      return matchedIndex;
    }
  }
  return attempts.length - 1;
}

function imageCommandTaskRef(taskArtifact) {
  const taskId = imageTaskId(taskArtifact);
  const taskPath = imageTaskPath(taskArtifact);
  if (!taskId || !taskPath) {
    throw new Error(
      `image-command task artifact 缺少 taskId/path: ${JSON.stringify(
        sanitizeJson(taskArtifact),
      )}`,
    );
  }
  if (!fs.existsSync(taskPath)) {
    throw new Error(`image-command task file 不存在: ${taskPath}`);
  }

  return {
    taskId,
    taskPath,
  };
}

async function waitForImageCommandTaskArtifactTerminal({
  page,
  workspace,
  appServerRequests,
  taskArtifact,
  options,
}) {
  const taskRef = imageCommandTaskRef(taskArtifact);
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const getResult = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_GET,
      {
        projectRootPath: workspace.rootPath,
        taskRef: taskRef.taskId,
      },
      appServerRequests,
    );
    const record = imageTaskRecord(getResult?.result);
    const attempts = Array.isArray(record?.attempts) ? record.attempts : [];
    const currentAttemptIndex = resolveCurrentAttemptIndex(record || {});
    const currentAttempt =
      currentAttemptIndex >= 0 ? attempts[currentAttemptIndex] : null;
    const resultImageCount = Array.isArray(record?.result?.images)
      ? record.result.images.length
      : 0;
    lastSnapshot = sanitizeJson({
      taskId: taskRef.taskId,
      taskPath: taskRef.taskPath,
      completeMethodUsed: "media_runtime_worker",
      completeReturned: Boolean(getResult?.result),
      status: record?.status ?? getResult?.result?.status ?? null,
      normalizedStatus:
        record?.normalized_status ??
        getResult?.result?.normalized_status ??
        null,
      sameTaskFileUpdated: true,
      resultImageCount,
      attemptCount: attempts.length,
      currentAttemptStatus: currentAttempt?.status ?? null,
      currentAttemptWorkerId: currentAttempt?.worker_id ?? null,
      currentAttemptHasResultSnapshot: Boolean(currentAttempt?.result_snapshot),
      response: getResult?.result ?? null,
    });
    if (
      lastSnapshot.normalizedStatus === IMAGE_COMMAND_TERMINAL_STATUS &&
      lastSnapshot.resultImageCount >= 1 &&
      lastSnapshot.currentAttemptWorkerId === IMAGE_COMMAND_WORKER_ID
    ) {
      return lastSnapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Claw @配图后端 worker 未把 task artifact 推进终态: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

export async function waitForGuiImageCommandTerminal(
  page,
  options,
  taskId,
  scenarioConfig = resolveImageIntentScenario(options.scenario),
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({
        prompt,
        imagePrompt,
        presentationCaption,
        presentationIntro,
        taskId,
        toolLabels,
        modelLabel,
        tokenLabel,
      }) => {
        const text = document.body?.innerText || "";
        const textarea = document.querySelector(
          'textarea[name="agent-chat-message"]',
        );
        const buttons = Array.from(document.querySelectorAll("button")).map(
          (button) => ({
            title: button.getAttribute("title") || "",
            text: button.textContent || "",
            aria: button.getAttribute("aria-label") || "",
            disabled: button.disabled,
          }),
        );
        const stopButtonVisible = buttons.some((button) => {
          const label = [button.title, button.text, button.aria].join("\n");
          return (
            !button.disabled &&
            (label.includes("停止") ||
              label.includes("终止") ||
              /\bStop\b/i.test(label))
          );
        });
        const cardSelector = `[data-testid="image-workbench-message-preview-${taskId}"]`;
        const introSelector = `[data-testid="image-workbench-assistant-intro-${taskId}"]`;
        const completionSelector = `[data-testid="image-workbench-completion-caption-${taskId}"]`;
        const mediaSelector = `[data-testid="image-workbench-message-preview-single-media-${taskId}"], [data-testid="image-workbench-message-preview-grid-${taskId}"]`;
        const cards = Array.from(document.querySelectorAll(cardSelector));
        const legacyIntroNodes = Array.from(
          document.querySelectorAll(introSelector),
        );
        const canonicalIntroNodes = Array.from(
          document.querySelectorAll(
            '[data-message-role="assistant"][data-thread-item-id]',
          ),
        ).filter((node) =>
          (node.textContent || "").includes(presentationIntro),
        );
        const introNodes =
          legacyIntroNodes.length > 0 ? legacyIntroNodes : canonicalIntroNodes;
        const completionNodes = Array.from(
          document.querySelectorAll(completionSelector),
        );
        const mediaNodes = Array.from(document.querySelectorAll(mediaSelector));
        const cardText = cards.map((card) => card.textContent || "").join("\n");
        const introText = introNodes
          .map((node) => node.textContent || "")
          .join("\n");
        const completionText = completionNodes
          .map((node) => node.textContent || "")
          .join("\n");
        const toolbarSelector = `[data-testid="image-workbench-message-preview-toolbar-${taskId}"]`;
        const toolbarText = Array.from(
          document.querySelectorAll(toolbarSelector),
        )
          .map((toolbar) => toolbar.textContent || "")
          .join("\n");
        const hasPreviewImage = cards.some((card) =>
          Boolean(card.querySelector("img")),
        );
        const imageMetrics = cards.flatMap((card) =>
          Array.from(card.querySelectorAll("img")).map((image) => {
            const rect = image.getBoundingClientRect();
            return {
              complete: image.complete,
              naturalWidth: image.naturalWidth,
              naturalHeight: image.naturalHeight,
              width: rect.width,
              height: rect.height,
              dataImage: image.currentSrc.startsWith("data:image/"),
            };
          }),
        );
        const hasLoadedVisiblePreviewImage = imageMetrics.some(
          (image) =>
            image.complete === true &&
            image.naturalWidth > 1 &&
            image.naturalHeight > 1 &&
            image.width > 16 &&
            image.height > 16,
        );
        const hasNaturalCompletionCaption =
          (completionText.includes("搞定") ||
            completionText.includes("完成了")) &&
          (completionText.includes("已经做好了") ||
            completionText.includes("已经生成")) &&
          !completionText.includes("任务ID") &&
          !completionText.includes("{task_id}");
        const introNode = introNodes[0] ?? null;
        const cardNode = cards[0] ?? null;
        const completionNode = completionNodes[0] ?? null;
        const introBeforeCard = Boolean(
          introNode &&
            cardNode &&
            (introNode.compareDocumentPosition(cardNode) &
              Node.DOCUMENT_POSITION_FOLLOWING),
        );
        const completionAfterCard = Boolean(
          cardNode &&
            completionNode &&
            (cardNode.compareDocumentPosition(completionNode) &
              Node.DOCUMENT_POSITION_FOLLOWING),
        );
        const visiblePendingStatus =
          text.includes("pending_submit") ||
          text.includes("排队中") ||
          cardText.includes("正在生成") ||
          cardText.includes("图片生成中");
        const describeElement = (element) => {
          const rect = element.getBoundingClientRect();
          const style = window.getComputedStyle(element);
          return {
            text: element.textContent || "",
            display: style.display,
            visibility: style.visibility,
            opacity: style.opacity,
            width: rect.width,
            height: rect.height,
          };
        };
        const tokenUsageNodes = Array.from(
          document.querySelectorAll('[data-testid="token-usage-display"]'),
        ).map(describeElement);
        const assistantMetaFooterNodes = Array.from(
          document.querySelectorAll(
            '[data-testid="assistant-message-meta-footer"]',
          ),
        ).map(describeElement);
        const assistantMessageNodes = Array.from(
          document.querySelectorAll('[data-message-role="assistant"]'),
        ).map((element) => ({
          ...describeElement(element),
          messageId: element.getAttribute("data-message-id") || "",
          runtimeTurnId:
            element.getAttribute("data-runtime-turn-id") || "",
          hasImageCard: Boolean(element.querySelector(cardSelector)),
          hasTokenUsage: Boolean(
            element.querySelector('[data-testid="token-usage-display"]'),
          ),
        }));
        return {
          url: window.location.href,
          hasPrompt:
            text.includes(prompt) ||
            (text.includes("@配图") && text.includes(imagePrompt)),
          hasPresentationIntro: text.includes(presentationIntro),
          hasPresentationIntroInAssistantText:
            introText.includes(presentationIntro),
          presentationIntroSource:
            legacyIntroNodes.length > 0
              ? "message_shell"
              : canonicalIntroNodes.length > 0
                ? "canonical_item"
                : null,
          hasToolStripLabel: toolLabels.some((label) =>
            toolbarText.includes(label),
          ),
          hasImageModelLabel: toolbarText.includes(modelLabel),
          hasTokenUsage: text.includes(tokenLabel),
          hasPresentationCaption:
            text.includes(presentationCaption) ||
            completionText.includes(presentationCaption),
          hasPresentationCaptionAfterCard:
            completionText.includes(presentationCaption) && completionAfterCard,
          cardHasPresentationCaption: cardText.includes(presentationCaption),
          introBeforeCard,
          completionAfterCard,
          hasNaturalCompletionCaption,
          templateTaskIdVisible: text.includes("{task_id}"),
          draftImageVisible: text.includes("draft-image-"),
          textareaDisabled:
            textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
          stopButtonVisible,
          cardCount: cards.length,
          mediaCount: mediaNodes.length,
          toolbarText,
          introText,
          completionText,
          hasPreviewImage,
          hasLoadedVisiblePreviewImage,
          imageMetrics,
          cardText,
          taskIdVisible: text.includes(taskId),
          terminalMessageVisible:
            text.includes("图片任务已完成，共生成 1 张") ||
            cardText.includes("已完成") ||
            cardText.includes("生成"),
          visiblePendingStatus,
          tokenUsageNodes,
          assistantMetaFooterNodes,
          assistantMessageNodes,
          bodyText: text,
        };
      },
      {
        prompt: scenarioConfig.inputPrompt,
        imagePrompt: scenarioConfig.imagePrompt,
        presentationCaption: IMAGE_COMMAND_PRESENTATION_CAPTION,
        presentationIntro: IMAGE_COMMAND_PRESENTATION_INTRO,
        taskId,
        toolLabels: IMAGE_COMMAND_TOOL_LABELS,
        modelLabel: IMAGE_COMMAND_MODEL_LABEL,
        tokenLabel: IMAGE_COMMAND_TOKEN_LABEL,
      },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    const hasTerminalResultCaption = Boolean(
      snapshot.hasPresentationCaption || snapshot.hasNaturalCompletionCaption,
    );
    if (
      snapshot.hasPrompt &&
      snapshot.cardCount === 1 &&
      snapshot.mediaCount >= 1 &&
      snapshot.hasPresentationIntro === true &&
      snapshot.hasPresentationIntroInAssistantText === true &&
      snapshot.hasToolStripLabel === true &&
      snapshot.hasImageModelLabel === true &&
      snapshot.hasTokenUsage === true &&
      snapshot.hasPreviewImage === true &&
      snapshot.hasLoadedVisiblePreviewImage === true &&
      hasTerminalResultCaption &&
      snapshot.hasPresentationCaptionAfterCard === true &&
      snapshot.cardHasPresentationCaption === false &&
      snapshot.introBeforeCard === true &&
      snapshot.completionAfterCard === true &&
      snapshot.visiblePendingStatus === false &&
      snapshot.templateTaskIdVisible === false &&
      snapshot.draftImageVisible === false &&
      snapshot.textareaDisabled === false &&
      snapshot.stopButtonVisible === false
    ) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  const recentNotificationEvidence = await page
    .evaluate(async () => {
      const response = await window.electronAPI?.invoke?.(
        "app_server_drain_events",
        { request: { includeRecent: true, limit: 500 } },
      );
      const lines = Array.isArray(response?.lines) ? response.lines : [];
      return lines.flatMap((line) => {
        try {
          const message = JSON.parse(String(line));
          if (
            message?.method !== "thread/tokenUsage/updated" &&
            message?.method !== "turn/completed"
          ) {
            return [];
          }
          const params = message.params;
          const last = params?.tokenUsage?.last;
          return [
            {
              method: message.method,
              threadId: params?.threadId ?? null,
              turnId: params?.turnId ?? params?.turn?.id ?? null,
              inputTokens: last?.inputTokens ?? null,
              outputTokens: last?.outputTokens ?? null,
            },
          ];
        } catch {
          return [];
        }
      });
    })
    .catch((error) => [{ diagnosticError: String(error?.message || error) }]);
  throw new Error(
    `Claw GUI 未把 @配图 task 轻卡推进终态: ${JSON.stringify(
      sanitizeJson({
        snapshot: lastSnapshot,
        recentNotifications: recentNotificationEvidence,
      }),
    )}`,
  );
}

async function readImageCommandTaskArtifact({
  page,
  appServerRequests,
  taskArtifact,
  workspace,
}) {
  const taskId = imageTaskId(taskArtifact);
  const taskRefs = Array.from(
    new Set(
      [taskId, imageTaskPath(taskArtifact), imageTaskArtifactPath(taskArtifact)]
        .filter((value) => typeof value === "string")
        .map((value) => value.trim())
        .filter(Boolean),
    ),
  );
  let getResult = null;
  const getErrors = [];
  for (const taskRef of taskRefs) {
    try {
      getResult = await invokeAppServerFromPage(
        page,
        APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_GET,
        {
          projectRootPath: workspace.rootPath,
          taskRef,
        },
        appServerRequests,
      );
      break;
    } catch (error) {
      getErrors.push({ taskRef, error: String(error?.message || error) });
    }
  }
  const listResult = await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_LIST,
    {
      projectRootPath: workspace.rootPath,
      taskType: "image_generate",
      limit: 20,
    },
    appServerRequests,
  );
  const taskPath = imageTaskPath(taskArtifact);
  let fileStatus = null;
  let fileNormalizedStatus = null;
  let fileResultImageCount = null;
  if (typeof taskPath === "string" && fs.existsSync(taskPath)) {
    try {
      const record = readTaskArtifactFile(taskPath);
      fileStatus = record.status ?? null;
      fileNormalizedStatus = record.normalized_status ?? null;
      fileResultImageCount = Array.isArray(record.result?.images)
        ? record.result.images.length
        : null;
    } catch {
      fileStatus = "unreadable";
    }
  }
  const getRecord = imageTaskRecord(getResult?.result);
  return sanitizeJson({
    taskId,
    taskPath,
    exists: typeof taskPath === "string" ? fs.existsSync(taskPath) : false,
    pathIncludesImageGenerate:
      typeof taskPath === "string" &&
      taskPath.includes(".lime/tasks/image_generate"),
    getReturned: Boolean(getResult?.result),
    getErrors,
    getTaskId: imageTaskId(getResult?.result),
    getStatus: getRecord?.status ?? getResult?.result?.status ?? null,
    getNormalizedStatus:
      getRecord?.normalized_status ??
      getResult?.result?.normalized_status ??
      null,
    listReturned: Boolean(listResult?.result),
    listContainsTask:
      JSON.stringify(listResult?.result || {}).includes(String(taskId || "")) &&
      Boolean(taskId),
    listContainsImageGenerate: JSON.stringify(
      listResult?.result || {},
    ).includes("image_generate"),
    fileStatus,
    fileNormalizedStatus,
    fileResultImageCount,
  });
}

async function reloadAndReopenImageCommandSession({
  page,
  options,
  appServerRequests,
}) {
  const reload = await reloadRendererDocument(page, options);
  const renderer = await waitForRendererReady(page, options);
  const session = await openFixtureSessionFromSidebar(
    page,
    options,
    appServerRequests,
  );
  return sanitizeJson({
    reload,
    renderer,
    session,
  });
}

async function waitForImageCommandAgentUiPerformanceTrace(page, options) {
  const startedAt = Date.now();
  const timeoutMs = Math.min(options.timeoutMs, 15_000);
  let lastEvidence = null;
  while (Date.now() - startedAt < timeoutMs) {
    const evidence = sanitizeJson(
      await collectAgentUiPerformanceTraceEvidence(page),
    );
    lastEvidence = evidence;
    if (traceEvidenceHasProviderAndClient(evidence)) {
      return evidence;
    }
    await sleep(options.intervalMs);
  }
  return lastEvidence;
}

export async function runImageCommandScenario({
  page,
  options,
  workspace,
  appServerRequests,
  imageFixtureProvider,
  summary,
}) {
  const scenarioConfig = resolveImageIntentScenario(options.scenario);
  const imageCommandInputSend = sanitizeJson(
    await sendPromptFromGui(page, options, scenarioConfig.inputPrompt, {
      expectedSessionId: options.sessionId,
    }),
  );
  if (summary) {
    summary.imageCommandInputSend = imageCommandInputSend;
  }
  const traceTurnStart =
    imageCommandInputSend.afterClick?.matchingTurnStartTrace ?? null;
  const imageTaskArtifact = await waitForImageCommandWorkflowTaskArtifact({
    page,
    workspace,
    appServerRequests,
    options,
    imageFixtureProvider,
    scenarioConfig,
  });
  const workflowTaskPayload = imageTaskPayload(imageTaskArtifact.response);
  const turnId = optionalPayloadString(workflowTaskPayload, [
    "turn_id",
    "turnId",
  ]);
  const workflowHarness = {
    image_command_intent: {
      kind: "image_command",
      image_task: workflowTaskPayload,
    },
  };
  const imageCommandWorkflowTurnStart = {
    kind: "workflow:image_command",
    sessionId: options.sessionId,
    turnId,
    inputText: scenarioConfig.routedPrompt,
    providerPreference: traceTurnStart?.providerPreference ?? null,
    modelPreference: traceTurnStart?.modelPreference ?? null,
    runtimeOptions: traceTurnStart?.runtimeOptions ?? {},
    metadata: traceTurnStart?.metadata ?? null,
    runtimeRequest: {
      metadata: {
        harness:
          traceTurnStart?.runtimeRequest?.metadata?.harness ??
          workflowHarness,
      },
    },
  };
  const imageCommandTaskArtifact = await readImageCommandTaskArtifact({
    page,
    appServerRequests,
    taskArtifact: imageTaskArtifact.response,
    workspace,
  });
  const readModelAfterTaskCreate = await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_SESSION_READ,
    {
      threadId: options.threadId,
      includeTurns: true,
    },
    appServerRequests,
  );
  const readModelImageCommandAfterTaskCreate = sanitizeJson({
    summary: serializeReadModelSummary(
      readModelAfterTaskCreate.result,
      scenarioConfig,
    ),
    readModel: readModelAfterTaskCreate.result,
  });
  if (summary) {
    summary.imageCommandTaskArtifact = imageCommandTaskArtifact;
    summary.readModelImageCommandAfterTaskCreate =
      readModelImageCommandAfterTaskCreate;
  }
  const guiImageCommandCompleted = await waitForGuiImageCommandCompleted(
    page,
    options,
    scenarioConfig,
  );
  const readModelImageCommandCompleted =
    await waitForSessionReadImageCommandCompleted(
      page,
      options,
      appServerRequests,
      scenarioConfig,
    );
  const imageCommandTaskArtifactTerminalPatch =
    await waitForImageCommandTaskArtifactTerminal({
      page,
      workspace,
      appServerRequests,
      taskArtifact: imageTaskArtifact.response,
      options,
    });
  const imageCommandTaskArtifactTerminal = await readImageCommandTaskArtifact({
    page,
    appServerRequests,
    taskArtifact: imageTaskArtifact.response,
    workspace,
  });
  const imageCommandTaskAuditLog = readImageCommandTaskAuditLog({
    workspace,
    taskPath: imageCommandTaskArtifactTerminalPatch.taskPath,
  });
  const imageCommandWorkflowRead = await readImageCommandWorkflowAudit({
    page,
    appServerRequests,
    sessionId: options.sessionId,
    turnId,
    taskId: imageCommandTaskArtifactTerminalPatch.taskId,
  });
  const guiImageCommandTerminal = await waitForGuiImageCommandTerminal(
    page,
    options,
    imageCommandTaskArtifactTerminalPatch.taskId,
    scenarioConfig,
  );
  const agentUiPerformanceTracePreReload =
    await waitForImageCommandAgentUiPerformanceTrace(page, options);
  const guiImageCommandReload = await reloadAndReopenImageCommandSession({
    page,
    options,
    appServerRequests,
  });
  const guiImageCommandRestoredAfterReload =
    await waitForGuiImageCommandTerminal(
      page,
      options,
      imageCommandTaskArtifactTerminalPatch.taskId,
      scenarioConfig,
    );
  const imageCommandTaskArtifactAfterReload =
    await readImageCommandTaskArtifact({
      page,
      appServerRequests,
      taskArtifact: imageTaskArtifact.response,
      workspace,
    });

  return sanitizeJson({
    imageCommandInputSend,
    imageCommandWorkflowUsed: true,
    imageCommandWorkflowTurnStart,
    imageCommandBackendTurnStart: {
      sessionId: imageCommandWorkflowTurnStart.sessionId,
      turnId,
      inputText: imageCommandWorkflowTurnStart.inputText,
      hasImageCommandIntentMetadata: JSON.stringify(
        imageCommandWorkflowTurnStart.runtimeRequest || {},
      ).includes("image_command_intent"),
      hasLegacyImageSkillLaunchMetadata: JSON.stringify(
        imageCommandWorkflowTurnStart.runtimeRequest || {},
      ).includes("image_skill_launch"),
      hasImageTaskMetadata: JSON.stringify(
        imageCommandWorkflowTurnStart.runtimeRequest || {},
      ).includes("image_task"),
      providerPreference: imageCommandWorkflowTurnStart.providerPreference,
      modelPreference: imageCommandWorkflowTurnStart.modelPreference,
    },
    imageCommandTaskCreateRequest: workflowTaskPayload,
    imageCommandFixtureProvider: imageFixtureProvider ?? null,
    imageCommandTaskArtifact,
    imageCommandTaskArtifactTerminalPatch,
    imageCommandTaskArtifactTerminal,
    imageCommandTaskAuditLog,
    imageCommandWorkflowRead,
    guiImageCommandCompleted,
    guiImageCommandTerminal,
    agentUiPerformanceTrace: agentUiPerformanceTracePreReload,
    agentUiPerformanceTracePreReload,
    guiImageCommandReload,
    guiImageCommandRestoredAfterReload,
    imageCommandTaskArtifactAfterReload,
    readModelImageCommandCompleted: readModelImageCommandCompleted.summary,
  });
}
