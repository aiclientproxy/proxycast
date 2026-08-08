import fs from "node:fs";
import {
  renderApprovalRequestResumeActionRespondScript,
  renderApprovalRequestResumeHelpersScript,
  renderApprovalRequestResumeTurnStartScript,
} from "./claw-chat-current-fixture-approval-backend-events.mjs";
import { renderBackendToolAndSkillEventScript } from "./claw-chat-current-fixture-backend-tool-skill-events.mjs";
import { renderUnknownItemBackendEventsExpression } from "./claw-chat-current-fixture-unknown-item.mjs";
import {
  APPROVAL_REQUEST_RESUME_PROMPT,
  APPROVAL_REQUEST_FULL_ACCESS_DONE_TEXT,
  APPROVAL_REQUEST_FULL_ACCESS_PROMPT,
  APPROVAL_REQUEST_FULL_ACCESS_RESULT_TEXT,
  APPROVAL_REQUEST_RESUME_SECOND_DONE_TEXT,
  APPROVAL_REQUEST_RESUME_SECOND_PROMPT,
  APPROVAL_REQUEST_RESUME_SECOND_RESULT_TEXT,
  ASSISTANT_DONE_TEXT,
  CONTINUE_DONE_TEXT,
  CONTINUE_PROMPT,
  EVENT_READ_PROBE_DONE_TEXT,
  EVENT_READ_PROBE_PROMPT,
  EVENT_READ_PROBE_READ_TEXT,
  EXPERT_PANEL_SKILLS_RUNTIME_SCENARIO,
  EXPERT_SKILLS_RUNTIME_DONE_TEXT,
  EXPERT_SKILLS_RUNTIME_PANEL_DONE_TEXT,
  EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
  EXPERT_SKILLS_RUNTIME_PROMPT,
  EXPERT_SKILLS_RUNTIME_SCENARIO,
  FIXTURE_MODEL,
  FIXTURE_PROVIDER,
  GOAL_DONE_TEXT,
  GOAL_PROMPT,
  GREETING_DONE_TEXT,
  GREETING_PROMPT,
  GREETING_SUMMARY_TEXT,
  IMAGE_COMMAND_PRESENTATION_CAPTION,
  IMAGE_COMMAND_PRESENTATION_INTRO,
  INPUTBAR_RICH_RESTORE_PROMPT,
  INPUTBAR_RICH_RESTORE_SCENARIO,
  LIVE_TAIL_COMMIT_DONE_TEXT,
  LIVE_TAIL_COMMIT_FIRST_TEXT,
  LIVE_TAIL_COMMIT_OVERFLOW_MARKER,
  LIVE_TAIL_COMMIT_PROMPT,
  LIVE_TAIL_COMMIT_TABLE_HEADER,
  LIVE_TAIL_COMMIT_TABLE_TAIL,
  MCP_STRUCTURED_CONTENT_DONE_TEXT,
  MCP_STRUCTURED_CONTENT_PROMPT,
  PLAN_DONE_TEXT,
  PLAN_PROMPT,
  PLAN_STEPS,
  PROPOSED_PLAN_BLOCK,
  REASONING_FIRST_VISIBLE_CONTENT_TEXT,
  REASONING_FIRST_VISIBLE_DONE_TEXT,
  REASONING_FIRST_VISIBLE_FINAL_TEXT,
  REASONING_FIRST_VISIBLE_PROMPT,
  REASONING_FIRST_VISIBLE_TEXT,
  TERMINAL_CANCELED_AFTER_ANSWER_CANCELED_TEXT,
  TERMINAL_CANCELED_AFTER_ANSWER_PARTIAL_TEXT,
  TERMINAL_CANCELED_AFTER_ANSWER_PROMPT,
  TERMINAL_CANCELED_AFTER_ANSWER_SCENARIO,
  TERMINAL_FAILED_AFTER_ANSWER_FAILURE_TEXT,
  TERMINAL_FAILED_AFTER_ANSWER_PARTIAL_TEXT,
  TERMINAL_FAILED_AFTER_ANSWER_PROMPT,
  TERMINAL_STALE_GUARD_DONE_TEXT,
  TERMINAL_STALE_GUARD_FIRST_DONE_TEXT,
  TERMINAL_STALE_GUARD_FIRST_PROMPT,
  TERMINAL_STALE_GUARD_FIRST_TEXT,
  TERMINAL_STALE_GUARD_SECOND_PROMPT,
  TERMINAL_STALE_GUARD_SECOND_TEXT,
  TERMINAL_STALE_GUARD_STALE_DONE_TEXT,
  THREAD_ID,
  TYPED_ERROR_RETRY_FAILURE_ERROR_TEXT,
  TYPED_ERROR_RETRY_FAILURE_PARTIAL_TEXT,
  TYPED_ERROR_RETRY_FAILURE_PROMPT,
  TYPED_ERROR_RETRY_FAILURE_SCENARIO,
  TYPED_ERROR_RETRY_FAILURE_TEXT,
  TYPED_ERROR_RETRY_MESSAGE,
  TYPED_ERROR_RETRY_SECOND_MESSAGE,
  TYPED_ERROR_RETRY_SUCCESS_DONE_TEXT,
  TYPED_ERROR_RETRY_SUCCESS_PROMPT,
  TYPED_ERROR_RETRY_SUCCESS_SCENARIO,
  TYPED_ERROR_RETRY_SUCCESS_TEXT,
  WEB_TOOLS_BROKEN_MARKDOWN_TEXT,
  WEB_TOOLS_RENDERING_DONE_TEXT,
  WEB_TOOLS_RENDERING_PROMPT,
  renderSkillsRuntimeBackendEvents,
  SKILLS_RUNTIME_DONE_TEXT,
  SKILLS_RUNTIME_EXPLICIT_DONE_TEXT,
  SKILLS_RUNTIME_EXPLICIT_PROMPT,
  SKILLS_RUNTIME_EXPLICIT_SCENARIO,
  SKILLS_RUNTIME_MANUAL_ENABLE_DONE_TEXT,
  SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT,
  SKILLS_RUNTIME_MANUAL_ENABLE_SCENARIO,
  SKILLS_RUNTIME_PROMPT,
  SKILLS_RUNTIME_SCENARIO,
} from "./claw-chat-current-fixture-constants.mjs";
import {
  MEDIA_REFERENCE_DONE_TEXT,
  MEDIA_REFERENCE_MIME_TYPE,
  MEDIA_REFERENCE_PROMPT,
  MEDIA_REFERENCE_SUMMARY_TEXT,
  MEDIA_REFERENCE_URI,
} from "./claw-chat-current-fixture-media-reference.mjs";

export function summarizeRequestInput(request) {
  const agentImages = Array.isArray(
    request?.runtimeOptions?.runtimeRequest?.images,
  )
    ? request.runtimeOptions.runtimeRequest.images
    : [];
  const runtimeMetadata =
    request?.runtimeOptions?.runtimeRequest?.metadata ?? {};
  const harnessMetadata = runtimeMetadata?.harness ?? {};
  const pathReferences = Array.isArray(runtimeMetadata?.path_references)
    ? runtimeMetadata.path_references
    : Array.isArray(runtimeMetadata?.pathReferences)
      ? runtimeMetadata.pathReferences
      : [];
  const harnessFileReferences = Array.isArray(harnessMetadata?.file_references)
    ? harnessMetadata.file_references
    : Array.isArray(harnessMetadata?.fileReferences)
      ? harnessMetadata.fileReferences
      : [];
  const fileReferences =
    pathReferences.length > 0 ? pathReferences : harnessFileReferences;
  const inputImageCount = runtimeInputImageCount(request);
  return {
    textLength: runtimeInputText(request).length,
    attachmentCount: inputImageCount + agentImages.length,
    imageAttachmentCount: inputImageCount + agentImages.length,
    fileReferenceCount: fileReferences.length,
    fileReferenceNames: fileReferences
      .map((reference) => reference?.name)
      .filter((value) => typeof value === "string"),
    fileReferencePaths: fileReferences
      .map((reference) => reference?.path)
      .filter((value) => typeof value === "string"),
  };
}

export function runtimeInputText(request) {
  const parts = Array.isArray(request?.input?.parts) ? request.input.parts : [];
  return parts
    .map((part) => (typeof part?.Text?.text === "string" ? part.Text.text : ""))
    .join("");
}

export function runtimeInputImageCount(request) {
  const parts = Array.isArray(request?.input?.parts) ? request.input.parts : [];
  return parts.filter((part) => part?.Image !== undefined).length;
}

export function buildCanonicalToolItem({
  sessionId,
  threadId,
  turnId,
  itemId,
  ordinal,
  callId,
  name,
  arguments: toolArguments = [],
  status = "inProgress",
  output = null,
  metadata = {},
}) {
  const timestamp = Date.now();
  const argumentsList = Array.isArray(toolArguments)
    ? toolArguments
    : Object.entries(toolArguments).map(([argumentName, value]) => ({
        name: argumentName,
        value:
          typeof value === "string"
            ? value
            : (JSON.stringify(value) ?? String(value)),
      }));
  const terminal = ["completed", "failed", "interrupted", "cancelled"].includes(
    status,
  );
  return {
    item: {
      sessionId,
      threadId,
      turnId,
      itemId,
      sequence: 0,
      ordinal,
      createdAtMs: timestamp,
      updatedAtMs: timestamp,
      ...(terminal ? { completedAtMs: timestamp } : {}),
      kind: "tool",
      status,
      payload: {
        type: "tool",
        call_id: callId,
        name,
        arguments: argumentsList,
        output,
      },
      metadata,
    },
  };
}

export function writeFixtureBackend(backendPath, options = {}) {
  const mediaReferenceSourcePath = String(
    options.mediaReferenceSourcePath ?? "",
  ).trim();
  const proposedPlanFixtureText = `${PROPOSED_PLAN_BLOCK}\n计划已写入右侧计划轨，等待你确认后再执行。\n`;
  const proposedPlanThreadItemText = PLAN_STEPS.map(
    (step) => `- ${step.step}`,
  ).join("\n");
  const webToolsRenderingFixtureText = `网页搜索渲染结论：搜索来源已展开，读取页面已归入同一过程，最终正文继续输出。\n${WEB_TOOLS_BROKEN_MARKDOWN_TEXT}\n${WEB_TOOLS_RENDERING_DONE_TEXT}\n`;
  const liveTailCommitFixtureText = [
    `${LIVE_TAIL_COMMIT_OVERFLOW_MARKER}: 长输出已经从 live tail commit 到消息历史。`,
    "live tail 长输出第 01 行：首字已经可见，后续追加不能重写前段。",
    "live tail 长输出第 02 行：滚动容器需要保持接近底部。",
    "live tail 长输出第 03 行：assistant bubble 不能挪到全局 overlay。",
    "live tail 长输出第 04 行：history item 与 live tail 同源。",
    "live tail 长输出第 05 行：commit 后 sequence 仍稳定。",
    "live tail 长输出第 06 行：completion 只关闭 running 状态。",
    "live tail 长输出第 07 行：table render 不能遮挡输入框。",
    "live tail 长输出第 08 行：read model 需要包含同一轮内容。",
    "live tail 长输出第 09 行：Electron IPC trace 仍归属 current method。",
    "live tail 长输出第 10 行：provider wait 与 client local output 分开记录。",
    "live tail 长输出第 11 行：首字 paint 不能等待最终 done。",
    "live tail 长输出第 12 行：长输出 commit 不丢任何中间文本。",
    "下面的表格用于证明 live table tail reflow 不会丢失：",
    `${LIVE_TAIL_COMMIT_TABLE_HEADER}`,
    "| --- | --- | --- |",
    "| 01 | first token before commit | stable |",
    "| 02 | live tail separated from history item | stable |",
    "| 03 | scroll anchor retained | stable |",
    "| 04 | overflow commit sequence | stable |",
    "| 05 | markdown table header | stable |",
    "| 06 | markdown table body | stable |",
    "| 07 | renderer projection | stable |",
    "| 08 | App Server read model | stable |",
    "| 09 | Electron IPC trace | stable |",
    "| 10 | no overlay global buffer | stable |",
    "| 11 | completion does not rewrite earlier text | stable |",
    "| 12 | table tail pending | stable |",
    "| 13 | row reflow after overflow | stable |",
    "| 14 | status surface during stream | stable |",
    "| 15 | inputbar remains below transcript | stable |",
    "| 16 | no startup note flash | stable |",
    "| 17 | first paint before done | stable |",
    "| 18 | renderer list item retained | stable |",
    "| 19 | markdown pipe row parsed | stable |",
    "| 20 | table body visible | stable |",
    "| 21 | tail row survives completion | stable |",
    "| 22 | bottom anchor retained | stable |",
    "| 23 | current terminal closes stream | stable |",
    `${LIVE_TAIL_COMMIT_TABLE_TAIL}`,
    `${LIVE_TAIL_COMMIT_DONE_TEXT}`,
    "",
  ].join("\n");
  const skillsRuntimeBackendEvents = renderSkillsRuntimeBackendEvents(
    SKILLS_RUNTIME_SCENARIO,
  );
  const explicitSkillsRuntimeBackendEvents = renderSkillsRuntimeBackendEvents({
    ...SKILLS_RUNTIME_EXPLICIT_SCENARIO,
    promptFlagName: "isExplicitSkillsRuntimePrompt",
  });
  const manualEnableSkillsRuntimeBackendEvents =
    renderSkillsRuntimeBackendEvents({
      ...SKILLS_RUNTIME_MANUAL_ENABLE_SCENARIO,
      promptFlagName: "isManualEnableSkillsRuntimePrompt",
    });
  const expertSkillsRuntimeBackendEvents = renderSkillsRuntimeBackendEvents({
    ...EXPERT_SKILLS_RUNTIME_SCENARIO,
    promptFlagName: "isExpertSkillsRuntimePrompt",
  });
  const expertPanelSkillsRuntimeBackendEvents =
    renderSkillsRuntimeBackendEvents({
      ...EXPERT_PANEL_SKILLS_RUNTIME_SCENARIO,
      promptFlagName: "isExpertPanelSkillsRuntimePrompt",
    });
  fs.writeFileSync(
    backendPath,
    `#!/usr/bin/env node
import { createHash } from "node:crypto";
import { appendFileSync, readFileSync } from "node:fs";

const ledgerPath = process.argv[2];
const cancelSignalPath = process.argv[3];
const typedErrorSignalPath = process.argv[4];
const mediaReferenceSourcePath = ${JSON.stringify(mediaReferenceSourcePath)};
const input = JSON.parse(readFileSync(0, "utf8"));
const runtimeRequest = input.request?.runtimeOptions?.runtimeRequest;

${renderApprovalRequestResumeHelpersScript()}

export function appendLedgerEntry(entry) {
  if (!ledgerPath) {
    return;
  }
  appendFileSync(ledgerPath, JSON.stringify({
    ...entry,
    recordedAt: new Date().toISOString()
  }) + "\\n");
}

export function readLedgerEntries() {
  if (!ledgerPath) {
    return [];
  }
  try {
    return readFileSync(ledgerPath, "utf8")
      .split("\\n")
      .filter((line) => line.trim().length > 0)
      .map((line) => JSON.parse(line));
  } catch {
    return [];
  }
}

export function emitEvents(events) {
  appendLedgerEntry({
    kind: "backendEmit",
    sessionId: input.request?.session?.sessionId,
    turnId: input.request?.turn?.turnId,
    eventCount: events.length,
    eventTypes: events.map((event) => event?.type).filter(Boolean)
  });
  console.log(JSON.stringify({ events }));
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function readTypedErrorSignals() {
  if (!typedErrorSignalPath) {
    return [];
  }
  try {
    return readFileSync(typedErrorSignalPath, "utf8")
      .split("\\n")
      .filter((line) => line.trim().length > 0)
      .map((line) => JSON.parse(line));
  } catch {
    return [];
  }
}

export async function waitForTypedErrorSignal(stage) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < 120000) {
    if (readTypedErrorSignals().some((entry) => entry?.stage === stage)) {
      return;
    }
    await sleep(50);
  }
  throw new Error("typed error scenario timed out waiting for signal: " + stage);
}

export function currentThreadId() {
  return input.request?.session?.threadId ??
    input.request?.session?.thread_id ??
    "${THREAD_ID}";
}

export function currentTurnId() {
  return input.request?.turn?.turnId ??
    input.request?.turn?.turn_id ??
    runtimeRequest?.turn_id ??
    runtimeRequest?.turnId ??
    "";
}

${summarizeRequestInput.toString()}
${runtimeInputText.toString()}
${runtimeInputImageCount.toString()}
${buildCanonicalToolItem.toString()}

export function summarizeActionRespondRequest(request) {
  const actionScope = request?.actionScope || request?.action_scope || {};
  return {
    requestId:
      request?.requestId ||
      request?.request_id ||
      request?.actionId ||
      request?.action_id,
    actionType: request?.actionType || request?.action_type,
    decision: request?.decision,
    decisionScope: request?.decisionScope || request?.decision_scope,
    confirmed: request?.confirmed,
    requestKeys: Object.keys(request || {}).sort(),
    actionScope: {
      sessionId: actionScope.sessionId || actionScope.session_id,
      session_id: actionScope.sessionId || actionScope.session_id,
      threadId: actionScope.threadId || actionScope.thread_id,
      thread_id: actionScope.threadId || actionScope.thread_id,
      turnId: actionScope.turnId || actionScope.turn_id,
      turn_id: actionScope.turnId || actionScope.turn_id
    }
  };
}

const actionRespondSummary =
  input.kind === "actionRespond"
    ? summarizeActionRespondRequest(input.request)
    : {};

appendLedgerEntry({
    kind: input.kind,
    sessionId: input.request?.session?.sessionId,
    turnId: input.request?.turn?.turnId,
    inputText: runtimeInputText(input.request),
    inputSummary: summarizeRequestInput(input.request),
    providerPreference: input.request?.providerPreference,
    modelPreference: input.request?.modelPreference,
    runtimeOptions: input.request?.runtimeOptions,
    runtimeRequest,
    ...actionRespondSummary
});

if (input.kind === "turnCancel") {
  if (cancelSignalPath) {
    appendFileSync(cancelSignalPath, JSON.stringify({
      sessionId: input.request?.session?.sessionId,
      turnId: input.request?.turn?.turnId,
      recordedAt: new Date().toISOString()
    }) + "\\n");
  }
  if (process.env.CLAW_CHAT_FIXTURE_SCENARIO === "${TERMINAL_CANCELED_AFTER_ANSWER_SCENARIO}") {
    appendLedgerEntry({
      kind: "terminalCanceledAfterAnswerTurnCanceled",
      sessionId: input.request?.session?.sessionId,
      turnId: input.request?.turn?.turnId,
      eventType: "turn.canceled",
      partialText: "${TERMINAL_CANCELED_AFTER_ANSWER_PARTIAL_TEXT}",
      canceledText: "${TERMINAL_CANCELED_AFTER_ANSWER_CANCELED_TEXT}"
    });
  }
  emitEvents([
    {
      type: "turn.canceled",
      payload: {
        status: "canceled",
        reason: "user_cancelled"
      }
    }
  ]);
  process.exit(0);
}

${renderApprovalRequestResumeActionRespondScript()}

if (input.kind === "turnStart") {
  const inputText = runtimeInputText(input.request);
  const serializedInput = JSON.stringify(input);
  const isImageTaskPresentationPrompt =
    inputText.includes("image_task_presentation.v1") ||
    inputText.includes("Generate user-visible copy for one image generation turn.") ||
    serializedInput.includes("image_task_presentation.v1") ||
    serializedInput.includes("Generate user-visible copy for one image generation turn.") ||
    serializedInput.includes("image_command_presentation");
  const isEventReadProbe = inputText.includes("${EVENT_READ_PROBE_PROMPT}");
  const isContinuePrompt = inputText.includes("${CONTINUE_PROMPT}");
  const isGreetingPrompt = inputText === "${GREETING_PROMPT}";
  const isPlanPrompt = inputText.includes("${PLAN_PROMPT}");
  const isGoalPrompt = inputText.includes("${GOAL_PROMPT}");
  const isInputbarRichRestorePrompt = inputText.includes("${INPUTBAR_RICH_RESTORE_PROMPT}");
  const isApprovalRequestResumePrompt = inputText.includes("${APPROVAL_REQUEST_RESUME_PROMPT}");
  const isApprovalRequestFullAccessPrompt = inputText.includes("${APPROVAL_REQUEST_FULL_ACCESS_PROMPT}");
  const isApprovalRequestResumeSecondPrompt = inputText.includes("${APPROVAL_REQUEST_RESUME_SECOND_PROMPT}");
  const isReasoningFirstVisiblePrompt = inputText.includes("${REASONING_FIRST_VISIBLE_PROMPT}");
  const isLiveTailCommitPrompt = inputText.includes("${LIVE_TAIL_COMMIT_PROMPT}");
  const isTerminalCanceledAfterAnswerPrompt = inputText.includes("${TERMINAL_CANCELED_AFTER_ANSWER_PROMPT}");
  const isTerminalFailedAfterAnswerPrompt = inputText.includes("${TERMINAL_FAILED_AFTER_ANSWER_PROMPT}");
  const isTerminalStaleGuardFirstPrompt = inputText.includes("${TERMINAL_STALE_GUARD_FIRST_PROMPT}");
  const isTerminalStaleGuardSecondPrompt = inputText.includes("${TERMINAL_STALE_GUARD_SECOND_PROMPT}");
  const isTypedErrorRetrySuccessPrompt = inputText.includes("${TYPED_ERROR_RETRY_SUCCESS_PROMPT}");
  const isTypedErrorRetryFailurePrompt = inputText.includes("${TYPED_ERROR_RETRY_FAILURE_PROMPT}");
  const isWebToolsRenderingPrompt = inputText.includes("${WEB_TOOLS_RENDERING_PROMPT}");
  const isMcpStructuredContentPrompt = inputText.includes("${MCP_STRUCTURED_CONTENT_PROMPT}");
  const isMediaReferencePrompt = inputText.includes("${MEDIA_REFERENCE_PROMPT}");
  const isManualEnableSkillsRuntimePrompt = inputText.includes("${SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT}");
  const isExpertPanelSkillsRuntimePrompt = inputText.includes("${EXPERT_SKILLS_RUNTIME_PANEL_PROMPT}");
  const isExpertSkillsRuntimePrompt =
    inputText.includes("${EXPERT_SKILLS_RUNTIME_PROMPT}") &&
    !isExpertPanelSkillsRuntimePrompt;
  const isExplicitSkillsRuntimePrompt = inputText.includes("${SKILLS_RUNTIME_EXPLICIT_PROMPT}");
  const isSkillsRuntimePrompt =
    inputText.includes("${SKILLS_RUNTIME_PROMPT}") &&
    !isExplicitSkillsRuntimePrompt &&
    !isManualEnableSkillsRuntimePrompt;
  const assistantDoneText = isEventReadProbe
    ? "${EVENT_READ_PROBE_DONE_TEXT}"
    : isContinuePrompt
      ? "${CONTINUE_DONE_TEXT}"
      : isGreetingPrompt
        ? "${GREETING_DONE_TEXT}"
      : isPlanPrompt
        ? "${PLAN_DONE_TEXT}"
        : isGoalPrompt
          ? "${GOAL_DONE_TEXT}"
          : isReasoningFirstVisiblePrompt
            ? "${REASONING_FIRST_VISIBLE_DONE_TEXT}"
          : isLiveTailCommitPrompt
            ? "${LIVE_TAIL_COMMIT_DONE_TEXT}"
          : isTerminalStaleGuardFirstPrompt
            ? "${TERMINAL_STALE_GUARD_FIRST_DONE_TEXT}"
          : isTerminalStaleGuardSecondPrompt
            ? "${TERMINAL_STALE_GUARD_DONE_TEXT}"
          : isWebToolsRenderingPrompt
            ? "${WEB_TOOLS_RENDERING_DONE_TEXT}"
          : isMcpStructuredContentPrompt
            ? "${MCP_STRUCTURED_CONTENT_DONE_TEXT}"
              : isMediaReferencePrompt
                ? "${MEDIA_REFERENCE_DONE_TEXT}"
                : isSkillsRuntimePrompt
                    ? "${SKILLS_RUNTIME_DONE_TEXT}"
                    : isExplicitSkillsRuntimePrompt
                      ? "${SKILLS_RUNTIME_EXPLICIT_DONE_TEXT}"
                      : isManualEnableSkillsRuntimePrompt
                        ? "${SKILLS_RUNTIME_MANUAL_ENABLE_DONE_TEXT}"
                        : isExpertSkillsRuntimePrompt
                          ? "${EXPERT_SKILLS_RUNTIME_DONE_TEXT}"
                          : isExpertPanelSkillsRuntimePrompt
                            ? "${EXPERT_SKILLS_RUNTIME_PANEL_DONE_TEXT}"
                            : isApprovalRequestFullAccessPrompt
                              ? "${APPROVAL_REQUEST_FULL_ACCESS_DONE_TEXT}"
                            : isApprovalRequestResumeSecondPrompt
                              ? "${APPROVAL_REQUEST_RESUME_SECOND_DONE_TEXT}"
                            : "${ASSISTANT_DONE_TEXT}";
  const hasProcessPrelude =
    isEventReadProbe ||
    isPlanPrompt ||
    isReasoningFirstVisiblePrompt ||
    isWebToolsRenderingPrompt ||
    isMcpStructuredContentPrompt ||
    isMediaReferencePrompt ||
    isSkillsRuntimePrompt ||
    isExplicitSkillsRuntimePrompt ||
    isManualEnableSkillsRuntimePrompt ||
    isExpertSkillsRuntimePrompt ||
    isExpertPanelSkillsRuntimePrompt;
  const currentTurnIdForItem = currentTurnId() || "turn";
  const commentaryItemId = \`agent-message-commentary-\${currentTurnIdForItem}\`;
  const finalAnswerItemId = \`agent-message-final-\${currentTurnIdForItem}\`;
  function messageDeltaPayload(text, phase, itemId) {
    return {
      text,
      item_id: itemId,
      itemId,
      phase,
      thread_id: currentThreadId(),
      threadId: currentThreadId(),
      turn_id: currentTurnId(),
      turnId: currentTurnId()
    };
  }
  function providerTracePayload(stage, elapsedMs, status, extra = {}) {
    return {
      stage,
      provider: "${FIXTURE_PROVIDER}",
      model: "${FIXTURE_MODEL}",
      attempt: 1,
      elapsed_ms: elapsedMs,
      elapsedMs,
      status,
      ...extra
    };
  }
  function approvalSessionCacheFromRequest() {
    const harness = runtimeRequest?.metadata?.harness ?? {};
    return harness.approval_session_cache ?? harness.approvalSessionCache ?? null;
  }
  function approvalSessionCacheEvents() {
    const cache = approvalSessionCacheFromRequest();
    if (!cache) {
      return [];
    }
    const turnId = currentTurnId() || "turn";
    const threadId = currentThreadId();
    const requestId = "permission-" + turnId;
    const sourceRequestId = cache.sourceRequestId ?? cache.source_request_id;
    const key = cache.key ?? {};
    const scope = {
      sessionId: input.request?.session?.sessionId,
      session_id: input.request?.session?.sessionId,
      threadId,
      thread_id: threadId,
      turnId,
      turn_id: turnId
    };
    const cachePayload = {
      backend: "runtime_core",
      requestId,
      request_id: requestId,
      actionId: requestId,
      action_id: requestId,
      actionType: "tool_confirmation",
      action_type: "tool_confirmation",
      actionKind: "permission_preflight",
      action_kind: "permission_preflight",
      toolName: "browser_control",
      tool_name: "browser_control",
      decision: cache.decision ?? "allow_for_session",
      decisionScope: cache.decisionScope ?? cache.decision_scope ?? "session",
      decision_scope: cache.decisionScope ?? cache.decision_scope ?? "session",
      sourceRequestId,
      source_request_id: sourceRequestId,
      key,
      cache,
      scope
    };
    return [
      {
        type: "approval.session_cache.hit",
        payload: cachePayload
      },
      {
        type: "action.resolved",
        payload: {
          ...cachePayload,
          source: "approval_session_cache",
          confirmed: true
        }
      }
    ];
  }
  if (isTypedErrorRetrySuccessPrompt || isTypedErrorRetryFailurePrompt) {
    const scenario = isTypedErrorRetrySuccessPrompt
      ? "${TYPED_ERROR_RETRY_SUCCESS_SCENARIO}"
      : "${TYPED_ERROR_RETRY_FAILURE_SCENARIO}";
    const retryEvents = [
      {
        type: "runtime.error",
        payload: {
          message: "${TYPED_ERROR_RETRY_MESSAGE}",
          errorCode: "server_overloaded",
          retryable: true,
          willRetry: true,
          retryAttempt: 1,
          retryMaxAttempts: isTypedErrorRetrySuccessPrompt ? 1 : 2
        }
      },
      ...(isTypedErrorRetryFailurePrompt
        ? [{
            type: "runtime.error",
            payload: {
              message: "${TYPED_ERROR_RETRY_SECOND_MESSAGE}",
              errorCode: "server_overloaded",
              retryable: true,
              willRetry: true,
              retryAttempt: 2,
              retryMaxAttempts: 2
            }
          }]
        : [])
    ];
    emitEvents(retryEvents);
    appendLedgerEntry({
      kind: "typedErrorRetryingAwaitingSignal",
      scenario,
      sessionId: input.request?.session?.sessionId,
      threadId: currentThreadId(),
      turnId: currentTurnId(),
      eventTypes: retryEvents.map((event) => event.type),
      retryCount: retryEvents.length,
      expectedWillRetry: true,
      signalStage: "retry-visible"
    });
    await waitForTypedErrorSignal("retry-visible");

    if (isTypedErrorRetrySuccessPrompt) {
      emitEvents([
        {
          type: "provider.first_text_delta.received",
          payload: providerTracePayload("first_text_delta_received", 180, "running", {
            attempt: 2,
            text_chars: "${TYPED_ERROR_RETRY_SUCCESS_TEXT}".length,
            textChars: "${TYPED_ERROR_RETRY_SUCCESS_TEXT}".length
          })
        },
        {
          type: "message.delta",
          payload: messageDeltaPayload(
            "${TYPED_ERROR_RETRY_SUCCESS_TEXT}\\n${TYPED_ERROR_RETRY_SUCCESS_DONE_TEXT}",
            "final_answer",
            finalAnswerItemId
          )
        },
        {
          type: "turn.completed",
          payload: {
            status: "completed",
            text: "${TYPED_ERROR_RETRY_SUCCESS_DONE_TEXT}"
          }
        }
      ]);
      appendLedgerEntry({
        kind: "typedErrorTerminalEmitted",
        scenario,
        sessionId: input.request?.session?.sessionId,
        threadId: currentThreadId(),
        turnId: currentTurnId(),
        eventType: "turn.completed",
        terminalStatus: "completed"
      });
      process.exit(0);
    }

    emitEvents([
      {
        type: "runtime.error",
        payload: {
          message: "${TYPED_ERROR_RETRY_FAILURE_ERROR_TEXT}",
          errorCode: "server_overloaded",
          retryable: true,
          willRetry: false,
          retryAttempt: 2,
          retryMaxAttempts: 2,
          failureCategory: "fixture_retry_exhausted"
        }
      }
    ]);
    appendLedgerEntry({
      kind: "typedErrorAwaitingTerminalSignal",
      scenario,
      sessionId: input.request?.session?.sessionId,
      threadId: currentThreadId(),
      turnId: currentTurnId(),
      eventType: "runtime.error",
      expectedWillRetry: false,
      retryable: true,
      signalStage: "terminal-visible"
    });
    await waitForTypedErrorSignal("terminal-visible");
    emitEvents([
      {
        type: "provider.first_text_delta.received",
        payload: providerTracePayload("first_text_delta_received", 270, "running", {
          attempt: 3,
          text_chars: "${TYPED_ERROR_RETRY_FAILURE_PARTIAL_TEXT}".length,
          textChars: "${TYPED_ERROR_RETRY_FAILURE_PARTIAL_TEXT}".length
        })
      },
      {
        type: "message.delta",
        payload: messageDeltaPayload(
          "${TYPED_ERROR_RETRY_FAILURE_PARTIAL_TEXT}",
          "final_answer",
          finalAnswerItemId
        )
      },
      {
        type: "turn.failed",
        payload: {
          status: "failed",
          message: "${TYPED_ERROR_RETRY_FAILURE_ERROR_TEXT}",
          errorCode: "server_overloaded",
          retryable: true,
          willRetry: false,
          failureCategory: "fixture_retry_exhausted"
        }
      }
    ]);
    appendLedgerEntry({
      kind: "typedErrorTerminalEmitted",
      scenario,
      sessionId: input.request?.session?.sessionId,
      threadId: currentThreadId(),
      turnId: currentTurnId(),
      eventType: "turn.failed",
      terminalStatus: "failed",
      failureText: "${TYPED_ERROR_RETRY_FAILURE_TEXT}"
    });
    process.exit(0);
  }
  if (isImageTaskPresentationPrompt) {
    const presentationText = JSON.stringify({
      assistant_intro: ${JSON.stringify(IMAGE_COMMAND_PRESENTATION_INTRO)},
      completion_caption: ${JSON.stringify(IMAGE_COMMAND_PRESENTATION_CAPTION)}
    });
    emitEvents([
      {
        type: "message.delta",
        payload: messageDeltaPayload(presentationText, "final_answer", finalAnswerItemId)
      },
      {
        type: "turn.completed",
        payload: {
          status: "completed",
          text: presentationText
        }
      }
    ]);
    process.exit(0);
  }
${renderApprovalRequestResumeTurnStartScript()}
  const initialMessageText = isEventReadProbe
    ? "事件流 probe 已进入 RuntimeCore：\\n"
    : isContinuePrompt
      ? "继续输出已恢复：\\n"
      : isGreetingPrompt
        ? "${GREETING_SUMMARY_TEXT}。\\n"
      : isPlanPrompt
        ? "我先给出计划，不会直接改代码：\\n"
          : isGoalPrompt
            ? "追求目标已进入当前回合：\\n"
            : isReasoningFirstVisiblePrompt
              ? ""
            : isLiveTailCommitPrompt
              ? "${LIVE_TAIL_COMMIT_FIRST_TEXT}\\n"
            : isTerminalCanceledAfterAnswerPrompt
              ? "${TERMINAL_CANCELED_AFTER_ANSWER_PARTIAL_TEXT}\\n"
            : isTerminalFailedAfterAnswerPrompt
              ? "${TERMINAL_FAILED_AFTER_ANSWER_PARTIAL_TEXT}\\n"
            : isApprovalRequestFullAccessPrompt
              ? "${APPROVAL_REQUEST_FULL_ACCESS_RESULT_TEXT}\\n"
            : isTerminalStaleGuardFirstPrompt
              ? "${TERMINAL_STALE_GUARD_FIRST_TEXT}\\n"
            : isTerminalStaleGuardSecondPrompt
              ? "第二轮已经开始，旧 terminal 到达时不能打断当前输出。\\n"
            : isApprovalRequestResumeSecondPrompt
              ? "${APPROVAL_REQUEST_RESUME_SECOND_RESULT_TEXT}\\n"
          : isWebToolsRenderingPrompt
            ? "我先联网核实目标页面来源。\\n"
            : isMcpStructuredContentPrompt
              ? "我先调用 MCP docs 诊断工具，并只把用户答案放在 structuredContent。\\n"
              : isMediaReferencePrompt
                ? "${MEDIA_REFERENCE_SUMMARY_TEXT}：\\n"
                : isSkillsRuntimePrompt
                    ? "我先搜索 Skills metadata，再按需加载单个 SKILL.md。\\n"
                    : isExplicitSkillsRuntimePrompt
                      ? "我识别到显式 Skill 提及，仍先检索 metadata，再按需加载单个 SKILL.md。\\n"
                      : isManualEnableSkillsRuntimePrompt
                        ? "我识别到本轮手动启用的 workspace Skill，仍先核对 metadata，再按需加载单个 SKILL.md。\\n"
                        : isExpertSkillsRuntimePrompt
                          ? "我识别到专家绑定的 skillRefs，但仍先通过 skill_search 选择，再按需加载单个 SKILL.md。\\n"
                          : isExpertPanelSkillsRuntimePrompt
                            ? "我识别到右侧专家面板更新后的 skillRefs，并继续通过 skill_search 选择单个 Skill。\\n"
                            : "以下是今日国际新闻简要整理：\\n";
  const initialEvents = [
    ...approvalSessionCacheEvents(),
${renderUnknownItemBackendEventsExpression()}
    ...(isLiveTailCommitPrompt
      ? [
          {
            type: "item.started",
            payload: {
              item: {
                id: finalAnswerItemId,
                item_id: finalAnswerItemId,
                itemId: finalAnswerItemId,
                thread_id: currentThreadId(),
                threadId: currentThreadId(),
                turn_id: currentTurnId(),
                turnId: currentTurnId(),
                type: "agentMessage",
                kind: "agentMessage",
                role: "assistant",
                status: "inProgress",
                phase: "final"
              }
            }
          }
        ]
      : []),
    {
      type: "provider.request.started",
      payload: providerTracePayload("request_started", 0, "running")
    },
    {
      type: "provider.first_event.received",
      payload: providerTracePayload("first_event_received", 40, "running")
    },
    {
      type: "provider.first_text_delta.received",
      payload: providerTracePayload("first_text_delta_received", 90, "running", {
        text_chars: initialMessageText.length,
        textChars: initialMessageText.length
      })
    },
    {
      type: "message.delta",
      payload: messageDeltaPayload(
        initialMessageText,
        hasProcessPrelude ? "commentary" : "final_answer",
        hasProcessPrelude ? commentaryItemId : finalAnswerItemId
      )
    }
  ];
  const followupText = isContinuePrompt
    ? "停止后的同一会话已经可以继续输出，并由 App Server current 终态收口。\\n"
    : isGreetingPrompt
      ? "输入区、会话页和流式回复都已经接上，可以继续发下一句话。\\n"
    : isPlanPrompt
        ? ${JSON.stringify(proposedPlanFixtureText)}
        : isGoalPrompt
          ? "目标已绑定到本轮请求，后续会围绕 ${GOAL_PROMPT} 收口。\\n"
        : isReasoningFirstVisiblePrompt
          ? "${REASONING_FIRST_VISIBLE_FINAL_TEXT}\\n"
        : isLiveTailCommitPrompt
          ? ${JSON.stringify(liveTailCommitFixtureText)}
        : isTerminalStaleGuardFirstPrompt
          ? "${TERMINAL_STALE_GUARD_FIRST_DONE_TEXT}\\n"
        : isTerminalStaleGuardSecondPrompt
          ? "${TERMINAL_STALE_GUARD_SECOND_TEXT}\\n${TERMINAL_STALE_GUARD_DONE_TEXT}\\n"
        : isWebToolsRenderingPrompt
          ? ${JSON.stringify(webToolsRenderingFixtureText)}
          : isMcpStructuredContentPrompt
          ? "MCP structuredContent 展示验证完成。\\n"
            : isMediaReferencePrompt
              ? ""
              : isSkillsRuntimePrompt
                  ? ${JSON.stringify(SKILLS_RUNTIME_SCENARIO.fixtureText)}
                  : isExplicitSkillsRuntimePrompt
                    ? ${JSON.stringify(SKILLS_RUNTIME_EXPLICIT_SCENARIO.fixtureText)}
                    : isManualEnableSkillsRuntimePrompt
                      ? ${JSON.stringify(SKILLS_RUNTIME_MANUAL_ENABLE_SCENARIO.fixtureText)}
                      : isExpertSkillsRuntimePrompt
                        ? ${JSON.stringify(EXPERT_SKILLS_RUNTIME_SCENARIO.fixtureText)}
                        : isExpertPanelSkillsRuntimePrompt
                          ? ${JSON.stringify(EXPERT_PANEL_SKILLS_RUNTIME_SCENARIO.fixtureText)}
                          : isApprovalRequestFullAccessPrompt
                            ? "${APPROVAL_REQUEST_FULL_ACCESS_DONE_TEXT}\\n"
                          : isApprovalRequestResumeSecondPrompt
                            ? "${APPROVAL_REQUEST_RESUME_SECOND_DONE_TEXT}\\n"
        : "1. 多国外交议题持续升温，地区安全与经贸协商仍是焦点。\\n2. 全球市场继续关注能源、供应链和主要央行政策变化。\\n3. 国际组织呼吁在气候、粮食与人道援助议题上保持协调。\\n";
  const shouldWaitForCancel =
    ((process.env.CLAW_CHAT_FIXTURE_SCENARIO === "cancel" ||
      process.env.CLAW_CHAT_FIXTURE_SCENARIO === "cancel-then-continue") &&
      !isEventReadProbe &&
      !isContinuePrompt) ||
    (process.env.CLAW_CHAT_FIXTURE_SCENARIO === "${INPUTBAR_RICH_RESTORE_SCENARIO}" &&
      isInputbarRichRestorePrompt) ||
    (process.env.CLAW_CHAT_FIXTURE_SCENARIO === "${TERMINAL_CANCELED_AFTER_ANSWER_SCENARIO}" &&
      isTerminalCanceledAfterAnswerPrompt);
  if (shouldWaitForCancel) {
    emitEvents(
      isInputbarRichRestorePrompt
        ? initialEvents.filter((event) => event.type !== "message.delta" && event.type !== "provider.first_text_delta.received")
        : initialEvents
    );
    const startedAt = Date.now();
    while (Date.now() - startedAt < 120000) {
      try {
        const cancelled = cancelSignalPath ? readFileSync(cancelSignalPath, "utf8").trim() : "";
        if (cancelled) {
          process.exit(0);
        }
      } catch {
        // 等待 turnCancel 写入 signal。
      }
      await sleep(100);
    }
    console.error("cancel scenario timed out waiting for turnCancel");
    process.exit(2);
  }

  emitEvents(
    isReasoningFirstVisiblePrompt
      ? initialEvents.filter(
          (event) =>
            event.type !== "message.delta" &&
            event.type !== "provider.first_text_delta.received",
        )
      : initialEvents,
  );
  await sleep(120);
  if (hasProcessPrelude && initialMessageText.length > 0) {
    emitEvents([
      {
        type: "message.completed",
        payload: {
          ...messageDeltaPayload(
            initialMessageText,
            "commentary",
            commentaryItemId
          ),
          role: "assistant",
          status: "completed"
        }
      }
    ]);
    await sleep(80);
  }
  if (isTerminalFailedAfterAnswerPrompt) {
    appendLedgerEntry({
      kind: "terminalFailedAfterAnswerTurnFailed",
      sessionId: input.request?.session?.sessionId,
      turnId: currentTurnId(),
      eventType: "turn.failed",
      partialText: "${TERMINAL_FAILED_AFTER_ANSWER_PARTIAL_TEXT}",
      failureText: "${TERMINAL_FAILED_AFTER_ANSWER_FAILURE_TEXT}"
    });
    emitEvents([
      {
        type: "turn.failed",
        payload: {
          status: "failed",
          message: "${TERMINAL_FAILED_AFTER_ANSWER_FAILURE_TEXT}",
          errorMessage: "${TERMINAL_FAILED_AFTER_ANSWER_FAILURE_TEXT}",
          error_message: "${TERMINAL_FAILED_AFTER_ANSWER_FAILURE_TEXT}",
          failureCategory: "fixture_failed_after_answer",
          retryable: false
        }
      }
    ]);
    process.exit(0);
  }
  if (isTerminalStaleGuardSecondPrompt) {
    const firstTurnEntry = readLedgerEntries().find(
      (entry) =>
        entry?.kind === "turnStart" &&
        entry?.inputText === "${TERMINAL_STALE_GUARD_FIRST_PROMPT}",
    );
    appendLedgerEntry({
      kind: "terminalStaleGuardStaleTerminal",
      sessionId: input.request?.session?.sessionId,
      currentTurnId: currentTurnId(),
      staleTurnId: firstTurnEntry?.turnId ?? "unknown",
      staleEventType: "turn.completed",
      staleDoneText: "${TERMINAL_STALE_GUARD_STALE_DONE_TEXT}"
    });
    await sleep(250);
  }
  if (isReasoningFirstVisiblePrompt) {
    const reasoningStartedAt = new Date().toISOString();
    const reasoningItemId = \`\${currentTurnId() || "turn"}:reasoning:first-visible\`;
    emitEvents([
      {
        type: "reasoning.started",
        payload: {
          reasoningId: reasoningItemId,
          reasoning_id: reasoningItemId,
          status: "in_progress",
          started_at: reasoningStartedAt,
          startedAt: reasoningStartedAt
        }
      },
      {
        type: "reasoning.final",
        payload: {
          reasoningId: reasoningItemId,
          reasoning_id: reasoningItemId,
          text: "${REASONING_FIRST_VISIBLE_TEXT}",
          providerMetadata: {
            backend: "reasoning_first_visible_fixture",
            signature: "reasoning-first-visible-final-signature"
          },
          provider_metadata: {
            backend: "reasoning_first_visible_fixture",
            signature: "reasoning-first-visible-final-signature"
          }
        }
      },
      {
        type: "item.updated",
        payload: {
          itemType: "reasoning",
          itemId: reasoningItemId,
          status: "in_progress",
          canonicalLifecycle: "runtime_message_reasoning.v1",
          item: {
            id: reasoningItemId,
            thread_id: currentThreadId(),
            threadId: currentThreadId(),
            turn_id: currentTurnId(),
            turnId: currentTurnId(),
            type: "reasoning",
            text: "${REASONING_FIRST_VISIBLE_TEXT}",
            summary: ["${REASONING_FIRST_VISIBLE_TEXT}"],
            sequence: 1,
            status: "in_progress",
            started_at: reasoningStartedAt,
            startedAt: reasoningStartedAt,
            updated_at: reasoningStartedAt,
            updatedAt: reasoningStartedAt,
            metadata: {
              provider_metadata: {
                backend: "reasoning_first_visible_fixture",
                signature: "reasoning-first-visible-item-signature"
              }
            }
          }
        }
      }
    ]);
    await sleep(5000);
    emitEvents([
      {
        type: "item.updated",
        payload: {
          itemType: "reasoning",
          itemId: reasoningItemId,
          status: "in_progress",
          canonicalLifecycle: "runtime_message_reasoning.v1",
          item: {
            id: reasoningItemId,
            thread_id: currentThreadId(),
            threadId: currentThreadId(),
            turn_id: currentTurnId(),
            turnId: currentTurnId(),
            type: "reasoning",
            text: "${REASONING_FIRST_VISIBLE_TEXT}",
            summary: ["${REASONING_FIRST_VISIBLE_TEXT}"],
            content: ["${REASONING_FIRST_VISIBLE_CONTENT_TEXT}"],
            sequence: 1,
            status: "in_progress",
            started_at: reasoningStartedAt,
            startedAt: reasoningStartedAt,
            updated_at: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            metadata: {
              provider_metadata: {
                backend: "reasoning_first_visible_fixture",
                signature: "reasoning-first-visible-item-signature"
              }
            }
          }
        }
      },
      {
        type: "reasoning.ended",
        payload: {
          reasoningId: reasoningItemId,
          reasoning_id: reasoningItemId,
          status: "completed"
        }
      },
      {
        type: "provider.first_text_delta.received",
        payload: providerTracePayload("first_text_delta_received", 5200, "running", {
          text_chars: followupText.length,
          textChars: followupText.length
        })
      },
      {
        type: "message.delta",
        payload: messageDeltaPayload(followupText, "final_answer", finalAnswerItemId)
      },
      {
        type: "turn.completed",
        payload: {
          status: "completed",
          text: "${REASONING_FIRST_VISIBLE_FINAL_TEXT}\\n${REASONING_FIRST_VISIBLE_DONE_TEXT}"
        }
      }
    ]);
    process.exit(0);
  }
  if (isLiveTailCommitPrompt) {
    await sleep(1400);
    const completedMessageText = (initialMessageText + followupText).trim();
    const completedAt = new Date().toISOString();
    const repairEventTypes = ["item.completed", "turn.completed"];
    emitEvents([
      {
        type: "item.completed",
        payload: {
          item: {
            id: finalAnswerItemId,
            item_id: finalAnswerItemId,
            itemId: finalAnswerItemId,
            thread_id: currentThreadId(),
            threadId: currentThreadId(),
            turn_id: currentTurnId(),
            turnId: currentTurnId(),
            type: "agentMessage",
            kind: "agentMessage",
            role: "assistant",
            status: "completed",
            text: completedMessageText,
            phase: "final",
            completed_at: completedAt,
            completedAt
          }
        }
      },
      {
        type: "turn.completed",
        payload: {
          status: "completed",
          text: "${LIVE_TAIL_COMMIT_DONE_TEXT}"
        }
      }
    ]);
    appendLedgerEntry({
      kind: "liveTailCommitCompleted",
      sessionId: input.request?.session?.sessionId,
      threadId: currentThreadId(),
      turnId: currentTurnId(),
      itemId: finalAnswerItemId,
      firstText: "${LIVE_TAIL_COMMIT_FIRST_TEXT}",
      overflowMarker: "${LIVE_TAIL_COMMIT_OVERFLOW_MARKER}",
      tableHeader: "${LIVE_TAIL_COMMIT_TABLE_HEADER}",
      tableTail: "${LIVE_TAIL_COMMIT_TABLE_TAIL}",
      droppedEventType: "message.delta",
      repairEventType: repairEventTypes[0],
      terminalEventType: repairEventTypes[1],
      eventType: "turn.completed",
      emittedEventTypes: repairEventTypes,
      completedTextLength: completedMessageText.length,
      completedTextSha256: createHash("sha256")
        .update(completedMessageText)
        .digest("hex")
    });
    process.exit(0);
  }
  if (isMediaReferencePrompt) {
    emitEvents([
      {
        type: "message.delta",
        payload: messageDeltaPayload(
          "${MEDIA_REFERENCE_SUMMARY_TEXT}",
          "final_answer",
          finalAnswerItemId,
        ),
      },
      {
        type: "item.completed",
        payload: {
          item: {
            id: "agent-media-reference-1",
            thread_id: currentThreadId(),
            threadId: currentThreadId(),
            turn_id: currentTurnId(),
            turnId: currentTurnId(),
            type: "media",
            status: "completed",
            uri: mediaReferenceSourcePath || "${MEDIA_REFERENCE_URI}",
            mime_type: "${MEDIA_REFERENCE_MIME_TYPE}"
          }
        }
      },
      {
        type: "turn.completed",
        payload: {
          status: "completed",
          text: "${MEDIA_REFERENCE_DONE_TEXT}"
        }
      }
    ]);
    process.exit(0);
  }
  ${renderBackendToolAndSkillEventScript({
    skillsRuntimeBackendEvents,
    explicitSkillsRuntimeBackendEvents,
    manualEnableSkillsRuntimeBackendEvents,
    expertSkillsRuntimeBackendEvents,
    expertPanelSkillsRuntimeBackendEvents,
  })}
  emitEvents([
    {
      type: "message.delta",
      payload: messageDeltaPayload(followupText, "final_answer", finalAnswerItemId)
    }
  ]);
  if (
    process.env.CLAW_CHAT_FIXTURE_SCENARIO === "complete" &&
    !isEventReadProbe
  ) {
    await sleep(80);
    emitEvents([
      {
        type: "message.completed",
        payload: {
          ...messageDeltaPayload(
            initialMessageText + followupText,
            "final_answer",
            finalAnswerItemId
          ),
          role: "assistant",
          status: "completed"
        }
      }
    ]);
  }
  await sleep(120);
  if (isPlanPrompt) {
    emitEvents([
      {
        type: "plan.final",
        payload: {
          text: ${JSON.stringify(proposedPlanThreadItemText)},
          revisionId: "proposed_plan:fixture-1",
          source: "proposed_plan",
          plan: ${JSON.stringify(PLAN_STEPS)}
        }
      }
    ]);
    await sleep(80);
  }
  emitEvents([
    {
      type: "turn.completed",
      payload: {
        status: "completed",
        text: isWebToolsRenderingPrompt ? followupText : assistantDoneText
      }
    }
  ]);
  process.exit(0);
}

emitEvents([]);
`,
    { mode: 0o755 },
  );
}
