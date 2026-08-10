import {
  ASSISTANT_DONE_TEXT,
  NEWS_PROMPT,
  PLAN_DONE_TEXT,
  PLAN_PROMPT,
  PLAN_STEPS,
  REASONING_FIRST_VISIBLE_CONTENT_TEXT,
  REASONING_FIRST_VISIBLE_DONE_TEXT,
  REASONING_FIRST_VISIBLE_FINAL_TEXT,
  REASONING_FIRST_VISIBLE_PROMPT,
  REASONING_FIRST_VISIBLE_TEXT,
  SKILLS_RUNTIME_SCENARIO,
} from "./claw-chat-current-fixture-constants.mjs";
import { evaluatePageSnapshot } from "./claw-chat-current-fixture-rpc.mjs";
import {
  assert,
  sanitizeJson,
  sleep,
} from "./claw-chat-current-fixture-utils.mjs";

export function countTextOccurrences(text, needle) {
  if (!text || !needle) {
    return 0;
  }
  return text.split(needle).length - 1;
}

export function isGuiCanceledSnapshotReady(
  snapshot,
  { partialText = "" } = {},
) {
  if (
    !snapshot?.hasPrompt ||
    snapshot.textareaVisible !== true ||
    snapshot.textareaDisabled !== false ||
    snapshot.stopButtonVisible !== false
  ) {
    return false;
  }

  if (partialText) {
    return snapshot.hasPartialText === true;
  }

  return snapshot.hasStoppedCopy === true;
}

export function isGuiChatCompletedSnapshotReady(
  snapshot,
  { requiredAssistantVisibleTexts = [], expectedIdentity = null } = {},
) {
  const hasRequiredAssistantVisibleTexts = (
    snapshot?.requiredVisibleTextHits || []
  ).every((hit) => hit.occurrences > 0);
  const scopedDedupeGuardHits =
    snapshot?.assistantScopeDedupeGuardHits || snapshot?.dedupeGuardHits || [];
  const scopedDisallowedVisibleTextHits =
    snapshot?.disallowedVisibleTextHits || [];
  const hasExpectedAssistantContent =
    requiredAssistantVisibleTexts.length > 0
      ? snapshot?.hasAssistantSummary && hasRequiredAssistantVisibleTexts
      : snapshot?.hasAssistantSummary || snapshot?.hasDoneText;
  const contentReady =
    snapshot?.hasPrompt &&
    hasExpectedAssistantContent &&
    snapshot?.textareaVisible === true &&
    snapshot?.textareaDisabled === false &&
    snapshot?.stopButtonVisible === false &&
    scopedDedupeGuardHits.every((hit) => hit.occurrences <= 1) &&
    scopedDisallowedVisibleTextHits.every((hit) => hit.occurrences === 0);
  if (!contentReady || !expectedIdentity) {
    return contentReady;
  }

  const expectedSessionId = String(expectedIdentity.sessionId || "").trim();
  const expectedTurnId = String(expectedIdentity.turnId || "").trim();
  const runtimeTurnStatus = String(
    snapshot.completionScope?.runtimeTurnStatus || "",
  )
    .trim()
    .toLowerCase();
  return (
    expectedSessionId.length > 0 &&
    expectedTurnId.length > 0 &&
    snapshot.textareaSessionId === expectedSessionId &&
    snapshot.completionScope?.runtimeTurnId === expectedTurnId &&
    snapshot.completionScope?.assistantRuntimeTurnId === expectedTurnId &&
    ["canceled", "cancelled", "completed", "failed", "interrupted"].includes(
      runtimeTurnStatus,
    )
  );
}

export async function waitForGuiChatCompleted(
  page,
  options,
  {
    prompt = NEWS_PROMPT,
    doneText = ASSISTANT_DONE_TEXT,
    summaryText = "今日国际新闻简要整理",
    requiredVisibleTexts,
    dedupeGuardTexts = [],
    disallowedVisibleTexts = ["legacy_tool_event"],
    expectedIdentity = null,
  } = {},
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  const requiredAssistantVisibleTexts =
    requiredVisibleTexts ??
    (prompt === NEWS_PROMPT && doneText === ASSISTANT_DONE_TEXT
      ? ["全球市场继续关注能源", "国际组织呼吁"]
      : []);
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({
        prompt,
        doneText,
        summaryText,
        requiredAssistantVisibleTexts,
        dedupeGuardTexts,
        disallowedVisibleTexts,
      }) => {
        const text = document.body?.innerText || "";
        const mainText = document.querySelector("main")?.innerText || text;
        const turnGroups = Array.from(
          document.querySelectorAll('[data-testid="message-turn-group"]'),
        );
        const promptTurnGroups = turnGroups.filter((group) =>
          (group.innerText || "").includes(prompt),
        );
        const messageListScope =
          document.querySelector('[data-testid="message-list-column"]') ||
          document;
        const readTurnGroupSnapshot = (group) => {
          const groupText = group?.innerText || mainText;
          const assistantBubbles = Array.from(
            (group || messageListScope).querySelectorAll(
              '[data-message-role="assistant"]',
            ),
          );
          const lastAssistantMessageId =
            group?.getAttribute("data-last-assistant-message-id") || "";
          const assistantScope =
            assistantBubbles.find(
              (bubble) =>
                bubble.getAttribute("data-message-id") ===
                lastAssistantMessageId,
            ) ??
            assistantBubbles[assistantBubbles.length - 1] ??
            group;
          const assistantScopeText = assistantScope?.innerText || groupText;
          const assistantTurnText = assistantBubbles
            .map((bubble) => bubble.innerText || "")
            .join("\n");
          const hasRequiredAssistantVisibleTexts =
            requiredAssistantVisibleTexts.every((requiredText) =>
              groupText.includes(requiredText),
            );

          return {
            group,
            groupText,
            assistantScope,
            assistantScopeText,
            assistantTurnText,
            hasExpectedAssistantContent:
              requiredAssistantVisibleTexts.length > 0
                ? assistantScopeText.includes(summaryText) &&
                  hasRequiredAssistantVisibleTexts
                : assistantScopeText.includes(summaryText) ||
                  assistantScopeText.includes(doneText),
          };
        };
        const scopedSnapshot =
          [...promptTurnGroups]
            .reverse()
            .map(readTurnGroupSnapshot)
            .find((candidate) => candidate.hasExpectedAssistantContent) ??
          (promptTurnGroups.length > 0
            ? readTurnGroupSnapshot(
                promptTurnGroups[promptTurnGroups.length - 1],
              )
            : readTurnGroupSnapshot(null));
        const scopedTurnGroup = scopedSnapshot.group;
        const scopedText = scopedSnapshot.groupText;
        const assistantScope = scopedSnapshot.assistantScope;
        const assistantScopeText = scopedSnapshot.assistantScopeText;
        const assistantTurnText = scopedSnapshot.assistantTurnText;
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
        const approvalRecordRoot = scopedTurnGroup ?? messageListScope;
        const compactTimelinePreviewCount = approvalRecordRoot.querySelectorAll(
          '[data-testid="message-list-historical-timeline-preview:leading"]',
        ).length;
        const approvalRecords = Array.from(
          approvalRecordRoot.querySelectorAll(
            '[data-testid="timeline-approval-record"]',
          ),
        ).map((record) => ({
          text: record.textContent || "",
          lineBreaks: ((record.textContent || "").match(/\n/gu) || []).length,
          innerTextLineBreaks: ((record.innerText || "").match(/\n/gu) || [])
            .length,
        }));
        const approvalRecordText = approvalRecords
          .map((record) => record.text)
          .join("\n");
        const readVisiblePlanOwner = ({ kind, selector, itemSelector }) => {
          const root = document.querySelector(selector);
          const ownerRect = root?.getBoundingClientRect();
          const ownerStyle = root ? window.getComputedStyle(root) : null;
          const visible = Boolean(
            root &&
            ownerRect &&
            ownerRect.width > 16 &&
            ownerRect.height > 8 &&
            ownerStyle?.visibility !== "hidden" &&
            ownerStyle?.display !== "none",
          );
          return {
            kind,
            visible,
            text: root?.textContent || "",
            itemCount: itemSelector
              ? root?.querySelectorAll(itemSelector).length || 0
              : 0,
          };
        };
        const planOwners = [
          readVisiblePlanOwner({
            kind: "run-control-plan",
            selector: '[data-testid="task-center-run-control-plan"]',
            itemSelector: '[data-testid="task-center-run-control-plan-item"]',
          }),
          readVisiblePlanOwner({
            kind: "task-rail-plan",
            selector: '[data-testid="task-center-task-rail-plan"]',
            itemSelector: '[data-testid="task-center-task-rail-plan-item"]',
          }),
          readVisiblePlanOwner({
            kind: "message-plan-block",
            selector: '[data-testid="agent-plan-block"]',
            itemSelector: null,
          }),
        ].filter((owner) => owner.visible);
        const planOwnerText = planOwners.map((owner) => owner.text).join("\n");
        const planDecisionPanel = document.querySelector(
          '[data-testid="plan-composer-decision-panel"]',
        );
        const planDecisionRect = planDecisionPanel?.getBoundingClientRect();
        const planDecisionStyle = planDecisionPanel
          ? window.getComputedStyle(planDecisionPanel)
          : null;
        const planDecisionVisible = Boolean(
          planDecisionPanel &&
          planDecisionRect &&
          planDecisionRect.width > 320 &&
          planDecisionRect.height > 48 &&
          planDecisionStyle?.visibility !== "hidden" &&
          planDecisionStyle?.display !== "none",
        );
        const legacyUpdatePlanVisibleHits = [
          "UpdatePlanTool",
          "update_plan",
        ].filter((label) =>
          [mainText, planOwnerText].some((textValue) =>
            textValue.includes(label),
          ),
        );
        const approvalLegacyFragments = [
          prompt,
          "历史记录只读",
          "歷史記錄只讀",
          "History records are read-only",
          "履歴記録は読み取り専用",
          "히스토리 기록은 읽기 전용",
          "请求",
          "請求",
          "Request",
          "リクエスト",
          "요청",
          "范围",
          "範圍",
          "範囲",
          "범위",
          "Scope",
          "ソース",
          "来源",
          "來源",
          "Source",
          "출처",
        ];
        return {
          url: window.location.href,
          hasPrompt: scopedText.includes(prompt),
          hasAssistantSummary: assistantScopeText.includes(summaryText),
          requiredVisibleTextHits: requiredAssistantVisibleTexts.map(
            (requiredText) => ({
              text: requiredText,
              occurrences: scopedText.split(requiredText).length - 1,
            }),
          ),
          summaryOccurrences: text.split(summaryText).length - 1,
          mainSummaryOccurrences: mainText.split(summaryText).length - 1,
          dedupeGuardHits: dedupeGuardTexts.map((guardText) => ({
            text: guardText,
            occurrences: mainText.split(guardText).length - 1,
          })),
          disallowedVisibleTextHits: disallowedVisibleTexts.map(
            (guardText) => ({
              text: guardText,
              occurrences: assistantTurnText.split(guardText).length - 1,
            }),
          ),
          scopedSummaryOccurrences: scopedText.split(summaryText).length - 1,
          assistantScopeSummaryOccurrences:
            assistantScopeText.split(summaryText).length - 1,
          scopedDedupeGuardHits: dedupeGuardTexts.map((guardText) => ({
            text: guardText,
            occurrences: scopedText.split(guardText).length - 1,
          })),
          assistantScopeDedupeGuardHits: dedupeGuardTexts.map((guardText) => ({
            text: guardText,
            occurrences: assistantTurnText.split(guardText).length - 1,
          })),
          completionScope: {
            foundTurnGroup: Boolean(scopedTurnGroup),
            runtimeTurnId:
              scopedTurnGroup?.getAttribute("data-runtime-turn-id") || "",
            runtimeTurnStatus:
              scopedTurnGroup?.getAttribute("data-runtime-turn-status") || "",
            lastAssistantMessageId:
              scopedTurnGroup?.getAttribute("data-last-assistant-message-id") ||
              "",
            assistantMessageId:
              assistantScope?.getAttribute("data-message-id") || "",
            assistantRuntimeTurnId:
              assistantScope?.getAttribute("data-runtime-turn-id") || "",
            text: scopedText,
            assistantText: assistantScopeText,
          },
          hasDoneText: assistantScopeText.includes(doneText),
          hasEpochFallbackTitle: text.includes("任务 1970/1/1"),
          textareaVisible,
          textareaDisabled:
            textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
          textareaSessionId:
            textarea instanceof HTMLTextAreaElement
              ? textarea.dataset.sessionId || null
              : null,
          textareaValue:
            textarea instanceof HTMLTextAreaElement ? textarea.value : null,
          stopButtonVisible,
          hasMessageList: Boolean(
            document.querySelector('[data-testid="message-list"]') ||
            document.querySelector('[data-testid="message-list-frame"]'),
          ),
          approvalRecordShape: {
            recordCount: approvalRecords.length,
            maxLineBreaks: Math.max(
              0,
              ...approvalRecords.map((record) => record.lineBreaks),
            ),
            promptInRecord: approvalRecordText.includes(prompt),
            legacyDetailFragmentHits: approvalLegacyFragments.filter(
              (fragment) =>
                typeof fragment === "string" &&
                fragment.trim() &&
                approvalRecordText.includes(fragment),
            ),
            texts: approvalRecords.map((record) => record.text),
          },
          compactTimelinePreviewCount,
          planUiAbsentWithoutProposedPlan:
            planOwners.length === 0 &&
            planDecisionVisible === false &&
            legacyUpdatePlanVisibleHits.length === 0,
          planUiAbsence: {
            planOwnerCount: planOwners.length,
            planOwnerKinds: planOwners.map((owner) => owner.kind),
            planOwnerItemCounts: planOwners.map((owner) => ({
              kind: owner.kind,
              itemCount: owner.itemCount,
            })),
            planDecisionVisible,
            legacyUpdatePlanVisibleHits,
          },
          bodyText: text,
          mainText,
        };
      },
      {
        prompt,
        doneText,
        summaryText,
        requiredAssistantVisibleTexts,
        dedupeGuardTexts,
        disallowedVisibleTexts,
      },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      isGuiChatCompletedSnapshotReady(snapshot, {
        requiredAssistantVisibleTexts,
        expectedIdentity,
      })
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Claw GUI 未完成输入闭环: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

export async function waitForGuiSkillsRuntimeCompleted(
  page,
  options,
  scenario = SKILLS_RUNTIME_SCENARIO,
  expectedIdentity = null,
) {
  return await waitForGuiChatCompleted(page, options, {
    prompt: scenario.prompt,
    doneText: scenario.doneText,
    summaryText: scenario.guiSummaryText ?? scenario.summaryText,
    dedupeGuardTexts: scenario.dedupeGuardTexts ?? [],
    disallowedVisibleTexts: scenario.disallowedVisibleTexts ?? [
      "legacy_tool_event",
    ],
    expectedIdentity,
  });
}

function reasoningFirstVisibleSnapshotFromDom({
  prompt,
  reasoningContentText,
  reasoningText,
  finalText,
  doneText,
}) {
  const text = document.body?.innerText || "";
  const turnGroups = Array.from(
    document.querySelectorAll('[data-testid="message-turn-group"]'),
  );
  const promptTurnGroup =
    [...turnGroups]
      .reverse()
      .find((group) => (group.innerText || "").includes(prompt)) ?? null;
  const scope = promptTurnGroup ?? document;
  const scopedText = promptTurnGroup?.innerText || text;
  const processBlocks = Array.from(
    scope.querySelectorAll(
      [
        '[data-testid="agent-thread-timeline:leading"]',
        '[data-testid="assistant-primary-timeline-shell"]',
        '[data-testid^="agent-thread-block:"][data-testid$=":process"]',
      ].join(","),
    ),
  ).map((node) => ({
    testId: node.getAttribute("data-testid") || "",
    text: node.textContent || "",
    open: node instanceof HTMLDetailsElement ? node.open : null,
  }));
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
  const reasoningIndex = scopedText.indexOf(reasoningText);
  const reasoningContentIndex = scopedText.indexOf(reasoningContentText);
  const finalAnswerIndex = scopedText.indexOf(finalText);
  const hasThinkingLabel =
    scopedText.includes("思考中") || scopedText.includes("已完成思考");
  const hasReasoningProcess =
    hasThinkingLabel ||
    processBlocks.some(
      (block) =>
        block.text.includes(reasoningText) ||
        block.text.includes("思考中") ||
        block.text.includes("已完成思考"),
    );
  const startupNoteVisible =
    text.includes("启动处理流程") || text.includes("已接收请求");

  return {
    url: window.location.href,
    hasPrompt: scopedText.includes(prompt),
    hasReasoningText: reasoningIndex >= 0,
    hasReasoningContentText: reasoningContentIndex >= 0,
    reasoningContentIndex,
    reasoningProcessOpen: processBlocks.some(
      (block) => block.open === true && block.text.includes(reasoningText),
    ),
    hasReasoningProcess,
    hasThinkingLabel,
    hasFinalText: finalAnswerIndex >= 0,
    hasDoneText: scopedText.includes(doneText),
    reasoningIndex,
    finalAnswerIndex,
    hasReasoningBeforeFinalAnswer:
      reasoningIndex >= 0 &&
      (finalAnswerIndex < 0 || reasoningIndex < finalAnswerIndex),
    startupNoteVisible,
    processBlockCount: processBlocks.length,
    processBlocks,
    scopedText,
    bodyText: text,
    textareaVisible,
    textareaDisabled:
      textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
    textareaValue:
      textarea instanceof HTMLTextAreaElement ? textarea.value : null,
    stopButtonVisible,
    hasMessageList: Boolean(
      document.querySelector('[data-testid="message-list"]') ||
      document.querySelector('[data-testid="message-list-frame"]'),
    ),
  };
}

async function evaluateReasoningFirstVisibleSnapshot(page) {
  return await evaluatePageSnapshot(
    page,
    reasoningFirstVisibleSnapshotFromDom,
    {
      prompt: REASONING_FIRST_VISIBLE_PROMPT,
      reasoningContentText: REASONING_FIRST_VISIBLE_CONTENT_TEXT,
      reasoningText: REASONING_FIRST_VISIBLE_TEXT,
      finalText: REASONING_FIRST_VISIBLE_FINAL_TEXT,
      doneText: REASONING_FIRST_VISIBLE_DONE_TEXT,
    },
  );
}

export async function waitForGuiReasoningFirstVisibleBeforeAnswer(
  page,
  options,
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 45_000)) {
    const snapshot = await evaluateReasoningFirstVisibleSnapshot(page);
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.hasPrompt &&
      snapshot.hasReasoningText &&
      snapshot.hasReasoningProcess &&
      snapshot.hasReasoningBeforeFinalAnswer &&
      snapshot.hasFinalText === false &&
      snapshot.hasDoneText === false &&
      snapshot.startupNoteVisible === false
    ) {
      return sanitizeJson({
        ...snapshot,
        reasoningFirstVisibleBeforeAnswerCaptured: true,
      });
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Claw GUI 未在最终回答前显示 reasoning: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

export async function waitForGuiReasoningFirstVisibleCompleted(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluateReasoningFirstVisibleSnapshot(page);
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.hasPrompt &&
      snapshot.hasReasoningText &&
      snapshot.hasFinalText &&
      snapshot.hasReasoningBeforeFinalAnswer &&
      snapshot.hasReasoningContentText === false &&
      snapshot.startupNoteVisible === false &&
      snapshot.textareaVisible &&
      snapshot.textareaDisabled === false &&
      snapshot.stopButtonVisible === false
    ) {
      const promptTurnGroup = page
        .locator('[data-testid="message-turn-group"]')
        .filter({ hasText: REASONING_FIRST_VISIBLE_PROMPT })
        .last();
      const historicalPreview = promptTurnGroup
        .locator('[data-testid^="message-list-historical-timeline-preview:"]')
        .last();
      const historicalPreviewCount = await historicalPreview.count();
      if (historicalPreviewCount > 0) {
        await historicalPreview.click();
      }

      const reasoningBlock = promptTurnGroup
        .locator(
          'details[data-testid*="agent-thread-block:"][data-testid$=":process"]',
        )
        .last();
      await reasoningBlock.waitFor({
        state: "visible",
        timeout: Math.min(options.timeoutMs, 30_000),
      });
      assert(
        (await reasoningBlock.count()) === 1,
        "Claw GUI 完成态缺少可展开的 reasoning 过程块",
      );
      const detailsAvailable =
        (await reasoningBlock.getAttribute("data-details-available")) ===
        "true";
      assert(detailsAvailable, "Claw GUI 完成态 reasoning 过程块不可展开");
      const reasoningSummary = reasoningBlock.locator(":scope > summary");
      let reasoningOpenedByClick = await reasoningBlock.evaluate(
        (node) => node instanceof HTMLDetailsElement && node.open,
      );
      if (!reasoningOpenedByClick) {
        for (let attempt = 0; attempt < 3; attempt += 1) {
          await reasoningSummary.click();
          reasoningOpenedByClick = await reasoningBlock.evaluate(
            (node) => node instanceof HTMLDetailsElement && node.open,
          );
          if (reasoningOpenedByClick) {
            break;
          }
        }
      }
      assert(
        reasoningOpenedByClick,
        "Claw GUI 完成态 reasoning 过程块点击后未展开",
      );

      const expandedStartedAt = Date.now();
      let expandedSnapshot = null;
      while (
        Date.now() - expandedStartedAt <
        Math.min(options.timeoutMs, 20_000)
      ) {
        expandedSnapshot = await evaluateReasoningFirstVisibleSnapshot(page);
        if (
          expandedSnapshot?.hasReasoningContentText === true &&
          expandedSnapshot?.reasoningProcessOpen === true
        ) {
          return sanitizeJson({
            ...expandedSnapshot,
            historicalReasoningPreviewExpanded: historicalPreviewCount > 0,
            reasoningDetailsAvailable: detailsAvailable,
            reasoningOpenedByClick,
            reasoningContentExpandedAfterCompletion: true,
          });
        }
        await sleep(options.intervalMs);
      }
      throw new Error(
        `Claw GUI 完成态 reasoning 展开后未显示 canonical content: ${JSON.stringify(
          sanitizeJson(expandedSnapshot),
        )}`,
      );
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Claw GUI reasoning-first 场景未完成: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

export async function waitForGuiPlanCompleted(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ prompt, doneText, planSteps }) => {
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
        const readVisiblePlanOwner = ({
          kind,
          selector,
          itemSelector,
          revisionSelector,
        }) => {
          const root = document.querySelector(selector);
          const rect = root?.getBoundingClientRect();
          const ownerStyle = root ? window.getComputedStyle(root) : null;
          const visible = Boolean(
            root &&
            rect &&
            rect.width > 16 &&
            rect.height > 8 &&
            ownerStyle?.visibility !== "hidden" &&
            ownerStyle?.display !== "none",
          );
          const revision = revisionSelector
            ? root?.querySelector(revisionSelector)
            : null;
          return {
            kind,
            visible,
            text: root?.textContent || "",
            itemCount: itemSelector
              ? root?.querySelectorAll(itemSelector).length || 0
              : 0,
            revisionId: revision?.getAttribute("data-plan-revision-id") || null,
            revisionSource: revision?.getAttribute("data-plan-source") || null,
            revisionTurnId: revision?.getAttribute("data-plan-turn-id") || null,
          };
        };
        const planOwners = [
          readVisiblePlanOwner({
            kind: "run-control-plan",
            selector: '[data-testid="task-center-run-control-plan"]',
            itemSelector: '[data-testid="task-center-run-control-plan-item"]',
            revisionSelector:
              '[data-testid="task-center-run-control-plan-revision"]',
          }),
          readVisiblePlanOwner({
            kind: "task-rail-plan",
            selector: '[data-testid="task-center-task-rail-plan"]',
            itemSelector: '[data-testid="task-center-task-rail-plan-item"]',
            revisionSelector:
              '[data-testid="task-center-task-rail-plan-revision"]',
          }),
          readVisiblePlanOwner({
            kind: "message-plan-block",
            selector: '[data-testid="agent-plan-block"]',
            itemSelector: null,
            revisionSelector: null,
          }),
        ].filter((owner) => owner.visible);
        const planOwnerText = planOwners.map((owner) => owner.text).join("\n");
        const planOwnerStepHits = planSteps.map((step) => ({
          step: step.step,
          visible: planOwnerText.includes(step.step),
          owners: planOwners
            .filter((owner) => owner.text.includes(step.step))
            .map((owner) => owner.kind),
        }));
        const planOwnerKindsWithAllSteps = planOwners
          .filter((owner) =>
            planSteps.every((step) => owner.text.includes(step.step)),
          )
          .map((owner) => owner.kind);
        const planDecisionPanel = document.querySelector(
          '[data-testid="plan-composer-decision-panel"][data-layout="composer-drawer"]',
        );
        const planDecisionText = planDecisionPanel?.textContent || "";
        const planDecisionRect = planDecisionPanel?.getBoundingClientRect();
        const planDecisionStyle = planDecisionPanel
          ? window.getComputedStyle(planDecisionPanel)
          : null;
        const planDecisionVisible = Boolean(
          planDecisionPanel &&
          planDecisionRect &&
          planDecisionRect.width > 320 &&
          planDecisionRect.height > 48 &&
          planDecisionStyle?.visibility !== "hidden" &&
          planDecisionStyle?.display !== "none",
        );
        const planDecisionRevision = planDecisionPanel?.querySelector(
          '[data-testid="plan-composer-revision-status"]',
        );
        const planDecisionRevisionId =
          planDecisionRevision?.getAttribute("data-plan-revision-id") || null;
        return {
          url: window.location.href,
          hasPrompt: text.includes(prompt),
          hasPlanIntro: text.includes("我先给出计划"),
          hasDoneText: text.includes(doneText),
          hasPlanSection: planOwners.length > 0,
          hasAllPlanSteps: planOwnerStepHits.every((hit) => hit.visible),
          planStepHits: planOwnerStepHits,
          proposedPlanVisible: planOwnerStepHits.every((hit) => hit.visible),
          planOwnerHasAllSteps: planOwnerStepHits.every((hit) => hit.visible),
          planOwnerKinds: planOwners.map((owner) => owner.kind),
          planOwnerKindsWithAllSteps,
          planOwnerRevisionIds: planOwners
            .map((owner) => owner.revisionId)
            .filter(Boolean),
          planOwnerRevisionSources: planOwners
            .map((owner) => owner.revisionSource)
            .filter(Boolean),
          planOwners,
          textareaVisible,
          textareaDisabled:
            textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
          textareaValue:
            textarea instanceof HTMLTextAreaElement ? textarea.value : null,
          stopButtonVisible,
          planDecisionVisible,
          planDecisionText,
          planDecisionHasTitle: planDecisionText.includes("实施此计划"),
          planDecisionHasAcceptOption:
            planDecisionText.includes("是，实施此计划"),
          planDecisionHasAdjustInput: Boolean(
            planDecisionPanel?.querySelector(
              '[data-testid="plan-composer-adjust-input"]',
            ),
          ),
          planDecisionHasEscHint: planDecisionText.includes("ESC"),
          planDecisionRevisionBound: Boolean(planDecisionRevisionId),
          planDecisionRevisionId,
          planDecisionRevisionSource:
            planDecisionRevision?.getAttribute("data-plan-source") || null,
          planDecisionRevisionTurnId:
            planDecisionRevision?.getAttribute("data-plan-turn-id") || null,
          bodyText: text,
          taskRailText: planOwnerText,
        };
      },
      { prompt: PLAN_PROMPT, doneText: PLAN_DONE_TEXT, planSteps: PLAN_STEPS },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.hasPrompt &&
      snapshot.planOwnerHasAllSteps &&
      snapshot.planDecisionVisible &&
      snapshot.planDecisionHasTitle &&
      snapshot.planDecisionHasAcceptOption &&
      snapshot.planDecisionHasAdjustInput &&
      snapshot.planDecisionRevisionBound &&
      snapshot.textareaVisible === false &&
      snapshot.stopButtonVisible === false
    ) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Claw GUI 未显示计划轨: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

export async function waitForStopButtonVisibleAndClick(
  page,
  options,
  {
    prompt = NEWS_PROMPT,
    visibleOutputText = "以下是今日国际新闻简要整理",
    requireVisibleOutput = false,
  } = {},
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ prompt, visibleOutputText }) => {
        const text = document.body?.innerText || "";
        const turnGroups = Array.from(
          document.querySelectorAll('[data-testid="message-turn-group"]'),
        );
        const promptTurnGroup =
          [...turnGroups]
            .reverse()
            .find((group) => (group.innerText || "").includes(prompt)) ?? null;
        const scope = promptTurnGroup ?? document;
        const scopedText = promptTurnGroup?.innerText || text;
        const statusNodes = Array.from(
          scope.querySelectorAll(
            [
              '[data-testid="assistant-first-token-runtime-status"]',
              '[data-testid="message-runtime-status-pill"]',
              '[data-testid="inputbar-runtime-status-line"]',
            ].join(","),
          ),
        );
        const statusSnapshots = statusNodes.map((node) => ({
          testId: node.getAttribute("data-testid") || "",
          status: node.getAttribute("data-status") || "",
          text: node.textContent || "",
        }));
        const startupNoteVisible =
          text.includes("启动处理流程") || text.includes("已接收请求");
        const buttons = Array.from(document.querySelectorAll("button")).map(
          (button, index) => {
            const label = [
              button.getAttribute("title") || "",
              button.textContent || "",
              button.getAttribute("aria-label") || "",
            ].join("\n");
            return {
              index,
              label,
              disabled: button.disabled,
              visible: Boolean(
                button.offsetParent ||
                button.getClientRects().length > 0 ||
                window.getComputedStyle(button).position === "fixed",
              ),
              isStop:
                !button.disabled &&
                (label.includes("停止") ||
                  label.includes("终止") ||
                  /\bStop\b/i.test(label)),
            };
          },
        );
        const hasRunningStatus =
          scopedText.includes("正在输出") ||
          scopedText.includes("正在生成") ||
          statusSnapshots.some(
            (entry) =>
              entry.status === "running" ||
              entry.text.includes("正在输出") ||
              entry.text.includes("正在生成"),
          ) ||
          buttons.some(
            (button) =>
              button.visible &&
              (button.label.includes("正在输出") ||
                button.label.includes("正在生成")),
          );
        return {
          url: window.location.href,
          hasPrompt: text.includes(prompt),
          hasAssistantSummary: text.includes("今日国际新闻简要整理"),
          hasVisibleAssistantOutput: visibleOutputText
            ? scopedText.includes(visibleOutputText)
            : true,
          hasRunningStatus,
          startupNoteVisible,
          statusSnapshots,
          stopButtons: buttons.filter((button) => button.isStop),
          buttonLabels: buttons
            .filter((button) => button.label.trim().length > 0)
            .slice(0, 80)
            .map((button) => button.label),
          scopedText,
          bodyText: text,
        };
      },
      { prompt, visibleOutputText },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.stopButtons?.length > 0 &&
      (!requireVisibleOutput || snapshot.hasVisibleAssistantOutput === true)
    ) {
      const clicked = await page.evaluate(() => {
        const buttons = Array.from(document.querySelectorAll("button"));
        const stopButton = buttons.find((button) => {
          const label = [
            button.getAttribute("title") || "",
            button.textContent || "",
            button.getAttribute("aria-label") || "",
          ].join("\n");
          return (
            !button.disabled &&
            (label.includes("停止") ||
              label.includes("终止") ||
              /\bStop\b/i.test(label))
          );
        });
        if (stopButton instanceof HTMLElement) {
          stopButton.click();
          return {
            clicked: true,
            label:
              stopButton.getAttribute("aria-label") ||
              stopButton.getAttribute("title") ||
              stopButton.textContent ||
              "stop",
          };
        }
        return { clicked: false };
      });
      assert(
        clicked?.clicked,
        `停止按钮出现但点击失败: ${JSON.stringify(sanitizeJson(clicked))}`,
      );
      return {
        beforeClick: sanitizeJson(snapshot),
        clicked: sanitizeJson(clicked),
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Claw GUI 未出现停止按钮: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

export async function waitForGuiChatCanceled(
  page,
  options,
  { prompt = NEWS_PROMPT, partialText = "" } = {},
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ prompt, partialText }) => {
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
        const approvalRecords = Array.from(
          document.querySelectorAll('[data-testid="timeline-approval-record"]'),
        ).map((record) => ({
          text: record.textContent || "",
          lineBreaks: ((record.textContent || "").match(/\n/gu) || []).length,
          innerTextLineBreaks: ((record.innerText || "").match(/\n/gu) || [])
            .length,
        }));
        const approvalRecordText = approvalRecords
          .map((record) => record.text)
          .join("\n");
        const approvalLegacyFragments = [
          prompt,
          "历史记录只读",
          "歷史記錄只讀",
          "History records are read-only",
          "履歴記録は読み取り専用",
          "히스토리 기록은 읽기 전용",
          "请求",
          "請求",
          "Request",
          "リクエスト",
          "요청",
          "范围",
          "範圍",
          "範囲",
          "범위",
          "Scope",
          "ソース",
          "来源",
          "來源",
          "Source",
          "출처",
        ];
        return {
          url: window.location.href,
          hasPrompt: text.includes(prompt),
          hasAssistantSummary: text.includes("今日国际新闻简要整理"),
          hasPartialText: partialText ? text.includes(partialText) : null,
          hasStoppedCopy:
            text.includes("已停止") ||
            text.includes("本轮已中止") ||
            /\bStopped\b/i.test(text) ||
            /\bCanceled\b/i.test(text),
          textareaVisible,
          textareaDisabled:
            textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
          textareaValue:
            textarea instanceof HTMLTextAreaElement ? textarea.value : null,
          stopButtonVisible,
          approvalRecordShape: {
            recordCount: approvalRecords.length,
            maxLineBreaks: Math.max(
              0,
              ...approvalRecords.map((record) => record.lineBreaks),
            ),
            promptInRecord: approvalRecordText.includes(prompt),
            legacyDetailFragmentHits: approvalLegacyFragments.filter(
              (fragment) =>
                typeof fragment === "string" &&
                fragment.trim() &&
                approvalRecordText.includes(fragment),
            ),
            texts: approvalRecords.map((record) => record.text),
          },
          compactTimelinePreviewCount: document.querySelectorAll(
            '[data-testid="message-list-historical-timeline-preview:leading"]',
          ).length,
          bodyText: text,
        };
      },
      { prompt, partialText },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (isGuiCanceledSnapshotReady(snapshot, { partialText })) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Claw GUI 未完成取消闭环: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}
