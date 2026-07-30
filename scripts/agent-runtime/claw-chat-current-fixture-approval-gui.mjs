import {
  APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT,
  APPROVAL_REQUEST_RESUME_COMMAND,
  APPROVAL_REQUEST_RESUME_REQUEST_ID,
  APPROVAL_REQUEST_RESUME_SECOND_DONE_TEXT,
  APPROVAL_REQUEST_RESUME_SECOND_PROMPT_MARKER,
  APPROVAL_REQUEST_RESUME_TOOL_NAME,
} from "./claw-chat-current-fixture-constants.mjs";
import { evaluatePageSnapshot } from "./claw-chat-current-fixture-rpc.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

export async function waitForGuiApprovalPending(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ requestId, promptText, commandText, toolName }) => {
        const isDecisionLabel = (decision, label) => {
          const normalized = String(label || "").trim();
          const labelTokens = normalized
            .split(/\s+/u)
            .map((token) => token.trim())
            .filter(Boolean);
          if (decision === "allow_for_session") {
            return (
              normalized.includes("本会话允许") ||
              normalized.includes("本會話允許") ||
              normalized.includes("このセッションで許可") ||
              normalized.includes("세션") ||
              normalized.includes(
                "agentChat.inputbar.approval.action.allowForSession",
              ) ||
              /\bAllow for session\b/i.test(normalized) ||
              labelTokens.some((token) => /^session$/i.test(token))
            );
          }
          if (decision === "decline") {
            return (
              normalized.includes("拒绝") ||
              normalized.includes("拒絕") ||
              normalized.includes("却下") ||
              normalized.includes("거부") ||
              normalized.includes(
                "agentChat.inputbar.approval.action.decline",
              ) ||
              /\bDecline\b/i.test(normalized) ||
              /\bDeny\b/i.test(normalized)
            );
          }
          if (decision === "cancel") {
            return (
              normalized.includes("取消") ||
              normalized.includes("中止") ||
              normalized.includes("キャンセル") ||
              normalized.includes("취소") ||
              normalized.includes(
                "agentChat.inputbar.approval.action.cancel",
              ) ||
              /\bCancel\b/i.test(normalized) ||
              /\bAbort\b/i.test(normalized)
            );
          }
          return false;
        };
        const bodyText = document.body?.innerText || "";
        const requestSection =
          Array.from(document.querySelectorAll("[data-request-id]")).find(
            (element) => element.getAttribute("data-request-id") === requestId,
          ) ?? null;
        const approvalPrompt = document.querySelector(
          '[data-testid="inputbar-approval-prompt"]',
        );
        const section = requestSection ?? approvalPrompt;
        const sectionText = section?.textContent || "";
        const searchRoot = section || document;
        const promptSummary = approvalPrompt?.querySelector(
          '[data-testid="inputbar-approval-summary"]',
        );
        const promptRect = approvalPrompt?.getBoundingClientRect();
        const textarea = document.querySelector(
          'textarea[name="agent-chat-message"]',
        );
        const textareaRect = textarea?.getBoundingClientRect();
        const textareaStyle = textarea
          ? window.getComputedStyle(textarea)
          : null;
        const textareaVisible = Boolean(
          textarea &&
          textareaRect &&
          textareaRect.width > 16 &&
          textareaRect.height > 16 &&
          textareaStyle?.visibility !== "hidden" &&
          textareaStyle?.display !== "none",
        );
        const buttons = Array.from(searchRoot.querySelectorAll("button")).map(
          (button) => {
            const label = [
              button.textContent || "",
              button.getAttribute("aria-label") || "",
              button.getAttribute("title") || "",
            ].join("\n");
            const decision = button.getAttribute("data-decision") || "";
            return {
              text: button.textContent || "",
              aria: button.getAttribute("aria-label") || "",
              title: button.getAttribute("title") || "",
              decision,
              disabled: button.disabled,
              approve:
                decision === "allow_for_session" ||
                isDecisionLabel("allow_for_session", label),
              decline:
                decision === "decline" || isDecisionLabel("decline", label),
              cancel: decision === "cancel" || isDecisionLabel("cancel", label),
            };
          },
        );
        const approveButton = buttons.find((button) => button.approve);
        const promptSummaryText = promptSummary?.textContent || "";
        return {
          hasSection: Boolean(approvalPrompt),
          hasApprovalContent: promptSummaryText.includes(promptText),
          hasPrompt: promptSummaryText.includes(promptText),
          hasRequestId: Boolean(requestSection),
          hasToolName: sectionText.includes(toolName),
          hasCommand: sectionText.includes(commandText),
          hasDetails: Boolean(approvalPrompt?.querySelector("details")),
          hasPreformattedArguments: Boolean(
            approvalPrompt?.querySelector("pre"),
          ),
          textareaVisible,
          textareaDisabled: textarea?.disabled ?? null,
          promptHeight: promptRect?.height ?? null,
          singleLine:
            Boolean(promptRect) &&
            (promptRect?.height ?? Number.POSITIVE_INFINITY) <= 48,
          approveButtonVisible: Boolean(approveButton),
          approveButtonDisabled: approveButton?.disabled ?? null,
          declineButtonVisible: buttons.some((button) => button.decline),
          cancelButtonVisible: buttons.some((button) => button.cancel),
          stopButtonVisible: buttons.some((button) => {
            const label = `${button.text}\n${button.aria}`;
            return (
              !button.disabled &&
              (label.includes("停止") ||
                label.includes("终止") ||
                /\bStop\b/i.test(label))
            );
          }),
          sectionText,
          bodyText,
          buttons: buttons.map(
            ({
              text,
              aria,
              title,
              decision,
              disabled,
              approve,
              decline,
              cancel,
            }) => ({
              text,
              aria,
              title,
              decision,
              disabled,
              approve,
              decline,
              cancel,
            }),
          ),
        };
      },
      {
        requestId: APPROVAL_REQUEST_RESUME_REQUEST_ID,
        promptText: APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT,
        commandText: APPROVAL_REQUEST_RESUME_COMMAND,
        toolName: APPROVAL_REQUEST_RESUME_TOOL_NAME,
      },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.hasSection &&
      snapshot.hasApprovalContent &&
      snapshot.hasPrompt &&
      snapshot.hasToolName === false &&
      snapshot.hasCommand === false &&
      snapshot.hasDetails === false &&
      snapshot.hasPreformattedArguments === false &&
      snapshot.textareaVisible === true &&
      snapshot.textareaDisabled === false &&
      snapshot.singleLine === true &&
      snapshot.approveButtonVisible &&
      snapshot.approveButtonDisabled === false
    ) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `审批 pending UI 未出现: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

export async function clickApprovalDecisionButton(
  page,
  options,
  decision = "allow_for_session",
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 30_000)) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ requestId, decision }) => {
        const isDecisionLabel = (decision, label) => {
          const normalized = String(label || "").trim();
          const labelTokens = normalized
            .split(/\s+/u)
            .map((token) => token.trim())
            .filter(Boolean);
          if (decision === "allow_for_session") {
            return (
              normalized.includes("本会话允许") ||
              normalized.includes("本會話允許") ||
              normalized.includes("このセッションで許可") ||
              normalized.includes("세션") ||
              normalized.includes(
                "agentChat.inputbar.approval.action.allowForSession",
              ) ||
              /\bAllow for session\b/i.test(normalized) ||
              labelTokens.some((token) => /^session$/i.test(token))
            );
          }
          if (decision === "decline") {
            return (
              normalized.includes("拒绝") ||
              normalized.includes("拒絕") ||
              normalized.includes("却下") ||
              normalized.includes("거부") ||
              normalized.includes(
                "agentChat.inputbar.approval.action.decline",
              ) ||
              /\bDecline\b/i.test(normalized) ||
              /\bDeny\b/i.test(normalized)
            );
          }
          if (decision === "cancel") {
            return (
              normalized.includes("取消") ||
              normalized.includes("中止") ||
              normalized.includes("キャンセル") ||
              normalized.includes("취소") ||
              normalized.includes(
                "agentChat.inputbar.approval.action.cancel",
              ) ||
              /\bCancel\b/i.test(normalized) ||
              /\bAbort\b/i.test(normalized)
            );
          }
          return false;
        };
        const requestSection =
          Array.from(document.querySelectorAll("[data-request-id]")).find(
            (element) => element.getAttribute("data-request-id") === requestId,
          ) ?? null;
        const section =
          requestSection ??
          document
            .querySelector(
              '[data-testid="decision-panel-tool-confirmation-summary"]',
            )
            ?.closest("[data-request-id]") ??
          document.querySelector('[data-harness-section="approvals"]') ??
          document;
        const buttons = Array.from(section.querySelectorAll("button"));
        const button = buttons.find((candidate) => {
          const label = [
            candidate.textContent || "",
            candidate.getAttribute("aria-label") || "",
            candidate.getAttribute("title") || "",
          ].join("\n");
          return (
            !candidate.disabled &&
            (candidate.getAttribute("data-decision") === decision ||
              isDecisionLabel(decision, label))
          );
        });
        if (!button) {
          return {
            clicked: false,
            decision,
            buttonCount: buttons.length,
            sectionText: section.textContent || "",
          };
        }
        button.click();
        return {
          clicked: true,
          text: button.textContent || "",
          aria: button.getAttribute("aria-label") || "",
          decision: button.getAttribute("data-decision") || decision,
        };
      },
      { requestId: APPROVAL_REQUEST_RESUME_REQUEST_ID, decision },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (snapshot.clicked === true) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `未能点击审批 ${decision} 按钮: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

export function clickApprovalApproveButton(page, options) {
  return clickApprovalDecisionButton(page, options, "allow_for_session");
}

export async function waitForGuiApprovalPromptAbsent(
  page,
  options,
  {
    requiredText = "",
    runtimePromptText = APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT,
  } = {},
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 30_000)) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ requiredText, runtimePromptText }) => {
        const bodyText = document.body?.innerText || "";
        const approvalPrompt = document.querySelector(
          '[data-testid="inputbar-approval-prompt"]',
        );
        const approvalRecordCount = document.querySelectorAll(
          '[data-testid="timeline-approval-record"]',
        ).length;
        const textarea = document.querySelector(
          'textarea[name="agent-chat-message"]',
        );
        const textareaRect = textarea?.getBoundingClientRect();
        const textareaStyle = textarea
          ? window.getComputedStyle(textarea)
          : null;
        const textareaVisible = Boolean(
          textarea &&
          textareaRect &&
          textareaRect.width > 16 &&
          textareaRect.height > 16 &&
          textareaStyle?.visibility !== "hidden" &&
          textareaStyle?.display !== "none",
        );
        const buttons = Array.from(document.querySelectorAll("button"));
        const stopButtonVisible = buttons.some((button) => {
          const label = [
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
        return {
          approvalPromptVisible: Boolean(approvalPrompt),
          approvalRecordCount,
          includesRuntimePermissionPrompt:
            bodyText.includes("需要确认浏览器控制权限"),
          includesRuntimeApprovalPrompt: runtimePromptText
            ? bodyText.includes(runtimePromptText)
            : false,
          hasRequiredText: requiredText
            ? bodyText.includes(requiredText)
            : true,
          textareaVisible,
          textareaDisabled:
            textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
          stopButtonVisible,
        };
      },
      { requiredText, runtimePromptText },
    );
    lastSnapshot = snapshot;
    if (
      snapshot?.hasRequiredText === true &&
      snapshot?.approvalPromptVisible === false &&
      snapshot?.includesRuntimePermissionPrompt === false &&
      snapshot?.includesRuntimeApprovalPrompt === false &&
      snapshot?.textareaVisible === true &&
      snapshot?.textareaDisabled === false &&
      snapshot?.stopButtonVisible === false
    ) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `不应出现审批输入区: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

export async function waitForGuiApprovalPromptAbsentAfterSecondTurn(
  page,
  options,
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 30_000)) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ promptText, secondPromptMarker, secondDoneText }) => {
        const bodyText = document.body?.innerText || "";
        const secondTurnGroup = [...document.querySelectorAll(
          '[data-testid="message-turn-group"]',
        )]
          .reverse()
          .find((group) =>
            (group.innerText || "").includes(secondPromptMarker),
          );
        const approvalPrompt = document.querySelector(
          '[data-testid="inputbar-approval-prompt"]',
        );
        const textarea = document.querySelector(
          'textarea[name="agent-chat-message"]',
        );
        const buttons = Array.from(document.querySelectorAll("button"));
        const stopButtonVisible = buttons.some((button) => {
          const label = [
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
        return {
          approvalPromptVisible: Boolean(approvalPrompt),
          includesFirstApprovalPrompt: bodyText.includes(promptText),
          includesRuntimePermissionPrompt:
            bodyText.includes("需要确认浏览器控制权限"),
          hasSecondPrompt: bodyText.includes(secondPromptMarker),
          hasSecondDoneText: bodyText.includes(secondDoneText),
          hasPendingApprovalStatus: (secondTurnGroup?.innerText || "").includes(
            "待确认",
          ),
          textareaVisible: Boolean(textarea),
          textareaDisabled: textarea?.disabled ?? null,
          stopButtonVisible,
        };
      },
      {
        promptText: APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT,
        secondPromptMarker: APPROVAL_REQUEST_RESUME_SECOND_PROMPT_MARKER,
        secondDoneText: APPROVAL_REQUEST_RESUME_SECOND_DONE_TEXT,
      },
    );
    lastSnapshot = snapshot;
    if (
      snapshot?.hasSecondPrompt === true &&
      snapshot?.hasSecondDoneText === true &&
      snapshot?.approvalPromptVisible === false &&
      snapshot?.includesRuntimePermissionPrompt === false &&
      snapshot?.hasPendingApprovalStatus === false &&
      snapshot?.textareaVisible === true &&
      snapshot?.textareaDisabled === false &&
      snapshot?.stopButtonVisible === false
    ) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `第二轮不应出现审批输入区: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}
