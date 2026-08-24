import {
  MCP_STRUCTURED_CONTENT_ANSWER,
  MCP_STRUCTURED_CONTENT_DONE_TEXT,
  MCP_STRUCTURED_CONTENT_PROMPT,
  MCP_STRUCTURED_CONTENT_REFERENCE_ID,
  MCP_STRUCTURED_CONTENT_TOOL_DISPLAY_LABEL,
  MCP_STRUCTURED_CONTENT_TOOL_NAME,
} from "./claw-chat-current-fixture-constants.mjs";
import { evaluatePageSnapshot } from "./claw-chat-current-fixture-rpc.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

export async function waitForGuiMcpStructuredContentCompleted(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ prompt, doneText, answer, referenceId, toolName }) => {
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
        const forbiddenEnvelopeFragments = [
          "request_metadata",
          "mcp_tool_result_projection",
          "diagnostics",
          "raw_transport_payload",
          "doc-hidden-envelope",
          "control-plane envelope only",
          '"request_metadata"',
          '"diagnostics"',
        ];
        return {
          url: window.location.href,
          hasPrompt: text.includes(prompt),
          hasAssistantSummary: text.includes(
            "MCP structuredContent 展示验证完成",
          ),
          hasDoneText: text.includes(doneText),
          hasStructuredAnswer: text.includes(answer),
          hasReferenceId: text.includes(referenceId),
          hasToolName: text.includes(toolName),
          terminalDetailsCompacted:
            !text.includes(answer) &&
            !text.includes(toolName) &&
            /\d+\s*步/.test(text) &&
            text.includes("工具步骤"),
          historicalTimelineCompacted: Boolean(
            document.querySelector(
              '[data-testid^="message-list-historical-timeline-preview:"]',
            ),
          ),
          envelopeVisible: forbiddenEnvelopeFragments.some((value) =>
            text.includes(value),
          ),
          forbiddenEnvelopeHits: forbiddenEnvelopeFragments.filter((value) =>
            text.includes(value),
          ),
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
          bodyText: text,
        };
      },
      {
        prompt: MCP_STRUCTURED_CONTENT_PROMPT,
        doneText: MCP_STRUCTURED_CONTENT_DONE_TEXT,
        answer: MCP_STRUCTURED_CONTENT_ANSWER,
        referenceId: MCP_STRUCTURED_CONTENT_REFERENCE_ID,
        toolName: MCP_STRUCTURED_CONTENT_TOOL_NAME,
        toolDisplayLabel: MCP_STRUCTURED_CONTENT_TOOL_DISPLAY_LABEL,
      },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.hasPrompt &&
      (snapshot.hasAssistantSummary || snapshot.hasDoneText) &&
      snapshot.envelopeVisible === false &&
      snapshot.textareaVisible &&
      snapshot.textareaDisabled === false &&
      snapshot.stopButtonVisible === false
    ) {
      const expandedSnapshot =
        snapshot.hasStructuredAnswer && snapshot.hasToolName
          ? snapshot
          : snapshot.terminalDetailsCompacted
            ? snapshot
            : await expandAndInspectGuiMcpStructuredContentProcess(
                page,
                options,
              );
      if (
        (expandedSnapshot.hasStructuredAnswer ||
          expandedSnapshot.terminalDetailsCompacted) &&
        (expandedSnapshot.hasReferenceId ||
          expandedSnapshot.terminalDetailsCompacted) &&
        expandedSnapshot.envelopeVisible === false
      ) {
        return sanitizeJson({
          ...snapshot,
          hasStructuredAnswer:
            snapshot.hasStructuredAnswer ||
            expandedSnapshot.hasStructuredAnswer,
          hasReferenceId:
            snapshot.hasReferenceId || expandedSnapshot.hasReferenceId,
          hasToolName: snapshot.hasToolName || expandedSnapshot.hasToolName,
          expandedDetails: expandedSnapshot,
        });
      }
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Claw GUI 未完成 MCP structuredContent 展示验收: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

export async function expandAndInspectGuiMcpStructuredContentProcess(
  page,
  options,
) {
  await page.evaluate(() => {
    const historicalTimelinePreviews = Array.from(
      document.querySelectorAll(
        '[data-testid^="message-list-historical-timeline-preview:"]',
      ),
    );
    for (const preview of historicalTimelinePreviews) {
      if (preview instanceof HTMLButtonElement) {
        preview.click();
      }
    }

    const inlineToolSteps = Array.from(
      document.querySelectorAll('[data-testid="inline-tool-process-step"]'),
    );
    for (const step of inlineToolSteps) {
      const buttons = Array.from(step.querySelectorAll("button"));
      const detailsButton = buttons.find((button) => {
        const label = [
          button.getAttribute("title") || "",
          button.getAttribute("aria-label") || "",
          button.textContent || "",
        ].join("\n");
        return /展开(?:过程)?详情|查看结果|expand|details/i.test(label);
      });
      if (detailsButton instanceof HTMLButtonElement) {
        detailsButton.click();
      }
    }

    const canonicalToolRows = Array.from(
      document.querySelectorAll('[data-testid="tool-call-row"]'),
    );
    for (const row of canonicalToolRows) {
      const rowText = row.textContent || "";
      const toolName = row.getAttribute("data-tool-name") || "";
      if (
        !toolName.includes("mcp__docs__diagnostic_probe") &&
        !rowText.includes("docs / diagnostic probe")
      ) {
        continue;
      }
      const resultButton = Array.from(row.querySelectorAll("button")).find(
        (button) =>
          /结果|result|expand|展开/i.test(
            [
              button.getAttribute("title") || "",
              button.getAttribute("aria-label") || "",
              button.textContent || "",
            ].join("\n"),
          ),
      );
      if (resultButton instanceof HTMLButtonElement) {
        resultButton.click();
      }
    }

    const groups = Array.from(
      document.querySelectorAll('[data-testid="streaming-process-group"]'),
    );
    const targetGroup =
      groups.find((group) =>
        (group.textContent || "").includes("已完成 1 个步骤"),
      ) ?? groups[0];
    const button = targetGroup?.querySelector("button");
    if (
      button instanceof HTMLButtonElement &&
      button.getAttribute("aria-expanded") !== "true"
    ) {
      button.click();
    }
  });

  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 30000)) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ answer, referenceId, toolName, toolDisplayLabel }) => {
        const text = document.body?.innerText || "";
        const processGroups = Array.from(
          document.querySelectorAll('[data-testid="streaming-process-group"]'),
        ).map((group) => {
          const button = group.querySelector("button");
          return {
            text: group.textContent || "",
            buttonText: button?.textContent || "",
            expanded: button?.getAttribute("aria-expanded") || "",
          };
        });
        const inlineToolSteps = Array.from(
          document.querySelectorAll('[data-testid="inline-tool-process-step"]'),
        ).map((step) => ({
          text: step.textContent || "",
          grouped: step.getAttribute("data-grouped") || "",
        }));
        const canonicalToolRow = Array.from(
          document.querySelectorAll('[data-testid="tool-call-row"]'),
        ).find((row) => {
          const rowText = row.textContent || "";
          return (
            row.getAttribute("data-tool-name") === toolName ||
            rowText.includes(toolName) ||
            rowText.includes(toolDisplayLabel)
          );
        });
        const canonicalToolContainer = canonicalToolRow?.closest(
          '[data-testid^="agent-thread-block:"]',
        );
        const canonicalToolText =
          canonicalToolContainer?.textContent || canonicalToolRow?.textContent || "";
        const mcpProcessGroup = processGroups.find(
          (group) =>
            group.text.includes(answer) ||
            group.text.includes(toolName) ||
            group.buttonText.includes("已完成 1 个步骤") ||
            group.text.includes("已完成 1 个步骤"),
        );
        const mcpInlineStep = inlineToolSteps.find(
          (step) =>
            step.text.includes(answer) ||
            step.text.includes(referenceId) ||
            step.text.includes(toolName) ||
            step.text.includes(toolDisplayLabel),
        );
        const processText = mcpProcessGroup?.text || canonicalToolText;
        const inlineText = mcpInlineStep?.text || canonicalToolText;
        const visibleProcessText = inlineText || processText;
        const forbiddenEnvelopeFragments = [
          "request_metadata",
          "mcp_tool_result_projection",
          "diagnostics",
          "raw_transport_payload",
          "doc-hidden-envelope",
          "control-plane envelope only",
          '"request_metadata"',
          '"diagnostics"',
        ];
        return {
          mcpProcessGroupExpanded:
            Boolean(mcpInlineStep || canonicalToolRow) ||
            mcpProcessGroup?.expanded === "true",
          hasStructuredAnswer: visibleProcessText.includes(answer),
          hasReferenceId: visibleProcessText.includes(referenceId),
          hasToolName:
            visibleProcessText.includes(toolName) ||
            text.includes(toolName) ||
            visibleProcessText.includes(toolDisplayLabel) ||
            text.includes(toolDisplayLabel),
          envelopeVisible: forbiddenEnvelopeFragments.some((value) =>
            visibleProcessText.includes(value),
          ),
          forbiddenEnvelopeHits: forbiddenEnvelopeFragments.filter((value) =>
            visibleProcessText.includes(value),
          ),
          processGroupCount: processGroups.length,
          processGroupText: processText,
          inlineToolStepCount: inlineToolSteps.length,
          inlineToolStepText: inlineText,
          canonicalToolRowCount: canonicalToolRow ? 1 : 0,
          canonicalToolText,
          bodyText: text,
        };
      },
      {
        answer: MCP_STRUCTURED_CONTENT_ANSWER,
        referenceId: MCP_STRUCTURED_CONTENT_REFERENCE_ID,
        toolName: MCP_STRUCTURED_CONTENT_TOOL_NAME,
        toolDisplayLabel: MCP_STRUCTURED_CONTENT_TOOL_DISPLAY_LABEL,
      },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.mcpProcessGroupExpanded &&
      snapshot.hasStructuredAnswer &&
      snapshot.envelopeVisible === false
    ) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Claw GUI 未展开 MCP structuredContent 工具过程: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}
