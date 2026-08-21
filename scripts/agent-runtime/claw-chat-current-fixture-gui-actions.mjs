export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

export function sanitizeJson(value, depth = 0) {
  if (depth > 6) {
    return "[truncated]";
  }
  if (value === null || typeof value !== "object") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.slice(0, 50).map((item) => sanitizeJson(item, depth + 1));
  }
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => {
      if (/^(system_prompt|systemPrompt)$/u.test(key)) {
        return [key, item ? "[redacted-prompt]" : null];
      }
      return [
        key,
        typeof item === "string" && item.length > 4000
          ? `${item.slice(0, 4000)}... [truncated ${item.length - 4000} chars]`
          : sanitizeJson(item, depth + 1),
      ];
    }),
  );
}

function isTransientPageEvaluationError(error) {
  const message = error instanceof Error ? error.message : String(error ?? "");
  return (
    message.includes("Execution context was destroyed") ||
    message.includes("most likely because of a navigation") ||
    message.includes("Cannot find context with specified id")
  );
}

export async function evaluatePageSnapshot(page, pageFunction, arg) {
  try {
    return await page.evaluate(pageFunction, arg);
  } catch (error) {
    if (isTransientPageEvaluationError(error)) {
      return null;
    }
    throw error;
  }
}

function isTaskCenterHomeText(text) {
  return (
    text.includes("青柠一下，灵感即来") ||
    text.includes("你可以从这些任务开始") ||
    text.includes("向下滑，看看 Lime 可以帮你做什么")
  );
}

function escapeCssAttributeValue(value) {
  return String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function buildInputSelector(expectedSessionId) {
  const baseSelector = 'textarea[name="agent-chat-message"]';
  if (!expectedSessionId) {
    return baseSelector;
  }
  return `${baseSelector}[data-session-id="${escapeCssAttributeValue(
    expectedSessionId,
  )}"]`;
}

function describeExpectedSession(expectedSessionId) {
  return expectedSessionId ? ` session=${expectedSessionId}` : "";
}

async function setControlledTextareaValue(page, prompt, expectedSessionId) {
  return await page.evaluate(
    ({ expectedPrompt, sessionId }) => {
      const input = Array.from(
        document.querySelectorAll('textarea[name="agent-chat-message"]'),
      ).find(
        (node) =>
          node instanceof HTMLTextAreaElement &&
          (!sessionId || node.dataset.sessionId === sessionId),
      );
      if (!(input instanceof HTMLTextAreaElement)) {
        return {
          ok: false,
          reason: "missing-textarea",
          expectedSessionId: sessionId,
          textareaSessionId: null,
          value: null,
        };
      }

      input.focus();
      const valueSetter = Object.getOwnPropertyDescriptor(
        HTMLTextAreaElement.prototype,
        "value",
      )?.set;
      if (valueSetter) {
        valueSetter.call(input, expectedPrompt);
      } else {
        input.value = expectedPrompt;
      }
      input.setSelectionRange(expectedPrompt.length, expectedPrompt.length);
      input.dispatchEvent(
        new InputEvent("input", {
          bubbles: true,
          inputType: "insertText",
          data: expectedPrompt,
        }),
      );
      input.dispatchEvent(new Event("change", { bubbles: true }));

      return {
        ok: input.value === expectedPrompt,
        reason: input.value === expectedPrompt ? null : "value-mismatch",
        expectedSessionId: sessionId,
        textareaSessionId: input.dataset.sessionId || null,
        value: input.value,
      };
    },
    { expectedPrompt: prompt, sessionId: expectedSessionId },
  );
}

export async function waitForInputReady(page, options, constraints = {}) {
  const expectedSessionId = constraints.expectedSessionId ?? null;
  const allowTaskCenterHomeInput =
    constraints.allowTaskCenterHomeInput === true;
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      (sessionId) => {
        const candidates = Array.from(
          document.querySelectorAll('textarea[name="agent-chat-message"]'),
        ).filter(
          (node) =>
            node instanceof HTMLTextAreaElement &&
            (!sessionId || node.dataset.sessionId === sessionId),
        );
        const textarea =
          candidates.find((node) => {
            const rect = node.getBoundingClientRect();
            const style = window.getComputedStyle(node);
            return Boolean(
              rect &&
              rect.width > 16 &&
              rect.height > 16 &&
              style.visibility !== "hidden" &&
              style.display !== "none",
            );
          }) ??
          candidates[0] ??
          null;
        const rect = textarea?.getBoundingClientRect();
        const style = textarea ? window.getComputedStyle(textarea) : null;
        const visible = Boolean(
          textarea &&
          rect &&
          rect.width > 16 &&
          rect.height > 16 &&
          style?.visibility !== "hidden" &&
          style?.display !== "none",
        );
        const container = textarea?.closest(
          '[data-testid="inputbar-core-container"]',
        );
        const mainArea = document.querySelector(
          '[data-testid="workspace-main-area"]',
        );
        const floatingOverlay = document.querySelector(
          '[data-testid="general-workbench-input-overlay"]',
        );
        return {
          url: window.location.href,
          expectedSessionId: sessionId,
          hasTextarea: Boolean(textarea),
          textareaCount: candidates.length,
          textareaSessionId:
            textarea instanceof HTMLTextAreaElement
              ? textarea.dataset.sessionId || null
              : null,
          textareaVisible: visible,
          textareaDisabled:
            textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
          textareaValue:
            textarea instanceof HTMLTextAreaElement ? textarea.value : null,
          hasInputbarCore: Boolean(container),
          sendButtonVisible: Boolean(
            container?.querySelector('[data-testid="send-btn"]'),
          ),
          workspaceMainAreaPresent: Boolean(mainArea),
          workspaceLayoutMode:
            mainArea instanceof HTMLElement
              ? mainArea.dataset.layoutMode || null
              : null,
          workspaceHasRightSurface:
            mainArea instanceof HTMLElement
              ? mainArea.dataset.hasRightSurface || null
              : null,
          workspaceFloatingInputOverlay:
            mainArea instanceof HTMLElement
              ? mainArea.dataset.floatingInputOverlay || null
              : null,
          floatingOverlayPresent: Boolean(floatingOverlay),
          bodyText: document.body?.innerText || "",
          mainText: document.querySelector("main")?.textContent || "",
        };
      },
      expectedSessionId,
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.hasTextarea &&
      snapshot.hasInputbarCore &&
      snapshot.textareaVisible &&
      snapshot.textareaDisabled === false &&
      (!expectedSessionId ||
        snapshot.textareaSessionId === expectedSessionId) &&
      !snapshot.mainText.includes("最近对话") &&
      !snapshot.mainText.includes("正在恢复生成会话") &&
      (allowTaskCenterHomeInput ||
        (!isTaskCenterHomeText(snapshot.mainText || "") &&
          !isTaskCenterHomeText(snapshot.bodyText || "")))
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Claw 输入框未就绪${describeExpectedSession(
      expectedSessionId,
    )}: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

async function waitForControlledInputValue(
  page,
  prompt,
  timeoutMs,
  expectedSessionId = null,
) {
  await page.waitForFunction(
    ({ expectedPrompt, sessionId }) => {
      const input = Array.from(
        document.querySelectorAll('textarea[name="agent-chat-message"]'),
      ).find(
        (node) =>
          node instanceof HTMLTextAreaElement &&
          (!sessionId || node.dataset.sessionId === sessionId),
      );
      return (
        input instanceof HTMLTextAreaElement && input.value === expectedPrompt
      );
    },
    { expectedPrompt: prompt, sessionId: expectedSessionId },
    { timeout: Math.min(timeoutMs, 10_000) },
  );
  await page.evaluate(
    () =>
      new Promise((resolve) => {
        window.requestAnimationFrame(() => {
          window.requestAnimationFrame(resolve);
        });
      }),
  );
}

async function waitForSendButtonReady(page, prompt, options, constraints = {}) {
  const expectedSessionId = constraints.expectedSessionId ?? null;
  const selector =
    '[data-testid="inputbar-core-container"] [data-testid="send-btn"]';
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 10_000)) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ expectedPrompt, sendSelector, sessionId }) => {
        const input = Array.from(
          document.querySelectorAll('textarea[name="agent-chat-message"]'),
        ).find(
          (node) =>
            node instanceof HTMLTextAreaElement &&
            (!sessionId || node.dataset.sessionId === sessionId),
        );
        const container = input?.closest(
          '[data-testid="inputbar-core-container"]',
        );
        const sendButton = sessionId
          ? container?.querySelector('[data-testid="send-btn"]')
          : document.querySelector(sendSelector);
        const inputValue =
          input instanceof HTMLTextAreaElement ? input.value : null;
        return {
          expectedSessionId: sessionId,
          textareaSessionId:
            input instanceof HTMLTextAreaElement
              ? input.dataset.sessionId || null
              : null,
          inputValue,
          promptVisibleInTextarea: inputValue === expectedPrompt,
          sendButtonExists: Boolean(sendButton),
          sendButtonDisabled:
            sendButton instanceof HTMLButtonElement
              ? sendButton.disabled
              : null,
          sendButtonLabel:
            sendButton instanceof HTMLElement
              ? sendButton.getAttribute("aria-label") ||
                sendButton.getAttribute("title") ||
                sendButton.textContent ||
                "send"
              : null,
        };
      },
      {
        expectedPrompt: prompt,
        sendSelector: selector,
        sessionId: expectedSessionId,
      },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.promptVisibleInTextarea &&
      snapshot.sendButtonExists &&
      snapshot.sendButtonDisabled === false &&
      (!expectedSessionId || snapshot.textareaSessionId === expectedSessionId)
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `输入栏发送按钮未进入可发送状态${describeExpectedSession(
      expectedSessionId,
    )}: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

async function clickInputbarSendButton(page, options, constraints = {}) {
  const expectedSessionId = constraints.expectedSessionId ?? null;
  const selector =
    '[data-testid="inputbar-core-container"] [data-testid="send-btn"]';
  const sendLocator = expectedSessionId
    ? page
        .locator(buildInputSelector(expectedSessionId))
        .locator(
          'xpath=ancestor::*[@data-testid="inputbar-core-container"][1]//*[@data-testid="send-btn"]',
        )
    : page.locator(selector).first();
  await sendLocator.waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 10_000),
  });
  const beforeClick = await page.evaluate(
    ({ sendSelector, sessionId }) => {
      const input = Array.from(
        document.querySelectorAll('textarea[name="agent-chat-message"]'),
      ).find(
        (node) =>
          node instanceof HTMLTextAreaElement &&
          (!sessionId || node.dataset.sessionId === sessionId),
      );
      const container = input?.closest(
        '[data-testid="inputbar-core-container"]',
      );
      const sendButton = sessionId
        ? container?.querySelector('[data-testid="send-btn"]')
        : document.querySelector(sendSelector);
      return {
        expectedSessionId: sessionId,
        textareaSessionId:
          input instanceof HTMLTextAreaElement
            ? input.dataset.sessionId || null
            : null,
        exists: Boolean(sendButton),
        disabled:
          sendButton instanceof HTMLButtonElement ? sendButton.disabled : null,
        label:
          sendButton instanceof HTMLElement
            ? sendButton.getAttribute("aria-label") ||
              sendButton.getAttribute("title") ||
              sendButton.textContent ||
              "send"
            : null,
      };
    },
    { sendSelector: selector, sessionId: expectedSessionId },
  );
  assert(
    beforeClick.exists &&
      beforeClick.disabled === false &&
      (!expectedSessionId ||
        beforeClick.textareaSessionId === expectedSessionId),
    `输入栏发送按钮不可用: ${JSON.stringify(sanitizeJson(beforeClick))}`,
  );
  const clickedAt = new Date().toISOString();
  await sendLocator.click();
  return {
    clicked: true,
    clickedAt,
    method: "inputbar-send-btn",
    disabledBeforeClick: beforeClick.disabled,
    label: beforeClick.label,
    sessionId: beforeClick.textareaSessionId,
  };
}

async function pressInputbarEnter(page, prompt, options, constraints = {}) {
  const expectedSessionId = constraints.expectedSessionId ?? null;
  const inputLocator = expectedSessionId
    ? page.locator(buildInputSelector(expectedSessionId))
    : page.locator(buildInputSelector(null)).first();
  await inputLocator.waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 10_000),
  });
  const beforePress = await page.evaluate(
    ({ expectedPrompt, sessionId }) => {
      const input = Array.from(
        document.querySelectorAll('textarea[name="agent-chat-message"]'),
      ).find(
        (node) =>
          node instanceof HTMLTextAreaElement &&
          (!sessionId || node.dataset.sessionId === sessionId),
      );
      return {
        expectedSessionId: sessionId,
        textareaSessionId:
          input instanceof HTMLTextAreaElement
            ? input.dataset.sessionId || null
            : null,
        value: input instanceof HTMLTextAreaElement ? input.value : null,
        disabled: input instanceof HTMLTextAreaElement ? input.disabled : null,
      };
    },
    { expectedPrompt: prompt, sessionId: expectedSessionId },
  );
  assert(
    beforePress.value === prompt &&
      beforePress.disabled === false &&
      (!expectedSessionId ||
        beforePress.textareaSessionId === expectedSessionId),
    `输入栏 Enter 提交不可用: ${JSON.stringify(sanitizeJson(beforePress))}`,
  );
  const clickedAt = new Date().toISOString();
  await inputLocator.press("Enter");
  return {
    clicked: true,
    clickedAt,
    method: "inputbar-textarea-enter",
    triggerSource: "enter",
    sessionId: beforePress.textareaSessionId,
  };
}

async function sampleInputbarSubmitState(
  page,
  prompt,
  expectedSessionId = null,
  expectedThreadId = null,
) {
  return await evaluatePageSnapshot(
    page,
    ({ expectedPrompt, sessionId, threadId }) => {
      const parseJsonArray = (raw) => {
        try {
          const parsed = JSON.parse(raw || "[]");
          return Array.isArray(parsed) ? parsed : [];
        } catch {
          return [];
        }
      };
      const decodeJsonRpcLines = (lines) => {
        if (!Array.isArray(lines)) {
          return [];
        }
        return lines
          .map((line) => {
            try {
              return JSON.parse(String(line));
            } catch {
              return null;
            }
          })
          .filter(Boolean);
      };
      const traceMessages = parseJsonArray(
        window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      );
      const appServerTrace = traceMessages
        .filter((entry) => entry?.command === "app_server_handle_json_lines")
        .map((entry) => {
          const messages = decodeJsonRpcLines(
            entry?.args_preview?.request?.lines,
          );
          return {
            timestamp: entry?.timestamp || null,
            transport: entry?.transport || null,
            status: entry?.status || null,
            durationMs: entry?.duration_ms ?? null,
            error: entry?.error || null,
            methods: messages.map((message) => message?.method).filter(Boolean),
            turnStarts: messages
              .filter((message) => message?.method === "turn/start")
              .map((message) => ({
                id: message?.id || null,
                sessionId: message?.params?.sessionId || null,
                threadId: message?.params?.threadId || null,
                clientUserMessageId:
                  message?.params?.clientUserMessageId || null,
                model: message?.params?.model || null,
                permissions: message?.params?.permissions || null,
                text: Array.isArray(message?.params?.input)
                  ? message.params.input
                      .filter(
                        (part) =>
                          part?.type === "text" &&
                          typeof part?.text === "string",
                      )
                      .map((part) => part.text)
                      .join("\n")
                  : null,
              })),
          };
        });
      const appServerTurnStartTrace = appServerTrace
        .flatMap((entry) =>
          entry.turnStarts.map((turnStart) => ({
            ...turnStart,
            timestamp: entry.timestamp,
            transport: entry.transport,
            status: entry.status,
            durationMs: entry.durationMs,
            error: entry.error,
          })),
        )
        .slice(-20);
      const matchingTurnStartTrace = appServerTurnStartTrace.find(
        (entry) =>
          (!threadId || entry.threadId === threadId) &&
          (!sessionId || !entry.sessionId || entry.sessionId === sessionId) &&
          (!expectedPrompt ||
            String(entry.text || "").includes(expectedPrompt)),
      );
      const inputs = Array.from(
        document.querySelectorAll('textarea[name="agent-chat-message"]'),
      ).filter(
        (node) =>
          node instanceof HTMLTextAreaElement &&
          (!sessionId || node.dataset.sessionId === sessionId),
      );
      const input = inputs[0] ?? null;
      const container = input?.closest(
        '[data-testid="inputbar-core-container"]',
      );
      const sendButton = container?.querySelector('[data-testid="send-btn"]');
      const bodyText = document.body?.innerText || "";
      const mainText = document.querySelector("main")?.textContent || "";
      const promptInBody = bodyText.includes(expectedPrompt);
      const promptInMain = mainText.includes(expectedPrompt);
      const pendingTexts = [
        "正在发送",
        "正在输出",
        "正在生成",
        "排队",
        "继续以「代码文学专家」身份",
      ].filter((text) => bodyText.includes(text));
      return {
        url: window.location.href,
        expectedSessionId: sessionId,
        textareaCount: inputs.length,
        textareaSessionId:
          input instanceof HTMLTextAreaElement
            ? input.dataset.sessionId || null
            : null,
        textareaValue:
          input instanceof HTMLTextAreaElement ? input.value : null,
        textareaDisabled:
          input instanceof HTMLTextAreaElement ? input.disabled : null,
        sendButtonDisabled:
          sendButton instanceof HTMLButtonElement ? sendButton.disabled : null,
        promptInBody,
        promptInMain,
        promptOccurrencesInBody: expectedPrompt
          ? bodyText.split(expectedPrompt).length - 1
          : 0,
        pendingTexts,
        hasInputbarCore: Boolean(container),
        hasWorkspaceInlineInputSlot: Boolean(
          document.querySelector('[data-testid="workspace-inline-input-slot"]'),
        ),
        hasEmptyStateComposer: Boolean(
          document.querySelector('[data-testid="empty-state-composer"]'),
        ),
        appServerTraceTail: appServerTrace.slice(-30),
        appServerTurnStartTrace,
        matchingTurnStartTrace: matchingTurnStartTrace || null,
      };
    },
    {
      expectedPrompt: prompt,
      sessionId: expectedSessionId,
      threadId: expectedThreadId,
    },
  );
}

async function waitForInputbarSubmitEffect(
  page,
  prompt,
  options,
  expectedSessionId = null,
  constraints = {},
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  let firstSubmitEffectSnapshot = null;
  const timeoutMs = constraints.requireTurnStart
    ? Math.min(options.timeoutMs, 120_000)
    : Math.min(options.timeoutMs, 10_000);
  while (Date.now() - startedAt < timeoutMs) {
    const snapshot = await sampleInputbarSubmitState(
      page,
      prompt,
      expectedSessionId,
      constraints.expectedThreadId ?? null,
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      !firstSubmitEffectSnapshot &&
      (snapshot.textareaValue !== prompt ||
        snapshot.promptOccurrencesInBody > 1 ||
        snapshot.pendingTexts.length > 0)
    ) {
      firstSubmitEffectSnapshot = snapshot;
    }
    if (snapshot.matchingTurnStartTrace) {
      return snapshot;
    }
    if (constraints.requireTurnStart !== true && firstSubmitEffectSnapshot) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  return lastSnapshot ?? firstSubmitEffectSnapshot;
}

export async function sendPromptFromGui(
  page,
  options,
  prompt,
  constraints = {},
) {
  const expectedSessionId = constraints.expectedSessionId ?? null;
  const before = await waitForInputReady(page, options, constraints);
  const appliedInput = await setControlledTextareaValue(
    page,
    prompt,
    expectedSessionId,
  );
  assert(
    appliedInput.ok &&
      (!expectedSessionId ||
        appliedInput.textareaSessionId === expectedSessionId),
    `输入框写入失败: ${JSON.stringify(sanitizeJson(appliedInput))}`,
  );
  await waitForControlledInputValue(
    page,
    prompt,
    options.timeoutMs,
    expectedSessionId,
  );
  const afterFill = await page.evaluate(
    ({ expectedPrompt, sessionId }) => {
      const input = Array.from(
        document.querySelectorAll('textarea[name="agent-chat-message"]'),
      ).find(
        (node) =>
          node instanceof HTMLTextAreaElement &&
          (!sessionId || node.dataset.sessionId === sessionId),
      );
      return {
        expectedSessionId: sessionId,
        textareaSessionId:
          input instanceof HTMLTextAreaElement
            ? input.dataset.sessionId || null
            : null,
        value: input instanceof HTMLTextAreaElement ? input.value : null,
        promptVisibleInTextarea:
          input instanceof HTMLTextAreaElement
            ? input.value === expectedPrompt
            : false,
      };
    },
    { expectedPrompt: prompt, sessionId: expectedSessionId },
  );
  assert(
    afterFill.promptVisibleInTextarea &&
      (!expectedSessionId || afterFill.textareaSessionId === expectedSessionId),
    `输入框未保留用户输入: ${JSON.stringify(sanitizeJson(afterFill))}`,
  );
  const afterFillStability =
    typeof constraints.collectAfterFillStability === "function"
      ? await constraints.collectAfterFillStability({ afterFill })
      : null;

  const submitWithEnter = constraints.submitWithEnter === true;
  const sendReady = submitWithEnter
    ? {
        promptVisibleInTextarea: afterFill.promptVisibleInTextarea,
        submitMethod: "inputbar-textarea-enter",
        textareaSessionId: afterFill.textareaSessionId,
      }
    : await waitForSendButtonReady(page, prompt, options, constraints);
  if (typeof constraints.beforeClick === "function") {
    await constraints.beforeClick({ afterFill, afterFillStability, sendReady });
  }
  const clicked = submitWithEnter
    ? await pressInputbarEnter(page, prompt, options, constraints)
    : await clickInputbarSendButton(page, options, constraints);
  if (typeof constraints.afterClick === "function") {
    await constraints.afterClick({
      afterFill,
      afterFillStability,
      clicked,
      sendReady,
    });
  }
  const afterClick = await waitForInputbarSubmitEffect(
    page,
    prompt,
    options,
    expectedSessionId,
    constraints,
  );
  return {
    before,
    afterFill,
    afterFillStability,
    sendReady,
    clicked,
    afterClick,
  };
}

export async function setInputbarAccessMode(
  page,
  options,
  accessMode,
  constraints = {},
) {
  const expectedSessionId = constraints.expectedSessionId ?? null;
  const selectLocator = expectedSessionId
    ? page
        .locator(buildInputSelector(expectedSessionId))
        .locator(
          'xpath=ancestor::*[@data-testid="inputbar-core-container"][1]//*[@data-testid="inputbar-access-mode-select"]',
        )
    : page.locator('[data-testid="inputbar-access-mode-select"]').first();

  await waitForInputReady(page, options, constraints);
  await selectLocator.waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 10_000),
  });

  const before = await evaluatePageSnapshot(
    page,
    ({ sessionId }) => {
      const input = Array.from(
        document.querySelectorAll('textarea[name="agent-chat-message"]'),
      ).find(
        (node) =>
          node instanceof HTMLTextAreaElement &&
          (!sessionId || node.dataset.sessionId === sessionId),
      );
      const container = input?.closest(
        '[data-testid="inputbar-core-container"]',
      );
      const select =
        container?.querySelector(
          '[data-testid="inputbar-access-mode-select"]',
        ) ??
        document.querySelector('[data-testid="inputbar-access-mode-select"]');
      return {
        expectedSessionId: sessionId,
        textareaSessionId:
          input instanceof HTMLTextAreaElement
            ? input.dataset.sessionId || null
            : null,
        selectExists: Boolean(select),
        value: select instanceof HTMLSelectElement ? select.value : null,
        options:
          select instanceof HTMLSelectElement
            ? Array.from(select.options).map((option) => option.value)
            : [],
      };
    },
    { sessionId: expectedSessionId },
  );

  await selectLocator.selectOption(accessMode);
  await page.waitForFunction(
    ({ expectedAccessMode, sessionId }) => {
      const input = Array.from(
        document.querySelectorAll('textarea[name="agent-chat-message"]'),
      ).find(
        (node) =>
          node instanceof HTMLTextAreaElement &&
          (!sessionId || node.dataset.sessionId === sessionId),
      );
      const container = input?.closest(
        '[data-testid="inputbar-core-container"]',
      );
      const select =
        container?.querySelector(
          '[data-testid="inputbar-access-mode-select"]',
        ) ??
        document.querySelector('[data-testid="inputbar-access-mode-select"]');
      return (
        select instanceof HTMLSelectElement &&
        select.value === expectedAccessMode
      );
    },
    { expectedAccessMode: accessMode, sessionId: expectedSessionId },
    { timeout: Math.min(options.timeoutMs, 10_000) },
  );

  const after = await evaluatePageSnapshot(
    page,
    ({ expectedAccessMode, sessionId }) => {
      const input = Array.from(
        document.querySelectorAll('textarea[name="agent-chat-message"]'),
      ).find(
        (node) =>
          node instanceof HTMLTextAreaElement &&
          (!sessionId || node.dataset.sessionId === sessionId),
      );
      const container = input?.closest(
        '[data-testid="inputbar-core-container"]',
      );
      const select =
        container?.querySelector(
          '[data-testid="inputbar-access-mode-select"]',
        ) ??
        document.querySelector('[data-testid="inputbar-access-mode-select"]');
      return {
        expectedSessionId: sessionId,
        expectedAccessMode,
        textareaSessionId:
          input instanceof HTMLTextAreaElement
            ? input.dataset.sessionId || null
            : null,
        selectExists: Boolean(select),
        value: select instanceof HTMLSelectElement ? select.value : null,
        matched:
          select instanceof HTMLSelectElement
            ? select.value === expectedAccessMode
            : false,
      };
    },
    { expectedAccessMode: accessMode, sessionId: expectedSessionId },
  );

  assert(
    after?.matched === true,
    `输入栏权限模式未切换: ${JSON.stringify(
      sanitizeJson({ before, after, accessMode }),
    )}`,
  );

  return sanitizeJson({
    before,
    after,
  });
}
