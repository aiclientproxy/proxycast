import {
  assert,
  evaluatePageSnapshot,
  sanitizeJson,
  sleep,
} from "./claw-chat-current-fixture-gui-actions.mjs";
import { reloadRendererDocument } from "./claw-chat-current-fixture-rpc.mjs";
import {
  EXPERT_SKILLS_RUNTIME_BASE_SKILL_REF,
  EXPERT_SKILLS_RUNTIME_ID,
  EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
  EXPERT_SKILLS_RUNTIME_PROMPT,
  EXPERT_SKILLS_RUNTIME_SKILL_REF,
  EXPERT_SKILLS_RUNTIME_TITLE,
} from "./skills-runtime-fixture-scenario.mjs";

export const EXPERT_PANEL_SKILLS_RUNTIME_UI_SKILL_REF =
  EXPERT_SKILLS_RUNTIME_SKILL_REF;
export const EXPERT_PANEL_SKILLS_RUNTIME_UI_SKILL_TEST_ID =
  "skill-capability-report";
export const EXPERT_PANEL_SKILLS_RUNTIME_UI_CANDIDATE_TEST_ID = `expert-skill-candidate-${EXPERT_PANEL_SKILLS_RUNTIME_UI_SKILL_TEST_ID}`;
export const EXPERT_PANEL_SKILLS_RUNTIME_UI_ADD_TEST_ID = `expert-skill-add-${EXPERT_PANEL_SKILLS_RUNTIME_UI_SKILL_TEST_ID}`;
export const EXPERT_PANEL_SKILLS_RUNTIME_UI_CHIP_TEST_ID = `expert-info-skill-chip-${EXPERT_PANEL_SKILLS_RUNTIME_UI_SKILL_TEST_ID}`;
export const EXPERT_PANEL_SKILLS_RUNTIME_UI_READINESS_TEST_ID = `expert-info-skill-readiness-${EXPERT_PANEL_SKILLS_RUNTIME_UI_SKILL_TEST_ID}`;

export async function reloadRendererAfterExpertPanelSkillCatalogInjection(
  page,
  options,
  waitForRendererReady,
  clearInvokeBuffers,
) {
  const reload = await reloadRendererDocument(page, options);
  const renderer = await waitForRendererReady(page, options);
  const catalogState = await evaluatePageSnapshot(page, () => {
    const raw = window.localStorage.getItem("lime:skill-catalog:v1") || "";
    return {
      skillCatalogRawPresent: raw.length > 0,
      skillCatalogRawIncludesCapabilityReport:
        raw.includes("capability-report"),
    };
  });
  await clearInvokeBuffers(page);
  return sanitizeJson({
    reload,
    renderer,
    catalogState,
  });
}

export async function launchExpertSkillsRuntimeFromExpertPlaza(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;

  await page.locator('[data-testid="app-sidebar-nav-plugins"]').click();
  await page.locator('[data-testid="plugin-workspace-tab-experts"]').click();

  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ expertId, expertTitle, skillRef, prompt }) => {
        const text = document.body?.innerText || "";
        const plaza = document.querySelector(
          '[data-testid="expert-plaza-page"]',
        );
        const card = document.querySelector(
          `[data-testid="expert-card-${expertId}"]`,
        );
        const startButton = document.querySelector(
          `[data-testid="expert-start-${expertId}"]`,
        );
        return {
          url: window.location.href,
          plazaVisible: Boolean(plaza),
          cardVisible: Boolean(card),
          startButtonVisible: Boolean(startButton),
          startButtonDisabled:
            startButton instanceof HTMLButtonElement
              ? startButton.disabled
              : null,
          titleVisible: text.includes(expertTitle),
          skillRefVisible: text.includes(skillRef),
          promptStarterVisible: text.includes(prompt),
          bodyText: text,
        };
      },
      {
        expertId: EXPERT_SKILLS_RUNTIME_ID,
        expertTitle: EXPERT_SKILLS_RUNTIME_TITLE,
        skillRef: EXPERT_SKILLS_RUNTIME_SKILL_REF,
        prompt: EXPERT_SKILLS_RUNTIME_PROMPT,
      },
    );
    lastSnapshot = snapshot;
    if (
      snapshot?.plazaVisible &&
      snapshot.cardVisible &&
      snapshot.startButtonVisible &&
      snapshot.startButtonDisabled === false &&
      snapshot.titleVisible
    ) {
      await page
        .locator(`[data-testid="expert-start-${EXPERT_SKILLS_RUNTIME_ID}"]`)
        .click();
      return sanitizeJson({
        ...snapshot,
        clicked: true,
      });
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `Expert Plaza 未出现可启动的专家卡片: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

export async function addExpertSkillsRuntimeSkillFromInfoPanel(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;

  while (Date.now() - startedAt < options.timeoutMs) {
    const ready = await evaluatePageSnapshot(
      page,
      ({
        expertTitle,
        baseSkillRef,
        addedSkillRef,
        addedSkillUiRef,
        addedSkillChipTestId,
        addedSkillReadinessTestId,
      }) => {
        const text = document.body?.innerText || "";
        const visibleElementSnapshot = (element) => {
          const rect = element?.getBoundingClientRect();
          const style = element ? window.getComputedStyle(element) : null;
          return {
            visible: Boolean(
              element &&
              rect &&
              rect.width > 8 &&
              rect.height > 8 &&
              style?.visibility !== "hidden" &&
              style?.display !== "none",
            ),
            label:
              element?.getAttribute?.("aria-label") ||
              element?.getAttribute?.("title") ||
              element?.textContent ||
              "",
          };
        };
        const panel = document.querySelector(
          '[data-testid="expert-info-panel"]',
        );
        const panelText = panel?.textContent || "";
        const addButton = document.querySelector(
          '[data-testid="expert-info-skills-add"]',
        );
        const addButtonSnapshot = visibleElementSnapshot(addButton);
        const baseChip = document.querySelector(
          '[data-testid="expert-info-skill-chip-skill-code-review"]',
        );
        const addedChip = document.querySelector(
          `[data-testid="${addedSkillChipTestId}"]`,
        );
        const addedReadiness = document.querySelector(
          `[data-testid="${addedSkillReadinessTestId}"]`,
        );
        return {
          panelVisible: Boolean(panel),
          titleVisible: text.includes(expertTitle),
          addButtonVisible: addButtonSnapshot.visible,
          addButtonLabel: addButtonSnapshot.label,
          addButtonDisabled:
            addButton instanceof HTMLButtonElement ? addButton.disabled : null,
          baseSkillVisible:
            panelText.includes(baseSkillRef) || Boolean(baseChip),
          addedSkillVisible:
            panelText.includes(addedSkillRef) ||
            panelText.includes(addedSkillUiRef) ||
            panelText.includes("Capability Report") ||
            Boolean(addedChip),
          addedChipVisible: Boolean(addedChip),
          addedReadinessVisible: Boolean(addedReadiness),
          addedReadinessText: addedReadiness?.textContent || "",
          bodyText: text,
        };
      },
      {
        expertTitle: EXPERT_SKILLS_RUNTIME_TITLE,
        baseSkillRef: EXPERT_SKILLS_RUNTIME_BASE_SKILL_REF,
        addedSkillRef: EXPERT_SKILLS_RUNTIME_SKILL_REF,
        addedSkillUiRef: EXPERT_PANEL_SKILLS_RUNTIME_UI_SKILL_REF,
        addedSkillChipTestId: EXPERT_PANEL_SKILLS_RUNTIME_UI_CHIP_TEST_ID,
        addedSkillReadinessTestId:
          EXPERT_PANEL_SKILLS_RUNTIME_UI_READINESS_TEST_ID,
      },
    );
    lastSnapshot = ready;
    if (
      ready?.panelVisible &&
      ready.titleVisible &&
      ready.addedSkillVisible &&
      (ready.addedChipVisible || ready.addedReadinessVisible)
    ) {
      return sanitizeJson({
        ready,
        alreadyAdded: true,
        pickerOpened: null,
        candidate: null,
        added: ready,
        pickerClosed: { skipped: true, reason: "skill already visible" },
      });
    }
    if (
      ready?.panelVisible &&
      ready.titleVisible &&
      ready.addButtonVisible &&
      ready.addButtonDisabled !== true
    ) {
      break;
    }
    await sleep(options.intervalMs);
  }

  assert(
    lastSnapshot?.panelVisible &&
      lastSnapshot?.addButtonVisible &&
      lastSnapshot?.addButtonDisabled !== true,
    `专家信息面板未出现可添加技能入口: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );

  const pickerTrigger = await clickExpertSkillPickerTrigger(page);
  assert(
    pickerTrigger.clicked === true,
    `专家技能选择器入口点击失败: ${JSON.stringify(
      sanitizeJson(pickerTrigger),
    )}`,
  );
  const pickerOpened = await waitForExpertSkillPickerState(page, options, {
    expectCandidate: false,
  });
  const pickerSearch = await setExpertSkillPickerQuery(
    page,
    "capability-report",
  );

  const candidate = await waitForExpertSkillPickerState(page, options, {
    expectCandidate: true,
  });
  await page
    .locator(`[data-testid="${EXPERT_PANEL_SKILLS_RUNTIME_UI_ADD_TEST_ID}"]`)
    .click();

  const added = await waitForExpertPanelAddedSkill(page, options);
  const pickerClosed = await closeExpertSkillPickerAfterSelection(
    page,
    options,
  );
  return sanitizeJson({
    ready: lastSnapshot,
    pickerTrigger,
    pickerOpened,
    pickerSearch,
    candidate,
    added,
    pickerClosed,
  });
}

async function clickExpertSkillPickerTrigger(page) {
  return sanitizeJson(
    await page.evaluate(() => {
      const visibleElementSnapshot = (element) => {
        const rect = element?.getBoundingClientRect();
        const style = element ? window.getComputedStyle(element) : null;
        return {
          visible: Boolean(
            element &&
            rect &&
            rect.width > 8 &&
            rect.height > 8 &&
            style?.visibility !== "hidden" &&
            style?.display !== "none",
          ),
          label:
            element?.getAttribute?.("aria-label") ||
            element?.getAttribute?.("title") ||
            element?.textContent ||
            "",
        };
      };
      const panel = document.querySelector('[data-testid="expert-info-panel"]');
      const mappingAction = panel?.querySelector(
        '[data-testid="expert-info-skills-runtime-action-skill-code-review"]',
      );
      const addButtons = Array.from(
        panel?.querySelectorAll('[data-testid="expert-info-skills-add"]') ?? [],
      );
      const candidates = [mappingAction, ...addButtons].filter(Boolean);
      const button = candidates.find(
        (candidate) => visibleElementSnapshot(candidate).visible,
      );
      if (!(button instanceof HTMLElement)) {
        return {
          clicked: false,
          reason: "missing-visible-trigger",
          panelVisible: Boolean(panel),
          triggerCount: candidates.length,
          triggers: candidates.map((candidate) =>
            visibleElementSnapshot(candidate),
          ),
          bodyText: document.body?.innerText || "",
        };
      }
      button.scrollIntoView({ block: "center", inline: "center" });
      button.click();
      const snapshot = visibleElementSnapshot(button);
      return {
        clicked: true,
        panelVisible: Boolean(panel),
        label: snapshot.label,
        triggerKind:
          button.getAttribute("data-testid") ===
          "expert-info-skills-runtime-action-skill-code-review"
            ? "mapping-action"
            : "add-button",
        triggerCount: candidates.length,
      };
    }),
  );
}

async function setExpertSkillPickerQuery(page, query) {
  return sanitizeJson(
    await page.evaluate((nextQuery) => {
      const input = document.querySelector(
        '[data-testid="expert-skill-picker-dialog"] input',
      );
      if (!(input instanceof HTMLInputElement)) {
        return { updated: false, reason: "missing-skill-picker-input" };
      }
      const previousValue = input.value;
      input.focus();
      const valueSetter = Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )?.set;
      if (valueSetter) {
        valueSetter.call(input, nextQuery);
      } else {
        input.value = nextQuery;
      }
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
      return {
        updated: true,
        previousValue,
        nextValue: input.value,
      };
    }, query),
  );
}

async function waitForExpertSkillPickerState(
  page,
  options,
  { expectCandidate },
) {
  const startedAt = Date.now();
  let lastSnapshot = null;

  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ candidateTestId, addTestId }) => {
        const text = document.body?.innerText || "";
        const panel = document.querySelector(
          '[data-testid="expert-info-panel"]',
        );
        const dialog = document.querySelector(
          '[data-testid="expert-skill-picker-dialog"]',
        );
        const input = dialog?.querySelector("input");
        const candidate = document.querySelector(
          `[data-testid="${candidateTestId}"]`,
        );
        const addButton = document.querySelector(
          `[data-testid="${addTestId}"]`,
        );
        const candidateTestIds = Array.from(
          dialog?.querySelectorAll(
            '[data-testid^="expert-skill-candidate-"]',
          ) ?? [],
        )
          .map((element) => element.getAttribute("data-testid") || "")
          .filter(Boolean);
        const rawCatalog =
          window.localStorage.getItem("lime:skill-catalog:v1") || "";
        let catalogTenantId = null;
        let catalogItemCount = null;
        try {
          const parsedCatalog = JSON.parse(rawCatalog);
          catalogTenantId = parsedCatalog?.tenantId ?? null;
          catalogItemCount = Array.isArray(parsedCatalog?.items)
            ? parsedCatalog.items.length
            : null;
        } catch {
          catalogTenantId = null;
          catalogItemCount = null;
        }
        const oemTenantId = window.__LIME_OEM_CLOUD__?.tenantId ?? null;
        return {
          expertPanelVisible: Boolean(panel),
          dialogVisible: Boolean(dialog),
          query: input instanceof HTMLInputElement ? input.value : null,
          candidateVisible: Boolean(candidate),
          addButtonVisible: Boolean(addButton),
          addButtonDisabled:
            addButton instanceof HTMLButtonElement ? addButton.disabled : null,
          expectedCandidateTestId: candidateTestId,
          candidateTestIds,
          skillCatalogRawPresent: rawCatalog.length > 0,
          skillCatalogRawIncludesCapabilityReport:
            rawCatalog.includes("capability-report"),
          catalogTenantId,
          catalogItemCount,
          oemTenantId,
          hasCapabilityReportText:
            text.includes("capability-report") ||
            text.includes("Capability Report"),
          bodyText: text,
        };
      },
      {
        candidateTestId: EXPERT_PANEL_SKILLS_RUNTIME_UI_CANDIDATE_TEST_ID,
        addTestId: EXPERT_PANEL_SKILLS_RUNTIME_UI_ADD_TEST_ID,
      },
    );
    lastSnapshot = snapshot;
    if (!snapshot?.expertPanelVisible && !snapshot?.dialogVisible) {
      break;
    }
    if (!snapshot?.dialogVisible) {
      await sleep(options.intervalMs);
      continue;
    }
    if (!expectCandidate) {
      return snapshot;
    }
    if (
      snapshot.candidateVisible &&
      snapshot.addButtonVisible &&
      snapshot.addButtonDisabled === false
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `专家技能选择器未达到预期状态: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function waitForExpertPanelAddedSkill(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({
        baseSkillRef,
        addedSkillRef,
        addedSkillUiRef,
        addedSkillChipTestId,
        addedSkillReadinessTestId,
      }) => {
        const text = document.body?.innerText || "";
        const baseChip = document.querySelector(
          '[data-testid="expert-info-skill-chip-skill-code-review"]',
        );
        const addedChip = document.querySelector(
          `[data-testid="${addedSkillChipTestId}"]`,
        );
        const readiness = document.querySelector(
          `[data-testid="${addedSkillReadinessTestId}"]`,
        );
        return {
          baseSkillVisible: text.includes(baseSkillRef) || Boolean(baseChip),
          addedSkillVisible:
            text.includes(addedSkillRef) ||
            text.includes(addedSkillUiRef) ||
            Boolean(addedChip),
          addedChipVisible: Boolean(addedChip),
          readinessText: readiness?.textContent || "",
          dialogVisible: Boolean(
            document.querySelector(
              '[data-testid="expert-skill-picker-dialog"]',
            ),
          ),
          bodyText: text,
        };
      },
      {
        baseSkillRef: EXPERT_SKILLS_RUNTIME_BASE_SKILL_REF,
        addedSkillRef: EXPERT_SKILLS_RUNTIME_SKILL_REF,
        addedSkillUiRef: EXPERT_PANEL_SKILLS_RUNTIME_UI_SKILL_REF,
        addedSkillChipTestId: EXPERT_PANEL_SKILLS_RUNTIME_UI_CHIP_TEST_ID,
        addedSkillReadinessTestId:
          EXPERT_PANEL_SKILLS_RUNTIME_UI_READINESS_TEST_ID,
      },
    );
    lastSnapshot = snapshot;
    if (snapshot?.addedChipVisible && snapshot.addedSkillVisible) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `专家面板未显示新增技能: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

async function closeExpertSkillPickerAfterSelection(page, options) {
  const closed = await page.evaluate(() => {
    const dialog = document.querySelector(
      '[data-testid="expert-skill-picker-dialog"]',
    );
    if (!dialog) {
      return { clicked: false, alreadyClosed: true };
    }
    const buttons = Array.from(dialog.querySelectorAll("button"));
    const closeButton =
      buttons.find((button) => {
        const label = [
          button.getAttribute("aria-label") || "",
          button.getAttribute("title") || "",
          button.textContent || "",
        ].join("\n");
        return label.includes("关闭") || /\bClose\b/i.test(label);
      }) ?? buttons[0];
    if (closeButton instanceof HTMLElement) {
      closeButton.click();
      return {
        clicked: true,
        alreadyClosed: false,
        label:
          closeButton.getAttribute("aria-label") ||
          closeButton.getAttribute("title") ||
          closeButton.textContent ||
          "",
      };
    }
    return {
      clicked: false,
      alreadyClosed: false,
      buttonCount: buttons.length,
    };
  });

  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(page, () => ({
      dialogVisible: Boolean(
        document.querySelector('[data-testid="expert-skill-picker-dialog"]'),
      ),
      textareaVisible: Boolean(
        document.querySelector('textarea[name="agent-chat-message"]'),
      ),
    }));
    lastSnapshot = snapshot;
    if (snapshot?.dialogVisible === false) {
      return sanitizeJson({
        ...closed,
        dialogVisible: false,
        textareaVisible: snapshot.textareaVisible,
      });
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `专家技能选择器未关闭: ${JSON.stringify(
      sanitizeJson({ clicked: closed, lastSnapshot }),
    )}`,
  );
}

export function selectExpertPanelSkillsRuntimeSessionId(
  expertPanelSkillsRuntimeBackendTurn,
  expertPlazaSkillsRuntimeSessionId,
) {
  return (
    expertPanelSkillsRuntimeBackendTurn?.entry?.sessionId ??
    expertPlazaSkillsRuntimeSessionId
  );
}

export function summarizeExpertPanelSkillsRuntimeTurnStart(
  expertPanelSkillsRuntimeBackendTurn,
) {
  return sanitizeJson({
    sessionId: expertPanelSkillsRuntimeBackendTurn?.entry?.sessionId ?? null,
    turnId: expertPanelSkillsRuntimeBackendTurn?.entry?.turnId ?? null,
    inputText: expertPanelSkillsRuntimeBackendTurn?.entry?.inputText ?? null,
    usesPanelPrompt:
      expertPanelSkillsRuntimeBackendTurn?.entry?.inputText ===
      EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
  });
}
