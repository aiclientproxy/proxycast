import fs from "node:fs";
import path from "node:path";
import { APP_SERVER_HANDLE_JSON_LINES_COMMAND } from "./claw-chat-current-fixture-constants.mjs";
import {
  decodeJsonRpcLines,
  evaluatePageSnapshot,
  readTraceMessages,
} from "./claw-chat-current-fixture-rpc.mjs";
import {
  assert,
  sanitizeJson,
  sleep,
  writeJsonFile,
} from "./claw-chat-current-fixture-utils.mjs";

const SKILLS_CHANGED_FIXTURE_SKILL_NAME = "notification-refresh";
const SKILL_LIST_METHOD = "skill/list";
const SKILLS_CHANGED_DEBUG_MARKER = "skillsChanged.received";

async function readSkillListTrace(page) {
  const traceRaw = await page.evaluate(() =>
    window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
  );
  const matchingEntries = readTraceMessages(traceRaw).filter((entry) => {
    if (entry?.command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
      return false;
    }
    return decodeJsonRpcLines(entry?.args_preview?.request?.lines).some(
      (message) => message?.method === SKILL_LIST_METHOD,
    );
  });
  return {
    method: SKILL_LIST_METHOD,
    totalCount: matchingEntries.length,
    electronIpcSuccessCount: matchingEntries.filter(
      (entry) =>
        entry?.transport === "electron-ipc" && entry?.status === "success",
    ).length,
  };
}

async function readSkillSelectorSnapshot(page) {
  return await evaluatePageSnapshot(
    page,
    (skillName) => {
      const panel = document.querySelector(
        '[data-testid="inputbar-plus-panel-skills"]',
      );
      const selector = document.querySelector(
        '[data-testid="skill-selector-inline"]',
      );
      const refreshClicks = Number(
        document.documentElement.dataset.skillsRuntimeRefreshClicks || "0",
      );
      return {
        panelVisible:
          panel instanceof HTMLElement && panel.offsetParent !== null,
        selectorVisible:
          selector instanceof HTMLElement && selector.offsetParent !== null,
        skillVisible: Boolean(
          selector?.textContent?.includes(String(skillName || "")),
        ),
        manualRefreshClickCount: Number.isFinite(refreshClicks)
          ? refreshClicks
          : null,
      };
    },
    SKILLS_CHANGED_FIXTURE_SKILL_NAME,
  );
}

async function installManualRefreshClickTracker(page) {
  await page.evaluate(() => {
    const runtimeWindow = window;
    runtimeWindow.__limeSkillsRuntimeRefreshTrackerCleanup?.();
    document.documentElement.dataset.skillsRuntimeRefreshClicks = "0";
    const onClick = (event) => {
      const target =
        event.target instanceof Element
          ? event.target.closest('[data-testid="skill-selector-refresh"]')
          : null;
      if (!target) {
        return;
      }
      const count = Number(
        document.documentElement.dataset.skillsRuntimeRefreshClicks || "0",
      );
      document.documentElement.dataset.skillsRuntimeRefreshClicks = String(
        Number.isFinite(count) ? count + 1 : 1,
      );
    };
    document.addEventListener("click", onClick, true);
    runtimeWindow.__limeSkillsRuntimeRefreshTrackerCleanup = () => {
      document.removeEventListener("click", onClick, true);
      delete runtimeWindow.__limeSkillsRuntimeRefreshTrackerCleanup;
    };
  });
}

async function removeManualRefreshClickTracker(page) {
  await page
    .evaluate(() => {
      window.__limeSkillsRuntimeRefreshTrackerCleanup?.();
    })
    .catch(() => undefined);
}

function writeSkillsChangedFixtureSkill(runtimeEnv) {
  const home = runtimeEnv?.env?.HOME;
  assert(home, "Skills changed fixture 缺少临时 HOME");
  const skillDirectory = path.join(
    home,
    ".agents",
    "skills",
    SKILLS_CHANGED_FIXTURE_SKILL_NAME,
  );
  fs.mkdirSync(skillDirectory, { recursive: true });
  fs.writeFileSync(
    path.join(skillDirectory, "SKILL.md"),
    [
      "---",
      `name: ${SKILLS_CHANGED_FIXTURE_SKILL_NAME}`,
      "description: Proves typed skills changed catalog invalidation.",
      "---",
      "",
      `# ${SKILLS_CHANGED_FIXTURE_SKILL_NAME}`,
      "",
      "Use only for the controlled Electron catalog refresh fixture.",
      "",
    ].join("\n"),
  );
  return { skillName: SKILLS_CHANGED_FIXTURE_SKILL_NAME };
}

export async function verifySkillsChangedCatalogRefresh(
  page,
  options,
  runtimeEnv,
) {
  const beforeOpenTrace = await readSkillListTrace(page);
  await page.locator('[data-testid="inputbar-plus-trigger"]').click();
  await page.locator('[data-testid="inputbar-plus-skills"]').click();
  await page
    .locator('[data-testid="inputbar-plus-panel-skills"]')
    .waitFor({ state: "visible", timeout: options.timeoutMs });
  await page
    .locator('[data-testid="skill-selector-inline"]')
    .waitFor({ state: "visible", timeout: options.timeoutMs });

  const initialStartedAt = Date.now();
  let initialTrace = beforeOpenTrace;
  let initialGui = null;
  while (Date.now() - initialStartedAt < options.timeoutMs) {
    initialTrace = await readSkillListTrace(page);
    initialGui = await readSkillSelectorSnapshot(page);
    if (
      initialGui?.panelVisible &&
      initialGui?.selectorVisible &&
      initialTrace.electronIpcSuccessCount >
        beforeOpenTrace.electronIpcSuccessCount
    ) {
      break;
    }
    await sleep(options.intervalMs);
  }
  assert(
    initialTrace.electronIpcSuccessCount >
      beforeOpenTrace.electronIpcSuccessCount,
    `技能面板未触发初始 current skill/list: ${JSON.stringify(
      sanitizeJson({ beforeOpenTrace, initialTrace, initialGui }),
    )}`,
  );

  await installManualRefreshClickTracker(page);
  const notificationMarkers = [];
  const collectNotificationMarker = (message) => {
    const text = String(message.text?.() || "");
    if (text.includes(SKILLS_CHANGED_DEBUG_MARKER)) {
      notificationMarkers.push(SKILLS_CHANGED_DEBUG_MARKER);
    }
  };
  page.on("console", collectNotificationMarker);

  let completed = null;
  let lastSnapshot = null;
  try {
    const fixtureSkill = writeSkillsChangedFixtureSkill(runtimeEnv);
    const startedAt = Date.now();
    while (Date.now() - startedAt < options.timeoutMs) {
      const trace = await readSkillListTrace(page);
      const gui = await readSkillSelectorSnapshot(page);
      lastSnapshot = { trace, gui, markerCount: notificationMarkers.length };
      if (
        notificationMarkers.length > 0 &&
        trace.electronIpcSuccessCount > initialTrace.electronIpcSuccessCount &&
        gui?.panelVisible &&
        gui?.selectorVisible &&
        gui?.skillVisible &&
        gui?.manualRefreshClickCount === 0
      ) {
        completed = {
          skillName: fixtureSkill.skillName,
          initialCatalog: {
            beforeOpen: beforeOpenTrace,
            afterPanelOpen: initialTrace,
            gui: initialGui,
          },
          notification: {
            method: "skills/changed",
            marker: SKILLS_CHANGED_DEBUG_MARKER,
            markerCount: notificationMarkers.length,
          },
          automaticRefresh: {
            beforeCount: initialTrace.electronIpcSuccessCount,
            afterCount: trace.electronIpcSuccessCount,
            increment:
              trace.electronIpcSuccessCount -
              initialTrace.electronIpcSuccessCount,
            method: SKILL_LIST_METHOD,
            transport: "electron-ipc",
          },
          gui,
          manualRefresh: {
            clickCount: gui.manualRefreshClickCount,
          },
        };
        break;
      }
      await sleep(options.intervalMs);
    }
  } finally {
    page.off("console", collectNotificationMarker);
    await removeManualRefreshClickTracker(page);
  }

  assert(
    completed,
    `skills/changed 未自动刷新 GUI catalog: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
  await page.keyboard.press("Escape").catch(() => undefined);
  return sanitizeJson(completed);
}

export async function waitForExpertPanelSkillsRuntimeSessionReady(
  page,
  options,
  expectedSessionId,
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      (sessionId) => {
        const text = document.body?.innerText || "";
        const textareas = Array.from(
          document.querySelectorAll('textarea[name="agent-chat-message"]'),
        ).filter((node) => node instanceof HTMLTextAreaElement);
        const textarea = textareas.find(
          (node) => !sessionId || node.dataset.sessionId === sessionId,
        );
        const fallbackTextarea = textareas[0] ?? null;
        const isVisibleTextarea = (node) =>
          node instanceof HTMLElement
            ? node.offsetParent !== null
            : Boolean(node);
        return {
          url: window.location.href,
          expectedSessionId: sessionId,
          hasExpertPrompt: text.includes(
            "请以「代码文学专家」身份，使用绑定技能完成一次最小代码审查。",
          ),
          hasExpertPanel:
            text.includes("专家信息") && text.includes("代码文学专家"),
          hasAddedSkill: text.includes("Capability Report"),
          textareaSessionId:
            textarea instanceof HTMLTextAreaElement
              ? textarea.dataset.sessionId || null
              : null,
          textareaVisible: isVisibleTextarea(textarea),
          textareaDisabled:
            textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
          fallbackTextareaSessionId:
            fallbackTextarea instanceof HTMLTextAreaElement
              ? fallbackTextarea.dataset.sessionId || null
              : null,
          fallbackTextareaVisible: isVisibleTextarea(fallbackTextarea),
          fallbackTextareaDisabled:
            fallbackTextarea instanceof HTMLTextAreaElement
              ? fallbackTextarea.disabled
              : null,
          textareaCount: textareas.length,
          bodyText: text,
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
      snapshot.hasExpertPrompt &&
      snapshot.hasExpertPanel &&
      snapshot.hasAddedSkill
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `GUI 未恢复专家面板 Skills runtime 会话: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

export function writeCapabilityReportSkillPackage(skillDirectory) {
  const skillFilePath = path.join(skillDirectory, "SKILL.md");
  fs.mkdirSync(skillDirectory, { recursive: true });
  fs.writeFileSync(
    skillFilePath,
    [
      "---",
      "name: Capability Report",
      "description: Fixture skill for Skills runtime manual enable evidence.",
      "allowed-tools: Read",
      "---",
      "",
      "# Capability Report",
      "",
      "Use this fixture skill only to prove workspace-local manual session enable.",
      "",
    ].join("\n"),
  );
  return { skillFilePath };
}

export function ensureManualEnableWorkspaceSkill(workspaceRoot) {
  const skillDirectory = path.join(
    workspaceRoot,
    ".agents",
    "skills",
    "capability-report",
  );
  const { skillFilePath } = writeCapabilityReportSkillPackage(skillDirectory);
  const registrationDirectory = path.join(skillDirectory, ".lime");
  const registrationFilePath = path.join(
    registrationDirectory,
    "registration.json",
  );
  fs.mkdirSync(registrationDirectory, { recursive: true });
  writeJsonFile(registrationFilePath, {
    registrationId: "capreg-fixture-capability-report",
    registeredAt: "2026-06-21T00:00:00.000Z",
    skillDirectory: "capability-report",
    registeredSkillDirectory: skillDirectory,
    sourceDraftId: "capdraft-fixture-capability-report",
    sourceVerificationReportId: "capver-fixture-capability-report",
    generatedFileCount: 1,
    permissionSummary: ["Level 0 read-only fixture"],
  });
  return {
    skillDirectory,
    skillFilePath,
    registrationFilePath,
  };
}

export function ensureUserVisibleCapabilityReportSkill(runtimeEnv) {
  const home = runtimeEnv?.env?.HOME;
  assert(home, "Expert Panel Skills Runtime fixture 缺少临时 HOME");
  const skillDirectory = path.join(
    home,
    ".agents",
    "skills",
    "capability-report",
  );
  const { skillFilePath } = writeCapabilityReportSkillPackage(skillDirectory);
  return {
    skillDirectory,
    skillFilePath,
  };
}

export async function launchSkillsRuntimeFromWorkspacePanel(
  page,
  options,
  workspace,
) {
  assert(
    workspace?.rootPath,
    "workspace panel fixture 缺少 workspace rootPath",
  );
  const workspaceSkill = ensureManualEnableWorkspaceSkill(workspace.rootPath);
  const startedAt = Date.now();
  let lastSnapshot = null;

  await page.locator('[data-testid="app-sidebar-nav-skills"]').click();

  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(page, () => {
      const text = document.body?.innerText || "";
      const installedView = document.querySelector(
        '[data-testid="skills-installed-view"]',
      );
      const installedTab = Array.from(document.querySelectorAll("button")).find(
        (button) => (button.textContent || "").includes("用户安装"),
      );
      if (!installedView && installedTab instanceof HTMLButtonElement) {
        installedTab.click();
      }
      const panel = document.querySelector(
        '[data-testid="workspace-registered-skills-panel"]',
      );
      const enableButton = document.querySelector(
        '[data-testid="workspace-registered-skill-enable-runtime"]',
      );
      return {
        text,
        skillsPageVisible:
          text.includes("Skills") ||
          text.includes("技能广场") ||
          text.includes("用户安装"),
        installedViewVisible: Boolean(installedView),
        registeredPanelVisible: Boolean(panel),
        registeredSkillVisible: text.includes("Capability Report"),
        enableButtonVisible: Boolean(enableButton),
        enableButtonDisabled:
          enableButton instanceof HTMLButtonElement
            ? enableButton.disabled
            : null,
      };
    });
    lastSnapshot = snapshot;
    if (
      snapshot.registeredPanelVisible &&
      snapshot.registeredSkillVisible &&
      snapshot.enableButtonVisible &&
      snapshot.enableButtonDisabled === false
    ) {
      await page
        .locator('[data-testid="workspace-registered-skill-enable-runtime"]')
        .click();
      return sanitizeJson({
        ...snapshot,
        clicked: true,
        workspaceSkill,
      });
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `Skills 工作台未出现可试用的已保存技能: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}
