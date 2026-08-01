import fs from "node:fs";
import { describe, expect, it } from "vitest";

function readSmokeScript() {
  return [
    "scripts/agent-runtime/current-fixture-regression-smoke.mjs",
    "scripts/lib/electron-fixture-build.mjs",
  ]
    .map((filePath) => fs.readFileSync(filePath, "utf8"))
    .join("\n");
}

function readContentFactoryFixture() {
  return fs.readFileSync(
    "scripts/agent-runtime/claw-chat-current-fixture-content-factory-article-workspace.mjs",
    "utf8",
  );
}

function readContentFactoryAssertions() {
  return fs.readFileSync(
    "scripts/agent-runtime/claw-chat-current-fixture-content-factory-assertions.mjs",
    "utf8",
  );
}

function readFixtureConstants() {
  return fs.readFileSync(
    "scripts/agent-runtime/claw-chat-current-fixture-constants.mjs",
    "utf8",
  );
}

describe("agent runtime current fixture regression smoke guard", () => {
  it("runs current Agent Runtime regression tests through Vitest smoke runner", () => {
    const content = readSmokeScript();

    expect(content).toContain("runVitestSmoke");
    expect(content).toContain("runNodeSmoke");
    expect(content).toContain("runElectronFixtureSmoke");
    expect(content).toContain(
      "src/components/agent/chat/hooks/agentChatHistory.test.ts",
    );
    expect(content).toContain(
      "src/components/agent/chat/hooks/agentSessionScopedStorage.test.ts",
    );
    expect(content).toContain(
      "src/components/agent/chat/hooks/agentStreamCompletionController.test.ts",
    );
    expect(content).toContain(
      "src/components/agent/chat/hooks/agentStreamRuntimeHandler.unit.test.ts",
    );
    expect(content).toContain(
      "src/components/agent/chat/hooks/agentStreamRuntimeHandler.typedError.test.ts",
    );
    expect(content).toContain(
      "src/components/agent/chat/components/MessageList.test.tsx",
    );
  });

  it("runs both typed error Electron Gate B fixtures", () => {
    const content = readSmokeScript();

    expect(content).toContain(
      "Claw typed error retry success Electron fixture",
    );
    expect(content).toContain("typed-error-retry-success");
    expect(content).toContain(
      "claw-chat-current-fixture-typed-error-retry-success-regression",
    );
    expect(content).toContain(
      "Claw typed error retry failure Electron fixture",
    );
    expect(content).toContain("typed-error-retry-failure");
    expect(content).toContain(
      "claw-chat-current-fixture-typed-error-retry-failure-regression",
    );
  });

  it("keeps active turn steer in the exact Electron regression matrix", () => {
    const content = readSmokeScript();

    expect(content).toContain(
      "Claw active turn steer same identity Electron fixture",
    );
    expect(content).toContain('"inputbar-active-steer"');
    expect(content).toContain(
      "claw-chat-current-fixture-active-steer-regression",
    );
    expect(content).toContain(
      "active turn 经 thread/read -> turn/steer 保持同一 Turn identity",
    );
  });

  it("keeps Electron fixture guards in the current regression set", () => {
    const content = readSmokeScript();

    expect(content).toContain(
      "scripts/electron/session-history-fixture-smoke.test.mjs",
    );
    expect(content).toContain(
      "scripts/electron/code-artifact-workbench-fixture-smoke.test.mjs",
    );
    expect(content).toContain(
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.test.mjs",
    );
    expect(content).toContain("Electron/App Server fixture smoke guard");
    expect(content).toContain("Claw GUI current fixture guard");
  });

  it("runs the real Electron cancel-then-continue Claw fixture", () => {
    const content = readSmokeScript();

    expect(content).toContain("Claw 停止后同会话继续输出 Electron fixture");
    expect(content).toContain(
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
    );
    expect(content).toContain("--scenario");
    expect(content).toContain("cancel-then-continue");
    expect(content).toContain(
      "claw-chat-current-fixture-cancel-then-continue-regression",
    );
    expect(content).toContain("停止后同会话继续输出 Electron fixture");
  });

  it("runs the real Electron home first-send hotpath fixture", () => {
    const content = readSmokeScript();

    expect(content).toContain("Claw 首页首发热路径 Electron fixture");
    expect(content).toContain(
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
    );
    expect(content).toContain("--scenario");
    expect(content).toContain("home-hotpath");
    expect(content).toContain(
      "claw-chat-current-fixture-home-hotpath-regression",
    );
    expect(content).toContain("--timeout-ms");
    expect(content).toContain("240000");
    expect(content).toContain(
      "首页首发热路径用户消息可见/无空对话/无模型回退 Electron fixture",
    );
  });

  it("runs the real Electron greeting home first-send hotpath fixture", () => {
    const content = readSmokeScript();

    expect(content).toContain("Claw 首页首发短问候热路径 Electron fixture");
    expect(content).toContain(
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
    );
    expect(content).toContain("--scenario");
    expect(content).toContain("home-hotpath-greeting");
    expect(content).toContain(
      "claw-chat-current-fixture-home-hotpath-greeting-regression",
    );
    expect(content).toContain("--timeout-ms");
    expect(content).toContain("240000");
  });

  it("runs the unknown Item fail-visible and recovery Electron fixture", () => {
    const content = readSmokeScript();

    expect(content).toContain(
      "Claw unknown Item fail-visible/recovery Electron fixture",
    );
    expect(content).toContain("--scenario");
    expect(content).toContain("unknown-item");
    expect(content).toContain(
      "claw-chat-current-fixture-unknown-item-regression",
    );
  });

  it("runs the real Electron approval decision fixtures in the current regression set", () => {
    const content = readSmokeScript();

    expect(content).toContain(
      "Claw approval allow-for-session resume Electron fixture",
    );
    expect(content).toContain("approval-request-resume");
    expect(content).toContain(
      "claw-chat-current-fixture-approval-request-resume-regression",
    );
    expect(content).toContain(
      "Claw approval decline-continue Electron fixture",
    );
    expect(content).toContain("approval-request-decline");
    expect(content).toContain(
      "claw-chat-current-fixture-approval-request-decline-regression",
    );
    expect(content).toContain("Claw approval cancel-turn Electron fixture");
    expect(content).toContain("approval-request-cancel");
    expect(content).toContain(
      "claw-chat-current-fixture-approval-request-cancel-regression",
    );
    expect(content).toContain(
      "Claw approval full-access no prompt Electron fixture",
    );
    expect(content).toContain("approval-request-full-access");
    expect(content).toContain(
      "claw-chat-current-fixture-approval-request-full-access-regression",
    );
    expect(content).toContain(
      "approval full-access 不弹 prompt / 不生成 timeline record Electron fixture",
    );
  });

  it("runs the real Electron Plan history hydrate Claw fixture", () => {
    const content = readSmokeScript();

    expect(content).toContain(
      "Claw Plan revisioned history hydrate Electron fixture",
    );
    expect(content).toContain(
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
    );
    expect(content).toContain("--scenario");
    expect(content).toContain("plan");
    expect(content).toContain(
      "claw-chat-current-fixture-plan-history-hydrate-regression",
    );
    expect(content).toContain(
      "Plan revisioned thread item + history hydrate Electron fixture",
    );
  });

  it("runs the real Electron image-command Claw fixture in the current regression set", () => {
    const content = readSmokeScript();

    expect(content).toContain("Claw 图片命令 GUI Electron fixture");
    expect(content).toContain(
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
    );
    expect(content).toContain("--scenario");
    expect(content).toContain("image-command");
    expect(content).toContain(
      "claw-chat-current-fixture-image-command-regression",
    );
    expect(content).toContain("真实 GUI 图片命令到 Claw Chat Electron fixture");
  });

  it("runs the real Electron plain image intent Claw fixture in the current regression set", () => {
    const content = readSmokeScript();

    expect(content).toContain("Claw 普通画图意图 GUI Electron fixture");
    expect(content).toContain(
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
    );
    expect(content).toContain("--scenario");
    expect(content).toContain("plain-image-intent");
    expect(content).toContain(
      "claw-chat-current-fixture-plain-image-intent-regression",
    );
    expect(content).toContain(
      "普通自然语言画图意图到同一图片 task 主链 Electron fixture",
    );
  });

  it("runs the real Coding Workbench Electron fixture in the current regression set", () => {
    const content = readSmokeScript();

    expect(content).toContain("Coding Workbench Electron fixture");
    expect(content).toContain(
      "scripts/electron/code-artifact-workbench-fixture-smoke.mjs",
    );
    expect(content).toContain("--scenario");
    expect(content).toContain("gui-coding-input");
    expect(content).toContain(
      "code-artifact-workbench-gui-coding-input-regression",
    );
    expect(content).toContain(
      "真实 GUI coding 输入到 Coding Workbench Electron fixture",
    );
  });

  it("runs the Content Factory article Article Editor Electron fixture", () => {
    const content = readSmokeScript();

    expect(content).toContain(
      "Content Factory article Article Editor Electron fixture",
    );
    expect(content).toContain(
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
    );
    expect(content).toContain("--scenario");
    expect(content).toContain("content-factory-article-workspace");
    expect(content).toContain(
      "claw-chat-current-fixture-content-factory-article-workspace-regression",
    );
    expect(content).toContain(
      "内容工厂文章 Article Editor / articleDraft 右侧产物闭环 Electron fixture",
    );
  });

  it("selects the Content Factory article frame by artifact identity", () => {
    const content = readContentFactoryFixture();

    expect(content).toContain(
      'candidate.getAttribute("data-artifact-ref") === artifactRef',
    );
    expect(content).toContain(
      "contentFactoryArticleWorkspaceArticleArtifactIdentity",
    );
    expect(content).toContain("articleObjectRef.artifactIds.find");
    expect(content).toContain("artifactRef: workerArticleArtifactRef");
    expect(content).toContain(
      "readModelArticleArtifactRef === workerArticleArtifactRef",
    );
    expect(content).toContain("targetFrameCount: targetFrames.length");
    expect(content).not.toContain(
      "const frame = frames.find(isVisible) ?? frames[0] ?? null",
    );
  });

  it("does not synthesize Content Factory workflow response or contract probe", () => {
    const content = readContentFactoryFixture();
    const assertions = readContentFactoryAssertions();
    const constants = readFixtureConstants();

    expect(content).not.toContain("APP_SERVER_METHOD_WORKFLOW_RESPOND");
    expect(content).not.toContain('type: "action.required"');
    expect(content).not.toContain(
      "startContentFactoryWorkflowRespondActionTurn",
    );
    expect(content).not.toContain("runRuntimeContractRejectionProbe");
    expect(constants).not.toContain(
      "CONTENT_FACTORY_ARTICLE_WORKSPACE_WORKFLOW_RESPOND_",
    );
    expect(constants).not.toContain(
      "CONTENT_FACTORY_ARTICLE_WORKSPACE_CONTRACT_REJECT_",
    );
    expect(assertions).not.toContain(
      "contentFactoryArticleWorkspaceWorkflowRespondProjected",
    );
    expect(assertions).not.toContain(
      "contentFactoryArticleWorkspaceRuntimeContractFailClosed",
    );
    expect(content).toContain("APP_SERVER_METHOD_ARTIFACT_WRITE");
    expect(content).not.toContain(
      "APP_SERVER_METHOD_AGENT_SESSION_RUNTIME_EVENTS_APPEND",
    );
    expect(content).not.toContain('type: "workflow.run.started"');
    expect(content).not.toContain('type: "runtime.error"');
  });

  it("runs the media item reference Claw fixture in the current regression set", () => {
    const content = readSmokeScript();

    expect(content).toContain(
      "Claw media item reference Agent Chat GUI Electron fixture",
    );
    expect(content).toContain(
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
    );
    expect(content).toContain("--scenario");
    expect(content).toContain("media-reference");
    expect(content).toContain(
      "claw-chat-current-fixture-media-reference-regression",
    );
    expect(content).toContain(
      "media item / imageView 引用到 Agent Chat 卡片与 Workbench source 预览 Electron fixture",
    );
  });

  it("keeps the aggregate fixture smoke diagnosable and app-url aware", () => {
    const content = readSmokeScript();

    expect(content).toContain("function printHelp()");
    expect(content).toContain('arg === "-h" || arg === "--help"');
    expect(content).toContain("--app-url <url>");
    expect(content).toContain("function nodeSmokeArgs(args, options)");
    expect(content).toContain('"--app-url", options.appUrl');
    expect(content).toContain("args: resolvedArgs");
  });

  it("prepares packaged renderer and Electron host assets before real Electron fixtures", () => {
    const content = readSmokeScript();

    expect(content).toContain("function ensureElectronFixtureBuild(options)");
    expect(content).toContain("../lib/electron-fixture-build.mjs");
    expect(content).toContain("ensurePackagedElectronFixtureBuild");
    expect(content).toContain(
      "function runElectronFixtureSmoke(label, args, options)",
    );
    expect(content).toContain('path.join(rootDir, "dist", "index.html")');
    expect(content).toContain(
      'path.join(rootDir, "dist-electron", "main", "main.js")',
    );
    expect(content).toContain(
      'path.join(rootDir, "dist-electron", "preload", "preload.cjs")',
    );
    expect(content).toContain(
      'path.join(rootDir, "dist-electron", "app-server.release.json")',
    );
    expect(content).toContain("electronAppServerBinaryDestination");
    expect(content).toContain('"build:renderer:electron:smoke"');
    expect(content).toContain("reusing fresh packaged fixture assets");
    expect(content).toContain("fresh-artifacts");
    expect(content).toContain("stale-source");
    expect(content).toContain("rebuilding stale packaged fixture assets");
    expect(content).toContain("ensureElectronFixtureBuild(options)");
    expect(content).toContain("LIME_ELECTRON_FIXTURE_BUILD_READY");
    expect(content.indexOf("ensureElectronFixtureBuild(options)")).toBeLessThan(
      content.indexOf("Coding Workbench Electron fixture"),
    );
    expect(content).toContain(
      "Claw Expert Plaza Skills Runtime click-through Electron fixture",
    );
    expect(content).not.toContain(
      "Claw Inputbar pending steer multi queue order Electron fixture",
    );
    expect(content).not.toContain('"inputbar-pending-steer-rich-restore"');
    expect(content).not.toContain('"inputbar-pending-steer-multi-queue"');
    expect(content).not.toContain(
      "Claw Inputbar pending steer pop-front resume hydrate Electron fixture",
    );
    expect(content).not.toContain('"inputbar-pending-steer-pop-front-resume"');
    for (const filePath of [
      "scripts/agent-runtime/claw-chat-current-fixture-inputbar-pending-steer.mjs",
      "scripts/agent-runtime/claw-chat-current-fixture-pending-steer-gui-actions.mjs",
      "scripts/agent-runtime/claw-chat-current-fixture-pending-steer-read-model.mjs",
      "scripts/agent-runtime/claw-chat-current-fixture-pending-steer-assertions.mjs",
    ]) {
      expect(fs.existsSync(filePath)).toBe(false);
    }
  });

  it("does not opt into live provider or mock backend evidence", () => {
    const content = readSmokeScript();

    expect(content).toContain('LIME_ALLOW_LIVE_PROVIDER_SMOKE: "0"');
    expect(content).toContain('LIME_REAL_API_TEST: "0"');
    expect(content).toContain("liveProviderUsed=false");
    expect(content).not.toContain("--allow-live-provider");
    expect(content).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
  });
});
