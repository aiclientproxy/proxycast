#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import {
  electronFixtureBuildReadyEnv,
  ensureElectronFixtureBuild as ensurePackagedElectronFixtureBuild,
} from "../lib/electron-fixture-build.mjs";
import { runVitestSmoke } from "../lib/vitest-smoke-runner.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "../..");
const LOG_PREFIX = "smoke:agent-runtime-current-fixture";

function printHelp() {
  console.log(`
Agent Runtime Current Fixture Regression Smoke

用途:
  聚合 current Agent Runtime / Claw GUI fixture 回归，验证 GUI 输入框、Electron
  Desktop Host bridge、App Server JSON-RPC 与 fixture backend 的 current 主链。

用法:
  node scripts/agent-runtime/current-fixture-regression-smoke.mjs [选项]

选项:
  --app-url <url>  可选 renderer dev server，例如 http://127.0.0.1:1420/
  -h, --help       显示帮助
`);
}

function parseArgs(argv) {
  const options = {
    appUrl: "",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "-h" || arg === "--help") {
      printHelp();
      process.exit(0);
    }
    if (arg === "--app-url" && next) {
      options.appUrl = next.trim();
      index += 1;
      continue;
    }
    throw new Error(`未知参数: ${arg}`);
  }
  return options;
}

function nodeSmokeArgs(args, options) {
  if (!options.appUrl) {
    return args;
  }
  return [args[0], "--app-url", options.appUrl, ...args.slice(1)];
}

function runVitest(label, args) {
  return runVitestSmoke({
    rootDir,
    label,
    args,
    logPrefix: LOG_PREFIX,
    env: {
      LIME_ALLOW_LIVE_PROVIDER_SMOKE: "0",
      LIME_REAL_API_TEST: "0",
    },
  });
}

function runNodeSmoke(label, args, options) {
  const startedAt = Date.now();
  const resolvedArgs = nodeSmokeArgs(args, options);
  const buildReadyEnv = electronFixtureBuildReadyEnv();

  console.log(`\n[${LOG_PREFIX}] > ${label}`);
  const result = spawnSync(process.execPath, resolvedArgs, {
    cwd: rootDir,
    stdio: "inherit",
    env: {
      ...process.env,
      LIME_ALLOW_LIVE_PROVIDER_SMOKE: "0",
      LIME_REAL_API_TEST: "0",
      ...(options.appUrl ? {} : { [buildReadyEnv]: "1" }),
    },
  });

  if (result.error) {
    throw result.error;
  }

  if (typeof result.status === "number" && result.status !== 0) {
    const error = new Error(`[${LOG_PREFIX}] ${label} 失败`);
    error.exitCode = result.status;
    throw error;
  }

  return {
    label,
    status: "pass",
    durationMs: Date.now() - startedAt,
    args: resolvedArgs,
  };
}

function runElectronFixtureSmoke(label, args, options) {
  ensureElectronFixtureBuild(options);
  return runNodeSmoke(label, args, options);
}

function ensureElectronFixtureBuild(options) {
  return ensurePackagedElectronFixtureBuild({
    appUrl: options.appUrl,
    logPrefix: LOG_PREFIX,
    rootDir,
  });
}

function main() {
  const options = parseArgs(process.argv.slice(2));

  runVitest("Agent 历史/缓存终态恢复", [
    "src/components/agent/chat/hooks/agentChatHistory.test.ts",
    "src/components/agent/chat/hooks/agentSessionScopedStorage.test.ts",
  ]);

  runVitest("Agent 流式完成与运行态收尾", [
    "src/components/agent/chat/hooks/agentStreamCompletionController.test.ts",
    "src/components/agent/chat/hooks/agentStreamRuntimeHandler.unit.test.ts",
    "src/components/agent/chat/hooks/agentStreamRuntimeHandler.typedError.test.ts",
    "-t",
    "turn_completed|usage|工具",
  ]);

  runVitest("Claw 消息列表终态 UI", [
    "src/components/agent/chat/components/MessageList.test.tsx",
    "-t",
    "远端 failed runtimeStatus|完成态 assistant 有正文|assistant 已有正文且仍在发送时",
  ]);

  runVitest("Electron/App Server fixture smoke guard", [
    "scripts/electron/session-history-fixture-smoke.test.mjs",
    "scripts/electron/code-artifact-workbench-fixture-smoke.test.mjs",
    "scripts/agent-runtime/claw-chat-current-fixture-smoke.test.mjs",
  ]);

  ensureElectronFixtureBuild(options);

  runElectronFixtureSmoke(
    "Claw 首页首发热路径 Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "home-hotpath",
      "--prefix",
      "claw-chat-current-fixture-home-hotpath-regression",
      "--timeout-ms",
      "240000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw 首页首发短问候热路径 Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "home-hotpath-greeting",
      "--prefix",
      "claw-chat-current-fixture-home-hotpath-greeting-regression",
      "--timeout-ms",
      "240000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Coding Workbench Electron fixture",
    [
      "scripts/electron/code-artifact-workbench-fixture-smoke.mjs",
      "--scenario",
      "gui-coding-input",
      "--prefix",
      "code-artifact-workbench-gui-coding-input-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw 图片命令 GUI Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "image-command",
      "--prefix",
      "claw-chat-current-fixture-image-command-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw 普通画图意图 GUI Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "plain-image-intent",
      "--prefix",
      "claw-chat-current-fixture-plain-image-intent-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw 停止后同会话继续输出 Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "cancel-then-continue",
      "--prefix",
      "claw-chat-current-fixture-cancel-then-continue-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw approval allow-for-session resume Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "approval-request-resume",
      "--prefix",
      "claw-chat-current-fixture-approval-request-resume-regression",
      "--timeout-ms",
      "240000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw approval decline-continue Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "approval-request-decline",
      "--prefix",
      "claw-chat-current-fixture-approval-request-decline-regression",
      "--timeout-ms",
      "240000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw approval cancel-turn Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "approval-request-cancel",
      "--prefix",
      "claw-chat-current-fixture-approval-request-cancel-regression",
      "--timeout-ms",
      "240000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw approval full-access no prompt Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "approval-request-full-access",
      "--prefix",
      "claw-chat-current-fixture-approval-request-full-access-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw Inputbar rich draft restore output-free cancel Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "inputbar-rich-restore",
      "--prefix",
      "claw-chat-current-fixture-inputbar-rich-restore-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw active turn steer same identity Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "inputbar-active-steer",
      "--prefix",
      "claw-chat-current-fixture-active-steer-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw Plan revisioned history hydrate Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "plan",
      "--prefix",
      "claw-chat-current-fixture-plan-history-hydrate-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw Skills Runtime natural + explicit $skill + Skills workspace try Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "skills-runtime",
      "--prefix",
      "claw-chat-current-fixture-skills-runtime-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw MCP structuredContent Agent Chat GUI Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "mcp-structured-content",
      "--prefix",
      "claw-chat-current-fixture-mcp-structured-content-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw media item reference Agent Chat GUI Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "media-reference",
      "--prefix",
      "claw-chat-current-fixture-media-reference-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw Expert Plaza Skills Runtime click-through Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "expert-plaza-skills-runtime",
      "--prefix",
      "claw-chat-current-fixture-expert-plaza-skills-runtime-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw Expert Panel Skills Runtime override Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "expert-panel-skills-runtime",
      "--prefix",
      "claw-chat-current-fixture-expert-panel-skills-runtime-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw typed error retry success Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "typed-error-retry-success",
      "--prefix",
      "claw-chat-current-fixture-typed-error-retry-success-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Claw typed error retry failure Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "typed-error-retry-failure",
      "--prefix",
      "claw-chat-current-fixture-typed-error-retry-failure-regression",
      "--timeout-ms",
      "180000",
    ],
    options,
  );

  runElectronFixtureSmoke(
    "Content Factory article Article Editor Electron fixture",
    [
      "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
      "--scenario",
      "content-factory-article-workspace",
      "--prefix",
      "claw-chat-current-fixture-content-factory-article-workspace-regression",
      "--timeout-ms",
      "240000",
    ],
    options,
  );

  console.log(
    `[${LOG_PREFIX}] summary: current Agent Runtime fixture regression 已覆盖 history/cache hydration、turn_completed 工具收尾、typed error retry success/failure、failed read model、Claw 终态 UI、Electron fixture guard、首页首发热路径用户消息可见/无空对话/无模型回退 Electron fixture、真实 GUI coding 输入到 Coding Workbench Electron fixture、真实 GUI 图片命令到 Claw Chat Electron fixture、普通自然语言画图意图到同一图片 task 主链 Electron fixture、Claw GUI current fixture guard、停止后同会话继续输出 Electron fixture、approval allow-for-session resume / decline continue / cancel turn 三类 Electron fixture、approval full-access 不弹 prompt / 不生成 timeline record Electron fixture、Inputbar rich draft 在 output-free cancel 后恢复 text/image/path/skill Electron fixture、active turn 经 thread/read -> turn/steer 保持同一 Turn identity Electron fixture、Plan revisioned thread item + history hydrate Electron fixture、Skills Runtime natural + 显式 $skill + 技能中心试用入口三入口按需加载 Electron fixture、MCP structuredContent 到 Agent Chat GUI 可见 Electron fixture、media item / imageView 引用到 Agent Chat 卡片与 Workbench source 预览 Electron fixture、Expert Plaza 点击专家卡片进入同一 Skills Runtime 闭环 Electron fixture、ExpertInfoPanel 调整 skillRefs 后下一轮继承同一 Skills Runtime 闭环并展示 Evidence Pack 复盘 Electron fixture、内容工厂文章 Article Editor / articleDraft 右侧产物闭环 Electron fixture；liveProviderUsed=false`,
  );
  console.log(`\n[${LOG_PREFIX}] 通过`);
}

main();
