#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import {
  assertLiveProviderSmokeAllowed,
  liveProviderSmokeAllowed,
} from "./lib/live-provider-smoke-gate.mjs";

const DEFAULTS = {
  invokeUrl: "http://127.0.0.1:3030/invoke",
  packName: "provider-e2e-persona",
  output: null,
};
const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const APP_SERVER_METHOD_MODEL_PROVIDER_LIST = "modelProvider/list";

const DEFAULT_SOURCE_TEXT = [
  "# 访谈摘录：林澈",
  "",
  "林澈是一个面向中小品牌的内容增长顾问，过去 8 年主要帮助茶饮、文旅和本地生活团队搭建内容栏目、私域转化和直播复盘机制。",
  "",
  "他说自己最常被客户找来解决的问题不是“写一篇爆款”，而是“让团队每周都能稳定产出能转化的内容”。他反复强调：内容不是灵感管理，而是经营节奏管理。",
  "",
  "代表案例：2025 年帮助一家地方茶饮品牌从零搭建小红书和视频号内容日历。三个月内完成 12 个栏目测试，沉淀 4 个可复用栏目，其中“门店真实一天”和“顾客点单理由”成为固定栏目。团队从临时追热点改为每周选题会 + 每日素材池。",
  "",
  "表达风格：直接、克制、偏实战，不喜欢“颠覆”“闭环神话”“私域收割”这类词。他常说：“内容的第一目标不是热闹，是让用户知道你为什么值得信任。”",
  "",
  "边界：不能声称服务过上市公司；不能夸大 GMV；不能把客户数据写成公开案例。",
].join("\n");

function printHelp() {
  console.log(`Usage: node scripts/knowledge-provider-e2e.mjs [options]

真实 Provider E2E，验证 app_server_handle_json_lines -> knowledgePack/compile -> Builder Skill -> documents/ -> compiled/splits -> persona context。
默认不会调用外部模型；必须显式传 --allow-live-provider 或 --allow-external-provider。

Options:
  --invoke-url <url>             DevBridge invoke URL，默认 ${DEFAULTS.invokeUrl}
  --provider <id>                API Key Provider id，例如 custom-xxx 或 lime-hub
  --model <model>                模型 id
  --working-dir <dir>            E2E workspace；默认创建临时目录
  --pack-name <name>             pack name，默认 ${DEFAULTS.packName}
  --source-file <path>           自定义 Markdown 来源；默认使用内置非敏感示例
  --output <path>                写出 sanitized evidence JSON
  --list-providers               只列出 provider 摘要，不调用模型
  --allow-live-provider          明确允许外部模型调用、资料外发与费用消耗
  --allow-external-provider      明确允许外部模型调用、资料外发与费用消耗
  --help, -h                     显示帮助
`);
}

function parseArgs(argv) {
  const options = {
    ...DEFAULTS,
    allowExternalProvider: false,
    allowLiveProvider: liveProviderSmokeAllowed(),
    listProviders: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      options.help = true;
      continue;
    }
    if (arg === "--allow-external-provider") {
      options.allowExternalProvider = true;
      options.allowLiveProvider = true;
      continue;
    }
    if (arg === "--allow-live-provider") {
      options.allowLiveProvider = true;
      continue;
    }
    if (arg === "--list-providers") {
      options.listProviders = true;
      continue;
    }
    if (
      [
        "--invoke-url",
        "--provider",
        "--model",
        "--working-dir",
        "--pack-name",
        "--source-file",
        "--output",
      ].includes(arg)
    ) {
      const value = argv[index + 1];
      if (!value) {
        throw new Error(`${arg} 需要参数`);
      }
      const key = arg
        .slice(2)
        .replace(/-([a-z])/g, (_, char) => char.toUpperCase());
      options[key] = value;
      index += 1;
      continue;
    }
    throw new Error(`未知参数：${arg}`);
  }

  return options;
}

async function invoke(invokeUrl, cmd, args = {}) {
  const response = await fetch(invokeUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ cmd, args }),
  });
  const text = await response.text();
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    throw new Error(`DevBridge 非 JSON 响应：${text.slice(0, 400)}`);
  }
  if (payload.error) {
    throw new Error(payload.error);
  }
  return payload.result;
}

let appServerRequestId = 1;

async function invokeAppServerMethod(invokeUrl, method, params = {}) {
  const id = `knowledge-provider-${appServerRequestId++}`;
  const result = await invoke(invokeUrl, APP_SERVER_HANDLE_JSON_LINES_COMMAND, {
    request: { lines: [`${JSON.stringify({ id, method, params })}\n`] },
  });
  const messages = (Array.isArray(result?.lines) ? result.lines : [])
    .map((line) => {
      try {
        return JSON.parse(String(line));
      } catch {
        return null;
      }
    })
    .filter(Boolean);
  const error = messages.find((message) => message.id === id && message.error);
  if (error) {
    throw new Error(
      `${method} error: ${error.error?.message || "App Server JSON-RPC error"}`,
    );
  }
  const response = messages.find(
    (message) => message.id === id && Object.hasOwn(message, "result"),
  );
  if (!response) {
    throw new Error(`${method} missing App Server response`);
  }
  return response.result;
}

function sanitizeProviders(providers) {
  return providers.map((provider) => ({
    id: provider.id,
    name: provider.name,
    type: provider.type,
    enabled: provider.enabled,
    api_key_count: provider.api_key_count,
    models: Array.isArray(provider.models)
      ? provider.models.slice(0, 8)
      : provider.models,
  }));
}

function loadSourceText(sourceFile) {
  if (!sourceFile) {
    return DEFAULT_SOURCE_TEXT;
  }
  return fs.readFileSync(path.resolve(sourceFile), "utf8");
}

function ensureProviderOptions(options) {
  if (!options.provider || !options.model) {
    throw new Error(
      "真实 Provider E2E 需要同时传 --provider 和 --model；可先用 --list-providers 查看摘要",
    );
  }
  assertLiveProviderSmokeAllowed({
    allowed: options.allowLiveProvider,
    scriptName: "scripts/knowledge-provider-e2e.mjs",
    flag: "--allow-live-provider",
  });
}

function evidenceOutputPath(options, workingDir) {
  if (options.output) {
    return path.resolve(options.output);
  }
  return path.join(workingDir, "provider-e2e-result.json");
}

function readJsonIfExists(filePath) {
  if (!fs.existsSync(filePath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

async function runProviderE2E(options) {
  ensureProviderOptions(options);

  const workingDir = options.workingDir
    ? path.resolve(options.workingDir)
    : fs.mkdtempSync(path.join(os.tmpdir(), "lime-knowledge-provider-e2e-"));
  fs.mkdirSync(workingDir, { recursive: true });

  const packName = options.packName;
  const sourceText = loadSourceText(options.sourceFile);
  const sessionId = `knowledge-provider-e2e-${Date.now()}`;

  await invokeAppServerMethod(
    options.invokeUrl,
    "knowledgePack/source/import",
    {
      workingDir,
      packName,
      description: "真实 Provider E2E：内容增长顾问个人 IP 知识包",
      packType: "personal-profile",
      language: "zh-CN",
      sourceFileName: "provider-e2e-source.md",
      sourceText,
    },
  );

  let compileResult;
  try {
    compileResult = await invokeAppServerMethod(
      options.invokeUrl,
      "knowledgePack/compile",
      {
        workingDir,
        name: packName,
        builderRuntime: {
          enabled: true,
          providerOverride: options.provider,
          modelOverride: options.model,
          sessionId,
        },
      },
    );
  } catch (error) {
    throw new Error(
      `Knowledge Provider E2E 尚未迁入 App Server current builder runtime binding: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }

  await invokeAppServerMethod(
    options.invokeUrl,
    "knowledgePack/status/update",
    {
      workingDir,
      name: packName,
      status: "ready",
    },
  );

  const detail = await invokeAppServerMethod(
    options.invokeUrl,
    "knowledgePack/read",
    {
      workingDir,
      name: packName,
    },
  );

  const context = await invokeAppServerMethod(
    options.invokeUrl,
    "knowledgeContext/resolve",
    {
      workingDir,
      name: packName,
      task: "根据个人 IP 知识库写一段不夸大的视频号简介",
      maxChars: 8000,
      activation: "explicit",
      writeRun: true,
      runReason: "provider-e2e-script",
    },
  );

  const runRecord = JSON.parse(
    fs.readFileSync(compileResult.run.absolutePath, "utf8"),
  );
  const docPath = detail.documents?.[0]?.absolutePath;
  const doc =
    docPath && fs.existsSync(docPath) ? fs.readFileSync(docPath, "utf8") : "";
  const indexPath = path.join(
    workingDir,
    ".lime",
    "knowledge",
    "packs",
    packName,
    "compiled",
    "index.json",
  );
  const compiledIndex = readJsonIfExists(indexPath);

  const evidence = {
    workingDir,
    packName,
    provider: {
      id: options.provider,
      model: options.model,
    },
    compile: {
      selectedSourceCount: compileResult.selectedSourceCount,
      runRelativePath: compileResult.run.relativePath,
      warnings: compileResult.warnings,
    },
    runtimeBinding: runRecord.builder_skill?.runtime_binding ?? null,
    artifacts: {
      documents: (detail.documents ?? []).map((file) => file.relativePath),
      compiled: (detail.compiled ?? []).map((file) => file.relativePath),
      runs: (detail.runs ?? []).map((file) => file.relativePath),
      compiledIndexExists: Boolean(compiledIndex),
      splitCount: compiledIndex?.splits?.length ?? 0,
    },
    context: {
      status: context.status,
      runPath: context.runPath,
      tokenEstimate: context.tokenEstimate,
      selectedFiles: context.selectedFiles,
      warningCount: context.warnings.length,
      fencedPrefix: context.fencedContext.slice(0, 180),
    },
    documentShape: {
      startsWithMarkdownHeading: doc.trimStart().startsWith("# "),
      startsWithJsonFence: doc.trimStart().startsWith("```json"),
      containsBoundary:
        doc.includes("上市公司") || doc.includes("GMV") || doc.includes("不得"),
      length: [...doc].length,
      preview: doc.slice(0, 600),
    },
  };

  const outputPath = evidenceOutputPath(options, workingDir);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(evidence, null, 2)}\n`);
  return { outputPath, evidence };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  if (options.listProviders) {
    const providerList = await invokeAppServerMethod(
      options.invokeUrl,
      APP_SERVER_METHOD_MODEL_PROVIDER_LIST,
    );
    const providers = providerList?.providers;
    console.log(JSON.stringify(sanitizeProviders(providers), null, 2));
    return;
  }

  const { outputPath, evidence } = await runProviderE2E(options);
  console.log(JSON.stringify({ outputPath, evidence }, null, 2));
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
