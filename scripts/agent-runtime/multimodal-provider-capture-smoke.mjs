#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { createAppServerStdioTransport } from "../harness/app-server-stdio-transport.mjs";
import { startOpenAiCompatibleFixtureServer } from "../lib/openai-compatible-fixture-server.mjs";

const PNG_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";
const TERMINAL_STATUSES = new Set([
  "completed",
  "failed",
  "canceled",
  "cancelled",
]);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "../..");

function parseArgs(argv) {
  const options = {
    allowLiveProvider: false,
    appServerBin:
      process.env.LIME_MULTIMODAL_APP_SERVER_BIN ||
      path.join(repoRoot, "lime-rs/target/debug/app-server"),
    appServerDataDir:
      process.env.LIME_MULTIMODAL_APP_SERVER_DATA_DIR ||
      path.join(os.homedir(), "Library/Application Support/lime/app-server"),
    intervalMs: 100,
    imagePath: "",
    modelPreference: "",
    providerPreference: "",
    timeoutMs: 60_000,
    logPrefix: "[smoke:agent-runtime-multimodal-capture]",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const value = argv[index + 1];
    if (arg === "--allow-live-provider") {
      options.allowLiveProvider = true;
      continue;
    }
    if (arg === "--app-server-bin" && value) {
      options.appServerBin = path.resolve(value);
      index += 1;
      continue;
    }
    if (arg === "--app-server-data-dir" && value) {
      options.appServerDataDir = path.resolve(value);
      index += 1;
      continue;
    }
    if (arg === "--timeout-ms" && value) {
      options.timeoutMs = Number(value);
      index += 1;
      continue;
    }
    if (arg === "--image" && value) {
      options.imagePath = path.resolve(value);
      index += 1;
      continue;
    }
    if (arg === "--model" && value) {
      options.modelPreference = value.trim();
      index += 1;
      continue;
    }
    if (arg === "--provider" && value) {
      options.providerPreference = value.trim();
      index += 1;
      continue;
    }
    throw new Error(`unknown argument: ${arg}`);
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 10_000) {
    throw new Error("--timeout-ms must be >= 10000");
  }
  if (
    options.allowLiveProvider &&
    (!options.providerPreference ||
      !options.modelPreference ||
      !options.imagePath)
  ) {
    throw new Error(
      "live provider mode requires --provider, --model, and --image",
    );
  }
  return options;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function imageUrls(value, output = []) {
  if (Array.isArray(value)) {
    for (const item of value) imageUrls(item, output);
    return output;
  }
  if (!value || typeof value !== "object") return output;
  if (typeof value.image_url === "string") output.push(value.image_url);
  if (typeof value.image_url?.url === "string")
    output.push(value.image_url.url);
  for (const child of Object.values(value)) imageUrls(child, output);
  return output;
}

function stringValues(value, output = []) {
  if (Array.isArray(value)) {
    for (const item of value) stringValues(item, output);
    return output;
  }
  if (!value || typeof value !== "object") {
    if (typeof value === "string" && value.trim()) output.push(value.trim());
    return output;
  }
  for (const child of Object.values(value)) stringValues(child, output);
  return output;
}

function imageDataUrl(imagePath) {
  const extension = path.extname(imagePath).toLowerCase();
  const mediaType =
    extension === ".jpg" || extension === ".jpeg"
      ? "image/jpeg"
      : extension === ".webp"
        ? "image/webp"
        : "image/png";
  return {
    dataUrl: `data:${mediaType};base64,${fs.readFileSync(imagePath).toString("base64")}`,
    mediaType,
  };
}

function turnFromRead(read, turnId) {
  const turns = Array.isArray(read?.thread?.turns) ? read.thread.turns : [];
  return turns.find((turn) => turn?.id === turnId) || null;
}

async function waitForTerminal(transport, options, threadId, turnId) {
  const startedAt = Date.now();
  let latestRead = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    latestRead = await transport.invoke(options, "thread/read", {
      threadId,
      includeTurns: true,
    });
    const turn = turnFromRead(latestRead, turnId);
    const status = String(turn?.status || "").toLowerCase();
    if (TERMINAL_STATUSES.has(status)) return { read: latestRead, status };
    await new Promise((resolve) => setTimeout(resolve, options.intervalMs));
  }
  let cancelStatus = "not_requested";
  try {
    await transport.invoke(options, "turn/interrupt", {
      threadId,
      turnId,
    });
    cancelStatus = "requested";
    const cancelDeadline = Date.now() + 10_000;
    while (Date.now() < cancelDeadline) {
      latestRead = await transport.invoke(options, "thread/read", {
        threadId,
        includeTurns: true,
      });
      const status = String(
        turnFromRead(latestRead, turnId)?.status || "",
      ).toLowerCase();
      if (TERMINAL_STATUSES.has(status)) {
        cancelStatus = status;
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, options.intervalMs));
    }
  } catch (error) {
    cancelStatus = `failed:${
      error instanceof Error ? error.message : String(error)
    }`;
  }
  const actualStatus = String(
    turnFromRead(latestRead, turnId)?.status || "missing",
  ).toLowerCase();
  throw new Error(
    `multimodal turn timeout: thread=${threadId} turn=${turnId} expected=terminal actual=${actualStatus} timeoutMs=${options.timeoutMs} cancelStatus=${cancelStatus}`,
  );
}

function providerWithImageInput(provider) {
  const modelCapabilities = provider.providerConfig.modelCapabilities || {};
  return {
    ...provider,
    providerConfig: {
      ...provider.providerConfig,
      modelCapabilities: {
        ...modelCapabilities,
        capabilities: {
          ...(modelCapabilities.capabilities || {}),
          vision: true,
        },
        taskFamilies: ["chat", "vision_understanding"],
        inputModalities: ["text", "image"],
      },
    },
  };
}

async function registerFixtureProvider(transport, options, fixture, suffix) {
  const descriptor = providerWithImageInput(fixture.provider);
  const created = await transport.invoke(options, "modelProvider/create", {
    name: `Multimodal capture ${suffix}`,
    providerType: "openai",
    apiHost: descriptor.providerConfig.baseUrl,
  });
  const providerId = String(created?.provider?.id || "").trim();
  assert(providerId, "modelProvider/create did not return provider.id");

  await transport.invoke(options, "modelProvider/update", {
    providerId,
    enabled: true,
    sortOrder: 0,
    models: [
      {
        id: descriptor.modelPreference,
        capability: descriptor.providerConfig.modelCapabilities,
      },
    ],
  });
  await transport.invoke(options, "modelProviderKey/create", {
    providerId,
    apiKey: descriptor.providerConfig.apiKey,
    alias: "multimodal-capture-fixture",
    replaceExisting: true,
  });

  return {
    modelPreference: descriptor.modelPreference,
    providerPreference: providerId,
    providerName: descriptor.providerName,
  };
}

async function assertCatalogRoute(transport, options, provider) {
  const catalog = await transport.invoke(options, "model/list", {
    includeHidden: true,
    limit: 500,
  });
  const models = Array.isArray(catalog?.data) ? catalog.data : [];
  const selected = models.find(
    (model) =>
      model?.providerId === provider.providerPreference &&
      model?.model === provider.modelPreference,
  );
  assert(
    selected,
    `model/list did not expose provider=${provider.providerPreference} model=${provider.modelPreference}`,
  );
  assert(
    Array.isArray(selected.inputModalities) &&
      selected.inputModalities.includes("image"),
    "model/list did not expose image input capability",
  );
  return selected;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const workspaceRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), "lime-multimodal-capture-"),
  );
  let fixture = null;
  let transport = null;
  try {
    const liveProviderUsed = options.allowLiveProvider;
    const scenarioId = liveProviderUsed ? "LIV-03" : "PRV-04/ITM-05";
    if (!liveProviderUsed) {
      fixture = await startOpenAiCompatibleFixtureServer({
        content: "MULTIMODAL_CAPTURE_OK",
      });
    }
    let provider = liveProviderUsed
      ? {
          providerPreference: options.providerPreference,
          providerName: options.providerPreference,
          modelPreference: options.modelPreference,
        }
      : null;
    const image = liveProviderUsed
      ? imageDataUrl(options.imagePath)
      : { dataUrl: PNG_DATA_URL, mediaType: "image/png" };
    const prompt = liveProviderUsed
      ? "Inspect the attached image directly. Name only the fruit-like object and its dominant color in one short sentence. Do not call tools, transcribe text, infer missing details, or report anything that is not visible."
      : "Describe the attached image in one sentence.";
    transport = await createAppServerStdioTransport({
      repoRoot,
      binaryPath: options.appServerBin,
      dataDir: options.appServerDataDir,
      timeoutMs: options.timeoutMs,
      logPrefix: options.logPrefix,
    });
    await transport.waitForReady();

    const workspaceResponse = await transport.invoke(
      options,
      "workspace/ensure",
      {
        name: "Multimodal provider capture",
        rootPath: workspaceRoot,
        workspaceType: "temporary",
      },
    );
    const workspaceId = String(
      workspaceResponse?.workspace?.id || workspaceResponse?.workspaceId || "",
    ).trim();
    assert(workspaceId, "workspace/ensure did not return workspace id");

    const suffix = `${Date.now()}-${process.pid}`;
    if (!provider) {
      provider = await registerFixtureProvider(
        transport,
        options,
        fixture,
        suffix,
      );
    }
    const catalogModel = await assertCatalogRoute(transport, options, provider);
    const started = await transport.invoke(options, "thread/start", {
      cwd: workspaceRoot,
      historyMode: "paginated",
      model: provider.modelPreference,
      modelProvider: provider.providerPreference,
      runtimeWorkspaceRoots: [workspaceRoot],
      serviceName: "Multimodal provider capture",
      threadSource: "appServer",
    });
    const threadId = String(started?.thread?.id || "").trim();
    const sessionId = String(started?.thread?.sessionId || "").trim();
    assert(
      threadId && sessionId,
      "thread/start did not return canonical identity",
    );

    const turnResponse = await transport.invoke(options, "turn/start", {
      threadId,
      clientUserMessageId: `multimodal-capture-${suffix}`,
      input: [
        { type: "text", text: prompt },
        { type: "image", url: image.dataUrl, detail: "auto" },
      ],
      cwd: workspaceRoot,
      runtimeWorkspaceRoots: [workspaceRoot],
      approvalPolicy: "never",
      sandboxPolicy: "danger-full-access",
      responsesapiClientMetadata: {
        source: "smoke:agent-runtime-multimodal-capture",
        scenarioId,
      },
    });
    const turnId = String(turnResponse?.turn?.id || "").trim();
    assert(turnId, "turn/start did not return canonical turn.id");
    const terminal = await waitForTerminal(
      transport,
      options,
      threadId,
      turnId,
    );
    const evidence = await transport.invoke(options, "evidence/export", {
      sessionId,
      turnId,
      includeEvents: true,
      includeArtifacts: false,
      includeEvidencePack: false,
    });

    assert(
      terminal.status === "completed",
      `turn terminal status=${terminal.status}`,
    );
    const readText = JSON.stringify(terminal.read);
    const evidenceText = JSON.stringify(evidence);
    assert(
      !readText.includes("base64,"),
      "thread/read leaked inline image payload",
    );
    assert(
      !evidenceText.includes("base64,"),
      "evidence/export leaked inline image payload",
    );
    assert(
      readText.includes("sidecar://"),
      "thread/read did not retain canonical sidecar reference",
    );
    let providerRequestPath = null;
    let providerImagePayloadObserved = null;
    let liveVisionAnswerObserved = null;
    let providerToolCount = null;
    let providerMaxOutputTokens = null;
    let providerThinkingEnabled = null;
    let historyImagePayloadObserved = null;
    let followUpTurnId = null;
    if (liveProviderUsed) {
      const visibleText = stringValues(terminal.read).join("\n");
      const normalized = visibleText.toLowerCase().replaceAll(/\s+/g, " ");
      liveVisionAnswerObserved =
        normalized.includes("apple") && normalized.includes("red");
      assert(
        liveVisionAnswerObserved,
        `Agnes response did not identify the visible object and color: ${visibleText.slice(-2000)}`,
      );
    } else {
      const followUp = await transport.invoke(options, "turn/start", {
        threadId,
        clientUserMessageId: `multimodal-follow-up-${suffix}`,
        input: [
          {
            type: "text",
            text: "Use the prior image context and answer with one short sentence.",
          },
        ],
        cwd: workspaceRoot,
        runtimeWorkspaceRoots: [workspaceRoot],
        approvalPolicy: "never",
        sandboxPolicy: "danger-full-access",
        responsesapiClientMetadata: {
          source: "smoke:agent-runtime-multimodal-capture",
          scenarioId: `${scenarioId}-history`,
        },
      });
      followUpTurnId = String(followUp?.turn?.id || "").trim();
      assert(followUpTurnId, "follow-up turn/start did not return turn.id");
      const followUpTerminal = await waitForTerminal(
        transport,
        options,
        threadId,
        followUpTurnId,
      );
      assert(
        followUpTerminal.status === "completed",
        `follow-up turn terminal status=${followUpTerminal.status}`,
      );
      assert(
        !JSON.stringify(followUpTerminal.read).includes("base64,"),
        "follow-up thread/read leaked hydrated history image payload",
      );
      assert(
        fixture.requests.length === 2,
        `provider request count=${fixture.requests.length}`,
      );
      const urls = imageUrls(fixture.requests[0]?.body);
      const historyUrls = imageUrls(fixture.requests[1]?.body);
      providerImagePayloadObserved = urls.includes(PNG_DATA_URL);
      historyImagePayloadObserved = historyUrls.includes(PNG_DATA_URL);
      providerRequestPath = fixture.requests[0]?.path || null;
      providerToolCount = Array.isArray(fixture.requests[0]?.body?.tools)
        ? fixture.requests[0].body.tools.length
        : 0;
      providerMaxOutputTokens = fixture.requests[0]?.body?.max_tokens ?? null;
      providerThinkingEnabled =
        fixture.requests[0]?.body?.chat_template_kwargs?.enable_thinking ??
        null;
      assert(
        providerImagePayloadObserved,
        "provider wire request did not contain hydrated image data",
      );
      assert(
        historyImagePayloadObserved,
        "follow-up provider request did not hydrate the canonical history image",
      );
    }

    console.log(
      JSON.stringify(
        {
          status: "passed",
          scenarioId,
          evidenceLevel: liveProviderUsed
            ? "App Server integration + live provider"
            : "App Server integration",
          liveProviderUsed,
          provider: provider.providerPreference,
          model: provider.modelPreference,
          catalogModelId: catalogModel.id,
          threadId,
          sessionId,
          turnId,
          providerRequestPath,
          providerImagePayloadObserved,
          providerToolCount,
          providerMaxOutputTokens,
          providerThinkingEnabled,
          historyImagePayloadObserved,
          followUpTurnId,
          liveVisionAnswerObserved,
          canonicalSidecarReferenceObserved: true,
          readModelInlinePayloadAbsent: true,
          evidenceInlinePayloadAbsent: true,
          terminalStatus: terminal.status,
        },
        null,
        2,
      ),
    );
  } finally {
    await transport?.close();
    await fixture?.close();
    fs.rmSync(workspaceRoot, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(
    `[smoke:agent-runtime-multimodal-capture] failed: ${
      error instanceof Error ? error.stack || error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
