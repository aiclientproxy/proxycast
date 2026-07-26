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
  const turns = Array.isArray(read?.turns)
    ? read.turns
    : Array.isArray(read?.detail?.thread_read?.turns)
      ? read.detail.thread_read.turns
      : [];
  return (
    turns.find((turn) => (turn?.turnId || turn?.turn_id) === turnId) || null
  );
}

async function waitForTerminal(
  transport,
  options,
  sessionId,
  turnId,
  turnPromise,
) {
  const startedAt = Date.now();
  let latestRead = null;
  let startError = null;
  void turnPromise.catch((error) => {
    startError = error;
  });
  while (Date.now() - startedAt < options.timeoutMs) {
    latestRead = await transport.invoke(options, "thread/read", {
      sessionId,
      historyLimit: 100,
    });
    const turn = turnFromRead(latestRead, turnId);
    const status = String(turn?.status || "").toLowerCase();
    if (TERMINAL_STATUSES.has(status)) return { read: latestRead, status };
    if (startError) throw startError;
    await new Promise((resolve) => setTimeout(resolve, options.intervalMs));
  }
  let cancelStatus = "not_requested";
  try {
    await transport.invoke(options, "turn/interrupt", {
      sessionId,
      turnId,
    });
    cancelStatus = "requested";
    const cancelDeadline = Date.now() + 10_000;
    while (Date.now() < cancelDeadline) {
      latestRead = await transport.invoke(options, "thread/read", {
        sessionId,
        historyLimit: 100,
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
    `multimodal turn timeout: session=${sessionId} turn=${turnId} expected=terminal actual=${actualStatus} timeoutMs=${options.timeoutMs} cancelStatus=${cancelStatus}`,
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
    const provider = liveProviderUsed
      ? {
          providerPreference: options.providerPreference,
          providerName: options.providerPreference,
          modelPreference: options.modelPreference,
          providerConfig: null,
        }
      : providerWithImageInput(fixture.provider);
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
    const sessionId = `multimodal-capture-${suffix}`;
    const turnId = `multimodal-capture-turn-${suffix}`;
    await transport.invoke(options, "thread/start", {
      sessionId,
      threadId: sessionId,
      appId: "desktop",
      workspaceId,
      workingDir: workspaceRoot,
      businessObjectRef: {
        kind: "agent.session",
        id: `agent-session:${workspaceId}:${sessionId}`,
        title: "Multimodal provider capture",
        metadata: {
          title: "Multimodal provider capture",
          executionStrategy: "react",
          runStartHooks: false,
          harness: {
            source: "smoke:agent-runtime-multimodal-capture",
            scenarioId,
          },
        },
      },
    });
    await transport.invoke(options, "thread/settings/update", {
      threadId: sessionId,
      modelProvider: provider.providerPreference,
      model: provider.modelPreference,
    });

    const turnPromise = transport.invoke(options, "turn/start", {
      sessionId,
      turnId,
      input: {
        text: prompt,
        attachments: [
          {
            kind: "image",
            uri: image.dataUrl,
            metadata: { mediaType: image.mediaType },
          },
        ],
      },
      runtimeOptions: {
        stream: true,
        eventName: `multimodal_capture_${suffix}`,
        runtimeRequest: {
          providerPreference: provider.providerPreference,
          modelPreference: provider.modelPreference,
          ...(provider.providerConfig
            ? { providerConfig: provider.providerConfig }
            : {}),
          approvalPolicy: "never",
          sandboxPolicy: "danger-full-access",
          executionStrategy: "react",
          workingDir: workspaceRoot,
          workspaceRoot,
          projectRoot: workspaceRoot,
          webSearch: false,
          searchMode: "disabled",
          metadata: {
            harness: {
              source: "smoke:agent-runtime-multimodal-capture",
              scenarioId,
              provider_budget: {
                max_provider_steps: 1,
              },
              generation: {
                max_output_tokens: 128,
                enable_thinking: false,
              },
              turn_policy: {
                tool_surface: "direct_answer",
              },
            },
          },
        },
      },
      queueIfBusy: false,
      skipPreSubmitResume: true,
    });
    const terminal = await waitForTerminal(
      transport,
      options,
      sessionId,
      turnId,
      turnPromise,
    );
    await turnPromise;
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
      assert(
        fixture.requests.length === 1,
        `provider request count=${fixture.requests.length}`,
      );
      const urls = imageUrls(fixture.requests[0]?.body);
      providerImagePayloadObserved = urls.includes(PNG_DATA_URL);
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
        providerToolCount === 0,
        `direct-answer provider request exposed ${providerToolCount} tools`,
      );
      assert(
        providerMaxOutputTokens === 128,
        `provider max_tokens=${providerMaxOutputTokens}`,
      );
      assert(
        providerThinkingEnabled === false,
        `provider enable_thinking=${providerThinkingEnabled}`,
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
          providerRequestPath,
          providerImagePayloadObserved,
          providerToolCount,
          providerMaxOutputTokens,
          providerThinkingEnabled,
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
