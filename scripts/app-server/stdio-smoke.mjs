#!/usr/bin/env node

import { access, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { localAppServerBinaryPath } from "../lib/electron-dev-sidecar.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "../..");
const clientDistPath = path.join(
  rootDir,
  "packages",
  "app-server-client",
  "dist",
  "index.js",
);
const {
  METHOD_AGENT_SESSION_EVENT,
  PROTOCOL_VERSION,
  connectAppServerSidecar,
  resolveSidecarBinaryPath,
  stdioSidecar,
} = await import(pathToFileURL(clientDistPath).href);
const devBinaryPath = localAppServerBinaryPath({ repoRoot: rootDir });
const binaryResolution = resolveSidecarBinaryPath({
  devBinaryPath,
});
const binaryPath = binaryResolution?.binaryPath;

async function main() {
  if (!binaryPath) {
    throw new Error("app-server binary path could not be resolved");
  }
  await assertBinaryExists(binaryPath);

  const tempDir = await mkdtemp(path.join(tmpdir(), "app-server-stdio-"));
  let connected;
  try {
    connected = await connectAppServerSidecar(
      {
        ...stdioSidecar(binaryPath, undefined, path.join(tempDir, "data")),
        backendMode: "unavailable",
      },
      {
        clientInfo: {
          name: "lime_stdio_smoke",
          version: "1.60.0",
        },
        capabilities: {
          eventMethods: [METHOD_AGENT_SESSION_EVENT],
        },
      },
      {
        initializeTimeoutMs: 5_000,
        expectedProtocolVersion: PROTOCOL_VERSION,
      },
    );

    const threadRequest = connected.client.startSession({
      cwd: tempDir,
      runtimeWorkspaceRoots: [tempDir],
    });
    connected.sidecar.send(threadRequest);
    const threadResponse = await nextResponseForRequest(
      connected.sidecar,
      threadRequest.id,
      "thread/start",
    );
    expectResponseError(
      threadResponse,
      threadRequest.id,
      "thread/start",
      "runtime model route is not executable",
    );

    console.log(
      `[smoke:app-server-stdio] ok binary=${binaryPath} source=${binaryResolution.source} protocol=${connected.initializeResponse.serverInfo.protocolVersion} backend=unavailable thread=fail-closed`,
    );
  } finally {
    await connected?.sidecar.close();
    await rm(tempDir, { recursive: true, force: true });
  }
}

async function nextResponseForRequest(sidecar, id, label) {
  const deadline = Date.now() + 5_000;
  while (Date.now() < deadline) {
    const message = await sidecar.nextMessage(deadline - Date.now());
    if (message?.id === id) {
      return message;
    }
  }
  throw new Error(`timed out waiting for ${label} response ${String(id)}`);
}

async function assertBinaryExists(targetPath) {
  try {
    await access(targetPath);
  } catch {
    throw new Error(
      [
        `app-server binary not found: ${targetPath}`,
        '先构建：cargo build --manifest-path "lime-rs/Cargo.toml" -p app-server',
        "或设置：APP_SERVER_BIN=/path/to/app-server",
      ].join("\n"),
    );
  }
}

function expectResponseError(message, id, label, expectedMessage) {
  if (!message || message.id !== id || !("error" in message)) {
    throw new Error(
      `expected ${label} error response for request ${String(id)}`,
    );
  }
  const actualMessage = String(message.error?.message ?? "");
  if (!actualMessage.includes(expectedMessage)) {
    throw new Error(
      `unexpected ${label} error: expected "${expectedMessage}", got "${actualMessage}"`,
    );
  }
  return message.error;
}

main().catch((error) => {
  console.error(
    `[smoke:app-server-stdio] failed: ${error instanceof Error ? error.message : String(error)}`,
  );
  process.exitCode = 1;
});
