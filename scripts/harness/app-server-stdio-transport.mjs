import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";
import { pathToFileURL } from "node:url";

import { ensureElectronAppServerRuntimeBinary } from "../lib/electron-app-server-assets.mjs";

const DEEPSWE_RUNTIME_EVENT_TYPES = new Set([
  "provider.request.started",
  "provider.step",
  "provider.usage",
]);

export async function createAppServerStdioTransport({
  repoRoot,
  binaryPath,
  dataDir,
  timeoutMs,
  logPrefix,
}) {
  if (!fs.existsSync(binaryPath)) {
    throw new Error(`app-server binary missing: ${binaryPath}`);
  }
  if (!fs.existsSync(dataDir)) {
    throw new Error(`app-server data dir missing: ${dataDir}`);
  }
  ensureElectronAppServerRuntimeBinary({ binaryPath });
  const isolatedDataDir = createIsolatedAppServerDataDir(dataDir);

  const clientModulePath = path.join(
    repoRoot,
    "packages/app-server-client/dist/index.js",
  );
  if (!fs.existsSync(clientModulePath)) {
    throw new Error(
      `app-server client dist missing: ${clientModulePath}; run npm --prefix packages/app-server-client run build`,
    );
  }
  const clientModule = await import(pathToFileURL(clientModulePath).href);
  let connected;
  try {
    const config = {
      ...clientModule.stdioSidecar(
        binaryPath,
        undefined,
        isolatedDataDir,
        "retain",
      ),
      backendMode: "runtime",
    };
    connected = await clientModule.connectAppServerSidecar(
      config,
      {
        clientInfo: {
          name: "lime_deepswe_adapter",
          version: "1.0.0",
        },
        capabilities: {
          eventMethods: [clientModule.METHOD_AGENT_SESSION_EVENT],
        },
      },
      {
        initializeTimeoutMs: Math.min(timeoutMs, 30_000),
        expectedProtocolVersion: clientModule.PROTOCOL_VERSION,
        cwd: repoRoot,
        env: {
          APP_SERVER_BACKEND_MODE: "runtime",
        },
      },
    );
  } catch (error) {
    fs.rmSync(isolatedDataDir, { recursive: true, force: true });
    throw error;
  }

  return {
    async waitForReady() {
      console.log(
        `${logPrefix} App Server stdio ready protocol=${connected.initializeResponse.serverInfo.protocolVersion} dataIsolation=sqlite-vacuum-snapshot`,
      );
      return {
        status: "ok",
        transport: "app-server-stdio",
        dataIsolation: "sqlite-vacuum-snapshot",
      };
    },
    async invoke(
      options,
      method,
      params,
      requestTimeoutMs = options.timeoutMs,
    ) {
      const request = connected.client.request(method, params);
      const response = await connected.connection.request(
        request,
        request.method,
        { timeoutMs: requestTimeoutMs },
      );
      return response.result;
    },
    readRuntimeEvents: (identity) =>
      readAppServerRuntimeEvents(isolatedDataDir, identity),
    async close() {
      try {
        await connected.sidecar.close();
      } finally {
        fs.rmSync(isolatedDataDir, { recursive: true, force: true });
      }
    },
  };
}

export async function readAppServerRuntimeEvents(
  dataDir,
  { sessionId, threadId, turnId },
) {
  const eventsDir = path.join(dataDir, "runtime/events/sessions");
  if (!fs.existsSync(eventsDir)) {
    return [];
  }
  const events = [];
  const files = fs
    .readdirSync(eventsDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".jsonl"))
    .map((entry) => path.join(eventsDir, entry.name))
    .sort();
  for (const file of files) {
    const lines = readline.createInterface({
      input: fs.createReadStream(file, { encoding: "utf8" }),
      crlfDelay: Infinity,
    });
    let lineNumber = 0;
    for await (const line of lines) {
      lineNumber += 1;
      if (!line.trim()) continue;
      let event;
      try {
        event = JSON.parse(line);
      } catch (error) {
        throw new Error(
          `invalid App Server runtime event JSON at ${path.basename(file)}:${lineNumber}: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
      if (
        !DEEPSWE_RUNTIME_EVENT_TYPES.has(event?.type) ||
        (sessionId && event.sessionId !== sessionId) ||
        (threadId && event.threadId !== threadId) ||
        (turnId && event.turnId !== turnId)
      ) {
        continue;
      }
      events.push(event);
    }
  }
  return events;
}

function createIsolatedAppServerDataDir(sourceDataDir) {
  const sourceDatabase = path.join(sourceDataDir, "lime.db");
  if (!fs.existsSync(sourceDatabase)) {
    throw new Error(`app-server seed database missing: ${sourceDatabase}`);
  }
  const isolatedDataDir = fs.mkdtempSync(
    path.join(os.tmpdir(), "lime-deepswe-app-server-"),
  );
  const isolatedDatabase = path.join(isolatedDataDir, "lime.db");
  const escapedDatabasePath = isolatedDatabase.replaceAll("'", "''");
  try {
    execFileSync(
      "sqlite3",
      [sourceDatabase, `VACUUM INTO '${escapedDatabasePath}'`],
      { stdio: ["ignore", "pipe", "pipe"] },
    );
    return isolatedDataDir;
  } catch (error) {
    fs.rmSync(isolatedDataDir, { recursive: true, force: true });
    throw error;
  }
}
