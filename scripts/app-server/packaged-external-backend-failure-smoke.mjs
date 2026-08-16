#!/usr/bin/env node

import {
  access,
  chmod,
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { copyElectronAppServerRuntimeLibraries } from "../lib/electron-app-server-assets.mjs";
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
  PROTOCOL_VERSION,
  defaultReleaseManifestPath,
  platformKey,
  sha256File,
  sidecarBinaryName,
  startPackagedAppServerSidecar,
} = await import(pathToFileURL(clientDistPath).href);

async function main() {
  const sourceBinaryPath = await resolveSourceBinaryPath();
  const tempDir = await mkdtemp(
    path.join(tmpdir(), "app-server-packaged-failure-smoke-"),
  );
  let lifecycle;

  try {
    const manifestVersion = await readPackageVersion();
    const platform = platformKey();
    const resourcesPath = path.join(tempDir, "resources");
    const packagedDir = path.join(resourcesPath, "app-server", platform);
    const packagedBinaryPath = path.join(packagedDir, sidecarBinaryName());
    const manifestPath = defaultReleaseManifestPath(resourcesPath);
    const backendPath = path.join(tempDir, "external-backend-fails.mjs");

    await mkdir(packagedDir, { recursive: true });
    await copyFile(sourceBinaryPath, packagedBinaryPath);
    await chmod(packagedBinaryPath, 0o755).catch(() => undefined);
    const runtimeLibraries = await copyElectronAppServerRuntimeLibraries({
      repoRoot: rootDir,
      sourceBinary: sourceBinaryPath,
      destinationDirectory: packagedDir,
    });
    await writeFailingExternalBackend(backendPath);

    await writeFile(
      manifestPath,
      `${JSON.stringify(
        {
          version: manifestVersion,
          protocolVersion: PROTOCOL_VERSION,
          artifacts: [
            {
              platform,
              url: `file://${packagedBinaryPath}`,
              sha256: await sha256File(packagedBinaryPath),
            },
          ],
        },
        null,
        2,
      )}\n`,
    );

    const started = await startPackagedAppServerSidecar(
      {
        clientInfo: {
          name: "content_studio_failure_smoke",
          version: manifestVersion,
        },
        capabilities: {},
      },
      {
        resourcesPath,
        dataDir: path.join(tempDir, "data"),
        backendMode: "external",
        backendCommand: process.execPath,
        backendArgs: [backendPath],
        backendTimeoutMs: 5_000,
        initializeTimeoutMs: 5_000,
        expectedProtocolVersion: PROTOCOL_VERSION,
        restartPolicy: {
          maxAttempts: 1,
          initialDelayMs: 0,
        },
      },
    );
    lifecycle = started.lifecycle;

    assertEqual(
      started.resolved.config.binaryPath,
      packagedBinaryPath,
      "packaged binary path",
    );
    const connection = started.connected.connection;
    const threadStart = await connection.startSession(
      {
        historyMode: "paginated",
        model: "fixture-model",
        modelProvider: "fixture-provider",
        serviceName: "content-studio",
        threadSource: "appServer",
      },
      { timeoutMs: 5_000 },
    );
    const threadId = requireNonEmptyString(
      threadStart.result.thread?.id,
      "thread/start thread id",
    );
    const sessionId = requireNonEmptyString(
      threadStart.result.thread?.sessionId,
      "thread/start session id",
    );

    const turnResult = await connection.startTurn(
      {
        threadId,
        input: [
          {
            type: "text",
            text: "packaged external backend failure smoke",
          },
        ],
      },
      { timeoutMs: 5_000 },
    );
    const turnId = requireNonEmptyString(
      turnResult.result.turn?.id,
      "turn/start turn id",
    );
    assertEqual(
      turnResult.result.turn.status,
      "inProgress",
      "turn/start status",
    );

    const clientNotifications = await collectRuntimeNotificationsUntilFailure(
      connection,
      5_000,
      { threadId, turnId },
    );
    const clientFailure = assertDirectFailureNotifications(
      clientNotifications,
      { threadId, turnId },
    );

    const readResult = await connection.readThread(
      { threadId, includeTurns: true },
      { timeoutMs: 5_000 },
    );
    assertEqual(
      readResult.result.thread.sessionId,
      sessionId,
      "read thread session id",
    );
    assertEqual(readResult.result.thread.id, threadId, "read thread id");
    const readTurns = Array.isArray(readResult.result.thread.turns)
      ? readResult.result.thread.turns
      : [];
    assertEqual(readTurns.length, 1, "read failed turn count");
    const readTurn = readTurns.find((turn) => turn?.id === turnId);
    if (!readTurn) {
      throw new Error(`read thread is missing failed turn ${turnId}`);
    }
    assertEqual(readTurn.status, "failed", "read failed turn status");
    if (!readTurn.completedAt) {
      throw new Error(`read failed turn ${turnId} is missing completedAt`);
    }

    const readTurnError = String(
      readTurn.error?.message ?? readTurn.error ?? readTurn.failure?.message ?? "",
    );
    if (
      !readTurnError.includes(
        "packaged external backend crashed after partial output",
      )
    ) {
      throw new Error(
        `read thread missing failure summary: ${JSON.stringify(readTurn)}`,
      );
    }

    await lifecycle.stop();

    console.log(
      [
        "[smoke:app-server-packaged-external-backend-failure] ok",
        `source=${sourceBinaryPath}`,
        `packaged=${packagedBinaryPath}`,
        `protocol=${started.connected.initializeResponse.serverInfo.protocolVersion}`,
        `clientNotifications=${clientNotifications.map((notification) => notification.method).join(",")}`,
        `threadId=${threadId}`,
        `turnId=${turnId}`,
        `readTurns=${readTurns.length}`,
        `readTurnStatus=${readTurn.status}`,
        `runtimeLibraries=${runtimeLibraries.length}`,
        `clientFailure=${JSON.stringify(clientFailure.params.turn.error.message)}`,
      ].join(" "),
    );
  } finally {
    await lifecycle?.stop().catch(() => undefined);
    await rm(tempDir, { recursive: true, force: true });
  }
}

async function resolveSourceBinaryPath() {
  const binaryPath =
    process.env.APP_SERVER_BIN ||
    localAppServerBinaryPath({ repoRoot: rootDir });
  try {
    await access(binaryPath);
    return binaryPath;
  } catch {
    throw new Error(
      [
        `app-server binary not found: ${binaryPath}`,
        '先构建：cargo build --manifest-path "lime-rs/Cargo.toml" -p app-server',
        "或设置：APP_SERVER_BIN=/path/to/app-server",
      ].join("\n"),
    );
  }
}

async function readPackageVersion() {
  const packageJson = JSON.parse(
    await readFile(path.join(rootDir, "package.json"), "utf8"),
  );
  return String(packageJson.version || "").trim();
}

async function writeFailingExternalBackend(backendPath) {
  await writeFile(
    backendPath,
    `#!/usr/bin/env node
console.log(JSON.stringify({
  type: "message.delta",
  payload: {
    text: "partial packaged failure"
  }
}));
console.error("packaged external backend crashed after partial output");
process.exit(7);
`,
  );
}

async function collectRuntimeNotificationsUntilFailure(
  connection,
  timeoutMs,
  identity,
) {
  const deadline = Date.now() + timeoutMs;
  const notifications = [];
  while (
    !notifications.some(
      (notification) =>
        notification.method === "turn/completed" &&
        notification.params?.threadId === identity.threadId &&
        notification.params?.turn?.id === identity.turnId &&
        notification.params?.turn?.status === "failed",
    )
  ) {
    const remainingMs = deadline - Date.now();
    if (remainingMs <= 0) {
      throw new Error(
        `timed out waiting for streamed failure notifications: ${JSON.stringify(notifications)}`,
      );
    }
    const notification = await connection.nextNotification(remainingMs);
    notifications.push(notification);
  }
  return notifications;
}

function assertDirectFailureNotifications(notifications, identity) {
  const delta = notifications.find(
    (notification) =>
      notification.method === "item/agentMessage/delta" &&
      notification.params?.threadId === identity.threadId &&
      notification.params?.turnId === identity.turnId,
  );
  if (!delta) {
    throw new Error(
      `client notifications missing item/agentMessage/delta: ${JSON.stringify(notifications)}`,
    );
  }
  if (!String(delta.params?.delta ?? "").includes("partial packaged failure")) {
    throw new Error(
      `client delta missing partial output: ${JSON.stringify(delta)}`,
    );
  }

  const failed = notifications.find(
    (notification) =>
      notification.method === "turn/completed" &&
      notification.params?.threadId === identity.threadId &&
      notification.params?.turn?.id === identity.turnId &&
      notification.params?.turn?.status === "failed",
  );
  if (!failed) {
    throw new Error(
      `client notifications missing failed turn/completed: ${JSON.stringify(notifications)}`,
    );
  }
  const message = String(failed.params?.turn?.error?.message ?? "");
  if (
    !message.includes("packaged external backend crashed after partial output")
  ) {
    throw new Error(
      `client turn/completed missing stderr summary: ${JSON.stringify(failed)}`,
    );
  }
  return failed;
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`unexpected ${label}: expected ${expected}, got ${actual}`);
  }
}

function requireNonEmptyString(value, label) {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`missing ${label}: ${JSON.stringify(value)}`);
  }
  return value.trim();
}

main().catch((error) => {
  console.error(
    `[smoke:app-server-packaged-external-backend-failure] failed: ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
