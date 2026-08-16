#!/usr/bin/env node

import { access, mkdtemp, rm, writeFile } from "node:fs/promises";
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

async function main() {
  const binaryResolution = resolveSidecarBinaryPath({
    devBinaryPath,
  });
  const binaryPath = binaryResolution?.binaryPath;
  if (!binaryPath) {
    throw new Error("app-server binary path could not be resolved");
  }
  await assertBinaryExists(binaryPath);

  const tempDir = await mkdtemp(
    path.join(tmpdir(), "app-server-external-backend-"),
  );
  let connected;
  try {
    const backendPath = path.join(tempDir, "query-loop-backend.mjs");
    const policyPath = path.join(tempDir, "content-studio.policy.json");
    await writeExternalBackend(backendPath);
    await writeFile(
      policyPath,
      `${JSON.stringify(
        {
          capabilities: [
            {
              id: "content.draft.generate",
              title: "Generate Draft",
              methods: ["turn/start"],
              appIds: ["content-studio"],
            },
          ],
        },
        null,
        2,
      )}\n`,
    );

    connected = await connectAppServerSidecar(
      {
        ...stdioSidecar(binaryPath, policyPath, path.join(tempDir, "data")),
        backendMode: "external",
        backendCommand: process.execPath,
        backendArgs: [backendPath],
        backendTimeoutMs: 5_000,
      },
      {
        clientInfo: {
          name: "content_studio",
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

    const connection = connected.connection;
    const sessionId = "appserver_external_backend_smoke_session";
    const threadId = "appserver_external_backend_smoke_thread";
    const turnId = "turn_external_backend_smoke";

    const capabilityResult = await connection.listCapabilities(
      {
        appId: "content-studio",
        workspaceId: "content-workspace",
      },
      { timeoutMs: 5_000 },
    );
    const capabilityIds = capabilityResult.result.capabilities.map(
      (capability) => capability.id,
    );
    if (!capabilityIds.includes("content.draft.generate")) {
      throw new Error(
        `content.draft.generate capability missing: ${capabilityIds.join(", ")}`,
      );
    }

    const sessionResult = await connection.startSession(
      {
        sessionId,
        threadId,
        appId: "content-studio",
        workspaceId: "content-workspace",
      },
      { timeoutMs: 5_000 },
    );
    assertEqual(
      sessionResult.result.session.sessionId,
      sessionId,
      "session id",
    );
    assertEqual(sessionResult.result.session.threadId, threadId, "thread id");

    const turnResult = await connection.startTurn(
      {
        sessionId,
        turnId,
        input: {
          text: "生成一段 content-studio 草稿",
        },
        runtimeOptions: {
          stream: true,
          capabilityId: "content.draft.generate",
          metadata: {
            smoke: true,
          },
        },
      },
      { timeoutMs: 5_000 },
    );
    assertEqual(turnResult.result.turn.sessionId, sessionId, "turn session id");
    assertEqual(turnResult.result.turn.turnId, turnId, "turn id");
    assertEqual(turnResult.result.turn.status, "completed", "turn status");
    if (!turnResult.result.turn.completedAt) {
      throw new Error(`turn ${turnId} is missing completedAt`);
    }

    const notificationTypes = [];
    for (const notification of turnResult.notifications) {
      if (notification.method === METHOD_AGENT_SESSION_EVENT) {
        notificationTypes.push(notification.params.event.type);
      }
    }
    while (
      !notificationTypes.includes("message.delta") ||
      !notificationTypes.includes("artifact.snapshot") ||
      !notificationTypes.includes("turn.completed")
    ) {
      const notification = await connection.nextNotification(5_000);
      assertEqual(
        notification.method,
        METHOD_AGENT_SESSION_EVENT,
        "event method",
      );
      assertEqual(
        notification.params.event.sessionId,
        sessionId,
        "event session id",
      );
      notificationTypes.push(notification.params.event.type);
    }

    const readResult = await connection.readSession(
      { sessionId },
      { timeoutMs: 5_000 },
    );
    assertEqual(
      readResult.result.session.sessionId,
      sessionId,
      "read session id",
    );
    assertEqual(readResult.result.session.threadId, threadId, "read thread id");
    assertEqual(
      readResult.result.session.appId,
      "content-studio",
      "read app id",
    );
    assertEqual(
      readResult.result.session.workspaceId,
      "content-workspace",
      "read workspace id",
    );
    const readTurns = Array.isArray(readResult.result.turns)
      ? readResult.result.turns
      : [];
    assertEqual(readTurns.length, 1, "read turn count");
    const readTurn = readTurns.find((turn) => turn?.turnId === turnId);
    if (!readTurn) {
      throw new Error(`read session is missing turn ${turnId}`);
    }
    assertEqual(readTurn.sessionId, sessionId, "read turn session id");
    assertEqual(readTurn.threadId, threadId, "read turn thread id");
    assertEqual(readTurn.status, "completed", "read turn status");
    if (!readTurn.completedAt) {
      throw new Error(`read turn ${turnId} is missing completedAt`);
    }

    const artifactResult = await connection.readArtifacts(
      {
        sessionId,
        turnId,
      },
      { timeoutMs: 5_000 },
    );
    assertEqual(artifactResult.result.artifacts.length, 1, "artifact count");
    assertEqual(
      artifactResult.result.artifacts[0].artifactRef,
      "content-draft-smoke",
      "artifact ref",
    );

    console.log(
      [
        "[smoke:app-server-external-backend] ok",
        `binary=${binaryPath}`,
        `source=${binaryResolution.source}`,
        `protocol=${connected.initializeResponse.serverInfo.protocolVersion}`,
        `capabilities=${capabilityIds.join(",")}`,
        `events=${notificationTypes.join(",")}`,
        `readSession=${readResult.result.session.sessionId}`,
        `readTurns=${readTurns.length}`,
        `readTurnStatus=${readTurn.status}`,
        `artifacts=${artifactResult.result.artifacts.length}`,
      ].join(" "),
    );
  } finally {
    await connected?.sidecar.close().catch(() => undefined);
    await rm(tempDir, { recursive: true, force: true });
  }
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

async function writeExternalBackend(backendPath) {
  await writeFile(
    backendPath,
    `#!/usr/bin/env node
import { readFileSync } from "node:fs";

const input = JSON.parse(readFileSync(0, "utf8"));

if (input.kind === "turnStart") {
  const text = input.request.input?.text ?? "";
  console.log(JSON.stringify({
    events: [
      {
        type: "message.delta",
        payload: {
          backend: "external",
          appId: input.request.session?.appId,
          text: \`草稿已生成：\${text}\`,
        },
      },
      {
        type: "artifact.snapshot",
        payload: {
          artifactId: "content-draft-smoke",
          title: "Content Draft Smoke",
          kind: "markdown",
          status: "ready",
          path: ".app-server/artifacts/content-draft-smoke.md",
          content: \`# Content Draft Smoke\\n\\n\${text}\`,
          metadata: {
            backend: "external",
            smoke: true,
          },
        },
      },
      {
        type: "turn.completed",
        payload: {
          backend: "external",
          status: "completed",
        },
      },
    ],
  }));
  process.exit(0);
}

console.log(JSON.stringify({ events: [] }));
`,
  );
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`unexpected ${label}: expected ${expected}, got ${actual}`);
  }
}

main().catch((error) => {
  console.error(
    `[smoke:app-server-external-backend] failed: ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
