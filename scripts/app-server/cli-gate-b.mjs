#!/usr/bin/env node

import { execFile } from "node:child_process";
import {
  access,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { localAppServerBinaryPath } from "../lib/electron-dev-sidecar.mjs";
import { writeTerminalExternalBackend } from "./terminal-gate-fixture.mjs";

const execFileAsync = promisify(execFile);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "../..");
const cliBinaryName = process.platform === "win32" ? "lime.exe" : "lime";
const defaultCliBinaryPath = path.join(
  rootDir,
  "lime-rs",
  "target",
  "debug",
  cliBinaryName,
);
const prompt = "cli gate b prompt";
const completedText = "cli gate b completed";

async function main() {
  const cliBinaryPath = path.resolve(
    process.env.LIME_CLI_BIN || defaultCliBinaryPath,
  );
  const appServerBinaryPath = path.resolve(
    process.env.APP_SERVER_BIN || localAppServerBinaryPath({ repoRoot: rootDir }),
  );
  await Promise.all([
    assertBinaryExists(cliBinaryPath, "lime"),
    assertBinaryExists(appServerBinaryPath, "app-server"),
  ]);

  const tempDir = await mkdtemp(path.join(tmpdir(), "cli-gate-b-"));
  try {
    const backendPath = path.join(tempDir, "cli-backend.mjs");
    const ledgerPath = path.join(tempDir, "cli-backend.jsonl");
    const dataDir = path.join(tempDir, "data");
    const appDataDir = path.join(tempDir, "app-data");
    await Promise.all(
      [
        "home",
        "xdg-config",
        "xdg-data",
        "roaming-app-data",
        "local-app-data",
      ].map((directory) =>
        mkdir(path.join(tempDir, directory), { recursive: true }),
      ),
    );
    await writeTerminalExternalBackend(backendPath, {
      completedText,
      command: "printf cli-gate-b",
    });

    const args = [
      "exec",
      prompt,
      "--json",
      "--cwd",
      tempDir,
      "--model",
      "fixture-model",
      "--provider",
      "fixture-provider",
      "--app-server",
      appServerBinaryPath,
      "--app-server-arg=--backend",
      "--app-server-arg=external",
      "--app-server-arg=--backend-command",
      `--app-server-arg=${process.execPath}`,
      "--app-server-arg=--backend-arg",
      `--app-server-arg=${backendPath}`,
      "--app-server-arg=--backend-arg",
      `--app-server-arg=${ledgerPath}`,
      "--app-server-arg=--backend-timeout-ms",
      "--app-server-arg=5000",
      "--app-server-arg=--data-dir",
      `--app-server-arg=${dataDir}`,
      "--app-server-arg=--app-data-dir",
      `--app-server-arg=${appDataDir}`,
    ];
    const { stdout, stderr } = await runCli(cliBinaryPath, args, tempDir);
    if (stderr.trim()) {
      throw new Error(`lime wrote unexpected stderr: ${stderr.trim()}`);
    }

    const envelope = JSON.parse(stdout);
    assertEqual(envelope.ok, true, "CLI envelope ok");
    assertEqual(envelope.result?.status, "ready", "turn status");
    assertEqual(envelope.result?.output, completedText, "CLI output");
    assertNonEmptyString(envelope.result?.thread_id, "thread id");
    assertNonEmptyString(envelope.result?.turn_id, "turn id");

    const ledger = await readJsonLines(ledgerPath);
    const turnStart = ledger.find((entry) => entry?.kind === "turnStart");
    if (!turnStart) {
      throw new Error("external backend did not record turnStart");
    }
    assertEqual(turnStart.inputText, prompt, "backend input");
    assertEqual(
      turnStart.threadId,
      envelope.result.thread_id,
      "canonical thread identity",
    );
    assertEqual(
      turnStart.turnId,
      envelope.result.turn_id,
      "canonical turn identity",
    );
    assertEqual(
      turnStart.eventTypes.join(","),
      "message.delta,item.started,item.completed,turn.completed",
      "runtime event sequence",
    );

    console.log(
      [
        "[smoke:cli-gate-b] ok",
        `cli=${cliBinaryPath}`,
        `appServer=${appServerBinaryPath}`,
        `thread=${envelope.result.thread_id}`,
        `turn=${envelope.result.turn_id}`,
        `status=${envelope.result.status}`,
        `events=${turnStart.eventTypes.join(",")}`,
      ].join(" "),
    );
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

async function runCli(cliBinaryPath, args, tempDir) {
  const environment = await isolatedEnvironment(tempDir);
  try {
    return await execFileAsync(cliBinaryPath, args, {
      cwd: rootDir,
      encoding: "utf8",
      env: environment,
      maxBuffer: 1024 * 1024,
      timeout: 20_000,
      windowsHide: true,
    });
  } catch (error) {
    const stdout = String(error?.stdout ?? "").trim();
    const stderr = String(error?.stderr ?? "").trim();
    throw new Error(
      [
        error instanceof Error ? error.message : String(error),
        stdout ? `stdout: ${stdout}` : "stdout: <empty>",
        stderr ? `stderr: ${stderr}` : "stderr: <empty>",
      ].join("\n"),
    );
  }
}

async function isolatedEnvironment(tempDir) {
  const home = path.join(tempDir, "home");
  const appData = path.join(tempDir, "roaming-app-data");
  const localAppData = path.join(tempDir, "local-app-data");
  const environment = {
    ...process.env,
    HOME: home,
    XDG_CONFIG_HOME: path.join(tempDir, "xdg-config"),
    XDG_DATA_HOME: path.join(tempDir, "xdg-data"),
    APPDATA: appData,
    LOCALAPPDATA: localAppData,
  };
  if (process.platform === "darwin") {
    const libraryDirs = [path.join(rootDir, "lime-rs", "target", "debug")];
    const prebuiltRoot = path.join(
      rootDir,
      "lime-rs",
      "target",
      "sherpa-onnx-prebuilt",
    );
    const entries = await readdir(prebuiltRoot, { withFileTypes: true }).catch(
      () => [],
    );
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const libDir = path.join(prebuiltRoot, entry.name, "lib");
      try {
        await access(libDir);
        libraryDirs.push(libDir);
      } catch {
        // Optional prebuilt variants are absent on most developer machines.
      }
    }
    if (process.env.DYLD_LIBRARY_PATH) {
      libraryDirs.push(process.env.DYLD_LIBRARY_PATH);
    }
    environment.DYLD_LIBRARY_PATH = libraryDirs.join(path.delimiter);
  }
  return environment;
}

async function assertBinaryExists(targetPath, label) {
  try {
    await access(targetPath);
  } catch {
    throw new Error(
      `${label} binary not found: ${targetPath}\n` +
        '先构建：cargo build --manifest-path "lime-rs/Cargo.toml" -p cli -p app-server',
    );
  }
}

async function readJsonLines(filePath) {
  const content = await readFile(filePath, "utf8");
  return content
    .split(/\r?\n/u)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(
      `unexpected ${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`,
    );
  }
}

function assertNonEmptyString(value, label) {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`missing ${label}: ${JSON.stringify(value)}`);
  }
}

main().catch((error) => {
  console.error(
    `[smoke:cli-gate-b] failed: ${error instanceof Error ? error.message : String(error)}`,
  );
  process.exitCode = 1;
});
