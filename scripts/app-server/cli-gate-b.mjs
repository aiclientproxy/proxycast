#!/usr/bin/env node

import { spawn } from "node:child_process";
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
import { fileURLToPath } from "node:url";
import { localAppServerBinaryPath } from "../lib/electron-dev-sidecar.mjs";
import { writeTerminalExternalBackend } from "./terminal-gate-fixture.mjs";

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
    process.env.APP_SERVER_BIN ||
      localAppServerBinaryPath({ repoRoot: rootDir }),
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
    if (process.env.LIME_CLI_GATE_B_USE_SIBLING_APP_SERVER !== "1") {
      args.splice(9, 0, "--app-server", appServerBinaryPath);
    }
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

    const jsonlArgs = args.map((argument) =>
      argument === "--json" ? "--jsonl" : argument,
    );
    const jsonl = await runCli(cliBinaryPath, jsonlArgs, tempDir);
    if (jsonl.stderr.trim()) {
      throw new Error(
        `jsonl exec wrote unexpected stderr: ${jsonl.stderr.trim()}`,
      );
    }
    assertEqual(
      jsonl.stdout.trim().split(/\r?\n/u).length,
      1,
      "single-line JSONL envelope",
    );
    const jsonlEnvelope = JSON.parse(jsonl.stdout);
    assertEqual(jsonlEnvelope.ok, true, "JSONL envelope ok");
    assertEqual(jsonlEnvelope.result?.output, completedText, "JSONL output");

    const stdinArgs = ["exec", ...args.slice(2)];
    const stdin = await runCli(cliBinaryPath, stdinArgs, tempDir, {
      input: `${prompt}\n`,
    });
    assertEqual(
      JSON.parse(stdin.stdout).result?.output,
      completedText,
      "stdin output",
    );

    const invalid = await runCliResult(
      cliBinaryPath,
      ["exec", "", "--json"],
      tempDir,
    );
    assertEqual(invalid.code, 1, "empty prompt exit code");
    assertEqual(JSON.parse(invalid.stdout).ok, false, "error envelope");
    assertEqual(invalid.stderr.trim(), "", "error envelope stderr");

    const completion = await runCli(
      cliBinaryPath,
      ["completion", "zsh"],
      tempDir,
    );
    if (
      !completion.stdout.includes("_lime") ||
      !completion.stdout.includes("completion")
    ) {
      throw new Error(
        "zsh completion did not describe the canonical lime command tree",
      );
    }

    console.log(
      [
        "[smoke:cli-gate-b] ok",
        `cli=${cliBinaryPath}`,
        `appServer=${appServerBinaryPath}`,
        `thread=${envelope.result.thread_id}`,
        `turn=${envelope.result.turn_id}`,
        `status=${envelope.result.status}`,
        `events=${turnStart.eventTypes.join(",")}`,
        "jsonl=ok",
        "stdin=ok",
        "error-exit=1",
        "completion=zsh",
      ].join(" "),
    );
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

async function runCli(cliBinaryPath, args, tempDir, options = {}) {
  const result = await runCliResult(cliBinaryPath, args, tempDir, options);
  if (result.code !== 0) {
    throw new Error(
      [
        `lime exited with code ${result.code}`,
        result.stdout ? `stdout: ${result.stdout.trim()}` : "stdout: <empty>",
        result.stderr ? `stderr: ${result.stderr.trim()}` : "stderr: <empty>",
      ].join("\n"),
    );
  }
  return result;
}

async function runCliResult(cliBinaryPath, args, tempDir, options = {}) {
  const environment = await isolatedEnvironment(tempDir);
  return new Promise((resolve, reject) => {
    const child = spawn(cliBinaryPath, args, {
      cwd: rootDir,
      env: environment,
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
      signal: AbortSignal.timeout(options.timeoutMs ?? 20_000),
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.once("error", reject);
    child.once("close", (code, signal) => {
      resolve({
        code: typeof code === "number" ? code : 1,
        signal,
        stdout,
        stderr,
      });
    });
    if (options.input != null) {
      child.stdin.end(options.input);
    } else {
      child.stdin.end();
    }
  });
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
