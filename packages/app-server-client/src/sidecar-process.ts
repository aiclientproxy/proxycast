import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { once } from "node:events";
import { createInterface, type Interface as ReadlineInterface } from "node:readline";

import { AppServerConnection } from "./connection.js";
import {
  METHOD_INITIALIZE,
  PROTOCOL_VERSION,
  decodeMessage,
  encodeMessage,
  isJsonRpcErrorResponse,
  isJsonRpcResponse,
  type InitializeParams,
  type InitializeResponse,
  type JsonRpcMessage,
  type RequestId,
} from "./protocol.js";
import { AppServerClient } from "./request-client.js";
import { assertSidecarFileSha256, sidecarArgs } from "./sidecar-manifest.js";
import type {
  ConnectedAppServerSidecar,
  ConnectSidecarOptions,
  SidecarLaunchConfig,
  SidecarProcessOptions,
} from "./sidecar-types.js";

export async function spawnAppServerSidecar(
  config: SidecarLaunchConfig,
  options: SidecarProcessOptions = {},
): Promise<AppServerSidecar> {
  if (options.verifySha256 ?? Boolean(config.expectedSha256)) {
    await assertSidecarFileSha256(config);
  }

  const child = spawn(config.binaryPath, options.args ?? sidecarArgs(config), {
    ...options.spawnOptions,
    cwd: options.cwd,
    env: options.env ? { ...process.env, ...options.env } : process.env,
    stdio: "pipe",
  });

  return new AppServerSidecar(child);
}

export async function connectAppServerSidecar(
  config: SidecarLaunchConfig,
  initializeParams: InitializeParams,
  options: ConnectSidecarOptions = {},
): Promise<ConnectedAppServerSidecar> {
  const client = options.client ?? new AppServerClient();
  const sidecar = await spawnAppServerSidecar(config, options);

  try {
    const initializeRequest = client.initialize(initializeParams);
    sidecar.send(initializeRequest);
    const initializeMessage = await sidecar.nextMessage(
      options.initializeTimeoutMs,
    );
    const initializeResponse = expectResponseResult<InitializeResponse>(
      initializeMessage,
      initializeRequest.id,
      METHOD_INITIALIZE,
    );
    assertInitializeResponseProtocol(
      initializeResponse,
      options.expectedProtocolVersion ?? PROTOCOL_VERSION,
    );
    sidecar.send(client.initialized());

    return {
      client,
      connection: new AppServerConnection(sidecar, client),
      sidecar,
      initializeResponse,
    };
  } catch (error) {
    appendSidecarStderr(error, sidecar.stderrLines);
    await sidecar.close().catch(() => undefined);
    throw error;
  }
}

const SIDECAR_STDERR_TAIL_LIMIT = 20;

function appendSidecarStderr(
  error: unknown,
  stderrLines: readonly string[],
): void {
  if (!(error instanceof Error) || stderrLines.length === 0) {
    return;
  }
  const tail = stderrLines.slice(-SIDECAR_STDERR_TAIL_LIMIT);
  error.message = `${error.message}; stderr=${tail.join("\n")}`;
  Object.assign(error, { stderrLines: tail });
}

export class AppServerSidecar {
  readonly child: ChildProcessWithoutNullStreams;
  readonly stderrLines: string[] = [];

  #stdout: ReadlineInterface;
  #stderr: ReadlineInterface;
  #messages: JsonRpcMessage[] = [];
  #waiters: Array<{
    resolve: (message: JsonRpcMessage) => void;
    reject: (error: Error) => void;
    timer: NodeJS.Timeout;
  }> = [];
  #closed = false;
  #closedError: Error | null = null;

  constructor(child: ChildProcessWithoutNullStreams) {
    this.child = child;
    this.#stdout = createInterface({ input: child.stdout });
    this.#stderr = createInterface({ input: child.stderr });

    this.#stdout.on("line", (line) => this.#receiveLine(line));
    this.#stderr.on("line", (line) => {
      this.stderrLines.push(line);
      if (this.stderrLines.length > SIDECAR_STDERR_TAIL_LIMIT) {
        this.stderrLines.splice(
          0,
          this.stderrLines.length - SIDECAR_STDERR_TAIL_LIMIT,
        );
      }
    });
    child.stdin.on("error", (error) =>
      this.#markClosedWithError(
        normalizeSidecarStdinError(error, "app-server sidecar stdin error"),
      ),
    );
    child.once("error", (error) => this.#markClosedWithError(error));
    child.once("exit", (code, signal) =>
      this.#markClosedWithError(
        new Error(
          `app-server exited before next message: code=${code}, signal=${signal}`,
        ),
      ),
    );
  }

  send(message: JsonRpcMessage): void {
    this.sendLine(encodeMessage(message));
  }

  sendLine(line: string): void {
    if (this.#closed || this.child.stdin.destroyed) {
      throw new Error("app-server sidecar stdin is closed");
    }
    try {
      this.child.stdin.write(line, (error) => {
        if (error) {
          this.#markClosedWithError(
            normalizeSidecarStdinError(
              error,
              "app-server sidecar stdin write failed",
            ),
          );
        }
      });
    } catch (error) {
      throw normalizeSidecarStdinError(
        error,
        "app-server sidecar stdin write failed",
      );
    }
  }

  nextMessage(timeoutMs = 30_000): Promise<JsonRpcMessage> {
    const message = this.#messages.shift();
    if (message) {
      return Promise.resolve(message);
    }
    if (this.#closed) {
      return Promise.reject(
        this.#closedError ?? new Error("app-server sidecar is closed"),
      );
    }

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.#waiters = this.#waiters.filter(
          (waiter) => waiter.timer !== timer,
        );
        reject(
          new Error(
            `timed out waiting for app-server message after ${timeoutMs}ms`,
          ),
        );
      }, timeoutMs);
      this.#waiters.push({ resolve, reject, timer });
    });
  }

  async waitForExit(timeoutMs = 5_000): Promise<void> {
    if (this.child.exitCode !== null || this.child.signalCode !== null) {
      return;
    }
    await withTimeout(
      once(this.child, "exit"),
      timeoutMs,
      "timed out waiting for app-server exit",
    );
  }

  async close(
    signal: NodeJS.Signals = "SIGTERM",
    timeoutMs = 5_000,
  ): Promise<void> {
    if (this.child.exitCode === null && this.child.signalCode === null) {
      this.child.kill(signal);
      try {
        await this.waitForExit(timeoutMs);
      } catch (error) {
        this.child.kill("SIGKILL");
        await this.waitForExit(timeoutMs);
        throw error;
      }
    }

    this.#stdout.close();
    this.#stderr.close();
  }

  #receiveLine(line: string): void {
    let message: JsonRpcMessage;
    try {
      message = decodeMessage(line);
    } catch (error) {
      this.#rejectWaiters(
        error instanceof Error ? error : new Error(String(error)),
      );
      return;
    }

    const waiter = this.#waiters.shift();
    if (!waiter) {
      this.#messages.push(message);
      return;
    }
    clearTimeout(waiter.timer);
    waiter.resolve(message);
  }

  #rejectWaiters(error: Error): void {
    for (const waiter of this.#waiters.splice(0)) {
      clearTimeout(waiter.timer);
      waiter.reject(error);
    }
  }

  #markClosedWithError(error: Error): void {
    this.#closed = true;
    this.#closedError = error;
    this.#rejectWaiters(error);
  }
}

function normalizeSidecarStdinError(error: unknown, fallback: string): Error {
  const message = error instanceof Error ? error.message : String(error || "");
  if (
    message.includes("EPIPE") ||
    message.includes("ERR_STREAM_DESTROYED") ||
    message.includes("write after end")
  ) {
    return new Error("app-server sidecar stdin is closed");
  }
  return error instanceof Error ? error : new Error(fallback);
}

function expectResponseResult<T>(
  message: JsonRpcMessage,
  id: RequestId,
  method: string,
): T {
  if (isJsonRpcErrorResponse(message)) {
    throw new Error(`${method} failed: ${message.error.message}`);
  }
  if (!isJsonRpcResponse(message) || message.id !== id) {
    throw new Error(`expected ${method} response for request ${String(id)}`);
  }
  return message.result as T;
}

function assertInitializeResponseProtocol(
  response: InitializeResponse,
  expectedProtocolVersion: string,
): void {
  if (response.serverInfo.protocolVersion !== expectedProtocolVersion) {
    throw new Error(
      `unsupported app-server protocol: expected ${expectedProtocolVersion}, got ${response.serverInfo.protocolVersion}`,
    );
  }
}

async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  message: string,
): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
  }
}
