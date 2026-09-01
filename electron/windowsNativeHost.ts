import { createHash } from "node:crypto";
import {
  readFileSync,
  statSync,
  existsSync,
} from "node:fs";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import path from "node:path";
import { app } from "./electronRuntime";

const RESOURCE_MANIFEST = "desktop-resources.manifest.json";
const RESOURCE_SCHEMA_VERSION = 1;
const RESOURCE_APPLICATION_ID = "com.limecloud.lime";
const REQUEST_TIMEOUT_MS = 10_000;

export type WindowsNativeHostRequest = {
  method: string;
  params?: Record<string, unknown>;
};

export type WindowsNativeHostEvent = {
  event: string;
  payload?: unknown;
};

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

type NativeHostResponse = {
  id?: string;
  event?: string;
  payload?: unknown;
  ok?: boolean;
  result?: unknown;
  error?: { code?: string; message?: string; data?: unknown };
};

export class WindowsNativeHostClient {
  #child: ChildProcessWithoutNullStreams | null = null;
  #stdoutBuffer = "";
  #nextRequestId = 1;
  readonly #pending = new Map<string, PendingRequest>();
  readonly #eventListeners = new Set<
    (event: WindowsNativeHostEvent) => void
  >();

  onEvent(listener: (event: WindowsNativeHostEvent) => void): () => void {
    this.#eventListeners.add(listener);
    return () => this.#eventListeners.delete(listener);
  }

  async invoke({
    method,
    params = {},
  }: WindowsNativeHostRequest): Promise<unknown> {
    if (process.platform !== "win32") {
      throw new NativeHostError(
        "unsupported",
        "Windows native host is only available on Windows.",
      );
    }
    const helperPath = resolveWindowsNativeHostPath();
    if (!helperPath) {
      throw new NativeHostError(
        "unavailable",
        "Windows native host resource is missing or failed integrity verification.",
      );
    }
    const child = this.#ensureChild(helperPath);
    const id = String(this.#nextRequestId++);
    return await new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.#pending.delete(id);
        reject(
          new NativeHostError(
            "timeout",
            `Windows native host request timed out: ${method}`,
          ),
        );
      }, REQUEST_TIMEOUT_MS);
      this.#pending.set(id, { resolve, reject, timer });
      child.stdin.write(
        `${JSON.stringify({ id, method, params })}\n`,
        "utf8",
        (error) => {
          if (error) this.#settleFailure(id, error);
        },
      );
    });
  }

  dispose(): void {
    const child = this.#child;
    this.#child = null;
    if (child && !child.killed) child.kill();
    this.#settleAllFailure(new Error("Windows native host stopped"));
    this.#eventListeners.clear();
  }

  #ensureChild(helperPath: string): ChildProcessWithoutNullStreams {
    if (this.#child && !this.#child.killed) return this.#child;
    const child = spawn(helperPath, [], {
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    this.#child = child;
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => this.#consumeStdout(chunk));
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", () => undefined);
    child.once("error", (error) => {
      this.#child = null;
      this.#settleAllFailure(error);
    });
    child.once("exit", () => {
      if (this.#child === child) this.#child = null;
      this.#settleAllFailure(new Error("Windows native host exited"));
    });
    return child;
  }

  #consumeStdout(chunk: string): void {
    this.#stdoutBuffer += chunk;
    while (true) {
      const newline = this.#stdoutBuffer.indexOf("\n");
      if (newline < 0) return;
      const line = this.#stdoutBuffer.slice(0, newline).trim();
      this.#stdoutBuffer = this.#stdoutBuffer.slice(newline + 1);
      if (!line) continue;
      let response: NativeHostResponse;
      try {
        response = JSON.parse(line) as NativeHostResponse;
      } catch {
        continue;
      }
      if (typeof response.event === "string" && response.event.length > 0) {
        const event = { event: response.event, payload: response.payload };
        for (const listener of this.#eventListeners) {
          try {
            listener(event);
          } catch {}
        }
        continue;
      }
      if (typeof response.id !== "string") continue;
      if (response.ok) {
        this.#settleSuccess(response.id, response.result);
      } else {
        const error = response.error;
        this.#settleFailure(
          response.id,
          new NativeHostError(
            error?.code ?? "native_host_error",
            error?.message ?? "Windows native host request failed.",
            error?.data,
          ),
        );
      }
    }
  }

  #settleSuccess(id: string, value: unknown): void {
    const pending = this.#pending.get(id);
    if (!pending) return;
    this.#pending.delete(id);
    clearTimeout(pending.timer);
    pending.resolve(value);
  }

  #settleFailure(id: string, error: Error): void {
    const pending = this.#pending.get(id);
    if (!pending) return;
    this.#pending.delete(id);
    clearTimeout(pending.timer);
    pending.reject(error);
  }

  #settleAllFailure(error: Error): void {
    for (const id of this.#pending.keys()) this.#settleFailure(id, error);
  }
}

export class NativeHostError extends Error {
  readonly code: string;
  readonly data: unknown;

  constructor(code: string, message: string, data?: unknown) {
    super(message);
    this.name = "NativeHostError";
    this.code = code;
    this.data = data;
  }
}

export function resolveWindowsNativeHostPath(): string | null {
  if (process.platform !== "win32") return null;
  const resourcesRoot =
    app?.isPackaged === true && process.resourcesPath?.trim()
      ? process.resourcesPath
      : path.resolve(process.cwd(), "dist-electron");
  const manifestPath = path.join(resourcesRoot, RESOURCE_MANIFEST);
  if (!existsSync(manifestPath)) return null;
  let manifest: {
    schemaVersion?: number;
    applicationId?: string;
    platform?: string;
    arch?: string;
    platformKey?: string;
    resources?: Array<{ id?: string; path?: string; sha256?: string }>;
    native?: { windowsHelper?: { id?: string; path?: string } | null };
  };
  try {
    manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as typeof manifest;
  } catch {
    return null;
  }
  const expectedKey = "win32-x64";
  if (
    manifest.schemaVersion !== RESOURCE_SCHEMA_VERSION ||
    manifest.applicationId !== RESOURCE_APPLICATION_ID ||
    manifest.platform !== "win32" ||
    manifest.arch !== "x64" ||
    manifest.platformKey !== expectedKey
  ) return null;
  const metadata = manifest.native?.windowsHelper;
  if (metadata?.id !== "windows-native-host" || typeof metadata.path !== "string") {
    return null;
  }
  const resource = manifest.resources?.find(
    (candidate) => candidate?.id === metadata.id,
  );
  if (!resource || resource.path !== metadata.path || !/^[a-f0-9]{64}$/u.test(resource.sha256 ?? "")) {
    return null;
  }
  const normalized = metadata.path.replace(/\\/g, "/");
  if (!normalized || normalized.startsWith("/") || normalized.split("/").includes("..")) return null;
  const helperPath = path.resolve(resourcesRoot, normalized);
  if (!helperPath.startsWith(`${path.resolve(resourcesRoot)}${path.sep}`) || !existsSync(helperPath) || !statSync(helperPath).isFile()) return null;
  try {
    if (createHash("sha256").update(readFileSync(helperPath)).digest("hex") !== resource.sha256) return null;
  } catch {
    return null;
  }
  return helperPath;
}
