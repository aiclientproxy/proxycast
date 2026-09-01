import { createHash } from "node:crypto";
import {
  execFileSync,
  spawn,
  type ChildProcessWithoutNullStreams,
} from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { app } from "./electronRuntime";

const RESOURCE_MANIFEST = "desktop-resources.manifest.json";
const RESOURCE_SCHEMA_VERSION = 1;
const RESOURCE_APPLICATION_ID = "com.limecloud.lime";
const REQUEST_TIMEOUT_MS = 10_000;

export type MacOSNativeHostRequest = {
  method: string;
  params?: Record<string, unknown>;
};

export type MacOSNativeHostEvent = {
  event: string;
  payload?: unknown;
};

export type MacOSNativeHostEventListener = (
  event: MacOSNativeHostEvent,
) => void;

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

export class MacOSNativeHostClient {
  #child: ChildProcessWithoutNullStreams | null = null;
  #stdoutBuffer = "";
  #nextRequestId = 1;
  readonly #pending = new Map<string, PendingRequest>();
  readonly #eventListeners = new Set<MacOSNativeHostEventListener>();

  onEvent(listener: MacOSNativeHostEventListener): () => void {
    this.#eventListeners.add(listener);
    return () => this.#eventListeners.delete(listener);
  }

  async invoke({
    method,
    params = {},
  }: MacOSNativeHostRequest): Promise<unknown> {
    if (process.platform !== "darwin") {
      throw new NativeHostError(
        "unsupported",
        "macOS native host is only available on macOS.",
      );
    }
    const helperPath = resolveMacOSNativeHostPath();
    if (!helperPath) {
      throw new NativeHostError(
        "unavailable",
        "macOS native host resource is missing or failed integrity verification.",
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
            `macOS native host request timed out: ${method}`,
          ),
        );
      }, REQUEST_TIMEOUT_MS);
      this.#pending.set(id, { resolve, reject, timer });
      child.stdin.write(
        `${JSON.stringify({ id, method, params })}\n`,
        "utf8",
        (error) => {
          if (!error) {
            return;
          }
          this.#settleFailure(id, error);
        },
      );
    });
  }

  dispose(): void {
    const child = this.#child;
    this.#child = null;
    if (child && !child.killed) {
      child.kill();
    }
    this.#settleAllFailure(new Error("macOS native host stopped"));
    this.#eventListeners.clear();
  }

  #ensureChild(helperPath: string): ChildProcessWithoutNullStreams {
    if (this.#child && !this.#child.killed) {
      return this.#child;
    }

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
      if (this.#child === child) {
        this.#child = null;
      }
      this.#settleAllFailure(new Error("macOS native host exited"));
    });
    return child;
  }

  #consumeStdout(chunk: string): void {
    this.#stdoutBuffer += chunk;
    while (true) {
      const newline = this.#stdoutBuffer.indexOf("\n");
      if (newline < 0) {
        return;
      }
      const line = this.#stdoutBuffer.slice(0, newline).trim();
      this.#stdoutBuffer = this.#stdoutBuffer.slice(newline + 1);
      if (!line) {
        continue;
      }
      let response: NativeHostResponse;
      try {
        response = JSON.parse(line) as NativeHostResponse;
      } catch {
        continue;
      }
      if (typeof response.event === "string" && response.event.length > 0) {
        const event = {
          event: response.event,
          payload: response.payload,
        } satisfies MacOSNativeHostEvent;
        for (const listener of this.#eventListeners) {
          try {
            listener(event);
          } catch {}
        }
        continue;
      }
      if (typeof response.id !== "string") {
        continue;
      }
      if (response.ok) {
        this.#settleSuccess(response.id, response.result);
      } else {
        const error = response.error;
        this.#settleFailure(
          response.id,
          new NativeHostError(
            error?.code ?? "native_host_error",
            error?.message ?? "macOS native host request failed.",
            error?.data,
          ),
        );
      }
    }
  }

  #settleSuccess(id: string, value: unknown): void {
    const pending = this.#pending.get(id);
    if (!pending) {
      return;
    }
    this.#pending.delete(id);
    clearTimeout(pending.timer);
    pending.resolve(value);
  }

  #settleFailure(id: string, error: Error): void {
    const pending = this.#pending.get(id);
    if (!pending) {
      return;
    }
    this.#pending.delete(id);
    clearTimeout(pending.timer);
    pending.reject(error);
  }

  #settleAllFailure(error: Error): void {
    for (const id of this.#pending.keys()) {
      this.#settleFailure(id, error);
    }
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

export function resolveMacOSNativeHostPath(): string | null {
  if (process.platform !== "darwin") {
    return null;
  }
  const packagedResourcesRoot =
    typeof process.resourcesPath === "string" && process.resourcesPath.trim()
      ? process.resourcesPath
      : null;
  const resourcesRoot =
    app?.isPackaged === true
      ? packagedResourcesRoot
      : path.resolve(process.cwd(), "dist-electron");
  if (!resourcesRoot) {
    return null;
  }
  const manifestPath = path.join(resourcesRoot, RESOURCE_MANIFEST);
  if (!existsSync(manifestPath)) {
    return null;
  }
  let manifest: {
    schemaVersion?: number;
    applicationId?: string;
    platform?: string;
    arch?: string;
    platformKey?: string;
    native?: {
      helper?: {
        id?: string;
        path?: string;
        signedByForge?: boolean;
        bundlePath?: string;
        bundleIdentifier?: string;
      };
    };
    resources?: Array<{ id?: string; path?: string; sha256?: string }>;
  };
  try {
    manifest = JSON.parse(
      readFileSync(manifestPath, "utf8"),
    ) as typeof manifest;
  } catch {
    return null;
  }
  if (
    manifest.schemaVersion !== RESOURCE_SCHEMA_VERSION ||
    manifest.applicationId !== RESOURCE_APPLICATION_ID ||
    manifest.platform !== "darwin" ||
    manifest.arch !== process.arch ||
    manifest.platformKey !==
      `darwin-${process.arch === "arm64" ? "arm64" : "x64"}`
  ) {
    return null;
  }
  const helperMetadata = manifest.native?.helper;
  const resource = manifest.resources?.find(
    (candidate) => candidate?.id === "macos-native-host",
  );
  const relativePath = resource?.path?.replace(/\\/g, "/");
  if (
    helperMetadata?.id !== "macos-native-host" ||
    helperMetadata.path?.replace(/\\/g, "/") !== relativePath ||
    helperMetadata.bundleIdentifier !==
      `${RESOURCE_APPLICATION_ID}.native-host` ||
    !relativePath ||
    relativePath.startsWith("/") ||
    relativePath.split("/").some((part) => part === "..") ||
    !/^[a-f0-9]{64}$/u.test(resource?.sha256 ?? "")
  ) {
    return null;
  }
  const declaredBundlePath = helperMetadata.bundlePath;
  if (declaredBundlePath) {
    const bundlePath = resolveManifestRelativePath(
      resourcesRoot,
      declaredBundlePath,
    );
    if (!bundlePath || !statSync(bundlePath).isDirectory()) {
      return null;
    }
  }
  const helperPath = path.resolve(resourcesRoot, relativePath);
  if (
    !helperPath.startsWith(`${path.resolve(resourcesRoot)}${path.sep}`) ||
    !existsSync(helperPath) ||
    !statSync(helperPath).isFile()
  ) {
    return null;
  }
  try {
    const digest = createHash("sha256")
      .update(readFileSync(helperPath))
      .digest("hex");
    const signaturePath = declaredBundlePath
      ? resolveManifestRelativePath(resourcesRoot, declaredBundlePath)
      : helperPath;
    if (!signaturePath) {
      return null;
    }
    if (digest === resource?.sha256) {
      if (manifest.native?.helper?.signedByForge !== true) {
        return helperPath;
      }
      execFileSync("codesign", ["--verify", "--strict", signaturePath], {
        stdio: "ignore",
      });
      return helperPath;
    }
    if (manifest.native?.helper?.signedByForge !== true) {
      return null;
    }
    execFileSync("codesign", ["--verify", "--strict", signaturePath], {
      stdio: "ignore",
    });
    return helperPath;
  } catch {
    return null;
  }
}

function resolveManifestRelativePath(resourcesRoot: string, value?: string) {
  const relativePath = value?.replace(/\\/g, "/");
  if (
    !relativePath ||
    relativePath.startsWith("/") ||
    relativePath.split("/").some((part) => part === "..")
  ) {
    return null;
  }
  const resolved = path.resolve(resourcesRoot, relativePath);
  return resolved.startsWith(`${path.resolve(resourcesRoot)}${path.sep}`) &&
    existsSync(resolved)
    ? resolved
    : null;
}
