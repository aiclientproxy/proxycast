import { existsSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { app } from "./electronRuntime";

const MANIFEST_NAME = "desktop-resources.manifest.json";
const SCHEMA_VERSION = 1;
const APPLICATION_ID = "com.limecloud.lime";

export type DesktopResourceReadiness = {
  status:
    | "ready"
    | "unverified"
    | "not_configured"
    | "unavailable"
    | "unsupported";
  reason: string;
  platformKey: string | null;
  resourceIds: string[];
};

type ResourceManifest = {
  schemaVersion?: number;
  applicationId?: string;
  platform?: string;
  arch?: string;
  platformKey?: string;
  resources?: Array<{ id?: string; path?: string; required?: boolean }>;
};

export function readDesktopResourceReadiness({
  platform = process.platform,
  arch = process.arch,
  packaged: _packaged = app.isPackaged,
  resourcesRoot,
}: {
  platform?: string;
  arch?: string;
  packaged?: boolean;
  resourcesRoot?: string;
} = {}): DesktopResourceReadiness {
  if (platform !== "darwin" && platform !== "win32") {
    return {
      status: "unsupported",
      reason:
        "Desktop resource readiness is only defined for macOS and Windows.",
      platformKey: null,
      resourceIds: [],
    };
  }

  const packagedRoot =
    typeof process.resourcesPath === "string" && process.resourcesPath.trim()
      ? process.resourcesPath
      : path.resolve(process.cwd(), "dist-electron");
  const root = resourcesRoot ?? packagedRoot;
  const manifestPath = path.join(root, MANIFEST_NAME);
  if (!existsSync(manifestPath)) {
    return {
      status: "not_configured",
      reason: "The packaged desktop resource manifest is not present.",
      platformKey: null,
      resourceIds: [],
    };
  }

  let manifest: ResourceManifest;
  try {
    manifest = JSON.parse(
      readFileSync(manifestPath, "utf8"),
    ) as ResourceManifest;
  } catch {
    return {
      status: "unavailable",
      reason: "The packaged desktop resource manifest is not valid JSON.",
      platformKey: null,
      resourceIds: [],
    };
  }

  const expectedPlatformKey =
    platform === "win32"
      ? "win32-x64"
      : `darwin-${arch === "arm64" ? "arm64" : "x64"}`;
  if (
    manifest.schemaVersion !== SCHEMA_VERSION ||
    manifest.applicationId !== APPLICATION_ID ||
    manifest.platform !== platform ||
    manifest.arch !== arch ||
    manifest.platformKey !== expectedPlatformKey ||
    !Array.isArray(manifest.resources)
  ) {
    return {
      status: "unavailable",
      reason:
        "The packaged desktop resource manifest identity does not match this runtime.",
      platformKey: manifest.platformKey ?? null,
      resourceIds: [],
    };
  }

  const resourceIds = manifest.resources
    .map((resource) => resource?.id)
    .filter((id): id is string => typeof id === "string" && id.length > 0);
  const requiredIds =
    platform === "win32"
      ? [
          "app-server",
          "code-mode-host",
          "windows-sandbox-setup",
          "windows-sandbox-runner",
          "windows-native-host",
        ]
      : ["app-server", "code-mode-host", "macos-native-host"];
  const missing = requiredIds.filter((id) => {
    const resource = manifest.resources?.find(
      (candidate) => candidate?.id === id,
    );
    const relativePath = normalizeRelativePath(resource?.path);
    if (!relativePath) {
      return true;
    }
    const absolutePath = path.resolve(root, relativePath);
    return (
      !absolutePath.startsWith(`${path.resolve(root)}${path.sep}`) ||
      !existsSync(absolutePath) ||
      !statSync(absolutePath).isFile()
    );
  });
  if (missing.length > 0) {
    return {
      status: "unavailable",
      reason: `Required desktop resources are missing: ${missing.join(", ")}.`,
      platformKey: expectedPlatformKey,
      resourceIds,
    };
  }

  return {
    status: "unverified",
    reason:
      "The manifest and required resource paths are present; digest, signature and runtime Gate B remain release checks.",
    platformKey: expectedPlatformKey,
    resourceIds,
  };
}

function normalizeRelativePath(value: unknown): string | null {
  const normalized = String(value ?? "").replace(/\\/g, "/");
  if (
    !normalized ||
    normalized.startsWith("/") ||
    normalized.split("/").some((part) => part === "..")
  ) {
    return null;
  }
  return normalized;
}
