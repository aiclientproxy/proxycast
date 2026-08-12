import type { SpawnOptionsWithoutStdio } from "node:child_process";

import type { AppServerConnection } from "./connection.js";
import type { InitializeResponse } from "./protocol.js";
import type { AppServerClient } from "./request-client.js";
import type { AppServerSidecarLifecycle } from "./sidecar-lifecycle.js";
import type { AppServerSidecar } from "./sidecar-process.js";

export const DEFAULT_LISTEN_URL = "stdio://";

export const DEFAULT_RELEASE_MANIFEST_NAME = "app-server.release.json";

export const DEFAULT_PROTOCOL_SCHEMA_MANIFEST_NAME = "manifest.json";

export type SidecarLaunchConfig = {
  binaryPath: string;
  listenUrl: string;
  dataDir?: string;
  backendMode?: "external" | "runtime" | "mock" | "unavailable";
  backendCommand?: string;
  backendArgs?: string[];
  backendTimeoutMs?: number;
  appPolicyPath?: string;
  expectedSha256?: string;
  artifact?: AppServerReleaseArtifact;
};

export type SidecarBinaryPathSource = "env" | "resources" | "dev";

export type SidecarBinaryPathResolution = {
  binaryPath: string;
  source: SidecarBinaryPathSource;
};

export type ResolveSidecarBinaryPathOptions = {
  env?: NodeJS.ProcessEnv;
  envVarName?: string;
  allowEnvOverride?: boolean;
  resourcesPath?: string;
  resourceRelativePath?: string;
  devBinaryPath?: string;
  platform?: NodeJS.Platform | string;
  arch?: NodeJS.Architecture | string;
};

export type ResolveSidecarFromManifestOptions =
  ResolveSidecarBinaryPathOptions & {
    listenUrl?: string;
    backendMode?: SidecarLaunchConfig["backendMode"];
    backendCommand?: string;
    backendArgs?: string[];
    backendTimeoutMs?: number;
    appPolicyPath?: string;
    dataDir?: string;
    expectedProtocolVersion?: string;
  };

export const DEFAULT_STANDALONE_BACKEND_MODE: NonNullable<
  SidecarLaunchConfig["backendMode"]
> = "unavailable";

export type ResolvedSidecarLaunchConfig = {
  config: SidecarLaunchConfig;
  artifact: AppServerReleaseArtifact;
  binaryPathSource: SidecarBinaryPathSource;
};

export type SidecarProcessOptions = {
  args?: string[];
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  verifySha256?: boolean;
  spawnOptions?: Omit<SpawnOptionsWithoutStdio, "stdio" | "cwd" | "env">;
};

export type ConnectSidecarOptions = SidecarProcessOptions & {
  client?: AppServerClient;
  initializeTimeoutMs?: number;
  expectedProtocolVersion?: string;
};

export type SidecarRestartPolicy = {
  maxAttempts?: number;
  initialDelayMs?: number;
  maxDelayMs?: number;
  factor?: number;
};

export type SidecarExitEvent = {
  attempt: number;
  code: number | null;
  signal: NodeJS.Signals | null;
  stderrLines: string[];
};

export type SidecarRestartScheduledEvent = SidecarExitEvent & {
  delayMs: number;
};

export type SidecarRestartFailedEvent = {
  attempt: number;
  error: unknown;
  stderrLines?: string[];
};

export type SidecarLifecycleOptions = ConnectSidecarOptions & {
  restartPolicy?: SidecarRestartPolicy;
  sleep?: (delayMs: number) => Promise<void>;
  onExit?: (event: SidecarExitEvent) => void;
  onRestartScheduled?: (event: SidecarRestartScheduledEvent) => void;
  onRestarted?: (connected: ConnectedAppServerSidecar, attempt: number) => void;
  onRestartFailed?: (event: SidecarRestartFailedEvent) => void;
};

export type PackagedSidecarLifecycleOptions = SidecarLifecycleOptions &
  ResolveSidecarFromManifestOptions & {
    resourcesPath: string;
    manifestPath?: string;
    manifestRelativePath?: string;
  };

export type ConnectedAppServerSidecar = {
  client: AppServerClient;
  connection: AppServerConnection;
  sidecar: AppServerSidecar;
  initializeResponse: InitializeResponse;
};

export type StartedPackagedAppServerSidecar = {
  resolved: ResolvedSidecarLaunchConfig;
  lifecycle: AppServerSidecarLifecycle;
  connected: ConnectedAppServerSidecar;
};

export type AppServerArtifactPlatform =
  | "darwin-arm64"
  | "darwin-x64"
  | "win32-x64"
  | "linux-x64"
  | string;

export type AppServerReleaseArtifact = {
  platform: AppServerArtifactPlatform;
  url: string;
  sha256: string;
  codeModeHostSha256?: string;
};

export type AppServerReleaseManifest = {
  version: string;
  protocolVersion: string;
  artifacts: AppServerReleaseArtifact[];
};
